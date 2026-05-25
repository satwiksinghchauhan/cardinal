#pragma once
// =============================================================================
// Cardinal - Episodic Retriever
// File: src/memory/episodic_retriever.h
//
// Unified retrieval interface for past episodes.
// Given a user message, returns the most relevant past episodes
// for injection into the inference prompt.
//
// Three modes (matching verifier pattern):
//   KEYWORD  -- SQLite FTS5 full-text search. Always available.
//               Best for exact or near-exact query matches.
//   SEMANTIC -- TF-IDF cosine similarity. Pure C++, no model needed.
//               CPU-based, zero VRAM cost. Better than keyword for
//               paraphrased or conceptually similar queries.
//   HYBRID   -- Weighted combination of both. Recommended default.
//               Scores are normalized to [0,1] before merging.
//
// TF-IDF index:
//   Built over user_message + response_summary of all episodes.
//   Cached in memory. Rebuilt according to cache_rebuild_strategy:
//     ON_DEMAND -- when corpus grows by cache_rebuild_threshold episodes
//     PERIODIC  -- every cache_rebuild_interval_seconds seconds
//     EXPLICIT  -- only when rebuild_index() is called directly
//
// Thread safety:
//   All public methods are protected by a shared_mutex.
//   Reads use shared lock, writes (index rebuild) use exclusive lock.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "memory/episodic_storage.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <chrono>

namespace cardinal {

    // -------------------------------------------------------------------------
    // RetrievalMode
    // Mirrors verifier mode pattern -- symbolic/neural/hybrid becomes
    // keyword/semantic/hybrid.
    // -------------------------------------------------------------------------
    enum class RetrievalMode {
        KEYWORD,   // FTS5 only
        SEMANTIC,  // TF-IDF cosine only
        HYBRID     // Weighted combination
    };

    // -------------------------------------------------------------------------
    // CacheRebuildStrategy
    // Controls when the TF-IDF index is rebuilt.
    // -------------------------------------------------------------------------
    enum class CacheRebuildStrategy {
        ON_DEMAND,  // Rebuild when corpus grows by threshold
        PERIODIC,   // Rebuild every N seconds
        EXPLICIT    // Rebuild only on explicit rebuild_index() call
    };

    // -------------------------------------------------------------------------
    // RetrievalResult
    // A single retrieved episode with its relevance scores.
    // -------------------------------------------------------------------------
    struct RetrievalResult {
        EpisodeRecord episode;
        float         score = 0.0f;  // Combined score in [0, 1]
        float         keyword_score = 0.0f;  // Normalized keyword component
        float         semantic_score = 0.0f;  // Normalized semantic component
        RetrievalMode mode_used = RetrievalMode::HYBRID;

        std::string score_summary() const;
    };

    // -------------------------------------------------------------------------
    // TfIdfVector
    // Sparse TF-IDF vector for a single document.
    // Maps term_index -> tfidf_weight.
    // -------------------------------------------------------------------------
    using TfIdfVector = std::unordered_map<int, float>;

    // -------------------------------------------------------------------------
    // IndexedEpisode
    // An episode with its precomputed TF-IDF vector and L2 norm.
    // -------------------------------------------------------------------------
    struct IndexedEpisode {
        std::string  episode_id;
        TfIdfVector  tfidf;
        float        norm = 0.0f;
    };

    // -------------------------------------------------------------------------
    // EpisodicRetriever
    //
    // Usage:
    //   EpisodicRetriever retriever(config, storage);
    //   retriever.init();
    //   auto results = retriever.retrieve(user_message);
    //   retriever.notify_new_episode(ep_id);
    // -------------------------------------------------------------------------
    class EpisodicRetriever {
    public:
        EpisodicRetriever(const CardinalConfig& config,
            EpisodicStorage& storage);

        EpisodicRetriever(const EpisodicRetriever&) = delete;
        EpisodicRetriever& operator=(const EpisodicRetriever&) = delete;

        // -- Lifecycle --

        // Build the initial TF-IDF index. Must be called before retrieve().
        void init();

        // -- Core retrieval --

        // Retrieve relevant past episodes for a query.
        // Mode comes from config.retriever.mode.
        std::vector<RetrievalResult> retrieve(const std::string& query) const;

        // Retrieve with explicit mode override.
        std::vector<RetrievalResult> retrieve(const std::string& query,
            RetrievalMode      mode) const;

        // -- Index management --

        // Notify that a new episode was written to storage.
        // Triggers rebuild check based on strategy.
        void notify_new_episode(const std::string& episode_id);

        // Force an immediate full index rebuild.
        void rebuild_index();

        // -- Runtime config --
        void          set_mode(RetrievalMode mode);
        RetrievalMode mode() const { return mode_; }

        void set_weights(float keyword_weight, float semantic_weight);

        // -- Stats --
        int  index_size()      const;
        int  vocabulary_size() const;
        bool index_ready()     const { return index_built_; }

    private:
        // -- Tokenization --
        static std::vector<std::string> tokenize(const std::string& text);
        static bool                     is_stopword(const std::string& token);

        // -- TF-IDF engine --
        void        build_vocabulary(
            const std::vector<std::pair<std::string,
            std::string>>&docs);
        TfIdfVector compute_tfidf(
            const std::vector<std::string>& tokens) const;
        static float l2_norm(const TfIdfVector& v);
        static float cosine_similarity(const TfIdfVector& a, float a_norm,
            const TfIdfVector& b, float b_norm);

        // -- Retrieval internals --
        std::vector<std::pair<std::string, float>>
            keyword_search(const std::string& query, int limit) const;

        std::vector<std::pair<std::string, float>>
            semantic_search(const std::string& query, int limit) const;

        std::vector<RetrievalResult>
            merge_results(
                const std::vector<std::pair<std::string, float>>& keyword_results,
                const std::vector<std::pair<std::string, float>>& semantic_results,
                RetrievalMode                                      mode) const;

        std::vector<EpisodeRecord>
            fetch_episodes(const std::vector<std::string>& ids) const;

        // -- Cache rebuild helpers --
        void maybe_rebuild();

        static RetrievalMode        parse_mode(const std::string& s);
        static CacheRebuildStrategy parse_strategy(const std::string& s);

        // -- Members --
        const CardinalConfig& config_;
        EpisodicStorage& storage_;

        RetrievalMode        mode_;
        CacheRebuildStrategy rebuild_strategy_;
        float                keyword_weight_;
        float                semantic_weight_;

        // TF-IDF index
        std::unordered_map<std::string, int> vocab_;     // term -> term_index
        std::unordered_map<int, float>       idf_;       // term_index -> idf weight
        std::vector<IndexedEpisode>          index_;     // one per episode
        std::unordered_map<std::string, int> index_map_; // episode_id -> index_ pos

        bool  index_built_ = false;
        int   episodes_since_rebuild_ = 0;

        std::chrono::steady_clock::time_point last_rebuild_time_;

        mutable std::shared_mutex mutex_;
    };

    // -------------------------------------------------------------------------
    // EpisodicRetrieverError
    // -------------------------------------------------------------------------
    class EpisodicRetrieverError : public std::runtime_error {
    public:
        explicit EpisodicRetrieverError(const std::string& message)
            : std::runtime_error("EpisodicRetrieverError: " + message) {}
    };

} // namespace cardinal