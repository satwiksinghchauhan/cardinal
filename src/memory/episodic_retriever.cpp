// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Episodic Retriever Implementation
// File: src/memory/episodic_retriever.cpp
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "memory/episodic_retriever.h"
#include "utils/logger.h"

#include <sqlite3.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cctype>
#include <numeric>
#include <unordered_set>

namespace cardinal {

    // =========================================================================
    // Stopword list
    // Common English words that carry no semantic signal for TF-IDF.
    // Kept minimal -- over-filtering hurts short episode texts.
    // =========================================================================
    static const std::unordered_set<std::string> STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "what", "which", "who", "how", "when",
        "where", "why", "not", "no", "so", "if", "as", "up", "out",
        "about", "into", "than", "then", "there", "can", "my", "your"
    };

    // =========================================================================
    // Constructor
    // =========================================================================

    EpisodicRetriever::EpisodicRetriever(const CardinalConfig& config,
        EpisodicStorage& storage)
        : config_(config)
        , storage_(storage)
        , mode_(parse_mode(config.retriever.mode))
        , rebuild_strategy_(parse_strategy(config.retriever.cache_rebuild_strategy))
        , keyword_weight_(config.retriever.keyword_weight)
        , semantic_weight_(config.retriever.semantic_weight)
        , last_rebuild_time_(std::chrono::steady_clock::now())
    {
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void EpisodicRetriever::init() {
        LOG_INFO("EpisodicRetriever initializing...");
        rebuild_index();
        LOG_INFO("EpisodicRetriever ready. Index size: " +
            std::to_string(index_size()) +
            " episodes, vocabulary: " +
            std::to_string(vocabulary_size()) + " terms");
    }

    // =========================================================================
    // Core retrieval
    // =========================================================================

    std::vector<RetrievalResult>
        EpisodicRetriever::retrieve(const std::string& query) const {
        return retrieve(query, mode_);
    }

    std::vector<RetrievalResult>
        EpisodicRetriever::retrieve(const std::string& query,
            RetrievalMode      mode) const {
        if (query.empty()) return {};

        std::shared_lock<std::shared_mutex> lock(mutex_);

        if (!index_built_ && mode != RetrievalMode::KEYWORD) {
            LOG_WARN("EpisodicRetriever::retrieve called before index is built "
                "-- falling back to keyword mode");
            mode = RetrievalMode::KEYWORD;
        }

        const int limit = config_.retriever.max_results * 3;
        // Fetch more than needed before merging so we have candidates to
        // score and filter -- final cap applied after merge

        std::vector<std::pair<std::string, float>> kw_results;
        std::vector<std::pair<std::string, float>> sem_results;

        if (mode == RetrievalMode::KEYWORD || mode == RetrievalMode::HYBRID) {
            kw_results = keyword_search(query, limit);
        }
        if (mode == RetrievalMode::SEMANTIC || mode == RetrievalMode::HYBRID) {
            sem_results = semantic_search(query, limit);
        }

        auto results = merge_results(kw_results, sem_results, mode);

        // Apply min_score filter
        const float min_score = config_.retriever.min_score;
        results.erase(
            std::remove_if(results.begin(), results.end(),
                [min_score](const RetrievalResult& r) {
                    return r.score < min_score;
                }),
            results.end()
        );

        // Cap at max_results
        if (static_cast<int>(results.size()) > config_.retriever.max_results) {
            results.resize(config_.retriever.max_results);
        }

        LOG_DEBUG("EpisodicRetriever: query='" + query +
            "' returned " + std::to_string(results.size()) + " results");

        return results;
    }

    // =========================================================================
    // Index management
    // =========================================================================

    void EpisodicRetriever::notify_new_episode(const std::string& episode_id) {
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            ++episodes_since_rebuild_;
        }
        // Suppress unused parameter warning in case we add id-based logic later
        (void)episode_id;
        maybe_rebuild();
    }

    void EpisodicRetriever::rebuild_index() {
        std::unique_lock<std::shared_mutex> lock(mutex_);

        LOG_DEBUG("EpisodicRetriever: rebuilding TF-IDF index...");

        // Fetch all episodes from storage for indexing
        // We use a large limit here -- the index holds everything
        EpisodeQuery q;
        q.max_results = 100000;
        q.recent_first = false;  // Stable order for deterministic vocab
        auto episodes = storage_.query(q);

        if (episodes.empty()) {
            LOG_DEBUG("EpisodicRetriever: no episodes to index");
            index_built_ = true;
            episodes_since_rebuild_ = 0;
            last_rebuild_time_ = std::chrono::steady_clock::now();
            return;
        }

        // Build (id, text) pairs for vocabulary construction
        // Text = user_message + " " + response_summary
        std::vector<std::pair<std::string, std::string>> docs;
        docs.reserve(episodes.size());
        for (const auto& ep : episodes) {
            docs.emplace_back(ep.id,
                ep.user_message + " " + ep.response_summary);
        }

        // Build vocabulary and IDF weights
        build_vocabulary(docs);

        // Build per-episode TF-IDF vectors
        index_.clear();
        index_map_.clear();
        index_.reserve(docs.size());

        for (size_t i = 0; i < docs.size(); ++i) {
            auto tokens = tokenize(docs[i].second);
            auto tfidf = compute_tfidf(tokens);
            float norm = l2_norm(tfidf);

            IndexedEpisode ie;
            ie.episode_id = docs[i].first;
            ie.tfidf = std::move(tfidf);
            ie.norm = norm;

            index_map_[ie.episode_id] = static_cast<int>(index_.size());
            index_.push_back(std::move(ie));
        }

        index_built_ = true;
        episodes_since_rebuild_ = 0;
        last_rebuild_time_ = std::chrono::steady_clock::now();

        LOG_DEBUG("EpisodicRetriever: index rebuilt. " +
            std::to_string(index_.size()) + " episodes, " +
            std::to_string(vocab_.size()) + " terms");
    }

    // =========================================================================
    // Runtime config
    // =========================================================================

    void EpisodicRetriever::set_mode(RetrievalMode mode) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        mode_ = mode;
    }

    void EpisodicRetriever::set_weights(float keyword_weight,
        float semantic_weight) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        keyword_weight_ = keyword_weight;
        semantic_weight_ = semantic_weight;
    }

    // =========================================================================
    // Stats
    // =========================================================================

    int EpisodicRetriever::index_size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return static_cast<int>(index_.size());
    }

    int EpisodicRetriever::vocabulary_size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return static_cast<int>(vocab_.size());
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    std::vector<std::string>
        EpisodicRetriever::tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string token;

        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                token += static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            }
            else {
                if (!token.empty()) {
                    if (!is_stopword(token) && token.size() > 1) {
                        tokens.push_back(token);
                    }
                    token.clear();
                }
            }
        }
        // Don't forget the last token
        if (!token.empty() && !is_stopword(token) && token.size() > 1) {
            tokens.push_back(token);
        }

        return tokens;
    }

    bool EpisodicRetriever::is_stopword(const std::string& token) {
        return STOPWORDS.count(token) > 0;
    }

    // =========================================================================
    // TF-IDF engine
    // =========================================================================

    void EpisodicRetriever::build_vocabulary(
        const std::vector<std::pair<std::string, std::string>>& docs)
    {
        vocab_.clear();
        idf_.clear();

        int N = static_cast<int>(docs.size());

        // Document frequency: term -> number of docs containing term
        std::unordered_map<std::string, int> df;

        for (const auto& [id, text] : docs) {
            auto tokens = tokenize(text);
            // Use a set per document to count df correctly (once per doc)
            std::unordered_set<std::string> seen;
            for (const auto& t : tokens) {
                if (seen.insert(t).second) {
                    ++df[t];
                }
            }
        }

        // Assign term indices and compute IDF
        // IDF formula: log((N + 1) / (df + 1)) + 1
        // The +1 smoothing prevents division by zero and keeps IDF positive
        int term_idx = 0;
        for (const auto& [term, doc_freq] : df) {
            vocab_[term] = term_idx;
            idf_[term_idx] = std::log(
                (static_cast<float>(N) + 1.0f) /
                (static_cast<float>(doc_freq) + 1.0f)
            ) + 1.0f;
            ++term_idx;
        }
    }

    TfIdfVector EpisodicRetriever::compute_tfidf(
        const std::vector<std::string>& tokens) const
    {
        if (tokens.empty() || vocab_.empty()) return {};

        // Term frequency for this document
        std::unordered_map<std::string, int> tf_counts;
        for (const auto& t : tokens) {
            ++tf_counts[t];
        }

        float doc_len = static_cast<float>(tokens.size());
        TfIdfVector result;

        for (const auto& [term, count] : tf_counts) {
            auto vocab_it = vocab_.find(term);
            if (vocab_it == vocab_.end()) continue;  // OOV term

            int   term_idx = vocab_it->second;
            auto  idf_it = idf_.find(term_idx);
            if (idf_it == idf_.end()) continue;

            float tf = static_cast<float>(count) / doc_len;
            float tfidf = tf * idf_it->second;
            result[term_idx] = tfidf;
        }

        return result;
    }

    float EpisodicRetriever::l2_norm(const TfIdfVector& v) {
        float sum = 0.0f;
        for (const auto& [idx, weight] : v) {
            sum += weight * weight;
        }
        return std::sqrt(sum);
    }

    float EpisodicRetriever::cosine_similarity(const TfIdfVector& a,
        float              a_norm,
        const TfIdfVector& b,
        float              b_norm) {
        if (a_norm < 1e-9f || b_norm < 1e-9f) return 0.0f;

        float dot = 0.0f;
        // Iterate over smaller vector for efficiency
        const TfIdfVector* smaller = (a.size() < b.size()) ? &a : &b;
        const TfIdfVector* larger = (a.size() < b.size()) ? &b : &a;

        for (const auto& [idx, weight] : *smaller) {
            auto it = larger->find(idx);
            if (it != larger->end()) {
                dot += weight * it->second;
            }
        }

        return dot / (a_norm * b_norm);
    }

    // =========================================================================
    // Keyword search (FTS5)
    // =========================================================================

    std::vector<std::pair<std::string, float>>
        EpisodicRetriever::keyword_search(const std::string& query,
            int                limit) const {
        // Access the underlying SQLite database via EpisodicStorage query.
        // We use EpisodeQuery with the keyword field set -- this routes
        // through FTS5 in episodic_storage.cpp.
        EpisodeQuery q;
        q.keyword = query;
        q.max_results = limit;
        q.recent_first = true;

        auto episodes = storage_.query(q);

        if (episodes.empty()) return {};

        // FTS5 doesn't return scores directly through our query interface.
        // Assign rank-based scores: 1.0 for best match, decaying linearly.
        // This is a reasonable approximation since FTS5 already ranked them.
        std::vector<std::pair<std::string, float>> results;
        results.reserve(episodes.size());

        float n = static_cast<float>(episodes.size());
        for (size_t i = 0; i < episodes.size(); ++i) {
            // Score: 1.0 for rank 0, decaying to 1/n for last rank
            float score = 1.0f - (static_cast<float>(i) / n) * 0.5f;
            results.emplace_back(episodes[i].id, score);
        }

        return results;
    }

    // =========================================================================
    // Semantic search (TF-IDF cosine)
    // =========================================================================

    std::vector<std::pair<std::string, float>>
        EpisodicRetriever::semantic_search(const std::string& query,
            int                limit) const {
        if (!index_built_ || index_.empty()) return {};

        // Compute query vector
        auto query_tokens = tokenize(query);
        if (query_tokens.empty()) return {};

        auto query_tfidf = compute_tfidf(query_tokens);
        float query_norm = l2_norm(query_tfidf);

        if (query_norm < 1e-9f) return {};

        // Score all indexed episodes
        std::vector<std::pair<std::string, float>> scored;
        scored.reserve(index_.size());

        for (const auto& ie : index_) {
            float sim = cosine_similarity(query_tfidf, query_norm,
                ie.tfidf, ie.norm);
            if (sim > 1e-6f) {
                scored.emplace_back(ie.episode_id, sim);
            }
        }

        // Sort by score descending
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });

        // Cap at limit
        if (static_cast<int>(scored.size()) > limit) {
            scored.resize(limit);
        }

        return scored;
    }

    // =========================================================================
    // Merge results
    // =========================================================================

    std::vector<RetrievalResult>
        EpisodicRetriever::merge_results(
            const std::vector<std::pair<std::string, float>>& keyword_results,
            const std::vector<std::pair<std::string, float>>& semantic_results,
            RetrievalMode                                      mode) const
    {
        // Build maps for fast lookup
        std::unordered_map<std::string, float> kw_map;
        std::unordered_map<std::string, float> sem_map;

        for (const auto& [id, score] : keyword_results)  kw_map[id] = score;
        for (const auto& [id, score] : semantic_results) sem_map[id] = score;

        // Union of all episode IDs seen in either result set
        std::unordered_set<std::string> all_ids;
        for (const auto& [id, _] : kw_map)  all_ids.insert(id);
        for (const auto& [id, _] : sem_map) all_ids.insert(id);

        // Compute combined scores
        std::vector<std::pair<std::string, RetrievalResult>> scored;
        scored.reserve(all_ids.size());

        for (const auto& id : all_ids) {
            float kw_score = 0.0f;
            float sem_score = 0.0f;

            auto kw_it = kw_map.find(id);
            auto sem_it = sem_map.find(id);

            if (kw_it != kw_map.end())  kw_score = kw_it->second;
            if (sem_it != sem_map.end()) sem_score = sem_it->second;

            float combined = 0.0f;
            switch (mode) {
            case RetrievalMode::KEYWORD:
                combined = kw_score;
                break;
            case RetrievalMode::SEMANTIC:
                combined = sem_score;
                break;
            case RetrievalMode::HYBRID:
                combined = (keyword_weight_ * kw_score) +
                    (semantic_weight_ * sem_score);
                break;
            }

            RetrievalResult r;
            r.score = combined;
            r.keyword_score = kw_score;
            r.semantic_score = sem_score;
            r.mode_used = mode;

            scored.emplace_back(id, std::move(r));
        }

        // Sort by combined score descending
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) {
                return a.second.score > b.second.score;
            });

        // Fetch full episode records for top results
        // Limit how many we fetch to avoid hammering storage
        const int fetch_limit = config_.retriever.max_results * 2;
        if (static_cast<int>(scored.size()) > fetch_limit) {
            scored.resize(fetch_limit);
        }

        std::vector<std::string> ids_to_fetch;
        ids_to_fetch.reserve(scored.size());
        for (const auto& [id, _] : scored) {
            ids_to_fetch.push_back(id);
        }

        auto episodes = fetch_episodes(ids_to_fetch);

        // Build episode_id -> record map
        std::unordered_map<std::string, EpisodeRecord> ep_map;
        for (auto& ep : episodes) {
            ep_map[ep.id] = std::move(ep);
        }

        // Assemble final results -- only include episodes we could fetch
        std::vector<RetrievalResult> results;
        results.reserve(scored.size());

        for (auto& [id, result] : scored) {
            auto it = ep_map.find(id);
            if (it != ep_map.end()) {
                result.episode = std::move(it->second);
                results.push_back(std::move(result));
            }
        }

        return results;
    }

    // =========================================================================
    // Episode fetch helper
    // =========================================================================

    std::vector<EpisodeRecord>
        EpisodicRetriever::fetch_episodes(
            const std::vector<std::string>& ids) const
    {
        std::vector<EpisodeRecord> results;
        results.reserve(ids.size());

        for (const auto& id : ids) {
            auto ep = storage_.get_episode(id);
            if (ep) {
                results.push_back(std::move(*ep));
            }
        }

        return results;
    }

    // =========================================================================
    // Cache rebuild logic
    // =========================================================================

    void EpisodicRetriever::maybe_rebuild() {
        bool should_rebuild = false;

        {
            std::shared_lock<std::shared_mutex> lock(mutex_);

            switch (rebuild_strategy_) {
            case CacheRebuildStrategy::ON_DEMAND:
                should_rebuild = (episodes_since_rebuild_ >=
                    config_.retriever.cache_rebuild_threshold);
                break;

            case CacheRebuildStrategy::PERIODIC: {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_rebuild_time_).count();
                should_rebuild = (elapsed >=
                    config_.retriever.cache_rebuild_interval_seconds);
                break;
            }

            case CacheRebuildStrategy::EXPLICIT:
                // Never auto-rebuild
                should_rebuild = false;
                break;
            }
        }

        if (should_rebuild) {
            LOG_DEBUG("EpisodicRetriever: auto-rebuild triggered (strategy=" +
                config_.retriever.cache_rebuild_strategy + ")");
            rebuild_index();
        }
    }

    // =========================================================================
    // RetrievalResult helpers
    // =========================================================================

    std::string RetrievalResult::score_summary() const {
        std::ostringstream oss;
        oss << "score=" << score
            << " kw=" << keyword_score
            << " sem=" << semantic_score;
        return oss.str();
    }

    // =========================================================================
    // Enum parsers
    // =========================================================================

    RetrievalMode EpisodicRetriever::parse_mode(const std::string& s) {
        if (s == "keyword")  return RetrievalMode::KEYWORD;
        if (s == "semantic") return RetrievalMode::SEMANTIC;
        if (s == "hybrid")   return RetrievalMode::HYBRID;
        // Default to hybrid -- safest choice
        LOG_WARN("EpisodicRetriever: unknown mode '" + s +
            "' -- defaulting to hybrid");
        return RetrievalMode::HYBRID;
    }

    CacheRebuildStrategy EpisodicRetriever::parse_strategy(const std::string& s) {
        if (s == "on_demand") return CacheRebuildStrategy::ON_DEMAND;
        if (s == "periodic")  return CacheRebuildStrategy::PERIODIC;
        if (s == "explicit")  return CacheRebuildStrategy::EXPLICIT;
        LOG_WARN("EpisodicRetriever: unknown cache strategy '" + s +
            "' -- defaulting to on_demand");
        return CacheRebuildStrategy::ON_DEMAND;
    }

} // namespace cardinal