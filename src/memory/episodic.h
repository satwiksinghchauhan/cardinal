// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Episodic Memory
// File: src/memory/episodic.h
// Logs every inference cycle as a timestamped episode.
// Episodic memory is Cardinal's ability to look back at its own reasoning,
// what it was asked, how it felt, what it concluded, what rules it flagged.
// Unlike rule_store (distilled knowledge), episodic is the raw experience log.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "utils/json_parser.h"

#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <optional>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // Episode
    // A single inference cycle record.
    // -----------------------------------------------------------------------------
    struct Episode {
        std::string  id;                   // Unique episode ID
        std::string  timestamp;            // ISO 8601 timestamp
        std::string  user_message;         // What the user asked
        std::string  response_summary;     // First 200 chars of response
        FeelingOutput feeling;             // Feeling output from Pass 1
        std::string  reasoning_domain;     // Domain from feeling output
        bool         rule_candidate;       // Was a rule candidate signaled?
        bool         contradiction;        // Was a contradiction flagged?
        bool         uncertainty;          // Was uncertainty flagged?
        float        confidence;           // Confidence score
        int          pass1_tokens;         // Tokens generated in Pass 1
        int          pass2_tokens;         // Tokens generated in Pass 2
        long long    total_ms;             // Total inference time in ms
        std::string  extracted_rule_id;    // ID of rule extracted (empty if none)
    };

    // -----------------------------------------------------------------------------
    // EpisodicStats
    // Summary of episodic memory contents.
    // -----------------------------------------------------------------------------
    struct EpisodicStats {
        int   total_episodes;
        int   rule_candidate_episodes;
        int   contradiction_episodes;
        int   uncertain_episodes;
        float avg_confidence;
        float avg_total_ms;
        int   total_tokens;
    };

    // -----------------------------------------------------------------------------
    // EpisodicMemory
    // Append-only log of inference episodes.
    // Writes to a plain text log file (human readable) and maintains an
    // in-memory index for fast querying of recent episodes.
    //
    // The log format is one JSON object per line (JSONL) for easy parsing.
    // -----------------------------------------------------------------------------
    class EpisodicMemory {
    public:
        explicit EpisodicMemory(const CardinalConfig& config);
        ~EpisodicMemory();

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        // Open log file - creates if not exists
        void open();

        // Flush and close log file
        void close();

        // -------------------------------------------------------------------------
        // Logging
        // -------------------------------------------------------------------------

        // Log a completed inference cycle
        // Returns the episode ID
        std::string log_episode(
            const std::string& user_message,
            const std::string& response,
            const FeelingOutput& feeling,
            int                  pass1_tokens,
            int                  pass2_tokens,
            long long            total_ms,
            const std::string& extracted_rule_id = "");

        // -------------------------------------------------------------------------
        // Query (in-memory index - recent episodes only)
        // -------------------------------------------------------------------------

        // Get N most recent episodes
        std::vector<Episode> get_recent(int n = 10) const;

        // Get episodes where rule_candidate was true
        std::vector<Episode> get_rule_candidates(int max = 20) const;

        // Get episodes where contradiction was flagged
        std::vector<Episode> get_contradictions(int max = 20) const;

        // Get episodes for a specific reasoning domain
        std::vector<Episode> get_by_domain(const std::string& domain,
            int max = 20) const;

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        EpisodicStats stats() const;
        int           episode_count() const;

    private:
        // -------------------------------------------------------------------------
        // Internal helpers
        // -------------------------------------------------------------------------
        Episode     build_episode(const std::string& user_message,
            const std::string& response,
            const FeelingOutput& feeling,
            int                  pass1_tokens,
            int                  pass2_tokens,
            long long            total_ms,
            const std::string& rule_id) const;

        std::string episode_to_jsonl(const Episode& ep) const;
        void        append_to_file(const std::string& line);

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        std::vector<Episode>     index_;       // In-memory index (recent episodes)
        std::ofstream            log_file_;
        mutable std::mutex       mutex_;
        bool                     open_ = false;
        static constexpr int     MAX_INDEX = 1000; // Max episodes in memory
    };

    // -----------------------------------------------------------------------------
    // EpisodicError
    // -----------------------------------------------------------------------------
    class EpisodicError : public std::runtime_error {
    public:
        explicit EpisodicError(const std::string& message)
            : std::runtime_error("EpisodicError: " + message) {}
    };

} // namespace cardinal