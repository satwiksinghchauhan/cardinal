// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Episodic Storage
// File: src/memory/episodic_storage.h
//
// SQLite-backed persistent storage for inference episodes.
// Complements EpisodicMemory (JSONL audit trail) with a searchable index.
//
// Responsibilities:
//   - Persist every episode to SQLite with full schema
//   - FTS5 full-text search over user_message and response_summary
//   - One-time migration of existing JSONL episodes at startup
//   - Query by domain, confidence threshold, recency, keyword
//
// Dual-write pattern:
//   EpisodicMemory  -- append-only JSONL, audit trail, never modified
//   EpisodicStorage -- SQLite index, searchable, supports queries
//
// Both are written on every inference. EpisodicStorage is the query layer.
//
// Thread safety:
//   All public methods are protected by a mutex.
//   SQLite is opened in serialized mode (SQLITE_OPEN_FULLMUTEX).
// =============================================================================

#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "utils/json_parser.h"

#include <sqlite3.h>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <functional>

namespace cardinal {

    // -------------------------------------------------------------------------
    // EpisodeRecord
    // A single episode as stored in and retrieved from SQLite.
    // Field names and types match the JSONL schema exactly so migration
    // is a direct mapping with no data transformation.
    // -------------------------------------------------------------------------
    struct EpisodeRecord {
        std::string id;
        std::string timestamp;
        std::string user_message;
        std::string response_summary;
        float       confidence = 0.0f;
        std::string reasoning_type;
        std::string reasoning_domain;
        bool        contradiction = false;
        bool        uncertainty = false;
        bool        rule_candidate = false;
        std::string extracted_rule_id;  // Empty string if no rule extracted
        int         pass1_tokens = 0;
        int         pass2_tokens = 0;
        int         total_ms = 0;
    };

    // -------------------------------------------------------------------------
    // EpisodeQuery
    // Parameters for querying the episode store.
    // All filters are AND-combined. Empty/zero fields are ignored.
    // -------------------------------------------------------------------------
    struct EpisodeQuery {
        std::string keyword;            // FTS5 search over user_message + response
        std::string domain;             // Filter by reasoning_domain
        float       min_confidence = 0.0f;
        int         max_results = 20;
        bool        recent_first = true;  // ORDER BY timestamp DESC
    };

    // -------------------------------------------------------------------------
    // EpisodeStats
    // Summary statistics for monitoring and export decisions.
    // -------------------------------------------------------------------------
    struct EpisodeStats {
        int   total_episodes = 0;
        int   migrated_episodes = 0;  // Episodes from JSONL migration
        int   high_conf_episodes = 0;  // confidence >= 0.7
        int   rule_candidate_count = 0;
        float avg_confidence = 0.0f;
    };

    // -------------------------------------------------------------------------
    // EpisodicStorage
    // SQLite-backed persistent episode store with FTS5 keyword search.
    //
    // Usage:
    //   EpisodicStorage storage(config);
    //   storage.open();                          // Creates DB, runs migration
    //   storage.insert_episode(record);          // Called after every inference
    //   auto results = storage.search(query);    // Keyword/domain/confidence query
    //   storage.close();                         // Clean shutdown
    // -------------------------------------------------------------------------
    class EpisodicStorage {
    public:
        explicit EpisodicStorage(const CardinalConfig& config);
        ~EpisodicStorage();

        // Not copyable -- owns a SQLite handle
        EpisodicStorage(const EpisodicStorage&) = delete;
        EpisodicStorage& operator=(const EpisodicStorage&) = delete;

        // Movable
        EpisodicStorage(EpisodicStorage&&) = default;
        EpisodicStorage& operator=(EpisodicStorage&&) = default;

        // -- Lifecycle --

        // Open (or create) the SQLite database.
        // Creates schema if not present.
        // Runs JSONL migration if not yet completed.
        // Throws EpisodicStorageError on failure.
        void open();

        // Flush WAL and close the database connection cleanly.
        void close();

        bool is_open() const { return db_ != nullptr; }

        // -- Write --

        // Insert a single episode. Ignores duplicate IDs (INSERT OR IGNORE).
        // Called after every inference as part of dual-write.
        // Returns true on success, false if episode already exists.
        bool insert_episode(const EpisodeRecord& record);

        // Update the extracted_rule_id on an existing episode.
        // Called by ConsistencyChecker after a rule is committed.
        bool set_extracted_rule_id(const std::string& episode_id,
            const std::string& rule_id);

        // -- Read --

        // Get a single episode by ID.
        std::optional<EpisodeRecord> get_episode(const std::string& id) const;

        // Query episodes with filters. See EpisodeQuery for options.
        std::vector<EpisodeRecord> query(const EpisodeQuery& q) const;

        // Get the N most recent episodes regardless of other filters.
        std::vector<EpisodeRecord> get_recent(int n) const;

        // Get episodes by reasoning domain.
        std::vector<EpisodeRecord> get_by_domain(const std::string& domain,
            int max_results = 20) const;

        // Get episodes above a confidence threshold.
        std::vector<EpisodeRecord> get_high_confidence(float threshold = 0.7f,
            int max_results = 50) const;

        // Count total episodes in the database.
        int count() const;

        // -- Migration --

        // Migrate episodes from JSONL file into SQLite.
        // Safe to call multiple times -- checks metadata flag first.
        // Returns number of episodes migrated (0 if already done).
        int migrate_from_jsonl(const std::string& jsonl_path);

        bool migration_complete() const;

        // -- Stats --
        EpisodeStats stats() const;

    private:
        // -- Schema --
        void create_schema();
        void create_fts_triggers();

        // -- Metadata --
        void        set_metadata(const std::string& key, const std::string& value);
        std::string get_metadata(const std::string& key,
            const std::string& default_val = "") const;

        // -- Statement helpers --

        // Execute a SQL string with no result set (CREATE, INSERT, UPDATE, etc.)
        void exec(const std::string& sql) const;

        // Bind and execute an insert for a single EpisodeRecord.
        bool insert_episode_internal(const EpisodeRecord& record);

        // Map a sqlite3_stmt row to an EpisodeRecord.
        static EpisodeRecord row_to_record(sqlite3_stmt* stmt);

        // Parse a single JSONL line into an EpisodeRecord.
        // Returns nullopt on parse failure (line is skipped, not fatal).
        static std::optional<EpisodeRecord> parse_jsonl_line(const std::string& line);

        // -- Members --
        const CardinalConfig& config_;
        sqlite3* db_ = nullptr;
        mutable std::mutex    mutex_;

        // Cached path so we don't reconstruct on every call
        std::string           db_path_;
        std::string           jsonl_path_;
    };

    // -------------------------------------------------------------------------
    // EpisodicStorageError
    // -------------------------------------------------------------------------
    class EpisodicStorageError : public std::runtime_error {
    public:
        explicit EpisodicStorageError(const std::string& message)
            : std::runtime_error("EpisodicStorageError: " + message) {}
    };

} // namespace cardinal