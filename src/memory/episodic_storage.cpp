// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Episodic Storage Implementation
// File: src/memory/episodic_storage.cpp
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "memory/episodic_storage.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

using json = nlohmann::json;

namespace cardinal {

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    EpisodicStorage::EpisodicStorage(const CardinalConfig& config)
        : config_(config)
        , db_(nullptr)
    {
        // Derive paths from config.
        // We store the SQLite DB alongside the existing memory files.
        // JSONL path comes from config -- same file EpisodicMemory writes to.

        std::filesystem::path memory_dir =
            std::filesystem::path(config_.memory.rule_store_path).parent_path();

        db_path_ = (memory_dir / "episodes.db").string();
        jsonl_path_ = config_.memory.episodic_log_path;
    }

    EpisodicStorage::~EpisodicStorage() {
        if (db_) {
            close();
        }
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void EpisodicStorage::open() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (db_) {
            LOG_WARN("EpisodicStorage::open() called but database already open");
            return;
        }

        // Ensure parent directory exists
        std::filesystem::path p(db_path_);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        int rc = sqlite3_open_v2(
            db_path_.c_str(),
            &db_,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            nullptr
        );

        if (rc != SQLITE_OK) {
            std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
            sqlite3_close(db_);
            db_ = nullptr;
            throw EpisodicStorageError("Failed to open database at '" +
                db_path_ + "': " + err);
        }

        // WAL mode -- better concurrent read performance, safer on crash
        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA synchronous=NORMAL");
        exec("PRAGMA foreign_keys=ON");

        create_schema();
        create_fts_triggers();

        LOG_INFO("EpisodicStorage opened: " + db_path_);

        // Run migration outside the constructor lock -- we already hold mutex_
        // here so call internal version directly
        if (!migration_complete()) {
            // Release lock before calling migrate (which re-acquires it)
            // Actually we are already locked -- call internal migration directly
            int migrated = 0;
            if (!std::filesystem::exists(jsonl_path_)) {
                LOG_INFO("No JSONL file found at '" + jsonl_path_ +
                    "' -- skipping migration");
            }
            else {
                std::ifstream file(jsonl_path_);
                if (!file.is_open()) {
                    LOG_WARN("Could not open JSONL for migration: " + jsonl_path_);
                }
                else {
                    std::string line;
                    int skipped = 0;
                    exec("BEGIN TRANSACTION");
                    while (std::getline(file, line)) {
                        if (line.empty()) continue;
                        auto record = parse_jsonl_line(line);
                        if (!record) {
                            ++skipped;
                            continue;
                        }
                        if (insert_episode_internal(*record)) {
                            ++migrated;
                        }
                    }
                    exec("COMMIT");

                    if (skipped > 0) {
                        LOG_WARN("Migration: skipped " + std::to_string(skipped) +
                            " malformed JSONL lines");
                    }
                }
            }

            set_metadata("migration_v1_complete", "1");
            set_metadata("migration_v1_count",
                std::to_string(migrated));

            LOG_INFO("JSONL migration complete: " +
                std::to_string(migrated) + " episodes imported");
        }
        else {
            LOG_DEBUG("EpisodicStorage: migration already complete, skipping");
        }
    }

    void EpisodicStorage::close() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!db_) return;

        // Checkpoint WAL before closing
        sqlite3_wal_checkpoint_v2(db_, nullptr, SQLITE_CHECKPOINT_TRUNCATE,
            nullptr, nullptr);
        sqlite3_close(db_);
        db_ = nullptr;
        LOG_INFO("EpisodicStorage closed");
    }

    // =========================================================================
    // Schema
    // =========================================================================

    void EpisodicStorage::create_schema() {
        // Main episodes table
        exec(R"sql(
            CREATE TABLE IF NOT EXISTS episodes (
                id                TEXT PRIMARY KEY,
                timestamp         TEXT NOT NULL,
                user_message      TEXT NOT NULL,
                response_summary  TEXT NOT NULL,
                confidence        REAL NOT NULL DEFAULT 0.0,
                reasoning_type    TEXT NOT NULL DEFAULT '',
                reasoning_domain  TEXT NOT NULL DEFAULT '',
                contradiction     INTEGER NOT NULL DEFAULT 0,
                uncertainty       INTEGER NOT NULL DEFAULT 0,
                rule_candidate    INTEGER NOT NULL DEFAULT 0,
                extracted_rule_id TEXT NOT NULL DEFAULT '',
                pass1_tokens      INTEGER NOT NULL DEFAULT 0,
                pass2_tokens      INTEGER NOT NULL DEFAULT 0,
                total_ms          INTEGER NOT NULL DEFAULT 0
            )
        )sql");

        // Indexes for common query patterns
        exec("CREATE INDEX IF NOT EXISTS idx_episodes_domain "
            "ON episodes(reasoning_domain)");
        exec("CREATE INDEX IF NOT EXISTS idx_episodes_confidence "
            "ON episodes(confidence)");
        exec("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp "
            "ON episodes(timestamp)");
        exec("CREATE INDEX IF NOT EXISTS idx_episodes_rule_candidate "
            "ON episodes(rule_candidate)");

        // FTS5 virtual table for keyword search
        // content= makes it a content table (reads from episodes)
        // content_rowid= links FTS rowid to episodes rowid
        exec(R"sql(
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
            USING fts5(
                id          UNINDEXED,
                user_message,
                response_summary,
                content='episodes',
                content_rowid='rowid'
            )
        )sql");

        // Metadata table -- migration flags, db version, etc.
        exec(R"sql(
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        )sql");

        LOG_DEBUG("EpisodicStorage schema ready");
    }

    void EpisodicStorage::create_fts_triggers() {
        // Keep FTS index in sync with episodes table automatically.
        // These triggers fire on INSERT, UPDATE, DELETE.

        exec(R"sql(
            CREATE TRIGGER IF NOT EXISTS episodes_fts_insert
            AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, id, user_message, response_summary)
                VALUES (new.rowid, new.id, new.user_message, new.response_summary);
            END
        )sql");

        exec(R"sql(
            CREATE TRIGGER IF NOT EXISTS episodes_fts_delete
            AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, id, user_message, response_summary)
                VALUES ('delete', old.rowid, old.id, old.user_message, old.response_summary);
            END
        )sql");

        exec(R"sql(
            CREATE TRIGGER IF NOT EXISTS episodes_fts_update
            AFTER UPDATE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, id, user_message, response_summary)
                VALUES ('delete', old.rowid, old.id, old.user_message, old.response_summary);
                INSERT INTO episodes_fts(rowid, id, user_message, response_summary)
                VALUES (new.rowid, new.id, new.user_message, new.response_summary);
            END
        )sql");

        LOG_DEBUG("EpisodicStorage FTS triggers ready");
    }

    // =========================================================================
    // Write
    // =========================================================================

    bool EpisodicStorage::insert_episode(const EpisodeRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        return insert_episode_internal(record);
    }

    // Internal version -- called from open() which already holds the lock
    bool EpisodicStorage::insert_episode_internal(const EpisodeRecord& record) {
        if (!db_) {
            throw EpisodicStorageError("insert_episode called on closed database");
        }

        const char* sql = R"sql(
            INSERT OR IGNORE INTO episodes (
                id, timestamp, user_message, response_summary,
                confidence, reasoning_type, reasoning_domain,
                contradiction, uncertainty, rule_candidate,
                extracted_rule_id, pass1_tokens, pass2_tokens, total_ms
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?
            )
        )sql";

        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            throw EpisodicStorageError(
                std::string("Failed to prepare insert: ") + sqlite3_errmsg(db_));
        }

        // Bind all fields in order
        sqlite3_bind_text(stmt, 1, record.id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.timestamp.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.user_message.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.response_summary.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 5, static_cast<double>(record.confidence));
        sqlite3_bind_text(stmt, 6, record.reasoning_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 7, record.reasoning_domain.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 8, record.contradiction ? 1 : 0);
        sqlite3_bind_int(stmt, 9, record.uncertainty ? 1 : 0);
        sqlite3_bind_int(stmt, 10, record.rule_candidate ? 1 : 0);
        sqlite3_bind_text(stmt, 11, record.extracted_rule_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 12, record.pass1_tokens);
        sqlite3_bind_int(stmt, 13, record.pass2_tokens);
        sqlite3_bind_int(stmt, 14, record.total_ms);

        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if (rc == SQLITE_DONE) {
            // SQLITE_DONE = row inserted or ignored (INSERT OR IGNORE)
            // sqlite3_changes() tells us which it was
            return sqlite3_changes(db_) > 0;
        }

        throw EpisodicStorageError(
            std::string("insert_episode failed: ") + sqlite3_errmsg(db_));
    }

    bool EpisodicStorage::set_extracted_rule_id(const std::string& episode_id,
        const std::string& rule_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return false;

        const char* sql =
            "UPDATE episodes SET extracted_rule_id = ? WHERE id = ?";

        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, rule_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, episode_id.c_str(), -1, SQLITE_TRANSIENT);

        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        return (rc == SQLITE_DONE && sqlite3_changes(db_) > 0);
    }

    // =========================================================================
    // Read
    // =========================================================================

    std::optional<EpisodeRecord>
        EpisodicStorage::get_episode(const std::string& id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return std::nullopt;

        const char* sql =
            "SELECT * FROM episodes WHERE id = ? LIMIT 1";

        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) return std::nullopt;

        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);

        std::optional<EpisodeRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            result = row_to_record(stmt);
        }

        sqlite3_finalize(stmt);
        return result;
    }

    std::vector<EpisodeRecord>
        EpisodicStorage::query(const EpisodeQuery& q) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return {};

        std::vector<EpisodeRecord> results;

        // If keyword is provided, use FTS5 then filter
        // Otherwise use direct table scan with indexes
        if (!q.keyword.empty()) {
            // FTS5 path -- join back to episodes for full row data
            std::ostringstream sql;
            sql << "SELECT e.* FROM episodes e "
                << "JOIN episodes_fts f ON e.rowid = f.rowid "
                << "WHERE episodes_fts MATCH ? ";

            if (!q.domain.empty()) {
                sql << "AND e.reasoning_domain = ? ";
            }
            if (q.min_confidence > 0.0f) {
                sql << "AND e.confidence >= ? ";
            }

            sql << "ORDER BY e.timestamp "
                << (q.recent_first ? "DESC" : "ASC")
                << " LIMIT ?";

            sqlite3_stmt* stmt = nullptr;
            int rc = sqlite3_prepare_v2(db_, sql.str().c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK) {
                LOG_WARN("EpisodicStorage::query FTS prepare failed: " +
                    std::string(sqlite3_errmsg(db_)));
                return results;
            }

            int bind_idx = 1;
            sqlite3_bind_text(stmt, bind_idx++, q.keyword.c_str(), -1, SQLITE_TRANSIENT);
            if (!q.domain.empty()) {
                sqlite3_bind_text(stmt, bind_idx++, q.domain.c_str(), -1, SQLITE_TRANSIENT);
            }
            if (q.min_confidence > 0.0f) {
                sqlite3_bind_double(stmt, bind_idx++,
                    static_cast<double>(q.min_confidence));
            }
            sqlite3_bind_int(stmt, bind_idx, q.max_results);

            while (sqlite3_step(stmt) == SQLITE_ROW) {
                results.push_back(row_to_record(stmt));
            }
            sqlite3_finalize(stmt);

        }
        else {
            // Direct scan path
            std::ostringstream sql;
            sql << "SELECT * FROM episodes WHERE 1=1 ";

            if (!q.domain.empty()) {
                sql << "AND reasoning_domain = ? ";
            }
            if (q.min_confidence > 0.0f) {
                sql << "AND confidence >= ? ";
            }

            sql << "ORDER BY timestamp "
                << (q.recent_first ? "DESC" : "ASC")
                << " LIMIT ?";

            sqlite3_stmt* stmt = nullptr;
            int rc = sqlite3_prepare_v2(db_, sql.str().c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK) {
                LOG_WARN("EpisodicStorage::query direct prepare failed: " +
                    std::string(sqlite3_errmsg(db_)));
                return results;
            }

            int bind_idx = 1;
            if (!q.domain.empty()) {
                sqlite3_bind_text(stmt, bind_idx++, q.domain.c_str(), -1, SQLITE_TRANSIENT);
            }
            if (q.min_confidence > 0.0f) {
                sqlite3_bind_double(stmt, bind_idx++,
                    static_cast<double>(q.min_confidence));
            }
            sqlite3_bind_int(stmt, bind_idx, q.max_results);

            while (sqlite3_step(stmt) == SQLITE_ROW) {
                results.push_back(row_to_record(stmt));
            }
            sqlite3_finalize(stmt);
        }

        return results;
    }

    std::vector<EpisodeRecord>
        EpisodicStorage::get_recent(int n) const {
        EpisodeQuery q;
        q.max_results = n;
        q.recent_first = true;
        return query(q);
    }

    std::vector<EpisodeRecord>
        EpisodicStorage::get_by_domain(const std::string& domain,
            int max_results) const {
        EpisodeQuery q;
        q.domain = domain;
        q.max_results = max_results;
        return query(q);
    }

    std::vector<EpisodeRecord>
        EpisodicStorage::get_high_confidence(float threshold, int max_results) const {
        EpisodeQuery q;
        q.min_confidence = threshold;
        q.max_results = max_results;
        return query(q);
    }

    int EpisodicStorage::count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return 0;

        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db_,
            "SELECT COUNT(*) FROM episodes", -1, &stmt, nullptr);
        if (rc != SQLITE_OK) return 0;

        int result = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            result = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
        return result;
    }

    // =========================================================================
    // Migration
    // =========================================================================

    int EpisodicStorage::migrate_from_jsonl(const std::string& jsonl_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (migration_complete()) {
            LOG_DEBUG("EpisodicStorage: migration already complete, skipping");
            return 0;
        }

        if (!std::filesystem::exists(jsonl_path)) {
            LOG_INFO("No JSONL file at '" + jsonl_path + "' -- nothing to migrate");
            set_metadata("migration_v1_complete", "1");
            set_metadata("migration_v1_count", "0");
            return 0;
        }

        std::ifstream file(jsonl_path);
        if (!file.is_open()) {
            LOG_WARN("Could not open JSONL for migration: " + jsonl_path);
            return 0;
        }

        int migrated = 0;
        int skipped = 0;
        std::string line;

        exec("BEGIN TRANSACTION");
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto record = parse_jsonl_line(line);
            if (!record) {
                ++skipped;
                continue;
            }
            if (insert_episode_internal(*record)) {
                ++migrated;
            }
        }
        exec("COMMIT");

        set_metadata("migration_v1_complete", "1");
        set_metadata("migration_v1_count", std::to_string(migrated));

        LOG_INFO("JSONL migration complete: " + std::to_string(migrated) +
            " episodes imported, " + std::to_string(skipped) + " skipped");
        return migrated;
    }

    bool EpisodicStorage::migration_complete() const {
        return get_metadata("migration_v1_complete") == "1";
    }

    // =========================================================================
    // Stats
    // =========================================================================

    EpisodeStats EpisodicStorage::stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return {};

        EpisodeStats s;

        // Total
        {
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(db_,
                "SELECT COUNT(*) FROM episodes",
                -1, &stmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(stmt) == SQLITE_ROW)
                    s.total_episodes = sqlite3_column_int(stmt, 0);
                sqlite3_finalize(stmt);
            }
        }

        // High confidence
        {
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(db_,
                "SELECT COUNT(*) FROM episodes WHERE confidence >= 0.7",
                -1, &stmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(stmt) == SQLITE_ROW)
                    s.high_conf_episodes = sqlite3_column_int(stmt, 0);
                sqlite3_finalize(stmt);
            }
        }

        // Rule candidates
        {
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(db_,
                "SELECT COUNT(*) FROM episodes WHERE rule_candidate = 1",
                -1, &stmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(stmt) == SQLITE_ROW)
                    s.rule_candidate_count = sqlite3_column_int(stmt, 0);
                sqlite3_finalize(stmt);
            }
        }

        // Average confidence
        {
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(db_,
                "SELECT AVG(confidence) FROM episodes",
                -1, &stmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(stmt) == SQLITE_ROW)
                    s.avg_confidence =
                    static_cast<float>(sqlite3_column_double(stmt, 0));
                sqlite3_finalize(stmt);
            }
        }

        // Migrated count
        {
            std::string val = get_metadata("migration_v1_count", "0");
            try { s.migrated_episodes = std::stoi(val); }
            catch (...) { s.migrated_episodes = 0; }
        }

        return s;
    }

    // =========================================================================
    // Metadata helpers
    // =========================================================================

    void EpisodicStorage::set_metadata(const std::string& key,
        const std::string& value) {
        if (!db_) return;

        const char* sql =
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;

        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    std::string EpisodicStorage::get_metadata(const std::string& key,
        const std::string& default_val) const {
        if (!db_) return default_val;

        const char* sql =
            "SELECT value FROM metadata WHERE key = ? LIMIT 1";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return default_val;

        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

        std::string result = default_val;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* val =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (val) result = val;
        }
        sqlite3_finalize(stmt);
        return result;
    }

    // =========================================================================
    // SQL exec helper
    // =========================================================================

    void EpisodicStorage::exec(const std::string& sql) const {
        if (!db_) {
            throw EpisodicStorageError("exec called on closed database");
        }
        char* errmsg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            std::string msg = errmsg ? errmsg : "unknown error";
            sqlite3_free(errmsg);
            throw EpisodicStorageError("SQL error: " + msg + "\nSQL: " + sql);
        }
    }

    // =========================================================================
    // Row mapping
    // =========================================================================

    EpisodeRecord EpisodicStorage::row_to_record(sqlite3_stmt* stmt) {
        EpisodeRecord r;

        // Column order must match SELECT * FROM episodes schema order:
        // 0:id, 1:timestamp, 2:user_message, 3:response_summary,
        // 4:confidence, 5:reasoning_type, 6:reasoning_domain,
        // 7:contradiction, 8:uncertainty, 9:rule_candidate,
        // 10:extracted_rule_id, 11:pass1_tokens, 12:pass2_tokens, 13:total_ms

        auto get_text = [&](int col) -> std::string {
            const char* v =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, col));
            return v ? v : "";
            };

        r.id = get_text(0);
        r.timestamp = get_text(1);
        r.user_message = get_text(2);
        r.response_summary = get_text(3);
        r.confidence = static_cast<float>(sqlite3_column_double(stmt, 4));
        r.reasoning_type = get_text(5);
        r.reasoning_domain = get_text(6);
        r.contradiction = sqlite3_column_int(stmt, 7) != 0;
        r.uncertainty = sqlite3_column_int(stmt, 8) != 0;
        r.rule_candidate = sqlite3_column_int(stmt, 9) != 0;
        r.extracted_rule_id = get_text(10);
        r.pass1_tokens = sqlite3_column_int(stmt, 11);
        r.pass2_tokens = sqlite3_column_int(stmt, 12);
        r.total_ms = sqlite3_column_int(stmt, 13);

        return r;
    }

    // =========================================================================
    // JSONL line parser
    // =========================================================================

    std::optional<EpisodeRecord>
        EpisodicStorage::parse_jsonl_line(const std::string& line) {
        json j;
        try {
            j = json::parse(line);
        }
        catch (...) {
            LOG_WARN("EpisodicStorage: failed to parse JSONL line: " + line);
            return std::nullopt;
        }

        // id is required -- skip lines without it
        if (!j.contains("id") || !j["id"].is_string()) {
            LOG_WARN("EpisodicStorage: JSONL line missing 'id' field, skipping");
            return std::nullopt;
        }

        EpisodeRecord r;

        auto get_str = [&](const char* key, const char* def = "") -> std::string {
            return (j.contains(key) && j[key].is_string())
                ? j[key].get<std::string>() : def;
            };
        auto get_bool_field = [&](const char* key) -> bool {
            return (j.contains(key) && j[key].is_boolean())
                ? j[key].get<bool>() : false;
            };
        auto get_int_field = [&](const char* key) -> int {
            return (j.contains(key) && j[key].is_number_integer())
                ? j[key].get<int>() : 0;
            };
        auto get_float_field = [&](const char* key) -> float {
            return (j.contains(key) && j[key].is_number())
                ? j[key].get<float>() : 0.0f;
            };

        r.id = get_str("id");
        r.timestamp = get_str("timestamp");
        r.user_message = get_str("user_message");
        r.response_summary = get_str("response_summary");
        r.confidence = get_float_field("confidence");
        r.reasoning_type = get_str("reasoning_type");
        r.reasoning_domain = get_str("reasoning_domain");
        r.contradiction = get_bool_field("contradiction");
        r.uncertainty = get_bool_field("uncertainty");
        r.rule_candidate = get_bool_field("rule_candidate");
        r.extracted_rule_id = get_str("extracted_rule_id");
        r.pass1_tokens = get_int_field("pass1_tokens");
        r.pass2_tokens = get_int_field("pass2_tokens");
        r.total_ms = get_int_field("total_ms");

        return r;
    }

} // namespace cardinal