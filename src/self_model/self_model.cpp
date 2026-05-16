// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Self Model (Layer 1) — Implementation
// File: src/self_model/self_model.cpp
// =============================================================================

#include "self_model/self_model.h"
#include "utils/logger.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

std::string utc_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// RAII wrapper for sqlite3_stmt.
struct Stmt {
    sqlite3_stmt* s = nullptr;
    ~Stmt() { if (s) sqlite3_finalize(s); }
    Stmt(const Stmt&)            = delete;
    Stmt& operator=(const Stmt&) = delete;
    Stmt() = default;

    void prepare(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK) {
            throw std::runtime_error(
                std::string("SelfModel SQL prepare failed: ") +
                sqlite3_errmsg(db) + "  SQL: " + sql);
        }
    }
};

void exec_sql(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "(unknown)";
        sqlite3_free(err);
        throw std::runtime_error("SelfModel exec failed: " + msg);
    }
}

} // anonymous namespace

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

SelfModel::SelfModel(const CardinalConfig& config)
    : config_(config)
    , db_(nullptr)
{}

SelfModel::~SelfModel() {
    if (db_) close();
}

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------

void SelfModel::open() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (db_) {
        LOG_INFO("SelfModel: already open");
        return;
    }

    const std::string& db_path = config_.self_improvement.self_model.db_path;

    // Ensure parent directory exists using std::filesystem (no system() call).
    {
        fs::path p(db_path);
        if (p.has_parent_path()) {
            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);
            if (ec) {
                LOG_ERROR("SelfModel: could not create directory " +
                          p.parent_path().string() + ": " + ec.message());
            }
        }
    }

    int rc = sqlite3_open_v2(
        db_path.c_str(), &db_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        nullptr);

    if (rc != SQLITE_OK) {
        std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error(
            "SelfModel: failed to open DB at '" + db_path + "': " + err);
    }

    exec_sql(db_, "PRAGMA journal_mode=WAL;");
    exec_sql(db_, "PRAGMA synchronous=NORMAL;");
    exec_sql(db_, "PRAGMA foreign_keys=ON;");

    create_schema();
    LOG_INFO("SelfModel: opened database at " + db_path);
}

// ---------------------------------------------------------------------------
// close()
// ---------------------------------------------------------------------------

void SelfModel::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_) return;
    exec_sql(db_, "PRAGMA wal_checkpoint(TRUNCATE);");
    sqlite3_close(db_);
    db_ = nullptr;
    LOG_INFO("SelfModel: database closed");
}

// ---------------------------------------------------------------------------
// create_schema()
// ---------------------------------------------------------------------------

void SelfModel::create_schema() {
    // domain_stats — running accumulators, one row per domain.
    exec_sql(db_, R"sql(
        CREATE TABLE IF NOT EXISTS domain_stats (
            domain               TEXT    NOT NULL PRIMARY KEY,
            total_inferences     INTEGER NOT NULL DEFAULT 0,
            total_contradictions INTEGER NOT NULL DEFAULT 0,
            total_uncertainties  INTEGER NOT NULL DEFAULT 0,
            total_rule_commits   INTEGER NOT NULL DEFAULT 0,
            confidence_sum       REAL    NOT NULL DEFAULT 0.0,
            last_updated         TEXT    NOT NULL DEFAULT ''
        );
    )sql");

    // reasoning_stats — running accumulators, one row per reasoning type.
    exec_sql(db_, R"sql(
        CREATE TABLE IF NOT EXISTS reasoning_stats (
            reasoning_type      TEXT    NOT NULL PRIMARY KEY,
            usage_count         INTEGER NOT NULL DEFAULT 0,
            confidence_sum      REAL    NOT NULL DEFAULT 0.0,
            contradiction_count INTEGER NOT NULL DEFAULT 0,
            last_updated        TEXT    NOT NULL DEFAULT ''
        );
    )sql");

    LOG_DEBUG("SelfModel: schema ready");
}

// ---------------------------------------------------------------------------
// record_inference()
// ---------------------------------------------------------------------------

void SelfModel::record_inference(const std::string& domain,
                                  const std::string& reasoning_type,
                                  float              confidence,
                                  bool               contradiction,
                                  bool               uncertainty,
                                  bool               rule_committed) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        LOG_ERROR("SelfModel::record_inference called before open()");
        return;
    }

    confidence = std::clamp(confidence, 0.0f, 1.0f);

    update_domain_stats(domain, confidence, contradiction, uncertainty, rule_committed);
    update_reasoning_stats(reasoning_type, confidence, contradiction);
}

// ---------------------------------------------------------------------------
// update_domain_stats() — called with mutex_ held
// ---------------------------------------------------------------------------

void SelfModel::update_domain_stats(const std::string& domain,
                                     float confidence,
                                     bool  contradiction,
                                     bool  uncertainty,
                                     bool  rule_committed) {
    static const char* sql = R"sql(
        INSERT INTO domain_stats
            (domain, total_inferences, total_contradictions,
             total_uncertainties, total_rule_commits, confidence_sum, last_updated)
        VALUES (?, 1, ?, ?, ?, ?, ?)
        ON CONFLICT(domain) DO UPDATE SET
            total_inferences     = total_inferences     + 1,
            total_contradictions = total_contradictions + excluded.total_contradictions,
            total_uncertainties  = total_uncertainties  + excluded.total_uncertainties,
            total_rule_commits   = total_rule_commits   + excluded.total_rule_commits,
            confidence_sum       = confidence_sum       + excluded.confidence_sum,
            last_updated         = excluded.last_updated;
    )sql";

    Stmt stmt;
    stmt.prepare(db_, sql);

    std::string ts = utc_now();
    sqlite3_bind_text  (stmt.s, 1, domain.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int   (stmt.s, 2, contradiction  ? 1 : 0);
    sqlite3_bind_int   (stmt.s, 3, uncertainty    ? 1 : 0);
    sqlite3_bind_int   (stmt.s, 4, rule_committed ? 1 : 0);
    sqlite3_bind_double(stmt.s, 5, static_cast<double>(confidence));
    sqlite3_bind_text  (stmt.s, 6, ts.c_str(),    -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt.s) != SQLITE_DONE)
        LOG_ERROR("SelfModel: domain_stats upsert failed: " + std::string(sqlite3_errmsg(db_)));
}

// ---------------------------------------------------------------------------
// update_reasoning_stats() — called with mutex_ held
// ---------------------------------------------------------------------------

void SelfModel::update_reasoning_stats(const std::string& reasoning_type,
                                        float confidence,
                                        bool  contradiction) {
    static const char* sql = R"sql(
        INSERT INTO reasoning_stats
            (reasoning_type, usage_count, confidence_sum, contradiction_count, last_updated)
        VALUES (?, 1, ?, ?, ?)
        ON CONFLICT(reasoning_type) DO UPDATE SET
            usage_count         = usage_count         + 1,
            confidence_sum      = confidence_sum      + excluded.confidence_sum,
            contradiction_count = contradiction_count + excluded.contradiction_count,
            last_updated        = excluded.last_updated;
    )sql";

    Stmt stmt;
    stmt.prepare(db_, sql);

    std::string ts = utc_now();
    sqlite3_bind_text  (stmt.s, 1, reasoning_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt.s, 2, static_cast<double>(confidence));
    sqlite3_bind_int   (stmt.s, 3, contradiction ? 1 : 0);
    sqlite3_bind_text  (stmt.s, 4, ts.c_str(),             -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt.s) != SQLITE_DONE)
        LOG_ERROR("SelfModel: reasoning_stats upsert failed: " + std::string(sqlite3_errmsg(db_)));
}

// ---------------------------------------------------------------------------
// get_snapshot()
// ---------------------------------------------------------------------------

SelfModelSnapshot SelfModel::get_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SelfModelSnapshot snap;
    if (!db_) return snap;

    // ── domain_stats ─────────────────────────────────────────────────────────
    {
        static const char* sql = R"sql(
            SELECT domain, total_inferences, total_contradictions,
                   total_uncertainties, total_rule_commits,
                   confidence_sum, last_updated
            FROM domain_stats ORDER BY domain;
        )sql";

        Stmt stmt;
        stmt.prepare(db_, sql);

        while (sqlite3_step(stmt.s) == SQLITE_ROW) {
            DomainStats ds;
            ds.domain               = reinterpret_cast<const char*>(sqlite3_column_text(stmt.s, 0));
            ds.total_inferences     = sqlite3_column_int   (stmt.s, 1);
            ds.total_contradictions = sqlite3_column_int   (stmt.s, 2);
            ds.total_uncertainties  = sqlite3_column_int   (stmt.s, 3);
            int   rule_commits      = sqlite3_column_int   (stmt.s, 4);
            double conf_sum         = sqlite3_column_double(stmt.s, 5);
            ds.last_updated         = reinterpret_cast<const char*>(sqlite3_column_text(stmt.s, 6));

            if (ds.total_inferences > 0) {
                float n = static_cast<float>(ds.total_inferences);
                ds.avg_confidence     = static_cast<float>(conf_sum) / n;
                ds.contradiction_rate = static_cast<float>(ds.total_contradictions) / n;
                ds.uncertainty_rate   = static_cast<float>(ds.total_uncertainties)  / n;
                ds.rule_commit_rate   = static_cast<float>(rule_commits)             / n;
            }

            snap.domain_stats.push_back(std::move(ds));
        }
    }

    // ── reasoning_stats ───────────────────────────────────────────────────────
    {
        static const char* sql = R"sql(
            SELECT reasoning_type, usage_count, confidence_sum, contradiction_count
            FROM reasoning_stats ORDER BY usage_count DESC;
        )sql";

        Stmt stmt;
        stmt.prepare(db_, sql);

        while (sqlite3_step(stmt.s) == SQLITE_ROW) {
            ReasoningTypeStats rs;
            rs.reasoning_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt.s, 0));
            rs.usage_count    = sqlite3_column_int   (stmt.s, 1);
            double conf_sum   = sqlite3_column_double(stmt.s, 2);
            int    contra_cnt = sqlite3_column_int   (stmt.s, 3);  // raw int from DB

            if (rs.usage_count > 0) {
                float n = static_cast<float>(rs.usage_count);
                rs.avg_confidence     = static_cast<float>(conf_sum) / n;
                rs.contradiction_rate = static_cast<float>(contra_cnt) / n; // rate, not count
            }

            snap.reasoning_stats.push_back(std::move(rs));
        }
    }

    return snap;
}

// ---------------------------------------------------------------------------
// get_weakest_domains()
// ---------------------------------------------------------------------------

std::vector<DomainStats> SelfModel::get_weakest_domains(int n) const {
    auto snap = get_snapshot();

    std::sort(snap.domain_stats.begin(), snap.domain_stats.end(),
              [](const DomainStats& a, const DomainStats& b) {
                  return a.weakness_score() > b.weakness_score();
              });

    if (n > 0 && static_cast<int>(snap.domain_stats.size()) > n)
        snap.domain_stats.resize(static_cast<std::size_t>(n));

    return snap.domain_stats;
}

// ---------------------------------------------------------------------------
// get_all_domain_stats()
// ---------------------------------------------------------------------------

std::vector<DomainStats> SelfModel::get_all_domain_stats() const {
    return get_snapshot().domain_stats;
}

// ---------------------------------------------------------------------------
// format_for_prompt()
// ---------------------------------------------------------------------------

std::string SelfModel::format_for_prompt() const {
    const int max_chars = config_.self_improvement.self_model.prompt_max_chars;
    return get_snapshot().format_for_prompt(max_chars);
}

// ---------------------------------------------------------------------------
// total_records()
// ---------------------------------------------------------------------------

int SelfModel::total_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_) return 0;

    Stmt stmt;
    stmt.prepare(db_, "SELECT COALESCE(SUM(total_inferences), 0) FROM domain_stats;");
    if (sqlite3_step(stmt.s) == SQLITE_ROW)
        return sqlite3_column_int(stmt.s, 0);
    return 0;
}

// ---------------------------------------------------------------------------
// SelfModelSnapshot::format_for_prompt()
// Defined here (not in the header) to keep <sstream>/<iomanip> out of the
// types header.
// ---------------------------------------------------------------------------

std::string SelfModelSnapshot::format_for_prompt(int max_chars) const {
    if (domain_stats.empty()) return {};

    std::ostringstream oss;
    oss << "[Self-Model]\n";

    std::string weak   = weakest_domain();
    std::string strong = strongest_domain();
    if (!weak.empty())   oss << "Weakest domain:   " << weak   << "\n";
    if (!strong.empty()) oss << "Strongest domain: " << strong << "\n";

    oss << "Domain          Conf  Contradict  Uncertain  Inferences\n";
    for (const auto& d : domain_stats) {
        if (d.total_inferences == 0) continue;
        oss << std::left  << std::setw(16) << d.domain
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(4)  << d.avg_confidence      << "  "
            << std::setw(10) << d.contradiction_rate  << "  "
            << std::setw(9)  << d.uncertainty_rate    << "  "
            << std::setw(10) << d.total_inferences    << "\n";
    }

    if (!reasoning_stats.empty()) {
        oss << "Top reasoning: ";
        int shown = 0;
        for (const auto& r : reasoning_stats) {
            if (shown >= 3) break;
            oss << r.reasoning_type << "(" << r.usage_count << ") ";
            ++shown;
        }
        oss << "\n";
    }

    std::string result = oss.str();
    if (max_chars > 0 && static_cast<int>(result.size()) > max_chars) {
        result.resize(static_cast<std::size_t>(max_chars));
        auto nl = result.rfind('\n');
        if (nl != std::string::npos && nl > 0)
            result.resize(nl + 1);
    }
    return result;
}

} // namespace cardinal
