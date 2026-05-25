// =============================================================================
// Cardinal - Agent Context Implementation
// File: src/agent/agent_context.cpp
// =============================================================================

#include "agent/agent_context.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using json = nlohmann::json;

namespace cardinal {

    AgentContext::AgentContext(const CardinalConfig& config,
                               const std::string& session_id,
                               const std::string& task_id)
        : config_(config)
        , session_id_(session_id)
        , task_id_(task_id)
    {
        // Sanitize task_id for use in filename
        std::string safe_id = task_id_.substr(0, 16);
        std::replace(safe_id.begin(), safe_id.end(), '/', '_');
        std::replace(safe_id.begin(), safe_id.end(), '\\', '_');

        db_path_ = config_.agent.working_memory_path + "." +
                   session_id_.substr(0, 8) + "." + safe_id + ".db";
    }

    AgentContext::~AgentContext() {
        close();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void AgentContext::open() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto parent = std::filesystem::path(db_path_).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        int rc = sqlite3_open(db_path_.c_str(), &db_);
        if (rc != SQLITE_OK) {
            throw std::runtime_error(
                "AgentContext: failed to open: " + db_path_);
        }

        sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
        create_schema();
        LOG_DEBUG("AgentContext: opened " + db_path_);
    }

    void AgentContext::close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    void AgentContext::create_schema() {
        const char* sql = R"(
            CREATE TABLE IF NOT EXISTS working_memory (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                entry_type TEXT DEFAULT 'observation',
                step_index TEXT DEFAULT '',
                timestamp  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS step_history (
                step_index  INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                rationale   TEXT,
                status      TEXT NOT NULL,
                result_summary TEXT,
                used_tool   INTEGER DEFAULT 0,
                tool_used   TEXT,
                duration_ms INTEGER DEFAULT 0,
                retries     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS plan (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                goal        TEXT NOT NULL,
                plan_json   TEXT NOT NULL
            );
        )";

        char* err = nullptr;
        sqlite3_exec(db_, sql, nullptr, nullptr, &err);
        if (err) sqlite3_free(err);
    }

    // =========================================================================
    // Working memory
    // =========================================================================

    void AgentContext::store(const std::string& key,
                              const std::string& value,
                              const std::string& entry_type,
                              const std::string& step_index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return;

        const char* sql =
            "INSERT OR REPLACE INTO working_memory "
            "(key, value, entry_type, step_index, timestamp) "
            "VALUES (?, ?, ?, ?, ?);";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, key.c_str(),        -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, value.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, entry_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, step_index.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5,
            JsonParser::current_timestamp().c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    std::string AgentContext::retrieve(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return "";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT value FROM working_memory WHERE key = ?;",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

        std::string result;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        sqlite3_finalize(stmt);
        return result;
    }

    bool AgentContext::has(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return false;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT COUNT(*) FROM working_memory WHERE key = ?;",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

        bool found = false;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            found = sqlite3_column_int(stmt, 0) > 0;
        sqlite3_finalize(stmt);
        return found;
    }

    void AgentContext::remove(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "DELETE FROM working_memory WHERE key = ?;",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    void AgentContext::clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return;
        sqlite3_exec(db_, "DELETE FROM working_memory;", nullptr, nullptr, nullptr);
    }

    std::vector<WorkingMemoryEntry> AgentContext::get_all() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<WorkingMemoryEntry> entries;
        if (!db_) return entries;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT key, value, entry_type, step_index, timestamp "
            "FROM working_memory ORDER BY timestamp ASC;",
            -1, &stmt, nullptr);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            WorkingMemoryEntry e;
            e.key        = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            e.value      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            e.entry_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            e.step_index = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            e.timestamp  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            entries.push_back(std::move(e));
        }
        sqlite3_finalize(stmt);
        return entries;
    }

    std::vector<WorkingMemoryEntry> AgentContext::get_for_step(
        const std::string& step_index) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<WorkingMemoryEntry> entries;
        if (!db_) return entries;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT key, value, entry_type, step_index, timestamp "
            "FROM working_memory WHERE step_index = ?;",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, step_index.c_str(), -1, SQLITE_TRANSIENT);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            WorkingMemoryEntry e;
            e.key        = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            e.value      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            e.entry_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            e.step_index = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            e.timestamp  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            entries.push_back(std::move(e));
        }
        sqlite3_finalize(stmt);
        return entries;
    }

    // =========================================================================
    // Step history
    // =========================================================================

    void AgentContext::record_step(const AgentStep& step) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return;

        const char* sql =
            "INSERT OR REPLACE INTO step_history "
            "(step_index, description, rationale, status, result_summary, "
            " used_tool, tool_used, duration_ms, retries) "
            "VALUES (?,?,?,?,?,?,?,?,?);";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int (stmt, 1, step.index);
        sqlite3_bind_text(stmt, 2, step.description.c_str(),    -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, step.rationale.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4,
            step_status_to_string(step.status).c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, step.result_summary.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 6, step.used_tool ? 1 : 0);
        sqlite3_bind_text(stmt, 7, step.tool_used.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 8, step.duration_ms);
        sqlite3_bind_int (stmt, 9, step.retries);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    std::vector<AgentStep> AgentContext::get_all_steps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<AgentStep> steps;
        if (!db_) return steps;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT step_index, description, rationale, status, "
            "result_summary, used_tool, tool_used, duration_ms, retries "
            "FROM step_history ORDER BY step_index ASC;",
            -1, &stmt, nullptr);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            AgentStep s;
            s.index          = sqlite3_column_int(stmt, 0);
            s.description    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            s.rationale      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            // status string → enum (simplified)
            std::string st   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            if (st == "completed")   s.status = AgentStepStatus::COMPLETED;
            else if (st == "failed") s.status = AgentStepStatus::FAILED;
            else if (st == "skipped") s.status = AgentStepStatus::SKIPPED;
            else s.status = AgentStepStatus::PENDING;
            s.result_summary = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            s.used_tool      = sqlite3_column_int(stmt, 5) != 0;
            s.tool_used      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
            s.duration_ms    = sqlite3_column_int(stmt, 7);
            s.retries        = sqlite3_column_int(stmt, 8);
            steps.push_back(std::move(s));
        }
        sqlite3_finalize(stmt);
        return steps;
    }

    int AgentContext::step_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return 0;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT COUNT(*) FROM step_history;", -1, &stmt, nullptr);
        int count = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            count = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        return count;
    }

    // =========================================================================
    // Plan persistence
    // =========================================================================

    void AgentContext::save_plan(const AgentPlan& plan) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return;

        json j;
        j["goal"]       = plan.goal;
        j["strategy"]   = plan.strategy;
        j["confidence"] = plan.confidence;
        j["valid"]      = plan.valid;

        json steps = json::array();
        for (const auto& s : plan.steps) {
            json sj;
            sj["index"]       = s.index;
            sj["description"] = s.description;
            sj["rationale"]   = s.rationale;
            sj["status"]      = step_status_to_string(s.status);
            steps.push_back(sj);
        }
        j["steps"] = steps;

        const char* sql =
            "INSERT OR REPLACE INTO plan (id, goal, plan_json) VALUES (1, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, plan.goal.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, j.dump().c_str(),  -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    AgentPlan AgentContext::load_plan() const {
        std::lock_guard<std::mutex> lock(mutex_);
        AgentPlan plan;
        if (!db_) return plan;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT plan_json FROM plan WHERE id = 1;", -1, &stmt, nullptr);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string plan_json =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            try {
                auto j         = json::parse(plan_json);
                plan.goal      = j["goal"].get<std::string>();
                plan.strategy  = j.value("strategy", "");
                plan.confidence = j.value("confidence", 0.0f);
                plan.valid     = j.value("valid", false);

                for (const auto& sj : j["steps"]) {
                    AgentStep s;
                    s.index       = sj["index"].get<int>();
                    s.description = sj["description"].get<std::string>();
                    s.rationale   = sj.value("rationale", "");
                    plan.steps.push_back(std::move(s));
                }
            } catch (...) {}
        }
        sqlite3_finalize(stmt);
        return plan;
    }

    bool AgentContext::has_plan() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return false;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT COUNT(*) FROM plan WHERE id = 1;", -1, &stmt, nullptr);
        bool found = false;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            found = sqlite3_column_int(stmt, 0) > 0;
        sqlite3_finalize(stmt);
        return found;
    }

    // =========================================================================
    // build_context_summary
    // =========================================================================

    std::string AgentContext::build_context_summary() const {
        std::ostringstream oss;

        auto steps = get_all_steps();
        auto memory = get_all();

        if (!steps.empty()) {
            oss << "## Completed Steps\n";
            for (const auto& s : steps) {
                if (s.status != AgentStepStatus::COMPLETED) continue;
                oss << (s.index + 1) << ". " << s.description << "\n";
                if (!s.result_summary.empty())
                    oss << "   Result: " << s.result_summary.substr(
                        0, std::min(s.result_summary.size(), size_t(200))) << "\n";
            }
            oss << "\n";
        }

        // Include recent working memory entries
        size_t max_entries = std::min(memory.size(),
            static_cast<size_t>(config_.agent.working_memory_size));
        if (!memory.empty()) {
            oss << "## Working Memory\n";
            size_t start = memory.size() > max_entries
                         ? memory.size() - max_entries : 0;
            for (size_t i = start; i < memory.size(); ++i) {
                const auto& e = memory[i];
                oss << "- [" << e.entry_type << "] "
                    << e.key << ": "
                    << e.value.substr(0, std::min(e.value.size(), size_t(200)))
                    << "\n";
            }
        }

        return oss.str();
    }

    // =========================================================================
    // destroy
    // =========================================================================

    void AgentContext::destroy() {
        close();
        std::filesystem::remove(db_path_);
        LOG_DEBUG("AgentContext: destroyed " + db_path_);
    }

} // namespace cardinal
