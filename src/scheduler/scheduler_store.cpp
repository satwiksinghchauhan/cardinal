// =============================================================================
// Cardinal - Scheduler Store Implementation
// File: src/scheduler/scheduler_store.cpp
// =============================================================================

#include "scheduler/scheduler_store.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <stdexcept>
#include <sstream>
#include <filesystem>

using json = nlohmann::json;

namespace cardinal {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void exec_sql(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown error";
        sqlite3_free(err);
        throw std::runtime_error("SchedulerStore SQL error: " + msg);
    }
}

// Utility: generate a random UUID-like string using SQLite's randomblob
static std::string new_uuid() {
    // Simple UUID v4 via hex encoding 16 random bytes
    static thread_local unsigned char buf[16];
    sqlite3_randomness(16, buf);
    char out[37];
    snprintf(out, sizeof(out),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[6],buf[7],
        buf[8],buf[9],buf[10],buf[11],buf[12],buf[13],buf[14],buf[15]);
    return std::string(out);
}

// ---------------------------------------------------------------------------
// TriggerSpec ↔ JSON
// ---------------------------------------------------------------------------

std::string SchedulerStore::trigger_to_json(const TriggerSpec& t) {
    json j;
    j["type"]             = static_cast<int>(t.type);
    j["cron_expression"]  = t.cron_expression;
    j["interval_seconds"] = t.interval_seconds;
    j["condition_expr"]   = t.condition_expr;
    j["idle_minutes"]     = t.idle_minutes;
    return j.dump();
}

TriggerSpec SchedulerStore::trigger_from_json(const std::string& s) {
    TriggerSpec t;
    try {
        auto j = json::parse(s);
        t.type             = static_cast<TriggerType>(j.value("type", 3));
        t.cron_expression  = j.value("cron_expression", "");
        t.interval_seconds = j.value("interval_seconds", 0);
        t.condition_expr   = j.value("condition_expr", "");
        t.idle_minutes     = j.value("idle_minutes", 30);
    } catch (...) {}
    return t;
}

// ---------------------------------------------------------------------------
// TaskAction ↔ JSON
// ---------------------------------------------------------------------------

std::string SchedulerStore::action_to_json(const TaskAction& a) {
    json j;
    j["type"]           = static_cast<int>(a.type);
    j["goal"]           = a.goal;
    j["max_iterations"] = a.max_iterations;
    j["stream"]         = a.stream;
    j["domain_hint"]    = a.domain_hint;
    j["shell_command"]  = a.shell_command;
    j["webhook_url"]    = a.webhook_url;
    j["output_target"]  = static_cast<int>(a.output_target);
    j["output_file"]    = a.output_file;
    return j.dump();
}

TaskAction SchedulerStore::action_from_json(const std::string& s) {
    TaskAction a;
    try {
        auto j = json::parse(s);
        a.type           = static_cast<TaskActionType>(j.value("type", 0));
        a.goal           = j.value("goal", "");
        a.max_iterations = j.value("max_iterations", 0);
        a.stream         = j.value("stream", false);
        a.domain_hint    = j.value("domain_hint", "");
        a.shell_command  = j.value("shell_command", "");
        a.webhook_url    = j.value("webhook_url", "");
        a.output_target  = static_cast<OutputTarget>(j.value("output_target", 0));
        a.output_file    = j.value("output_file", "");
    } catch (...) {}
    return a;
}

// ---------------------------------------------------------------------------
// Row readers
// ---------------------------------------------------------------------------

ScheduledTask SchedulerStore::row_to_task(sqlite3_stmt* stmt) {
    ScheduledTask t;
    int col = 0;
    auto str = [&](int c) -> std::string {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, c));
        return v ? v : "";
    };
    t.id              = str(col++);
    t.name            = str(col++);
    t.description     = str(col++);
    t.enabled         = sqlite3_column_int(stmt, col++) != 0;
    t.trigger         = trigger_from_json(str(col++));
    t.action          = action_from_json(str(col++));
    t.run_count       = sqlite3_column_int(stmt, col++);
    t.fail_count      = sqlite3_column_int(stmt, col++);
    t.last_run_at     = str(col++);
    t.next_run_at     = str(col++);
    t.created_at      = str(col++);
    t.updated_at      = str(col++);
    t.created_from    = str(col++);
    t.created_in_session = str(col++);

    // Safety overrides stored as nullable ints: -1 = nullopt, 0 = false, 1 = true
    auto opt_bool = [&](int c) -> std::optional<bool> {
        if (sqlite3_column_type(stmt, c) == SQLITE_NULL) return std::nullopt;
        return sqlite3_column_int(stmt, c) != 0;
    };
    t.allow_file_write       = opt_bool(col++);
    t.allow_web_access       = opt_bool(col++);
    t.require_confirmation   = opt_bool(col++);
    t.full_autonomy          = opt_bool(col++);

    // allowed_apps_override stored as comma-separated string
    std::string apps = str(col++);
    if (!apps.empty()) {
        std::istringstream ss(apps);
        std::string app;
        while (std::getline(ss, app, ',')) {
            if (!app.empty()) t.allowed_apps_override.push_back(app);
        }
    }
    return t;
}

TaskRun SchedulerStore::row_to_run(sqlite3_stmt* stmt) {
    TaskRun r;
    int col = 0;
    auto str = [&](int c) -> std::string {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, c));
        return v ? v : "";
    };
    r.run_id         = str(col++);
    r.task_id        = str(col++);
    r.task_name      = str(col++);
    r.status         = static_cast<TaskRunStatus>(sqlite3_column_int(stmt, col++));
    r.started_at     = str(col++);
    r.finished_at    = str(col++);
    r.result_summary = str(col++);
    r.error_message  = str(col++);
    r.duration_ms    = sqlite3_column_int(stmt, col++);
    r.output_path    = str(col++);
    r.inference_id   = str(col++);
    r.session_id     = str(col++);
    return r;
}

TaskActionLog SchedulerStore::row_to_action_log(sqlite3_stmt* stmt) {
    TaskActionLog e;
    int col = 0;
    auto str = [&](int c) -> std::string {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, c));
        return v ? v : "";
    };
    // skip id column (col 0)
    col++;
    // skip run_id column (col 1)
    col++;
    e.sequence              = sqlite3_column_int(stmt, col++);
    e.action_type           = str(col++);
    e.description           = str(col++);
    e.input_summary         = str(col++);
    e.output_summary        = str(col++);
    e.success               = sqlite3_column_int(stmt, col++) != 0;
    e.required_confirmation = sqlite3_column_int(stmt, col++) != 0;
    e.confirmation_granted  = sqlite3_column_int(stmt, col++) != 0;
    e.duration_ms           = sqlite3_column_int(stmt, col++);
    e.timestamp             = str(col++);
    return e;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

SchedulerStore::SchedulerStore(const std::string& db_path)
    : db_path_(db_path) {}

SchedulerStore::~SchedulerStore() { close(); }

void SchedulerStore::open() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Ensure parent directory exists
    auto parent = std::filesystem::path(db_path_).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);

    if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
        throw std::runtime_error("SchedulerStore: cannot open DB at " + db_path_);
    }
    exec_sql(db_, "PRAGMA journal_mode=WAL;");
    exec_sql(db_, "PRAGMA foreign_keys=ON;");
    create_tables();
    apply_migrations();
    LOG_INFO("SchedulerStore: opened " + db_path_);
}

void SchedulerStore::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

void SchedulerStore::create_tables() {
    exec_sql(db_, R"sql(
        CREATE TABLE IF NOT EXISTS tasks (
            id                    TEXT PRIMARY KEY,
            name                  TEXT NOT NULL,
            description           TEXT NOT NULL DEFAULT '',
            enabled               INTEGER NOT NULL DEFAULT 1,
            trigger_json          TEXT NOT NULL DEFAULT '{}',
            action_json           TEXT NOT NULL DEFAULT '{}',
            run_count             INTEGER NOT NULL DEFAULT 0,
            fail_count            INTEGER NOT NULL DEFAULT 0,
            last_run_at           TEXT NOT NULL DEFAULT '',
            next_run_at           TEXT NOT NULL DEFAULT '',
            created_at            TEXT NOT NULL DEFAULT '',
            updated_at            TEXT NOT NULL DEFAULT '',
            created_from          TEXT NOT NULL DEFAULT '',
            created_in_session    TEXT NOT NULL DEFAULT '',
            allow_file_write      INTEGER,
            allow_web_access      INTEGER,
            require_confirmation  INTEGER,
            full_autonomy         INTEGER,
            allowed_apps_override TEXT NOT NULL DEFAULT ''
        );
    )sql");

    exec_sql(db_, R"sql(
        CREATE TABLE IF NOT EXISTS task_runs (
            run_id         TEXT PRIMARY KEY,
            task_id        TEXT NOT NULL,
            task_name      TEXT NOT NULL DEFAULT '',
            status         INTEGER NOT NULL DEFAULT 0,
            started_at     TEXT NOT NULL DEFAULT '',
            finished_at    TEXT NOT NULL DEFAULT '',
            result_summary TEXT NOT NULL DEFAULT '',
            error_message  TEXT NOT NULL DEFAULT '',
            duration_ms    INTEGER NOT NULL DEFAULT 0,
            output_path    TEXT NOT NULL DEFAULT '',
            inference_id   TEXT NOT NULL DEFAULT '',
            session_id     TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        );
    )sql");

    exec_sql(db_, R"sql(
        CREATE TABLE IF NOT EXISTS task_action_logs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                TEXT NOT NULL,
            sequence              INTEGER NOT NULL DEFAULT 0,
            action_type           TEXT NOT NULL DEFAULT '',
            description           TEXT NOT NULL DEFAULT '',
            input_summary         TEXT NOT NULL DEFAULT '',
            output_summary        TEXT NOT NULL DEFAULT '',
            success               INTEGER NOT NULL DEFAULT 0,
            required_confirmation INTEGER NOT NULL DEFAULT 0,
            confirmation_granted  INTEGER NOT NULL DEFAULT 0,
            duration_ms           INTEGER NOT NULL DEFAULT 0,
            timestamp             TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(run_id) REFERENCES task_runs(run_id) ON DELETE CASCADE
        );
    )sql");

    // Indices for common query patterns
    exec_sql(db_, "CREATE INDEX IF NOT EXISTS idx_runs_task_id ON task_runs(task_id);");
    exec_sql(db_, "CREATE INDEX IF NOT EXISTS idx_runs_started_at ON task_runs(started_at DESC);");
    exec_sql(db_, "CREATE INDEX IF NOT EXISTS idx_logs_run_id ON task_action_logs(run_id);");
}

void SchedulerStore::apply_migrations() {
    // Version tracking in user_version pragma.
    // Current schema version: 1
    int version = 0;
    {
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "PRAGMA user_version;", -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW)
                version = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
        }
    }
    if (version < 1) {
        // v1: initial schema already created by create_tables()
        exec_sql(db_, "PRAGMA user_version = 1;");
    }
    // Future: if (version < 2) { ... exec_sql("PRAGMA user_version = 2;"); }
}

// ---------------------------------------------------------------------------
// ScheduledTask CRUD
// ---------------------------------------------------------------------------

std::string SchedulerStore::insert_task(const ScheduledTask& task) {
    std::lock_guard<std::mutex> lock(mutex_);

    const char* sql = R"sql(
        INSERT INTO tasks
            (id, name, description, enabled, trigger_json, action_json,
             run_count, fail_count, last_run_at, next_run_at,
             created_at, updated_at, created_from, created_in_session,
             allow_file_write, allow_web_access, require_confirmation,
             full_autonomy, allowed_apps_override)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
    )sql";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    std::string trigger_json = trigger_to_json(task.trigger);
    std::string action_json  = action_to_json(task.action);
    std::string apps_str;
    for (size_t i = 0; i < task.allowed_apps_override.size(); ++i) {
        if (i) apps_str += ',';
        apps_str += task.allowed_apps_override[i];
    }

    int col = 1;
    sqlite3_bind_text(stmt, col++, task.id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, task.enabled ? 1 : 0);
    sqlite3_bind_text(stmt, col++, trigger_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, action_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, task.run_count);
    sqlite3_bind_int (stmt, col++, task.fail_count);
    sqlite3_bind_text(stmt, col++, task.last_run_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.next_run_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.created_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.updated_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.created_from.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.created_in_session.c_str(), -1, SQLITE_TRANSIENT);

    auto bind_opt = [&](std::optional<bool> v) {
        if (v.has_value()) sqlite3_bind_int(stmt, col++, *v ? 1 : 0);
        else               sqlite3_bind_null(stmt, col++);
    };
    bind_opt(task.allow_file_write);
    bind_opt(task.allow_web_access);
    bind_opt(task.require_confirmation);
    bind_opt(task.full_autonomy);
    sqlite3_bind_text(stmt, col++, apps_str.c_str(), -1, SQLITE_TRANSIENT);

    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return task.id;
}

bool SchedulerStore::update_task(const ScheduledTask& task) {
    std::lock_guard<std::mutex> lock(mutex_);

    const char* sql = R"sql(
        UPDATE tasks SET
            name=?, description=?, enabled=?, trigger_json=?, action_json=?,
            run_count=?, fail_count=?, last_run_at=?, next_run_at=?,
            updated_at=?, created_from=?, created_in_session=?,
            allow_file_write=?, allow_web_access=?, require_confirmation=?,
            full_autonomy=?, allowed_apps_override=?
        WHERE id=?;
    )sql";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    std::string trigger_json = trigger_to_json(task.trigger);
    std::string action_json  = action_to_json(task.action);
    std::string apps_str;
    for (size_t i = 0; i < task.allowed_apps_override.size(); ++i) {
        if (i) apps_str += ',';
        apps_str += task.allowed_apps_override[i];
    }

    int col = 1;
    sqlite3_bind_text(stmt, col++, task.name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, task.enabled ? 1 : 0);
    sqlite3_bind_text(stmt, col++, trigger_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, action_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, task.run_count);
    sqlite3_bind_int (stmt, col++, task.fail_count);
    sqlite3_bind_text(stmt, col++, task.last_run_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.next_run_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.updated_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.created_from.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.created_in_session.c_str(), -1, SQLITE_TRANSIENT);
    auto bind_opt = [&](std::optional<bool> v) {
        if (v.has_value()) sqlite3_bind_int(stmt, col++, *v ? 1 : 0);
        else               sqlite3_bind_null(stmt, col++);
    };
    bind_opt(task.allow_file_write);
    bind_opt(task.allow_web_access);
    bind_opt(task.require_confirmation);
    bind_opt(task.full_autonomy);
    sqlite3_bind_text(stmt, col++, apps_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, task.id.c_str(), -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool SchedulerStore::delete_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "DELETE FROM tasks WHERE id=?;", -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, task_id.c_str(), -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

std::optional<ScheduledTask> SchedulerStore::get_task(const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT id,name,description,enabled,trigger_json,action_json,
               run_count,fail_count,last_run_at,next_run_at,
               created_at,updated_at,created_from,created_in_session,
               allow_file_write,allow_web_access,require_confirmation,
               full_autonomy,allowed_apps_override
        FROM tasks WHERE id=?;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, task_id.c_str(), -1, SQLITE_TRANSIENT);
    std::optional<ScheduledTask> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) result = row_to_task(stmt);
    sqlite3_finalize(stmt);
    return result;
}

std::vector<ScheduledTask> SchedulerStore::list_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT id,name,description,enabled,trigger_json,action_json,
               run_count,fail_count,last_run_at,next_run_at,
               created_at,updated_at,created_from,created_in_session,
               allow_file_write,allow_web_access,require_confirmation,
               full_autonomy,allowed_apps_override
        FROM tasks ORDER BY created_at ASC;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    std::vector<ScheduledTask> result;
    while (sqlite3_step(stmt) == SQLITE_ROW) result.push_back(row_to_task(stmt));
    sqlite3_finalize(stmt);
    return result;
}

std::vector<ScheduledTask> SchedulerStore::list_enabled_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT id,name,description,enabled,trigger_json,action_json,
               run_count,fail_count,last_run_at,next_run_at,
               created_at,updated_at,created_from,created_in_session,
               allow_file_write,allow_web_access,require_confirmation,
               full_autonomy,allowed_apps_override
        FROM tasks WHERE enabled=1 ORDER BY created_at ASC;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    std::vector<ScheduledTask> result;
    while (sqlite3_step(stmt) == SQLITE_ROW) result.push_back(row_to_task(stmt));
    sqlite3_finalize(stmt);
    return result;
}

bool SchedulerStore::update_task_stats(const std::string& task_id,
                                       int run_count, int fail_count,
                                       const std::string& last_run_at,
                                       const std::string& next_run_at) {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        UPDATE tasks SET run_count=?, fail_count=?, last_run_at=?, next_run_at=?,
                         updated_at=datetime('now')
        WHERE id=?;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int (stmt, 1, run_count);
    sqlite3_bind_int (stmt, 2, fail_count);
    sqlite3_bind_text(stmt, 3, last_run_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, next_run_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, task_id.c_str(), -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool SchedulerStore::set_task_enabled(const std::string& task_id, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = "UPDATE tasks SET enabled=?, updated_at=datetime('now') WHERE id=?;";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int (stmt, 1, enabled ? 1 : 0);
    sqlite3_bind_text(stmt, 2, task_id.c_str(), -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

// ---------------------------------------------------------------------------
// TaskRun CRUD
// ---------------------------------------------------------------------------

std::string SchedulerStore::insert_run(const TaskRun& run) {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        INSERT INTO task_runs
            (run_id,task_id,task_name,status,started_at,finished_at,
             result_summary,error_message,duration_ms,output_path,
             inference_id,session_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?);
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    int col = 1;
    sqlite3_bind_text(stmt, col++, run.run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.task_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.task_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, static_cast<int>(run.status));
    sqlite3_bind_text(stmt, col++, run.started_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.finished_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.result_summary.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.error_message.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, run.duration_ms);
    sqlite3_bind_text(stmt, col++, run.output_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.inference_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return run.run_id;
}

bool SchedulerStore::update_run(const TaskRun& run) {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        UPDATE task_runs SET
            status=?,finished_at=?,result_summary=?,error_message=?,
            duration_ms=?,output_path=?,inference_id=?,session_id=?
        WHERE run_id=?;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    int col = 1;
    sqlite3_bind_int (stmt, col++, static_cast<int>(run.status));
    sqlite3_bind_text(stmt, col++, run.finished_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.result_summary.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.error_message.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, run.duration_ms);
    sqlite3_bind_text(stmt, col++, run.output_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.inference_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, run.run_id.c_str(), -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

std::optional<TaskRun> SchedulerStore::get_run(const std::string& run_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT run_id,task_id,task_name,status,started_at,finished_at,
               result_summary,error_message,duration_ms,output_path,
               inference_id,session_id
        FROM task_runs WHERE run_id=?;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    std::optional<TaskRun> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) result = row_to_run(stmt);
    sqlite3_finalize(stmt);
    return result;
}

std::vector<TaskRun> SchedulerStore::get_task_runs(const std::string& task_id, int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT run_id,task_id,task_name,status,started_at,finished_at,
               result_summary,error_message,duration_ms,output_path,
               inference_id,session_id
        FROM task_runs WHERE task_id=? ORDER BY started_at DESC LIMIT ?;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, task_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, 2, limit);
    std::vector<TaskRun> result;
    while (sqlite3_step(stmt) == SQLITE_ROW) result.push_back(row_to_run(stmt));
    sqlite3_finalize(stmt);
    return result;
}

std::vector<TaskRun> SchedulerStore::get_recent_runs(int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT run_id,task_id,task_name,status,started_at,finished_at,
               result_summary,error_message,duration_ms,output_path,
               inference_id,session_id
        FROM task_runs ORDER BY started_at DESC LIMIT ?;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, limit);
    std::vector<TaskRun> result;
    while (sqlite3_step(stmt) == SQLITE_ROW) result.push_back(row_to_run(stmt));
    sqlite3_finalize(stmt);
    return result;
}

// ---------------------------------------------------------------------------
// TaskActionLog
// ---------------------------------------------------------------------------

void SchedulerStore::append_action_log(const std::string& run_id, const TaskActionLog& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        INSERT INTO task_action_logs
            (run_id,sequence,action_type,description,input_summary,output_summary,
             success,required_confirmation,confirmation_granted,duration_ms,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?);
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    int col = 1;
    sqlite3_bind_text(stmt, col++, run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, e.sequence);
    sqlite3_bind_text(stmt, col++, e.action_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, e.description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, e.input_summary.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, col++, e.output_summary.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, col++, e.success ? 1 : 0);
    sqlite3_bind_int (stmt, col++, e.required_confirmation ? 1 : 0);
    sqlite3_bind_int (stmt, col++, e.confirmation_granted ? 1 : 0);
    sqlite3_bind_int (stmt, col++, e.duration_ms);
    sqlite3_bind_text(stmt, col++, e.timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::vector<TaskActionLog> SchedulerStore::get_action_logs(const std::string& run_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = R"sql(
        SELECT id,run_id,sequence,action_type,description,input_summary,
               output_summary,success,required_confirmation,confirmation_granted,
               duration_ms,timestamp
        FROM task_action_logs WHERE run_id=? ORDER BY sequence ASC;
    )sql";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    std::vector<TaskActionLog> result;
    while (sqlite3_step(stmt) == SQLITE_ROW) result.push_back(row_to_action_log(stmt));
    sqlite3_finalize(stmt);
    return result;
}

// ---------------------------------------------------------------------------
// Housekeeping
// ---------------------------------------------------------------------------

void SchedulerStore::prune_run_history(int max_entries_per_task) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Delete runs ranked beyond max_entries_per_task when ordered by started_at DESC
    const char* sql = R"sql(
        DELETE FROM task_runs
        WHERE run_id IN (
            SELECT run_id FROM (
                SELECT run_id,
                       ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY started_at DESC) AS rn
                FROM task_runs
            ) WHERE rn > ?
        );
    )sql";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, max_entries_per_task);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

int SchedulerStore::count_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM tasks;", -1, &stmt, nullptr);
    int n = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) n = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return n;
}

int SchedulerStore::count_enabled_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM tasks WHERE enabled=1;", -1, &stmt, nullptr);
    int n = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) n = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return n;
}

int SchedulerStore::count_runs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM task_runs;", -1, &stmt, nullptr);
    int n = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) n = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return n;
}

} // namespace cardinal
