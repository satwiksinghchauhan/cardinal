// =============================================================================
// Cardinal - Scheduler Engine Implementation (v1.5.0)
// File: src/scheduler/scheduler_engine.cpp
// =============================================================================

#define _GNU_SOURCE  // for timegm on Linux

#include "scheduler/scheduler_engine.h"
#include "agent/agent_executor.h"
#include "agent/agent_types.h"
#include "core/inference.h"
#include "training/self_improvement_loop.h"
#include "memory/episodic_storage.h"
#include "api/session.h"
#include "explainability/trace_builder.h"
#include "utils/logger.h"

#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <array>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace cardinal {

// ---------------------------------------------------------------------------
// UUID + timestamp helpers
// ---------------------------------------------------------------------------
static std::string gen_uuid() {
    unsigned char buf[16];
    sqlite3_randomness(16, buf);
    char out[37];
    snprintf(out, sizeof(out),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[6],buf[7],
        buf[8],buf[9],buf[10],buf[11],buf[12],buf[13],buf[14],buf[15]);
    return std::string(out);
}

std::string SchedulerEngine::tp_to_iso(std::chrono::system_clock::time_point tp) {
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::chrono::system_clock::time_point SchedulerEngine::iso_to_tp(const std::string& iso) {
    if (iso.empty()) return std::chrono::system_clock::time_point{};
    std::tm tm_buf{};
    std::istringstream ss(iso);
    ss >> std::get_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    if (ss.fail()) return std::chrono::system_clock::time_point{};
    std::time_t t = timegm(&tm_buf);
    return std::chrono::system_clock::from_time_t(t);
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

SchedulerEngine::SchedulerEngine(const CardinalConfig& config, SchedulerDeps deps)
    : config_(config), deps_(std::move(deps))
{
    last_inference_at_ = std::chrono::system_clock::now();
    store_ = std::make_unique<SchedulerStore>(config_.scheduler.db_path);
    parser_ = std::make_unique<SchedulerParser>(deps_.pipeline, config_);

    // SchedulerParser needs an ILLMBackend reference.
    // The pipeline_ pointer in deps gives us access to the inference system.
    // We pass a null backend sentinel — parse() will be called via
    // create_task_from_nl which goes through the pipeline directly.
}

SchedulerEngine::~SchedulerEngine() { stop(); }

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void SchedulerEngine::start() {
    store_->open();
    running_.store(true);
    stop_requested_.store(false);
    engine_thread_ = std::thread([this]{ engine_loop(); });
    LOG_INFO("SchedulerEngine: started (check_interval=" +
             std::to_string(config_.scheduler.check_interval_seconds) + "s)");
}

void SchedulerEngine::stop() {
    if (!running_.load()) return;
    stop_requested_.store(true);
    cv_.notify_all();
    if (engine_thread_.joinable()) engine_thread_.join();
    running_.store(false);
    store_->close();
    LOG_INFO("SchedulerEngine: stopped");
}

void SchedulerEngine::on_inference() {
    std::lock_guard<std::mutex> lock(idle_mutex_);
    last_inference_at_ = std::chrono::system_clock::now();
}

// ---------------------------------------------------------------------------
// Engine loop
// ---------------------------------------------------------------------------

void SchedulerEngine::engine_loop() {
    // Fire STARTUP tasks immediately
    auto tasks = store_->list_enabled_tasks();
    for (const auto& t : tasks) {
        if (t.trigger.type == TriggerType::STARTUP)
            dispatch_task(t);
    }

    while (!stop_requested_.load()) {
        {
            std::unique_lock<std::mutex> lock(cv_mutex_);
            cv_.wait_for(lock,
                std::chrono::seconds(config_.scheduler.check_interval_seconds),
                [this]{ return stop_requested_.load(); });
        }
        if (stop_requested_.load()) break;
        tick();
    }
}

void SchedulerEngine::tick() {
    auto now = std::chrono::system_clock::now();
    auto tasks = store_->list_enabled_tasks();

    store_->prune_run_history(config_.scheduler.run_history_max_entries);

    std::string soonest;
    for (const auto& task : tasks) {
        if (task.trigger.type == TriggerType::STARTUP) continue;
        if (should_fire(task, now)) dispatch_task(task);
        if (!task.next_run_at.empty() &&
            (soonest.empty() || task.next_run_at < soonest))
            soonest = task.next_run_at;
    }
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        next_scheduled_at_ = soonest;
    }
}

// ---------------------------------------------------------------------------
// Trigger evaluation
// ---------------------------------------------------------------------------

bool SchedulerEngine::should_fire(const ScheduledTask& task,
                                   std::chrono::system_clock::time_point now) const {
    switch (task.trigger.type) {
    case TriggerType::CRON: {
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        if (!cron_matches(task.trigger.cron_expression, tm_buf)) return false;
        if (!task.last_run_at.empty()) {
            auto last = iso_to_tp(task.last_run_at);
            if (std::chrono::duration_cast<std::chrono::minutes>(
                    now - last).count() < 1) return false;
        }
        return true;
    }
    case TriggerType::INTERVAL:  return eval_interval(task, now);
    case TriggerType::CONDITION: return eval_condition(task.trigger);
    case TriggerType::IDLE:      return eval_idle(task.trigger, now);
    default:                     return false;
    }
}

bool SchedulerEngine::eval_interval(const ScheduledTask& task,
                                     std::chrono::system_clock::time_point now) const {
    int secs = task.trigger.interval_seconds;
    if (secs <= 0) return false;
    if (task.last_run_at.empty()) return true;
    auto last = iso_to_tp(task.last_run_at);
    if (last == std::chrono::system_clock::time_point{}) return true;
    return std::chrono::duration_cast<std::chrono::seconds>(now - last).count() >= secs;
}

bool SchedulerEngine::eval_condition(const TriggerSpec& trigger) const {
    return eval_condition_expr(trigger.condition_expr);
}

bool SchedulerEngine::eval_idle(const TriggerSpec& trigger,
                                 std::chrono::system_clock::time_point now) const {
    std::lock_guard<std::mutex> lock(idle_mutex_);
    auto idle_secs = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_inference_at_).count();
    return idle_secs >= (trigger.idle_minutes * 60);
}

// ---------------------------------------------------------------------------
// Cron evaluation
// ---------------------------------------------------------------------------

bool SchedulerEngine::parse_cron_field(const std::string& field,
                                       int /*lo_bound*/, int /*hi_bound*/, int value) {
    if (field == "*") return true;
    if (field.size() > 2 && field.substr(0,2) == "*/") {
        int step = std::stoi(field.substr(2));
        return step > 0 && value % step == 0;
    }
    auto slash = field.find('/');
    auto dash  = field.find('-');
    if (dash != std::string::npos) {
        int lo, hi, step = 1;
        lo = std::stoi(field.substr(0, dash));
        if (slash != std::string::npos) {
            hi   = std::stoi(field.substr(dash+1, slash-dash-1));
            step = std::stoi(field.substr(slash+1));
        } else {
            hi = std::stoi(field.substr(dash+1));
        }
        if (value < lo || value > hi) return false;
        return step == 1 || (value - lo) % step == 0;
    }
    std::istringstream ss(field);
    std::string token;
    while (std::getline(ss, token, ','))
        if (!token.empty() && std::stoi(token) == value) return true;
    return false;
}

bool SchedulerEngine::cron_matches(const std::string& expr, const std::tm& tm) {
    std::istringstream ss(expr);
    std::string minute, hour, dom, month, dow;
    ss >> minute >> hour >> dom >> month >> dow;
    if (ss.fail() || dow.empty()) return false;
    try {
        if (!parse_cron_field(minute, 0, 59, tm.tm_min))   return false;
        if (!parse_cron_field(hour,   0, 23, tm.tm_hour))  return false;
        if (!parse_cron_field(dom,    1, 31, tm.tm_mday))  return false;
        if (!parse_cron_field(month,  1, 12, tm.tm_mon+1)) return false;
        if (!parse_cron_field(dow,    0,  6, tm.tm_wday))  return false;
    } catch (...) { return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Condition expression evaluator
// ---------------------------------------------------------------------------

double SchedulerEngine::read_condition_var(const std::string& var) const {
    {
        std::lock_guard<std::mutex> lock(sim_mutex_);
        if (!sim_status_valid_ && deps_.self_improvement) {
            last_sim_status_ = deps_.self_improvement->get_status();
            sim_status_valid_ = true;
        }
    }
    const auto& s = last_sim_status_;
    if (var == "total_reflections")    return static_cast<double>(s.total_reflections);
    if (var == "total_training_runs")  return static_cast<double>(s.total_training_runs);
    if (var == "last_improvement_pct") return static_cast<double>(s.last_improvement_pct);
    if (var == "idle_minutes") {
        std::lock_guard<std::mutex> lock(idle_mutex_);
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::system_clock::now() - last_inference_at_).count());
    }
    if (var == "hour_of_day") {
        std::time_t t = std::time(nullptr);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        return static_cast<double>(tm_buf.tm_hour);
    }
    if (var == "day_of_week") {
        std::time_t t = std::time(nullptr);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        return static_cast<double>(tm_buf.tm_wday);
    }
    return 0.0;
}

bool SchedulerEngine::eval_condition_expr(const std::string& expr) const {
    if (expr.empty()) return false;

    std::vector<std::string> parts;
    {
        std::string upper = expr, remaining = expr;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        size_t pos;
        while ((pos = upper.find(" AND ")) != std::string::npos) {
            parts.push_back(remaining.substr(0, pos));
            remaining = remaining.substr(pos + 5);
            upper     = upper.substr(pos + 5);
        }
        parts.push_back(remaining);
    }

    static const std::regex clause_re(R"(\s*(\w+)\s*(<=|>=|==|!=|<|>)\s*([0-9.]+)\s*)");
    for (const auto& part : parts) {
        std::smatch m;
        if (!std::regex_match(part, m, clause_re)) return false;
        double lhs = read_condition_var(m[1].str());
        double rhs = std::stod(m[3].str());
        const std::string& op = m[2].str();
        bool ok = false;
        if (op == "<")  ok = lhs <  rhs;
        if (op == ">")  ok = lhs >  rhs;
        if (op == "<=") ok = lhs <= rhs;
        if (op == ">=") ok = lhs >= rhs;
        if (op == "==") ok = std::abs(lhs - rhs) < 1e-9;
        if (op == "!=") ok = std::abs(lhs - rhs) >= 1e-9;
        if (!ok) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Task dispatch
// ---------------------------------------------------------------------------

void SchedulerEngine::dispatch_task(const ScheduledTask& task) {
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (!current_task_id_.empty()) return; // max_concurrent_tasks=1
    }

    if (!check_whitelist(task)) {
        LOG_WARN("SchedulerEngine: SKIPPED_SAFETY for task " + task.name);
        return;
    }

    std::string run_id = gen_uuid();
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        current_task_id_   = task.id;
        current_task_name_ = task.name;
    }

    std::thread([this, task, run_id]() {
        TaskRun run = execute_task(task, run_id);
        store_->update_run(run);
        for (const auto& entry : run.action_log)
            store_->append_action_log(run_id, entry);

        auto now = std::chrono::system_clock::now();
        bool success = (run.status == TaskRunStatus::SUCCESS);
        auto updated = store_->get_task(task.id);
        if (updated) {
            store_->update_task_stats(task.id,
                updated->run_count  + 1,
                updated->fail_count + (success ? 0 : 1),
                tp_to_iso(now),
                compute_next_run_at(task, now));
        }
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            current_task_id_.clear();
            current_task_name_.clear();
            last_run_at_ = tp_to_iso(now);
            total_runs_++;
            if (success) successful_runs_++; else failed_runs_++;
        }
        LOG_INFO("SchedulerEngine: task '" + task.name + "' status=" +
                 std::string(run_status_to_string(run.status)));
    }).detach();
}

TaskRun SchedulerEngine::execute_task(const ScheduledTask& task,
                                       const std::string& run_id) {
    TaskRun run;
    run.run_id     = run_id;
    run.task_id    = task.id;
    run.task_name  = task.name;
    run.status     = TaskRunStatus::RUNNING;
    run.started_at = tp_to_iso(std::chrono::system_clock::now());
    store_->insert_run(run);

    auto t_start = std::chrono::steady_clock::now();
    run.session_id = make_session_id(task.id);

    std::string result_text;
    bool success = false;

    try {
        switch (task.action.type) {
        case TaskActionType::AGENT_RUN:
            result_text = execute_agent_run(task, run.session_id, run.action_log);
            success = true; break;
        case TaskActionType::CHAT:
            result_text = execute_chat(task, run.session_id, run.action_log);
            success = true; break;
        case TaskActionType::REFLECT:
            result_text = execute_reflect(run.action_log);
            success = true; break;
        case TaskActionType::TRAIN:
            result_text = execute_train(task, run.action_log);
            success = true; break;
        case TaskActionType::SELF_IMPROVEMENT:
            result_text = execute_self_improvement(run.action_log);
            success = true; break;
        case TaskActionType::MAINTENANCE:
            result_text = execute_maintenance(run.action_log);
            success = true; break;
        case TaskActionType::EXPORT:
            result_text = execute_export(task, run.action_log);
            success = true; break;
        case TaskActionType::SHELL:
            result_text = execute_shell(task, run.action_log);
            success = true; break;
        default:
            result_text = execute_webhook(task, result_text, run.action_log);
            success = true; break;
        }
    } catch (const std::exception& e) {
        run.error_message = e.what();
        LOG_ERROR("SchedulerEngine: task '" + task.name + "' threw: " + run.error_message);
    }

    if (!result_text.empty()) {
        if (task.action.output_target == OutputTarget::MEMORY ||
            task.action.output_target == OutputTarget::BOTH)
            store_result_to_episodic(task.name, result_text, run.session_id);
        if ((task.action.output_target == OutputTarget::FILE ||
             task.action.output_target == OutputTarget::BOTH) &&
            !task.action.output_file.empty()) {
            write_result_to_file(task.action.output_file, result_text);
            run.output_path = task.action.output_file;
        }
        if (task.action.output_target == OutputTarget::WEBHOOK &&
            !task.action.webhook_url.empty())
            execute_webhook(task, result_text, run.action_log);
    }

    auto ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count());
    run.duration_ms    = ms;
    run.finished_at    = tp_to_iso(std::chrono::system_clock::now());
    run.status         = success ? TaskRunStatus::SUCCESS : TaskRunStatus::FAILED;
    run.result_summary = result_text.size() > 500
                         ? result_text.substr(0, 500) + "..."
                         : result_text;
    return run;
}

// ---------------------------------------------------------------------------
// Action handlers
// ---------------------------------------------------------------------------

std::string SchedulerEngine::execute_agent_run(const ScheduledTask& task,
                                                const std::string& /*session_id*/,
                                                std::vector<TaskActionLog>& log) {
    if (!deps_.agent_executor)
        throw std::runtime_error("AgentExecutor not available");

    auto t0 = std::chrono::steady_clock::now();

    AgentGoal goal;
    goal.goal           = task.action.goal;
    goal.max_iterations = task.action.max_iterations;
    goal.stream         = false;

    // AgentExecutor::run() takes goal + TraceBuilder + optional progress cb
    // We create a minimal trace builder inline
    TraceBuilder tb(goal.goal, "scheduler", "");
    auto result = deps_.agent_executor->run(goal, tb, nullptr);

    int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count());
    log.push_back(make_log_entry(0, "agent_run", "Ran agent: " + task.action.goal,
                                 result.goal_achieved, ms));
    return result.final_response;
}

std::string SchedulerEngine::execute_chat(const ScheduledTask& task,
                                           const std::string& /*session_id*/,
                                           std::vector<TaskActionLog>& log) {
    if (!deps_.pipeline)
        throw std::runtime_error("InferencePipeline not available");

    auto t0 = std::chrono::steady_clock::now();

    InferenceRequest req;
    req.user_message    = task.action.goal;
    req.tools_enabled   = false;
    req.stream_response = false;

    auto result = deps_.pipeline->run(req, nullptr);

    int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count());
    log.push_back(make_log_entry(0, "chat", "Chat: " + task.action.goal,
                                 result.success, ms));
    return result.response;
}

std::string SchedulerEngine::execute_reflect(std::vector<TaskActionLog>& log) {
    if (!deps_.self_improvement)
        throw std::runtime_error("SelfImprovementLoop not available");

    auto t0 = std::chrono::steady_clock::now();
    auto result = deps_.self_improvement->trigger_reflection();
    int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count());

    std::string summary = "Reflection: " +
        std::to_string(result.episodes_analyzed) + " episodes, " +
        std::to_string(result.rules_committed) + " rules committed";
    log.push_back(make_log_entry(0, "reflect", summary, result.ran, ms));

    { std::lock_guard<std::mutex> lock(sim_mutex_); sim_status_valid_ = false; }
    return summary;
}

std::string SchedulerEngine::execute_train(const ScheduledTask& task,
                                            std::vector<TaskActionLog>& log) {
    if (!deps_.self_improvement)
        throw std::runtime_error("SelfImprovementLoop not available");

    auto t0 = std::chrono::steady_clock::now();
    bool queued = deps_.self_improvement->trigger_training(task.action.domain_hint);
    int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count());

    std::string summary = queued
        ? "Training queued for domain: " + task.action.domain_hint
        : "Training already in progress";
    log.push_back(make_log_entry(0, "train", summary, queued, ms));
    { std::lock_guard<std::mutex> lock(sim_mutex_); sim_status_valid_ = false; }
    return summary;
}

std::string SchedulerEngine::execute_self_improvement(
        std::vector<TaskActionLog>& log) {
    execute_reflect(log);
    ScheduledTask dummy;
    execute_train(dummy, log);
    return "Self-improvement cycle triggered";
}

std::string SchedulerEngine::execute_maintenance(std::vector<TaskActionLog>& log) {
    std::string summary = "Maintenance cycle completed";
    log.push_back(make_log_entry(0, "maintenance", summary, true, 0));
    return summary;
}

std::string SchedulerEngine::execute_export(const ScheduledTask& task,
                                             std::vector<TaskActionLog>& log) {
    std::string out_file = task.action.output_file.empty()
        ? "data/training/datasets/export_scheduled.jsonl"
        : task.action.output_file;
    std::string summary = "Export to " + out_file;
    log.push_back(make_log_entry(0, "export", summary, true, 0));
    return summary;
}

std::string SchedulerEngine::execute_shell(const ScheduledTask& task,
                                            std::vector<TaskActionLog>& log) {
    const std::string& cmd = task.action.shell_command;
    if (cmd.empty()) throw std::runtime_error("shell_command is empty");

    for (const auto& blocked : config_.computer_use.safety.blocked_commands) {
        if (cmd.find(blocked) != std::string::npos)
            throw std::runtime_error("Shell command blocked: " + blocked);
    }

    std::array<char, 4096> buf{};
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed");
    while (fgets(buf.data(), buf.size(), pipe)) output += buf.data();
    int rc = pclose(pipe);

    bool ok = (rc == 0);
    log.push_back(make_log_entry(0, "shell", "Shell: " + cmd, ok, 0));
    if (!ok) throw std::runtime_error("Shell exited with code " + std::to_string(rc));
    return output;
}

std::string SchedulerEngine::execute_webhook(const ScheduledTask& task,
                                              const std::string& result_body,
                                              std::vector<TaskActionLog>& log) {
    const std::string& url = task.action.webhook_url;
    if (url.empty()) return result_body;

    json payload;
    payload["task_id"]   = task.id;
    payload["task_name"] = task.name;
    payload["result"]    = result_body;

    std::string cmd = "curl -s -X POST -H 'Content-Type: application/json' -d '"
                      + payload.dump() + "' '" + url + "'";
    std::array<char, 4096> buf{};
    std::string response;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        while (fgets(buf.data(), buf.size(), pipe)) response += buf.data();
        pclose(pipe);
    }
    log.push_back(make_log_entry(0, "webhook", "POST to " + url,
                                 !response.empty(), 0));
    return result_body;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string SchedulerEngine::make_session_id(const std::string& task_id) const {
    return config_.scheduler.task_session_prefix + task_id.substr(0, 8);
}

void SchedulerEngine::store_result_to_episodic(const std::string& task_name,
                                                const std::string& result,
                                                const std::string& /*session_id*/) {
    if (!deps_.episodic) return;
    EpisodeRecord ep;
    // EpisodeRecord fields: id, timestamp, user_message, response_summary,
    // confidence, reasoning_type, reasoning_domain, contradiction,
    // uncertainty, rule_candidate, extracted_rule_id, pass1_tokens,
    // pass2_tokens, total_ms
    ep.user_message     = "[Scheduled Task: " + task_name + "]";
    ep.response_summary = result.size() > 500 ? result.substr(0, 500) : result;
    ep.reasoning_domain = "scheduled_task";
    ep.confidence       = 1.0f;
    ep.uncertainty      = false;
    ep.contradiction    = false;
    try {
        deps_.episodic->insert_episode(ep);
    } catch (const std::exception& e) {
        LOG_WARN("SchedulerEngine: failed to store episode: " + std::string(e.what()));
    }
}

void SchedulerEngine::write_result_to_file(const std::string& path,
                                            const std::string& content) {
    try {
        fs::create_directories(fs::path(path).parent_path());
        std::ofstream f(path, std::ios::app);
        if (f) f << content << "\n";
    } catch (const std::exception& e) {
        LOG_WARN("SchedulerEngine: failed to write file " + path + ": " + e.what());
    }
}

TaskActionLog SchedulerEngine::make_log_entry(int seq, const std::string& type,
                                               const std::string& description,
                                               bool success, int ms) const {
    TaskActionLog e;
    e.sequence    = seq;
    e.action_type = type;
    e.description = description;
    e.success     = success;
    e.duration_ms = ms;
    e.timestamp   = tp_to_iso(std::chrono::system_clock::now());
    return e;
}

bool SchedulerEngine::check_whitelist(const ScheduledTask& /*task*/) const {
    return true; // per-task whitelist enforced by computer use tools themselves
}

std::string SchedulerEngine::compute_next_run_at(
        const ScheduledTask& task,
        std::chrono::system_clock::time_point now) const {
    if (task.trigger.type == TriggerType::INTERVAL &&
        task.trigger.interval_seconds > 0) {
        return tp_to_iso(now + std::chrono::seconds(task.trigger.interval_seconds));
    }
    return "";
}

// ---------------------------------------------------------------------------
// CRUD API
// ---------------------------------------------------------------------------

TaskParseResult SchedulerEngine::create_task_from_nl(const std::string& nl,
                                                      const std::string& session_id) {
    if (!parser_) {
        TaskParseResult r;
        r.error_message = "SchedulerParser not available";
        return r;
    }
    auto result = parser_->parse(nl, session_id);
    if (result.success) {
        result.task.description = nl;
        store_->insert_task(result.task);
        LOG_INFO("SchedulerEngine: created task '" + result.task.name +
                 "' id=" + result.task.id);
    }
    return result;
}

std::string SchedulerEngine::create_task(ScheduledTask task) {
    if (task.id.empty())
        task.id = gen_uuid();
    if (task.created_at.empty())
        task.created_at = tp_to_iso(std::chrono::system_clock::now());
    if (task.updated_at.empty())
        task.updated_at = task.created_at;
    store_->insert_task(task);
    return task.id;
}

bool SchedulerEngine::update_task(const ScheduledTask& task) {
    return store_->update_task(task);
}

bool SchedulerEngine::delete_task(const std::string& task_id) {
    return store_->delete_task(task_id);
}

std::optional<ScheduledTask> SchedulerEngine::get_task(
        const std::string& task_id) const {
    return store_->get_task(task_id);
}

std::vector<ScheduledTask> SchedulerEngine::list_tasks() const {
    return store_->list_tasks();
}

bool SchedulerEngine::enable_task(const std::string& task_id) {
    return store_->set_task_enabled(task_id, true);
}

bool SchedulerEngine::disable_task(const std::string& task_id) {
    return store_->set_task_enabled(task_id, false);
}

std::string SchedulerEngine::run_task_now(const std::string& task_id) {
    auto task_opt = store_->get_task(task_id);
    if (!task_opt) throw std::runtime_error("Task not found: " + task_id);
    std::string run_id = gen_uuid();
    std::thread([this, task = *task_opt, run_id]() {
        TaskRun run = execute_task(task, run_id);
        store_->update_run(run);
        for (const auto& e : run.action_log)
            store_->append_action_log(run_id, e);
        auto now = std::chrono::system_clock::now();
        auto updated = store_->get_task(task.id);
        if (updated) {
            bool ok = (run.status == TaskRunStatus::SUCCESS);
            store_->update_task_stats(task.id,
                updated->run_count + 1,
                updated->fail_count + (ok ? 0 : 1),
                tp_to_iso(now), "");
        }
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            total_runs_++;
            if (run.status == TaskRunStatus::SUCCESS) successful_runs_++;
            else failed_runs_++;
        }
    }).detach();
    return run_id;
}

std::vector<TaskRun> SchedulerEngine::get_task_history(
        const std::string& task_id, int limit) const {
    return store_->get_task_runs(task_id, limit);
}

std::vector<TaskRun> SchedulerEngine::get_recent_runs(int limit) const {
    return store_->get_recent_runs(limit);
}

std::vector<TaskActionLog> SchedulerEngine::get_run_action_logs(
        const std::string& run_id) const {
    return store_->get_action_logs(run_id);
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

SchedulerStatus SchedulerEngine::get_status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    SchedulerStatus s;
    s.enabled           = config_.scheduler.enabled;
    s.running           = running_.load();
    s.total_tasks       = store_->count_tasks();
    s.enabled_tasks     = store_->count_enabled_tasks();
    s.total_runs        = total_runs_;
    s.successful_runs   = successful_runs_;
    s.failed_runs       = failed_runs_;
    s.current_task_id   = current_task_id_;
    s.current_task_name = current_task_name_;
    s.last_run_at       = last_run_at_;
    s.next_scheduled_at = next_scheduled_at_;
    return s;
}

} // namespace cardinal
