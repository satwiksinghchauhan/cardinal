// =============================================================================
// Cardinal - Scheduler Parser Implementation (v1.5.0)
// File: src/scheduler/scheduler_parser.cpp
// =============================================================================

#include "scheduler/scheduler_parser.h"
#include "core/inference.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <sstream>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include <algorithm>

using json = nlohmann::json;

namespace cardinal {

// ---------------------------------------------------------------------------
// Helpers
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

static std::string iso_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

static TriggerType parse_trigger_type(const std::string& s) {
    if (s == "cron")      return TriggerType::CRON;
    if (s == "interval")  return TriggerType::INTERVAL;
    if (s == "condition") return TriggerType::CONDITION;
    if (s == "startup")   return TriggerType::STARTUP;
    if (s == "idle")      return TriggerType::IDLE;
    return TriggerType::MANUAL;
}

static TaskActionType parse_action_type(const std::string& s) {
    if (s == "agent_run")        return TaskActionType::AGENT_RUN;
    if (s == "chat")             return TaskActionType::CHAT;
    if (s == "reflect")          return TaskActionType::REFLECT;
    if (s == "train")            return TaskActionType::TRAIN;
    if (s == "self_improvement") return TaskActionType::SELF_IMPROVEMENT;
    if (s == "maintenance")      return TaskActionType::MAINTENANCE;
    if (s == "export")           return TaskActionType::EXPORT;
    if (s == "shell")            return TaskActionType::SHELL;
    if (s == "webhook")          return TaskActionType::WEBHOOK;
    return TaskActionType::AGENT_RUN;
}

static OutputTarget parse_output_target(const std::string& s) {
    if (s == "memory")  return OutputTarget::MEMORY;
    if (s == "file")    return OutputTarget::FILE;
    if (s == "webhook") return OutputTarget::WEBHOOK;
    if (s == "discard") return OutputTarget::DISCARD;
    if (s == "both")    return OutputTarget::BOTH;
    return OutputTarget::MEMORY;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SchedulerParser::SchedulerParser(InferencePipeline*    pipeline,
                                 const CardinalConfig& config,
                                 float                 min_confidence)
    : pipeline_(pipeline), config_(config), min_confidence_(min_confidence)
{}

// ---------------------------------------------------------------------------
// build_system_prompt — plain string, no raw literals with special chars
// ---------------------------------------------------------------------------

std::string SchedulerParser::build_system_prompt() {
    return
        "You are a task scheduling assistant for the Cardinal AI system.\n\n"
        "Your job is to parse a natural language task description and output "
        "a JSON object representing a ScheduledTask.\n\n"
        "Respond ONLY with a valid JSON object. No preamble, no markdown.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"name\": \"short human-readable task name (max 60 chars)\",\n"
        "  \"description\": \"the original user description verbatim\",\n"
        "  \"trigger\": {\n"
        "    \"type\": \"cron|interval|condition|manual|startup|idle\",\n"
        "    \"cron_expression\": \"\",\n"
        "    \"interval_seconds\": 0,\n"
        "    \"condition_expr\": \"\",\n"
        "    \"idle_minutes\": 30\n"
        "  },\n"
        "  \"action\": {\n"
        "    \"type\": \"agent_run|chat|reflect|train|self_improvement|maintenance|export|shell|webhook\",\n"
        "    \"goal\": \"\",\n"
        "    \"max_iterations\": 0,\n"
        "    \"domain_hint\": \"\",\n"
        "    \"shell_command\": \"\",\n"
        "    \"webhook_url\": \"\",\n"
        "    \"output_target\": \"memory|file|webhook|discard|both\",\n"
        "    \"output_file\": \"\"\n"
        "  },\n"
        "  \"allow_file_write\": null,\n"
        "  \"allow_web_access\": null,\n"
        "  \"require_confirmation\": null,\n"
        "  \"full_autonomy\": null,\n"
        "  \"confidence\": 0.85\n"
        "}\n\n"
        "Trigger type rules:\n"
        "- Cron schedule (daily/weekly/hourly at specific time): type=cron, fill cron_expression\n"
        "- Every N seconds/minutes/hours: type=interval, fill interval_seconds\n"
        "- When [metric] crosses threshold: type=condition, fill condition_expr\n"
        "- On startup: type=startup\n"
        "- When idle N minutes: type=idle, fill idle_minutes\n"
        "- Ambiguous: type=manual\n\n"
        "Condition variables: factual_confidence, total_contradictions, "
        "total_reflections, total_training_runs, idle_minutes, hour_of_day, day_of_week\n"
        "Operators: <, >, <=, >=, ==, !=, AND\n\n"
        "Action type rules:\n"
        "- Search/research/browse goal: agent_run\n"
        "- Single question: chat\n"
        "- Reflect/meta-cognition: reflect\n"
        "- Train/fine-tune: train\n"
        "- Full self-improvement: self_improvement\n"
        "- Maintenance/scan/cleanup: maintenance\n"
        "- Export training data: export\n"
        "- Shell command: shell\n"
        "- POST to URL: webhook\n\n"
        "Output target rules:\n"
        "- Store in memory: memory (default)\n"
        "- Write to file: file, fill output_file\n"
        "- POST to URL: webhook, fill webhook_url\n"
        "- Silent: discard\n"
        "- Memory + file: both\n\n"
        "Set confidence 0.0-1.0. If ambiguous, set confidence < 0.7 and add "
        "\"clarification_needed\" field explaining what is unclear.\n\n"
        "Cron examples: daily 9am=\"0 9 * * *\", every 30min=\"*/30 * * * *\", "
        "weekdays 8am=\"0 8 * * 1-5\"\n";
}

std::string SchedulerParser::build_user_message(const std::string& nl_description) {
    return "Parse this task description into a ScheduledTask JSON:\n\n" + nl_description;
}

// ---------------------------------------------------------------------------
// extract_from_json (public static)
// ---------------------------------------------------------------------------

TaskParseResult SchedulerParser::extract_from_json(const std::string& model_output,
                                                    const std::string& session_id,
                                                    float min_confidence) {
    TaskParseResult result;

    // Strip markdown code fences if present
    std::string raw = model_output;
    {
        std::regex fence(R"(```(?:json)?\s*([\s\S]*?)```)");
        std::smatch m;
        if (std::regex_search(raw, m, fence)) raw = m[1].str();
    }
    // Trim whitespace
    auto trim = [](std::string& s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            [](unsigned char c){ return !std::isspace(c); }));
        s.erase(std::find_if(s.rbegin(), s.rend(),
            [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
    };
    trim(raw);

    json j;
    try { j = json::parse(raw); }
    catch (const json::parse_error& e) {
        result.error_message = std::string("JSON parse error: ") + e.what();
        return result;
    }

    if (j.contains("clarification_needed") && j["clarification_needed"].is_string()) {
        result.clarification_needed = j["clarification_needed"].get<std::string>();
        result.confidence = j.value("confidence", 0.0f);
        return result;
    }

    float confidence = j.value("confidence", 0.5f);
    result.confidence = confidence;

    if (confidence < min_confidence) {
        result.clarification_needed =
            "I'm not confident about some details. "
            "Could you clarify the schedule or action?";
        return result;
    }

    ScheduledTask& task = result.task;
    task.name        = j.value("name", "Unnamed Task");
    task.description = j.value("description", "");
    task.enabled     = true;

    if (j.contains("trigger") && j["trigger"].is_object()) {
        const auto& jt = j["trigger"];
        task.trigger.type             = parse_trigger_type(jt.value("type", "manual"));
        task.trigger.cron_expression  = jt.value("cron_expression", "");
        task.trigger.interval_seconds = jt.value("interval_seconds", 0);
        task.trigger.condition_expr   = jt.value("condition_expr", "");
        task.trigger.idle_minutes     = jt.value("idle_minutes", 30);
    }

    if (j.contains("action") && j["action"].is_object()) {
        const auto& ja = j["action"];
        task.action.type           = parse_action_type(ja.value("type", "agent_run"));
        task.action.goal           = ja.value("goal", "");
        task.action.max_iterations = ja.value("max_iterations", 0);
        task.action.domain_hint    = ja.value("domain_hint", "");
        task.action.shell_command  = ja.value("shell_command", "");
        task.action.webhook_url    = ja.value("webhook_url", "");
        task.action.output_target  = parse_output_target(ja.value("output_target", "memory"));
        task.action.output_file    = ja.value("output_file", "");
    }

    auto opt_bool = [&](const char* key) -> std::optional<bool> {
        if (!j.contains(key) || j[key].is_null()) return std::nullopt;
        return j[key].get<bool>();
    };
    task.allow_file_write     = opt_bool("allow_file_write");
    task.allow_web_access     = opt_bool("allow_web_access");
    task.require_confirmation = opt_bool("require_confirmation");
    task.full_autonomy        = opt_bool("full_autonomy");

    finalise_task(task, session_id);
    result.success = true;
    return result;
}

// ---------------------------------------------------------------------------
// finalise_task
// ---------------------------------------------------------------------------

void SchedulerParser::finalise_task(ScheduledTask& task,
                                    const std::string& session_id) {
    task.id                 = gen_uuid();
    task.created_at         = iso_now();
    task.updated_at         = task.created_at;
    task.created_from       = session_id.empty() ? "api" : "chat";
    task.created_in_session = session_id;
    task.run_count          = 0;
    task.fail_count         = 0;
}

// ---------------------------------------------------------------------------
// parse — uses InferencePipeline::run()
// ---------------------------------------------------------------------------

TaskParseResult SchedulerParser::parse(const std::string& nl_description,
                                       const std::string& session_id) {
    LOG_DEBUG("SchedulerParser: parsing: " + nl_description.substr(0, 80));

    if (!pipeline_) {
        TaskParseResult r;
        r.error_message = "InferencePipeline not available";
        return r;
    }

    // Inject system prompt via the history field
    InferenceRequest req;
    req.user_message    = build_user_message(nl_description);
    req.tools_enabled   = false;
    req.stream_response = false;
    // Prepend system message in history so the pipeline sees it
    req.history.push_back({ "system", build_system_prompt() });

    InferenceResponse resp;
    try {
        resp = pipeline_->run(req, nullptr);
    } catch (const std::exception& e) {
        TaskParseResult r;
        r.error_message = std::string("LLM call failed: ") + e.what();
        return r;
    }

    if (!resp.success) {
        TaskParseResult r;
        r.error_message = "Inference failed: " + resp.error_message;
        return r;
    }

    TaskParseResult result = extract_from_json(resp.response, session_id,
                                               min_confidence_);
    if (result.success) {
        result.task.description = nl_description;
        LOG_INFO("SchedulerParser: parsed '" + result.task.name +
                 "' confidence=" + std::to_string(result.confidence));
    }
    return result;
}

// ---------------------------------------------------------------------------
// parse_json — direct JSON input, no LLM call
// ---------------------------------------------------------------------------

TaskParseResult SchedulerParser::parse_json(const std::string& json_str,
                                             const std::string& session_id) {
    TaskParseResult result = extract_from_json(json_str, session_id, 0.0f);
    if (result.success) result.task.created_from = "api";
    return result;
}

} // namespace cardinal
