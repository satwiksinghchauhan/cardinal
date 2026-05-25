// =============================================================================
// Cardinal - Tool: schedule_task Implementation
// =============================================================================

#include "tools/builtin/computer/tool_schedule_task.h"
#include "scheduler/scheduler_engine.h"

#include <sstream>
#include <chrono>

namespace cardinal {

ToolDefinition make_schedule_task_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "schedule_task";
    def.description =
        "Create, list, manage, or run scheduled tasks.\n\n"
        "Actions:\n"
        "  create    — schedule a new task from a natural language description\n"
        "  list      — list all scheduled tasks\n"
        "  delete    — delete a task by ID\n"
        "  enable    — enable a disabled task\n"
        "  disable   — pause a task without deleting it\n"
        "  run_now   — trigger a task immediately\n\n"
        "Examples for create:\n"
        "  'every morning at 9am search for AI news and save a summary'\n"
        "  'every hour run maintenance'\n"
        "  'when factual_confidence < 0.65 trigger training'";
    def.confirmation_required = false;

    def.parameters.push_back({
        "action", ToolParameterType::STRING,
        "Action: create|list|delete|enable|disable|run_now",
        true, ""
    });
    def.parameters.push_back({
        "description", ToolParameterType::STRING,
        "Natural language description of the task (for create action)",
        false, ""
    });
    def.parameters.push_back({
        "task_id", ToolParameterType::STRING,
        "Task ID (for delete, enable, disable, run_now)",
        false, ""
    });
    return def;
}

ToolResult execute_schedule_task(const ToolCall&    call,
                                  SchedulerEngine&   scheduler,
                                  const std::string& session_id) {
    ToolResult result;
    result.tool_name = "schedule_task";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string action  = get("action");
        std::string desc    = get("description");
        std::string task_id = get("task_id");

        if (action == "create") {
            if (desc.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "create requires 'description'";
                return result;
            }
            auto parse = scheduler.create_task_from_nl(desc, session_id);
            if (parse.success) {
                result.status = ToolStatus::SUCCESS;
                result.output = "Task created successfully!\n"
                    "ID:          " + parse.task.id + "\n"
                    "Name:        " + parse.task.name + "\n"
                    "Trigger:     " + std::string(trigger_type_to_string(parse.task.trigger.type));
                if (!parse.task.trigger.cron_expression.empty())
                    result.output += " (" + parse.task.trigger.cron_expression + ")";
                result.output += "\nAction:      " +
                    std::string(action_type_to_string(parse.task.action.type));
                if (!parse.task.action.goal.empty())
                    result.output += ": " + parse.task.action.goal.substr(0, 80);
                result.output += "\nConfidence:  " + std::to_string(int(parse.confidence * 100)) + "%";
            } else if (!parse.clarification_needed.empty()) {
                result.status = ToolStatus::SUCCESS; // not a failure, needs clarification
                result.output = "I need a bit more detail to schedule this task:\n" +
                                parse.clarification_needed;
            } else {
                result.status = ToolStatus::FAILURE;
                result.output = "Failed to parse task: " + parse.error_message;
            }

        } else if (action == "list") {
            auto tasks = scheduler.list_tasks();
            if (tasks.empty()) {
                result.status = ToolStatus::SUCCESS;
                result.output = "No scheduled tasks. Use create to add one.";
            } else {
                std::ostringstream oss;
                oss << "Scheduled tasks (" << tasks.size() << "):\n\n";
                for (const auto& t : tasks) {
                    oss << (t.enabled ? "✓" : "✗") << " [" << t.id.substr(0, 8) << "] "
                        << t.name << "\n"
                        << "  Trigger: " << trigger_type_to_string(t.trigger.type);
                    if (!t.trigger.cron_expression.empty())
                        oss << " (" << t.trigger.cron_expression << ")";
                    oss << "\n"
                        << "  Action:  " << action_type_to_string(t.action.type);
                    if (!t.action.goal.empty())
                        oss << " — " << t.action.goal.substr(0, 60);
                    oss << "\n"
                        << "  Runs: " << t.run_count
                        << "  Fails: " << t.fail_count;
                    if (!t.last_run_at.empty()) oss << "  Last: " << t.last_run_at;
                    oss << "\n\n";
                }
                result.status = ToolStatus::SUCCESS;
                result.output = oss.str();
            }

        } else if (action == "delete") {
            if (task_id.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "delete requires task_id. Use list to find IDs.";
                return result;
            }
            bool ok = scheduler.delete_task(task_id);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? "Task deleted: " + task_id
                               : "Task not found: " + task_id;

        } else if (action == "enable") {
            if (task_id.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "enable requires task_id";
                return result;
            }
            bool ok = scheduler.enable_task(task_id);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? "Task enabled: " + task_id : "Task not found: " + task_id;

        } else if (action == "disable") {
            if (task_id.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "disable requires task_id";
                return result;
            }
            bool ok = scheduler.disable_task(task_id);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? "Task disabled: " + task_id : "Task not found: " + task_id;

        } else if (action == "run_now") {
            if (task_id.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "run_now requires task_id";
                return result;
            }
            std::string run_id = scheduler.run_task_now(task_id);
            result.status = ToolStatus::SUCCESS;
            result.output = "Task dispatched. Run ID: " + run_id;

        } else {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Unknown action: " + action;
        }

    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "schedule_task failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
