#pragma once
// =============================================================================
// Cardinal - Scheduler & Computer Use Types (v1.5.0)
// File: src/scheduler/scheduler_types.h
//
// All types for:
//   - ScheduledTask      — persistent task definition
//   - TriggerSpec        — what makes a task fire
//   - TaskAction         — what the task does
//   - TaskRun            — one execution record
//   - TaskActionLog      — per-step action log within a run
//   - SchedulerStatus    — current engine state
//
// No dependencies on other Cardinal headers — safe to include anywhere.
// =============================================================================

#include <string>
#include <vector>
#include <optional>

namespace cardinal {

    // =========================================================================
    // TriggerType
    // =========================================================================
    enum class TriggerType : int {
        CRON       = 0,   // cron expression  e.g. "0 9 * * *"
        INTERVAL   = 1,   // fixed interval   e.g. every 3600 seconds
        CONDITION  = 2,   // self-model condition e.g. "factual_confidence < 0.7"
        MANUAL     = 3,   // only via API or chat — never fires automatically
        STARTUP    = 4,   // once on Cardinal start
        IDLE       = 5,   // fires when no inference for N minutes
    };

    inline const char* trigger_type_to_string(TriggerType t) {
        switch (t) {
            case TriggerType::CRON:      return "cron";
            case TriggerType::INTERVAL:  return "interval";
            case TriggerType::CONDITION: return "condition";
            case TriggerType::MANUAL:    return "manual";
            case TriggerType::STARTUP:   return "startup";
            case TriggerType::IDLE:      return "idle";
            default:                     return "unknown";
        }
    }

    // =========================================================================
    // TriggerSpec
    // Describes what makes a task fire.
    // =========================================================================
    struct TriggerSpec {
        TriggerType type = TriggerType::MANUAL;

        // CRON: standard five-field cron expression.
        // Examples: "0 9 * * *" = daily at 9am, "*/30 * * * *" = every 30 min.
        std::string cron_expression;

        // INTERVAL: seconds between runs (0 = disabled).
        int interval_seconds = 0;

        // CONDITION: expression evaluated against SelfImprovementStatus and
        // system state. Supported operators: <, >, <=, >=, ==, !=.
        // Supported variables:
        //   factual_confidence, ethical_confidence, spatial_confidence,
        //   temporal_confidence, social_confidence, mathematical_confidence,
        //   total_contradictions, total_reflections, total_training_runs,
        //   last_improvement_pct, idle_minutes, hour_of_day, day_of_week
        // Examples:
        //   "factual_confidence < 0.7"
        //   "total_contradictions > 10"
        //   "idle_minutes > 30 AND hour_of_day >= 22"
        std::string condition_expr;

        // IDLE: minutes of no inference activity before firing.
        int idle_minutes = 30;
    };

    // =========================================================================
    // TaskActionType
    // =========================================================================
    enum class TaskActionType : int {
        AGENT_RUN        = 0,   // full agentic loop with a goal
        CHAT             = 1,   // single inference
        REFLECT          = 2,   // Layer 2 meta-cognition pass
        TRAIN            = 3,   // Layer 3 LoRA training cycle
        SELF_IMPROVEMENT = 4,   // full self-improvement (all layers)
        MAINTENANCE      = 5,   // run_scan() + run_maintenance()
        EXPORT           = 6,   // export training data to JSONL
        SHELL            = 7,   // run a shell command directly
        WEBHOOK          = 8,   // HTTP POST result to a URL
    };

    inline const char* action_type_to_string(TaskActionType t) {
        switch (t) {
            case TaskActionType::AGENT_RUN:        return "agent_run";
            case TaskActionType::CHAT:             return "chat";
            case TaskActionType::REFLECT:          return "reflect";
            case TaskActionType::TRAIN:            return "train";
            case TaskActionType::SELF_IMPROVEMENT: return "self_improvement";
            case TaskActionType::MAINTENANCE:      return "maintenance";
            case TaskActionType::EXPORT:           return "export";
            case TaskActionType::SHELL:            return "shell";
            case TaskActionType::WEBHOOK:          return "webhook";
            default:                               return "unknown";
        }
    }

    // =========================================================================
    // OutputTarget
    // Where the task result goes.
    // =========================================================================
    enum class OutputTarget : int {
        MEMORY   = 0,   // EpisodicStorage as reasoning_domain="scheduled_task"
        FILE     = 1,   // write to output_file path
        WEBHOOK  = 2,   // HTTP POST to webhook_url
        DISCARD  = 3,   // run silently, no output stored
        BOTH     = 4,   // MEMORY + FILE
    };

    // =========================================================================
    // TaskAction
    // What the task does when triggered.
    // =========================================================================
    struct TaskAction {
        TaskActionType type = TaskActionType::AGENT_RUN;

        // AGENT_RUN / CHAT
        std::string goal;               // goal or message
        int         max_iterations = 0; // 0 = config default
        bool        stream         = false;

        // TRAIN
        std::string domain_hint;        // empty = CurriculumBuilder decides

        // SHELL
        std::string shell_command;      // raw shell command to execute

        // WEBHOOK
        std::string webhook_url;

        // Output routing
        OutputTarget output_target = OutputTarget::MEMORY;
        std::string  output_file;       // used when target == FILE or BOTH
    };

    // =========================================================================
    // TaskRunStatus
    // =========================================================================
    enum class TaskRunStatus : int {
        RUNNING           = 0,
        SUCCESS           = 1,
        FAILED            = 2,
        FAILED_TIMEOUT    = 3,
        SKIPPED_SAFETY    = 4,   // blocked by whitelist
        SKIPPED_NO_CONFIRM= 5,   // confirmation timed out
        SKIPPED_BUSY      = 6,   // inference was in progress, deferred
        CANCELLED         = 7,
    };

    inline const char* run_status_to_string(TaskRunStatus s) {
        switch (s) {
            case TaskRunStatus::RUNNING:            return "running";
            case TaskRunStatus::SUCCESS:            return "success";
            case TaskRunStatus::FAILED:             return "failed";
            case TaskRunStatus::FAILED_TIMEOUT:     return "failed_timeout";
            case TaskRunStatus::SKIPPED_SAFETY:     return "skipped_safety";
            case TaskRunStatus::SKIPPED_NO_CONFIRM: return "skipped_no_confirm";
            case TaskRunStatus::SKIPPED_BUSY:       return "skipped_busy";
            case TaskRunStatus::CANCELLED:          return "cancelled";
            default:                                return "unknown";
        }
    }

    // =========================================================================
    // TaskActionLog
    // One step within a run (for watch mode / audit).
    // =========================================================================
    struct TaskActionLog {
        int         sequence      = 0;
        std::string action_type;        // tool name or action description
        std::string description;        // human-readable: what was done
        std::string input_summary;      // brief description of inputs
        std::string output_summary;     // brief description of result
        bool        success       = false;
        bool        required_confirmation = false;
        bool        confirmation_granted  = false;
        int         duration_ms   = 0;
        std::string timestamp;
    };

    // =========================================================================
    // TaskRun
    // One execution record for a scheduled task.
    // Persisted to SQLite in task_runs table.
    // =========================================================================
    struct TaskRun {
        std::string   run_id;           // uuid
        std::string   task_id;          // parent task uuid
        std::string   task_name;        // denormalised for display
        TaskRunStatus status = TaskRunStatus::RUNNING;
        std::string   started_at;
        std::string   finished_at;
        std::string   result_summary;   // what the task produced
        std::string   error_message;
        int           duration_ms = 0;
        std::string   output_path;      // if written to file
        std::string   inference_id;     // for audit log linkage (if applicable)
        std::string   session_id;       // internal session used for this run
        std::vector<TaskActionLog> action_log;  // populated in watch mode
    };

    // =========================================================================
    // ScheduledTask
    // The persistent unit. Stored in SQLite. Survives restarts.
    // =========================================================================
    struct ScheduledTask {
        std::string   id;               // uuid
        std::string   name;             // human-readable name
        std::string   description;      // user's original natural language request
        bool          enabled    = true;

        TriggerSpec   trigger;
        TaskAction    action;

        // Execution stats
        int           run_count  = 0;
        int           fail_count = 0;
        std::string   last_run_at;
        std::string   next_run_at;      // pre-computed by engine
        std::string   created_at;
        std::string   updated_at;

        // Provenance
        std::string   created_from;     // "chat" | "api" | "config"
        std::string   created_in_session; // session that created it

        // Safety overrides (per-task, take precedence over global config)
        // If not set (nullopt), global config applies.
        std::optional<bool> allow_file_write;
        std::optional<bool> allow_web_access;
        std::optional<bool> require_confirmation;
        std::optional<bool> full_autonomy;
        std::vector<std::string> allowed_apps_override;  // empty = use global
    };

    // =========================================================================
    // SchedulerStatus
    // Current state of the scheduler engine.
    // =========================================================================
    struct SchedulerStatus {
        bool        enabled           = false;
        bool        running           = false;
        int         total_tasks       = 0;
        int         enabled_tasks     = 0;
        int         total_runs        = 0;
        int         successful_runs   = 0;
        int         failed_runs       = 0;
        std::string current_task_id;    // empty if idle
        std::string current_task_name;
        std::string last_run_at;
        std::string next_scheduled_at;  // soonest upcoming trigger
    };

    // =========================================================================
    // TaskParseResult
    // Output of SchedulerParser::parse().
    // =========================================================================
    struct TaskParseResult {
        bool          success = false;
        ScheduledTask task;
        std::string   error_message;
        std::string   clarification_needed; // non-empty → ask user to clarify
        float         confidence = 0.0f;    // parser's confidence in the result
    };

} // namespace cardinal
