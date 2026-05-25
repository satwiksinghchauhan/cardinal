#pragma once
// =============================================================================
// Cardinal - Agent Types
// File: src/agent/agent_types.h
//
// Shared types for the agentic execution system.
// No dependencies on other Cardinal headers — safe to include anywhere.
// =============================================================================

#include <string>
#include <vector>
#include <optional>

namespace cardinal {

    // -------------------------------------------------------------------------
    // AgentStepStatus
    // -------------------------------------------------------------------------
    enum class AgentStepStatus {
        PENDING,
        IN_PROGRESS,
        COMPLETED,
        FAILED,
        SKIPPED
    };

    inline std::string step_status_to_string(AgentStepStatus s) {
        switch (s) {
            case AgentStepStatus::PENDING:     return "pending";
            case AgentStepStatus::IN_PROGRESS: return "in_progress";
            case AgentStepStatus::COMPLETED:   return "completed";
            case AgentStepStatus::FAILED:      return "failed";
            case AgentStepStatus::SKIPPED:     return "skipped";
            default:                           return "unknown";
        }
    }

    // -------------------------------------------------------------------------
    // AgentStep
    // One discrete step in an agent's plan.
    // -------------------------------------------------------------------------
    struct AgentStep {
        int         index       = 0;
        std::string description;    // what needs to be done
        std::string rationale;      // why this step is needed
        AgentStepStatus status  = AgentStepStatus::PENDING;

        // Set after execution
        std::string result_summary;
        bool        used_tool   = false;
        std::string tool_used;
        int         duration_ms = 0;
        int         retries     = 0;
    };

    // -------------------------------------------------------------------------
    // AgentPlan
    // The decomposed plan for a goal.
    // -------------------------------------------------------------------------
    struct AgentPlan {
        std::string              goal;
        std::vector<AgentStep>   steps;
        std::string              strategy;    // high-level approach
        float                    confidence = 0.0f;  // model's confidence in plan
        bool                     valid      = false;

        int pending_count() const {
            int n = 0;
            for (const auto& s : steps)
                if (s.status == AgentStepStatus::PENDING) ++n;
            return n;
        }

        int completed_count() const {
            int n = 0;
            for (const auto& s : steps)
                if (s.status == AgentStepStatus::COMPLETED) ++n;
            return n;
        }
    };

    // -------------------------------------------------------------------------
    // WorkingMemoryEntry
    // A single entry in the agent's scratchpad.
    // Persisted to SQLite so tasks can be resumed.
    // -------------------------------------------------------------------------
    struct WorkingMemoryEntry {
        std::string key;            // unique key for this entry
        std::string value;          // serialized content
        std::string entry_type;     // "observation" | "result" | "note" | "error"
        std::string step_index;     // which step produced this
        std::string timestamp;
    };

    // -------------------------------------------------------------------------
    // AgentGoal
    // Input to the agentic pipeline.
    // -------------------------------------------------------------------------
    struct AgentGoal {
        std::string session_id;
        std::string goal;           // high-level goal description
        int         max_iterations = 0;  // 0 = use config default
        bool        stream         = false;
        std::vector<std::string> context_hints; // optional caller-provided hints
    };

    // -------------------------------------------------------------------------
    // AgentResult
    // Output of a completed agentic execution.
    // -------------------------------------------------------------------------
    struct AgentResult {
        // Outcome
        bool        goal_achieved   = false;
        std::string final_response;     // synthesized answer/report
        std::string failure_reason;     // set if goal not achieved

        // Execution summary
        int         iterations_used = 0;
        int         tools_called    = 0;
        int         steps_completed = 0;
        int         steps_total     = 0;
        int         total_tokens    = 0;
        int         total_ms        = 0;

        // Plan reference
        AgentPlan   plan;

        // Working memory snapshot (for audit)
        std::vector<WorkingMemoryEntry> memory_snapshot;
    };

} // namespace cardinal
