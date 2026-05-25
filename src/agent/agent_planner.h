#pragma once
// =============================================================================
// Cardinal - Agent Planner
// File: src/agent/agent_planner.h
//
// Decomposes a high-level goal into a structured AgentPlan using the LLM.
// Uses the same ILLMBackend as inference — no separate model needed.
//
// The planner:
//   1. Prompts the LLM to decompose the goal into discrete steps
//   2. Parses the structured JSON plan from the response
//   3. Validates the plan against available tools
//   4. Returns AgentPlan for execution by AgentExecutor
// =============================================================================

#include "agent/agent_types.h"
#include "tools/tool_registry.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>

namespace cardinal {

    // Forward declaration
    class ILLMBackend;

    class AgentPlanner {
    public:
        AgentPlanner(const CardinalConfig& config,
                     ILLMBackend&          backend,
                     const ToolRegistry&   registry);

        // ------------------------------------------------------------------
        // Decompose a goal into an AgentPlan.
        // Returns a valid plan or plan.valid=false on failure.
        // ------------------------------------------------------------------
        AgentPlan decompose(const std::string& goal,
                            const std::string& context_hints = "") const;

        // ------------------------------------------------------------------
        // Re-plan from current state (called when plan needs updating mid-task)
        // ------------------------------------------------------------------
        AgentPlan replan(const std::string& goal,
                         const std::string& progress_summary,
                         const std::string& failure_reason) const;

    private:
        // Build the planning prompt
        std::string build_planning_prompt(const std::string& goal,
                                          const std::string& context,
                                          bool is_replan) const;

        // Parse LLM output into AgentPlan
        AgentPlan parse_plan(const std::string& llm_output,
                             const std::string& goal) const;

        const CardinalConfig& config_;
        ILLMBackend&          backend_;
        const ToolRegistry&   registry_;
    };

} // namespace cardinal
