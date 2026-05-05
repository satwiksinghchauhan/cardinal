// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Agent Executor
// File: src/agent/agent_executor.h
//
// Core agentic execution loop.
//
// For each step in the plan:
//   1. Prompt LLM with step description + working memory context
//   2. Detect tool calls in output
//   3. Execute tools (with confirmation if required)
//   4. Store results in working memory
//   5. Check if step succeeded; self-correct if not
//   6. Update TraceBuilder with step record
//   7. Check goal achievement after each step
//
// After all steps (or on max_iterations):
//   - Synthesize final response from working memory + step history
//   - Return AgentResult
//
// Thread-safe per session — concurrent sessions use separate AgentContexts.
// =============================================================================

#include "agent/agent_types.h"
#include "agent/agent_context.h"
#include "agent/agent_planner.h"
#include "tools/tool_executor.h"
#include "tools/tool_registry.h"
#include "explainability/trace_builder.h"
#include "utils/config_loader.h"

#include <string>
#include <functional>
#include <memory>

namespace cardinal {

    // Forward declarations
    class ILLMBackend;
    class InferencePipeline;

    // -------------------------------------------------------------------------
    // AgentStreamCallback
    // Called during agent execution to stream intermediate progress.
    // Different from the token-level StreamCallback — this is step-level.
    // -------------------------------------------------------------------------
    struct AgentProgressUpdate {
        int         step_index;
        std::string step_description;
        std::string status;      // "started" | "tool_called" | "completed" | "failed"
        std::string detail;      // brief detail for the update
    };
    using AgentProgressCallback = std::function<void(const AgentProgressUpdate&)>;

    // -------------------------------------------------------------------------
    // AgentExecutor
    // -------------------------------------------------------------------------
    class AgentExecutor {
    public:
        AgentExecutor(const CardinalConfig& config,
                      ILLMBackend&          backend,
                      const ToolRegistry&   registry,
                      const ToolExecutor&   tool_executor);

        // ------------------------------------------------------------------
        // Run a complete agentic task from goal to result.
        // ------------------------------------------------------------------
        AgentResult run(const AgentGoal&       goal,
                        TraceBuilder&          trace_builder,
                        AgentProgressCallback  progress_cb = nullptr);

        // ------------------------------------------------------------------
        // Set confirmation callback (forwarded to ToolExecutor)
        // ------------------------------------------------------------------
        void set_confirmation_callback(ConfirmationCallback cb) {
            confirmation_cb_ = std::move(cb);
        }

    private:
        // ------------------------------------------------------------------
        // Internal execution
        // ------------------------------------------------------------------

        // Execute a single plan step. Returns updated step.
        AgentStep execute_step(AgentStep&          step,
                               AgentContext&        ctx,
                               const AgentPlan&     plan,
                               TraceBuilder&        trace_builder);

        // Prompt LLM to act on a step given current context
        std::string think_for_step(const AgentStep&   step,
                                   const AgentContext& ctx,
                                   const AgentPlan&   plan) const;

        // Check if the goal has been achieved based on step history
        bool check_goal_achieved(const AgentGoal&    goal,
                                  const AgentContext& ctx,
                                  const AgentPlan&    plan) const;

        // Synthesize final response from completed plan + working memory
        std::string synthesize_response(const AgentGoal&    goal,
                                         const AgentContext& ctx,
                                         const AgentPlan&    plan,
                                         bool                goal_achieved) const;

        // Self-correction: prompt LLM to retry a failed step differently
        AgentStep self_correct(AgentStep&          step,
                               AgentContext&        ctx,
                               const std::string&  failure_reason,
                               TraceBuilder&        trace_builder);

        // Build step prompt with full context
        std::string build_step_prompt(const AgentStep&   step,
                                       const AgentContext& ctx,
                                       const AgentPlan&   plan,
                                       bool is_correction = false,
                                       const std::string& correction_hint = "") const;

        // Notify progress
        void notify(AgentProgressCallback& cb,
                    int step_index,
                    const std::string& description,
                    const std::string& status,
                    const std::string& detail = "") const;

        // ------------------------------------------------------------------
        // Members
        // ------------------------------------------------------------------
        const CardinalConfig& config_;
        ILLMBackend&          backend_;
        const ToolRegistry&   registry_;
        const ToolExecutor&   tool_executor_;
        AgentPlanner          planner_;
        ConfirmationCallback  confirmation_cb_;
    };

} // namespace cardinal
