#pragma once
// =============================================================================
// Cardinal - Trace Builder
// File: src/explainability/trace_builder.h
//
// Incrementally builds a ReasoningTrace during a single inference cycle.
// InferencePipeline calls record_* methods as each stage completes.
// finalize() computes totals and hands off to AuditLog for signing + storage.
//
// One TraceBuilder instance per inference cycle — not thread-shared.
// =============================================================================

#include "explainability/reasoning_trace.h"
#include "core/feeling_output.h"
#include "tools/tool_result.h"
#include "memory/rule.h"

#include <string>
#include <vector>
#include <chrono>

namespace cardinal {

    // Forward declarations
    struct RetrievalResult;
    struct ConsistencyCheckResult;

    class TraceBuilder {
    public:
        // Start a new trace for this inference cycle
        explicit TraceBuilder(const std::string& session_id,
                              const std::string& backend_type,
                              const std::string& model_name);

        // ------------------------------------------------------------------
        // Record methods — called by InferencePipeline in order
        // ------------------------------------------------------------------

        void record_query(const std::string& query, bool agent_mode = false,
                          const std::string& goal = "");

        void record_active_rules(const std::vector<Rule>& rules);

        void record_retrieved_episodes(
            const std::vector<RetrievalResult>& results);

        // Pass 1
        void record_pass1_start();
        void record_pass1_complete(const FeelingOutput& feeling,
                                   bool valid,
                                   int retries,
                                   int tokens);

        // Tool calls
        void record_tool_call(const ToolResult& result);
        void record_tool_iteration();

        // Symbolic check
        void record_symbolic_check(const ConsistencyCheckResult& result);

        // Rule extraction
        void record_rule_committed(const std::string& rule_id,
                                   const Rule& rule);

        // Agent steps
        void record_agent_step(const AgentStepRecord& step);
        void record_agent_complete(bool goal_achieved, int iterations);

        // Pass 2
        void record_pass2_start();
        void record_pass2_complete(const std::string& response, int tokens);

        // Episode ID (set after storage)
        void set_episode_id(const std::string& id) { trace_.episode_id = id; }

        // ------------------------------------------------------------------
        // Finalize
        // Computes totals. Returns completed trace ready for signing + storage.
        // ------------------------------------------------------------------
        ReasoningTrace finalize();

        // Access in-progress trace (read-only)
        const ReasoningTrace& trace() const { return trace_; }

    private:
        ReasoningTrace trace_;

        // Timing
        std::chrono::steady_clock::time_point inference_start_;
        std::chrono::steady_clock::time_point pass1_start_;
        std::chrono::steady_clock::time_point pass2_start_;
    };

} // namespace cardinal
