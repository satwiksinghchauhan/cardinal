#pragma once
// =============================================================================
// Cardinal - Inference Pipeline (v1.2.0)
// File: src/core/inference.h
//
// Changes from v1.1.0:
//   - InferenceRequest: agent_mode, goal, max_iterations, extra_tools,
//     tools_enabled fields added
//   - InferenceResponse: trace field added (ReasoningTrace)
//   - InferencePipeline: tool loop in Pass 2, trace building, agent routing
//   - set_tool_executor() and set_audit_log() injection points added
// =============================================================================

#include "utils/config_loader.h"
#include "utils/json_parser.h"
#include "core/feeling_output.h"
#include "core/llm_backend.h"
#include "memory/episodic_retriever.h"
#include "tools/tool_result.h"
#include "tools/tool_registry.h"
#include "explainability/reasoning_trace.h"

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <memory>

namespace cardinal {

    // Forward declarations
    class ToolExecutor;
    class AuditLog;
    class TraceBuilder;

    // -------------------------------------------------------------------------
    // InferenceRequest
    // -------------------------------------------------------------------------
    struct InferenceRequest {
        // Core
        std::string              user_message;
        std::vector<ChatMessage> history;
        std::vector<Rule>        active_rules;
        bool                     stream_response = false;

        // Tools
        bool                         tools_enabled = true;
        std::vector<ToolDefinition>  extra_tools;

        // Agent mode
        bool        agent_mode     = false;
        std::string goal;
        int         max_iterations = 0;
    };

    // -------------------------------------------------------------------------
    // InferenceResponse
    // -------------------------------------------------------------------------
    struct InferenceResponse {
        std::string      response;
        FeelingOutput    feeling;
        bool             success       = false;
        std::string      error_message;
        InferenceMetrics metrics;
        ReasoningTrace   trace;

        bool needs_rule_extraction() const {
            return success && feeling.rule_candidate_signal;
        }
        bool has_contradiction() const {
            return success && feeling.contradiction_flag;
        }
        bool is_uncertain() const {
            return success && feeling.uncertainty_flag;
        }
    };

    using StreamCallback = std::function<bool(const std::string& token)>;

    // -------------------------------------------------------------------------
    // InferencePipeline
    // -------------------------------------------------------------------------
    class InferencePipeline {
    public:
        InferencePipeline(const CardinalConfig& config, ILLMBackend& backend);

        void set_retriever(EpisodicRetriever* retriever) { retriever_     = retriever; }
        void set_tool_executor(ToolExecutor* executor)   { tool_executor_ = executor; }
        void set_tool_registry(ToolRegistry* registry)   { tool_registry_ = registry; }
        void set_audit_log(AuditLog* audit_log)          { audit_log_     = audit_log; }

        bool has_retriever()     const { return retriever_     != nullptr; }
        bool has_tool_executor() const { return tool_executor_ != nullptr; }

        InferenceResponse run(const InferenceRequest& request,
                              const StreamCallback&   stream_cb = nullptr);

        void reset_context();
        bool context_near_limit() const;

        void               set_system_prompt(const std::string& prompt);
        const std::string& system_prompt() const { return system_prompt_; }

        int total_inferences()  const { return total_inferences_; }
        int total_retries()     const { return total_retries_; }
        int failed_inferences() const { return failed_inferences_; }

    private:
        std::vector<ChatMessage> build_messages(
            const InferenceRequest& request) const;

        std::string format_rules(const std::vector<Rule>& rules) const;
        std::string format_episodes(const std::vector<RetrievalResult>& results) const;

        bool run_pass1(FeelingContext& ctx,
                       const std::vector<ChatMessage>& messages,
                       TraceBuilder& trace_builder);

        bool run_pass2(FeelingContext& ctx,
                       std::vector<ChatMessage>& messages,
                       const InferenceRequest& request,
                       const StreamCallback& stream_cb,
                       TraceBuilder& trace_builder);

        InferenceResponse build_response(const FeelingContext& ctx,
                                          bool success,
                                          ReasoningTrace trace,
                                          const std::string& error = "") const;

        const CardinalConfig& config_;
        ILLMBackend&          backend_;
        FeelingContext        feeling_ctx_;

        EpisodicRetriever*   retriever_     = nullptr;
        ToolExecutor*        tool_executor_ = nullptr;
        ToolRegistry*        tool_registry_ = nullptr;
        AuditLog*            audit_log_     = nullptr;

        std::string          system_prompt_;

        int total_inferences_  = 0;
        int total_retries_     = 0;
        int failed_inferences_ = 0;

        static const char* DEFAULT_SYSTEM_PROMPT;
    };

    class InferenceError : public std::runtime_error {
    public:
        explicit InferenceError(const std::string& message)
            : std::runtime_error("InferenceError: " + message) {}
    };

} // namespace cardinal
