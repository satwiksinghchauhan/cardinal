// =============================================================================
// Cardinal - Inference Pipeline Implementation (v1.2.0)
// File: src/core/inference.cpp
//
// Changes from v1.1.0:
//   - Tool loop added to Pass 2
//   - TraceBuilder used throughout
//   - Audit log appended at end of each inference
//   - Extra tools registered per-request then cleaned up
// =============================================================================

#include "inference.h"
#include "core/llm_backend.h"
#include "utils/logger.h"
#include "memory/episodic_retriever.h"
#include "tools/tool_executor.h"
#include "tools/tool_registry.h"
#include "explainability/trace_builder.h"
#include "explainability/audit_log.h"

#include <thread>
#include <chrono>
#include <sstream>

namespace cardinal {

    const char* InferencePipeline::DEFAULT_SYSTEM_PROMPT =
        "You are Cardinal, a neurosymbolic AI assistant. "
        "You reason carefully before responding. "
        "You are honest about uncertainty and flag contradictions when you detect them. "
        "When you encounter patterns that suggest a general rule, you signal this. "
        "Your responses are precise, thoughtful, and grounded in logic.";

    InferencePipeline::InferencePipeline(const CardinalConfig& config,
                                         ILLMBackend&          backend)
        : config_(config)
        , backend_(backend)
        , feeling_ctx_(config)
        , system_prompt_(DEFAULT_SYSTEM_PROMPT)
    {
        LOG_INFO("InferencePipeline initialized (v1.2.0)");
    }

    // =========================================================================
    // run
    // =========================================================================

    InferenceResponse InferencePipeline::run(const InferenceRequest& request,
                                              const StreamCallback&   stream_cb)
    {
        ++total_inferences_;

        LOG_INFO("Inference #" + std::to_string(total_inferences_) +
                 " - user: \"" + request.user_message.substr(
                     0, std::min(50, (int)request.user_message.size())) + "...\"");

        // Start trace
        auto info = backend_.get_info();
        TraceBuilder trace_builder(
            "",  // session_id set by caller via ChatResponse
            info.name,
            info.model_name);

        trace_builder.record_query(
            request.agent_mode ? request.goal : request.user_message,
            request.agent_mode,
            request.goal);

        trace_builder.record_active_rules(request.active_rules);

        // Register per-request extra tools
        if (tool_registry_ && !request.extra_tools.empty()) {
            tool_registry_->register_tools(request.extra_tools);
        }

        feeling_ctx_.reset();

        // Build messages
        auto messages = build_messages(request);

        // -------------------------------------------------------------------------
        // Pass 1: feeling output
        // -------------------------------------------------------------------------
        trace_builder.record_pass1_start();
        bool pass1_ok = run_pass1(feeling_ctx_, messages, trace_builder);

        if (!pass1_ok) {
            ++failed_inferences_;
            LOG_WARN("Pass 1 failed");
            auto trace = trace_builder.finalize();
            if (audit_log_) audit_log_->append(trace);
            return build_response(feeling_ctx_, false, std::move(trace),
                "Pass 1 failed: could not generate valid feeling output");
        }

        LOG_DEBUG("Pass 1 success: " + feeling_ctx_.feeling().to_string());

        // -------------------------------------------------------------------------
        // Pass 2: response with tool loop
        // -------------------------------------------------------------------------
        trace_builder.record_pass2_start();

        // messages is passed by value so the tool loop can append to it
        auto mutable_messages = messages;
        bool pass2_ok = run_pass2(feeling_ctx_, mutable_messages,
                                   request, stream_cb, trace_builder);

        if (!pass2_ok) {
            ++failed_inferences_;
            LOG_WARN("Pass 2 failed");
            auto trace = trace_builder.finalize();
            if (audit_log_) audit_log_->append(trace);
            return build_response(feeling_ctx_, false, std::move(trace),
                "Pass 2 failed: could not generate response");
        }

        // Finalize trace
        trace_builder.record_pass2_complete(
            feeling_ctx_.final_response(),
            feeling_ctx_.metrics().pass2_tokens_generated);

        LOG_INFO("Inference complete - " + feeling_ctx_.metrics().to_string());

        auto trace = trace_builder.finalize();
        if (audit_log_) audit_log_->append(trace);

        // Clean up per-request extra tools
        if (tool_registry_) {
            for (const auto& t : request.extra_tools)
                tool_registry_->unregister_tool(t.name);
        }

        return build_response(feeling_ctx_, true, std::move(trace));
    }

    // =========================================================================
    // run_pass1
    // =========================================================================

    bool InferencePipeline::run_pass1(FeelingContext&                 ctx,
                                       const std::vector<ChatMessage>& messages,
                                       TraceBuilder&                   trace_builder)
    {
        int retries = 0;
        while (true) {
            auto result = backend_.generate_feeling(ctx, messages);

            if (result.success && ctx.has_valid_feeling()) {
                trace_builder.record_pass1_complete(
                    ctx.feeling(), true, retries,
                    result.tokens_generated);
                return true;
            }

            if (!ctx.should_retry()) {
                trace_builder.record_pass1_complete(
                    FeelingOutput{}, false, retries, 0);
                LOG_WARN("Pass 1 exhausted retries");
                return false;
            }

            ctx.increment_retry();
            ++total_retries_;
            ++retries;

            if (config_.feedback.retry_delay_ms > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config_.feedback.retry_delay_ms));
            }

            LOG_DEBUG("Pass 1 retry " + std::to_string(ctx.retry_count()));
        }
    }

    // =========================================================================
    // run_pass2 — with tool loop
    // =========================================================================

    bool InferencePipeline::run_pass2(FeelingContext&          ctx,
                                       std::vector<ChatMessage>& messages,
                                       const InferenceRequest&   request,
                                       const StreamCallback&     stream_cb,
                                       TraceBuilder&             trace_builder)
    {
        int max_tool_iter = config_.agent.max_iterations;
        int tool_iter     = 0;

        // Inject synthetic turn (feeling state) before generation
        ctx.prepare_synthetic_turn();

        while (true) {
            // Generate response
            TokenCallback token_cb = nullptr;
            if (stream_cb) {
                token_cb = [&stream_cb](const std::string& token_text,
                                        int, int) -> bool {
                    return stream_cb(token_text);
                };
            }

            auto result = backend_.generate_response(ctx, messages, token_cb);
            if (!result.success) return false;

            // Check for tool calls if tools are enabled
            if (request.tools_enabled && tool_executor_ &&
                tool_iter < max_tool_iter)
            {
                auto detected = tool_executor_->detect_tool_calls(result.text);

                if (detected.has_tool_calls) {
                    ++tool_iter;
                    trace_builder.record_tool_iteration();

                    // Execute all tool calls
                    auto tool_results = tool_executor_->execute_all(
                        detected.tool_calls);

                    for (const auto& tr : tool_results)
                        trace_builder.record_tool_call(tr);

                    // Inject tool results back into context
                    std::string tool_context =
                        tool_executor_->format_results_for_context(tool_results);

                    // Append tool results as assistant + user turn
                    messages.push_back({ "assistant", result.text });
                    messages.push_back({ "user",
                        tool_context +
                        "\nContinue your response based on these tool results." });

                    // Check if any tool requires confirmation (pending)
                    bool has_pending = false;
                    for (const auto& tr : tool_results) {
                        if (tr.pending()) { has_pending = true; break; }
                    }
                    if (has_pending) {
                        // Pause — store partial response and return
                        ctx.set_final_response(detected.text_before);
                        ctx.set_state(PassState::COMPLETE);
                        return true;
                    }

                    // Loop back — generate response with tool results
                    continue;
                }
            }

            // No tool calls (or tools disabled) — final response
            ctx.set_final_response(result.text);
            ctx.set_state(PassState::COMPLETE);
            ctx.metrics().pass2_tokens_generated = result.tokens_generated;
            return ctx.has_response();
        }
    }

    // =========================================================================
    // build_messages
    // =========================================================================

    std::vector<ChatMessage> InferencePipeline::build_messages(
        const InferenceRequest& request) const
    {
        std::vector<ChatMessage> messages;

        std::string system_content = system_prompt_;

        // Inject active rules
        if (!request.active_rules.empty()) {
            system_content += "\n\n" + format_rules(request.active_rules);
        }

        // Inject tool definitions
        if (request.tools_enabled && tool_registry_) {
            auto tools = tool_registry_->get_enabled_tools();
            if (!request.extra_tools.empty()) {
                for (const auto& t : request.extra_tools)
                    tools.push_back(t);
            }
            if (!tools.empty()) {
                system_content += "\n\n" +
                    tool_registry_->format_tools_for_prompt(tools);
            }
        }

        messages.push_back({ "system", system_content });

        // Memory context injection
        if (retriever_ != nullptr && retriever_->index_ready()) {
            try {
                auto results = retriever_->retrieve(request.user_message);
                if (!results.empty()) {
                    std::string context_block = format_episodes(results);
                    if (!context_block.empty()) {
                        messages.push_back({
                            "user",
                            "[MEMORY CONTEXT]\n" + context_block +
                            "\n[END MEMORY CONTEXT]"
                        });
                        messages.push_back({
                            "assistant",
                            "I have reviewed the relevant past context "
                            "and will use it to inform my response."
                        });
                        LOG_DEBUG("Injected " + std::to_string(results.size()) +
                                  " episodes into prompt context");
                    }
                }
            } catch (const std::exception& e) {
                LOG_WARN("Prompt injection failed: " + std::string(e.what()));
            }
        }

        for (const auto& turn : request.history)
            messages.push_back(turn);

        messages.push_back({ "user", request.user_message });

        // -------------------------------------------------------------------------
        // Context budget trimming
        // Ensures the prompt never exceeds the safe generation budget regardless
        // of hardware. Trims oldest history turns first, preserving:
        //   - messages[0]: system prompt (never removed)
        //   - messages[1..2]: memory context pair (never removed)
        //   - messages.back(): current user message (never removed)
        //
        // Budget = 70% of context length, leaving 30% for generation output.
        // Token estimate: 1 token per 3.5 chars (conservative for chat templates).
        // This works on any hardware — on a 128K context GPU it simply never fires.
        // -------------------------------------------------------------------------
        const int ctx_len = backend_.get_info().context_length;
        if (ctx_len > 0) {
            const int budget = static_cast<int>(ctx_len * 0.70f);

            auto estimate_tokens = [](const std::vector<ChatMessage>& msgs) -> int {
                int total = 0;
                for (const auto& m : msgs)
                    total += static_cast<int>(m.content.size() / 3.5f) + 4; // +4 for role tokens
                return total;
            };

            // Find the index of the first trimable turn.
            // messages[0] = system (fixed)
            // messages[1..2] = memory context pair if injected (we skip these too)
            // messages[3..N-1] = conversation history (trimmable)
            // messages[N] = current user message (fixed)
            int trim_start = 1;
            // Skip past memory context pair if present
            if (messages.size() > 2 &&
                messages[1].role == "user" &&
                messages[1].content.find("[MEMORY CONTEXT]") != std::string::npos) {
                trim_start = 3;
            }

            int iterations = 0;
            const int max_trim_iterations = 50; // safety cap

            while (estimate_tokens(messages) > budget &&
                   static_cast<int>(messages.size()) > trim_start + 2 &&
                   iterations < max_trim_iterations)
            {
                // Remove the oldest trimmable turn (user or assistant)
                messages.erase(messages.begin() + trim_start);
                ++iterations;
            }

            if (iterations > 0) {
                LOG_DEBUG("Context trimming: removed " + std::to_string(iterations) +
                          " history turns to fit " + std::to_string(budget) +
                          " token budget (ctx=" + std::to_string(ctx_len) + ")");
            }
        }

        return messages;
    }

    // =========================================================================
    // format_rules / format_episodes (unchanged from v1.1.0)
    // =========================================================================

    std::string InferencePipeline::format_rules(
        const std::vector<Rule>& rules) const
    {
        if (rules.empty()) return "";
        std::ostringstream oss;
        oss << "## Active Rules\n";
        oss << "The following rules have been derived from prior reasoning. "
               "Apply them when relevant:\n\n";
        for (size_t i = 0; i < rules.size(); ++i) {
            const auto& rule = rules[i];
            oss << (i + 1) << ". [" << rule.domain << "] "
                << "IF " << rule.condition
                << " THEN " << rule.consequence
                << " (confidence: " << static_cast<int>(rule.confidence * 100) << "%)\n";
        }
        return oss.str();
    }

    std::string InferencePipeline::format_episodes(
        const std::vector<RetrievalResult>& results) const
    {
        if (results.empty()) return "";
        std::ostringstream oss;
        oss << "The following past interactions are relevant to the current query.\n"
               "Use them as reference context where appropriate:\n\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r  = results[i];
            const auto& ep = r.episode;
            int conf_pct   = static_cast<int>(ep.confidence * 100);
            oss << (i + 1) << ". ["
                << ep.reasoning_domain
                << " | confidence: " << conf_pct << "%"
                << " | score: " << static_cast<int>(r.score * 100) << "%"
                << "]\n";
            oss << "   Q: " << ep.user_message << "\n";
            std::string summary = ep.response_summary;
            const size_t MAX_SUMMARY = 300;
            if (summary.size() > MAX_SUMMARY) {
                size_t cutoff = summary.rfind('.', MAX_SUMMARY);
                if (cutoff == std::string::npos || cutoff < MAX_SUMMARY / 2)
                    cutoff = MAX_SUMMARY;
                summary = summary.substr(0, cutoff + 1) + "...";
            }
            oss << "   A: " << summary << "\n";
            if (i + 1 < results.size()) oss << "\n";
        }
        return oss.str();
    }

    // =========================================================================
    // build_response
    // =========================================================================

    InferenceResponse InferencePipeline::build_response(
        const FeelingContext& ctx,
        bool                  success,
        ReasoningTrace        trace,
        const std::string&    error) const
    {
        InferenceResponse resp;
        resp.success       = success;
        resp.error_message = error;
        resp.metrics       = ctx.metrics();
        resp.trace         = std::move(trace);

        if (success && ctx.has_valid_feeling()) {
            resp.feeling  = ctx.feeling();
            resp.response = ctx.final_response();
        }
        return resp;
    }

    // =========================================================================
    // Context management / system prompt (unchanged)
    // =========================================================================

    void InferencePipeline::reset_context() {
        backend_.clear_kv_cache();
        feeling_ctx_.reset();
        LOG_INFO("Pipeline context reset");
    }

    bool InferencePipeline::context_near_limit() const {
        return backend_.context_near_limit();
    }

    void InferencePipeline::set_system_prompt(const std::string& prompt) {
        system_prompt_ = prompt;
        LOG_INFO("System prompt updated (" +
                 std::to_string(prompt.size()) + " chars)");
    }

} // namespace cardinal
