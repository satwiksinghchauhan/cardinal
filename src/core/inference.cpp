// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Inference Pipeline Implementation
// File: src/core/inference.cpp
//
// Changes from original:
//   - #include "core/llm_engine.h" removed (LLMEngine no longer exists)
//   - #include "core/llm_backend.h" added (ILLMBackend lives here)
//   - Constructor: LLMEngine& engine → ILLMBackend& backend, engine_ → backend_
//   - All pipeline logic is identical — zero behavioural changes
// =============================================================================

#include "inference.h"
#include "core/llm_backend.h"               // ILLMBackend, GenerationResult, TokenCallback
#include "utils/logger.h"
#include "memory/episodic_retriever.h"

#include <thread>
#include <chrono>
#include <sstream>

namespace cardinal {

    // =========================================================================
    // Default system prompt
    // =========================================================================

    const char* InferencePipeline::DEFAULT_SYSTEM_PROMPT =
        "You are Cardinal, a neurosymbolic AI assistant. "
        "You reason carefully before responding. "
        "You are honest about uncertainty and flag contradictions when you detect them. "
        "When you encounter patterns that suggest a general rule, you signal this. "
        "Your responses are precise, thoughtful, and grounded in logic.";

    // =========================================================================
    // Constructor
    // =========================================================================

    InferencePipeline::InferencePipeline(const CardinalConfig& config,
                                         ILLMBackend&          backend)
        : config_(config)
        , backend_(backend)
        , feeling_ctx_(config)
        , system_prompt_(DEFAULT_SYSTEM_PROMPT)
    {
        LOG_INFO("InferencePipeline initialized");
    }

    // =========================================================================
    // run — full two-pass inference cycle
    // =========================================================================

    InferenceResponse InferencePipeline::run(const InferenceRequest& request,
                                              const StreamCallback&   stream_cb)
    {
        ++total_inferences_;

        LOG_INFO("Inference #" + std::to_string(total_inferences_) +
                 " - user: \"" + request.user_message.substr(
                     0, std::min(50, (int)request.user_message.size())) + "...\"");

        feeling_ctx_.reset();
        auto messages = build_messages(request);

        // ---------------------------------------------------------------------
        // Pass 1: feeling output (constrained / retry loop depending on backend)
        // ---------------------------------------------------------------------
        bool pass1_ok = run_pass1(feeling_ctx_, messages);

        if (!pass1_ok) {
            ++failed_inferences_;
            LOG_WARN("Pass 1 failed after " +
                     std::to_string(config_.feedback.max_retries) + " retries");
            return build_response(feeling_ctx_, false,
                "Pass 1 failed: could not generate valid feeling output");
        }

        LOG_DEBUG("Pass 1 success: " + feeling_ctx_.feeling().to_string());

        // ---------------------------------------------------------------------
        // Pass 2: final response (free decoding, synthetic turn injected)
        // ---------------------------------------------------------------------
        bool pass2_ok = run_pass2(feeling_ctx_, messages, stream_cb);

        if (!pass2_ok) {
            ++failed_inferences_;
            LOG_WARN("Pass 2 failed");
            return build_response(feeling_ctx_, false,
                "Pass 2 failed: could not generate response");
        }

        LOG_INFO("Inference complete - " + feeling_ctx_.metrics().to_string());
        return build_response(feeling_ctx_, true);
    }

    // =========================================================================
    // run_pass1
    // =========================================================================

    bool InferencePipeline::run_pass1(FeelingContext&                 ctx,
                                       const std::vector<ChatMessage>& messages)
    {
        while (true) {
            auto result = backend_.generate_feeling(ctx, messages);

            if (result.success && ctx.has_valid_feeling())
                return true;

            if (!ctx.should_retry()) {
                LOG_WARN("Pass 1 exhausted retries");
                return false;
            }

            ctx.increment_retry();
            ++total_retries_;

            if (config_.feedback.retry_delay_ms > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config_.feedback.retry_delay_ms));
            }

            LOG_DEBUG("Pass 1 retry " + std::to_string(ctx.retry_count()));
        }
    }

    // =========================================================================
    // run_pass2
    // =========================================================================

    bool InferencePipeline::run_pass2(FeelingContext&                 ctx,
                                       const std::vector<ChatMessage>& messages,
                                       const StreamCallback&           stream_cb)
    {
        TokenCallback token_cb = nullptr;
        if (stream_cb) {
            token_cb = [&stream_cb](const std::string& token_text,
                                    int /*token_id*/,
                                    int /*tokens_generated*/) -> bool {
                return stream_cb(token_text);
            };
        }

        auto result = backend_.generate_response(ctx, messages, token_cb);
        return result.success && ctx.has_response();
    }

    // =========================================================================
    // build_messages
    // =========================================================================

    std::vector<ChatMessage> InferencePipeline::build_messages(
        const InferenceRequest& request) const
    {
        std::vector<ChatMessage> messages;

        std::string system_content = system_prompt_;
        if (!request.active_rules.empty()) {
            system_content += "\n\n" + format_rules(request.active_rules);
        }
        messages.push_back({ "system", system_content });

        // Phase 6: Memory context injection
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
                LOG_WARN("Prompt injection: retrieval failed: " +
                         std::string(e.what()) + " -- continuing without context");
            }
        }

        for (const auto& turn : request.history)
            messages.push_back(turn);

        messages.push_back({ "user", request.user_message });
        return messages;
    }

    // =========================================================================
    // format_rules
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

    // =========================================================================
    // format_episodes
    // =========================================================================

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
        const std::string&    error) const
    {
        InferenceResponse resp;
        resp.success       = success;
        resp.error_message = error;
        resp.metrics       = ctx.metrics();

        if (success && ctx.has_valid_feeling()) {
            resp.feeling  = ctx.feeling();
            resp.response = ctx.final_response();
        }
        return resp;
    }

    // =========================================================================
    // Context management
    // =========================================================================

    void InferencePipeline::reset_context() {
        backend_.clear_kv_cache();
        feeling_ctx_.reset();
        LOG_INFO("Pipeline context reset");
    }

    bool InferencePipeline::context_near_limit() const {
        return backend_.context_near_limit();
    }

    // =========================================================================
    // System prompt
    // =========================================================================

    void InferencePipeline::set_system_prompt(const std::string& prompt) {
        system_prompt_ = prompt;
        LOG_INFO("System prompt updated (" +
                 std::to_string(prompt.size()) + " chars)");
    }

} // namespace cardinal
