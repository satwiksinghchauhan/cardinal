// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Inference Pipeline
// File: src/core/inference.h
//
// Orchestrates the full two-pass inference cycle:
//   Pass 1: constrained decoding → feeling output
//   Pass 2: free decoding with synthetic turn → final response
//
// Manages retries, memory context injection, and feeds results to verifier.
//
// Change from original: LLMEngine& replaced with ILLMBackend& so the pipeline
// is backend-agnostic. No other logic changes.
// =============================================================================

#include "utils/config_loader.h"
#include "utils/json_parser.h"
#include "core/feeling_output.h"
#include "core/llm_backend.h"           // ILLMBackend, GenerationResult, TokenCallback
#include "memory/episodic_retriever.h"

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <memory>

namespace cardinal {

    // -------------------------------------------------------------------------
    // InferenceRequest
    // -------------------------------------------------------------------------
    struct InferenceRequest {
        std::string              user_message;
        std::vector<ChatMessage> history;
        std::vector<Rule>        active_rules;
        bool                     stream_response;
    };

    // -------------------------------------------------------------------------
    // InferenceResponse
    // -------------------------------------------------------------------------
    struct InferenceResponse {
        std::string              response;
        FeelingOutput            feeling;
        bool                     success;
        std::string              error_message;
        InferenceMetrics         metrics;

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

    // -------------------------------------------------------------------------
    // StreamCallback
    // Return false to abort generation.
    // -------------------------------------------------------------------------
    using StreamCallback = std::function<bool(const std::string& token)>;

    // -------------------------------------------------------------------------
    // InferencePipeline
    // Backend-agnostic two-pass orchestrator.
    // Takes ILLMBackend& — works with any registered backend.
    // -------------------------------------------------------------------------
    class InferencePipeline {
    public:
        // backend must outlive this pipeline (owned by CardinalAPI)
        InferencePipeline(const CardinalConfig& config, ILLMBackend& backend);

        // -------------------------------------------------------------------------
        // Core inference
        // -------------------------------------------------------------------------
        InferenceResponse run(const InferenceRequest& request,
                              const StreamCallback& stream_cb = nullptr);

        // -------------------------------------------------------------------------
        // Context management
        // -------------------------------------------------------------------------
        void reset_context();
        bool context_near_limit() const;

        // -------------------------------------------------------------------------
        // Memory retrieval
        // -------------------------------------------------------------------------
        void set_retriever(EpisodicRetriever* retriever) { retriever_ = retriever; }
        bool has_retriever() const { return retriever_ != nullptr; }

        // -------------------------------------------------------------------------
        // System prompt
        // -------------------------------------------------------------------------
        void        set_system_prompt(const std::string& prompt);
        const std::string& system_prompt() const { return system_prompt_; }

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        int total_inferences()  const { return total_inferences_; }
        int total_retries()     const { return total_retries_; }
        int failed_inferences() const { return failed_inferences_; }

    private:
        std::vector<ChatMessage> build_messages(const InferenceRequest& request) const;
        std::string              format_rules(const std::vector<Rule>& rules) const;
        std::string              format_episodes(const std::vector<RetrievalResult>& results) const;

        bool run_pass1(FeelingContext& ctx, const std::vector<ChatMessage>& messages);
        bool run_pass2(FeelingContext& ctx, const std::vector<ChatMessage>& messages,
                       const StreamCallback& stream_cb);

        InferenceResponse build_response(const FeelingContext& ctx,
                                         bool success,
                                         const std::string& error = "") const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        ILLMBackend&          backend_;          // ← was LLMEngine&
        FeelingContext        feeling_ctx_;
        EpisodicRetriever*   retriever_ = nullptr;

        std::string          system_prompt_;

        int total_inferences_  = 0;
        int total_retries_     = 0;
        int failed_inferences_ = 0;

        static const char* DEFAULT_SYSTEM_PROMPT;
    };

    // -------------------------------------------------------------------------
    // InferenceError
    // -------------------------------------------------------------------------
    class InferenceError : public std::runtime_error {
    public:
        explicit InferenceError(const std::string& message)
            : std::runtime_error("InferenceError: " + message) {}
    };

} // namespace cardinal
