// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Inference Pipeline
// File: src/core/inference.h
// Orchestrates the full two-pass inference cycle:
//   Pass 1: constrained decoding -> feeling output
//   Pass 2: free decoding with synthetic turn -> final response
// Manages retries, context, and feeds results to the verifier.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

// Forward declarations - avoid deep include chains
// Full includes are in inference.cpp
#include "utils/config_loader.h"
#include "utils/json_parser.h"
#include "core/feeling_output.h"
#include "memory/episodic_retriever.h"

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <memory>

namespace cardinal {

    // Forward declare LLMEngine - full definition in llm_engine.h
    // included only in inference.cpp
    class LLMEngine;

    // -----------------------------------------------------------------------------
    // InferenceRequest
    // Input to a single inference cycle.
    // -----------------------------------------------------------------------------
    struct InferenceRequest {
        std::string              user_message;      // The user's input
        std::vector<ChatMessage> history;           // Prior conversation turns
        std::vector<Rule>        active_rules;      // Rules injected from rule store
        bool                     stream_response;   // Stream Pass 2 tokens as generated
    };

    // -----------------------------------------------------------------------------
    // InferenceResponse
    // Output of a single complete inference cycle.
    // -----------------------------------------------------------------------------
    struct InferenceResponse {
        // Core outputs
        std::string              response;          // Final response text
        FeelingOutput            feeling;           // Feeling output from Pass 1

        // Status
        bool                     success;           // Overall success
        std::string              error_message;     // Set on failure

        // Metrics
        InferenceMetrics         metrics;           // Timing + token counts

        // Flags for downstream processing
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

    // -----------------------------------------------------------------------------
    // StreamCallback
    // Called for each token during Pass 2 streaming.
    // Return false to abort generation.
    // -----------------------------------------------------------------------------
    using StreamCallback = std::function<bool(const std::string& token)>;

    // -----------------------------------------------------------------------------
    // InferencePipeline
    // The central orchestrator for Cardinal's two-pass inference.
    //
    // Usage:
    //   InferencePipeline pipeline(config, engine);
    //   auto response = pipeline.run(request);
    //
    // The pipeline:
    //   1. Builds the full message list with system prompt + rules
    //   2. Runs Pass 1 (feeling output, constrained decoding)
    //   3. Retries Pass 1 up to config.feedback.max_retries on failure
    //   4. Runs Pass 2 (final response, free decoding, synthetic turn injected)
    //   5. Returns InferenceResponse with both outputs + metrics
    // -----------------------------------------------------------------------------
    class InferencePipeline {
    public:
        InferencePipeline(const CardinalConfig& config, LLMEngine& engine);

        // -------------------------------------------------------------------------
        // Core inference
        // -------------------------------------------------------------------------

        // Run a full two-pass inference cycle
        InferenceResponse run(const InferenceRequest& request,
            const StreamCallback& stream_cb = nullptr);

        // -------------------------------------------------------------------------
        // Context management
        // -------------------------------------------------------------------------

        // Clear conversation context between sessions
        void reset_context();

        // Check if context is near its limit
        bool context_near_limit() const;

        // -------------------------------------------------------------------------
        // Memory retrieval
        // -------------------------------------------------------------------------
        
        // Set the episodic retriever for memory context injection.
        // Optional -- if not set, no memory context is injected.
        // Retriever must outlive this pipeline.
        void set_retriever(EpisodicRetriever* retriever) {
            retriever_ = retriever;
        }

        bool has_retriever() const { return retriever_ != nullptr; }

        // -------------------------------------------------------------------------
        // System prompt management
        // -------------------------------------------------------------------------

        // Set the base system prompt - persists across inference calls
        void set_system_prompt(const std::string& prompt);

        // Get the current system prompt
        const std::string& system_prompt() const { return system_prompt_; }

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        int  total_inferences()  const { return total_inferences_; }
        int  total_retries()     const { return total_retries_; }
        int  failed_inferences() const { return failed_inferences_; }

    private:
        // -------------------------------------------------------------------------
        // Internal pipeline steps
        // -------------------------------------------------------------------------

        // Build full message list: system prompt + rules + history + user message
        std::vector<ChatMessage> build_messages(const InferenceRequest& request) const;

        // Format active rules for injection into system prompt
        std::string format_rules(const std::vector<Rule>& rules) const;

        std::string format_episodes(
            const std::vector<RetrievalResult>& results) const;

        // Run Pass 1 with retry logic
        // Returns true if feeling output was successfully obtained
        bool run_pass1(FeelingContext& ctx,
            const std::vector<ChatMessage>& messages);

        // Run Pass 2
        // Returns true if response was successfully generated
        bool run_pass2(FeelingContext& ctx,
            const std::vector<ChatMessage>& messages,
            const StreamCallback& stream_cb);

        // Build InferenceResponse from completed FeelingContext
        InferenceResponse build_response(const FeelingContext& ctx,
            bool success,
            const std::string& error = "") const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        LLMEngine& engine_;
        FeelingContext         feeling_ctx_;
        EpisodicRetriever* retriever_ = nullptr; //phase 6

        std::string            system_prompt_;

        // Stats
        int                    total_inferences_ = 0;
        int                    total_retries_ = 0;
        int                    failed_inferences_ = 0;

        // Default system prompt used if none is set
        static const char* DEFAULT_SYSTEM_PROMPT;
    };

    // -----------------------------------------------------------------------------
    // InferenceError
    // -----------------------------------------------------------------------------
    class InferenceError : public std::runtime_error {
    public:
        explicit InferenceError(const std::string& message)
            : std::runtime_error("InferenceError: " + message) {}
    };

} // namespace cardinal