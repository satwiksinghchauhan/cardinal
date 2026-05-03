// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - LLM Engine
// File: src/core/llm_engine.h
// Full llama.cpp wrapper. Owns model, context, sampler chain.
// Handles constrained (Pass 1) and free (Pass 2) decoding.
// Model-agnostic - all model-specific settings come from CardinalConfig.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "utils/logger.h"
#include "core/feeling_output.h"

// llama.cpp headers
#include "llama.h"
#include "llama-cpp.h"
#include "common.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // TokenCallback
    // Called for each generated token during inference.
    // Used to stream output and track token counts for metrics.
    // Return false to stop generation early.
    // -----------------------------------------------------------------------------
    using TokenCallback = std::function<bool(const std::string& token_text,
        int token_id,
        int tokens_generated)>;

    // -----------------------------------------------------------------------------
    // GenerationResult
    // Result of a single generation call (one pass).
    // -----------------------------------------------------------------------------
    struct GenerationResult {
        std::string text;               // Full generated text
        int         tokens_generated;   // Number of tokens generated
        int         tokens_prompted;    // Tokens in the prompt
        bool        stopped_eos;        // Stopped on end-of-sequence token
        bool        stopped_limit;      // Stopped on max token limit
        bool        stopped_abort;      // Stopped by callback returning false
        bool        success;            // Overall success flag

        // Check if generation completed naturally
        bool completed_naturally() const { return stopped_eos && !stopped_abort; }
    };

    // ChatMessage is defined in utils/json_parser.h - included via feeling_output.h

    // -----------------------------------------------------------------------------
    // LLMEngine
    // Core inference engine. One instance per process.
    // Thread-safe for concurrent reads, serialized writes via internal mutex.
    //
    // Usage:
    //   LLMEngine engine(config);
    //   engine.load_model();
    //   auto result = engine.generate_feeling(ctx, messages);
    //   auto result = engine.generate_response(ctx, messages);
    // -----------------------------------------------------------------------------
    class LLMEngine {
    public:
        explicit LLMEngine(const CardinalConfig& config);
        ~LLMEngine();

        // -------------------------------------------------------------------------
        // Initialization
        // -------------------------------------------------------------------------

        // Load model and initialize context - call once at startup
        // Throws LLMError on failure
        void load_model();

        // Check if model is loaded and ready
        bool is_ready() const;

        // Unload model and free all resources
        void unload();

        // -------------------------------------------------------------------------
        // Two-pass inference
        // -------------------------------------------------------------------------

        // Pass 1: Constrained decoding with GBNF grammar
        // Generates the feeling output JSON.
        // Applies grammar from ctx.grammar(), enforces max_tokens_feeling.
        // Stores result in ctx via set_raw_feeling().
        GenerationResult generate_feeling(FeelingContext& ctx,
            const std::vector<ChatMessage>& messages);

        // Pass 2: Free decoding
        // Generates the final response.
        // Injects synthetic turn from ctx into context before generating.
        // Enforces max_tokens_response.
        GenerationResult generate_response(FeelingContext& ctx,
            const std::vector<ChatMessage>& messages,
            const TokenCallback& callback = nullptr);

        // -------------------------------------------------------------------------
        // Context management
        // -------------------------------------------------------------------------

        // Clear KV cache - call between unrelated conversations
        void clear_kv_cache();

        // Get current context usage (tokens used / context length)
        int  context_tokens_used() const;
        int  context_tokens_remaining() const;
        bool context_near_limit(float threshold = 0.9f) const;

        // -------------------------------------------------------------------------
        // Tokenization utilities
        // -------------------------------------------------------------------------

        // Tokenize a string - returns token IDs
        std::vector<llama_token> tokenize(const std::string& text,
            bool add_special = true) const;

        // Detokenize token IDs to string
        std::string detokenize(const std::vector<llama_token>& tokens) const;

        // Count tokens in a string without full tokenization
        int count_tokens(const std::string& text) const;

        // -------------------------------------------------------------------------
        // Chat template formatting
        // -------------------------------------------------------------------------

        // Apply the model's chat template to a list of messages
        // Produces the formatted prompt string ready for inference
        std::string apply_chat_template(const std::vector<ChatMessage>& messages,
            bool add_generation_prompt = true) const;

        // -------------------------------------------------------------------------
        // Model info
        // -------------------------------------------------------------------------

        std::string model_name()      const;
        int         context_length()  const;
        int         n_vocab()         const;
        int         n_embd()          const;

        // Disable copy/move - owns non-copyable llama resources
        LLMEngine(const LLMEngine&) = delete;
        LLMEngine& operator=(const LLMEngine&) = delete;
        LLMEngine(LLMEngine&&) = delete;
        LLMEngine& operator=(LLMEngine&&) = delete;

    private:
        // -------------------------------------------------------------------------
        // Internal generation - shared by both passes
        // -------------------------------------------------------------------------
        GenerationResult generate_internal(
            llama_context* ctx,
            const std::string& prompt,
            int                   max_tokens,
            bool                  use_grammar,
            const std::string& grammar_text,
            const TokenCallback& callback);

        // Build sampler chain for Pass 1 (constrained) or Pass 2 (free)
        llama_sampler* build_sampler(bool use_grammar,
            const std::string& grammar_text);

        // Format messages with synthetic turn injected (for Pass 2)
        std::vector<ChatMessage> inject_synthetic_turn(
            const std::vector<ChatMessage>& messages,
            const SyntheticTurn& turn) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;

        // llama.cpp owned resources
        // Model is shared - contexts are separate per pass
        llama_model* model_ = nullptr;
        llama_context* ctx_pass1_ = nullptr;  // Pass 1: constrained (grammar)
        llama_context* ctx_pass2_ = nullptr;  // Pass 2: free decoding

        // State
        bool                    ready_ = false;
        mutable std::mutex      mutex_;

        // Internal helpers
        llama_context* create_context();  // Creates a fresh context from model

        // Internal tokenize - caller must hold mutex_
        std::vector<llama_token> tokenize_internal(const std::string& text,
            bool add_special) const;

        // Token counters for metrics
        std::atomic<int>        total_tokens_generated_{ 0 };
    };

    // -----------------------------------------------------------------------------
    // LLMError - thrown on engine failures
    // -----------------------------------------------------------------------------
    class LLMError : public std::runtime_error {
    public:
        explicit LLMError(const std::string& message)
            : std::runtime_error("LLMError: " + message) {}
    };

} // namespace cardinal