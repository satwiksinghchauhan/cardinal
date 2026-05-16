// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - llama.cpp Backend
// File: src/core/backends/llama_cpp_backend.h
//
// Concrete ILLMBackend implementation wrapping llama.cpp.
// Owns the llama_model and two llama_contexts (one per pass).
// Handles GBNF-constrained decoding natively for Pass 1.
//
// Replaces: src/core/llm_engine.h / llm_engine.cpp
//
// v1.4.0: added get_llama_model() / get_llama_context() for LlamaCppTrainer.
// =============================================================================

#include "core/llm_backend.h"
#include "utils/config_loader.h"
#include "utils/logger.h"
#include "core/feeling_output.h"

// llama.cpp headers — llama_model, llama_context, llama_token all declared here.
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

    // -------------------------------------------------------------------------
    // LlamaCppBackend
    // One instance per process. Thread-safe via internal mutex_.
    //
    // Usage (handled by BackendFactory — do not instantiate directly):
    //   auto backend = std::make_unique<LlamaCppBackend>(config);
    //   backend->load_model();
    //   GenerationResult r = backend->generate_feeling(ctx, messages);
    // -------------------------------------------------------------------------
    class LlamaCppBackend final : public ILLMBackend {
    public:
        explicit LlamaCppBackend(const CardinalConfig& config);
        ~LlamaCppBackend() override;

        // ------------------------------------------------------------------
        // ILLMBackend — Lifecycle
        // ------------------------------------------------------------------
        void load_model() override;
        bool is_ready()   const override;
        void unload()     override;

        // ------------------------------------------------------------------
        // ILLMBackend — Two-pass inference
        // ------------------------------------------------------------------
        GenerationResult generate_feeling(
            FeelingContext&                 ctx,
            const std::vector<ChatMessage>& messages) override;

        GenerationResult generate_response(
            FeelingContext&                 ctx,
            const std::vector<ChatMessage>& messages,
            const TokenCallback&            callback = nullptr) override;

        // ------------------------------------------------------------------
        // ILLMBackend — Context management
        // ------------------------------------------------------------------
        void clear_kv_cache()                         override;
        bool context_near_limit(float t = 0.9f) const override;
        int  context_tokens_used()               const override;
        int  context_tokens_remaining()          const override;

        // ------------------------------------------------------------------
        // ILLMBackend — Capability / info
        // ------------------------------------------------------------------

        // Always true — llama.cpp enforces GBNF grammar during sampling.
        bool        supports_constrained_decoding() const override { return true; }
        BackendInfo get_info()                      const override;

        // ------------------------------------------------------------------
        // Tokenization utilities (llama.cpp specific — not on ILLMBackend)
        // ------------------------------------------------------------------
        std::vector<llama_token> tokenize(const std::string& text,
                                          bool add_special = true) const;
        std::string              detokenize(const std::vector<llama_token>& tokens) const;
        int                      count_tokens(const std::string& text) const;

        // Apply the model's built-in chat template to a message list.
        std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                        bool add_generation_prompt = true) const;

        // Model metadata
        std::string model_name()     const;
        int         context_length() const;
        int         n_vocab()        const;
        int         n_embd()         const;

        // ------------------------------------------------------------------
        // LoRA adapter access — used by LlamaCppTrainer (Layer 3).
        // llama.h is already included above so these types are fully declared.
        // The adapter is applied to ctx_pass2_ (free-decoding context only).
        // ------------------------------------------------------------------
        llama_model*   get_llama_model()   const { return model_; }
        llama_context* get_llama_context() const { return ctx_pass2_; }

    private:
        // ------------------------------------------------------------------
        // Internal generation — shared by both passes
        // ------------------------------------------------------------------
        GenerationResult generate_internal(
            llama_context*       ctx,
            const std::string&   prompt,
            int                  max_tokens,
            bool                 use_grammar,
            const std::string&   grammar_text,
            const TokenCallback& callback);

        // Build sampler chain:
        //   use_grammar=true  → GBNF constraint + greedy  (Pass 1)
        //   use_grammar=false → temperature + top_p + dist (Pass 2)
        llama_sampler* build_sampler(bool               use_grammar,
                                     const std::string& grammar_text);

        // Insert the feeling synthetic turn before the last user message.
        std::vector<ChatMessage> inject_synthetic_turn(
            const std::vector<ChatMessage>& messages,
            const SyntheticTurn&            turn) const;

        // Create a fresh llama_context from the loaded model.
        llama_context* create_context();

        // Tokenize without acquiring mutex_ — caller must hold it.
        std::vector<llama_token> tokenize_internal(const std::string& text,
                                                   bool add_special) const;

        // ------------------------------------------------------------------
        // Members
        // ------------------------------------------------------------------
        const CardinalConfig& config_;

        llama_model*   model_     = nullptr;
        llama_context* ctx_pass1_ = nullptr;   // constrained (GBNF)
        llama_context* ctx_pass2_ = nullptr;   // free decoding

        bool                 ready_ = false;
        mutable std::mutex   mutex_;

        std::atomic<int>     total_tokens_generated_{ 0 };
    };

} // namespace cardinal
