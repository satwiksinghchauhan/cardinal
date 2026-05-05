// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - llama.cpp Backend Implementation
// File: src/core/backends/llama_cpp_backend.cpp
//
// All logic is identical to the original llm_engine.cpp.
// Changes from original:
//   - Class renamed LLMEngine → LlamaCppBackend
//   - Include path updated to llama_cpp_backend.h
//   - Windows preprocessor guards removed (Linux-only build)
//   - LLMError no longer defined here — lives in llm_backend.h
// =============================================================================

#include "core/backends/llama_cpp_backend.h"

#include <stdexcept>
#include <sstream>
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>

namespace cardinal {

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    LlamaCppBackend::LlamaCppBackend(const CardinalConfig& config)
        : config_(config)
    {
        LOG_INFO("LlamaCppBackend created - model: " + config_.backend.llama_cpp.model_path);
    }

    LlamaCppBackend::~LlamaCppBackend() {
        unload();
    }

    // =========================================================================
    // create_context
    // =========================================================================

    llama_context* LlamaCppBackend::create_context() {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx           = config_.backend.llama_cpp.context_length;
        ctx_params.n_batch         = config_.backend.llama_cpp.context_length;
        ctx_params.n_ubatch        = 512;
        ctx_params.n_threads       = config_.backend.llama_cpp.threads;
        ctx_params.n_threads_batch = config_.backend.llama_cpp.threads;

        llama_context* ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw LLMError("Failed to create llama context");
        }
        return ctx;
    }

    // =========================================================================
    // load_model
    // =========================================================================

    void LlamaCppBackend::load_model() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (ready_) {
            LOG_WARN("load_model called but model already loaded");
            return;
        }

        LOG_INFO("Loading model: " + config_.backend.llama_cpp.model_path);
        LOG_INFO("GPU layers: "    + std::to_string(config_.backend.llama_cpp.gpu_layers));
        LOG_INFO("Context length: "+ std::to_string(config_.backend.llama_cpp.context_length));
        LOG_INFO("Threads: "       + std::to_string(config_.backend.llama_cpp.threads));

        llama_backend_init();

        llama_model_params model_params  = llama_model_default_params();
        model_params.n_gpu_layers        = config_.backend.llama_cpp.gpu_layers;

        model_ = llama_model_load_from_file(
            config_.backend.llama_cpp.model_path.c_str(), model_params);
        if (!model_) {
            throw LLMError("Failed to load model from: " +
                           config_.backend.llama_cpp.model_path);
        }
        LOG_INFO("Model loaded: " + model_name());

        // Two separate contexts — one per pass — so KV state never leaks
        // between constrained (Pass 1) and free (Pass 2) decoding.
        ctx_pass1_ = create_context();
        ctx_pass2_ = create_context();

        ready_ = true;
        LOG_INFO("LlamaCppBackend ready - vocab: " + std::to_string(n_vocab()) +
                 ", ctx: "  + std::to_string(context_length()) +
                 ", embd: " + std::to_string(n_embd()));
    }

    // =========================================================================
    // is_ready / unload
    // =========================================================================

    bool LlamaCppBackend::is_ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_;
    }

    void LlamaCppBackend::unload() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!ready_) return;

        if (ctx_pass1_) { llama_free(ctx_pass1_); ctx_pass1_ = nullptr; }
        if (ctx_pass2_) { llama_free(ctx_pass2_); ctx_pass2_ = nullptr; }
        if (model_)     { llama_model_free(model_); model_ = nullptr; }

        llama_backend_free();
        ready_ = false;

        LOG_INFO("LlamaCppBackend unloaded");
    }

    // =========================================================================
    // get_info
    // =========================================================================

    BackendInfo LlamaCppBackend::get_info() const {
        BackendInfo info;
        info.type           = BackendType::LLAMA_CPP;
        info.name           = "llama.cpp";
        info.version        = std::to_string(LLAMA_BUILD_NUMBER);
        info.model_name     = ready_ ? model_name() : "(not loaded)";
        info.context_length = ready_ ? context_length() : 0;
        info.n_vocab        = ready_ ? n_vocab() : 0;
        info.gpu_enabled    = config_.backend.llama_cpp.gpu_layers > 0;
        info.gpu_layers     = config_.backend.llama_cpp.gpu_layers;
        return info;
    }

    // =========================================================================
    // Pass 1: generate_feeling — GBNF constrained decoding
    // =========================================================================

    GenerationResult LlamaCppBackend::generate_feeling(
        FeelingContext&                 ctx,
        const std::vector<ChatMessage>& messages)
    {
        if (!ready_) throw LLMError("Engine not ready - call load_model() first");

        ctx.set_state(PassState::PASS1_FEELING);
        ctx.start_pass1_timer();

        LOG_DEBUG("Pass 1: generating feeling output (GBNF constrained decoding)");

        // Augment system message with feeling-output instruction
        std::vector<ChatMessage> feeling_messages = messages;
        bool has_system = false;
        for (auto& msg : feeling_messages) {
            if (msg.role == "system") {
                msg.content +=
                    "\n\nBefore responding, output your internal state as "
                    "a JSON object with exactly these fields: "
                    "confidence (float 0-1), reasoning_type "
                    "(analogical|causal|deductive|inductive|abductive|associative), "
                    "uncertainty_flag (bool), rule_candidate_signal (bool), "
                    "contradiction_flag (bool), reasoning_domain "
                    "(factual|ethical|spatial|temporal|social|mathematical). "
                    "Output ONLY the JSON object, nothing else.";
                has_system = true;
                break;
            }
        }
        if (!has_system) {
            feeling_messages.insert(feeling_messages.begin(), {
                "system",
                "You are Cardinal, a neurosymbolic AI. Before responding, output your "
                "internal state as a JSON object with exactly these fields: "
                "confidence (float 0-1), reasoning_type "
                "(analogical|causal|deductive|inductive|abductive|associative), "
                "uncertainty_flag (bool), rule_candidate_signal (bool), "
                "contradiction_flag (bool), reasoning_domain "
                "(factual|ethical|spatial|temporal|social|mathematical). "
                "Output ONLY the JSON object, nothing else."
            });
        }

        std::string prompt = apply_chat_template(feeling_messages, true);

        auto result = generate_internal(
            ctx_pass1_,
            prompt,
            config_.feeling_schema.max_tokens,
            true,                       // use_grammar — GBNF path
            ctx.grammar().content,
            nullptr
        );

        ctx.stop_pass1_timer();
        ctx.metrics().pass1_tokens_generated = result.tokens_generated;
        ctx.metrics().prompt_tokens          = result.tokens_prompted;

        if (result.success) {
            ctx.set_raw_feeling(result.text);
            if (!ctx.parse_feeling()) {
                // Should not happen under grammar constraint — log and fail
                LOG_WARN("GBNF-constrained output failed JSON parsing — unexpected");
                result.success = false;
            }
        }

        LOG_DEBUG("Pass 1 complete: " + result.text +
                  " (" + std::to_string(result.tokens_generated) + " tokens)");
        return result;
    }

    // =========================================================================
    // Pass 2: generate_response — free decoding
    // =========================================================================

    GenerationResult LlamaCppBackend::generate_response(
        FeelingContext&                 ctx,
        const std::vector<ChatMessage>& messages,
        const TokenCallback&            callback)
    {
        if (!ready_)                throw LLMError("Engine not ready - call load_model() first");
        if (!ctx.has_valid_feeling()) throw LLMError("Pass 2 called without valid feeling output");

        ctx.set_state(PassState::PASS2_RESPONSE);
        ctx.start_pass2_timer();

        LOG_DEBUG("Pass 2: generating final response (free decoding)");

        ctx.prepare_synthetic_turn();
        auto augmented = inject_synthetic_turn(messages, ctx.synthetic_turn());
        std::string prompt = apply_chat_template(augmented, true);

        auto result = generate_internal(
            ctx_pass2_,
            prompt,
            config_.inference.max_tokens_response,
            false,  // free decoding
            "",
            callback
        );

        ctx.stop_pass2_timer();
        ctx.metrics().pass2_tokens_generated = result.tokens_generated;

        if (result.success) {
            ctx.set_final_response(result.text);
            ctx.set_state(PassState::COMPLETE);
        } else {
            ctx.set_state(PassState::FAILED);
        }

        LOG_DEBUG("Pass 2 complete: " + std::to_string(result.tokens_generated) + " tokens");
        LOG_INFO(ctx.metrics().to_string());
        return result;
    }

    // =========================================================================
    // generate_internal — shared generation loop
    // =========================================================================

    GenerationResult LlamaCppBackend::generate_internal(
        llama_context*       ctx,
        const std::string&   prompt,
        int                  max_tokens,
        bool                 use_grammar,
        const std::string&   grammar_text,
        const TokenCallback& callback)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        GenerationResult result{};
        result.success       = false;
        result.stopped_eos   = false;
        result.stopped_limit = false;
        result.stopped_abort = false;

        // Tokenize prompt
        std::vector<llama_token> prompt_tokens = tokenize_internal(prompt, true);
        result.tokens_prompted = static_cast<int>(prompt_tokens.size());

        if (prompt_tokens.empty()) {
            LOG_WARN("generate_internal: empty prompt after tokenization");
            return result;
        }

        // Guard context limit
        int ctx_size = llama_n_ctx(ctx);
        if (result.tokens_prompted + max_tokens > ctx_size) {
            LOG_WARN("Prompt + max_tokens (" +
                     std::to_string(result.tokens_prompted + max_tokens) +
                     ") exceeds context length (" +
                     std::to_string(ctx_size) + ") — truncating max_tokens");
            max_tokens = ctx_size - result.tokens_prompted - 4;
            if (max_tokens <= 0)
                throw LLMError("Context full — cannot generate. Clear KV cache.");
        }

        // Fill batch with prompt tokens
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()));
        if (llama_decode(ctx, batch) != 0)
            throw LLMError("llama_decode failed on prompt");

        // Build sampler chain
        llama_sampler* sampler = build_sampler(use_grammar, grammar_text);

        // Generation loop
        std::string generated_text;
        generated_text.reserve(max_tokens * 4);

        int tokens_generated = 0;
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        while (tokens_generated < max_tokens) {
            llama_token token_id = llama_sampler_sample(sampler, ctx, -1);

            // For non-grammar samplers we must call accept() to advance state.
            // Grammar samplers advance during sample() — calling accept() again
            // would double-advance and corrupt grammar state.
            if (!use_grammar) {
                llama_sampler_accept(sampler, token_id);
            }

            if (llama_vocab_is_eog(vocab, token_id)) {
                result.stopped_eos = true;
                break;
            }

            char token_buf[256] = {};
            int  token_len = llama_token_to_piece(
                vocab, token_id, token_buf, sizeof(token_buf) - 1, 0, true);

            if (token_len < 0) {
                LOG_WARN("Token buffer too small for token " + std::to_string(token_id));
            } else {
                std::string token_text(token_buf, token_len);
                generated_text += token_text;
                ++tokens_generated;
                ++total_tokens_generated_;

                if (callback) {
                    bool cont = callback(token_text, token_id, tokens_generated);
                    if (!cont) {
                        result.stopped_abort = true;
                        break;
                    }
                }
            }

            llama_batch next_batch = llama_batch_get_one(&token_id, 1);
            if (llama_decode(ctx, next_batch) != 0) {
                LOG_WARN("llama_decode failed mid-generation — stopping");
                break;
            }
        }

        if (tokens_generated >= max_tokens && !result.stopped_eos)
            result.stopped_limit = true;

        llama_sampler_free(sampler);

        result.text             = generated_text;
        result.tokens_generated = tokens_generated;
        result.success          = true;
        return result;
    }

    // =========================================================================
    // build_sampler
    // =========================================================================

    llama_sampler* LlamaCppBackend::build_sampler(bool               use_grammar,
                                                   const std::string& grammar_text)
    {
        llama_sampler* chain = llama_sampler_chain_init(
            llama_sampler_chain_default_params());

        if (use_grammar && !grammar_text.empty()) {
            const llama_vocab* vocab = llama_model_get_vocab(model_);
            llama_sampler_chain_add(chain,
                llama_sampler_init_grammar(vocab, grammar_text.c_str(), "root"));
            llama_sampler_chain_add(chain,
                llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(chain,
                llama_sampler_init_temp(config_.inference.temperature));
            llama_sampler_chain_add(chain,
                llama_sampler_init_top_p(config_.inference.top_p, 1));
            llama_sampler_chain_add(chain,
                llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }

        return chain;
    }

    // =========================================================================
    // inject_synthetic_turn
    // =========================================================================

    std::vector<ChatMessage> LlamaCppBackend::inject_synthetic_turn(
        const std::vector<ChatMessage>& messages,
        const SyntheticTurn&            turn) const
    {
        std::vector<ChatMessage> augmented;
        augmented.reserve(messages.size() + 1);

        if (!messages.empty() && messages.back().role == "user") {
            for (size_t i = 0; i < messages.size() - 1; ++i)
                augmented.push_back(messages[i]);
            augmented.push_back({ turn.role, turn.format() });
            augmented.push_back(messages.back());
        } else {
            augmented = messages;
            augmented.push_back({ turn.role, turn.format() });
        }

        return augmented;
    }

    // =========================================================================
    // Context management
    // =========================================================================

    void LlamaCppBackend::clear_kv_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return;
        llama_memory_clear(llama_get_memory(ctx_pass1_), false);
        llama_memory_clear(llama_get_memory(ctx_pass2_), false);
        LOG_DEBUG("KV cache cleared (both contexts)");
    }

    int LlamaCppBackend::context_tokens_used() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return 0;
        return static_cast<int>(total_tokens_generated_.load());
    }

    int LlamaCppBackend::context_tokens_remaining() const {
        return context_length() - context_tokens_used();
    }

    bool LlamaCppBackend::context_near_limit(float threshold) const {
        if (context_length() == 0) return false;
        return static_cast<float>(context_tokens_used()) /
               static_cast<float>(context_length()) >= threshold;
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    std::vector<llama_token> LlamaCppBackend::tokenize(
        const std::string& text, bool add_special) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return tokenize_internal(text, add_special);
    }

    std::vector<llama_token> LlamaCppBackend::tokenize_internal(
        const std::string& text, bool add_special) const
    {
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        int n = llama_tokenize(vocab,
            text.c_str(), static_cast<int>(text.size()),
            nullptr, 0, add_special, true);
        if (n < 0) n = -n;

        std::vector<llama_token> tokens(n);
        llama_tokenize(vocab,
            text.c_str(), static_cast<int>(text.size()),
            tokens.data(), n, add_special, true);

        return tokens;
    }

    std::string LlamaCppBackend::detokenize(
        const std::vector<llama_token>& tokens) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return "";

        const llama_vocab* vocab = llama_model_get_vocab(model_);
        std::string result;
        result.reserve(tokens.size() * 4);

        char buf[256] = {};
        for (llama_token tok : tokens) {
            int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf) - 1, 0, true);
            if (len > 0) result.append(buf, len);
        }
        return result;
    }

    int LlamaCppBackend::count_tokens(const std::string& text) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(tokenize_internal(text, false).size());
    }

    // =========================================================================
    // apply_chat_template
    // =========================================================================

    std::string LlamaCppBackend::apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool                            add_generation_prompt) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) throw LLMError("Engine not ready");

        std::vector<llama_chat_message> llama_msgs;
        llama_msgs.reserve(messages.size());
        for (const auto& msg : messages) {
            llama_msgs.push_back({ msg.role.c_str(), msg.content.c_str() });
        }

        // Pass nullptr to use the model's own embedded template
        const char* tmpl = nullptr;

        int32_t required = llama_chat_apply_template(
            tmpl,
            llama_msgs.data(), llama_msgs.size(),
            add_generation_prompt,
            nullptr, 0);

        if (required < 0)
            throw LLMError("llama_chat_apply_template failed — model may lack chat template");

        std::string result(required + 1, '\0');
        llama_chat_apply_template(
            tmpl,
            llama_msgs.data(), llama_msgs.size(),
            add_generation_prompt,
            result.data(), static_cast<int32_t>(result.size()));

        result.resize(required);
        return result;
    }

    // =========================================================================
    // Model metadata
    // =========================================================================

    std::string LlamaCppBackend::model_name() const {
        if (!model_) return "none";
        std::string desc(256, '\0');
        llama_model_desc(model_, desc.data(), 256);
        return desc.c_str();
    }

    int LlamaCppBackend::context_length() const {
        if (!ctx_pass2_) return 0;
        return static_cast<int>(llama_n_ctx(ctx_pass2_));
    }

    int LlamaCppBackend::n_vocab() const {
        if (!model_) return 0;
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        return llama_vocab_n_tokens(vocab);
    }

    int LlamaCppBackend::n_embd() const {
        if (!model_) return 0;
        return llama_model_n_embd(model_);
    }

} // namespace cardinal
