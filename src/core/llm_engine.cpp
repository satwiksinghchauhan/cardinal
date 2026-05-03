// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - LLM Engine Implementation
// File: src/core/llm_engine.cpp
// =============================================================================

#include "llm_engine.h"

#include <stdexcept>
#include <sstream>
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>

namespace cardinal {

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    LLMEngine::LLMEngine(const CardinalConfig& config)
        : config_(config)
    {
        LOG_INFO("LLMEngine created - model: " + config_.model.path);
    }

    LLMEngine::~LLMEngine() {
        unload();
    }

    // =============================================================================
    // create_context - creates a fresh llama context from loaded model
    // =============================================================================

    llama_context* LLMEngine::create_context() {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = config_.model.context_length;
        ctx_params.n_threads = config_.model.threads;
        ctx_params.n_threads_batch = config_.model.threads;

        llama_context* ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw LLMError("Failed to create llama context");
        }
        return ctx;
    }

    // =============================================================================
    // load_model
    // =============================================================================

    void LLMEngine::load_model() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (ready_) {
            LOG_WARN("load_model called but model already loaded");
            return;
        }

        LOG_INFO("Loading model: " + config_.model.path);
        LOG_INFO("GPU layers: " + std::to_string(config_.model.gpu_layers));
        LOG_INFO("Context length: " + std::to_string(config_.model.context_length));
        LOG_INFO("Threads: " + std::to_string(config_.model.threads));

        // -------------------------------------------------------------------------
        // Initialize llama backend
        // -------------------------------------------------------------------------
        llama_backend_init();

        // -------------------------------------------------------------------------
        // Model parameters
        // -------------------------------------------------------------------------
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.model.gpu_layers;

        // -------------------------------------------------------------------------
        // Load model from file
        // -------------------------------------------------------------------------
        model_ = llama_model_load_from_file(config_.model.path.c_str(), model_params);
        if (!model_) {
            throw LLMError("Failed to load model from: " + config_.model.path);
        }
        LOG_INFO("Model loaded: " + model_name());

        // -------------------------------------------------------------------------
        // Create two separate contexts - one per pass
        // Pass 1 (grammar/constrained) and Pass 2 (free) must not share state
        // -------------------------------------------------------------------------
        ctx_pass1_ = create_context();
        ctx_pass2_ = create_context();

        ready_ = true;
        LOG_INFO("LLMEngine ready - vocab: " + std::to_string(n_vocab()) +
            ", ctx: " + std::to_string(context_length()) +
            ", embd: " + std::to_string(n_embd()));
    }

    // =============================================================================
    // is_ready / unload
    // =============================================================================

    bool LLMEngine::is_ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_;
    }

    void LLMEngine::unload() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!ready_) return;

        if (ctx_pass1_) {
            llama_free(ctx_pass1_);
            ctx_pass1_ = nullptr;
        }
        if (ctx_pass2_) {
            llama_free(ctx_pass2_);
            ctx_pass2_ = nullptr;
        }
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }

        llama_backend_free();
        ready_ = false;

        LOG_INFO("LLMEngine unloaded");
    }

    // =============================================================================
    // Pass 1: generate_feeling
    // Constrained decoding with GBNF grammar
    // =============================================================================

    GenerationResult LLMEngine::generate_feeling(
        FeelingContext& ctx,
        const std::vector<ChatMessage>& messages)
    {
        if (!ready_) throw LLMError("Engine not ready - call load_model() first");

        ctx.set_state(PassState::PASS1_FEELING);
        ctx.start_pass1_timer();

        LOG_DEBUG("Pass 1: generating feeling output (constrained decoding)");

        // Build system prompt that instructs model to output feeling schema
        std::vector<ChatMessage> feeling_messages = messages;

        // Prepend feeling instruction to system message (or insert one)
        bool has_system = false;
        for (auto& msg : feeling_messages) {
            if (msg.role == "system") {
                msg.content += "\n\nBefore responding, output your internal state as "
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
            true,
            ctx.grammar().content,
            nullptr
        );

        ctx.stop_pass1_timer();
        ctx.metrics().pass1_tokens_generated = result.tokens_generated;
        ctx.metrics().prompt_tokens = result.tokens_prompted;

        if (result.success) {
            ctx.set_raw_feeling(result.text);
            if (!ctx.parse_feeling()) {
                // Parse failed despite grammar constraint - shouldn't happen
                // but handle gracefully
                LOG_WARN("Grammar-constrained output failed parsing - unexpected");
                result.success = false;
            }
        }

        LOG_DEBUG("Pass 1 complete: " + result.text +
            " (" + std::to_string(result.tokens_generated) + " tokens)");

        return result;
    }

    // =============================================================================
    // Pass 2: generate_response
    // Free decoding with synthetic turn injected
    // =============================================================================

    GenerationResult LLMEngine::generate_response(
        FeelingContext& ctx,
        const std::vector<ChatMessage>& messages,
        const TokenCallback& callback)
    {
        if (!ready_) throw LLMError("Engine not ready - call load_model() first");
        if (!ctx.has_valid_feeling()) {
            throw LLMError("Pass 2 called without valid feeling output");
        }

        ctx.set_state(PassState::PASS2_RESPONSE);
        ctx.start_pass2_timer();

        LOG_DEBUG("Pass 2: generating final response (free decoding)");

        // Prepare synthetic turn and inject into message history
        ctx.prepare_synthetic_turn();
        auto augmented_messages = inject_synthetic_turn(messages, ctx.synthetic_turn());

        std::string prompt = apply_chat_template(augmented_messages, true);

        auto result = generate_internal(
            ctx_pass2_,
            prompt,
            config_.inference.max_tokens_response,
            false,
            "",
            callback
        );

        ctx.stop_pass2_timer();
        ctx.metrics().pass2_tokens_generated = result.tokens_generated;

        if (result.success) {
            ctx.set_final_response(result.text);
            ctx.set_state(PassState::COMPLETE);
        }
        else {
            ctx.set_state(PassState::FAILED);
        }

        LOG_DEBUG("Pass 2 complete: " +
            std::to_string(result.tokens_generated) + " tokens");
        LOG_INFO(ctx.metrics().to_string());

        return result;
    }

    // =============================================================================
    // generate_internal - shared generation loop
    // =============================================================================

    GenerationResult LLMEngine::generate_internal(
        llama_context* ctx,
        const std::string& prompt,
        int                  max_tokens,
        bool                 use_grammar,
        const std::string& grammar_text,
        const TokenCallback& callback)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        GenerationResult result{};
        result.success = false;
        result.stopped_eos = false;
        result.stopped_limit = false;
        result.stopped_abort = false;

        // -------------------------------------------------------------------------
        // Tokenize prompt
        // -------------------------------------------------------------------------
        std::vector<llama_token> prompt_tokens = tokenize_internal(prompt, true);
        result.tokens_prompted = static_cast<int>(prompt_tokens.size());

        if (prompt_tokens.empty()) {
            LOG_WARN("generate_internal: empty prompt after tokenization");
            return result;
        }

        // Check context limit
        int ctx_size = llama_n_ctx(ctx);
        if (result.tokens_prompted + max_tokens > ctx_size) {
            LOG_WARN("Prompt + max_tokens (" +
                std::to_string(result.tokens_prompted + max_tokens) +
                ") exceeds context length (" +
                std::to_string(ctx_size) + ") - truncating max_tokens");
            max_tokens = ctx_size - result.tokens_prompted - 4;
            if (max_tokens <= 0) {
                throw LLMError("Context full - cannot generate. Clear KV cache.");
            }
        }

        // -------------------------------------------------------------------------
        // Build and fill batch with prompt tokens
        // -------------------------------------------------------------------------
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size())
        );

        if (llama_decode(ctx, batch) != 0) {
            throw LLMError("llama_decode failed on prompt");
        }

        // -------------------------------------------------------------------------
        // Build sampler chain
        // -------------------------------------------------------------------------
        llama_sampler* sampler = build_sampler(use_grammar, grammar_text);

        // -------------------------------------------------------------------------
        // Generation loop
        // -------------------------------------------------------------------------
        std::string generated_text;
        generated_text.reserve(max_tokens * 4);

        int tokens_generated = 0;
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        while (tokens_generated < max_tokens) {
            // 1. Sample next token
            llama_token token_id = llama_sampler_sample(sampler, ctx, -1);

            // 2. Accept into sampler (grammar state update)
            // NOTE: In b8660, grammar samplers advance state during sample(),
            // so accept() must only be called for non-grammar samplers
            if (!use_grammar) {
                llama_sampler_accept(sampler, token_id);
            }

            // 3. Check for EOS
            if (llama_vocab_is_eog(vocab, token_id)) {
                result.stopped_eos = true;
                break;
            }

            // 4. Convert token to text
            char token_buf[256] = {};
            int  token_len = llama_token_to_piece(
                vocab, token_id, token_buf, sizeof(token_buf) - 1, 0, true);

            if (token_len < 0) {
                LOG_WARN("Token buffer too small for token " +
                    std::to_string(token_id));
            }
            else {
                std::string token_text(token_buf, token_len);
                generated_text += token_text;
                ++tokens_generated;
                ++total_tokens_generated_;

                // 5. Fire callback
                if (callback) {
                    bool cont = callback(token_text, token_id, tokens_generated);
                    if (!cont) {
                        result.stopped_abort = true;
                        break;
                    }
                }
            }

            // 6. Decode next token position
            llama_batch next_batch = llama_batch_get_one(&token_id, 1);
            if (llama_decode(ctx, next_batch) != 0) {
                LOG_WARN("llama_decode failed mid-generation - stopping");
                break;
            }
        }

        if (tokens_generated >= max_tokens && !result.stopped_eos) {
            result.stopped_limit = true;
        }

        // -------------------------------------------------------------------------
        // Cleanup sampler
        // -------------------------------------------------------------------------
        llama_sampler_free(sampler);

        result.text = generated_text;
        result.tokens_generated = tokens_generated;
        result.success = true;

        return result;
    }

    // =============================================================================
    // build_sampler
    // Pass 1: grammar + greedy (deterministic, schema-constrained)
    // Pass 2: temperature + top_p (creative, free)
    // =============================================================================

    llama_sampler* LLMEngine::build_sampler(bool use_grammar,
        const std::string& grammar_text)
    {
        llama_sampler* chain = llama_sampler_chain_init(
            llama_sampler_chain_default_params());

        if (use_grammar && !grammar_text.empty()) {
            // Pass 1: GBNF grammar constraint + greedy sampling
            // Grammar ensures valid JSON schema output
            const llama_vocab* vocab = llama_model_get_vocab(model_);
            llama_sampler_chain_add(chain,
                llama_sampler_init_grammar(vocab,
                    grammar_text.c_str(), "root"));
            llama_sampler_chain_add(chain,
                llama_sampler_init_greedy());
        }
        else {
            // Pass 2: temperature + top_p for natural response generation
            llama_sampler_chain_add(chain,
                llama_sampler_init_temp(config_.inference.temperature));
            llama_sampler_chain_add(chain,
                llama_sampler_init_top_p(config_.inference.top_p, 1));
            llama_sampler_chain_add(chain,
                llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }

        return chain;
    }

    // =============================================================================
    // inject_synthetic_turn
    // Inserts the feeling state as a synthetic assistant turn before
    // the final user message, so the model treats it as its prior thought.
    // =============================================================================

    std::vector<ChatMessage> LLMEngine::inject_synthetic_turn(
        const std::vector<ChatMessage>& messages,
        const SyntheticTurn& turn) const
    {
        std::vector<ChatMessage> augmented;
        augmented.reserve(messages.size() + 1);

        // Find the last user message and inject synthetic turn before it
        // Pattern: [..., user_msg] -> [..., synthetic_assistant, user_msg]
        if (!messages.empty() && messages.back().role == "user") {
            // Copy all but last
            for (size_t i = 0; i < messages.size() - 1; ++i) {
                augmented.push_back(messages[i]);
            }
            // Inject synthetic turn
            augmented.push_back({ turn.role, turn.format() });
            // Add last user message
            augmented.push_back(messages.back());
        }
        else {
            // No trailing user message - just append synthetic turn
            augmented = messages;
            augmented.push_back({ turn.role, turn.format() });
        }

        return augmented;
    }

    // =============================================================================
    // Context management
    // =============================================================================

    void LLMEngine::clear_kv_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return;
        llama_memory_clear(llama_get_memory(ctx_pass1_), false);
        llama_memory_clear(llama_get_memory(ctx_pass2_), false);
        LOG_DEBUG("KV cache cleared (both contexts)");
    }

    int LLMEngine::context_tokens_used() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return 0;
        return static_cast<int>(total_tokens_generated_.load());
    }

    int LLMEngine::context_tokens_remaining() const {
        return context_length() - context_tokens_used();
    }

    bool LLMEngine::context_near_limit(float threshold) const {
        if (context_length() == 0) return false;
        return static_cast<float>(context_tokens_used()) /
            static_cast<float>(context_length()) >= threshold;
    }

    // =============================================================================
    // Tokenization
    // =============================================================================

    std::vector<llama_token> LLMEngine::tokenize(const std::string& text,
        bool add_special) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tokenize_internal(text, add_special);
    }

    // Internal version - caller must hold mutex_
    std::vector<llama_token> LLMEngine::tokenize_internal(
        const std::string& text, bool add_special) const
    {
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        // Get token count first
        int n = llama_tokenize(vocab,
            text.c_str(), static_cast<int>(text.size()),
            nullptr, 0, add_special, true);

        if (n < 0) n = -n; // llama returns negative count if buffer too small

        std::vector<llama_token> tokens(n);
        llama_tokenize(vocab,
            text.c_str(), static_cast<int>(text.size()),
            tokens.data(), n, add_special, true);

        return tokens;
    }

    std::string LLMEngine::detokenize(
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

    int LLMEngine::count_tokens(const std::string& text) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(tokenize_internal(text, false).size());
    }

    // =============================================================================
    // Chat template
    // =============================================================================

    std::string LLMEngine::apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) throw LLMError("Engine not ready");

        // Build llama_chat_message array
        std::vector<llama_chat_message> llama_messages;
        llama_messages.reserve(messages.size());

        for (const auto& msg : messages) {
            llama_messages.push_back({
                msg.role.c_str(),
                msg.content.c_str()
                });
        }

        // Get model's built-in template name
        // Pass nullptr to use model's own template
        const char* tmpl = nullptr;

        // First call: get required buffer size
        int32_t required = llama_chat_apply_template(
            tmpl,
            llama_messages.data(),
            llama_messages.size(),
            add_generation_prompt,
            nullptr, 0
        );

        if (required < 0) {
            throw LLMError("llama_chat_apply_template failed - model may lack chat template");
        }

        // Second call: fill buffer
        std::string result(required + 1, '\0');
        llama_chat_apply_template(
            tmpl,
            llama_messages.data(),
            llama_messages.size(),
            add_generation_prompt,
            result.data(),
            static_cast<int32_t>(result.size())
        );

        // Trim to actual length
        result.resize(required);
        return result;
    }

    // =============================================================================
    // Model info
    // =============================================================================

    std::string LLMEngine::model_name() const {
        if (!model_) return "none";
        std::string desc(256, '\0');
        llama_model_desc(model_, desc.data(), 256);
        return desc.c_str();
    }

    int LLMEngine::context_length() const {
        if (!ctx_pass2_) return 0;
        return static_cast<int>(llama_n_ctx(ctx_pass2_));
    }

    int LLMEngine::n_vocab() const {
        if (!model_) return 0;
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        return llama_vocab_n_tokens(vocab);
    }

    int LLMEngine::n_embd() const {
        if (!model_) return 0;
        return llama_model_n_embd(model_);
    }

} // namespace cardinal