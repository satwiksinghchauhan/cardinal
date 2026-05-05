// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - TensorRT-LLM Backend
// File: src/core/backends/tensorrt_backend.h
//
// Concrete ILLMBackend implementation using NVIDIA TensorRT-LLM.
//
// API used: tensorrt_llm::executor (TRT-LLM >= 0.8, the modern C++ executor).
// Tokenization: HuggingFace tokenizers-cpp (same lib TRT-LLM uses internally).
// Chat template: manual Jinja2-style rendering (no llama.cpp dependency).
//
// Constrained decoding strategy:
//   supports_constrained_decoding() returns false for now.
//   generate_feeling() uses JSON-schema validation + retry loop (Approach C).
//   When lm-format-enforcer / Outlines is integrated, flip the flag and wire
//   the logits processor into build_sampling_config() — nothing else changes.
//
// Prerequisites (Ubuntu):
//   - CUDA 12.x + cuDNN 9.x + TensorRT 10.x
//   - TensorRT-LLM 0.8+ built from source (sm_86 target for RTX 3050)
//   - tokenizers-cpp (FetchContent, see CMakeLists.txt)
//   - Engine file pre-built via trtllm-build from HuggingFace weights
//
// Build:
//   cmake -DCARDINAL_ENABLE_TENSORRT=ON ...
// =============================================================================

#ifdef CARDINAL_ENABLE_TENSORRT

#include "core/llm_backend.h"
#include "utils/config_loader.h"
#include "utils/logger.h"
#include "core/feeling_output.h"

// TensorRT-LLM executor API
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

// HuggingFace tokenizers-cpp
#include "tokenizers_cpp.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>
#include <optional>
#include <filesystem>

namespace cardinal {

namespace trtllm = tensorrt_llm::executor;

// -----------------------------------------------------------------------------
// FeelingSchema
// Mirrors the JSON fields in feeling_schema.gbnf.
// Used for validation in the retry loop.
// -----------------------------------------------------------------------------
struct FeelingSchema {
    static const char* VALID_REASONING_TYPES[];
    static const char* VALID_DOMAINS[];

    // Validate a parsed JSON string contains all required fields with valid values.
    // Returns empty string on success, error description on failure.
    static std::string validate(const std::string& json_text);
};

// -----------------------------------------------------------------------------
// TensorRTBackend
// One instance per process. Owns the TRT-LLM executor and tokenizer.
// Thread-safe via internal mutex_ for all public methods.
// -----------------------------------------------------------------------------
class TensorRTBackend final : public ILLMBackend {
public:
    explicit TensorRTBackend(const CardinalConfig& config);
    ~TensorRTBackend() override;

    // -------------------------------------------------------------------------
    // ILLMBackend — Lifecycle
    // -------------------------------------------------------------------------
    void load_model() override;
    bool is_ready()   const override;
    void unload()     override;

    // -------------------------------------------------------------------------
    // ILLMBackend — Two-pass inference
    // -------------------------------------------------------------------------
    GenerationResult generate_feeling(
        FeelingContext&                 ctx,
        const std::vector<ChatMessage>& messages) override;

    GenerationResult generate_response(
        FeelingContext&                 ctx,
        const std::vector<ChatMessage>& messages,
        const TokenCallback&            callback = nullptr) override;

    // -------------------------------------------------------------------------
    // ILLMBackend — Context management
    // -------------------------------------------------------------------------
    void clear_kv_cache()                         override;
    bool context_near_limit(float t = 0.9f) const override;
    int  context_tokens_used()               const override;
    int  context_tokens_remaining()          const override;

    // -------------------------------------------------------------------------
    // ILLMBackend — Capability / info
    // -------------------------------------------------------------------------

    // Returns false — constrained decoding via logits processor not yet wired.
    // generate_feeling() uses JSON validation + retry loop instead.
    // Flip to true when lm-format-enforcer is integrated.
    bool        supports_constrained_decoding() const override { return false; }
    BackendInfo get_info()                      const override;

private:
    // -------------------------------------------------------------------------
    // Internal generation
    // -------------------------------------------------------------------------

    // Core synchronous generation — used by Pass 1 (no streaming).
    // Submits request to executor, polls until complete, returns full text.
    GenerationResult generate_sync(
        const std::string& prompt,
        int                max_tokens,
        float              temperature,
        float              top_p);

    // Streaming generation — used by Pass 2.
    // Polls executor response queue and fires callback per token chunk.
    GenerationResult generate_stream(
        const std::string&   prompt,
        int                  max_tokens,
        float                temperature,
        float                top_p,
        const TokenCallback& callback);

    // Pass 1 retry loop: generate → validate JSON schema → retry on failure.
    // Returns GenerationResult of the first successful attempt, or the last
    // failed attempt if all retries exhausted.
    GenerationResult generate_feeling_with_retry(
        const std::string& prompt,
        int                max_retries,
        int                retry_delay_ms);

    // Build the TRT-LLM SamplingConfig for a request.
    trtllm::SamplingConfig build_sampling_config(float temperature,
                                                  float top_p,
                                                  int   max_tokens) const;

    // Build the OutputConfig (controls what the executor returns).
    trtllm::OutputConfig build_output_config(bool return_log_probs = false) const;

    // -------------------------------------------------------------------------
    // Tokenization
    // -------------------------------------------------------------------------

    // Encode text → token IDs. Adds BOS if add_special=true.
    std::vector<int32_t> encode(const std::string& text,
                                bool               add_special = true) const;

    // Decode token IDs → text.
    std::string decode(const std::vector<int32_t>& token_ids) const;

    // Count tokens without full encode/decode round-trip.
    int count_tokens(const std::string& text) const;

    // -------------------------------------------------------------------------
    // Chat template
    // -------------------------------------------------------------------------

    // Apply chat template for the configured model family.
    // Supports: qwen3, llama3, chatml (detected from config.backend.tensorrt.chat_template).
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt = true) const;

    // Template renderers per model family
    std::string render_qwen3(const std::vector<ChatMessage>& messages,
                              bool add_generation_prompt) const;
    std::string render_llama3(const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt) const;
    std::string render_chatml(const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt) const;

    // -------------------------------------------------------------------------
    // Synthetic turn injection (Pass 2)
    // Identical logic to LlamaCppBackend — inserts feeling state before last
    // user message so model treats it as its prior internal monologue.
    // -------------------------------------------------------------------------
    std::vector<ChatMessage> inject_synthetic_turn(
        const std::vector<ChatMessage>& messages,
        const SyntheticTurn&            turn) const;

    // -------------------------------------------------------------------------
    // Feeling instruction injection (Pass 1)
    // Augments system message with JSON output instruction.
    // Same text as LlamaCppBackend for consistency.
    // -------------------------------------------------------------------------
    std::vector<ChatMessage> inject_feeling_instruction(
        const std::vector<ChatMessage>& messages) const;

    // -------------------------------------------------------------------------
    // Members
    // -------------------------------------------------------------------------
    const CardinalConfig& config_;

    // TRT-LLM executor — manages the inference engine, KV cache, batching.
    std::unique_ptr<trtllm::Executor> executor_;

    // HuggingFace tokenizer — loaded from config.backend.tensorrt.tokenizer_path.
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    // Engine metadata (populated during load_model())
    int  max_seq_len_       = 0;
    int  vocab_size_        = 0;
    bool ready_             = false;

    // Running token counter for context_tokens_used()
    std::atomic<int> total_tokens_generated_{ 0 };

    // Approximate prompt tokens for the current session (reset on clear_kv_cache)
    std::atomic<int> session_prompt_tokens_{ 0 };

    mutable std::mutex mutex_;
};

} // namespace cardinal

#endif // CARDINAL_ENABLE_TENSORRT
