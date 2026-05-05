// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - LLM Backend Interface
// File: src/core/llm_backend.h
//
// Pure abstract interface every inference backend must implement.
// InferencePipeline and CardinalAPI talk ONLY to this — never to a concrete
// backend directly. Swap backends by changing config.json "backend.type".
//
// Backends:
//   LlamaCppBackend   — llama.cpp + GBNF constrained decoding
//   TensorRTBackend   — TensorRT-LLM, JSON-schema retry fallback
//
// Adding a new backend:
//   1. Create src/core/backends/your_backend.h/.cpp
//   2. Inherit ILLMBackend, implement all pure virtuals
//   3. Add a case in BackendFactory::create()
//   4. Add a config block under "backend" in config.json
// =============================================================================

#include "utils/config_loader.h"
#include "core/feeling_output.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace cardinal {

    // -------------------------------------------------------------------------
    // TokenCallback
    // Called for each token during streaming generation (Pass 2).
    // Return false to abort generation early.
    // Signature is identical to the original LLMEngine definition so all
    // existing call sites in InferencePipeline compile unchanged.
    // -------------------------------------------------------------------------
    using TokenCallback = std::function<bool(const std::string& token_text,
                                             int                token_id,
                                             int                tokens_generated)>;

    // -------------------------------------------------------------------------
    // GenerationResult
    // Returned by both generate_feeling() and generate_response().
    // Identical to the original struct — no callers need changes.
    // -------------------------------------------------------------------------
    struct GenerationResult {
        std::string text;               // Full generated text
        int         tokens_generated;   // Tokens produced
        int         tokens_prompted;    // Tokens in the prompt
        bool        stopped_eos;        // Natural end-of-sequence stop
        bool        stopped_limit;      // Hit max_tokens ceiling
        bool        stopped_abort;      // Callback returned false
        bool        success;            // Overall flag

        bool completed_naturally() const { return stopped_eos && !stopped_abort; }
    };

    // -------------------------------------------------------------------------
    // BackendType
    // Matches the string values accepted in config.json "backend.type".
    // -------------------------------------------------------------------------
    enum class BackendType {
        LLAMA_CPP,
        TENSORRT,
        UNKNOWN
    };

    inline BackendType backend_type_from_string(const std::string& s) {
        if (s == "llama_cpp")  return BackendType::LLAMA_CPP;
        if (s == "tensorrt")   return BackendType::TENSORRT;
        return BackendType::UNKNOWN;
    }

    inline std::string backend_type_to_string(BackendType t) {
        switch (t) {
            case BackendType::LLAMA_CPP: return "llama_cpp";
            case BackendType::TENSORRT:  return "tensorrt";
            default:                     return "unknown";
        }
    }

    // -------------------------------------------------------------------------
    // BackendInfo
    // Returned by ILLMBackend::get_info() — used for logging and /stats.
    // -------------------------------------------------------------------------
    struct BackendInfo {
        BackendType type;
        std::string name;           // Human-readable ("llama.cpp", "TensorRT-LLM")
        std::string version;        // Library version string
        std::string model_name;     // Model identifier after load
        int         context_length; // Max context tokens
        int         n_vocab;        // Vocabulary size
        bool        gpu_enabled;    // Whether a GPU is in use
        int         gpu_layers;     // Layers offloaded to GPU (if applicable)
    };

    // -------------------------------------------------------------------------
    // ILLMBackend
    // Pure abstract base. One instance per process, owned by CardinalAPI via
    // unique_ptr<ILLMBackend>. InferencePipeline holds a reference.
    //
    // Thread-safety contract: implementations must serialise generate_* calls
    // internally (as LlamaCppBackend already does via mutex_).
    // -------------------------------------------------------------------------
    class ILLMBackend {
    public:
        virtual ~ILLMBackend() = default;

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------

        // Load model and initialise all resources. Throws on failure.
        // Called once by CardinalAPI::init().
        virtual void load_model() = 0;

        // True after load_model() succeeds and before unload().
        virtual bool is_ready() const = 0;

        // Release all resources. Safe to call from destructor.
        virtual void unload() = 0;

        // ------------------------------------------------------------------
        // Two-pass inference
        // Signatures are drop-in replacements for the original LLMEngine
        // methods — InferencePipeline compiles without modification.
        // ------------------------------------------------------------------

        // Pass 1: constrained decoding → feeling output JSON.
        // Backends with supports_constrained_decoding() == true use native
        // grammar enforcement (GBNF, Outlines, etc.).
        // Backends without it fall back to generate + JSON validation + retry.
        virtual GenerationResult generate_feeling(
            FeelingContext&                 ctx,
            const std::vector<ChatMessage>& messages) = 0;

        // Pass 2: free decoding with synthetic turn injected.
        // callback may be nullptr (non-streaming).
        virtual GenerationResult generate_response(
            FeelingContext&                 ctx,
            const std::vector<ChatMessage>& messages,
            const TokenCallback&            callback = nullptr) = 0;

        // ------------------------------------------------------------------
        // Context management
        // ------------------------------------------------------------------

        // Clear KV cache between unrelated sessions.
        virtual void clear_kv_cache() = 0;

        // Returns true when token usage exceeds threshold (default 90%).
        virtual bool context_near_limit(float threshold = 0.9f) const = 0;

        // Raw token counts — used by InferencePipeline::context_near_limit().
        virtual int context_tokens_used()      const = 0;
        virtual int context_tokens_remaining() const = 0;

        // ------------------------------------------------------------------
        // Capability query
        // ------------------------------------------------------------------

        // True  → backend enforces the feeling schema natively during sampling
        //         (llama.cpp GBNF, future: Outlines for TRT).
        // False → ILLMBackend caller must use JSON-schema retry loop.
        //
        // InferencePipeline does NOT check this — it is used by each backend's
        // own generate_feeling() implementation to choose its internal path.
        // Exposed here so CardinalAPI can log the active strategy at startup.
        virtual bool supports_constrained_decoding() const = 0;

        // Metadata snapshot — logged at startup and returned in /stats.
        virtual BackendInfo get_info() const = 0;

        // ------------------------------------------------------------------
        // Non-copyable / non-movable (owns hardware resources)
        // ------------------------------------------------------------------
        ILLMBackend(const ILLMBackend&)            = delete;
        ILLMBackend& operator=(const ILLMBackend&) = delete;
        ILLMBackend(ILLMBackend&&)                 = delete;
        ILLMBackend& operator=(ILLMBackend&&)      = delete;

    protected:
        ILLMBackend() = default;
    };

    // -------------------------------------------------------------------------
    // LLMError — thrown by any backend on unrecoverable engine failure.
    // Kept here so all backends and callers share one exception type.
    // -------------------------------------------------------------------------
    class LLMError : public std::runtime_error {
    public:
        explicit LLMError(const std::string& msg)
            : std::runtime_error("LLMError: " + msg) {}
    };

} // namespace cardinal
