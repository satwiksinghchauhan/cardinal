// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Config Loader
// File: src/utils/config_loader.h
//
// Changes from original:
//   - ModelConfig removed from CardinalConfig (fields now inside BackendConfig)
//   - BackendLlamaCppConfig, BackendTensorRTConfig, BackendConfig added
//   - CardinalConfig.model replaced with CardinalConfig.backend
//   - parse_backend() / parse_backend_llama_cpp() / parse_backend_tensorrt()
//     added to ConfigLoader private interface
//   - All other structs and methods are identical to original
// =============================================================================

#include <string>
#include <vector>
#include <stdexcept>

namespace cardinal {

    // -------------------------------------------------------------------------
    // BackendLlamaCppConfig
    // Fields previously in ModelConfig, now nested under backend.llama_cpp
    // -------------------------------------------------------------------------
    struct BackendLlamaCppConfig {
        std::string model_path;       // Path to .gguf file
        std::string chat_template;    // "qwen3", "llama3", "chatml", etc.
        int         context_length;
        int         gpu_layers;
        int         threads;
    };

    // -------------------------------------------------------------------------
    // BackendTensorRTConfig
    // -------------------------------------------------------------------------
    struct BackendTensorRTConfig {
        std::string engine_path;      // Path to pre-built .engine directory
        std::string tokenizer_path;   // Path to HuggingFace tokenizer directory
        std::string chat_template;    // "qwen3", "llama3", "chatml"
        int         max_batch_size;
        int         max_seq_len;
        float       kv_cache_fraction;
        bool        use_int8;
    };

    // -------------------------------------------------------------------------
    // BackendConfig
    // Top-level backend block. "type" selects which sub-config is active.
    // Both sub-configs are always parsed so config.json is validated fully
    // regardless of which backend is currently selected.
    // -------------------------------------------------------------------------
    struct BackendConfig {
        std::string            type;       // "llama_cpp" | "tensorrt"
        BackendLlamaCppConfig  llama_cpp;
        BackendTensorRTConfig  tensorrt;
    };

    // -------------------------------------------------------------------------
    // All structs below are unchanged from original
    // -------------------------------------------------------------------------

    struct InferenceConfig {
        float temperature;
        float top_p;
        int   max_tokens_feeling;
        int   max_tokens_response;
    };

    struct FeelingSchemaField {
        std::string name;
        std::string type_description;
    };

    struct FeelingSchemaConfig {
        std::string                      type;
        std::string                      grammar_path;
        std::vector<FeelingSchemaField>  fields;
        int                              max_tokens;
    };

    struct MemoryConfig {
        std::string rule_store_path;
        std::string knowledge_graph_path;
        std::string episodic_log_path;
        int         max_rules;
    };

    struct VerifierConfig {
        std::string mode;
        std::string neural_model_path;
        int         neural_gpu_layers;
        int         neural_max_tokens;
        float       contradiction_threshold;
        float       rule_confidence_decay;
        float       min_rule_confidence;
    };

    struct FeedbackConfig {
        int         max_retries;
        int         retry_delay_ms;
        std::string rule_injection_format;
    };

    struct RetrieverConfig {
        std::string mode;
        float       keyword_weight;
        float       semantic_weight;
        int         max_results;
        float       min_score;
        std::string cache_rebuild_strategy;
        int         cache_rebuild_threshold;
        int         cache_rebuild_interval_seconds;
    };

    struct ApiConfig {
        bool        http_enabled;
        std::string host;
        int         port;
        bool        auth_enabled;
        std::string api_key;
        bool        stream_enabled;
        int         max_request_size_kb;
        int         request_timeout_seconds;
    };

    struct ToolsConfig {
        bool browser_enabled;
        int  max_browse_depth;
        int  search_results_limit;
    };

    struct BenchmarkConfig {
        std::string              dataset;
        std::vector<std::string> metrics;
        int                      eval_frequency_seconds;
    };

    struct LoggingConfig {
        std::string level;
        std::string path;
    };

    // -------------------------------------------------------------------------
    // CardinalConfig
    // Note: `model` field is gone. Use `backend.llama_cpp` or `backend.tensorrt`
    // depending on `backend.type`. BackendFactory reads `backend.type` to decide
    // which concrete ILLMBackend to construct.
    // -------------------------------------------------------------------------
    struct CardinalConfig {
        BackendConfig       backend;          // ← replaces ModelConfig model
        InferenceConfig     inference;
        FeelingSchemaConfig feeling_schema;
        MemoryConfig        memory;
        VerifierConfig      verifier;
        FeedbackConfig      feedback;
        RetrieverConfig     retriever;
        ApiConfig           api;
        ToolsConfig         tools;
        BenchmarkConfig     benchmark;
        LoggingConfig       logging;
    };

    // -------------------------------------------------------------------------
    // ConfigLoader
    // -------------------------------------------------------------------------
    class ConfigLoader {
    public:
        static CardinalConfig load(const std::string& path);
        static CardinalConfig reload(const std::string& path);
        static void           validate(const CardinalConfig& config);
        static std::string    to_json_string(const CardinalConfig& config);

    private:
        // New backend parsers
        static BackendConfig          parse_backend(const auto& j);
        static BackendLlamaCppConfig  parse_backend_llama_cpp(const auto& j);
        static BackendTensorRTConfig  parse_backend_tensorrt(const auto& j);

        // Unchanged parsers
        static InferenceConfig     parse_inference(const auto& j);
        static FeelingSchemaConfig parse_feeling_schema(const auto& j);
        static MemoryConfig        parse_memory(const auto& j);
        static VerifierConfig      parse_verifier(const auto& j);
        static FeedbackConfig      parse_feedback(const auto& j);
        static RetrieverConfig     parse_retriever(const auto& j);
        static ApiConfig           parse_api(const auto& j);
        static ToolsConfig         parse_tools(const auto& j);
        static BenchmarkConfig     parse_benchmark(const auto& j);
        static LoggingConfig       parse_logging(const auto& j);
    };

    class ConfigError : public std::runtime_error {
    public:
        explicit ConfigError(const std::string& message)
            : std::runtime_error("ConfigError: " + message) {}
    };

} // namespace cardinal
