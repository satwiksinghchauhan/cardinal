// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Config Loader
// File: src/utils/config_loader.h
// =============================================================================

#include <string>
#include <vector>
#include <stdexcept>

namespace cardinal {

    struct ModelConfig {
        std::string path;
        std::string chat_template;
        int         context_length;
        int         gpu_layers;
        int         threads;
    };

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

    // -------------------------------------------------------------------------
    // ApiConfig
    // Controls the HTTP server used by TypeScript (Interface 2) and
    // any other external consumers. Auth is optional -- disable for
    // pure localhost development, enable for any networked deployment.
    // -------------------------------------------------------------------------
    struct ApiConfig {
        bool        http_enabled;               // Enable HTTP server on startup
        std::string host;                       // Bind address (e.g. "127.0.0.1")
        int         port;                       // Listen port (e.g. 8080)
        bool        auth_enabled;               // Require API key on all requests
        std::string api_key;                    // Bearer token for auth
        bool        stream_enabled;             // Allow SSE streaming responses
        int         max_request_size_kb;        // Max request body size in KB
        int         request_timeout_seconds;    // Per-request timeout
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
    // -------------------------------------------------------------------------
    struct CardinalConfig {
        ModelConfig         model;
        InferenceConfig     inference;
        FeelingSchemaConfig feeling_schema;
        MemoryConfig        memory;
        VerifierConfig      verifier;
        FeedbackConfig      feedback;
        RetrieverConfig     retriever;
        ApiConfig           api;           // Phase 6 API layer
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
        static ModelConfig         parse_model(const auto& j);
        static InferenceConfig     parse_inference(const auto& j);
        static FeelingSchemaConfig parse_feeling_schema(const auto& j);
        static MemoryConfig        parse_memory(const auto& j);
        static VerifierConfig      parse_verifier(const auto& j);
        static FeedbackConfig      parse_feedback(const auto& j);
        static RetrieverConfig     parse_retriever(const auto& j);
        static ApiConfig           parse_api(const auto& j);     // Phase 6
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