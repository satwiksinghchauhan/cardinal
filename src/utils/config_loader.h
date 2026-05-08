// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Config Loader (v1.2.0)
// File: src/utils/config_loader.h
//
// Changes from v1.1.0:
//   - ToolsConfig expanded: each tool has its own config struct
//   - AgentConfig added
//   - ExplainabilityConfig added
//   - CardinalConfig gains agent and explainability fields
// =============================================================================

#include <string>
#include <vector>
#include <stdexcept>

namespace cardinal {

    // -------------------------------------------------------------------------
    // Backend configs (unchanged from v1.1.0)
    // -------------------------------------------------------------------------
    struct BackendLlamaCppConfig {
        std::string model_path;
        std::string chat_template;
        int         context_length;
        int         gpu_layers;
        int         threads;
    };

    struct BackendTensorRTConfig {
        std::string engine_path;
        std::string tokenizer_path;
        std::string chat_template;
        int         max_batch_size;
        int         max_seq_len;
        float       kv_cache_fraction;
        bool        use_int8;
    };

    struct BackendConfig {
        std::string           type;
        BackendLlamaCppConfig llama_cpp;
        BackendTensorRTConfig tensorrt;
    };

    // -------------------------------------------------------------------------
    // Per-tool config structs (new in v1.2.0)
    // -------------------------------------------------------------------------
    struct WebSearchToolConfig {
        bool        enabled               = true;
        bool        confirmation_required = false;
        int         max_results           = 5;
        int         timeout_seconds       = 10;
    };

    struct WebFetchToolConfig {
        bool                     enabled               = true;
        bool                     confirmation_required = false;
        std::vector<std::string> allowed_domains;
        std::vector<std::string> blocked_domains;
        int                      timeout_seconds       = 15;
        int                      max_content_kb        = 512;
    };

    struct CalculatorToolConfig {
        bool enabled               = true;
        bool confirmation_required = false;
    };

    struct RunPythonToolConfig {
        bool        enabled               = true;
        bool        confirmation_required = true;
        std::string sandbox_mode          = "subprocess"; // "subprocess" | "docker"
        std::string docker_image          = "python:3.12-slim";
        int         timeout_seconds       = 30;
        int         memory_limit_mb       = 256;
        bool        network_enabled       = false;
    };

    struct FileReadToolConfig {
        bool                     enabled               = true;
        bool                     confirmation_required = false;
        std::vector<std::string> allowed_paths;
    };

    struct FileWriteToolConfig {
        bool                     enabled               = true;
        bool                     confirmation_required = true;
        std::vector<std::string> allowed_paths;
    };

    struct KnowledgeGraphToolConfig {
        bool enabled               = true;
        bool confirmation_required = false;
    };

    struct EpisodicSearchToolConfig {
        bool enabled               = true;
        bool confirmation_required = false;
        int  max_results           = 5;
    };

    // -------------------------------------------------------------------------
    // ToolsConfig (replaces original flat ToolsConfig)
    // -------------------------------------------------------------------------
    struct ToolsConfig {
        bool home_access = false;
        WebSearchToolConfig      web_search;
        WebFetchToolConfig       web_fetch;
        CalculatorToolConfig     calculator;
        RunPythonToolConfig      run_python;
        FileReadToolConfig       file_read;
        FileWriteToolConfig      file_write;
        KnowledgeGraphToolConfig knowledge_graph;
        EpisodicSearchToolConfig episodic_search;
    };

    // -------------------------------------------------------------------------
    // VisionConfig (new in v1.3.0)
    // -------------------------------------------------------------------------
    struct VisionConfig {
        std::string              model_path;       // path to moondream2 GGUF
        std::string              mmproj_path;      // path to mmproj GGUF
        int                      gpu_layers      = 0;    // 0 = CPU only
        int                      threads         = 4;
        int                      max_tokens      = 512;
        std::string              cache_path      = "data/vision_cache";
        int                      cache_ttl_hours = 24;   // 0 = keep forever
        int                      download_timeout_seconds = 30;
        bool                     confirmation_required    = false;
        std::vector<std::string> allowed_paths;   // for file:// inputs
    };

    // -------------------------------------------------------------------------
    // AgentConfig (new in v1.2.0)
    // -------------------------------------------------------------------------
    struct AgentConfig {
        bool        enabled                      = true;
        int         max_iterations               = 10;
        int         max_iterations_hard_cap      = 50;
        std::string working_memory_path          = "data/memory/agent_working_memory";
        int         working_memory_size          = 50;
        bool        self_correction_enabled      = true;
        int         self_correction_max_attempts = 3;
        bool        plan_before_execute          = true;
        bool        summarize_on_cap             = true;
    };

    // -------------------------------------------------------------------------
    // ExplainabilityConfig (new in v1.2.0)
    // -------------------------------------------------------------------------
    struct ExplainabilityConfig {
        bool        enabled                  = true;
        std::string audit_log_path           = "data/explainability/audit.db";
        bool        signing_enabled          = true;
        std::string private_key_path         = "data/explainability/cardinal_private.pem";
        std::string public_key_path          = "data/explainability/cardinal_public.pem";
        bool        auto_generate_keys       = true;
        std::string export_path              = "data/explainability/exports";
        bool        attach_trace_to_response = true;
    };

    // -------------------------------------------------------------------------
    // Unchanged structs from v1.1.0
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
        std::string                     type;
        std::string                     grammar_path;
        std::vector<FeelingSchemaField> fields;
        int                             max_tokens;
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
    // CardinalConfig (v1.2.0)
    // -------------------------------------------------------------------------
    struct CardinalConfig {
        BackendConfig         backend;
        InferenceConfig       inference;
        FeelingSchemaConfig   feeling_schema;
        MemoryConfig          memory;
        VerifierConfig        verifier;
        FeedbackConfig        feedback;
        RetrieverConfig       retriever;
        ApiConfig             api;
        ToolsConfig           tools;           // expanded per-tool configs
        AgentConfig           agent;
        ExplainabilityConfig  explainability;
        VisionConfig          vision;          // new in v1.3.0
        BenchmarkConfig       benchmark;
        LoggingConfig         logging;
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
        static BackendConfig          parse_backend(const auto& j);
        static BackendLlamaCppConfig  parse_backend_llama_cpp(const auto& j);
        static BackendTensorRTConfig  parse_backend_tensorrt(const auto& j);
        static InferenceConfig        parse_inference(const auto& j);
        static FeelingSchemaConfig    parse_feeling_schema(const auto& j);
        static MemoryConfig           parse_memory(const auto& j);
        static VerifierConfig         parse_verifier(const auto& j);
        static FeedbackConfig         parse_feedback(const auto& j);
        static RetrieverConfig        parse_retriever(const auto& j);
        static ApiConfig              parse_api(const auto& j);
        static ToolsConfig            parse_tools(const auto& j);       // expanded
        static VisionConfig           parse_vision(const auto& j);      // new in v1.3.0
        static AgentConfig            parse_agent(const auto& j);
        static ExplainabilityConfig   parse_explainability(const auto& j); // new
        static BenchmarkConfig        parse_benchmark(const auto& j);
        static LoggingConfig          parse_logging(const auto& j);
    };

    class ConfigError : public std::runtime_error {
    public:
        explicit ConfigError(const std::string& message)
            : std::runtime_error("ConfigError: " + message) {}
    };

} // namespace cardinal
