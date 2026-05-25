#pragma once
// =============================================================================
// Cardinal - Config Loader (v1.5.0)
// File: src/utils/config_loader.h
//
// Changes from v1.4.0:
//   - SchedulerConfig           added
//   - ComputerUseSafetyConfig   added
//   - ComputerUseScreenConfig   added
//   - ComputerUseBrowserConfig  added
//   - ComputerUseShellConfig    added
//   - ComputerUseEmailConfig    added
//   - ComputerUseAtSpiConfig    added
//   - ComputerUseConfig         added  (top-level wrapper)
//   - CardinalConfig gains       scheduler + computer_use fields
//   - ConfigLoader gains         parse_scheduler / parse_computer_use
// =============================================================================

#include <string>
#include <vector>
#include <stdexcept>

namespace cardinal {

    // -------------------------------------------------------------------------
    // Backend configs (unchanged)
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
    // Per-tool configs (unchanged from v1.3.0)
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
        std::string sandbox_mode          = "subprocess";
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
    // VisionConfig (unchanged from v1.3.0)
    // -------------------------------------------------------------------------
    struct VisionConfig {
        std::string              model_path;
        std::string              mmproj_path;
        int                      gpu_layers              = 0;
        int                      threads                 = 4;
        int                      max_tokens              = 512;
        std::string              cache_path              = "data/vision_cache";
        int                      cache_ttl_hours         = 24;
        int                      download_timeout_seconds = 30;
        bool                     confirmation_required   = false;
        std::vector<std::string> allowed_paths;
    };

    // -------------------------------------------------------------------------
    // AgentConfig (unchanged from v1.2.0)
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
    // ExplainabilityConfig (unchanged from v1.2.0)
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
    // SelfModelConfig (new in v1.4.0)
    // -------------------------------------------------------------------------
    struct SelfModelConfig {
        bool        enabled            = true;
        std::string db_path            = "data/self_model/self_model.db";
        bool        inject_into_prompt = true;
        int         prompt_max_chars   = 500;
        int         history_window     = 100;
    };

    struct MetaCognitionLoaderConfig {
        bool  enabled                            = true;
        int   trigger_every_n_inferences         = 20;
        float trigger_on_contradiction_rate_pct  = 30.0f;
        bool  on_demand_via_api                  = true;
        int   min_failures_to_reflect            = 5;
        int   max_corrective_rules_per_session   = 10;
        float corrective_rule_confidence         = 0.6f;
    };

    struct TrainingConfig {
        bool        enabled                          = true;
        int         lora_rank                        = 8;
        int         lora_alpha                       = 16;
        float       learning_rate                    = 0.0001f;
        int         epochs                           = 3;
        int         batch_size                       = 4;
        int         min_episodes_for_training        = 50;
        float       min_quality_confidence           = 0.75f;
        int         max_examples                     = 0;
        int         trigger_every_n_episodes         = 100;
        int         trigger_every_n_hours            = 24;
        float       trigger_on_domain_confidence_below = 0.5f;
        std::string adapter_load_policy              = "session_boundary";
        float       eval_improvement_threshold_pct   = 5.0f;
        int         eval_holdout_episodes            = 20;
        std::string adapter_output_dir               = "data/training/adapters";
        std::string dataset_output_dir               = "data/training/datasets";
        std::string export_script_dir                = "data/training/scripts";
        std::string hf_model_path                    = "models/qwen3.5-4b-hf";
        std::string python_venv                      = "~/cardinal/cardinal-train-venv";
        std::string convert_lora_script              = "vendor/llama.cpp/convert_lora_to_gguf.py";
        std::string llama_finetune_binary            = "vendor/llama.cpp/build/bin/llama-finetune";
    };

    struct SelfImprovementConfig {
        bool                     enabled        = true;
        SelfModelConfig          self_model;
        MetaCognitionLoaderConfig meta_cognition;
        TrainingConfig           training;
    };

    // -------------------------------------------------------------------------
    // Unchanged structs from v1.3.0
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
    // SchedulerConfig  (new in v1.5.0)
    // -------------------------------------------------------------------------
    struct SchedulerConfig {
        bool        enabled                      = false;
        std::string db_path                      = "data/scheduler/scheduler.db";
        int         check_interval_seconds       = 30;
        int         max_concurrent_tasks         = 1;
        int         idle_threshold_minutes       = 5;
        std::string task_session_prefix          = "scheduler_";
        int         run_history_max_entries      = 1000;
        int         max_task_duration_seconds    = 300;
    };

    // -------------------------------------------------------------------------
    // ComputerUseConfig  (new in v1.5.0)
    // -------------------------------------------------------------------------
    struct ComputerUseSafetyConfig {
        bool                     whitelist_enabled            = true;
        std::vector<std::string> allowed_apps;
        std::vector<std::string> allowed_domains;
        std::vector<std::string> allowed_paths;
        std::vector<std::string> blocked_commands             = {
            "rm -rf /", "mkfs", "dd if=", "sudo rm -rf", ":(){:|:&};:"
        };
        bool                     confirmation_required        = true;
        int                      confirmation_timeout_seconds = 30;
        bool                     watch_mode                   = true;
        bool                     full_autonomy                = false;
        bool                     allow_file_write             = false;
    };

    struct ComputerUseScreenConfig {
        std::string screenshot_tool        = "auto";
        bool        vision_analysis        = true;
        int         watch_interval_seconds = 2;
    };

    struct ComputerUseBrowserConfig {
        std::string executable              = "google-chrome";
        std::string venv_path               = "~/cardinal/cardinal-browser-venv";
        int         playwright_timeout_ms   = 10000;
        bool        headless                = false;
        std::string user_data_dir           = "data/browser_profile";
    };

    struct ComputerUseShellConfig {
        bool        enabled           = true;
        std::string shell             = "/bin/bash";
        int         timeout_seconds   = 30;
        std::string working_directory = "~";
    };

    struct ComputerUseEmailConfig {
        bool        enabled                  = false;
        std::string mode                     = "imap_smtp";
        std::string imap_host;
        int         imap_port                = 993;
        std::string smtp_host;
        int         smtp_port                = 587;
        std::string address;
        bool        gmail_api_enabled        = false;
        std::string gmail_credentials_path   = "data/gmail_credentials.json";
    };

    struct ComputerUseAtSpiConfig {
        bool enabled             = true;
        bool fallback_to_vision  = true;
    };

    struct ComputerUseConfig {
        bool                       enabled  = false;
        ComputerUseSafetyConfig    safety;
        ComputerUseScreenConfig    screen;
        ComputerUseBrowserConfig   browser;
        ComputerUseShellConfig     shell;
        ComputerUseEmailConfig     email;
        ComputerUseAtSpiConfig     atspi;
    };

    // -------------------------------------------------------------------------
    // CardinalConfig (v1.5.0)
    // -------------------------------------------------------------------------
    struct CardinalConfig {
        BackendConfig          backend;
        InferenceConfig        inference;
        FeelingSchemaConfig    feeling_schema;
        MemoryConfig           memory;
        VerifierConfig         verifier;
        FeedbackConfig         feedback;
        RetrieverConfig        retriever;
        ApiConfig              api;
        ToolsConfig            tools;
        AgentConfig            agent;
        ExplainabilityConfig   explainability;
        VisionConfig           vision;
        SelfImprovementConfig  self_improvement;
        SchedulerConfig        scheduler;       // ← new in v1.5.0
        ComputerUseConfig      computer_use;    // ← new in v1.5.0
        BenchmarkConfig        benchmark;
        LoggingConfig          logging;
    };

    // -------------------------------------------------------------------------
    // ConfigLoader (v1.5.0)
    // -------------------------------------------------------------------------
    class ConfigLoader {
    public:
        static CardinalConfig load(const std::string& path);
        static CardinalConfig reload(const std::string& path);
        static void           validate(const CardinalConfig& config);
        static std::string    to_json_string(const CardinalConfig& config);

    private:
        static BackendConfig              parse_backend(const auto& j);
        static BackendLlamaCppConfig      parse_backend_llama_cpp(const auto& j);
        static BackendTensorRTConfig      parse_backend_tensorrt(const auto& j);
        static InferenceConfig            parse_inference(const auto& j);
        static FeelingSchemaConfig        parse_feeling_schema(const auto& j);
        static MemoryConfig               parse_memory(const auto& j);
        static VerifierConfig             parse_verifier(const auto& j);
        static FeedbackConfig             parse_feedback(const auto& j);
        static RetrieverConfig            parse_retriever(const auto& j);
        static ApiConfig                  parse_api(const auto& j);
        static ToolsConfig                parse_tools(const auto& j);
        static VisionConfig               parse_vision(const auto& j);
        static AgentConfig                parse_agent(const auto& j);
        static ExplainabilityConfig       parse_explainability(const auto& j);
        static SelfImprovementConfig      parse_self_improvement(const auto& j);
        static SchedulerConfig            parse_scheduler(const auto& j);      // new
        static ComputerUseConfig          parse_computer_use(const auto& j);   // new
        static BenchmarkConfig            parse_benchmark(const auto& j);
        static LoggingConfig              parse_logging(const auto& j);
    };

    class ConfigError : public std::runtime_error {
    public:
        explicit ConfigError(const std::string& message)
            : std::runtime_error("ConfigError: " + message) {}
    };

} // namespace cardinal
