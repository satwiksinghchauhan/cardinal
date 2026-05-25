// =============================================================================
// Cardinal - Config Loader Implementation (v1.5.0)
// File: src/utils/config_loader.cpp
//
// Changes from v1.3.0:
//   v1.4.0: parse_self_improvement() added; load() wired;
//           to_json_string() extended with self_improvement block
//   v1.5.0: parse_scheduler() + parse_computer_use() added; load() wired;
//           to_json_string() extended with scheduler + computer_use blocks
// =============================================================================

#include "config_loader.h"
#include "logger.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>

using json = nlohmann::json;

namespace cardinal {

    namespace {
        template<typename T>
        T require(const json& j, const std::string& section,
                  const std::string& key) {
            if (!j.contains(key))
                throw ConfigError("Missing required field '" + key +
                                  "' in section '" + section + "'");
            try { return j.at(key).get<T>(); }
            catch (const json::exception& e) {
                throw ConfigError("Invalid type for '" + key +
                                  "' in '" + section + "': " + e.what());
            }
        }

        template<typename T>
        T opt(const json& j, const std::string& key, T def) {
            if (!j.contains(key)) return def;
            try { return j.at(key).get<T>(); }
            catch (...) { return def; }
        }

        std::vector<std::string> opt_string_array(
            const json& j, const std::string& key)
        {
            std::vector<std::string> result;
            if (!j.contains(key) || !j[key].is_array()) return result;
            for (const auto& v : j[key])
                if (v.is_string()) result.push_back(v.get<std::string>());
            return result;
        }
    }

    // =========================================================================
    // load
    // =========================================================================

    CardinalConfig ConfigLoader::load(const std::string& path) {
        LOG_INFO("Loading config from: " + path);

        if (!std::filesystem::exists(path))
            throw ConfigError("Config file not found: " + path);

        std::ifstream file(path);
        if (!file.is_open())
            throw ConfigError("Failed to open config file: " + path);

        json j;
        try { file >> j; }
        catch (const json::parse_error& e) {
            throw ConfigError("JSON parse error in '" + path + "': " + e.what());
        }

        CardinalConfig config;
        try {
            config.backend          = parse_backend(j.at("backend"));
            config.inference        = parse_inference(j.at("inference"));
            config.feeling_schema   = parse_feeling_schema(j.at("feeling_schema"));
            config.memory           = parse_memory(j.at("memory"));
            config.verifier         = parse_verifier(j.at("verifier"));
            config.feedback         = parse_feedback(j.at("feedback"));
            config.retriever        = parse_retriever(j.at("retriever"));
            config.api              = parse_api(j.at("api"));
            config.tools            = parse_tools(j.at("tools"));
            config.vision           = parse_vision(j.value("vision", json::object()));
            config.agent            = parse_agent(j.value("agent", json::object()));
            config.explainability   = parse_explainability(
                                          j.value("explainability", json::object()));
            config.self_improvement = parse_self_improvement(
                                          j.value("self_improvement", json::object()));
            config.scheduler        = parse_scheduler(j);       // v1.5.0
            config.computer_use     = parse_computer_use(j);    // v1.5.0
            config.benchmark        = parse_benchmark(j.at("benchmark"));
            config.logging          = parse_logging(j.at("logging"));
        }
        catch (const ConfigError&) { throw; }
        catch (const json::exception& e) {
            throw ConfigError("Missing required section: " + std::string(e.what()));
        }

        validate(config);
        LOG_INFO("Config loaded successfully");
        return config;
    }

    CardinalConfig ConfigLoader::reload(const std::string& path) {
        LOG_INFO("Reloading config from: " + path);
        return load(path);
    }

    // =========================================================================
    // validate
    // =========================================================================

    void ConfigLoader::validate(const CardinalConfig& config) {
        // Backend
        const std::vector<std::string> valid_types = { "llama_cpp", "tensorrt" };
        bool valid = false;
        for (const auto& t : valid_types)
            if (config.backend.type == t) { valid = true; break; }
        if (!valid)
            throw ConfigError("backend.type must be llama_cpp or tensorrt");

        if (config.backend.type == "llama_cpp") {
            const auto& lc = config.backend.llama_cpp;
            if (lc.model_path.empty())
                throw ConfigError("backend.llama_cpp.model_path cannot be empty");
            if (!std::filesystem::exists(lc.model_path))
                throw ConfigError("backend.llama_cpp.model_path not found: " +
                                  lc.model_path);
            if (lc.context_length < 512)
                throw ConfigError("backend.llama_cpp.context_length must be >= 512");
            if (lc.gpu_layers < 0)
                throw ConfigError("backend.llama_cpp.gpu_layers must be >= 0");
            if (lc.threads < 1)
                throw ConfigError("backend.llama_cpp.threads must be >= 1");
        }

        if (config.backend.type == "tensorrt") {
            const auto& tc = config.backend.tensorrt;
            if (tc.engine_path.empty())
                throw ConfigError("backend.tensorrt.engine_path cannot be empty");
            if (!std::filesystem::exists(tc.engine_path))
                throw ConfigError("backend.tensorrt.engine_path not found: " +
                                  tc.engine_path);
            if (tc.tokenizer_path.empty())
                throw ConfigError("backend.tensorrt.tokenizer_path cannot be empty");
            if (!std::filesystem::exists(tc.tokenizer_path))
                throw ConfigError("backend.tensorrt.tokenizer_path not found: " +
                                  tc.tokenizer_path);
            if (tc.kv_cache_fraction <= 0.0f || tc.kv_cache_fraction > 1.0f)
                throw ConfigError("backend.tensorrt.kv_cache_fraction must be in (0,1]");
        }

        // Inference
        if (config.inference.temperature < 0.0f || config.inference.temperature > 2.0f)
            throw ConfigError("inference.temperature must be 0.0-2.0");
        if (config.inference.top_p <= 0.0f || config.inference.top_p > 1.0f)
            throw ConfigError("inference.top_p must be 0.0-1.0");
        if (config.inference.max_tokens_feeling < 32)
            throw ConfigError("inference.max_tokens_feeling must be >= 32");
        if (config.inference.max_tokens_response < 64)
            throw ConfigError("inference.max_tokens_response must be >= 64");

        // Grammar path: only required for llama_cpp
        if (config.backend.type == "llama_cpp" &&
            !std::filesystem::exists(config.feeling_schema.grammar_path))
            throw ConfigError("feeling_schema.grammar_path not found: " +
                              config.feeling_schema.grammar_path);

        // Agent
        if (config.agent.max_iterations < 1)
            throw ConfigError("agent.max_iterations must be >= 1");
        if (config.agent.max_iterations_hard_cap < config.agent.max_iterations)
            throw ConfigError("agent.max_iterations_hard_cap must be >= max_iterations");
        if (config.agent.working_memory_size < 1)
            throw ConfigError("agent.working_memory_size must be >= 1");
        if (config.agent.self_correction_max_attempts < 0)
            throw ConfigError("agent.self_correction_max_attempts must be >= 0");

        // Run python sandbox mode
        const auto& rp = config.tools.run_python;
        if (rp.enabled) {
            if (rp.sandbox_mode != "subprocess" && rp.sandbox_mode != "docker")
                throw ConfigError(
                    "tools.run_python.sandbox_mode must be 'subprocess' or 'docker'");
            if (rp.sandbox_mode == "docker" && rp.docker_image.empty())
                throw ConfigError(
                    "tools.run_python.docker_image cannot be empty when "
                    "sandbox_mode is 'docker'");
        }

        // API
        if (config.api.port < 1 || config.api.port > 65535)
            throw ConfigError("api.port must be 1-65535");
        if (config.api.auth_enabled && config.api.api_key.empty())
            throw ConfigError("api.api_key cannot be empty when auth_enabled=true");

        // Scheduler (v1.5.0) — soft validation only, no hard errors for optional fields
        if (config.scheduler.enabled) {
            if (config.scheduler.check_interval_seconds < 1)
                throw ConfigError("scheduler.check_interval_seconds must be >= 1");
            if (config.scheduler.max_concurrent_tasks < 1)
                throw ConfigError("scheduler.max_concurrent_tasks must be >= 1");
        }

        // Computer use (v1.5.0) — no hard errors, all optional
        if (config.computer_use.enabled &&
            config.computer_use.safety.confirmation_timeout_seconds < 1)
            throw ConfigError(
                "computer_use.safety.confirmation_timeout_seconds must be >= 1");

        LOG_DEBUG("Config validation passed");
    }

    // =========================================================================
    // Backend parsers
    // =========================================================================

    BackendConfig ConfigLoader::parse_backend(const auto& j) {
        BackendConfig c;
        c.type = require<std::string>(j, "backend", "type");
        if (j.contains("llama_cpp") && j["llama_cpp"].is_object())
            c.llama_cpp = parse_backend_llama_cpp(j.at("llama_cpp"));
        else if (c.type == "llama_cpp")
            throw ConfigError("backend.llama_cpp block missing");
        if (j.contains("tensorrt") && j["tensorrt"].is_object())
            c.tensorrt = parse_backend_tensorrt(j.at("tensorrt"));
        else if (c.type == "tensorrt")
            throw ConfigError("backend.tensorrt block missing");
        return c;
    }

    BackendLlamaCppConfig ConfigLoader::parse_backend_llama_cpp(const auto& j) {
        BackendLlamaCppConfig c;
        c.model_path     = require<std::string>(j, "backend.llama_cpp", "model_path");
        c.chat_template  = require<std::string>(j, "backend.llama_cpp", "chat_template");
        c.context_length = require<int>(j, "backend.llama_cpp", "context_length");
        c.gpu_layers     = require<int>(j, "backend.llama_cpp", "gpu_layers");
        c.threads        = require<int>(j, "backend.llama_cpp", "threads");
        return c;
    }

    BackendTensorRTConfig ConfigLoader::parse_backend_tensorrt(const auto& j) {
        BackendTensorRTConfig c;
        c.engine_path       = require<std::string>(j, "backend.tensorrt", "engine_path");
        c.tokenizer_path    = require<std::string>(j, "backend.tensorrt", "tokenizer_path");
        c.chat_template     = require<std::string>(j, "backend.tensorrt", "chat_template");
        c.max_batch_size    = require<int>(j, "backend.tensorrt", "max_batch_size");
        c.max_seq_len       = require<int>(j, "backend.tensorrt", "max_seq_len");
        c.kv_cache_fraction = require<float>(j, "backend.tensorrt", "kv_cache_fraction");
        c.use_int8          = opt<bool>(j, "use_int8", false);
        return c;
    }

    // =========================================================================
    // parse_tools
    // =========================================================================

    ToolsConfig ConfigLoader::parse_tools(const auto& j) {
        ToolsConfig c;
        c.home_access = opt<bool>(j, "home_access", false);

        if (j.contains("web_search") && j["web_search"].is_object()) {
            const auto& ws = j["web_search"];
            c.web_search.enabled               = opt<bool>(ws, "enabled", true);
            c.web_search.confirmation_required = opt<bool>(ws, "confirmation_required", false);
            c.web_search.max_results           = opt<int>(ws, "max_results", 5);
            c.web_search.timeout_seconds       = opt<int>(ws, "timeout_seconds", 10);
        }

        if (j.contains("web_fetch") && j["web_fetch"].is_object()) {
            const auto& wf = j["web_fetch"];
            c.web_fetch.enabled               = opt<bool>(wf, "enabled", true);
            c.web_fetch.confirmation_required = opt<bool>(wf, "confirmation_required", false);
            c.web_fetch.timeout_seconds       = opt<int>(wf, "timeout_seconds", 15);
            c.web_fetch.max_content_kb        = opt<int>(wf, "max_content_kb", 512);
            c.web_fetch.allowed_domains       = opt_string_array(wf, "allowed_domains");
            c.web_fetch.blocked_domains       = opt_string_array(wf, "blocked_domains");
        }

        if (j.contains("calculator") && j["calculator"].is_object()) {
            const auto& calc = j["calculator"];
            c.calculator.enabled               = opt<bool>(calc, "enabled", true);
            c.calculator.confirmation_required = opt<bool>(calc, "confirmation_required", false);
        }

        if (j.contains("run_python") && j["run_python"].is_object()) {
            const auto& rp = j["run_python"];
            c.run_python.enabled               = opt<bool>(rp, "enabled", true);
            c.run_python.confirmation_required = opt<bool>(rp, "confirmation_required", true);
            c.run_python.sandbox_mode          = opt<std::string>(rp, "sandbox_mode", "subprocess");
            c.run_python.docker_image          = opt<std::string>(rp, "docker_image", "python:3.12-slim");
            c.run_python.timeout_seconds       = opt<int>(rp, "timeout_seconds", 30);
            c.run_python.memory_limit_mb       = opt<int>(rp, "memory_limit_mb", 256);
            c.run_python.network_enabled       = opt<bool>(rp, "network_enabled", false);
        }

        if (j.contains("file_read") && j["file_read"].is_object()) {
            const auto& fr = j["file_read"];
            c.file_read.enabled               = opt<bool>(fr, "enabled", true);
            c.file_read.confirmation_required = opt<bool>(fr, "confirmation_required", false);
            c.file_read.allowed_paths         = opt_string_array(fr, "allowed_paths");
        }

        if (j.contains("file_write") && j["file_write"].is_object()) {
            const auto& fw = j["file_write"];
            c.file_write.enabled               = opt<bool>(fw, "enabled", true);
            c.file_write.confirmation_required = opt<bool>(fw, "confirmation_required", true);
            c.file_write.allowed_paths         = opt_string_array(fw, "allowed_paths");
        }

        if (j.contains("knowledge_graph_query") &&
            j["knowledge_graph_query"].is_object()) {
            const auto& kg = j["knowledge_graph_query"];
            c.knowledge_graph.enabled               = opt<bool>(kg, "enabled", true);
            c.knowledge_graph.confirmation_required = opt<bool>(kg, "confirmation_required", false);
        }

        if (j.contains("episodic_search") && j["episodic_search"].is_object()) {
            const auto& es = j["episodic_search"];
            c.episodic_search.enabled               = opt<bool>(es, "enabled", true);
            c.episodic_search.confirmation_required = opt<bool>(es, "confirmation_required", false);
            c.episodic_search.max_results           = opt<int>(es, "max_results", 5);
        }

        return c;
    }

    // =========================================================================
    // parse_vision (v1.3.0)
    // =========================================================================

    VisionConfig ConfigLoader::parse_vision(const auto& j) {
        VisionConfig c;
        c.model_path               = opt<std::string>(j, "model_path", "");
        c.mmproj_path              = opt<std::string>(j, "mmproj_path", "");
        c.gpu_layers               = opt<int>(j, "gpu_layers", 0);
        c.threads                  = opt<int>(j, "threads", 4);
        c.max_tokens               = opt<int>(j, "max_tokens", 512);
        c.cache_path               = opt<std::string>(j, "cache_path", "data/vision_cache");
        c.cache_ttl_hours          = opt<int>(j, "cache_ttl_hours", 24);
        c.download_timeout_seconds = opt<int>(j, "download_timeout_seconds", 30);
        c.confirmation_required    = opt<bool>(j, "confirmation_required", false);
        c.allowed_paths            = opt_string_array(j, "allowed_paths");
        return c;
    }

    // =========================================================================
    // parse_agent
    // =========================================================================

    AgentConfig ConfigLoader::parse_agent(const auto& j) {
        AgentConfig c;
        c.enabled                      = opt<bool>(j, "enabled", true);
        c.max_iterations               = opt<int>(j, "max_iterations", 10);
        c.max_iterations_hard_cap      = opt<int>(j, "max_iterations_hard_cap", 50);
        c.working_memory_path          = opt<std::string>(j, "working_memory_path",
                                             "data/memory/agent_working_memory");
        c.working_memory_size          = opt<int>(j, "working_memory_size", 50);
        c.self_correction_enabled      = opt<bool>(j, "self_correction_enabled", true);
        c.self_correction_max_attempts = opt<int>(j, "self_correction_max_attempts", 3);
        c.plan_before_execute          = opt<bool>(j, "plan_before_execute", true);
        c.summarize_on_cap             = opt<bool>(j, "summarize_on_cap", true);
        return c;
    }

    // =========================================================================
    // parse_explainability
    // =========================================================================

    ExplainabilityConfig ConfigLoader::parse_explainability(const auto& j) {
        ExplainabilityConfig c;
        c.enabled                  = opt<bool>(j, "enabled", true);
        c.audit_log_path           = opt<std::string>(j, "audit_log_path",
                                         "data/explainability/audit.db");
        c.signing_enabled          = opt<bool>(j, "signing_enabled", true);
        c.private_key_path         = opt<std::string>(j, "private_key_path",
                                         "data/explainability/cardinal_private.pem");
        c.public_key_path          = opt<std::string>(j, "public_key_path",
                                         "data/explainability/cardinal_public.pem");
        c.auto_generate_keys       = opt<bool>(j, "auto_generate_keys", true);
        c.export_path              = opt<std::string>(j, "export_path",
                                         "data/explainability/exports");
        c.attach_trace_to_response = opt<bool>(j, "attach_trace_to_response", true);
        return c;
    }

    // =========================================================================
    // parse_self_improvement (v1.4.0)
    // =========================================================================

    SelfImprovementConfig ConfigLoader::parse_self_improvement(const auto& j) {
        SelfImprovementConfig c;
        c.enabled = opt<bool>(j, "enabled", true);

        if (j.contains("self_model") && j["self_model"].is_object()) {
            const auto& sm = j["self_model"];
            c.self_model.enabled            = opt<bool>(sm, "enabled", true);
            c.self_model.db_path            = opt<std::string>(sm, "db_path",
                                                 "data/self_model/self_model.db");
            c.self_model.inject_into_prompt = opt<bool>(sm, "inject_into_prompt", true);
            c.self_model.prompt_max_chars   = opt<int>(sm, "prompt_max_chars", 500);
            c.self_model.history_window     = opt<int>(sm, "history_window", 100);
        }

        if (j.contains("meta_cognition") && j["meta_cognition"].is_object()) {
            const auto& mc = j["meta_cognition"];
            c.meta_cognition.enabled                           =
                opt<bool>(mc, "enabled", true);
            c.meta_cognition.trigger_every_n_inferences        =
                opt<int>(mc, "trigger_every_n_inferences", 20);
            c.meta_cognition.trigger_on_contradiction_rate_pct =
                opt<float>(mc, "trigger_on_contradiction_rate_pct", 30.0f);
            c.meta_cognition.on_demand_via_api                 =
                opt<bool>(mc, "on_demand_via_api", true);
            c.meta_cognition.min_failures_to_reflect           =
                opt<int>(mc, "min_failures_to_reflect", 5);
            c.meta_cognition.max_corrective_rules_per_session  =
                opt<int>(mc, "max_corrective_rules_per_session", 10);
            c.meta_cognition.corrective_rule_confidence        =
                opt<float>(mc, "corrective_rule_confidence", 0.6f);
        }

        if (j.contains("training") && j["training"].is_object()) {
            const auto& tr = j["training"];
            c.training.enabled                           = opt<bool>(tr, "enabled", true);
            c.training.lora_rank                         = opt<int>(tr, "lora_rank", 8);
            c.training.lora_alpha                        = opt<int>(tr, "lora_alpha", 16);
            c.training.learning_rate                     = opt<float>(tr, "learning_rate", 0.0001f);
            c.training.epochs                            = opt<int>(tr, "epochs", 3);
            c.training.batch_size                        = opt<int>(tr, "batch_size", 4);
            c.training.min_episodes_for_training         = opt<int>(tr, "min_episodes_for_training", 50);
            c.training.min_quality_confidence            = opt<float>(tr, "min_quality_confidence", 0.75f);
            c.training.max_examples                      = opt<int>(tr, "max_examples", 0);
            c.training.trigger_every_n_episodes          = opt<int>(tr, "trigger_every_n_episodes", 100);
            c.training.trigger_every_n_hours             = opt<int>(tr, "trigger_every_n_hours", 24);
            c.training.trigger_on_domain_confidence_below =
                opt<float>(tr, "trigger_on_domain_confidence_below", 0.5f);
            c.training.adapter_load_policy               = opt<std::string>(tr, "adapter_load_policy",
                                                               "session_boundary");
            c.training.eval_improvement_threshold_pct    = opt<float>(tr, "eval_improvement_threshold_pct", 5.0f);
            c.training.eval_holdout_episodes             = opt<int>(tr, "eval_holdout_episodes", 20);
            c.training.adapter_output_dir                = opt<std::string>(tr, "adapter_output_dir",
                                                               "data/training/adapters");
            c.training.dataset_output_dir                = opt<std::string>(tr, "dataset_output_dir",
                                                               "data/training/datasets");
            c.training.export_script_dir                 = opt<std::string>(tr, "export_script_dir",
                                                               "data/training/scripts");
            c.training.hf_model_path                     = opt<std::string>(tr, "hf_model_path",
                                                               "models/qwen3.5-4b-hf");
            c.training.python_venv                       = opt<std::string>(tr, "python_venv",
                                                               "~/cardinal/cardinal-train-venv");
            c.training.convert_lora_script               = opt<std::string>(tr, "convert_lora_script",
                                                               "vendor/llama.cpp/convert_lora_to_gguf.py");
            c.training.llama_finetune_binary             = opt<std::string>(tr, "llama_finetune_binary",
                                                               "vendor/llama.cpp/build/bin/llama-finetune");
        }

        return c;
    }

    // =========================================================================
    // parse_scheduler (v1.5.0)
    // =========================================================================

    SchedulerConfig ConfigLoader::parse_scheduler(const auto& j) {
        SchedulerConfig c;
        if (!j.contains("scheduler")) return c;
        const auto& s = j["scheduler"];
        c.enabled                   = opt<bool>(s, "enabled", false);
        c.db_path                   = opt<std::string>(s, "db_path",
                                          "data/scheduler/scheduler.db");
        c.check_interval_seconds    = opt<int>(s, "check_interval_seconds", 30);
        c.max_concurrent_tasks      = opt<int>(s, "max_concurrent_tasks", 1);
        c.idle_threshold_minutes    = opt<int>(s, "idle_threshold_minutes", 5);
        c.task_session_prefix       = opt<std::string>(s, "task_session_prefix", "scheduler_");
        c.run_history_max_entries   = opt<int>(s, "run_history_max_entries", 1000);
        c.max_task_duration_seconds = opt<int>(s, "max_task_duration_seconds", 300);
        return c;
    }

    // =========================================================================
    // parse_computer_use (v1.5.0)
    // =========================================================================

    ComputerUseConfig ConfigLoader::parse_computer_use(const auto& j) {
        ComputerUseConfig c;
        if (!j.contains("computer_use")) return c;
        const auto& cu = j["computer_use"];
        c.enabled = opt<bool>(cu, "enabled", false);

        if (cu.contains("safety") && cu["safety"].is_object()) {
            const auto& sf = cu["safety"];
            c.safety.whitelist_enabled            = opt<bool>(sf, "whitelist_enabled", true);
            c.safety.confirmation_required        = opt<bool>(sf, "confirmation_required", true);
            c.safety.confirmation_timeout_seconds = opt<int>(sf, "confirmation_timeout_seconds", 30);
            c.safety.watch_mode                   = opt<bool>(sf, "watch_mode", true);
            c.safety.full_autonomy                = opt<bool>(sf, "full_autonomy", false);
            c.safety.allow_file_write             = opt<bool>(sf, "allow_file_write", false);
            c.safety.allowed_apps     = opt_string_array(sf, "allowed_apps");
            c.safety.allowed_domains  = opt_string_array(sf, "allowed_domains");
            c.safety.allowed_paths    = opt_string_array(sf, "allowed_paths");
            if (sf.contains("blocked_commands") && sf["blocked_commands"].is_array()) {
                c.safety.blocked_commands.clear();
                c.safety.blocked_commands = opt_string_array(sf, "blocked_commands");
            }
        }

        if (cu.contains("screen") && cu["screen"].is_object()) {
            const auto& sc = cu["screen"];
            c.screen.screenshot_tool        = opt<std::string>(sc, "screenshot_tool", "auto");
            c.screen.vision_analysis        = opt<bool>(sc, "vision_analysis", true);
            c.screen.watch_interval_seconds = opt<int>(sc, "watch_interval_seconds", 2);
        }

        if (cu.contains("browser") && cu["browser"].is_object()) {
            const auto& br = cu["browser"];
            c.browser.executable            = opt<std::string>(br, "executable", "google-chrome");
            c.browser.venv_path             = opt<std::string>(br, "venv_path",
                                                 "~/cardinal/cardinal-browser-venv");
            c.browser.playwright_timeout_ms = opt<int>(br, "playwright_timeout_ms", 10000);
            c.browser.headless              = opt<bool>(br, "headless", false);
            c.browser.user_data_dir         = opt<std::string>(br, "user_data_dir",
                                                 "data/browser_profile");
        }

        if (cu.contains("shell") && cu["shell"].is_object()) {
            const auto& sh = cu["shell"];
            c.shell.enabled           = opt<bool>(sh, "enabled", true);
            c.shell.shell             = opt<std::string>(sh, "shell", "/bin/bash");
            c.shell.timeout_seconds   = opt<int>(sh, "timeout_seconds", 30);
            c.shell.working_directory = opt<std::string>(sh, "working_directory", "~");
        }

        if (cu.contains("email") && cu["email"].is_object()) {
            const auto& em = cu["email"];
            c.email.enabled                = opt<bool>(em, "enabled", false);
            c.email.mode                   = opt<std::string>(em, "mode", "imap_smtp");
            c.email.imap_host              = opt<std::string>(em, "imap_host", "");
            c.email.imap_port              = opt<int>(em, "imap_port", 993);
            c.email.smtp_host              = opt<std::string>(em, "smtp_host", "");
            c.email.smtp_port              = opt<int>(em, "smtp_port", 587);
            c.email.address                = opt<std::string>(em, "address", "");
            c.email.gmail_api_enabled      = opt<bool>(em, "gmail_api_enabled", false);
            c.email.gmail_credentials_path = opt<std::string>(em, "gmail_credentials_path",
                                                 "data/gmail_credentials.json");
        }

        if (cu.contains("atspi") && cu["atspi"].is_object()) {
            const auto& at = cu["atspi"];
            c.atspi.enabled            = opt<bool>(at, "enabled", true);
            c.atspi.fallback_to_vision = opt<bool>(at, "fallback_to_vision", true);
        }

        return c;
    }

    // =========================================================================
    // Unchanged parsers from v1.1.0
    // =========================================================================

    InferenceConfig ConfigLoader::parse_inference(const auto& j) {
        InferenceConfig c;
        c.temperature         = require<float>(j, "inference", "temperature");
        c.top_p               = require<float>(j, "inference", "top_p");
        c.max_tokens_feeling  = require<int>(j, "inference", "max_tokens_feeling");
        c.max_tokens_response = require<int>(j, "inference", "max_tokens_response");
        return c;
    }

    FeelingSchemaConfig ConfigLoader::parse_feeling_schema(const auto& j) {
        FeelingSchemaConfig c;
        c.type         = require<std::string>(j, "feeling_schema", "type");
        c.grammar_path = require<std::string>(j, "feeling_schema", "grammar_path");
        c.max_tokens   = require<int>(j, "feeling_schema", "max_tokens");
        if (!j.contains("fields") || !j["fields"].is_object())
            throw ConfigError("feeling_schema.fields must be a JSON object");
        for (const auto& [key, val] : j["fields"].items()) {
            FeelingSchemaField field;
            field.name             = key;
            field.type_description = val.template get<std::string>();
            c.fields.push_back(field);
        }
        return c;
    }

    MemoryConfig ConfigLoader::parse_memory(const auto& j) {
        MemoryConfig c;
        c.rule_store_path      = require<std::string>(j, "memory", "rule_store_path");
        c.knowledge_graph_path = require<std::string>(j, "memory", "knowledge_graph_path");
        c.episodic_log_path    = require<std::string>(j, "memory", "episodic_log_path");
        c.max_rules            = require<int>(j, "memory", "max_rules");
        return c;
    }

    VerifierConfig ConfigLoader::parse_verifier(const auto& j) {
        VerifierConfig c;
        c.mode                    = require<std::string>(j, "verifier", "mode");
        c.neural_model_path       = opt<std::string>(j, "neural_model_path", "");
        c.neural_gpu_layers       = opt<int>(j, "neural_gpu_layers", 0);
        c.neural_max_tokens       = opt<int>(j, "neural_max_tokens", 512);
        c.contradiction_threshold = require<float>(j, "verifier", "contradiction_threshold");
        c.rule_confidence_decay   = require<float>(j, "verifier", "rule_confidence_decay");
        c.min_rule_confidence     = require<float>(j, "verifier", "min_rule_confidence");
        return c;
    }

    FeedbackConfig ConfigLoader::parse_feedback(const auto& j) {
        FeedbackConfig c;
        c.max_retries           = require<int>(j, "feedback", "max_retries");
        c.retry_delay_ms        = require<int>(j, "feedback", "retry_delay_ms");
        c.rule_injection_format = require<std::string>(j, "feedback", "rule_injection_format");
        return c;
    }

    RetrieverConfig ConfigLoader::parse_retriever(const auto& j) {
        RetrieverConfig c;
        c.mode                           = require<std::string>(j, "retriever", "mode");
        c.keyword_weight                 = require<float>(j, "retriever", "keyword_weight");
        c.semantic_weight                = require<float>(j, "retriever", "semantic_weight");
        c.max_results                    = require<int>(j, "retriever", "max_results");
        c.min_score                      = require<float>(j, "retriever", "min_score");
        c.cache_rebuild_strategy         = require<std::string>(j, "retriever", "cache_rebuild_strategy");
        c.cache_rebuild_threshold        = require<int>(j, "retriever", "cache_rebuild_threshold");
        c.cache_rebuild_interval_seconds = require<int>(j, "retriever", "cache_rebuild_interval_seconds");
        return c;
    }

    ApiConfig ConfigLoader::parse_api(const auto& j) {
        ApiConfig c;
        c.http_enabled            = require<bool>(j, "api", "http_enabled");
        c.host                    = require<std::string>(j, "api", "host");
        c.port                    = require<int>(j, "api", "port");
        c.auth_enabled            = require<bool>(j, "api", "auth_enabled");
        c.api_key                 = opt<std::string>(j, "api_key", "");
        c.stream_enabled          = require<bool>(j, "api", "stream_enabled");
        c.max_request_size_kb     = require<int>(j, "api", "max_request_size_kb");
        c.request_timeout_seconds = require<int>(j, "api", "request_timeout_seconds");
        return c;
    }

    BenchmarkConfig ConfigLoader::parse_benchmark(const auto& j) {
        BenchmarkConfig c;
        c.dataset                = require<std::string>(j, "benchmark", "dataset");
        c.eval_frequency_seconds = require<int>(j, "benchmark", "eval_frequency_seconds");
        if (!j.contains("metrics") || !j["metrics"].is_array())
            throw ConfigError("benchmark.metrics must be a JSON array");
        for (const auto& m : j["metrics"])
            c.metrics.push_back(m.template get<std::string>());
        return c;
    }

    LoggingConfig ConfigLoader::parse_logging(const auto& j) {
        LoggingConfig c;
        c.level = require<std::string>(j, "logging", "level");
        c.path  = require<std::string>(j, "logging", "path");
        return c;
    }

    // =========================================================================
    // to_json_string
    // =========================================================================

    std::string ConfigLoader::to_json_string(const CardinalConfig& config) {
        json j;

        // Backend
        j["backend"]["type"] = config.backend.type;
        j["backend"]["llama_cpp"]["model_path"]       = config.backend.llama_cpp.model_path;
        j["backend"]["llama_cpp"]["chat_template"]    = config.backend.llama_cpp.chat_template;
        j["backend"]["llama_cpp"]["context_length"]   = config.backend.llama_cpp.context_length;
        j["backend"]["llama_cpp"]["gpu_layers"]       = config.backend.llama_cpp.gpu_layers;
        j["backend"]["llama_cpp"]["threads"]          = config.backend.llama_cpp.threads;
        j["backend"]["tensorrt"]["engine_path"]       = config.backend.tensorrt.engine_path;
        j["backend"]["tensorrt"]["tokenizer_path"]    = config.backend.tensorrt.tokenizer_path;
        j["backend"]["tensorrt"]["chat_template"]     = config.backend.tensorrt.chat_template;
        j["backend"]["tensorrt"]["max_batch_size"]    = config.backend.tensorrt.max_batch_size;
        j["backend"]["tensorrt"]["max_seq_len"]       = config.backend.tensorrt.max_seq_len;
        j["backend"]["tensorrt"]["kv_cache_fraction"] = config.backend.tensorrt.kv_cache_fraction;
        j["backend"]["tensorrt"]["use_int8"]          = config.backend.tensorrt.use_int8;

        // Inference
        j["inference"]["temperature"]         = config.inference.temperature;
        j["inference"]["top_p"]               = config.inference.top_p;
        j["inference"]["max_tokens_feeling"]  = config.inference.max_tokens_feeling;
        j["inference"]["max_tokens_response"] = config.inference.max_tokens_response;

        // Tools
        const auto& t = config.tools;
        j["tools"]["web_search"]["enabled"]               = t.web_search.enabled;
        j["tools"]["web_search"]["confirmation_required"] = t.web_search.confirmation_required;
        j["tools"]["web_search"]["max_results"]           = t.web_search.max_results;
        j["tools"]["web_search"]["timeout_seconds"]       = t.web_search.timeout_seconds;
        j["tools"]["web_fetch"]["enabled"]                = t.web_fetch.enabled;
        j["tools"]["web_fetch"]["confirmation_required"]  = t.web_fetch.confirmation_required;
        j["tools"]["web_fetch"]["timeout_seconds"]        = t.web_fetch.timeout_seconds;
        j["tools"]["web_fetch"]["max_content_kb"]         = t.web_fetch.max_content_kb;
        j["tools"]["calculator"]["enabled"]               = t.calculator.enabled;
        j["tools"]["calculator"]["confirmation_required"] = t.calculator.confirmation_required;
        j["tools"]["run_python"]["enabled"]               = t.run_python.enabled;
        j["tools"]["run_python"]["confirmation_required"] = t.run_python.confirmation_required;
        j["tools"]["run_python"]["sandbox_mode"]          = t.run_python.sandbox_mode;
        j["tools"]["run_python"]["docker_image"]          = t.run_python.docker_image;
        j["tools"]["run_python"]["timeout_seconds"]       = t.run_python.timeout_seconds;
        j["tools"]["run_python"]["memory_limit_mb"]       = t.run_python.memory_limit_mb;
        j["tools"]["run_python"]["network_enabled"]       = t.run_python.network_enabled;
        j["tools"]["file_read"]["enabled"]                = t.file_read.enabled;
        j["tools"]["file_read"]["confirmation_required"]  = t.file_read.confirmation_required;
        j["tools"]["file_read"]["allowed_paths"]          = t.file_read.allowed_paths;
        j["tools"]["file_write"]["enabled"]               = t.file_write.enabled;
        j["tools"]["file_write"]["confirmation_required"] = t.file_write.confirmation_required;
        j["tools"]["file_write"]["allowed_paths"]         = t.file_write.allowed_paths;
        j["tools"]["knowledge_graph_query"]["enabled"]               = t.knowledge_graph.enabled;
        j["tools"]["knowledge_graph_query"]["confirmation_required"] = t.knowledge_graph.confirmation_required;
        j["tools"]["episodic_search"]["enabled"]               = t.episodic_search.enabled;
        j["tools"]["episodic_search"]["confirmation_required"] = t.episodic_search.confirmation_required;
        j["tools"]["episodic_search"]["max_results"]           = t.episodic_search.max_results;

        // Agent
        const auto& a = config.agent;
        j["agent"]["enabled"]                      = a.enabled;
        j["agent"]["max_iterations"]               = a.max_iterations;
        j["agent"]["max_iterations_hard_cap"]      = a.max_iterations_hard_cap;
        j["agent"]["working_memory_path"]          = a.working_memory_path;
        j["agent"]["working_memory_size"]          = a.working_memory_size;
        j["agent"]["self_correction_enabled"]      = a.self_correction_enabled;
        j["agent"]["self_correction_max_attempts"] = a.self_correction_max_attempts;
        j["agent"]["plan_before_execute"]          = a.plan_before_execute;
        j["agent"]["summarize_on_cap"]             = a.summarize_on_cap;

        // Explainability
        const auto& e = config.explainability;
        j["explainability"]["enabled"]                  = e.enabled;
        j["explainability"]["audit_log_path"]           = e.audit_log_path;
        j["explainability"]["signing_enabled"]          = e.signing_enabled;
        j["explainability"]["private_key_path"]         = e.private_key_path;
        j["explainability"]["public_key_path"]          = e.public_key_path;
        j["explainability"]["auto_generate_keys"]       = e.auto_generate_keys;
        j["explainability"]["export_path"]              = e.export_path;
        j["explainability"]["attach_trace_to_response"] = e.attach_trace_to_response;

        // Vision
        const auto& v = config.vision;
        j["vision"]["model_path"]               = v.model_path;
        j["vision"]["mmproj_path"]              = v.mmproj_path;
        j["vision"]["gpu_layers"]               = v.gpu_layers;
        j["vision"]["threads"]                  = v.threads;
        j["vision"]["max_tokens"]               = v.max_tokens;
        j["vision"]["cache_path"]               = v.cache_path;
        j["vision"]["cache_ttl_hours"]          = v.cache_ttl_hours;
        j["vision"]["download_timeout_seconds"] = v.download_timeout_seconds;
        j["vision"]["confirmation_required"]    = v.confirmation_required;
        j["vision"]["allowed_paths"]            = v.allowed_paths;

        // Self-improvement (v1.4.0)
        const auto& si = config.self_improvement;
        j["self_improvement"]["enabled"]                                        = si.enabled;
        j["self_improvement"]["self_model"]["enabled"]                          = si.self_model.enabled;
        j["self_improvement"]["self_model"]["db_path"]                          = si.self_model.db_path;
        j["self_improvement"]["self_model"]["inject_into_prompt"]               = si.self_model.inject_into_prompt;
        j["self_improvement"]["self_model"]["prompt_max_chars"]                 = si.self_model.prompt_max_chars;
        j["self_improvement"]["self_model"]["history_window"]                   = si.self_model.history_window;
        j["self_improvement"]["meta_cognition"]["enabled"]                      = si.meta_cognition.enabled;
        j["self_improvement"]["meta_cognition"]["trigger_every_n_inferences"]   = si.meta_cognition.trigger_every_n_inferences;
        j["self_improvement"]["meta_cognition"]["trigger_on_contradiction_rate_pct"] = si.meta_cognition.trigger_on_contradiction_rate_pct;
        j["self_improvement"]["meta_cognition"]["on_demand_via_api"]            = si.meta_cognition.on_demand_via_api;
        j["self_improvement"]["meta_cognition"]["min_failures_to_reflect"]      = si.meta_cognition.min_failures_to_reflect;
        j["self_improvement"]["meta_cognition"]["max_corrective_rules_per_session"] = si.meta_cognition.max_corrective_rules_per_session;
        j["self_improvement"]["meta_cognition"]["corrective_rule_confidence"]   = si.meta_cognition.corrective_rule_confidence;
        j["self_improvement"]["training"]["enabled"]                            = si.training.enabled;
        j["self_improvement"]["training"]["lora_rank"]                          = si.training.lora_rank;
        j["self_improvement"]["training"]["lora_alpha"]                         = si.training.lora_alpha;
        j["self_improvement"]["training"]["learning_rate"]                      = si.training.learning_rate;
        j["self_improvement"]["training"]["epochs"]                             = si.training.epochs;
        j["self_improvement"]["training"]["batch_size"]                         = si.training.batch_size;
        j["self_improvement"]["training"]["min_episodes_for_training"]          = si.training.min_episodes_for_training;
        j["self_improvement"]["training"]["min_quality_confidence"]             = si.training.min_quality_confidence;
        j["self_improvement"]["training"]["adapter_output_dir"]                 = si.training.adapter_output_dir;
        j["self_improvement"]["training"]["dataset_output_dir"]                 = si.training.dataset_output_dir;
        j["self_improvement"]["training"]["hf_model_path"]                      = si.training.hf_model_path;
        j["self_improvement"]["training"]["python_venv"]                        = si.training.python_venv;

        // Scheduler (v1.5.0)
        const auto& sc = config.scheduler;
        j["scheduler"]["enabled"]                   = sc.enabled;
        j["scheduler"]["db_path"]                   = sc.db_path;
        j["scheduler"]["check_interval_seconds"]    = sc.check_interval_seconds;
        j["scheduler"]["max_concurrent_tasks"]      = sc.max_concurrent_tasks;
        j["scheduler"]["idle_threshold_minutes"]    = sc.idle_threshold_minutes;
        j["scheduler"]["task_session_prefix"]       = sc.task_session_prefix;
        j["scheduler"]["run_history_max_entries"]   = sc.run_history_max_entries;
        j["scheduler"]["max_task_duration_seconds"] = sc.max_task_duration_seconds;

        // Computer use (v1.5.0)
        const auto& cu = config.computer_use;
        j["computer_use"]["enabled"]                                       = cu.enabled;
        j["computer_use"]["safety"]["whitelist_enabled"]                   = cu.safety.whitelist_enabled;
        j["computer_use"]["safety"]["allowed_apps"]                        = cu.safety.allowed_apps;
        j["computer_use"]["safety"]["allowed_domains"]                     = cu.safety.allowed_domains;
        j["computer_use"]["safety"]["allowed_paths"]                       = cu.safety.allowed_paths;
        j["computer_use"]["safety"]["blocked_commands"]                    = cu.safety.blocked_commands;
        j["computer_use"]["safety"]["confirmation_required"]               = cu.safety.confirmation_required;
        j["computer_use"]["safety"]["confirmation_timeout_seconds"]        = cu.safety.confirmation_timeout_seconds;
        j["computer_use"]["safety"]["watch_mode"]                          = cu.safety.watch_mode;
        j["computer_use"]["safety"]["full_autonomy"]                       = cu.safety.full_autonomy;
        j["computer_use"]["safety"]["allow_file_write"]                    = cu.safety.allow_file_write;
        j["computer_use"]["screen"]["screenshot_tool"]                     = cu.screen.screenshot_tool;
        j["computer_use"]["screen"]["vision_analysis"]                     = cu.screen.vision_analysis;
        j["computer_use"]["screen"]["watch_interval_seconds"]              = cu.screen.watch_interval_seconds;
        j["computer_use"]["browser"]["executable"]                         = cu.browser.executable;
        j["computer_use"]["browser"]["venv_path"]                          = cu.browser.venv_path;
        j["computer_use"]["browser"]["playwright_timeout_ms"]              = cu.browser.playwright_timeout_ms;
        j["computer_use"]["browser"]["headless"]                           = cu.browser.headless;
        j["computer_use"]["browser"]["user_data_dir"]                      = cu.browser.user_data_dir;
        j["computer_use"]["shell"]["enabled"]                              = cu.shell.enabled;
        j["computer_use"]["shell"]["shell"]                                = cu.shell.shell;
        j["computer_use"]["shell"]["timeout_seconds"]                      = cu.shell.timeout_seconds;
        j["computer_use"]["shell"]["working_directory"]                    = cu.shell.working_directory;
        j["computer_use"]["email"]["enabled"]                              = cu.email.enabled;
        j["computer_use"]["email"]["mode"]                                 = cu.email.mode;
        j["computer_use"]["email"]["imap_host"]                            = cu.email.imap_host;
        j["computer_use"]["email"]["imap_port"]                            = cu.email.imap_port;
        j["computer_use"]["email"]["smtp_host"]                            = cu.email.smtp_host;
        j["computer_use"]["email"]["smtp_port"]                            = cu.email.smtp_port;
        j["computer_use"]["email"]["address"]                              = cu.email.address;
        j["computer_use"]["email"]["gmail_api_enabled"]                    = cu.email.gmail_api_enabled;
        j["computer_use"]["email"]["gmail_credentials_path"]               = cu.email.gmail_credentials_path;
        j["computer_use"]["atspi"]["enabled"]                              = cu.atspi.enabled;
        j["computer_use"]["atspi"]["fallback_to_vision"]                   = cu.atspi.fallback_to_vision;

        // Logging
        j["logging"]["level"] = config.logging.level;
        j["logging"]["path"]  = config.logging.path;

        return j.dump(2);
    }

} // namespace cardinal
