// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Config Loader Implementation
// File: src/utils/config_loader.cpp
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
            if (!j.contains(key)) {
                throw ConfigError("Missing required field '" + key +
                    "' in section '" + section + "'");
            }
            try {
                return j.at(key).get<T>();
            }
            catch (const json::exception& e) {
                throw ConfigError("Invalid type for field '" + key +
                    "' in section '" + section + "': " + e.what());
            }
        }

        template<typename T>
        T optional_field(const json& j, const std::string& key,
            T default_value) {
            if (!j.contains(key)) return default_value;
            try {
                return j.at(key).get<T>();
            }
            catch (...) {
                return default_value;
            }
        }

    } // anonymous namespace

    // =========================================================================
    // load
    // =========================================================================

    CardinalConfig ConfigLoader::load(const std::string& path) {
        LOG_INFO("Loading config from: " + path);

        if (!std::filesystem::exists(path)) {
            throw ConfigError("Config file not found: " + path);
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            throw ConfigError("Failed to open config file: " + path);
        }

        json j;
        try {
            file >> j;
        }
        catch (const json::parse_error& e) {
            throw ConfigError("JSON parse error in '" + path +
                "': " + e.what());
        }

        CardinalConfig config;

        try {
            config.model = parse_model(j.at("model"));
            config.inference = parse_inference(j.at("inference"));
            config.feeling_schema = parse_feeling_schema(j.at("feeling_schema"));
            config.memory = parse_memory(j.at("memory"));
            config.verifier = parse_verifier(j.at("verifier"));
            config.feedback = parse_feedback(j.at("feedback"));
            config.retriever = parse_retriever(j.at("retriever"));
            config.api = parse_api(j.at("api"));
            config.tools = parse_tools(j.at("tools"));
            config.benchmark = parse_benchmark(j.at("benchmark"));
            config.logging = parse_logging(j.at("logging"));
        }
        catch (const ConfigError&) {
            throw;
        }
        catch (const json::exception& e) {
            throw ConfigError("Missing required section in config: " +
                std::string(e.what()));
        }

        validate(config);

        LOG_INFO("Config loaded successfully");
        return config;
    }

    // =========================================================================
    // reload
    // =========================================================================

    CardinalConfig ConfigLoader::reload(const std::string& path) {
        LOG_INFO("Reloading config from: " + path);
        return load(path);
    }

    // =========================================================================
    // validate
    // =========================================================================

    void ConfigLoader::validate(const CardinalConfig& config) {
        // Model
        if (config.model.path.empty())
            throw ConfigError("model.path cannot be empty");
        if (!std::filesystem::exists(config.model.path))
            throw ConfigError("model.path does not exist: " + config.model.path);
        if (config.model.context_length < 512)
            throw ConfigError("model.context_length must be >= 512");
        if (config.model.gpu_layers < 0)
            throw ConfigError("model.gpu_layers must be >= 0");
        if (config.model.threads < 1)
            throw ConfigError("model.threads must be >= 1");

        // Inference
        if (config.inference.temperature < 0.0f ||
            config.inference.temperature > 2.0f)
            throw ConfigError(
                "inference.temperature must be between 0.0 and 2.0");
        if (config.inference.top_p <= 0.0f || config.inference.top_p > 1.0f)
            throw ConfigError(
                "inference.top_p must be between 0.0 and 1.0");
        if (config.inference.max_tokens_feeling < 32)
            throw ConfigError(
                "inference.max_tokens_feeling must be >= 32");
        if (config.inference.max_tokens_response < 64)
            throw ConfigError(
                "inference.max_tokens_response must be >= 64");

        // Feeling schema
        if (config.feeling_schema.grammar_path.empty())
            throw ConfigError(
                "feeling_schema.grammar_path cannot be empty");
        if (!std::filesystem::exists(config.feeling_schema.grammar_path))
            throw ConfigError(
                "feeling_schema.grammar_path does not exist: " +
                config.feeling_schema.grammar_path);
        if (config.feeling_schema.fields.empty())
            throw ConfigError("feeling_schema.fields cannot be empty");

        // Memory
        if (config.memory.rule_store_path.empty())
            throw ConfigError("memory.rule_store_path cannot be empty");
        if (config.memory.knowledge_graph_path.empty())
            throw ConfigError("memory.knowledge_graph_path cannot be empty");
        if (config.memory.episodic_log_path.empty())
            throw ConfigError("memory.episodic_log_path cannot be empty");
        if (config.memory.max_rules < 1)
            throw ConfigError("memory.max_rules must be >= 1");

        // Verifier
        const std::vector<std::string> valid_verifier_modes =
        { "symbolic", "neural", "hybrid" };
        bool valid_vmode = false;
        for (const auto& m : valid_verifier_modes) {
            if (config.verifier.mode == m) { valid_vmode = true; break; }
        }
        if (!valid_vmode)
            throw ConfigError(
                "verifier.mode must be one of: symbolic, neural, hybrid");
        if ((config.verifier.mode == "neural" ||
            config.verifier.mode == "hybrid") &&
            config.verifier.neural_model_path.empty()) {
            LOG_WARN("verifier.mode is '" + config.verifier.mode +
                "' but neural_model_path is empty -- "
                "neural verifier will be disabled");
        }
        if (config.verifier.contradiction_threshold < 0.0f ||
            config.verifier.contradiction_threshold > 1.0f)
            throw ConfigError(
                "verifier.contradiction_threshold must be between 0.0 and 1.0");
        if (config.verifier.rule_confidence_decay < 0.0f)
            throw ConfigError(
                "verifier.rule_confidence_decay must be >= 0.0");
        if (config.verifier.min_rule_confidence < 0.0f ||
            config.verifier.min_rule_confidence > 1.0f)
            throw ConfigError(
                "verifier.min_rule_confidence must be between 0.0 and 1.0");

        // Feedback
        if (config.feedback.max_retries < 0)
            throw ConfigError("feedback.max_retries must be >= 0");
        if (config.feedback.retry_delay_ms < 0)
            throw ConfigError("feedback.retry_delay_ms must be >= 0");

        // Retriever
        const std::vector<std::string> valid_retriever_modes =
        { "keyword", "semantic", "hybrid" };
        bool valid_rmode = false;
        for (const auto& m : valid_retriever_modes) {
            if (config.retriever.mode == m) { valid_rmode = true; break; }
        }
        if (!valid_rmode)
            throw ConfigError(
                "retriever.mode must be one of: keyword, semantic, hybrid");
        if (config.retriever.keyword_weight < 0.0f ||
            config.retriever.keyword_weight > 1.0f)
            throw ConfigError(
                "retriever.keyword_weight must be between 0.0 and 1.0");
        if (config.retriever.semantic_weight < 0.0f ||
            config.retriever.semantic_weight > 1.0f)
            throw ConfigError(
                "retriever.semantic_weight must be between 0.0 and 1.0");
        if (config.retriever.max_results < 1)
            throw ConfigError("retriever.max_results must be >= 1");
        if (config.retriever.min_score < 0.0f ||
            config.retriever.min_score > 1.0f)
            throw ConfigError(
                "retriever.min_score must be between 0.0 and 1.0");
        const std::vector<std::string> valid_rebuild_strategies =
        { "on_demand", "periodic", "explicit" };
        bool valid_strat = false;
        for (const auto& s : valid_rebuild_strategies) {
            if (config.retriever.cache_rebuild_strategy == s) {
                valid_strat = true; break;
            }
        }
        if (!valid_strat)
            throw ConfigError(
                "retriever.cache_rebuild_strategy must be one of: "
                "on_demand, periodic, explicit");
        if (config.retriever.cache_rebuild_threshold < 1)
            throw ConfigError(
                "retriever.cache_rebuild_threshold must be >= 1");
        if (config.retriever.cache_rebuild_interval_seconds < 1)
            throw ConfigError(
                "retriever.cache_rebuild_interval_seconds must be >= 1");

        // API
        if (config.api.port < 1 || config.api.port > 65535)
            throw ConfigError("api.port must be between 1 and 65535");
        if (config.api.host.empty())
            throw ConfigError("api.host cannot be empty");
        if (config.api.auth_enabled && config.api.api_key.empty())
            throw ConfigError(
                "api.api_key cannot be empty when api.auth_enabled is true");
        if (config.api.max_request_size_kb < 1)
            throw ConfigError("api.max_request_size_kb must be >= 1");
        if (config.api.request_timeout_seconds < 1)
            throw ConfigError("api.request_timeout_seconds must be >= 1");

        // Tools
        if (config.tools.max_browse_depth < 1)
            throw ConfigError("tools.max_browse_depth must be >= 1");
        if (config.tools.search_results_limit < 1)
            throw ConfigError("tools.search_results_limit must be >= 1");

        // Benchmark
        if (config.benchmark.dataset.empty())
            throw ConfigError("benchmark.dataset cannot be empty");
        if (config.benchmark.metrics.empty())
            throw ConfigError("benchmark.metrics cannot be empty");
        if (config.benchmark.eval_frequency_seconds < 1)
            throw ConfigError(
                "benchmark.eval_frequency_seconds must be >= 1");

        // Logging
        const std::vector<std::string> valid_levels =
        { "trace", "debug", "info", "warn", "error", "fatal" };
        bool valid_level = false;
        for (const auto& l : valid_levels) {
            if (config.logging.level == l) { valid_level = true; break; }
        }
        if (!valid_level)
            throw ConfigError(
                "logging.level must be one of: "
                "trace, debug, info, warn, error, fatal");

        LOG_DEBUG("Config validation passed");
    }

    // =========================================================================
    // to_json_string
    // =========================================================================

    std::string ConfigLoader::to_json_string(const CardinalConfig& config) {
        json j;

        j["model"]["path"] = config.model.path;
        j["model"]["chat_template"] = config.model.chat_template;
        j["model"]["context_length"] = config.model.context_length;
        j["model"]["gpu_layers"] = config.model.gpu_layers;
        j["model"]["threads"] = config.model.threads;

        j["inference"]["temperature"] = config.inference.temperature;
        j["inference"]["top_p"] = config.inference.top_p;
        j["inference"]["max_tokens_feeling"] = config.inference.max_tokens_feeling;
        j["inference"]["max_tokens_response"] = config.inference.max_tokens_response;

        j["feeling_schema"]["type"] = config.feeling_schema.type;
        j["feeling_schema"]["grammar_path"] = config.feeling_schema.grammar_path;
        j["feeling_schema"]["max_tokens"] = config.feeling_schema.max_tokens;

        j["memory"]["rule_store_path"] = config.memory.rule_store_path;
        j["memory"]["knowledge_graph_path"] = config.memory.knowledge_graph_path;
        j["memory"]["episodic_log_path"] = config.memory.episodic_log_path;
        j["memory"]["max_rules"] = config.memory.max_rules;

        j["verifier"]["mode"] = config.verifier.mode;
        j["verifier"]["neural_model_path"] = config.verifier.neural_model_path;
        j["verifier"]["neural_gpu_layers"] = config.verifier.neural_gpu_layers;
        j["verifier"]["neural_max_tokens"] = config.verifier.neural_max_tokens;
        j["verifier"]["contradiction_threshold"] = config.verifier.contradiction_threshold;
        j["verifier"]["rule_confidence_decay"] = config.verifier.rule_confidence_decay;
        j["verifier"]["min_rule_confidence"] = config.verifier.min_rule_confidence;

        j["feedback"]["max_retries"] = config.feedback.max_retries;
        j["feedback"]["retry_delay_ms"] = config.feedback.retry_delay_ms;
        j["feedback"]["rule_injection_format"] = config.feedback.rule_injection_format;

        j["retriever"]["mode"] = config.retriever.mode;
        j["retriever"]["keyword_weight"] = config.retriever.keyword_weight;
        j["retriever"]["semantic_weight"] = config.retriever.semantic_weight;
        j["retriever"]["max_results"] = config.retriever.max_results;
        j["retriever"]["min_score"] = config.retriever.min_score;
        j["retriever"]["cache_rebuild_strategy"] = config.retriever.cache_rebuild_strategy;
        j["retriever"]["cache_rebuild_threshold"] = config.retriever.cache_rebuild_threshold;
        j["retriever"]["cache_rebuild_interval_seconds"] =
            config.retriever.cache_rebuild_interval_seconds;

        j["api"]["http_enabled"] = config.api.http_enabled;
        j["api"]["host"] = config.api.host;
        j["api"]["port"] = config.api.port;
        j["api"]["auth_enabled"] = config.api.auth_enabled;
        j["api"]["api_key"] = config.api.api_key;
        j["api"]["stream_enabled"] = config.api.stream_enabled;
        j["api"]["max_request_size_kb"] = config.api.max_request_size_kb;
        j["api"]["request_timeout_seconds"] = config.api.request_timeout_seconds;

        j["tools"]["browser_enabled"] = config.tools.browser_enabled;
        j["tools"]["max_browse_depth"] = config.tools.max_browse_depth;
        j["tools"]["search_results_limit"] = config.tools.search_results_limit;

        j["benchmark"]["dataset"] = config.benchmark.dataset;
        j["benchmark"]["metrics"] = config.benchmark.metrics;
        j["benchmark"]["eval_frequency_seconds"] = config.benchmark.eval_frequency_seconds;

        j["logging"]["level"] = config.logging.level;
        j["logging"]["path"] = config.logging.path;

        return j.dump(2);
    }

    // =========================================================================
    // Section parsers
    // =========================================================================

    ModelConfig ConfigLoader::parse_model(const auto& j) {
        ModelConfig c;
        c.path = require<std::string>(j, "model", "path");
        c.chat_template = require<std::string>(j, "model", "chat_template");
        c.context_length = require<int>(j, "model", "context_length");
        c.gpu_layers = require<int>(j, "model", "gpu_layers");
        c.threads = require<int>(j, "model", "threads");
        return c;
    }

    InferenceConfig ConfigLoader::parse_inference(const auto& j) {
        InferenceConfig c;
        c.temperature = require<float>(j, "inference", "temperature");
        c.top_p = require<float>(j, "inference", "top_p");
        c.max_tokens_feeling = require<int>(j, "inference", "max_tokens_feeling");
        c.max_tokens_response = require<int>(j, "inference", "max_tokens_response");
        return c;
    }

    FeelingSchemaConfig ConfigLoader::parse_feeling_schema(const auto& j) {
        FeelingSchemaConfig c;
        c.type = require<std::string>(j, "feeling_schema", "type");
        c.grammar_path = require<std::string>(j, "feeling_schema", "grammar_path");
        c.max_tokens = require<int>(j, "feeling_schema", "max_tokens");

        if (!j.contains("fields") || !j["fields"].is_object()) {
            throw ConfigError("feeling_schema.fields must be a JSON object");
        }
        for (const auto& [key, val] : j["fields"].items()) {
            FeelingSchemaField field;
            field.name = key;
            field.type_description = val.template get<std::string>();
            c.fields.push_back(field);
        }
        return c;
    }

    MemoryConfig ConfigLoader::parse_memory(const auto& j) {
        MemoryConfig c;
        c.rule_store_path = require<std::string>(j, "memory", "rule_store_path");
        c.knowledge_graph_path = require<std::string>(j, "memory", "knowledge_graph_path");
        c.episodic_log_path = require<std::string>(j, "memory", "episodic_log_path");
        c.max_rules = require<int>(j, "memory", "max_rules");
        return c;
    }

    VerifierConfig ConfigLoader::parse_verifier(const auto& j) {
        VerifierConfig c;
        c.mode = require<std::string>(j, "verifier", "mode");
        c.neural_model_path = optional_field<std::string>(j, "neural_model_path", "");
        c.neural_gpu_layers = optional_field<int>(j, "neural_gpu_layers", 0);
        c.neural_max_tokens = optional_field<int>(j, "neural_max_tokens", 512);
        c.contradiction_threshold = require<float>(j, "verifier", "contradiction_threshold");
        c.rule_confidence_decay = require<float>(j, "verifier", "rule_confidence_decay");
        c.min_rule_confidence = require<float>(j, "verifier", "min_rule_confidence");
        return c;
    }

    FeedbackConfig ConfigLoader::parse_feedback(const auto& j) {
        FeedbackConfig c;
        c.max_retries = require<int>(j, "feedback", "max_retries");
        c.retry_delay_ms = require<int>(j, "feedback", "retry_delay_ms");
        c.rule_injection_format = require<std::string>(j, "feedback", "rule_injection_format");
        return c;
    }

    RetrieverConfig ConfigLoader::parse_retriever(const auto& j) {
        RetrieverConfig c;
        c.mode = require<std::string>(j, "retriever", "mode");
        c.keyword_weight = require<float>(j, "retriever", "keyword_weight");
        c.semantic_weight = require<float>(j, "retriever", "semantic_weight");
        c.max_results = require<int>(j, "retriever", "max_results");
        c.min_score = require<float>(j, "retriever", "min_score");
        c.cache_rebuild_strategy =
            require<std::string>(j, "retriever", "cache_rebuild_strategy");
        c.cache_rebuild_threshold =
            require<int>(j, "retriever", "cache_rebuild_threshold");
        c.cache_rebuild_interval_seconds =
            require<int>(j, "retriever", "cache_rebuild_interval_seconds");
        return c;
    }

    ApiConfig ConfigLoader::parse_api(const auto& j) {
        ApiConfig c;
        c.http_enabled = require<bool>(j, "api", "http_enabled");
        c.host = require<std::string>(j, "api", "host");
        c.port = require<int>(j, "api", "port");
        c.auth_enabled = require<bool>(j, "api", "auth_enabled");
        c.api_key = optional_field<std::string>(j, "api_key", "");
        c.stream_enabled = require<bool>(j, "api", "stream_enabled");
        c.max_request_size_kb = require<int>(j, "api", "max_request_size_kb");
        c.request_timeout_seconds = require<int>(j, "api", "request_timeout_seconds");
        return c;
    }

    ToolsConfig ConfigLoader::parse_tools(const auto& j) {
        ToolsConfig c;
        c.browser_enabled = require<bool>(j, "tools", "browser_enabled");
        c.max_browse_depth = require<int>(j, "tools", "max_browse_depth");
        c.search_results_limit = require<int>(j, "tools", "search_results_limit");
        return c;
    }

    BenchmarkConfig ConfigLoader::parse_benchmark(const auto& j) {
        BenchmarkConfig c;
        c.dataset = require<std::string>(j, "benchmark", "dataset");
        c.eval_frequency_seconds = require<int>(j, "benchmark", "eval_frequency_seconds");

        if (!j.contains("metrics") || !j["metrics"].is_array()) {
            throw ConfigError("benchmark.metrics must be a JSON array");
        }
        for (const auto& m : j["metrics"]) {
            c.metrics.push_back(m.template get<std::string>());
        }
        return c;
    }

    LoggingConfig ConfigLoader::parse_logging(const auto& j) {
        LoggingConfig c;
        c.level = require<std::string>(j, "logging", "level");
        c.path = require<std::string>(j, "logging", "path");
        return c;
    }

} // namespace cardinal