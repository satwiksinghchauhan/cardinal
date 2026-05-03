// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Settings Manager Implementation
// File: src/api/cardinal_settings.cpp
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "api/cardinal_settings.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace cardinal {

    // =========================================================================
    // Constructor
    // =========================================================================

    SettingsManager::SettingsManager(const CardinalConfig& config,
        EpisodicRetriever& retriever,
        InferencePipeline& pipeline)
        : config_(config)
        , retriever_(retriever)
        , pipeline_(pipeline)
    {
        // Initialize current settings from config
        current_.retriever_mode = config.retriever.mode;
        current_.keyword_weight = config.retriever.keyword_weight;
        current_.semantic_weight = config.retriever.semantic_weight;
        current_.max_retrieval_results = config.retriever.max_results;
        current_.min_retrieval_score = config.retriever.min_score;
        current_.verifier_mode = config.verifier.mode;
        current_.min_rule_confidence = config.verifier.min_rule_confidence;
        current_.contradiction_threshold = config.verifier.contradiction_threshold;
        current_.temperature = config.inference.temperature;
        current_.top_p = config.inference.top_p;
        current_.stream_responses = config.api.stream_enabled;
        current_.log_level = config.logging.level;

        LOG_INFO("SettingsManager initialized");
    }

    // =========================================================================
    // get
    // =========================================================================

    CardinalSettings SettingsManager::get() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return current_;
    }

    // =========================================================================
    // update
    // =========================================================================

    CardinalVoidResult SettingsManager::update(const CardinalSettings& settings) {
        // Validate first -- if invalid, nothing changes
        auto validation = validate(settings);
        if (!validation.ok()) return validation;

        std::unique_lock<std::shared_mutex> lock(mutex_);
        current_ = settings;
        propagate(settings);

        LOG_INFO("SettingsManager: settings updated");
        return CardinalVoidResult::success();
    }

    // =========================================================================
    // set -- single field by name
    // =========================================================================

    CardinalVoidResult SettingsManager::set(const std::string& key,
        const std::string& value)
    {
        // Read current settings, modify the one field, then update
        CardinalSettings s = get();

        try {
            if (key == "retriever_mode") {
                s.retriever_mode = value;
            }
            else if (key == "keyword_weight") {
                s.keyword_weight = std::stof(value);
            }
            else if (key == "semantic_weight") {
                s.semantic_weight = std::stof(value);
            }
            else if (key == "max_retrieval_results") {
                s.max_retrieval_results = std::stoi(value);
            }
            else if (key == "min_retrieval_score") {
                s.min_retrieval_score = std::stof(value);
            }
            else if (key == "verifier_mode") {
                s.verifier_mode = value;
            }
            else if (key == "min_rule_confidence") {
                s.min_rule_confidence = std::stof(value);
            }
            else if (key == "contradiction_threshold") {
                s.contradiction_threshold = std::stof(value);
            }
            else if (key == "temperature") {
                s.temperature = std::stof(value);
            }
            else if (key == "top_p") {
                s.top_p = std::stof(value);
            }
            else if (key == "stream_responses") {
                s.stream_responses = (value == "true" || value == "1");
            }
            else if (key == "log_level") {
                s.log_level = value;
            }
            else {
                return CardinalVoidResult::failure(
                    CardinalStatus::INVALID_INPUT,
                    "Unknown setting key: '" + key + "'");
            }
        }
        catch (const std::invalid_argument&) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "Invalid value for '" + key + "': '" + value + "'");
        }
        catch (const std::out_of_range&) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "Value out of range for '" + key + "': '" + value + "'");
        }

        return update(s);
    }

    // =========================================================================
    // reset
    // =========================================================================

    CardinalVoidResult SettingsManager::reset() {
        CardinalSettings defaults;
        defaults.retriever_mode = config_.retriever.mode;
        defaults.keyword_weight = config_.retriever.keyword_weight;
        defaults.semantic_weight = config_.retriever.semantic_weight;
        defaults.max_retrieval_results = config_.retriever.max_results;
        defaults.min_retrieval_score = config_.retriever.min_score;
        defaults.verifier_mode = config_.verifier.mode;
        defaults.min_rule_confidence = config_.verifier.min_rule_confidence;
        defaults.contradiction_threshold = config_.verifier.contradiction_threshold;
        defaults.temperature = config_.inference.temperature;
        defaults.top_p = config_.inference.top_p;
        defaults.stream_responses = config_.api.stream_enabled;
        defaults.log_level = config_.logging.level;

        auto result = update(defaults);
        if (result.ok()) {
            LOG_INFO("SettingsManager: settings reset to config defaults");
        }
        return result;
    }

    // =========================================================================
    // to_json
    // =========================================================================

    std::string SettingsManager::to_json() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);

        json j;
        j["retriever_mode"] = current_.retriever_mode;
        j["keyword_weight"] = current_.keyword_weight;
        j["semantic_weight"] = current_.semantic_weight;
        j["max_retrieval_results"] = current_.max_retrieval_results;
        j["min_retrieval_score"] = current_.min_retrieval_score;
        j["verifier_mode"] = current_.verifier_mode;
        j["min_rule_confidence"] = current_.min_rule_confidence;
        j["contradiction_threshold"] = current_.contradiction_threshold;
        j["temperature"] = current_.temperature;
        j["top_p"] = current_.top_p;
        j["stream_responses"] = current_.stream_responses;
        j["log_level"] = current_.log_level;

        return j.dump(2);
    }

    // =========================================================================
    // from_json
    // =========================================================================

    CardinalResult<CardinalSettings>
        SettingsManager::from_json(const std::string& json_str) const {
        json j;
        try {
            j = json::parse(json_str);
        }
        catch (const json::parse_error& e) {
            return CardinalResult<CardinalSettings>::failure(
                CardinalStatus::INVALID_INPUT,
                "JSON parse error: " + std::string(e.what()));
        }

        // Start from current settings so partial updates work --
        // only fields present in the JSON are changed
        CardinalSettings s = get();

        try {
            if (j.contains("retriever_mode") && j["retriever_mode"].is_string())
                s.retriever_mode = j["retriever_mode"].get<std::string>();
            if (j.contains("keyword_weight") && j["keyword_weight"].is_number())
                s.keyword_weight = j["keyword_weight"].get<float>();
            if (j.contains("semantic_weight") && j["semantic_weight"].is_number())
                s.semantic_weight = j["semantic_weight"].get<float>();
            if (j.contains("max_retrieval_results") && j["max_retrieval_results"].is_number_integer())
                s.max_retrieval_results = j["max_retrieval_results"].get<int>();
            if (j.contains("min_retrieval_score") && j["min_retrieval_score"].is_number())
                s.min_retrieval_score = j["min_retrieval_score"].get<float>();
            if (j.contains("verifier_mode") && j["verifier_mode"].is_string())
                s.verifier_mode = j["verifier_mode"].get<std::string>();
            if (j.contains("min_rule_confidence") && j["min_rule_confidence"].is_number())
                s.min_rule_confidence = j["min_rule_confidence"].get<float>();
            if (j.contains("contradiction_threshold") && j["contradiction_threshold"].is_number())
                s.contradiction_threshold = j["contradiction_threshold"].get<float>();
            if (j.contains("temperature") && j["temperature"].is_number())
                s.temperature = j["temperature"].get<float>();
            if (j.contains("top_p") && j["top_p"].is_number())
                s.top_p = j["top_p"].get<float>();
            if (j.contains("stream_responses") && j["stream_responses"].is_boolean())
                s.stream_responses = j["stream_responses"].get<bool>();
            if (j.contains("log_level") && j["log_level"].is_string())
                s.log_level = j["log_level"].get<std::string>();
        }
        catch (const json::exception& e) {
            return CardinalResult<CardinalSettings>::failure(
                CardinalStatus::INVALID_INPUT,
                "JSON field error: " + std::string(e.what()));
        }

        return CardinalResult<CardinalSettings>::success(s);
    }

    // =========================================================================
    // validate
    // =========================================================================

    CardinalVoidResult
        SettingsManager::validate(const CardinalSettings& s) const {
        // Retriever mode
        if (s.retriever_mode != "keyword" &&
            s.retriever_mode != "semantic" &&
            s.retriever_mode != "hybrid") {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "retriever_mode must be: keyword, semantic, or hybrid");
        }

        // Weights
        if (s.keyword_weight < 0.0f || s.keyword_weight > 1.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "keyword_weight must be between 0.0 and 1.0");
        }
        if (s.semantic_weight < 0.0f || s.semantic_weight > 1.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "semantic_weight must be between 0.0 and 1.0");
        }

        // Results
        if (s.max_retrieval_results < 1) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "max_retrieval_results must be >= 1");
        }
        if (s.min_retrieval_score < 0.0f || s.min_retrieval_score > 1.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "min_retrieval_score must be between 0.0 and 1.0");
        }

        // Verifier mode
        if (s.verifier_mode != "symbolic" &&
            s.verifier_mode != "neural" &&
            s.verifier_mode != "hybrid") {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "verifier_mode must be: symbolic, neural, or hybrid");
        }

        // Confidence thresholds
        if (s.min_rule_confidence < 0.0f || s.min_rule_confidence > 1.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "min_rule_confidence must be between 0.0 and 1.0");
        }
        if (s.contradiction_threshold < 0.0f ||
            s.contradiction_threshold > 1.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "contradiction_threshold must be between 0.0 and 1.0");
        }

        // Inference
        if (s.temperature < 0.0f || s.temperature > 2.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "temperature must be between 0.0 and 2.0");
        }
        if (s.top_p <= 0.0f || s.top_p > 1.0f) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "top_p must be between 0.0 and 1.0");
        }

        // Log level
        const std::vector<std::string> valid_levels =
        { "trace", "debug", "info", "warn", "error", "fatal" };
        bool valid = false;
        for (const auto& l : valid_levels) {
            if (s.log_level == l) { valid = true; break; }
        }
        if (!valid) {
            return CardinalVoidResult::failure(
                CardinalStatus::INVALID_INPUT,
                "log_level must be: trace, debug, info, warn, error, or fatal");
        }

        return CardinalVoidResult::success();
    }

    // =========================================================================
    // propagate
    // Apply validated settings to core components immediately.
    // Called inside update() after validation passes and lock is held.
    // =========================================================================

    void SettingsManager::propagate(const CardinalSettings& s) {
        // Retriever -- mode and weights
        RetrievalMode mode = RetrievalMode::HYBRID;
        if (s.retriever_mode == "keyword")  mode = RetrievalMode::KEYWORD;
        if (s.retriever_mode == "semantic") mode = RetrievalMode::SEMANTIC;

        retriever_.set_mode(mode);
        retriever_.set_weights(s.keyword_weight, s.semantic_weight);

        // Log level -- propagate to logger
        // Logger::instance().set_level() if your logger supports it
        // Skipped for now -- logger level is set at startup
        // Will add when Logger gets a set_level() method

        LOG_DEBUG("SettingsManager: propagated -- retriever_mode=" +
            s.retriever_mode +
            " verifier_mode=" + s.verifier_mode +
            " temperature=" + std::to_string(s.temperature));
    }

} // namespace cardinal