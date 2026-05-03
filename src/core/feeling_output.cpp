// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Feeling Output Pipeline Context Implementation
// File: src/core/feeling_output.cpp
// =============================================================================

#include "feeling_output.h"
#include "utils/logger.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cardinal {

    // =============================================================================
    // PassState
    // =============================================================================

    std::string pass_state_to_string(PassState state) {
        switch (state) {
        case PassState::IDLE:           return "IDLE";
        case PassState::PASS1_FEELING:  return "PASS1_FEELING";
        case PassState::PASS2_RESPONSE: return "PASS2_RESPONSE";
        case PassState::COMPLETE:       return "COMPLETE";
        case PassState::FAILED:         return "FAILED";
        default:                        return "UNKNOWN";
        }
    }

    // =============================================================================
    // GrammarBuffer
    // =============================================================================

    void GrammarBuffer::load(const std::string& grammar_path) {
        path = grammar_path;

        std::ifstream file(grammar_path);
        if (!file.is_open()) {
            throw std::runtime_error("GrammarBuffer: failed to open grammar file: " +
                grammar_path);
        }

        std::ostringstream oss;
        oss << file.rdbuf();
        content = oss.str();

        if (content.empty()) {
            throw std::runtime_error("GrammarBuffer: grammar file is empty: " +
                grammar_path);
        }

        loaded = true;
        LOG_INFO("Grammar loaded: " + grammar_path +
            " (" + std::to_string(content.size()) + " bytes)");
    }

    // =============================================================================
    // SyntheticTurn
    // =============================================================================

    void SyntheticTurn::build(const FeelingOutput& feeling) {
        content = JsonParser::serialize_feeling_output(feeling);
        injected = false;
        LOG_DEBUG("Synthetic turn built: " + content);
    }

    std::string SyntheticTurn::format() const {
        // Format as a clean assistant turn for context injection
        // The model will treat this JSON as its own prior thought
        return "<feeling_state>\n" + content + "\n</feeling_state>";
    }

    // =============================================================================
    // InferenceMetrics
    // =============================================================================

    float InferenceMetrics::pass1_tps() const {
        if (pass1_duration.count() == 0) return 0.0f;
        return static_cast<float>(pass1_tokens_generated) /
            (static_cast<float>(pass1_duration.count()) / 1000.0f);
    }

    float InferenceMetrics::pass2_tps() const {
        if (pass2_duration.count() == 0) return 0.0f;
        return static_cast<float>(pass2_tokens_generated) /
            (static_cast<float>(pass2_duration.count()) / 1000.0f);
    }

    std::string InferenceMetrics::to_string() const {
        std::ostringstream oss;
        oss << "InferenceMetrics{"
            << "pass1=" << pass1_duration.count() << "ms"
            << " (" << pass1_tokens_generated << " tok, "
            << static_cast<int>(pass1_tps()) << " tps)"
            << ", pass2=" << pass2_duration.count() << "ms"
            << " (" << pass2_tokens_generated << " tok, "
            << static_cast<int>(pass2_tps()) << " tps)"
            << ", total=" << total_duration.count() << "ms"
            << ", retries=" << retry_count
            << ", feeling_valid=" << (feeling_valid ? "true" : "false")
            << "}";
        return oss.str();
    }

    // =============================================================================
    // FeelingContext
    // =============================================================================

    FeelingContext::FeelingContext(const CardinalConfig& config)
        : config_(config)
        , state_(PassState::IDLE)
        , retry_count_(0)
    {
        // Load grammar at construction - fail fast if grammar file is missing
        grammar_.load(config_.feeling_schema.grammar_path);
    }

    // -----------------------------------------------------------------------------
    // reset - clear per-cycle state, keep grammar
    // -----------------------------------------------------------------------------
    void FeelingContext::reset() {
        state_ = PassState::IDLE;
        retry_count_ = 0;
        raw_feeling_.clear();
        feeling_output_.reset();
        synthetic_turn_ = SyntheticTurn{};
        final_response_.clear();
        metrics_ = InferenceMetrics{};

        LOG_DEBUG("FeelingContext reset");
    }

    // -----------------------------------------------------------------------------
    // set_raw_feeling
    // -----------------------------------------------------------------------------
    void FeelingContext::set_raw_feeling(const std::string& raw_json) {
        raw_feeling_ = raw_json;
        LOG_DEBUG("Raw feeling received: " + raw_json);
    }

    // -----------------------------------------------------------------------------
    // parse_feeling
    // -----------------------------------------------------------------------------
    bool FeelingContext::parse_feeling() {
        if (raw_feeling_.empty()) {
            LOG_WARN("parse_feeling called with empty raw feeling");
            return false;
        }

        try {
            FeelingOutput parsed = JsonParser::parse_feeling_output(raw_feeling_);

            // Logical validation
            std::string validation_error;
            if (!JsonParser::validate_feeling_output(parsed, validation_error)) {
                LOG_WARN("Feeling output failed validation: " + validation_error);
                // Don't reject - log and continue with the parsed output
                // Validation failures are soft warnings, not hard errors
            }

            feeling_output_ = parsed;
            metrics_.feeling_valid = true;

            LOG_DEBUG("Feeling parsed successfully: " + parsed.to_string());
            return true;

        }
        catch (const ParseError& e) {
            LOG_WARN("Failed to parse feeling output (attempt " +
                std::to_string(retry_count_ + 1) + "): " + e.what());
            feeling_output_.reset();
            metrics_.feeling_valid = false;
            return false;
        }
    }

    // -----------------------------------------------------------------------------
    // feeling
    // -----------------------------------------------------------------------------
    const FeelingOutput& FeelingContext::feeling() const {
        if (!feeling_output_.has_value()) {
            throw std::runtime_error("FeelingContext: no valid feeling output available");
        }
        return feeling_output_.value();
    }

    // -----------------------------------------------------------------------------
    // prepare_synthetic_turn
    // -----------------------------------------------------------------------------
    void FeelingContext::prepare_synthetic_turn() {
        if (!has_valid_feeling()) {
            throw std::runtime_error(
                "FeelingContext: cannot prepare synthetic turn without valid feeling");
        }

        synthetic_turn_.build(feeling_output_.value());
        synthetic_turn_.injected = true;

        LOG_DEBUG("Synthetic turn prepared: " + synthetic_turn_.format());
    }

    // -----------------------------------------------------------------------------
    // set_final_response
    // -----------------------------------------------------------------------------
    void FeelingContext::set_final_response(const std::string& response) {
        final_response_ = response;
        LOG_DEBUG("Final response set (" +
            std::to_string(response.size()) + " chars)");
    }

    // -----------------------------------------------------------------------------
    // Retry logic
    // -----------------------------------------------------------------------------
    bool FeelingContext::should_retry() const {
        return retry_count_ < config_.feedback.max_retries;
    }

    void FeelingContext::increment_retry() {
        ++retry_count_;
        ++metrics_.retry_count;
        LOG_WARN("Inference retry " + std::to_string(retry_count_) +
            "/" + std::to_string(config_.feedback.max_retries));
    }

    // -----------------------------------------------------------------------------
    // Timer methods
    // -----------------------------------------------------------------------------
    void FeelingContext::start_pass1_timer() {
        pass1_start_ = std::chrono::steady_clock::now();
    }

    void FeelingContext::stop_pass1_timer() {
        auto end = std::chrono::steady_clock::now();
        metrics_.pass1_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - pass1_start_);
    }

    void FeelingContext::start_pass2_timer() {
        pass2_start_ = std::chrono::steady_clock::now();
    }

    void FeelingContext::stop_pass2_timer() {
        auto end = std::chrono::steady_clock::now();
        metrics_.pass2_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - pass2_start_);
        metrics_.total_duration =
            metrics_.pass1_duration + metrics_.pass2_duration;
    }

    // -----------------------------------------------------------------------------
    // to_string
    // -----------------------------------------------------------------------------
    std::string FeelingContext::to_string() const {
        std::ostringstream oss;
        oss << "FeelingContext{"
            << "state=" << pass_state_to_string(state_)
            << ", retries=" << retry_count_
            << ", has_feeling=" << (has_valid_feeling() ? "true" : "false")
            << ", has_response=" << (has_response() ? "true" : "false")
            << ", " << metrics_.to_string()
            << "}";
        return oss.str();
    }

} // namespace cardinal