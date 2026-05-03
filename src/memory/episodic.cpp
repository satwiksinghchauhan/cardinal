// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Episodic Memory Implementation
// File: src/memory/episodic.cpp
// =============================================================================

#include "episodic.h"
#include "utils/logger.h"

#include <filesystem>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace cardinal {

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    EpisodicMemory::EpisodicMemory(const CardinalConfig& config)
        : config_(config)
    {
        LOG_INFO("EpisodicMemory created - path: " +
            config_.memory.episodic_log_path);
    }

    EpisodicMemory::~EpisodicMemory() {
        close();
    }

    // =============================================================================
    // open
    // =============================================================================

    void EpisodicMemory::open() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (open_) return;

        // Create parent directory if needed
        std::filesystem::path p(config_.memory.episodic_log_path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        // Open in append mode - never overwrite existing episodes
        log_file_.open(config_.memory.episodic_log_path,
            std::ios::app | std::ios::out);
        if (!log_file_.is_open()) {
            throw EpisodicError("Failed to open episodic log: " +
                config_.memory.episodic_log_path);
        }

        open_ = true;
        LOG_INFO("EpisodicMemory opened");
    }

    // =============================================================================
    // close
    // =============================================================================

    void EpisodicMemory::close() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!open_) return;

        log_file_.flush();
        log_file_.close();
        open_ = false;

        LOG_DEBUG("EpisodicMemory closed");
    }

    // =============================================================================
    // log_episode
    // =============================================================================

    std::string EpisodicMemory::log_episode(
        const std::string& user_message,
        const std::string& response,
        const FeelingOutput& feeling,
        int                  pass1_tokens,
        int                  pass2_tokens,
        long long            total_ms,
        const std::string& extracted_rule_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!open_) {
            LOG_WARN("EpisodicMemory: log_episode called but not open");
            return "";
        }

        Episode ep = build_episode(user_message, response, feeling,
            pass1_tokens, pass2_tokens,
            total_ms, extracted_rule_id);

        // Write to file
        std::string line = episode_to_jsonl(ep);
        append_to_file(line);

        // Add to in-memory index
        index_.push_back(ep);

        // Trim index if too large - keep most recent
        if (static_cast<int>(index_.size()) > MAX_INDEX) {
            index_.erase(index_.begin(),
                index_.begin() + (index_.size() - MAX_INDEX));
        }

        LOG_DEBUG("Episode logged: " + ep.id +
            " [" + ep.reasoning_domain + "]" +
            " conf=" + std::to_string(ep.confidence) +
            " rule=" + (ep.rule_candidate ? "yes" : "no"));

        return ep.id;
    }

    // =============================================================================
    // Query methods
    // =============================================================================

    std::vector<Episode> EpisodicMemory::get_recent(int n) const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (index_.empty()) return {};

        int start = std::max(0, static_cast<int>(index_.size()) - n);
        return std::vector<Episode>(index_.begin() + start, index_.end());
    }

    std::vector<Episode> EpisodicMemory::get_rule_candidates(int max) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Episode> result;
        for (auto it = index_.rbegin(); it != index_.rend(); ++it) {
            if (it->rule_candidate) {
                result.push_back(*it);
                if (static_cast<int>(result.size()) >= max) break;
            }
        }
        return result;
    }

    std::vector<Episode> EpisodicMemory::get_contradictions(int max) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Episode> result;
        for (auto it = index_.rbegin(); it != index_.rend(); ++it) {
            if (it->contradiction) {
                result.push_back(*it);
                if (static_cast<int>(result.size()) >= max) break;
            }
        }
        return result;
    }

    std::vector<Episode> EpisodicMemory::get_by_domain(
        const std::string& domain, int max) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Episode> result;
        for (auto it = index_.rbegin(); it != index_.rend(); ++it) {
            if (it->reasoning_domain == domain) {
                result.push_back(*it);
                if (static_cast<int>(result.size()) >= max) break;
            }
        }
        return result;
    }

    // =============================================================================
    // stats
    // =============================================================================

    EpisodicStats EpisodicMemory::stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        EpisodicStats s{};
        s.total_episodes = static_cast<int>(index_.size());

        if (index_.empty()) return s;

        float conf_sum = 0.0f;
        float ms_sum = 0.0f;

        for (const auto& ep : index_) {
            if (ep.rule_candidate) ++s.rule_candidate_episodes;
            if (ep.contradiction)  ++s.contradiction_episodes;
            if (ep.uncertainty)    ++s.uncertain_episodes;
            conf_sum += ep.confidence;
            ms_sum += static_cast<float>(ep.total_ms);
            s.total_tokens += ep.pass1_tokens + ep.pass2_tokens;
        }

        s.avg_confidence = conf_sum / static_cast<float>(s.total_episodes);
        s.avg_total_ms = ms_sum / static_cast<float>(s.total_episodes);

        return s;
    }

    int EpisodicMemory::episode_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(index_.size());
    }

    // =============================================================================
    // Internal helpers
    // =============================================================================

    Episode EpisodicMemory::build_episode(
        const std::string& user_message,
        const std::string& response,
        const FeelingOutput& feeling,
        int                  pass1_tokens,
        int                  pass2_tokens,
        long long            total_ms,
        const std::string& rule_id) const
    {
        Episode ep;
        ep.id = JsonParser::generate_id();
        ep.timestamp = JsonParser::current_timestamp();
        ep.user_message = user_message;
        ep.feeling = feeling;
        ep.reasoning_domain = feeling.reasoning_domain;
        ep.rule_candidate = feeling.rule_candidate_signal;
        ep.contradiction = feeling.contradiction_flag;
        ep.uncertainty = feeling.uncertainty_flag;
        ep.confidence = feeling.confidence;
        ep.pass1_tokens = pass1_tokens;
        ep.pass2_tokens = pass2_tokens;
        ep.total_ms = total_ms;
        ep.extracted_rule_id = rule_id;

        // Summarize response - first 200 chars
        ep.response_summary = response.substr(
            0, std::min(200, static_cast<int>(response.size())));
        if (response.size() > 200) ep.response_summary += "...";

        return ep;
    }

    std::string EpisodicMemory::episode_to_jsonl(const Episode& ep) const {
        json j;
        j["id"] = ep.id;
        j["timestamp"] = ep.timestamp;
        j["user_message"] = ep.user_message;
        j["response_summary"] = ep.response_summary;
        j["reasoning_domain"] = ep.reasoning_domain;
        j["reasoning_type"] = ep.feeling.reasoning_type;
        j["confidence"] = ep.confidence;
        j["rule_candidate"] = ep.rule_candidate;
        j["contradiction"] = ep.contradiction;
        j["uncertainty"] = ep.uncertainty;
        j["pass1_tokens"] = ep.pass1_tokens;
        j["pass2_tokens"] = ep.pass2_tokens;
        j["total_ms"] = ep.total_ms;
        j["extracted_rule_id"] = ep.extracted_rule_id;
        return j.dump();  // Single line JSON
    }

    void EpisodicMemory::append_to_file(const std::string& line) {
        if (!log_file_.is_open()) return;
        log_file_ << line << "\n";
        log_file_.flush(); // Each episode is flushed immediately
    }

} // namespace cardinal