// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Rule Store Implementation
// File: src/memory/rule_store.cpp
// =============================================================================

#include "rule_store.h"
#include "utils/logger.h"

#include <algorithm>
#include <sstream>
#include <cmath>
#include <set>

namespace cardinal {

    // =============================================================================
    // Constructor
    // =============================================================================

    RuleStore::RuleStore(const CardinalConfig& config)
        : config_(config)
    {
        LOG_INFO("RuleStore created - path: " + config_.memory.rule_store_path);
    }

    // =============================================================================
    // load
    // =============================================================================

    void RuleStore::load() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto rules_vec = JsonParser::load_rules(config_.memory.rule_store_path);
        rules_.clear();
        for (const auto& rule : rules_vec) {
            rules_[rule.id] = rule;
        }

        loaded_ = true;
        dirty_ = false;

        LOG_INFO("RuleStore loaded - " + std::to_string(rules_.size()) + " rules");
    }

    // =============================================================================
    // save
    // =============================================================================

    void RuleStore::save() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!dirty_) {
            LOG_DEBUG("RuleStore: no changes, skipping save");
            return;
        }

        std::vector<Rule> rules_vec;
        rules_vec.reserve(rules_.size());
        for (const auto& [id, rule] : rules_) {
            rules_vec.push_back(rule);
        }

        // Sort by confidence descending for readability
        std::sort(rules_vec.begin(), rules_vec.end(),
            [](const Rule& a, const Rule& b) {
                return a.confidence > b.confidence;
            });

        JsonParser::save_rules(config_.memory.rule_store_path, rules_vec);
        dirty_ = false;

        LOG_INFO("RuleStore saved - " + std::to_string(rules_vec.size()) + " rules");
    }

    // =============================================================================
    // add_rule
    // =============================================================================

    std::string RuleStore::add_rule(const std::string& domain,
        const std::string& condition,
        const std::string& consequence,
        float              initial_confidence,
        const std::string& episode_id,
        const std::string& reasoning_type) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Build candidate rule
        Rule candidate;
        candidate.id = JsonParser::generate_id();
        candidate.domain = domain;
        candidate.condition = condition;
        candidate.consequence = consequence;
        candidate.confidence = std::clamp(initial_confidence, 0.0f, 1.0f);
        candidate.trigger_count = 0;
        candidate.created_at = JsonParser::current_timestamp();
        candidate.updated_at = candidate.created_at;

        // Phase 6: provenance -- empty string for rules created without episode context
        candidate.episode_id = episode_id;
        candidate.reasoning_type = reasoning_type;

        // Check for similar existing rules -- merge instead of duplicate
        for (auto& [id, existing] : rules_) {
            if (existing.domain == domain && is_similar(existing, candidate, 0.7f)) {
                merge_rule(existing, candidate);
                dirty_ = true;
                LOG_DEBUG("RuleStore: merged rule into existing: " + id);
                return id;
            }
        }

        // Enforce max_rules before adding
        if (static_cast<int>(rules_.size()) >= config_.memory.max_rules) {
            auto min_it = std::min_element(rules_.begin(), rules_.end(),
                [](const auto& a, const auto& b) {
                    return a.second.confidence < b.second.confidence;
                });
            if (min_it != rules_.end() &&
                min_it->second.confidence < candidate.confidence) {
                LOG_DEBUG("RuleStore: evicting low-confidence rule: " +
                    min_it->first);
                rules_.erase(min_it);
            }
            else {
                LOG_WARN("RuleStore: at capacity, new rule has lower confidence "
                    "than all existing rules -- discarding");
                return "";
            }
        }

        rules_[candidate.id] = candidate;
        dirty_ = true;

        LOG_DEBUG("RuleStore: added rule [" + domain + "] id=" +
            candidate.id +
            (episode_id.empty() ? "" : " from episode=" + episode_id));

        return candidate.id;
    }

    // =============================================================================
    // update_confidence
    // =============================================================================

    bool RuleStore::update_confidence(const std::string& rule_id, float delta) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rules_.find(rule_id);
        if (it == rules_.end()) return false;

        it->second.confidence = std::clamp(
            it->second.confidence + delta, 0.0f, 1.0f);
        it->second.updated_at = JsonParser::current_timestamp();
        dirty_ = true;

        return true;
    }

    // =============================================================================
    // record_trigger
    // =============================================================================

    bool RuleStore::record_trigger(const std::string& rule_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rules_.find(rule_id);
        if (it == rules_.end()) return false;

        ++it->second.trigger_count;
        // Triggering a rule slightly boosts its confidence
        it->second.confidence = std::clamp(
            it->second.confidence + 0.01f, 0.0f, 1.0f);
        it->second.updated_at = JsonParser::current_timestamp();
        dirty_ = true;

        return true;
    }

    // =============================================================================
    // remove_rule
    // =============================================================================

    bool RuleStore::remove_rule(const std::string& rule_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rules_.find(rule_id);
        if (it == rules_.end()) return false;

        rules_.erase(it);
        dirty_ = true;
        return true;
    }

    // =============================================================================
    // get_rule
    // =============================================================================

    std::optional<Rule> RuleStore::get_rule(const std::string& rule_id) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rules_.find(rule_id);
        if (it == rules_.end()) return std::nullopt;
        return it->second;
    }

    // =============================================================================
    // query
    // =============================================================================

    std::vector<Rule> RuleStore::query(const RuleQuery& q) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Rule> results;
        float min_conf = q.active_only
            ? std::max(q.min_confidence, config_.verifier.min_rule_confidence)
            : q.min_confidence;

        for (const auto& [id, rule] : rules_) {
            // Domain filter
            if (!q.domain.empty() && rule.domain != q.domain) continue;

            // Confidence filter
            if (rule.confidence < min_conf) continue;

            // Condition hint filter (simple substring match)
            if (!q.condition_hint.empty()) {
                if (rule.condition.find(q.condition_hint) == std::string::npos &&
                    rule.consequence.find(q.condition_hint) == std::string::npos) {
                    continue;
                }
            }

            results.push_back(rule);
        }

        // Sort by confidence descending
        std::sort(results.begin(), results.end(),
            [](const Rule& a, const Rule& b) {
                return a.confidence > b.confidence;
            });

        // Limit results
        if (q.max_results > 0 &&
            static_cast<int>(results.size()) > q.max_results) {
            results.resize(q.max_results);
        }

        return results;
    }

    // =============================================================================
    // get_top_rules
    // =============================================================================

    std::vector<Rule> RuleStore::get_top_rules(const std::string& domain,
        int n) const {
        RuleQuery q;
        q.domain = domain;
        q.max_results = n;
        q.active_only = true;
        return query(q);
    }

    // =============================================================================
    // get_all
    // =============================================================================

    std::vector<Rule> RuleStore::get_all() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Rule> result;
        result.reserve(rules_.size());
        for (const auto& [id, rule] : rules_) {
            result.push_back(rule);
        }
        return result;
    }

    // =============================================================================
    // decay_confidence
    // Called after each inference cycle to age out stale rules
    // =============================================================================

    void RuleStore::decay_confidence() {
        std::lock_guard<std::mutex> lock(mutex_);

        float decay = config_.verifier.rule_confidence_decay;
        int   decayed = 0;

        for (auto& [id, rule] : rules_) {
            float old_conf = rule.confidence;
            rule.confidence = std::max(0.0f, rule.confidence - decay);
            if (rule.confidence != old_conf) ++decayed;
        }

        if (decayed > 0) {
            dirty_ = true;
            LOG_DEBUG("RuleStore: decayed " + std::to_string(decayed) +
                " rules by " + std::to_string(decay));
        }
    }

    // =============================================================================
    // prune
    // =============================================================================

    int RuleStore::prune() {
        std::lock_guard<std::mutex> lock(mutex_);

        float threshold = config_.verifier.min_rule_confidence;
        int   pruned = 0;

        auto it = rules_.begin();
        while (it != rules_.end()) {
            if (it->second.confidence < threshold) {
                LOG_DEBUG("RuleStore: pruning rule [" +
                    it->second.domain + "] conf=" +
                    std::to_string(it->second.confidence));
                it = rules_.erase(it);
                ++pruned;
            }
            else {
                ++it;
            }
        }

        if (pruned > 0) {
            dirty_ = true;
            LOG_INFO("RuleStore: pruned " + std::to_string(pruned) +
                " rules below threshold " + std::to_string(threshold));
        }

        return pruned;
    }

    // =============================================================================
    // enforce_limit
    // =============================================================================

    int RuleStore::enforce_limit() {
        std::lock_guard<std::mutex> lock(mutex_);

        int max_rules = config_.memory.max_rules;
        if (static_cast<int>(rules_.size()) <= max_rules) return 0;

        // Build sorted list by confidence
        std::vector<std::pair<std::string, float>> sorted;
        sorted.reserve(rules_.size());
        for (const auto& [id, rule] : rules_) {
            sorted.push_back({ id, rule.confidence });
        }
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second; // ascending - remove lowest first
            });

        int to_remove = static_cast<int>(rules_.size()) - max_rules;
        int removed = 0;

        for (int i = 0; i < to_remove && i < static_cast<int>(sorted.size()); ++i) {
            rules_.erase(sorted[i].first);
            ++removed;
        }

        if (removed > 0) {
            dirty_ = true;
            LOG_INFO("RuleStore: enforced limit, removed " +
                std::to_string(removed) + " excess rules");
        }

        return removed;
    }

    // =============================================================================
    // stats
    // =============================================================================

    RuleStoreStats RuleStore::stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        RuleStoreStats s{};
        s.total_rules = static_cast<int>(rules_.size());

        const std::string domain_names[6] = {
            "factual", "ethical", "spatial",
            "temporal", "social", "mathematical"
        };

        float conf_sum = 0.0f;
        float trigger_sum = 0.0f;

        for (const auto& [id, rule] : rules_) {
            conf_sum += rule.confidence;
            trigger_sum += static_cast<float>(rule.trigger_count);

            if (rule.confidence >= config_.verifier.min_rule_confidence) {
                ++s.active_rules;
            }
            else {
                ++s.pruned_rules;
            }

            for (int i = 0; i < 6; ++i) {
                if (rule.domain == domain_names[i]) {
                    ++s.rules_by_domain[i];
                    break;
                }
            }
        }

        s.avg_confidence = s.total_rules > 0 ? conf_sum / s.total_rules : 0.0f;
        s.avg_trigger_count = s.total_rules > 0 ? trigger_sum / s.total_rules : 0.0f;

        return s;
    }

    int RuleStore::size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(rules_.size());
    }

    bool RuleStore::empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rules_.empty();
    }

    // =============================================================================
    // Internal helpers
    // =============================================================================

    bool RuleStore::is_similar(const Rule& a, const Rule& b,
        float threshold) const {
        if (a.domain != b.domain) return false;
        float overlap = word_overlap(a.condition, b.condition);
        return overlap >= threshold;
    }

    void RuleStore::merge_rule(Rule& existing, const Rule& incoming) {
        // Boost confidence toward the higher value
        existing.confidence = std::clamp(
            std::max(existing.confidence, incoming.confidence) + 0.05f,
            0.0f, 1.0f);
        existing.updated_at = JsonParser::current_timestamp();

        // If incoming consequence is longer/more detailed, prefer it
        if (incoming.consequence.size() > existing.consequence.size()) {
            existing.consequence = incoming.consequence;
        }
    }

    float RuleStore::word_overlap(const std::string& a,
        const std::string& b) const {
        // Tokenize both strings into word sets
        auto tokenize = [](const std::string& s) {
            std::set<std::string> words;
            std::istringstream iss(s);
            std::string word;
            while (iss >> word) {
                // Lowercase
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                // Strip punctuation
                word.erase(std::remove_if(word.begin(), word.end(),
                    [](char c) { return !std::isalnum(c); }), word.end());
                if (!word.empty()) words.insert(word);
            }
            return words;
            };

        auto words_a = tokenize(a);
        auto words_b = tokenize(b);

        if (words_a.empty() || words_b.empty()) return 0.0f;

        // Intersection size
        int intersection = 0;
        for (const auto& w : words_a) {
            if (words_b.count(w)) ++intersection;
        }

        // Jaccard similarity
        int union_size = static_cast<int>(words_a.size() + words_b.size()) -
            intersection;
        return union_size > 0
            ? static_cast<float>(intersection) / static_cast<float>(union_size)
            : 0.0f;
    }

} // namespace cardinal