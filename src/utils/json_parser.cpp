// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - JSON Parser Implementation
// File: src/utils/json_parser.cpp
// =============================================================================

#include "json_parser.h"
#include "logger.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <atomic>
#include <iomanip>

using json = nlohmann::json;

namespace cardinal {

    // =============================================================================
    // FeelingOutput methods
    // =============================================================================

    std::string FeelingOutput::to_string() const {
        std::ostringstream oss;
        oss << "FeelingOutput{"
            << "confidence=" << confidence
            << ", reasoning_type=" << reasoning_type
            << ", reasoning_domain=" << reasoning_domain
            << ", uncertainty=" << (uncertainty_flag ? "true" : "false")
            << ", rule_candidate=" << (rule_candidate_signal ? "true" : "false")
            << ", contradiction=" << (contradiction_flag ? "true" : "false")
            << "}";
        return oss.str();
    }

    // =============================================================================
    // Feeling output parsing
    // =============================================================================

    FeelingOutput JsonParser::parse_feeling_output(const std::string& json_str) {
        if (json_str.empty()) {
            throw ParseError("Empty JSON string for feeling output");
        }

        json j;
        try {
            j = json::parse(json_str);
        }
        catch (const json::parse_error& e) {
            throw ParseError("Failed to parse feeling output JSON: " +
                std::string(e.what()) +
                "\nRaw input: " + json_str);
        }

        FeelingOutput feeling;

        // confidence
        if (!j.contains("confidence") || !j["confidence"].is_number()) {
            throw ParseError("Missing or invalid 'confidence' field in feeling output");
        }
        feeling.confidence = j["confidence"].get<float>();
        if (feeling.confidence < 0.0f || feeling.confidence > 1.0f) {
            throw ParseError("'confidence' must be between 0.0 and 1.0, got: " +
                std::to_string(feeling.confidence));
        }

        // reasoning_type
        if (!j.contains("reasoning_type") || !j["reasoning_type"].is_string()) {
            throw ParseError("Missing or invalid 'reasoning_type' field in feeling output");
        }
        feeling.reasoning_type = j["reasoning_type"].get<std::string>();

        static const std::vector<std::string> valid_reasoning_types = {
            "analogical", "causal", "deductive",
            "inductive", "abductive", "associative"
        };
        bool valid_rt = false;
        for (const auto& t : valid_reasoning_types) {
            if (feeling.reasoning_type == t) { valid_rt = true; break; }
        }
        if (!valid_rt) {
            throw ParseError("Invalid reasoning_type: '" + feeling.reasoning_type +
                "'. Must be one of: analogical, causal, deductive, "
                "inductive, abductive, associative");
        }

        // uncertainty_flag
        if (!j.contains("uncertainty_flag") || !j["uncertainty_flag"].is_boolean()) {
            throw ParseError("Missing or invalid 'uncertainty_flag' field in feeling output");
        }
        feeling.uncertainty_flag = j["uncertainty_flag"].get<bool>();

        // rule_candidate_signal
        if (!j.contains("rule_candidate_signal") || !j["rule_candidate_signal"].is_boolean()) {
            throw ParseError("Missing or invalid 'rule_candidate_signal' field in feeling output");
        }
        feeling.rule_candidate_signal = j["rule_candidate_signal"].get<bool>();

        // contradiction_flag
        if (!j.contains("contradiction_flag") || !j["contradiction_flag"].is_boolean()) {
            throw ParseError("Missing or invalid 'contradiction_flag' field in feeling output");
        }
        feeling.contradiction_flag = j["contradiction_flag"].get<bool>();

        // reasoning_domain
        if (!j.contains("reasoning_domain") || !j["reasoning_domain"].is_string()) {
            throw ParseError("Missing or invalid 'reasoning_domain' field in feeling output");
        }
        feeling.reasoning_domain = j["reasoning_domain"].get<std::string>();

        static const std::vector<std::string> valid_domains = {
            "factual", "ethical", "spatial",
            "temporal", "social", "mathematical"
        };
        bool valid_domain = false;
        for (const auto& d : valid_domains) {
            if (feeling.reasoning_domain == d) { valid_domain = true; break; }
        }
        if (!valid_domain) {
            throw ParseError("Invalid reasoning_domain: '" + feeling.reasoning_domain +
                "'. Must be one of: factual, ethical, spatial, "
                "temporal, social, mathematical");
        }

        LOG_DEBUG("Parsed feeling output: " + feeling.to_string());
        return feeling;
    }

    // -----------------------------------------------------------------------------
    // validate_feeling_output
    // Checks logical consistency beyond field-level validation
    // -----------------------------------------------------------------------------
    bool JsonParser::validate_feeling_output(const FeelingOutput& feeling,
        std::string& error_msg) {
        // High confidence + uncertainty flag is contradictory
        if (feeling.confidence > 0.8f && feeling.uncertainty_flag) {
            error_msg = "Contradictory feeling: high confidence (" +
                std::to_string(feeling.confidence) +
                ") with uncertainty_flag=true";
            return false;
        }

        // Low confidence should usually have uncertainty flag set
        // This is a warning, not a hard failure - log it but don't reject
        if (feeling.confidence < 0.3f && !feeling.uncertainty_flag) {
            LOG_WARN("Low confidence (" + std::to_string(feeling.confidence) +
                ") without uncertainty_flag - possible model inconsistency");
        }

        return true;
    }

    // -----------------------------------------------------------------------------
    // serialize_feeling_output
    // Used to inject feeling output as synthetic assistant turn in Pass 2
    // -----------------------------------------------------------------------------
    std::string JsonParser::serialize_feeling_output(const FeelingOutput& feeling) {
        json j;
        j["confidence"] = feeling.confidence;
        j["reasoning_type"] = feeling.reasoning_type;
        j["uncertainty_flag"] = feeling.uncertainty_flag;
        j["rule_candidate_signal"] = feeling.rule_candidate_signal;
        j["contradiction_flag"] = feeling.contradiction_flag;
        j["reasoning_domain"] = feeling.reasoning_domain;
        return j.dump();
    }

    // =============================================================================
    // Rule store
    // =============================================================================

    std::vector<Rule> JsonParser::load_rules(const std::string& path) {
        std::vector<Rule> rules;

        // If file doesn't exist yet, return empty set (first run)
        if (!std::filesystem::exists(path)) {
            LOG_INFO("Rule store not found at '" + path + "' - starting fresh");
            return rules;
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            throw ParseError("Failed to open rule store: " + path);
        }

        json j;
        try {
            file >> j;
        }
        catch (const json::parse_error& e) {
            throw ParseError("Failed to parse rule store JSON: " + std::string(e.what()));
        }

        if (!j.is_array()) {
            throw ParseError("Rule store must be a JSON array, got: " +
                std::string(j.type_name()));
        }

        for (const auto& item : j) {
            try {
                rules.push_back(rule_from_json(item));
            }
            catch (const ParseError& e) {
                // Log and skip corrupted rules rather than failing entirely
                LOG_WARN("Skipping corrupted rule entry: " + std::string(e.what()));
            }
        }

        LOG_INFO("Loaded " + std::to_string(rules.size()) + " rules from: " + path);
        return rules;
    }

    void JsonParser::save_rules(const std::string& path,
        const std::vector<Rule>& rules) {
        json j = json::array();
        for (const auto& rule : rules) {
            j.push_back(rule_to_json(rule));
        }

        // Ensure parent directory exists
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        atomic_write(path, j.dump(2));
        LOG_DEBUG("Saved " + std::to_string(rules.size()) + " rules to: " + path);
    }

    // Phase 6: added episode_id and reasoning_type provenance fields
    json JsonParser::rule_to_json(const Rule& rule) {
        json j;
        j["id"] = rule.id;
        j["domain"] = rule.domain;
        j["condition"] = rule.condition;
        j["consequence"] = rule.consequence;
        j["confidence"] = rule.confidence;
        j["trigger_count"] = rule.trigger_count;
        j["created_at"] = rule.created_at;
        j["updated_at"] = rule.updated_at;
        // Provenance -- always written, empty string for legacy rules
        j["episode_id"] = rule.episode_id;
        j["reasoning_type"] = rule.reasoning_type;
        return j;
    }

    // Phase 6: reads provenance fields with safe empty-string defaults
    // so that existing rules.json loads without error
    Rule JsonParser::rule_from_json(const json& j) {
        Rule rule;
        rule.id = get_string(j, "id");
        rule.domain = get_string(j, "domain");
        rule.condition = get_string(j, "condition");
        rule.consequence = get_string(j, "consequence");
        rule.confidence = get_float(j, "confidence", 0.5f);
        rule.trigger_count = j.contains("trigger_count") ?
            j["trigger_count"].get<int>() : 0;
        rule.created_at = get_string(j, "created_at");
        rule.updated_at = get_string(j, "updated_at");

        // Provenance -- safe defaults for rules that predate Phase 6
        rule.episode_id = get_string(j, "episode_id", "");
        rule.reasoning_type = get_string(j, "reasoning_type", "");

        if (rule.id.empty())
            throw ParseError("Rule missing 'id' field");
        if (rule.condition.empty())
            throw ParseError("Rule '" + rule.id + "' missing 'condition' field");
        if (rule.consequence.empty())
            throw ParseError("Rule '" + rule.id + "' missing 'consequence' field");

        return rule;
    }

    // =============================================================================
    // Knowledge graph
    // =============================================================================

    std::vector<KnowledgeNode> JsonParser::load_knowledge(const std::string& path) {
        std::vector<KnowledgeNode> nodes;

        if (!std::filesystem::exists(path)) {
            LOG_INFO("Knowledge graph not found at '" + path + "' - starting fresh");
            return nodes;
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            throw ParseError("Failed to open knowledge graph: " + path);
        }

        json j;
        try {
            file >> j;
        }
        catch (const json::parse_error& e) {
            throw ParseError("Failed to parse knowledge graph JSON: " +
                std::string(e.what()));
        }

        if (!j.is_array()) {
            throw ParseError("Knowledge graph must be a JSON array");
        }

        for (const auto& item : j) {
            try {
                nodes.push_back(node_from_json(item));
            }
            catch (const ParseError& e) {
                LOG_WARN("Skipping corrupted knowledge node: " + std::string(e.what()));
            }
        }

        LOG_INFO("Loaded " + std::to_string(nodes.size()) +
            " knowledge nodes from: " + path);
        return nodes;
    }

    void JsonParser::save_knowledge(const std::string& path,
        const std::vector<KnowledgeNode>& nodes) {
        json j = json::array();
        for (const auto& node : nodes) {
            j.push_back(node_to_json(node));
        }

        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        atomic_write(path, j.dump(2));
        LOG_DEBUG("Saved " + std::to_string(nodes.size()) +
            " knowledge nodes to: " + path);
    }

    json JsonParser::node_to_json(const KnowledgeNode& node) {
        json j;
        j["id"] = node.id;
        j["label"] = node.label;
        j["type"] = node.type;
        j["content"] = node.content;
        j["related_ids"] = node.related_ids;
        j["confidence"] = node.confidence;
        j["source"] = node.source;
        j["created_at"] = node.created_at;
        j["updated_at"] = node.updated_at;
        return j;
    }

    KnowledgeNode JsonParser::node_from_json(const json& j) {
        KnowledgeNode node;
        node.id = get_string(j, "id");
        node.label = get_string(j, "label");
        node.type = get_string(j, "type");
        node.content = get_string(j, "content");
        node.confidence = get_float(j, "confidence", 0.5f);
        node.source = get_string(j, "source");
        node.created_at = get_string(j, "created_at");
        node.updated_at = get_string(j, "updated_at");

        if (j.contains("related_ids") && j["related_ids"].is_array()) {
            for (const auto& id : j["related_ids"]) {
                node.related_ids.push_back(id.get<std::string>());
            }
        }

        if (node.id.empty())
            throw ParseError("Knowledge node missing 'id' field");

        return node;
    }

    // =============================================================================
    // General utilities
    // =============================================================================

    std::optional<json> JsonParser::try_parse(const std::string& json_str) {
        try {
            return json::parse(json_str);
        }
        catch (...) {
            return std::nullopt;
        }
    }

    std::string JsonParser::pretty_print(const std::string& json_str, int indent) {
        auto j = try_parse(json_str);
        if (!j) return json_str; // Return as-is if unparseable
        return j->dump(indent);
    }

    bool JsonParser::is_valid_json(const std::string& json_str) {
        return try_parse(json_str).has_value();
    }

    std::string JsonParser::get_string(const json& j,
        const std::string& key,
        const std::string& default_val) {
        if (!j.contains(key) || !j[key].is_string()) return default_val;
        return j[key].get<std::string>();
    }

    float JsonParser::get_float(const json& j,
        const std::string& key,
        float default_val) {
        if (!j.contains(key) || !j[key].is_number()) return default_val;
        return j[key].get<float>();
    }

    bool JsonParser::get_bool(const json& j,
        const std::string& key,
        bool default_val) {
        if (!j.contains(key) || !j[key].is_boolean()) return default_val;
        return j[key].get<bool>();
    }

    // -----------------------------------------------------------------------------
    // generate_id
    // Simple monotonic ID: timestamp_ms + atomic counter
    // Not UUID but unique enough for a local rule/node store
    // -----------------------------------------------------------------------------
    std::string JsonParser::generate_id() {
        static std::atomic<uint32_t> counter{ 0 };

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        uint32_t count = counter.fetch_add(1, std::memory_order_relaxed);

        std::ostringstream oss;
        oss << std::hex << now_ms << "_" << std::setw(4) << std::setfill('0') << count;
        return oss.str();
    }

    // -----------------------------------------------------------------------------
    // current_timestamp - ISO 8601
    // -----------------------------------------------------------------------------
    std::string JsonParser::current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::tm tm_info{};
#ifdef _WIN32
        localtime_s(&tm_info, &time_t);
#else
        localtime_r(&time_t, &tm_info);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_info, "%Y-%m-%dT%H:%M:%S");
        return oss.str();
    }

    // -----------------------------------------------------------------------------
    // atomic_write
    // Writes to a temp file then renames - prevents corruption on crash mid-write
    // -----------------------------------------------------------------------------
    void JsonParser::atomic_write(const std::string& path,
        const std::string& content) {
        std::string tmp_path = path + ".tmp";

        std::ofstream file(tmp_path, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            throw ParseError("Failed to open temp file for atomic write: " + tmp_path);
        }

        file << content;
        file.flush();
        file.close();

        // Atomic rename
        std::filesystem::rename(tmp_path, path);
    }

} // namespace cardinal