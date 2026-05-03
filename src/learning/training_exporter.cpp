// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Training Exporter Implementation
// File: src/learning/training_exporter.cpp
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "learning/training_exporter.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <numeric>

using json = nlohmann::json;

namespace cardinal {

    // =========================================================================
    // Constructor
    // =========================================================================

    TrainingExporter::TrainingExporter(const CardinalConfig& config,
        EpisodicStorage& storage,
        RuleStore& rule_store)
        : config_(config)
        , storage_(storage)
        , rule_store_(rule_store)
    {
        LOG_INFO("TrainingExporter initialized");
    }

    // =========================================================================
    // export_to_file
    // =========================================================================

    ExportStats TrainingExporter::export_to_file(const std::string& output_path,
        const ExportFilter& filter)
    {
        // Collect examples
        auto examples = collect(filter);

        // Ensure parent directory exists
        std::filesystem::path p(output_path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        // Write JSONL -- one example per line, no trailing newline issues
        std::ofstream file(output_path, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            throw TrainingExporterError(
                "Failed to open output file: " + output_path);
        }

        int episodes_written = 0;
        int rules_written = 0;
        float conf_sum = 0.0f;

        for (const auto& ex : examples) {
            file << to_jsonl_line(ex) << "\n";

            if (ex.source == "episode") {
                ++episodes_written;
                conf_sum += ex.confidence;
            }
            else {
                ++rules_written;
            }
        }

        file.flush();
        file.close();

        // Build stats
        ExportStats stats;
        stats.episodes_exported = episodes_written;
        stats.rules_exported = rules_written;
        stats.total_exported = episodes_written + rules_written;
        stats.total_episodes_checked = storage_.count();
        stats.avg_confidence = episodes_written > 0
            ? conf_sum / static_cast<float>(episodes_written)
            : 0.0f;
        stats.output_path = output_path;
        stats.timestamp = JsonParser::current_timestamp();

        LOG_INFO("TrainingExporter: exported " +
            std::to_string(stats.total_exported) +
            " examples to " + output_path +
            " (episodes=" + std::to_string(episodes_written) +
            " rules=" + std::to_string(rules_written) +
            " avg_conf=" + std::to_string(stats.avg_confidence) + ")");

        return stats;
    }

    // =========================================================================
    // collect
    // =========================================================================

    std::vector<TrainingExample>
        TrainingExporter::collect(const ExportFilter& filter) const
    {
        std::vector<TrainingExample> examples;

        // -- Episodes --
        EpisodeQuery q;
        q.min_confidence = filter.min_confidence;
        q.domain = filter.domain;
        q.recent_first = filter.recent_first;
        // Fetch generously -- we'll cap after combining with rules
        q.max_results = (filter.max_examples > 0)
            ? filter.max_examples * 2
            : 100000;

        auto episodes = storage_.query(q);

        examples.reserve(episodes.size());
        for (const auto& ep : episodes) {
            // Skip episodes with empty response -- nothing useful to train on
            if (ep.response_summary.empty()) continue;
            // Skip episodes with very short responses -- likely errors
            if (ep.response_summary.size() < 20) continue;

            examples.push_back(episode_to_example(ep));
        }

        // -- Rules (optional) --
        if (filter.include_rules) {
            RuleQuery rq;
            rq.domain = filter.domain;
            rq.min_confidence = filter.min_confidence;
            rq.active_only = true;

            auto rules = rule_store_.query(rq);
            for (const auto& rule : rules) {
                examples.push_back(rule_to_example(rule));
            }
        }

        // -- Apply max_examples cap --
        if (filter.max_examples > 0 &&
            static_cast<int>(examples.size()) > filter.max_examples) {
            examples.resize(filter.max_examples);
        }

        return examples;
    }

    // =========================================================================
    // Stats queries
    // =========================================================================

    int TrainingExporter::available_episode_count(float min_confidence) const {
        EpisodeQuery q;
        q.min_confidence = min_confidence;
        q.max_results = 100000;
        return static_cast<int>(storage_.query(q).size());
    }

    int TrainingExporter::available_rule_count() const {
        return rule_store_.size();
    }

    ExportStats TrainingExporter::dry_run(const ExportFilter& filter) const {
        auto examples = collect(filter);

        int episodes = 0;
        int rules = 0;
        float conf_sum = 0.0f;

        for (const auto& ex : examples) {
            if (ex.source == "episode") {
                ++episodes;
                conf_sum += ex.confidence;
            }
            else {
                ++rules;
            }
        }

        ExportStats stats;
        stats.episodes_exported = episodes;
        stats.rules_exported = rules;
        stats.total_exported = episodes + rules;
        stats.total_episodes_checked = storage_.count();
        stats.avg_confidence = episodes > 0
            ? conf_sum / static_cast<float>(episodes)
            : 0.0f;
        stats.output_path = "(dry run)";
        stats.timestamp = JsonParser::current_timestamp();

        return stats;
    }

    // =========================================================================
    // Conversion helpers
    // =========================================================================

    TrainingExample
        TrainingExporter::episode_to_example(const EpisodeRecord& ep)
    {
        TrainingExample ex;
        ex.instruction = ep.user_message;
        ex.input = "";
        ex.output = clean_response(ep.response_summary);
        ex.episode_id = ep.id;
        ex.domain = ep.reasoning_domain;
        ex.reasoning_type = ep.reasoning_type;
        ex.confidence = ep.confidence;
        ex.timestamp = ep.timestamp;
        ex.source = "episode";
        return ex;
    }

    TrainingExample
        TrainingExporter::rule_to_example(const Rule& rule)
    {
        TrainingExample ex;
        // Frame the rule as a question-answer pair so the model learns
        // to apply the rule when queried about its condition
        ex.instruction = "What rule applies when: " + rule.condition + "?";
        ex.input = "";
        ex.output = rule.consequence;
        ex.episode_id = rule.episode_id;
        ex.domain = rule.domain;
        ex.reasoning_type = rule.reasoning_type;
        ex.confidence = rule.confidence;
        ex.timestamp = rule.created_at;
        ex.source = "rule";
        return ex;
    }

    std::string TrainingExporter::to_jsonl_line(const TrainingExample& example) {
        json j;
        j["instruction"] = example.instruction;
        j["input"] = example.input;
        j["output"] = example.output;
        // Compact -- one line, no pretty print
        return j.dump();
    }

    // =========================================================================
    // clean_response
    // Strips artifacts from response_summary before writing to training data.
    // Removes:
    //   <think>...</think> blocks (Qwen3 chain-of-thought)
    //   <feeling_state>...</feeling_state> blocks (Cardinal synthetic turn)
    //   Leading/trailing whitespace
    // =========================================================================

    std::string TrainingExporter::clean_response(const std::string& text) {
        std::string result = text;

        // Strip <think>...</think> blocks
        // These are Qwen3 internal reasoning traces -- not useful for training
        auto strip_tag = [&](const std::string& open_tag,
            const std::string& close_tag) {
                while (true) {
                    size_t start = result.find(open_tag);
                    if (start == std::string::npos) break;
                    size_t end = result.find(close_tag, start);
                    if (end == std::string::npos) {
                        // Unclosed tag -- strip from start to end of string
                        result = result.substr(0, start);
                        break;
                    }
                    result.erase(start, end + close_tag.size() - start);
                }
            };

        strip_tag("<think>", "</think>");
        strip_tag("<feeling_state>", "</feeling_state>");

        // Normalize whitespace -- collapse multiple newlines to two max
        std::string normalized;
        normalized.reserve(result.size());
        int consecutive_newlines = 0;

        for (char c : result) {
            if (c == '\n') {
                ++consecutive_newlines;
                if (consecutive_newlines <= 2) normalized += c;
            }
            else {
                consecutive_newlines = 0;
                normalized += c;
            }
        }

        // Trim leading and trailing whitespace
        size_t start = normalized.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = normalized.find_last_not_of(" \t\r\n");
        return normalized.substr(start, end - start + 1);
    }

} // namespace cardinal