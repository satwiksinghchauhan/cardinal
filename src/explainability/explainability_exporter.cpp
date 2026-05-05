// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Explainability Exporter Implementation
// File: src/explainability/explainability_exporter.cpp
// =============================================================================

#include "explainability/explainability_exporter.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <ctime>

using json = nlohmann::json;

namespace cardinal {

    ExplainabilityExporter::ExplainabilityExporter(const CardinalConfig& config)
        : config_(config)
    {}

    // =========================================================================
    // trace_to_json (static — canonical, used for hashing)
    // =========================================================================

    std::string ExplainabilityExporter::trace_to_json(const ReasoningTrace& t) {
        json j;

        // Identity
        j["inference_id"]  = t.inference_id;
        j["session_id"]    = t.session_id;
        j["episode_id"]    = t.episode_id;
        j["timestamp"]     = t.timestamp;
        j["backend_type"]  = t.backend_type;
        j["model_name"]    = t.model_name;
        j["agent_mode"]    = t.agent_mode;

        // Input
        j["query"]         = t.query;

        json active_rules = json::array();
        for (const auto& r : t.active_rules) {
            json rj;
            rj["id"]          = r.id;
            rj["domain"]      = r.domain;
            rj["condition"]   = r.condition;
            rj["consequence"] = r.consequence;
            rj["confidence"]  = r.confidence;
            active_rules.push_back(rj);
        }
        j["active_rules"] = active_rules;

        json episodes = json::array();
        for (const auto& ep : t.retrieved_episodes) {
            json ej;
            ej["id"]                   = ep.id;
            ej["user_message_preview"] = ep.user_message_preview;
            ej["reasoning_domain"]     = ep.reasoning_domain;
            ej["confidence"]           = ep.confidence;
            ej["retrieval_score"]      = ep.retrieval_score;
            episodes.push_back(ej);
        }
        j["retrieved_episodes"] = episodes;

        // Pass 1
        json p1;
        p1["valid"]       = t.feeling_valid;
        p1["retries"]     = t.pass1_retries;
        p1["tokens"]      = t.pass1_tokens;
        p1["duration_ms"] = t.pass1_duration_ms;
        if (t.feeling_valid) {
            json f;
            f["confidence"]            = t.feeling.confidence;
            f["reasoning_type"]        = t.feeling.reasoning_type;
            f["reasoning_domain"]      = t.feeling.reasoning_domain;
            f["uncertainty_flag"]      = t.feeling.uncertainty_flag;
            f["contradiction_flag"]    = t.feeling.contradiction_flag;
            f["rule_candidate_signal"] = t.feeling.rule_candidate_signal;
            p1["feeling"] = f;
        }
        j["pass1"] = p1;

        // Tool calls
        json tools = json::array();
        for (const auto& tr : t.tool_calls) {
            json tj;
            tj["tool_name"]     = tr.tool_name;
            tj["status"]        = tool_status_to_string(tr.status);
            tj["duration_ms"]   = tr.duration_ms;
            tj["timestamp"]     = tr.timestamp;
            if (tr.ok()) {
                // Truncate large outputs for JSON export
                std::string out = tr.output;
                if (out.size() > 2000)
                    out = out.substr(0, 2000) + "...[truncated]";
                tj["output"] = out;
            } else {
                tj["error"] = tr.error_message;
            }
            // Include arguments
            json args;
            for (const auto& [k, v] : tr.call.arguments)
                args[k] = v;
            tj["arguments"] = args;
            tools.push_back(tj);
        }
        j["tool_calls"]      = tools;
        j["tool_iterations"] = t.tool_iterations;

        // Symbolic check
        json sc;
        sc["ran"]                      = t.symbolic_check.ran;
        sc["contradictions_found"]     = t.symbolic_check.contradictions_found;
        sc["contradictions_resolved"]  = t.symbolic_check.contradictions_resolved;
        sc["contradictions_flagged"]   = t.symbolic_check.contradictions_flagged;
        sc["rules_fired"]              = t.symbolic_check.rules_fired;
        sc["contradictions"]           = t.symbolic_check.contradictions;
        sc["duration_ms"]              = t.symbolic_check.duration_ms;
        j["symbolic_check"] = sc;

        // Rule extraction
        json re;
        re["committed"] = t.rule_committed;
        if (t.rule_committed) {
            re["rule_id"] = t.committed_rule_id;
            if (t.committed_rule) {
                json rj;
                rj["id"]          = t.committed_rule->id;
                rj["domain"]      = t.committed_rule->domain;
                rj["condition"]   = t.committed_rule->condition;
                rj["consequence"] = t.committed_rule->consequence;
                rj["confidence"]  = t.committed_rule->confidence;
                re["rule"] = rj;
            }
        }
        j["rule_extraction"] = re;

        // Agent steps
        if (t.agent_mode) {
            j["agent_goal"]          = t.agent_goal;
            j["agent_goal_achieved"] = t.agent_goal_achieved;
            j["agent_iterations"]    = t.agent_iterations;

            json steps = json::array();
            for (const auto& s : t.agent_steps) {
                json sj;
                sj["step_index"]        = s.step_index;
                sj["description"]       = s.description;
                sj["action_taken"]      = s.action_taken;
                sj["tool_called"]       = s.tool_called;
                if (s.tool_called) {
                    sj["tool_name"]           = s.tool_name;
                    sj["tool_input_preview"]  = s.tool_input_json.substr(
                        0, std::min(s.tool_input_json.size(), size_t(300)));
                    sj["tool_output_preview"] = s.tool_output_preview;
                }
                sj["succeeded"]     = s.succeeded;
                sj["duration_ms"]   = s.duration_ms;
                if (!s.failure_reason.empty())
                    sj["failure_reason"] = s.failure_reason;
                steps.push_back(sj);
            }
            j["agent_steps"] = steps;
        }

        // Pass 2
        json p2;
        p2["tokens"]      = t.pass2_tokens;
        p2["duration_ms"] = t.pass2_duration_ms;
        j["pass2"] = p2;

        // Response (truncated for canonical hash, full in pretty export)
        j["final_response"] = t.final_response.substr(
            0, std::min(t.final_response.size(), size_t(500)));

        // Totals
        j["total_tokens"]      = t.total_tokens;
        j["total_duration_ms"] = t.total_duration_ms;

        // Integrity
        json integrity;
        integrity["sha256_hash"]   = t.integrity.sha256_hash;
        integrity["signed"]        = t.integrity.signed_;
        if (t.integrity.signed_) {
            integrity["signature"]     = t.integrity.signature;
            integrity["public_key_id"] = t.integrity.public_key_id;
            integrity["signed_at"]     = t.integrity.signed_at;
        }
        j["integrity"] = integrity;

        // Compact, sorted keys for deterministic hashing
        return j.dump();
    }

    // =========================================================================
    // export_json (pretty-printed for human consumption)
    // =========================================================================

    std::string ExplainabilityExporter::export_json(
        const ReasoningTrace& trace) const
    {
        // Parse the canonical JSON and re-dump with pretty printing
        json j = json::parse(trace_to_json(trace));

        // Add full response (canonical version truncates it)
        j["final_response"] = trace.final_response;

        return j.dump(2);
    }

    // =========================================================================
    // export_to_file
    // =========================================================================

    std::string ExplainabilityExporter::export_to_file(
        const ReasoningTrace& trace,
        const std::string& filename) const
    {
        auto export_dir = config_.explainability.export_path;
        std::filesystem::create_directories(export_dir);

        std::string fname = filename;
        if (fname.empty()) {
            // Generate filename: inference_<id>_<timestamp>.json
            std::string ts = trace.timestamp;
            // Replace colons and spaces for safe filename
            std::replace(ts.begin(), ts.end(), ':', '-');
            std::replace(ts.begin(), ts.end(), ' ', '_');
            fname = "inference_" + trace.inference_id.substr(0, 8) +
                    "_" + ts + ".json";
        }

        std::string path = export_dir + "/" + fname;

        std::ofstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error(
                "ExplainabilityExporter: cannot write to: " + path);
        }

        f << export_json(trace);

        LOG_INFO("ExplainabilityExporter: exported trace to " + path);
        return path;
    }

} // namespace cardinal
