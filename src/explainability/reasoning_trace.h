#pragma once
// =============================================================================
// Cardinal - Reasoning Trace
// File: src/explainability/reasoning_trace.h
//
// Complete audit record for a single inference cycle (chat or agent step).
// Every inference produces one ReasoningTrace regardless of backend or mode.
//
// The trace captures:
//   - Input context (query, rules active, episodes retrieved)
//   - Pass 1 feeling output
//   - All tool calls and their results
//   - Symbolic verification results
//   - Rule extraction outcome
//   - Pass 2 final response
//   - Full timing breakdown
//   - Ed25519 cryptographic signature for tamper-evidence
//
// Stored in the audit log and optionally attached to ChatResponse.
// Exportable as JSON via ExplainabilityExporter.
// =============================================================================

#include "core/feeling_output.h"
#include "tools/tool_result.h"

#include <string>
#include <vector>
#include <optional>
#include <chrono>

namespace cardinal {

    // -------------------------------------------------------------------------
    // SymbolicCheckRecord
    // What the symbolic engine did during this inference.
    // -------------------------------------------------------------------------
    struct SymbolicCheckRecord {
        bool                     ran               = false;
        int                      contradictions_found = 0;
        int                      contradictions_resolved = 0;
        int                      contradictions_flagged  = 0;
        std::vector<std::string> rules_fired;       // rule IDs that matched
        std::vector<std::string> contradictions;    // description of each
        int                      duration_ms       = 0;
    };

    // -------------------------------------------------------------------------
    // EpisodeReference
    // Lightweight reference to an episode retrieved during this inference.
    // -------------------------------------------------------------------------
    struct EpisodeReference {
        std::string id;
        std::string user_message_preview;   // first 100 chars
        std::string reasoning_domain;
        float       confidence   = 0.0f;
        float       retrieval_score = 0.0f;
    };

    // -------------------------------------------------------------------------
    // RuleReference
    // A rule that was active (injected into context) during this inference.
    // -------------------------------------------------------------------------
    struct RuleReference {
        std::string id;
        std::string domain;
        std::string condition;
        std::string consequence;
        float       confidence = 0.0f;
    };

    // -------------------------------------------------------------------------
    // AgentStepRecord
    // One step of an agentic execution. A ReasoningTrace for an agent task
    // contains one AgentStepRecord per iteration.
    // -------------------------------------------------------------------------
    struct AgentStepRecord {
        int         step_index   = 0;
        std::string description;             // what this step was trying to do
        std::string action_taken;            // what the model decided to do
        bool        tool_called  = false;
        std::string tool_name;
        std::string tool_input_json;
        std::string tool_output_preview;     // first 300 chars
        bool        succeeded    = false;
        std::string failure_reason;
        int         duration_ms  = 0;
    };

    // -------------------------------------------------------------------------
    // TraceIntegrity
    // Cryptographic integrity record.
    // -------------------------------------------------------------------------
    struct TraceIntegrity {
        std::string sha256_hash;    // SHA256 of canonical JSON (always present)
        std::string signature;      // Ed25519 signature (base64, if signing enabled)
        std::string public_key_id;  // fingerprint of signing key
        bool        signed_  = false;
        std::string signed_at;
    };

    // -------------------------------------------------------------------------
    // ReasoningTrace
    // Complete audit record. One per inference cycle.
    // -------------------------------------------------------------------------
    struct ReasoningTrace {
        // ------------------------------------------------------------------
        // Identity
        // ------------------------------------------------------------------
        std::string inference_id;       // UUID, unique per inference
        std::string session_id;
        std::string episode_id;         // set after storage
        std::string timestamp;
        std::string backend_type;       // "llama_cpp" | "tensorrt"
        std::string model_name;
        bool        agent_mode = false;

        // ------------------------------------------------------------------
        // Input
        // ------------------------------------------------------------------
        std::string              query;              // user message / agent goal
        std::vector<RuleReference>   active_rules;
        std::vector<EpisodeReference> retrieved_episodes;

        // ------------------------------------------------------------------
        // Pass 1 — Feeling output
        // ------------------------------------------------------------------
        FeelingOutput            feeling;
        bool                     feeling_valid    = false;
        int                      pass1_retries    = 0;
        int                      pass1_tokens     = 0;
        int                      pass1_duration_ms = 0;

        // ------------------------------------------------------------------
        // Tool calls (may be empty for non-tool inferences)
        // ------------------------------------------------------------------
        std::vector<ToolResult>  tool_calls;
        int                      tool_iterations  = 0;  // how many loop iterations

        // ------------------------------------------------------------------
        // Symbolic verification
        // ------------------------------------------------------------------
        SymbolicCheckRecord      symbolic_check;

        // ------------------------------------------------------------------
        // Rule extraction
        // ------------------------------------------------------------------
        bool                     rule_committed   = false;
        std::string              committed_rule_id;
        std::optional<RuleReference> committed_rule;

        // ------------------------------------------------------------------
        // Agent steps (only set when agent_mode = true)
        // ------------------------------------------------------------------
        std::vector<AgentStepRecord> agent_steps;
        std::string              agent_goal;
        bool                     agent_goal_achieved = false;
        int                      agent_iterations    = 0;

        // ------------------------------------------------------------------
        // Pass 2 — Final response
        // ------------------------------------------------------------------
        std::string              final_response;
        int                      pass2_tokens     = 0;
        int                      pass2_duration_ms = 0;

        // ------------------------------------------------------------------
        // Totals
        // ------------------------------------------------------------------
        int                      total_tokens     = 0;
        int                      total_duration_ms = 0;

        // ------------------------------------------------------------------
        // Integrity
        // ------------------------------------------------------------------
        TraceIntegrity           integrity;

        // ------------------------------------------------------------------
        // Convenience
        // ------------------------------------------------------------------
        bool has_tool_calls()   const { return !tool_calls.empty(); }
        bool has_agent_steps()  const { return !agent_steps.empty(); }
        bool is_signed()        const { return integrity.signed_; }

        // Returns a one-line summary for logging
        std::string summary() const {
            std::string s = "inference_id=" + inference_id +
                " tokens=" + std::to_string(total_tokens) +
                " duration=" + std::to_string(total_duration_ms) + "ms" +
                " tools=" + std::to_string(tool_calls.size()) +
                " rule_committed=" + (rule_committed ? "yes" : "no") +
                " signed=" + (is_signed() ? "yes" : "no");
            return s;
        }
    };

} // namespace cardinal
