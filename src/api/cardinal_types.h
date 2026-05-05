// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - API Types
// File: src/api/cardinal_types.h
//
// All types that cross the API boundary.
// Designed for:
//   - Direct C++ use (Interface 1, Interface 2 via HTTP)
//   - Python bindings via pybind11 (Interface 3, SEAL)
//   - TypeScript via HTTP/JSON (Interface 2 Agent)
//
// Constraints (pybind11 compatibility):
//   - All structs are copyable -- no unique_ptr, no move-only members
//   - All enums have explicit underlying type int
//   - All strings are std::string
//   - All collections are std::vector
//   - No raw pointers in any public type
//   - Every struct has a default constructor with sensible defaults
//
// No core types (FeelingOutput, Rule, EpisodeRecord, etc.) appear here.
// The API layer translates between core types and these API types.
// =============================================================================

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include "agent/agent_types.h"

namespace cardinal {

    // =========================================================================
    // CardinalStatus
    // Every API operation returns one of these.
    // Interface code checks status before reading value.
    // =========================================================================
    enum class CardinalStatus : int {
        OK = 0,
        NOT_INITIALIZED = 1,   // API not yet init()'d
        ALREADY_INITIALIZED = 2,   // init() called twice
        INFERENCE_FAILED = 3,   // LLM inference failed
        STORAGE_ERROR = 4,   // SQLite or file I/O error
        CONFIG_ERROR = 5,   // Config load or validation failed
        INVALID_INPUT = 6,   // Empty or malformed input
        EXPORT_FAILED = 7,   // Training export write failed
        SESSION_NOT_FOUND = 8,   // Session ID does not exist
        AUTH_FAILED = 9,   // Invalid or missing API key
        TIMEOUT = 10,  // Operation timed out
        SHUTDOWN = 11,  // API is shutting down
        NOT_FOUND = 12, // Requested resource not found
        INTERNAL_ERROR = 99   // Unexpected internal error
    };

    // Human-readable status string -- useful for logging and error messages
    inline const char* status_to_string(CardinalStatus s) {
        switch (s) {
        case CardinalStatus::OK:                  return "OK";
        case CardinalStatus::NOT_INITIALIZED:     return "NOT_INITIALIZED";
        case CardinalStatus::ALREADY_INITIALIZED: return "ALREADY_INITIALIZED";
        case CardinalStatus::INFERENCE_FAILED:    return "INFERENCE_FAILED";
        case CardinalStatus::STORAGE_ERROR:       return "STORAGE_ERROR";
        case CardinalStatus::CONFIG_ERROR:        return "CONFIG_ERROR";
        case CardinalStatus::INVALID_INPUT:       return "INVALID_INPUT";
        case CardinalStatus::EXPORT_FAILED:       return "EXPORT_FAILED";
        case CardinalStatus::SESSION_NOT_FOUND:   return "SESSION_NOT_FOUND";
        case CardinalStatus::AUTH_FAILED:         return "AUTH_FAILED";
        case CardinalStatus::TIMEOUT:             return "TIMEOUT";
        case CardinalStatus::SHUTDOWN:            return "SHUTDOWN";
        case CardinalStatus::NOT_FOUND:           return "NOT_FOUND";
        case CardinalStatus::INTERNAL_ERROR:      return "INTERNAL_ERROR";
        default:                                  return "UNKNOWN";
        }
    }

    // =========================================================================
    // CardinalResult<T>
    // Structured result type -- no exceptions cross the API boundary.
    //
    // Usage:
    //   auto result = api.chat("hello");
    //   if (!result.ok()) { log(result.error_message); return; }
    //   use(result.value);
    // =========================================================================
    template<typename T>
    struct CardinalResult {
        CardinalStatus status = CardinalStatus::OK;
        std::string    error_message;
        T              value = T{};

        bool ok() const { return status == CardinalStatus::OK; }

        // Convenience factory methods
        static CardinalResult<T> success(T val) {
            CardinalResult<T> r;
            r.status = CardinalStatus::OK;
            r.value = std::move(val);
            return r;
        }

        static CardinalResult<T> failure(CardinalStatus s,
            const std::string& msg) {
            CardinalResult<T> r;
            r.status = s;
            r.error_message = msg;
            return r;
        }
    };

    // Specialization for operations with no return value
    struct CardinalVoidResult {
        CardinalStatus status = CardinalStatus::OK;
        std::string    error_message;

        bool ok() const { return status == CardinalStatus::OK; }

        static CardinalVoidResult success() {
            return CardinalVoidResult{};
        }

        static CardinalVoidResult failure(CardinalStatus s,
            const std::string& msg) {
            CardinalVoidResult r;
            r.status = s;
            r.error_message = msg;
            return r;
        }
    };

    // =========================================================================
    // FeelingInfo
    // API-boundary version of FeelingOutput.
    // Carries all six fields from Pass 1 constrained decoding.
    // =========================================================================
    struct FeelingInfo {
        float       confidence = 0.0f;
        std::string reasoning_type;           // causal|deductive|etc.
        std::string reasoning_domain;         // factual|ethical|etc.
        bool        uncertainty_flag = false;
        bool        contradiction_flag = false;
        bool        rule_candidate = false;
    };

    // =========================================================================
    // ChatResponse
    // Result of a single chat() call.
    // =========================================================================
    struct ChatResponse {
        std::string  session_id;
        std::string  response;           // Final response text
        FeelingInfo  feeling;            // Pass 1 introspective state
        std::string  episode_id;         // ID of the logged episode
        std::string              inference_id;       // for trace lookup
        std::optional<AgentResult> agent_result;    // set when agent mode

        // Verifier outputs
        bool         rule_committed = false;
        std::string  committed_rule_id;
        int          contradictions_found = 0;
        int          contradictions_resolved = 0;
        int          contradictions_flagged = 0;

        // Performance
        int          pass1_tokens = 0;
        int          pass2_tokens = 0;
        int          total_ms = 0;
    };

    // =========================================================================
    // ChatTurn
    // A single turn in a conversation -- used to expose history.
    // =========================================================================
    struct ChatTurn {
        std::string role;       // "user" or "assistant"
        std::string content;
        std::string timestamp;
    };

    // =========================================================================
    // SessionInfo
    // Snapshot of a conversation session's current state.
    // =========================================================================
    struct SessionInfo {
        std::string            session_id;
        int                    turn_count = 0;
        std::vector<ChatTurn>  history;
        std::string            created_at;
        std::string            last_active_at;
    };

    // =========================================================================
    // RuleInfo
    // API-boundary version of Rule.
    // =========================================================================
    struct RuleInfo {
        std::string id;
        std::string domain;
        std::string condition;
        std::string consequence;
        float       confidence = 0.0f;
        int         trigger_count = 0;
        std::string episode_id;      // Provenance -- which episode created this
        std::string reasoning_type;  // Provenance -- reasoning type at extraction
        std::string created_at;
        std::string updated_at;
        bool        has_provenance = false;
    };

    // =========================================================================
    // EpisodeInfo
    // API-boundary version of EpisodeRecord.
    // =========================================================================
    struct EpisodeInfo {
        std::string id;
        std::string timestamp;
        std::string user_message;
        std::string response_summary;
        float       confidence = 0.0f;
        std::string reasoning_type;
        std::string reasoning_domain;
        bool        contradiction = false;
        bool        uncertainty = false;
        bool        rule_candidate = false;
        std::string extracted_rule_id;
        int         pass1_tokens = 0;
        int         pass2_tokens = 0;
        int         total_ms = 0;
    };

    // =========================================================================
    // MemoryStats
    // Snapshot of memory system health.
    // =========================================================================
    struct MemoryStats {
        // Episodes
        int   total_episodes = 0;
        int   migrated_episodes = 0;
        int   high_conf_episodes = 0;
        int   rule_candidate_count = 0;
        float avg_episode_confidence = 0.0f;

        // Rules
        int   total_rules = 0;
        int   active_rules = 0;
        float avg_rule_confidence = 0.0f;

        // Retriever
        int   index_size = 0;
        int   vocabulary_size = 0;
        bool  index_ready = false;
    };

    // =========================================================================
    // VerifierStats
    // Snapshot of verifier pipeline activity.
    // =========================================================================
    struct VerifierStats {
        int total_checks = 0;
        int total_rules_extracted = 0;
        int total_contradictions = 0;
        int total_resolved = 0;
        int total_flagged = 0;
        int total_maintenance_runs = 0;
    };

    // =========================================================================
    // SystemStats
    // Combined stats returned by the /stats endpoint and api.get_stats().
    // =========================================================================
    struct SystemStats {
        MemoryStats   memory;
        VerifierStats verifier;
        std::string   uptime_seconds;   // Formatted uptime string
        std::string   version;          // Cardinal version string
        bool          initialized = false;
    };

    // =========================================================================
    // ExportRequest
    // Parameters for a training data export operation.
    // =========================================================================
    struct ExportRequest {
        std::string output_path;
        float       min_confidence = 0.7f;
        std::string domain;          // Empty = all domains
        int         max_examples = 0;     // 0 = no limit
        bool        include_rules = true;
    };

    // =========================================================================
    // ExportInfo
    // Result of a completed export operation.
    // =========================================================================
    struct ExportInfo {
        int         episodes_exported = 0;
        int         rules_exported = 0;
        int         total_exported = 0;
        float       avg_confidence = 0.0f;
        std::string output_path;
        std::string timestamp;
    };

    // =========================================================================
    // ScanResult
    // Result of a full contradiction scan.
    // =========================================================================
    struct ScanResult {
        int total_contradictions = 0;
        int resolved = 0;
        int flagged = 0;
        int skipped = 0;
    };

    // =========================================================================
    // StreamToken
    // A single token from a streaming inference response.
    // Used by the streaming callback and SSE endpoint.
    // =========================================================================
    struct StreamToken {
        std::string  session_id;
        std::string  token;
        bool         is_final = false;   // True on last token
        FeelingInfo  feeling;                // Only populated on final token
    };

    // =========================================================================
    // StreamCallback
    // Called for each token during streaming inference.
    // Return false to abort generation.
    // Signature compatible with std::function for pybind11 wrapping.
    // =========================================================================
    using ApiStreamCallback = std::function<bool(const StreamToken&)>;

} // namespace cardinal
