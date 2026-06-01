#pragma once
// =============================================================================
// Cardinal - API Types (v1.5.0)
// File: src/api/cardinal_types.h
// =============================================================================

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include "agent/agent_types.h"

namespace cardinal {

    // =========================================================================
    // CardinalStatus
    // =========================================================================
    enum class CardinalStatus : int {
        OK                  = 0,
        NOT_INITIALIZED     = 1,
        ALREADY_INITIALIZED = 2,
        INFERENCE_FAILED    = 3,
        STORAGE_ERROR       = 4,
        CONFIG_ERROR        = 5,
        INVALID_INPUT       = 6,
        EXPORT_FAILED       = 7,
        SESSION_NOT_FOUND   = 8,
        AUTH_FAILED         = 9,
        TIMEOUT             = 10,
        SHUTDOWN            = 11,
        NOT_FOUND           = 12,
        TRAINING_FAILED     = 13,
        SELF_MODEL_ERROR    = 14,
        SCHEDULER_ERROR     = 15,
        COMPUTER_USE_ERROR  = 16,
        INVALID_REQUEST     = 17,
        VOICE_ERROR         = 18,
        INTERNAL_ERROR      = 99
    };

    inline const char* status_to_string(CardinalStatus s) {
        switch (s) {
        case CardinalStatus::OK:                   return "OK";
        case CardinalStatus::NOT_INITIALIZED:      return "NOT_INITIALIZED";
        case CardinalStatus::ALREADY_INITIALIZED:  return "ALREADY_INITIALIZED";
        case CardinalStatus::INFERENCE_FAILED:     return "INFERENCE_FAILED";
        case CardinalStatus::STORAGE_ERROR:        return "STORAGE_ERROR";
        case CardinalStatus::CONFIG_ERROR:         return "CONFIG_ERROR";
        case CardinalStatus::INVALID_INPUT:        return "INVALID_INPUT";
        case CardinalStatus::EXPORT_FAILED:        return "EXPORT_FAILED";
        case CardinalStatus::SESSION_NOT_FOUND:    return "SESSION_NOT_FOUND";
        case CardinalStatus::AUTH_FAILED:          return "AUTH_FAILED";
        case CardinalStatus::TIMEOUT:              return "TIMEOUT";
        case CardinalStatus::SHUTDOWN:             return "SHUTDOWN";
        case CardinalStatus::NOT_FOUND:            return "NOT_FOUND";
        case CardinalStatus::TRAINING_FAILED:      return "TRAINING_FAILED";
        case CardinalStatus::SELF_MODEL_ERROR:     return "SELF_MODEL_ERROR";
        case CardinalStatus::SCHEDULER_ERROR:      return "SCHEDULER_ERROR";
        case CardinalStatus::COMPUTER_USE_ERROR:   return "COMPUTER_USE_ERROR";
        case CardinalStatus::INVALID_REQUEST:      return "INVALID_REQUEST";
        case CardinalStatus::INTERNAL_ERROR:       return "INTERNAL_ERROR";
        default:                                   return "UNKNOWN";
        }
    }

    // =========================================================================
    // CardinalResult<T>
    // =========================================================================
    template<typename T>
    struct CardinalResult {
        CardinalStatus status        = CardinalStatus::OK;
        std::string    error_message;
        std::string    message;
        T              value         = T{};

        bool ok() const { return status == CardinalStatus::OK; }

        static CardinalResult<T> success(T val) {
            CardinalResult<T> r;
            r.status = CardinalStatus::OK;
            r.value  = std::move(val);
            return r;
        }

        static CardinalResult<T> failure(CardinalStatus s, const std::string& msg) {
            CardinalResult<T> r;
            r.status        = s;
            r.error_message = msg;
            r.message       = msg;
            return r;
        }

        // Alias used throughout v1.5.0 code
        static CardinalResult<T> error(CardinalStatus s, const std::string& msg) {
            return failure(s, msg);
        }
    };

    // =========================================================================
    // CardinalVoidResult
    // =========================================================================
    struct CardinalVoidResult {
        CardinalStatus status        = CardinalStatus::OK;
        std::string    error_message;
        std::string    message;

        bool ok() const { return status == CardinalStatus::OK; }

        static CardinalVoidResult success() {
            return CardinalVoidResult{};
        }

        static CardinalVoidResult failure(CardinalStatus s, const std::string& msg) {
            CardinalVoidResult r;
            r.status        = s;
            r.error_message = msg;
            r.message       = msg;
            return r;
        }

        // v1.5.0 alias — NOTE: cannot be named ok() as that clashes with
        // the bool ok() const member. Use success() for the no-error factory.
        static CardinalVoidResult error(CardinalStatus s, const std::string& msg) {
            return failure(s, msg);
        }
    };

    // =========================================================================
    // FeelingInfo
    // =========================================================================
    struct FeelingInfo {
        float       confidence       = 0.0f;
        std::string reasoning_type;
        std::string reasoning_domain;
        bool        uncertainty_flag   = false;
        bool        contradiction_flag = false;
        bool        rule_candidate     = false;
    };

    // =========================================================================
    // ChatResponse
    // =========================================================================
    struct ChatResponse {
        std::string  session_id;
        std::string  response;
        FeelingInfo  feeling;
        std::string  episode_id;
        std::string  inference_id;
        std::optional<AgentResult> agent_result;

        bool        rule_committed          = false;
        std::string committed_rule_id;
        int         contradictions_found    = 0;
        int         contradictions_resolved = 0;
        int         contradictions_flagged  = 0;

        int         pass1_tokens = 0;
        int         pass2_tokens = 0;
        int         total_ms     = 0;
    };

    struct ChatTurn {
        std::string role;
        std::string content;
        std::string timestamp;
    };

    struct SessionInfo {
        std::string           session_id;
        int                   turn_count = 0;
        std::vector<ChatTurn> history;
        std::string           created_at;
        std::string           last_active_at;
    };

    struct RuleInfo {
        std::string id;
        std::string domain;
        std::string condition;
        std::string consequence;
        float       confidence    = 0.0f;
        int         trigger_count = 0;
        std::string episode_id;
        std::string reasoning_type;
        std::string created_at;
        std::string updated_at;
        bool        has_provenance = false;
    };

    struct EpisodeInfo {
        std::string id;
        std::string timestamp;
        std::string user_message;
        std::string response_summary;
        float       confidence       = 0.0f;
        std::string reasoning_type;
        std::string reasoning_domain;
        bool        contradiction    = false;
        bool        uncertainty      = false;
        bool        rule_candidate   = false;
        std::string extracted_rule_id;
        int         pass1_tokens     = 0;
        int         pass2_tokens     = 0;
        int         total_ms         = 0;
    };

    struct MemoryStats {
        int   total_episodes         = 0;
        int   migrated_episodes      = 0;
        int   high_conf_episodes     = 0;
        int   rule_candidate_count   = 0;
        float avg_episode_confidence = 0.0f;
        int   total_rules            = 0;
        int   active_rules           = 0;
        float avg_rule_confidence    = 0.0f;
        int   index_size             = 0;
        int   vocabulary_size        = 0;
        bool  index_ready            = false;
    };

    struct VerifierStats {
        int total_checks           = 0;
        int total_rules_extracted  = 0;
        int total_contradictions   = 0;
        int total_resolved         = 0;
        int total_flagged          = 0;
        int total_maintenance_runs = 0;
    };

    struct SystemStats {
        MemoryStats   memory;
        VerifierStats verifier;
        std::string   uptime_seconds;
        std::string   version;
        bool          initialized = false;
    };

    struct ExportRequest {
        std::string output_path;
        float       min_confidence = 0.7f;
        std::string domain;
        int         max_examples   = 0;
        bool        include_rules  = true;
    };

    struct ExportInfo {
        int         episodes_exported = 0;
        int         rules_exported    = 0;
        int         total_exported    = 0;
        float       avg_confidence    = 0.0f;
        std::string output_path;
        std::string timestamp;
    };

    struct ScanResult {
        int total_contradictions = 0;
        int resolved             = 0;
        int flagged              = 0;
        int skipped              = 0;
    };

    struct StreamToken {
        std::string session_id;
        std::string token;
        bool        is_final = false;
        FeelingInfo feeling;
    };

    using ApiStreamCallback = std::function<bool(const StreamToken&)>;

} // namespace cardinal
