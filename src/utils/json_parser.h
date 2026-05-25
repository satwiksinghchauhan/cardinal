#pragma once
// =============================================================================
// Cardinal - JSON Parser
// File: src/utils/json_parser.h
// Centralized JSON utilities for the entire codebase.
// Handles:
//   - Feeling schema output parsing (Pass 1 result)
//   - Rule store serialization/deserialization
//   - Knowledge graph serialization/deserialization
//   - General JSON helpers used across modules
//
// Phase 6 note:
//   Rule struct moved to src/memory/rule.h for clean separation.
//   Including json_parser.h still gives you Rule transitively via rule.h.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "memory/rule.h"   // Rule struct -- moved here in Phase 6

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ChatMessage
    // A single turn in the conversation history.
    // Defined here as a shared type used across core modules.
    // -------------------------------------------------------------------------
    struct ChatMessage {
        std::string role;     // "system" | "user" | "assistant"
        std::string content;
    };

    // -------------------------------------------------------------------------
    // FeelingOutput
    // Parsed result of Pass 1 constrained decoding.
    // This is the core data structure that flows from LLM core to verifier.
    // -------------------------------------------------------------------------
    struct FeelingOutput {
        float       confidence;             // 0.0 - 1.0
        std::string reasoning_type;         // analogical|causal|deductive|
        //   inductive|abductive|associative
        bool        uncertainty_flag;       // LLM is uncertain about this response
        bool        rule_candidate_signal;  // A rule might exist here
        bool        contradiction_flag;     // Conflict detected with existing knowledge
        std::string reasoning_domain;       // factual|ethical|spatial|
        //   temporal|social|mathematical

// Convenience checks
        bool is_high_confidence() const { return confidence >= 0.7f; }
        bool is_low_confidence()  const { return confidence < 0.4f; }
        bool needs_verification() const {
            return rule_candidate_signal || contradiction_flag;
        }

        // String representation for logging
        std::string to_string() const;
    };

    // -------------------------------------------------------------------------
    // KnowledgeNode
    // A single node in the knowledge graph.
    // Stays in json_parser.h -- no reason to move it yet.
    // -------------------------------------------------------------------------
    struct KnowledgeNode {
        std::string              id;
        std::string              label;
        std::string              type;        // "concept"|"fact"|"entity"|"relation"
        std::string              content;
        std::vector<std::string> related_ids;
        float                    confidence;
        std::string              source;
        std::string              created_at;
        std::string              updated_at;
    };

    // -------------------------------------------------------------------------
    // JsonParser
    // All methods are static -- no state, pure utilities.
    // -------------------------------------------------------------------------
    class JsonParser {
    public:
        // -- Feeling output parsing --

        // Parse raw JSON string from Pass 1 constrained decoding.
        // Returns FeelingOutput on success, throws ParseError on failure.
        static FeelingOutput parse_feeling_output(const std::string& json_str);

        // Validate a parsed FeelingOutput for logical consistency.
        // Returns true if valid, false + fills error_msg if not.
        static bool validate_feeling_output(const FeelingOutput& feeling,
            std::string& error_msg);

        // Serialize FeelingOutput back to JSON string (for logging/injection).
        static std::string serialize_feeling_output(const FeelingOutput& feeling);

        // -- Rule store serialization --

        // Load all rules from rules.json.
        // Phase 6: also handles legacy rules missing provenance fields gracefully.
        static std::vector<Rule> load_rules(const std::string& path);

        // Save all rules to rules.json (atomic write).
        static void save_rules(const std::string& path,
            const std::vector<Rule>& rules);

        // Serialize a single Rule to JSON object.
        // Phase 6: includes episode_id and reasoning_type fields.
        static nlohmann::json rule_to_json(const Rule& rule);

        // Deserialize a single Rule from JSON object.
        // Phase 6: reads provenance fields with safe defaults for legacy rules.
        static Rule rule_from_json(const nlohmann::json& j);

        // -- Knowledge graph serialization --

        static std::vector<KnowledgeNode> load_knowledge(const std::string& path);
        static void save_knowledge(const std::string& path,
            const std::vector<KnowledgeNode>& nodes);
        static nlohmann::json  node_to_json(const KnowledgeNode& node);
        static KnowledgeNode   node_from_json(const nlohmann::json& j);

        // -- General utilities --

        static std::optional<nlohmann::json> try_parse(const std::string& json_str);
        static std::string pretty_print(const std::string& json_str, int indent = 2);
        static bool        is_valid_json(const std::string& json_str);

        static std::string get_string(const nlohmann::json& j,
            const std::string& key,
            const std::string& default_val = "");
        static float       get_float(const nlohmann::json& j,
            const std::string& key,
            float default_val = 0.0f);
        static bool        get_bool(const nlohmann::json& j,
            const std::string& key,
            bool default_val = false);

        static std::string generate_id();
        static std::string current_timestamp();

    private:
        static void atomic_write(const std::string& path,
            const std::string& content);
    };

    // -------------------------------------------------------------------------
    // ParseError
    // -------------------------------------------------------------------------
    class ParseError : public std::runtime_error {
    public:
        explicit ParseError(const std::string& message)
            : std::runtime_error("ParseError: " + message) {}
    };

} // namespace cardinal