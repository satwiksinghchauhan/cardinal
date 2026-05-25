#pragma once
// =============================================================================
// Cardinal - Symbolic Engine
// File: src/verifier/symbolic_engine.h
// Full SWI-Prolog interface for Cardinal's symbolic verification layer.
// Manages a live Prolog engine, loads Cardinal's knowledge base, and exposes
// rule assertion, contradiction detection, and rule querying to C++.
//
// The Prolog knowledge base (cardinal_kb.pl) is loaded at startup and contains
// the core inference rules. Dynamic facts (rules from rule_store) are asserted
// at runtime via the C++ interface.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "utils/json_parser.h"

// SWI-Prolog C interface
#include <SWI-Prolog.h>

#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <functional>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // PrologResult
    // Result of a single Prolog query.
    // -----------------------------------------------------------------------------
    struct PrologResult {
        bool        success;            // Query succeeded (at least one solution)
        bool        deterministic;      // No choice points remain
        std::string error_message;      // Set on failure

        // Bound variable values from the query (as strings)
        std::vector<std::pair<std::string, std::string>> bindings;
    };

    // -----------------------------------------------------------------------------
    // ContradictionResult
    // Result of running contradiction detection.
    // -----------------------------------------------------------------------------
    struct ContradictionResult {
        bool        has_contradiction;
        std::string rule_id_a;          // First contradicting rule
        std::string rule_id_b;          // Second contradicting rule
        std::string explanation;        // Natural language explanation
        float       severity;           // 0.0 - 1.0
    };

    // -----------------------------------------------------------------------------
    // RuleAssertResult
    // Result of asserting a rule into Prolog.
    // -----------------------------------------------------------------------------
    struct RuleAssertResult {
        bool        success;
        std::string prolog_fact;        // The asserted Prolog fact (for logging)
        std::string error_message;
    };

    // -----------------------------------------------------------------------------
    // SymbolicEngine
    // Wraps the SWI-Prolog engine for Cardinal's verifier.
    // Thread-safe ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all Prolog calls are serialized via internal mutex since
    // SWI-Prolog's engine is not thread-safe by default.
    //
    // Lifecycle:
    //   1. init() ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â initializes SWI-Prolog, loads cardinal_kb.pl
    //   2. load_rules() ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â asserts all rules from rule_store into Prolog
    //   3. check_contradiction() ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â called after each inference with new rule candidate
    //   4. query_rules() ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â retrieves rules matching a domain for prompt injection
    //   5. shutdown() ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â cleanly terminates Prolog engine
    // -----------------------------------------------------------------------------
    class SymbolicEngine {
    public:
        explicit SymbolicEngine(const CardinalConfig& config);
        ~SymbolicEngine();

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        // Initialize SWI-Prolog engine and load knowledge base
        // kb_path: path to cardinal_kb.pl (default: src/verifier/cardinal_kb.pl)
        void init(const std::string& kb_path = "");

        // Check if engine is initialized and ready
        bool is_ready() const { return initialized_; }

        // Shutdown Prolog engine
        void shutdown();

        // -------------------------------------------------------------------------
        // Rule management
        // -------------------------------------------------------------------------

        // Assert a rule into Prolog's dynamic database
        // Creates a fact: cardinal_rule(Id, Domain, Condition, Consequence, Confidence)
        RuleAssertResult assert_rule(const Rule& rule);

        // Assert all rules from a vector (bulk load at startup)
        int assert_rules(const std::vector<Rule>& rules);

        // Retract a rule by ID
        bool retract_rule(const std::string& rule_id);

        // Retract all rules (for reload)
        void retract_all_rules();

        // -------------------------------------------------------------------------
        // Contradiction detection
        // -------------------------------------------------------------------------

        // Check if a new rule candidate contradicts existing rules
        // Called by ConsistencyCheck when rule_candidate_signal is true
        ContradictionResult check_contradiction(
            const std::string& domain,
            const std::string& condition,
            const std::string& consequence) const;

        // Check consistency of entire rule base
        std::vector<ContradictionResult> check_all_contradictions() const;

        // -------------------------------------------------------------------------
        // Rule querying
        // -------------------------------------------------------------------------

        // Query rules for a given domain above confidence threshold
        std::vector<Rule> query_rules(const std::string& domain,
            float min_confidence = 0.0f,
            int   max_results = 10) const;

        // Check if a specific claim is supported by the rule base
        // Returns confidence score (0.0 if not supported)
        float check_claim(const std::string& domain,
            const std::string& claim) const;

        // -------------------------------------------------------------------------
        // Direct Prolog query interface
        // For advanced use ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â run arbitrary Prolog goals
        // -------------------------------------------------------------------------
        PrologResult query(const std::string& goal) const;

        // Run a goal that should succeed exactly once
        bool query_once(const std::string& goal) const;

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        int  asserted_rule_count() const { return asserted_rules_; }
        bool prolog_initialized()  const { return initialized_; }

        // Disable copy/move ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â owns Prolog engine state
        SymbolicEngine(const SymbolicEngine&) = delete;
        SymbolicEngine& operator=(const SymbolicEngine&) = delete;
        SymbolicEngine(SymbolicEngine&&) = delete;
        SymbolicEngine& operator=(SymbolicEngine&&) = delete;

    private:
        // -------------------------------------------------------------------------
        // Internal helpers
        // -------------------------------------------------------------------------

        // Escape a string for safe use in Prolog atom
        std::string escape_prolog_atom(const std::string& s) const;

        // Convert a Rule to a Prolog fact string
        std::string rule_to_prolog_fact(const Rule& rule) const;

        // Parse Prolog query result term to string
        std::string term_to_string(term_t t) const;

        // Handle Prolog exception ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â extract error message
        std::string get_prolog_error() const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        bool                    initialized_ = false;
        int                     asserted_rules_ = 0;
        mutable std::mutex      mutex_;

        // SWI-Prolog argc/argv storage (must outlive PL_initialise)
        std::vector<std::string>  pl_argv_strings_;
        std::vector<char*>        pl_argv_;
    };

    // -----------------------------------------------------------------------------
    // SymbolicEngineError
    // -----------------------------------------------------------------------------
    class SymbolicEngineError : public std::runtime_error {
    public:
        explicit SymbolicEngineError(const std::string& message)
            : std::runtime_error("SymbolicEngineError: " + message) {}
    };

} // namespace cardinal