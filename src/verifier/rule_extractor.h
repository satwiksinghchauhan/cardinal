// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Rule Extractor
// File: src/verifier/rule_extractor.h
// Extracts rule candidates from inference cycles when rule_candidate_signal
// is true. The extractor bridges the LLM core and the symbolic engine:
//   - Takes feeling output + response text as input
//   - Derives a structured Rule (condition + consequence + domain)
//   - Runs contradiction check via SymbolicEngine before committing
//   - Writes confirmed rules to RuleStore and asserts into Prolog
//
// Design principle: the LLM signals WHERE a rule might exist, but does NOT
// propose WHAT the rule is. The extractor derives the rule from the response
// text using NLP heuristics, keeping LLM bias out of the rule base.
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
#include "memory/rule_store.h"
#include "verifier/symbolic_engine.h"

#include <string>
#include <vector>
#include <optional>
#include <mutex>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // ExtractionInput
    // Input to the rule extractor for a single inference cycle.
    // -----------------------------------------------------------------------------
    struct ExtractionInput {
        FeelingOutput feeling;          // Feeling output from Pass 1
        std::string   user_message;     // Original user query
        std::string   response_text;    // Full response from Pass 2
        std::string   episode_id;       // Episodic ID for traceability
    };

    // -----------------------------------------------------------------------------
    // CandidateRule
    // A rule candidate before it has been validated and committed.
    // May be rejected by contradiction check or confidence threshold.
    // -----------------------------------------------------------------------------
    struct CandidateRule {
        std::string domain;
        std::string condition;
        std::string consequence;
        float       confidence;
        std::string extraction_method;  // How it was extracted
        std::string source_sentence;    // The sentence it was derived from

        // Provenance (Phase 6) -- carried from ExtractionInput
        std::string episode_id;         // Episode that triggered this extraction
        std::string reasoning_type;     // Reasoning type from FeelingOutput
    };

    // -----------------------------------------------------------------------------
    // ExtractionResult
    // Result of a single rule extraction attempt.
    // -----------------------------------------------------------------------------
    struct ExtractionResult {
        bool                         extracted;          // Was a rule candidate found?
        bool                         committed;          // Was it written to rule store?
        bool                         contradiction_found; // Did it contradict existing rules?
        std::optional<CandidateRule> candidate;          // The candidate (if found)
        std::string                  committed_rule_id;  // ID if committed
        std::string                  rejection_reason;   // Why it was rejected (if any)
        ContradictionResult          contradiction;      // Details if contradiction found
    };

    // -----------------------------------------------------------------------------
    // RuleExtractor
    // Extracts, validates, and commits rules from inference cycles.
    //
    // Usage:
    //   RuleExtractor extractor(config, rule_store, symbolic_engine);
    //   auto result = extractor.extract(input);
    //   if (result.committed) { ... }
    // -----------------------------------------------------------------------------
    class RuleExtractor {
    public:
        RuleExtractor(const CardinalConfig& config,
            RuleStore& rule_store,
            SymbolicEngine& symbolic_engine);

        // -------------------------------------------------------------------------
        // Core extraction
        // -------------------------------------------------------------------------

        // Extract a rule candidate from an inference cycle
        // Only called when feeling.rule_candidate_signal == true
        ExtractionResult extract(const ExtractionInput& input);

        // -------------------------------------------------------------------------
        // Batch operations
        // -------------------------------------------------------------------------

        // Re-load all rules from rule store into Prolog engine
        // Called at startup to sync rule_store with symbolic_engine
        void sync_rules_to_prolog();

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        int total_extracted()    const { return total_extracted_; }
        int total_committed()    const { return total_committed_; }
        int total_rejected()     const { return total_rejected_; }
        int total_contradicted() const { return total_contradicted_; }

    private:
        // -------------------------------------------------------------------------
        // Extraction pipeline
        // -------------------------------------------------------------------------

        // Step 1: Extract candidate rule from response text
        std::optional<CandidateRule> extract_candidate(
            const ExtractionInput& input) const;

        // Step 2: Validate candidate Ã¢â‚¬â€ confidence threshold, non-empty fields
        bool validate_candidate(const CandidateRule& candidate,
            std::string& rejection_reason) const;

        // Step 3: Check for contradictions via symbolic engine
        ContradictionResult check_contradictions(
            const CandidateRule& candidate) const;

        // Step 4: Commit to rule store and assert into Prolog
        std::string commit_rule(const CandidateRule& candidate);

        // -------------------------------------------------------------------------
        // NLP extraction heuristics
        // -------------------------------------------------------------------------

        // Extract condition/consequence from causal sentence patterns
        // "if X then Y", "X causes Y", "X leads to Y", "X results in Y"
        std::optional<CandidateRule> extract_causal(
            const std::string& text,
            const std::string& domain) const;

        // Extract from deductive patterns
        // "therefore X", "thus X", "it follows that X"
        std::optional<CandidateRule> extract_deductive(
            const std::string& text,
            const std::string& domain) const;

        // Extract from general declarative statements
        // Fallback: use first meaningful sentence as condition,
        // last conclusion sentence as consequence
        std::optional<CandidateRule> extract_declarative(
            const std::string& text,
            const std::string& user_message,
            const std::string& domain) const;

        // -------------------------------------------------------------------------
        // Text processing utilities
        // -------------------------------------------------------------------------

        // Split response into sentences
        std::vector<std::string> split_sentences(const std::string& text) const;

        // Find sentence matching a pattern
        std::string find_pattern_sentence(
            const std::vector<std::string>& sentences,
            const std::vector<std::string>& patterns) const;

        // Clean extracted text Ã¢â‚¬â€ remove markdown, truncate, normalize
        std::string clean_text(const std::string& text,
            int max_length = 200) const;

        // Check if a string contains any of the given keywords
        bool contains_any(const std::string& text,
            const std::vector<std::string>& keywords) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        RuleStore& rule_store_;
        SymbolicEngine& symbolic_engine_;
        mutable std::mutex    mutex_;

        // Stats
        int total_extracted_ = 0;
        int total_committed_ = 0;
        int total_rejected_ = 0;
        int total_contradicted_ = 0;
    };

    // -----------------------------------------------------------------------------
    // RuleExtractorError
    // -----------------------------------------------------------------------------
    class RuleExtractorError : public std::runtime_error {
    public:
        explicit RuleExtractorError(const std::string& message)
            : std::runtime_error("RuleExtractorError: " + message) {}
    };

} // namespace cardinal