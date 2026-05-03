// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Rule Store
// File: src/memory/rule_store.h
// Manages the persistent rule base - the symbolic memory of Cardinal.
// Rules are derived from feeling output when rule_candidate_signal is true.
// The rule store handles CRUD, confidence decay, pruning, and atomic persistence.
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

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <functional>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // RuleQuery
    // Parameters for querying the rule store.
    // -----------------------------------------------------------------------------
    struct RuleQuery {
        std::string              domain;          // Filter by domain (empty = all)
        std::string              condition_hint;  // Fuzzy match on condition text
        float                    min_confidence;  // Minimum confidence threshold
        int                      max_results;     // Maximum rules to return (0 = all)
        bool                     active_only;     // Only rules above min_confidence

        // Defaults
        RuleQuery()
            : min_confidence(0.0f)
            , max_results(0)
            , active_only(true) {
        }
    };

    // -----------------------------------------------------------------------------
    // RuleStoreStats
    // Snapshot of rule store health.
    // -----------------------------------------------------------------------------
    struct RuleStoreStats {
        int   total_rules;
        int   active_rules;       // Above min_confidence
        int   pruned_rules;       // Below min_confidence (still in store until next save)
        float avg_confidence;
        float avg_trigger_count;
        int   rules_by_domain[6]; // factual, ethical, spatial, temporal, social, mathematical
    };

    // -----------------------------------------------------------------------------
    // RuleStore
    // Thread-safe persistent rule base.
    //
    // Lifecycle:
    //   1. load() at startup - reads rules.json, returns empty set on first run
    //   2. add_rule() / update_rule() as verifier extracts rules from inferences
    //   3. decay_confidence() called periodically to age out stale rules
    //   4. prune() removes rules below min_confidence threshold
    //   5. save() persists to disk atomically
    //   6. query() retrieves relevant rules for injection into prompts
    // -----------------------------------------------------------------------------
    class RuleStore {
    public:
        explicit RuleStore(const CardinalConfig& config);

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        // Load rules from disk - safe to call on empty/nonexistent file
        void load();

        // Save rules to disk atomically
        void save();

        // -------------------------------------------------------------------------
        // Rule management
        // -------------------------------------------------------------------------

        // Add a new rule - returns assigned rule ID
        // If a semantically similar rule exists, merges instead of duplicating
        std::string add_rule(const std::string& domain,
            const std::string& condition,
            const std::string& consequence,
            float              initial_confidence = 0.5f,
            const std::string& episode_id = "",
            const std::string& reasoning_type = "");

        // Update an existing rule's confidence
        bool update_confidence(const std::string& rule_id, float delta);

        // Record that a rule was triggered (used in inference)
        bool record_trigger(const std::string& rule_id);

        // Remove a rule by ID
        bool remove_rule(const std::string& rule_id);

        // Get a rule by ID
        std::optional<Rule> get_rule(const std::string& rule_id) const;

        // -------------------------------------------------------------------------
        // Query
        // -------------------------------------------------------------------------

        // Query rules for injection into inference prompts
        std::vector<Rule> query(const RuleQuery& q) const;

        // Get top N rules by confidence for a given domain
        std::vector<Rule> get_top_rules(const std::string& domain, int n) const;

        // Get all rules
        std::vector<Rule> get_all() const;

        // -------------------------------------------------------------------------
        // Maintenance
        // -------------------------------------------------------------------------

        // Apply confidence decay to all rules (call after each inference cycle)
        // decay = config.verifier.rule_confidence_decay per cycle
        void decay_confidence();

        // Remove rules below min_confidence threshold
        // Returns number of rules pruned
        int prune();

        // Enforce max_rules cap - removes lowest confidence rules if over limit
        int enforce_limit();

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        RuleStoreStats stats() const;
        int            size()  const;
        bool           empty() const;

        // Check if rule store has been modified since last save
        bool is_dirty() const { return dirty_; }

    private:
        // -------------------------------------------------------------------------
        // Internal helpers
        // -------------------------------------------------------------------------

        // Check if two rules are semantically similar enough to merge
        // Simple heuristic: same domain + condition overlap > threshold
        bool is_similar(const Rule& a, const Rule& b, float threshold = 0.8f) const;

        // Merge rule b into rule a - boosts confidence, updates timestamps
        void merge_rule(Rule& existing, const Rule& incoming);

        // Calculate word overlap ratio between two strings (for similarity check)
        float word_overlap(const std::string& a, const std::string& b) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        std::unordered_map<std::string, Rule>    rules_;   // ID -> Rule
        mutable std::mutex                       mutex_;
        bool                                     dirty_ = false;
        bool                                     loaded_ = false;
    };

    // -----------------------------------------------------------------------------
    // RuleStoreError
    // -----------------------------------------------------------------------------
    class RuleStoreError : public std::runtime_error {
    public:
        explicit RuleStoreError(const std::string& message)
            : std::runtime_error("RuleStoreError: " + message) {}
    };

} // namespace cardinal