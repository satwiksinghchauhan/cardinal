// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Consistency Check
// File: src/verifier/consistency_check.h
// Orchestrates consistency verification across the full verifier pipeline.
// Sits above SymbolicEngine and RuleExtractor â€” called after each inference
// cycle to maintain rule base integrity.
//
// Responsibilities:
//   1. Trigger rule extraction when feeling.rule_candidate_signal is true
//   2. Run contradiction scan when feeling.contradiction_flag is true
//   3. Periodic full rule base consistency sweep
//   4. Confidence adjustment based on contradiction resolution
//   5. Route results back to episodic memory for traceability
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
#include "memory/episodic.h"
#include "verifier/symbolic_engine.h"
#include "verifier/rule_extractor.h"
#include "verifier/neural_verifier.h"

#include <string>
#include <vector>
#include <mutex>
#include <chrono>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ResolutionOutcome
    // Result of a single contradiction resolution attempt.
    // -------------------------------------------------------------------------
    enum class ResolutionOutcome {
        RESOLVED,       // One rule deprecated (confidence set to 0.0)
        FLAGGED,        // Confidence delta too small -- flagged for review
        SKIPPED,        // One or both rules not found in store
        NO_CONTRADICTION // ContradictionResult had no contradiction
    };

    // -----------------------------------------------------------------------------
    // ConsistencyCheckInput
    // Input for a post-inference consistency check.
    // -----------------------------------------------------------------------------
    struct ConsistencyCheckInput {
        FeelingOutput feeling;          // Feeling output from Pass 1
        std::string   user_message;     // Original user query
        std::string   response_text;    // Full response from Pass 2
        std::string   episode_id;       // Episodic ID for traceability
    };

    // -----------------------------------------------------------------------------
    // ConsistencyCheckResult
    // Combined result of all checks run for one inference cycle.
    // -----------------------------------------------------------------------------
    struct ConsistencyCheckResult {
        // Rule extraction results
        bool             rule_extracted;
        bool             rule_committed;
        std::string      committed_rule_id;
        ExtractionResult extraction;

        // Contradiction results
        bool                              contradiction_found;
        std::vector<ContradictionResult>  contradictions;
        int                               contradictions_resolved = 0;  // Auto-resolved by confidence delta
        int                               contradictions_flagged = 0;  // Flagged for review (delta too small)

        // Neural verifier results (populated when mode includes neural)
        bool                     neural_ran;
        NeuralVerificationResult neural_result;

        // Maintenance
        int              rules_decayed;
        int              rules_pruned;
        bool             periodic_sweep_ran;

        // Overall
        bool             success;
        std::string      summary;
    };

    // -----------------------------------------------------------------------------
    // ConsistencyChecker
    // The top-level verifier orchestrator. One instance per Cardinal session.
    // Supports three modes via config.verifier.mode:
    //   "symbolic" â€” SWI-Prolog only
    //   "neural"   â€” NeuralVerifier only (requires neural_model_path)
    //   "hybrid"   â€” Both run, results merged via weighted consensus
    // -----------------------------------------------------------------------------
    class ConsistencyChecker {
    public:
        ConsistencyChecker(const CardinalConfig& config,
            RuleStore& rule_store,
            EpisodicMemory& episodic,
            SymbolicEngine& symbolic_engine,
            RuleExtractor& rule_extractor,
            NeuralVerifier& neural_verifier);

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        // Initialize â€” syncs rules from store to Prolog engine
        void init();

        // -------------------------------------------------------------------------
        // Core check â€” called after each inference cycle
        // -------------------------------------------------------------------------
        ConsistencyCheckResult check(const ConsistencyCheckInput& input);

        // -------------------------------------------------------------------------
        // Manual operations
        // -------------------------------------------------------------------------

        // Run a full contradiction scan across all rules
        std::vector<ContradictionResult> run_full_scan();

        // Force rule decay + prune cycle
        int run_maintenance();

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        int total_checks()          const { return total_checks_; }
        int total_rules_extracted() const { return total_rules_extracted_; }
        int total_contradictions()  const { return total_contradictions_; }
        int total_maintenance_runs() const { return total_maintenance_runs_; }
        int total_resolved()  const { return total_resolved_; }
        int total_flagged()   const { return total_flagged_; }

    private:
        // -------------------------------------------------------------------------
        // Internal pipeline
        // -------------------------------------------------------------------------

        // Handle rule extraction if signaled
        ExtractionResult handle_rule_extraction(
            const ConsistencyCheckInput& input);

        // Handle contradiction check if flagged
        std::vector<ContradictionResult> handle_contradiction_check(
            const ConsistencyCheckInput& input);

        // Attempt to auto-resolve a detected contradiction.
        // Strategy: confidence-based -- lower confidence rule is deprecated.
        // If confidence delta < resolution_threshold, flags for review instead.
        // Returns the outcome of the resolution attempt.
        ResolutionOutcome resolve_contradiction(
            const ContradictionResult& contradiction,
            float                      resolution_threshold = 0.2f);

        // Run periodic maintenance if due
        bool run_periodic_maintenance();

        // Build human-readable summary of check result
        std::string build_summary(const ConsistencyCheckResult& result) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        RuleStore& rule_store_;
        EpisodicMemory& episodic_;
        SymbolicEngine& symbolic_engine_;
        RuleExtractor& rule_extractor_;
        NeuralVerifier& neural_verifier_;
        mutable std::mutex      mutex_;

        // Periodic maintenance tracking
        std::chrono::steady_clock::time_point last_maintenance_;
        static constexpr int MAINTENANCE_INTERVAL_INFERENCES = 10;
        int inferences_since_maintenance_ = 0;

        // Stats
        int total_checks_ = 0;
        int total_rules_extracted_ = 0;
        int total_contradictions_ = 0;
        int total_maintenance_runs_ = 0;
        int total_resolved_ = 0;
        int total_flagged_ = 0;

    };

    // -----------------------------------------------------------------------------
    // ConsistencyCheckError
    // -----------------------------------------------------------------------------
    class ConsistencyCheckError : public std::runtime_error {
    public:
        explicit ConsistencyCheckError(const std::string& message)
            : std::runtime_error("ConsistencyCheckError: " + message) {}
    };

} // namespace cardinal