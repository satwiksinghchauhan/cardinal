// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Consistency Check Implementation
// File: src/verifier/consistency_check.cpp
// =============================================================================

#include "consistency_check.h"
#include "utils/logger.h"

#include <sstream>

namespace cardinal {

    // =============================================================================
    // Constructor
    // =============================================================================

    ConsistencyChecker::ConsistencyChecker(const CardinalConfig& config,
        RuleStore& rule_store,
        EpisodicMemory& episodic,
        SymbolicEngine& symbolic_engine,
        RuleExtractor& rule_extractor,
        NeuralVerifier& neural_verifier)
        : config_(config)
        , rule_store_(rule_store)
        , episodic_(episodic)
        , symbolic_engine_(symbolic_engine)
        , rule_extractor_(rule_extractor)
        , neural_verifier_(neural_verifier)
        , last_maintenance_(std::chrono::steady_clock::now())
    {
        LOG_INFO("ConsistencyChecker initialized â€” mode: " + config_.verifier.mode);
    }

    // =============================================================================
    // init
    // =============================================================================

    void ConsistencyChecker::init() {
        // Sync all persisted rules to the Prolog engine
        rule_extractor_.sync_rules_to_prolog();

        LOG_INFO("ConsistencyChecker ready â€” " +
            std::to_string(rule_store_.size()) + " rules loaded");
    }

    // =============================================================================
    // check â€” main entry point, called after each inference cycle
    // =============================================================================

    ConsistencyCheckResult ConsistencyChecker::check(
        const ConsistencyCheckInput& input)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        ++total_checks_;
        ++inferences_since_maintenance_;

        ConsistencyCheckResult result{};
        result.success = true;
        result.rule_extracted = false;
        result.rule_committed = false;
        result.contradiction_found = false;
        result.neural_ran = false;
        result.rules_decayed = 0;
        result.rules_pruned = 0;
        result.periodic_sweep_ran = false;

        const std::string& mode = config_.verifier.mode;

        LOG_DEBUG("ConsistencyChecker: check #" + std::to_string(total_checks_) +
            " mode=" + mode + " episode=" + input.episode_id);

        // -------------------------------------------------------------------------
        // 1. Rule extraction (if signaled) â€” mode-agnostic
        // -------------------------------------------------------------------------
        if (input.feeling.rule_candidate_signal) {
            result.extraction = handle_rule_extraction(input);
            result.rule_extracted = result.extraction.extracted;
            result.rule_committed = result.extraction.committed;

            if (result.rule_committed) {
                result.committed_rule_id = result.extraction.committed_rule_id;
                ++total_rules_extracted_;
            }

            if (result.extraction.contradiction_found) {
                result.contradiction_found = true;
                result.contradictions.push_back(result.extraction.contradiction);
                ++total_contradictions_;
            }
        }

        // -------------------------------------------------------------------------
        // 2. Contradiction check â€” routed by mode
        // -------------------------------------------------------------------------
        if (input.feeling.contradiction_flag || result.rule_committed) {

            bool use_symbolic = (mode == "symbolic" || mode == "hybrid");
            bool use_neural = (mode == "neural" || mode == "hybrid") &&
                neural_verifier_.is_available();

            // --- Symbolic path ---
            if (use_symbolic && input.feeling.contradiction_flag) {
                int resolved_before = total_resolved_;
                int flagged_before = total_flagged_;

                auto contradictions = handle_contradiction_check(input);

                if (!contradictions.empty()) {
                    result.contradiction_found = true;
                    for (const auto& c : contradictions)
                        result.contradictions.push_back(c);
                    total_contradictions_ +=
                        static_cast<int>(contradictions.size());

                    // Capture resolution stats for this cycle
                    result.contradictions_resolved =
                        total_resolved_ - resolved_before;
                    result.contradictions_flagged =
                        total_flagged_ - flagged_before;
                }
            }

            // --- Neural path ---
            if (use_neural && result.rule_committed) {
                auto domain_rules = rule_store_.get_top_rules(
                    input.feeling.reasoning_domain, 5);

                auto nr = neural_verifier_.verify_rule_candidate(
                    input.feeling.reasoning_domain,
                    result.extraction.candidate.has_value()
                    ? result.extraction.candidate->condition : "",
                    result.extraction.candidate.has_value()
                    ? result.extraction.candidate->consequence : "",
                    domain_rules);

                result.neural_ran = true;
                result.neural_result = nr;

                if (nr.available && nr.contradiction_detected) {
                    result.contradiction_found = true;

                    // Build a ContradictionResult from neural output
                    ContradictionResult cr;
                    cr.has_contradiction = true;
                    cr.explanation = "[neural] " + nr.reasoning;
                    cr.severity = nr.contradiction_score;
                    result.contradictions.push_back(cr);
                    ++total_contradictions_;

                    LOG_WARN("ConsistencyChecker: neural contradiction detected "
                        "(score=" + std::to_string(nr.contradiction_score) +
                        ") â€” " + nr.reasoning);
                }

                // In hybrid mode: boost/reduce rule confidence based on neural score
                if (mode == "hybrid" && result.rule_committed &&
                    !result.committed_rule_id.empty()) {
                    float quality = nr.available ? nr.rule_quality_score : 0.5f;
                    float adj = (quality - 0.5f) * 0.1f; // -0.05 to +0.05
                    rule_store_.update_confidence(result.committed_rule_id, adj);
                    LOG_DEBUG("ConsistencyChecker: hybrid confidence adjustment " +
                        std::to_string(adj) + " for rule " +
                        result.committed_rule_id);
                }
            }
        }

        // -------------------------------------------------------------------------
        // 3. Periodic maintenance
        // -------------------------------------------------------------------------
        if (inferences_since_maintenance_ >= MAINTENANCE_INTERVAL_INFERENCES) {
            result.periodic_sweep_ran = run_periodic_maintenance();
            inferences_since_maintenance_ = 0;
        }

        // -------------------------------------------------------------------------
        // 4. Build summary
        // -------------------------------------------------------------------------
        result.summary = build_summary(result);
        LOG_DEBUG("ConsistencyChecker: " + result.summary);

        return result;
    }

    // =============================================================================
    // run_full_scan
    // =============================================================================

    std::vector<ContradictionResult> ConsistencyChecker::run_full_scan() {
        LOG_INFO("ConsistencyChecker: running full contradiction scan...");

        auto contradictions = symbolic_engine_.check_all_contradictions();

        if (contradictions.empty()) {
            LOG_INFO("ConsistencyChecker: rule base is consistent ("
                + std::to_string(rule_store_.size()) + " rules)");
            return contradictions;
        }

        LOG_WARN("ConsistencyChecker: found " +
            std::to_string(contradictions.size()) +
            " contradiction(s) in rule base -- attempting resolution");

        int resolved = 0;
        int flagged = 0;

        for (const auto& c : contradictions) {
            auto outcome = resolve_contradiction(c);

            switch (outcome) {
            case ResolutionOutcome::RESOLVED:
                ++resolved;
                ++total_resolved_;
                LOG_INFO("  Resolved: deprecated weaker rule in pair "
                    + c.rule_id_a + " / " + c.rule_id_b);
                break;

            case ResolutionOutcome::FLAGGED:
                ++flagged;
                ++total_flagged_;
                LOG_WARN("  Flagged for review: " + c.rule_id_a +
                    " vs " + c.rule_id_b +
                    " -- " + c.explanation);
                break;

            case ResolutionOutcome::SKIPPED:
                LOG_WARN("  Skipped (rule not found): "
                    + c.rule_id_a + " / " + c.rule_id_b);
                break;

            case ResolutionOutcome::NO_CONTRADICTION:
                break;
            }
        }

        LOG_INFO("ConsistencyChecker: scan complete -- resolved=" +
            std::to_string(resolved) +
            " flagged=" + std::to_string(flagged));

        // If any rules were deprecated, re-sync Prolog and save
        if (resolved > 0) {
            rule_extractor_.sync_rules_to_prolog();
            rule_store_.save();
        }

        return contradictions;
    }

    // =============================================================================
    // run_maintenance
    // =============================================================================

    int ConsistencyChecker::run_maintenance() {
        LOG_INFO("ConsistencyChecker: running maintenance...");

        // Apply confidence decay
        rule_store_.decay_confidence();

        // Prune rules below threshold
        int pruned = rule_store_.prune();

        // Enforce max rules cap
        int evicted = rule_store_.enforce_limit();

        // Save changes
        rule_store_.save();

        // Re-sync Prolog engine after pruning
        if (pruned > 0 || evicted > 0) {
            rule_extractor_.sync_rules_to_prolog();
            LOG_INFO("ConsistencyChecker: pruned=" + std::to_string(pruned) +
                " evicted=" + std::to_string(evicted) +
                " â€” re-synced Prolog engine");
        }

        ++total_maintenance_runs_;

        auto stats = rule_store_.stats();
        LOG_INFO("ConsistencyChecker: maintenance complete â€” "
            + std::to_string(stats.active_rules) + " active rules, "
            + std::to_string(stats.pruned_rules) + " below threshold");

        return pruned + evicted;
    }

    // =============================================================================
    // Internal pipeline
    // =============================================================================

    ExtractionResult ConsistencyChecker::handle_rule_extraction(
        const ConsistencyCheckInput& input)
    {
        ExtractionInput ei;
        ei.feeling = input.feeling;
        ei.user_message = input.user_message;
        ei.response_text = input.response_text;
        ei.episode_id = input.episode_id;

        auto result = rule_extractor_.extract(ei);

        if (result.committed) {
            LOG_INFO("ConsistencyChecker: rule committed from episode " +
                input.episode_id + " â€” id=" + result.committed_rule_id);
        }
        else if (result.extracted && !result.committed) {
            LOG_DEBUG("ConsistencyChecker: rule extracted but not committed â€” " +
                result.rejection_reason);
        }

        return result;
    }

    std::vector<ContradictionResult> ConsistencyChecker::handle_contradiction_check(
        const ConsistencyCheckInput& input)
    {
        std::vector<ContradictionResult> results;

        auto domain_rules = rule_store_.get_top_rules(
            input.feeling.reasoning_domain, 20);

        for (const auto& rule : domain_rules) {
            auto cr = symbolic_engine_.check_contradiction(
                rule.domain, rule.condition, rule.consequence);

            if (!cr.has_contradiction) continue;

            results.push_back(cr);

            // Attempt auto-resolution
            auto outcome = resolve_contradiction(cr);

            switch (outcome) {
            case ResolutionOutcome::RESOLVED:
                ++total_resolved_;
                LOG_INFO("ConsistencyChecker: auto-resolved contradiction "
                    "between " + cr.rule_id_a + " and " + cr.rule_id_b);
                break;

            case ResolutionOutcome::FLAGGED:
                ++total_flagged_;
                LOG_WARN("ConsistencyChecker: contradiction flagged for review "
                    "(confidence delta too small) -- "
                    + cr.rule_id_a + " vs " + cr.rule_id_b);
                // Still penalize both slightly to surface the conflict
                if (!cr.rule_id_a.empty())
                    rule_store_.update_confidence(cr.rule_id_a, -0.05f);
                if (!cr.rule_id_b.empty())
                    rule_store_.update_confidence(cr.rule_id_b, -0.05f);
                break;

            case ResolutionOutcome::SKIPPED:
                LOG_WARN("ConsistencyChecker: contradiction resolution skipped "
                    "-- one or both rules not found: "
                    + cr.rule_id_a + " / " + cr.rule_id_b);
                break;

            case ResolutionOutcome::NO_CONTRADICTION:
                break;
            }
        }

        if (!results.empty()) {
            LOG_WARN("ConsistencyChecker: " + std::to_string(results.size()) +
                " contradiction(s) in domain " +
                input.feeling.reasoning_domain);
        }

        return results;
    }

    bool ConsistencyChecker::run_periodic_maintenance() {
        LOG_DEBUG("ConsistencyChecker: periodic maintenance triggered");
        run_maintenance();
        return true;
    }

    std::string ConsistencyChecker::build_summary(
        const ConsistencyCheckResult& result) const
    {
        std::ostringstream oss;
        oss << "check #" << total_checks_
            << " [" << config_.verifier.mode << "]";

        if (result.rule_extracted) {
            oss << " | rule_extracted";
            if (result.rule_committed)
                oss << " committed=" << result.committed_rule_id;
            else
                oss << " (not committed)";
        }

        if (result.neural_ran) {
            oss << " | neural score="
                << std::to_string(result.neural_result.contradiction_score).substr(0, 4);
        }

        if (result.contradiction_found) {
            oss << " | contradictions=" << result.contradictions.size();
        }

        if (result.contradictions_resolved > 0) {
            oss << " | resolved=" << result.contradictions_resolved;
        }
        if (result.contradictions_flagged > 0) {
            oss << " | flagged_for_review=" << result.contradictions_flagged;
        }

        if (result.periodic_sweep_ran) {
            oss << " | maintenance_ran";
        }

        if (!result.rule_extracted && !result.contradiction_found &&
            !result.neural_ran) {
            oss << " | clean";
        }

        return oss.str();
    }

    // -----------------------------------------------------------------------------
    // resolve_contradiction() implementation.
    // -----------------------------------------------------------------------------

    ResolutionOutcome ConsistencyChecker::resolve_contradiction(
        const ContradictionResult& contradiction,
        float                      resolution_threshold)
    {
        if (!contradiction.has_contradiction) {
            return ResolutionOutcome::NO_CONTRADICTION;
        }

        // Both IDs must be present for confidence-based resolution
        if (contradiction.rule_id_a.empty() || contradiction.rule_id_b.empty()) {
            LOG_DEBUG("ConsistencyChecker::resolve_contradiction: "
                "missing rule IDs -- skipping");
            return ResolutionOutcome::SKIPPED;
        }

        auto rule_a = rule_store_.get_rule(contradiction.rule_id_a);
        auto rule_b = rule_store_.get_rule(contradiction.rule_id_b);

        if (!rule_a.has_value() || !rule_b.has_value()) {
            LOG_DEBUG("ConsistencyChecker::resolve_contradiction: "
                "one or both rules not found in store");
            return ResolutionOutcome::SKIPPED;
        }

        float conf_a = rule_a->confidence;
        float conf_b = rule_b->confidence;
        float delta = std::abs(conf_a - conf_b);

        if (delta < resolution_threshold) {
            // Too close to call -- flag for human review, keep both
            LOG_DEBUG("ConsistencyChecker::resolve_contradiction: "
                "delta=" + std::to_string(delta) +
                " < threshold=" + std::to_string(resolution_threshold) +
                " -- flagging for review");
            return ResolutionOutcome::FLAGGED;
        }

        // Deprecate the lower-confidence rule by setting confidence to 0.0
        // It will be pruned on the next maintenance cycle
        const std::string& loser_id = (conf_a < conf_b)
            ? contradiction.rule_id_a
            : contradiction.rule_id_b;
        const std::string& winner_id = (conf_a < conf_b)
            ? contradiction.rule_id_b
            : contradiction.rule_id_a;
        float loser_conf = std::min(conf_a, conf_b);
        float winner_conf = std::max(conf_a, conf_b);

        // Set confidence to 0.0 -- soft delete, pruned on next maintenance
        // update_confidence uses a delta, so we pass -(current confidence)
        rule_store_.update_confidence(loser_id, -loser_conf);

        LOG_INFO("ConsistencyChecker::resolve_contradiction: "
            "deprecated rule " + loser_id +
            " (conf=" + std::to_string(loser_conf) + ")"
            " in favor of " + winner_id +
            " (conf=" + std::to_string(winner_conf) + ")"
            " delta=" + std::to_string(delta));

        return ResolutionOutcome::RESOLVED;
    }

} // namespace cardinal