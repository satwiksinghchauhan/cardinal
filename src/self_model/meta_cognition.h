// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Meta-Cognition (Layer 2)
// File: src/self_model/meta_cognition.h
//
// Scheduled reflection pass that analyses recent inference failures and
// generates corrective rules stored in RuleStore with type "meta_correction".
//
// Trigger conditions (any fires a reflection pass):
//   1. Every N inferences  (mc_cfg_.trigger_every_n_inferences)
//   2. Contradiction rate  exceeds threshold per domain
//   3. On-demand via API   (CardinalAPI::reflect())
//
// Reflection pass steps:
//   1. Pull recent failure episodes from EpisodicStorage
//   2. Build a structured reflection prompt from those episodes + SelfModel snapshot
//   3. Run a single LLM pass via ILLMBackend::generate_response()
//   4. Parse findings (JSON array) from the LLM response
//   5. For each finding above corrective_rule_confidence:
//        - Insert a "meta_correction" rule into RuleStore
//   6. Persist RuleStore
//   7. Return ReflectionResult
//
// Thread safety:
//   reflect_mutex_ prevents concurrent reflection passes (try_to_lock on hot path).
//   window_mutex_ protects domain_windows_ updated on every inference.
//   ts_mutex_ protects last_reflection_at_ string.
// =============================================================================

#include "self_model/self_model_types.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>

namespace cardinal {

    class EpisodicStorage;
    struct EpisodeRecord;
    class RuleStore;
    class SelfModel;
    struct SelfModelSnapshot;
    class ILLMBackend;

    // -------------------------------------------------------------------------
    // MetaCognitionSettings
    // Extracted from CardinalConfig at construction and stored by value so
    // MetaCognition has no dangling reference if the config is ever reloaded.
    // -------------------------------------------------------------------------
    struct MetaCognitionSettings {
        bool  enabled                            = true;
        int   trigger_every_n_inferences         = 20;
        float trigger_on_contradiction_rate_pct  = 30.0f;
        bool  on_demand_via_api                  = true;
        int   min_failures_to_reflect            = 5;
        int   max_corrective_rules_per_session   = 10;
        float corrective_rule_confidence         = 0.6f;
    };

    // -------------------------------------------------------------------------
    // MetaCognition
    // -------------------------------------------------------------------------
    class MetaCognition {
    public:
        // All dependencies are injected as non-owning references.
        // CardinalAPI owns all subsystems and guarantees their lifetime.
        MetaCognition(const CardinalConfig& config,
                      EpisodicStorage&      storage,
                      RuleStore&            rule_store,
                      SelfModel&            self_model,
                      ILLMBackend&          backend);

        ~MetaCognition() = default;

        MetaCognition(const MetaCognition&)            = delete;
        MetaCognition& operator=(const MetaCognition&) = delete;

        // ------------------------------------------------------------------
        // Called after every inference.
        // Increments counters, checks triggers, runs reflection if triggered.
        // Returns empty ReflectionResult (ran==false) if no trigger fired.
        // ------------------------------------------------------------------
        ReflectionResult on_inference(const std::string& domain,
                                      bool               contradiction,
                                      bool               uncertainty);

        // ------------------------------------------------------------------
        // Force an immediate reflection pass (API / unit test entry point).
        // ------------------------------------------------------------------
        ReflectionResult reflect(const std::string& trigger = "manual");

        // ------------------------------------------------------------------
        // Accessors
        // ------------------------------------------------------------------
        bool        is_enabled()            const { return mc_cfg_.enabled; }
        int         inference_count()       const { return inference_counter_.load(); }
        int         total_reflections()     const;
        int         total_corrective_rules() const;
        std::string last_reflection_at()    const;

        void reset_inference_counter();

    private:
        // -- Trigger checks --------------------------------------------------
        bool should_trigger_by_count() const;
        bool should_trigger_by_contradiction_rate(const std::string& domain) const;

        // -- Core reflection (acquires reflect_mutex_) -----------------------
        ReflectionResult run_reflection(const std::string& trigger);

        // -- Prompt construction ---------------------------------------------
        std::string build_reflection_prompt(
            const std::vector<EpisodeRecord>& failures,
            const SelfModelSnapshot&          snapshot) const;

        // -- Response parsing (nlohmann/json) --------------------------------
        std::vector<ReflectionFinding> parse_findings(
            const std::string& llm_response) const;

        // -- Rule commitment -------------------------------------------------
        int commit_corrective_rules(
            const std::vector<ReflectionFinding>& findings,
            const std::string&                    reflection_timestamp);

        // -- Per-domain rolling contradiction window -------------------------
        struct DomainWindow {
            int inferences     = 0;
            int contradictions = 0;
        };
        std::unordered_map<std::string, DomainWindow> domain_windows_;

        // -- Members ---------------------------------------------------------
        MetaCognitionSettings  mc_cfg_;        // extracted sub-config (by value)
        const CardinalConfig&  config_;        // full config — needed for FeelingContext
        EpisodicStorage&       storage_;
        RuleStore&             rule_store_;
        SelfModel&             self_model_;
        ILLMBackend&           backend_;

        std::atomic<int>  inference_counter_{ 0 };
        std::atomic<int>  total_reflections_{ 0 };
        std::atomic<int>  total_corrective_rules_{ 0 };

        mutable std::mutex reflect_mutex_;
        mutable std::mutex window_mutex_;
        mutable std::mutex ts_mutex_;
        std::string        last_reflection_at_;
    };

} // namespace cardinal
