// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Curriculum Builder (Layer 3)
// File: src/training/curriculum_builder.h
// =============================================================================

#include "self_model/self_model_types.h"   // DomainStats, LoRAConfig
#include "utils/config_loader.h"

#include <chrono>
#include <mutex>                           // std::mutex — was missing
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cardinal {

    class SelfModel;

    // -------------------------------------------------------------------------
    // CurriculumPlan
    // Output of CurriculumBuilder::build_plan().
    // -------------------------------------------------------------------------
    struct CurriculumPlan {
        std::string target_domain;       // "" = all domains
        std::string reason;
        LoRAConfig  lora_cfg;
        float       min_episode_confidence = 0.0f;
        int         max_episodes           = 0;    // 0 = use global default
        float       priority_score         = 0.0f;

        bool is_domain_focused() const { return !target_domain.empty(); }
    };

    // -------------------------------------------------------------------------
    // CurriculumBuilder
    // -------------------------------------------------------------------------
    class CurriculumBuilder {
    public:
        CurriculumBuilder(const CardinalConfig& config,
                          SelfModel&            self_model);

        // Build the next training plan from current SelfModel state.
        // Never throws — returns a fallback "general" plan on any error.
        CurriculumPlan build_plan() const;

        // Quick check: is any domain weak enough to warrant a focused run?
        // Returns ("", false) if none exceeds the weakness threshold.
        std::pair<std::string, bool> should_focus_domain() const;

        // Record that a training cycle just completed for a domain.
        void record_trained(const std::string& domain);

        float weakness_threshold() const { return weakness_threshold_; }

    private:
        LoRAConfig build_lora_cfg(float weakness_score) const;
        float      priority_score(const DomainStats& ds) const;
        long       seconds_since_trained(const std::string& domain) const;

        LoRAConfig base_lora_cfg_;
        float      weakness_threshold_;
        int        cooldown_seconds_;

        SelfModel& self_model_;

        mutable std::unordered_map<
            std::string,
            std::chrono::steady_clock::time_point> last_trained_;

        mutable std::mutex last_trained_mutex_;
    };

} // namespace cardinal
