// =============================================================================
// Cardinal - Curriculum Builder — Implementation
// File: src/training/curriculum_builder.cpp
// =============================================================================

#include "training/curriculum_builder.h"
#include "self_model/self_model.h"
#include "utils/logger.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <sstream>

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

CurriculumBuilder::CurriculumBuilder(const CardinalConfig& config,
                                     SelfModel&            self_model)
    : self_model_(self_model)
{
    const auto& tc = config.self_improvement.training;

    // Base LoRA config from global training settings.
    base_lora_cfg_.rank           = tc.lora_rank;
    base_lora_cfg_.alpha          = tc.lora_alpha;
    base_lora_cfg_.learning_rate  = tc.learning_rate;
    base_lora_cfg_.epochs         = tc.epochs;
    base_lora_cfg_.batch_size     = tc.batch_size;

    // Weakness threshold: domains above this score are candidates for focus.
    // Default 0.4 — roughly "contradicting or uncertain on ~40% of inferences
    // in a domain, or averaging 60% confidence".
    weakness_threshold_ = 0.4f;

    // Cooldown: don't re-select a domain for at least N seconds after training.
    // Default 3600s (1 hour) — prevents back-to-back cycles on the same domain
    // when the improvement hasn't had time to show up in the self-model.
    cooldown_seconds_ = 3600;

    LOG_INFO("CurriculumBuilder: initialised (weakness_threshold=" +
             std::to_string(weakness_threshold_) +
             ", cooldown_seconds=" + std::to_string(cooldown_seconds_) + ")");
}

// ---------------------------------------------------------------------------
// build_plan()
// ---------------------------------------------------------------------------

CurriculumPlan CurriculumBuilder::build_plan() const {
    CurriculumPlan plan;

    try {
        auto all_stats = self_model_.get_all_domain_stats();

        if (all_stats.empty()) {
            // Not enough data yet — return a general plan to warm up.
            plan.target_domain          = "";
            plan.reason                 = "no domain data yet — general warm-up";
            plan.lora_cfg               = base_lora_cfg_;
            plan.min_episode_confidence = 0.7f;
            plan.priority_score         = 0.0f;
            return plan;
        }

        // Score each domain and find the highest-priority candidate that is
        // not in cooldown.
        std::string best_domain;
        float       best_score = -1.0f;

        for (const auto& ds : all_stats) {
            // Skip domains with too few inferences to be meaningful.
            if (ds.total_inferences < 5) continue;

            // Skip domains in cooldown.
            if (seconds_since_trained(ds.domain) < static_cast<long>(cooldown_seconds_)) {
                LOG_DEBUG("CurriculumBuilder: domain '" + ds.domain +
                          "' in cooldown, skipping");
                continue;
            }

            float score = priority_score(ds);
            if (score > best_score) {
                best_score  = score;
                best_domain = ds.domain;
            }
        }

        if (best_domain.empty() || best_score < weakness_threshold_) {
            // No domain weak enough — train on the full corpus.
            plan.target_domain          = "";
            plan.reason                 = "no domain exceeds weakness threshold (" +
                                          std::to_string(weakness_threshold_) +
                                          ") — general training";
            plan.lora_cfg               = base_lora_cfg_;
            plan.min_episode_confidence = 0.7f;
            plan.max_episodes           = 0;
            plan.priority_score         = best_score;
        } else {
            // Focused domain run.
            // Find the DomainStats for logging / LoRA tuning.
            const DomainStats* target_ds = nullptr;
            for (const auto& ds : all_stats) {
                if (ds.domain == best_domain) { target_ds = &ds; break; }
            }

            std::ostringstream reason;
            reason << "domain '" << best_domain << "' priority_score="
                   << std::to_string(best_score);
            if (target_ds) {
                reason << " (contradiction_rate="
                       << std::to_string(target_ds->contradiction_rate)
                       << ", avg_conf="
                       << std::to_string(target_ds->avg_confidence) << ")";
            }

            plan.target_domain          = best_domain;
            plan.reason                 = reason.str();
            plan.lora_cfg               = build_lora_cfg(best_score);
            // Include some lower-confidence episodes so the model sees failures.
            plan.min_episode_confidence = 0.4f;
            plan.max_episodes           = 200; // cap to keep training time bounded
            plan.priority_score         = best_score;
        }

    } catch (const std::exception& ex) {
        LOG_ERROR("CurriculumBuilder::build_plan exception: " +
                  std::string(ex.what()) + " — falling back to general plan");
        plan.target_domain          = "";
        plan.reason                 = "fallback (exception in build_plan)";
        plan.lora_cfg               = base_lora_cfg_;
        plan.min_episode_confidence = 0.7f;
    }

    LOG_INFO("CurriculumBuilder: plan — domain='" + plan.target_domain +
             "' reason='" + plan.reason + "'");
    return plan;
}

// ---------------------------------------------------------------------------
// should_focus_domain()
// ---------------------------------------------------------------------------

std::pair<std::string, bool> CurriculumBuilder::should_focus_domain() const {
    try {
        auto weakest = self_model_.get_weakest_domains(1);
        if (weakest.empty()) return {"", false};

        const auto& ds = weakest.front();
        if (ds.total_inferences < 5) return {"", false};
        if (ds.weakness_score() < weakness_threshold_) return {"", false};
        if (seconds_since_trained(ds.domain) < static_cast<long>(cooldown_seconds_)) {
            return {"", false};
        }
        return {ds.domain, true};

    } catch (...) {
        return {"", false};
    }
}

// ---------------------------------------------------------------------------
// record_trained()
// ---------------------------------------------------------------------------

void CurriculumBuilder::record_trained(const std::string& domain) {
    std::lock_guard<std::mutex> lock(last_trained_mutex_);
    last_trained_[domain] = std::chrono::steady_clock::now();
    LOG_DEBUG("CurriculumBuilder: recorded training for domain '" + domain + "'");
}

// ---------------------------------------------------------------------------
// build_lora_cfg()
// ---------------------------------------------------------------------------

LoRAConfig CurriculumBuilder::build_lora_cfg(float ws) const {
    LoRAConfig cfg = base_lora_cfg_;

    if (ws > 0.6f) {
        // Severely weak domain: more epochs, lower LR to avoid over-fitting
        // on noisy failure data.
        cfg.epochs        = std::min(base_lora_cfg_.epochs + 1, 6);
        cfg.learning_rate = base_lora_cfg_.learning_rate * 0.5f;
        LOG_DEBUG("CurriculumBuilder: applying conservative LoRA overrides for "
                  "high-weakness domain (score=" + std::to_string(ws) + ")");
    } else if (ws > 0.4f) {
        // Moderately weak: slight LR reduction only.
        cfg.learning_rate = base_lora_cfg_.learning_rate * 0.75f;
    }
    // Below 0.4 (general plan): use base config unchanged.

    return cfg;
}

// ---------------------------------------------------------------------------
// priority_score()
// ---------------------------------------------------------------------------

float CurriculumBuilder::priority_score(const DomainStats& ds) const {
    // Start with the DomainStats weakness score [0, 1].
    float score = ds.weakness_score();

    // Apply a recency bonus: domains not trained recently get up to +0.2
    // on top of their weakness score, decaying linearly from cooldown to
    // 5× cooldown. This prevents a domain from being perpetually skipped
    // if its weakness score is just below the threshold.
    long secs = seconds_since_trained(ds.domain);
    if (secs > cooldown_seconds_) {
        long   bonus_window   = static_cast<long>(cooldown_seconds_) * 4;
        float  recency_bonus  = std::min(
            0.2f,
            0.2f * static_cast<float>(secs - cooldown_seconds_) /
                   static_cast<float>(bonus_window));
        score += recency_bonus;
    }

    return std::min(score, 1.0f);
}

// ---------------------------------------------------------------------------
// seconds_since_trained()
// ---------------------------------------------------------------------------

long CurriculumBuilder::seconds_since_trained(const std::string& domain) const {
    std::lock_guard<std::mutex> lock(last_trained_mutex_);
    auto it = last_trained_.find(domain);
    if (it == last_trained_.end()) {
        // Never trained — treat as very old.
        return LONG_MAX;
    }
    auto elapsed = std::chrono::steady_clock::now() - it->second;
    return static_cast<long>(
        std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
}

} // namespace cardinal
