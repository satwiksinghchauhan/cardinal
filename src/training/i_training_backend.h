// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Training Backend Interface (Layer 3)
// File: src/training/i_training_backend.h
// =============================================================================

#include "self_model/self_model_types.h"   // TrainingDataset, TrainingResult, LoRAConfig
#include "memory/episodic_storage.h"       // EpisodeRecord — needed for evaluate()

#include <string>
#include <functional>
#include <vector>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ProgressCallback
    // Return false to request graceful abort.
    // -------------------------------------------------------------------------
    using ProgressCallback = std::function<bool(int step, int total_steps, float loss)>;

    // -------------------------------------------------------------------------
    // ITrainingBackend
    // -------------------------------------------------------------------------
    class ITrainingBackend {
    public:
        virtual ~ITrainingBackend() = default;

        virtual std::string name()              const = 0;
        virtual bool        can_train_locally() const = 0;

        virtual TrainingResult prepare(const TrainingDataset& dataset,
                                       const LoRAConfig&      lora_cfg) = 0;

        virtual TrainingResult train(const TrainingDataset& dataset,
                                     const LoRAConfig&      lora_cfg,
                                     ProgressCallback       progress_cb = nullptr) = 0;

        virtual TrainingResult evaluate(const std::string&                adapter_path,
                                        const std::vector<EpisodeRecord>& eval_episodes) = 0;

        virtual TrainingResult load_adapter(const std::string& adapter_path) = 0;

        virtual void unload_adapter() = 0;

        virtual bool        has_adapter()         const = 0;
        virtual std::string active_adapter_path() const = 0;

        // Convenience: full pipeline in one call.
        // Default implementation chains the four virtual methods.
        virtual TrainingResult run_full_cycle(
            const TrainingDataset&            dataset,
            const LoRAConfig&                 lora_cfg,
            const std::vector<EpisodeRecord>& eval_episodes,
            float                             improvement_threshold_pct,
            ProgressCallback                  progress_cb = nullptr);
    };

    inline TrainingResult ITrainingBackend::run_full_cycle(
            const TrainingDataset&            dataset,
            const LoRAConfig&                 lora_cfg,
            const std::vector<EpisodeRecord>& eval_episodes,
            float                             improvement_threshold_pct,
            ProgressCallback                  progress_cb) {

        TrainingResult r = prepare(dataset, lora_cfg);
        if (!r.success) return r;

        r = train(dataset, lora_cfg, std::move(progress_cb));
        if (!r.success) return r;

        if (!can_train_locally()) return r;

        std::string adapter_path = r.adapter_path;
        r = evaluate(adapter_path, eval_episodes);
        if (!r.success) return r;

        r.improvement_pct = (r.baseline_score > 0.0f)
            ? ((r.eval_score - r.baseline_score) / r.baseline_score) * 100.0f
            : 0.0f;

        if (r.improvement_pct >= improvement_threshold_pct) {
            TrainingResult load_r = load_adapter(adapter_path);
            r.adapter_loaded = load_r.success;
            if (!load_r.success)
                r.error_message = "load_adapter failed: " + load_r.error_message;
        } else {
            r.error_message = "improvement " + std::to_string(r.improvement_pct) +
                              "% below threshold " +
                              std::to_string(improvement_threshold_pct) +
                              "% — adapter not loaded";
        }
        return r;
    }

} // namespace cardinal
