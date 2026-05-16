// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Adapter Evaluator (Layer 3)
// File: src/training/adapter_evaluator.h
// =============================================================================

#include "self_model/self_model_types.h"   // TrainingResult
#include "memory/episodic_storage.h"       // EpisodeRecord
#include "utils/config_loader.h"

#include <mutex>
#include <string>
#include <vector>

namespace cardinal {

    class ITrainingBackend;

    // -------------------------------------------------------------------------
    // EvalResult
    // -------------------------------------------------------------------------
    struct EvalResult {
        bool        approved         = false;
        bool        loaded           = false;
        bool        pending          = false;
        float       eval_score       = 0.0f;
        float       baseline_score   = 0.0f;
        float       improvement_pct  = 0.0f;
        float       threshold_pct    = 0.0f;
        std::string adapter_path;
        std::string load_policy;
        std::string error_message;
    };

    // -------------------------------------------------------------------------
    // AdapterEvaluator
    // -------------------------------------------------------------------------
    class AdapterEvaluator {
    public:
        AdapterEvaluator(const CardinalConfig& config,
                         ITrainingBackend&     backend);

        ~AdapterEvaluator() = default;

        AdapterEvaluator(const AdapterEvaluator&)            = delete;
        AdapterEvaluator& operator=(const AdapterEvaluator&) = delete;

        // Evaluate adapter, apply improvement threshold gate, honour load policy.
        // Never throws — errors land in EvalResult.error_message.
        EvalResult evaluate_and_gate(
            const std::string&                adapter_path,
            const std::vector<EpisodeRecord>& eval_episodes);

        // Apply pending adapter at session boundary.
        // Returns true if an adapter was loaded.
        bool apply_pending_adapter();

        // Accessors
        std::string pending_adapter_path() const;
        bool        has_pending_adapter()  const;
        float       improvement_threshold_pct() const { return improvement_threshold_pct_; }
        std::string load_policy() const { return load_policy_; }

    private:
        ITrainingBackend& backend_;
        float             improvement_threshold_pct_;
        std::string       load_policy_;

        mutable std::mutex pending_mutex_;
        std::string        pending_adapter_path_;
    };

} // namespace cardinal
