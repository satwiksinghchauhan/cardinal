// =============================================================================
// Cardinal - Adapter Evaluator — Implementation
// File: src/training/adapter_evaluator.cpp
// =============================================================================

#include "training/adapter_evaluator.h"
#include "training/i_training_backend.h"
#include "utils/logger.h"

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

AdapterEvaluator::AdapterEvaluator(const CardinalConfig& config,
                                   ITrainingBackend&     backend)
    : backend_(backend)
{
    const auto& tc = config.self_improvement.training;
    improvement_threshold_pct_ = tc.eval_improvement_threshold_pct;
    load_policy_               = tc.adapter_load_policy;

    if (load_policy_ != "immediate" && load_policy_ != "session_boundary") {
        LOG_INFO("AdapterEvaluator: unknown load_policy '" + load_policy_ +
                 "' — defaulting to 'session_boundary'");
        load_policy_ = "session_boundary";
    }

    LOG_INFO("AdapterEvaluator: initialised (threshold=" +
             std::to_string(improvement_threshold_pct_) +
             "%, policy=" + load_policy_ + ")");
}

// ---------------------------------------------------------------------------
// evaluate_and_gate()
// ---------------------------------------------------------------------------

EvalResult AdapterEvaluator::evaluate_and_gate(
        const std::string&                adapter_path,
        const std::vector<EpisodeRecord>& eval_episodes) {

    EvalResult result;
    result.adapter_path  = adapter_path;
    result.load_policy   = load_policy_;
    result.threshold_pct = improvement_threshold_pct_;

    TrainingResult tr = backend_.evaluate(adapter_path, eval_episodes);

    if (!tr.success) {
        result.error_message = "backend evaluate() failed: " + tr.error_message;
        LOG_ERROR("AdapterEvaluator: " + result.error_message);
        return result;
    }

    result.eval_score      = tr.eval_score;
    result.baseline_score  = tr.baseline_score;
    result.improvement_pct = tr.improvement_pct;

    LOG_INFO("AdapterEvaluator: eval complete — baseline=" +
             std::to_string(result.baseline_score) +
             " eval=" + std::to_string(result.eval_score) +
             " improvement=" + std::to_string(result.improvement_pct) +
             "% (threshold=" + std::to_string(improvement_threshold_pct_) + "%)");

    if (result.improvement_pct < improvement_threshold_pct_) {
        result.approved      = false;
        result.error_message = "improvement " +
            std::to_string(result.improvement_pct) +
            "% below threshold " +
            std::to_string(improvement_threshold_pct_) +
            "% — adapter rejected";
        LOG_INFO("AdapterEvaluator: " + result.error_message);
        return result;
    }

    result.approved = true;

    if (load_policy_ == "immediate") {
        LOG_INFO("AdapterEvaluator: policy=immediate — loading adapter now");
        TrainingResult load_r = backend_.load_adapter(adapter_path);
        if (load_r.success) {
            result.loaded = true;
            LOG_INFO("AdapterEvaluator: adapter loaded immediately from " + adapter_path);
        } else {
            result.error_message = "immediate load failed: " + load_r.error_message;
            LOG_ERROR("AdapterEvaluator: " + result.error_message);
        }
    } else {
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_adapter_path_ = adapter_path;
        }
        result.pending = true;
        LOG_INFO("AdapterEvaluator: policy=session_boundary — adapter queued: " +
                 adapter_path);
    }

    return result;
}

// ---------------------------------------------------------------------------
// apply_pending_adapter()
// ---------------------------------------------------------------------------

bool AdapterEvaluator::apply_pending_adapter() {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_adapter_path_.empty()) return false;
        path = pending_adapter_path_;
    }

    LOG_INFO("AdapterEvaluator: applying pending adapter at session boundary: " + path);

    TrainingResult r = backend_.load_adapter(path);
    if (r.success) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_adapter_path_.clear();
        LOG_INFO("AdapterEvaluator: pending adapter loaded successfully");
        return true;
    }

    LOG_ERROR("AdapterEvaluator: pending adapter load failed: " + r.error_message);
    return false;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::string AdapterEvaluator::pending_adapter_path() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return pending_adapter_path_;
}

bool AdapterEvaluator::has_pending_adapter() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return !pending_adapter_path_.empty();
}

} // namespace cardinal
