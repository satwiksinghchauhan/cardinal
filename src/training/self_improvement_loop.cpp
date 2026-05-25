// =============================================================================
// Cardinal - Self-Improvement Loop — Implementation
// File: src/training/self_improvement_loop.cpp
// =============================================================================

#include "training/self_improvement_loop.h"
#include "training/curriculum_builder.h"
#include "training/dataset_curator.h"
#include "training/training_factory.h"
#include "training/adapter_evaluator.h"
#include "self_model/self_model.h"
#include "self_model/meta_cognition.h"
#include "memory/episodic_storage.h"
#include "memory/rule_store.h"
#include "core/llm_backend.h"
#include "utils/logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

std::string utc_now_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

} // anonymous namespace

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SelfImprovementLoop::SelfImprovementLoop(const CardinalConfig& config,
                                         EpisodicStorage&      storage,
                                         RuleStore&            rule_store,
                                         ILLMBackend&          backend)
    : storage_(storage)
    , rule_store_(rule_store)
    , backend_(backend)
    , config_(config)
{
    const auto& si = config.self_improvement;

    cfg_.enabled                        = si.enabled;
    cfg_.self_model_enabled             = si.self_model.enabled;
    cfg_.inject_into_prompt             = si.self_model.inject_into_prompt;
    cfg_.meta_cognition_enabled         = si.meta_cognition.enabled;
    cfg_.training_enabled               = si.training.enabled;
    cfg_.trigger_every_n_episodes       = si.training.trigger_every_n_episodes;
    cfg_.trigger_every_n_hours          = si.training.trigger_every_n_hours;
    cfg_.trigger_on_domain_conf_below   = si.training.trigger_on_domain_confidence_below;
    cfg_.eval_improvement_threshold_pct = si.training.eval_improvement_threshold_pct;
    cfg_.adapter_load_policy            = si.training.adapter_load_policy;
    cfg_.min_episodes_for_training      = si.training.min_episodes_for_training;

    last_training_time_ = std::chrono::steady_clock::now();

    LOG_INFO("SelfImprovementLoop: constructed (enabled=" +
             std::string(cfg_.enabled ? "true" : "false") + ")");
}

SelfImprovementLoop::~SelfImprovementLoop() {
    if (running_.load()) {
        stop();
    }
}

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

void SelfImprovementLoop::start() {
    if (!cfg_.enabled) {
        LOG_INFO("SelfImprovementLoop: self-improvement disabled in config — not starting");
        return;
    }

    if (running_.load()) {
        LOG_INFO("SelfImprovementLoop: already running");
        return;
    }

    // ── Construct Layer 1 ─────────────────────────────────────────────────
    if (cfg_.self_model_enabled) {
        self_model_ = std::make_unique<SelfModel>(config_);
        self_model_->open();
        LOG_INFO("SelfImprovementLoop: Layer 1 (SelfModel) started");
    }

    // ── Construct Layer 2 ─────────────────────────────────────────────────
    if (cfg_.meta_cognition_enabled && self_model_) {
        meta_cognition_ = std::make_unique<MetaCognition>(
            config_, storage_, rule_store_, *self_model_, backend_);
        LOG_INFO("SelfImprovementLoop: Layer 2 (MetaCognition) started");
    }

    // ── Construct Layer 3 ─────────────────────────────────────────────────
    if (cfg_.training_enabled) {
        try {
            trainer_   = TrainingFactory::create(config_, backend_);
            curriculum_ = std::make_unique<CurriculumBuilder>(config_, *self_model_);
            curator_    = std::make_unique<DatasetCurator>(config_, storage_, rule_store_);
            evaluator_  = std::make_unique<AdapterEvaluator>(config_, *trainer_);
            LOG_INFO("SelfImprovementLoop: Layer 3 (Training) started — backend=" +
                     trainer_->name());
        } catch (const std::exception& ex) {
            LOG_ERROR("SelfImprovementLoop: Layer 3 init failed: " +
                      std::string(ex.what()) + " — training disabled for this run");
            cfg_.training_enabled = false;
        }
    }

    // ── Start training thread ─────────────────────────────────────────────
    running_.store(true);
    training_thread_ = std::thread(&SelfImprovementLoop::training_thread_fn, this);

    LOG_INFO("SelfImprovementLoop: started");
}

// ---------------------------------------------------------------------------
// stop()
// ---------------------------------------------------------------------------

void SelfImprovementLoop::stop() {
    if (!running_.load()) return;

    LOG_INFO("SelfImprovementLoop: stopping...");

    {
        std::lock_guard<std::mutex> lock(training_mutex_);
        stop_requested_ = true;
    }
    training_cv_.notify_one();

    if (training_thread_.joinable()) {
        training_thread_.join();
    }

    if (self_model_ && self_model_->is_open()) {
        self_model_->close();
    }

    running_.store(false);
    LOG_INFO("SelfImprovementLoop: stopped");
}

// ---------------------------------------------------------------------------
// training_thread_fn()
// ---------------------------------------------------------------------------

void SelfImprovementLoop::training_thread_fn() {
    LOG_INFO("SelfImprovementLoop: training thread started");

    // Wake every 60 seconds to check the interval trigger, even if no
    // explicit request was posted.
    const auto poll_interval = std::chrono::seconds(60);

    while (true) {
        TrainingRequest req;
        bool            do_train = false;

        {
            std::unique_lock<std::mutex> lock(training_mutex_);
            training_cv_.wait_for(lock, poll_interval, [this] {
                return training_requested_ || stop_requested_;
            });

            if (stop_requested_) break;

            if (training_requested_) {
                req             = pending_request_;
                do_train        = true;
                training_requested_ = false;
            }
        }

        // Even if no explicit request, check the time-based interval trigger.
        if (!do_train && check_interval_trigger()) {
            req      = { "interval", "" };
            do_train = true;
        }

        if (!do_train) continue;
        if (!cfg_.training_enabled) continue;

        // Gate: need minimum episodes before training is worthwhile.
        int total_eps = storage_.count();
        if (total_eps < cfg_.min_episodes_for_training) {
            LOG_INFO("SelfImprovementLoop: training trigger fired but only " +
                     std::to_string(total_eps) + "/" +
                     std::to_string(cfg_.min_episodes_for_training) +
                     " episodes — deferring");
            continue;
        }

        training_in_progress_.store(true);
        TrainingResult result = run_training_cycle(req);
        training_in_progress_.store(false);

        // Update stats.
        {
            std::lock_guard<std::mutex> sl(stats_mutex_);
            ++total_training_runs_;
            last_training_at_    = utc_now_str();
            last_improvement_pct_ = result.improvement_pct;
            if (result.adapter_loaded) {
                active_adapter_path_ = result.adapter_path;
            }
        }

        // Update timing so the interval trigger resets.
        {
            std::lock_guard<std::mutex> tl(timing_mutex_);
            last_training_time_ = std::chrono::steady_clock::now();
        }

        // Notify curriculum builder so this domain enters cooldown.
        if (curriculum_ && !result.trigger.empty()) {
            curriculum_->record_trained(result.trigger);
        }
    }

    LOG_INFO("SelfImprovementLoop: training thread exiting");
}

// ---------------------------------------------------------------------------
// run_training_cycle()
// ---------------------------------------------------------------------------

TrainingResult SelfImprovementLoop::run_training_cycle(const TrainingRequest& req) {
    LOG_INFO("SelfImprovementLoop: starting training cycle (trigger=" +
             req.trigger + ", domain_hint='" + req.domain_hint + "')");

    TrainingResult result;
    result.trigger    = req.trigger;
    result.timestamp  = utc_now_str();

    try {
        // Step 1 — Build curriculum plan.
        CurriculumPlan plan = curriculum_->build_plan();
        if (!req.domain_hint.empty()) {
            plan.target_domain = req.domain_hint;
            plan.reason        = "manual trigger with domain_hint=" + req.domain_hint;
        }

        LOG_INFO("SelfImprovementLoop: curriculum plan — domain='" +
                 plan.target_domain + "' reason='" + plan.reason + "'");

        // Step 2 — Curate dataset.
        CurationStats cstats;
        TrainingDataset dataset = curator_->curate(plan, cstats);

        if (dataset.examples.empty()) {
            result.error_message = "dataset is empty after curation — skipping cycle";
            LOG_INFO("SelfImprovementLoop: " + result.error_message);
            return result;
        }

        LOG_INFO("SelfImprovementLoop: dataset ready — " +
                 std::to_string(cstats.final_count) + " examples");

        // Step 3 — Prepare + train.
        TrainingResult prep_r = trainer_->prepare(dataset, plan.lora_cfg);
        if (!prep_r.success) {
            result.error_message = "prepare() failed: " + prep_r.error_message;
            LOG_ERROR("SelfImprovementLoop: " + result.error_message);
            return result;
        }

        TrainingResult train_r = trainer_->train(dataset, plan.lora_cfg,
            [](int step, int total, float loss) -> bool {
                // Log every 10 steps to avoid spam.
                if (step % 10 == 0 || step == total) {
                    LOG_DEBUG("SelfImprovementLoop: training step " +
                              std::to_string(step) + "/" +
                              std::to_string(total) +
                              " loss=" + std::to_string(loss));
                }
                return true; // continue
            });

        if (!train_r.success) {
            result.error_message = "train() failed: " + train_r.error_message;
            LOG_ERROR("SelfImprovementLoop: " + result.error_message);
            return result;
        }

        result = train_r;  // carry adapter_path, examples_trained, duration_seconds

        // For script-export backends, nothing more to do locally.
        if (!trainer_->can_train_locally()) {
            LOG_INFO("SelfImprovementLoop: script-export mode — cycle complete, "
                     "script at " + train_r.script_path);
            result.success = true;
            return result;
        }

        // Step 4 — Evaluate + gate + load (via AdapterEvaluator).
        auto holdout = curator_->get_eval_holdout();
        EvalResult eval = evaluator_->evaluate_and_gate(train_r.adapter_path, holdout);

        result.eval_score      = eval.eval_score;
        result.baseline_score  = eval.baseline_score;
        result.improvement_pct = eval.improvement_pct;
        result.adapter_loaded  = eval.loaded;

        if (!eval.error_message.empty()) {
            result.error_message = eval.error_message;
        }

        if (eval.pending) {
            LOG_INFO("SelfImprovementLoop: adapter queued for session boundary: " +
                     train_r.adapter_path);
        } else if (eval.loaded) {
            std::lock_guard<std::mutex> sl(stats_mutex_);
            active_adapter_path_ = train_r.adapter_path;
            LOG_INFO("SelfImprovementLoop: adapter loaded immediately");
        }

    } catch (const std::exception& ex) {
        result.error_message = std::string("exception in training cycle: ") + ex.what();
        LOG_ERROR("SelfImprovementLoop: " + result.error_message);
    }

    LOG_INFO("SelfImprovementLoop: training cycle complete — improvement=" +
             std::to_string(result.improvement_pct) + "% loaded=" +
             std::string(result.adapter_loaded ? "yes" : "no"));

    return result;
}

// ---------------------------------------------------------------------------
// on_inference() — inference thread hook
// ---------------------------------------------------------------------------

void SelfImprovementLoop::on_inference(const std::string& domain,
                                        const std::string& reasoning_type,
                                        float              confidence,
                                        bool               contradiction,
                                        bool               uncertainty,
                                        bool               rule_committed) {
    if (!cfg_.enabled) return;

    // Layer 1 — record in SelfModel (fast SQLite UPSERT, own mutex).
    if (self_model_ && cfg_.self_model_enabled) {
        self_model_->record_inference(domain, reasoning_type, confidence,
                                      contradiction, uncertainty, rule_committed);
    }

    // Layer 2 — MetaCognition (may run a synchronous reflection pass).
    if (meta_cognition_ && cfg_.meta_cognition_enabled) {
        // Result is intentionally discarded here; CardinalAPI surfaces it
        // via the /reflect endpoint. If a reflection ran, its corrective
        // rules are already committed to RuleStore.
        meta_cognition_->on_inference(domain, contradiction, uncertainty);
    }

    // Layer 3 — increment episode counter and check triggers.
    if (!cfg_.training_enabled) return;

    int count = ++episode_counter_;
    (void)count;

    // Coalesced trigger: only post if not already training.
    if (training_in_progress_.load()) return;

    std::string trigger;
    std::string domain_hint;

    if (check_episode_count_trigger()) {
        trigger = "episode_count";
    } else if (check_confidence_trigger()) {
        trigger     = "confidence";
        domain_hint = domain;
    }

    if (!trigger.empty()) {
        post_training_request({ trigger, domain_hint });
    }
}

// ---------------------------------------------------------------------------
// on_session_boundary()
// ---------------------------------------------------------------------------

void SelfImprovementLoop::on_session_boundary() {
    if (!cfg_.enabled || !evaluator_) return;
    evaluator_->apply_pending_adapter();
}

// ---------------------------------------------------------------------------
// trigger_reflection() — manual Layer 2 trigger
// ---------------------------------------------------------------------------

ReflectionResult SelfImprovementLoop::trigger_reflection() {
    if (!meta_cognition_) {
        ReflectionResult r;
        r.error_message = "MetaCognition not initialised (disabled in config)";
        return r;
    }
    return meta_cognition_->reflect("manual");
}

// ---------------------------------------------------------------------------
// trigger_training() — manual Layer 3 trigger
// ---------------------------------------------------------------------------

bool SelfImprovementLoop::trigger_training(const std::string& domain_hint) {
    if (!cfg_.training_enabled || !trainer_) return false;
    if (training_in_progress_.load()) return false;
    post_training_request({ "manual", domain_hint });
    return true;
}

// ---------------------------------------------------------------------------
// format_self_model_for_prompt()
// ---------------------------------------------------------------------------

std::string SelfImprovementLoop::format_self_model_for_prompt() const {
    if (!self_model_ || !cfg_.self_model_enabled || !cfg_.inject_into_prompt) {
        return "";
    }
    return self_model_->format_for_prompt();
}

// ---------------------------------------------------------------------------
// get_status()
// ---------------------------------------------------------------------------

SelfImprovementStatus SelfImprovementLoop::get_status() const {
    SelfImprovementStatus s;

    // Layer 1.
    s.self_model_enabled = cfg_.self_model_enabled && self_model_ != nullptr;
    if (self_model_) {
        auto snap           = self_model_->get_snapshot();
        s.weakest_domain    = snap.weakest_domain();
        s.strongest_domain  = snap.strongest_domain();
        s.total_domain_stats = static_cast<int>(snap.domain_stats.size());
    }

    // Layer 2.
    s.meta_cognition_enabled = cfg_.meta_cognition_enabled && meta_cognition_ != nullptr;
    if (meta_cognition_) {
        s.total_reflections      = meta_cognition_->total_reflections();
        s.total_corrective_rules = meta_cognition_->total_corrective_rules();
        s.last_reflection_at     = meta_cognition_->last_reflection_at();
    }

    // Layer 3.
    s.training_enabled = cfg_.training_enabled && trainer_ != nullptr;
    {
        std::lock_guard<std::mutex> sl(stats_mutex_);
        s.total_training_runs   = total_training_runs_;
        s.last_training_at      = last_training_at_;
        s.active_adapter_path   = active_adapter_path_;
        s.last_improvement_pct  = last_improvement_pct_;
    }

    return s;
}

// ---------------------------------------------------------------------------
// Trigger checks
// ---------------------------------------------------------------------------

bool SelfImprovementLoop::check_episode_count_trigger() const {
    int n = cfg_.trigger_every_n_episodes;
    if (n <= 0) return false;
    return (episode_counter_.load() % n) == 0 && episode_counter_.load() > 0;
}

bool SelfImprovementLoop::check_interval_trigger() const {
    int hours = cfg_.trigger_every_n_hours;
    if (hours <= 0) return false;
    std::lock_guard<std::mutex> tl(timing_mutex_);
    auto elapsed = std::chrono::steady_clock::now() - last_training_time_;
    return elapsed >= std::chrono::hours(hours);
}

bool SelfImprovementLoop::check_confidence_trigger() const {
    float threshold = cfg_.trigger_on_domain_conf_below;
    if (threshold <= 0.0f || !self_model_) return false;

    auto snap = self_model_->get_snapshot();
    for (const auto& ds : snap.domain_stats) {
        if (ds.total_inferences >= 10 && ds.avg_confidence < threshold) {
            LOG_INFO("SelfImprovementLoop: confidence trigger — domain='" +
                     ds.domain + "' avg_conf=" + std::to_string(ds.avg_confidence) +
                     " below " + std::to_string(threshold));
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// post_training_request()
// ---------------------------------------------------------------------------

void SelfImprovementLoop::post_training_request(TrainingRequest req) {
    {
        std::lock_guard<std::mutex> lock(training_mutex_);
        if (training_requested_) {
            // Coalesce: a request is already queued.
            // Upgrade trigger string if the new one is more specific.
            if (req.trigger == "manual") {
                pending_request_ = req;  // manual always wins
            }
            return;
        }
        pending_request_    = req;
        training_requested_ = true;
    }
    training_cv_.notify_one();
    LOG_INFO("SelfImprovementLoop: training request posted (trigger=" +
             req.trigger + ")");
}

} // namespace cardinal
