#pragma once
// =============================================================================
// Cardinal - Self-Improvement Loop (Layer 3 Orchestrator)
// File: src/training/self_improvement_loop.h
//
// Owns and coordinates all three self-improvement layers:
//
//   Layer 1 — SelfModel       (record_inference, format_for_prompt)
//   Layer 2 — MetaCognition   (on_inference, reflect)
//   Layer 3 — Training cycle  (CurriculumBuilder → DatasetCurator →
//                              ITrainingBackend → AdapterEvaluator)
//
// Called from CardinalAPI on two hooks:
//
//   on_inference(domain, reasoning_type, confidence, contradiction,
//                uncertainty, rule_committed)
//     — fired synchronously after every inference, on the inference thread.
//     — updates Layer 1 (fast, in-process SQLite write).
//     — forwards to Layer 2; MetaCognition::on_inference() may trigger a
//       synchronous reflection pass if a trigger condition fires.
//     — checks Layer 3 triggers (episode count, wall-clock interval,
//       domain confidence). If a trigger fires, posts to the training
//       thread (non-blocking).
//
//   on_session_boundary()
//     — fired by CardinalAPI between sessions (before a new session's
//       first inference, or on destroy_session).
//     — calls AdapterEvaluator::apply_pending_adapter() to hot-swap any
//       approved adapter that was waiting for a clean cut-point.
//
// Training thread:
//   A single std::thread (training_thread_) runs the Layer 3 cycle
//   asynchronously so it never blocks inference.
//   The thread sleeps on a condition_variable and wakes when:
//     (a) a trigger posts a training request, or
//     (b) the wall-clock interval fires (checked every 60 s).
//   Only one training cycle runs at a time; overlapping triggers are
//   coalesced into a single pending request.
//
// Trigger conditions for Layer 3 (any one fires):
//   T1 — every N episodes   (config.training.trigger_every_n_episodes)
//   T2 — every N hours      (config.training.trigger_every_n_hours)
//   T3 — any domain's avg_confidence drops below threshold
//          (config.training.trigger_on_domain_confidence_below)
//   T4 — manual via API     (CardinalAPI::trigger_training())
//
// Shutdown:
//   stop() signals the training thread to finish its current cycle (if any)
//   and then exit.  CardinalAPI calls stop() inside shutdown().
//
// Ownership:
//   SelfImprovementLoop owns SelfModel, MetaCognition, CurriculumBuilder,
//   DatasetCurator, ITrainingBackend (via TrainingFactory), and
//   AdapterEvaluator.  All subsystems are constructed in start() so that
//   config is fully validated before any SQLite file is opened.
//
// Thread safety:
//   on_inference() is called on the inference thread — must be lock-free for
//   the common (no-trigger) case.  Atomic counters + a single try_to_lock
//   for the trigger check satisfy this.
//   on_session_boundary() is also on the inference thread; it acquires only
//   the AdapterEvaluator's pending_mutex_ (very short hold time).
// =============================================================================

#include "self_model/self_model_types.h"    // SelfImprovementStatus, ReflectionResult
#include "utils/config_loader.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace cardinal {

    class SelfModel;
    class MetaCognition;
    class CurriculumBuilder;
    class DatasetCurator;
    class ITrainingBackend;
    class AdapterEvaluator;
    class EpisodicStorage;
    class RuleStore;
    class ILLMBackend;

    // -------------------------------------------------------------------------
    // TrainingRequest
    // Posted to the training thread when a trigger fires.
    // -------------------------------------------------------------------------
    struct TrainingRequest {
        std::string trigger;      // "episode_count"|"interval"|"confidence"|"manual"
        std::string domain_hint;  // "" = let CurriculumBuilder decide
    };

    // -------------------------------------------------------------------------
    // SelfImprovementLoop
    // -------------------------------------------------------------------------
    class SelfImprovementLoop {
    public:
        SelfImprovementLoop(const CardinalConfig& config,
                            EpisodicStorage&      storage,
                            RuleStore&            rule_store,
                            ILLMBackend&          backend);

        ~SelfImprovementLoop();

        SelfImprovementLoop(const SelfImprovementLoop&)            = delete;
        SelfImprovementLoop& operator=(const SelfImprovementLoop&) = delete;

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------

        // Open SelfModel DB, start training thread. Safe to call only once.
        void start();

        // Signal training thread to stop, join, close SelfModel DB.
        void stop();

        bool is_running() const { return running_.load(); }

        // ------------------------------------------------------------------
        // Inference hooks (called on inference thread)
        // ------------------------------------------------------------------

        // Layer 1 + 2 update; Layer 3 trigger check.
        // Non-blocking for the common case (no trigger fires).
        void on_inference(const std::string& domain,
                          const std::string& reasoning_type,
                          float              confidence,
                          bool               contradiction,
                          bool               uncertainty,
                          bool               rule_committed);

        // Apply any pending adapter at a clean session boundary.
        void on_session_boundary();

        // ------------------------------------------------------------------
        // Manual triggers (called from CardinalAPI endpoint handlers)
        // ------------------------------------------------------------------

        // Force an immediate reflection pass (Layer 2).
        // Runs synchronously on the calling thread and returns the result.
        ReflectionResult trigger_reflection();

        // Post a training request to the background thread (Layer 3).
        // Returns immediately; training runs asynchronously.
        // Returns false if training is disabled or a cycle is already running.
        bool trigger_training(const std::string& domain_hint = "");

        // ------------------------------------------------------------------
        // Prompt injection (Layer 1)
        // ------------------------------------------------------------------

        // Returns formatted self-model context for system prompt injection.
        // Returns "" if self_model is disabled or not yet open.
        std::string format_self_model_for_prompt() const;

        // ------------------------------------------------------------------
        // Status (for CardinalAPI::get_stats)
        // ------------------------------------------------------------------
        SelfImprovementStatus get_status() const;

    private:
        // -- Training thread entry point -----------------------------------
        void training_thread_fn();

        // -- Run one full Layer 3 training cycle --------------------------
        // Called on the training thread with a specific request.
        TrainingResult run_training_cycle(const TrainingRequest& req);

        // -- Trigger checks (called on inference thread) ------------------
        bool check_episode_count_trigger()   const;
        bool check_interval_trigger()         const;
        bool check_confidence_trigger()       const;

        // -- Post a request to the training thread (non-blocking) ---------
        void post_training_request(TrainingRequest req);

        // -- Config sub-structs (stored by value) -------------------------
        struct LoopCfg {
            bool        enabled                         = true;
            // Layer 1
            bool        self_model_enabled              = true;
            bool        inject_into_prompt              = true;
            // Layer 2
            bool        meta_cognition_enabled          = true;
            // Layer 3
            bool        training_enabled                = true;
            int         trigger_every_n_episodes        = 100;
            int         trigger_every_n_hours           = 24;
            float       trigger_on_domain_conf_below    = 0.5f;
            float       eval_improvement_threshold_pct  = 5.0f;
            std::string adapter_load_policy             = "session_boundary";
            int         min_episodes_for_training       = 50;
        } cfg_;

        // -- Subsystem ownership ------------------------------------------
        std::unique_ptr<SelfModel>        self_model_;
        std::unique_ptr<MetaCognition>    meta_cognition_;
        std::unique_ptr<CurriculumBuilder> curriculum_;
        std::unique_ptr<DatasetCurator>   curator_;
        std::unique_ptr<ITrainingBackend> trainer_;
        std::unique_ptr<AdapterEvaluator> evaluator_;

        // Non-owning references to shared subsystems.
        EpisodicStorage& storage_;
        RuleStore&       rule_store_;
        ILLMBackend&     backend_;
        const CardinalConfig& config_;

        // -- Training thread state ----------------------------------------
        std::thread              training_thread_;
        std::mutex               training_mutex_;
        std::condition_variable  training_cv_;
        bool                     training_requested_ = false;
        bool                     stop_requested_     = false;
        TrainingRequest          pending_request_;

        // -- Atomic counters (written on inference thread) ----------------
        std::atomic<int>  episode_counter_{ 0 };
        std::atomic<bool> training_in_progress_{ false };
        std::atomic<bool> running_{ false };

        // -- Timing -------------------------------------------------------
        std::chrono::steady_clock::time_point last_training_time_;
        mutable std::mutex                     timing_mutex_;

        // -- Stats (training thread writes, get_status reads) -------------
        mutable std::mutex stats_mutex_;
        int         total_training_runs_  = 0;
        std::string last_training_at_;
        std::string active_adapter_path_;
        float       last_improvement_pct_ = 0.0f;
    };

} // namespace cardinal
