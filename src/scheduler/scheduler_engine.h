#pragma once
// =============================================================================
// Cardinal - Scheduler Engine (v1.5.0)
// File: src/scheduler/scheduler_engine.h
// =============================================================================

#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_store.h"
#include "scheduler/scheduler_parser.h"
#include "self_model/self_model_types.h"
#include "utils/config_loader.h"

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <optional>

namespace cardinal {

    class AgentExecutor;
    class InferencePipeline;
    class SelfImprovementLoop;
    class EpisodicStorage;
    class SessionManager;
    class TraceBuilder;

    struct SchedulerDeps {
        AgentExecutor*        agent_executor   = nullptr;
        InferencePipeline*    pipeline         = nullptr;
        SelfImprovementLoop*  self_improvement = nullptr;
        EpisodicStorage*      episodic         = nullptr;
        SessionManager*       sessions         = nullptr;
        std::function<bool()> inference_busy_fn;
    };

    class SchedulerEngine {
    public:
        explicit SchedulerEngine(const CardinalConfig& config, SchedulerDeps deps);
        ~SchedulerEngine();

        SchedulerEngine(const SchedulerEngine&)            = delete;
        SchedulerEngine& operator=(const SchedulerEngine&) = delete;

        void start();
        void stop();
        bool is_running() const { return running_.load(); }
        void on_inference();

        TaskParseResult              create_task_from_nl(const std::string& nl,
                                                          const std::string& session_id = "");
        std::string                  create_task(ScheduledTask task);
        bool                         update_task(const ScheduledTask& task);
        bool                         delete_task(const std::string& task_id);
        std::optional<ScheduledTask> get_task(const std::string& task_id) const;
        std::vector<ScheduledTask>   list_tasks() const;
        bool                         enable_task(const std::string& task_id);
        bool                         disable_task(const std::string& task_id);
        std::string                  run_task_now(const std::string& task_id);

        std::vector<TaskRun>       get_task_history(const std::string& task_id,
                                                     int limit = 50) const;
        std::vector<TaskRun>       get_recent_runs(int limit = 100) const;
        std::vector<TaskActionLog> get_run_action_logs(const std::string& run_id) const;

        SchedulerStatus get_status() const;

    private:
        void engine_loop();
        void tick();

        bool should_fire(const ScheduledTask& task,
                         std::chrono::system_clock::time_point now) const;
        bool eval_interval(const ScheduledTask& task,
                           std::chrono::system_clock::time_point now) const;
        bool eval_condition(const TriggerSpec& trigger) const;
        bool eval_idle(const TriggerSpec& trigger,
                       std::chrono::system_clock::time_point now) const;
        std::string compute_next_run_at(const ScheduledTask& task,
                                        std::chrono::system_clock::time_point now) const;

        static bool parse_cron_field(const std::string& field,
                                     int lo_bound, int hi_bound, int value);
        static bool cron_matches(const std::string& expr, const std::tm& tm);

        bool   eval_condition_expr(const std::string& expr) const;
        double read_condition_var(const std::string& var) const;

        void    dispatch_task(const ScheduledTask& task);
        TaskRun execute_task(const ScheduledTask& task, const std::string& run_id);

        std::string execute_agent_run(const ScheduledTask& task,
                                      const std::string& session_id,
                                      std::vector<TaskActionLog>& log);
        std::string execute_chat(const ScheduledTask& task,
                                 const std::string& session_id,
                                 std::vector<TaskActionLog>& log);
        std::string execute_reflect(std::vector<TaskActionLog>& log);
        std::string execute_train(const ScheduledTask& task,
                                  std::vector<TaskActionLog>& log);
        std::string execute_self_improvement(std::vector<TaskActionLog>& log);
        std::string execute_maintenance(std::vector<TaskActionLog>& log);
        std::string execute_export(const ScheduledTask& task,
                                   std::vector<TaskActionLog>& log);
        std::string execute_shell(const ScheduledTask& task,
                                  std::vector<TaskActionLog>& log);
        std::string execute_webhook(const ScheduledTask& task,
                                    const std::string& result_body,
                                    std::vector<TaskActionLog>& log);

        std::string   make_session_id(const std::string& task_id) const;
        void          store_result_to_episodic(const std::string& task_name,
                                               const std::string& result,
                                               const std::string& session_id);
        void          write_result_to_file(const std::string& path,
                                           const std::string& content);
        TaskActionLog make_log_entry(int seq, const std::string& type,
                                     const std::string& description,
                                     bool success, int ms) const;
        bool          check_whitelist(const ScheduledTask& task) const;

        static std::string tp_to_iso(std::chrono::system_clock::time_point tp);
        static std::chrono::system_clock::time_point iso_to_tp(const std::string& iso);

        const CardinalConfig&  config_;
        SchedulerDeps          deps_;

        std::unique_ptr<SchedulerStore>   store_;
        std::unique_ptr<SchedulerParser>  parser_;

        std::thread             engine_thread_;
        std::atomic<bool>       running_{ false };
        std::atomic<bool>       stop_requested_{ false };
        mutable std::mutex      cv_mutex_;
        std::condition_variable cv_;

        mutable std::mutex                    idle_mutex_;
        std::chrono::system_clock::time_point last_inference_at_;

        mutable std::mutex  status_mutex_;
        std::string         current_task_id_;
        std::string         current_task_name_;
        std::string         last_run_at_;
        std::string         next_scheduled_at_;
        int                 total_runs_      = 0;
        int                 successful_runs_ = 0;
        int                 failed_runs_     = 0;

        mutable std::mutex            sim_mutex_;
        mutable SelfImprovementStatus last_sim_status_;
        mutable bool                  sim_status_valid_ = false;
    };

} // namespace cardinal
