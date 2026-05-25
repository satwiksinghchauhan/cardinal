#pragma once
// =============================================================================
// Cardinal - Scheduler Store
// File: src/scheduler/scheduler_store.h
// =============================================================================

#include "scheduler/scheduler_types.h"

#include <sqlite3.h>   // must be included here — sqlite3_stmt used in declarations

#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <memory>

namespace cardinal {

    class SchedulerStore {
    public:
        explicit SchedulerStore(const std::string& db_path);
        ~SchedulerStore();

        SchedulerStore(const SchedulerStore&)            = delete;
        SchedulerStore& operator=(const SchedulerStore&) = delete;

        void open();
        void close();

        // ScheduledTask CRUD
        std::string              insert_task(const ScheduledTask& task);
        bool                     update_task(const ScheduledTask& task);
        bool                     delete_task(const std::string& task_id);
        std::optional<ScheduledTask> get_task(const std::string& task_id) const;
        std::vector<ScheduledTask>   list_tasks() const;
        std::vector<ScheduledTask>   list_enabled_tasks() const;
        bool update_task_stats(const std::string& task_id,
                               int run_count, int fail_count,
                               const std::string& last_run_at,
                               const std::string& next_run_at);
        bool set_task_enabled(const std::string& task_id, bool enabled);

        // TaskRun CRUD
        std::string           insert_run(const TaskRun& run);
        bool                  update_run(const TaskRun& run);
        std::optional<TaskRun> get_run(const std::string& run_id) const;
        std::vector<TaskRun>  get_task_runs(const std::string& task_id,
                                             int limit = 50) const;
        std::vector<TaskRun>  get_recent_runs(int limit = 100) const;

        // TaskActionLog
        void                       append_action_log(const std::string& run_id,
                                                     const TaskActionLog& entry);
        std::vector<TaskActionLog> get_action_logs(const std::string& run_id) const;

        // Housekeeping
        void prune_run_history(int max_entries_per_task);
        int  count_tasks()         const;
        int  count_enabled_tasks() const;
        int  count_runs()          const;

    private:
        void create_tables();
        void apply_migrations();

        static std::string   trigger_to_json(const TriggerSpec& t);
        static TriggerSpec   trigger_from_json(const std::string& json);
        static std::string   action_to_json(const TaskAction& a);
        static TaskAction    action_from_json(const std::string& json);

        static ScheduledTask  row_to_task(sqlite3_stmt* stmt);
        static TaskRun        row_to_run(sqlite3_stmt* stmt);
        static TaskActionLog  row_to_action_log(sqlite3_stmt* stmt);

        std::string        db_path_;
        sqlite3*           db_  = nullptr;
        mutable std::mutex mutex_;
    };

} // namespace cardinal
