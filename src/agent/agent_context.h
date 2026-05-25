#pragma once
// =============================================================================
// Cardinal - Agent Context
// File: src/agent/agent_context.h
//
// SQLite-backed working memory for agentic execution.
// Persists between restarts — interrupted tasks can be resumed.
//
// One AgentContext per active agent task (keyed by session_id + task_id).
// AgentExecutor owns and manages AgentContext instances.
// =============================================================================

#include "agent/agent_types.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <mutex>

struct sqlite3;

namespace cardinal {

    class AgentContext {
    public:
        AgentContext(const CardinalConfig& config,
                     const std::string&   session_id,
                     const std::string&   task_id);
        ~AgentContext();

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------
        void open();
        void close();
        bool is_open() const { return db_ != nullptr; }

        // ------------------------------------------------------------------
        // Working memory — key-value scratchpad
        // ------------------------------------------------------------------
        void store(const std::string& key,
                   const std::string& value,
                   const std::string& entry_type = "observation",
                   const std::string& step_index = "");

        std::string retrieve(const std::string& key) const;
        bool        has(const std::string& key) const;
        void        remove(const std::string& key);
        void        clear();

        // Get all entries for context injection
        std::vector<WorkingMemoryEntry> get_all() const;

        // Get entries for a specific step
        std::vector<WorkingMemoryEntry> get_for_step(
            const std::string& step_index) const;

        // ------------------------------------------------------------------
        // Step history
        // ------------------------------------------------------------------
        void        record_step(const AgentStep& step);
        AgentStep   get_step(int index) const;
        std::vector<AgentStep> get_all_steps() const;
        int         step_count() const;

        // ------------------------------------------------------------------
        // Plan persistence
        // ------------------------------------------------------------------
        void save_plan(const AgentPlan& plan);
        AgentPlan load_plan() const;
        bool      has_plan() const;

        // ------------------------------------------------------------------
        // Context summary for prompt injection
        // Formats working memory + step history as a text block the model
        // can use to understand what has been done so far.
        // ------------------------------------------------------------------
        std::string build_context_summary() const;

        // ------------------------------------------------------------------
        // Cleanup
        // ------------------------------------------------------------------
        void destroy(); // delete from disk entirely

        // Accessors
        const std::string& session_id() const { return session_id_; }
        const std::string& task_id()    const { return task_id_; }

    private:
        void create_schema();

        const CardinalConfig& config_;
        std::string           session_id_;
        std::string           task_id_;
        std::string           db_path_;

        sqlite3*              db_ = nullptr;
        mutable std::mutex    mutex_;
    };

} // namespace cardinal
