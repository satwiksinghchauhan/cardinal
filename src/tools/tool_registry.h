#pragma once
// =============================================================================
// Cardinal - Tool Registry (v1.5.0)
// File: src/tools/tool_registry.h
//
// Changes from v1.3.0:
//   v1.5.0: register_builtin_computer_tools() + register_builtin_schedule_task()
//           added to private section; init() calls them when enabled in config
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

namespace cardinal {

    class ToolRegistry {
    public:
        explicit ToolRegistry(const CardinalConfig& config);

        // ------------------------------------------------------------------
        // Initialization
        // ------------------------------------------------------------------
        void init();

        // ------------------------------------------------------------------
        // Registration
        // ------------------------------------------------------------------
        void register_tool(const ToolDefinition& def);
        void register_tools(const std::vector<ToolDefinition>& defs);
        bool unregister_tool(const std::string& name);

        // ------------------------------------------------------------------
        // Lookup
        // ------------------------------------------------------------------
        std::optional<ToolDefinition> get(const std::string& name) const;
        bool is_available(const std::string& name) const;
        bool requires_confirmation(const std::string& name) const;
        std::vector<ToolDefinition> get_enabled_tools() const;
        int count() const;

        // ------------------------------------------------------------------
        // Prompt formatting
        // ------------------------------------------------------------------
        std::string format_tools_for_prompt(
            const std::vector<ToolDefinition>& tools) const;
        std::string format_all_for_prompt() const;

        // ------------------------------------------------------------------
        // Validation
        // ------------------------------------------------------------------
        std::string validate_call(const ToolCall& call) const;

    private:
        // Built-in tools — original (v1.2.0 / v1.3.0)
        void register_builtin_web_search();
        void register_builtin_web_fetch();
        void register_builtin_calculator();
        void register_builtin_run_python();
        void register_builtin_file_read();
        void register_builtin_file_write();
        void register_builtin_kg_query();
        void register_builtin_episodic_search();
        void register_builtin_analyze_image();

        // Built-in tools — computer use + scheduler (v1.5.0)
        void register_builtin_computer_tools();
        void register_builtin_schedule_task();

        // Format a single tool as JSON Schema
        std::string format_tool(const ToolDefinition& def) const;

        // Members
        const CardinalConfig&                            config_;
        std::unordered_map<std::string, ToolDefinition>  tools_;
        mutable std::shared_mutex                        mutex_;
    };

} // namespace cardinal
