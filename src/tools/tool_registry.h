// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Tool Registry
// File: src/tools/tool_registry.h
//
// Single source of truth for all tool definitions.
// Built-in tools are registered at startup from config.
// User-defined tools can be registered per-request via the API.
//
// Thread-safe — multiple inference threads can read concurrently.
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ToolRegistry
    // -------------------------------------------------------------------------
    class ToolRegistry {
    public:
        explicit ToolRegistry(const CardinalConfig& config);

        // ------------------------------------------------------------------
        // Initialization
        // Registers all enabled built-in tools from config.
        // Call once at CardinalAPI::init().
        // ------------------------------------------------------------------
        void init();

        // ------------------------------------------------------------------
        // Registration
        // ------------------------------------------------------------------

        // Register a built-in or user-defined tool.
        // Overwrites existing definition if name already registered.
        void register_tool(const ToolDefinition& def);

        // Register multiple tools at once (e.g. per-request user tools)
        void register_tools(const std::vector<ToolDefinition>& defs);

        // Unregister a tool by name
        bool unregister_tool(const std::string& name);

        // ------------------------------------------------------------------
        // Lookup
        // ------------------------------------------------------------------

        // Get tool definition by name. Returns nullopt if not found.
        std::optional<ToolDefinition> get(const std::string& name) const;

        // True if tool exists and is enabled
        bool is_available(const std::string& name) const;

        // True if tool requires human confirmation
        bool requires_confirmation(const std::string& name) const;

        // Get all enabled tool definitions (for prompt injection)
        std::vector<ToolDefinition> get_enabled_tools() const;

        // Get count
        int count() const;

        // ------------------------------------------------------------------
        // Prompt formatting
        // Formats all enabled tools as a JSON Schema block for injection
        // into the system prompt. Model-agnostic — every backend uses this.
        // ------------------------------------------------------------------
        std::string format_tools_for_prompt(
            const std::vector<ToolDefinition>& tools) const;

        std::string format_all_for_prompt() const;

        // ------------------------------------------------------------------
        // Validation
        // Validate a parsed ToolCall against its definition.
        // Returns empty string on success, error description on failure.
        // ------------------------------------------------------------------
        std::string validate_call(const ToolCall& call) const;

    private:
        // Register all built-in tools based on config
        void register_builtin_web_search();
        void register_builtin_web_fetch();
        void register_builtin_calculator();
        void register_builtin_run_python();
        void register_builtin_file_read();
        void register_builtin_file_write();
        void register_builtin_kg_query();
        void register_builtin_episodic_search();

        // Format a single tool as JSON Schema
        std::string format_tool(const ToolDefinition& def) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig&                          config_;
        std::unordered_map<std::string, ToolDefinition> tools_;
        mutable std::shared_mutex                      mutex_;
    };

} // namespace cardinal
