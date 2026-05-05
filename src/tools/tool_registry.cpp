// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Tool Registry Implementation
// File: src/tools/tool_registry.cpp
// =============================================================================

#include "tools/tool_registry.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

namespace cardinal {

    ToolRegistry::ToolRegistry(const CardinalConfig& config)
        : config_(config)
    {}

    // =========================================================================
    // init
    // =========================================================================

    void ToolRegistry::init() {
        LOG_INFO("ToolRegistry: registering built-in tools...");

        const auto& tc = config_.tools;

        if (tc.web_search.enabled)         register_builtin_web_search();
        if (tc.web_fetch.enabled)          register_builtin_web_fetch();
        if (tc.calculator.enabled)         register_builtin_calculator();
        if (tc.run_python.enabled)         register_builtin_run_python();
        if (tc.file_read.enabled)          register_builtin_file_read();
        if (tc.file_write.enabled)         register_builtin_file_write();
        if (tc.knowledge_graph.enabled)    register_builtin_kg_query();
        if (tc.episodic_search.enabled)    register_builtin_episodic_search();

        LOG_INFO("ToolRegistry: " + std::to_string(tools_.size()) +
                 " tools registered");
    }

    // =========================================================================
    // Registration
    // =========================================================================

    void ToolRegistry::register_tool(const ToolDefinition& def) {
        std::unique_lock lock(mutex_);
        tools_[def.name] = def;
        LOG_DEBUG("ToolRegistry: registered tool '" + def.name + "'");
    }

    void ToolRegistry::register_tools(const std::vector<ToolDefinition>& defs) {
        for (const auto& def : defs) register_tool(def);
    }

    bool ToolRegistry::unregister_tool(const std::string& name) {
        std::unique_lock lock(mutex_);
        return tools_.erase(name) > 0;
    }

    // =========================================================================
    // Lookup
    // =========================================================================

    std::optional<ToolDefinition> ToolRegistry::get(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) return std::nullopt;
        return it->second;
    }

    bool ToolRegistry::is_available(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        return it != tools_.end() && it->second.enabled;
    }

    bool ToolRegistry::requires_confirmation(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) return false;
        return it->second.confirmation_required;
    }

    std::vector<ToolDefinition> ToolRegistry::get_enabled_tools() const {
        std::shared_lock lock(mutex_);
        std::vector<ToolDefinition> result;
        for (const auto& [name, def] : tools_) {
            if (def.enabled) result.push_back(def);
        }
        return result;
    }

    int ToolRegistry::count() const {
        std::shared_lock lock(mutex_);
        return static_cast<int>(tools_.size());
    }

    // =========================================================================
    // Prompt formatting
    // =========================================================================

    std::string ToolRegistry::format_tool(const ToolDefinition& def) const {
        json tool;
        tool["name"]        = def.name;
        tool["description"] = def.description;

        json params;
        params["type"] = "object";
        json props;
        json required_arr = json::array();

        for (const auto& p : def.parameters) {
            json param;
            param["type"]        = tool_param_type_to_string(p.type);
            param["description"] = p.description;
            if (!p.enum_values.empty()) {
                param["enum"] = p.enum_values;
            }
            if (!p.default_val.empty()) {
                param["default"] = p.default_val;
            }
            props[p.name] = param;
            if (p.required) required_arr.push_back(p.name);
        }

        params["properties"] = props;
        if (!required_arr.empty()) {
            params["required"] = required_arr;
        }
        tool["parameters"] = params;

        return tool.dump(2);
    }

    std::string ToolRegistry::format_tools_for_prompt(
        const std::vector<ToolDefinition>& tools) const
    {
        if (tools.empty()) return "";

        std::ostringstream oss;
        oss << "## Available Tools\n\n";
        oss << "You have access to the following tools. To use a tool, output a "
               "JSON object wrapped in <tool_call> tags:\n\n";
        oss << "<tool_call>\n"
               "{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
               "</tool_call>\n\n";
        oss << "Available tools:\n\n";

        for (const auto& def : tools) {
            if (!def.enabled) continue;
            oss << "### " << def.name << "\n";
            oss << def.description << "\n";
            oss << "```json\n" << format_tool(def) << "\n```\n\n";
        }

        return oss.str();
    }

    std::string ToolRegistry::format_all_for_prompt() const {
        return format_tools_for_prompt(get_enabled_tools());
    }

    // =========================================================================
    // Validation
    // =========================================================================

    std::string ToolRegistry::validate_call(const ToolCall& call) const {
        auto def_opt = get(call.tool_name);
        if (!def_opt) return "Tool not found: " + call.tool_name;
        if (!def_opt->enabled) return "Tool disabled: " + call.tool_name;

        // Check required parameters
        for (const auto& param : def_opt->parameters) {
            if (param.required &&
                call.arguments.find(param.name) == call.arguments.end()) {
                return "Missing required parameter '" + param.name +
                       "' for tool '" + call.tool_name + "'";
            }
        }

        return ""; // valid
    }

    // =========================================================================
    // Built-in tool registration
    // =========================================================================

    void ToolRegistry::register_builtin_web_search() {
        ToolDefinition def;
        def.name        = "web_search";
        def.description = "Search the web using DuckDuckGo. Returns a list of "
                          "relevant results with titles, URLs, and snippets.";
        def.confirmation_required =
            config_.tools.web_search.confirmation_required;

        def.parameters.push_back({
            "query", ToolParameterType::STRING,
            "The search query string", true, ""
        });
        def.parameters.push_back({
            "max_results", ToolParameterType::NUMBER,
            "Maximum number of results to return (default: 5)",
            false,
            std::to_string(config_.tools.web_search.max_results)
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_web_fetch() {
        ToolDefinition def;
        def.name        = "web_fetch";
        def.description = "Fetch the content of a URL and return it as plain text. "
                          "HTML is stripped, only readable content is returned.";
        def.confirmation_required =
            config_.tools.web_fetch.confirmation_required;

        def.parameters.push_back({
            "url", ToolParameterType::STRING,
            "The full URL to fetch (must start with http:// or https://)",
            true, ""
        });
        def.parameters.push_back({
            "max_kb", ToolParameterType::NUMBER,
            "Maximum content size in KB to return",
            false,
            std::to_string(config_.tools.web_fetch.max_content_kb)
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_calculator() {
        ToolDefinition def;
        def.name        = "calculator";
        def.description = "Evaluate a mathematical expression and return the result. "
                          "Supports arithmetic, trigonometry, logarithms, and "
                          "common mathematical functions.";
        def.confirmation_required =
            config_.tools.calculator.confirmation_required;

        def.parameters.push_back({
            "expression", ToolParameterType::STRING,
            "The mathematical expression to evaluate. "
            "Examples: '2 + 2', 'sqrt(144)', 'sin(pi/2)', '(3^4) / 2'",
            true, ""
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_run_python() {
        ToolDefinition def;
        def.name        = "run_python";
        def.description = "Execute Python code in a sandboxed environment and "
                          "return stdout output. Network access is disabled. "
                          "Use for data analysis, calculations, and scripting.";
        def.confirmation_required =
            config_.tools.run_python.confirmation_required;

        def.parameters.push_back({
            "code", ToolParameterType::STRING,
            "The Python code to execute. Print results to stdout.",
            true, ""
        });
        def.parameters.push_back({
            "timeout_seconds", ToolParameterType::NUMBER,
            "Execution timeout in seconds",
            false,
            std::to_string(config_.tools.run_python.timeout_seconds)
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_file_read() {
        ToolDefinition def;
        def.name        = "file_read";
        def.description = "Read the contents of a file from the allowed paths. "
                          "Returns the file content as text.";
        def.confirmation_required =
            config_.tools.file_read.confirmation_required;

        def.parameters.push_back({
            "path", ToolParameterType::STRING,
            "The file path to read. Must be within allowed paths.",
            true, ""
        });
        def.parameters.push_back({
            "max_kb", ToolParameterType::NUMBER,
            "Maximum content size in KB to return (default: 512)",
            false, "512"
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_file_write() {
        ToolDefinition def;
        def.name        = "file_write";
        def.description = "Write content to a file in the allowed output paths. "
                          "Creates the file if it doesn't exist, overwrites if it does.";
        def.confirmation_required =
            config_.tools.file_write.confirmation_required;

        def.parameters.push_back({
            "path", ToolParameterType::STRING,
            "The file path to write. Must be within allowed output paths.",
            true, ""
        });
        def.parameters.push_back({
            "content", ToolParameterType::STRING,
            "The content to write to the file.",
            true, ""
        });
        def.parameters.push_back({
            "append", ToolParameterType::BOOLEAN,
            "If true, append to existing file instead of overwriting.",
            false, "false"
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_kg_query() {
        ToolDefinition def;
        def.name        = "knowledge_graph_query";
        def.description = "Query Cardinal's internal knowledge graph for facts, "
                          "entities, and relationships in a specific domain.";
        def.confirmation_required =
            config_.tools.knowledge_graph.confirmation_required;

        def.parameters.push_back({
            "query", ToolParameterType::STRING,
            "The query string to search the knowledge graph",
            true, ""
        });
        def.parameters.push_back({
            "domain", ToolParameterType::STRING,
            "Optional domain filter (factual|ethical|spatial|temporal|social|mathematical)",
            false, ""
        });
        def.parameters.push_back({
            "max_results", ToolParameterType::NUMBER,
            "Maximum number of results",
            false, "10"
        });

        register_tool(def);
    }

    void ToolRegistry::register_builtin_episodic_search() {
        ToolDefinition def;
        def.name        = "episodic_search";
        def.description = "Search Cardinal's episodic memory for past interactions "
                          "and learned knowledge relevant to a query.";
        def.confirmation_required =
            config_.tools.episodic_search.confirmation_required;

        def.parameters.push_back({
            "query", ToolParameterType::STRING,
            "The query to search past episodes for",
            true, ""
        });
        def.parameters.push_back({
            "max_results", ToolParameterType::NUMBER,
            "Maximum number of episodes to return",
            false,
            std::to_string(config_.tools.episodic_search.max_results)
        });
        def.parameters.push_back({
            "min_confidence", ToolParameterType::NUMBER,
            "Minimum confidence score filter (0.0-1.0)",
            false, "0.0"
        });

        register_tool(def);
    }

} // namespace cardinal
