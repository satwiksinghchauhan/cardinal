// =============================================================================
// Cardinal - Tool: shell_run Implementation
// =============================================================================

#include "tools/builtin/computer/tool_shell_run.h"
#include "computer/shell_executor.h"

namespace cardinal {

ToolDefinition make_shell_run_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "shell_run";
    def.description = "Run a shell command and return its output. "
                      "Commands are sandboxed and checked against a blocked-command list. "
                      "Network access is available. Use for system tasks, file operations, "
                      "running scripts, checking status, installing packages, etc.";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "command", ToolParameterType::STRING,
        "The shell command to run, e.g. 'ls -la ~/Documents', 'pip install numpy'",
        true, ""
    });
    def.parameters.push_back({
        "timeout_seconds", ToolParameterType::NUMBER,
        "Execution timeout in seconds. Default: uses config value.",
        false, "0"
    });
    def.parameters.push_back({
        "working_dir", ToolParameterType::STRING,
        "Working directory for the command. Default: config working_directory.",
        false, ""
    });
    return def;
}

ToolResult execute_shell_run(const ToolCall& call, ShellExecutor& shell) {
    ToolResult result;
    result.tool_name = "shell_run";
    result.call      = call;

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    std::string command     = get("command");
    int         timeout     = 0;
    try { timeout = std::stoi(get("timeout_seconds", "0")); } catch (...) {}
    std::string working_dir = get("working_dir");

    if (command.empty()) {
        result.status = ToolStatus::INVALID_ARGS;
        result.output = "Missing required parameter: command";
        return result;
    }

    auto sr = shell.run(command, timeout, working_dir);
    result.duration_ms = sr.duration_ms;

    if (!sr.success) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = sr.stderr_text.empty()
            ? "Exit code: " + std::to_string(sr.exit_code)
            : sr.stderr_text;
        result.output = "Command failed (exit=" + std::to_string(sr.exit_code) + ")";
        if (!sr.stdout_text.empty()) result.output += ":\n" + sr.stdout_text;
        if (!sr.stderr_text.empty()) result.output += "\nError: " + sr.stderr_text;
    } else {
        result.status = ToolStatus::SUCCESS;
        result.output = sr.stdout_text.empty() ? "(no output)" : sr.stdout_text;
    }
    return result;
}

} // namespace cardinal
