#pragma once
// =============================================================================
// Cardinal - Tool: shell_run
// File: src/tools/builtin/computer/tool_shell_run.h
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class ShellExecutor;
    ToolDefinition make_shell_run_tool_def(const CardinalConfig& config);
    ToolResult     execute_shell_run(const ToolCall& call, ShellExecutor& shell);
} // namespace cardinal
