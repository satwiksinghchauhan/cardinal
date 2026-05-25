#pragma once
// =============================================================================
// Cardinal - Tool: system_control
// File: src/tools/builtin/computer/tool_system_control.h
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class SystemController;
    ToolDefinition make_system_control_tool_def(const CardinalConfig& config);
    ToolResult     execute_system_control(const ToolCall& call, SystemController& sys);
} // namespace cardinal
