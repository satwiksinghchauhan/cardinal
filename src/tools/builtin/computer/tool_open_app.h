#pragma once
// =============================================================================
// Cardinal - Tool: open_app
// File: src/tools/builtin/computer/tool_open_app.h
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class AppController;
    ToolDefinition make_open_app_tool_def(const CardinalConfig& config);
    ToolResult     execute_open_app(const ToolCall& call, AppController& apps);
} // namespace cardinal
