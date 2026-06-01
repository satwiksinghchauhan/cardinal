#pragma once
// =============================================================================
// Cardinal - Tool: close_app
// File: src/tools/builtin/computer/tool_close_app.h
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class AppController;
    ToolDefinition make_close_app_tool_def(const CardinalConfig& config);
    ToolResult     execute_close_app(const ToolCall& call, AppController& apps);
} // namespace cardinal
