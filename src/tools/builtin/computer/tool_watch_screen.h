#pragma once
// =============================================================================
// Cardinal - Tool: watch_screen
// File: src/tools/builtin/computer/tool_watch_screen.h
//
// Watch the screen for a visual change matching a description.
// Polls for up to timeout_seconds, returns when screen changes or times out.
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class ScreenReader;
    ToolDefinition make_watch_screen_tool_def(const CardinalConfig& config);
    ToolResult     execute_watch_screen(const ToolCall& call, ScreenReader& reader);
} // namespace cardinal
