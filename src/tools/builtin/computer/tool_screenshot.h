#pragma once
// =============================================================================
// Cardinal - Tool: screenshot
// File: src/tools/builtin/computer/tool_screenshot.h
//
// Captures the screen (full or region) and optionally analyzes it with vision.
// Returns path + description injected into the model context.
// =============================================================================

#include "tools/tool_result.h"
#include "computer/computer_types.h"

namespace cardinal {

    class ScreenReader;

    // Register the tool definition into the registry
    ToolDefinition make_screenshot_tool_def(const CardinalConfig& config);

    // Execute a validated ToolCall
    ToolResult execute_screenshot(const ToolCall& call, ScreenReader& reader);

} // namespace cardinal
