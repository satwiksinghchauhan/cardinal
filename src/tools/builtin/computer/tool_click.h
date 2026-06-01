#pragma once
// =============================================================================
// Cardinal - Tool: click
// File: src/tools/builtin/computer/tool_click.h
//
// Click at pixel coordinates OR find + click by visual description.
// AT-SPI2 element lookup is attempted first; vision fallback if not found.
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {

    class InputController;
    class ScreenReader;
    class AtSpiReader;

    ToolDefinition make_click_tool_def(const CardinalConfig& config);

    ToolResult execute_click(const ToolCall&    call,
                             InputController&   input,
                             ScreenReader&      screen,
                             AtSpiReader*       atspi = nullptr);

} // namespace cardinal
