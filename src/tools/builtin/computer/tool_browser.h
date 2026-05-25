#pragma once
// =============================================================================
// Cardinal - Tool: browser
// File: src/tools/builtin/computer/tool_browser.h
//
// Single tool with multiple actions for browser control via Playwright.
// Actions: navigate, click, click_text, type, scroll, screenshot,
//          get_content, execute_js, new_tab, close_tab, back, forward, reload
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class BrowserController;
    ToolDefinition make_browser_tool_def(const CardinalConfig& config);
    ToolResult     execute_browser(const ToolCall& call, BrowserController& browser);
} // namespace cardinal
