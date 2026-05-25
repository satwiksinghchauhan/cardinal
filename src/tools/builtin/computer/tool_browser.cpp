// =============================================================================
// Cardinal - Tool: browser Implementation
// =============================================================================

#include "tools/builtin/computer/tool_browser.h"
#include "computer/browser_controller.h"
#include "utils/logger.h"

#include <chrono>
#include <sstream>

namespace cardinal {

ToolDefinition make_browser_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "browser";
    def.description =
        "Control the web browser via Playwright. Supports navigation, clicking, "
        "typing, reading page content, taking screenshots, and running JavaScript.\n\n"
        "Actions:\n"
        "  navigate    — go to a URL\n"
        "  click       — click a CSS selector\n"
        "  click_text  — click an element by visible text\n"
        "  type        — fill a field (CSS selector + text)\n"
        "  scroll      — scroll the page\n"
        "  get_content — return page text content\n"
        "  screenshot  — take a browser screenshot\n"
        "  execute_js  — run JavaScript and return result\n"
        "  new_tab     — open a new tab\n"
        "  close_tab   — close current tab\n"
        "  back        — browser back\n"
        "  forward     — browser forward\n"
        "  reload      — reload current page";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "action", ToolParameterType::STRING,
        "Action to perform: navigate|click|click_text|type|scroll|get_content|"
        "screenshot|execute_js|new_tab|close_tab|back|forward|reload",
        true, ""
    });
    def.parameters.push_back({
        "url", ToolParameterType::STRING,
        "URL for navigate action, e.g. 'https://example.com'",
        false, ""
    });
    def.parameters.push_back({
        "selector", ToolParameterType::STRING,
        "CSS selector for click or type actions, e.g. '#search-input', 'button.submit'",
        false, ""
    });
    def.parameters.push_back({
        "text", ToolParameterType::STRING,
        "Text to type (for type action) or visible text to click (for click_text action)",
        false, ""
    });
    def.parameters.push_back({
        "script", ToolParameterType::STRING,
        "JavaScript code to execute (for execute_js action)",
        false, ""
    });
    def.parameters.push_back({
        "scroll_y", ToolParameterType::NUMBER,
        "Vertical scroll amount in scroll units (positive = down). For scroll action.",
        false, "3"
    });
    def.parameters.push_back({
        "timeout_ms", ToolParameterType::NUMBER,
        "Action timeout in milliseconds. Default: uses config value.",
        false, "0"
    });
    return def;
}

ToolResult execute_browser(const ToolCall& call, BrowserController& browser) {
    ToolResult result;
    result.tool_name = "browser";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") -> std::string {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string action = get("action");
        if (action.empty()) {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Missing required parameter: action";
            return result;
        }

        BrowserAction ba;
        ba.timeout_ms = 0;
        try { ba.timeout_ms = std::stoi(get("timeout_ms", "0")); } catch (...) {}

        if (action == "navigate") {
            ba.type = BrowserActionType::NAVIGATE;
            ba.url  = get("url");
            if (ba.url.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "navigate action requires 'url'";
                return result;
            }
        } else if (action == "click") {
            ba.type     = BrowserActionType::CLICK;
            ba.selector = get("selector");
        } else if (action == "click_text") {
            ba.type = BrowserActionType::CLICK;
            ba.text = get("text");
        } else if (action == "type") {
            ba.type     = BrowserActionType::TYPE;
            ba.selector = get("selector");
            ba.text     = get("text");
        } else if (action == "scroll") {
            ba.type = BrowserActionType::SCROLL;
            try { ba.scroll.delta_y = std::stoi(get("scroll_y", "3")); } catch (...) {}
        } else if (action == "get_content") {
            ba.type = BrowserActionType::GET_CONTENT;
        } else if (action == "screenshot") {
            ba.type = BrowserActionType::SCREENSHOT;
        } else if (action == "execute_js") {
            ba.type = BrowserActionType::EXECUTE_JS;
            ba.text = get("script");
        } else if (action == "new_tab") {
            ba.type = BrowserActionType::NEW_TAB;
        } else if (action == "close_tab") {
            ba.type = BrowserActionType::CLOSE_TAB;
        } else if (action == "back") {
            ba.type = BrowserActionType::BACK;
        } else if (action == "forward") {
            ba.type = BrowserActionType::FORWARD;
        } else if (action == "reload") {
            ba.type = BrowserActionType::RELOAD;
        } else {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Unknown browser action: " + action;
            return result;
        }

        auto br = browser.execute(ba);

        if (!br.success) {
            result.status        = ToolStatus::FAILURE;
            result.error_message = br.error_message;
            result.output        = "Browser action '" + action + "' failed: " + br.error_message;
        } else {
            result.status = ToolStatus::SUCCESS;
            std::ostringstream oss;
            oss << "Browser action '" << action << "' succeeded.";
            if (!br.url.empty())              oss << "\nURL: " << br.url;
            if (!br.title.empty())            oss << "\nTitle: " << br.title;
            if (!br.content.empty())          oss << "\nContent:\n" << br.content.substr(0, 8000);
            if (!br.screenshot_path.empty())  oss << "\nScreenshot: " << br.screenshot_path;
            result.output = oss.str();
        }

        result.duration_ms = br.duration_ms;
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "browser tool failed: " + std::string(e.what());
    }

    if (result.duration_ms == 0)
        result.duration_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
