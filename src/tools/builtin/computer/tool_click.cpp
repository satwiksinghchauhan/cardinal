// =============================================================================
// Cardinal - Tool: click Implementation
// File: src/tools/builtin/computer/tool_click.cpp
// =============================================================================

#include "tools/builtin/computer/tool_click.h"
#include "computer/input_controller.h"
#include "computer/screen_reader.h"
#include "computer/atspi_reader.h"
#include "utils/logger.h"

#include <chrono>
#include <sstream>

namespace cardinal {

ToolDefinition make_click_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "click";
    def.description = "Click on a UI element or screen position. "
                      "Provide either (x, y) pixel coordinates OR a natural language "
                      "description of the element to click (e.g. 'the Submit button', "
                      "'search box', 'File menu'). "
                      "When a description is given, AT-SPI accessibility tree is searched "
                      "first; vision fallback is used if not found.";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "description", ToolParameterType::STRING,
        "Natural language description of the element to click, e.g. 'Submit button', "
        "'username field', 'File menu'. Use this OR x/y coordinates.",
        false, ""
    });
    def.parameters.push_back({
        "x", ToolParameterType::NUMBER,
        "X pixel coordinate to click. Use with y instead of description.",
        false, ""
    });
    def.parameters.push_back({
        "y", ToolParameterType::NUMBER,
        "Y pixel coordinate to click.",
        false, ""
    });
    def.parameters.push_back({
        "button", ToolParameterType::STRING,
        "Mouse button: 'left' (default), 'right', 'middle'",
        false, "left"
    });
    def.parameters.push_back({
        "double_click", ToolParameterType::BOOLEAN,
        "If true, perform a double-click. Default: false.",
        false, "false"
    });
    def.parameters.push_back({
        "app", ToolParameterType::STRING,
        "Optional: application name to scope AT-SPI search (e.g. 'firefox')",
        false, ""
    });
    return def;
}

ToolResult execute_click(const ToolCall&  call,
                          InputController& input,
                          ScreenReader&    screen,
                          AtSpiReader*     atspi) {
    ToolResult result;
    result.tool_name = "click";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& def = "") -> std::string {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : def;
    };

    try {
        std::string desc   = get("description");
        std::string sx     = get("x");
        std::string sy     = get("y");
        std::string btn_s  = get("button", "left");
        bool dbl           = get("double_click", "false") == "true";
        std::string app    = get("app");

        MouseButton button = MouseButton::LEFT;
        if (btn_s == "right")  button = MouseButton::RIGHT;
        if (btn_s == "middle") button = MouseButton::MIDDLE;
        int clicks = dbl ? 2 : 1;

        int cx = -1, cy = -1;

        if (!sx.empty() && !sy.empty()) {
            // Direct coordinates
            cx = std::stoi(sx);
            cy = std::stoi(sy);
        } else if (!desc.empty()) {
            // Try AT-SPI first
            if (atspi && atspi->is_available()) {
                std::string search_app = app.empty() ? "" : app;
                // Try to get bounds — need app name
                // If no app specified, try focused app via screen description
                if (!search_app.empty()) {
                    auto bounds = atspi->get_bounds(search_app, desc);
                    if (bounds) {
                        cx = bounds->cx();
                        cy = bounds->cy();
                        LOG_DEBUG("click: AT-SPI found '" + desc + "' at " +
                                  std::to_string(cx) + "," + std::to_string(cy));
                    }
                }
            }

            // Vision fallback
            if (cx < 0 || cy < 0) {
                auto pt = screen.find_element(desc);
                if (pt) {
                    cx = pt->x;
                    cy = pt->y;
                    LOG_DEBUG("click: vision found '" + desc + "' at " +
                              std::to_string(cx) + "," + std::to_string(cy));
                }
            }

            if (cx < 0 || cy < 0) {
                result.status        = ToolStatus::FAILURE;
                result.error_message = "Could not locate element: " + desc;
                result.output        = "Could not find '" + desc +
                                       "' on screen. Try taking a screenshot first to verify the element is visible.";
                result.duration_ms   = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count());
                return result;
            }
        } else {
            result.status        = ToolStatus::INVALID_ARGS;
            result.error_message = "Provide either 'description' or 'x' and 'y'";
            result.output        = result.error_message;
            return result;
        }

        input.mouse_click(cx, cy, button, clicks);

        std::ostringstream oss;
        oss << "Clicked at (" << cx << ", " << cy << ")";
        if (!desc.empty()) oss << " ['" << desc << "']";
        if (dbl) oss << " (double-click)";

        result.status = ToolStatus::SUCCESS;
        result.output = oss.str();

    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "Click failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
