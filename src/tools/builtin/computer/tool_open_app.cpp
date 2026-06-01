// =============================================================================
// Cardinal - Tool: open_app Implementation
// =============================================================================

#include "tools/builtin/computer/tool_open_app.h"
#include "computer/app_controller.h"
#include <chrono>

namespace cardinal {

ToolDefinition make_open_app_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "open_app";
    def.description = "Open a desktop application by name. "
                      "Examples: 'google-chrome', 'firefox', 'nautilus', 'gnome-terminal', "
                      "'gedit', 'thunderbird'. Uses .desktop files when available.";
    def.confirmation_required = config.computer_use.safety.confirmation_required;
    def.parameters.push_back({
        "app", ToolParameterType::STRING,
        "Application name or executable to launch, e.g. 'google-chrome', 'nautilus'",
        true, ""
    });
    def.parameters.push_back({
        "focus", ToolParameterType::BOOLEAN,
        "If true and app is already running, focus its window instead of launching a new instance. Default: false.",
        false, "false"
    });
    return def;
}

ToolResult execute_open_app(const ToolCall& call, AppController& apps) {
    ToolResult result;
    result.tool_name = "open_app";
    result.call      = call;
    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string app   = get("app");
        bool        focus = get("focus", "false") == "true";

        if (app.empty()) {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Missing required parameter: app";
            return result;
        }

        if (!apps.is_app_allowed(app)) {
            result.status = ToolStatus::FAILURE;
            result.output = "App '" + app + "' is not in the allowed apps list.";
            return result;
        }

        // Try focus first if requested and already running
        if (focus) {
            auto existing = apps.get_app(app);
            if (existing) {
                apps.focus_app(app);
                result.status = ToolStatus::SUCCESS;
                result.output = "Focused existing window: " + app;
                result.duration_ms = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count());
                return result;
            }
        }

        bool ok = apps.open_app(app);
        result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
        result.output = ok ? "Launched: " + app
                           : "Failed to launch: " + app + ". Check that it is installed.";
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "open_app failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
