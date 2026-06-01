// =============================================================================
// Cardinal - Tool: close_app Implementation
// =============================================================================

#include "tools/builtin/computer/tool_close_app.h"
#include "computer/app_controller.h"
#include <chrono>

namespace cardinal {

ToolDefinition make_close_app_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "close_app";
    def.description = "Close a running application by name. Sends a graceful close signal first.";
    def.confirmation_required = config.computer_use.safety.confirmation_required;
    def.parameters.push_back({
        "app", ToolParameterType::STRING,
        "Application name or window title to close, e.g. 'firefox', 'gedit'",
        true, ""
    });
    return def;
}

ToolResult execute_close_app(const ToolCall& call, AppController& apps) {
    ToolResult result;
    result.tool_name = "close_app";
    result.call      = call;
    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string app = get("app");
        if (app.empty()) {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Missing required parameter: app";
            return result;
        }

        bool ok = apps.close_app(app);
        result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
        result.output = ok ? "Closed: " + app : "Failed to close: " + app;
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "close_app failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
