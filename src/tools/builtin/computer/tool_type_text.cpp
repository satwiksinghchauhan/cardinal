// =============================================================================
// Cardinal - Tool: type_text Implementation
// =============================================================================

#include "tools/builtin/computer/tool_type_text.h"
#include "computer/input_controller.h"
#include "utils/logger.h"

#include <chrono>
#include <sstream>

namespace cardinal {

ToolDefinition make_type_text_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "type_text";
    def.description = "Type text or send a keyboard shortcut. "
                      "Use 'text' to type literal characters. "
                      "Use 'key' to send a key combination like 'ctrl+c', 'Return', "
                      "'alt+F4', 'ctrl+shift+t', 'Escape', 'Tab', 'F5'.";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "text", ToolParameterType::STRING,
        "Literal text to type, e.g. 'Hello world'. Use this OR key, not both.",
        false, ""
    });
    def.parameters.push_back({
        "key", ToolParameterType::STRING,
        "Key combination to send, e.g. 'ctrl+c', 'Return', 'alt+F4', 'ctrl+z'.",
        false, ""
    });
    def.parameters.push_back({
        "delay_ms", ToolParameterType::NUMBER,
        "Delay in milliseconds between each keystroke when typing text. Default: 0.",
        false, "0"
    });
    return def;
}

ToolResult execute_type_text(const ToolCall& call, InputController& input) {
    ToolResult result;
    result.tool_name = "type_text";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& def = "") -> std::string {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : def;
    };

    try {
        std::string text     = get("text");
        std::string key      = get("key");
        int         delay_ms = 0;
        try { delay_ms = std::stoi(get("delay_ms", "0")); } catch (...) {}

        if (text.empty() && key.empty()) {
            result.status        = ToolStatus::INVALID_ARGS;
            result.error_message = "Provide either 'text' or 'key'";
            result.output        = result.error_message;
            return result;
        }

        if (!key.empty()) {
            input.send_key(key);
            result.output = "Sent key: " + key;
        } else {
            input.type_text(text, delay_ms);
            std::string preview = text.size() > 50 ? text.substr(0, 50) + "…" : text;
            result.output = "Typed: " + preview;
        }

        result.status = ToolStatus::SUCCESS;
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "type_text failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
