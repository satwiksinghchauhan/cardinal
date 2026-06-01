// =============================================================================
// Cardinal - Tool: system_control Implementation
// =============================================================================

#include "tools/builtin/computer/tool_system_control.h"
#include "computer/system_controller.h"

#include <sstream>
#include <chrono>

namespace cardinal {

ToolDefinition make_system_control_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "system_control";
    def.description =
        "Read or change system settings: volume, brightness, wifi, bluetooth, "
        "notifications. Also reads CPU/RAM usage and system info.\n\n"
        "Actions: get_state|set_volume|set_mute|set_brightness|"
        "set_wifi|set_bluetooth|set_notifications";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "action", ToolParameterType::STRING,
        "Action: get_state|set_volume|set_mute|set_brightness|"
        "set_wifi|set_bluetooth|set_notifications",
        true, ""
    });
    def.parameters.push_back({
        "value", ToolParameterType::STRING,
        "Value for set actions. For volume/brightness: 0-100. "
        "For mute/wifi/bluetooth/notifications: 'true' or 'false'.",
        false, ""
    });
    return def;
}

ToolResult execute_system_control(const ToolCall& call, SystemController& sys) {
    ToolResult result;
    result.tool_name = "system_control";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string action = get("action");
        std::string value  = get("value");

        if (action == "get_state") {
            auto s = sys.get_state();
            std::ostringstream oss;
            oss << "System state:\n"
                << "  Volume:     " << s.volume_pct << "%" << (s.muted ? " (muted)" : "") << "\n"
                << "  Brightness: " << s.brightness_pct << "%\n"
                << "  Wi-Fi:      " << (s.wifi_enabled ? "on" : "off");
            if (!s.wifi_ssid.empty()) oss << " (" << s.wifi_ssid << ")";
            oss << "\n"
                << "  Bluetooth:  " << (s.bluetooth_enabled ? "on" : "off") << "\n"
                << "  CPU:        " << static_cast<int>(s.cpu_pct) << "%\n"
                << "  RAM:        " << s.ram_used_mb << "/" << s.ram_total_mb << " MB\n"
                << "  OS:         " << s.os_name << "\n"
                << "  Hostname:   " << s.hostname;
            result.status = ToolStatus::SUCCESS;
            result.output = oss.str();

        } else if (action == "set_volume") {
            int pct = 50;
            try { pct = std::stoi(value); } catch (...) {}
            bool ok = sys.set_volume(pct);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? "Volume set to " + std::to_string(pct) + "%"
                               : "Failed to set volume";

        } else if (action == "set_mute") {
            bool mute = (value == "true" || value == "1");
            bool ok = sys.set_mute(mute);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? std::string(mute ? "Muted" : "Unmuted")
                               : "Failed to set mute";

        } else if (action == "set_brightness") {
            int pct = 50;
            try { pct = std::stoi(value); } catch (...) {}
            bool ok = sys.set_brightness(pct);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? "Brightness set to " + std::to_string(pct) + "%"
                               : "Failed to set brightness";

        } else if (action == "set_wifi") {
            bool en = (value == "true" || value == "on" || value == "1");
            bool ok = sys.set_wifi(en);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? std::string(en ? "Wi-Fi enabled" : "Wi-Fi disabled")
                               : "Failed to set Wi-Fi";

        } else if (action == "set_bluetooth") {
            bool en = (value == "true" || value == "on" || value == "1");
            bool ok = sys.set_bluetooth(en);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? std::string(en ? "Bluetooth enabled" : "Bluetooth disabled")
                               : "Failed to set Bluetooth";

        } else if (action == "set_notifications") {
            bool en = (value == "true" || value == "on" || value == "1");
            bool ok = sys.set_notifications(en);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok ? std::string(en ? "Notifications enabled" : "Notifications disabled (DND on)")
                               : "Failed to set notifications";

        } else {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Unknown action: " + action;
        }
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "system_control failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
