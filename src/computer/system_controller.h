#pragma once
// =============================================================================
// Cardinal - System Controller
// File: src/computer/system_controller.h
//
// Controls system-level settings:
//   volume, brightness, wifi, bluetooth, notifications
// Uses pactl (PulseAudio/PipeWire), brightnessctl, nmcli, bluetoothctl.
// Reads CPU/RAM stats from /proc.
// =============================================================================

#include "computer/computer_types.h"
#include "utils/config_loader.h"

namespace cardinal {

    class SystemController {
    public:
        explicit SystemController(const CardinalConfig& config);
        ~SystemController() = default;

        SystemController(const SystemController&)            = delete;
        SystemController& operator=(const SystemController&) = delete;

        // State
        SystemState get_state() const;

        // Volume (0-100)
        bool set_volume(int pct);
        bool set_mute(bool muted);

        // Brightness (0-100)
        bool set_brightness(int pct);

        // Wi-Fi
        bool set_wifi(bool enabled);

        // Bluetooth
        bool set_bluetooth(bool enabled);

        // Notifications (Do Not Disturb via gsettings)
        bool set_notifications(bool enabled);

    private:
        static std::string run_cmd(const std::string& cmd);
        static int         parse_int_output(const std::string& out);

        const CardinalConfig& config_;
    };

} // namespace cardinal
