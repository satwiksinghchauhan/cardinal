#pragma once
// =============================================================================
// Cardinal - Display Detector
// File: src/computer/display_detector.h
//
// Detects at runtime whether the system is running X11, Wayland, or headless.
// Checks environment variables + tool availability.
// Result is cached after first call — detection is done once at startup.
// =============================================================================

#include "computer/computer_types.h"
#include "utils/config_loader.h"

#include <string>

namespace cardinal {

    class DisplayDetector {
    public:
        DisplayDetector() = default;

        // Detect and cache the display server. Safe to call multiple times.
        void detect();

        DisplayServer server()      const { return server_; }
        const ScreenInfo& info()    const { return info_; }
        bool is_headless()          const { return server_ == DisplayServer::HEADLESS; }
        bool is_x11()               const { return server_ == DisplayServer::X11; }
        bool is_wayland()           const { return server_ == DisplayServer::WAYLAND; }

        // Tool availability (populated during detect())
        bool has_scrot()       const { return has_scrot_; }
        bool has_grim()        const { return has_grim_; }
        bool has_xdotool()     const { return has_xdotool_; }
        bool has_ydotool()     const { return has_ydotool_; }
        bool has_wtype()       const { return has_wtype_; }
        bool has_wmctrl()      const { return has_wmctrl_; }
        bool has_swaymsg()     const { return has_swaymsg_; }
        bool has_xrandr()      const { return has_xrandr_; }
        bool has_xprop()       const { return has_xprop_; }
        bool has_atspi()       const { return has_atspi_; }

        // Screenshot tool to use for this session
        std::string screenshot_tool() const;

        // Input tool for typing text
        std::string input_tool() const;

        // Returns a human-readable summary for logging
        std::string summary() const;

    private:
        static bool tool_exists(const std::string& name);
        static int  get_screen_dimension(const std::string& xrandr_field);

        DisplayServer server_   = DisplayServer::HEADLESS;
        ScreenInfo    info_;
        bool          detected_ = false;

        bool has_scrot_    = false;
        bool has_grim_     = false;
        bool has_xdotool_  = false;
        bool has_ydotool_  = false;
        bool has_wtype_    = false;
        bool has_wmctrl_   = false;
        bool has_swaymsg_  = false;
        bool has_xrandr_   = false;
        bool has_xprop_    = false;
        bool has_atspi_    = false;
    };

} // namespace cardinal
