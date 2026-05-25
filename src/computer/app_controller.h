#pragma once
// =============================================================================
// Cardinal - App Controller
// File: src/computer/app_controller.h
//
// Open, close, focus, and list desktop applications.
// X11: wmctrl + xdotool + xprop
// Wayland: swaymsg (Sway/wlroots) with gdbus fallback for GNOME
// =============================================================================

#include "computer/computer_types.h"
#include "computer/display_detector.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <optional>

namespace cardinal {

    class AppController {
    public:
        explicit AppController(const DisplayDetector& display,
                               const CardinalConfig&  config);
        ~AppController() = default;

        AppController(const AppController&)            = delete;
        AppController& operator=(const AppController&) = delete;

        // ------------------------------------------------------------------
        // Application lifecycle
        // ------------------------------------------------------------------

        // Launch an application by executable name or .desktop app id.
        // Returns true if launch command succeeded (does not wait for window).
        bool open_app(const std::string& app_name);

        // Close application by name or PID. Sends SIGTERM first, SIGKILL on timeout.
        bool close_app(const std::string& app_name);

        // Focus the window of an application (bring to foreground).
        bool focus_app(const std::string& app_name);

        // ------------------------------------------------------------------
        // Window enumeration
        // ------------------------------------------------------------------

        // List all visible windows/apps.
        std::vector<AppInfo> list_apps() const;

        // Get info for a specific app by name (fuzzy match).
        std::optional<AppInfo> get_app(const std::string& app_name) const;

        // Get the currently focused window.
        std::optional<AppInfo> get_focused_app() const;

        // ------------------------------------------------------------------
        // Safety check
        // ------------------------------------------------------------------
        bool is_app_allowed(const std::string& app_name) const;

    private:
        // X11 helpers
        std::vector<AppInfo> list_apps_x11() const;
        bool focus_app_x11(const std::string& app_name);
        bool close_app_x11(const std::string& app_name);
        std::optional<AppInfo> get_focused_x11() const;

        // Wayland helpers
        std::vector<AppInfo> list_apps_wayland() const;
        bool focus_app_wayland(const std::string& app_name);
        bool close_app_wayland(const std::string& app_name);
        std::optional<AppInfo> get_focused_wayland() const;

        static std::string run_cmd(const std::string& cmd);
        static bool        fuzzy_match(const std::string& haystack,
                                       const std::string& needle);

        const DisplayDetector& display_;
        const CardinalConfig&  config_;
    };

} // namespace cardinal
