// =============================================================================
// Cardinal - Display Detector Implementation
// File: src/computer/display_detector.cpp
// =============================================================================

#include "computer/display_detector.h"
#include "utils/logger.h"

#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <array>

namespace cardinal {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool DisplayDetector::tool_exists(const std::string& name) {
    std::string cmd = "command -v " + name + " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

static std::string run_cmd(const std::string& cmd) {
    std::array<char, 512> buf{};
    std::string result;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "";
    while (fgets(buf.data(), buf.size(), p)) result += buf.data();
    pclose(p);
    // trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

// ---------------------------------------------------------------------------
// detect()
// ---------------------------------------------------------------------------

void DisplayDetector::detect() {
    if (detected_) return;
    detected_ = true;

    // Check env vars to determine display server
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    const char* x11     = std::getenv("DISPLAY");

    if (wayland && *wayland) {
        server_           = DisplayServer::WAYLAND;
        info_.server      = DisplayServer::WAYLAND;
        info_.display_var = wayland;
    } else if (x11 && *x11) {
        server_           = DisplayServer::X11;
        info_.server      = DisplayServer::X11;
        info_.display_var = x11;
    } else {
        server_           = DisplayServer::HEADLESS;
        info_.server      = DisplayServer::HEADLESS;
        info_.display_var = "";
    }

    // Tool detection
    has_scrot_   = tool_exists("scrot");
    has_grim_    = tool_exists("grim");
    has_xdotool_ = tool_exists("xdotool");
    has_ydotool_ = tool_exists("ydotool");
    has_wtype_   = tool_exists("wtype");
    has_wmctrl_  = tool_exists("wmctrl");
    has_swaymsg_ = tool_exists("swaymsg");
    has_xrandr_  = tool_exists("xrandr");
    has_xprop_   = tool_exists("xprop");

    // AT-SPI: check for python3-gi and pyatspi
    has_atspi_ = (std::system(
        "python3 -c 'import pyatspi' >/dev/null 2>&1") == 0);

    // Screen resolution
    if (server_ == DisplayServer::X11 && has_xrandr_) {
        // xrandr | grep ' connected' | grep -o '[0-9]*x[0-9]*'
        std::string res = run_cmd(
            "xrandr 2>/dev/null | grep ' connected primary' | "
            "grep -o '[0-9]*x[0-9]*' | head -1");
        if (res.empty())
            res = run_cmd(
                "xrandr 2>/dev/null | grep ' connected' | "
                "grep -o '[0-9]*x[0-9]*' | head -1");
        if (!res.empty()) {
            auto pos = res.find('x');
            if (pos != std::string::npos) {
                info_.width  = std::stoi(res.substr(0, pos));
                info_.height = std::stoi(res.substr(pos + 1));
            }
        }
    } else if (server_ == DisplayServer::WAYLAND && has_swaymsg_) {
        // swaymsg -t get_outputs  — simplified: just try wlr-randr
        if (tool_exists("wlr-randr")) {
            std::string res = run_cmd(
                "wlr-randr 2>/dev/null | grep 'current' | "
                "grep -o '[0-9]*x[0-9]*' | head -1");
            if (!res.empty()) {
                auto pos = res.find('x');
                if (pos != std::string::npos) {
                    info_.width  = std::stoi(res.substr(0, pos));
                    info_.height = std::stoi(res.substr(pos + 1));
                }
            }
        }
        // Fallback: assume 1920x1080
        if (info_.width == 0) { info_.width = 1920; info_.height = 1080; }
    }

    // Default resolution if still unknown
    if (info_.width == 0)  info_.width  = 1920;
    if (info_.height == 0) info_.height = 1080;

    // HiDPI scale via GDK_SCALE or XCURSOR_SIZE heuristic
    const char* gdk_scale = std::getenv("GDK_SCALE");
    if (gdk_scale) {
        try { info_.scale_factor = std::stof(gdk_scale); } catch (...) {}
    }

    LOG_INFO("DisplayDetector: " + summary());
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::string DisplayDetector::screenshot_tool() const {
    if (server_ == DisplayServer::X11     && has_scrot_) return "scrot";
    if (server_ == DisplayServer::WAYLAND && has_grim_)  return "grim";
    return "";
}

std::string DisplayDetector::input_tool() const {
    if (server_ == DisplayServer::X11     && has_xdotool_) return "xdotool";
    if (server_ == DisplayServer::WAYLAND && has_ydotool_) return "ydotool";
    if (server_ == DisplayServer::WAYLAND && has_wtype_)   return "wtype";
    return "";
}

std::string DisplayDetector::summary() const {
    std::ostringstream oss;
    oss << display_server_to_string(server_)
        << " " << info_.width << "x" << info_.height;
    if (info_.scale_factor != 1.0f) oss << " @" << info_.scale_factor << "x";
    oss << " | screenshot=" << screenshot_tool()
        << " input=" << input_tool()
        << " atspi=" << (has_atspi_ ? "yes" : "no");
    return oss.str();
}

} // namespace cardinal
