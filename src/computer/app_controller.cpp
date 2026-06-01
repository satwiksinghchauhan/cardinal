// =============================================================================
// Cardinal - App Controller Implementation
// File: src/computer/app_controller.cpp
// =============================================================================

#include "computer/app_controller.h"
#include "utils/logger.h"

#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <array>
#include <cctype>

namespace cardinal {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string AppController::run_cmd(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string result;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "";
    while (fgets(buf.data(), buf.size(), p)) result += buf.data();
    pclose(p);
    return result;
}

bool AppController::fuzzy_match(const std::string& haystack,
                                 const std::string& needle) {
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    return lower(haystack).find(lower(needle)) != std::string::npos;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

AppController::AppController(const DisplayDetector& display,
                             const CardinalConfig&  config)
    : display_(display), config_(config)
{}

// ---------------------------------------------------------------------------
// Safety check
// ---------------------------------------------------------------------------

bool AppController::is_app_allowed(const std::string& app_name) const {
    if (!config_.computer_use.safety.whitelist_enabled) return true;
    const auto& allowed = config_.computer_use.safety.allowed_apps;
    if (allowed.empty()) return true;
    for (const auto& a : allowed) {
        if (fuzzy_match(app_name, a) || fuzzy_match(a, app_name)) return true;
    }
    LOG_WARN("AppController: app not in whitelist: " + app_name);
    return false;
}

// ---------------------------------------------------------------------------
// open_app
// ---------------------------------------------------------------------------

bool AppController::open_app(const std::string& app_name) {
    if (!is_app_allowed(app_name)) return false;

    // Try gtk-launch (uses .desktop files) first, then raw exec
    std::string cmd_gtk  = "gtk-launch " + app_name + " >/dev/null 2>&1 &";
    std::string cmd_exec = app_name + " >/dev/null 2>&1 &";

    if (std::system(cmd_gtk.c_str()) == 0) {
        LOG_INFO("AppController: launched via gtk-launch: " + app_name);
        return true;
    }
    int rc = std::system(cmd_exec.c_str());
    if (rc == 0) {
        LOG_INFO("AppController: launched: " + app_name);
        return true;
    }
    LOG_WARN("AppController: failed to launch: " + app_name);
    return false;
}

// ---------------------------------------------------------------------------
// close_app
// ---------------------------------------------------------------------------

bool AppController::close_app(const std::string& app_name) {
    if (display_.is_x11())     return close_app_x11(app_name);
    if (display_.is_wayland()) return close_app_wayland(app_name);
    // Headless: pkill by name
    int rc = std::system(("pkill -f " + app_name + " 2>/dev/null").c_str());
    return rc == 0;
}

bool AppController::close_app_x11(const std::string& app_name) {
    // wmctrl -c closes by window title pattern
    int rc = std::system(("wmctrl -c " + app_name + " 2>/dev/null").c_str());
    if (rc == 0) return true;
    // Fallback: pkill
    rc = std::system(("pkill -f " + app_name + " 2>/dev/null").c_str());
    return rc == 0;
}

bool AppController::close_app_wayland(const std::string& app_name) {
    if (display_.has_swaymsg()) {
        std::string cmd = "swaymsg '[app_id=\"" + app_name + "\"] kill' 2>/dev/null";
        if (std::system(cmd.c_str()) == 0) return true;
    }
    int rc = std::system(("pkill -f " + app_name + " 2>/dev/null").c_str());
    return rc == 0;
}

// ---------------------------------------------------------------------------
// focus_app
// ---------------------------------------------------------------------------

bool AppController::focus_app(const std::string& app_name) {
    if (display_.is_x11())     return focus_app_x11(app_name);
    if (display_.is_wayland()) return focus_app_wayland(app_name);
    return false;
}

bool AppController::focus_app_x11(const std::string& app_name) {
    if (!display_.has_wmctrl()) {
        LOG_WARN("AppController: wmctrl not available");
        return false;
    }
    // wmctrl -a activates window by name pattern
    int rc = std::system(("wmctrl -a " + app_name + " 2>/dev/null").c_str());
    if (rc == 0) return true;
    // Try xdotool
    if (display_.has_xdotool()) {
        std::string cmd = "xdotool search --name " + app_name +
                          " windowactivate --sync 2>/dev/null";
        rc = std::system(cmd.c_str());
        return rc == 0;
    }
    return false;
}

bool AppController::focus_app_wayland(const std::string& app_name) {
    if (display_.has_swaymsg()) {
        std::string cmd = "swaymsg '[app_id=\"" + app_name + "\"] focus' 2>/dev/null";
        int rc = std::system(cmd.c_str());
        return rc == 0;
    }
    return false;
}

// ---------------------------------------------------------------------------
// list_apps
// ---------------------------------------------------------------------------

std::vector<AppInfo> AppController::list_apps() const {
    if (display_.is_x11())     return list_apps_x11();
    if (display_.is_wayland()) return list_apps_wayland();

    // Headless: list running processes
    std::vector<AppInfo> result;
    std::string out = run_cmd("ps -eo pid,comm --no-headers 2>/dev/null | head -40");
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        AppInfo a;
        int pid = 0;
        char name[256]{};
        if (sscanf(line.c_str(), "%d %255s", &pid, name) == 2) {
            a.pid  = pid;
            a.name = name;
            result.push_back(a);
        }
    }
    return result;
}

std::vector<AppInfo> AppController::list_apps_x11() const {
    std::vector<AppInfo> result;
    if (!display_.has_wmctrl()) return result;

    // wmctrl -l -G prints: ID desktop X Y W H hostname title
    std::string out = run_cmd("wmctrl -l -G 2>/dev/null");
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        AppInfo a;
        char wid[32]{}, hostname[128]{}, title[512]{};
        int desktop = 0, x = 0, y = 0, w = 0, h = 0;
        if (sscanf(line.c_str(), "%s %d %d %d %d %d %127s %511[^\n]",
                   wid, &desktop, &x, &y, &w, &h, hostname, title) >= 7) {
            a.window_title = title;
            a.name         = title;
            a.window_rect  = {x, y, w, h};
            result.push_back(a);
        }
    }
    return result;
}

std::vector<AppInfo> AppController::list_apps_wayland() const {
    std::vector<AppInfo> result;
    if (!display_.has_swaymsg()) return result;

    // swaymsg -t get_tree — simplified: list top-level app_ids
    std::string out = run_cmd(
        "swaymsg -t get_tree 2>/dev/null | "
        "python3 -c \""
        "import sys,json;"
        "t=json.load(sys.stdin);"
        "[print(n.get('app_id',''),n.get('name',''),n.get('pid',0))"
        " for n in t.get('nodes',[])"
        " for n in n.get('nodes',[])"
        " for n in n.get('nodes',[])"
        " if n.get('app_id')]"
        "\" 2>/dev/null");

    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        AppInfo a;
        int pid = 0;
        std::string app_id, name;
        ls >> app_id >> name >> pid;
        a.executable   = app_id;
        a.name         = name.empty() ? app_id : name;
        a.window_title = a.name;
        a.pid          = pid;
        result.push_back(a);
    }
    return result;
}

// ---------------------------------------------------------------------------
// get_app / get_focused_app
// ---------------------------------------------------------------------------

std::optional<AppInfo> AppController::get_app(const std::string& app_name) const {
    auto apps = list_apps();
    for (const auto& a : apps) {
        if (fuzzy_match(a.name, app_name) ||
            fuzzy_match(a.executable, app_name) ||
            fuzzy_match(a.window_title, app_name))
            return a;
    }
    return std::nullopt;
}

std::optional<AppInfo> AppController::get_focused_app() const {
    if (display_.is_x11())     return get_focused_x11();
    if (display_.is_wayland()) return get_focused_wayland();
    return std::nullopt;
}

std::optional<AppInfo> AppController::get_focused_x11() const {
    if (!display_.has_xprop()) return std::nullopt;
    std::string wid = run_cmd(
        "xdotool getactivewindow 2>/dev/null");
    while (!wid.empty() && (wid.back() == '\n' || wid.back() == '\r'))
        wid.pop_back();
    if (wid.empty()) return std::nullopt;

    AppInfo a;
    std::string title = run_cmd(
        "xdotool getwindowname " + wid + " 2>/dev/null");
    while (!title.empty() && (title.back() == '\n' || title.back() == '\r'))
        title.pop_back();
    a.window_title = title;
    a.name         = title;
    a.focused      = true;
    return a;
}

std::optional<AppInfo> AppController::get_focused_wayland() const {
    if (!display_.has_swaymsg()) return std::nullopt;
    std::string out = run_cmd(
        "swaymsg -t get_tree 2>/dev/null | "
        "python3 -c \""
        "import sys,json;"
        "def find(n):"
        "  if n.get('focused'):print(n.get('app_id',''),n.get('name',''));"
        "  [find(c) for c in n.get('nodes',[])+n.get('floating_nodes',[])];"
        "find(json.load(sys.stdin))"
        "\" 2>/dev/null | head -1");
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    if (out.empty()) return std::nullopt;

    std::istringstream ss(out);
    std::string app_id, name;
    ss >> app_id >> name;
    AppInfo a;
    a.executable   = app_id;
    a.name         = name.empty() ? app_id : name;
    a.window_title = a.name;
    a.focused      = true;
    return a;
}

} // namespace cardinal
