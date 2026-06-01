// =============================================================================
// Cardinal - System Controller Implementation
// File: src/computer/system_controller.cpp
// =============================================================================

#include "computer/system_controller.h"
#include "utils/logger.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <array>
#include <algorithm>
#include <cctype>

namespace cardinal {

SystemController::SystemController(const CardinalConfig& config)
    : config_(config)
{}

std::string SystemController::run_cmd(const std::string& cmd) {
    std::array<char, 1024> buf{};
    std::string result;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "";
    while (fgets(buf.data(), buf.size(), p)) result += buf.data();
    pclose(p);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

int SystemController::parse_int_output(const std::string& out) {
    std::string s = out;
    s.erase(std::remove_if(s.begin(), s.end(),
            [](char c){ return !std::isdigit(c); }), s.end());
    if (s.empty()) return -1;
    try { return std::stoi(s); } catch (...) { return -1; }
}

// ---------------------------------------------------------------------------
// get_state
// ---------------------------------------------------------------------------

SystemState SystemController::get_state() const {
    SystemState s;

    // Volume via pactl
    std::string vol_out = run_cmd(
        "pactl get-sink-volume @DEFAULT_SINK@ 2>/dev/null | "
        "grep -o '[0-9]*%' | head -1");
    s.volume_pct = parse_int_output(vol_out);

    std::string mute_out = run_cmd(
        "pactl get-sink-mute @DEFAULT_SINK@ 2>/dev/null");
    s.muted = mute_out.find("yes") != std::string::npos;

    // Brightness via brightnessctl
    std::string bright_out = run_cmd("brightnessctl g 2>/dev/null");
    std::string bright_max = run_cmd("brightnessctl m 2>/dev/null");
    int cur = parse_int_output(bright_out);
    int max = parse_int_output(bright_max);
    if (cur >= 0 && max > 0)
        s.brightness_pct = (cur * 100) / max;

    // Wi-Fi via nmcli
    std::string wifi_out = run_cmd("nmcli radio wifi 2>/dev/null");
    s.wifi_enabled = wifi_out.find("enabled") != std::string::npos;
    s.wifi_ssid = run_cmd(
        "nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | "
        "grep '^yes' | cut -d: -f2 | head -1");

    // Bluetooth
    std::string bt_out = run_cmd(
        "bluetoothctl show 2>/dev/null | grep 'Powered' | awk '{print $2}'");
    s.bluetooth_enabled = bt_out.find("yes") != std::string::npos;

    // OS
    s.os_name = run_cmd("lsb_release -sd 2>/dev/null");
    if (s.os_name.empty()) s.os_name = run_cmd("uname -sr 2>/dev/null");
    s.hostname = run_cmd("hostname 2>/dev/null");

    // CPU from /proc/stat (simple one-shot reading — not a true average)
    {
        std::ifstream f("/proc/stat");
        if (f) {
            std::string tag;
            long long user, nice, system, idle, iowait, irq, softirq;
            f >> tag >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
            long long total = user + nice + system + idle + iowait + irq + softirq;
            long long busy  = total - idle - iowait;
            if (total > 0)
                s.cpu_pct = static_cast<float>(busy) / static_cast<float>(total) * 100.0f;
        }
    }

    // RAM from /proc/meminfo
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        long long mem_total_kb = 0, mem_avail_kb = 0;
        while (std::getline(f, line)) {
            if (line.find("MemTotal:") == 0)
                mem_total_kb = std::stoll(line.substr(9));
            if (line.find("MemAvailable:") == 0)
                mem_avail_kb = std::stoll(line.substr(13));
        }
        s.ram_total_mb = mem_total_kb / 1024;
        s.ram_used_mb  = (mem_total_kb - mem_avail_kb) / 1024;
    }

    return s;
}

// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------

bool SystemController::set_volume(int pct) {
    pct = std::max(0, std::min(100, pct));
    std::string cmd = "pactl set-sink-volume @DEFAULT_SINK@ " +
                      std::to_string(pct) + "% 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

bool SystemController::set_mute(bool muted) {
    std::string cmd = "pactl set-sink-mute @DEFAULT_SINK@ " +
                      std::string(muted ? "1" : "0") + " 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

// ---------------------------------------------------------------------------
// Brightness
// ---------------------------------------------------------------------------

bool SystemController::set_brightness(int pct) {
    pct = std::max(1, std::min(100, pct));
    std::string cmd = "brightnessctl set " + std::to_string(pct) + "% 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

// ---------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------

bool SystemController::set_wifi(bool enabled) {
    std::string cmd = "nmcli radio wifi " +
                      std::string(enabled ? "on" : "off") + " 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

// ---------------------------------------------------------------------------
// Bluetooth
// ---------------------------------------------------------------------------

bool SystemController::set_bluetooth(bool enabled) {
    std::string cmd = "bluetoothctl power " +
                      std::string(enabled ? "on" : "off") + " 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

// ---------------------------------------------------------------------------
// Notifications (GNOME DND via gsettings)
// ---------------------------------------------------------------------------

bool SystemController::set_notifications(bool enabled) {
    // GNOME: disable-notifications is inverted
    std::string val = enabled ? "false" : "true";
    std::string cmd = "gsettings set org.gnome.desktop.notifications "
                      "show-banners " + val + " 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

} // namespace cardinal
