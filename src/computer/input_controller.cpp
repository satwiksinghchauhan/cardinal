// =============================================================================
// Cardinal - Input Controller Implementation
// File: src/computer/input_controller.cpp
// =============================================================================

#include "computer/input_controller.h"
#include "utils/logger.h"

#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <chrono>

namespace cardinal {

InputController::InputController(const DisplayDetector& display,
                                 const CardinalConfig&  config)
    : display_(display), config_(config)
{}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void InputController::check_display() const {
    if (display_.is_headless())
        throw std::runtime_error("InputController: cannot send input in headless mode");
}

std::string InputController::shell_escape(const std::string& s) {
    // Wrap in single quotes, escaping any existing single quotes
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

int InputController::mouse_button_to_xdotool(MouseButton b) {
    switch (b) {
        case MouseButton::LEFT:   return 1;
        case MouseButton::MIDDLE: return 2;
        case MouseButton::RIGHT:  return 3;
        default:                  return 1;
    }
}

std::string InputController::mouse_button_to_ydotool(MouseButton b) {
    switch (b) {
        case MouseButton::LEFT:   return "0x110";
        case MouseButton::MIDDLE: return "0x112";
        case MouseButton::RIGHT:  return "0x111";
        default:                  return "0x110";
    }
}

// ---------------------------------------------------------------------------
// Public dispatch methods
// ---------------------------------------------------------------------------

void InputController::mouse_move(int x, int y) {
    check_display();
    if (display_.is_x11())     x11_mouse_move(x, y);
    else                       wl_mouse_move(x, y);
}

void InputController::mouse_click(int x, int y, MouseButton button, int clicks) {
    check_display();
    if (display_.is_x11())     x11_mouse_click(x, y, button, clicks);
    else                       wl_mouse_click(x, y, button, clicks);
}

void InputController::mouse_scroll(int x, int y, int delta_x, int delta_y) {
    check_display();
    if (display_.is_x11())     x11_mouse_scroll(x, y, delta_x, delta_y);
    else                       wl_mouse_scroll(x, y, delta_x, delta_y);
}

void InputController::type_text(const std::string& text, int delay_ms) {
    check_display();
    if (display_.is_x11())     x11_type_text(text, delay_ms);
    else                       wl_type_text(text, delay_ms);
}

void InputController::send_key(const std::string& key_combo) {
    check_display();
    if (display_.is_x11())     x11_send_key(key_combo);
    else                       wl_send_key(key_combo);
}

void InputController::send_key_event(const KeyEvent& ev) {
    if (!ev.key_combo.empty())
        send_key(ev.key_combo);
    else if (!ev.text.empty())
        type_text(ev.text, ev.delay_ms);
}

// ---------------------------------------------------------------------------
// X11 implementations (xdotool)
// ---------------------------------------------------------------------------

void InputController::x11_mouse_move(int x, int y) {
    std::string cmd = "xdotool mousemove " +
                      std::to_string(x) + " " + std::to_string(y);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("InputController: xdotool mousemove failed");
}

void InputController::x11_mouse_click(int x, int y, MouseButton button, int clicks) {
    int btn = mouse_button_to_xdotool(button);
    std::string cmd = "xdotool mousemove " +
                      std::to_string(x) + " " + std::to_string(y) +
                      " click --clearmodifiers ";
    if (clicks == 2) cmd += "--repeat 2 ";
    cmd += std::to_string(btn);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("InputController: xdotool click failed");
}

void InputController::x11_mouse_scroll(int x, int y, int delta_x, int delta_y) {
    // xdotool scroll: button 4=up 5=down 6=left 7=right
    x11_mouse_move(x, y);
    auto scroll_once = [&](int btn, int times) {
        for (int i = 0; i < std::abs(times); ++i) {
            std::string cmd = "xdotool click " + std::to_string(btn);
            std::system(cmd.c_str());
        }
    };
    if (delta_y < 0) scroll_once(4, -delta_y); // up
    if (delta_y > 0) scroll_once(5,  delta_y); // down
    if (delta_x < 0) scroll_once(6, -delta_x); // left
    if (delta_x > 0) scroll_once(7,  delta_x); // right
}

void InputController::x11_type_text(const std::string& text, int delay_ms) {
    std::string cmd = "xdotool type --clearmodifiers ";
    if (delay_ms > 0)
        cmd += "--delay " + std::to_string(delay_ms) + " ";
    cmd += shell_escape(text);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("InputController: xdotool type failed");
}

void InputController::x11_send_key(const std::string& key_combo) {
    // xdotool uses "ctrl+c" format natively
    std::string cmd = "xdotool key --clearmodifiers " + shell_escape(key_combo);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("InputController: xdotool key failed: " + key_combo);
}

// ---------------------------------------------------------------------------
// Wayland implementations (ydotool / wtype)
// ---------------------------------------------------------------------------

void InputController::wl_mouse_move(int x, int y) {
    if (!display_.has_ydotool()) {
        LOG_WARN("InputController: ydotool not available for mouse move");
        return;
    }
    // ydotool mousemove uses absolute coords with --absolute
    std::string cmd = "ydotool mousemove --absolute -x " +
                      std::to_string(x) + " -y " + std::to_string(y);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("InputController: ydotool mousemove failed");
}

void InputController::wl_mouse_click(int x, int y, MouseButton button, int clicks) {
    wl_mouse_move(x, y);
    if (!display_.has_ydotool()) return;

    std::string btn = mouse_button_to_ydotool(button);
    for (int i = 0; i < clicks; ++i) {
        // click = key down + key up
        std::string cmd = "ydotool click " + btn;
        std::system(cmd.c_str());
        if (clicks > 1)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void InputController::wl_mouse_scroll(int x, int y, int delta_x, int delta_y) {
    wl_mouse_move(x, y);
    if (!display_.has_ydotool()) return;
    // ydotool scroll: positive y = scroll down
    if (delta_y != 0) {
        std::string cmd = "ydotool scroll -y " + std::to_string(delta_y * 3);
        std::system(cmd.c_str());
    }
    if (delta_x != 0) {
        std::string cmd = "ydotool scroll -x " + std::to_string(delta_x * 3);
        std::system(cmd.c_str());
    }
}

void InputController::wl_type_text(const std::string& text, int delay_ms) {
    // Prefer wtype (handles unicode better), fall back to ydotool
    if (display_.has_wtype()) {
        std::string cmd = "wtype ";
        if (delay_ms > 0)
            cmd += "-d " + std::to_string(delay_ms) + " ";
        cmd += shell_escape(text);
        int rc = std::system(cmd.c_str());
        if (rc != 0) LOG_WARN("InputController: wtype failed");
    } else if (display_.has_ydotool()) {
        std::string cmd = "ydotool type -- " + shell_escape(text);
        int rc = std::system(cmd.c_str());
        if (rc != 0) LOG_WARN("InputController: ydotool type failed");
    } else {
        LOG_WARN("InputController: no wayland typing tool available");
    }
}

void InputController::wl_send_key(const std::string& key_combo) {
    // wtype uses -k for key names; ydotool uses key with Linux keycodes.
    // We use wtype when available since it handles XKB key names.
    if (display_.has_wtype()) {
        std::string cmd = "wtype -k " + shell_escape(key_combo);
        int rc = std::system(cmd.c_str());
        if (rc != 0) LOG_WARN("InputController: wtype key failed: " + key_combo);
    } else if (display_.has_ydotool()) {
        std::string cmd = "ydotool key " + shell_escape(key_combo);
        int rc = std::system(cmd.c_str());
        if (rc != 0) LOG_WARN("InputController: ydotool key failed: " + key_combo);
    } else {
        LOG_WARN("InputController: no wayland key tool available");
    }
}

} // namespace cardinal
