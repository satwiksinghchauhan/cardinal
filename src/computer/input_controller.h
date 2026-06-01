#pragma once
// =============================================================================
// Cardinal - Input Controller
// File: src/computer/input_controller.h
//
// Abstracts mouse and keyboard input across X11 (xdotool) and
// Wayland (ydotool / wtype). Headless mode throws on any input attempt.
//
// Mouse:  move, click, double-click, right-click, scroll
// Keyboard: type text, send key combos (ctrl+c, Return, F5, etc.)
// =============================================================================

#include "computer/computer_types.h"
#include "computer/display_detector.h"
#include "utils/config_loader.h"

namespace cardinal {

    class InputController {
    public:
        explicit InputController(const DisplayDetector& display,
                                 const CardinalConfig&  config);
        ~InputController() = default;

        InputController(const InputController&)            = delete;
        InputController& operator=(const InputController&) = delete;

        // ------------------------------------------------------------------
        // Mouse
        // ------------------------------------------------------------------
        void mouse_move(int x, int y);
        void mouse_click(int x, int y,
                         MouseButton button = MouseButton::LEFT,
                         int         clicks = 1);
        void mouse_scroll(int x, int y, int delta_x, int delta_y);

        // ------------------------------------------------------------------
        // Keyboard
        // ------------------------------------------------------------------

        // Type a literal string (character by character, with optional delay)
        void type_text(const std::string& text, int delay_ms = 0);

        // Send a key combination: "ctrl+c", "Return", "alt+F4", etc.
        void send_key(const std::string& key_combo);

        // High-level: dispatch a KeyEvent (either type_text or send_key)
        void send_key_event(const KeyEvent& ev);

    private:
        void check_display() const;

        // X11 implementations
        void x11_mouse_move(int x, int y);
        void x11_mouse_click(int x, int y, MouseButton button, int clicks);
        void x11_mouse_scroll(int x, int y, int delta_x, int delta_y);
        void x11_type_text(const std::string& text, int delay_ms);
        void x11_send_key(const std::string& key_combo);

        // Wayland implementations
        void wl_mouse_move(int x, int y);
        void wl_mouse_click(int x, int y, MouseButton button, int clicks);
        void wl_mouse_scroll(int x, int y, int delta_x, int delta_y);
        void wl_type_text(const std::string& text, int delay_ms);
        void wl_send_key(const std::string& key_combo);

        static std::string shell_escape(const std::string& s);
        static int         mouse_button_to_xdotool(MouseButton b);
        static std::string mouse_button_to_ydotool(MouseButton b);

        const DisplayDetector& display_;
        const CardinalConfig&  config_;
    };

} // namespace cardinal
