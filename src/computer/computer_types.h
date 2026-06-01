#pragma once
// =============================================================================
// Cardinal - Computer Use Types
// File: src/computer/computer_types.h
//
// Shared types across all computer use controllers:
//   DisplayDetector, ScreenReader, InputController, AppController,
//   BrowserController, ShellExecutor, FileManager, SystemController,
//   EmailController, AtSpiReader
//
// No dependencies on other Cardinal headers — safe to include anywhere.
// =============================================================================

#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <functional>

namespace cardinal {

    // =========================================================================
    // Display / session types
    // =========================================================================

    enum class DisplayServer {
        X11,
        WAYLAND,
        HEADLESS   // no display — shell + browser only
    };

    inline const char* display_server_to_string(DisplayServer d) {
        switch (d) {
            case DisplayServer::X11:      return "x11";
            case DisplayServer::WAYLAND:  return "wayland";
            case DisplayServer::HEADLESS: return "headless";
            default:                      return "unknown";
        }
    }

    // =========================================================================
    // Geometry
    // =========================================================================

    struct Point {
        int x = 0;
        int y = 0;
    };

    struct ScreenRegion {
        int x      = 0;
        int y      = 0;
        int width  = 0;
        int height = 0;

        bool contains(int px, int py) const {
            return px >= x && px < x + width &&
                   py >= y && py < y + height;
        }
        int cx() const { return x + width  / 2; }
        int cy() const { return y + height / 2; }
    };

    struct ScreenInfo {
        int           width        = 0;
        int           height       = 0;
        float         scale_factor = 1.0f;  // HiDPI
        DisplayServer server       = DisplayServer::HEADLESS;
        std::string   display_var; // e.g. ":0" or "wayland-0"
    };

    // =========================================================================
    // Screenshot
    // =========================================================================

    struct Screenshot {
        std::string   path;          // absolute path to captured PNG
        int           width  = 0;
        int           height = 0;
        std::string   timestamp;

        // Optional crop region used when screenshot was partial
        std::optional<ScreenRegion> region;

        // Vision analysis result (populated by ScreenReader if requested)
        std::string   description;
        bool          analyzed = false;
    };

    // =========================================================================
    // Mouse / click events
    // =========================================================================

    enum class MouseButton { LEFT, MIDDLE, RIGHT };

    struct MouseEvent {
        int         x      = 0;
        int         y      = 0;
        MouseButton button = MouseButton::LEFT;
        int         clicks = 1;     // 1 = single, 2 = double
        bool        move_only = false; // just move, no click
    };

    struct ScrollEvent {
        int x         = 0;
        int y         = 0;
        int delta_x   = 0;   // horizontal scroll (positive = right)
        int delta_y   = 0;   // vertical scroll (positive = down)
    };

    // =========================================================================
    // Keyboard events
    // =========================================================================

    struct KeyEvent {
        std::string text;       // type this literal text, OR
        std::string key_combo;  // send this key combo e.g. "ctrl+c", "Return", "F5"
        int         delay_ms = 0; // delay between keystrokes (for text)
    };

    // =========================================================================
    // Application info
    // =========================================================================

    struct AppInfo {
        std::string name;           // display name
        std::string executable;     // e.g. "google-chrome"
        std::string window_title;   // current window title
        int         pid     = 0;
        bool        focused = false;
        bool        visible = true;
        ScreenRegion window_rect;
    };

    // =========================================================================
    // AT-SPI node (accessibility tree)
    // =========================================================================

    struct AtSpiNode {
        std::string              role;         // "button", "text", "menu", etc.
        std::string              name;         // accessible name
        std::string              description;
        std::string              value;        // current value (for inputs)
        ScreenRegion             bounds;
        bool                     focusable = false;
        bool                     focused   = false;
        bool                     enabled   = true;
        std::vector<std::string> states;      // AT-SPI state set
        std::vector<AtSpiNode>   children;

        // Path from root — useful for logging
        std::string path;
    };

    // =========================================================================
    // Browser types
    // =========================================================================

    enum class BrowserActionType {
        NAVIGATE,
        CLICK,
        TYPE,
        SCROLL,
        SCREENSHOT,
        GET_CONTENT,
        EXECUTE_JS,
        WAIT_FOR,
        NEW_TAB,
        CLOSE_TAB,
        BACK,
        FORWARD,
        RELOAD,
    };

    inline const char* browser_action_to_string(BrowserActionType t) {
        switch (t) {
            case BrowserActionType::NAVIGATE:    return "navigate";
            case BrowserActionType::CLICK:       return "click";
            case BrowserActionType::TYPE:        return "type";
            case BrowserActionType::SCROLL:      return "scroll";
            case BrowserActionType::SCREENSHOT:  return "screenshot";
            case BrowserActionType::GET_CONTENT: return "get_content";
            case BrowserActionType::EXECUTE_JS:  return "execute_js";
            case BrowserActionType::WAIT_FOR:    return "wait_for";
            case BrowserActionType::NEW_TAB:     return "new_tab";
            case BrowserActionType::CLOSE_TAB:   return "close_tab";
            case BrowserActionType::BACK:        return "back";
            case BrowserActionType::FORWARD:     return "forward";
            case BrowserActionType::RELOAD:      return "reload";
            default:                             return "unknown";
        }
    }

    struct BrowserAction {
        BrowserActionType type    = BrowserActionType::NAVIGATE;
        std::string       url;            // NAVIGATE
        std::string       selector;       // CSS selector for CLICK / TYPE / WAIT_FOR
        std::string       text;           // TYPE / EXECUTE_JS
        std::string       description;    // vision-based click description (fallback)
        int               timeout_ms = 0; // 0 = use config default
        ScrollEvent       scroll;         // SCROLL
    };

    struct BrowserResult {
        bool        success = false;
        std::string url;            // current URL after action
        std::string title;          // current page title
        std::string content;        // page text / JS result
        std::string screenshot_path;
        std::string error_message;
        int         duration_ms = 0;
    };

    // =========================================================================
    // Shell result
    // =========================================================================

    struct ShellResult {
        bool        success    = false;
        int         exit_code  = 0;
        std::string stdout_text;
        std::string stderr_text;
        int         duration_ms = 0;
        std::string command;
    };

    // =========================================================================
    // File operation types
    // =========================================================================

    enum class FileOpType {
        LIST,
        MOVE,
        COPY,
        DELETE,
        MKDIR,
        EXISTS,
        STAT,
        WATCH_START,
        WATCH_STOP,
    };

    struct FileEntry {
        std::string path;
        std::string name;
        bool        is_dir     = false;
        long long   size_bytes = 0;
        std::string modified_at;
        std::string permissions; // e.g. "rwxr-xr-x"
    };

    struct FileOpResult {
        bool                    success = false;
        std::string             error_message;
        std::vector<FileEntry>  entries;     // for LIST
        std::string             dest_path;   // for MOVE / COPY
        int                     duration_ms = 0;
    };

    // =========================================================================
    // System control
    // =========================================================================

    struct SystemState {
        int         volume_pct      = -1;    // 0-100, -1 = unknown
        bool        muted           = false;
        int         brightness_pct  = -1;    // 0-100
        bool        wifi_enabled    = false;
        std::string wifi_ssid;
        bool        bluetooth_enabled = false;
        std::string os_name;
        std::string hostname;
        float       cpu_pct         = 0.0f;
        long long   ram_used_mb     = 0;
        long long   ram_total_mb    = 0;
    };

    // =========================================================================
    // Email types
    // =========================================================================

    struct EmailMessage {
        std::string              message_id;
        std::string              from;
        std::vector<std::string> to;
        std::vector<std::string> cc;
        std::string              subject;
        std::string              body_text;
        std::string              body_html;
        std::string              date;
        bool                     read   = false;
        bool                     starred = false;
        std::vector<std::string> labels;
    };

    struct EmailQuery {
        std::string              folder     = "INBOX";
        std::string              subject_contains;
        std::string              from_contains;
        std::string              body_contains;
        bool                     unread_only = false;
        int                      max_results = 10;
        int                      offset      = 0;
    };

    struct EmailSendRequest {
        std::vector<std::string> to;
        std::vector<std::string> cc;
        std::string              subject;
        std::string              body;
        bool                     html = false;
    };

    // =========================================================================
    // Computer use result — generic wrapper returned by all tool handlers
    // =========================================================================

    struct ComputerUseResult {
        bool        success = false;
        std::string output;          // human-readable result injected to LLM context
        std::string error_message;
        int         duration_ms = 0;

        // Optional structured payloads (only one will be set)
        std::optional<Screenshot>   screenshot;
        std::optional<ShellResult>  shell;
        std::optional<BrowserResult> browser;
        std::optional<FileOpResult> file_op;
        std::optional<SystemState>  system_state;
    };

    // =========================================================================
    // Watch event (used by watch/ subsystem)
    // =========================================================================

    enum class WatchEventType {
        FILE_CREATED,
        FILE_MODIFIED,
        FILE_DELETED,
        FILE_MOVED,
        SCREEN_CHANGED,
        PROCESS_STARTED,
        PROCESS_STOPPED,
    };

    inline const char* watch_event_to_string(WatchEventType t) {
        switch (t) {
            case WatchEventType::FILE_CREATED:    return "file_created";
            case WatchEventType::FILE_MODIFIED:   return "file_modified";
            case WatchEventType::FILE_DELETED:    return "file_deleted";
            case WatchEventType::FILE_MOVED:      return "file_moved";
            case WatchEventType::SCREEN_CHANGED:  return "screen_changed";
            case WatchEventType::PROCESS_STARTED: return "process_started";
            case WatchEventType::PROCESS_STOPPED: return "process_stopped";
            default:                              return "unknown";
        }
    }

    struct WatchEvent {
        WatchEventType type;
        std::string    path;         // file path (for file events)
        std::string    dest_path;    // for FILE_MOVED
        std::string    process_name; // for process events
        int            pid    = 0;
        std::string    timestamp;
        std::string    description;  // human-readable summary
    };

    using WatchCallback = std::function<void(const WatchEvent&)>;

} // namespace cardinal
