#pragma once
// =============================================================================
// Cardinal - Browser Controller
// File: src/computer/browser_controller.h
//
// Controls a browser via Playwright (Python subprocess in cardinal-browser-venv).
// A persistent Python helper process is spawned on first use and kept alive for
// the session lifetime. Communication is via stdin/stdout JSON lines.
//
// Supports: navigate, click (CSS selector or visual description), type,
//           scroll, screenshot, get_content, execute_js, new/close tab,
//           back, forward, reload.
//
// If a selector-based click fails and a VisionEncoder is available, falls back
// to screenshot + vision to locate and click by visual description.
// =============================================================================

#include "computer/computer_types.h"
#include "utils/config_loader.h"

#include <string>
#include <memory>
#include <mutex>
#include <optional>

namespace cardinal {

    class VisionEncoder;

    class BrowserController {
    public:
        explicit BrowserController(const CardinalConfig& config,
                                   VisionEncoder*        vision = nullptr);
        ~BrowserController();

        BrowserController(const BrowserController&)            = delete;
        BrowserController& operator=(const BrowserController&) = delete;

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------

        // Start the Playwright helper process. Called lazily on first use.
        bool start();

        // Stop the helper process.
        void stop();

        bool is_running() const { return proc_ != nullptr; }

        // ------------------------------------------------------------------
        // Actions (all blocking, return on completion)
        // ------------------------------------------------------------------

        BrowserResult execute(const BrowserAction& action);

        // Convenience wrappers
        BrowserResult navigate(const std::string& url);
        BrowserResult click(const std::string& selector);
        BrowserResult click_text(const std::string& visible_text);
        BrowserResult type(const std::string& selector, const std::string& text);
        BrowserResult get_content();
        BrowserResult screenshot(const std::string& save_path = "");
        BrowserResult execute_js(const std::string& js);

        // ------------------------------------------------------------------
        // Domain safety check
        // ------------------------------------------------------------------
        bool is_domain_allowed(const std::string& url) const;

    private:
        bool        ensure_started();
        std::string send_action(const std::string& json_request);
        BrowserResult parse_response(const std::string& json_response) const;

        static std::string build_helper_script();
        static std::string venv_python(const std::string& venv_path);

        const CardinalConfig& config_;
        VisionEncoder*        vision_;

        FILE*      proc_    = nullptr;   // popen handle to helper stdin
        FILE*      proc_out_= nullptr;   // read end of helper stdout
        int        proc_pid_= 0;
        std::mutex mutex_;

        std::string helper_script_path_;
    };

} // namespace cardinal
