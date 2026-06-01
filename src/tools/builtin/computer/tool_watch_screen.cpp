// =============================================================================
// Cardinal - Tool: watch_screen Implementation
// =============================================================================

#include "tools/builtin/computer/tool_watch_screen.h"
#include "computer/screen_reader.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdio>
#include <array>

namespace fs = std::filesystem;

namespace cardinal {

ToolDefinition make_watch_screen_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "watch_screen";
    def.description =
        "Watch the screen and wait until a visual change occurs, or until a "
        "specific element or text appears. Polls at regular intervals.\n\n"
        "Useful for: waiting for a page to load, a dialog to appear, "
        "a download to finish, a process to complete.";
    def.confirmation_required = false;

    def.parameters.push_back({
        "wait_for", ToolParameterType::STRING,
        "Optional: description of what to wait for, e.g. 'the download complete notification', "
        "'a Save dialog'. If omitted, returns on any visual change.",
        false, ""
    });
    def.parameters.push_back({
        "timeout_seconds", ToolParameterType::NUMBER,
        "Maximum seconds to wait. Default: 30.",
        false, "30"
    });
    def.parameters.push_back({
        "poll_seconds", ToolParameterType::NUMBER,
        "How often to check the screen in seconds. Default: 2.",
        false, "2"
    });
    def.parameters.push_back({
        "analyze", ToolParameterType::BOOLEAN,
        "If true, describe what changed when a change is detected. Default: true.",
        false, "true"
    });
    (void)config;
    return def;
}

// Simple pixel diff using ImageMagick (same approach as screen_watcher)
static float quick_diff(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty() || !fs::exists(a) || !fs::exists(b)) return 1.0f;
    std::string cmd = "compare -metric PSNR " + a + " " + b + " /dev/null 2>&1 | "
                      "grep -oE '[0-9]+\\.[0-9]+' | head -1";
    char buf[64]{};
    float psnr = 0.0f;
    FILE* p = popen(cmd.c_str(), "r");
    if (p) {
        if (fgets(buf, sizeof(buf), p)) {
            try { psnr = std::stof(buf); } catch (...) {}
        }
        pclose(p);
    }
    if (psnr <= 0) return 1.0f;
    float norm = psnr / 50.0f;
    if (norm > 1.0f) norm = 1.0f;
    return 1.0f - norm;
}

ToolResult execute_watch_screen(const ToolCall& call, ScreenReader& reader) {
    ToolResult result;
    result.tool_name = "watch_screen";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string wait_for    = get("wait_for");
        int         timeout_s   = 30;
        int         poll_s      = 2;
        bool        analyze     = get("analyze", "true") != "false";
        try { timeout_s = std::stoi(get("timeout_seconds", "30")); } catch (...) {}
        try { poll_s    = std::stoi(get("poll_seconds", "2"));      } catch (...) {}

        // Take baseline screenshot
        std::string prev_path;
        try {
            auto baseline = reader.capture(false);
            prev_path = baseline.path;
        } catch (...) {
            result.status = ToolStatus::FAILURE;
            result.output = "Cannot capture screen (headless mode?)";
            return result;
        }

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_s);

        bool changed = false;
        std::string changed_path;
        std::string description;

        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::seconds(poll_s));

            try {
                auto s = reader.capture(false);
                float diff = quick_diff(prev_path, s.path);

                // Clean up old prev
                if (!prev_path.empty() && fs::exists(prev_path) && prev_path != s.path)
                    fs::remove(prev_path);
                prev_path = s.path;

                constexpr float CHANGE_THRESHOLD = 0.05f;
                if (diff < CHANGE_THRESHOLD) continue;

                // Screen changed — check if it matches wait_for
                if (!wait_for.empty() && analyze) {
                    description = reader.analyze(s.path,
                        "Is '" + wait_for + "' visible on screen? "
                        "Reply YES or NO, then briefly describe what you see.");
                    // Only count as match if model says YES
                    if (description.size() >= 2 &&
                        (description.substr(0, 2) == "YE" ||
                         description.substr(0, 2) == "ye" ||
                         description.substr(0, 2) == "Ye")) {
                        changed = true;
                        changed_path = s.path;
                        break;
                    }
                    // Not a match yet — keep waiting
                } else {
                    // Any change counts
                    changed = true;
                    changed_path = s.path;
                    if (analyze) description = reader.analyze(s.path);
                    break;
                }
            } catch (...) {}
        }

        // Cleanup prev
        if (!prev_path.empty() && prev_path != changed_path && fs::exists(prev_path))
            fs::remove(prev_path);

        if (changed) {
            result.status = ToolStatus::SUCCESS;
            result.output = "Screen change detected";
            if (!wait_for.empty()) result.output += " — '" + wait_for + "' found";
            result.output += ".\nScreenshot: " + changed_path;
            if (!description.empty()) result.output += "\nDescription: " + description;
        } else {
            result.status = ToolStatus::SUCCESS; // timeout is not an error
            result.output = "Timeout after " + std::to_string(timeout_s) +
                            "s — no matching screen change detected.";
            if (!wait_for.empty())
                result.output += " '" + wait_for + "' was not found.";
        }

    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "watch_screen failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
