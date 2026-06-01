// =============================================================================
// Cardinal - Screen Watcher Implementation
// File: src/watch/screen_watcher.cpp
// =============================================================================

#include "watch/screen_watcher.h"
#include "computer/screen_reader.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

namespace cardinal {

static std::string sw_now_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

ScreenWatcher::ScreenWatcher(const ScreenWatchConfig& cfg,
                             ScreenReader&            reader,
                             WatchCallback            cb)
    : config_(cfg), reader_(reader), callback_(std::move(cb))
{}

ScreenWatcher::~ScreenWatcher() { stop(); }

void ScreenWatcher::start() {
    stop_.store(false);
    running_.store(true);
    thread_ = std::thread([this]{ watch_loop(); });
    LOG_INFO("ScreenWatcher: started (interval=" +
             std::to_string(config_.interval_seconds) + "s)");
}

void ScreenWatcher::stop() {
    if (!running_.load()) return;
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

WatcherStatus ScreenWatcher::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    WatcherStatus s;
    s.running       = running_.load();
    s.watcher_type  = "screen";
    s.events_fired  = events_fired_;
    s.last_event_at = last_event_at_;
    return s;
}

// Simple pixel diff using ImageMagick compare if available, else 0
float ScreenWatcher::compute_diff(const std::string& path_a,
                                   const std::string& path_b) {
    if (path_a.empty() || path_b.empty()) return 1.0f;
    if (!fs::exists(path_a) || !fs::exists(path_b)) return 1.0f;

    // Use ImageMagick compare -metric PSNR — if not available, assume changed
    std::string cmd = "compare -metric PSNR " + path_a + " " + path_b +
                      " /dev/null 2>&1 | grep -oE '[0-9]+\\.[0-9]+'";
    char buf[64]{};
    float psnr = 0.0f;
    FILE* p = popen(cmd.c_str(), "r");
    if (p) {
        if (fgets(buf, sizeof(buf), p)) {
            try { psnr = std::stof(buf); } catch (...) {}
        }
        pclose(p);
    }
    // PSNR > 40 = very similar, < 30 = significant change
    // Normalize: diff = 1 - clamp(psnr/50, 0, 1)
    if (psnr <= 0) return 1.0f;
    float norm = psnr / 50.0f;
    if (norm > 1.0f) norm = 1.0f;
    return 1.0f - norm;
}

void ScreenWatcher::watch_loop() {
    while (!stop_.load()) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.interval_seconds));
        if (stop_.load()) break;

        try {
            auto s = reader_.capture(false);
            float diff = compute_diff(prev_screenshot_, s.path);

            if (diff >= config_.change_threshold) {
                WatchEvent we;
                we.type        = WatchEventType::SCREEN_CHANGED;
                we.timestamp   = sw_now_iso();
                we.description = "Screen changed (diff=" +
                                 std::to_string(static_cast<int>(diff * 100)) + "%)";
                we.path        = s.path;

                if (config_.analyze_changes && !s.description.empty()) {
                    we.description += ": " + s.description;
                }

                if (callback_) callback_(we);

                std::lock_guard<std::mutex> lock(mutex_);
                ++events_fired_;
                last_event_at_ = we.timestamp;
            }

            // Cleanup old prev screenshot
            if (!prev_screenshot_.empty() && fs::exists(prev_screenshot_))
                fs::remove(prev_screenshot_);
            prev_screenshot_ = s.path;

        } catch (const std::exception& e) {
            LOG_DEBUG("ScreenWatcher: capture error: " + std::string(e.what()));
        }
    }
    running_.store(false);
}

} // namespace cardinal
