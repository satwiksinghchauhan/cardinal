#pragma once
// =============================================================================
// Cardinal - Screen Watcher
// File: src/watch/screen_watcher.h
//
// Periodically takes a screenshot and computes pixel diff.
// Fires WatchCallback(SCREEN_CHANGED) when change exceeds threshold.
// =============================================================================

#include "watch/watch_types.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <string>

namespace cardinal {

    class ScreenReader;

    class ScreenWatcher {
    public:
        explicit ScreenWatcher(const ScreenWatchConfig& cfg,
                               ScreenReader&            reader,
                               WatchCallback            cb);
        ~ScreenWatcher();

        ScreenWatcher(const ScreenWatcher&)            = delete;
        ScreenWatcher& operator=(const ScreenWatcher&) = delete;

        void start();
        void stop();
        bool is_running() const { return running_.load(); }
        WatcherStatus status() const;

    private:
        void watch_loop();
        static float compute_diff(const std::string& path_a,
                                  const std::string& path_b);

        ScreenWatchConfig  config_;
        ScreenReader&      reader_;
        WatchCallback      callback_;
        std::thread        thread_;
        std::atomic<bool>  running_{false};
        std::atomic<bool>  stop_{false};
        mutable std::mutex mutex_;
        int                events_fired_ = 0;
        std::string        last_event_at_;
        std::string        prev_screenshot_;
    };

} // namespace cardinal
