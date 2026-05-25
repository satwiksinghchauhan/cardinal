#pragma once
// =============================================================================
// Cardinal - Process Watcher
// File: src/watch/process_watcher.h
//
// Polls /proc at a configurable interval and fires WatchCallback when a
// watched process starts or stops.
// =============================================================================

#include "watch/watch_types.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace cardinal {

    class ProcessWatcher {
    public:
        explicit ProcessWatcher(const ProcessWatchConfig& cfg, WatchCallback cb);
        ~ProcessWatcher();

        ProcessWatcher(const ProcessWatcher&)            = delete;
        ProcessWatcher& operator=(const ProcessWatcher&) = delete;

        void start();
        void stop();
        bool is_running() const { return running_.load(); }
        WatcherStatus status() const;

    private:
        void watch_loop();

        // Returns map of pid → process_name for all running processes
        static std::unordered_map<int, std::string> snapshot_processes();

        ProcessWatchConfig  config_;
        WatchCallback       callback_;
        std::thread         thread_;
        std::atomic<bool>   running_{false};
        std::atomic<bool>   stop_{false};
        mutable std::mutex  mutex_;
        int                 events_fired_ = 0;
        std::string         last_event_at_;

        // Previous snapshot: pid → name
        std::unordered_map<int, std::string> prev_procs_;
    };

} // namespace cardinal
