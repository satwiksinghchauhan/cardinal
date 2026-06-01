#pragma once
// =============================================================================
// Cardinal - File Watcher
// File: src/watch/file_watcher.h
//
// inotify-based file system event watcher (Linux).
// Calls WatchCallback on any matching file event.
// =============================================================================

#include "watch/watch_types.h"
#include <thread>
#include <atomic>
#include <mutex>

namespace cardinal {

    class FileWatcher {
    public:
        explicit FileWatcher(const FileWatchConfig& cfg, WatchCallback cb);
        ~FileWatcher();

        FileWatcher(const FileWatcher&)            = delete;
        FileWatcher& operator=(const FileWatcher&) = delete;

        void start();
        void stop();
        bool is_running() const { return running_.load(); }
        WatcherStatus status() const;

    private:
        void watch_loop();
        void add_watches(int inotify_fd);

        FileWatchConfig   config_;
        WatchCallback     callback_;
        std::thread       thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> stop_{false};
        mutable std::mutex mutex_;
        int               events_fired_ = 0;
        std::string       last_event_at_;
    };

} // namespace cardinal
