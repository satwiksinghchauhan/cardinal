// =============================================================================
// Cardinal - File Watcher Implementation
// File: src/watch/file_watcher.cpp
// =============================================================================

#include "watch/file_watcher.h"
#include "utils/logger.h"

#include <sys/inotify.h>
#include <unistd.h>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace cardinal {

static std::string now_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

FileWatcher::FileWatcher(const FileWatchConfig& cfg, WatchCallback cb)
    : config_(cfg), callback_(std::move(cb))
{}

FileWatcher::~FileWatcher() { stop(); }

void FileWatcher::start() {
    stop_.store(false);
    running_.store(true);
    thread_ = std::thread([this]{ watch_loop(); });
    LOG_INFO("FileWatcher: started");
}

void FileWatcher::stop() {
    if (!running_.load()) return;
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
    LOG_INFO("FileWatcher: stopped");
}

WatcherStatus FileWatcher::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    WatcherStatus s;
    s.running       = running_.load();
    s.watcher_type  = "file";
    s.events_fired  = events_fired_;
    s.last_event_at = last_event_at_;
    return s;
}

void FileWatcher::add_watches(int inotify_fd) {
    uint32_t mask = 0;
    if (config_.watch_creates)  mask |= IN_CREATE;
    if (config_.watch_modifies) mask |= IN_MODIFY | IN_CLOSE_WRITE;
    if (config_.watch_deletes)  mask |= IN_DELETE;
    if (config_.watch_moves)    mask |= IN_MOVED_FROM | IN_MOVED_TO;

    for (const auto& path : config_.paths) {
        if (!fs::exists(path)) continue;
        inotify_add_watch(inotify_fd, path.c_str(), mask);
        if (config_.recursive && fs::is_directory(path)) {
            for (const auto& entry : fs::recursive_directory_iterator(path)) {
                if (entry.is_directory())
                    inotify_add_watch(inotify_fd, entry.path().c_str(), mask);
            }
        }
    }
}

void FileWatcher::watch_loop() {
    int inotify_fd = inotify_init1(IN_NONBLOCK);
    if (inotify_fd < 0) {
        LOG_ERROR("FileWatcher: inotify_init1 failed");
        running_.store(false);
        return;
    }

    add_watches(inotify_fd);

    constexpr size_t BUF_LEN = 4096;
    char buf[BUF_LEN] __attribute__((aligned(__alignof__(struct inotify_event))));

    while (!stop_.load()) {
        ssize_t len = read(inotify_fd, buf, BUF_LEN);
        if (len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            break;
        }

        const struct inotify_event* event;
        for (char* ptr = buf; ptr < buf + len;
             ptr += sizeof(struct inotify_event) + event->len) {
            event = reinterpret_cast<const struct inotify_event*>(ptr);

            WatchEvent we;
            we.timestamp = now_iso();
            std::string name = (event->len > 0) ? event->name : "";

            if (event->mask & IN_CREATE)       { we.type = WatchEventType::FILE_CREATED;  we.path = name; }
            else if (event->mask & (IN_MODIFY | IN_CLOSE_WRITE)) { we.type = WatchEventType::FILE_MODIFIED; we.path = name; }
            else if (event->mask & IN_DELETE)  { we.type = WatchEventType::FILE_DELETED;  we.path = name; }
            else if (event->mask & IN_MOVED_FROM) { we.type = WatchEventType::FILE_MOVED; we.path = name; }
            else if (event->mask & IN_MOVED_TO)   { we.type = WatchEventType::FILE_MOVED; we.dest_path = name; }
            else continue;

            we.description = std::string(watch_event_to_string(we.type)) + ": " + name;

            if (callback_) callback_(we);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++events_fired_;
                last_event_at_ = we.timestamp;
            }
        }
    }

    close(inotify_fd);
    running_.store(false);
}

} // namespace cardinal
