// =============================================================================
// Cardinal - Process Watcher Implementation
// File: src/watch/process_watcher.cpp
// =============================================================================

#include "watch/process_watcher.h"
#include "utils/logger.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace cardinal {

static std::string pw_now_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

ProcessWatcher::ProcessWatcher(const ProcessWatchConfig& cfg, WatchCallback cb)
    : config_(cfg), callback_(std::move(cb))
{}

ProcessWatcher::~ProcessWatcher() { stop(); }

void ProcessWatcher::start() {
    prev_procs_ = snapshot_processes();
    stop_.store(false);
    running_.store(true);
    thread_ = std::thread([this]{ watch_loop(); });
    LOG_INFO("ProcessWatcher: started");
}

void ProcessWatcher::stop() {
    if (!running_.load()) return;
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

WatcherStatus ProcessWatcher::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    WatcherStatus s;
    s.running       = running_.load();
    s.watcher_type  = "process";
    s.events_fired  = events_fired_;
    s.last_event_at = last_event_at_;
    return s;
}

// ---------------------------------------------------------------------------
// snapshot_processes — reads /proc
// ---------------------------------------------------------------------------

std::unordered_map<int, std::string> ProcessWatcher::snapshot_processes() {
    std::unordered_map<int, std::string> result;
    for (const auto& entry : fs::directory_iterator("/proc")) {
        const auto& p = entry.path().filename().string();
        bool all_digits = !p.empty() &&
            std::all_of(p.begin(), p.end(), ::isdigit);
        if (!all_digits) continue;

        int pid = std::stoi(p);
        std::ifstream comm("/proc/" + p + "/comm");
        if (!comm) continue;
        std::string name;
        std::getline(comm, name);
        if (!name.empty()) result[pid] = name;
    }
    return result;
}

// ---------------------------------------------------------------------------
// watch_loop
// ---------------------------------------------------------------------------

void ProcessWatcher::watch_loop() {
    while (!stop_.load()) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.poll_interval_seconds));
        if (stop_.load()) break;

        auto current = snapshot_processes();

        // Detect started processes
        for (const auto& [pid, name] : current) {
            if (prev_procs_.find(pid) == prev_procs_.end()) {
                bool watch_it = config_.watch_all;
                if (!watch_it) {
                    for (const auto& wn : config_.watch_names) {
                        if (name.find(wn) != std::string::npos) { watch_it = true; break; }
                    }
                }
                if (watch_it && callback_) {
                    WatchEvent we;
                    we.type         = WatchEventType::PROCESS_STARTED;
                    we.process_name = name;
                    we.pid          = pid;
                    we.timestamp    = pw_now_iso();
                    we.description  = "Process started: " + name +
                                      " (pid=" + std::to_string(pid) + ")";
                    callback_(we);
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++events_fired_;
                    last_event_at_ = we.timestamp;
                }
            }
        }

        // Detect stopped processes
        for (const auto& [pid, name] : prev_procs_) {
            if (current.find(pid) == current.end()) {
                bool watch_it = config_.watch_all;
                if (!watch_it) {
                    for (const auto& wn : config_.watch_names) {
                        if (name.find(wn) != std::string::npos) { watch_it = true; break; }
                    }
                }
                if (watch_it && callback_) {
                    WatchEvent we;
                    we.type         = WatchEventType::PROCESS_STOPPED;
                    we.process_name = name;
                    we.pid          = pid;
                    we.timestamp    = pw_now_iso();
                    we.description  = "Process stopped: " + name +
                                      " (pid=" + std::to_string(pid) + ")";
                    callback_(we);
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++events_fired_;
                    last_event_at_ = we.timestamp;
                }
            }
        }

        prev_procs_ = std::move(current);
    }
    running_.store(false);
}

} // namespace cardinal
