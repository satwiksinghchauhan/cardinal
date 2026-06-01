#pragma once
// =============================================================================
// Cardinal - Watch Types
// File: src/watch/watch_types.h
//
// Shared configuration and status types for the watch/ subsystem.
// WatchEvent and WatchCallback are in computer_types.h.
// =============================================================================

#include "computer/computer_types.h"
#include <string>
#include <vector>

namespace cardinal {

    // -------------------------------------------------------------------------
    // FileWatchConfig
    // -------------------------------------------------------------------------
    struct FileWatchConfig {
        std::vector<std::string> paths;          // directories or files to watch
        bool                     recursive = false;
        bool                     watch_creates  = true;
        bool                     watch_modifies = true;
        bool                     watch_deletes  = true;
        bool                     watch_moves    = true;
    };

    // -------------------------------------------------------------------------
    // ScreenWatchConfig
    // -------------------------------------------------------------------------
    struct ScreenWatchConfig {
        int   interval_seconds = 2;   // how often to take a screenshot
        float change_threshold = 0.05f; // fraction of pixels that must differ
        bool  analyze_changes  = false; // run vision on changed frames
    };

    // -------------------------------------------------------------------------
    // ProcessWatchConfig
    // -------------------------------------------------------------------------
    struct ProcessWatchConfig {
        std::vector<std::string> watch_names;    // process names to watch
        bool                     watch_all = false; // watch any start/stop
        int                      poll_interval_seconds = 2;
    };

    // -------------------------------------------------------------------------
    // WatcherStatus
    // -------------------------------------------------------------------------
    struct WatcherStatus {
        bool        running       = false;
        std::string watcher_type; // "file", "screen", "process"
        int         events_fired  = 0;
        std::string last_event_at;
    };

} // namespace cardinal
