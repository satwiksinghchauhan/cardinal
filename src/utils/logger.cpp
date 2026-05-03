// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Logger Implementation
// File: src/utils/logger.cpp
// =============================================================================

#include "logger.h"

#include <filesystem>
#include <stdexcept>
#include <array>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // ANSI color codes for console output
    // Windows 10+ supports ANSI via ENABLE_VIRTUAL_TERMINAL_PROCESSING
    // -----------------------------------------------------------------------------
    static constexpr const char* COLOR_RESET = "\033[0m";
    static constexpr const char* COLOR_GREY = "\033[90m";
    static constexpr const char* COLOR_CYAN = "\033[36m";
    static constexpr const char* COLOR_GREEN = "\033[32m";
    static constexpr const char* COLOR_YELLOW = "\033[33m";
    static constexpr const char* COLOR_RED = "\033[31m";
    static constexpr const char* COLOR_BOLD_RED = "\033[1;31m";

    // -----------------------------------------------------------------------------
    // Enable ANSI colors on Windows console
    // -----------------------------------------------------------------------------
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// Windows headers define ERROR as a macro - this collides with LogLevel::ERROR.
// Undefine it immediately after inclusion.
#ifdef ERROR
#undef ERROR
#endif
    static void enable_ansi_console() {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut == INVALID_HANDLE_VALUE) return;
        DWORD dwMode = 0;
        if (!GetConsoleMode(hOut, &dwMode)) return;
        SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#else
    static void enable_ansi_console() {}
#endif

    // -----------------------------------------------------------------------------
    // Singleton instance
    // -----------------------------------------------------------------------------
    Logger& Logger::instance() {
        static Logger inst;
        return inst;
    }

    // -----------------------------------------------------------------------------
    // Destructor - ensure clean shutdown
    // -----------------------------------------------------------------------------
    Logger::~Logger() {
        shutdown();
    }

    // -----------------------------------------------------------------------------
    // init
    // -----------------------------------------------------------------------------
    void Logger::init(const std::string& log_path,
        LogLevel level,
        bool console) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (initialized_) return; // Already initialized - ignore duplicate calls

        level_ = level;
        console_ = console;

        // Enable ANSI colors on Windows
        enable_ansi_console();

        // Create log directory if it doesn't exist
        std::filesystem::path path(log_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        // Open log file (append mode - preserves logs across restarts)
        file_.open(log_path, std::ios::app);
        if (!file_.is_open()) {
            throw std::runtime_error("Logger: failed to open log file: " + log_path);
        }

        initialized_ = true;

        // Write session separator so log file is readable across runs
        std::string sep(80, '=');
        file_ << "\n" << sep << "\n";
        file_ << "  Cardinal Session Started: " << timestamp() << "\n";
        file_ << sep << "\n\n";
        file_.flush();
    }

    // -----------------------------------------------------------------------------
    // log - core logging function
    // -----------------------------------------------------------------------------
    void Logger::log(LogLevel level,
        const std::string& message,
        const std::source_location& loc) {
        if (!initialized_) return;
        if (level < level_)  return; // Below minimum level - discard

        std::string formatted = format_message(level, message, loc);

        std::lock_guard<std::mutex> lock(mutex_);

        // Write to file (no colors)
        file_ << formatted << "\n";

        // Write to console (with colors)
        if (console_) {
            // FATAL and ERROR go to stderr, rest to stdout
            std::ostream& out = (level >= LogLevel::ERROR) ? std::cerr : std::cout;
            out << level_to_color(level) << formatted << COLOR_RESET << "\n";
        }

        // Auto-flush on WARN and above to ensure critical messages are written
        if (level >= LogLevel::WARN) {
            file_.flush();
            if (console_) {
                std::cout.flush();
                std::cerr.flush();
            }
        }
    }

    // -----------------------------------------------------------------------------
    // set_level / get_level
    // -----------------------------------------------------------------------------
    void Logger::set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    LogLevel Logger::get_level() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return level_;
    }

    // -----------------------------------------------------------------------------
    // flush
    // -----------------------------------------------------------------------------
    void Logger::flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.flush();
    }

    // -----------------------------------------------------------------------------
    // shutdown
    // -----------------------------------------------------------------------------
    void Logger::shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;

        if (file_.is_open()) {
            file_ << "\n[" << timestamp() << "] Cardinal session ended.\n";
            file_.flush();
            file_.close();
        }

        initialized_ = false;
    }

    // -----------------------------------------------------------------------------
    // is_initialized
    // -----------------------------------------------------------------------------
    bool Logger::is_initialized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_;
    }

    // -----------------------------------------------------------------------------
    // format_message
    // Produces: [TIMESTAMP] [LEVEL] message  (file:line)
    // Example:  [2026-04-03 14:22:01.342] [INFO ] Model loaded  (llm_engine.cpp:87)
    // -----------------------------------------------------------------------------
    std::string Logger::format_message(LogLevel level,
        const std::string& message,
        const std::source_location& loc) const {
        std::ostringstream oss;

        // Timestamp
        oss << "[" << timestamp() << "] ";

        // Level (fixed width 5 chars)
        oss << "[" << level_to_string(level) << "] ";

        // Message
        oss << message;

        // Source location (filename only, not full path - keeps logs readable)
        std::string filename = std::filesystem::path(loc.file_name()).filename().string();
        oss << "  (" << filename << ":" << loc.line() << ")";

        return oss.str();
    }

    // -----------------------------------------------------------------------------
    // level_to_string - fixed 5-char width for alignment
    // -----------------------------------------------------------------------------
    std::string Logger::level_to_string(LogLevel level) const {
        switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default:              return "?????";
        }
    }

    // -----------------------------------------------------------------------------
    // level_to_color - ANSI color per level
    // -----------------------------------------------------------------------------
    std::string Logger::level_to_color(LogLevel level) const {
        switch (level) {
        case LogLevel::TRACE: return COLOR_GREY;
        case LogLevel::DEBUG: return COLOR_CYAN;
        case LogLevel::INFO:  return COLOR_GREEN;
        case LogLevel::WARN:  return COLOR_YELLOW;
        case LogLevel::ERROR: return COLOR_RED;
        case LogLevel::FATAL: return COLOR_BOLD_RED;
        default:              return COLOR_RESET;
        }
    }

    // -----------------------------------------------------------------------------
    // timestamp - millisecond precision
    // -----------------------------------------------------------------------------
    std::string Logger::timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::tm tm_info{};
#ifdef _WIN32
        localtime_s(&tm_info, &time_t);
#else
        localtime_r(&time_t, &tm_info);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

} // namespace cardinal