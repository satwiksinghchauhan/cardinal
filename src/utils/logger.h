// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Logger
// File: src/utils/logger.h
// Thread-safe logging with levels, file + console output, and timestamps.
// No external dependencies - STL only.
// =============================================================================

// Must come before any Windows headers to prevent macro pollution
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <string>
#include <fstream>
#include <mutex>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <source_location>

// Windows defines ERROR as a macro - undefine after any header inclusion
#ifdef ERROR
#undef ERROR
#endif

namespace cardinal {

    // -----------------------------------------------------------------------------
    // Log levels - ordered by severity
    // -----------------------------------------------------------------------------
    enum class LogLevel {
        TRACE = 0,  // Fine-grained debug info
        DEBUG = 1,  // General debug info
        INFO = 2,  // Normal operation
        WARN = 3,  // Unexpected but recoverable
        ERROR = 4,  // Errors that affect operation
        FATAL = 5   // Unrecoverable - program should exit
    };

    // -----------------------------------------------------------------------------
    // Logger
    // Singleton - one instance per process, shared across all modules.
    // Usage:
    //   Logger::instance().init("logs/cardinal.log", LogLevel::INFO);
    //   LOG_INFO("Engine initialized");
    //   LOG_DEBUG("Confidence score: {}", 0.87f);  // fmt-style via stream
    // -----------------------------------------------------------------------------
    class Logger {
    public:
        // Get singleton instance
        static Logger& instance();

        // Initialize logger - call once at startup
        // log_path: path to log file (created if not exists)
        // level:    minimum level to log (messages below this are ignored)
        // console:  also print to stdout/stderr (default true)
        void init(const std::string& log_path,
            LogLevel level = LogLevel::INFO,
            bool console = true);

        // Core log function - prefer macros below
        void log(LogLevel level,
            const std::string& message,
            const std::source_location& loc = std::source_location::current());

        // Change log level at runtime
        void set_level(LogLevel level);
        LogLevel get_level() const;

        // Flush pending writes
        void flush();

        // Shut down logger - flushes and closes file
        void shutdown();

        // Check if initialized
        bool is_initialized() const;

        // Disable copy/move - singleton
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;
        Logger(Logger&&) = delete;
        Logger& operator=(Logger&&) = delete;

    private:
        Logger() = default;
        ~Logger();

        std::ofstream       file_;
        LogLevel            level_ = LogLevel::INFO;
        bool                console_ = true;
        bool                initialized_ = false;
        mutable std::mutex  mutex_;

        // Internal helpers
        std::string format_message(LogLevel level,
            const std::string& message,
            const std::source_location& loc) const;
        std::string level_to_string(LogLevel level) const;
        std::string level_to_color(LogLevel level) const;
        std::string timestamp() const;
    };

    // -----------------------------------------------------------------------------
    // Convenience macros
    // These capture source location automatically.
    // Usage:
    //   LOG_INFO("Model loaded from: " + path);
    //   LOG_WARN("Confidence below threshold: " + std::to_string(conf));
    //   LOG_ERROR("Failed to parse feeling schema");
    // -----------------------------------------------------------------------------
#define LOG_TRACE(msg) \
    ::cardinal::Logger::instance().log(::cardinal::LogLevel::TRACE, (msg), std::source_location::current())

#define LOG_DEBUG(msg) \
    ::cardinal::Logger::instance().log(::cardinal::LogLevel::DEBUG, (msg), std::source_location::current())

#define LOG_INFO(msg) \
    ::cardinal::Logger::instance().log(::cardinal::LogLevel::INFO,  (msg), std::source_location::current())

#define LOG_WARN(msg) \
    ::cardinal::Logger::instance().log(::cardinal::LogLevel::WARN,  (msg), std::source_location::current())

#define LOG_ERROR(msg) \
    ::cardinal::Logger::instance().log(::cardinal::LogLevel::ERROR, (msg), std::source_location::current())

#define LOG_FATAL(msg) \
    ::cardinal::Logger::instance().log(::cardinal::LogLevel::FATAL, (msg), std::source_location::current())

// Stream-style macro for building messages inline:
// LOG_S(INFO) << "Loaded " << n << " rules from store";
#define LOG_S(level) \
    ::cardinal::LogStream(::cardinal::LogLevel::level, std::source_location::current())

// -----------------------------------------------------------------------------
// LogStream - enables LOG_S(INFO) << "value: " << val syntax
// -----------------------------------------------------------------------------
    class LogStream {
    public:
        LogStream(LogLevel level, std::source_location loc)
            : level_(level), loc_(loc) {
        }

        ~LogStream() {
            Logger::instance().log(level_, stream_.str(), loc_);
        }

        template<typename T>
        LogStream& operator<<(const T& val) {
            stream_ << val;
            return *this;
        }

    private:
        LogLevel              level_;
        std::source_location  loc_;
        std::ostringstream    stream_;
    };

} // namespace cardinal