#pragma once
// =============================================================================
// Cardinal - Shell Executor
// File: src/computer/shell_executor.h
//
// Sandboxed shell command execution.
// - Checks against config blocked_commands before running
// - Captures stdout + stderr separately via pipe pair
// - Enforces timeout (SIGKILL on expiry)
// - Working directory defaults to config.computer_use.shell.working_directory
// =============================================================================

#include "computer/computer_types.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <chrono>

namespace cardinal {

    class ShellExecutor {
    public:
        explicit ShellExecutor(const CardinalConfig& config);
        ~ShellExecutor() = default;

        ShellExecutor(const ShellExecutor&)            = delete;
        ShellExecutor& operator=(const ShellExecutor&) = delete;

        // Run command. Blocks until complete or timeout.
        ShellResult run(const std::string& command,
                        int                timeout_seconds = 0,
                        const std::string& working_dir    = "");

        // Check if command is blocked by config
        bool is_blocked(const std::string& command) const;

    private:
        static std::string expand_home(const std::string& path);

        const CardinalConfig& config_;
    };

} // namespace cardinal
