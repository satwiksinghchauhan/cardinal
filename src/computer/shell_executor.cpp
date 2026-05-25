// =============================================================================
// Cardinal - Shell Executor Implementation
// File: src/computer/shell_executor.cpp
// =============================================================================

#include "computer/shell_executor.h"
#include "utils/logger.h"

#include <stdexcept>
#include <sstream>
#include <array>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>

namespace fs = std::filesystem;

namespace cardinal {

ShellExecutor::ShellExecutor(const CardinalConfig& config)
    : config_(config)
{}

std::string ShellExecutor::expand_home(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

bool ShellExecutor::is_blocked(const std::string& command) const {
    for (const auto& blocked : config_.computer_use.safety.blocked_commands) {
        if (command.find(blocked) != std::string::npos) {
            LOG_WARN("ShellExecutor: blocked command pattern '" + blocked + "'");
            return true;
        }
    }
    return false;
}

ShellResult ShellExecutor::run(const std::string& command,
                                int                timeout_seconds,
                                const std::string& working_dir) {
    ShellResult result;
    result.command = command;

    if (!config_.computer_use.shell.enabled) {
        result.stderr_text = "Shell execution disabled in config";
        return result;
    }

    if (is_blocked(command)) {
        result.exit_code     = -1;
        result.stderr_text   = "Command blocked by safety config";
        return result;
    }

    int timeout = timeout_seconds > 0
        ? timeout_seconds
        : config_.computer_use.shell.timeout_seconds;

    std::string cwd = working_dir.empty()
        ? expand_home(config_.computer_use.shell.working_directory)
        : working_dir;
    if (cwd.empty()) cwd = ".";

    // Build shell: cd to working dir, then run command
    std::string full_cmd = "cd " + cwd + " && " + command;

    // Use popen for simplicity; capture stdout+stderr combined
    // For production, replace with posix_spawn + pipe pair for separate streams
    auto t0 = std::chrono::steady_clock::now();

    // Add timeout wrapper via 'timeout' coreutils command if available
    std::string timed_cmd;
    if (timeout > 0)
        timed_cmd = "timeout " + std::to_string(timeout) + "s bash -c " +
                    "'" + full_cmd + "' 2>&1";
    else
        timed_cmd = "bash -c '" + full_cmd + "' 2>&1";

    FILE* pipe = popen(timed_cmd.c_str(), "r");
    if (!pipe) {
        result.exit_code   = -1;
        result.stderr_text = "popen failed: " + std::string(strerror(errno));
        return result;
    }

    std::array<char, 4096> buf{};
    std::string output;
    while (fgets(buf.data(), buf.size(), pipe)) {
        output += buf.data();
        // Cap output at 1MB
        if (output.size() > 1024 * 1024) {
            output += "\n[output truncated]";
            break;
        }
    }

    int status  = pclose(pipe);
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    auto ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count());

    result.stdout_text  = output;
    result.exit_code    = exit_code;
    result.success      = (exit_code == 0);
    result.duration_ms  = ms;

    if (exit_code == 124) {
        result.success     = false;
        result.stderr_text = "Command timed out after " + std::to_string(timeout) + "s";
    }

    LOG_DEBUG("ShellExecutor: exit=" + std::to_string(exit_code) +
              " ms=" + std::to_string(ms) + " cmd=" + command.substr(0, 80));
    return result;
}

} // namespace cardinal
