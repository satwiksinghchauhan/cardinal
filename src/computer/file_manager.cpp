// =============================================================================
// Cardinal - File Manager Implementation
// File: src/computer/file_manager.cpp
// =============================================================================

#include "computer/file_manager.h"
#include "utils/logger.h"

#include <filesystem>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;

namespace cardinal {

FileManager::FileManager(const CardinalConfig& config)
    : config_(config)
{}

std::string FileManager::expand_home(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

std::string FileManager::permissions_string(const fs::perms& p) {
    auto bit = [&](fs::perms perm, char c) -> char {
        return (p & perm) != fs::perms::none ? c : '-';
    };
    std::string s;
    s += bit(fs::perms::owner_read,    'r');
    s += bit(fs::perms::owner_write,   'w');
    s += bit(fs::perms::owner_exec,    'x');
    s += bit(fs::perms::group_read,    'r');
    s += bit(fs::perms::group_write,   'w');
    s += bit(fs::perms::group_exec,    'x');
    s += bit(fs::perms::others_read,   'r');
    s += bit(fs::perms::others_write,  'w');
    s += bit(fs::perms::others_exec,   'x');
    return s;
}

std::string FileManager::iso_time(const fs::file_time_type& ft) {
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    std::time_t t = std::chrono::system_clock::to_time_t(sctp);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

bool FileManager::is_write_allowed() const {
    return config_.computer_use.safety.allow_file_write;
}

bool FileManager::is_path_allowed(const std::string& raw_path) const {
    if (!config_.computer_use.safety.whitelist_enabled) return true;
    std::string path = expand_home(raw_path);
    fs::path abs;
    try { abs = fs::absolute(path); } catch (...) { return false; }

    const auto& allowed = config_.computer_use.safety.allowed_paths;
    if (allowed.empty()) return true;
    for (const auto& a : allowed) {
        fs::path allowed_abs;
        try { allowed_abs = fs::absolute(expand_home(a)); } catch (...) { continue; }
        // Check if abs starts with allowed_abs
        auto [it, end] = std::mismatch(allowed_abs.begin(), allowed_abs.end(),
                                        abs.begin());
        if (it == allowed_abs.end()) return true;
    }
    LOG_WARN("FileManager: path not in allowed_paths: " + path);
    return false;
}

bool FileManager::exists(const std::string& path) const {
    return fs::exists(expand_home(path));
}

FileOpResult FileManager::list(const std::string& raw_path, bool recursive) const {
    FileOpResult r;
    std::string path = expand_home(raw_path);
    if (!is_path_allowed(path)) { r.error_message = "Path not allowed"; return r; }
    if (!fs::exists(path))      { r.error_message = "Path does not exist"; return r; }

    try {
        auto add_entry = [&](const fs::directory_entry& de) {
            FileEntry e;
            e.path     = de.path().string();
            e.name     = de.path().filename().string();
            e.is_dir   = de.is_directory();
            if (!e.is_dir) {
                try { e.size_bytes = static_cast<long long>(de.file_size()); } catch (...) {}
            }
            try { e.modified_at  = iso_time(de.last_write_time()); } catch (...) {}
            try { e.permissions  = permissions_string(de.status().permissions()); } catch (...) {}
            r.entries.push_back(e);
        };

        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(path))
                add_entry(entry);
        } else {
            for (const auto& entry : fs::directory_iterator(path))
                add_entry(entry);
        }
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    return r;
}

FileOpResult FileManager::move(const std::string& raw_src, const std::string& raw_dst) {
    FileOpResult r;
    if (!is_write_allowed()) { r.error_message = "File write not allowed"; return r; }
    std::string src = expand_home(raw_src);
    std::string dst = expand_home(raw_dst);
    if (!is_path_allowed(src) || !is_path_allowed(dst)) {
        r.error_message = "Path not allowed"; return r;
    }
    try {
        fs::create_directories(fs::path(dst).parent_path());
        fs::rename(src, dst);
        r.success   = true;
        r.dest_path = dst;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    return r;
}

FileOpResult FileManager::copy(const std::string& raw_src, const std::string& raw_dst) {
    FileOpResult r;
    if (!is_write_allowed()) { r.error_message = "File write not allowed"; return r; }
    std::string src = expand_home(raw_src);
    std::string dst = expand_home(raw_dst);
    if (!is_path_allowed(src) || !is_path_allowed(dst)) {
        r.error_message = "Path not allowed"; return r;
    }
    try {
        fs::create_directories(fs::path(dst).parent_path());
        fs::copy(src, dst, fs::copy_options::overwrite_existing |
                           fs::copy_options::recursive);
        r.success   = true;
        r.dest_path = dst;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    return r;
}

FileOpResult FileManager::remove(const std::string& raw_path) {
    FileOpResult r;
    if (!is_write_allowed()) { r.error_message = "File write not allowed"; return r; }
    std::string path = expand_home(raw_path);
    if (!is_path_allowed(path)) { r.error_message = "Path not allowed"; return r; }
    try {
        fs::remove_all(path);
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    return r;
}

FileOpResult FileManager::mkdir(const std::string& raw_path) {
    FileOpResult r;
    if (!is_write_allowed()) { r.error_message = "File write not allowed"; return r; }
    std::string path = expand_home(raw_path);
    if (!is_path_allowed(path)) { r.error_message = "Path not allowed"; return r; }
    try {
        fs::create_directories(path);
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    return r;
}

FileOpResult FileManager::stat(const std::string& raw_path) const {
    FileOpResult r;
    std::string path = expand_home(raw_path);
    if (!is_path_allowed(path)) { r.error_message = "Path not allowed"; return r; }
    try {
        fs::directory_entry de(path);
        if (!de.exists()) { r.error_message = "Does not exist"; return r; }
        FileEntry e;
        e.path    = path;
        e.name    = fs::path(path).filename().string();
        e.is_dir  = de.is_directory();
        if (!e.is_dir) {
            try { e.size_bytes = static_cast<long long>(de.file_size()); } catch (...) {}
        }
        try { e.modified_at = iso_time(de.last_write_time()); } catch (...) {}
        try { e.permissions = permissions_string(de.status().permissions()); } catch (...) {}
        r.entries.push_back(e);
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    return r;
}

} // namespace cardinal
