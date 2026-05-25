#pragma once
// =============================================================================
// Cardinal - File Manager
// File: src/computer/file_manager.h
//
// Extends the existing file_read / file_write tools with:
//   list, move, copy, delete, mkdir, stat, exists
// All operations validate against config allowed_paths.
// =============================================================================

#include "computer/computer_types.h"
#include <filesystem>
#include "utils/config_loader.h"

#include <string>
#include <vector>

namespace cardinal {

    class FileManager {
    public:
        explicit FileManager(const CardinalConfig& config);
        ~FileManager() = default;

        FileManager(const FileManager&)            = delete;
        FileManager& operator=(const FileManager&) = delete;

        FileOpResult list(const std::string& path, bool recursive = false) const;
        FileOpResult move(const std::string& src, const std::string& dst);
        FileOpResult copy(const std::string& src, const std::string& dst);
        FileOpResult remove(const std::string& path);
        FileOpResult mkdir(const std::string& path);
        FileOpResult stat(const std::string& path) const;
        bool         exists(const std::string& path) const;

        // Safety
        bool is_path_allowed(const std::string& path) const;
        bool is_write_allowed() const;

    private:
        static std::string expand_home(const std::string& path);
        static std::string permissions_string(const std::filesystem::perms& p);
        static std::string iso_time(const std::filesystem::file_time_type& ft);

        const CardinalConfig& config_;
    };

} // namespace cardinal
