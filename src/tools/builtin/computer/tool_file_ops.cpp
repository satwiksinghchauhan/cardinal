// =============================================================================
// Cardinal - Tool: file_ops Implementation
// =============================================================================

#include "tools/builtin/computer/tool_file_ops.h"
#include "computer/file_manager.h"

#include <sstream>
#include <chrono>

namespace cardinal {

ToolDefinition make_file_ops_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "file_ops";
    def.description =
        "File system operations: list directory contents, move, copy, delete files, "
        "create directories, get file stats, check existence.\n\n"
        "Actions: list|move|copy|delete|mkdir|stat|exists";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "action", ToolParameterType::STRING,
        "Operation: list|move|copy|delete|mkdir|stat|exists",
        true, ""
    });
    def.parameters.push_back({
        "path", ToolParameterType::STRING,
        "Source path (required for all actions)",
        true, ""
    });
    def.parameters.push_back({
        "dest", ToolParameterType::STRING,
        "Destination path (required for move and copy actions)",
        false, ""
    });
    def.parameters.push_back({
        "recursive", ToolParameterType::BOOLEAN,
        "For list action: include subdirectories recursively. Default: false.",
        false, "false"
    });
    return def;
}

ToolResult execute_file_ops(const ToolCall& call, FileManager& fm) {
    ToolResult result;
    result.tool_name = "file_ops";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        std::string action    = get("action");
        std::string path      = get("path");
        std::string dest      = get("dest");
        bool        recursive = get("recursive", "false") == "true";

        if (action.empty() || path.empty()) {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Required: action and path";
            return result;
        }

        if (action == "list") {
            auto r = fm.list(path, recursive);
            if (!r.success) {
                result.status = ToolStatus::FAILURE;
                result.output = "list failed: " + r.error_message;
            } else {
                result.status = ToolStatus::SUCCESS;
                std::ostringstream oss;
                oss << "Contents of " << path << " (" << r.entries.size() << " items):\n";
                for (const auto& e : r.entries) {
                    oss << (e.is_dir ? "d" : "-")
                        << " " << e.permissions
                        << " " << e.size_bytes
                        << " " << e.modified_at
                        << " " << e.name << "\n";
                }
                result.output = oss.str();
            }
        } else if (action == "move") {
            auto r = fm.move(path, dest);
            result.status = r.success ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = r.success ? "Moved: " + path + " → " + dest
                                      : "move failed: " + r.error_message;
        } else if (action == "copy") {
            auto r = fm.copy(path, dest);
            result.status = r.success ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = r.success ? "Copied: " + path + " → " + dest
                                      : "copy failed: " + r.error_message;
        } else if (action == "delete") {
            auto r = fm.remove(path);
            result.status = r.success ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = r.success ? "Deleted: " + path
                                      : "delete failed: " + r.error_message;
        } else if (action == "mkdir") {
            auto r = fm.mkdir(path);
            result.status = r.success ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = r.success ? "Created directory: " + path
                                      : "mkdir failed: " + r.error_message;
        } else if (action == "stat") {
            auto r = fm.stat(path);
            if (!r.success || r.entries.empty()) {
                result.status = ToolStatus::FAILURE;
                result.output = "stat failed: " + r.error_message;
            } else {
                const auto& e = r.entries[0];
                result.status = ToolStatus::SUCCESS;
                std::ostringstream oss;
                oss << "Path: "        << e.path        << "\n"
                    << "Type: "        << (e.is_dir ? "directory" : "file") << "\n"
                    << "Size: "        << e.size_bytes  << " bytes\n"
                    << "Modified: "    << e.modified_at << "\n"
                    << "Permissions: " << e.permissions;
                result.output = oss.str();
            }
        } else if (action == "exists") {
            bool ex = fm.exists(path);
            result.status = ToolStatus::SUCCESS;
            result.output = path + (ex ? " exists" : " does not exist");
        } else {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Unknown action: " + action;
        }
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "file_ops failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
