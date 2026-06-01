#pragma once
// =============================================================================
// Cardinal - Tool: file_ops
// File: src/tools/builtin/computer/tool_file_ops.h
//
// List, move, copy, delete, mkdir, stat files.
// Actions: list|move|copy|delete|mkdir|stat|exists
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class FileManager;
    ToolDefinition make_file_ops_tool_def(const CardinalConfig& config);
    ToolResult     execute_file_ops(const ToolCall& call, FileManager& fm);
} // namespace cardinal
