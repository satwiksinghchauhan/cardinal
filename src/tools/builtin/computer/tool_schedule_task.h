#pragma once
// =============================================================================
// Cardinal - Tool: schedule_task
// File: src/tools/builtin/computer/tool_schedule_task.h
//
// Create, list, enable, disable, delete scheduled tasks from within a chat.
// Actions: create|list|delete|enable|disable|run_now
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class SchedulerEngine;
    ToolDefinition make_schedule_task_tool_def(const CardinalConfig& config);
    ToolResult     execute_schedule_task(const ToolCall&   call,
                                         SchedulerEngine&  scheduler,
                                         const std::string& session_id = "");
} // namespace cardinal
