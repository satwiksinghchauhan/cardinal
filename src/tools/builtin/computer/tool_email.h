#pragma once
// =============================================================================
// Cardinal - Tool: email
// File: src/tools/builtin/computer/tool_email.h
// Actions: read|send
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {
    class EmailController;
    ToolDefinition make_email_tool_def(const CardinalConfig& config);
    ToolResult     execute_email(const ToolCall& call, EmailController& email);
} // namespace cardinal
