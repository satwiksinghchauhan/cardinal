#pragma once
// =============================================================================
// Cardinal - Tool: type_text
// File: src/tools/builtin/computer/tool_type_text.h
//
// Type literal text or send a key combination (ctrl+c, Return, F5, etc.)
// =============================================================================

#include "tools/tool_result.h"
#include "utils/config_loader.h"

namespace cardinal {

    class InputController;

    ToolDefinition make_type_text_tool_def(const CardinalConfig& config);
    ToolResult     execute_type_text(const ToolCall& call, InputController& input);

} // namespace cardinal
