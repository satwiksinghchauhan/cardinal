#pragma once
// =============================================================================
// Cardinal - voice_control Tool (v1.6.0)
// File: src/tools/builtin/voice/tool_voice_control.h
//
// Allows the LLM to control the voice subsystem at runtime:
//   set_mode | set_voice | set_volume | stop_speaking
// =============================================================================

#include "tools/tool_result.h"
#include <string>

namespace cardinal {
    class VoiceLoop;

    ToolDefinition make_voice_control_tool_definition();

    ToolResult execute_voice_control(const ToolCall& call, VoiceLoop* voice_loop);

} // namespace cardinal
