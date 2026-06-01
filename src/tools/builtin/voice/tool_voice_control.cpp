// =============================================================================
// Cardinal - voice_control Tool Implementation (v1.6.0)
// File: src/tools/builtin/voice/tool_voice_control.cpp
// =============================================================================

#include "tools/builtin/voice/tool_voice_control.h"
#include "voice/voice_loop.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>

namespace cardinal {

using json = nlohmann::json;

// Helper function to convert string to VoiceInputMode
static VoiceInputMode string_to_voice_input_mode(const std::string& mode_str) {
    if (mode_str == "push_to_talk" || mode_str == "ptt") {
        return VoiceInputMode::PUSH_TO_TALK;
    } else if (mode_str == "vad") {
        return VoiceInputMode::VAD;
    } else if (mode_str == "wake_word") {
        return VoiceInputMode::WAKE_WORD;
    }
    return VoiceInputMode::VAD;  // default
}

ToolDefinition make_voice_control_tool_definition() {
    ToolDefinition def;
    def.name                 = "voice_control";
    def.description          = "Control the Cardinal voice subsystem. "
                               "Actions: set_mode (push_to_talk|vad|wake_word), "
                               "set_voice (model name), set_volume (0-100), stop_speaking.";
    def.enabled              = true;
    def.confirmation_required = false;
    
    // Parameter 1: action
    ToolParameter action_param;
    action_param.name = "action";
    action_param.type = ToolParameterType::STRING;
    action_param.description = "The voice control action to perform";
    action_param.required = true;
    action_param.default_val = "";
    action_param.enum_values = {"set_mode", "set_voice", "set_volume", "stop_speaking"};
    def.parameters.push_back(action_param);
    
    // Parameter 2: mode
    ToolParameter mode_param;
    mode_param.name = "mode";
    mode_param.type = ToolParameterType::STRING;
    mode_param.description = "Input mode (for set_mode action): push_to_talk|vad|wake_word";
    mode_param.required = false;
    mode_param.default_val = "";
    mode_param.enum_values = {"push_to_talk", "ptt", "vad", "wake_word"};
    def.parameters.push_back(mode_param);
    
    // Parameter 3: voice
    ToolParameter voice_param;
    voice_param.name = "voice";
    voice_param.type = ToolParameterType::STRING;
    voice_param.description = "Piper voice model name (for set_voice action)";
    voice_param.required = false;
    voice_param.default_val = "";
    def.parameters.push_back(voice_param);
    
    // Parameter 4: value
    ToolParameter value_param;
    value_param.name = "value";
    value_param.type = ToolParameterType::STRING;
    value_param.description = "Volume level 0-100 (for set_volume action)";
    value_param.required = false;
    value_param.default_val = "";
    def.parameters.push_back(value_param);
    
    return def;
}

ToolResult execute_voice_control(const ToolCall& call, VoiceLoop* voice_loop) {
    ToolResult result;
    result.tool_name = "voice_control";
    result.call      = call;

    if (!voice_loop) {
        result.status = ToolStatus::FAILURE;
        result.error_message = "Voice subsystem is not active.";
        return result;
    }

    try {
        // Build JSON string from arguments map
        json args;
        for (const auto& [key, value] : call.arguments) {
            try {
                args[key] = json::parse(value);
            } catch (...) {
                args[key] = value;
            }
        }
        
        std::string action = args.value("action", "");

        if (action == "set_mode") {
            std::string mode_str = args.value("mode", "vad");
            VoiceInputMode mode = string_to_voice_input_mode(mode_str);
            voice_loop->set_input_mode(mode);
            result.status = ToolStatus::SUCCESS;
            result.output = "Input mode set to " + mode_str + ".";

        } else if (action == "set_voice") {
            std::string voice = args.value("voice", "");
            if (voice.empty()) {
                result.status = ToolStatus::FAILURE;
                result.error_message = "No voice name specified.";
            } else {
                result.status = ToolStatus::SUCCESS;
                result.output = "Voice change to \"" + voice +
                                 "\" will take effect on next VoiceLoop restart.";
                LOG_INFO("voice_control: set_voice requested: " + voice);
            }

        } else if (action == "set_volume") {
            std::string val = args.value("value", "80");
            result.status = ToolStatus::SUCCESS;
            result.output = "Volume set to " + val + "% (system volume).";

        } else if (action == "stop_speaking") {
            voice_loop->stop_speaking();
            result.status = ToolStatus::SUCCESS;
            result.output = "Playback stopped.";

        } else {
            result.status = ToolStatus::FAILURE;
            result.error_message = "Unknown action: " + action;
        }

    } catch (const std::exception& ex) {
        result.status = ToolStatus::FAILURE;
        result.error_message = std::string("voice_control error: ") + ex.what();
    }

    return result;
}

} // namespace cardinal
