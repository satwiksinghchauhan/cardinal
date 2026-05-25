#pragma once
// =============================================================================
// Cardinal - Tool Shared Types
// File: src/tools/tool_result.h
//
// Shared types used across ToolRegistry, ToolExecutor, InferencePipeline,
// AgentExecutor, and the explainability system.
//
// No dependencies on other Cardinal headers — safe to include anywhere.
// =============================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <chrono>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ToolParameterType
    // JSON Schema primitive types for tool parameter definitions.
    // -------------------------------------------------------------------------
    enum class ToolParameterType {
        STRING,
        NUMBER,
        BOOLEAN,
        ARRAY,
        OBJECT
    };

    inline std::string tool_param_type_to_string(ToolParameterType t) {
        switch (t) {
            case ToolParameterType::STRING:  return "string";
            case ToolParameterType::NUMBER:  return "number";
            case ToolParameterType::BOOLEAN: return "boolean";
            case ToolParameterType::ARRAY:   return "array";
            case ToolParameterType::OBJECT:  return "object";
            default:                         return "string";
        }
    }

    // -------------------------------------------------------------------------
    // ToolParameter
    // Single parameter in a tool's JSON Schema definition.
    // -------------------------------------------------------------------------
    struct ToolParameter {
        std::string       name;
        ToolParameterType type;
        std::string       description;
        bool              required    = true;
        std::string       default_val = "";     // serialized default value

        // For enum parameters — allowed values
        std::vector<std::string> enum_values;
    };

    // -------------------------------------------------------------------------
    // ToolDefinition
    // Complete definition of a tool — name, description, parameters.
    // This is what gets injected into the system prompt so the model
    // knows what tools are available and how to call them.
    //
    // Built-in tools are registered by ToolRegistry at startup.
    // User-defined tools can be added per-request via the API.
    // -------------------------------------------------------------------------
    struct ToolDefinition {
        std::string                  name;           // Unique identifier
        std::string                  description;    // Human + model readable
        std::vector<ToolParameter>   parameters;
        bool                         enabled         = true;
        bool                         confirmation_required = false;

        // Returns true if a parameter with this name exists and is required
        bool has_required_param(const std::string& param_name) const {
            for (const auto& p : parameters) {
                if (p.name == param_name && p.required) return true;
            }
            return false;
        }
    };

    // -------------------------------------------------------------------------
    // ToolCall
    // A tool invocation parsed from the model's output.
    // The model outputs a JSON block; we parse it into this struct.
    // -------------------------------------------------------------------------
    struct ToolCall {
        std::string                                  tool_name;
        std::unordered_map<std::string, std::string> arguments; // key → serialized value
        std::string                                  raw_json;  // original model output
        int                                          call_index = 0; // position in response
    };

    // -------------------------------------------------------------------------
    // ToolStatus
    // -------------------------------------------------------------------------
    enum class ToolStatus {
        SUCCESS,
        FAILURE,
        TIMEOUT,
        CONFIRMATION_REQUIRED,  // paused, waiting for human approval
        DISABLED,               // tool disabled in config
        NOT_FOUND,              // tool name not in registry
        INVALID_ARGS,           // argument validation failed
        SANDBOX_ERROR           // Docker/subprocess error
    };

    inline std::string tool_status_to_string(ToolStatus s) {
        switch (s) {
            case ToolStatus::SUCCESS:               return "success";
            case ToolStatus::FAILURE:               return "failure";
            case ToolStatus::TIMEOUT:               return "timeout";
            case ToolStatus::CONFIRMATION_REQUIRED: return "confirmation_required";
            case ToolStatus::DISABLED:              return "disabled";
            case ToolStatus::NOT_FOUND:             return "not_found";
            case ToolStatus::INVALID_ARGS:          return "invalid_args";
            case ToolStatus::SANDBOX_ERROR:         return "sandbox_error";
            default:                                return "unknown";
        }
    }

    // -------------------------------------------------------------------------
    // ToolResult
    // Result of a single tool execution.
    // Stored in ReasoningTrace and injected back into model context.
    // -------------------------------------------------------------------------
    struct ToolResult {
        // Identity
        std::string  tool_name;
        ToolCall     call;              // the call that produced this result

        // Outcome
        ToolStatus   status;
        std::string  output;            // tool output text (injected to context)
        std::string  error_message;     // set on failure

        // Metrics
        int          duration_ms   = 0;
        std::string  timestamp;

        // Convenience
        bool ok()      const { return status == ToolStatus::SUCCESS; }
        bool failed()  const { return status == ToolStatus::FAILURE ||
                                      status == ToolStatus::TIMEOUT ||
                                      status == ToolStatus::SANDBOX_ERROR; }
        bool pending() const { return status == ToolStatus::CONFIRMATION_REQUIRED; }
    };

    // -------------------------------------------------------------------------
    // ToolCallDetectionResult
    // Returned by ToolExecutor::detect_tool_calls() when scanning model output.
    // -------------------------------------------------------------------------
    struct ToolCallDetectionResult {
        bool                   has_tool_calls = false;
        std::vector<ToolCall>  tool_calls;
        std::string            text_before;   // text before first tool call
        std::string            text_after;    // text after last tool call
    };

    // -------------------------------------------------------------------------
    // ConfirmationRequest
    // Sent to the API layer when a tool requires human confirmation.
    // The agent pauses and waits for approve/deny via HTTP.
    // -------------------------------------------------------------------------
    struct ConfirmationRequest {
        std::string  request_id;     // unique ID for this confirmation
        std::string  session_id;
        std::string  tool_name;
        std::string  arguments_json;
        std::string  reason;         // why confirmation is needed
        std::string  timestamp;
    };

} // namespace cardinal
