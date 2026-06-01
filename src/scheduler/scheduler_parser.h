#pragma once
// =============================================================================
// Cardinal - Scheduler Parser (v1.5.0)
// File: src/scheduler/scheduler_parser.h
//
// Converts natural language task descriptions into ScheduledTask structs.
// Uses InferencePipeline::run() for LLM calls.
// extract_from_json() is public static — used by SchedulerEngine directly.
// =============================================================================

#include "scheduler/scheduler_types.h"
#include "utils/config_loader.h"

#include <string>

namespace cardinal {

    class InferencePipeline;

    class SchedulerParser {
    public:
        // pipeline may be nullptr — parse() will fail gracefully if so
        explicit SchedulerParser(InferencePipeline*    pipeline,
                                 const CardinalConfig& config,
                                 float                 min_confidence = 0.70f);

        ~SchedulerParser() = default;

        SchedulerParser(const SchedulerParser&)            = delete;
        SchedulerParser& operator=(const SchedulerParser&) = delete;

        // Parse natural language into a ScheduledTask
        TaskParseResult parse(const std::string& nl_description,
                              const std::string& session_id = "");

        // Parse from a JSON string directly (HTTP API path)
        TaskParseResult parse_json(const std::string& json_str,
                                   const std::string& session_id = "");

        // Public static — used by SchedulerEngine::create_task_from_nl()
        static TaskParseResult extract_from_json(const std::string& model_output,
                                                  const std::string& session_id,
                                                  float              min_confidence);

        static std::string build_system_prompt();
        static std::string build_user_message(const std::string& nl_description);

    private:
        static void finalise_task(ScheduledTask& task, const std::string& session_id);

        InferencePipeline*    pipeline_;
        const CardinalConfig& config_;
        float                 min_confidence_;
    };

} // namespace cardinal
