// =============================================================================
// Cardinal - Agent Planner Implementation
// File: src/agent/agent_planner.cpp
// =============================================================================

#include "agent/agent_planner.h"
#include "core/llm_backend.h"
#include "core/feeling_output.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>

using json = nlohmann::json;

namespace cardinal {

    AgentPlanner::AgentPlanner(const CardinalConfig& config,
                               ILLMBackend&          backend,
                               const ToolRegistry&   registry)
        : config_(config)
        , backend_(backend)
        , registry_(registry)
    {}

    // =========================================================================
    // decompose
    // =========================================================================

    AgentPlan AgentPlanner::decompose(const std::string& goal,
                                       const std::string& context_hints) const
    {
        LOG_INFO("AgentPlanner: decomposing goal: " +
                 goal.substr(0, std::min(goal.size(), size_t(80))));

        std::string prompt = build_planning_prompt(goal, context_hints, false);

        // Use a direct generation call — we don't need feeling output for planning
        std::vector<ChatMessage> messages = {
            { "system",
              "You are a precise task planner. You decompose goals into "
              "clear, actionable steps. You always respond with valid JSON." },
            { "user", prompt }
        };

        FeelingContext feeling_ctx(config_);
        auto result = backend_.generate_response(feeling_ctx, messages, nullptr);

        if (!result.success || result.text.empty()) {
            LOG_WARN("AgentPlanner: LLM generation failed");
            AgentPlan failed;
            failed.goal  = goal;
            failed.valid = false;
            return failed;
        }

        return parse_plan(result.text, goal);
    }

    // =========================================================================
    // replan
    // =========================================================================

    AgentPlan AgentPlanner::replan(const std::string& goal,
                                    const std::string& progress_summary,
                                    const std::string& failure_reason) const
    {
        LOG_INFO("AgentPlanner: replanning after failure: " + failure_reason);

        std::string context =
            "Previous progress:\n" + progress_summary +
            "\n\nFailure reason: " + failure_reason +
            "\n\nGenerate a revised plan to complete the remaining goal.";

        return decompose(goal, context);
    }

    // =========================================================================
    // build_planning_prompt
    // =========================================================================

    std::string AgentPlanner::build_planning_prompt(
        const std::string& goal,
        const std::string& context,
        bool               is_replan) const
    {
        std::ostringstream oss;

        if (is_replan) {
            oss << "CONTEXT FROM PREVIOUS ATTEMPT:\n" << context << "\n\n";
        }

        oss << "GOAL: " << goal << "\n\n";

        // List available tools
        auto tools = registry_.get_enabled_tools();
        if (!tools.empty()) {
            oss << "AVAILABLE TOOLS:\n";
            for (const auto& t : tools) {
                oss << "- " << t.name << ": " << t.description << "\n";
            }
            oss << "\n";
        }

        if (!context.empty() && !is_replan) {
            oss << "CONTEXT HINTS:\n" << context << "\n\n";
        }

        oss << "Decompose this goal into a sequence of concrete steps.\n"
               "Each step should be specific and achievable.\n"
               "Respond ONLY with a JSON object in this exact format:\n\n"
               "{\n"
               "  \"strategy\": \"<brief description of overall approach>\",\n"
               "  \"confidence\": <float 0.0-1.0>,\n"
               "  \"steps\": [\n"
               "    {\n"
               "      \"index\": 0,\n"
               "      \"description\": \"<what to do>\",\n"
               "      \"rationale\": \"<why this step is needed>\"\n"
               "    }\n"
               "  ]\n"
               "}\n\n"
               "Output ONLY the JSON object. No other text.";

        return oss.str();
    }

    // =========================================================================
    // parse_plan
    // =========================================================================

    AgentPlan AgentPlanner::parse_plan(const std::string& llm_output,
                                        const std::string& goal) const
    {
        AgentPlan plan;
        plan.goal  = goal;
        plan.valid = false;

        // Find JSON object in output
        size_t start = llm_output.find('{');
        size_t end   = llm_output.rfind('}');

        if (start == std::string::npos || end == std::string::npos) {
            LOG_WARN("AgentPlanner: no JSON in planner output");
            return plan;
        }

        std::string json_str = llm_output.substr(start, end - start + 1);

        try {
            auto j = json::parse(json_str);

            plan.strategy   = j.value("strategy", "");
            plan.confidence = j.value("confidence", 0.5f);

            if (!j.contains("steps") || !j["steps"].is_array()) {
                LOG_WARN("AgentPlanner: plan has no steps array");
                return plan;
            }

            for (const auto& sj : j["steps"]) {
                AgentStep step;
                step.index       = sj.value("index", static_cast<int>(plan.steps.size()));
                step.description = sj.value("description", "");
                step.rationale   = sj.value("rationale", "");
                step.status      = AgentStepStatus::PENDING;

                if (step.description.empty()) continue;
                plan.steps.push_back(std::move(step));
            }

            if (plan.steps.empty()) {
                LOG_WARN("AgentPlanner: plan has zero valid steps");
                return plan;
            }

            plan.valid = true;
            LOG_INFO("AgentPlanner: plan with " +
                     std::to_string(plan.steps.size()) +
                     " steps, confidence=" +
                     std::to_string(static_cast<int>(plan.confidence * 100)) + "%");

        } catch (const json::exception& e) {
            LOG_WARN("AgentPlanner: JSON parse failed: " + std::string(e.what()));
        }

        return plan;
    }

} // namespace cardinal
