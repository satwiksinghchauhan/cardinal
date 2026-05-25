// =============================================================================
// Cardinal - Agent Executor Implementation
// File: src/agent/agent_executor.cpp
// =============================================================================

#include "agent/agent_executor.h"
#include "core/llm_backend.h"
#include "core/feeling_output.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <sstream>
#include <chrono>
#include <algorithm>

namespace cardinal {

    AgentExecutor::AgentExecutor(const CardinalConfig& config,
                                 ILLMBackend&          backend,
                                 const ToolRegistry&   registry,
                                 const ToolExecutor&   tool_executor)
        : config_(config)
        , backend_(backend)
        , registry_(registry)
        , tool_executor_(tool_executor)
        , planner_(config, backend, registry)
    {}

    // =========================================================================
    // run — main agentic loop
    // =========================================================================

    AgentResult AgentExecutor::run(const AgentGoal&      goal,
                                    TraceBuilder&         trace_builder,
                                    AgentProgressCallback progress_cb)
    {
        AgentResult result;
        result.goal_achieved = false;

        auto task_start = std::chrono::steady_clock::now();

        LOG_INFO("AgentExecutor: starting task: " +
                 goal.goal.substr(0, std::min(goal.goal.size(), size_t(80))));

        // Determine max iterations
        int max_iter = (goal.max_iterations > 0)
            ? std::min(goal.max_iterations, config_.agent.max_iterations_hard_cap)
            : config_.agent.max_iterations;

        // ------------------------------------------------------------------
        // 1. Open working memory (persistent SQLite)
        // ------------------------------------------------------------------
        std::string task_id = JsonParser::generate_id();
        AgentContext ctx(config_, goal.session_id, task_id);
        ctx.open();

        // Store goal in working memory
        ctx.store("__goal__", goal.goal, "note", "0");
        for (size_t i = 0; i < goal.context_hints.size(); ++i) {
            ctx.store("__hint_" + std::to_string(i) + "__",
                      goal.context_hints[i], "note", "0");
        }

        // ------------------------------------------------------------------
        // 2. Plan
        // ------------------------------------------------------------------
        notify(progress_cb, -1, "Planning", "started", goal.goal);

        AgentPlan plan;
        if (config_.agent.plan_before_execute) {
            std::string hints;
            for (const auto& h : goal.context_hints)
                hints += h + "\n";
            plan = planner_.decompose(goal.goal, hints);

            if (!plan.valid) {
                LOG_WARN("AgentExecutor: planning failed");
                result.failure_reason = "Failed to generate a valid plan";
                result.final_response =
                    "I was unable to create a plan to achieve this goal. "
                    "Please try rephrasing or breaking it into smaller tasks.";
                ctx.destroy();
                return result;
            }

            ctx.save_plan(plan);
            result.plan       = plan;
            result.steps_total = static_cast<int>(plan.steps.size());
        } else {
            // No planning — single-step execution
            AgentStep single;
            single.index       = 0;
            single.description = goal.goal;
            single.rationale   = "Direct execution without planning";
            plan.steps.push_back(single);
            plan.goal  = goal.goal;
            plan.valid = true;
            result.steps_total = 1;
        }

        trace_builder.record_agent_step(AgentStepRecord{
            -1, "Planning", "Decomposed goal into " +
            std::to_string(plan.steps.size()) + " steps",
            false, "", "", "", true, "", 0
        });

        notify(progress_cb, -1, "Planning", "completed",
               std::to_string(plan.steps.size()) + " steps planned");

        // ------------------------------------------------------------------
        // 3. Execute loop
        // ------------------------------------------------------------------
        int iteration = 0;

        for (auto& step : plan.steps) {
            if (iteration >= max_iter) {
                LOG_WARN("AgentExecutor: max iterations reached (" +
                         std::to_string(max_iter) + ")");
                if (config_.agent.summarize_on_cap) {
                    result.failure_reason = "Maximum iterations reached";
                }
                break;
            }

            notify(progress_cb, step.index, step.description, "started");

            step = execute_step(step, ctx, plan, trace_builder);

            ++iteration;
            result.steps_completed += (step.status == AgentStepStatus::COMPLETED) ? 1 : 0;

            notify(progress_cb, step.index, step.description,
                   step_status_to_string(step.status), step.result_summary);

            // Check goal achievement after each step
            if (check_goal_achieved(goal, ctx, plan)) {
                result.goal_achieved = true;
                LOG_INFO("AgentExecutor: goal achieved after step " +
                         std::to_string(step.index));
                break;
            }
        }

        // If all steps complete, goal is achieved
        if (!result.goal_achieved &&
            plan.completed_count() == static_cast<int>(plan.steps.size())) {
            result.goal_achieved = true;
        }

        // ------------------------------------------------------------------
        // 4. Synthesize final response
        // ------------------------------------------------------------------
        notify(progress_cb, -1, "Synthesizing response", "started");

        result.final_response = synthesize_response(
            goal, ctx, plan, result.goal_achieved);

        // ------------------------------------------------------------------
        // 5. Populate result stats
        // ------------------------------------------------------------------
        result.iterations_used = iteration;
        result.memory_snapshot = ctx.get_all();

        result.total_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - task_start).count());

        // Count tool calls across all steps
        for (const auto& s : plan.steps)
            if (s.used_tool) ++result.tools_called;

        // Update trace
        trace_builder.record_agent_complete(result.goal_achieved, iteration);

        LOG_INFO("AgentExecutor: task complete — goal_achieved=" +
                 std::string(result.goal_achieved ? "yes" : "no") +
                 " iterations=" + std::to_string(iteration) +
                 " ms=" + std::to_string(result.total_ms));

        ctx.destroy(); // Clean up working memory DB
        return result;
    }

    // =========================================================================
    // execute_step
    // =========================================================================

    AgentStep AgentExecutor::execute_step(AgentStep&       step,
                                           AgentContext&    ctx,
                                           const AgentPlan& plan,
                                           TraceBuilder&    trace_builder)
    {
        auto step_start = std::chrono::steady_clock::now();
        step.status = AgentStepStatus::IN_PROGRESS;

        LOG_INFO("AgentExecutor: step " + std::to_string(step.index) +
                 ": " + step.description);

        // Think: prompt LLM for this step
        std::string llm_output = think_for_step(step, ctx, plan);

        // Detect tool calls
        auto detected = tool_executor_.detect_tool_calls(llm_output);

        AgentStepRecord step_record;
        step_record.step_index   = step.index;
        step_record.description  = step.description;
        step_record.action_taken = llm_output.substr(
            0, std::min(llm_output.size(), size_t(200)));

        if (detected.has_tool_calls) {
            // Execute first tool call (sequential for now)
            const auto& call = detected.tool_calls[0];
            step_record.tool_called = true;
            step_record.tool_name   = call.tool_name;
            step_record.tool_input_json = call.raw_json;

            auto tool_result = tool_executor_.execute(call);
            trace_builder.record_tool_call(tool_result);

            if (tool_result.ok()) {
                // Store result in working memory
                ctx.store("step_" + std::to_string(step.index) + "_result",
                          tool_result.output,
                          "result",
                          std::to_string(step.index));

                step.used_tool      = true;
                step.tool_used      = call.tool_name;
                step.result_summary = tool_result.output.substr(
                    0, std::min(tool_result.output.size(), size_t(300)));
                step.status         = AgentStepStatus::COMPLETED;

                step_record.tool_output_preview = step.result_summary;
                step_record.succeeded           = true;

            } else {
                // Tool failed — self-correct if enabled
                if (config_.agent.self_correction_enabled && step.retries <
                    config_.agent.self_correction_max_attempts) {

                    LOG_WARN("AgentExecutor: tool failed, self-correcting: " +
                             tool_result.error_message);
                    step = self_correct(step, ctx,
                                        tool_result.error_message,
                                        trace_builder);
                } else {
                    step.status         = AgentStepStatus::FAILED;
                    step.result_summary = "Tool failed: " + tool_result.error_message;

                    ctx.store("step_" + std::to_string(step.index) + "_error",
                              tool_result.error_message,
                              "error",
                              std::to_string(step.index));

                    step_record.succeeded    = false;
                    step_record.failure_reason = tool_result.error_message;
                }
            }
        } else {
            // No tool call — LLM gave a direct response
            // Treat the response as the step result
            std::string response = detected.text_before.empty()
                ? llm_output : detected.text_before;

            ctx.store("step_" + std::to_string(step.index) + "_result",
                      response,
                      "observation",
                      std::to_string(step.index));

            step.result_summary = response.substr(
                0, std::min(response.size(), size_t(300)));
            step.status         = AgentStepStatus::COMPLETED;
            step_record.succeeded = true;
        }

        step.duration_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - step_start).count());
        step_record.duration_ms = step.duration_ms;

        ctx.record_step(step);
        trace_builder.record_agent_step(step_record);

        return step;
    }

    // =========================================================================
    // think_for_step
    // =========================================================================

    std::string AgentExecutor::think_for_step(const AgentStep&   step,
                                               const AgentContext& ctx,
                                               const AgentPlan&   plan) const
    {
        std::string prompt = build_step_prompt(step, ctx, plan);

        std::vector<ChatMessage> messages = {
            { "system",
              "You are Cardinal, an autonomous AI agent. "
              "You execute tasks step by step using available tools. "
              "When you need to use a tool, output a tool call in "
              "<tool_call>{...}</tool_call> format. "
              "When you have gathered enough information, respond directly.\n\n" +
              registry_.format_all_for_prompt() },
            { "user", prompt }
        };

        FeelingContext feeling_ctx(config_);
        auto result = backend_.generate_response(feeling_ctx, messages, nullptr);

        if (!result.success) {
            LOG_WARN("AgentExecutor: LLM generation failed for step " +
                     std::to_string(step.index));
            return "";
        }

        return result.text;
    }

    // =========================================================================
    // build_step_prompt
    // =========================================================================

    std::string AgentExecutor::build_step_prompt(
        const AgentStep&   step,
        const AgentContext& ctx,
        const AgentPlan&   plan,
        bool               is_correction,
        const std::string& correction_hint) const
    {
        std::ostringstream oss;

        oss << "## Goal\n" << plan.goal << "\n\n";

        // Show overall plan context
        oss << "## Plan Overview\n";
        for (const auto& s : plan.steps) {
            std::string marker =
                s.index == step.index ? "→ [CURRENT] " :
                s.status == AgentStepStatus::COMPLETED ? "✓ " :
                s.status == AgentStepStatus::FAILED    ? "✗ " : "  ";
            oss << marker << (s.index + 1) << ". " << s.description << "\n";
        }
        oss << "\n";

        // Working memory context
        std::string mem_summary = ctx.build_context_summary();
        if (!mem_summary.empty()) {
            oss << mem_summary << "\n";
        }

        // Current step
        oss << "## Current Step\n";
        oss << "Step " << (step.index + 1) << ": " << step.description << "\n";
        if (!step.rationale.empty())
            oss << "Why: " << step.rationale << "\n";
        oss << "\n";

        if (is_correction) {
            oss << "## Correction Needed\n"
                << "Your previous attempt failed: " << correction_hint << "\n"
                << "Please try a different approach.\n\n";
        }

        oss << "Execute this step. Use a tool if needed, or respond directly "
               "if you already have the information.";

        return oss.str();
    }

    // =========================================================================
    // self_correct
    // =========================================================================

    AgentStep AgentExecutor::self_correct(AgentStep&         step,
                                           AgentContext&       ctx,
                                           const std::string& failure_reason,
                                           TraceBuilder&       trace_builder)
    {
        ++step.retries;
        LOG_INFO("AgentExecutor: self-correction attempt " +
                 std::to_string(step.retries) + " for step " +
                 std::to_string(step.index));

        // Re-think with correction context
        std::string corrected_output = [&]() {
            std::string prompt = build_step_prompt(
                step, ctx, AgentPlan{}, true, failure_reason);

            std::vector<ChatMessage> messages = {
                { "system",
                  "You are Cardinal, an autonomous AI agent performing self-correction. "
                  "A previous attempt at this step failed. Try a different approach.\n\n" +
                  registry_.format_all_for_prompt() },
                { "user", prompt }
            };

            FeelingContext feeling_ctx(config_);
            auto result = backend_.generate_response(feeling_ctx, messages, nullptr);
            return result.success ? result.text : std::string("");
        }();

        if (corrected_output.empty()) {
            step.status = AgentStepStatus::FAILED;
            step.result_summary = "Self-correction failed: no output";
            return step;
        }

        // Try executing the corrected approach
        auto detected = tool_executor_.detect_tool_calls(corrected_output);

        if (detected.has_tool_calls) {
            auto tool_result = tool_executor_.execute(detected.tool_calls[0]);
            trace_builder.record_tool_call(tool_result);

            if (tool_result.ok()) {
                ctx.store("step_" + std::to_string(step.index) + "_corrected",
                          tool_result.output, "result",
                          std::to_string(step.index));

                step.used_tool      = true;
                step.tool_used      = detected.tool_calls[0].tool_name;
                step.result_summary = tool_result.output.substr(
                    0, std::min(tool_result.output.size(), size_t(300)));
                step.status         = AgentStepStatus::COMPLETED;
            } else {
                step.status = AgentStepStatus::FAILED;
                step.result_summary = "Correction also failed: " +
                                       tool_result.error_message;
            }
        } else {
            ctx.store("step_" + std::to_string(step.index) + "_corrected",
                      corrected_output, "observation",
                      std::to_string(step.index));
            step.result_summary = corrected_output.substr(
                0, std::min(corrected_output.size(), size_t(300)));
            step.status = AgentStepStatus::COMPLETED;
        }

        return step;
    }

    // =========================================================================
    // check_goal_achieved
    // =========================================================================

    bool AgentExecutor::check_goal_achieved(const AgentGoal&    goal,
                                             const AgentContext& ctx,
                                             const AgentPlan&    plan) const
    {
        // Check if all steps completed
        auto steps = ctx.get_all_steps();
        int completed = 0;
        for (const auto& s : steps)
            if (s.status == AgentStepStatus::COMPLETED) ++completed;

        if (completed >= static_cast<int>(plan.steps.size()) &&
            !plan.steps.empty())
            return true;

        // Ask the LLM if goal is achieved based on working memory
        // (only if we have some results)
        if (steps.empty()) return false;

        std::ostringstream oss;
        oss << "GOAL: " << goal.goal << "\n\n";
        oss << ctx.build_context_summary();
        oss << "\nHas the goal been fully achieved? "
               "Respond with only 'YES' or 'NO'.";

        std::vector<ChatMessage> messages = {
            { "system", "You are a goal completion evaluator. "
                        "Answer only YES or NO." },
            { "user", oss.str() }
        };

        FeelingContext feeling_ctx(config_);
        auto result = backend_.generate_response(feeling_ctx, messages, nullptr);

        if (!result.success) return false;

        std::string answer = result.text;
        // Trim and uppercase
        answer.erase(0, answer.find_first_not_of(" \t\n\r"));
        answer.erase(answer.find_last_not_of(" \t\n\r") + 1);
        for (auto& c : answer) c = static_cast<char>(std::toupper(c));

        return answer.substr(0, 3) == "YES";
    }

    // =========================================================================
    // synthesize_response
    // =========================================================================

    std::string AgentExecutor::synthesize_response(
        const AgentGoal&    goal,
        const AgentContext& ctx,
        const AgentPlan&    plan,
        bool                goal_achieved) const
    {
        std::ostringstream oss;
        oss << "GOAL: " << goal.goal << "\n\n";
        oss << ctx.build_context_summary() << "\n";
        oss << "Steps completed: " << ctx.step_count() << "/" <<
               plan.steps.size() << "\n";
        oss << "Goal achieved: " << (goal_achieved ? "YES" : "NO") << "\n\n";
        oss << "Based on everything above, provide a comprehensive response "
               "that directly addresses the original goal. "
               "Include key findings, results, and conclusions.";

        std::vector<ChatMessage> messages = {
            { "system",
              "You are Cardinal, a neurosymbolic AI agent. "
              "Synthesize the results of your task execution into a "
              "clear, comprehensive response." },
            { "user", oss.str() }
        };

        FeelingContext feeling_ctx(config_);
        auto result = backend_.generate_response(feeling_ctx, messages, nullptr);

        if (!result.success || result.text.empty()) {
            return goal_achieved
                ? "Task completed successfully."
                : "Task could not be completed. Please try again.";
        }

        return result.text;
    }

    // =========================================================================
    // notify
    // =========================================================================

    void AgentExecutor::notify(AgentProgressCallback& cb,
                                int step_index,
                                const std::string& description,
                                const std::string& status,
                                const std::string& detail) const
    {
        if (!cb) return;
        AgentProgressUpdate update;
        update.step_index       = step_index;
        update.step_description = description;
        update.status           = status;
        update.detail           = detail;
        cb(update);
    }

} // namespace cardinal
