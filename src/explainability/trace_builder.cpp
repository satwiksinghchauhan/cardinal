// =============================================================================
// Cardinal - Trace Builder Implementation
// File: src/explainability/trace_builder.cpp
// =============================================================================

#include "explainability/trace_builder.h"
#include "memory/episodic_retriever.h"
#include "verifier/consistency_check.h"
#include "utils/json_parser.h"
#include "utils/logger.h"

#include <algorithm>

namespace cardinal {

    TraceBuilder::TraceBuilder(const std::string& session_id,
                               const std::string& backend_type,
                               const std::string& model_name)
    {
        inference_start_ = std::chrono::steady_clock::now();

        trace_.inference_id  = JsonParser::generate_id();
        trace_.session_id    = session_id;
        trace_.timestamp     = JsonParser::current_timestamp();
        trace_.backend_type  = backend_type;
        trace_.model_name    = model_name;
    }

    // =========================================================================
    // record_query
    // =========================================================================

    void TraceBuilder::record_query(const std::string& query,
                                     bool agent_mode,
                                     const std::string& goal)
    {
        trace_.query      = query;
        trace_.agent_mode = agent_mode;
        trace_.agent_goal = goal;
    }

    // =========================================================================
    // record_active_rules
    // =========================================================================

    void TraceBuilder::record_active_rules(const std::vector<Rule>& rules) {
        trace_.active_rules.clear();
        trace_.active_rules.reserve(rules.size());
        for (const auto& r : rules) {
            RuleReference ref;
            ref.id          = r.id;
            ref.domain      = r.domain;
            ref.condition   = r.condition;
            ref.consequence = r.consequence;
            ref.confidence  = r.confidence;
            trace_.active_rules.push_back(std::move(ref));
        }
    }

    // =========================================================================
    // record_retrieved_episodes
    // =========================================================================

    void TraceBuilder::record_retrieved_episodes(
        const std::vector<RetrievalResult>& results)
    {
        trace_.retrieved_episodes.clear();
        trace_.retrieved_episodes.reserve(results.size());
        for (const auto& r : results) {
            EpisodeReference ref;
            ref.id = r.episode.id;
            ref.user_message_preview = r.episode.user_message.substr(
                0, std::min(r.episode.user_message.size(), size_t(100)));
            ref.reasoning_domain  = r.episode.reasoning_domain;
            ref.confidence        = r.episode.confidence;
            ref.retrieval_score   = r.score;
            trace_.retrieved_episodes.push_back(std::move(ref));
        }
    }

    // =========================================================================
    // Pass 1
    // =========================================================================

    void TraceBuilder::record_pass1_start() {
        pass1_start_ = std::chrono::steady_clock::now();
    }

    void TraceBuilder::record_pass1_complete(const FeelingOutput& feeling,
                                              bool valid,
                                              int retries,
                                              int tokens)
    {
        trace_.feeling       = feeling;
        trace_.feeling_valid = valid;
        trace_.pass1_retries = retries;
        trace_.pass1_tokens  = tokens;
        trace_.pass1_duration_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pass1_start_).count());
    }

    // =========================================================================
    // Tool calls
    // =========================================================================

    void TraceBuilder::record_tool_call(const ToolResult& result) {
        trace_.tool_calls.push_back(result);
    }

    void TraceBuilder::record_tool_iteration() {
        ++trace_.tool_iterations;
    }

    // =========================================================================
    // Symbolic check
    // =========================================================================

    void TraceBuilder::record_symbolic_check(
        const ConsistencyCheckResult& result)
    {
        trace_.symbolic_check.ran = true;
        trace_.symbolic_check.contradictions_found    =
            static_cast<int>(result.contradictions.size());
        trace_.symbolic_check.contradictions_resolved = result.contradictions_resolved;
        trace_.symbolic_check.contradictions_flagged  = result.contradictions_flagged;

        if (result.rule_committed && !result.committed_rule_id.empty()) {
            trace_.symbolic_check.rules_fired.push_back(result.committed_rule_id);
        }

        for (const auto& c : result.contradictions) {
            trace_.symbolic_check.contradictions.push_back(
                c.explanation.empty() ? "contradiction detected" : c.explanation);
        }
    }

    // =========================================================================
    // Rule extraction
    // =========================================================================

    void TraceBuilder::record_rule_committed(const std::string& rule_id,
                                              const Rule& rule)
    {
        trace_.rule_committed      = true;
        trace_.committed_rule_id   = rule_id;

        RuleReference ref;
        ref.id          = rule.id;
        ref.domain      = rule.domain;
        ref.condition   = rule.condition;
        ref.consequence = rule.consequence;
        ref.confidence  = rule.confidence;
        trace_.committed_rule = ref;
    }

    // =========================================================================
    // Agent steps
    // =========================================================================

    void TraceBuilder::record_agent_step(const AgentStepRecord& step) {
        trace_.agent_steps.push_back(step);
    }

    void TraceBuilder::record_agent_complete(bool goal_achieved, int iterations) {
        trace_.agent_goal_achieved = goal_achieved;
        trace_.agent_iterations    = iterations;
    }

    // =========================================================================
    // Pass 2
    // =========================================================================

    void TraceBuilder::record_pass2_start() {
        pass2_start_ = std::chrono::steady_clock::now();
    }

    void TraceBuilder::record_pass2_complete(const std::string& response,
                                              int tokens)
    {
        trace_.final_response  = response;
        trace_.pass2_tokens    = tokens;
        trace_.pass2_duration_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pass2_start_).count());
    }

    // =========================================================================
    // finalize
    // =========================================================================

    ReasoningTrace TraceBuilder::finalize() {
        trace_.total_tokens = trace_.pass1_tokens + trace_.pass2_tokens;
        trace_.total_duration_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - inference_start_).count());

        LOG_DEBUG("TraceBuilder: finalized " + trace_.summary());
        return trace_;
    }

} // namespace cardinal
