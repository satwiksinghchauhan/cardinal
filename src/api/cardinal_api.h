// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - API Facade (v1.2.0)
// File: src/api/cardinal_api.h
//
// Changes from v1.1.0:
//   - ToolRegistry, ToolExecutor owned here
//   - AuditLog, ExplainabilityExporter owned here
//   - AgentExecutor owned here
//   - agent() method added
//   - register_tools(), get_trace(), export_trace() added
//   - ChatResponse extended with trace and agent_result
// =============================================================================

#include "utils/config_loader.h"
#include "utils/json_parser.h"
#include "core/llm_backend.h"
#include "core/inference.h"
#include "core/feeling_output.h"
#include "api/cardinal_types.h"
#include "api/session.h"
#include "api/cardinal_settings.h"
#include "tools/tool_result.h"
#include "agent/agent_types.h"
#include "explainability/reasoning_trace.h"

namespace cardinal {
    class RuleStore;
    class KnowledgeGraph;
    class EpisodicMemory;
    class EpisodicStorage;
    class EpisodicRetriever;
    class SymbolicEngine;
    class RuleExtractor;
    class NeuralVerifier;
    class ConsistencyChecker;
    class TrainingExporter;
    class ToolRegistry;
    class ToolExecutor;
    class AuditLog;
    class ExplainabilityExporter;
    class AgentExecutor;
    struct Rule;
    struct EpisodeRecord;
    struct FeelingOutput;
}

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <functional>
#include <chrono>

namespace cardinal {

    using ApiStreamCallback = std::function<bool(const StreamToken&)>;

    class CardinalAPI {
    public:
        CardinalAPI();
        ~CardinalAPI();

        CardinalAPI(const CardinalAPI&)            = delete;
        CardinalAPI& operator=(const CardinalAPI&) = delete;

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------
        CardinalVoidResult init(const std::string& config_path);
        CardinalVoidResult shutdown();

        // ------------------------------------------------------------------
        // Session management (unchanged)
        // ------------------------------------------------------------------
        CardinalResult<std::string>              create_session(const std::string& id = "");
        CardinalVoidResult                       destroy_session(const std::string& id);
        CardinalVoidResult                       reset_session(const std::string& id);
        CardinalResult<SessionInfo>              get_session(const std::string& id) const;
        CardinalResult<std::vector<std::string>> list_sessions() const;

        // ------------------------------------------------------------------
        // Inference (unchanged signature, richer response)
        // ------------------------------------------------------------------
        CardinalResult<ChatResponse> chat(const std::string& session_id,
                                          const std::string& message);
        CardinalResult<ChatResponse> chat_stream(const std::string&       session_id,
                                                  const std::string&       message,
                                                  const ApiStreamCallback& stream_cb);

        // ------------------------------------------------------------------
        // Agentic execution (new in v1.2.0)
        // ------------------------------------------------------------------
        CardinalResult<ChatResponse> agent(const std::string& session_id,
                                           const std::string& goal,
                                           int                max_iterations = 0);

        // ------------------------------------------------------------------
        // Tool management (new in v1.2.0)
        // ------------------------------------------------------------------
        CardinalVoidResult register_tools(const std::vector<ToolDefinition>& tools);
        CardinalVoidResult unregister_tool(const std::string& name);
        CardinalResult<std::vector<ToolDefinition>> list_tools() const;

        // ------------------------------------------------------------------
        // Explainability (new in v1.2.0)
        // ------------------------------------------------------------------
        CardinalResult<std::string>  get_trace(const std::string& inference_id) const;
        CardinalResult<std::string>  export_trace(const std::string& inference_id) const;
        CardinalResult<bool>         verify_trace(const std::string& inference_id) const;
        CardinalResult<std::string>  get_public_key() const;

        // ------------------------------------------------------------------
        // Memory / verifier (unchanged)
        // ------------------------------------------------------------------
        CardinalResult<SystemStats>              get_stats()    const;
        CardinalResult<std::vector<RuleInfo>>    get_rules()    const;
        CardinalResult<std::vector<EpisodeInfo>> get_episodes(
            const std::string& keyword    = "",
            const std::string& domain     = "",
            float              min_conf   = 0.0f,
            int                max_results = 50) const;

        CardinalResult<ScanResult> run_scan();
        CardinalVoidResult         run_maintenance();

        // ------------------------------------------------------------------
        // Training export (unchanged)
        // ------------------------------------------------------------------
        CardinalResult<ExportInfo> export_training_data(const ExportRequest& request);
        CardinalResult<ExportInfo> export_dry_run(const ExportRequest& request) const;

        // ------------------------------------------------------------------
        // Settings (unchanged)
        // ------------------------------------------------------------------
        CardinalResult<CardinalSettings> get_settings() const;
        CardinalVoidResult               update_settings(const CardinalSettings& settings);
        CardinalVoidResult               set_setting(const std::string& key,
                                                      const std::string& value);
        CardinalVoidResult               reset_settings();

        // ------------------------------------------------------------------
        // Health (unchanged)
        // ------------------------------------------------------------------
        CardinalVoidResult health_check() const;
        std::string        uptime_string() const;

    private:
        ChatResponse     run_post_inference(const std::string&       session_id,
                                            const std::string&       user_message,
                                            const InferenceResponse& resp);
        FeelingInfo      to_feeling_info(const FeelingOutput& f);
        RuleInfo         to_rule_info(const Rule& r) const;
        EpisodeInfo      to_episode_info(const EpisodeRecord& ep) const;
        CardinalVoidResult check_initialized() const;

        // ------------------------------------------------------------------
        // Subsystem ownership
        // ------------------------------------------------------------------
        std::unique_ptr<CardinalConfig>        config_;
        std::unique_ptr<ILLMBackend>           backend_;
        std::unique_ptr<InferencePipeline>     pipeline_;

        // Memory
        std::unique_ptr<RuleStore>             rule_store_;
        std::unique_ptr<KnowledgeGraph>        knowledge_graph_;
        std::unique_ptr<EpisodicMemory>        episodic_;
        std::unique_ptr<EpisodicStorage>       storage_;
        std::unique_ptr<EpisodicRetriever>     retriever_;

        // Verifier
        std::unique_ptr<SymbolicEngine>        symbolic_;
        std::unique_ptr<RuleExtractor>         extractor_;
        std::unique_ptr<NeuralVerifier>        neural_verifier_;
        std::unique_ptr<ConsistencyChecker>    checker_;

        // Tools (new)
        std::unique_ptr<ToolRegistry>          tool_registry_;
        std::unique_ptr<ToolExecutor>          tool_executor_;

        // Explainability (new)
        std::unique_ptr<AuditLog>              audit_log_;
        std::unique_ptr<ExplainabilityExporter> exporter_;

        // Agent (new)
        std::unique_ptr<AgentExecutor>         agent_executor_;

        // API layer
        std::unique_ptr<TrainingExporter>      training_exporter_;
        std::unique_ptr<SettingsManager>       settings_;
        std::unique_ptr<SessionManager>        sessions_;

        // ------------------------------------------------------------------
        // Synchronisation
        // ------------------------------------------------------------------
        mutable std::mutex        api_mutex_;
        mutable std::shared_mutex session_mutex_;
        std::mutex                inference_mutex_;

        std::atomic<bool> initialized_{ false };
        std::atomic<bool> shutting_down_{ false };

        std::chrono::steady_clock::time_point start_time_;
    };

} // namespace cardinal
