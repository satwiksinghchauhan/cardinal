// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - API Facade (v1.4.0)
// File: src/api/cardinal_api.h
//
// Changes from v1.3.0:
//   - SelfImprovementLoop owned here
//   - on_inference post-hook calls loop_.on_inference()
//   - on_session_boundary() calls loop_.on_session_boundary()
//   - New API methods:
//       get_self_model_status()  → SelfImprovementStatus
//       reflect()                → ReflectionResult   (Layer 2 on-demand)
//       trigger_training()       → bool               (Layer 3 on-demand)
//   - CardinalStatus extended: TRAINING_FAILED = 13, SELF_MODEL_ERROR = 14
//   - SystemStats extended with SelfImprovementStatus field
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
#include "self_model/self_model_types.h"   // SelfImprovementStatus, ReflectionResult

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
    class VisionEncoder;
    class VisionCache;
    class ToolExecutor;
    class AuditLog;
    class ExplainabilityExporter;
    class AgentExecutor;
    class SelfImprovementLoop;      // ← new in v1.4.0
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
        // Inference (unchanged signature)
        // ------------------------------------------------------------------
        CardinalResult<ChatResponse> chat(const std::string& session_id,
                                          const std::string& message);
        CardinalResult<ChatResponse> chat_stream(const std::string&       session_id,
                                                  const std::string&       message,
                                                  const ApiStreamCallback& stream_cb);

        // ------------------------------------------------------------------
        // Agentic execution (unchanged from v1.2.0)
        // ------------------------------------------------------------------
        CardinalResult<ChatResponse> agent(const std::string& session_id,
                                           const std::string& goal,
                                           int                max_iterations = 0);

        // ------------------------------------------------------------------
        // Tool management (unchanged from v1.2.0)
        // ------------------------------------------------------------------
        CardinalVoidResult register_tools(const std::vector<ToolDefinition>& tools);
        CardinalVoidResult unregister_tool(const std::string& name);
        CardinalResult<std::vector<ToolDefinition>> list_tools() const;

        // ------------------------------------------------------------------
        // Explainability (unchanged from v1.2.0)
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
        // Self-Improvement (new in v1.4.0)
        // ------------------------------------------------------------------

        // Returns the current state of all three self-improvement layers.
        // Mapped to GET /self_model by the HTTP server.
        CardinalResult<SelfImprovementStatus> get_self_model_status() const;

        // Trigger an on-demand Layer 2 reflection pass.
        // Runs synchronously; may take several seconds.
        // Mapped to POST /reflect by the HTTP server.
        CardinalResult<ReflectionResult> reflect();

        // Post a Layer 3 training request to the background thread.
        // Returns immediately (training is async).
        // domain_hint="" lets CurriculumBuilder decide the target domain.
        // Mapped to POST /train by the HTTP server.
        CardinalResult<bool> trigger_training(const std::string& domain_hint = "");

        // Called by the HTTP server / session manager between sessions.
        // Applies any pending adapter approved by AdapterEvaluator.
        void on_session_boundary();

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

        // Tools
        std::unique_ptr<ToolRegistry>          tool_registry_;
        std::unique_ptr<ToolExecutor>          tool_executor_;

        // Vision
        std::unique_ptr<VisionEncoder>         vision_encoder_;
        std::unique_ptr<VisionCache>           vision_cache_;

        // Explainability
        std::unique_ptr<AuditLog>              audit_log_;
        std::unique_ptr<ExplainabilityExporter> exporter_;

        // Agent
        std::unique_ptr<AgentExecutor>         agent_executor_;

        // Self-Improvement (new in v1.4.0)
        // Owns Layers 1-3: SelfModel, MetaCognition, CurriculumBuilder,
        // DatasetCurator, ITrainingBackend, AdapterEvaluator.
        std::unique_ptr<SelfImprovementLoop>   self_improvement_;

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
