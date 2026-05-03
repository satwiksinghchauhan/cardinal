// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - API Facade
// File: src/api/cardinal_api.h
//
// Top-level entry point for all Cardinal operations.
// Owns all subsystem instances and exposes a clean public API.
//
// Changes from original:
//   - #include "core/llm_engine.h" replaced with "core/llm_backend.h"
//   - engine_ member changed from unique_ptr<LLMEngine>
//     to unique_ptr<ILLMBackend> (constructed by BackendFactory)
//   - Windows preprocessor guards removed (Linux-only build)
//   - Startup log now prints backend type and constrained-decoding capability
// =============================================================================

#include "utils/config_loader.h"
#include "utils/json_parser.h"
#include "core/llm_backend.h"
#include "core/inference.h"
#include "core/feeling_output.h"
#include "api/cardinal_types.h"
#include "api/session.h"
#include "api/cardinal_settings.h"

// Forward declarations — full definitions only in cardinal_api.cpp
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

    // -------------------------------------------------------------------------
    // ApiStreamCallback
    // -------------------------------------------------------------------------
    using ApiStreamCallback = std::function<bool(const StreamToken&)>;

    // -------------------------------------------------------------------------
    // CardinalAPI
    // -------------------------------------------------------------------------
    class CardinalAPI {
    public:
        CardinalAPI();
        ~CardinalAPI();

        // Non-copyable / non-movable
        CardinalAPI(const CardinalAPI&)            = delete;
        CardinalAPI& operator=(const CardinalAPI&) = delete;

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------
        CardinalVoidResult init(const std::string& config_path);
        CardinalVoidResult shutdown();

        // ------------------------------------------------------------------
        // Session management
        // ------------------------------------------------------------------
        CardinalResult<std::string>              create_session(const std::string& session_id = "");
        CardinalVoidResult                       destroy_session(const std::string& session_id);
        CardinalVoidResult                       reset_session(const std::string& session_id);
        CardinalResult<SessionInfo>              get_session(const std::string& session_id) const;
        CardinalResult<std::vector<std::string>> list_sessions() const;

        // ------------------------------------------------------------------
        // Inference
        // ------------------------------------------------------------------
        CardinalResult<ChatResponse> chat(const std::string& session_id,
                                          const std::string& message);
        CardinalResult<ChatResponse> chat_stream(const std::string&       session_id,
                                                  const std::string&       message,
                                                  const ApiStreamCallback& stream_cb);

        // ------------------------------------------------------------------
        // Memory / verifier
        // ------------------------------------------------------------------
        CardinalResult<SystemStats>             get_stats()    const;
        CardinalResult<std::vector<RuleInfo>>   get_rules()    const;
        CardinalResult<std::vector<EpisodeInfo>> get_episodes(
            const std::string& keyword    = "",
            const std::string& domain     = "",
            float              min_conf   = 0.0f,
            int                max_results = 50) const;

        CardinalResult<ScanResult> run_scan();
        CardinalVoidResult         run_maintenance();

        // ------------------------------------------------------------------
        // Training export
        // ------------------------------------------------------------------
        CardinalResult<ExportInfo> export_training_data(const ExportRequest& request);
        CardinalResult<ExportInfo> export_dry_run(const ExportRequest& request) const;

        // ------------------------------------------------------------------
        // Settings
        // ------------------------------------------------------------------
        CardinalResult<CardinalSettings> get_settings() const;
        CardinalVoidResult               update_settings(const CardinalSettings& settings);
        CardinalVoidResult               set_setting(const std::string& key,
                                                      const std::string& value);
        CardinalVoidResult               reset_settings();

        // ------------------------------------------------------------------
        // Health
        // ------------------------------------------------------------------
        CardinalVoidResult health_check() const;
        std::string        uptime_string() const;

    private:
        // ------------------------------------------------------------------
        // Internal helpers
        // ------------------------------------------------------------------
        ChatResponse     run_post_inference(const std::string&       session_id,
                                            const std::string&       user_message,
                                            const InferenceResponse& resp);
        FeelingInfo      to_feeling_info(const FeelingOutput&  f) const;
        RuleInfo         to_rule_info(const Rule&             r) const;
        EpisodeInfo      to_episode_info(const EpisodeRecord&  ep) const;
        CardinalVoidResult check_initialized() const;

        // ------------------------------------------------------------------
        // Subsystem ownership
        // ------------------------------------------------------------------
        std::unique_ptr<CardinalConfig>   config_;

        // LLM backend — constructed by BackendFactory::create() in init().
        // Type is determined by config.backend.type at runtime.
        // Everything downstream uses ILLMBackend& — never the concrete type.
        std::unique_ptr<ILLMBackend>      backend_;

        std::unique_ptr<InferencePipeline> pipeline_;

        // Memory
        std::unique_ptr<RuleStore>         rule_store_;
        std::unique_ptr<KnowledgeGraph>    knowledge_graph_;
        std::unique_ptr<EpisodicMemory>    episodic_;
        std::unique_ptr<EpisodicStorage>   storage_;
        std::unique_ptr<EpisodicRetriever> retriever_;

        // Verifier pipeline
        std::unique_ptr<SymbolicEngine>    symbolic_;
        std::unique_ptr<RuleExtractor>     extractor_;
        std::unique_ptr<NeuralVerifier>    neural_verifier_;
        std::unique_ptr<ConsistencyChecker> checker_;

        // API layer
        std::unique_ptr<TrainingExporter>  exporter_;
        std::unique_ptr<SettingsManager>   settings_;
        std::unique_ptr<SessionManager>    sessions_;

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
