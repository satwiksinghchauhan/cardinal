// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - API Facade
// File: src/api/cardinal_api.h
//
// The single entry point for all interfaces.
// Owns all core components -- one object, one lifetime.
//
// Usage:
//   CardinalAPI api;
//   auto init_result = api.init("config.json");
//   if (!init_result.ok()) { handle_error(); }
//
//   auto result = api.chat("default", "What is entropy?");
//   if (result.ok()) { display(result.value.response); }
//
//   api.shutdown();
//
// Thread safety:
//   All public methods are thread-safe.
//   Session operations use per-session locking where possible.
//   Heavy operations (inference) serialize per session.
//
// Error handling:
//   No exceptions cross this boundary.
//   All methods return CardinalResult<T> or CardinalVoidResult.
//
// Ownership:
//   CardinalAPI owns all core components via unique_ptr.
//   Components are constructed in init() and destroyed in shutdown().
//   Raw references to components are never exposed externally.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "api/cardinal_types.h"
#include "api/cardinal_settings.h"
#include "api/session.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <functional>

// Forward declarations -- full definitions in cardinal_api.cpp
namespace cardinal {
    class LLMEngine;
    class InferencePipeline;
    class RuleStore;
    class KnowledgeGraph;
    class EpisodicMemory;
    class EpisodicStorage;
    class EpisodicRetriever;
    class SymbolicEngine;
    class NeuralVerifier;
    class RuleExtractor;
    class ConsistencyChecker;
    class TrainingExporter;
}

namespace cardinal {

    // =========================================================================
    // CardinalAPI
    // =========================================================================
    class CardinalAPI {
    public:
        CardinalAPI();
        ~CardinalAPI();

        // Not copyable or movable -- owns unique resources
        CardinalAPI(const CardinalAPI&) = delete;
        CardinalAPI& operator=(const CardinalAPI&) = delete;
        CardinalAPI(CardinalAPI&&) = delete;
        CardinalAPI& operator=(CardinalAPI&&) = delete;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        // Initialize all components from config file.
        // Must be called before any other method.
        // Safe to check is_initialized() first.
        CardinalVoidResult init(const std::string& config_path = "config.json");

        // Clean shutdown -- saves state, closes DB, stops HTTP server.
        // Safe to call multiple times.
        CardinalVoidResult shutdown();

        bool is_initialized() const { return initialized_.load(); }

        // =====================================================================
        // Session management
        // =====================================================================

        // Create a new named session. Returns the session ID.
        // If session_id is empty, one is generated automatically.
        // If session_id already exists, returns existing session ID.
        CardinalResult<std::string> create_session(
            const std::string& session_id = "");

        // Destroy a session and its history.
        CardinalVoidResult destroy_session(const std::string& session_id);

        // Reset a session's history without destroying it.
        CardinalVoidResult reset_session(const std::string& session_id);

        // Get session info snapshot.
        CardinalResult<SessionInfo> get_session(
            const std::string& session_id) const;

        // Get all active session IDs.
        CardinalResult<std::vector<std::string>> list_sessions() const;

        // =====================================================================
        // Inference
        // =====================================================================

        // Send a message and get a response.
        // Creates session if it doesn't exist.
        // Runs full two-pass inference + memory write + consistency check.
        CardinalResult<ChatResponse> chat(
            const std::string& session_id,
            const std::string& message);

        // Send a message with streaming callback.
        // stream_cb is called for each token as it is generated.
        // Final ChatResponse is returned after streaming completes.
        CardinalResult<ChatResponse> chat_stream(
            const std::string& session_id,
            const std::string& message,
            const ApiStreamCallback& stream_cb);

        // =====================================================================
        // Memory
        // =====================================================================

        // Get current memory and verifier stats.
        CardinalResult<SystemStats> get_stats() const;

        // Get all rules in the rule store.
        CardinalResult<std::vector<RuleInfo>> get_rules() const;

        // Get all episodes matching a query.
        // keyword: FTS5 search (empty = all)
        // domain:  filter by reasoning domain (empty = all)
        CardinalResult<std::vector<EpisodeInfo>> get_episodes(
            const std::string& keyword = "",
            const std::string& domain = "",
            float              min_conf = 0.0f,
            int                max_results = 20) const;

        // Run a full contradiction scan across all rules.
        CardinalResult<ScanResult> run_scan();

        // Force a maintenance cycle (decay, prune, save).
        CardinalVoidResult run_maintenance();

        // =====================================================================
        // Training export
        // =====================================================================

        // Export training data to a JSONL file.
        CardinalResult<ExportInfo> export_training_data(
            const ExportRequest& request);

        // Dry run -- returns what would be exported without writing.
        CardinalResult<ExportInfo> export_dry_run(
            const ExportRequest& request) const;

        // =====================================================================
        // Settings
        // =====================================================================

        // Get current runtime settings.
        CardinalResult<CardinalSettings> get_settings() const;

        // Update runtime settings.
        CardinalVoidResult update_settings(const CardinalSettings& settings);

        // Update a single setting by key/value string.
        CardinalVoidResult set_setting(const std::string& key,
            const std::string& value);

        // Reset all settings to config file defaults.
        CardinalVoidResult reset_settings();

        // =====================================================================
        // Health
        // =====================================================================

        // Quick alive check -- returns OK if initialized and not shutting down.
        CardinalVoidResult health_check() const;

        // Get uptime as a human-readable string.
        std::string uptime_string() const;

    private:
        // =====================================================================
        // Internal helpers
        // =====================================================================

        // Run the full post-inference pipeline:
        // dual-write, retriever notify, consistency check, rule linkback, save.
        // Returns the populated ChatResponse.
        ChatResponse run_post_inference(
            const std::string& session_id,
            const std::string& user_message,
            const struct InferenceResponse& resp);

        // Convert core FeelingOutput to API FeelingInfo.
        static FeelingInfo to_feeling_info(const struct FeelingOutput& f);

        // Convert core Rule to API RuleInfo.
        static RuleInfo to_rule_info(const struct Rule& r);

        // Convert core EpisodeRecord to API EpisodeInfo.
        static EpisodeInfo to_episode_info(const struct EpisodeRecord& ep);

        // Validate that the API is initialized before any operation.
        CardinalVoidResult check_initialized() const;

        // =====================================================================
        // Owned components (constructed in init, destroyed in shutdown)
        // =====================================================================

        // Config -- loaded once, never modified
        std::unique_ptr<CardinalConfig>      config_;

        // Memory layer
        std::unique_ptr<RuleStore>           rule_store_;
        std::unique_ptr<KnowledgeGraph>      knowledge_graph_;
        std::unique_ptr<EpisodicMemory>      episodic_;
        std::unique_ptr<EpisodicStorage>     storage_;
        std::unique_ptr<EpisodicRetriever>   retriever_;

        // Verifier pipeline
        std::unique_ptr<SymbolicEngine>      symbolic_;
        std::unique_ptr<NeuralVerifier>      neural_verifier_;
        std::unique_ptr<RuleExtractor>       extractor_;
        std::unique_ptr<ConsistencyChecker>  checker_;

        // LLM
        std::unique_ptr<LLMEngine>           engine_;
        std::unique_ptr<InferencePipeline>   pipeline_;

        // API layer
        std::unique_ptr<SettingsManager>     settings_;
        std::unique_ptr<SessionManager>      sessions_;
        std::unique_ptr<TrainingExporter>    exporter_;

        // =====================================================================
        // State
        // =====================================================================
        std::atomic<bool>                    initialized_{ false };
        std::atomic<bool>                    shutting_down_{ false };

        std::chrono::steady_clock::time_point start_time_;

        // Protects inference -- one inference at a time per session
        // Map of session_id -> per-session mutex would be ideal but
        // for now a single inference mutex is safe and simple
        mutable std::mutex                   inference_mutex_;

        // Protects session map reads/writes
        mutable std::shared_mutex            session_mutex_;

        // General API state mutex
        mutable std::mutex                   api_mutex_;
    };

} // namespace cardinal