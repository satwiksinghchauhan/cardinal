#pragma once
// =============================================================================
// Cardinal - API Facade (v1.6.0)
// File: src/api/cardinal_api.h
//
// Changes from v1.5.0:
//   - VoiceLoop owned here via unique_ptr
//   - voice_types.h included
//   - New voice API methods: enable_voice, disable_voice, is_voice_active,
//     get_voice_status, voice_speak, voice_transcribe
//   - check_voice() private helper
//   - voice_mutex_ added
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
#include "self_model/self_model_types.h"
#include "scheduler/scheduler_types.h"
#include "computer/computer_types.h"
#include "voice/voice_types.h"           // ← new in v1.6.0

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
    class SelfImprovementLoop;
    class SchedulerEngine;
    class DisplayDetector;
    class ScreenReader;
    class InputController;
    class AppController;
    class BrowserController;
    class ShellExecutor;
    class FileManager;
    class SystemController;
    class EmailController;
    class AtSpiReader;
    class VoiceLoop;                     // ← new in v1.6.0
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
#include <optional>

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
        // Session management
        // ------------------------------------------------------------------
        CardinalResult<std::string>              create_session(const std::string& id = "");
        CardinalVoidResult                       destroy_session(const std::string& id);
        CardinalVoidResult                       reset_session(const std::string& id);
        CardinalResult<SessionInfo>              get_session(const std::string& id) const;
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
        // Agent
        // ------------------------------------------------------------------
        CardinalResult<ChatResponse> agent(const std::string& session_id,
                                           const std::string& goal,
                                           int                max_iterations = 0);

        // ------------------------------------------------------------------
        // Tool management
        // ------------------------------------------------------------------
        CardinalVoidResult register_tools(const std::vector<ToolDefinition>& tools);
        CardinalVoidResult unregister_tool(const std::string& name);
        CardinalResult<std::vector<ToolDefinition>> list_tools() const;

        // ------------------------------------------------------------------
        // Explainability
        // ------------------------------------------------------------------
        CardinalResult<std::string>  get_trace(const std::string& inference_id) const;
        CardinalResult<std::string>  export_trace(const std::string& inference_id) const;
        CardinalResult<bool>         verify_trace(const std::string& inference_id) const;
        CardinalResult<std::string>  get_public_key() const;

        // ------------------------------------------------------------------
        // Memory / verifier
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
        // Training export
        // ------------------------------------------------------------------
        CardinalResult<ExportInfo> export_training_data(const ExportRequest& request);
        CardinalResult<ExportInfo> export_dry_run(const ExportRequest& request) const;

        // ------------------------------------------------------------------
        // Self-Improvement
        // ------------------------------------------------------------------
        CardinalResult<SelfImprovementStatus> get_self_model_status() const;
        CardinalResult<ReflectionResult>      reflect();
        CardinalResult<bool>                  trigger_training(
                                                  const std::string& domain_hint = "");
        void on_session_boundary();

        // ------------------------------------------------------------------
        // Scheduler (v1.5.0)
        // ------------------------------------------------------------------
        CardinalResult<SchedulerStatus>            get_scheduler_status() const;
        CardinalResult<std::vector<ScheduledTask>> list_tasks() const;
        CardinalResult<ScheduledTask>              get_task(const std::string& task_id) const;
        CardinalResult<TaskParseResult>            create_task(const std::string& nl,
                                                               const std::string& session_id = "");
        CardinalResult<std::string>                create_task_direct(const ScheduledTask& task);
        CardinalVoidResult                         update_task(const ScheduledTask& task);
        CardinalVoidResult                         delete_task(const std::string& task_id);
        CardinalVoidResult                         enable_task(const std::string& task_id);
        CardinalVoidResult                         disable_task(const std::string& task_id);
        CardinalResult<std::string>                run_task_now(const std::string& task_id);
        CardinalResult<std::vector<TaskRun>>       get_task_history(const std::string& task_id,
                                                                     int limit = 50) const;
        CardinalResult<std::vector<TaskRun>>       get_recent_runs(int limit = 100) const;
        CardinalResult<std::vector<TaskActionLog>> get_run_action_logs(const std::string& run_id) const;

        // ------------------------------------------------------------------
        // Computer Use (v1.5.0)
        // ------------------------------------------------------------------
        CardinalResult<ScreenInfo>  get_computer_status() const;
        CardinalResult<Screenshot>  take_screenshot(bool analyze = false,
                                                    const std::string& prompt = "");
        CardinalResult<std::string> computer_click(int x, int y,
                                                   const std::string& description = "");
        CardinalResult<std::string> computer_type(const std::string& text,
                                                  const std::string& key = "");
        CardinalResult<ShellResult> computer_shell(const std::string& command,
                                                   int timeout_seconds = 0);

        // ------------------------------------------------------------------
        // Voice (v1.6.0)
        // ------------------------------------------------------------------
        CardinalVoidResult               enable_voice(const std::string& input_mode = "");
        CardinalVoidResult               disable_voice();
        bool                             is_voice_active() const;
        VoiceStatus                      get_voice_status() const;
        CardinalResult<TTSResult>        voice_speak(const std::string& text);
        CardinalResult<TranscriptResult> voice_transcribe(const AudioChunk& audio);

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
        ChatResponse       run_post_inference(const std::string&       session_id,
                                              const std::string&       user_message,
                                              const InferenceResponse& resp);
        FeelingInfo        to_feeling_info(const FeelingOutput& f);
        RuleInfo           to_rule_info(const Rule& r) const;
        EpisodeInfo        to_episode_info(const EpisodeRecord& ep) const;
        CardinalVoidResult check_initialized() const;
        CardinalVoidResult check_scheduler() const;
        CardinalVoidResult check_computer_use() const;
        CardinalVoidResult check_voice() const;           // ← new in v1.6.0

        // ------------------------------------------------------------------
        // Subsystem ownership
        // ------------------------------------------------------------------
        std::unique_ptr<CardinalConfig>        config_;
        std::unique_ptr<ILLMBackend>           backend_;
        std::unique_ptr<InferencePipeline>     pipeline_;

        std::unique_ptr<RuleStore>             rule_store_;
        std::unique_ptr<KnowledgeGraph>        knowledge_graph_;
        std::unique_ptr<EpisodicMemory>        episodic_;
        std::unique_ptr<EpisodicStorage>       storage_;
        std::unique_ptr<EpisodicRetriever>     retriever_;

        std::unique_ptr<SymbolicEngine>        symbolic_;
        std::unique_ptr<RuleExtractor>         extractor_;
        std::unique_ptr<NeuralVerifier>        neural_verifier_;
        std::unique_ptr<ConsistencyChecker>    checker_;

        std::unique_ptr<ToolRegistry>          tool_registry_;
        std::unique_ptr<ToolExecutor>          tool_executor_;

        std::unique_ptr<VisionEncoder>         vision_encoder_;
        std::unique_ptr<VisionCache>           vision_cache_;

        std::unique_ptr<AuditLog>              audit_log_;
        std::unique_ptr<ExplainabilityExporter> exporter_;

        std::unique_ptr<AgentExecutor>         agent_executor_;
        std::unique_ptr<SelfImprovementLoop>   self_improvement_;

        // Scheduler (v1.5.0)
        std::unique_ptr<SchedulerEngine>       scheduler_;

        // Computer Use (v1.5.0)
        std::unique_ptr<DisplayDetector>       display_detector_;
        std::unique_ptr<ScreenReader>          screen_reader_;
        std::unique_ptr<InputController>       input_controller_;
        std::unique_ptr<AppController>         app_controller_;
        std::unique_ptr<BrowserController>     browser_controller_;
        std::unique_ptr<ShellExecutor>         shell_executor_;
        std::unique_ptr<FileManager>           file_manager_;
        std::unique_ptr<SystemController>      system_controller_;
        std::unique_ptr<EmailController>       email_controller_;
        std::unique_ptr<AtSpiReader>           atspi_reader_;

        // Voice (v1.6.0)
        std::unique_ptr<VoiceLoop>             voice_loop_;

        std::unique_ptr<TrainingExporter>      training_exporter_;
        std::unique_ptr<SettingsManager>       settings_;
        std::unique_ptr<SessionManager>        sessions_;

        // ------------------------------------------------------------------
        // Synchronisation
        // ------------------------------------------------------------------
        mutable std::mutex        api_mutex_;
        mutable std::shared_mutex session_mutex_;
        std::mutex                inference_mutex_;
        mutable std::mutex        voice_mutex_;           // ← new in v1.6.0

        std::atomic<bool> initialized_{ false };
        std::atomic<bool> shutting_down_{ false };

        std::chrono::steady_clock::time_point start_time_;
    };

} // namespace cardinal
