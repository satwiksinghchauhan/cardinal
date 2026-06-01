// =============================================================================
// Cardinal - API Facade Implementation (v1.6.0)
// File: src/api/cardinal_api.cpp
//
// Changes from v1.5.0:
//   - VoiceLoop included and initialised in init() if voice.enabled
//   - shutdown() stops VoiceLoop
//   - enable_voice / disable_voice / is_voice_active / get_voice_status
//     / voice_speak / voice_transcribe / check_voice added
// =============================================================================

#include "api/cardinal_api.h"

#include "utils/logger.h"
#include "utils/json_parser.h"
#include "core/backend_factory.h"
#include "core/inference.h"
#include "memory/rule_store.h"
#include "memory/knowledge_graph.h"
#include "memory/episodic.h"
#include "memory/episodic_storage.h"
#include "memory/episodic_retriever.h"
#include "verifier/symbolic_engine.h"
#include "verifier/neural_verifier.h"
#include "verifier/rule_extractor.h"
#include "verifier/consistency_check.h"
#include "learning/training_exporter.h"
#include "tools/tool_registry.h"
#include "tools/tool_executor.h"
#include "vision/vision_encoder.h"
#include "vision/vision_cache.h"
#include "agent/agent_executor.h"
#include "explainability/audit_log.h"
#include "explainability/explainability_exporter.h"
#include "training/self_improvement_loop.h"

// v1.5.0
#include "scheduler/scheduler_engine.h"
#include "computer/display_detector.h"
#include "computer/screen_reader.h"
#include "computer/input_controller.h"
#include "computer/app_controller.h"
#include "computer/browser_controller.h"
#include "computer/shell_executor.h"
#include "computer/file_manager.h"
#include "computer/system_controller.h"
#include "computer/email_controller.h"
#include "computer/atspi_reader.h"

// v1.6.0
#include "voice/voice_loop.h"

#include <sstream>
#include <iomanip>

namespace cardinal {

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    CardinalAPI::CardinalAPI() {}

    CardinalAPI::~CardinalAPI() {
        if (initialized_.load()) shutdown();
    }

    // =========================================================================
    // init
    // =========================================================================

    CardinalVoidResult CardinalAPI::init(const std::string& config_path) {
        std::lock_guard<std::mutex> lock(api_mutex_);

        if (initialized_.load())
            return CardinalVoidResult::failure(
                CardinalStatus::ALREADY_INITIALIZED,
                "CardinalAPI::init() called twice");

        try {
            LOG_INFO("CardinalAPI v1.6.0 initializing from: " + config_path);
            start_time_ = std::chrono::steady_clock::now();

            // Config
            config_ = std::make_unique<CardinalConfig>(
                ConfigLoader::load(config_path));

            // Memory
            rule_store_      = std::make_unique<RuleStore>(*config_);
            knowledge_graph_ = std::make_unique<KnowledgeGraph>(*config_);
            episodic_        = std::make_unique<EpisodicMemory>(*config_);
            storage_         = std::make_unique<EpisodicStorage>(*config_);

            rule_store_->load();
            knowledge_graph_->load();
            episodic_->open();
            storage_->open();

            // Retriever
            retriever_ = std::make_unique<EpisodicRetriever>(*config_, *storage_);
            retriever_->init();

            // Verifier
            symbolic_ = std::make_unique<SymbolicEngine>(*config_);
            symbolic_->init("src/verifier/cardinal_kb.pl");

            extractor_ = std::make_unique<RuleExtractor>(
                *config_, *rule_store_, *symbolic_);

            neural_verifier_ = std::make_unique<NeuralVerifier>(*config_);
            neural_verifier_->load();

            checker_ = std::make_unique<ConsistencyChecker>(
                *config_, *rule_store_, *episodic_,
                *symbolic_, *extractor_, *neural_verifier_);
            checker_->init();

            // LLM Backend
            backend_ = BackendFactory::create(*config_);
            backend_->load_model();
            {
                auto info = backend_->get_info();
                LOG_INFO("Backend: " + info.name +
                         " | model: " + info.model_name +
                         " | constrained_decoding: " +
                         (backend_->supports_constrained_decoding()
                              ? "native" : "retry_loop"));
            }

            // Tools
            tool_registry_ = std::make_unique<ToolRegistry>(*config_);
            tool_registry_->init();

            tool_executor_ = std::make_unique<ToolExecutor>(*config_, *tool_registry_);
            tool_executor_->set_knowledge_graph(knowledge_graph_.get());
            tool_executor_->set_retriever(retriever_.get());

            // Vision encoder
            vision_cache_   = std::make_unique<VisionCache>(*config_);
            vision_cache_->init();

            vision_encoder_ = std::make_unique<VisionEncoder>(*config_);
            vision_encoder_->load();

            if (vision_encoder_->is_ready()) {
                tool_executor_->set_vision_encoder(vision_encoder_.get());
                tool_executor_->set_vision_cache(vision_cache_.get());
                LOG_INFO("Vision encoder ready (moondream2, CPU)");
            } else {
                LOG_INFO("Vision encoder not loaded — analyze_image tool disabled");
            }

            // Explainability
            audit_log_ = std::make_unique<AuditLog>(*config_);
            audit_log_->open();

            exporter_ = std::make_unique<ExplainabilityExporter>(*config_);

            // Inference pipeline
            pipeline_ = std::make_unique<InferencePipeline>(*config_, *backend_);
            pipeline_->set_retriever(retriever_.get());
            pipeline_->set_tool_executor(tool_executor_.get());
            pipeline_->set_tool_registry(tool_registry_.get());
            pipeline_->set_audit_log(audit_log_.get());

            // Agent executor
            agent_executor_ = std::make_unique<AgentExecutor>(
                *config_, *backend_, *tool_registry_, *tool_executor_);

            // API layer
            training_exporter_ = std::make_unique<TrainingExporter>(
                *config_, *storage_, *rule_store_);
            settings_ = std::make_unique<SettingsManager>(
                *config_, *retriever_, *pipeline_);
            sessions_ = std::make_unique<SessionManager>();
            sessions_->create();

            // Self-Improvement (v1.4.0)
            if (config_->self_improvement.enabled) {
                self_improvement_ = std::make_unique<SelfImprovementLoop>(
                    *config_, *storage_, *rule_store_, *backend_);
                self_improvement_->start();
                LOG_INFO("Self-Improvement loop started (Layers 1-3)");
            } else {
                LOG_INFO("Self-Improvement disabled in config");
            }

            // ---- Scheduler (v1.5.0) ----
            if (config_->scheduler.enabled) {
                SchedulerDeps deps;
                deps.agent_executor   = agent_executor_.get();
                deps.pipeline         = pipeline_.get();
                deps.self_improvement = self_improvement_.get();
                deps.episodic         = storage_.get();
                deps.sessions         = sessions_.get();
                deps.inference_busy_fn = [this]() -> bool {
                    if (inference_mutex_.try_lock()) {
                        inference_mutex_.unlock();
                        return false;
                    }
                    return true;
                };
                scheduler_ = std::make_unique<SchedulerEngine>(
                    *config_, std::move(deps));
                scheduler_->start();
                LOG_INFO("SchedulerEngine started");
            }

            // ---- Computer Use (v1.5.0) ----
            if (config_->computer_use.enabled) {
                display_detector_ = std::make_unique<DisplayDetector>();
                display_detector_->detect();

                screen_reader_ = std::make_unique<ScreenReader>(
                    *display_detector_, *config_,
                    vision_encoder_->is_ready() ? vision_encoder_.get() : nullptr);

                input_controller_  = std::make_unique<InputController>(
                    *display_detector_, *config_);
                app_controller_    = std::make_unique<AppController>(
                    *display_detector_, *config_);
                shell_executor_    = std::make_unique<ShellExecutor>(*config_);
                file_manager_      = std::make_unique<FileManager>(*config_);
                system_controller_ = std::make_unique<SystemController>(*config_);
                email_controller_  = std::make_unique<EmailController>(*config_);
                atspi_reader_      = std::make_unique<AtSpiReader>(*config_);
                browser_controller_= std::make_unique<BrowserController>(
                    *config_,
                    vision_encoder_->is_ready() ? vision_encoder_.get() : nullptr);

                LOG_INFO("Computer use initialised (" +
                    std::string(display_server_to_string(
                        display_detector_->server())) + ")");

                tool_executor_->set_screen_reader(screen_reader_.get());
                tool_executor_->set_input_controller(input_controller_.get());
                tool_executor_->set_app_controller(app_controller_.get());
                tool_executor_->set_browser_controller(browser_controller_.get());
                tool_executor_->set_shell_executor(shell_executor_.get());
                tool_executor_->set_file_manager(file_manager_.get());
                tool_executor_->set_system_controller(system_controller_.get());
                tool_executor_->set_email_controller(email_controller_.get());
            }

            // Wire scheduler into tool_executor
            if (scheduler_)
                tool_executor_->set_scheduler(scheduler_.get());

            // ---- Voice (v1.6.0) ----
            if (config_->voice.enabled) {
                auto vr = enable_voice();
                if (!vr.ok()) {
                    LOG_WARN("Voice subsystem failed to start: " + vr.error_message +
                                " — continuing without voice");
                }
            }

            initialized_.store(true);

            LOG_INFO("CardinalAPI initialized — episodes=" +
                std::to_string(storage_->count()) +
                " rules="    + std::to_string(rule_store_->size()) +
                " tools="    + std::to_string(tool_registry_->count()) +
                " audit_log=" + std::to_string(audit_log_->count()) +
                " signing="  + (audit_log_->public_key_fingerprint().empty()
                                 ? "disabled" : "enabled"));

            return CardinalVoidResult::success();
        }
        catch (const std::exception& e) {
            LOG_FATAL("CardinalAPI::init failed: " + std::string(e.what()));
            return CardinalVoidResult::failure(
                CardinalStatus::CONFIG_ERROR,
                std::string("Initialization failed: ") + e.what());
        }
    }

    // =========================================================================
    // shutdown
    // =========================================================================

    CardinalVoidResult CardinalAPI::shutdown() {
        std::lock_guard<std::mutex> lock(api_mutex_);
        if (!initialized_.load()) return CardinalVoidResult::success();

        shutting_down_.store(true);
        try {
            LOG_INFO("CardinalAPI shutting down...");

            // v1.6.0 — voice first (has its own audio thread)
            if (voice_loop_) {
                voice_loop_->stop();
                voice_loop_.reset();
            }

            // v1.5.0 — stop scheduler and browser
            if (scheduler_) {
                scheduler_->stop();
                scheduler_.reset();
            }
            if (browser_controller_) {
                browser_controller_->stop();
            }

            if (checker_)    checker_->run_maintenance();
            if (rule_store_) rule_store_->save();
            if (sessions_)   sessions_->destroy_all();
            if (storage_)    storage_->close();
            if (audit_log_)  audit_log_->close();

            initialized_.store(false);
            shutting_down_.store(false);
            LOG_INFO("CardinalAPI shutdown complete");
            return CardinalVoidResult::success();
        }
        catch (const std::exception& e) {
            initialized_.store(false);
            return CardinalVoidResult::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("Shutdown error: ") + e.what());
        }
    }

    // =========================================================================
    // Session management
    // =========================================================================

    CardinalResult<std::string>
        CardinalAPI::create_session(const std::string& session_id) {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::string>::failure(check.status, check.error_message);
        std::unique_lock lock(session_mutex_);
        if (!session_id.empty() && sessions_->exists(session_id))
            return CardinalResult<std::string>::success(session_id);
        return CardinalResult<std::string>::success(sessions_->create());
    }

    CardinalVoidResult CardinalAPI::destroy_session(const std::string& id) {
        auto check = check_initialized();
        if (!check.ok()) return check;
        std::unique_lock lock(session_mutex_);
        if (!sessions_->destroy(id))
            return CardinalVoidResult::failure(
                CardinalStatus::SESSION_NOT_FOUND, "Session not found: " + id);
        on_session_boundary();
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::reset_session(const std::string& id) {
        auto check = check_initialized();
        if (!check.ok()) return check;
        std::unique_lock lock(session_mutex_);
        auto* session = sessions_->get(id);
        if (!session)
            return CardinalVoidResult::failure(
                CardinalStatus::SESSION_NOT_FOUND, "Session not found: " + id);
        session->reset();
        return CardinalVoidResult::success();
    }

    CardinalResult<SessionInfo>
        CardinalAPI::get_session(const std::string& id) const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<SessionInfo>::failure(check.status, check.error_message);
        std::shared_lock lock(session_mutex_);
        const auto* s = sessions_->get(id);
        if (!s)
            return CardinalResult<SessionInfo>::failure(
                CardinalStatus::SESSION_NOT_FOUND, "Session not found: " + id);
        return CardinalResult<SessionInfo>::success(s->to_session_info());
    }

    CardinalResult<std::vector<std::string>> CardinalAPI::list_sessions() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::vector<std::string>>::failure(
                check.status, check.error_message);
        std::shared_lock lock(session_mutex_);
        return CardinalResult<std::vector<std::string>>::success(
            sessions_->all_ids());
    }

    // =========================================================================
    // chat / chat_stream
    // =========================================================================

    CardinalResult<ChatResponse>
        CardinalAPI::chat(const std::string& session_id,
                          const std::string& message) {
        return chat_stream(session_id, message, nullptr);
    }

    CardinalResult<ChatResponse>
        CardinalAPI::chat_stream(const std::string&       session_id,
                                  const std::string&       message,
                                  const ApiStreamCallback& stream_cb)
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ChatResponse>::failure(
                check.status, check.error_message);

        if (message.empty())
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INVALID_INPUT, "Message cannot be empty");

        if (shutting_down_.load())
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::SHUTDOWN, "API is shutting down");

        {
            std::unique_lock lock(session_mutex_);
            if (!sessions_->exists(session_id)) sessions_->create();
        }

        std::lock_guard<std::mutex> inf_lock(inference_mutex_);

        try {
            std::vector<ChatMessage> history;
            {
                std::shared_lock lock(session_mutex_);
                auto* session = sessions_->get(session_id);
                if (session) history = session->get_history();
            }

            InferenceRequest req;
            req.user_message    = message;
            req.history         = history;
            req.stream_response = (stream_cb != nullptr);
            req.tools_enabled   = config_->tools.web_search.enabled ||
                                   config_->tools.calculator.enabled ||
                                   config_->tools.run_python.enabled;

            StreamCallback core_cb = nullptr;
            if (stream_cb) {
                core_cb = [&stream_cb, &session_id](const std::string& token) -> bool {
                    StreamToken st;
                    st.session_id = session_id;
                    st.token      = token;
                    st.is_final   = false;
                    return stream_cb(st);
                };
            }

            auto resp = pipeline_->run(req, core_cb);

            if (!resp.success)
                return CardinalResult<ChatResponse>::failure(
                    CardinalStatus::INFERENCE_FAILED,
                    resp.error_message.empty() ? "Inference failed" : resp.error_message);

            ChatResponse chat_resp = run_post_inference(session_id, message, resp);

            if (stream_cb) {
                StreamToken ft;
                ft.session_id = session_id;
                ft.token      = "";
                ft.is_final   = true;
                ft.feeling    = chat_resp.feeling;
                stream_cb(ft);
            }

            {
                std::unique_lock lock(session_mutex_);
                auto* session = sessions_->get(session_id);
                if (session) {
                    session->add_user_turn(message);
                    session->add_assistant_turn(resp.response);
                }
            }

            return CardinalResult<ChatResponse>::success(chat_resp);
        }
        catch (const std::exception& e) {
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("Internal error: ") + e.what());
        }
    }

    // =========================================================================
    // agent
    // =========================================================================

    CardinalResult<ChatResponse>
        CardinalAPI::agent(const std::string& session_id,
                           const std::string& goal,
                           int                max_iterations)
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ChatResponse>::failure(
                check.status, check.error_message);

        if (goal.empty())
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INVALID_INPUT, "Goal cannot be empty");

        if (!config_->agent.enabled)
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INVALID_INPUT,
                "Agent mode disabled in config (agent.enabled=false)");

        std::lock_guard<std::mutex> inf_lock(inference_mutex_);

        try {
            auto info = backend_->get_info();
            TraceBuilder trace_builder(session_id, info.name, info.model_name);
            trace_builder.record_query(goal, true, goal);

            AgentGoal agent_goal;
            agent_goal.session_id     = session_id;
            agent_goal.goal           = goal;
            agent_goal.max_iterations = max_iterations;

            AgentResult agent_result = agent_executor_->run(
                agent_goal, trace_builder, nullptr);

            auto trace = trace_builder.finalize();
            if (audit_log_) audit_log_->append(trace);

            ChatResponse chat_resp;
            chat_resp.session_id   = session_id;
            chat_resp.response     = agent_result.final_response;
            chat_resp.agent_result = agent_result;

            {
                std::unique_lock lock(session_mutex_);
                if (!sessions_->exists(session_id)) sessions_->create();
                auto* session = sessions_->get(session_id);
                if (session) {
                    session->add_user_turn("[AGENT GOAL] " + goal);
                    session->add_assistant_turn(agent_result.final_response);
                }
            }

            return CardinalResult<ChatResponse>::success(chat_resp);
        }
        catch (const std::exception& e) {
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("Agent error: ") + e.what());
        }
    }

    // =========================================================================
    // Tool management
    // =========================================================================

    CardinalVoidResult CardinalAPI::register_tools(
        const std::vector<ToolDefinition>& tools)
    {
        auto check = check_initialized();
        if (!check.ok()) return check;
        tool_registry_->register_tools(tools);
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::unregister_tool(const std::string& name) {
        auto check = check_initialized();
        if (!check.ok()) return check;
        tool_registry_->unregister_tool(name);
        return CardinalVoidResult::success();
    }

    CardinalResult<std::vector<ToolDefinition>> CardinalAPI::list_tools() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::vector<ToolDefinition>>::failure(
                check.status, check.error_message);
        return CardinalResult<std::vector<ToolDefinition>>::success(
            tool_registry_->get_enabled_tools());
    }

    // =========================================================================
    // Explainability
    // =========================================================================

    CardinalResult<std::string>
        CardinalAPI::get_trace(const std::string& inference_id) const
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::string>::failure(check.status, check.error_message);
        try {
            auto trace = audit_log_->get(inference_id);
            if (trace.inference_id.empty())
                return CardinalResult<std::string>::failure(
                    CardinalStatus::NOT_FOUND, "Trace not found: " + inference_id);
            return CardinalResult<std::string>::success(exporter_->export_json(trace));
        }
        catch (const std::exception& e) {
            return CardinalResult<std::string>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<std::string>
        CardinalAPI::export_trace(const std::string& inference_id) const
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::string>::failure(check.status, check.error_message);
        try {
            auto trace = audit_log_->get(inference_id);
            if (trace.inference_id.empty())
                return CardinalResult<std::string>::failure(
                    CardinalStatus::NOT_FOUND, "Trace not found: " + inference_id);
            return CardinalResult<std::string>::success(exporter_->export_to_file(trace));
        }
        catch (const std::exception& e) {
            return CardinalResult<std::string>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<bool>
        CardinalAPI::verify_trace(const std::string& inference_id) const
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<bool>::failure(check.status, check.error_message);
        return CardinalResult<bool>::success(audit_log_->verify(inference_id));
    }

    CardinalResult<std::string> CardinalAPI::get_public_key() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::string>::failure(check.status, check.error_message);
        return CardinalResult<std::string>::success(audit_log_->public_key_pem());
    }

    // =========================================================================
    // Memory / verifier / export / settings / health
    // =========================================================================

    CardinalResult<SystemStats> CardinalAPI::get_stats() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<SystemStats>::failure(check.status, check.error_message);
        try {
            SystemStats stats;
            auto ep_stats   = storage_->stats();
            auto rule_stats = rule_store_->stats();

            stats.memory.total_episodes         = ep_stats.total_episodes;
            stats.memory.migrated_episodes      = ep_stats.migrated_episodes;
            stats.memory.high_conf_episodes     = ep_stats.high_conf_episodes;
            stats.memory.rule_candidate_count   = ep_stats.rule_candidate_count;
            stats.memory.avg_episode_confidence = ep_stats.avg_confidence;
            stats.memory.total_rules            = rule_stats.total_rules;
            stats.memory.active_rules           = rule_stats.active_rules;
            stats.memory.avg_rule_confidence    = rule_stats.avg_confidence;
            stats.memory.index_size             = retriever_->index_size();
            stats.memory.vocabulary_size        = retriever_->vocabulary_size();
            stats.memory.index_ready            = retriever_->index_ready();

            stats.verifier.total_checks           = checker_->total_checks();
            stats.verifier.total_rules_extracted  = checker_->total_rules_extracted();
            stats.verifier.total_contradictions   = checker_->total_contradictions();
            stats.verifier.total_resolved         = checker_->total_resolved();
            stats.verifier.total_flagged          = checker_->total_flagged();
            stats.verifier.total_maintenance_runs = checker_->total_maintenance_runs();

            stats.uptime_seconds = uptime_string();
            stats.version        = "1.6.0";
            stats.initialized    = true;
            return CardinalResult<SystemStats>::success(stats);
        }
        catch (const std::exception& e) {
            return CardinalResult<SystemStats>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<std::vector<RuleInfo>> CardinalAPI::get_rules() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::vector<RuleInfo>>::failure(
                check.status, check.error_message);
        try {
            auto rules = rule_store_->get_all();
            std::vector<RuleInfo> result;
            for (const auto& r : rules) result.push_back(to_rule_info(r));
            return CardinalResult<std::vector<RuleInfo>>::success(result);
        }
        catch (const std::exception& e) {
            return CardinalResult<std::vector<RuleInfo>>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<std::vector<EpisodeInfo>>
        CardinalAPI::get_episodes(const std::string& keyword,
                                   const std::string& domain,
                                   float min_conf, int max_results) const
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::vector<EpisodeInfo>>::failure(
                check.status, check.error_message);
        try {
            EpisodeQuery q;
            q.keyword        = keyword;
            q.domain         = domain;
            q.min_confidence = min_conf;
            q.max_results    = max_results;
            q.recent_first   = true;
            auto episodes    = storage_->query(q);
            std::vector<EpisodeInfo> result;
            for (const auto& ep : episodes)
                result.push_back(to_episode_info(ep));
            return CardinalResult<std::vector<EpisodeInfo>>::success(result);
        }
        catch (const std::exception& e) {
            return CardinalResult<std::vector<EpisodeInfo>>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<ScanResult> CardinalAPI::run_scan() {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ScanResult>::failure(check.status, check.error_message);
        try {
            int rb = checker_->total_resolved(), fb = checker_->total_flagged();
            auto contradictions = checker_->run_full_scan();
            ScanResult result;
            result.total_contradictions = static_cast<int>(contradictions.size());
            result.resolved = checker_->total_resolved() - rb;
            result.flagged  = checker_->total_flagged() - fb;
            result.skipped  = result.total_contradictions - result.resolved - result.flagged;
            if (result.resolved > 0) rule_store_->save();
            return CardinalResult<ScanResult>::success(result);
        }
        catch (const std::exception& e) {
            return CardinalResult<ScanResult>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalVoidResult CardinalAPI::run_maintenance() {
        auto check = check_initialized();
        if (!check.ok()) return check;
        try { checker_->run_maintenance(); return CardinalVoidResult::success(); }
        catch (const std::exception& e) {
            return CardinalVoidResult::failure(CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<ExportInfo>
        CardinalAPI::export_training_data(const ExportRequest& request)
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ExportInfo>::failure(check.status, check.error_message);
        if (request.output_path.empty())
            return CardinalResult<ExportInfo>::failure(
                CardinalStatus::INVALID_INPUT, "output_path cannot be empty");
        try {
            ExportFilter filter;
            filter.min_confidence = request.min_confidence;
            filter.domain         = request.domain;
            filter.max_examples   = request.max_examples;
            filter.include_rules  = request.include_rules;
            auto stats = training_exporter_->export_to_file(request.output_path, filter);
            ExportInfo info;
            info.episodes_exported = stats.episodes_exported;
            info.rules_exported    = stats.rules_exported;
            info.total_exported    = stats.total_exported;
            info.avg_confidence    = stats.avg_confidence;
            info.output_path       = stats.output_path;
            info.timestamp         = stats.timestamp;
            return CardinalResult<ExportInfo>::success(info);
        }
        catch (const std::exception& e) {
            return CardinalResult<ExportInfo>::failure(
                CardinalStatus::EXPORT_FAILED, e.what());
        }
    }

    CardinalResult<ExportInfo>
        CardinalAPI::export_dry_run(const ExportRequest& request) const
    {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ExportInfo>::failure(check.status, check.error_message);
        try {
            ExportFilter filter;
            filter.min_confidence = request.min_confidence;
            filter.domain         = request.domain;
            filter.max_examples   = request.max_examples;
            filter.include_rules  = request.include_rules;
            auto stats = training_exporter_->dry_run(filter);
            ExportInfo info;
            info.episodes_exported = stats.episodes_exported;
            info.rules_exported    = stats.rules_exported;
            info.total_exported    = stats.total_exported;
            info.avg_confidence    = stats.avg_confidence;
            info.output_path       = "(dry run)";
            info.timestamp         = stats.timestamp;
            return CardinalResult<ExportInfo>::success(info);
        }
        catch (const std::exception& e) {
            return CardinalResult<ExportInfo>::failure(
                CardinalStatus::INTERNAL_ERROR, e.what());
        }
    }

    CardinalResult<CardinalSettings> CardinalAPI::get_settings() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<CardinalSettings>::failure(
                check.status, check.error_message);
        return CardinalResult<CardinalSettings>::success(settings_->get());
    }

    CardinalVoidResult CardinalAPI::update_settings(const CardinalSettings& s) {
        auto check = check_initialized(); if (!check.ok()) return check;
        return settings_->update(s);
    }
    CardinalVoidResult CardinalAPI::set_setting(const std::string& k,
                                                 const std::string& v) {
        auto check = check_initialized(); if (!check.ok()) return check;
        return settings_->set(k, v);
    }
    CardinalVoidResult CardinalAPI::reset_settings() {
        auto check = check_initialized(); if (!check.ok()) return check;
        return settings_->reset();
    }

    CardinalVoidResult CardinalAPI::health_check() const {
        if (!initialized_.load())
            return CardinalVoidResult::failure(
                CardinalStatus::NOT_INITIALIZED, "Not initialized");
        if (shutting_down_.load())
            return CardinalVoidResult::failure(
                CardinalStatus::SHUTDOWN, "Shutting down");
        return CardinalVoidResult::success();
    }

    std::string CardinalAPI::uptime_string() const {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_).count();
        std::ostringstream oss;
        oss << std::setfill('0')
            << std::setw(2) << elapsed / 3600 << "h "
            << std::setw(2) << (elapsed % 3600) / 60 << "m "
            << std::setw(2) << elapsed % 60 << "s";
        return oss.str();
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    ChatResponse CardinalAPI::run_post_inference(
        const std::string&       session_id,
        const std::string&       user_message,
        const InferenceResponse& resp)
    {
        std::string ep_id = episodic_->log_episode(
            user_message, resp.response, resp.feeling,
            resp.metrics.pass1_tokens_generated,
            resp.metrics.pass2_tokens_generated,
            resp.metrics.total_duration.count());

        EpisodeRecord record;
        record.id               = ep_id;
        record.timestamp        = JsonParser::current_timestamp();
        record.user_message     = user_message;
        record.response_summary = resp.response;
        record.confidence       = resp.feeling.confidence;
        record.reasoning_type   = resp.feeling.reasoning_type;
        record.reasoning_domain = resp.feeling.reasoning_domain;
        record.contradiction    = resp.feeling.contradiction_flag;
        record.uncertainty      = resp.feeling.uncertainty_flag;
        record.rule_candidate   = resp.feeling.rule_candidate_signal;
        record.pass1_tokens     = resp.metrics.pass1_tokens_generated;
        record.pass2_tokens     = resp.metrics.pass2_tokens_generated;
        record.total_ms         = static_cast<int>(resp.metrics.total_duration.count());

        storage_->insert_episode(record);
        retriever_->notify_new_episode(ep_id);

        ConsistencyCheckInput ci;
        ci.feeling       = resp.feeling;
        ci.user_message  = user_message;
        ci.response_text = resp.response;
        ci.episode_id    = ep_id;

        auto cr = checker_->check(ci);
        if (cr.rule_committed && !cr.committed_rule_id.empty())
            storage_->set_extracted_rule_id(ep_id, cr.committed_rule_id);
        if (rule_store_->is_dirty()) rule_store_->save();

        if (self_improvement_) {
            self_improvement_->on_inference(
                resp.feeling.reasoning_domain,
                resp.feeling.reasoning_type,
                resp.feeling.confidence,
                resp.feeling.contradiction_flag,
                resp.feeling.uncertainty_flag,
                cr.rule_committed);
        }

        if (scheduler_) scheduler_->on_inference();

        ChatResponse chat_resp;
        chat_resp.session_id              = session_id;
        chat_resp.response                = resp.response;
        chat_resp.feeling                 = to_feeling_info(resp.feeling);
        chat_resp.episode_id              = ep_id;
        chat_resp.rule_committed          = cr.rule_committed;
        chat_resp.committed_rule_id       = cr.committed_rule_id;
        chat_resp.contradictions_found    = static_cast<int>(cr.contradictions.size());
        chat_resp.contradictions_resolved = cr.contradictions_resolved;
        chat_resp.contradictions_flagged  = cr.contradictions_flagged;
        chat_resp.pass1_tokens            = resp.metrics.pass1_tokens_generated;
        chat_resp.pass2_tokens            = resp.metrics.pass2_tokens_generated;
        chat_resp.total_ms                = static_cast<int>(resp.metrics.total_duration.count());
        chat_resp.inference_id            = resp.trace.inference_id;

        return chat_resp;
    }

    FeelingInfo CardinalAPI::to_feeling_info(const FeelingOutput& f) {
        FeelingInfo info;
        info.confidence         = f.confidence;
        info.reasoning_type     = f.reasoning_type;
        info.reasoning_domain   = f.reasoning_domain;
        info.uncertainty_flag   = f.uncertainty_flag;
        info.contradiction_flag = f.contradiction_flag;
        info.rule_candidate     = f.rule_candidate_signal;
        return info;
    }

    RuleInfo CardinalAPI::to_rule_info(const Rule& r) const {
        RuleInfo info;
        info.id             = r.id;
        info.domain         = r.domain;
        info.condition      = r.condition;
        info.consequence    = r.consequence;
        info.confidence     = r.confidence;
        info.trigger_count  = r.trigger_count;
        info.episode_id     = r.episode_id;
        info.reasoning_type = r.reasoning_type;
        info.created_at     = r.created_at;
        info.updated_at     = r.updated_at;
        info.has_provenance = r.has_provenance();
        return info;
    }

    EpisodeInfo CardinalAPI::to_episode_info(const EpisodeRecord& ep) const {
        EpisodeInfo info;
        info.id                = ep.id;
        info.timestamp         = ep.timestamp;
        info.user_message      = ep.user_message;
        info.response_summary  = ep.response_summary;
        info.confidence        = ep.confidence;
        info.reasoning_type    = ep.reasoning_type;
        info.reasoning_domain  = ep.reasoning_domain;
        info.contradiction     = ep.contradiction;
        info.uncertainty       = ep.uncertainty;
        info.rule_candidate    = ep.rule_candidate;
        info.extracted_rule_id = ep.extracted_rule_id;
        info.pass1_tokens      = ep.pass1_tokens;
        info.pass2_tokens      = ep.pass2_tokens;
        info.total_ms          = ep.total_ms;
        return info;
    }

    CardinalVoidResult CardinalAPI::check_initialized() const {
        if (!initialized_.load())
            return CardinalVoidResult::failure(
                CardinalStatus::NOT_INITIALIZED,
                "CardinalAPI not initialized — call init() first");
        return CardinalVoidResult::success();
    }

    // =========================================================================
    // Self-Improvement (v1.4.0)
    // =========================================================================

    CardinalResult<SelfImprovementStatus> CardinalAPI::get_self_model_status() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<SelfImprovementStatus>::failure(
                check.status, check.error_message);
        if (!self_improvement_) {
            SelfImprovementStatus s;
            return CardinalResult<SelfImprovementStatus>::success(s);
        }
        return CardinalResult<SelfImprovementStatus>::success(
            self_improvement_->get_status());
    }

    CardinalResult<ReflectionResult> CardinalAPI::reflect() {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ReflectionResult>::failure(
                check.status, check.error_message);
        if (!self_improvement_) {
            ReflectionResult r;
            r.error_message = "Self-improvement is disabled in config";
            return CardinalResult<ReflectionResult>::success(r);
        }
        return CardinalResult<ReflectionResult>::success(
            self_improvement_->trigger_reflection());
    }

    CardinalResult<bool> CardinalAPI::trigger_training(
            const std::string& domain_hint) {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<bool>::failure(check.status, check.error_message);
        if (!self_improvement_)
            return CardinalResult<bool>::success(false);
        return CardinalResult<bool>::success(
            self_improvement_->trigger_training(domain_hint));
    }

    void CardinalAPI::on_session_boundary() {
        if (self_improvement_) self_improvement_->on_session_boundary();
    }

    // =========================================================================
    // Scheduler API (v1.5.0)
    // =========================================================================

    CardinalVoidResult CardinalAPI::check_scheduler() const {
        auto init_check = check_initialized();
        if (!init_check.ok()) return init_check;
        if (!scheduler_)
            return CardinalVoidResult::error(CardinalStatus::SCHEDULER_ERROR,
                                              "Scheduler is disabled in config");
        return CardinalVoidResult::success();
    }

    CardinalResult<SchedulerStatus> CardinalAPI::get_scheduler_status() const {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<SchedulerStatus>::error(guard.status, guard.message);
        return CardinalResult<SchedulerStatus>::success(scheduler_->get_status());
    }

    CardinalResult<std::vector<ScheduledTask>> CardinalAPI::list_tasks() const {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<std::vector<ScheduledTask>>::error(
                guard.status, guard.message);
        return CardinalResult<std::vector<ScheduledTask>>::success(
            scheduler_->list_tasks());
    }

    CardinalResult<ScheduledTask> CardinalAPI::get_task(
            const std::string& task_id) const {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<ScheduledTask>::error(guard.status, guard.message);
        auto t = scheduler_->get_task(task_id);
        if (!t)
            return CardinalResult<ScheduledTask>::error(
                CardinalStatus::NOT_FOUND, "Task not found: " + task_id);
        return CardinalResult<ScheduledTask>::success(*t);
    }

    CardinalResult<TaskParseResult> CardinalAPI::create_task(
            const std::string& nl_description, const std::string& session_id) {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<TaskParseResult>::error(guard.status, guard.message);
        try {
            auto result = scheduler_->create_task_from_nl(nl_description, session_id);
            return CardinalResult<TaskParseResult>::success(result);
        } catch (const std::exception& e) {
            return CardinalResult<TaskParseResult>::error(
                CardinalStatus::SCHEDULER_ERROR, e.what());
        }
    }

    CardinalResult<std::string> CardinalAPI::create_task_direct(
            const ScheduledTask& task) {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<std::string>::error(guard.status, guard.message);
        try {
            return CardinalResult<std::string>::success(
                scheduler_->create_task(task));
        } catch (const std::exception& e) {
            return CardinalResult<std::string>::error(
                CardinalStatus::SCHEDULER_ERROR, e.what());
        }
    }

    CardinalVoidResult CardinalAPI::update_task(const ScheduledTask& task) {
        auto guard = check_scheduler();
        if (!guard.ok()) return guard;
        if (!scheduler_->update_task(task))
            return CardinalVoidResult::error(CardinalStatus::NOT_FOUND,
                                              "Task not found: " + task.id);
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::delete_task(const std::string& task_id) {
        auto guard = check_scheduler();
        if (!guard.ok()) return guard;
        if (!scheduler_->delete_task(task_id))
            return CardinalVoidResult::error(CardinalStatus::NOT_FOUND,
                                              "Task not found: " + task_id);
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::enable_task(const std::string& task_id) {
        auto guard = check_scheduler();
        if (!guard.ok()) return guard;
        if (!scheduler_->enable_task(task_id))
            return CardinalVoidResult::error(CardinalStatus::NOT_FOUND,
                                              "Task not found: " + task_id);
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::disable_task(const std::string& task_id) {
        auto guard = check_scheduler();
        if (!guard.ok()) return guard;
        if (!scheduler_->disable_task(task_id))
            return CardinalVoidResult::error(CardinalStatus::NOT_FOUND,
                                              "Task not found: " + task_id);
        return CardinalVoidResult::success();
    }

    CardinalResult<std::string> CardinalAPI::run_task_now(
            const std::string& task_id) {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<std::string>::error(guard.status, guard.message);
        try {
            return CardinalResult<std::string>::success(
                scheduler_->run_task_now(task_id));
        } catch (const std::exception& e) {
            return CardinalResult<std::string>::error(
                CardinalStatus::SCHEDULER_ERROR, e.what());
        }
    }

    CardinalResult<std::vector<TaskRun>> CardinalAPI::get_task_history(
            const std::string& task_id, int limit) const {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<std::vector<TaskRun>>::error(
                guard.status, guard.message);
        return CardinalResult<std::vector<TaskRun>>::success(
            scheduler_->get_task_history(task_id, limit));
    }

    CardinalResult<std::vector<TaskRun>> CardinalAPI::get_recent_runs(
            int limit) const {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<std::vector<TaskRun>>::error(
                guard.status, guard.message);
        return CardinalResult<std::vector<TaskRun>>::success(
            scheduler_->get_recent_runs(limit));
    }

    CardinalResult<std::vector<TaskActionLog>> CardinalAPI::get_run_action_logs(
            const std::string& run_id) const {
        auto guard = check_scheduler();
        if (!guard.ok())
            return CardinalResult<std::vector<TaskActionLog>>::error(
                guard.status, guard.message);
        return CardinalResult<std::vector<TaskActionLog>>::success(
            scheduler_->get_run_action_logs(run_id));
    }

    // =========================================================================
    // Computer Use API (v1.5.0)
    // =========================================================================

    CardinalVoidResult CardinalAPI::check_computer_use() const {
        auto init_check = check_initialized();
        if (!init_check.ok()) return init_check;
        if (!display_detector_)
            return CardinalVoidResult::error(CardinalStatus::COMPUTER_USE_ERROR,
                                              "Computer use is disabled in config");
        return CardinalVoidResult::success();
    }

    CardinalResult<ScreenInfo> CardinalAPI::get_computer_status() const {
        auto guard = check_computer_use();
        if (!guard.ok())
            return CardinalResult<ScreenInfo>::error(guard.status, guard.message);
        return CardinalResult<ScreenInfo>::success(display_detector_->info());
    }

    CardinalResult<Screenshot> CardinalAPI::take_screenshot(
            bool analyze, const std::string& prompt) {
        auto guard = check_computer_use();
        if (!guard.ok())
            return CardinalResult<Screenshot>::error(guard.status, guard.message);
        try {
            auto s = screen_reader_->capture(analyze);
            if (analyze && !prompt.empty() && s.description.empty())
                s.description = screen_reader_->analyze(s.path, prompt);
            return CardinalResult<Screenshot>::success(s);
        } catch (const std::exception& e) {
            return CardinalResult<Screenshot>::error(
                CardinalStatus::COMPUTER_USE_ERROR, e.what());
        }
    }

    CardinalResult<std::string> CardinalAPI::computer_click(
            int x, int y, const std::string& description) {
        auto guard = check_computer_use();
        if (!guard.ok())
            return CardinalResult<std::string>::error(guard.status, guard.message);
        try {
            if (!description.empty() && (x < 0 || y < 0)) {
                auto pt = screen_reader_->find_element(description);
                if (!pt)
                    return CardinalResult<std::string>::error(
                        CardinalStatus::COMPUTER_USE_ERROR,
                        "Could not locate element: " + description);
                x = pt->x; y = pt->y;
            }
            input_controller_->mouse_click(x, y);
            return CardinalResult<std::string>::success(
                "Clicked at (" + std::to_string(x) + "," + std::to_string(y) + ")");
        } catch (const std::exception& e) {
            return CardinalResult<std::string>::error(
                CardinalStatus::COMPUTER_USE_ERROR, e.what());
        }
    }

    CardinalResult<std::string> CardinalAPI::computer_type(
            const std::string& text, const std::string& key) {
        auto guard = check_computer_use();
        if (!guard.ok())
            return CardinalResult<std::string>::error(guard.status, guard.message);
        try {
            if (!key.empty())  input_controller_->send_key(key);
            if (!text.empty()) input_controller_->type_text(text);
            return CardinalResult<std::string>::success(
                key.empty() ? "Typed: " + text : "Sent key: " + key);
        } catch (const std::exception& e) {
            return CardinalResult<std::string>::error(
                CardinalStatus::COMPUTER_USE_ERROR, e.what());
        }
    }

    CardinalResult<ShellResult> CardinalAPI::computer_shell(
            const std::string& command, int timeout_seconds) {
        auto guard = check_computer_use();
        if (!guard.ok())
            return CardinalResult<ShellResult>::error(guard.status, guard.message);
        try {
            auto result = shell_executor_->run(command, timeout_seconds);
            if (!result.success)
                return CardinalResult<ShellResult>::error(
                    CardinalStatus::COMPUTER_USE_ERROR, result.stderr_text);
            return CardinalResult<ShellResult>::success(result);
        } catch (const std::exception& e) {
            return CardinalResult<ShellResult>::error(
                CardinalStatus::COMPUTER_USE_ERROR, e.what());
        }
    }

    // =========================================================================
    // Voice API (v1.6.0)
    // =========================================================================

    CardinalVoidResult CardinalAPI::check_voice() const {
        auto init_check = check_initialized();
        if (!init_check.ok()) return init_check;
        if (!voice_loop_ || !voice_loop_->is_running())
            return CardinalVoidResult::error(CardinalStatus::VOICE_ERROR,
                                              "Voice subsystem is not active");
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::enable_voice(const std::string& input_mode_override) {
        auto init_check = check_initialized();
        if (!init_check.ok()) return init_check;

        std::lock_guard<std::mutex> lock(voice_mutex_);

        if (voice_loop_ && voice_loop_->is_running())
            return CardinalVoidResult::success();   // already active

        VoiceConfig vc = config_->voice;
        if (!input_mode_override.empty())
            vc.input_mode = voice_input_mode_from_string(input_mode_override);

        // Provide a thin chat_stream lambda so VoiceLoop doesn't depend on CardinalAPI
        VoiceChatStreamFn chat_fn =
            [this](const std::string& session_id,
                   const std::string& message,
                   std::function<bool(const std::string&, bool)> cb) -> bool
        {
            ApiStreamCallback api_cb = [&cb](const StreamToken& token) -> bool {
                return cb(token.token, token.is_final);
            };
            auto result = chat_stream(session_id, message, api_cb);
            return result.ok();
        };

        voice_loop_ = std::make_unique<VoiceLoop>(vc, std::move(chat_fn));

        if (!voice_loop_->start()) {
            voice_loop_.reset();
            return CardinalVoidResult::error(CardinalStatus::VOICE_ERROR,
                                              "VoiceLoop failed to start");
        }

        LOG_INFO("Voice subsystem enabled (mode=" +
                 voice_input_mode_to_string(vc.input_mode) + ")");
        return CardinalVoidResult::success();
    }

    CardinalVoidResult CardinalAPI::disable_voice() {
        std::lock_guard<std::mutex> lock(voice_mutex_);
        if (!voice_loop_) return CardinalVoidResult::success();
        voice_loop_->stop();
        voice_loop_.reset();
        LOG_INFO("Voice subsystem disabled");
        return CardinalVoidResult::success();
    }

    bool CardinalAPI::is_voice_active() const {
        std::lock_guard<std::mutex> lock(voice_mutex_);
        return voice_loop_ && voice_loop_->is_running();
    }

    VoiceStatus CardinalAPI::get_voice_status() const {
        std::lock_guard<std::mutex> lock(voice_mutex_);
        if (!voice_loop_) { VoiceStatus vs; vs.active = false; return vs; }
        return voice_loop_->get_status();
    }

    CardinalResult<TTSResult> CardinalAPI::voice_speak(const std::string& text) {
        {
            std::lock_guard<std::mutex> lock(voice_mutex_);
            auto vc = check_voice();
            if (!vc.ok())
                return CardinalResult<TTSResult>::error(vc.status, vc.error_message);
        }
        TTSResult tts = voice_loop_->speak(text);
        if (!tts.success)
            return CardinalResult<TTSResult>::error(CardinalStatus::VOICE_ERROR,
                                                     tts.error_message);
        return CardinalResult<TTSResult>::success(std::move(tts));
    }

    CardinalResult<TranscriptResult> CardinalAPI::voice_transcribe(const AudioChunk& audio) {
        {
            std::lock_guard<std::mutex> lock(voice_mutex_);
            auto vc = check_voice();
            if (!vc.ok())
                return CardinalResult<TranscriptResult>::error(vc.status, vc.error_message);
        }
        TranscriptResult tr = voice_loop_->transcribe(audio);
        if (!tr.success)
            return CardinalResult<TranscriptResult>::error(CardinalStatus::VOICE_ERROR,
                                                            tr.error_message);
        return CardinalResult<TranscriptResult>::success(std::move(tr));
    }

} // namespace cardinal
