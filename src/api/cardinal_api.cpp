// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - API Facade Implementation
// File: src/api/cardinal_api.cpp
//
// Changes from original:
//   - #include "core/llm_engine.h" replaced with "core/backend_factory.h"
//   - engine_ construction:
//       was: engine_ = std::make_unique<LLMEngine>(*config_); engine_->load_model();
//       now: backend_ = BackendFactory::create(*config_); backend_->load_model();
//   - InferencePipeline constructor arg: *engine_ → *backend_
//   - Startup log prints backend type and constrained-decoding capability
//   - Windows preprocessor guards removed (Linux-only build)
//   - All other logic is identical to original
// =============================================================================

#include "api/cardinal_api.h"

// Core includes — full definitions needed here
#include "utils/logger.h"
#include "utils/json_parser.h"
#include "core/backend_factory.h"      // ← replaces llm_engine.h
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

#include <sstream>
#include <iomanip>

namespace cardinal {

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    CardinalAPI::CardinalAPI() {
        // Nothing constructed here -- init() does all the work
    }

    CardinalAPI::~CardinalAPI() {
        if (initialized_.load()) {
            shutdown();
        }
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    CardinalVoidResult CardinalAPI::init(const std::string& config_path) {
        std::lock_guard<std::mutex> lock(api_mutex_);

        if (initialized_.load()) {
            return CardinalVoidResult::failure(
                CardinalStatus::ALREADY_INITIALIZED,
                "CardinalAPI::init() called twice");
        }

        try {
            LOG_INFO("CardinalAPI initializing from: " + config_path);
            start_time_ = std::chrono::steady_clock::now();

            // -- Config --
            config_ = std::make_unique<CardinalConfig>(
                ConfigLoader::load(config_path));

            // -- Memory layer --
            rule_store_      = std::make_unique<RuleStore>(*config_);
            knowledge_graph_ = std::make_unique<KnowledgeGraph>(*config_);
            episodic_        = std::make_unique<EpisodicMemory>(*config_);
            storage_         = std::make_unique<EpisodicStorage>(*config_);

            rule_store_->load();
            knowledge_graph_->load();
            episodic_->open();
            storage_->open();

            // -- Retriever --
            retriever_ = std::make_unique<EpisodicRetriever>(*config_, *storage_);
            retriever_->init();

            // -- Verifier pipeline --
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

            // -- LLM Backend --
            // BackendFactory reads config.backend.type and constructs the right
            // implementation. This is the only place in the codebase that
            // knows concrete backend types exist.
            backend_ = BackendFactory::create(*config_);
            backend_->load_model();

            {
                auto info = backend_->get_info();
                LOG_INFO("Backend: " + info.name +
                         " | model: " + info.model_name +
                         " | gpu_layers: " + std::to_string(info.gpu_layers) +
                         " | constrained_decoding: " +
                         (backend_->supports_constrained_decoding()
                              ? "native (GBNF)"
                              : "retry loop (JSON schema)"));
            }

            pipeline_ = std::make_unique<InferencePipeline>(*config_, *backend_);
            pipeline_->set_retriever(retriever_.get());

            // -- API layer --
            exporter_ = std::make_unique<TrainingExporter>(
                *config_, *storage_, *rule_store_);

            settings_ = std::make_unique<SettingsManager>(
                *config_, *retriever_, *pipeline_);

            sessions_ = std::make_unique<SessionManager>();

            // Create a default session
            sessions_->create();

            initialized_.store(true);

            LOG_INFO("CardinalAPI initialized -- episodes=" +
                std::to_string(storage_->count()) +
                " rules=" + std::to_string(rule_store_->size()) +
                " index=" + std::to_string(retriever_->index_size()));

            return CardinalVoidResult::success();
        }
        catch (const std::exception& e) {
            LOG_FATAL("CardinalAPI::init failed: " + std::string(e.what()));
            return CardinalVoidResult::failure(
                CardinalStatus::CONFIG_ERROR,
                std::string("Initialization failed: ") + e.what());
        }
    }

    CardinalVoidResult CardinalAPI::shutdown() {
        std::lock_guard<std::mutex> lock(api_mutex_);

        if (!initialized_.load()) return CardinalVoidResult::success();

        shutting_down_.store(true);

        try {
            LOG_INFO("CardinalAPI shutting down...");

            if (checker_)    checker_->run_maintenance();
            if (rule_store_) rule_store_->save();
            if (sessions_)   sessions_->destroy_all();
            if (storage_)    storage_->close();

            initialized_.store(false);
            shutting_down_.store(false);

            LOG_INFO("CardinalAPI shutdown complete");
            return CardinalVoidResult::success();
        }
        catch (const std::exception& e) {
            LOG_WARN("CardinalAPI::shutdown error: " + std::string(e.what()));
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
            return CardinalResult<std::string>::failure(
                check.status, check.error_message);

        std::unique_lock<std::shared_mutex> lock(session_mutex_);

        if (!session_id.empty() && sessions_->exists(session_id)) {
            return CardinalResult<std::string>::success(session_id);
        }

        std::string new_id = sessions_->create();
        return CardinalResult<std::string>::success(new_id);
    }

    CardinalVoidResult
        CardinalAPI::destroy_session(const std::string& session_id) {
        auto check = check_initialized();
        if (!check.ok()) return check;

        std::unique_lock<std::shared_mutex> lock(session_mutex_);

        if (!sessions_->destroy(session_id)) {
            return CardinalVoidResult::failure(
                CardinalStatus::SESSION_NOT_FOUND,
                "Session not found: " + session_id);
        }
        return CardinalVoidResult::success();
    }

    CardinalVoidResult
        CardinalAPI::reset_session(const std::string& session_id) {
        auto check = check_initialized();
        if (!check.ok()) return check;

        std::unique_lock<std::shared_mutex> lock(session_mutex_);

        auto* session = sessions_->get(session_id);
        if (!session) {
            return CardinalVoidResult::failure(
                CardinalStatus::SESSION_NOT_FOUND,
                "Session not found: " + session_id);
        }
        session->reset();
        return CardinalVoidResult::success();
    }

    CardinalResult<SessionInfo>
        CardinalAPI::get_session(const std::string& session_id) const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<SessionInfo>::failure(
                check.status, check.error_message);

        std::shared_lock<std::shared_mutex> lock(session_mutex_);

        const auto* session = sessions_->get(session_id);
        if (!session) {
            return CardinalResult<SessionInfo>::failure(
                CardinalStatus::SESSION_NOT_FOUND,
                "Session not found: " + session_id);
        }
        return CardinalResult<SessionInfo>::success(session->to_session_info());
    }

    CardinalResult<std::vector<std::string>>
        CardinalAPI::list_sessions() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<std::vector<std::string>>::failure(
                check.status, check.error_message);

        std::shared_lock<std::shared_mutex> lock(session_mutex_);
        return CardinalResult<std::vector<std::string>>::success(
            sessions_->all_ids());
    }

    // =========================================================================
    // Inference
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

        if (message.empty()) {
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INVALID_INPUT,
                "Message cannot be empty");
        }

        if (shutting_down_.load()) {
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::SHUTDOWN,
                "API is shutting down");
        }

        {
            std::unique_lock<std::shared_mutex> lock(session_mutex_);
            if (!sessions_->exists(session_id)) {
                sessions_->create();
            }
        }

        std::lock_guard<std::mutex> inf_lock(inference_mutex_);

        try {
            std::vector<ChatMessage> history;
            {
                std::shared_lock<std::shared_mutex> lock(session_mutex_);
                auto* session = sessions_->get(session_id);
                if (session) history = session->get_history();
            }

            InferenceRequest req;
            req.user_message   = message;
            req.history        = history;
            req.stream_response = (stream_cb != nullptr);

            StreamCallback core_cb = nullptr;
            if (stream_cb) {
                core_cb = [&stream_cb, &session_id](
                    const std::string& token) -> bool {
                        StreamToken st;
                        st.session_id = session_id;
                        st.token      = token;
                        st.is_final   = false;
                        return stream_cb(st);
                    };
            }

            auto resp = pipeline_->run(req, core_cb);

            if (!resp.success) {
                return CardinalResult<ChatResponse>::failure(
                    CardinalStatus::INFERENCE_FAILED,
                    resp.error_message.empty()
                        ? "Inference failed"
                        : resp.error_message);
            }

            ChatResponse chat_resp = run_post_inference(
                session_id, message, resp);

            if (stream_cb) {
                StreamToken final_token;
                final_token.session_id = session_id;
                final_token.token      = "";
                final_token.is_final   = true;
                final_token.feeling    = chat_resp.feeling;
                stream_cb(final_token);
            }

            {
                std::unique_lock<std::shared_mutex> lock(session_mutex_);
                auto* session = sessions_->get(session_id);
                if (session) {
                    session->add_user_turn(message);
                    session->add_assistant_turn(resp.response);
                }
            }

            return CardinalResult<ChatResponse>::success(chat_resp);
        }
        catch (const std::exception& e) {
            LOG_WARN("CardinalAPI::chat_stream exception: " +
                     std::string(e.what()));
            return CardinalResult<ChatResponse>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("Internal error: ") + e.what());
        }
    }

    // =========================================================================
    // Memory / verifier
    // =========================================================================

    CardinalResult<SystemStats> CardinalAPI::get_stats() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<SystemStats>::failure(
                check.status, check.error_message);

        try {
            SystemStats stats;

            auto ep_stats   = storage_->stats();
            auto rule_stats = rule_store_->stats();

            stats.memory.total_episodes       = ep_stats.total_episodes;
            stats.memory.migrated_episodes    = ep_stats.migrated_episodes;
            stats.memory.high_conf_episodes   = ep_stats.high_conf_episodes;
            stats.memory.rule_candidate_count = ep_stats.rule_candidate_count;
            stats.memory.avg_episode_confidence = ep_stats.avg_confidence;
            stats.memory.total_rules          = rule_stats.total_rules;
            stats.memory.active_rules         = rule_stats.active_rules;
            stats.memory.avg_rule_confidence  = rule_stats.avg_confidence;
            stats.memory.index_size           = retriever_->index_size();
            stats.memory.vocabulary_size      = retriever_->vocabulary_size();
            stats.memory.index_ready          = retriever_->index_ready();

            stats.verifier.total_checks            = checker_->total_checks();
            stats.verifier.total_rules_extracted   = checker_->total_rules_extracted();
            stats.verifier.total_contradictions    = checker_->total_contradictions();
            stats.verifier.total_resolved          = checker_->total_resolved();
            stats.verifier.total_flagged           = checker_->total_flagged();
            stats.verifier.total_maintenance_runs  = checker_->total_maintenance_runs();

            stats.uptime_seconds = uptime_string();
            stats.version        = "0.6.0";
            stats.initialized    = true;

            return CardinalResult<SystemStats>::success(stats);
        }
        catch (const std::exception& e) {
            return CardinalResult<SystemStats>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("Stats error: ") + e.what());
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
            result.reserve(rules.size());
            for (const auto& r : rules)
                result.push_back(to_rule_info(r));
            return CardinalResult<std::vector<RuleInfo>>::success(result);
        }
        catch (const std::exception& e) {
            return CardinalResult<std::vector<RuleInfo>>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("get_rules error: ") + e.what());
        }
    }

    CardinalResult<std::vector<EpisodeInfo>>
        CardinalAPI::get_episodes(const std::string& keyword,
                                   const std::string& domain,
                                   float              min_conf,
                                   int                max_results) const
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

            auto episodes = storage_->query(q);
            std::vector<EpisodeInfo> result;
            result.reserve(episodes.size());
            for (const auto& ep : episodes)
                result.push_back(to_episode_info(ep));
            return CardinalResult<std::vector<EpisodeInfo>>::success(result);
        }
        catch (const std::exception& e) {
            return CardinalResult<std::vector<EpisodeInfo>>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("get_episodes error: ") + e.what());
        }
    }

    CardinalResult<ScanResult> CardinalAPI::run_scan() {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ScanResult>::failure(
                check.status, check.error_message);

        try {
            int resolved_before = checker_->total_resolved();
            int flagged_before  = checker_->total_flagged();

            auto contradictions = checker_->run_full_scan();

            ScanResult result;
            result.total_contradictions = static_cast<int>(contradictions.size());
            result.resolved = checker_->total_resolved() - resolved_before;
            result.flagged  = checker_->total_flagged() - flagged_before;
            result.skipped  = result.total_contradictions
                              - result.resolved - result.flagged;

            if (result.resolved > 0) rule_store_->save();

            return CardinalResult<ScanResult>::success(result);
        }
        catch (const std::exception& e) {
            return CardinalResult<ScanResult>::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("run_scan error: ") + e.what());
        }
    }

    CardinalVoidResult CardinalAPI::run_maintenance() {
        auto check = check_initialized();
        if (!check.ok()) return check;

        try {
            checker_->run_maintenance();
            return CardinalVoidResult::success();
        }
        catch (const std::exception& e) {
            return CardinalVoidResult::failure(
                CardinalStatus::INTERNAL_ERROR,
                std::string("run_maintenance error: ") + e.what());
        }
    }

    // =========================================================================
    // Training export
    // =========================================================================

    CardinalResult<ExportInfo>
        CardinalAPI::export_training_data(const ExportRequest& request) {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ExportInfo>::failure(
                check.status, check.error_message);

        if (request.output_path.empty()) {
            return CardinalResult<ExportInfo>::failure(
                CardinalStatus::INVALID_INPUT,
                "output_path cannot be empty");
        }

        try {
            ExportFilter filter;
            filter.min_confidence = request.min_confidence;
            filter.domain         = request.domain;
            filter.max_examples   = request.max_examples;
            filter.include_rules  = request.include_rules;

            auto stats = exporter_->export_to_file(request.output_path, filter);

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
                CardinalStatus::EXPORT_FAILED,
                std::string("Export failed: ") + e.what());
        }
    }

    CardinalResult<ExportInfo>
        CardinalAPI::export_dry_run(const ExportRequest& request) const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<ExportInfo>::failure(
                check.status, check.error_message);

        try {
            ExportFilter filter;
            filter.min_confidence = request.min_confidence;
            filter.domain         = request.domain;
            filter.max_examples   = request.max_examples;
            filter.include_rules  = request.include_rules;

            auto stats = exporter_->dry_run(filter);

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
                CardinalStatus::INTERNAL_ERROR,
                std::string("Dry run error: ") + e.what());
        }
    }

    // =========================================================================
    // Settings
    // =========================================================================

    CardinalResult<CardinalSettings> CardinalAPI::get_settings() const {
        auto check = check_initialized();
        if (!check.ok())
            return CardinalResult<CardinalSettings>::failure(
                check.status, check.error_message);
        return CardinalResult<CardinalSettings>::success(settings_->get());
    }

    CardinalVoidResult
        CardinalAPI::update_settings(const CardinalSettings& settings) {
        auto check = check_initialized();
        if (!check.ok()) return check;
        return settings_->update(settings);
    }

    CardinalVoidResult
        CardinalAPI::set_setting(const std::string& key,
                                  const std::string& value) {
        auto check = check_initialized();
        if (!check.ok()) return check;
        return settings_->set(key, value);
    }

    CardinalVoidResult CardinalAPI::reset_settings() {
        auto check = check_initialized();
        if (!check.ok()) return check;
        return settings_->reset();
    }

    // =========================================================================
    // Health
    // =========================================================================

    CardinalVoidResult CardinalAPI::health_check() const {
        if (!initialized_.load()) {
            return CardinalVoidResult::failure(
                CardinalStatus::NOT_INITIALIZED,
                "CardinalAPI not initialized");
        }
        if (shutting_down_.load()) {
            return CardinalVoidResult::failure(
                CardinalStatus::SHUTDOWN,
                "CardinalAPI is shutting down");
        }
        return CardinalVoidResult::success();
    }

    std::string CardinalAPI::uptime_string() const {
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - start_time_).count();

        long long h = elapsed / 3600;
        long long m = (elapsed % 3600) / 60;
        long long s = elapsed % 60;

        std::ostringstream oss;
        oss << std::setfill('0')
            << std::setw(2) << h << "h "
            << std::setw(2) << m << "m "
            << std::setw(2) << s << "s";
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
            user_message,
            resp.response,
            resp.feeling,
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

        if (cr.rule_committed && !cr.committed_rule_id.empty()) {
            storage_->set_extracted_rule_id(ep_id, cr.committed_rule_id);
        }

        if (rule_store_->is_dirty()) {
            rule_store_->save();
        }

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

        LOG_DEBUG("CardinalAPI: ep=" + ep_id +
                  " rule_committed=" + (cr.rule_committed ? "yes" : "no") +
                  " contradictions=" + std::to_string(cr.contradictions.size()));

        return chat_resp;
    }

    FeelingInfo CardinalAPI::to_feeling_info(const FeelingOutput& f) const {
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
        info.id                  = ep.id;
        info.timestamp           = ep.timestamp;
        info.user_message        = ep.user_message;
        info.response_summary    = ep.response_summary;
        info.confidence          = ep.confidence;
        info.reasoning_type      = ep.reasoning_type;
        info.reasoning_domain    = ep.reasoning_domain;
        info.contradiction       = ep.contradiction;
        info.uncertainty         = ep.uncertainty;
        info.rule_candidate      = ep.rule_candidate;
        info.extracted_rule_id   = ep.extracted_rule_id;
        info.pass1_tokens        = ep.pass1_tokens;
        info.pass2_tokens        = ep.pass2_tokens;
        info.total_ms            = ep.total_ms;
        return info;
    }

    CardinalVoidResult CardinalAPI::check_initialized() const {
        if (!initialized_.load()) {
            return CardinalVoidResult::failure(
                CardinalStatus::NOT_INITIALIZED,
                "CardinalAPI not initialized -- call init() first");
        }
        return CardinalVoidResult::success();
    }

} // namespace cardinal
