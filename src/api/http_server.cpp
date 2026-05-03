// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - HTTP Server Implementation
// File: src/api/http_server.cpp
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

// httplib must be included before any Windows headers that define ERROR
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include "api/http_server.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

namespace cardinal {

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    HttpServer::HttpServer(CardinalAPI& api, const CardinalConfig& config)
        : api_(api)
        , config_(config)
        , server_(std::make_unique<httplib::Server>())
        , host_(config.api.host)
        , port_(config.api.port)
        , auth_enabled_(config.api.auth_enabled)
        , api_key_(config.api.api_key)
    {
        // Set request size limit
        server_->set_payload_max_length(
            static_cast<size_t>(config.api.max_request_size_kb) * 1024);

        // CORS headers -- needed for TypeScript browser clients
        server_->set_default_headers({
            { "Access-Control-Allow-Origin",  "*" },
            { "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS" },
            { "Access-Control-Allow-Headers",
              "Content-Type, Authorization, Accept" }
            });

        // Handle OPTIONS preflight for CORS
        server_->Options(".*", [](const httplib::Request&,
            httplib::Response& res) {
                res.status = 204;
            });

        register_routes();
        LOG_INFO("HttpServer configured on " + host_ + ":" + std::to_string(port_));
    }

    HttpServer::~HttpServer() {
        if (running_.load()) stop();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool HttpServer::start() {
        LOG_INFO("HttpServer starting on " + host_ + ":" + std::to_string(port_));
        running_.store(true);

        bool ok = server_->listen(host_.c_str(), port_);

        running_.store(false);

        if (!ok) {
            LOG_WARN("HttpServer failed to bind on " +
                host_ + ":" + std::to_string(port_));
        }
        else {
            LOG_INFO("HttpServer stopped");
        }

        return ok;
    }

    void HttpServer::stop() {
        LOG_INFO("HttpServer stopping...");
        server_->stop();
        running_.store(false);
    }

    // =========================================================================
    // Route registration
    // =========================================================================

    void HttpServer::register_routes() {
        // Health -- always public
        server_->Get("/api/health",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_health(req, res);
            });

        // Chat -- SSE streaming or regular JSON
        server_->Post("/api/chat",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_chat(req, res);
            });

        // Session management
        server_->Post("/api/sessions",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_create_session(req, res);
            });

        server_->Delete(R"(/api/sessions/(.+))",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_destroy_session(req, res);
            });

        server_->Post(R"(/api/sessions/(.+)/reset)",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_reset_session(req, res);
            });

        server_->Post("/api/reset",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_reset(req, res);
            });

        // Memory
        server_->Get("/api/stats",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_stats(req, res);
            });

        server_->Get("/api/rules",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_rules(req, res);
            });

        server_->Get("/api/episodes",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_episodes(req, res);
            });

        server_->Post("/api/scan",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_scan(req, res);
            });

        server_->Post("/api/maintenance",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_maintenance(req, res);
            });

        // Settings
        server_->Get("/api/settings",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_get_settings(req, res);
            });

        server_->Post("/api/settings",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_post_settings(req, res);
            });

        // Export
        server_->Post("/api/export",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_export(req, res);
            });
    }

    // =========================================================================
    // Auth middleware
    // =========================================================================

    bool HttpServer::check_auth(const httplib::Request& req,
        httplib::Response& res) const {
        if (!auth_enabled_) return true;

        auto auth_header = req.get_header_value("Authorization");
        if (auth_header.empty()) {
            send_error(res, 401, CardinalStatus::AUTH_FAILED,
                "Missing Authorization header");
            return false;
        }

        // Expect: "Bearer <api_key>"
        const std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) != prefix) {
            send_error(res, 401, CardinalStatus::AUTH_FAILED,
                "Invalid Authorization format -- use: Bearer <api_key>");
            return false;
        }

        std::string provided_key = auth_header.substr(prefix.size());
        if (provided_key != api_key_) {
            send_error(res, 401, CardinalStatus::AUTH_FAILED,
                "Invalid API key");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Route handlers
    // =========================================================================

    void HttpServer::handle_health(const httplib::Request&,
        httplib::Response& res) {
        auto result = api_.health_check();
        if (result.ok()) {
            json j;
            j["status"] = "ok";
            j["uptime"] = api_.uptime_string();
            j["version"] = "0.6.0";
            send_ok(res, j.dump());
        }
        else {
            send_error(res, 503, result.status, result.error_message);
        }
    }

    // -------------------------------------------------------------------------
    // handle_chat
    // Supports two modes:
    //   Regular:   Accept: application/json  -> returns full ChatResponse JSON
    //   Streaming: Accept: text/event-stream -> SSE token stream
    //
    // Request body (JSON):
    //   { "session_id": "...", "message": "..." }
    //   session_id is optional -- defaults to "default"
    // -------------------------------------------------------------------------
    void HttpServer::handle_chat(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        // Parse request body
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT,
                "Invalid JSON body");
            return;
        }

        std::string session_id = "default";
        std::string message;

        if (body.contains("session_id") && body["session_id"].is_string())
            session_id = body["session_id"].get<std::string>();
        if (body.contains("message") && body["message"].is_string())
            message = body["message"].get<std::string>();

        if (message.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT,
                "message field is required and cannot be empty");
            return;
        }

        // Check if client wants SSE streaming
        bool wants_stream = false;
        auto accept = req.get_header_value("Accept");
        if (accept.find("text/event-stream") != std::string::npos) {
            wants_stream = true;
        }
        // Also check explicit stream flag in body
        if (body.contains("stream") && body["stream"].is_boolean()) {
            wants_stream = body["stream"].get<bool>();
        }

        if (wants_stream && config_.api.stream_enabled) {
            // SSE streaming response
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");

            // Accumulate SSE data to send as chunked response
            // cpp-httplib doesn't support true SSE push natively
            // so we collect tokens then send as one chunked body
            std::ostringstream sse_body;

            ApiStreamCallback cb = [&sse_body](const StreamToken& token) -> bool {
                json t;
                t["token"] = token.token;
                t["is_final"] = token.is_final;
                if (token.is_final) {
                    // Include feeling on final token
                    json f;
                    f["confidence"] = token.feeling.confidence;
                    f["reasoning_type"] = token.feeling.reasoning_type;
                    f["reasoning_domain"] = token.feeling.reasoning_domain;
                    f["uncertainty_flag"] = token.feeling.uncertainty_flag;
                    f["contradiction_flag"] = token.feeling.contradiction_flag;
                    f["rule_candidate"] = token.feeling.rule_candidate;
                    t["feeling"] = f;
                }
                sse_body << sse_event(t.dump());
                return true;
                };

            auto result = api_.chat_stream(session_id, message, cb);

            if (!result.ok()) {
                sse_body << sse_event(
                    error_json(result.status, result.error_message));
            }

            res.status = 200;
            res.set_content(sse_body.str(), "text/event-stream");

        }
        else {
            // Regular JSON response
            auto result = api_.chat(session_id, message);

            if (!result.ok()) {
                send_error(res, 500, result.status, result.error_message);
                return;
            }

            send_ok(res, chat_response_to_json(result.value));
        }
    }

    void HttpServer::handle_reset(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        std::string session_id = "default";
        try {
            auto body = json::parse(req.body);
            if (body.contains("session_id") && body["session_id"].is_string())
                session_id = body["session_id"].get<std::string>();
        }
        catch (...) {}

        auto result = api_.reset_session(session_id);
        if (!result.ok()) {
            send_error(res, 404, result.status, result.error_message);
            return;
        }

        json j;
        j["status"] = "ok";
        j["session_id"] = session_id;
        send_ok(res, j.dump());
    }

    void HttpServer::handle_stats(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        auto result = api_.get_stats();
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }
        send_ok(res, stats_to_json(result.value));
    }

    void HttpServer::handle_rules(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        auto result = api_.get_rules();
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }

        json arr = json::array();
        for (const auto& r : result.value) {
            arr.push_back(json::parse(rule_to_json(r)));
        }

        json j;
        j["rules"] = arr;
        j["count"] = static_cast<int>(result.value.size());
        send_ok(res, j.dump());
    }

    void HttpServer::handle_episodes(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        // Query params: keyword, domain, min_conf, max_results
        std::string keyword;
        std::string domain;
        float       min_conf = 0.0f;
        int         max_results = 20;

        if (req.has_param("keyword"))     keyword = req.get_param_value("keyword");
        if (req.has_param("domain"))      domain = req.get_param_value("domain");
        if (req.has_param("min_conf")) {
            try { min_conf = std::stof(req.get_param_value("min_conf")); }
            catch (...) {}
        }
        if (req.has_param("max_results")) {
            try { max_results = std::stoi(req.get_param_value("max_results")); }
            catch (...) {}
        }

        auto result = api_.get_episodes(keyword, domain, min_conf, max_results);
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }

        json arr = json::array();
        for (const auto& ep : result.value) {
            arr.push_back(json::parse(episode_to_json(ep)));
        }

        json j;
        j["episodes"] = arr;
        j["count"] = static_cast<int>(result.value.size());
        send_ok(res, j.dump());
    }

    void HttpServer::handle_scan(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        auto result = api_.run_scan();
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }
        send_ok(res, scan_result_to_json(result.value));
    }

    void HttpServer::handle_maintenance(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        auto result = api_.run_maintenance();
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }

        json j;
        j["status"] = "ok";
        j["message"] = "Maintenance cycle completed";
        send_ok(res, j.dump());
    }

    void HttpServer::handle_get_settings(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        auto result = api_.get_settings();
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }
        send_ok(res, settings_to_json(result.value));
    }

    void HttpServer::handle_post_settings(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        if (req.body.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT,
                "Request body is required");
            return;
        }

        // Get current settings, apply partial JSON patch
        auto current_result = api_.get_settings();
        if (!current_result.ok()) {
            send_error(res, 500, current_result.status,
                current_result.error_message);
            return;
        }

        // Parse incoming JSON and apply as partial update
        // We route through SettingsManager::from_json via the API
        // For now parse key/value pairs directly
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT,
                "Invalid JSON body");
            return;
        }

        // Apply each field individually via set_setting()
        for (const auto& [key, val] : body.items()) {
            std::string str_val;
            if (val.is_string())      str_val = val.get<std::string>();
            else if (val.is_number()) str_val = std::to_string(val.get<double>());
            else if (val.is_boolean()) str_val = val.get<bool>() ? "true" : "false";
            else continue;

            auto set_result = api_.set_setting(key, str_val);
            if (!set_result.ok()) {
                send_error(res, 400, set_result.status, set_result.error_message);
                return;
            }
        }

        // Return updated settings
        auto updated = api_.get_settings();
        if (!updated.ok()) {
            send_error(res, 500, updated.status, updated.error_message);
            return;
        }

        send_ok(res, settings_to_json(updated.value));
    }

    void HttpServer::handle_export(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        ExportRequest export_req;
        export_req.output_path = "data/training_export.jsonl";
        export_req.min_confidence = 0.7f;
        export_req.include_rules = true;

        try {
            auto body = json::parse(req.body);
            if (body.contains("output_path") && body["output_path"].is_string())
                export_req.output_path = body["output_path"].get<std::string>();
            if (body.contains("min_confidence") && body["min_confidence"].is_number())
                export_req.min_confidence = body["min_confidence"].get<float>();
            if (body.contains("domain") && body["domain"].is_string())
                export_req.domain = body["domain"].get<std::string>();
            if (body.contains("max_examples") && body["max_examples"].is_number_integer())
                export_req.max_examples = body["max_examples"].get<int>();
            if (body.contains("include_rules") && body["include_rules"].is_boolean())
                export_req.include_rules = body["include_rules"].get<bool>();
            if (body.contains("dry_run") && body["dry_run"].get<bool>()) {
                auto result = api_.export_dry_run(export_req);
                if (!result.ok()) {
                    send_error(res, 500, result.status, result.error_message);
                    return;
                }
                send_ok(res, export_info_to_json(result.value));
                return;
            }
        }
        catch (...) {}

        auto result = api_.export_training_data(export_req);
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }
        send_ok(res, export_info_to_json(result.value));
    }

    void HttpServer::handle_create_session(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        std::string requested_id;
        try {
            auto body = json::parse(req.body);
            if (body.contains("session_id") && body["session_id"].is_string())
                requested_id = body["session_id"].get<std::string>();
        }
        catch (...) {}

        auto result = api_.create_session(requested_id);
        if (!result.ok()) {
            send_error(res, 500, result.status, result.error_message);
            return;
        }

        json j;
        j["status"] = "ok";
        j["session_id"] = result.value;
        send_ok(res, j.dump());
    }

    void HttpServer::handle_destroy_session(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        // Extract session ID from URL: /api/sessions/<id>
        std::string session_id;
        if (req.matches.size() > 1) {
            session_id = req.matches[1].str();
        }

        if (session_id.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT,
                "Session ID required in URL");
            return;
        }

        auto result = api_.destroy_session(session_id);
        if (!result.ok()) {
            send_error(res, 404, result.status, result.error_message);
            return;
        }

        json j;
        j["status"] = "ok";
        j["session_id"] = session_id;
        j["message"] = "Session destroyed";
        send_ok(res, j.dump());
    }

    void HttpServer::handle_reset_session(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        std::string session_id;
        if (req.matches.size() > 1) {
            session_id = req.matches[1].str();
        }

        if (session_id.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT,
                "Session ID required in URL");
            return;
        }

        auto result = api_.reset_session(session_id);
        if (!result.ok()) {
            send_error(res, 404, result.status, result.error_message);
            return;
        }

        json j;
        j["status"] = "ok";
        j["session_id"] = session_id;
        j["message"] = "Session history cleared";
        send_ok(res, j.dump());
    }

    // =========================================================================
    // JSON response helpers
    // =========================================================================

    void HttpServer::send_ok(httplib::Response& res,
        const std::string& json_body) {
        res.status = 200;
        res.set_content(json_body, "application/json");
    }

    void HttpServer::send_error(httplib::Response& res,
        int                http_code,
        CardinalStatus     status,
        const std::string& message) {
        res.status = http_code;
        res.set_content(error_json(status, message), "application/json");
    }

    std::string HttpServer::error_json(CardinalStatus     status,
        const std::string& message) {
        json j;
        j["error"] = message;
        j["status"] = static_cast<int>(status);
        j["code"] = status_to_string(status);
        return j.dump();
    }

    std::string HttpServer::sse_event(const std::string& json_data) {
        return "data: " + json_data + "\n\n";
    }

    // =========================================================================
    // JSON serialization helpers
    // =========================================================================

    std::string HttpServer::feeling_to_json(const FeelingInfo& f) {
        json j;
        j["confidence"] = f.confidence;
        j["reasoning_type"] = f.reasoning_type;
        j["reasoning_domain"] = f.reasoning_domain;
        j["uncertainty_flag"] = f.uncertainty_flag;
        j["contradiction_flag"] = f.contradiction_flag;
        j["rule_candidate"] = f.rule_candidate;
        return j.dump();
    }

    std::string HttpServer::chat_response_to_json(const ChatResponse& r) {
        json j;
        j["session_id"] = r.session_id;
        j["response"] = r.response;
        j["episode_id"] = r.episode_id;
        j["feeling"] = json::parse(feeling_to_json(r.feeling));
        j["rule_committed"] = r.rule_committed;
        j["committed_rule_id"] = r.committed_rule_id;
        j["contradictions_found"] = r.contradictions_found;
        j["contradictions_resolved"] = r.contradictions_resolved;
        j["contradictions_flagged"] = r.contradictions_flagged;
        j["pass1_tokens"] = r.pass1_tokens;
        j["pass2_tokens"] = r.pass2_tokens;
        j["total_ms"] = r.total_ms;
        return j.dump();
    }

    std::string HttpServer::stats_to_json(const SystemStats& s) {
        json j;
        j["memory"]["total_episodes"] = s.memory.total_episodes;
        j["memory"]["migrated_episodes"] = s.memory.migrated_episodes;
        j["memory"]["high_conf_episodes"] = s.memory.high_conf_episodes;
        j["memory"]["rule_candidate_count"] = s.memory.rule_candidate_count;
        j["memory"]["avg_episode_confidence"] = s.memory.avg_episode_confidence;
        j["memory"]["total_rules"] = s.memory.total_rules;
        j["memory"]["active_rules"] = s.memory.active_rules;
        j["memory"]["avg_rule_confidence"] = s.memory.avg_rule_confidence;
        j["memory"]["index_size"] = s.memory.index_size;
        j["memory"]["vocabulary_size"] = s.memory.vocabulary_size;
        j["memory"]["index_ready"] = s.memory.index_ready;
        j["verifier"]["total_checks"] = s.verifier.total_checks;
        j["verifier"]["total_rules_extracted"] = s.verifier.total_rules_extracted;
        j["verifier"]["total_contradictions"] = s.verifier.total_contradictions;
        j["verifier"]["total_resolved"] = s.verifier.total_resolved;
        j["verifier"]["total_flagged"] = s.verifier.total_flagged;
        j["verifier"]["total_maintenance_runs"] = s.verifier.total_maintenance_runs;
        j["uptime"] = s.uptime_seconds;
        j["version"] = s.version;
        j["initialized"] = s.initialized;
        return j.dump(2);
    }

    std::string HttpServer::rule_to_json(const RuleInfo& r) {
        json j;
        j["id"] = r.id;
        j["domain"] = r.domain;
        j["condition"] = r.condition;
        j["consequence"] = r.consequence;
        j["confidence"] = r.confidence;
        j["trigger_count"] = r.trigger_count;
        j["episode_id"] = r.episode_id;
        j["reasoning_type"] = r.reasoning_type;
        j["created_at"] = r.created_at;
        j["updated_at"] = r.updated_at;
        j["has_provenance"] = r.has_provenance;
        return j.dump();
    }

    std::string HttpServer::episode_to_json(const EpisodeInfo& ep) {
        json j;
        j["id"] = ep.id;
        j["timestamp"] = ep.timestamp;
        j["user_message"] = ep.user_message;
        j["response_summary"] = ep.response_summary;
        j["confidence"] = ep.confidence;
        j["reasoning_type"] = ep.reasoning_type;
        j["reasoning_domain"] = ep.reasoning_domain;
        j["contradiction"] = ep.contradiction;
        j["uncertainty"] = ep.uncertainty;
        j["rule_candidate"] = ep.rule_candidate;
        j["extracted_rule_id"] = ep.extracted_rule_id;
        j["pass1_tokens"] = ep.pass1_tokens;
        j["pass2_tokens"] = ep.pass2_tokens;
        j["total_ms"] = ep.total_ms;
        return j.dump();
    }

    std::string HttpServer::export_info_to_json(const ExportInfo& info) {
        json j;
        j["episodes_exported"] = info.episodes_exported;
        j["rules_exported"] = info.rules_exported;
        j["total_exported"] = info.total_exported;
        j["avg_confidence"] = info.avg_confidence;
        j["output_path"] = info.output_path;
        j["timestamp"] = info.timestamp;
        return j.dump();
    }

    std::string HttpServer::scan_result_to_json(const ScanResult& r) {
        json j;
        j["total_contradictions"] = r.total_contradictions;
        j["resolved"] = r.resolved;
        j["flagged"] = r.flagged;
        j["skipped"] = r.skipped;
        return j.dump();
    }

    std::string HttpServer::settings_to_json(const CardinalSettings& s) {
        json j;
        j["retriever_mode"] = s.retriever_mode;
        j["keyword_weight"] = s.keyword_weight;
        j["semantic_weight"] = s.semantic_weight;
        j["max_retrieval_results"] = s.max_retrieval_results;
        j["min_retrieval_score"] = s.min_retrieval_score;
        j["verifier_mode"] = s.verifier_mode;
        j["min_rule_confidence"] = s.min_rule_confidence;
        j["contradiction_threshold"] = s.contradiction_threshold;
        j["temperature"] = s.temperature;
        j["top_p"] = s.top_p;
        j["stream_responses"] = s.stream_responses;
        j["log_level"] = s.log_level;
        return j.dump(2);
    }

    std::string HttpServer::session_info_to_json(const SessionInfo& info) {
        json j;
        j["session_id"] = info.session_id;
        j["turn_count"] = info.turn_count;
        j["created_at"] = info.created_at;
        j["last_active_at"] = info.last_active_at;

        json history = json::array();
        for (const auto& turn : info.history) {
            json t;
            t["role"] = turn.role;
            t["content"] = turn.content;
            t["timestamp"] = turn.timestamp;
            history.push_back(t);
        }
        j["history"] = history;
        return j.dump();
    }

} // namespace cardinal