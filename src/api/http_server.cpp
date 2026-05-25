// =============================================================================
// Cardinal - HTTP Server Implementation (v1.5.0)
// File: src/api/http_server.cpp
//
// Changes from v1.4.0:
//   - Scheduler endpoints registered in register_routes()
//   - Computer use endpoints registered in register_routes()
//   - All new handler implementations added
//   - All new JSON serializer implementations added
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

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
        server_->set_payload_max_length(
            static_cast<size_t>(config.api.max_request_size_kb) * 1024);

        server_->set_default_headers({
            { "Access-Control-Allow-Origin",  "*" },
            { "Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS" },
            { "Access-Control-Allow-Headers",
              "Content-Type, Authorization, Accept" }
        });

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
        if (!ok)
            LOG_WARN("HttpServer failed to bind on " + host_ + ":" + std::to_string(port_));
        else
            LOG_INFO("HttpServer stopped");
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
        // Health — always public
        server_->Get("/api/health",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_health(req, res);
            });

        // Chat
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

        // v1.4.0 — Self-Improvement
        server_->Get("/api/self_model",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_self_model(req, res);
            });
        server_->Post("/api/reflect",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_reflect(req, res);
            });
        server_->Post("/api/train",
            [this](const httplib::Request& req, httplib::Response& res) {
                handle_train(req, res);
            });

        // v1.5.0 — Scheduler
        // NOTE: more-specific patterns must be registered BEFORE less-specific ones
        server_->Get("/api/scheduler/status",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_status(req, res);
            });
        server_->Get("/api/scheduler/tasks",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_tasks_list(req, res);
            });
        server_->Post("/api/scheduler/tasks",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_create(req, res);
            });
        server_->Get("/api/scheduler/runs",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_runs_list(req, res);
            });
        server_->Get(R"(/api/scheduler/runs/(.+)/actions)",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_run_actions(req, res);
            });
        server_->Get(R"(/api/scheduler/tasks/(.+)/history)",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_history(req, res);
            });
        server_->Post(R"(/api/scheduler/tasks/(.+)/run)",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_run(req, res);
            });
        server_->Post(R"(/api/scheduler/tasks/(.+)/enable)",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_enable(req, res);
            });
        server_->Post(R"(/api/scheduler/tasks/(.+)/disable)",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_disable(req, res);
            });
        server_->Get(R"(/api/scheduler/tasks/(.+))",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_get(req, res);
            });
        server_->Put(R"(/api/scheduler/tasks/(.+))",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_put(req, res);
            });
        server_->Delete(R"(/api/scheduler/tasks/(.+))",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_scheduler_task_delete(req, res);
            });

        // v1.5.0 — Computer Use
        server_->Get("/api/computer/status",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_computer_status(req, res);
            });
        server_->Post("/api/computer/screenshot",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_computer_screenshot(req, res);
            });
        server_->Post("/api/computer/click",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_computer_click(req, res);
            });
        server_->Post("/api/computer/type",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_computer_type(req, res);
            });
        server_->Post("/api/computer/shell",
            [this](const httplib::Request& req, httplib::Response& res) {
                if (check_auth(req, res)) handle_computer_shell(req, res);
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

        const std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) != prefix) {
            send_error(res, 401, CardinalStatus::AUTH_FAILED,
                "Invalid Authorization format — use: Bearer <api_key>");
            return false;
        }

        if (auth_header.substr(prefix.size()) != api_key_) {
            send_error(res, 401, CardinalStatus::AUTH_FAILED, "Invalid API key");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Existing route handlers (v1.4.0, unchanged)
    // =========================================================================

    void HttpServer::handle_health(const httplib::Request&,
        httplib::Response& res) {
        auto result = api_.health_check();
        if (result.ok()) {
            json j;
            j["status"]  = "ok";
            j["uptime"]  = api_.uptime_string();
            j["version"] = "1.5.0";
            send_ok(res, j.dump());
        } else {
            send_error(res, 503, result.status, result.error_message);
        }
    }

    void HttpServer::handle_chat(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;

        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT, "Invalid JSON body");
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

        bool wants_stream = false;
        auto accept = req.get_header_value("Accept");
        if (accept.find("text/event-stream") != std::string::npos)
            wants_stream = true;
        if (body.contains("stream") && body["stream"].is_boolean())
            wants_stream = body["stream"].get<bool>();

        if (wants_stream && config_.api.stream_enabled) {
            res.set_header("Content-Type",    "text/event-stream");
            res.set_header("Cache-Control",   "no-cache");
            res.set_header("Connection",      "keep-alive");
            res.set_header("X-Accel-Buffering", "no");

            std::ostringstream sse_body;

            ApiStreamCallback cb = [&sse_body](const StreamToken& token) -> bool {
                json t;
                t["token"]    = token.token;
                t["is_final"] = token.is_final;
                if (token.is_final) {
                    json f;
                    f["confidence"]        = token.feeling.confidence;
                    f["reasoning_type"]    = token.feeling.reasoning_type;
                    f["reasoning_domain"]  = token.feeling.reasoning_domain;
                    f["uncertainty_flag"]  = token.feeling.uncertainty_flag;
                    f["contradiction_flag"]= token.feeling.contradiction_flag;
                    f["rule_candidate"]    = token.feeling.rule_candidate;
                    t["feeling"] = f;
                }
                sse_body << sse_event(t.dump());
                return true;
            };

            auto result = api_.chat_stream(session_id, message, cb);
            if (!result.ok())
                sse_body << sse_event(error_json(result.status, result.error_message));

            res.status = 200;
            res.set_content(sse_body.str(), "text/event-stream");
        } else {
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
        } catch (...) {}
        auto result = api_.reset_session(session_id);
        if (!result.ok()) { send_error(res, 404, result.status, result.error_message); return; }
        json j; j["status"] = "ok"; j["session_id"] = session_id;
        send_ok(res, j.dump());
    }

    void HttpServer::handle_stats(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.get_stats();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        send_ok(res, stats_to_json(result.value));
    }

    void HttpServer::handle_rules(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.get_rules();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        json arr = json::array();
        for (const auto& r : result.value) arr.push_back(json::parse(rule_to_json(r)));
        json j; j["rules"] = arr; j["count"] = static_cast<int>(result.value.size());
        send_ok(res, j.dump());
    }

    void HttpServer::handle_episodes(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        std::string keyword, domain;
        float min_conf = 0.0f; int max_results = 20;
        if (req.has_param("keyword"))     keyword = req.get_param_value("keyword");
        if (req.has_param("domain"))      domain  = req.get_param_value("domain");
        if (req.has_param("min_conf"))    try { min_conf    = std::stof(req.get_param_value("min_conf")); } catch (...) {}
        if (req.has_param("max_results")) try { max_results = std::stoi(req.get_param_value("max_results")); } catch (...) {}
        auto result = api_.get_episodes(keyword, domain, min_conf, max_results);
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        json arr = json::array();
        for (const auto& ep : result.value) arr.push_back(json::parse(episode_to_json(ep)));
        json j; j["episodes"] = arr; j["count"] = static_cast<int>(result.value.size());
        send_ok(res, j.dump());
    }

    void HttpServer::handle_scan(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.run_scan();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        send_ok(res, scan_result_to_json(result.value));
    }

    void HttpServer::handle_maintenance(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.run_maintenance();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        json j; j["status"] = "ok"; j["message"] = "Maintenance cycle completed";
        send_ok(res, j.dump());
    }

    void HttpServer::handle_get_settings(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.get_settings();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        send_ok(res, settings_to_json(result.value));
    }

    void HttpServer::handle_post_settings(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        if (req.body.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT, "Request body is required");
            return;
        }
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT, "Invalid JSON body");
            return;
        }
        for (const auto& [key, val] : body.items()) {
            std::string str_val;
            if (val.is_string())       str_val = val.get<std::string>();
            else if (val.is_number())  str_val = std::to_string(val.get<double>());
            else if (val.is_boolean()) str_val = val.get<bool>() ? "true" : "false";
            else continue;
            auto set_result = api_.set_setting(key, str_val);
            if (!set_result.ok()) {
                send_error(res, 400, set_result.status, set_result.error_message);
                return;
            }
        }
        auto updated = api_.get_settings();
        if (!updated.ok()) { send_error(res, 500, updated.status, updated.error_message); return; }
        send_ok(res, settings_to_json(updated.value));
    }

    void HttpServer::handle_export(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        ExportRequest export_req;
        export_req.output_path    = "data/training_export.jsonl";
        export_req.min_confidence = 0.7f;
        export_req.include_rules  = true;
        try {
            auto body = json::parse(req.body);
            if (body.contains("output_path")    && body["output_path"].is_string())
                export_req.output_path    = body["output_path"].get<std::string>();
            if (body.contains("min_confidence") && body["min_confidence"].is_number())
                export_req.min_confidence = body["min_confidence"].get<float>();
            if (body.contains("domain")         && body["domain"].is_string())
                export_req.domain         = body["domain"].get<std::string>();
            if (body.contains("max_examples")   && body["max_examples"].is_number_integer())
                export_req.max_examples   = body["max_examples"].get<int>();
            if (body.contains("include_rules")  && body["include_rules"].is_boolean())
                export_req.include_rules  = body["include_rules"].get<bool>();
            if (body.contains("dry_run") && body["dry_run"].get<bool>()) {
                auto result = api_.export_dry_run(export_req);
                if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
                send_ok(res, export_info_to_json(result.value)); return;
            }
        } catch (...) {}
        auto result = api_.export_training_data(export_req);
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
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
        } catch (...) {}
        auto result = api_.create_session(requested_id);
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        json j; j["status"] = "ok"; j["session_id"] = result.value;
        send_ok(res, j.dump());
    }

    void HttpServer::handle_destroy_session(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        std::string session_id;
        if (req.matches.size() > 1) session_id = req.matches[1].str();
        if (session_id.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT, "Session ID required in URL");
            return;
        }
        auto result = api_.destroy_session(session_id);
        if (!result.ok()) { send_error(res, 404, result.status, result.error_message); return; }
        json j; j["status"] = "ok"; j["session_id"] = session_id; j["message"] = "Session destroyed";
        send_ok(res, j.dump());
    }

    void HttpServer::handle_reset_session(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        std::string session_id;
        if (req.matches.size() > 1) session_id = req.matches[1].str();
        if (session_id.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_INPUT, "Session ID required in URL");
            return;
        }
        auto result = api_.reset_session(session_id);
        if (!result.ok()) { send_error(res, 404, result.status, result.error_message); return; }
        json j; j["status"] = "ok"; j["session_id"] = session_id; j["message"] = "Session history cleared";
        send_ok(res, j.dump());
    }

    // =========================================================================
    // v1.4.0 — Self-Improvement handlers
    // =========================================================================

    void HttpServer::handle_self_model(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.get_self_model_status();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        send_ok(res, self_improvement_status_to_json(result.value));
    }

    void HttpServer::handle_reflect(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        auto result = api_.reflect();
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        send_ok(res, reflection_result_to_json(result.value));
    }

    void HttpServer::handle_train(const httplib::Request& req,
        httplib::Response& res) {
        if (!check_auth(req, res)) return;
        std::string domain_hint;
        if (!req.body.empty()) {
            try {
                auto body = json::parse(req.body);
                domain_hint = body.value("domain_hint", std::string(""));
            } catch (...) {}
        }
        auto result = api_.trigger_training(domain_hint);
        if (!result.ok()) { send_error(res, 500, result.status, result.error_message); return; }
        json j;
        j["accepted"]    = result.value;
        j["domain_hint"] = domain_hint;
        j["message"]     = result.value
            ? "Training request posted to background thread"
            : "Training not started (disabled or already running)";
        send_ok(res, j.dump());
    }

    // =========================================================================
    // v1.5.0 — Scheduler handlers
    // =========================================================================

    void HttpServer::handle_scheduler_status(const httplib::Request&,
        httplib::Response& res) {
        auto result = api_.get_scheduler_status();
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        send_ok(res, scheduler_status_to_json(result.value));
    }

    void HttpServer::handle_scheduler_tasks_list(const httplib::Request&,
        httplib::Response& res) {
        auto result = api_.list_tasks();
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json arr = json::array();
        for (const auto& t : result.value)
            arr.push_back(json::parse(scheduled_task_to_json(t)));
        send_ok(res, arr.dump());
    }

    void HttpServer::handle_scheduler_task_create(const httplib::Request& req,
        httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_REQUEST, "Invalid JSON"); return;
        }
        if (body.contains("description") && body["description"].is_string()) {
            auto result = api_.create_task(body["description"].get<std::string>(),
                                           body.value("session_id", std::string("")));
            if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
            send_ok(res, task_parse_result_to_json(result.value));
        } else {
            ScheduledTask task;
            task.name         = body.value("name", "Unnamed");
            task.description  = body.value("description", "");
            task.enabled      = body.value("enabled", true);
            task.created_from = "api";
            if (body.contains("trigger"))
                task.trigger = trigger_spec_from_json(body["trigger"].dump());
            if (body.contains("action"))
                task.action = task_action_from_json(body["action"].dump());
            auto result = api_.create_task_direct(task);
            if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
            json resp; resp["id"] = result.value; resp["success"] = true;
            send_ok(res, resp.dump());
        }
    }

    void HttpServer::handle_scheduler_task_get(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        auto result = api_.get_task(id);
        if (!result.ok()) { send_error(res, 404, result.status, result.message); return; }
        send_ok(res, scheduled_task_to_json(result.value));
    }

    void HttpServer::handle_scheduler_task_put(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_REQUEST, "Invalid JSON"); return;
        }
        auto existing = api_.get_task(id);
        if (!existing.ok()) { send_error(res, 404, existing.status, existing.message); return; }
        ScheduledTask task = existing.value;
        if (body.contains("name"))    task.name    = body["name"].get<std::string>();
        if (body.contains("enabled")) task.enabled = body["enabled"].get<bool>();
        if (body.contains("trigger")) task.trigger = trigger_spec_from_json(body["trigger"].dump());
        if (body.contains("action"))  task.action  = task_action_from_json(body["action"].dump());
        auto result = api_.update_task(task);
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        send_ok(res, scheduled_task_to_json(task));
    }

    void HttpServer::handle_scheduler_task_delete(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        auto result = api_.delete_task(id);
        if (!result.ok()) { send_error(res, 404, result.status, result.message); return; }
        json resp; resp["deleted"] = id;
        send_ok(res, resp.dump());
    }

    void HttpServer::handle_scheduler_task_run(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        auto result = api_.run_task_now(id);
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json resp; resp["run_id"] = result.value;
        send_ok(res, resp.dump());
    }

    void HttpServer::handle_scheduler_task_enable(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        auto result = api_.enable_task(id);
        if (!result.ok()) { send_error(res, 404, result.status, result.message); return; }
        json resp; resp["enabled"] = true; resp["id"] = id;
        send_ok(res, resp.dump());
    }

    void HttpServer::handle_scheduler_task_disable(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        auto result = api_.disable_task(id);
        if (!result.ok()) { send_error(res, 404, result.status, result.message); return; }
        json resp; resp["enabled"] = false; resp["id"] = id;
        send_ok(res, resp.dump());
    }

    void HttpServer::handle_scheduler_task_history(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        int limit = 50;
        if (req.has_param("limit")) try { limit = std::stoi(req.get_param_value("limit")); } catch (...) {}
        auto result = api_.get_task_history(id, limit);
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json arr = json::array();
        for (const auto& r : result.value) arr.push_back(json::parse(task_run_to_json(r)));
        send_ok(res, arr.dump());
    }

    void HttpServer::handle_scheduler_runs_list(const httplib::Request& req,
        httplib::Response& res) {
        int limit = 100;
        if (req.has_param("limit")) try { limit = std::stoi(req.get_param_value("limit")); } catch (...) {}
        auto result = api_.get_recent_runs(limit);
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json arr = json::array();
        for (const auto& r : result.value) arr.push_back(json::parse(task_run_to_json(r)));
        send_ok(res, arr.dump());
    }

    void HttpServer::handle_scheduler_run_actions(const httplib::Request& req,
        httplib::Response& res) {
        std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
        auto result = api_.get_run_action_logs(id);
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json arr = json::array();
        for (const auto& e : result.value) arr.push_back(json::parse(task_action_log_to_json(e)));
        send_ok(res, arr.dump());
    }

    // =========================================================================
    // v1.5.0 — Computer Use handlers
    // =========================================================================

    void HttpServer::handle_computer_status(const httplib::Request&,
        httplib::Response& res) {
        auto result = api_.get_computer_status();
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        send_ok(res, screen_info_to_json(result.value));
    }

    void HttpServer::handle_computer_screenshot(const httplib::Request& req,
        httplib::Response& res) {
        bool analyze = true; std::string prompt;
        if (!req.body.empty()) {
            try {
                auto body = json::parse(req.body);
                analyze   = body.value("analyze", true);
                prompt    = body.value("prompt", "");
            } catch (...) {}
        }
        auto result = api_.take_screenshot(analyze, prompt);
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        send_ok(res, screenshot_to_json(result.value));
    }

    void HttpServer::handle_computer_click(const httplib::Request& req,
        httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_REQUEST, "Invalid JSON"); return;
        }
        auto result = api_.computer_click(body.value("x", -1),
                                           body.value("y", -1),
                                           body.value("description", std::string("")));
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json resp; resp["result"] = result.value;
        send_ok(res, resp.dump());
    }

    void HttpServer::handle_computer_type(const httplib::Request& req,
        httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_REQUEST, "Invalid JSON"); return;
        }
        auto result = api_.computer_type(body.value("text", std::string("")),
                                          body.value("key",  std::string("")));
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        json resp; resp["result"] = result.value;
        send_ok(res, resp.dump());
    }

    void HttpServer::handle_computer_shell(const httplib::Request& req,
        httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            send_error(res, 400, CardinalStatus::INVALID_REQUEST, "Invalid JSON"); return;
        }
        std::string command = body.value("command", "");
        if (command.empty()) {
            send_error(res, 400, CardinalStatus::INVALID_REQUEST, "Missing command"); return;
        }
        auto result = api_.computer_shell(command, body.value("timeout_seconds", 0));
        if (!result.ok()) { send_error(res, 503, result.status, result.message); return; }
        send_ok(res, shell_result_to_json(result.value));
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
        int http_code, CardinalStatus status,
        const std::string& message) {
        res.status = http_code;
        res.set_content(error_json(status, message), "application/json");
    }

    std::string HttpServer::error_json(CardinalStatus status,
        const std::string& message) {
        json j;
        j["error"]  = message;
        j["status"] = static_cast<int>(status);
        j["code"]   = status_to_string(status);
        return j.dump();
    }

    std::string HttpServer::sse_event(const std::string& json_data) {
        return "data: " + json_data + "\n\n";
    }

    // =========================================================================
    // JSON serialization helpers — existing (v1.4.0)
    // =========================================================================

    std::string HttpServer::feeling_to_json(const FeelingInfo& f) {
        json j;
        j["confidence"]        = f.confidence;
        j["reasoning_type"]    = f.reasoning_type;
        j["reasoning_domain"]  = f.reasoning_domain;
        j["uncertainty_flag"]  = f.uncertainty_flag;
        j["contradiction_flag"]= f.contradiction_flag;
        j["rule_candidate"]    = f.rule_candidate;
        return j.dump();
    }

    std::string HttpServer::chat_response_to_json(const ChatResponse& r) {
        json j;
        j["session_id"]              = r.session_id;
        j["response"]                = r.response;
        j["episode_id"]              = r.episode_id;
        j["feeling"]                 = json::parse(feeling_to_json(r.feeling));
        j["rule_committed"]          = r.rule_committed;
        j["committed_rule_id"]       = r.committed_rule_id;
        j["contradictions_found"]    = r.contradictions_found;
        j["contradictions_resolved"] = r.contradictions_resolved;
        j["contradictions_flagged"]  = r.contradictions_flagged;
        j["pass1_tokens"]            = r.pass1_tokens;
        j["pass2_tokens"]            = r.pass2_tokens;
        j["total_ms"]                = r.total_ms;
        return j.dump();
    }

    std::string HttpServer::stats_to_json(const SystemStats& s) {
        json j;
        j["memory"]["total_episodes"]         = s.memory.total_episodes;
        j["memory"]["migrated_episodes"]      = s.memory.migrated_episodes;
        j["memory"]["high_conf_episodes"]     = s.memory.high_conf_episodes;
        j["memory"]["rule_candidate_count"]   = s.memory.rule_candidate_count;
        j["memory"]["avg_episode_confidence"] = s.memory.avg_episode_confidence;
        j["memory"]["total_rules"]            = s.memory.total_rules;
        j["memory"]["active_rules"]           = s.memory.active_rules;
        j["memory"]["avg_rule_confidence"]    = s.memory.avg_rule_confidence;
        j["memory"]["index_size"]             = s.memory.index_size;
        j["memory"]["vocabulary_size"]        = s.memory.vocabulary_size;
        j["memory"]["index_ready"]            = s.memory.index_ready;
        j["verifier"]["total_checks"]           = s.verifier.total_checks;
        j["verifier"]["total_rules_extracted"]  = s.verifier.total_rules_extracted;
        j["verifier"]["total_contradictions"]   = s.verifier.total_contradictions;
        j["verifier"]["total_resolved"]         = s.verifier.total_resolved;
        j["verifier"]["total_flagged"]          = s.verifier.total_flagged;
        j["verifier"]["total_maintenance_runs"] = s.verifier.total_maintenance_runs;
        j["uptime"]      = s.uptime_seconds;
        j["version"]     = s.version;
        j["initialized"] = s.initialized;
        return j.dump(2);
    }

    std::string HttpServer::rule_to_json(const RuleInfo& r) {
        json j;
        j["id"]             = r.id;
        j["domain"]         = r.domain;
        j["condition"]      = r.condition;
        j["consequence"]    = r.consequence;
        j["confidence"]     = r.confidence;
        j["trigger_count"]  = r.trigger_count;
        j["episode_id"]     = r.episode_id;
        j["reasoning_type"] = r.reasoning_type;
        j["created_at"]     = r.created_at;
        j["updated_at"]     = r.updated_at;
        j["has_provenance"] = r.has_provenance;
        return j.dump();
    }

    std::string HttpServer::episode_to_json(const EpisodeInfo& ep) {
        json j;
        j["id"]                = ep.id;
        j["timestamp"]         = ep.timestamp;
        j["user_message"]      = ep.user_message;
        j["response_summary"]  = ep.response_summary;
        j["confidence"]        = ep.confidence;
        j["reasoning_type"]    = ep.reasoning_type;
        j["reasoning_domain"]  = ep.reasoning_domain;
        j["contradiction"]     = ep.contradiction;
        j["uncertainty"]       = ep.uncertainty;
        j["rule_candidate"]    = ep.rule_candidate;
        j["extracted_rule_id"] = ep.extracted_rule_id;
        j["pass1_tokens"]      = ep.pass1_tokens;
        j["pass2_tokens"]      = ep.pass2_tokens;
        j["total_ms"]          = ep.total_ms;
        return j.dump();
    }

    std::string HttpServer::export_info_to_json(const ExportInfo& info) {
        json j;
        j["episodes_exported"] = info.episodes_exported;
        j["rules_exported"]    = info.rules_exported;
        j["total_exported"]    = info.total_exported;
        j["avg_confidence"]    = info.avg_confidence;
        j["output_path"]       = info.output_path;
        j["timestamp"]         = info.timestamp;
        return j.dump();
    }

    std::string HttpServer::scan_result_to_json(const ScanResult& r) {
        json j;
        j["total_contradictions"] = r.total_contradictions;
        j["resolved"] = r.resolved;
        j["flagged"]  = r.flagged;
        j["skipped"]  = r.skipped;
        return j.dump();
    }

    std::string HttpServer::settings_to_json(const CardinalSettings& s) {
        json j;
        j["retriever_mode"]         = s.retriever_mode;
        j["keyword_weight"]         = s.keyword_weight;
        j["semantic_weight"]        = s.semantic_weight;
        j["max_retrieval_results"]  = s.max_retrieval_results;
        j["min_retrieval_score"]    = s.min_retrieval_score;
        j["verifier_mode"]          = s.verifier_mode;
        j["min_rule_confidence"]    = s.min_rule_confidence;
        j["contradiction_threshold"]= s.contradiction_threshold;
        j["temperature"]            = s.temperature;
        j["top_p"]                  = s.top_p;
        j["stream_responses"]       = s.stream_responses;
        j["log_level"]              = s.log_level;
        return j.dump(2);
    }

    std::string HttpServer::session_info_to_json(const SessionInfo& info) {
        json j;
        j["session_id"]    = info.session_id;
        j["turn_count"]    = info.turn_count;
        j["created_at"]    = info.created_at;
        j["last_active_at"]= info.last_active_at;
        json history = json::array();
        for (const auto& turn : info.history) {
            json t; t["role"] = turn.role; t["content"] = turn.content;
            t["timestamp"] = turn.timestamp; history.push_back(t);
        }
        j["history"] = history;
        return j.dump();
    }

    std::string HttpServer::self_improvement_status_to_json(
        const SelfImprovementStatus& s) {
        json j;
        j["self_model_enabled"]    = s.self_model_enabled;
        j["weakest_domain"]        = s.weakest_domain;
        j["strongest_domain"]      = s.strongest_domain;
        j["total_domain_stats"]    = s.total_domain_stats;
        j["meta_cognition_enabled"]= s.meta_cognition_enabled;
        j["total_reflections"]     = s.total_reflections;
        j["total_corrective_rules"]= s.total_corrective_rules;
        j["last_reflection_at"]    = s.last_reflection_at;
        j["training_enabled"]      = s.training_enabled;
        j["total_training_runs"]   = s.total_training_runs;
        j["last_training_at"]      = s.last_training_at;
        j["active_adapter_path"]   = s.active_adapter_path;
        j["last_improvement_pct"]  = s.last_improvement_pct;
        return j.dump(2);
    }

    std::string HttpServer::reflection_result_to_json(const ReflectionResult& r) {
        json j;
        j["ran"]               = r.ran;
        j["trigger"]           = r.trigger;
        j["episodes_analyzed"] = r.episodes_analyzed;
        j["failures_analyzed"] = r.failures_analyzed;
        j["rules_committed"]   = r.rules_committed;
        j["duration_ms"]       = r.duration_ms;
        j["timestamp"]         = r.timestamp;
        j["error_message"]     = r.error_message;
        json findings = json::array();
        for (const auto& f : r.findings) {
            json fi;
            fi["domain"]         = f.domain;
            fi["pattern"]        = f.pattern;
            fi["recommendation"] = f.recommendation;
            fi["confidence"]     = f.confidence;
            fi["timestamp"]      = f.timestamp;
            findings.push_back(fi);
        }
        j["findings"] = findings;
        return j.dump(2);
    }

    // =========================================================================
    // JSON serialization helpers — v1.5.0
    // =========================================================================

    std::string HttpServer::scheduler_status_to_json(const SchedulerStatus& s) {
        json j;
        j["enabled"]           = s.enabled;
        j["running"]           = s.running;
        j["total_tasks"]       = s.total_tasks;
        j["enabled_tasks"]     = s.enabled_tasks;
        j["total_runs"]        = s.total_runs;
        j["successful_runs"]   = s.successful_runs;
        j["failed_runs"]       = s.failed_runs;
        j["current_task_id"]   = s.current_task_id;
        j["current_task_name"] = s.current_task_name;
        j["last_run_at"]       = s.last_run_at;
        j["next_scheduled_at"] = s.next_scheduled_at;
        return j.dump();
    }

    std::string HttpServer::trigger_spec_to_json(const TriggerSpec& t) {
        json j;
        j["type"]             = trigger_type_to_string(t.type);
        j["cron_expression"]  = t.cron_expression;
        j["interval_seconds"] = t.interval_seconds;
        j["condition_expr"]   = t.condition_expr;
        j["idle_minutes"]     = t.idle_minutes;
        return j.dump();
    }

    std::string HttpServer::task_action_to_json(const TaskAction& a) {
        json j;
        j["type"]           = action_type_to_string(a.type);
        j["goal"]           = a.goal;
        j["max_iterations"] = a.max_iterations;
        j["domain_hint"]    = a.domain_hint;
        j["shell_command"]  = a.shell_command;
        j["webhook_url"]    = a.webhook_url;
        j["output_file"]    = a.output_file;
        return j.dump();
    }

    TriggerSpec HttpServer::trigger_spec_from_json(const std::string& s) {
        TriggerSpec t;
        try {
            auto j = json::parse(s);
            std::string ts = j.value("type", "manual");
            if      (ts == "cron")      t.type = TriggerType::CRON;
            else if (ts == "interval")  t.type = TriggerType::INTERVAL;
            else if (ts == "condition") t.type = TriggerType::CONDITION;
            else if (ts == "startup")   t.type = TriggerType::STARTUP;
            else if (ts == "idle")      t.type = TriggerType::IDLE;
            else                        t.type = TriggerType::MANUAL;
            t.cron_expression  = j.value("cron_expression", "");
            t.interval_seconds = j.value("interval_seconds", 0);
            t.condition_expr   = j.value("condition_expr", "");
            t.idle_minutes     = j.value("idle_minutes", 30);
        } catch (...) {}
        return t;
    }

    TaskAction HttpServer::task_action_from_json(const std::string& s) {
        TaskAction a;
        try {
            auto j = json::parse(s);
            std::string ts = j.value("type", "agent_run");
            if      (ts == "chat")             a.type = TaskActionType::CHAT;
            else if (ts == "reflect")          a.type = TaskActionType::REFLECT;
            else if (ts == "train")            a.type = TaskActionType::TRAIN;
            else if (ts == "self_improvement") a.type = TaskActionType::SELF_IMPROVEMENT;
            else if (ts == "maintenance")      a.type = TaskActionType::MAINTENANCE;
            else if (ts == "export")           a.type = TaskActionType::EXPORT;
            else if (ts == "shell")            a.type = TaskActionType::SHELL;
            else if (ts == "webhook")          a.type = TaskActionType::WEBHOOK;
            else                               a.type = TaskActionType::AGENT_RUN;
            a.goal           = j.value("goal", "");
            a.max_iterations = j.value("max_iterations", 0);
            a.domain_hint    = j.value("domain_hint", "");
            a.shell_command  = j.value("shell_command", "");
            a.webhook_url    = j.value("webhook_url", "");
            a.output_file    = j.value("output_file", "");
        } catch (...) {}
        return a;
    }

    std::string HttpServer::scheduled_task_to_json(const ScheduledTask& t) {
        json j;
        j["id"]           = t.id;
        j["name"]         = t.name;
        j["description"]  = t.description;
        j["enabled"]      = t.enabled;
        j["trigger"]      = json::parse(trigger_spec_to_json(t.trigger));
        j["action"]       = json::parse(task_action_to_json(t.action));
        j["run_count"]    = t.run_count;
        j["fail_count"]   = t.fail_count;
        j["last_run_at"]  = t.last_run_at;
        j["next_run_at"]  = t.next_run_at;
        j["created_at"]   = t.created_at;
        j["updated_at"]   = t.updated_at;
        j["created_from"] = t.created_from;
        return j.dump();
    }

    std::string HttpServer::task_run_to_json(const TaskRun& r) {
        json j;
        j["run_id"]         = r.run_id;
        j["task_id"]        = r.task_id;
        j["task_name"]      = r.task_name;
        j["status"]         = run_status_to_string(r.status);
        j["started_at"]     = r.started_at;
        j["finished_at"]    = r.finished_at;
        j["result_summary"] = r.result_summary;
        j["error_message"]  = r.error_message;
        j["duration_ms"]    = r.duration_ms;
        j["output_path"]    = r.output_path;
        j["session_id"]     = r.session_id;
        return j.dump();
    }

    std::string HttpServer::task_action_log_to_json(const TaskActionLog& e) {
        json j;
        j["sequence"]              = e.sequence;
        j["action_type"]           = e.action_type;
        j["description"]           = e.description;
        j["input_summary"]         = e.input_summary;
        j["output_summary"]        = e.output_summary;
        j["success"]               = e.success;
        j["required_confirmation"] = e.required_confirmation;
        j["confirmation_granted"]  = e.confirmation_granted;
        j["duration_ms"]           = e.duration_ms;
        j["timestamp"]             = e.timestamp;
        return j.dump();
    }

    std::string HttpServer::task_parse_result_to_json(const TaskParseResult& r) {
        json j;
        j["success"]              = r.success;
        j["confidence"]           = r.confidence;
        j["error_message"]        = r.error_message;
        j["clarification_needed"] = r.clarification_needed;
        if (r.success)
            j["task"] = json::parse(scheduled_task_to_json(r.task));
        return j.dump();
    }

    std::string HttpServer::screen_info_to_json(const ScreenInfo& s) {
        json j;
        j["width"]        = s.width;
        j["height"]       = s.height;
        j["scale_factor"] = s.scale_factor;
        j["server"]       = display_server_to_string(s.server);
        j["display_var"]  = s.display_var;
        return j.dump();
    }

    std::string HttpServer::screenshot_to_json(const Screenshot& s) {
        json j;
        j["path"]        = s.path;
        j["width"]       = s.width;
        j["height"]      = s.height;
        j["timestamp"]   = s.timestamp;
        j["analyzed"]    = s.analyzed;
        j["description"] = s.description;
        return j.dump();
    }

    std::string HttpServer::shell_result_to_json(const ShellResult& r) {
        json j;
        j["success"]     = r.success;
        j["exit_code"]   = r.exit_code;
        j["stdout"]      = r.stdout_text;
        j["stderr"]      = r.stderr_text;
        j["duration_ms"] = r.duration_ms;
        j["command"]     = r.command;
        return j.dump();
    }

} // namespace cardinal
