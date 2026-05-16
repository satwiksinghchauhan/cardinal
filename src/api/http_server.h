// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - HTTP Server
// File: src/api/http_server.h
//
// Local HTTP server exposing CardinalAPI to TypeScript (Interface 2)
// and any other external consumers.
//
// Built on cpp-httplib (already linked). All responses are JSON.
// Streaming uses Server-Sent Events (SSE) over chunked HTTP.
//
// Lifecycle:
//   HttpServer server(api);
//   server.start();   // Blocks until stop() is called from another thread
//   server.stop();    // Called from signal handler or shutdown sequence
//
// Authentication:
//   When config.api.auth_enabled is true, all requests must include:
//   Authorization: Bearer <api_key>
//   Requests without a valid key receive 401 Unauthorized.
//   Health check endpoint (/api/health) is always public.
//
// Endpoints:
//   GET  /api/health           -- alive check (always public)
//   POST /api/chat             -- send message, get response or SSE stream
//   POST /api/reset            -- reset session history
//   GET  /api/stats            -- system stats
//   GET  /api/rules            -- rule store contents
//   GET  /api/episodes         -- episode query
//   POST /api/scan             -- run contradiction scan
//   POST /api/maintenance      -- run maintenance cycle
//   GET  /api/settings         -- get current settings
//   POST /api/settings         -- update settings (partial JSON accepted)
//   POST /api/export           -- export training data
//   POST /api/sessions         -- create session
//   DELETE /api/sessions/:id   -- destroy session
//   POST /api/sessions/:id/reset -- reset session
//
// SSE Streaming (POST /api/chat with Accept: text/event-stream):
//   Each token:  data: {"token":"...","is_final":false}\n\n
//   Final token: data: {"token":"","is_final":true,"feeling":{...},"episode_id":"..."}\n\n
//   On error:    data: {"error":"..."}\n\n
// =============================================================================

#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "api/cardinal_api.h"
#include "utils/config_loader.h"
#include "self_model/self_model_types.h"   // SelfImprovementStatus, ReflectionResult

#include <string>
#include <atomic>
#include <thread>
#include <memory>
#include <functional>

// Forward declare httplib types to avoid polluting headers
namespace httplib {
    class Server;
    struct Request;
    struct Response;
}

namespace cardinal {

    // =========================================================================
    // HttpServer
    // =========================================================================
    class HttpServer {
    public:
        explicit HttpServer(CardinalAPI& api, const CardinalConfig& config);
        ~HttpServer();

        // Not copyable
        HttpServer(const HttpServer&) = delete;
        HttpServer& operator=(const HttpServer&) = delete;

        // -- Lifecycle --

        // Start the server. Blocks until stop() is called.
        // Call this from a dedicated thread if you want non-blocking behavior:
        //   std::thread t([&]{ server.start(); });
        // Returns false if server fails to bind to host:port.
        bool start();

        // Stop the server. Safe to call from any thread.
        void stop();

        bool is_running() const { return running_.load(); }

        // -- Config --
        const std::string& host() const { return host_; }
        int                port() const { return port_; }

    private:
        // -- Route registration --
        void register_routes();

        // -- Auth middleware --
        // Returns true if request is authorized (or auth is disabled).
        // Sets response to 401 and returns false if unauthorized.
        bool check_auth(const httplib::Request& req,
            httplib::Response& res) const;

        // -- Route handlers --
        void handle_health(const httplib::Request& req, httplib::Response& res);
        void handle_chat(const httplib::Request& req, httplib::Response& res);
        void handle_reset(const httplib::Request& req, httplib::Response& res);
        void handle_stats(const httplib::Request& req, httplib::Response& res);
        void handle_rules(const httplib::Request& req, httplib::Response& res);
        void handle_episodes(const httplib::Request& req, httplib::Response& res);
        void handle_scan(const httplib::Request& req, httplib::Response& res);
        void handle_maintenance(const httplib::Request& req, httplib::Response& res);
        void handle_get_settings(const httplib::Request& req, httplib::Response& res);
        void handle_post_settings(const httplib::Request& req, httplib::Response& res);
        void handle_export(const httplib::Request& req, httplib::Response& res);
        void handle_create_session(const httplib::Request& req, httplib::Response& res);
        void handle_destroy_session(const httplib::Request& req, httplib::Response& res);
        void handle_reset_session(const httplib::Request& req, httplib::Response& res);

        // v1.4.0 — Self-Improvement
        void handle_self_model(const httplib::Request& req, httplib::Response& res);
        void handle_reflect   (const httplib::Request& req, httplib::Response& res);
        void handle_train     (const httplib::Request& req, httplib::Response& res);

        // -- JSON response helpers --

        // Send a success JSON response
        static void send_ok(httplib::Response& res,
            const std::string& json_body);

        // Send an error JSON response
        static void send_error(httplib::Response& res,
            int                http_code,
            CardinalStatus     status,
            const std::string& message);

        // Serialize CardinalStatus + message to JSON error body
        static std::string error_json(CardinalStatus     status,
            const std::string& message);

        // -- SSE helpers --

        // Format a single SSE event line
        static std::string sse_event(const std::string& json_data);

        // -- JSON serialization helpers --
        static std::string feeling_to_json(const FeelingInfo& f);
        static std::string chat_response_to_json(const ChatResponse& r);
        static std::string stats_to_json(const SystemStats& s);
        static std::string rule_to_json(const RuleInfo& r);
        static std::string episode_to_json(const EpisodeInfo& ep);
        static std::string export_info_to_json(const ExportInfo& info);
        static std::string scan_result_to_json(const ScanResult& r);
        static std::string settings_to_json(const CardinalSettings& s);
        static std::string session_info_to_json(const SessionInfo& info);
        static std::string self_improvement_status_to_json(const SelfImprovementStatus& s);
        static std::string reflection_result_to_json(const ReflectionResult& r);

        // -- Members --
        CardinalAPI& api_;
        const CardinalConfig& config_;
        std::unique_ptr<httplib::Server> server_;

        std::string                host_;
        int                        port_;
        bool                       auth_enabled_;
        std::string                api_key_;

        std::atomic<bool>          running_{ false };
    };

    // =========================================================================
    // HttpServerError
    // =========================================================================
    class HttpServerError : public std::runtime_error {
    public:
        explicit HttpServerError(const std::string& message)
            : std::runtime_error("HttpServerError: " + message) {}
    };

} // namespace cardinal