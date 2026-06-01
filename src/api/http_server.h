#pragma once
// =============================================================================
// Cardinal - HTTP Server (v1.6.0)
// File: src/api/http_server.h
//
// Changes from v1.5.0:
//   - Voice endpoints (5 routes)
//   - voice_status_to_json() serializer
//   - handle_voice_* handler declarations
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
#include "self_model/self_model_types.h"
#include "scheduler/scheduler_types.h"
#include "computer/computer_types.h"
#include "voice/voice_types.h"           // ← new in v1.6.0

#include <string>
#include <atomic>
#include <thread>
#include <memory>
#include <functional>

namespace httplib {
    class Server;
    struct Request;
    struct Response;
}

namespace cardinal {

    class HttpServer {
    public:
        explicit HttpServer(CardinalAPI& api, const CardinalConfig& config);
        ~HttpServer();

        HttpServer(const HttpServer&)            = delete;
        HttpServer& operator=(const HttpServer&) = delete;

        bool start();
        void stop();
        bool is_running() const { return running_.load(); }

        const std::string& host() const { return host_; }
        int                port() const { return port_; }

    private:
        void register_routes();
        bool check_auth(const httplib::Request& req, httplib::Response& res) const;

        // -- Existing route handlers (v1.4.0) --
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
        void handle_self_model(const httplib::Request& req, httplib::Response& res);
        void handle_reflect(const httplib::Request& req, httplib::Response& res);
        void handle_train(const httplib::Request& req, httplib::Response& res);

        // -- Scheduler handlers (v1.5.0) --
        void handle_scheduler_status  (const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_tasks_list(const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_create(const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_get  (const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_put  (const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_delete(const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_run  (const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_enable(const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_disable(const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_task_history(const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_runs_list (const httplib::Request& req, httplib::Response& res);
        void handle_scheduler_run_actions(const httplib::Request& req, httplib::Response& res);

        // -- Computer use handlers (v1.5.0) --
        void handle_computer_status    (const httplib::Request& req, httplib::Response& res);
        void handle_computer_screenshot(const httplib::Request& req, httplib::Response& res);
        void handle_computer_click     (const httplib::Request& req, httplib::Response& res);
        void handle_computer_type      (const httplib::Request& req, httplib::Response& res);
        void handle_computer_shell     (const httplib::Request& req, httplib::Response& res);

        // -- Voice handlers (v1.6.0) --
        void handle_voice_status    (const httplib::Request& req, httplib::Response& res);
        void handle_voice_enable    (const httplib::Request& req, httplib::Response& res);
        void handle_voice_disable   (const httplib::Request& req, httplib::Response& res);
        void handle_voice_speak     (const httplib::Request& req, httplib::Response& res);
        void handle_voice_transcribe(const httplib::Request& req, httplib::Response& res);

        // -- JSON response helpers --
        static void        send_ok(httplib::Response& res, const std::string& json_body);
        static void        send_error(httplib::Response& res, int http_code,
                                      CardinalStatus status, const std::string& message);
        static std::string error_json(CardinalStatus status, const std::string& message);
        static std::string sse_event(const std::string& json_data);

        // -- Serializers (v1.4.0) --
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

        // -- Serializers (v1.5.0) --
        static std::string scheduler_status_to_json(const SchedulerStatus& s);
        static std::string scheduled_task_to_json(const ScheduledTask& t);
        static std::string task_run_to_json(const TaskRun& r);
        static std::string task_action_log_to_json(const TaskActionLog& e);
        static std::string task_parse_result_to_json(const TaskParseResult& r);
        static std::string screen_info_to_json(const ScreenInfo& s);
        static std::string screenshot_to_json(const Screenshot& s);
        static std::string shell_result_to_json(const ShellResult& r);
        static std::string trigger_spec_to_json(const TriggerSpec& t);
        static std::string task_action_to_json(const TaskAction& a);
        static TriggerSpec  trigger_spec_from_json(const std::string& json);
        static TaskAction   task_action_from_json(const std::string& json);

        // -- Serializers (v1.6.0) --
        static std::string voice_status_to_json(const VoiceStatus& s);

        // -- Members --
        CardinalAPI&          api_;
        const CardinalConfig& config_;
        std::unique_ptr<httplib::Server> server_;

        std::string       host_;
        int               port_;
        bool              auth_enabled_;
        std::string       api_key_;
        std::atomic<bool> running_{ false };
    };

    class HttpServerError : public std::runtime_error {
    public:
        explicit HttpServerError(const std::string& message)
            : std::runtime_error("HttpServerError: " + message) {}
    };

} // namespace cardinal
