// =============================================================================
// Cardinal - Tool Executor Implementation
// File: src/tools/tool_executor.cpp
// =============================================================================

#include "tools/tool_executor.h"
#include "tools/builtin/analyze_image.h"
#include "vision/vision_encoder.h"
#include "vision/vision_cache.h"
#include "memory/knowledge_graph.h"
#include "memory/episodic_retriever.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <nlohmann/json.hpp>
#include <httplib.h>

#include <sstream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <regex>
#include <cstdio>
#include <array>
#include <stdexcept>

// muparser for safe math evaluation
#include <muParser.h>

// v1.5.0 — Computer Use + Scheduler tool executors
#include "computer/screen_reader.h"
#include "computer/input_controller.h"
#include "computer/app_controller.h"
#include "computer/browser_controller.h"
#include "computer/shell_executor.h"
#include "computer/file_manager.h"
#include "computer/system_controller.h"
#include "computer/email_controller.h"
#include "computer/atspi_reader.h"
#include "scheduler/scheduler_engine.h"

using json = nlohmann::json;

namespace cardinal {

    namespace {
        // Timestamp helper
        std::string now_ts() {
            return JsonParser::current_timestamp();
        }

        // Elapsed ms since a start point
        int elapsed_ms(std::chrono::steady_clock::time_point start) {
            return static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count());
        }

        // Resolve a path to absolute:
        //   ~/foo  -> /home/user/foo
        //   /foo   -> /foo  (unchanged)
        //   foo    -> foo   (relative, left for weakly_canonical vs CWD)
        std::string resolve_path(const std::string& path) {
            if (path.empty()) return path;
            if (path[0] == '~') {
                const char* home = std::getenv("HOME");
                if (home && path.size() > 1)
                    return std::string(home) + path.substr(1);
                if (home)
                    return std::string(home);
            }
            return path;
        }
    }

    // =========================================================================
    // Constructor
    // =========================================================================

    ToolExecutor::ToolExecutor(const CardinalConfig& config,
                               const ToolRegistry&   registry)
        : config_(config)
        , registry_(registry)
    {}

    // =========================================================================
    // detect_tool_calls
    // Scans model output for <tool_call>...</tool_call> blocks.
    // Falls back to scanning for raw JSON objects matching tool schema.
    // =========================================================================

    ToolCallDetectionResult ToolExecutor::detect_tool_calls(
        const std::string& model_output) const
    {
        ToolCallDetectionResult result;
        result.has_tool_calls = false;

        // Primary: <tool_call>JSON</tool_call> pattern
        std::regex tool_call_re(
            R"(<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>)",
            std::regex::ECMAScript);

        auto begin = std::sregex_iterator(
            model_output.begin(), model_output.end(), tool_call_re);
        auto end = std::sregex_iterator();

        size_t last_pos = 0;
        bool first = true;

        for (auto it = begin; it != end; ++it) {
            const auto& match = *it;

            if (first) {
                result.text_before = model_output.substr(0, match.position());
                first = false;
            }

            std::string json_str = match[1].str();
            last_pos = match.position() + match.length();

            try {
                auto j = json::parse(json_str);

                ToolCall call;
                call.raw_json   = json_str;
                call.call_index = static_cast<int>(result.tool_calls.size());

                if (j.contains("name") && j["name"].is_string()) {
                    call.tool_name = j["name"].get<std::string>();
                } else {
                    continue; // malformed
                }

                if (j.contains("arguments") && j["arguments"].is_object()) {
                    for (auto& [key, val] : j["arguments"].items()) {
                        if (val.is_string())
                            call.arguments[key] = val.get<std::string>();
                        else
                            call.arguments[key] = val.dump();
                    }
                }

                result.tool_calls.push_back(std::move(call));
                result.has_tool_calls = true;

            } catch (const json::exception&) {
                LOG_DEBUG("ToolExecutor: malformed tool call JSON, skipping");
            }
        }

        if (result.has_tool_calls && last_pos < model_output.size()) {
            result.text_after = model_output.substr(last_pos);
        } else if (!result.has_tool_calls) {
            result.text_before = model_output;
        }

        return result;
    }

    // =========================================================================
    // execute
    // =========================================================================

    ToolResult ToolExecutor::execute(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        LOG_INFO("ToolExecutor: executing '" + call.tool_name + "'");

        // Validate
        std::string val_err = registry_.validate_call(call);
        if (!val_err.empty()) {
            return make_error(call, ToolStatus::INVALID_ARGS, val_err);
        }

        // Confirmation check
        if (registry_.requires_confirmation(call.tool_name) && confirmation_cb_) {
            ConfirmationRequest req;
            req.request_id    = JsonParser::generate_id();
            req.tool_name     = call.tool_name;
            req.arguments_json = call.raw_json;
            req.reason        = "Tool '" + call.tool_name +
                                "' requires human confirmation before execution";
            req.timestamp     = now_ts();

            bool approved = confirmation_cb_(req);
            if (!approved) {
                return make_error(call, ToolStatus::CONFIRMATION_REQUIRED,
                                  "Tool execution denied by user");
            }
        }

        return dispatch(call);
    }

    std::vector<ToolResult> ToolExecutor::execute_all(
        const std::vector<ToolCall>& calls) const
    {
        std::vector<ToolResult> results;
        results.reserve(calls.size());
        for (const auto& call : calls) {
            results.push_back(execute(call));
        }
        return results;
    }

    // =========================================================================
    // dispatch
    // =========================================================================

    ToolResult ToolExecutor::dispatch(const ToolCall& call) const {
        const auto& name = call.tool_name;

        if (name == "web_search")            return execute_web_search(call);
        if (name == "web_fetch")             return execute_web_fetch(call);
        if (name == "calculator")            return execute_calculator(call);
        if (name == "run_python")            return execute_run_python(call);
        if (name == "file_read")             return execute_file_read(call);
        if (name == "file_write")            return execute_file_write(call);
        if (name == "knowledge_graph_query") return execute_kg_query(call);
        if (name == "episodic_search")       return execute_episodic_search(call);
        if (name == "analyze_image") {
            if (!vision_encoder_ || !vision_cache_)
                return make_error(call, ToolStatus::FAILURE,
                    "analyze_image: vision subsystem not initialized");
            return execute_analyze_image(call, config_,
                                         *vision_encoder_, *vision_cache_);
        }

        // v1.5.0 — Computer Use tools
        if (name == "screenshot")       return execute_screenshot(call);
        if (name == "click")            return execute_click(call);
        if (name == "type_text")        return execute_type_text(call);
        if (name == "open_app")         return execute_open_app(call);
        if (name == "close_app")        return execute_close_app(call);
        if (name == "browser")          return execute_browser(call);
        if (name == "shell_run")        return execute_shell_run(call);
        if (name == "file_ops")         return execute_file_ops(call);
        if (name == "system_control")   return execute_system_control(call);
        if (name == "email")            return execute_email(call);
        if (name == "watch_screen")     return execute_watch_screen(call);

        // v1.5.0 — Scheduler tool
        if (name == "schedule_task")    return execute_schedule_task(call);

        return make_error(call, ToolStatus::NOT_FOUND,
                          "No executor for tool: " + name);
    }

    // =========================================================================
    // web_search — DuckDuckGo HTML scrape (no API key)
    // =========================================================================

    ToolResult ToolExecutor::execute_web_search(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        std::string query   = get_arg(call, "query");
        int max_results     = std::stoi(
            get_arg(call, "max_results",
                    std::to_string(config_.tools.web_search.max_results)));
        int timeout         = config_.tools.web_search.timeout_seconds;

        if (query.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "web_search: query cannot be empty");

        try {
            // URL-encode query
            std::string encoded;
            for (char c : query) {
                if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
                    encoded += c;
                else
                    encoded += '%' + [c]() {
                        char buf[3];
                        snprintf(buf, sizeof(buf), "%02X",
                                 static_cast<unsigned char>(c));
                        return std::string(buf);
                    }();
            }

            httplib::Client cli("https://html.duckduckgo.com");
            cli.set_connection_timeout(timeout);
            cli.set_read_timeout(timeout);

            auto res = cli.Get("/html/?q=" + encoded);

            if (!res || res->status != 200) {
                return make_error(call, ToolStatus::FAILURE,
                                  "web_search: HTTP request failed");
            }

            // Extract results from DDG HTML response
            // Parse <a class="result__a"> links and snippets
            std::string body = res->body;
            std::regex  title_re(R"REGEX(<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>)REGEX",
                                  std::regex::ECMAScript);
            std::regex  snippet_re(R"REGEX(<a class="result__snippet"[^>]*>(.*?)</a>)REGEX",
                                    std::regex::ECMAScript);

            // Strip HTML tags helper
            auto strip_tags = [](const std::string& html) {
                return std::regex_replace(html, std::regex("<[^>]*>"), "");
            };

            std::ostringstream oss;
            oss << "Search results for: " << query << "\n\n";

            auto title_begin = std::sregex_iterator(body.begin(), body.end(), title_re);
            auto title_end   = std::sregex_iterator();

            int count = 0;
            for (auto it = title_begin; it != title_end && count < max_results;
                 ++it, ++count) {
                const auto& m = *it;
                std::string url   = m[1].str();
                std::string title = strip_tags(m[2].str());

                // Clean DDG redirect URLs
                if (url.find("//duckduckgo.com/l/?uddg=") != std::string::npos) {
                    auto uddg_pos = url.find("uddg=");
                    if (uddg_pos != std::string::npos) {
                        url = url.substr(uddg_pos + 5);
                        // URL decode %xx
                        std::string decoded;
                        for (size_t i = 0; i < url.size(); ++i) {
                            if (url[i] == '%' && i + 2 < url.size()) {
                                int val = std::stoi(url.substr(i+1, 2), nullptr, 16);
                                decoded += static_cast<char>(val);
                                i += 2;
                            } else {
                                decoded += url[i];
                            }
                        }
                        url = decoded;
                    }
                }

                oss << (count + 1) << ". " << title << "\n"
                    << "   URL: " << url << "\n\n";
            }

            if (count == 0) {
                oss << "No results found.";
            }

            return make_success(call, oss.str(), elapsed_ms(start));

        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                              std::string("web_search error: ") + e.what());
        }
    }

    // =========================================================================
    // web_fetch
    // =========================================================================

    ToolResult ToolExecutor::execute_web_fetch(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        std::string url     = get_arg(call, "url");
        int max_kb          = std::stoi(get_arg(call, "max_kb",
            std::to_string(config_.tools.web_fetch.max_content_kb)));
        int timeout         = config_.tools.web_fetch.timeout_seconds;

        if (url.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "web_fetch: url cannot be empty");

        // Check blocked domains
        for (const auto& blocked : config_.tools.web_fetch.blocked_domains) {
            if (url.find(blocked) != std::string::npos)
                return make_error(call, ToolStatus::FAILURE,
                                  "web_fetch: domain is blocked: " + blocked);
        }

        // Check allowed domains (if list is non-empty)
        if (!config_.tools.web_fetch.allowed_domains.empty()) {
            bool allowed = false;
            for (const auto& domain : config_.tools.web_fetch.allowed_domains) {
                if (url.find(domain) != std::string::npos) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed)
                return make_error(call, ToolStatus::FAILURE,
                                  "web_fetch: domain not in allowed list");
        }

        try {
            // Parse URL into host + path
            std::string host, path;
            bool https = url.substr(0, 8) == "https://";
            std::string stripped = url.substr(https ? 8 : 7);
            auto slash = stripped.find('/');
            if (slash == std::string::npos) {
                host = stripped;
                path = "/";
            } else {
                host = stripped.substr(0, slash);
                path = stripped.substr(slash);
            }

            httplib::Client cli((https ? "https://" : "http://") + host);
            cli.set_connection_timeout(timeout);
            cli.set_read_timeout(timeout);
            cli.set_follow_location(true);

            auto res = cli.Get(path);
            if (!res || res->status != 200) {
                return make_error(call, ToolStatus::FAILURE,
                                  "web_fetch: HTTP " +
                                  (res ? std::to_string(res->status) : "failed"));
            }

            std::string body = res->body;

            // Strip HTML tags
            body = std::regex_replace(body, std::regex("<script[^>]*>[\\s\\S]*?</script>"), "");
            body = std::regex_replace(body, std::regex("<style[^>]*>[\\s\\S]*?</style>"), "");
            body = std::regex_replace(body, std::regex("<[^>]*>"), " ");
            body = std::regex_replace(body, std::regex("&nbsp;"), " ");
            body = std::regex_replace(body, std::regex("&amp;"), "&");
            body = std::regex_replace(body, std::regex("&lt;"), "<");
            body = std::regex_replace(body, std::regex("&gt;"), ">");
            body = std::regex_replace(body, std::regex("\\s{2,}"), " ");

            // Trim to max_kb
            size_t max_bytes = static_cast<size_t>(max_kb) * 1024;
            if (body.size() > max_bytes) {
                body = body.substr(0, max_bytes) + "\n[Content truncated]";
            }

            return make_success(call, body, elapsed_ms(start));

        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                              std::string("web_fetch error: ") + e.what());
        }
    }

    // =========================================================================
    // calculator — muparser safe evaluation
    // =========================================================================

    ToolResult ToolExecutor::execute_calculator(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        std::string expr = get_arg(call, "expression");
        if (expr.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "calculator: expression cannot be empty");

        try {
            mu::Parser parser;
            parser.SetExpr(expr);
            double result = parser.Eval();

            std::ostringstream oss;
            oss << expr << " = " << result;

            return make_success(call, oss.str(), elapsed_ms(start));

        } catch (const mu::Parser::exception_type& e) {
            return make_error(call, ToolStatus::FAILURE,
                              "calculator error: " + std::string(e.GetMsg()));
        }
    }

    // =========================================================================
    // run_python — routes to subprocess or docker
    // =========================================================================

    ToolResult ToolExecutor::execute_run_python(const ToolCall& call) const {
        std::string code    = get_arg(call, "code");
        int timeout         = std::stoi(get_arg(call, "timeout_seconds",
            std::to_string(config_.tools.run_python.timeout_seconds)));

        if (code.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "run_python: code cannot be empty");

        const auto& mode = config_.tools.run_python.sandbox_mode;

        if (mode == "docker")
            return run_python_docker(code, timeout);
        else
            return run_python_subprocess(code, timeout);
    }

    ToolResult ToolExecutor::run_python_subprocess(
        const std::string& code, int timeout_seconds) const
    {
        auto start = std::chrono::steady_clock::now();

        try {
            // Write code to temp file
            std::string tmp_path = "/tmp/cardinal_python_" +
                                   JsonParser::generate_id().substr(0, 8) + ".py";

            {
                std::ofstream f(tmp_path);
                if (!f.is_open())
                    return ToolResult{
                        "run_python", {}, ToolStatus::SANDBOX_ERROR,
                        "", "Failed to create temp file", 0, now_ts()
                    };
                f << code;
            }

            // Build command with resource limits
            // ulimit: memory (256MB), CPU time (timeout), no network via unshare
            std::string cmd =
                "timeout " + std::to_string(timeout_seconds) +
                " bash -c 'ulimit -v " +
                std::to_string(config_.tools.run_python.memory_limit_mb * 1024) +
                " && ulimit -t " + std::to_string(timeout_seconds) +
                " && python3 " + tmp_path + " 2>&1'";

            std::array<char, 4096> buf{};
            std::string output;
            FILE* pipe = popen(cmd.c_str(), "r");

            if (!pipe) {
                std::filesystem::remove(tmp_path);
                return ToolResult{
                    "run_python", {}, ToolStatus::SANDBOX_ERROR,
                    "", "Failed to start subprocess", 0, now_ts()
                };
            }

            while (fgets(buf.data(), buf.size(), pipe) != nullptr)
                output += buf.data();

            int exit_code = pclose(pipe);
            std::filesystem::remove(tmp_path);

            // exit code 124 = timeout
            if (exit_code == 124 * 256 || exit_code == 124)
                return make_error(
                    ToolCall{"run_python", {}, "", 0},
                    ToolStatus::TIMEOUT,
                    "Python execution timed out after " +
                    std::to_string(timeout_seconds) + "s");

            return make_success(
                ToolCall{"run_python", {}, "", 0},
                output.empty() ? "(no output)" : output,
                elapsed_ms(start));

        } catch (const std::exception& e) {
            return ToolResult{
                "run_python", {}, ToolStatus::SANDBOX_ERROR,
                "", std::string("subprocess error: ") + e.what(),
                elapsed_ms(start), now_ts()
            };
        }
    }

    ToolResult ToolExecutor::run_python_docker(
        const std::string& code, int timeout_seconds) const
    {
        auto start = std::chrono::steady_clock::now();

        try {
            // Write to temp file accessible to Docker
            std::string tmp_dir  = "/tmp/cardinal_docker_" +
                                    JsonParser::generate_id().substr(0, 8);
            std::string tmp_file = tmp_dir + "/code.py";
            std::filesystem::create_directories(tmp_dir);

            {
                std::ofstream f(tmp_file);
                if (!f.is_open()) {
                    std::filesystem::remove_all(tmp_dir);
                    return ToolResult{
                        "run_python", {}, ToolStatus::SANDBOX_ERROR,
                        "", "Failed to create temp dir", 0, now_ts()
                    };
                }
                f << code;
            }

            std::string mem_limit =
                std::to_string(config_.tools.run_python.memory_limit_mb) + "m";
            std::string network =
                config_.tools.run_python.network_enabled ? "bridge" : "none";

            std::string cmd =
                "docker run --rm "
                "--network=" + network + " "
                "--memory=" + mem_limit + " "
                "--cpus=1 "
                "--security-opt=no-new-privileges "
                "-v " + tmp_dir + ":/code:ro "
                "--timeout=" + std::to_string(timeout_seconds) + " " +
                config_.tools.run_python.docker_image + " "
                "timeout " + std::to_string(timeout_seconds) +
                " python3 /code/code.py 2>&1";

            std::array<char, 4096> buf{};
            std::string output;
            FILE* pipe = popen(cmd.c_str(), "r");

            if (!pipe) {
                std::filesystem::remove_all(tmp_dir);
                return ToolResult{
                    "run_python", {}, ToolStatus::SANDBOX_ERROR,
                    "", "Failed to start Docker container", 0, now_ts()
                };
            }

            while (fgets(buf.data(), buf.size(), pipe) != nullptr)
                output += buf.data();

            int exit_code = pclose(pipe);
            std::filesystem::remove_all(tmp_dir);

            if (exit_code == 124 * 256 || exit_code == 124)
                return make_error(
                    ToolCall{"run_python", {}, "", 0},
                    ToolStatus::TIMEOUT,
                    "Python execution timed out after " +
                    std::to_string(timeout_seconds) + "s");

            return make_success(
                ToolCall{"run_python", {}, "", 0},
                output.empty() ? "(no output)" : output,
                elapsed_ms(start));

        } catch (const std::exception& e) {
            return ToolResult{
                "run_python", {}, ToolStatus::SANDBOX_ERROR,
                "", std::string("docker error: ") + e.what(),
                elapsed_ms(start), now_ts()
            };
        }
    }

    // =========================================================================
    // file_read
    // =========================================================================

    ToolResult ToolExecutor::execute_file_read(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        std::string path   = resolve_path(get_arg(call, "path"));
        int max_kb         = std::stoi(get_arg(call, "max_kb", "512"));

        if (path.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "file_read: path cannot be empty");

        if (!is_path_allowed(path, config_.tools.file_read.allowed_paths))
            return make_error(call, ToolStatus::FAILURE,
                              "file_read: path not in allowed list: " + path);

        if (!std::filesystem::exists(path))
            return make_error(call, ToolStatus::FAILURE,
                              "file_read: file not found: " + path);

        try {
            std::ifstream f(path);
            if (!f.is_open())
                return make_error(call, ToolStatus::FAILURE,
                                  "file_read: cannot open file: " + path);

            std::string content((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());

            size_t max_bytes = static_cast<size_t>(max_kb) * 1024;
            if (content.size() > max_bytes) {
                content = content.substr(0, max_bytes) + "\n[Truncated]";
            }

            return make_success(call, content, elapsed_ms(start));

        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                              std::string("file_read error: ") + e.what());
        }
    }

    // =========================================================================
    // file_write
    // =========================================================================

    ToolResult ToolExecutor::execute_file_write(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        std::string path    = resolve_path(get_arg(call, "path"));
        std::string content = get_arg(call, "content");
        bool append         = get_arg(call, "append", "false") == "true";

        if (path.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "file_write: path cannot be empty");

        if (!is_path_allowed(path, config_.tools.file_write.allowed_paths))
            return make_error(call, ToolStatus::FAILURE,
                              "file_write: path not in allowed list: " + path);

        try {
            // Create parent directories if needed
            auto parent = std::filesystem::path(path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);

            auto mode = append
                ? (std::ios::out | std::ios::app)
                : (std::ios::out | std::ios::trunc);

            std::ofstream f(path, mode);
            if (!f.is_open())
                return make_error(call, ToolStatus::FAILURE,
                                  "file_write: cannot open for writing: " + path);

            f << content;

            std::string msg = "Written " + std::to_string(content.size()) +
                              " bytes to " + path;
            return make_success(call, msg, elapsed_ms(start));

        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                              std::string("file_write error: ") + e.what());
        }
    }

    // =========================================================================
    // knowledge_graph_query
    // =========================================================================

    ToolResult ToolExecutor::execute_kg_query(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        if (!kg_)
            return make_error(call, ToolStatus::FAILURE,
                              "knowledge_graph_query: KG not available");

        std::string query  = get_arg(call, "query");
        std::string domain = get_arg(call, "domain", "");
        int max_results    = std::stoi(get_arg(call, "max_results", "10"));

        if (query.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "knowledge_graph_query: query cannot be empty");

        try {
            GraphQuery gq;
            gq.content_hint    = query;
            gq.label_hint      = query;
            gq.max_results     = max_results;
            if (!domain.empty()) {
                // domain filter maps to label_hint narrowing
                gq.label_hint = domain + " " + query;
            }
            auto results = kg_->query(gq);

            std::ostringstream oss;
            oss << "Knowledge graph results for: " << query << "\n\n";

            if (results.empty()) {
                oss << "No results found.";
            } else {
                for (size_t i = 0; i < results.size(); ++i) {
                    const auto& node = results[i];
                    oss << (i + 1) << ". [" << node_type_to_string(node.type) << "] "
                        << node.data.label << ": "
                        << node.data.content << "\n";
                }
            }

            return make_success(call, oss.str(), elapsed_ms(start));

        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                              std::string("kg_query error: ") + e.what());
        }
    }

    // =========================================================================
    // episodic_search
    // =========================================================================

    ToolResult ToolExecutor::execute_episodic_search(const ToolCall& call) const {
        auto start = std::chrono::steady_clock::now();

        if (!retriever_)
            return make_error(call, ToolStatus::FAILURE,
                              "episodic_search: retriever not available");

        std::string query       = get_arg(call, "query");
        int max_results         = std::stoi(
            get_arg(call, "max_results",
                    std::to_string(config_.tools.episodic_search.max_results)));
        float min_conf          = std::stof(
            get_arg(call, "min_confidence", "0.0"));

        if (query.empty())
            return make_error(call, ToolStatus::INVALID_ARGS,
                              "episodic_search: query cannot be empty");

        try {
            auto results = retriever_->retrieve(query);

            std::ostringstream oss;
            oss << "Episodic memory results for: " << query << "\n\n";

            int shown = 0;
            for (const auto& r : results) {
                if (r.episode.confidence < min_conf) continue;
                if (shown >= max_results) break;

                oss << (shown + 1) << ". ["
                    << r.episode.reasoning_domain
                    << " | confidence: "
                    << static_cast<int>(r.episode.confidence * 100) << "%]\n"
                    << "   Q: " << r.episode.user_message << "\n"
                    << "   A: " << r.episode.response_summary.substr(
                        0, std::min(r.episode.response_summary.size(), size_t(200)))
                    << "\n\n";
                ++shown;
            }

            if (shown == 0) oss << "No relevant episodes found.";

            return make_success(call, oss.str(), elapsed_ms(start));

        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                              std::string("episodic_search error: ") + e.what());
        }
    }

    // =========================================================================
    // format_results_for_context
    // =========================================================================

    std::string ToolExecutor::format_results_for_context(
        const std::vector<ToolResult>& results) const
    {
        if (results.empty()) return "";

        std::ostringstream oss;
        oss << "\n[TOOL RESULTS]\n";

        for (const auto& r : results) {
            oss << "Tool: " << r.tool_name << "\n";
            oss << "Status: " << tool_status_to_string(r.status) << "\n";
            if (r.ok()) {
                oss << "Output:\n" << r.output << "\n";
            } else {
                oss << "Error: " << r.error_message << "\n";
            }
            oss << "---\n";
        }

        oss << "[END TOOL RESULTS]\n";
        return oss.str();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    bool ToolExecutor::is_path_allowed(
        const std::string& path,
        const std::vector<std::string>& allowed_paths) const
    {
        // home_access flag: allow entire home directory (dev convenience)
        if (config_.tools.home_access) {
            const char* home = std::getenv("HOME");
            if (home) {
                std::string resolved = resolve_path(path);
                if (resolved.substr(0, std::string(home).size()) == std::string(home))
                    return true;
            }
        }

        if (allowed_paths.empty()) return false;

        // Resolve ~ and then canonicalize the input path
        std::string resolved_input = resolve_path(path);
        std::filesystem::path canonical_path;
        try {
            canonical_path = std::filesystem::weakly_canonical(resolved_input);
        } catch (...) {
            return false;
        }

        for (const auto& allowed : allowed_paths) {
            try {
                // Also resolve ~ in each allowed path entry
                std::string resolved_allowed = resolve_path(allowed);
                auto canonical_allowed = std::filesystem::weakly_canonical(resolved_allowed);
                auto path_str    = canonical_path.string();
                auto allowed_str = canonical_allowed.string();
                // Ensure allowed_str ends with / so we don't match /home/foo against /home/foobar
                if (allowed_str.back() != '/') allowed_str += '/';
                if (path_str.substr(0, allowed_str.size()) == allowed_str
                    || path_str == allowed_str.substr(0, allowed_str.size() - 1))
                    return true;
            } catch (...) {
                continue;
            }
        }
        return false;
    }

    std::string ToolExecutor::get_arg(const ToolCall& call,
                                       const std::string& key,
                                       const std::string& default_val) const
    {
        auto it = call.arguments.find(key);
        if (it == call.arguments.end()) return default_val;
        // Strip surrounding quotes if present (JSON string values)
        std::string val = it->second;
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        return val;
    }

    ToolResult ToolExecutor::make_error(const ToolCall& call,
                                         ToolStatus status,
                                         const std::string& message) const
    {
        ToolResult r;
        r.tool_name     = call.tool_name;
        r.call          = call;
        r.status        = status;
        r.error_message = message;
        r.timestamp     = now_ts();
        LOG_WARN("ToolExecutor: " + message);
        return r;
    }

    ToolResult ToolExecutor::make_success(const ToolCall& call,
                                           const std::string& output,
                                           int duration_ms) const
    {
        ToolResult r;
        r.tool_name   = call.tool_name;
        r.call        = call;
        r.status      = ToolStatus::SUCCESS;
        r.output      = output;
        r.duration_ms = duration_ms;
        r.timestamp   = now_ts();
        LOG_DEBUG("ToolExecutor: '" + call.tool_name + "' succeeded in " +
                  std::to_string(duration_ms) + "ms");
        return r;
    }

    // =========================================================================
    // v1.5.0 — Computer Use tool implementations
    // =========================================================================

    ToolResult ToolExecutor::execute_screenshot(const ToolCall& call) const {
        if (!screen_reader_)
            return make_error(call, ToolStatus::FAILURE,
                "screenshot: computer use not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            bool analyze = (get_arg(call, "analyze", "true") != "false");
            std::string prompt = get_arg(call, "prompt", "");

            // Optional region
            std::string rx = get_arg(call, "region_x", "");
            Screenshot s;
            if (!rx.empty()) {
                ScreenRegion region;
                region.x      = std::stoi(rx);
                region.y      = std::stoi(get_arg(call, "region_y", "0"));
                region.width  = std::stoi(get_arg(call, "region_w", "800"));
                region.height = std::stoi(get_arg(call, "region_h", "600"));
                s = screen_reader_->capture_region(region, analyze);
            } else {
                s = screen_reader_->capture(analyze);
            }
            if (analyze && !prompt.empty() && s.description.empty())
                s.description = screen_reader_->analyze(s.path, prompt);

            std::string out = "Screenshot saved: " + s.path;
            if (!s.description.empty()) out += "\n\n" + s.description;
            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, out, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "screenshot: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_click(const ToolCall& call) const {
        if (!input_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "click: computer use not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string desc = get_arg(call, "description", "");
            int x = -1, y = -1;
            std::string xs = get_arg(call, "x", "");
            std::string ys = get_arg(call, "y", "");
            if (!xs.empty()) x = std::stoi(xs);
            if (!ys.empty()) y = std::stoi(ys);

            if (!desc.empty() && (x < 0 || y < 0)) {
                if (!screen_reader_)
                    return make_error(call, ToolStatus::FAILURE,
                        "click: screen_reader needed for element lookup");
                auto pt = screen_reader_->find_element(desc);
                if (!pt)
                    return make_error(call, ToolStatus::FAILURE,
                        "click: could not locate element: " + desc);
                x = pt->x; y = pt->y;
            }
            if (x < 0 || y < 0)
                return make_error(call, ToolStatus::FAILURE,
                    "click: provide description or x/y coordinates");

            bool dbl = (get_arg(call, "double_click", "false") == "true");
            std::string btn = get_arg(call, "button", "left");
            MouseButton mb = MouseButton::LEFT;
            if (btn == "right")  mb = MouseButton::RIGHT;
            if (btn == "middle") mb = MouseButton::MIDDLE;

            if (dbl) input_controller_->mouse_click(x, y, mb, 2);
            else     input_controller_->mouse_click(x, y, mb, 1);

            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            std::string out = "Clicked at (" + std::to_string(x) + ", " +
                              std::to_string(y) + ")";
            if (!desc.empty()) out += " [" + desc + "]";
            return make_success(call, out, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "click: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_type_text(const ToolCall& call) const {
        if (!input_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "type_text: computer use not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string text = get_arg(call, "text", "");
            std::string key  = get_arg(call, "key",  "");
            if (text.empty() && key.empty())
                return make_error(call, ToolStatus::FAILURE,
                    "type_text: provide text or key");
            if (!key.empty())  input_controller_->send_key(key);
            if (!text.empty()) input_controller_->type_text(text);
            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call,
                key.empty() ? "Typed: " + text : "Sent key: " + key, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "type_text: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_open_app(const ToolCall& call) const {
        if (!app_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "open_app: computer use not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string app = get_arg(call, "app", "");
            if (app.empty())
                return make_error(call, ToolStatus::FAILURE,
                    "open_app: app name required");
            bool focus = (get_arg(call, "focus", "false") == "true");
            if (focus) {
                auto info = app_controller_->get_app(app);
                if (info) app_controller_->focus_app(app);
                else      app_controller_->open_app(app);
            } else {
                app_controller_->open_app(app);
            }
            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, "Opened: " + app, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "open_app: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_close_app(const ToolCall& call) const {
        if (!app_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "close_app: computer use not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string app = get_arg(call, "app", "");
            if (app.empty())
                return make_error(call, ToolStatus::FAILURE,
                    "close_app: app name required");
            app_controller_->close_app(app);
            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, "Closed: " + app, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "close_app: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_browser(const ToolCall& call) const {
        if (!browser_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "browser: browser controller not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string action   = get_arg(call, "action", "");
            std::string url      = get_arg(call, "url", "");
            std::string selector = get_arg(call, "selector", "");
            std::string text     = get_arg(call, "text", "");
            std::string script   = get_arg(call, "script", "");
            int scroll_y = 0;
            std::string sy = get_arg(call, "scroll_y", "");
            if (!sy.empty()) scroll_y = std::stoi(sy);

            // BrowserController uses BrowserAction enum — build action and call execute()
            BrowserAction ba;
            ba.url         = url;
            ba.selector    = selector;
            ba.text        = text.empty() ? script : text;
            ba.timeout_ms  = 0;

            if      (action == "navigate")    ba.type = BrowserActionType::NAVIGATE;
            else if (action == "click")       ba.type = BrowserActionType::CLICK;
            else if (action == "click_text")  { ba.type = BrowserActionType::CLICK; ba.description = text; }
            else if (action == "type")        ba.type = BrowserActionType::TYPE;
            else if (action == "scroll")      ba.type = BrowserActionType::SCROLL;
            else if (action == "get_content") ba.type = BrowserActionType::GET_CONTENT;
            else if (action == "screenshot")  ba.type = BrowserActionType::SCREENSHOT;
            else if (action == "execute_js")  { ba.type = BrowserActionType::EXECUTE_JS; ba.text = script; }
            else if (action == "new_tab")     ba.type = BrowserActionType::NEW_TAB;
            else if (action == "close_tab")   ba.type = BrowserActionType::CLOSE_TAB;
            else if (action == "back")        ba.type = BrowserActionType::BACK;
            else if (action == "forward")     ba.type = BrowserActionType::FORWARD;
            else if (action == "reload")      ba.type = BrowserActionType::RELOAD;
            else return make_error(call, ToolStatus::FAILURE,
                "browser: unknown action: " + action);

            BrowserResult br = browser_controller_->execute(ba);
            if (!br.success)
                return make_error(call, ToolStatus::FAILURE,
                    "browser: " + br.error_message);
            std::string result = br.content.empty() ? br.url : br.content;
            if (!br.screenshot_path.empty()) result += "\nScreenshot: " + br.screenshot_path;

            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, result, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "browser: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_shell_run(const ToolCall& call) const {
        if (!shell_executor_)
            return make_error(call, ToolStatus::FAILURE,
                "shell_run: shell executor not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string cmd = get_arg(call, "command", "");
            if (cmd.empty())
                return make_error(call, ToolStatus::FAILURE,
                    "shell_run: command required");
            int timeout = 0;
            std::string ts = get_arg(call, "timeout_seconds", "");
            if (!ts.empty()) timeout = std::stoi(ts);
            std::string working_dir = get_arg(call, "working_dir", "");

            auto result = shell_executor_->run(cmd, timeout, working_dir);
            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());

            if (!result.success)
                return make_error(call, ToolStatus::FAILURE,
                    "shell_run: " + result.stderr_text);

            std::string out = result.stdout_text;
            if (!result.stderr_text.empty())
                out += "\n[stderr]: " + result.stderr_text;
            return make_success(call, out, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "shell_run: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_file_ops(const ToolCall& call) const {
        if (!file_manager_)
            return make_error(call, ToolStatus::FAILURE,
                "file_ops: file manager not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string action = get_arg(call, "action", "");
            std::string path   = get_arg(call, "path",   "");
            std::string dest   = get_arg(call, "dest",   "");
            bool recursive = (get_arg(call, "recursive", "false") == "true");

            FileOpResult fr;
            if      (action == "list")   fr = file_manager_->list(path, recursive);
            else if (action == "move")   fr = file_manager_->move(path, dest);
            else if (action == "copy")   fr = file_manager_->copy(path, dest);
            else if (action == "delete") fr = file_manager_->remove(path);
            else if (action == "mkdir")  fr = file_manager_->mkdir(path);
            else if (action == "stat")   fr = file_manager_->stat(path);
            else if (action == "exists") {
                bool ex = file_manager_->exists(path);
                fr.success = true;
                fr.error_message = ex ? "true" : "false";
            }
            else return make_error(call, ToolStatus::FAILURE,
                "file_ops: unknown action: " + action);

            if (!fr.success)
                return make_error(call, ToolStatus::FAILURE,
                    "file_ops: " + fr.error_message);

            // Format result string
            std::string result;
            if (action == "list") {
                result = std::to_string(fr.entries.size()) + " entries:\n";
                for (const auto& e : fr.entries)
                    result += (e.is_dir ? "d " : "f ") + e.name +
                              "  " + e.permissions + "  " + e.modified_at + "\n";
            } else if (action == "exists") {
                result = fr.error_message; // "true" or "false"
            } else if (action == "stat") {
                if (!fr.entries.empty()) {
                    const auto& e = fr.entries[0];
                    result = e.path + "\ntype: " + (e.is_dir ? "directory" : "file") +
                             "\npermissions: " + e.permissions +
                             "\nmodified: " + e.modified_at;
                }
            } else {
                result = "OK: " + (fr.dest_path.empty() ? path : fr.dest_path);
            }

            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, result, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "file_ops: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_system_control(const ToolCall& call) const {
        if (!system_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "system_control: system controller not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string action = get_arg(call, "action", "");
            std::string value  = get_arg(call, "value",  "");

            std::string result;
            if (action == "get_state") {
                SystemState ss = system_controller_->get_state();
                result  = "volume=" + std::to_string(ss.volume_pct) + "%";
                result += " muted=" + std::string(ss.muted ? "true" : "false");
                result += " brightness=" + std::to_string(ss.brightness_pct) + "%";
                result += " wifi=" + std::string(ss.wifi_enabled ? "on" : "off");
                if (!ss.wifi_ssid.empty()) result += "(" + ss.wifi_ssid + ")";
                result += " bluetooth=" + std::string(ss.bluetooth_enabled ? "on" : "off");
            } else if (action == "set_volume") {
                bool ok = system_controller_->set_volume(std::stoi(value));
                result = ok ? "Volume set to " + value + "%" : "set_volume failed";
            } else if (action == "set_mute") {
                bool ok = system_controller_->set_mute(value == "true");
                result = ok ? std::string(value == "true" ? "Muted" : "Unmuted") : "set_mute failed";
            } else if (action == "set_brightness") {
                bool ok = system_controller_->set_brightness(std::stoi(value));
                result = ok ? "Brightness set to " + value + "%" : "set_brightness failed";
            } else if (action == "set_wifi") {
                bool ok = system_controller_->set_wifi(value == "true");
                result = ok ? "WiFi " + std::string(value == "true" ? "enabled" : "disabled") : "set_wifi failed";
            } else if (action == "set_bluetooth") {
                bool ok = system_controller_->set_bluetooth(value == "true");
                result = ok ? "Bluetooth " + std::string(value == "true" ? "enabled" : "disabled") : "set_bluetooth failed";
            } else if (action == "set_notifications") {
                bool ok = system_controller_->set_notifications(value == "true");
                result = ok ? "Notifications " + std::string(value == "true" ? "enabled" : "disabled") : "set_notifications failed";
            } else return make_error(call, ToolStatus::FAILURE,
                "system_control: unknown action: " + action);

            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, result, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "system_control: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_email(const ToolCall& call) const {
        if (!email_controller_)
            return make_error(call, ToolStatus::FAILURE,
                "email: email controller not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string action = get_arg(call, "action", "");
            std::string result;

            if (action == "read") {
                EmailQuery req;
                req.folder           = get_arg(call, "folder", "INBOX");
                req.subject_contains = get_arg(call, "subject", "");
                req.from_contains    = get_arg(call, "from", "");
                req.unread_only      = (get_arg(call, "unread_only", "false") == "true");
                std::string mr       = get_arg(call, "max_results", "10");
                req.max_results      = mr.empty() ? 10 : std::stoi(mr);
                auto msgs = email_controller_->read(req);
                result = std::to_string(msgs.size()) + " message(s):\n";
                for (const auto& m : msgs) {
                    result += "From: " + m.from + "\n";
                    result += "Subject: " + m.subject + "\n";
                    result += "Date: " + m.date + "\n";
                    if (!m.body_text.empty())
                        result += m.body_text.substr(0, 500) + "\n";
                    result += "---\n";
                }
            } else if (action == "send") {
                EmailSendRequest req;
                std::string to_str = get_arg(call, "to", "");
                if (to_str.empty())
                    return make_error(call, ToolStatus::FAILURE,
                        "email: 'to' required for send");
                req.to.push_back(to_str);
                req.subject = get_arg(call, "send_subject", "");
                req.body    = get_arg(call, "body", "");
                bool ok = email_controller_->send(req);
                result = ok ? "Email sent to " + to_str : "send failed";
            } else {
                return make_error(call, ToolStatus::FAILURE,
                    "email: unknown action: " + action);
            }

            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, result, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "email: " + std::string(e.what()));
        }
    }

    ToolResult ToolExecutor::execute_watch_screen(const ToolCall& call) const {
        if (!screen_reader_)
            return make_error(call, ToolStatus::FAILURE,
                "watch_screen: computer use not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string wait_for = get_arg(call, "wait_for", "");
            int timeout = 30;
            int poll    = 2;
            std::string ts = get_arg(call, "timeout_seconds", "");
            std::string ps = get_arg(call, "poll_seconds", "");
            if (!ts.empty()) timeout = std::stoi(ts);
            if (!ps.empty()) poll    = std::stoi(ps);
            bool analyze = (get_arg(call, "analyze", "true") != "false");

            auto s0 = screen_reader_->capture(false);
            auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(timeout);

            while (std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::seconds(poll));
                auto s1 = screen_reader_->capture(false);
                if (s1.path != s0.path) {
                    std::string out = "Screen changed after " +
                        std::to_string(static_cast<int>(
                            std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - t0).count()))
                        + "s";
                    if (analyze) {
                        std::string desc = screen_reader_->analyze(
                            s1.path,
                            wait_for.empty() ? "Describe what changed on the screen"
                                             : "Did this happen: " + wait_for +
                                               "? Describe what you see.");
                        if (!desc.empty()) out += "\n" + desc;
                    }
                    int ms = static_cast<int>(std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count());
                    return make_success(call, out, ms);
                }
                s0 = s1;
            }
            return make_error(call, ToolStatus::TIMEOUT,
                "watch_screen: timed out after " + std::to_string(timeout) + "s");
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "watch_screen: " + std::string(e.what()));
        }
    }

    // =========================================================================
    // v1.5.0 — Scheduler tool implementation
    // =========================================================================

    ToolResult ToolExecutor::execute_schedule_task(const ToolCall& call) const {
        if (!scheduler_)
            return make_error(call, ToolStatus::FAILURE,
                "schedule_task: scheduler not initialised");
        auto t0 = std::chrono::steady_clock::now();
        try {
            std::string action = get_arg(call, "action", "");
            std::string result;

            if (action == "create") {
                std::string desc = get_arg(call, "description", "");
                if (desc.empty())
                    return make_error(call, ToolStatus::FAILURE,
                        "schedule_task: description required for create");
                auto pr = scheduler_->create_task_from_nl(desc, "");
                if (!pr.success) {
                    if (!pr.clarification_needed.empty())
                        result = "Clarification needed: " + pr.clarification_needed;
                    else
                        result = "Failed to parse task: " + pr.error_message;
                } else {
                    result = "Task created: '" + pr.task.name +
                             "' (id=" + pr.task.id + ")";
                }
            } else if (action == "list") {
                auto tasks = scheduler_->list_tasks();
                if (tasks.empty()) {
                    result = "No scheduled tasks.";
                } else {
                    result = std::to_string(tasks.size()) + " task(s):\n";
                    for (const auto& t : tasks)
                        result += "  - [" + t.id.substr(0, 8) + "] " + t.name +
                                  " (" + (t.enabled ? "enabled" : "disabled") + ")\n";
                }
            } else if (action == "enable" || action == "disable") {
                std::string id = get_arg(call, "task_id", "");
                if (id.empty())
                    return make_error(call, ToolStatus::FAILURE,
                        "schedule_task: task_id required for " + action);
                bool ok = (action == "enable") ? scheduler_->enable_task(id)
                                               : scheduler_->disable_task(id);
                result = ok ? "Task " + action + "d: " + id
                            : "Task not found: " + id;
            } else if (action == "delete") {
                std::string id = get_arg(call, "task_id", "");
                if (id.empty())
                    return make_error(call, ToolStatus::FAILURE,
                        "schedule_task: task_id required for delete");
                bool ok = scheduler_->delete_task(id);
                result = ok ? "Task deleted: " + id : "Task not found: " + id;
            } else if (action == "run_now") {
                std::string id = get_arg(call, "task_id", "");
                if (id.empty())
                    return make_error(call, ToolStatus::FAILURE,
                        "schedule_task: task_id required for run_now");
                std::string run_id = scheduler_->run_task_now(id);
                result = "Task dispatched, run_id=" + run_id;
            } else {
                return make_error(call, ToolStatus::FAILURE,
                    "schedule_task: unknown action: " + action);
            }

            int ms = static_cast<int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count());
            return make_success(call, result, ms);
        } catch (const std::exception& e) {
            return make_error(call, ToolStatus::FAILURE,
                "schedule_task: " + std::string(e.what()));
        }
    }


} // namespace cardinal
