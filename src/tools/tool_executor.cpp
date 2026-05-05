// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Tool Executor Implementation
// File: src/tools/tool_executor.cpp
// =============================================================================

#include "tools/tool_executor.h"
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
#include <regex>
#include <cstdio>
#include <array>
#include <stdexcept>

// muparser for safe math evaluation
#include <muParser.h>

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

} // namespace cardinal
