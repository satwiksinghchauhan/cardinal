// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Main Entry Point
// File: src/main.cpp
// Phase 6 -- CardinalAPI facade with interactive loop and HTTP server
//
// Startup sequence:
//   1. CardinalAPI::init() -- constructs and initializes all components
//   2. Optional HTTP server start (explicit, in background thread)
//   3. Interactive loop
//   4. CardinalAPI::shutdown() on exit
// =============================================================================

#include "api/cardinal_api.h"
#include "api/http_server.h"
#include "utils/config_loader.h"
#include "utils/logger.h"

#include <iostream>
#include <string>
#include <thread>
#include <csignal>
#include <atomic>

// =============================================================================
// Signal handling for clean shutdown
// =============================================================================

static std::atomic<bool> g_shutdown_requested{ false };

static void signal_handler(int) {
    g_shutdown_requested.store(true);
}

// =============================================================================
// print_banner
// =============================================================================

static void print_banner(bool http_enabled,
    const std::string& host, int port) {
    std::cout << "\n";
    std::cout << "  +===========================================+\n";
    std::cout << "  |         C A R D I N A L  v1.3.0           |\n";
    std::cout << "  |    Neurosymbolic AGI Architecture         |\n";
    std::cout << "  +===========================================+\n";
    std::cout << "\n";
    if (http_enabled) {
        std::cout << "  HTTP API: http://" << host << ":" << port << "\n";
        std::cout << "  TypeScript bridge: ready\n";
    }
    std::cout << "\n";
    std::cout << "  Commands:\n";
    std::cout << "    /exit        -- quit\n";
    std::cout << "    /reset       -- clear conversation history\n";
    std::cout << "    /rules       -- show active rule store\n";
    std::cout << "    /stats       -- show memory and verifier stats\n";
    std::cout << "    /export      -- export training data to JSONL\n";
    std::cout << "    /scan        -- run full contradiction scan\n";
    std::cout << "    /http start  -- start HTTP server\n";
    std::cout << "    /http stop   -- stop HTTP server\n";
    std::cout << "\n";
}

// =============================================================================
// Command helpers
// =============================================================================

static void print_stats(cardinal::CardinalAPI& api) {
    auto result = api.get_stats();
    if (!result.ok()) {
        std::cout << "  [stats error: " << result.error_message << "]\n\n";
        return;
    }
    const auto& s = result.value;
    std::cout << "\n  -- Memory --\n";
    std::cout << "  Episodes:      " << s.memory.total_episodes << "\n";
    std::cout << "  High-conf:     " << s.memory.high_conf_episodes << "\n";
    std::cout << "  Active rules:  " << s.memory.active_rules << "\n";
    std::cout << "  Index size:    " << s.memory.index_size << "\n";
    std::cout << "  Vocabulary:    " << s.memory.vocabulary_size
        << " terms\n";
    std::cout << "\n  -- Verifier --\n";
    std::cout << "  Checks:        " << s.verifier.total_checks << "\n";
    std::cout << "  Extracted:     " << s.verifier.total_rules_extracted << "\n";
    std::cout << "  Contradictions:" << s.verifier.total_contradictions << "\n";
    std::cout << "  Resolved:      " << s.verifier.total_resolved << "\n";
    std::cout << "  Flagged:       " << s.verifier.total_flagged << "\n";
    std::cout << "  Uptime:        " << s.uptime_seconds << "\n\n";
}

static void print_rules(cardinal::CardinalAPI& api) {
    auto result = api.get_rules();
    if (!result.ok()) {
        std::cout << "  [rules error: " << result.error_message << "]\n\n";
        return;
    }
    if (result.value.empty()) {
        std::cout << "\n  No rules in store.\n\n";
        return;
    }
    std::cout << "\n  -- Rule Store (" << result.value.size()
        << " rules) --\n";
    for (size_t i = 0; i < result.value.size(); ++i) {
        const auto& r = result.value[i];
        std::cout << "  " << (i + 1) << ". [" << r.domain << "] "
            << "conf=" << static_cast<int>(r.confidence * 100) << "%";
        if (r.has_provenance)
            std::cout << " ep=" << r.episode_id.substr(0, 12) << "...";
        std::cout << "\n";
        std::cout << "     IF:   "
            << r.condition.substr(0, 80)
            << (r.condition.size() > 80 ? "..." : "") << "\n";
        std::cout << "     THEN: "
            << r.consequence.substr(0, 80)
            << (r.consequence.size() > 80 ? "..." : "") << "\n";
    }
    std::cout << "\n";
}

static void handle_export(cardinal::CardinalAPI& api) {
    cardinal::ExportRequest req;
    req.min_confidence = 0.7f;
    req.include_rules = true;

    auto dry = api.export_dry_run(req);
    if (!dry.ok()) {
        std::cout << "  [export error: " << dry.error_message << "]\n\n";
        return;
    }

    std::cout << "\n  -- Export Preview --\n";
    std::cout << "  Episodes: " << dry.value.episodes_exported << "\n";
    std::cout << "  Rules:    " << dry.value.rules_exported << "\n";
    std::cout << "  Total:    " << dry.value.total_exported << "\n";

    if (dry.value.total_exported == 0) {
        std::cout << "  Nothing to export yet.\n\n";
        return;
    }

    std::cout << "\n  Export to data/training_export.jsonl? (y/n): ";
    std::string answer;
    std::getline(std::cin, answer);

    if (answer != "y" && answer != "Y") {
        std::cout << "  Export cancelled.\n\n";
        return;
    }

    req.output_path = "data/training_export.jsonl";
    auto result = api.export_training_data(req);
    if (!result.ok()) {
        std::cout << "  Export failed: " << result.error_message << "\n\n";
        return;
    }
    std::cout << "  Exported " << result.value.total_exported
        << " examples to " << result.value.output_path << "\n\n";
}

// =============================================================================
// main
// =============================================================================

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // =====================================================================
        // 1. Initialize
        // =====================================================================
        cardinal::Logger::instance().init("logs/cardinal.log");
        LOG_INFO("Cardinal v0.6 starting...");

        cardinal::CardinalAPI api;
        auto init_result = api.init("config.json");

        if (!init_result.ok()) {
            std::cerr << "\n[FATAL] Init failed: "
                << init_result.error_message << "\n";
            return 1;
        }

        // =====================================================================
        // 2. HTTP server setup
        // =====================================================================
        auto config = cardinal::ConfigLoader::load("config.json");
        cardinal::HttpServer http_server(api, config);
        std::thread http_thread;
        bool http_running = false;

        auto start_http = [&]() {
            if (http_running) {
                std::cout << "  HTTP server already running.\n\n";
                return;
            }
            if (!config.api.http_enabled) {
                std::cout << "  HTTP server disabled in config "
                    "(set api.http_enabled=true to enable).\n\n";
                return;
            }
            http_thread = std::thread([&]() {
                http_server.start();
                });
            http_running = true;
            std::cout << "  HTTP server started on "
                << config.api.host << ":"
                << config.api.port << "\n\n";
            };

        auto stop_http = [&]() {
            if (!http_running) {
                std::cout << "  HTTP server not running.\n\n";
                return;
            }
            http_server.stop();
            if (http_thread.joinable()) http_thread.join();
            http_running = false;
            std::cout << "  HTTP server stopped.\n\n";
            };

        // Auto-start if enabled in config
        if (config.api.http_enabled) {
            start_http();
        }

        // =====================================================================
        // 3. Interactive loop
        // =====================================================================
        print_banner(config.api.http_enabled,
            config.api.host, config.api.port);

        const std::string session_id = "default";
        std::cout << "  Cardinal is ready. Type your message.\n\n";

        while (!g_shutdown_requested.load()) {
            std::cout << "You: ";
            std::cout.flush();

            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;

            // Trim
            size_t s = user_input.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            size_t e = user_input.find_last_not_of(" \t\r\n");
            user_input = user_input.substr(s, e - s + 1);
            if (user_input.empty()) continue;

            // -- Commands --
            if (user_input == "/exit") {
                std::cout << "\n  Cardinal: Farewell.\n\n";
                break;
            }
            if (user_input == "/reset") {
                api.reset_session(session_id);
                std::cout << "\n  [Conversation history cleared]\n\n";
                continue;
            }
            if (user_input == "/rules") {
                print_rules(api);
                continue;
            }
            if (user_input == "/stats") {
                print_stats(api);
                continue;
            }
            if (user_input == "/export") {
                handle_export(api);
                continue;
            }
            if (user_input == "/scan") {
                std::cout << "\n  Running full contradiction scan...\n";
                auto result = api.run_scan();
                if (result.ok()) {
                    std::cout << "  Contradictions: "
                        << result.value.total_contradictions
                        << " | Resolved: " << result.value.resolved
                        << " | Flagged: " << result.value.flagged
                        << "\n\n";
                }
                continue;
            }
            if (user_input == "/http start") {
                start_http();
                continue;
            }
            if (user_input == "/http stop") {
                stop_http();
                continue;
            }

            // -- Inference --
            std::cout << "\nCardinal: ";
            std::cout.flush();

            cardinal::ApiStreamCallback stream_cb =
                [](const cardinal::StreamToken& token) -> bool {
                if (!token.is_final) {
                    std::cout << token.token << std::flush;
                }
                return true;
                };

            auto result = api.chat_stream(session_id, user_input, stream_cb);

            std::cout << "\n\n";

            if (!result.ok()) {
                std::cout << "  [Error: " << result.error_message << "]\n\n";
                continue;
            }

            const auto& r = result.value;

            // Feeling summary line
            std::cout << "  [" << r.feeling.reasoning_domain
                << " | " << r.feeling.reasoning_type
                << " | conf="
                << static_cast<int>(r.feeling.confidence * 100) << "%";
            if (r.feeling.uncertainty_flag)   std::cout << " | uncertain";
            if (r.feeling.contradiction_flag) std::cout << " | contradiction";
            if (r.feeling.rule_candidate)     std::cout << " | rule_candidate";
            if (r.rule_committed)
                std::cout << " | rule=" << r.committed_rule_id;
            if (r.contradictions_resolved > 0)
                std::cout << " | resolved=" << r.contradictions_resolved;
            if (r.contradictions_flagged > 0)
                std::cout << " | flagged=" << r.contradictions_flagged;
            std::cout << "]\n\n";
        }

        // =====================================================================
        // 4. Clean shutdown
        // =====================================================================
        if (http_running) stop_http();
        api.shutdown();
        LOG_INFO("Cardinal shutdown complete");
    }
    catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
