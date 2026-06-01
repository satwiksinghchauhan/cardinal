// =============================================================================
// Cardinal - Main Entry Point (v2.0.0)
// File: src/main.cpp
//
// Changes from v1.5.0:
//   - int main() → int main(int argc, char* argv[])
//   - --voice [mode] flag parsed
//   - Voice subsystem started after api.init() when flag is set
//   - /voice on/off/status/speak commands added
//   - Banner updated with voice commands
//   - Version bumped to 1.6.0
// =============================================================================

#include "api/cardinal_api.h"
#include "api/http_server.h"
#include "utils/config_loader.h"
#include "utils/logger.h"
#include "computer/computer_types.h"

#include <iostream>
#include <string>
#include <thread>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_shutdown_requested{ false };
static void signal_handler(int) { g_shutdown_requested.store(true); }

// =============================================================================
// print_banner
// =============================================================================

static void print_banner(bool http_enabled,
                          const std::string& host, int port,
                          bool voice_mode)
{
    std::cout << "\n";
    std::cout << "  +===========================================+\n";
    std::cout << "  |         C A R D I N A L  v2.0.0           |\n";
    std::cout << "  |    Neurosymbolic AGI Architecture         |\n";
    std::cout << "  +===========================================+\n";
    std::cout << "\n";
    if (http_enabled)
        std::cout << "  HTTP API: http://" << host << ":" << port << "\n"
                  << "  TypeScript bridge: ready\n";
    if (voice_mode)
        std::cout << "  Voice mode: active\n";
    std::cout << "\n";
    std::cout << "  Commands:\n";
    std::cout << "    /exit              -- quit\n";
    std::cout << "    /reset             -- clear conversation history\n";
    std::cout << "    /rules             -- show active rule store\n";
    std::cout << "    /stats             -- show memory and verifier stats\n";
    std::cout << "    /export            -- export training data to JSONL\n";
    std::cout << "    /scan              -- run full contradiction scan\n";
    std::cout << "    /http start        -- start HTTP server\n";
    std::cout << "    /http stop         -- stop HTTP server\n";
    std::cout << "    /self_model        -- show self-improvement status\n";
    std::cout << "    /reflect           -- trigger meta-cognition (Layer 2)\n";
    std::cout << "    /train [domain]    -- trigger LoRA fine-tuning (Layer 3)\n";
    std::cout << "    /scheduler         -- show scheduler status\n";
    std::cout << "    /tasks             -- list scheduled tasks\n";
    std::cout << "    /computer          -- show computer use status\n";
    std::cout << "    /voice on [mode]   -- enable voice (ptt/vad/wake)\n";
    std::cout << "    /voice off         -- disable voice\n";
    std::cout << "    /voice status      -- show voice subsystem status\n";
    std::cout << "    /voice speak <txt> -- test TTS directly\n";
    std::cout << "\n";
}

// =============================================================================
// Command helpers (unchanged from v1.5.0)
// =============================================================================

static void print_stats(cardinal::CardinalAPI& api) {
    auto result = api.get_stats();
    if (!result.ok()) {
        std::cout << "  [stats error: " << result.error_message << "]\n\n";
        return;
    }
    const auto& s = result.value;
    std::cout << "\n  -- Memory --\n";
    std::cout << "  Episodes:       " << s.memory.total_episodes     << "\n";
    std::cout << "  High-conf:      " << s.memory.high_conf_episodes << "\n";
    std::cout << "  Active rules:   " << s.memory.active_rules       << "\n";
    std::cout << "  Index size:     " << s.memory.index_size         << "\n";
    std::cout << "  Vocabulary:     " << s.memory.vocabulary_size    << " terms\n";
    std::cout << "\n  -- Verifier --\n";
    std::cout << "  Checks:         " << s.verifier.total_checks           << "\n";
    std::cout << "  Extracted:      " << s.verifier.total_rules_extracted  << "\n";
    std::cout << "  Contradictions: " << s.verifier.total_contradictions   << "\n";
    std::cout << "  Resolved:       " << s.verifier.total_resolved         << "\n";
    std::cout << "  Flagged:        " << s.verifier.total_flagged          << "\n";
    std::cout << "  Uptime:         " << s.uptime_seconds                  << "\n\n";
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
    std::cout << "\n  -- Rule Store (" << result.value.size() << " rules) --\n";
    for (size_t i = 0; i < result.value.size(); ++i) {
        const auto& r = result.value[i];
        std::cout << "  " << (i+1) << ". [" << r.domain << "] "
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
    req.include_rules  = true;

    auto dry = api.export_dry_run(req);
    if (!dry.ok()) {
        std::cout << "  [export error: " << dry.error_message << "]\n\n";
        return;
    }
    std::cout << "\n  -- Export Preview --\n";
    std::cout << "  Episodes: " << dry.value.episodes_exported << "\n";
    std::cout << "  Rules:    " << dry.value.rules_exported    << "\n";
    std::cout << "  Total:    " << dry.value.total_exported    << "\n";

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

static void print_self_model(cardinal::CardinalAPI& api) {
    auto result = api.get_self_model_status();
    if (!result.ok()) {
        std::cout << "  [self_model error: " << result.error_message << "]\n\n";
        return;
    }
    const auto& s = result.value;
    std::cout << "\n  -- Layer 1: Self-Model --\n";
    std::cout << "  Enabled:          " << (s.self_model_enabled ? "yes" : "no") << "\n";
    if (s.self_model_enabled) {
        std::cout << "  Domain records:   " << s.total_domain_stats  << "\n";
        if (!s.weakest_domain.empty())
            std::cout << "  Weakest domain:   " << s.weakest_domain   << "\n";
        if (!s.strongest_domain.empty())
            std::cout << "  Strongest domain: " << s.strongest_domain << "\n";
    }
    std::cout << "\n  -- Layer 2: Meta-Cognition --\n";
    std::cout << "  Enabled:          " << (s.meta_cognition_enabled ? "yes" : "no") << "\n";
    if (s.meta_cognition_enabled) {
        std::cout << "  Reflections run:  " << s.total_reflections      << "\n";
        std::cout << "  Corrective rules: " << s.total_corrective_rules << "\n";
        if (!s.last_reflection_at.empty())
            std::cout << "  Last reflection:  " << s.last_reflection_at << "\n";
    }
    std::cout << "\n  -- Layer 3: LoRA Training --\n";
    std::cout << "  Enabled:          " << (s.training_enabled ? "yes" : "no") << "\n";
    if (s.training_enabled) {
        std::cout << "  Training runs:    " << s.total_training_runs << "\n";
        if (!s.last_training_at.empty())
            std::cout << "  Last training:    " << s.last_training_at << "\n";
        if (!s.active_adapter_path.empty())
            std::cout << "  Active adapter:   " << s.active_adapter_path << "\n";
        if (s.last_improvement_pct > 0.0f)
            std::cout << "  Last improvement: +" << s.last_improvement_pct << "%\n";
    }
    std::cout << "\n";
}

static void handle_reflect(cardinal::CardinalAPI& api) {
    std::cout << "\n  Running meta-cognition reflection pass...\n";
    auto result = api.reflect();
    if (!result.ok()) {
        std::cout << "  [reflect error: " << result.error_message << "]\n\n";
        return;
    }
    const auto& r = result.value;
    if (!r.ran) {
        std::cout << "  Reflection skipped";
        if (!r.error_message.empty())
            std::cout << ": " << r.error_message;
        std::cout << "\n\n";
        return;
    }
    std::cout << "  Trigger:          " << r.trigger            << "\n";
    std::cout << "  Episodes analyzed:" << r.episodes_analyzed  << "\n";
    std::cout << "  Failures analyzed:" << r.failures_analyzed  << "\n";
    std::cout << "  Findings:         " << r.findings.size()    << "\n";
    std::cout << "  Rules committed:  " << r.rules_committed    << "\n";
    std::cout << "  Duration:         " << r.duration_ms        << "ms\n";
    if (!r.findings.empty()) {
        std::cout << "\n  Findings:\n";
        for (const auto& f : r.findings) {
            std::cout << "  [" << f.domain << "] " << f.pattern << "\n";
            std::cout << "    -> " << f.recommendation << "\n";
        }
    }
    std::cout << "\n";
}

static void handle_train(cardinal::CardinalAPI& api,
                          const std::string& domain_hint)
{
    std::cout << "\n  Queuing LoRA training";
    if (!domain_hint.empty())
        std::cout << " (domain hint: " << domain_hint << ")";
    std::cout << "...\n";
    std::cout << "  Training runs asynchronously in the background.\n";
    std::cout << "  Use /self_model to check status.\n\n";

    auto result = api.trigger_training(domain_hint);
    if (!result.ok()) {
        std::cout << "  [train error: " << result.error_message << "]\n\n";
        return;
    }
    if (result.value) {
        std::cout << "  Training cycle queued successfully.\n\n";
    } else {
        std::cout << "  Training not queued (check config or episode count).\n\n";
    }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // =========================================================================
    // Parse arguments
    // =========================================================================
    bool        voice_mode          = false;
    std::string voice_mode_override;   // "ptt", "vad", or "wake_word"

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--voice") {
            voice_mode = true;
        } else if (arg == "--voice=ptt"  || arg == "--voice=push_to_talk") {
            voice_mode          = true;
            voice_mode_override = "ptt";
        } else if (arg == "--voice=vad") {
            voice_mode          = true;
            voice_mode_override = "vad";
        } else if (arg == "--voice=wake" || arg == "--voice=wake_word") {
            voice_mode          = true;
            voice_mode_override = "wake_word";
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: cardinal [--voice[=ptt|vad|wake]]\n";
            return 0;
        }
    }

    try {
        // =====================================================================
        // 1. Initialize
        // =====================================================================
        cardinal::Logger::instance().init("logs/cardinal.log");
        LOG_INFO("Cardinal v1.6.0 starting...");

        cardinal::CardinalAPI api;
        auto init_result = api.init("config.json");
        if (!init_result.ok()) {
            std::cerr << "\n[FATAL] Init failed: "
                      << init_result.error_message << "\n";
            return 1;
        }

        // =====================================================================
        // 2. Voice subsystem (--voice flag)
        // =====================================================================
        if (voice_mode) {
            LOG_INFO("--voice flag: enabling voice subsystem");
            auto vr = api.enable_voice(voice_mode_override);
            if (!vr.ok()) {
                std::cerr << "\n  [WARNING] Voice failed to start: "
                          << vr.error_message << "\n";
                std::cerr << "  Continuing in text-only mode.\n\n";
                voice_mode = false;
            } else {
                std::cout << "  Voice mode active";
                if (!voice_mode_override.empty())
                    std::cout << " (" << voice_mode_override << ")";
                std::cout << "\n";
            }
        }

        // =====================================================================
        // 3. HTTP server setup
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
                std::cout << "  HTTP server disabled in config.\n\n";
                return;
            }
            http_thread = std::thread([&]() { http_server.start(); });
            http_running = true;
            std::cout << "  HTTP server started on "
                      << config.api.host << ":" << config.api.port << "\n\n";
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

        if (config.api.http_enabled) start_http();

        // =====================================================================
        // 4. Interactive loop
        // =====================================================================
        print_banner(config.api.http_enabled, config.api.host, config.api.port,
                     voice_mode);

        const std::string session_id = "default";
        std::cout << "  Cardinal is ready. Type your message.\n\n";

        while (!g_shutdown_requested.load()) {
            std::cout << "You: ";
            std::cout.flush();

            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;

            // Trim
            auto s = user_input.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            auto e = user_input.find_last_not_of(" \t\r\n");
            user_input = user_input.substr(s, e - s + 1);
            if (user_input.empty()) continue;

            // ------------------------------------------------------------------
            // Commands
            // ------------------------------------------------------------------
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
                    std::cout << "  Contradictions: "  << result.value.total_contradictions
                              << " | Resolved: "       << result.value.resolved
                              << " | Flagged: "        << result.value.flagged
                              << "\n\n";
                }
                continue;
            }
            if (user_input == "/http start") { start_http(); continue; }
            if (user_input == "/http stop")  { stop_http();  continue; }

            // v1.4.0 commands
            if (user_input == "/self_model") {
                print_self_model(api);
                continue;
            }
            if (user_input == "/reflect") {
                handle_reflect(api);
                continue;
            }
            if (user_input == "/train" ||
                user_input.substr(0, 7) == "/train ")
            {
                std::string domain_hint;
                if (user_input.size() > 7)
                    domain_hint = user_input.substr(7);
                handle_train(api, domain_hint);
                api.on_session_boundary();
                continue;
            }

            // v1.5.0 commands
            if (user_input == "/scheduler") {
                auto result = api.get_scheduler_status();
                if (!result.ok()) {
                    std::cout << "  [scheduler error: " << result.error_message << "]\n\n";
                } else {
                    const auto& sc = result.value;
                    std::cout << "\n  -- Scheduler --\n";
                    std::cout << "  Running:  " << (sc.running ? "yes" : "no") << "\n";
                    std::cout << "  Tasks:    " << sc.total_tasks
                              << " (" << sc.enabled_tasks << " enabled)\n";
                    std::cout << "  Runs:     " << sc.total_runs
                              << " | ok=" << sc.successful_runs
                              << " | fail=" << sc.failed_runs << "\n";
                    if (!sc.current_task_name.empty())
                        std::cout << "  Active:   " << sc.current_task_name << "\n";
                    if (!sc.last_run_at.empty())
                        std::cout << "  Last run: " << sc.last_run_at << "\n";
                    if (!sc.next_scheduled_at.empty())
                        std::cout << "  Next:     " << sc.next_scheduled_at << "\n";
                    std::cout << "\n";
                }
                continue;
            }
            if (user_input == "/tasks") {
                auto result = api.list_tasks();
                if (!result.ok()) {
                    std::cout << "  [tasks error: " << result.error_message << "]\n\n";
                } else if (result.value.empty()) {
                    std::cout << "\n  No scheduled tasks.\n\n";
                } else {
                    std::cout << "\n  -- Scheduled Tasks (" << result.value.size() << ") --\n";
                    for (const auto& t : result.value) {
                        std::cout << "  [" << t.id.substr(0, 8) << "] "
                                  << (t.enabled ? "ON " : "OFF") << "  "
                                  << t.name;
                        if (!t.last_run_at.empty())
                            std::cout << "  (last: " << t.last_run_at << ")";
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
                continue;
            }
            if (user_input == "/computer") {
                auto result = api.get_computer_status();
                if (!result.ok()) {
                    std::cout << "  [computer error: " << result.message << "]\n\n";
                } else {
                    const auto& cs = result.value;
                    std::cout << "\n  -- Computer Use --\n";
                    std::cout << "  Display: " << cs.display_var << "\n";
                    std::cout << "  Server:  " << display_server_to_string(cs.server) << "\n";
                    std::cout << "  Size:    " << cs.width << "x" << cs.height << "\n\n";
                }
                continue;
            }

            // v1.6.0 commands — voice
            if (user_input == "/voice status") {
                auto vs = api.get_voice_status();
                std::cout << "\n  -- Voice Subsystem --\n";
                std::cout << "  Active:      " << (vs.active ? "yes" : "no") << "\n";
                if (vs.active) {
                    std::cout << "  Mode:        " << vs.input_mode    << "\n";
                    std::cout << "  State:       " << vs.current_state << "\n";
                    std::cout << "  STT ready:   " << (vs.stt_ready       ? "yes" : "no") << "\n";
                    std::cout << "  TTS ready:   " << (vs.tts_ready       ? "yes" : "no") << "\n";
                    std::cout << "  Wake ready:  " << (vs.wake_word_ready  ? "yes" : "no") << "\n";
                    std::cout << "  Transcripts: " << vs.transcriptions << "\n";
                    std::cout << "  Utterances:  " << vs.utterances     << "\n";
                }
                std::cout << "\n";
                continue;
            }

            if (user_input == "/voice off") {
                auto r = api.disable_voice();
                if (!r.ok())
                    std::cout << "  [voice error: " << r.error_message << "]\n\n";
                else
                    std::cout << "  Voice subsystem disabled.\n\n";
                continue;
            }

            if (user_input.substr(0, 9) == "/voice on") {
                std::string mode_hint;
                if (user_input.size() > 10) {
                    mode_hint = user_input.substr(10);
                    // trim leading spaces
                    auto lt = mode_hint.find_first_not_of(" \t");
                    if (lt != std::string::npos) mode_hint = mode_hint.substr(lt);
                }
                auto r = api.enable_voice(mode_hint);
                if (!r.ok()) {
                    std::cout << "  [voice error: " << r.error_message << "]\n\n";
                } else {
                    std::cout << "  Voice subsystem enabled";
                    if (!mode_hint.empty()) std::cout << " (" << mode_hint << ")";
                    std::cout << ".\n\n";
                }
                continue;
            }

            if (user_input.size() >= 13 &&
                user_input.substr(0, 12) == "/voice speak")
            {
                std::string text;
                if (user_input.size() > 13) text = user_input.substr(13);
                if (text.empty()) {
                    std::cout << "  Usage: /voice speak <text>\n\n";
                } else {
                    auto r = api.voice_speak(text);
                    if (!r.ok())
                        std::cout << "  [voice error: " << r.error_message << "]\n\n";
                    else
                        std::cout << "  Spoken (" << r.value.duration_ms << "ms synthesis).\n\n";
                }
                continue;
            }

            // ------------------------------------------------------------------
            // Inference
            // ------------------------------------------------------------------
            std::cout << "\nCardinal: ";
            std::cout.flush();

            cardinal::ApiStreamCallback stream_cb =
                [](const cardinal::StreamToken& token) -> bool {
                    if (!token.is_final)
                        std::cout << token.token << std::flush;
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
                      << " | conf=" << static_cast<int>(r.feeling.confidence * 100) << "%";
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
        // 5. Clean shutdown
        // =====================================================================
        api.on_session_boundary();

        if (api.is_voice_active())
            api.disable_voice();

        if (http_running) stop_http();
        api.shutdown();
        LOG_INFO("Cardinal shutdown complete");
    }
    catch (const std::exception& ex) {
        std::cerr << "\n[FATAL] " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
