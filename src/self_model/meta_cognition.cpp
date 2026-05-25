// =============================================================================
// Cardinal - Meta-Cognition (Layer 2) — Implementation
// File: src/self_model/meta_cognition.cpp
// =============================================================================

#include "self_model/meta_cognition.h"
#include "memory/episodic_storage.h"
#include "memory/rule_store.h"
#include "self_model/self_model.h"
#include "core/llm_backend.h"
#include "core/feeling_output.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

std::string utc_now_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

int elapsed_ms(std::chrono::steady_clock::time_point start) {
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count());
}

} // anonymous namespace

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor
// Member initialisation order must match declaration order in the header:
//   mc_cfg_, config_, storage_, rule_store_, self_model_, backend_,
//   inference_counter_, total_reflections_, total_corrective_rules_,
//   reflect_mutex_, window_mutex_, ts_mutex_, last_reflection_at_,
//   domain_windows_
// ---------------------------------------------------------------------------

MetaCognition::MetaCognition(const CardinalConfig&  config,
                             EpisodicStorage&       storage,
                             RuleStore&             rule_store,
                             SelfModel&             self_model,
                             ILLMBackend&           backend)
    : config_(config)
    , storage_(storage)
    , rule_store_(rule_store)
    , self_model_(self_model)
    , backend_(backend)
{
    const auto& mc = config.self_improvement.meta_cognition;
    mc_cfg_.enabled                           = mc.enabled;
    mc_cfg_.trigger_every_n_inferences        = mc.trigger_every_n_inferences;
    mc_cfg_.trigger_on_contradiction_rate_pct = mc.trigger_on_contradiction_rate_pct;
    mc_cfg_.on_demand_via_api                 = mc.on_demand_via_api;
    mc_cfg_.min_failures_to_reflect           = mc.min_failures_to_reflect;
    mc_cfg_.max_corrective_rules_per_session  = mc.max_corrective_rules_per_session;
    mc_cfg_.corrective_rule_confidence        = mc.corrective_rule_confidence;

    LOG_INFO("MetaCognition: initialised (enabled=" +
             std::string(mc_cfg_.enabled ? "true" : "false") +
             ", trigger_every=" +
             std::to_string(mc_cfg_.trigger_every_n_inferences) + ")");
}

// ---------------------------------------------------------------------------
// on_inference — hot path
// ---------------------------------------------------------------------------

ReflectionResult MetaCognition::on_inference(const std::string& domain,
                                              bool               contradiction,
                                              bool               uncertainty) {
    if (!mc_cfg_.enabled) return {};

    // Update rolling domain window.
    {
        std::lock_guard<std::mutex> wl(window_mutex_);
        auto& w = domain_windows_[domain];
        w.inferences++;
        if (contradiction) w.contradictions++;

        if (w.inferences >= mc_cfg_.trigger_every_n_inferences) {
            w.inferences     = 0;
            w.contradictions = 0;
        }
    }

    int count = ++inference_counter_;
    (void)uncertainty;

    std::string trigger;
    if (should_trigger_by_count()) {
        reset_inference_counter();
        trigger = "scheduled";
    } else if (should_trigger_by_contradiction_rate(domain)) {
        trigger = "contradiction_rate";
    }

    if (trigger.empty()) return {};

    LOG_INFO("MetaCognition: trigger fired (" + trigger +
             ") at inference #" + std::to_string(count));

    return run_reflection(trigger);
}

// ---------------------------------------------------------------------------
// reflect — on-demand API entry point
// ---------------------------------------------------------------------------

ReflectionResult MetaCognition::reflect(const std::string& trigger) {
    if (!mc_cfg_.enabled) {
        ReflectionResult r;
        r.error_message = "MetaCognition is disabled in config";
        return r;
    }
    if (!mc_cfg_.on_demand_via_api && trigger == "manual") {
        ReflectionResult r;
        r.error_message = "on_demand_via_api is disabled in config";
        return r;
    }
    return run_reflection(trigger);
}

// ---------------------------------------------------------------------------
// Trigger predicates
// ---------------------------------------------------------------------------

bool MetaCognition::should_trigger_by_count() const {
    int n = mc_cfg_.trigger_every_n_inferences;
    if (n <= 0) return false;
    return (inference_counter_.load() % n) == 0;
}

bool MetaCognition::should_trigger_by_contradiction_rate(
        const std::string& domain) const {
    float threshold_pct = mc_cfg_.trigger_on_contradiction_rate_pct;
    if (threshold_pct <= 0.0f || threshold_pct > 100.0f) return false;

    std::lock_guard<std::mutex> wl(window_mutex_);
    auto it = domain_windows_.find(domain);
    if (it == domain_windows_.end()) return false;

    const auto& w = it->second;
    if (w.inferences < 5) return false;

    float rate_pct = (static_cast<float>(w.contradictions) /
                      static_cast<float>(w.inferences)) * 100.0f;
    return rate_pct >= threshold_pct;
}

void MetaCognition::reset_inference_counter() {
    inference_counter_.store(0);
}

// ---------------------------------------------------------------------------
// run_reflection — core pass, serialised by reflect_mutex_
// ---------------------------------------------------------------------------

ReflectionResult MetaCognition::run_reflection(const std::string& trigger) {
    std::unique_lock<std::mutex> lock(reflect_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        LOG_INFO("MetaCognition: reflection already in progress, skipping");
        return {};
    }

    auto t_start = std::chrono::steady_clock::now();
    std::string ts = utc_now_str();

    ReflectionResult result;
    result.ran       = true;
    result.trigger   = trigger;
    result.timestamp = ts;

    try {
        // ── Step 1: Query failure episodes ───────────────────────────────────
        auto recent = storage_.get_recent(mc_cfg_.trigger_every_n_inferences * 2);

        std::vector<EpisodeRecord> failures;
        failures.reserve(recent.size());
        for (auto& ep : recent)
            if (ep.contradiction || ep.uncertainty)
                failures.push_back(std::move(ep));

        result.episodes_analyzed = static_cast<int>(recent.size());
        result.failures_analyzed = static_cast<int>(failures.size());

        if (result.failures_analyzed < mc_cfg_.min_failures_to_reflect) {
            LOG_INFO("MetaCognition: only " +
                     std::to_string(result.failures_analyzed) +
                     " failures, below min=" +
                     std::to_string(mc_cfg_.min_failures_to_reflect) +
                     " — skipping");
            result.ran = false;
            return result;
        }

        // ── Step 2: Get self-model snapshot ──────────────────────────────────
        SelfModelSnapshot snapshot = self_model_.get_snapshot();

        // ── Step 3: Build reflection prompt ──────────────────────────────────
        std::string prompt = build_reflection_prompt(failures, snapshot);

        // ── Step 4: Run LLM pass ─────────────────────────────────────────────
        FeelingContext ctx(config_);
        std::vector<ChatMessage> msgs;
        msgs.push_back({ "user", prompt });

        std::string llm_output;

        // TokenCallback: bool(const std::string& token, int token_id, int pos)
        GenerationResult gen = backend_.generate_response(
            ctx, msgs,
            [&llm_output](const std::string& tok, int /*id*/, int /*pos*/) -> bool {
                llm_output += tok;
                return true;
            });

        // GenerationResult::success is a bool — check it directly.
        // The generated text accumulates in llm_output via the callback above.
        if (!gen.success) {
            result.error_message = "LLM generation failed (GenerationResult::success=false)";
            LOG_ERROR("MetaCognition: " + result.error_message);
            result.ran = false;
            return result;
        }

        // ── Step 5: Parse findings ────────────────────────────────────────────
        auto findings = parse_findings(llm_output);
        result.findings = findings;

        // ── Step 6: Commit corrective rules ───────────────────────────────────
        int committed = commit_corrective_rules(findings, ts);
        result.rules_committed = committed;

        // ── Step 7: Persist rule store ────────────────────────────────────────
        if (committed > 0) rule_store_.save();

    } catch (const std::exception& ex) {
        result.error_message = std::string("exception: ") + ex.what();
        LOG_ERROR("MetaCognition: " + result.error_message);
        result.ran = false;
    }

    result.duration_ms = elapsed_ms(t_start);

    ++total_reflections_;
    total_corrective_rules_.fetch_add(result.rules_committed);
    {
        std::lock_guard<std::mutex> tl(ts_mutex_);
        last_reflection_at_ = result.timestamp;
    }

    LOG_INFO("MetaCognition: reflection complete — findings=" +
             std::to_string(result.findings.size()) +
             " rules_committed=" + std::to_string(result.rules_committed) +
             " duration_ms=" + std::to_string(result.duration_ms));

    return result;
}

// ---------------------------------------------------------------------------
// build_reflection_prompt
// ---------------------------------------------------------------------------

std::string MetaCognition::build_reflection_prompt(
        const std::vector<EpisodeRecord>& failures,
        const SelfModelSnapshot&          snapshot) const {
    std::ostringstream oss;

    oss << "You are Cardinal's meta-cognitive reflection module. Analyse recent "
           "inference failures and produce structured corrective insights.\n\n";

    std::string sm = snapshot.format_for_prompt(400);
    if (!sm.empty()) oss << "## Current Self-Model\n" << sm << "\n\n";

    oss << "## Recent Failure Episodes (" << failures.size() << " total)\n";
    int shown = 0;
    for (const auto& ep : failures) {
        if (shown >= 15) {
            oss << "... (" << (failures.size() - shown) << " more not shown)\n";
            break;
        }
        oss << "---\n"
            << "Domain:        " << ep.reasoning_domain << "\n"
            << "Type:          " << ep.reasoning_type   << "\n"
            << "Confidence:    " << ep.confidence        << "\n"
            << "Contradiction: " << (ep.contradiction ? "yes" : "no") << "\n"
            << "Uncertainty:   " << (ep.uncertainty   ? "yes" : "no") << "\n";

        std::string msg = ep.user_message;
        if (msg.size() > 200) msg = msg.substr(0, 200) + "...";
        oss << "Query:         " << msg << "\n";

        std::string resp = ep.response_summary;
        if (resp.size() > 200) resp = resp.substr(0, 200) + "...";
        oss << "Response:      " << resp << "\n";
        ++shown;
    }

    oss << "\n## Instructions\n"
           "Reply ONLY with a JSON array. No preamble, no explanation outside "
           "the JSON. Each object must have exactly these fields:\n"
           "  \"domain\"         — reasoning domain "
               "(factual|ethical|spatial|temporal|social|mathematical)\n"
           "  \"pattern\"        — failure pattern description (max 120 chars)\n"
           "  \"recommendation\" — corrective action (max 200 chars)\n"
           "  \"confidence\"     — your confidence this insight is correct (0.0-1.0)\n\n"
           "If you find no actionable patterns, reply with: []\n";

    return oss.str();
}

// ---------------------------------------------------------------------------
// parse_findings
// ---------------------------------------------------------------------------

std::vector<ReflectionFinding> MetaCognition::parse_findings(
        const std::string& llm_response) const {
    std::vector<ReflectionFinding> findings;

    // Strip Markdown fences if present.
    std::string raw = llm_response;
    {
        auto fence_start = raw.find("```");
        if (fence_start != std::string::npos) {
            auto nl = raw.find('\n', fence_start);
            if (nl != std::string::npos) raw = raw.substr(nl + 1);
            auto close_fence = raw.rfind("```");
            if (close_fence != std::string::npos)
                raw = raw.substr(0, close_fence);
        }
    }

    // Trim whitespace.
    {
        auto start = raw.find_first_not_of(" \t\r\n");
        auto end   = raw.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return findings;
        raw = raw.substr(start, end - start + 1);
    }

    std::string ts = utc_now_str();

    try {
        json arr = json::parse(raw);
        if (!arr.is_array()) {
            LOG_ERROR("MetaCognition: parse_findings — response is not a JSON array");
            return findings;
        }

        for (const auto& item : arr) {
            if (!item.is_object()) continue;

            ReflectionFinding f;
            f.domain         = item.value("domain",         std::string(""));
            f.pattern        = item.value("pattern",        std::string(""));
            f.recommendation = item.value("recommendation", std::string(""));
            f.confidence     = item.value("confidence",     0.0f);
            f.timestamp      = ts;

            if (f.domain.empty() || f.pattern.empty() || f.recommendation.empty()) {
                LOG_DEBUG("MetaCognition: skipping incomplete finding");
                continue;
            }

            f.confidence = std::clamp(f.confidence, 0.0f, 1.0f);
            findings.push_back(std::move(f));
        }

    } catch (const json::parse_error& ex) {
        LOG_ERROR("MetaCognition: JSON parse error — " + std::string(ex.what()) +
                  "\nRaw (first 500): " + raw.substr(0, 500));
    } catch (const std::exception& ex) {
        LOG_ERROR("MetaCognition: parse_findings exception — " + std::string(ex.what()));
    }

    LOG_INFO("MetaCognition: parsed " + std::to_string(findings.size()) + " findings");
    return findings;
}

// ---------------------------------------------------------------------------
// commit_corrective_rules
// ---------------------------------------------------------------------------

int MetaCognition::commit_corrective_rules(
        const std::vector<ReflectionFinding>& findings,
        const std::string&                    reflection_timestamp) {
    int committed = 0;
    int cap       = mc_cfg_.max_corrective_rules_per_session;

    for (const auto& f : findings) {
        if (committed >= cap) {
            LOG_INFO("MetaCognition: reached max_corrective_rules_per_session=" +
                     std::to_string(cap));
            break;
        }

        if (f.confidence < mc_cfg_.corrective_rule_confidence) {
            LOG_DEBUG("MetaCognition: skipping low-confidence finding (" +
                      std::to_string(f.confidence) + ")");
            continue;
        }

        std::string rule_id = rule_store_.add_rule(
            f.domain,
            "[pattern] " + f.pattern,
            "[corrective] " + f.recommendation,
            f.confidence,
            "reflection_" + reflection_timestamp,
            "meta_correction"
        );

        if (rule_id.empty()) {
            LOG_ERROR("MetaCognition: add_rule returned empty ID for domain=" + f.domain);
            continue;
        }

        ++committed;
        LOG_DEBUG("MetaCognition: committed rule " + rule_id +
                  " domain=" + f.domain);
    }

    return committed;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int MetaCognition::total_reflections() const {
    return total_reflections_.load();
}

int MetaCognition::total_corrective_rules() const {
    return total_corrective_rules_.load();
}

std::string MetaCognition::last_reflection_at() const {
    std::lock_guard<std::mutex> tl(ts_mutex_);
    return last_reflection_at_;
}

} // namespace cardinal
