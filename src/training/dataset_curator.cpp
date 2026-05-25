// =============================================================================
// Cardinal - Dataset Curator — Implementation
// File: src/training/dataset_curator.cpp
// =============================================================================

#include "training/dataset_curator.h"
#include "memory/episodic_storage.h"
#include "memory/rule_store.h"
#include "utils/logger.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <iomanip>
#include <optional>
#include <random>
#include <sstream>

namespace {

std::string utc_now_str() {
    auto now  = std::chrono::system_clock::now();
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

} // anonymous namespace

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DatasetCurator::DatasetCurator(const CardinalConfig& config,
                               EpisodicStorage&      storage,
                               RuleStore&            rule_store)
    : storage_(storage)
    , rule_store_(rule_store)
{
    const auto& tc = config.self_improvement.training;
    cfg_.eval_holdout_episodes  = tc.eval_holdout_episodes;
    cfg_.max_examples           = tc.max_examples;
    cfg_.min_quality_confidence = tc.min_quality_confidence;
    cfg_.include_rules          = true;   // always on; individual rules still filtered
    cfg_.max_rule_examples      = 50;

    LOG_INFO("DatasetCurator: initialised (holdout=" +
             std::to_string(cfg_.eval_holdout_episodes) +
             ", min_conf=" + std::to_string(cfg_.min_quality_confidence) + ")");
}

// ---------------------------------------------------------------------------
// curate() — main entry point
// ---------------------------------------------------------------------------

TrainingDataset DatasetCurator::curate(const CurriculumPlan& plan,
                                        CurationStats&        stats_out) const {
    TrainingDataset dataset;
    dataset.domain_focus = plan.target_domain;
    dataset.created_at   = utc_now_str();

    CurationStats stats;

    try {
        // ── Step 1: Query episodes ────────────────────────────────────────────
        // Fetch a generous batch; we'll filter down in subsequent steps.
        // Use the plan's min_confidence (may be lower than global default
        // for failure-inclusive training runs).
        EpisodeQuery q;
        q.domain         = plan.target_domain;
        q.min_confidence = plan.min_episode_confidence;
        q.recent_first   = true;

        // If plan caps examples, fetch 3× to allow for dedup / holdout losses.
        int fetch_limit = (plan.max_episodes > 0)
            ? plan.max_episodes * 3
            : 2000;
        q.max_results = fetch_limit;

        auto episodes = storage_.query(q);
        stats.total_queried = static_cast<int>(episodes.size());

        if (episodes.empty()) {
            LOG_INFO("DatasetCurator: no episodes found for domain='" +
                     plan.target_domain + "' min_conf=" +
                     std::to_string(plan.min_episode_confidence));
            stats_out = stats;
            return dataset;
        }

        // ── Step 2: Reserve eval holdout ──────────────────────────────────────
        // The most-recent N episodes are held out for evaluation.
        // They are already at the front because recent_first=true.
        int holdout_n = std::min(cfg_.eval_holdout_episodes,
                                 static_cast<int>(episodes.size()));
        std::unordered_set<std::string> holdout_ids;
        holdout_ids.reserve(static_cast<std::size_t>(holdout_n));
        for (int i = 0; i < holdout_n; ++i) {
            holdout_ids.insert(episodes[static_cast<std::size_t>(i)].id);
        }
        stats.holdout_reserved = holdout_n;

        // ── Step 3: Quality filter + dedup ────────────────────────────────────
        std::unordered_set<std::size_t> seen_hashes;
        std::vector<TrainingExample>    examples;

        // Effective confidence threshold: plan may lower it below the global
        // min for failure-inclusive cycles; global min is the hard floor.
        float conf_floor = std::max(plan.min_episode_confidence,
                                    cfg_.min_quality_confidence * 0.5f);

        for (const auto& ep : episodes) {
            // Skip holdout.
            if (holdout_ids.count(ep.id)) continue;

            // Confidence filter.
            if (ep.confidence < conf_floor) continue;

            // Completeness filter.
            if (ep.user_message.empty() || ep.response_summary.empty()) continue;

            // Dedup on user_message hash.
            std::size_t h = message_hash(ep.user_message);
            if (seen_hashes.count(h)) { ++stats.deduped; continue; }
            seen_hashes.insert(h);

            auto maybe_ex = episode_to_example(ep);
            if (!maybe_ex) continue;

            examples.push_back(std::move(*maybe_ex));
            ++stats.passed_quality;

            // Respect plan and global max-examples caps.
            int cap = (plan.max_episodes > 0) ? plan.max_episodes : cfg_.max_examples;
            if (cap > 0 && static_cast<int>(examples.size()) >= cap) break;
        }

        // ── Step 4: Rule-augmented examples ──────────────────────────────────
        if (cfg_.include_rules) {
            auto rule_exs = build_rule_examples(plan.target_domain,
                                                cfg_.max_rule_examples);
            stats.rule_examples_added = static_cast<int>(rule_exs.size());
            for (auto& rex : rule_exs) {
                examples.push_back(std::move(rex));
            }
        }

        // ── Step 5: Shuffle ───────────────────────────────────────────────────
        {
            std::mt19937 rng(static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            std::shuffle(examples.begin(), examples.end(), rng);
        }

        dataset.examples    = std::move(examples);
        stats.final_count   = static_cast<int>(dataset.examples.size());

    } catch (const std::exception& ex) {
        LOG_ERROR("DatasetCurator::curate exception: " + std::string(ex.what()));
    }

    LOG_INFO("DatasetCurator: curated dataset — queried=" +
             std::to_string(stats.total_queried) +
             " passed=" + std::to_string(stats.passed_quality) +
             " deduped=" + std::to_string(stats.deduped) +
             " holdout=" + std::to_string(stats.holdout_reserved) +
             " rules=" + std::to_string(stats.rule_examples_added) +
             " final=" + std::to_string(stats.final_count));

    stats_out = stats;
    return dataset;
}

// ---------------------------------------------------------------------------
// curate() — convenience overload
// ---------------------------------------------------------------------------

TrainingDataset DatasetCurator::curate(const CurriculumPlan& plan) const {
    CurationStats stats;
    return curate(plan, stats);
}

// ---------------------------------------------------------------------------
// get_eval_holdout()
// ---------------------------------------------------------------------------

std::vector<EpisodeRecord> DatasetCurator::get_eval_holdout() const {
    // Return the N most-recent episodes across all domains.
    // These are the episodes reserved from every training dataset.
    return storage_.get_recent(cfg_.eval_holdout_episodes);
}

// ---------------------------------------------------------------------------
// episode_to_example()
// ---------------------------------------------------------------------------

std::optional<TrainingExample> DatasetCurator::episode_to_example(
        const EpisodeRecord& ep) const {

    // Hard completeness requirements.
    if (ep.user_message.size() < 10)     return std::nullopt;
    if (ep.response_summary.size() < 10) return std::nullopt;

    TrainingExample ex;
    ex.instruction = ep.user_message;
    ex.output      = ep.response_summary;
    ex.domain      = ep.reasoning_domain;
    ex.confidence  = ep.confidence;
    ex.episode_id  = ep.id;

    // If a rule was committed during this episode, inject the rule context
    // into the `input` field so the model learns to apply it.
    if (!ep.extracted_rule_id.empty()) {
        // We don't re-query the rule store here to keep curate() fast;
        // the rule ID is sufficient as a provenance tag. Rule text is
        // injected at the rule-augmented example level (build_rule_examples).
        ex.input = "[rule:" + ep.extracted_rule_id + "]";
    }

    return ex;
}

// ---------------------------------------------------------------------------
// build_rule_examples()
// ---------------------------------------------------------------------------

std::vector<TrainingExample> DatasetCurator::build_rule_examples(
        const std::string& domain,
        int                max_rule_examples) const {

    std::vector<TrainingExample> examples;

    RuleQuery q;
    q.domain         = domain;
    q.min_confidence = 0.5f;
    q.active_only    = true;
    q.max_results    = max_rule_examples;

    auto rules = rule_store_.query(q);

    for (const auto& rule : rules) {
        if (rule.condition.empty() || rule.consequence.empty()) continue;

        TrainingExample ex;
        ex.instruction = "Apply the following guideline in your response:";
        ex.input       = rule.condition + " → " + rule.consequence;
        // The expected output is intentionally abstract — the model should
        // learn to acknowledge and apply the rule, not memorise a specific
        // response. We use a templated compliant-response stub here.
        ex.output      = "I will keep in mind: " + rule.consequence +
                         " when reasoning about " + rule.domain + " topics.";
        ex.domain      = rule.domain;
        ex.confidence  = rule.confidence;
        ex.episode_id  = rule.episode_id;  // provenance

        examples.push_back(std::move(ex));
    }

    return examples;
}

// ---------------------------------------------------------------------------
// message_hash()
// ---------------------------------------------------------------------------

std::size_t DatasetCurator::message_hash(const std::string& s) {
    // FNV-1a 64-bit — fast, good distribution, no dependencies.
    std::size_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= static_cast<std::size_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace cardinal
