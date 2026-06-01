#pragma once
// =============================================================================
// Cardinal - Dataset Curator (Layer 3)
// File: src/training/dataset_curator.h
// =============================================================================

#include "self_model/self_model_types.h"   // TrainingDataset, TrainingExample
#include "memory/episodic_storage.h"       // EpisodeRecord — required by get_eval_holdout / episode_to_example
#include "training/curriculum_builder.h"   // CurriculumPlan
#include "utils/config_loader.h"

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cardinal {

    class RuleStore;

    // -------------------------------------------------------------------------
    // CurationStats
    // -------------------------------------------------------------------------
    struct CurationStats {
        int total_queried       = 0;
        int passed_quality      = 0;
        int deduped             = 0;
        int holdout_reserved    = 0;
        int rule_examples_added = 0;
        int final_count         = 0;
    };

    // -------------------------------------------------------------------------
    // DatasetCurator
    // -------------------------------------------------------------------------
    class DatasetCurator {
    public:
        DatasetCurator(const CardinalConfig& config,
                       EpisodicStorage&      storage,
                       RuleStore&            rule_store);

        // Build a curated TrainingDataset from the given CurriculumPlan.
        // Never throws — returns an empty dataset on any error.
        TrainingDataset curate(const CurriculumPlan& plan,
                               CurationStats&        stats_out) const;

        // Convenience overload without stats output.
        TrainingDataset curate(const CurriculumPlan& plan) const;

        // Return the N most-recent episodes reserved as eval holdout.
        std::vector<EpisodeRecord> get_eval_holdout() const;

    private:
        // Convert a single episode to a TrainingExample.
        std::optional<TrainingExample> episode_to_example(
            const EpisodeRecord& ep) const;

        // Build rule-augmented examples from RuleStore.
        std::vector<TrainingExample> build_rule_examples(
            const std::string& domain,
            int                max_rule_examples) const;

        // FNV-1a hash for dedup.
        static std::size_t message_hash(const std::string& s);

        struct CuratorCfg {
            int   eval_holdout_episodes  = 20;
            int   max_examples           = 0;
            float min_quality_confidence = 0.75f;
            bool  include_rules          = true;
            int   max_rule_examples      = 50;
        } cfg_;

        EpisodicStorage& storage_;
        RuleStore&       rule_store_;
    };

} // namespace cardinal
