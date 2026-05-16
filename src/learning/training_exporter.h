// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Training Exporter
// File: src/learning/training_exporter.h
//
// Exports high-confidence episodes and rules as training data for
// external LoRA fine-tuning.
//
// Output format: Alpaca JSONL
//   One JSON object per line:
//   {"instruction": "<user_message>", "input": "", "output": "<response>"}
//
// v1.4.0: TrainingExample moved to self_model_types.h (canonical definition).
//         This header now includes it from there instead of redefining it.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "self_model/self_model_types.h"   // TrainingExample — canonical definition
#include "utils/config_loader.h"
#include "memory/episodic_storage.h"
#include "memory/rule_store.h"

#include <string>
#include <vector>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ExportFilter
    // Controls what gets exported.
    // -------------------------------------------------------------------------
    struct ExportFilter {
        float       min_confidence = 0.7f;  // Minimum episode confidence
        std::string domain;                  // Filter by domain (empty = all)
        int         max_examples   = 0;      // 0 = no limit
        bool        include_rules  = false;  // Also export rules as examples
        bool        recent_first   = true;   // ORDER BY timestamp DESC
    };

    // -------------------------------------------------------------------------
    // ExportStats
    // Summary of a completed export operation.
    // -------------------------------------------------------------------------
    struct ExportStats {
        int         total_episodes_checked = 0;
        int         episodes_exported      = 0;
        int         rules_exported         = 0;
        int         total_exported         = 0;
        float       avg_confidence         = 0.0f;
        std::string output_path;
        std::string timestamp;
    };

    // -------------------------------------------------------------------------
    // TrainingExporter
    // -------------------------------------------------------------------------
    class TrainingExporter {
    public:
        TrainingExporter(const CardinalConfig& config,
                         EpisodicStorage&      storage,
                         RuleStore&            rule_store);

        // Export training examples to a JSONL file.
        // Creates parent directories if needed.
        // Throws TrainingExporterError on file write failure.
        ExportStats export_to_file(const std::string&  output_path,
                                   const ExportFilter& filter = ExportFilter{});

        // Collect training examples without writing to disk.
        std::vector<TrainingExample> collect(
            const ExportFilter& filter = ExportFilter{}) const;

        // How many episodes are available above the confidence threshold.
        int available_episode_count(float min_confidence = 0.7f) const;

        // How many rules are available for export.
        int available_rule_count() const;

        // Full stats snapshot without exporting.
        ExportStats dry_run(const ExportFilter& filter = ExportFilter{}) const;

    private:
        static TrainingExample episode_to_example(const EpisodeRecord& ep);
        static TrainingExample rule_to_example   (const Rule& rule);
        static std::string     to_jsonl_line     (const TrainingExample& example);
        static std::string     clean_response    (const std::string& text);

        const CardinalConfig& config_;
        EpisodicStorage&      storage_;
        RuleStore&            rule_store_;
    };

    // -------------------------------------------------------------------------
    // TrainingExporterError
    // -------------------------------------------------------------------------
    class TrainingExporterError : public std::runtime_error {
    public:
        explicit TrainingExporterError(const std::string& message)
            : std::runtime_error("TrainingExporterError: " + message) {}
    };

} // namespace cardinal
