#pragma once
// =============================================================================
// Cardinal - Self-Model Types
// File: src/self_model/self_model_types.h
//
// Shared types across all three self-improvement layers:
//   Layer 1: SelfModel      — symbolic self-knowledge
//   Layer 2: MetaCognition  — reflection + corrective rules
//   Layer 3: LoRA training  — weight updates
// =============================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace cardinal {

    // -------------------------------------------------------------------------
    // DomainStats
    // Per reasoning-domain performance statistics.
    // Updated after every inference from episodic data.
    // -------------------------------------------------------------------------
    struct DomainStats {
        std::string domain;                  // factual|ethical|spatial|etc.
        float       avg_confidence    = 0.0f;
        float       contradiction_rate = 0.0f; // fraction of inferences with contradiction
        float       uncertainty_rate  = 0.0f;  // fraction with uncertainty_flag
        float       rule_commit_rate  = 0.0f;  // fraction that produced a committed rule
        int         total_inferences  = 0;
        int         total_contradictions = 0;
        int         total_uncertainties  = 0;
        std::string last_updated;

        // Weakness score: higher = more room for improvement
        float weakness_score() const {
            if (total_inferences == 0) return 0.0f;
            return (contradiction_rate * 0.4f) +
                   (uncertainty_rate   * 0.3f) +
                   ((1.0f - avg_confidence) * 0.3f);
        }
    };

    // -------------------------------------------------------------------------
    // ReasoningTypeStats
    // How often each reasoning type is used and how reliable it is.
    // -------------------------------------------------------------------------
    struct ReasoningTypeStats {
        std::string reasoning_type;     // causal|deductive|analogical|etc.
        int         usage_count     = 0;
        float       avg_confidence  = 0.0f;
        float       contradiction_rate = 0.0f;
    };

    // -------------------------------------------------------------------------
    // PerformanceTrend
    // Rolling average of confidence over recent N inferences.
    // Used to detect degradation that should trigger training.
    // -------------------------------------------------------------------------
    struct PerformanceTrend {
        std::string domain;
        float       confidence_7d   = 0.0f;  // last 7 days or N inferences
        float       confidence_30d  = 0.0f;
        float       trend_direction = 0.0f;  // positive = improving, negative = degrading
        std::string computed_at;
    };

    // -------------------------------------------------------------------------
    // SelfModelSnapshot
    // Point-in-time snapshot of Cardinal's self-knowledge.
    // Injected into the system prompt as calibration context.
    // -------------------------------------------------------------------------
    struct SelfModelSnapshot {
        std::vector<DomainStats>        domain_stats;
        std::vector<ReasoningTypeStats> reasoning_stats;
        std::vector<PerformanceTrend>   trends;

        // Weakest domain (highest weakness_score)
        std::string weakest_domain() const {
            float worst = -1.0f;
            std::string result;
            for (const auto& d : domain_stats) {
                if (d.weakness_score() > worst) {
                    worst  = d.weakness_score();
                    result = d.domain;
                }
            }
            return result;
        }

        // Strongest domain (lowest weakness_score, enough data)
        std::string strongest_domain() const {
            float best = 2.0f;
            std::string result;
            for (const auto& d : domain_stats) {
                if (d.total_inferences >= 5 && d.weakness_score() < best) {
                    best   = d.weakness_score();
                    result = d.domain;
                }
            }
            return result;
        }

        // Format as text for system prompt injection
        std::string format_for_prompt(int max_chars = 500) const;
    };

    // -------------------------------------------------------------------------
    // ReflectionFinding
    // A single insight from a meta-cognition reflection pass.
    // -------------------------------------------------------------------------
    struct ReflectionFinding {
        std::string domain;
        std::string pattern;          // description of the failure pattern
        std::string recommendation;   // what to do differently
        float       confidence = 0.0f;
        std::string timestamp;
    };

    // -------------------------------------------------------------------------
    // ReflectionResult
    // Output of a single meta-cognition reflection pass.
    // -------------------------------------------------------------------------
    struct ReflectionResult {
        bool                          ran              = false;
        std::string                   trigger;         // "scheduled"|"contradiction_rate"|"manual"
        int                           episodes_analyzed = 0;
        int                           failures_analyzed = 0;
        std::vector<ReflectionFinding> findings;
        int                           rules_committed  = 0;
        int                           duration_ms      = 0;
        std::string                   timestamp;
        std::string                   error_message;
    };

    // -------------------------------------------------------------------------
    // LoRAConfig
    // Parameters for a LoRA fine-tuning run.
    // -------------------------------------------------------------------------
    struct LoRAConfig {
        int         rank          = 8;
        int         alpha         = 16;
        float       learning_rate = 0.0001f;
        int         epochs        = 3;
        int         batch_size    = 4;
        float       val_split     = 0.05f;
        std::string optimizer     = "adamw";
        std::string target_modules = "q_proj,v_proj"; // which layers to adapt
    };

    // -------------------------------------------------------------------------
    // TrainingExample
    // A single supervised training example in alpaca format.
    // -------------------------------------------------------------------------
    struct TrainingExample {
        std::string instruction;    // the prompt / task
        std::string input;          // optional additional context
        std::string output;         // the target response
        std::string domain;         // source domain
        float       confidence = 0.0f; // source episode confidence
        std::string episode_id;     // provenance
        // Extended fields used by TrainingExporter (v1.4.0 merge)
        std::string reasoning_type; // e.g. "causal", "deductive"
        std::string timestamp;      // ISO-8601 timestamp of source episode/rule
        std::string source;         // "episode" | "rule"
    };

    // -------------------------------------------------------------------------
    // TrainingDataset
    // A curated set of training examples ready for fine-tuning.
    // -------------------------------------------------------------------------
    struct TrainingDataset {
        std::vector<TrainingExample> examples;
        std::string                  domain_focus;   // empty = all domains
        std::string                  created_at;
        std::string                  jsonl_path;     // path to exported JSONL file

        int size() const { return static_cast<int>(examples.size()); }
    };

    // -------------------------------------------------------------------------
    // TrainingResult
    // Output of a complete training cycle.
    // -------------------------------------------------------------------------
    struct TrainingResult {
        bool        success          = false;
        std::string adapter_path;    // path to output .gguf adapter
        std::string script_path;     // path to generated training script (TRT)
        float       eval_score       = 0.0f;  // held-out eval score
        float       baseline_score   = 0.0f;  // pre-training baseline
        float       improvement_pct  = 0.0f;  // (eval - baseline) / baseline * 100
        bool        adapter_loaded   = false;
        int         examples_trained = 0;
        int         duration_seconds = 0;
        std::string trigger;         // what triggered this training run
        std::string timestamp;
        std::string error_message;
    };

    // -------------------------------------------------------------------------
    // SelfImprovementStatus
    // Current state of all three layers — returned by CardinalAPI::get_stats()
    // -------------------------------------------------------------------------
    struct SelfImprovementStatus {
        // Layer 1
        bool        self_model_enabled  = false;
        std::string weakest_domain;
        std::string strongest_domain;
        int         total_domain_stats  = 0;

        // Layer 2
        bool        meta_cognition_enabled = false;
        int         total_reflections      = 0;
        int         total_corrective_rules = 0;
        std::string last_reflection_at;

        // Layer 3
        bool        training_enabled    = false;
        int         total_training_runs = 0;
        std::string last_training_at;
        std::string active_adapter_path;
        float       last_improvement_pct = 0.0f;
    };

} // namespace cardinal
