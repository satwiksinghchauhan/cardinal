#pragma once
// =============================================================================
// Cardinal - TensorRT Training Backend (Layer 3)
// File: src/training/tensorrt_trainer.h
//
// Script-export backend for production environments where fine-tuning runs
// on a separate orchestration cluster, not inside the Cardinal process.
//
// What this backend does:
//   prepare()      — validates dataset, writes JSONL to dataset_output_dir.
//   train()        — writes a self-contained shell script to export_script_dir
//                    that a cluster operator can inspect and submit.
//                    Returns immediately with result.script_path set.
//                    does NOT launch any subprocess.
//   evaluate()     — not supported locally; returns a placeholder result with
//                    a descriptive error_message (not an error — just a note).
//   load_adapter() — not supported; returns success with a note. The cluster
//                    operator is responsible for deploying the new engine.
//   unload_adapter()— no-op.
//
// Generated shell script structure:
//   #!/usr/bin/env bash
//   # Cardinal LoRA training script — generated <timestamp>
//   # Dataset:  <jsonl_path>
//   # Domain:   <domain_focus>
//   # Examples: <count>
//
//   set -euo pipefail
//
//   # --- Step 1: PEFT fine-tuning ---
//   python -m cardinal_train --model <hf_model_path> --data <jsonl> ...
//
//   # --- Step 2: Convert to GGUF ---
//   python <convert_lora_script> <adapter_dir> --outfile <adapter.gguf>
//
//   # --- Step 3: (Optional) TensorRT engine rebuild ---
//   # Uncomment and adapt for your TRT-LLM environment:
//   # trtllm-build --checkpoint_dir <adapter_dir> \
//   #              --output_dir     <engine_dir>  \
//   #              --max_batch_size 1             \
//   #              --max_seq_len    8192
//
//   echo "Training complete. Adapter: <adapter.gguf>"
//
// Thread safety:
//   All public methods are protected by trainer_mutex_.
// =============================================================================

#include "training/i_training_backend.h"
#include "utils/config_loader.h"

#include <string>
#include <mutex>

namespace cardinal {

    class TensorRTTrainer final : public ITrainingBackend {
    public:
        explicit TensorRTTrainer(const CardinalConfig& config);
        ~TensorRTTrainer() override = default;

        TensorRTTrainer(const TensorRTTrainer&)            = delete;
        TensorRTTrainer& operator=(const TensorRTTrainer&) = delete;

        // -- ITrainingBackend interface ----------------------------------------
        std::string name()             const override { return "tensorrt"; }
        bool        can_train_locally() const override { return false; }

        TrainingResult prepare (const TrainingDataset& dataset,
                                const LoRAConfig&      lora_cfg) override;

        TrainingResult train   (const TrainingDataset& dataset,
                                const LoRAConfig&      lora_cfg,
                                ProgressCallback       progress_cb = nullptr) override;

        // Not supported — returns a non-error placeholder result.
        TrainingResult evaluate(const std::string&                adapter_path,
                                const std::vector<EpisodeRecord>& eval_episodes) override;

        // No-op — adapter deployment is handled externally.
        TrainingResult load_adapter  (const std::string& adapter_path) override;
        void           unload_adapter()                                 override {}

        bool        has_adapter()         const override { return false; }
        std::string active_adapter_path() const override { return ""; }

    private:
        // Write JSONL dataset and return its path.
        std::string write_jsonl(const TrainingDataset& dataset,
                                const std::string&     run_id) const;

        // Render and write the shell script; return its path.
        std::string write_script(const std::string& jsonl_path,
                                 const std::string& run_id,
                                 const TrainingDataset& dataset,
                                 const LoRAConfig&      lora_cfg) const;

        // Expand "~/" to $HOME.
        static std::string expand_home(const std::string& path);

        static std::string make_run_id();
        static std::string utc_now_str();

        // Training sub-config (stored by value).
        struct TrainCfg {
            std::string adapter_output_dir;
            std::string dataset_output_dir;
            std::string export_script_dir;
            std::string hf_model_path;
            std::string python_venv;
            std::string convert_lora_script;
            float       min_quality_confidence = 0.75f;
        } cfg_;

        mutable std::mutex trainer_mutex_;
    };

} // namespace cardinal
