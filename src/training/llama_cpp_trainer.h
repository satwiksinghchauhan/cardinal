#pragma once
// =============================================================================
// Cardinal - LlamaCpp Training Backend (Layer 3)
// File: src/training/llama_cpp_trainer.h
// =============================================================================

#include "training/i_training_backend.h"
#include "utils/config_loader.h"

#include <mutex>
#include <string>
#include <vector>

struct llama_adapter_lora;

namespace cardinal {

    class ILLMBackend;
    class LlamaCppBackend;

    class LlamaCppTrainer final : public ITrainingBackend {
    public:
        LlamaCppTrainer(const CardinalConfig& config,
                        ILLMBackend&          backend);
        ~LlamaCppTrainer() override;

        LlamaCppTrainer(const LlamaCppTrainer&)            = delete;
        LlamaCppTrainer& operator=(const LlamaCppTrainer&) = delete;

        std::string name()              const override { return "llama_cpp"; }
        bool        can_train_locally() const override { return true; }

        TrainingResult prepare (const TrainingDataset& dataset,
                                const LoRAConfig&      lora_cfg) override;
        TrainingResult train   (const TrainingDataset& dataset,
                                const LoRAConfig&      lora_cfg,
                                ProgressCallback       progress_cb = nullptr) override;
        TrainingResult evaluate(const std::string&                adapter_path,
                                const std::vector<EpisodeRecord>& eval_episodes) override;
        TrainingResult load_adapter  (const std::string& adapter_path) override;
        void           unload_adapter()                                 override;
        bool           has_adapter()         const override;
        std::string    active_adapter_path() const override;

    private:
        std::string    write_jsonl          (const TrainingDataset& dataset,
                                             const std::string&     run_id) const;
        int            run_subprocess       (const std::string&     cmd,
                                             ProgressCallback       cb) const;
        std::string    build_peft_command   (const std::string&     jsonl_path,
                                             const std::string&     out_dir,
                                             const std::string&     run_id,
                                             const LoRAConfig&      lora_cfg) const;
        std::string    build_convert_command(const std::string&     hf_dir,
                                             const std::string&     gguf_path) const;

        TrainingResult load_adapter_internal  (const std::string& adapter_path);
        void           unload_adapter_internal();

        static std::string make_run_id();
        static std::string expand_home(const std::string& path);

        struct TrainCfg {
            int         lora_rank;
            int         lora_alpha;
            float       learning_rate;
            int         epochs;
            int         batch_size;
            float       min_quality_confidence;
            std::string adapter_output_dir;
            std::string dataset_output_dir;
            std::string hf_model_path;
            std::string python_venv;
            std::string convert_lora_script;
        } cfg_;

        ILLMBackend&          backend_;
        const CardinalConfig& config_;

        llama_adapter_lora*  active_lora_handle_  = nullptr;
        bool                 adapter_loaded_       = false;
        std::string          active_adapter_path_;
        float                cached_baseline_      = -1.0f;

        mutable std::mutex   trainer_mutex_;
    };

} // namespace cardinal
