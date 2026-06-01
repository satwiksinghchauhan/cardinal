// =============================================================================
// Cardinal - TensorRT Training Backend — Implementation
// File: src/training/tensorrt_trainer.cpp
// =============================================================================

#include "training/tensorrt_trainer.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace cardinal {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::string TensorRTTrainer::utc_now_str() {
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

std::string TensorRTTrainer::make_run_id() {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    return "run_" + std::to_string(ms);
}

std::string TensorRTTrainer::expand_home(const std::string& path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TensorRTTrainer::TensorRTTrainer(const CardinalConfig& config) {
    const auto& tc = config.self_improvement.training;
    cfg_.adapter_output_dir     = tc.adapter_output_dir;
    cfg_.dataset_output_dir     = tc.dataset_output_dir;
    cfg_.export_script_dir      = tc.export_script_dir;
    cfg_.hf_model_path          = tc.hf_model_path;
    cfg_.python_venv            = expand_home(tc.python_venv);
    cfg_.convert_lora_script    = tc.convert_lora_script;
    cfg_.min_quality_confidence = tc.min_quality_confidence;

    LOG_INFO("TensorRTTrainer: initialised (script-export mode, script_dir=" +
             cfg_.export_script_dir + ")");
}

// ---------------------------------------------------------------------------
// prepare()
// ---------------------------------------------------------------------------

TrainingResult TensorRTTrainer::prepare(const TrainingDataset& dataset,
                                         const LoRAConfig&      /*lora_cfg*/) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);

    TrainingResult r;
    r.timestamp = utc_now_str();

    if (dataset.examples.empty()) {
        r.error_message = "dataset is empty";
        LOG_ERROR("TensorRTTrainer::prepare: " + r.error_message);
        return r;
    }

    // Ensure output dirs exist.
    try {
        fs::create_directories(cfg_.dataset_output_dir);
        fs::create_directories(cfg_.adapter_output_dir);
        fs::create_directories(cfg_.export_script_dir);
    } catch (const std::exception& ex) {
        r.error_message = std::string("failed to create output dirs: ") + ex.what();
        LOG_ERROR("TensorRTTrainer::prepare: " + r.error_message);
        return r;
    }

    r.success          = true;
    r.examples_trained = static_cast<int>(dataset.examples.size());
    return r;
}

// ---------------------------------------------------------------------------
// write_jsonl()
// ---------------------------------------------------------------------------

std::string TensorRTTrainer::write_jsonl(const TrainingDataset& dataset,
                                          const std::string&     run_id) const {
    std::string path = cfg_.dataset_output_dir + "/" + run_id + ".jsonl";
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) throw std::runtime_error("cannot open JSONL output: " + path);

    int written = 0;
    for (const auto& ex : dataset.examples) {
        if (ex.confidence < cfg_.min_quality_confidence) continue;

        json obj;
        obj["instruction"] = ex.instruction;
        obj["input"]       = ex.input;
        obj["output"]      = ex.output;
        obj["domain"]      = ex.domain;
        obj["confidence"]  = ex.confidence;
        obj["episode_id"]  = ex.episode_id;
        ofs << obj.dump() << "\n";
        ++written;
    }

    ofs.close();
    if (!ofs) throw std::runtime_error("JSONL write failed (flush error): " + path);

    LOG_INFO("TensorRTTrainer: wrote " + std::to_string(written) +
             " examples to " + path);
    return path;
}

// ---------------------------------------------------------------------------
// write_script()
// ---------------------------------------------------------------------------

std::string TensorRTTrainer::write_script(const std::string&     jsonl_path,
                                           const std::string&     run_id,
                                           const TrainingDataset& dataset,
                                           const LoRAConfig&      lora_cfg) const {
    std::string script_path = cfg_.export_script_dir + "/train_" + run_id + ".sh";
    std::ofstream ofs(script_path, std::ios::out | std::ios::trunc);
    if (!ofs) throw std::runtime_error("cannot open script output: " + script_path);

    std::string adapter_dir  = cfg_.adapter_output_dir + "/" + run_id;
    std::string adapter_gguf = cfg_.adapter_output_dir + "/" + run_id + ".gguf";
    std::string python_bin   = cfg_.python_venv + "/bin/python";

    ofs << "#!/usr/bin/env bash\n"
        << "# =================================================================\n"
        << "# Cardinal LoRA training script\n"
        << "# Generated:  " << utc_now_str() << "\n"
        << "# Run ID:     " << run_id << "\n"
        << "# Dataset:    " << jsonl_path << "\n"
        << "# Domain:     " << (dataset.domain_focus.empty() ? "all" : dataset.domain_focus) << "\n"
        << "# Examples:   " << dataset.examples.size() << "\n"
        << "# LoRA rank:  " << lora_cfg.rank << "\n"
        << "# LoRA alpha: " << lora_cfg.alpha << "\n"
        << "# LR:         " << lora_cfg.learning_rate << "\n"
        << "# Epochs:     " << lora_cfg.epochs << "\n"
        << "# =================================================================\n\n"
        << "set -euo pipefail\n\n"

        << "# ── Step 1: PEFT fine-tuning ─────────────────────────────────────\n"
        << python_bin << " -m cardinal_train"
        << " \\\n    --model "   << cfg_.hf_model_path
        << " \\\n    --data "    << jsonl_path
        << " \\\n    --output "  << adapter_dir
        << " \\\n    --rank "    << lora_cfg.rank
        << " \\\n    --alpha "   << lora_cfg.alpha
        << " \\\n    --lr "      << lora_cfg.learning_rate
        << " \\\n    --epochs "  << lora_cfg.epochs
        << " \\\n    --batch "   << lora_cfg.batch_size
        << " \\\n    --modules " << lora_cfg.target_modules
        << "\n\n"

        << "# ── Step 2: Convert HF adapter to GGUF ──────────────────────────\n"
        << python_bin << " " << cfg_.convert_lora_script
        << " \\\n    " << adapter_dir
        << " \\\n    --outfile " << adapter_gguf
        << "\n\n"

        << "# ── Step 3: (Optional) TensorRT-LLM engine rebuild ──────────────\n"
        << "# Uncomment and adapt the block below for your TRT-LLM environment.\n"
        << "#\n"
        << "# ENGINE_DIR=\"" << cfg_.adapter_output_dir << "/" << run_id << "_engine\"\n"
        << "# trtllm-build \\\n"
        << "#     --checkpoint_dir " << adapter_dir << " \\\n"
        << "#     --output_dir     \"${ENGINE_DIR}\" \\\n"
        << "#     --max_batch_size 1 \\\n"
        << "#     --max_seq_len    8192 \\\n"
        << "#     --use_paged_context_fmha enable\n"
        << "\n"

        << "echo \"Training complete.\"\n"
        << "echo \"Adapter GGUF: " << adapter_gguf << "\"\n";

    ofs.close();
    if (!ofs) throw std::runtime_error("script write failed (flush error): " + script_path);

    // Make the script executable.
    fs::permissions(script_path,
                    fs::perms::owner_read  | fs::perms::owner_write  | fs::perms::owner_exec |
                    fs::perms::group_read  | fs::perms::group_exec   |
                    fs::perms::others_read | fs::perms::others_exec,
                    fs::perm_options::replace);

    LOG_INFO("TensorRTTrainer: wrote training script to " + script_path);
    return script_path;
}

// ---------------------------------------------------------------------------
// train()
// ---------------------------------------------------------------------------

TrainingResult TensorRTTrainer::train(const TrainingDataset& dataset,
                                       const LoRAConfig&      lora_cfg,
                                       ProgressCallback       progress_cb) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);

    TrainingResult r;
    r.timestamp = utc_now_str();
    r.trigger   = dataset.domain_focus.empty() ? "general" : dataset.domain_focus;

    try {
        std::string run_id = make_run_id();

        // Write JSONL.
        std::string jsonl_path = write_jsonl(dataset, run_id);

        // Write shell script.
        std::string script_path = write_script(jsonl_path, run_id, dataset, lora_cfg);

        r.success          = true;
        r.script_path      = script_path;
        r.examples_trained = static_cast<int>(dataset.examples.size());
        r.duration_seconds = 0;  // script-only — no time spent training

        // Signal single completion to any progress observer.
        if (progress_cb) {
            progress_cb(1, 1, 0.0f);
        }

        LOG_INFO("TensorRTTrainer::train: script written to " + script_path +
                 " (submit to cluster to run training)");

    } catch (const std::exception& ex) {
        r.error_message = std::string("exception: ") + ex.what();
        LOG_ERROR("TensorRTTrainer::train: " + r.error_message);
    }

    return r;
}

// ---------------------------------------------------------------------------
// evaluate() — not supported locally
// ---------------------------------------------------------------------------

TrainingResult TensorRTTrainer::evaluate(
        const std::string&                /*adapter_path*/,
        const std::vector<EpisodeRecord>& /*eval_episodes*/) {
    TrainingResult r;
    r.timestamp     = utc_now_str();
    r.success       = true;     // not a hard failure — just unsupported locally
    r.eval_score    = 0.0f;
    r.error_message = "eval not supported in TensorRT script-export mode; "
                      "run evaluate_adapter.py on the cluster after training";
    LOG_INFO("TensorRTTrainer::evaluate: " + r.error_message);
    return r;
}

// ---------------------------------------------------------------------------
// load_adapter() — no-op with informational message
// ---------------------------------------------------------------------------

TrainingResult TensorRTTrainer::load_adapter(const std::string& adapter_path) {
    TrainingResult r;
    r.timestamp      = utc_now_str();
    r.success        = true;
    r.adapter_path   = adapter_path;
    r.adapter_loaded = false;
    r.error_message  = "adapter loading is managed externally in TensorRT mode; "
                       "deploy the new engine via your cluster orchestrator";
    LOG_INFO("TensorRTTrainer::load_adapter: " + r.error_message);
    return r;
}

} // namespace cardinal
