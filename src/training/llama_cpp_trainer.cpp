// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - LlamaCpp Training Backend — Implementation
// File: src/training/llama_cpp_trainer.cpp
// =============================================================================

#include "training/llama_cpp_trainer.h"
#include "core/llm_backend.h"
#include "core/backends/llama_cpp_backend.h"
#include "core/feeling_output.h"
#include "utils/logger.h"

#include <llama.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json   = nlohmann::json;

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

int elapsed_s(std::chrono::steady_clock::time_point start) {
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count());
}

FILE* safe_popen(const std::string& cmd) {
#ifdef _WIN32
    return _popen(cmd.c_str(), "r");
#else
    return popen(cmd.c_str(), "r");
#endif
}

int safe_pclose(FILE* f) {
#ifdef _WIN32
    return _pclose(f);
#else
    return pclose(f);
#endif
}

bool parse_progress_line(const std::string& line,
                         int& step, int& total, float& loss) {
    std::istringstream iss(line);
    std::string tok;
    if (!(iss >> tok) || tok != "STEP") return false;
    std::string step_tok;
    if (!(iss >> step_tok)) return false;
    auto slash = step_tok.find('/');
    if (slash == std::string::npos) return false;
    try {
        step  = std::stoi(step_tok.substr(0, slash));
        total = std::stoi(step_tok.substr(slash + 1));
    } catch (...) { return false; }
    if (!(iss >> tok) || tok != "LOSS") return false;
    if (!(iss >> loss)) return false;
    return true;
}

} // anonymous namespace

namespace cardinal {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

LlamaCppTrainer::LlamaCppTrainer(const CardinalConfig& config,
                                 ILLMBackend&          backend)
    : backend_(backend)
    , config_(config)
{
    const auto& tc = config.self_improvement.training;
    cfg_.lora_rank              = tc.lora_rank;
    cfg_.lora_alpha             = tc.lora_alpha;
    cfg_.learning_rate          = tc.learning_rate;
    cfg_.epochs                 = tc.epochs;
    cfg_.batch_size             = tc.batch_size;
    cfg_.min_quality_confidence = tc.min_quality_confidence;
    cfg_.adapter_output_dir     = tc.adapter_output_dir;
    cfg_.dataset_output_dir     = tc.dataset_output_dir;
    cfg_.hf_model_path          = tc.hf_model_path;
    cfg_.python_venv            = expand_home(tc.python_venv);
    cfg_.convert_lora_script    = tc.convert_lora_script;

    LOG_INFO("LlamaCppTrainer: initialised (venv=" + cfg_.python_venv + ")");
}

LlamaCppTrainer::~LlamaCppTrainer() {
    if (adapter_loaded_) {
        std::lock_guard<std::mutex> lock(trainer_mutex_);
        unload_adapter_internal();
    }
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::string LlamaCppTrainer::make_run_id() {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
    return "run_" + std::to_string(ms);
}

std::string LlamaCppTrainer::expand_home(const std::string& path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

// ---------------------------------------------------------------------------
// prepare()
// ---------------------------------------------------------------------------

TrainingResult LlamaCppTrainer::prepare(const TrainingDataset& dataset,
                                         const LoRAConfig& /*lora_cfg*/) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);

    TrainingResult r;
    r.timestamp = utc_now_str();

    if (dataset.examples.empty()) {
        r.error_message = "dataset is empty";
        LOG_ERROR("LlamaCppTrainer::prepare: " + r.error_message);
        return r;
    }

    int low_conf = 0;
    for (const auto& ex : dataset.examples)
        if (ex.confidence < cfg_.min_quality_confidence) ++low_conf;

    if (low_conf == static_cast<int>(dataset.examples.size())) {
        r.error_message = "all examples below min_quality_confidence";
        LOG_ERROR("LlamaCppTrainer::prepare: " + r.error_message);
        return r;
    }

    if (!fs::exists(cfg_.hf_model_path)) {
        r.error_message = "hf_model_path not found: " + cfg_.hf_model_path;
        LOG_ERROR("LlamaCppTrainer::prepare: " + r.error_message);
        return r;
    }

    if (!fs::exists(cfg_.python_venv + "/bin/python")) {
        r.error_message = "python not found in venv: " + cfg_.python_venv;
        LOG_ERROR("LlamaCppTrainer::prepare: " + r.error_message);
        return r;
    }

    if (!fs::exists(cfg_.convert_lora_script)) {
        r.error_message = "convert_lora_to_gguf.py not found: " + cfg_.convert_lora_script;
        LOG_ERROR("LlamaCppTrainer::prepare: " + r.error_message);
        return r;
    }

    fs::create_directories(cfg_.dataset_output_dir);
    fs::create_directories(cfg_.adapter_output_dir);

    r.success          = true;
    r.examples_trained = static_cast<int>(dataset.examples.size());
    return r;
}

// ---------------------------------------------------------------------------
// write_jsonl()
// ---------------------------------------------------------------------------

std::string LlamaCppTrainer::write_jsonl(const TrainingDataset& dataset,
                                          const std::string&     run_id) const {
    std::string path = cfg_.dataset_output_dir + "/" + run_id + ".jsonl";
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) throw std::runtime_error("cannot open JSONL output: " + path);

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
    }

    ofs.close();
    if (!ofs) throw std::runtime_error("JSONL write failed: " + path);
    LOG_INFO("LlamaCppTrainer: wrote JSONL to " + path);
    return path;
}

// ---------------------------------------------------------------------------
// Command builders
// ---------------------------------------------------------------------------

std::string LlamaCppTrainer::build_peft_command(const std::string& jsonl_path,
                                                  const std::string& out_dir,
                                                  const std::string& run_id,
                                                  const LoRAConfig&  lora_cfg) const {
    std::ostringstream cmd;
    cmd << cfg_.python_venv << "/bin/python -m cardinal_train"
        << " --model "   << cfg_.hf_model_path
        << " --data "    << jsonl_path
        << " --output "  << out_dir << "/" << run_id
        << " --rank "    << lora_cfg.rank
        << " --alpha "   << lora_cfg.alpha
        << " --lr "      << lora_cfg.learning_rate
        << " --epochs "  << lora_cfg.epochs
        << " --batch "   << lora_cfg.batch_size
        << " --modules " << lora_cfg.target_modules
        << " 2>&1";
    return cmd.str();
}

std::string LlamaCppTrainer::build_convert_command(const std::string& hf_dir,
                                                    const std::string& gguf_path) const {
    std::ostringstream cmd;
    cmd << cfg_.python_venv << "/bin/python "
        << cfg_.convert_lora_script << " "
        << hf_dir
        << " --outfile " << gguf_path
        << " 2>&1";
    return cmd.str();
}

// ---------------------------------------------------------------------------
// run_subprocess()
// ---------------------------------------------------------------------------

int LlamaCppTrainer::run_subprocess(const std::string& cmd,
                                     ProgressCallback   progress_cb) const {
    LOG_INFO("LlamaCppTrainer: running: " + cmd);
    FILE* pipe = safe_popen(cmd);
    if (!pipe) throw std::runtime_error("popen failed: " + cmd);

    std::array<char, 512> buf{};
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
        std::string line(buf.data());
        if (!line.empty() && line.back() == '\n') line.pop_back();
        LOG_DEBUG("LlamaCppTrainer [subprocess]: " + line);

        if (progress_cb) {
            int step = 0, total = 0; float loss = 0.0f;
            if (parse_progress_line(line, step, total, loss))
                if (!progress_cb(step, total, loss)) {
                    safe_pclose(pipe);
                    return -1;
                }
        }
    }

    int rc = safe_pclose(pipe);
#ifndef _WIN32
    if (WIFEXITED(rc)) rc = WEXITSTATUS(rc);
#endif
    return rc;
}

// ---------------------------------------------------------------------------
// train()
// ---------------------------------------------------------------------------

TrainingResult LlamaCppTrainer::train(const TrainingDataset& dataset,
                                       const LoRAConfig&      lora_cfg,
                                       ProgressCallback       progress_cb) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);

    TrainingResult r;
    r.timestamp = utc_now_str();
    r.trigger   = dataset.domain_focus.empty() ? "general" : dataset.domain_focus;
    auto t_start = std::chrono::steady_clock::now();

    try {
        std::string run_id     = make_run_id();
        std::string jsonl_path = write_jsonl(dataset, run_id);

        std::string adapter_hf_dir = cfg_.adapter_output_dir + "/" + run_id;
        int rc = run_subprocess(
            build_peft_command(jsonl_path, cfg_.adapter_output_dir, run_id, lora_cfg),
            progress_cb);
        if (rc != 0) {
            r.error_message = "PEFT subprocess exited with code " + std::to_string(rc);
            LOG_ERROR("LlamaCppTrainer::train: " + r.error_message);
            return r;
        }
        if (!fs::exists(adapter_hf_dir)) {
            r.error_message = "PEFT did not create adapter dir: " + adapter_hf_dir;
            LOG_ERROR("LlamaCppTrainer::train: " + r.error_message);
            return r;
        }

        std::string gguf_path = cfg_.adapter_output_dir + "/" + run_id + ".gguf";
        rc = run_subprocess(build_convert_command(adapter_hf_dir, gguf_path), nullptr);
        if (rc != 0) {
            r.error_message = "convert_lora_to_gguf.py exited with code " + std::to_string(rc);
            LOG_ERROR("LlamaCppTrainer::train: " + r.error_message);
            return r;
        }
        if (!fs::exists(gguf_path)) {
            r.error_message = "GGUF not found after conversion: " + gguf_path;
            LOG_ERROR("LlamaCppTrainer::train: " + r.error_message);
            return r;
        }

        r.success          = true;
        r.adapter_path     = gguf_path;
        r.examples_trained = static_cast<int>(dataset.examples.size());
        r.duration_seconds = elapsed_s(t_start);
        LOG_INFO("LlamaCppTrainer::train: adapter at " + gguf_path);

    } catch (const std::exception& ex) {
        r.error_message = std::string("exception: ") + ex.what();
        LOG_ERROR("LlamaCppTrainer::train: " + r.error_message);
    }
    return r;
}

// ---------------------------------------------------------------------------
// evaluate()
//
// Scoring strategy — avoids any FeelingContext field access issues:
//
//   Baseline (pre-adapter):
//     Average of the stored episode confidence values from the holdout set.
//     ep.confidence IS the model's self-assessed quality at inference time,
//     making it a valid pre-training reference score. Cached after first call.
//
//   Eval (post-adapter):
//     After loading the adapter, run generate_feeling() on each holdout
//     episode. generate_feeling() returns a GenerationResult that contains
//     the parsed FeelingOutput directly — no FeelingContext field access needed.
//     If GenerationResult doesn't carry feeling, fall back to stored confidence.
// ---------------------------------------------------------------------------

TrainingResult LlamaCppTrainer::evaluate(
        const std::string&                adapter_path,
        const std::vector<EpisodeRecord>& eval_episodes) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);

    TrainingResult r;
    r.timestamp = utc_now_str();

    if (eval_episodes.empty()) {
        r.error_message = "eval_episodes is empty";
        return r;
    }

    try {
        // Baseline: average stored episode confidence (no inference needed).
        if (cached_baseline_ < 0.0f) {
            float sum = 0.0f;
            for (const auto& ep : eval_episodes) sum += ep.confidence;
            cached_baseline_ = sum / static_cast<float>(eval_episodes.size());
            LOG_INFO("LlamaCppTrainer::evaluate: baseline=" +
                     std::to_string(cached_baseline_));
        }
        r.baseline_score = cached_baseline_;

        // Load adapter.
        TrainingResult load_r = load_adapter_internal(adapter_path);
        if (!load_r.success) {
            r.error_message = "temporary adapter load failed: " + load_r.error_message;
            return r;
        }

        // Eval score: run generate_feeling() per episode.
        // generate_feeling() runs only Pass 1 (constrained decoding) which
        // is fast and produces the confidence score directly in its return value.
        float conf_sum = 0.0f;
        int   count    = 0;
        for (const auto& ep : eval_episodes) {
            std::vector<ChatMessage> msgs;
            msgs.push_back({ "user", ep.user_message });

            FeelingContext ctx(config_);
            GenerationResult gen = backend_.generate_feeling(ctx, msgs);

            float conf = ep.confidence;  // fallback: use stored confidence
            if (gen.success && ctx.has_valid_feeling()) {
                // feeling() is the correct public accessor — returns const FeelingOutput&
                // Gated by has_valid_feeling() so no exception risk.
                conf = ctx.feeling().confidence;
            }

            conf_sum += conf;
            ++count;
        }
        r.eval_score = count > 0 ? conf_sum / static_cast<float>(count) : 0.0f;

        unload_adapter_internal();

        r.improvement_pct = (r.baseline_score > 0.0f)
            ? ((r.eval_score - r.baseline_score) / r.baseline_score) * 100.0f
            : 0.0f;
        r.adapter_path = adapter_path;
        r.success      = true;

        LOG_INFO("LlamaCppTrainer::evaluate: baseline=" +
                 std::to_string(r.baseline_score) +
                 " eval=" + std::to_string(r.eval_score) +
                 " improvement=" + std::to_string(r.improvement_pct) + "%");

    } catch (const std::exception& ex) {
        r.error_message = std::string("exception: ") + ex.what();
        LOG_ERROR("LlamaCppTrainer::evaluate: " + r.error_message);
        try { unload_adapter_internal(); } catch (...) {}
    }
    return r;
}

// ---------------------------------------------------------------------------
// load_adapter() — public
// ---------------------------------------------------------------------------

TrainingResult LlamaCppTrainer::load_adapter(const std::string& adapter_path) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);
    return load_adapter_internal(adapter_path);
}

// ---------------------------------------------------------------------------
// load_adapter_internal() — called with lock held
// ---------------------------------------------------------------------------

TrainingResult LlamaCppTrainer::load_adapter_internal(const std::string& adapter_path) {
    TrainingResult r;
    r.timestamp = utc_now_str();

    if (!fs::exists(adapter_path)) {
        r.error_message = "adapter GGUF not found: " + adapter_path;
        LOG_ERROR("LlamaCppTrainer::load_adapter: " + r.error_message);
        return r;
    }

    auto* lcb = dynamic_cast<LlamaCppBackend*>(&backend_);
    if (!lcb) {
        r.error_message = "backend is not LlamaCppBackend";
        LOG_ERROR("LlamaCppTrainer::load_adapter: " + r.error_message);
        return r;
    }

    llama_model*   model = lcb->get_llama_model();
    llama_context* ctx   = lcb->get_llama_context();

    if (!model || !ctx) {
        r.error_message = "backend model/context null — call load_model() first";
        LOG_ERROR("LlamaCppTrainer::load_adapter: " + r.error_message);
        return r;
    }

    if (adapter_loaded_) unload_adapter_internal();

    llama_adapter_lora* adapter =
        llama_adapter_lora_init(model, adapter_path.c_str());
    if (!adapter) {
        r.error_message = "llama_adapter_lora_init failed for: " + adapter_path;
        LOG_ERROR("LlamaCppTrainer::load_adapter: " + r.error_message);
        return r;
    }

    float scale = 1.0f;
    llama_set_adapters_lora(ctx, &adapter, 1, &scale);

    active_lora_handle_  = adapter;
    adapter_loaded_      = true;
    active_adapter_path_ = adapter_path;

    r.success        = true;
    r.adapter_path   = adapter_path;
    r.adapter_loaded = true;
    LOG_INFO("LlamaCppTrainer: adapter loaded from " + adapter_path);
    return r;
}

// ---------------------------------------------------------------------------
// unload_adapter() — public
// ---------------------------------------------------------------------------

void LlamaCppTrainer::unload_adapter() {
    std::lock_guard<std::mutex> lock(trainer_mutex_);
    unload_adapter_internal();
}

// ---------------------------------------------------------------------------
// unload_adapter_internal() — called with lock held
// ---------------------------------------------------------------------------

void LlamaCppTrainer::unload_adapter_internal() {
    if (!adapter_loaded_ || !active_lora_handle_) return;

    auto* lcb = dynamic_cast<LlamaCppBackend*>(&backend_);
    if (lcb) {
        llama_context* ctx = lcb->get_llama_context();
        if (ctx) llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
    }

    llama_adapter_lora_free(active_lora_handle_);
    active_lora_handle_  = nullptr;
    adapter_loaded_      = false;
    active_adapter_path_ = "";
    LOG_INFO("LlamaCppTrainer: adapter unloaded");
}

// ---------------------------------------------------------------------------
// has_adapter / active_adapter_path
// ---------------------------------------------------------------------------

bool LlamaCppTrainer::has_adapter() const {
    std::lock_guard<std::mutex> lock(trainer_mutex_);
    return adapter_loaded_;
}

std::string LlamaCppTrainer::active_adapter_path() const {
    std::lock_guard<std::mutex> lock(trainer_mutex_);
    return active_adapter_path_;
}

} // namespace cardinal
