// =============================================================================
// Cardinal - TensorRT-LLM Backend Implementation
// File: src/core/backends/tensorrt_backend.cpp
// =============================================================================

#ifdef CARDINAL_ENABLE_TENSORRT

#include "core/backends/tensorrt_backend.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace cardinal {

using json = nlohmann::json;
namespace trtllm = tensorrt_llm::executor;

// =============================================================================
// FeelingSchema
// =============================================================================

const char* FeelingSchema::VALID_REASONING_TYPES[] = {
    "analogical", "causal", "deductive",
    "inductive", "abductive", "associative", nullptr
};

const char* FeelingSchema::VALID_DOMAINS[] = {
    "factual", "ethical", "spatial",
    "temporal", "social", "mathematical", nullptr
};

std::string FeelingSchema::validate(const std::string& json_text) {
    // Trim whitespace
    auto trimmed = json_text;
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    trimmed.erase(trimmed.begin(),
        std::find_if(trimmed.begin(), trimmed.end(), not_space));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), not_space).base(),
        trimmed.end());

    json j;
    try {
        j = json::parse(trimmed);
    } catch (const json::exception& e) {
        return std::string("JSON parse error: ") + e.what();
    }

    if (!j.is_object())
        return "Root must be a JSON object";

    // --- confidence ---
    if (!j.contains("confidence"))
        return "Missing field: confidence";
    if (!j["confidence"].is_number())
        return "Field 'confidence' must be a number";
    float conf = j["confidence"].get<float>();
    if (conf < 0.0f || conf > 1.0f)
        return "Field 'confidence' must be in [0, 1]";

    // --- reasoning_type ---
    if (!j.contains("reasoning_type"))
        return "Missing field: reasoning_type";
    if (!j["reasoning_type"].is_string())
        return "Field 'reasoning_type' must be a string";
    {
        auto rt = j["reasoning_type"].get<std::string>();
        bool valid = false;
        for (int i = 0; VALID_REASONING_TYPES[i] != nullptr; ++i)
            if (rt == VALID_REASONING_TYPES[i]) { valid = true; break; }
        if (!valid)
            return "Invalid reasoning_type: " + rt;
    }

    // --- uncertainty_flag ---
    if (!j.contains("uncertainty_flag"))
        return "Missing field: uncertainty_flag";
    if (!j["uncertainty_flag"].is_boolean())
        return "Field 'uncertainty_flag' must be a boolean";

    // --- rule_candidate_signal ---
    if (!j.contains("rule_candidate_signal"))
        return "Missing field: rule_candidate_signal";
    if (!j["rule_candidate_signal"].is_boolean())
        return "Field 'rule_candidate_signal' must be a boolean";

    // --- contradiction_flag ---
    if (!j.contains("contradiction_flag"))
        return "Missing field: contradiction_flag";
    if (!j["contradiction_flag"].is_boolean())
        return "Field 'contradiction_flag' must be a boolean";

    // --- reasoning_domain ---
    if (!j.contains("reasoning_domain"))
        return "Missing field: reasoning_domain";
    if (!j["reasoning_domain"].is_string())
        return "Field 'reasoning_domain' must be a string";
    {
        auto rd = j["reasoning_domain"].get<std::string>();
        bool valid = false;
        for (int i = 0; VALID_DOMAINS[i] != nullptr; ++i)
            if (rd == VALID_DOMAINS[i]) { valid = true; break; }
        if (!valid)
            return "Invalid reasoning_domain: " + rd;
    }

    return ""; // all good
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

TensorRTBackend::TensorRTBackend(const CardinalConfig& config)
    : config_(config)
{
    LOG_INFO("TensorRTBackend created");
    LOG_INFO("  Engine:    " + config_.backend.tensorrt.engine_path);
    LOG_INFO("  Tokenizer: " + config_.backend.tensorrt.tokenizer_path);
}

TensorRTBackend::~TensorRTBackend() {
    unload();
}

// =============================================================================
// load_model
// =============================================================================

void TensorRTBackend::load_model() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (ready_) {
        LOG_WARN("TensorRTBackend::load_model called but already loaded");
        return;
    }

    // -------------------------------------------------------------------------
    // Validate engine path
    // -------------------------------------------------------------------------
    const auto& engine_path = config_.backend.tensorrt.engine_path;
    if (!std::filesystem::exists(engine_path)) {
        throw LLMError("TRT engine not found: " + engine_path +
                       "\nBuild it with: trtllm-build --checkpoint_dir <hf_model> "
                       "--output_dir " + engine_path);
    }

    // -------------------------------------------------------------------------
    // Load tokenizer
    // -------------------------------------------------------------------------
    const auto& tok_path = config_.backend.tensorrt.tokenizer_path;
    LOG_INFO("Loading tokenizer from: " + tok_path);

    // tokenizers-cpp can load from a directory containing tokenizer.json
    // (HuggingFace format) or from a single .json file directly.
    std::string tok_file = tok_path;
    if (std::filesystem::is_directory(tok_path)) {
        tok_file = tok_path + "/tokenizer.json";
    }

    if (!std::filesystem::exists(tok_file)) {
        throw LLMError("Tokenizer file not found: " + tok_file);
    }

    std::ifstream ifs(tok_file);
    if (!ifs.is_open()) {
        throw LLMError("Cannot open tokenizer file: " + tok_file);
    }
    std::string tok_json((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());

    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tok_json);
    if (!tokenizer_) {
        throw LLMError("Failed to initialise tokenizer from: " + tok_file);
    }
    LOG_INFO("Tokenizer loaded");

    // -------------------------------------------------------------------------
    // Build TRT-LLM executor
    // The executor manages the engine lifecycle, KV cache, and request queue.
    // We use in-process mode (single GPU, no MPI) appropriate for sm_86.
    // -------------------------------------------------------------------------
    LOG_INFO("Initialising TRT-LLM executor...");
    LOG_INFO("  max_batch_size:    " +
             std::to_string(config_.backend.tensorrt.max_batch_size));
    LOG_INFO("  max_seq_len:       " +
             std::to_string(config_.backend.tensorrt.max_seq_len));
    LOG_INFO("  kv_cache_fraction: " +
             std::to_string(config_.backend.tensorrt.kv_cache_fraction));

    trtllm::ExecutorConfig exec_config;

    // KV cache configuration
    trtllm::KvCacheConfig kv_config;
    kv_config.setFreeGpuMemoryFraction(config_.backend.tensorrt.kv_cache_fraction);
    exec_config.setKvCacheConfig(kv_config);

    // Scheduling: GUARANTEED_NO_EVICT keeps requests in flight until complete.
    // Better for low-latency single-user use than STATIC_BATCH.
    exec_config.setSchedulerConfig(
        trtllm::SchedulerConfig(trtllm::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT));

    // Batching mode: INFLIGHT allows the executor to overlap prefill + decode
    // across concurrent requests. For Cardinal (single session) this gives
    // better GPU utilisation on the 3050 than STATIC mode.
    exec_config.setBatchingType(trtllm::BatchingType::kINFLIGHT);

    // Parallel config: single GPU, no tensor/pipeline parallelism
    trtllm::ParallelConfig par_config;
    par_config.setCommunicationMode(trtllm::CommunicationMode::kLEADER);
    exec_config.setParallelConfig(par_config);

    try {
        executor_ = std::make_unique<trtllm::Executor>(
            engine_path,
            trtllm::ModelType::kDECODER_ONLY,
            exec_config
        );
    } catch (const std::exception& e) {
        throw LLMError(std::string("TRT-LLM executor init failed: ") + e.what());
    }

    max_seq_len_ = config_.backend.tensorrt.max_seq_len;
    ready_       = true;

    LOG_INFO("TensorRTBackend ready");
    LOG_INFO("  max_seq_len: " + std::to_string(max_seq_len_));
}

// =============================================================================
// is_ready / unload
// =============================================================================

bool TensorRTBackend::is_ready() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_;
}

void TensorRTBackend::unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ready_) return;

    executor_.reset();
    tokenizer_.reset();
    ready_ = false;

    LOG_INFO("TensorRTBackend unloaded");
}

// =============================================================================
// get_info
// =============================================================================

BackendInfo TensorRTBackend::get_info() const {
    BackendInfo info;
    info.type           = BackendType::TENSORRT;
    info.name           = "TensorRT-LLM";
    info.version        = "0.8+";   // populated at build time if desired
    info.model_name     = config_.backend.tensorrt.engine_path;
    info.context_length = max_seq_len_;
    info.n_vocab        = vocab_size_;
    info.gpu_enabled    = true;     // TRT-LLM always runs on GPU
    info.gpu_layers     = -1;       // N/A — whole model is on GPU
    return info;
}

// =============================================================================
// Pass 1: generate_feeling — JSON validation + retry loop
// =============================================================================

GenerationResult TensorRTBackend::generate_feeling(
    FeelingContext&                 ctx,
    const std::vector<ChatMessage>& messages)
{
    if (!ready_) throw LLMError("Engine not ready - call load_model() first");

    ctx.set_state(PassState::PASS1_FEELING);
    ctx.start_pass1_timer();

    LOG_DEBUG("Pass 1 (TRT): generating feeling output (JSON retry loop)");

    // Augment messages with feeling instruction
    auto feeling_messages = inject_feeling_instruction(messages);
    std::string prompt    = apply_chat_template(feeling_messages, true);

    // Track prompt tokens for metrics
    int prompt_tok = count_tokens(prompt);
    ctx.metrics().prompt_tokens = prompt_tok;
    session_prompt_tokens_.fetch_add(prompt_tok, std::memory_order_relaxed);

    // Retry loop
    auto result = generate_feeling_with_retry(
        prompt,
        config_.feedback.max_retries,
        config_.feedback.retry_delay_ms
    );

    ctx.stop_pass1_timer();
    ctx.metrics().pass1_tokens_generated = result.tokens_generated;

    if (result.success) {
        ctx.set_raw_feeling(result.text);
        if (!ctx.parse_feeling()) {
            // Validation passed but parse failed — shouldn't happen
            LOG_WARN("TRT Pass 1: validation passed but ctx.parse_feeling() failed");
            result.success = false;
        }
    }

    LOG_DEBUG("Pass 1 (TRT) complete: " + result.text);
    return result;
}

// =============================================================================
// generate_feeling_with_retry
// =============================================================================

GenerationResult TensorRTBackend::generate_feeling_with_retry(
    const std::string& prompt,
    int                max_retries,
    int                retry_delay_ms)
{
    // Use temperature=0 for Pass 1 — deterministic, matches GBNF greedy behaviour
    // on the llama.cpp side.
    constexpr float FEELING_TEMPERATURE = 0.0f;
    constexpr float FEELING_TOP_P       = 1.0f;

    GenerationResult last_result{};
    last_result.success = false;

    std::string correction_suffix;  // injected after first failure

    for (int attempt = 0; attempt <= max_retries; ++attempt) {
        if (attempt > 0) {
            LOG_DEBUG("TRT Pass 1 retry " + std::to_string(attempt) +
                      "/" + std::to_string(max_retries));
            if (retry_delay_ms > 0)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(retry_delay_ms));
        }

        // On retry, append a corrective note to the prompt
        std::string effective_prompt = prompt;
        if (attempt > 0 && !correction_suffix.empty()) {
            effective_prompt += correction_suffix;
        }

        last_result = generate_sync(
            effective_prompt,
            config_.feeling_schema.max_tokens,
            FEELING_TEMPERATURE,
            FEELING_TOP_P
        );

        if (!last_result.success) {
            LOG_WARN("TRT Pass 1 attempt " + std::to_string(attempt) +
                     ": generation failed");
            continue;
        }

        // Validate JSON schema
        std::string err = FeelingSchema::validate(last_result.text);
        if (err.empty()) {
            // Valid — done
            LOG_DEBUG("TRT Pass 1 attempt " + std::to_string(attempt) +
                      ": schema valid");
            return last_result;
        }

        LOG_WARN("TRT Pass 1 attempt " + std::to_string(attempt) +
                 ": schema invalid (" + err + ")");

        // Build corrective suffix for next attempt
        correction_suffix =
            "\n\nYour previous output was invalid: " + err +
            "\nOutput ONLY a valid JSON object with exactly these fields: "
            "{\"confidence\": <float 0-1>, "
            "\"reasoning_type\": <analogical|causal|deductive|inductive|abductive|associative>, "
            "\"uncertainty_flag\": <true|false>, "
            "\"rule_candidate_signal\": <true|false>, "
            "\"contradiction_flag\": <true|false>, "
            "\"reasoning_domain\": <factual|ethical|spatial|temporal|social|mathematical>}";

        last_result.success = false;
    }

    LOG_WARN("TRT Pass 1: all " + std::to_string(max_retries + 1) +
             " attempts exhausted");
    return last_result;
}

// =============================================================================
// Pass 2: generate_response
// =============================================================================

GenerationResult TensorRTBackend::generate_response(
    FeelingContext&                 ctx,
    const std::vector<ChatMessage>& messages,
    const TokenCallback&            callback)
{
    if (!ready_)                throw LLMError("Engine not ready - call load_model() first");
    if (!ctx.has_valid_feeling()) throw LLMError("Pass 2 called without valid feeling output");

    ctx.set_state(PassState::PASS2_RESPONSE);
    ctx.start_pass2_timer();

    LOG_DEBUG("Pass 2 (TRT): generating final response");

    ctx.prepare_synthetic_turn();
    auto augmented    = inject_synthetic_turn(messages, ctx.synthetic_turn());
    std::string prompt = apply_chat_template(augmented, true);

    GenerationResult result;
    if (callback) {
        result = generate_stream(
            prompt,
            config_.inference.max_tokens_response,
            config_.inference.temperature,
            config_.inference.top_p,
            callback
        );
    } else {
        result = generate_sync(
            prompt,
            config_.inference.max_tokens_response,
            config_.inference.temperature,
            config_.inference.top_p
        );
    }

    ctx.stop_pass2_timer();
    ctx.metrics().pass2_tokens_generated = result.tokens_generated;

    if (result.success) {
        ctx.set_final_response(result.text);
        ctx.set_state(PassState::COMPLETE);
    } else {
        ctx.set_state(PassState::FAILED);
    }

    LOG_DEBUG("Pass 2 (TRT) complete: " +
              std::to_string(result.tokens_generated) + " tokens");
    LOG_INFO(ctx.metrics().to_string());
    return result;
}

// =============================================================================
// generate_sync — blocking single-request generation
// =============================================================================

GenerationResult TensorRTBackend::generate_sync(
    const std::string& prompt,
    int                max_tokens,
    float              temperature,
    float              top_p)
{
    GenerationResult result{};
    result.success       = false;
    result.stopped_eos   = false;
    result.stopped_limit = false;
    result.stopped_abort = false;

    // Tokenize prompt
    auto input_ids = encode(prompt, true);
    result.tokens_prompted = static_cast<int>(input_ids.size());

    if (input_ids.empty()) {
        LOG_WARN("TRT generate_sync: empty token list after encoding");
        return result;
    }

    // Guard context limit
    int effective_max = max_tokens;
    if (result.tokens_prompted + effective_max > max_seq_len_) {
        effective_max = max_seq_len_ - result.tokens_prompted - 4;
        LOG_WARN("TRT: truncating max_tokens to " + std::to_string(effective_max));
        if (effective_max <= 0) {
            throw LLMError("TRT: context full — cannot generate");
        }
    }

    // Build request
    trtllm::Request req(
        trtllm::VecTokens(input_ids.begin(), input_ids.end()),
        effective_max
    );
    req.setSamplingConfig(build_sampling_config(temperature, top_p, effective_max));
    req.setOutputConfig(build_output_config());

    // Submit to executor
    auto req_id = executor_->enqueueRequest(req);

    // Poll for completion
    std::string generated_text;
    int tokens_generated = 0;

    while (true) {
        // Wait up to 60 seconds total; poll in 1ms increments
        auto responses = executor_->awaitResponses(req_id,
            std::chrono::milliseconds(60000));

        if (responses.empty()) {
            LOG_WARN("TRT generate_sync: timeout waiting for response");
            break;
        }

        for (auto& resp : responses) {
            if (resp.hasError()) {
                LOG_WARN("TRT executor error: " + resp.getErrorMsg());
                return result;
            }

            auto& result_data = resp.getResult();

            // Decode the newly generated tokens in this response
            auto& output_ids = result_data.outputTokenIds;
            if (!output_ids.empty() && !output_ids[0].empty()) {
                // output_ids[beam][token_position] — we use beam 0
                auto new_text = decode(
                    std::vector<int32_t>(output_ids[0].begin(),
                                         output_ids[0].end()));
                generated_text  += new_text;
                tokens_generated += static_cast<int>(output_ids[0].size());
            }

            // Check stop reason
            if (result_data.isFinal) {
                if (result_data.finishReasons[0] ==
                    trtllm::FinishReason::kEND_ID) {
                    result.stopped_eos = true;
                } else if (result_data.finishReasons[0] ==
                           trtllm::FinishReason::kLENGTH) {
                    result.stopped_limit = true;
                }
                goto done;
            }
        }
    }

done:
    total_tokens_generated_.fetch_add(tokens_generated, std::memory_order_relaxed);

    result.text             = generated_text;
    result.tokens_generated = tokens_generated;
    result.success          = true;
    return result;
}

// =============================================================================
// generate_stream — streaming generation with per-token callback
// =============================================================================

GenerationResult TensorRTBackend::generate_stream(
    const std::string&   prompt,
    int                  max_tokens,
    float                temperature,
    float                top_p,
    const TokenCallback& callback)
{
    GenerationResult result{};
    result.success       = false;
    result.stopped_eos   = false;
    result.stopped_limit = false;
    result.stopped_abort = false;

    auto input_ids = encode(prompt, true);
    result.tokens_prompted = static_cast<int>(input_ids.size());

    if (input_ids.empty()) {
        LOG_WARN("TRT generate_stream: empty token list after encoding");
        return result;
    }

    int effective_max = max_tokens;
    if (result.tokens_prompted + effective_max > max_seq_len_) {
        effective_max = max_seq_len_ - result.tokens_prompted - 4;
        if (effective_max <= 0)
            throw LLMError("TRT: context full — cannot generate");
    }

    // Streaming: set streaming=true so executor returns incremental responses
    trtllm::Request req(
        trtllm::VecTokens(input_ids.begin(), input_ids.end()),
        effective_max
    );
    req.setSamplingConfig(build_sampling_config(temperature, top_p, effective_max));
    req.setOutputConfig(build_output_config());
    req.setStreaming(true);  // enable incremental token delivery

    auto req_id = executor_->enqueueRequest(req);

    std::string full_text;
    int tokens_generated = 0;
    // Track previously decoded position to extract only new tokens per chunk
    std::vector<int32_t> accumulated_ids;

    while (true) {
        auto responses = executor_->awaitResponses(req_id,
            std::chrono::milliseconds(60000));

        if (responses.empty()) {
            LOG_WARN("TRT generate_stream: timeout");
            break;
        }

        for (auto& resp : responses) {
            if (resp.hasError()) {
                LOG_WARN("TRT stream error: " + resp.getErrorMsg());
                goto stream_done;
            }

            auto& result_data = resp.getResult();
            auto& output_ids  = result_data.outputTokenIds;

            if (!output_ids.empty() && !output_ids[0].empty()) {
                // TRT-LLM streaming returns cumulative output_ids each chunk.
                // Extract only new tokens by diffing against accumulated_ids.
                auto& all_ids = output_ids[0];
                size_t prev_size = accumulated_ids.size();

                if (all_ids.size() > prev_size) {
                    std::vector<int32_t> new_ids(
                        all_ids.begin() + static_cast<ptrdiff_t>(prev_size),
                        all_ids.end());

                    accumulated_ids.insert(accumulated_ids.end(),
                                           new_ids.begin(), new_ids.end());

                    std::string token_text = decode(new_ids);
                    full_text += token_text;
                    tokens_generated += static_cast<int>(new_ids.size());

                    if (callback) {
                        bool cont = callback(token_text,
                                             static_cast<int>(new_ids.back()),
                                             tokens_generated);
                        if (!cont) {
                            result.stopped_abort = true;
                            // Cancel remaining generation
                            executor_->cancelRequest(req_id);
                            goto stream_done;
                        }
                    }
                }
            }

            if (result_data.isFinal) {
                if (result_data.finishReasons[0] == trtllm::FinishReason::kEND_ID)
                    result.stopped_eos = true;
                else if (result_data.finishReasons[0] == trtllm::FinishReason::kLENGTH)
                    result.stopped_limit = true;
                goto stream_done;
            }
        }
    }

stream_done:
    total_tokens_generated_.fetch_add(tokens_generated, std::memory_order_relaxed);

    result.text             = full_text;
    result.tokens_generated = tokens_generated;
    result.success          = !result.stopped_abort || tokens_generated > 0;
    return result;
}

// =============================================================================
// build_sampling_config
// =============================================================================

trtllm::SamplingConfig TensorRTBackend::build_sampling_config(
    float temperature, float top_p, int max_tokens) const
{
    trtllm::SamplingConfig cfg;
    cfg.setTemperature(temperature);
    cfg.setTopP(top_p);
    // beamWidth=1 for greedy/sampling (no beam search)
    cfg.setBeamWidth(1);
    return cfg;
}

// =============================================================================
// build_output_config
// =============================================================================

trtllm::OutputConfig TensorRTBackend::build_output_config(bool return_log_probs) const {
    trtllm::OutputConfig cfg;
    cfg.returnLogProbs  = return_log_probs;
    cfg.returnContextLogits = false;
    cfg.returnGenerationLogits = false;
    cfg.excludeInputFromOutput = true;  // output_ids contains only generated tokens
    return cfg;
}

// =============================================================================
// Context management
// =============================================================================

void TensorRTBackend::clear_kv_cache() {
    // TRT-LLM's executor manages KV cache per-request automatically.
    // There is no global "clear all" API — instead we reset our token counters
    // so context_near_limit() reflects the fresh session state.
    total_tokens_generated_.store(0, std::memory_order_relaxed);
    session_prompt_tokens_.store(0, std::memory_order_relaxed);
    LOG_DEBUG("TRT: session token counters reset (KV cache is per-request)");
}

int TensorRTBackend::context_tokens_used() const {
    return session_prompt_tokens_.load(std::memory_order_relaxed) +
           total_tokens_generated_.load(std::memory_order_relaxed);
}

int TensorRTBackend::context_tokens_remaining() const {
    return max_seq_len_ - context_tokens_used();
}

bool TensorRTBackend::context_near_limit(float threshold) const {
    if (max_seq_len_ == 0) return false;
    return static_cast<float>(context_tokens_used()) /
           static_cast<float>(max_seq_len_) >= threshold;
}

// =============================================================================
// Tokenization
// =============================================================================

std::vector<int32_t> TensorRTBackend::encode(
    const std::string& text, bool add_special) const
{
    if (!tokenizer_) throw LLMError("Tokenizer not loaded");

    auto encoded = tokenizer_->Encode(text);
    std::vector<int32_t> ids;
    ids.reserve(encoded.size());
    for (auto id : encoded) ids.push_back(static_cast<int32_t>(id));
    return ids;
}

std::string TensorRTBackend::decode(const std::vector<int32_t>& token_ids) const {
    if (!tokenizer_) throw LLMError("Tokenizer not loaded");

    std::vector<int> ids(token_ids.begin(), token_ids.end());
    return tokenizer_->Decode(ids);
}

int TensorRTBackend::count_tokens(const std::string& text) const {
    return static_cast<int>(encode(text, false).size());
}

// =============================================================================
// Chat template
// =============================================================================

std::string TensorRTBackend::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt) const
{
    const auto& tmpl = config_.backend.tensorrt.chat_template;

    if (tmpl == "qwen3")
        return render_qwen3(messages, add_generation_prompt);
    if (tmpl == "llama3")
        return render_llama3(messages, add_generation_prompt);
    if (tmpl == "chatml")
        return render_chatml(messages, add_generation_prompt);

    // Unknown template — fall back to chatml which is the most common
    LOG_WARN("TRT: unknown chat_template '" + tmpl +
             "' — falling back to chatml");
    return render_chatml(messages, add_generation_prompt);
}

// Qwen3 uses the same im_start/im_end tokens as ChatML but with
// a <think> block pattern. For inference we just need the standard
// im_start/im_end wrapping; the model handles its own chain-of-thought.
std::string TensorRTBackend::render_qwen3(
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt) const
{
    std::ostringstream oss;
    for (const auto& msg : messages) {
        oss << "<|im_start|>" << msg.role << "\n"
            << msg.content
            << "<|im_end|>\n";
    }
    if (add_generation_prompt)
        oss << "<|im_start|>assistant\n";
    return oss.str();
}

std::string TensorRTBackend::render_llama3(
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt) const
{
    std::ostringstream oss;
    oss << "<|begin_of_text|>";
    for (const auto& msg : messages) {
        oss << "<|start_header_id|>" << msg.role << "<|end_header_id|>\n\n"
            << msg.content
            << "<|eot_id|>";
    }
    if (add_generation_prompt)
        oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return oss.str();
}

std::string TensorRTBackend::render_chatml(
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt) const
{
    std::ostringstream oss;
    for (const auto& msg : messages) {
        oss << "<|im_start|>" << msg.role << "\n"
            << msg.content
            << "<|im_end|>\n";
    }
    if (add_generation_prompt)
        oss << "<|im_start|>assistant\n";
    return oss.str();
}

// =============================================================================
// inject_feeling_instruction
// =============================================================================

std::vector<ChatMessage> TensorRTBackend::inject_feeling_instruction(
    const std::vector<ChatMessage>& messages) const
{
    std::vector<ChatMessage> out = messages;
    bool has_system = false;

    for (auto& msg : out) {
        if (msg.role == "system") {
            msg.content +=
                "\n\nBefore responding, output your internal state as "
                "a JSON object with exactly these fields: "
                "confidence (float 0-1), reasoning_type "
                "(analogical|causal|deductive|inductive|abductive|associative), "
                "uncertainty_flag (bool), rule_candidate_signal (bool), "
                "contradiction_flag (bool), reasoning_domain "
                "(factual|ethical|spatial|temporal|social|mathematical). "
                "Output ONLY the JSON object, nothing else.";
            has_system = true;
            break;
        }
    }

    if (!has_system) {
        out.insert(out.begin(), {
            "system",
            "You are Cardinal, a neurosymbolic AI. Before responding, output your "
            "internal state as a JSON object with exactly these fields: "
            "confidence (float 0-1), reasoning_type "
            "(analogical|causal|deductive|inductive|abductive|associative), "
            "uncertainty_flag (bool), rule_candidate_signal (bool), "
            "contradiction_flag (bool), reasoning_domain "
            "(factual|ethical|spatial|temporal|social|mathematical). "
            "Output ONLY the JSON object, nothing else."
        });
    }

    return out;
}

// =============================================================================
// inject_synthetic_turn
// =============================================================================

std::vector<ChatMessage> TensorRTBackend::inject_synthetic_turn(
    const std::vector<ChatMessage>& messages,
    const SyntheticTurn&            turn) const
{
    std::vector<ChatMessage> augmented;
    augmented.reserve(messages.size() + 1);

    if (!messages.empty() && messages.back().role == "user") {
        for (size_t i = 0; i < messages.size() - 1; ++i)
            augmented.push_back(messages[i]);
        augmented.push_back({ turn.role, turn.format() });
        augmented.push_back(messages.back());
    } else {
        augmented = messages;
        augmented.push_back({ turn.role, turn.format() });
    }

    return augmented;
}

} // namespace cardinal

#endif // CARDINAL_ENABLE_TENSORRT
