// =============================================================================
// Cardinal - Vision Encoder Implementation
// File: src/vision/vision_encoder.cpp
// =============================================================================

#include "vision/vision_encoder.h"
#include "utils/logger.h"

#include <chrono>
#include <filesystem>

namespace cardinal {

VisionEncoder::VisionEncoder(const CardinalConfig& config)
    : config_(config)
{}

VisionEncoder::~VisionEncoder() {
    unload();
}

// =========================================================================
// load
// =========================================================================

void VisionEncoder::load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) return;

    const auto& vc = config_.vision;

    if (vc.model_path.empty()) {
        LOG_INFO("VisionEncoder: model_path empty — vision disabled");
        return;
    }
    if (!std::filesystem::exists(vc.model_path)) {
        LOG_WARN("VisionEncoder: model not found: " + vc.model_path);
        return;
    }
    if (!std::filesystem::exists(vc.mmproj_path)) {
        LOG_WARN("VisionEncoder: mmproj not found: " + vc.mmproj_path);
        return;
    }

#ifndef CARDINAL_MTMD_AVAILABLE
    LOG_WARN("VisionEncoder: built without mtmd support — vision disabled");
    LOG_WARN("  Add vendor/llama.cpp/tools/mtmd to include path in CMakeLists");
    return;
#else
    LOG_INFO("VisionEncoder: loading text model: " + vc.model_path);
    LOG_INFO("VisionEncoder: loading mmproj:     " + vc.mmproj_path);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = vc.gpu_layers;

    model_ = llama_model_load_from_file(vc.model_path.c_str(), model_params);
    if (!model_) {
        LOG_WARN("VisionEncoder: failed to load text model");
        return;
    }

    llama_context_params ctx_params  = llama_context_default_params();
    ctx_params.n_ctx                 = 4096;
    ctx_params.n_batch               = 4096;
    ctx_params.n_ubatch              = 512;
    ctx_params.n_threads             = vc.threads;
    ctx_params.n_threads_batch       = vc.threads;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_); model_ = nullptr;
        LOG_WARN("VisionEncoder: failed to create context");
        return;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu             = (vc.gpu_layers > 0);
    mtmd_params.n_threads           = vc.threads;
    mtmd_params.print_timings       = false;
    mtmd_params.warmup              = false;

    mtmd_ = mtmd_init_from_file(vc.mmproj_path.c_str(), model_, mtmd_params);
    if (!mtmd_) {
        llama_free(ctx_); ctx_ = nullptr;
        llama_model_free(model_); model_ = nullptr;
        LOG_WARN("VisionEncoder: failed to init mtmd context");
        return;
    }

    if (!mtmd_support_vision(mtmd_)) {
        LOG_WARN("VisionEncoder: mmproj does not support vision");
        mtmd_free(mtmd_); mtmd_ = nullptr;
        llama_free(ctx_); ctx_ = nullptr;
        llama_model_free(model_); model_ = nullptr;
        return;
    }

    ready_ = true;
    LOG_INFO("VisionEncoder: ready (" +
             std::string(vc.gpu_layers > 0 ? "GPU" : "CPU") +
             ", " + std::to_string(vc.threads) + " threads)");
#endif // CARDINAL_MTMD_AVAILABLE
}

// =========================================================================
// unload
// =========================================================================

void VisionEncoder::unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ready_) return;

#ifdef CARDINAL_MTMD_AVAILABLE
    if (mtmd_)  { mtmd_free(mtmd_);         mtmd_  = nullptr; }
#endif
    if (ctx_)   { llama_free(ctx_);          ctx_   = nullptr; }
    if (model_) { llama_model_free(model_);  model_ = nullptr; }

    ready_ = false;
    LOG_INFO("VisionEncoder: unloaded");
}

// =========================================================================
// encode
// =========================================================================

VisionResult VisionEncoder::encode(const std::string&   image_path,
                                    const std::string&   user_prompt,
                                    const ImageMetadata& metadata) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    VisionResult result;
    result.metadata = metadata;

    if (!ready_) {
        result.success       = false;
        result.error_message = "Vision encoder not loaded";
        return result;
    }

#ifndef CARDINAL_MTMD_AVAILABLE
    result.success       = false;
    result.error_message = "Vision encoding requires mtmd support (CARDINAL_MTMD_AVAILABLE)";
    return result;
#else
    auto start = std::chrono::steady_clock::now();

    const std::string& prompt_text =
        user_prompt.empty() ? DEFAULT_PROMPT : user_prompt;

    LOG_INFO("VisionEncoder: encoding " + image_path);

    // Load image as bitmap
    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(mtmd_,
                                                              image_path.c_str());
    if (!bitmap) {
        result.success       = false;
        result.error_message = "Failed to load image: " + image_path;
        return result;
    }

    // Build prompt with image marker
    std::string marker      = mtmd_default_marker();
    std::string full_prompt = marker + "\n\nQuestion: " + prompt_text +
                              "\n\nAnswer:";

    // Tokenize prompt + image
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
        mtmd_bitmap_free(bitmap);
        result.success       = false;
        result.error_message = "Failed to allocate input chunks";
        return result;
    }

    mtmd_input_text input_text;
    input_text.text          = full_prompt.c_str();
    input_text.add_special   = true;
    input_text.parse_special = true;

    const mtmd_bitmap* bitmaps_arr[] = { bitmap };

    int32_t tok_ret = mtmd_tokenize(mtmd_, chunks, &input_text,
                                     bitmaps_arr, 1);
    mtmd_bitmap_free(bitmap);

    if (tok_ret != 0) {
        mtmd_input_chunks_free(chunks);
        result.success       = false;
        result.error_message = "mtmd_tokenize failed (code " +
                               std::to_string(tok_ret) + ")";
        return result;
    }

    // Clear KV cache and eval all chunks
    llama_memory_clear(llama_get_memory(ctx_), false);

    llama_pos n_past     = 0;
    llama_pos new_n_past = 0;

    int32_t eval_ret = mtmd_helper_eval_chunks(
        mtmd_, ctx_, chunks,
        n_past, /*seq_id=*/0,
        static_cast<int32_t>(llama_n_batch(ctx_)),
        /*logits_last=*/true,
        &new_n_past);

    mtmd_input_chunks_free(chunks);

    if (eval_ret != 0) {
        result.success       = false;
        result.error_message = "mtmd_helper_eval_chunks failed (code " +
                               std::to_string(eval_ret) + ")";
        return result;
    }

    // Sampling loop
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    llama_sampler* sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.3f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    std::string description;
    description.reserve(config_.vision.max_tokens * 4);

    for (int i = 0; i < config_.vision.max_tokens; ++i) {
        llama_token token_id = llama_sampler_sample(sampler, ctx_, -1);
        if (llama_vocab_is_eog(vocab, token_id)) break;

        char buf[256] = {};
        int len = llama_token_to_piece(vocab, token_id,
                                        buf, sizeof(buf)-1, 0, true);
        if (len > 0) description.append(buf, len);

        llama_batch next = llama_batch_get_one(&token_id, 1);
        if (llama_decode(ctx_, next) != 0) break;
    }

    llama_sampler_free(sampler);

    while (!description.empty() &&
           (description.back() == '\n' || description.back() == ' '))
        description.pop_back();

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count());

    result.success     = !description.empty();
    result.description = description;

    if (!result.success)
        result.error_message = "No description generated";
    else
        LOG_INFO("VisionEncoder: encoded in " +
                 std::to_string(result.duration_ms) + "ms");

    return result;
#endif // CARDINAL_MTMD_AVAILABLE
}

// =========================================================================
// ctx_params_n_batch
// =========================================================================

int VisionEncoder::ctx_params_n_batch() const {
    if (!ctx_) return 512;
    return static_cast<int>(llama_n_batch(ctx_));
}

} // namespace cardinal
