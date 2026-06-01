// =============================================================================
// Cardinal - STT Engine Implementation (v1.6.0)
// File: src/voice/stt_engine.cpp
// =============================================================================

#include "voice/stt_engine.h"
#include "utils/logger.h"

#include <whisper.h>

#include <chrono>
#include <algorithm>

namespace cardinal {

STTEngine::STTEngine(const VoiceSTTConfig& config)
    : config_(config)
{}

STTEngine::~STTEngine() {
    shutdown();
}

bool STTEngine::init() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) return true;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = true;
    cparams.gpu_device = 0;

    ctx_ = whisper_init_from_file_with_params(config_.model_path.c_str(), cparams);
    if (!ctx_) {
        LOG_ERROR("STTEngine: failed to load model: " + config_.model_path);
        return false;
    }

    ready_ = true;
    LOG_INFO("STTEngine ready — model: " + config_.model_path +
             " lang: " + config_.language);
    return true;
}

void STTEngine::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ctx_) {
        whisper_free(ctx_);
        ctx_   = nullptr;
        ready_ = false;
        LOG_INFO("STTEngine shut down");
    }
}

TranscriptResult STTEngine::transcribe(const AudioChunk& audio) {
    TranscriptResult result;

    if (!ready_ || !ctx_) {
        result.error_message = "STTEngine not initialised";
        return result;
    }
    if (audio.empty()) {
        result.error_message = "Empty audio chunk";
        return result;
    }

    auto float_samples = to_float32(audio.samples);

    std::lock_guard<std::mutex> lock(mutex_);

    whisper_full_params params        = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language                   = config_.language.c_str();
    params.translate                  = false;
    params.no_context                 = true;
    params.single_segment             = false;
    params.print_progress             = false;
    params.print_realtime             = false;
    params.print_timestamps           = false;
    params.n_threads                  = config_.threads;
    params.beam_search.beam_size      = config_.beam_size;
    if (!config_.initial_prompt.empty())
        params.initial_prompt = config_.initial_prompt.c_str();

    auto t0 = std::chrono::steady_clock::now();

    int rc = whisper_full(ctx_, params,
                          float_samples.data(),
                          static_cast<int>(float_samples.size()));

    auto t1 = std::chrono::steady_clock::now();

    if (rc != 0) {
        result.error_message = "whisper_full() returned " + std::to_string(rc);
        return result;
    }

    int n_seg = whisper_full_n_segments(ctx_);
    std::string text;
    for (int i = 0; i < n_seg; ++i) {
        const char* seg = whisper_full_get_segment_text(ctx_, i);
        if (seg) text += seg;
    }

    // Trim whitespace
    auto lt = text.find_first_not_of(" \t\r\n");
    if (lt == std::string::npos) {
        text.clear();
    } else {
        auto rt = text.find_last_not_of(" \t\r\n");
        text = text.substr(lt, rt - lt + 1);
    }

    result.text        = text;
    result.success     = true;
    result.confidence  = 1.0f;
    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    LOG_DEBUG("STT: \"" + text + "\" (" + std::to_string(result.duration_ms) + "ms)");
    return result;
}

std::vector<float> STTEngine::to_float32(const std::vector<int16_t>& samples) {
    std::vector<float> out(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
        out[i] = samples[i] / 32768.0f;
    return out;
}

} // namespace cardinal
