// =============================================================================
// Cardinal - TTS Engine Implementation (v1.6.0)
// File: src/voice/tts_engine.cpp
// =============================================================================

#include "voice/tts_engine.h"
#include "utils/logger.h"

// Piper public API — pre-built, installed to vendor/piper/install
#include <piper.hpp>

#include <chrono>

namespace cardinal {

TTSEngine::TTSEngine(const VoiceTTSConfig& config)
    : config_(config)
    , piper_config_(std::make_unique<piper::PiperConfig>())
    , voice_(std::make_unique<piper::Voice>())
{}

TTSEngine::~TTSEngine() {
    shutdown();
}

bool TTSEngine::init() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) return true;

    piper::initialize(*piper_config_);

    std::optional<piper::SpeakerId> speaker_id;
    if (config_.speaker_id >= 0)
        speaker_id = static_cast<piper::SpeakerId>(config_.speaker_id);

    try {
        // useCuda = false — piper uses CPU ONNX runtime
        piper::loadVoice(*piper_config_,
                          config_.model_path,
                          config_.config_path,
                          *voice_,
                          speaker_id,
                          false);
    } catch (const std::exception& ex) {
        LOG_ERROR("TTSEngine: failed to load voice: " + std::string(ex.what()));
        return false;
    }

    // Apply config overrides
    voice_->synthesisConfig.lengthScale = config_.length_scale;
    voice_->synthesisConfig.noiseScale  = config_.noise_scale;
    voice_->synthesisConfig.noiseW      = config_.noise_w;

    ready_ = true;
    LOG_INFO("TTSEngine ready — model: " + config_.model_path);
    return true;
}

void TTSEngine::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) {
        piper::terminate(*piper_config_);
        ready_ = false;
        LOG_INFO("TTSEngine shut down");
    }
}

TTSResult TTSEngine::synthesise(const TTSRequest& request) {
    TTSResult result;
    if (!ready_) { result.error_message = "TTSEngine not initialised"; return result; }
    if (request.text.empty()) { result.error_message = "Empty text"; return result; }

    std::lock_guard<std::mutex> lock(mutex_);

    // Apply per-request overrides
    voice_->synthesisConfig.lengthScale = request.length_scale;
    voice_->synthesisConfig.noiseScale  = request.noise_scale;
    voice_->synthesisConfig.noiseW      = request.noise_w;

    auto t0 = std::chrono::steady_clock::now();

    piper::SynthesisResult piper_result;
    std::vector<int16_t>   audio_buffer;

    try {
        // audioCallback is called each time a chunk is ready — we accumulate
        piper::textToAudio(*piper_config_, *voice_,
                            request.text, audio_buffer, piper_result,
                            []{} /* no-op chunk callback */);
    } catch (const std::exception& ex) {
        result.error_message = ex.what();
        return result;
    }

    auto t1 = std::chrono::steady_clock::now();

    result.success     = true;
    result.samples     = std::move(audio_buffer);
    result.sample_rate = voice_->synthesisConfig.sampleRate;
    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    return result;
}

void TTSEngine::synthesise_streaming(const std::string&  text,
                                      const TTSRequest&   request_template,
                                      TTSSentenceCallback on_sentence)
{
    if (!ready_ || text.empty() || !on_sentence) return;
    for (const auto& sentence : split_sentences(text)) {
        if (sentence.empty()) continue;
        TTSRequest req = request_template;
        req.text       = sentence;
        on_sentence(synthesise(req));
    }
}

// =============================================================================
// split_sentences — splits on '.', '?', '!', '\n'; minimum 3 words
// =============================================================================

std::vector<std::string> TTSEngine::split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string              current;

    auto word_count = [](const std::string& s) -> int {
        int  count   = 0;
        bool in_word = false;
        for (char c : s) {
            bool ws = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
            if (!ws && !in_word) { ++count; in_word = true; }
            else if (ws)          { in_word = false; }
        }
        return count;
    };

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;
        bool boundary = (c == '.' || c == '?' || c == '!' || c == '\n');

        if (boundary) {
            auto lt = current.find_first_not_of(" \t\r\n");
            if (lt == std::string::npos) { current.clear(); continue; }
            auto rt = current.find_last_not_of(" \t\r\n");
            std::string t = current.substr(lt, rt - lt + 1);
            if (word_count(t) >= 3) { sentences.push_back(t); current.clear(); }
        }
    }

    if (!current.empty()) {
        auto lt = current.find_first_not_of(" \t\r\n");
        if (lt != std::string::npos) {
            auto rt = current.find_last_not_of(" \t\r\n");
            std::string t = current.substr(lt, rt - lt + 1);
            if (!t.empty()) sentences.push_back(t);
        }
    }

    return sentences;
}

} // namespace cardinal
