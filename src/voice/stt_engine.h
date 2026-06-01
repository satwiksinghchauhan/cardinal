#pragma once
// =============================================================================
// Cardinal - STT Engine (v1.6.0)
// File: src/voice/stt_engine.h
//
// Whisper.cpp wrapper — CUDA accelerated, per-utterance transcription.
// Uses whisper_full() with no_context=true so each call is independent.
// =============================================================================

#include "voice/voice_types.h"

#include <string>
#include <vector>
#include <memory>
#include <mutex>

struct whisper_context;

namespace cardinal {

    class STTEngine {
    public:
        explicit STTEngine(const VoiceSTTConfig& config);
        ~STTEngine();

        STTEngine(const STTEngine&)            = delete;
        STTEngine& operator=(const STTEngine&) = delete;

        bool init();
        void shutdown();
        bool is_ready() const { return ready_; }

        // Blocks until Whisper finishes. Call from a dedicated worker thread.
        // audio must be 16-bit PCM mono at 16 kHz.
        TranscriptResult transcribe(const AudioChunk& audio);

    private:
        static std::vector<float> to_float32(const std::vector<int16_t>& samples);

        const VoiceSTTConfig config_;
        whisper_context*     ctx_{ nullptr };
        bool                 ready_{ false };
        mutable std::mutex   mutex_;
    };

} // namespace cardinal
