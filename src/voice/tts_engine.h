#pragma once
// =============================================================================
// Cardinal - TTS Engine (v1.6.0)
// File: src/voice/tts_engine.h
//
// Piper TTS wrapper — uses piper::textToAudio() from piper.hpp directly.
// ONNX runtime, CPU inference.
// Sentence-streaming: split text on boundary chars, synthesise each sentence
// independently for low-latency playback.
// =============================================================================

#include "voice/voice_types.h"

#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <memory>

namespace piper {
    struct PiperConfig;
    struct Voice;
}

namespace cardinal {

    using TTSSentenceCallback = std::function<void(TTSResult)>;

    class TTSEngine {
    public:
        explicit TTSEngine(const VoiceTTSConfig& config);
        ~TTSEngine();

        TTSEngine(const TTSEngine&)            = delete;
        TTSEngine& operator=(const TTSEngine&) = delete;

        bool init();
        void shutdown();
        bool is_ready() const { return ready_; }

        // Blocks until Piper finishes.
        TTSResult synthesise(const TTSRequest& request);

        // Sentence-streaming: splits text, calls on_sentence for each chunk.
        void synthesise_streaming(const std::string&  text,
                                   const TTSRequest&   request_template,
                                   TTSSentenceCallback on_sentence);

        // Splits text into speakable sentences (public for VoiceLoop use).
        static std::vector<std::string> split_sentences(const std::string& text);

    private:
        const VoiceTTSConfig                config_;
        std::unique_ptr<piper::PiperConfig> piper_config_;
        std::unique_ptr<piper::Voice>       voice_;
        bool                                ready_{ false };
        mutable std::mutex                  mutex_;
    };

} // namespace cardinal
