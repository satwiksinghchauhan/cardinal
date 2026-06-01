#pragma once
// =============================================================================
// Cardinal - Voice Types (v1.6.0)
// File: src/voice/voice_types.h
//
// Shared types for the voice subsystem.
// No dependencies on other Cardinal headers — safe to include anywhere.
// =============================================================================

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace cardinal {

    // =========================================================================
    // VoiceInputMode
    // =========================================================================
    enum class VoiceInputMode {
        PUSH_TO_TALK,       // hold key to record
        VAD,                // voice activity detection, auto start/stop
        WAKE_WORD           // passive listening, activates on wake phrase
    };

    inline std::string voice_input_mode_to_string(VoiceInputMode m) {
        switch (m) {
        case VoiceInputMode::PUSH_TO_TALK: return "push_to_talk";
        case VoiceInputMode::VAD:          return "vad";
        case VoiceInputMode::WAKE_WORD:    return "wake_word";
        default:                           return "vad";
        }
    }

    inline VoiceInputMode voice_input_mode_from_string(const std::string& s) {
        if (s == "push_to_talk" || s == "ptt") return VoiceInputMode::PUSH_TO_TALK;
        if (s == "wake_word")                   return VoiceInputMode::WAKE_WORD;
        return VoiceInputMode::VAD;
    }

    // =========================================================================
    // TTSStreamingMode
    // =========================================================================
    enum class TTSStreamingMode {
        SENTENCE,   // speak each sentence as response generates
        FULL        // wait for complete response, then speak
    };

    inline std::string tts_streaming_mode_to_string(TTSStreamingMode m) {
        return m == TTSStreamingMode::SENTENCE ? "sentence" : "full";
    }

    // =========================================================================
    // VoiceLoopState
    // =========================================================================
    enum class VoiceLoopState {
        IDLE,
        PASSIVE_LISTENING,  // wake word thread active
        LISTENING,          // VAD active, waiting for speech
        RECORDING,          // capturing audio to buffer
        TRANSCRIBING,       // whisper.cpp processing
        INFERRING,          // CardinalAPI::chat_stream() running
        SPEAKING,           // TTS audio playing
        INTERRUPTED,        // barge-in detected while SPEAKING
        STOPPING            // shutdown requested
    };

    inline std::string voice_loop_state_to_string(VoiceLoopState s) {
        switch (s) {
        case VoiceLoopState::IDLE:              return "idle";
        case VoiceLoopState::PASSIVE_LISTENING: return "passive_listening";
        case VoiceLoopState::LISTENING:         return "listening";
        case VoiceLoopState::RECORDING:         return "recording";
        case VoiceLoopState::TRANSCRIBING:      return "transcribing";
        case VoiceLoopState::INFERRING:         return "inferring";
        case VoiceLoopState::SPEAKING:          return "speaking";
        case VoiceLoopState::INTERRUPTED:       return "interrupted";
        case VoiceLoopState::STOPPING:          return "stopping";
        default:                                return "unknown";
        }
    }

    // =========================================================================
    // AudioChunk
    // Raw PCM audio samples (16-bit signed, mono, 16kHz)
    // =========================================================================
    struct AudioChunk {
        std::vector<int16_t> samples;
        int                  sample_rate = 16000;
        int                  channels    = 1;

        float duration_ms() const {
            if (sample_rate == 0 || samples.empty()) return 0.0f;
            return static_cast<float>(samples.size()) /
                   static_cast<float>(sample_rate) * 1000.0f;
        }

        bool empty() const { return samples.empty(); }
    };

    // =========================================================================
    // TranscriptResult
    // Output of STTEngine::transcribe()
    // =========================================================================
    struct TranscriptResult {
        bool        success        = false;
        std::string text;               // trimmed transcript
        float       confidence     = 0.0f;
        int         duration_ms    = 0;
        std::string error_message;

        bool empty() const {
            return text.empty() ||
                   text == "[BLANK_AUDIO]" ||
                   text == "(blank)";
        }
    };

    // =========================================================================
    // VADResult
    // Output of one VAD frame evaluation
    // =========================================================================
    struct VADResult {
        bool  speech_detected = false;
        float rms             = 0.0f;   // root mean square energy 0..1
        float threshold       = 0.0f;
    };

    // =========================================================================
    // WakeWordResult
    // =========================================================================
    struct WakeWordResult {
        bool        detected      = false;
        std::string phrase;
        float       score         = 0.0f;
    };

    // =========================================================================
    // TTSRequest
    // =========================================================================
    struct TTSRequest {
        std::string text;
        int         speaker_id  = 0;
        float       length_scale = 1.0f;  // speed: < 1 = faster, > 1 = slower
        float       noise_scale  = 0.667f;
        float       noise_w      = 0.8f;
    };

    // =========================================================================
    // TTSResult
    // Raw PCM audio output from TTS synthesis
    // =========================================================================
    struct TTSResult {
        bool                 success = false;
        std::vector<int16_t> samples;
        int                  sample_rate  = 22050;
        int                  duration_ms  = 0;
        std::string          error_message;
    };

    // =========================================================================
    // VoiceStatus
    // Current state of the voice subsystem (for API/HTTP)
    // =========================================================================
    struct VoiceStatus {
        bool          active          = false;
        std::string   input_mode;
        std::string   current_state;
        bool          stt_ready       = false;
        bool          tts_ready       = false;
        bool          wake_word_ready = false;
        std::string   session_id;
        int           transcriptions  = 0;
        int           utterances      = 0;
    };

    // =========================================================================
    // Config structs (populated from config.json by ConfigLoader)
    // =========================================================================

    struct VoiceSTTConfig {
        std::string model_path;
        std::string language        = "en";
        int         gpu_layers      = 8;
        int         threads         = 4;
        int         beam_size       = 5;
        std::string initial_prompt;
    };

    struct VoiceTTSConfig {
        std::string model_path;
        std::string config_path;
        int         speaker_id      = 0;
        float       length_scale    = 1.0f;
        float       noise_scale     = 0.667f;
        float       noise_w         = 0.8f;
        int         sample_rate     = 22050;
    };

    struct VoiceVADConfig {
        float energy_threshold      = 0.02f;
        int   pre_speech_ms         = 300;
        int   post_speech_ms        = 800;
        int   min_speech_ms         = 200;
        int   max_speech_ms         = 30000;
    };

    struct VoicePTTConfig {
        std::string key             = "space";
    };

    struct VoiceWakeWordConfig {
        std::string phrase          = "hey cardinal";
        std::string acoustic_model;
        std::string dictionary;
        float       sensitivity     = 1e-20f;
    };

    struct VoiceAudioConfig {
        int input_device            = -1;   // -1 = default
        int output_device           = -1;
        int sample_rate             = 16000;
        int channels                = 1;
        int frames_per_buffer       = 512;
    };

    struct VoiceConfig {
        bool             enabled         = false;
        VoiceInputMode   input_mode      = VoiceInputMode::VAD;
        TTSStreamingMode tts_streaming   = TTSStreamingMode::SENTENCE;
        std::string      session_id      = "voice_session";

        VoiceSTTConfig      stt;
        VoiceTTSConfig      tts;
        VoiceVADConfig      vad;
        VoicePTTConfig      ptt;
        VoiceWakeWordConfig wake_word;
        VoiceAudioConfig    audio;
    };

    // =========================================================================
    // Callbacks
    // =========================================================================
    using TranscriptCallback   = std::function<void(const TranscriptResult&)>;
    using WakeWordCallback     = std::function<void(const WakeWordResult&)>;
    using VoiceStateCallback   = std::function<void(VoiceLoopState)>;
    using BargeInCallback      = std::function<void()>;

} // namespace cardinal
