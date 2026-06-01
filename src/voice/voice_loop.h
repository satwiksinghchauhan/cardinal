#pragma once
// =============================================================================
// Cardinal - VoiceLoop (v1.6.0)
// File: src/voice/voice_loop.h
//
// Top-level voice subsystem state machine.
// Owned by CardinalAPI; runs its own thread.
// Owns: AudioDevice, VADDetector, STTEngine, TTSEngine, WakeWordDetector.
//
// State machine:
//   IDLE
//    ├─ (push_to_talk) → RECORDING → TRANSCRIBING → INFERRING → SPEAKING → IDLE
//    ├─ (vad)          → LISTENING → RECORDING    → TRANSCRIBING → INFERRING → SPEAKING → IDLE
//    └─ (wake_word)    → PASSIVE_LISTENING → LISTENING → RECORDING → TRANSCRIBING → INFERRING → SPEAKING → IDLE
//
// Barge-in: VAD onset during SPEAKING → stop_playback() immediately → RECORDING
// =============================================================================

#include "voice/voice_types.h"
#include "voice/audio_device.h"
#include "voice/vad_detector.h"
#include "voice/stt_engine.h"
#include "voice/tts_engine.h"
#include "voice/wake_word_detector.h"

#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <queue>

namespace cardinal {

    // Thin wrapper around CardinalAPI::chat_stream provided to VoiceLoop.
    using VoiceChatStreamFn = std::function<
        bool(const std::string& session_id,
             const std::string& message,
             std::function<bool(const std::string& token, bool is_final)> cb)
    >;

    class VoiceLoop {
    public:
        VoiceLoop(const VoiceConfig& config, VoiceChatStreamFn chat_fn);
        ~VoiceLoop();

        VoiceLoop(const VoiceLoop&)            = delete;
        VoiceLoop& operator=(const VoiceLoop&) = delete;

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------
        bool start();
        void stop();
        bool is_running() const { return running_.load(); }

        // ------------------------------------------------------------------
        // Runtime control
        // ------------------------------------------------------------------
        void           set_input_mode(VoiceInputMode mode);
        VoiceInputMode input_mode() const;

        TTSResult        speak(const std::string& text);
        TranscriptResult transcribe(const AudioChunk& audio);
        void             stop_speaking();

        // ------------------------------------------------------------------
        // State query
        // ------------------------------------------------------------------
        VoiceLoopState state() const;
        VoiceStatus    get_status() const;

        // ------------------------------------------------------------------
        // Callbacks (set before start())
        // ------------------------------------------------------------------
        void set_state_callback(VoiceStateCallback cb);
        void set_transcript_callback(TranscriptCallback cb);

    private:
        void loop_thread();
        void ptt_thread_fn();
        void ptt_key_down();
        void ptt_key_up();

        void handle_speech_segment(AudioChunk segment);
        void run_inference_and_speak(const std::string& transcript);

        void set_state(VoiceLoopState s);
        bool is_barge_in_candidate() const;

        VoiceConfig       config_;
        VoiceChatStreamFn chat_fn_;

        std::unique_ptr<AudioDevice>      audio_;
        std::unique_ptr<VADDetector>      vad_;
        std::unique_ptr<STTEngine>        stt_;
        std::unique_ptr<TTSEngine>        tts_;
        std::unique_ptr<WakeWordDetector> wake_word_;

        mutable std::mutex state_mutex_;
        VoiceLoopState     state_{ VoiceLoopState::IDLE };
        std::atomic<bool>  running_{ false };
        std::atomic<bool>  stop_requested_{ false };

        std::mutex              work_mutex_;
        std::condition_variable work_cv_;
        std::queue<AudioChunk>  pending_segments_;

        std::atomic<bool>    ptt_recording_{ false };
        std::vector<int16_t> ptt_buffer_;
        std::mutex           ptt_buf_mutex_;

        std::thread loop_thread_;
        std::thread ptt_thread_;

        VoiceStateCallback  state_cb_;
        TranscriptCallback  transcript_cb_;

        std::atomic<int> transcription_count_{ 0 };
        std::atomic<int> utterance_count_{ 0 };
    };

} // namespace cardinal
