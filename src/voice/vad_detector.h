#pragma once
// =============================================================================
// Cardinal - VAD Detector (v1.6.0)
// File: src/voice/vad_detector.h
//
// Energy-based Voice Activity Detector.
// Accumulates frames into pre-speech and post-speech windows using
// configurable RMS thresholds. No external library — pure C++.
// =============================================================================

#include "voice/voice_types.h"

#include <vector>
#include <deque>
#include <mutex>
#include <functional>

namespace cardinal {

    using SpeechSegmentCallback = std::function<void(AudioChunk)>;

    class VADDetector {
    public:
        explicit VADDetector(const VoiceVADConfig& config, int sample_rate);

        // Feed one frame of PCM samples.
        // Thread-safe; may be called from the PortAudio callback thread.
        void push_frame(const int16_t* samples, int num_samples);

        void set_segment_callback(SpeechSegmentCallback cb);
        void set_onset_callback(std::function<void()> cb);
        void set_offset_callback(std::function<void()> cb);

        bool is_speech_active() const;

        static float compute_rms(const int16_t* samples, int num_samples);

        void reset();

    private:
        const VoiceVADConfig config_;
        int                  sample_rate_;

        enum class State { IDLE, SPEECH };
        State state_{ State::IDLE };

        int                       pre_roll_samples_;
        std::deque<int16_t>       pre_roll_buffer_;
        std::vector<int16_t>      speech_buffer_;
        int                       post_silence_samples_{ 0 };
        int                       post_silence_needed_{ 0 };
        int                       min_speech_samples_{ 0 };
        int                       max_speech_samples_{ 0 };

        mutable std::mutex       mutex_;
        SpeechSegmentCallback    segment_cb_;
        std::function<void()>    onset_cb_;
        std::function<void()>    offset_cb_;
    };

} // namespace cardinal
