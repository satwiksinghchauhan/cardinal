#pragma once
// =============================================================================
// Cardinal - Wake Word Detector (v1.6.0)
// File: src/voice/wake_word_detector.h
//
// PocketSphinx wrapper — continuous keyword spotting, offline, no API key.
// Runs its own thread; calls WakeWordCallback on detection.
// AudioDevice capture callback feeds frames here via push_frame().
// =============================================================================

#include "voice/voice_types.h"

#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <deque>
#include <vector>

typedef struct ps_decoder_s ps_decoder_t;

namespace cardinal {

    class WakeWordDetector {
    public:
        explicit WakeWordDetector(const VoiceWakeWordConfig& config, int sample_rate);
        ~WakeWordDetector();

        WakeWordDetector(const WakeWordDetector&)            = delete;
        WakeWordDetector& operator=(const WakeWordDetector&) = delete;

        bool init();
        void shutdown();
        bool is_ready() const { return ready_.load(); }

        bool start(WakeWordCallback on_detected);
        void stop();
        bool is_listening() const { return listening_.load(); }

        // Thread-safe; called from audio capture callback.
        void push_frame(const int16_t* samples, int num_samples);

    private:
        void recognition_loop();

        const VoiceWakeWordConfig config_;
        int                       sample_rate_;
        ps_decoder_t*             ps_{ nullptr };
        std::atomic<bool>         ready_{ false };
        std::atomic<bool>         listening_{ false };

        mutable std::mutex        queue_mutex_;
        std::condition_variable   queue_cv_;
        std::deque<int16_t>       frame_queue_;
        static constexpr int      kMaxQueueSamples = 16000 * 2;

        std::thread          recog_thread_;
        WakeWordCallback     on_detected_;
    };

} // namespace cardinal
