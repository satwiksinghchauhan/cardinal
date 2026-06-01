#pragma once
// =============================================================================
// Cardinal - Audio Device (v1.6.0)
// File: src/voice/audio_device.h
//
// PortAudio wrapper for microphone capture and speaker playback.
// Two independent streams: input (capture) and output (playback).
// Both are 16-bit signed PCM, mono, configurable sample rate.
// =============================================================================

#include "voice/voice_types.h"

#include <portaudio.h>      // needed for PaStreamCallbackTimeInfo, PaStreamCallbackFlags

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <memory>

namespace cardinal {

    class AudioDevice {
    public:
        explicit AudioDevice(const VoiceAudioConfig& config);
        ~AudioDevice();

        AudioDevice(const AudioDevice&)            = delete;
        AudioDevice& operator=(const AudioDevice&) = delete;

        // ------------------------------------------------------------------
        // Initialisation
        // ------------------------------------------------------------------
        bool init();
        void shutdown();
        bool is_ready() const { return ready_; }

        // ------------------------------------------------------------------
        // Capture
        // ------------------------------------------------------------------
        bool start_capture(std::function<void(const int16_t*, int)> callback);
        void stop_capture();
        bool is_capturing() const { return capturing_.load(); }

        // ------------------------------------------------------------------
        // Playback
        // ------------------------------------------------------------------
        void play(const std::vector<int16_t>& samples,
                  int                          sample_rate,
                  std::function<void()>        on_done = nullptr);
        void stop_playback();
        bool is_playing() const { return playing_.load(); }
        void wait_until_done();

        // ------------------------------------------------------------------
        // Device enumeration
        // ------------------------------------------------------------------
        struct DeviceInfo {
            int         index;
            std::string name;
            int         max_input_channels;
            int         max_output_channels;
            float       default_sample_rate;
        };

        std::vector<DeviceInfo> list_devices() const;
        int default_input_device()  const { return default_input_;  }
        int default_output_device() const { return default_output_; }

    private:
        // PortAudio callbacks — must match PaStreamCallback typedef exactly
        static int capture_callback(const void*                     input,
                                    void*                           output,
                                    unsigned long                   frames_per_buffer,
                                    const PaStreamCallbackTimeInfo* time_info,
                                    PaStreamCallbackFlags           status_flags,
                                    void*                           user_data);

        static int playback_callback(const void*                     input,
                                     void*                           output,
                                     unsigned long                   frames_per_buffer,
                                     const PaStreamCallbackTimeInfo* time_info,
                                     PaStreamCallbackFlags           status_flags,
                                     void*                           user_data);

        bool open_capture_stream();
        bool open_playback_stream();
        void close_streams();

        const VoiceAudioConfig& config_;

        PaStream* capture_stream_  = nullptr;
        PaStream* playback_stream_ = nullptr;

        bool ready_          = false;
        int  default_input_  = -1;
        int  default_output_ = -1;

        // Capture
        std::atomic<bool>                        capturing_{ false };
        std::function<void(const int16_t*, int)> capture_cb_;

        // Playback queue
        struct PlaybackChunk {
            std::vector<int16_t>  samples;
            int                   sample_rate;
            std::function<void()> on_done;
            size_t                position = 0;
        };

        std::atomic<bool>            playing_{ false };
        mutable std::mutex           playback_mutex_;
        std::condition_variable      playback_cv_;
        std::deque<PlaybackChunk>    playback_queue_;

        int output_sample_rate_ = 44100;
    };

} // namespace cardinal
