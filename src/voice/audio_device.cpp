// =============================================================================
// Cardinal - Audio Device Implementation (v1.6.0)
// File: src/voice/audio_device.cpp
// =============================================================================

#include "voice/audio_device.h"
#include "utils/logger.h"

#include <portaudio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace cardinal {

// =============================================================================
// Constructor / Destructor
// =============================================================================

AudioDevice::AudioDevice(const VoiceAudioConfig& config)
    : config_(config)
{}

AudioDevice::~AudioDevice() {
    shutdown();
}

// =============================================================================
// init / shutdown
// =============================================================================

bool AudioDevice::init() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        LOG_ERROR("PortAudio init failed: " + std::string(Pa_GetErrorText(err)));
        return false;
    }

    default_input_  = (config_.input_device  >= 0) ? config_.input_device
                                                    : Pa_GetDefaultInputDevice();
    default_output_ = (config_.output_device >= 0) ? config_.output_device
                                                    : Pa_GetDefaultOutputDevice();

    if (default_input_ == paNoDevice)
        LOG_WARN("AudioDevice: no input device available");
    if (default_output_ == paNoDevice)
        LOG_WARN("AudioDevice: no output device available");

    const PaDeviceInfo* out_info = Pa_GetDeviceInfo(default_output_);
    if (out_info)
        output_sample_rate_ = static_cast<int>(out_info->defaultSampleRate);

    ready_ = true;
    LOG_INFO("AudioDevice init: input=" + std::to_string(default_input_) +
             " output=" + std::to_string(default_output_) +
             " out_rate=" + std::to_string(output_sample_rate_));
    return true;
}

void AudioDevice::shutdown() {
    stop_capture();
    stop_playback();
    close_streams();
    if (ready_) {
        Pa_Terminate();
        ready_ = false;
        LOG_INFO("AudioDevice shut down");
    }
}

// =============================================================================
// Capture
// =============================================================================

bool AudioDevice::open_capture_stream() {
    if (capture_stream_) return true;
    if (default_input_ == paNoDevice) return false;

    PaStreamParameters p{};
    p.device                    = default_input_;
    p.channelCount              = config_.channels;
    p.sampleFormat              = paInt16;
    p.suggestedLatency          = Pa_GetDeviceInfo(default_input_)->defaultLowInputLatency;
    p.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&capture_stream_, &p, nullptr,
                                config_.sample_rate, config_.frames_per_buffer,
                                paClipOff, &AudioDevice::capture_callback, this);
    if (err != paNoError) {
        LOG_ERROR("AudioDevice: open capture stream failed: " +
                  std::string(Pa_GetErrorText(err)));
        capture_stream_ = nullptr;
        return false;
    }
    return true;
}

bool AudioDevice::start_capture(std::function<void(const int16_t*, int)> callback) {
    if (!ready_) return false;
    if (capturing_.load()) return true;
    if (!open_capture_stream()) return false;

    capture_cb_ = std::move(callback);
    capturing_.store(true);

    PaError err = Pa_StartStream(capture_stream_);
    if (err != paNoError) {
        LOG_ERROR("AudioDevice: start capture failed: " +
                  std::string(Pa_GetErrorText(err)));
        capturing_.store(false);
        return false;
    }
    LOG_INFO("AudioDevice: capture started (rate=" +
             std::to_string(config_.sample_rate) + ")");
    return true;
}

void AudioDevice::stop_capture() {
    if (!capturing_.exchange(false)) return;
    if (capture_stream_) Pa_StopStream(capture_stream_);
    capture_cb_ = nullptr;
    LOG_INFO("AudioDevice: capture stopped");
}

// PortAudio capture callback — uses real PortAudio types
int AudioDevice::capture_callback(const void*                     input,
                                   void*                           /*output*/,
                                   unsigned long                   frames,
                                   const PaStreamCallbackTimeInfo* /*time_info*/,
                                   PaStreamCallbackFlags           /*status*/,
                                   void*                           user_data)
{
    auto* self = static_cast<AudioDevice*>(user_data);
    if (!self->capturing_.load()) return paContinue;
    if (self->capture_cb_ && input)
        self->capture_cb_(static_cast<const int16_t*>(input), static_cast<int>(frames));
    return paContinue;
}

// =============================================================================
// Playback
// =============================================================================

bool AudioDevice::open_playback_stream() {
    if (playback_stream_) return true;
    if (default_output_ == paNoDevice) return false;

    PaStreamParameters p{};
    p.device                    = default_output_;
    p.channelCount              = 1;
    p.sampleFormat              = paInt16;
    p.suggestedLatency          = Pa_GetDeviceInfo(default_output_)->defaultLowOutputLatency;
    p.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&playback_stream_, nullptr, &p,
                                output_sample_rate_, config_.frames_per_buffer,
                                paClipOff, &AudioDevice::playback_callback, this);
    if (err != paNoError) {
        LOG_ERROR("AudioDevice: open playback stream failed: " +
                  std::string(Pa_GetErrorText(err)));
        playback_stream_ = nullptr;
        return false;
    }
    return true;
}

void AudioDevice::play(const std::vector<int16_t>& samples,
                        int                          sample_rate,
                        std::function<void()>        on_done)
{
    if (!ready_) return;
    if (samples.empty()) { if (on_done) on_done(); return; }
    if (!open_playback_stream()) return;

    PlaybackChunk chunk;
    chunk.sample_rate = sample_rate;
    chunk.on_done     = std::move(on_done);

    // Linear-interpolation resample when rates differ
    if (sample_rate != output_sample_rate_) {
        double ratio  = static_cast<double>(output_sample_rate_) / sample_rate;
        size_t out_len = static_cast<size_t>(samples.size() * ratio);
        chunk.samples.resize(out_len);
        for (size_t i = 0; i < out_len; ++i) {
            double src_pos = i / ratio;
            size_t src_i   = static_cast<size_t>(src_pos);
            double frac    = src_pos - src_i;
            int16_t a = samples[src_i];
            int16_t b = (src_i + 1 < samples.size()) ? samples[src_i + 1] : a;
            chunk.samples[i] = static_cast<int16_t>(a + frac * (b - a));
        }
    } else {
        chunk.samples = samples;
    }

    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        playback_queue_.push_back(std::move(chunk));
    }

    if (!playing_.exchange(true)) {
        PaError err = Pa_StartStream(playback_stream_);
        if (err != paNoError) {
            LOG_ERROR("AudioDevice: start playback failed: " +
                      std::string(Pa_GetErrorText(err)));
            playing_.store(false);
        }
    }
    playback_cv_.notify_all();
}

void AudioDevice::stop_playback() {
    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        for (auto& c : playback_queue_)
            if (c.on_done) c.on_done();
        playback_queue_.clear();
    }
    if (playing_.exchange(false)) {
        if (playback_stream_) Pa_StopStream(playback_stream_);
    }
    playback_cv_.notify_all();
}

void AudioDevice::wait_until_done() {
    std::unique_lock<std::mutex> lock(playback_mutex_);
    playback_cv_.wait(lock, [this] {
        return playback_queue_.empty() || !playing_.load();
    });
}

// PortAudio playback callback — uses real PortAudio types
int AudioDevice::playback_callback(const void*                     /*input*/,
                                    void*                           output,
                                    unsigned long                   frames,
                                    const PaStreamCallbackTimeInfo* /*time_info*/,
                                    PaStreamCallbackFlags           /*status*/,
                                    void*                           user_data)
{
    auto*         self    = static_cast<AudioDevice*>(user_data);
    auto*         out_buf = static_cast<int16_t*>(output);
    unsigned long left    = frames;

    std::lock_guard<std::mutex> lock(self->playback_mutex_);

    while (left > 0 && !self->playback_queue_.empty()) {
        PlaybackChunk& chunk = self->playback_queue_.front();
        size_t avail   = chunk.samples.size() - chunk.position;
        size_t to_copy = std::min(static_cast<size_t>(left), avail);

        std::memcpy(out_buf, chunk.samples.data() + chunk.position,
                    to_copy * sizeof(int16_t));
        out_buf        += to_copy;
        left           -= static_cast<unsigned long>(to_copy);
        chunk.position += to_copy;

        if (chunk.position >= chunk.samples.size()) {
            if (chunk.on_done) chunk.on_done();
            self->playback_queue_.pop_front();
        }
    }

    if (left > 0)
        std::memset(out_buf, 0, left * sizeof(int16_t));

    if (self->playback_queue_.empty()) {
        self->playing_.store(false);
        self->playback_cv_.notify_all();
        return paComplete;
    }
    return paContinue;
}

// =============================================================================
// Stream close
// =============================================================================

void AudioDevice::close_streams() {
    if (capture_stream_)  { Pa_CloseStream(capture_stream_);  capture_stream_  = nullptr; }
    if (playback_stream_) { Pa_CloseStream(playback_stream_); playback_stream_ = nullptr; }
}

// =============================================================================
// Device enumeration
// =============================================================================

std::vector<AudioDevice::DeviceInfo> AudioDevice::list_devices() const {
    std::vector<DeviceInfo> out;
    if (!ready_) return out;
    int n = Pa_GetDeviceCount();
    for (int i = 0; i < n; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info) continue;
        DeviceInfo d;
        d.index               = i;
        d.name                = info->name ? info->name : "";
        d.max_input_channels  = info->maxInputChannels;
        d.max_output_channels = info->maxOutputChannels;
        d.default_sample_rate = static_cast<float>(info->defaultSampleRate);
        out.push_back(d);
    }
    return out;
}

} // namespace cardinal
