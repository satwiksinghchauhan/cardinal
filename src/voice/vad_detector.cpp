// =============================================================================
// Cardinal - VAD Detector Implementation (v1.6.0)
// File: src/voice/vad_detector.cpp
// =============================================================================

#include "voice/vad_detector.h"
#include "utils/logger.h"

#include <cmath>
#include <algorithm>

namespace cardinal {

VADDetector::VADDetector(const VoiceVADConfig& config, int sample_rate)
    : config_(config)
    , sample_rate_(sample_rate)
{
    pre_roll_samples_    = (config_.pre_speech_ms  * sample_rate_) / 1000;
    post_silence_needed_ = (config_.post_speech_ms * sample_rate_) / 1000;
    min_speech_samples_  = (config_.min_speech_ms  * sample_rate_) / 1000;
    max_speech_samples_  = (config_.max_speech_ms  * sample_rate_) / 1000;
}

float VADDetector::compute_rms(const int16_t* samples, int num_samples) {
    if (num_samples <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        double s = samples[i] / 32768.0;
        sum += s * s;
    }
    return static_cast<float>(std::sqrt(sum / num_samples));
}

void VADDetector::push_frame(const int16_t* samples, int num_samples) {
    float rms      = compute_rms(samples, num_samples);
    bool  is_voice = rms >= config_.energy_threshold;

    std::unique_lock<std::mutex> lock(mutex_);

    switch (state_) {

    case State::IDLE:
        pre_roll_buffer_.insert(pre_roll_buffer_.end(),
                                samples, samples + num_samples);
        while (static_cast<int>(pre_roll_buffer_.size()) > pre_roll_samples_)
            pre_roll_buffer_.pop_front();

        if (is_voice) {
            state_ = State::SPEECH;
            speech_buffer_.assign(pre_roll_buffer_.begin(), pre_roll_buffer_.end());
            speech_buffer_.insert(speech_buffer_.end(), samples, samples + num_samples);
            pre_roll_buffer_.clear();
            post_silence_samples_ = 0;

            auto cb = onset_cb_;
            lock.unlock();
            if (cb) cb();
            lock.lock();
        }
        break;

    case State::SPEECH:
        speech_buffer_.insert(speech_buffer_.end(), samples, samples + num_samples);

        // Hard cap — emit immediately regardless of silence
        if (static_cast<int>(speech_buffer_.size()) >= max_speech_samples_) {
            auto seg_cb = segment_cb_;
            AudioChunk chunk;
            chunk.samples.swap(speech_buffer_);
            chunk.sample_rate = sample_rate_;
            state_ = State::IDLE;
            auto off_cb = offset_cb_;
            lock.unlock();
            if (off_cb) off_cb();
            if (seg_cb) seg_cb(std::move(chunk));
            lock.lock();
            break;
        }

        if (!is_voice) {
            post_silence_samples_ += num_samples;
            if (post_silence_samples_ >= post_silence_needed_) {
                if (static_cast<int>(speech_buffer_.size()) < min_speech_samples_) {
                    // Too short — discard
                    speech_buffer_.clear();
                    state_ = State::IDLE;
                } else {
                    auto seg_cb = segment_cb_;
                    AudioChunk chunk;
                    chunk.samples.swap(speech_buffer_);
                    chunk.sample_rate = sample_rate_;
                    state_ = State::IDLE;
                    auto off_cb = offset_cb_;
                    lock.unlock();
                    if (off_cb) off_cb();
                    if (seg_cb) seg_cb(std::move(chunk));
                    lock.lock();
                }
            }
        } else {
            post_silence_samples_ = 0;
        }
        break;
    }
}

void VADDetector::set_segment_callback(SpeechSegmentCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    segment_cb_ = std::move(cb);
}

void VADDetector::set_onset_callback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    onset_cb_ = std::move(cb);
}

void VADDetector::set_offset_callback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    offset_cb_ = std::move(cb);
}

bool VADDetector::is_speech_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == State::SPEECH;
}

void VADDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = State::IDLE;
    pre_roll_buffer_.clear();
    speech_buffer_.clear();
    post_silence_samples_ = 0;
}

} // namespace cardinal
