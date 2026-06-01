// =============================================================================
// Cardinal - Wake Word Detector Implementation (v1.6.0)
// File: src/voice/wake_word_detector.cpp
// =============================================================================

#include "voice/wake_word_detector.h"
#include "utils/logger.h"

#include <pocketsphinx.h>

#include <algorithm>
#include <cstring>

namespace cardinal {

WakeWordDetector::WakeWordDetector(const VoiceWakeWordConfig& config, int sample_rate)
    : config_(config)
    , sample_rate_(sample_rate)
{}

WakeWordDetector::~WakeWordDetector() {
    shutdown();
}

bool WakeWordDetector::init() {
    if (ready_.load()) return true;

    ps_config_t* ps_cfg = ps_config_init(nullptr);
    if (!ps_cfg) {
        LOG_ERROR("WakeWordDetector: ps_config_init() failed");
        return false;
    }

    if (!config_.acoustic_model.empty())
        ps_config_set_str(ps_cfg, "hmm", config_.acoustic_model.c_str());
    if (!config_.dictionary.empty())
        ps_config_set_str(ps_cfg, "dict", config_.dictionary.c_str());

    ps_config_set_str(ps_cfg, "keyphrase", config_.phrase.c_str());
    ps_config_set_float(ps_cfg, "kws_threshold",
                        static_cast<double>(config_.sensitivity));
    ps_config_set_int(ps_cfg, "samprate", sample_rate_);

    ps_ = ps_init(ps_cfg);
    ps_config_free(ps_cfg);

    if (!ps_) {
        LOG_ERROR("WakeWordDetector: ps_init() failed");
        return false;
    }

    ready_.store(true);
    LOG_INFO("WakeWordDetector ready — phrase: \"" + config_.phrase + "\"");
    return true;
}

void WakeWordDetector::shutdown() {
    stop();
    if (ps_) {
        ps_free(ps_);
        ps_ = nullptr;
        ready_.store(false);
        LOG_INFO("WakeWordDetector shut down");
    }
}

bool WakeWordDetector::start(WakeWordCallback on_detected) {
    if (!ready_.load()) return false;
    if (listening_.exchange(true)) return true;

    on_detected_  = std::move(on_detected);
    recog_thread_ = std::thread(&WakeWordDetector::recognition_loop, this);
    LOG_INFO("WakeWordDetector: listening for \"" + config_.phrase + "\"");
    return true;
}

void WakeWordDetector::stop() {
    if (!listening_.exchange(false)) return;
    queue_cv_.notify_all();
    if (recog_thread_.joinable()) recog_thread_.join();
    LOG_INFO("WakeWordDetector: stopped");
}

void WakeWordDetector::push_frame(const int16_t* samples, int num_samples) {
    if (!listening_.load()) return;
    std::lock_guard<std::mutex> lock(queue_mutex_);
    int overflow = static_cast<int>(frame_queue_.size()) + num_samples - kMaxQueueSamples;
    if (overflow > 0)
        frame_queue_.erase(frame_queue_.begin(), frame_queue_.begin() + overflow);
    frame_queue_.insert(frame_queue_.end(), samples, samples + num_samples);
    queue_cv_.notify_one();
}

void WakeWordDetector::recognition_loop() {
    if (ps_start_utt(ps_) < 0) {
        LOG_ERROR("WakeWordDetector: ps_start_utt() failed");
        listening_.store(false);
        return;
    }

    constexpr int kChunk = 512;
    std::vector<int16_t> buf(kChunk);

    while (listening_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return !frame_queue_.empty() || !listening_.load();
        });
        if (!listening_.load()) break;

        int to_take = std::min(kChunk, static_cast<int>(frame_queue_.size()));
        for (int i = 0; i < to_take; ++i) {
            buf[i] = frame_queue_.front();
            frame_queue_.pop_front();
        }
        lock.unlock();

        int rc = ps_process_raw(ps_, buf.data(), to_take, FALSE, FALSE);
        if (rc < 0) {
            LOG_WARN("WakeWordDetector: ps_process_raw() error");
            continue;
        }

        int32_t score = 0;
        const char* hyp = ps_get_hyp(ps_, &score);
        if (hyp && std::string(hyp).find(config_.phrase) != std::string::npos) {
            LOG_INFO("WakeWordDetector: detected \"" + config_.phrase + "\"");

            WakeWordResult result;
            result.detected = true;
            result.phrase   = config_.phrase;
            result.score    = static_cast<float>(score);

            if (on_detected_) on_detected_(result);

            ps_end_utt(ps_);
            ps_start_utt(ps_);
        }
    }

    ps_end_utt(ps_);
}

} // namespace cardinal
