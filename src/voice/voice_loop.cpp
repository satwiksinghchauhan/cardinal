// =============================================================================
// Cardinal - VoiceLoop Implementation (v1.6.0)
// File: src/voice/voice_loop.cpp
// =============================================================================

#include "voice/voice_loop.h"
#include "utils/logger.h"

#include <termios.h>
#include <unistd.h>

namespace cardinal {

// =============================================================================
// Constructor / Destructor
// =============================================================================

VoiceLoop::VoiceLoop(const VoiceConfig& config, VoiceChatStreamFn chat_fn)
    : config_(config)
    , chat_fn_(std::move(chat_fn))
{}

VoiceLoop::~VoiceLoop() {
    stop();
}

// =============================================================================
// start
// =============================================================================

bool VoiceLoop::start() {
    if (running_.load()) return true;

    LOG_INFO("VoiceLoop: initialising...");

    audio_ = std::make_unique<AudioDevice>(config_.audio);
    if (!audio_->init()) {
        LOG_ERROR("VoiceLoop: AudioDevice init failed");
        return false;
    }

    vad_ = std::make_unique<VADDetector>(config_.vad, config_.audio.sample_rate);

    vad_->set_segment_callback([this](AudioChunk seg) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        pending_segments_.push(std::move(seg));
        work_cv_.notify_one();
    });

    vad_->set_onset_callback([this] {
        if (is_barge_in_candidate()) {
            LOG_INFO("VoiceLoop: barge-in");
            audio_->stop_playback();
            set_state(VoiceLoopState::INTERRUPTED);
        }
    });

    stt_ = std::make_unique<STTEngine>(config_.stt);
    if (!stt_->init()) {
        LOG_ERROR("VoiceLoop: STTEngine init failed");
        return false;
    }

    tts_ = std::make_unique<TTSEngine>(config_.tts);
    if (!tts_->init()) {
        LOG_ERROR("VoiceLoop: TTSEngine init failed");
        return false;
    }

    if (config_.input_mode == VoiceInputMode::WAKE_WORD) {
        wake_word_ = std::make_unique<WakeWordDetector>(
            config_.wake_word, config_.audio.sample_rate);
        if (!wake_word_->init()) {
            LOG_ERROR("VoiceLoop: WakeWordDetector init failed");
            return false;
        }
    }

    stop_requested_.store(false);
    running_.store(true);

    loop_thread_ = std::thread(&VoiceLoop::loop_thread, this);

    if (config_.input_mode == VoiceInputMode::PUSH_TO_TALK)
        ptt_thread_ = std::thread(&VoiceLoop::ptt_thread_fn, this);

    LOG_INFO("VoiceLoop started — mode=" +
             voice_input_mode_to_string(config_.input_mode));
    return true;
}

// =============================================================================
// stop
// =============================================================================

void VoiceLoop::stop() {
    if (!running_.exchange(false)) return;

    stop_requested_.store(true);
    work_cv_.notify_all();

    audio_->stop_playback();
    audio_->stop_capture();

    if (wake_word_) wake_word_->stop();

    if (loop_thread_.joinable()) loop_thread_.join();
    if (ptt_thread_.joinable())  ptt_thread_.join();

    stt_->shutdown();
    tts_->shutdown();
    audio_->shutdown();

    set_state(VoiceLoopState::STOPPING);
    LOG_INFO("VoiceLoop stopped");
}

// =============================================================================
// loop_thread — main state machine
// =============================================================================

void VoiceLoop::loop_thread() {
    // Start audio capture; route frames to VAD and wake word
    audio_->start_capture([this](const int16_t* frames, int n) {
        VoiceLoopState s = state();

        if (wake_word_ && wake_word_->is_listening())
            wake_word_->push_frame(frames, n);

        if (s == VoiceLoopState::LISTENING  ||
            s == VoiceLoopState::RECORDING  ||
            s == VoiceLoopState::SPEAKING   ||
            s == VoiceLoopState::INTERRUPTED)
        {
            vad_->push_frame(frames, n);
        }

        if (ptt_recording_.load()) {
            std::lock_guard<std::mutex> lock(ptt_buf_mutex_);
            ptt_buffer_.insert(ptt_buffer_.end(), frames, frames + n);
        }
    });

    // Enter initial state based on input mode
    switch (config_.input_mode) {
    case VoiceInputMode::PUSH_TO_TALK:
        set_state(VoiceLoopState::IDLE);
        break;
    case VoiceInputMode::VAD:
        set_state(VoiceLoopState::LISTENING);
        break;
    case VoiceInputMode::WAKE_WORD:
        set_state(VoiceLoopState::PASSIVE_LISTENING);
        wake_word_->start([this](const WakeWordResult&) {
            LOG_INFO("VoiceLoop: wake word detected — activating VAD");
            vad_->reset();
            set_state(VoiceLoopState::LISTENING);
        });
        break;
    }

    // Work loop: process completed speech segments
    while (!stop_requested_.load()) {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_cv_.wait(lock, [this] {
            return !pending_segments_.empty() || stop_requested_.load();
        });
        if (stop_requested_.load()) break;

        AudioChunk segment = std::move(pending_segments_.front());
        pending_segments_.pop();
        lock.unlock();

        handle_speech_segment(std::move(segment));

        if (!stop_requested_.load()) {
            if (config_.input_mode == VoiceInputMode::WAKE_WORD)
                set_state(VoiceLoopState::PASSIVE_LISTENING);
            else
                set_state(VoiceLoopState::LISTENING);
        }
    }
}

// =============================================================================
// handle_speech_segment
// =============================================================================

void VoiceLoop::handle_speech_segment(AudioChunk segment) {
    set_state(VoiceLoopState::TRANSCRIBING);

    TranscriptResult transcript = stt_->transcribe(segment);
    ++transcription_count_;

    if (transcript_cb_) transcript_cb_(transcript);

    if (!transcript.success || transcript.empty()) {
        LOG_DEBUG("VoiceLoop: blank transcription — skipping");
        return;
    }

    LOG_INFO("VoiceLoop: transcript: \"" + transcript.text + "\"");
    run_inference_and_speak(transcript.text);
}

// =============================================================================
// run_inference_and_speak
// =============================================================================

void VoiceLoop::run_inference_and_speak(const std::string& transcript) {
    set_state(VoiceLoopState::INFERRING);
    ++utterance_count_;

    if (config_.tts_streaming == TTSStreamingMode::SENTENCE) {
        // Accumulate tokens; speak each complete sentence immediately
        std::string sentence_buf;
        std::mutex  buf_mutex;

        auto speak_sentence = [&](const std::string& s) {
            if (s.empty() || stop_requested_.load()) return;
            set_state(VoiceLoopState::SPEAKING);
            TTSRequest req;
            req.text         = s;
            req.speaker_id   = config_.tts.speaker_id;
            req.length_scale = config_.tts.length_scale;
            req.noise_scale  = config_.tts.noise_scale;
            req.noise_w      = config_.tts.noise_w;
            TTSResult r = tts_->synthesise(req);
            if (r.success && !stop_requested_.load()) {
                audio_->play(r.samples, r.sample_rate);
                audio_->wait_until_done();
            }
        };

        chat_fn_(config_.session_id, transcript,
                 [&](const std::string& token, bool is_final) -> bool {
                     if (stop_requested_.load()) return false;
                     std::lock_guard<std::mutex> lock(buf_mutex);
                     sentence_buf += token;

                     auto sentences = TTSEngine::split_sentences(sentence_buf);

                     if (!is_final && sentences.size() >= 2) {
                         // Speak all complete sentences (all but last)
                         for (size_t i = 0; i + 1 < sentences.size(); ++i)
                             speak_sentence(sentences[i]);
                         sentence_buf = sentences.back();
                     } else if (is_final) {
                         for (const auto& s : sentences)
                             speak_sentence(s);
                         sentence_buf.clear();
                     }
                     return !stop_requested_.load();
                 });

    } else {
        // Full mode: collect entire response, then speak
        std::string full_response;
        chat_fn_(config_.session_id, transcript,
                 [&](const std::string& token, bool /*is_final*/) -> bool {
                     full_response += token;
                     return !stop_requested_.load();
                 });

        if (!full_response.empty() && !stop_requested_.load()) {
            set_state(VoiceLoopState::SPEAKING);
            TTSRequest req;
            req.text         = full_response;
            req.speaker_id   = config_.tts.speaker_id;
            req.length_scale = config_.tts.length_scale;
            req.noise_scale  = config_.tts.noise_scale;
            req.noise_w      = config_.tts.noise_w;

            tts_->synthesise_streaming(full_response, req,
                [this](TTSResult r) {
                    if (r.success && !stop_requested_.load()) {
                        audio_->play(r.samples, r.sample_rate);
                        audio_->wait_until_done();
                    }
                });
        }
    }

    if (!stop_requested_.load())
        set_state(VoiceLoopState::IDLE);
}

// =============================================================================
// Push-to-talk keyboard thread
// Reads raw stdin; space down = start recording, any other key = stop.
// =============================================================================

void VoiceLoop::ptt_thread_fn() {
    struct termios oldt{}, newt{};
    tcgetattr(STDIN_FILENO, &oldt);
    newt          = oldt;
    newt.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    newt.c_cc[VMIN]  = 1;
    newt.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    bool space_down = false;

    while (running_.load() && !stop_requested_.load()) {
        char c = 0;
        if (read(STDIN_FILENO, &c, 1) <= 0) break;

        if (c == ' ') {
            if (!space_down) { space_down = true;  ptt_key_down(); }
        } else {
            if (space_down)  { space_down = false; ptt_key_up();   }
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

void VoiceLoop::ptt_key_down() {
    if (ptt_recording_.exchange(true)) return;
    LOG_DEBUG("VoiceLoop: PTT key down");
    set_state(VoiceLoopState::RECORDING);
    std::lock_guard<std::mutex> lock(ptt_buf_mutex_);
    ptt_buffer_.clear();
}

void VoiceLoop::ptt_key_up() {
    if (!ptt_recording_.exchange(false)) return;
    LOG_DEBUG("VoiceLoop: PTT key up");

    AudioChunk chunk;
    {
        std::lock_guard<std::mutex> lock(ptt_buf_mutex_);
        chunk.samples.swap(ptt_buffer_);
    }
    chunk.sample_rate = config_.audio.sample_rate;
    chunk.channels    = config_.audio.channels;

    if (!chunk.empty()) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        pending_segments_.push(std::move(chunk));
        work_cv_.notify_one();
    }
}

// =============================================================================
// Public API
// =============================================================================

TTSResult VoiceLoop::speak(const std::string& text) {
    TTSRequest req;
    req.text         = text;
    req.speaker_id   = config_.tts.speaker_id;
    req.length_scale = config_.tts.length_scale;
    req.noise_scale  = config_.tts.noise_scale;
    req.noise_w      = config_.tts.noise_w;
    TTSResult result = tts_->synthesise(req);
    if (result.success)
        audio_->play(result.samples, result.sample_rate);
    return result;
}

TranscriptResult VoiceLoop::transcribe(const AudioChunk& audio) {
    return stt_->transcribe(audio);
}

void VoiceLoop::stop_speaking() {
    audio_->stop_playback();
    if (state() == VoiceLoopState::SPEAKING)
        set_state(VoiceLoopState::IDLE);
}

void VoiceLoop::set_input_mode(VoiceInputMode mode) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.input_mode = mode;
}

VoiceInputMode VoiceLoop::input_mode() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return config_.input_mode;
}

// =============================================================================
// State helpers
// =============================================================================

void VoiceLoop::set_state(VoiceLoopState s) {
    { std::lock_guard<std::mutex> lock(state_mutex_); state_ = s; }
    if (state_cb_) state_cb_(s);
    LOG_DEBUG("VoiceLoop: state -> " + voice_loop_state_to_string(s));
}

VoiceLoopState VoiceLoop::state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

bool VoiceLoop::is_barge_in_candidate() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_ == VoiceLoopState::SPEAKING;
}

VoiceStatus VoiceLoop::get_status() const {
    VoiceStatus vs;
    vs.active          = running_.load();
    vs.input_mode      = voice_input_mode_to_string(config_.input_mode);
    vs.current_state   = voice_loop_state_to_string(state());
    vs.stt_ready       = stt_ && stt_->is_ready();
    vs.tts_ready       = tts_ && tts_->is_ready();
    vs.wake_word_ready = wake_word_ && wake_word_->is_ready();
    vs.session_id      = config_.session_id;
    vs.transcriptions  = transcription_count_.load();
    vs.utterances      = utterance_count_.load();
    return vs;
}

void VoiceLoop::set_state_callback(VoiceStateCallback cb) { state_cb_      = std::move(cb); }
void VoiceLoop::set_transcript_callback(TranscriptCallback cb) { transcript_cb_ = std::move(cb); }

} // namespace cardinal
