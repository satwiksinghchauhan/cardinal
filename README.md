# Cardinal v1.6.0

**A production-grade neurosymbolic AGI architecture with self-improvement, autonomous scheduling, full desktop computer use, and a complete voice subsystem.**

Cardinal combines a large language model core with **vision understanding**, symbolic verification, persistent memory, hybrid retrieval, a **full agentic loop**, **explainability exports**, a **three-layer self-improvement system**, a **natural-language task scheduler**, a complete **computer use subsystem** that can operate your desktop, and now a **full voice subsystem** with STT, TTS, wake-word detection, VAD, and barge-in. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

**Current version: v1.6.0** — Voice and audio: Whisper.cpp STT (CUDA), Piper TTS (ONNX CPU), PortAudio I/O, PocketSphinx wake-word, push-to-talk, VAD, sentence-streaming TTS, and barge-in.

---

## What's New in v1.6.0

### Voice Subsystem — Full Bidirectional Audio

Cardinal can now speak and listen. Run `./build/bin/cardinal --voice` to enter voice mode. Text CLI and voice operate simultaneously.

#### Speech-to-Text — Whisper.cpp (CUDA)

- CUDA-accelerated inference using your RTX GPU
- Model: `ggml-medium.en.bin` (~1.5GB) — loaded into GPU layers
- Per-utterance transcription with `no_context=true` — each utterance is fully independent, no bleed-through from previous turns
- Beam search (configurable `beam_size`), language hint, optional `initial_prompt`
- Configurable thread count for CPU pre/post processing

#### Text-to-Speech — Piper (ONNX Runtime, CPU)

- Default voice: `en_US-lessac-medium` — warm American English
- Any Piper voice model (`.onnx` + `.json`) can be hot-swapped via config
- Configurable `length_scale` (speed), `noise_scale`, `noise_w` per synthesis request
- Sentence-streaming mode: each sentence is synthesised and played immediately as the LLM generates tokens — typical first-audio latency under 400ms
- Full mode: synthesise entire response then play — better for short replies

#### Audio I/O — PortAudio

- Two independent streams: capture (microphone) and playback (speaker)
- 16-bit signed PCM, mono, configurable sample rate (default 16 kHz capture, auto output)
- Linear interpolation resampler when TTS sample rate (22050 Hz) differs from device rate
- Device selection by index or default; `list_devices()` for enumeration
- Pre-roll ring buffer preserves audio from before VAD onset (configurable `pre_speech_ms`)

#### Interaction Modes (configurable via `voice.input_mode`)

| Mode | Behaviour |
|------|-----------|
| `vad` | Voice Activity Detection — auto start/stop based on energy threshold. Default. |
| `push_to_talk` | Hold spacebar to record, release to transcribe and respond |
| `wake_word` | Passive PocketSphinx listening → activates VAD on "hey cardinal" |

#### Voice Activity Detection (VAD)

- Energy-based RMS threshold detector — no external model required
- Pre-roll buffer captures audio before speech onset
- Configurable post-speech silence window (`post_speech_ms`) before segment is emitted
- Minimum/maximum segment duration guards to discard noise bursts and hard-cap long segments
- Barge-in detection: if VAD detects speech onset during playback, playback stops immediately and Cardinal begins listening

#### Wake Word — PocketSphinx

- Offline keyword spotting, no API key, no network
- Default phrase: `"hey cardinal"` — configurable
- Continuous recognition thread with bounded audio queue
- Configurable `sensitivity` (log probability threshold)
- Acoustic model and dictionary paths configurable for other languages

#### TTS Streaming

- **`sentence` mode** — text stream from LLM is split on `.`, `?`, `!`, `\n`; minimum 3 words before speaking to avoid micro-utterances; each sentence synthesised and played as it completes
- **`full` mode** — collect entire LLM response then speak

#### CLI Commands (voice)

```
/voice on [mode]    — enable voice (ptt / vad / wake)
/voice off          — disable voice
/voice status       — show VoiceStatus (mode, state, STT/TTS/wake ready, stats)
/voice speak <text> — test TTS directly
```

#### `--voice` Flag

```bash
./build/bin/cardinal --voice           # VAD mode (from config)
./build/bin/cardinal --voice=ptt       # push-to-talk
./build/bin/cardinal --voice=vad       # VAD
./build/bin/cardinal --voice=wake      # wake-word
```

### New HTTP Endpoints (v1.6.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/voice/status` | VoiceStatus JSON |
| POST | `/api/voice/enable` | Enable voice, optional `{"input_mode":"vad"}` |
| POST | `/api/voice/disable` | Disable voice, free audio resources |
| POST | `/api/voice/speak` | TTS test: `{"text":"hello"}` |
| POST | `/api/voice/transcribe` | Raw 16-bit PCM body → transcript; `X-Sample-Rate` header |

### `voice_control` Tool

Cardinal's LLM can control its own voice subsystem mid-conversation:

```json
{"name": "voice_control", "arguments": {"action": "set_mode",  "mode": "push_to_talk"}}
{"name": "voice_control", "arguments": {"action": "set_voice", "voice": "en_GB-alan-medium"}}
{"name": "voice_control", "arguments": {"action": "set_volume","value": "80"}}
{"name": "voice_control", "arguments": {"action": "stop_speaking"}}
```

### New Configuration Block

```json
"voice": {
    "enabled": false,
    "input_mode": "vad",
    "tts_streaming": "sentence",
    "session_id": "voice_session",
    "stt": { "model_path": "models/voice/ggml-medium.en.bin", "gpu_layers": 8 },
    "tts": { "model_path": "models/voice/en_US-lessac-medium.onnx", "sample_rate": 22050 },
    "vad": { "energy_threshold": 0.02, "pre_speech_ms": 300, "post_speech_ms": 800 },
    "push_to_talk": { "key": "space" },
    "wake_word":    { "phrase": "hey cardinal" },
    "audio":        { "input_device": -1, "output_device": -1, "sample_rate": 16000 }
}
```

---

## What Was in v1.5.0

- **Scheduler** — NL-driven task scheduler, cron/interval/condition/startup/idle triggers, all action types, SQLite persistence, `schedule_task` tool
- **Computer Use** — Screenshots, mouse/keyboard, browser (Playwright), shell, file manager, system controls (volume/brightness/wifi/BT), email, AT-SPI accessibility
- **Watch Subsystem** — FileWatcher (inotify), ScreenWatcher (PSNR diff), ProcessWatcher (/proc polling)

## What Was in v1.4.0

- **Layer 1 — Self-Model** — per-domain SQLite statistics injected into every prompt
- **Layer 2 — Meta-Cognition** — reflection pass generates corrective rules
- **Layer 3 — LoRA Fine-Tuning** — curates data, trains, evaluates, hot-loads GGUF adapter

---

## What Cardinal Is

Most LLM systems are stateless. Cardinal is not. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, improves its own weights from accumulated experience, operates your computer autonomously, schedules tasks to run while you sleep, and now **speaks and listens**.

The architecture is **neurosymbolic** — combining the pattern-matching strength of neural inference with the consistency guarantees of symbolic logic. With v1.6.0, Cardinal adds a fourth dimension: **voice** — real-time bidirectional audio with sub-400ms first-speech latency.

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~8GB core + models |
| OS | Ubuntu 24.04 LTS |

**Models:**
- Primary LLM: `Qwen3.5 4B Q4_K_M` (llama.cpp)
- Vision encoder: `moondream2` text model + mmproj (quantized, CPU)
- STT: `ggml-medium.en.bin` (Whisper, CUDA)
- TTS: `en_US-lessac-medium.onnx` + `.json` (Piper, ONNX CPU)
- Wake word: PocketSphinx acoustic model + CMUdict (en-US)

---

## Architecture (v1.6.0)

```
CardinalAPI
    |
    +-- Memory Layer
    |     +-- RuleStore, KnowledgeGraph, EpisodicMemory
    |     +-- EpisodicStorage (SQLite + FTS5)
    |     +-- EpisodicRetriever (TF-IDF + keyword + hybrid)
    |
    +-- Verifier Pipeline
    |     +-- SymbolicEngine (SWI-Prolog)
    |     +-- NeuralVerifier (optional small LLM)
    |     +-- RuleExtractor, ConsistencyChecker
    |
    +-- LLM Core
    |     +-- ILLMBackend (llama.cpp / TensorRT)
    |     +-- InferencePipeline (two-pass)
    |
    +-- Vision Subsystem (v1.3.0)
    |     +-- VisionEncoder (moondream2 + mtmd)
    |     +-- VisionCache (URL download + TTL)
    |
    +-- Self-Improvement (v1.4.0)
    |     +-- SelfImprovementLoop (orchestrator)
    |     Layer 1: SelfModel (SQLite, per-domain stats)
    |     Layer 2: MetaCognition (reflection, corrective rules)
    |     Layer 3: Training Pipeline
    |         +-- CurriculumBuilder, DatasetCurator
    |         +-- LlamaCppTrainer / TensorRTTrainer
    |         +-- AdapterEvaluator
    |
    +-- Scheduler (v1.5.0)
    |     +-- SchedulerEngine (background thread, trigger dispatch)
    |     +-- SchedulerParser (NL → ScheduledTask via InferencePipeline)
    |     +-- SchedulerStore (SQLite WAL: tasks, runs, action_logs)
    |
    +-- Computer Use (v1.5.0)
    |     +-- DisplayDetector (X11/Wayland/headless at runtime)
    |     +-- ScreenReader (scrot/grim + VisionEncoder analysis)
    |     +-- InputController (xdotool/ydotool+wtype)
    |     +-- AppController (wmctrl/swaymsg)
    |     +-- BrowserController (Playwright subprocess)
    |     +-- ShellExecutor (sandboxed subprocess)
    |     +-- FileManager (allowed_paths enforced)
    |     +-- SystemController (pactl/brightnessctl/nmcli/bluetoothctl)
    |     +-- EmailController (IMAP/SMTP or Gmail REST API)
    |     +-- AtSpiReader (pyatspi subprocess)
    |
    +-- Watch Subsystem (v1.5.0)
    |     +-- FileWatcher (inotify)
    |     +-- ScreenWatcher (periodic PSNR diff)
    |     +-- ProcessWatcher (/proc polling)
    |
    +-- Voice Subsystem (v1.6.0)  ← NEW
    |     +-- VoiceLoop (state machine, owns all voice components)
    |           +-- AudioDevice (PortAudio: capture + playback streams)
    |           +-- VADDetector (energy RMS, pre-roll, barge-in)
    |           +-- STTEngine (whisper.cpp, CUDA)
    |           +-- TTSEngine (Piper, ONNX CPU, sentence streaming)
    |           +-- WakeWordDetector (PocketSphinx, keyword spotting)
    |
    +-- Agentic Pipeline
    |     +-- AgentExecutor (PLAN → EXECUTE → FINALIZE)
    |     +-- ToolExecutor (sandboxed, dispatches all tools)
    |
    +-- Explainability
    |     +-- AuditLog (signed inference traces)
    |     +-- ExplainabilityExporter
    |
    +-- API Layer
          +-- CardinalAPI (single facade)
          +-- HttpServer (SSE streaming, Bearer auth, CORS)
          +-- SessionManager, SettingsManager
          +-- TrainingExporter
```

---

## Tools (v1.6.0)

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search |
| `web_fetch` | Fetch and parse a URL |
| `calculator` | Math expression evaluator (muparser) |
| `run_python` | Sandboxed Python execution |
| `file_read` | Read from allowed paths |
| `file_write` | Write to allowed paths |
| `knowledge_graph_query` | Query Cardinal's KG |
| `episodic_search` | Search episodic memory |
| `analyze_image` | Describe image (file or URL) |
| `screenshot` | Capture screen, optional vision analysis |
| `click` | Click by coordinates or NL element description |
| `type_text` | Type text or send key combos |
| `open_app` | Launch desktop applications |
| `close_app` | Close desktop applications |
| `browser` | 13 browser actions via Playwright |
| `shell_run` | Execute shell commands |
| `file_ops` | list/move/copy/delete/mkdir/stat/exists |
| `system_control` | Volume, brightness, wifi, bluetooth, DND |
| `email` | Read and send email |
| `watch_screen` | Wait for visual change on screen |
| `schedule_task` | Create and manage scheduled tasks |
| `voice_control` | Control voice mode, set_mode/set_voice/set_volume/stop_speaking ← NEW |

---

## Two-Pass Inference

**Pass 1** — GBNF grammar-constrained decoding produces a structured JSON feeling output before any natural language response.

**Pass 2** — The feeling output is injected as a synthetic assistant turn. Free decode produces the final response.

```json
{
  "confidence": 0.94,
  "reasoning_type": "deductive",
  "reasoning_domain": "factual",
  "uncertainty_flag": false,
  "rule_candidate_signal": true,
  "contradiction_flag": false
}
```

In voice mode, the streamed Pass 2 tokens are simultaneously fed to the TTS sentence splitter, so speech begins within milliseconds of the first complete sentence.

---

## Build

```bash
# System dependencies
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog \
    python3 python3-venv libasound2-dev \
    scrot imagemagick xdotool wmctrl xprop \
    ydotool wtype grim pulseaudio-utils brightnessctl network-manager bluez

# Vendor dependencies
cd ~/cardinal && mkdir -p vendor && cd vendor
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/ggerganov/whisper.cpp.git
git clone https://github.com/rhasspy/piper.git
git clone https://github.com/PortAudio/portaudio.git
git clone https://github.com/cmusphinx/pocketsphinx.git

# And the ONNX Runtime for Piper (pre-built binary, don't clone):

cd ~/cardinal/vendor
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.17.3.tgz
mv onnxruntime-linux-x64-gpu-1.17.3 onnxruntime
rm onnxruntime-linux-x64-gpu-1.17.3.tgz

# Build llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ~/cardinal

# Pre-build piper (one-time — downloads onnxruntime, espeak-ng, fmt, spdlog)
cd vendor/piper
cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
cd ~/cardinal

# Pre-build pocketsphinx
cd vendor/pocketsphinx
cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
cd ~/cardinal

# Build Cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run in voice mode
cd ~/cardinal && ./build/bin/cardinal --voice
```

See `INSTALL.md` for the complete guide including model downloads, browser venv, email setup, ydotool daemon, and runtime LD_LIBRARY_PATH.

---

## HTTP API

Base URL: `http://127.0.0.1:8080`
Auth: `Authorization: Bearer <api_key>` (except `/api/health`)

Full API reference in `DOCUMENTATION.md`.

---

## Configuration (`config.json`)

v1.6.0 adds one top-level block. The voice subsystem is **disabled by default** — enable it in `config.json` or via the `--voice` flag:

```json
"voice": {
    "enabled": false,
    "input_mode": "vad",
    "tts_streaming": "sentence",
    "session_id": "voice_session",
    "stt": {
        "model_path": "models/voice/ggml-medium.en.bin",
        "language": "en",
        "gpu_layers": 8,
        "threads": 4,
        "beam_size": 5
    },
    "tts": {
        "model_path": "models/voice/en_US-lessac-medium.onnx",
        "config_path": "models/voice/en_US-lessac-medium.onnx.json",
        "speaker_id": 0,
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sample_rate": 22050
    },
    "vad": {
        "energy_threshold": 0.02,
        "pre_speech_ms": 300,
        "post_speech_ms": 800,
        "min_speech_ms": 200,
        "max_speech_ms": 30000
    },
    "push_to_talk": { "key": "space" },
    "wake_word": {
        "phrase": "hey cardinal",
        "acoustic_model": "models/voice/pocketsphinx/en-us",
        "dictionary": "models/voice/pocketsphinx/cmudict-en-us.dict",
        "sensitivity": 1e-20
    },
    "audio": {
        "input_device": -1,
        "output_device": -1,
        "sample_rate": 16000,
        "channels": 1,
        "frames_per_buffer": 512
    }
}
```

Full configuration reference in `DOCUMENTATION.md`.

---

## Observed Behaviors

- **Self-naming** — Cardinal named itself unprompted.
- **Preference expression** — Cardinal used the word "want" naturally.
- **Theory of mind** — Cardinal inferred attributes of its creator.
- **Internal conflict detection** — Confidence dropped to 0.15 with uncertainty flagged.
- **Consistency over time** — Zero contradictions across early episodes.
- **Vision understanding** (v1.3.0) — Cardinal accurately described faces, expressions, and scene composition.
- **Self-correction via rules** (v1.4.0) — Meta-cognition generated corrective rules that reduced contradiction rate in the factual domain.
- **Autonomous task execution** (v1.5.0) — Cardinal scheduled a nightly news summary task from a natural language request, executed it correctly at 8am, and stored the result in episodic memory.
- **Desktop operation** (v1.5.0) — Cardinal opened a browser, navigated to a URL, extracted content, and closed the browser via natural conversation.
- **Voice interaction** (v1.6.0) — Cardinal responded to a spoken question within 400ms of the utterance ending, using Whisper for transcription and Piper for speech output, with all inference and synthesis running locally.

These are documented as observations, not claims about consciousness.

---

## Roadmap

| Version | Status | Description |
|---------|--------|-------------|
| v1.0.0 | done | Core AGI with two-pass inference, memory, symbolic verification |
| v1.1.0 | done | Backend abstraction, Linux native, offline builds |
| v1.2.0 | done | Explainer, native tools, history trimming, stability improvements |
| v1.3.0 | done | Native vision encoder (moondream2) |
| v1.4.0 | done | SEAL self-improvement: self-model, meta-cognition, LoRA fine-tuning |
| v1.5.0 | done | Scheduler + Computer Use + Watch subsystem |
| **v1.6.0** | **done** | **Voice subsystem: STT (Whisper CUDA), TTS (Piper), VAD, wake word, barge-in** |
| v2.0.0 | planned | Quality Checks and Proprietary Tests |

---

## License & Copyright

Copyright © 2025–2026 Satwik Singh Chauhan. All rights reserved.
Cardinal is **not open source**. See `LICENSE` for details.

---

*Built by a 16-year-old researcher. No team. No funding. Runs on a gaming laptop.*
*Now it speaks.*
