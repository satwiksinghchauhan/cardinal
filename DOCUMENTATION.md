# Cardinal v1.6.0 — Technical Documentation

**Architecture:** Neurosymbolic AGI Core + Agentic Loop + Explainability + Vision + Self-Improvement + Scheduler + Computer Use + Watch + Voice
**Language:** C++20
**Platform:** Linux (Ubuntu 24.04 LTS)
**GPU:** NVIDIA CUDA (llama.cpp / TensorRT for LLM; CUDA for Whisper STT)
**Vision:** moondream2 via `llama.cpp` `mtmd` subsystem
**Self-Improvement:** Three-layer SEAL system (self-model, meta-cognition, LoRA fine-tuning)
**v1.6.0 additions:** Full voice subsystem — Whisper.cpp STT (CUDA), Piper TTS (ONNX CPU), PortAudio I/O, PocketSphinx wake-word, VAD, push-to-talk, sentence-streaming TTS, barge-in

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Two-Pass Inference](#2-two-pass-inference)
3. [Feeling Output Schema](#3-feeling-output-schema)
4. [Memory Systems](#4-memory-systems)
5. [Retrieval System](#5-retrieval-system)
6. [Verifier Pipeline](#6-verifier-pipeline)
7. [Rule System](#7-rule-system)
8. [Backend Abstraction](#8-backend-abstraction)
9. [Vision Subsystem (v1.3.0)](#9-vision-subsystem-v130)
10. [Self-Improvement Subsystem (v1.4.0)](#10-self-improvement-subsystem-v140)
11. [Scheduler Subsystem (v1.5.0)](#11-scheduler-subsystem-v150)
12. [Computer Use Subsystem (v1.5.0)](#12-computer-use-subsystem-v150)
13. [Watch Subsystem (v1.5.0)](#13-watch-subsystem-v150)
14. [Voice Subsystem (v1.6.0)](#14-voice-subsystem-v160)
15. [Agentic Pipeline](#15-agentic-pipeline)
16. [Tools System](#16-tools-system)
17. [Explainability Exports](#17-explainability-exports)
18. [Training Export](#18-training-export)
19. [Configuration Reference](#19-configuration-reference)
20. [CardinalAPI Reference](#20-cardinalapi-reference)
21. [HTTP API Reference](#21-http-api-reference)
22. [Settings Manager](#22-settings-manager)
23. [Session Manager](#23-session-manager)
24. [Type Reference](#24-type-reference)
25. [Error Handling](#25-error-handling)
26. [Offline Builds & Vendoring](#26-offline-builds--vendoring)
27. [Module Reference](#27-module-reference)
28. [Threading Model](#28-threading-model)
29. [Lifecycle and Startup Sequence](#29-lifecycle-and-startup-sequence)

---

## 1. Architecture Overview

Cardinal is structured in four layers. Each layer depends only on the layers below it.

```
Layer 4 — Interfaces
    CLI (interactive loop, /commands, --voice flag)
    HTTP API (SSE streaming, Bearer auth)

Layer 3 — API Layer
    CardinalAPI         single facade, owns all components
    HttpServer          SSE streaming, Bearer auth, CORS
    SettingsManager     runtime-mutable config
    SessionManager      multi-session conversation state
    TrainingExporter    Alpaca JSONL export
    Explainability      audit log, cryptographic signing

Layer 2 — Core Systems
    InferencePipeline   two-pass orchestrator, prompt injection
    AgentExecutor       PLAN → EXECUTE loop
    ToolExecutor        sandboxed tool execution (all tools including voice control)
    ILLMBackend         abstract backend (llama.cpp / TensorRT)
    ConsistencyChecker  verifier orchestrator
    SymbolicEngine      SWI-Prolog integration
    SchedulerEngine     background thread, NL parsing, task dispatch    ← v1.5.0
    Computer Use Layer  display, input, browser, shell, file, email     ← v1.5.0
    Watch Subsystem     file, screen, process watchers                  ← v1.5.0
    VoiceLoop           STT, TTS, VAD, wake-word, barge-in             ← v1.6.0

Layer 1 — Foundation
    RuleStore           persistent rule base
    KnowledgeGraph      typed node graph
    EpisodicMemory      JSONL audit trail
    EpisodicStorage     SQLite + FTS5 searchable index
    EpisodicRetriever   TF-IDF + keyword + hybrid retrieval
    VisionEncoder       moondream2 via mtmd API                        ← v1.3.0
    VisionCache         URL download cache, TTL eviction               ← v1.3.0
    SelfImprovementLoop SEAL orchestrator                              ← v1.4.0
    SelfModel           per-domain statistics SQLite                   ← v1.4.0
    MetaCognition       reflection, corrective rules                   ← v1.4.0
    Training Pipeline   CurriculumBuilder, DatasetCurator, trainers   ← v1.4.0
    SchedulerStore      SQLite WAL: tasks, runs, action_logs           ← v1.5.0
    AudioDevice         PortAudio capture + playback streams           ← v1.6.0
    VADDetector         energy RMS, pre-roll, barge-in detection       ← v1.6.0
    STTEngine           whisper.cpp, CUDA                              ← v1.6.0
    TTSEngine           Piper ONNX CPU, sentence streaming             ← v1.6.0
    WakeWordDetector    PocketSphinx keyword spotting                  ← v1.6.0
    ConfigLoader        typed config, validated at startup
    Logger              thread-safe, 6 levels
```

### Component Ownership

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary — callers see only types defined in `cardinal_types.h`.

The voice subsystem is owned differently from all other subsystems: `CardinalAPI` owns a single `VoiceLoop` which in turn owns all five voice components (`AudioDevice`, `VADDetector`, `STTEngine`, `TTSEngine`, `WakeWordDetector`). This containment means all audio resources are freed atomically when `VoiceLoop::stop()` is called.

### Dependency Graph (v1.6.0)

```
CardinalAPI
    owns --> ILLMBackend
    owns --> InferencePipeline --> ILLMBackend
                               --> EpisodicRetriever
    owns --> AgentExecutor --> InferencePipeline
                           --> ToolExecutor
    owns --> ToolExecutor --> [all controllers below via setters]
    owns --> RuleStore, KnowledgeGraph, EpisodicMemory, EpisodicStorage
    owns --> EpisodicRetriever --> EpisodicStorage
    owns --> SymbolicEngine, NeuralVerifier, RuleExtractor, ConsistencyChecker
    owns --> VisionCache, VisionEncoder --> mtmd library
    owns --> SelfImprovementLoop
                --> SelfModel, MetaCognition
                --> CurriculumBuilder, DatasetCurator
                --> ITrainingBackend, AdapterEvaluator
    owns --> SchedulerEngine (v1.5.0)
                --> SchedulerStore  --> scheduler.db
                --> SchedulerParser --> InferencePipeline
                --> AgentExecutor, SelfImprovementLoop, EpisodicStorage
    owns --> DisplayDetector (v1.5.0)
    owns --> ScreenReader    --> DisplayDetector, VisionEncoder
    owns --> InputController --> DisplayDetector
    owns --> AppController   --> DisplayDetector
    owns --> BrowserController --> VisionEncoder
    owns --> ShellExecutor
    owns --> FileManager
    owns --> SystemController
    owns --> EmailController
    owns --> AtSpiReader
    owns --> FileWatcher, ScreenWatcher, ProcessWatcher (v1.5.0)
    owns --> VoiceLoop (v1.6.0)  ← NEW
                owns --> AudioDevice   --> PortAudio
                owns --> VADDetector
                owns --> STTEngine     --> whisper.cpp (CUDA)
                owns --> TTSEngine     --> Piper (ONNX CPU)
                owns --> WakeWordDetector --> PocketSphinx
    owns --> AuditLog, ExplainabilityExporter
    owns --> TrainingExporter, SettingsManager, SessionManager
    owns --> HttpServer
```

---

## 2. Two-Pass Inference

Every inference Cardinal performs consists of exactly two passes through the language model using separate inference contexts to prevent grammar state contamination.

### Pass 1 — Constrained Decoding (Feeling Output)

GBNF grammar-constrained decoding forces the model to produce a structured JSON object before any natural language. The grammar is in `src/prompts/feeling_schema.gbnf`.

### Synthetic Turn Injection

The feeling JSON is injected as a synthetic assistant turn — the model reads its own feeling as something it already said, not an instruction.

### Pass 2 — Free Decoding (Final Response)

No grammar constraint. Streaming callback fires per token.

**v1.6.0 Voice mode:** In sentence-streaming TTS mode, the Pass 2 stream callback simultaneously feeds tokens to the TTS sentence splitter. When a complete sentence boundary is detected (`.`, `?`, `!`, `\n`) and the buffer has ≥3 words, synthesis begins immediately — overlapping with the remaining LLM generation.

### v1.4.0 Addition — Self-Improvement Hook

After every `run_post_inference()`, `SelfImprovementLoop::on_inference()` is called with feeling output fields:
- Layer 1: O(1) SQLite upsert to `domain_stats` and `reasoning_stats`
- Layer 2: checks inference counter and contradiction rate thresholds
- Layer 3: checks episode counter and domain confidence thresholds

### v1.5.0 Addition — Scheduler Hook

After every inference, `SchedulerEngine::on_inference()` updates the idle tracker timestamp.

### Prompt Injection (v1.4.0)

If `inject_into_prompt=true`, a `[Self-Model]` block is prepended:

```
[Self-Model]
Weakest domain:   factual
Strongest domain: mathematical
Domain          Conf  Contradict  Uncertain  Inferences
factual         0.71      0.12       0.08         142
mathematical    0.93      0.01       0.02          38
```

### Retry Logic

If Pass 1 fails to produce valid JSON, retried up to `max_retries` times with `retry_delay_ms` delay.

### Metrics

Every `ChatResponse` carries: `pass1_tokens`, `pass2_tokens`, `total_ms`.

---

## 3. Feeling Output Schema

**Fields:** `confidence` (float 0–1), `reasoning_type` (causal/deductive/inductive/abductive/analogical/associative), `reasoning_domain` (factual/ethical/spatial/temporal/social/mathematical), `uncertainty_flag` (bool), `contradiction_flag` (bool), `rule_candidate_signal` (bool).

**Validation:** `confidence > 0.8` with `uncertainty_flag = true` is rejected as contradictory and triggers a retry.

---

## 4. Memory Systems

### 4.1 RuleStore

Persistent symbolic memory. Rules: `id`, `domain`, `condition`, `consequence`, `confidence`, `trigger_count`, timestamps, provenance (`episode_id`, `reasoning_type`).
Storage: `data/memory/rules.json` (atomic writes).
v1.4.0: corrective rules stored here with `reasoning_type = "meta_correction"`.

### 4.2 KnowledgeGraph

Typed nodes: `concept`, `fact`, `entity`, `relation`. Storage: `data/memory/knowledge.json`.

### 4.3 EpisodicMemory

Append-only JSONL audit trail (`logs/episodic.log`). Never modified after write.

### 4.4 EpisodicStorage

SQLite + FTS5 searchable index. Synchronised with JSONL via migration on first open. Used by `DatasetCurator` (Layer 3), `MetaCognition` (Layer 2), and `SchedulerEngine` (v1.5.0) to store scheduled task results.

### 4.5 SelfModel DB (v1.4.0)

SQLite at `data/self_model/self_model.db`. Two tables: `domain_stats` and `reasoning_stats`. All writes are single-statement `INSERT ... ON CONFLICT DO UPDATE` upserts — O(1) regardless of inference count.

### 4.6 SchedulerStore (v1.5.0)

SQLite WAL at `data/scheduler/scheduler.db`. Three tables:

**`scheduled_tasks`** — task definitions with trigger spec, action, run stats.
**`task_runs`** — every run execution record with status, timing, result summary.
**`task_action_logs`** — per-step action log within a run.

WAL mode enabled for concurrent read during active run logging.

---

## 5. Retrieval System

`EpisodicRetriever` provides three modes:
- **KEYWORD** — SQLite FTS5, BM25 ranking
- **SEMANTIC** — TF-IDF cosine similarity (in-memory index)
- **HYBRID** — weighted combination (default: 0.7 keyword, 0.3 semantic)

Retrieved episodes inject as a `[MEMORY CONTEXT]` block. Retrieval failure is non-fatal.

---

## 6. Verifier Pipeline

Runs after every inference to maintain rule base integrity.

- **Modes:** `symbolic` (SWI-Prolog), `neural` (small LLM), `hybrid`
- **Per-inference sequence:** rule extraction → consistency check → contradiction resolution → rule commit → `set_extracted_rule_id()` on episode

---

## 7. Rule System

Rules: `id`, `domain`, `condition`, `consequence`, `confidence`, `trigger_count`, `episode_id`, `reasoning_type`.

**Lifecycle:** extraction → consistency check → commit → confidence decay → pruning.
**v1.4.0:** `meta_correction` type — rules from MetaCognition reflection passes.
**Dedup:** Jaccard overlap > 0.8 on condition text → merge rather than duplicate.

---

## 8. Backend Abstraction

`ILLMBackend` abstract interface. Two implementations:

| Method | LlamaCppBackend | TensorRTBackend |
|--------|-----------------|-----------------|
| `generate_feeling()` | GBNF constrained, ctx_pass1_ | GBNF constrained |
| `generate_response()` | Free decode, ctx_pass2_ | Free decode |
| `load_lora_adapter()` | `llama_adapter_lora_init` + `llama_set_adapters_lora` | no-op |
| `unload_lora_adapter()` | `llama_set_adapters_lora(0)` + free | no-op |

The adapter is applied only to `ctx_pass2_`. `ctx_pass1_` reflects the base model's intrinsic reasoning quality unmodified by fine-tuning.

---

## 9. Vision Subsystem (v1.3.0)

Unchanged from v1.3.0.

- **`VisionEncoder`** — moondream2 via `mtmd` API. Method: `encode(image_path, prompt, ImageMetadata) → VisionResult`
- **`VisionCache`** — URL download cache with TTL eviction
- **`analyze_image` tool** — accepts local paths and HTTP/HTTPS URLs

**v1.5.0 usage:** `ScreenReader::analyze()` calls `VisionEncoder::encode()` with `ImageSource::FILE` to describe screenshots. Vision-based element finding uses the same encoder with a coordinate-extraction prompt.

Performance on RTX 3050 4GB: ~9–10s first image, ~2–3s cached.

---

## 10. Self-Improvement Subsystem (v1.4.0)

All three layers owned and orchestrated by `SelfImprovementLoop`.

### 10.1 SelfImprovementLoop

**Hooks:**
- `on_inference(domain, reasoning_type, confidence, contradiction, uncertainty, rule_committed)` — after every inference. Fast path: updates atomics, may trigger reflection or post to training thread.
- `on_session_boundary()` — at `destroy_session()`. Applies pending LoRA adapter at a clean cut-point.

**Training thread:** sleeps on `condition_variable`, wakes on trigger or 60-second poll. One cycle at a time; overlapping triggers coalesced.

### 10.2 Layer 1 — SelfModel

Maintains `data/self_model/self_model.db`. O(1) upsert after every inference. Provides `format_for_prompt()` for system prompt injection.

**Weakness score:** `(contradiction_rate × 0.4) + (uncertainty_rate × 0.3) + ((1 - avg_confidence) × 0.3)`

### 10.3 Layer 2 — MetaCognition

Reflection pass steps:
1. Query recent failure episodes from `EpisodicStorage`
2. Check `min_failures_to_reflect` threshold
3. Build structured reflection prompt
4. Single LLM pass (no tools, no feeling)
5. Parse JSON findings array
6. Commit corrective rules with `reasoning_type = "meta_correction"`

Uses `std::try_to_lock` — silently drops trigger if reflection already running.

### 10.4 Layer 3 — Training Pipeline

**CurriculumBuilder** — decides target domain by weakness score + recency bonus.

**DatasetCurator** — converts episodes to alpaca format. Quality filters: confidence floor, minimum text length, FNV-1a dedup, holdout reservation. Rule augmentation adds corrective rules as training examples.

**LlamaCppTrainer** — PEFT subprocess → `convert_lora_to_gguf.py` → `llama_set_adapters_lora` on `ctx_pass2_` only.

**AdapterEvaluator** — improvement threshold gate (default 5%). Load policies: `"immediate"` or `"session_boundary"` (default).

---

## 11. Scheduler Subsystem (v1.5.0)

### 11.1 Overview

The Scheduler allows Cardinal to execute tasks autonomously on a defined schedule. Tasks are created from natural language descriptions, parsed into structured `ScheduledTask` objects, persisted in SQLite, and dispatched by a background engine thread.

**Files:**
- `src/scheduler/scheduler_types.h` — all shared types
- `src/scheduler/scheduler_store.h/.cpp` — SQLite persistence
- `src/scheduler/scheduler_parser.h/.cpp` — NL → ScheduledTask via InferencePipeline
- `src/scheduler/scheduler_engine.h/.cpp` — background thread, trigger evaluation, dispatch

### 11.2 ScheduledTask Structure

```cpp
struct ScheduledTask {
    std::string id;               // UUID
    std::string name;
    std::string description;
    bool        enabled;
    TriggerSpec trigger;
    TaskAction  action;
    int         run_count;
    int         fail_count;
    std::string last_run_at;
    std::string next_run_at;
    std::string created_at;
    std::string updated_at;
    std::string created_from;     // "chat" | "api"
    std::string created_in_session;
};
```

### 11.3 Trigger Types

| Type | Description | Required fields |
|------|-------------|-----------------|
| `CRON` | Five-field cron expression | `cron_expression` |
| `INTERVAL` | Every N seconds | `interval_seconds` |
| `CONDITION` | Expression threshold (e.g. `episode_count > 100`) | `condition_expr` |
| `STARTUP` | Fires once on engine start | — |
| `IDLE` | Fires after N minutes without inference | `idle_minutes` |
| `MANUAL` | Only fires on explicit `run_task_now()` | — |

### 11.4 Action Types

| Type | Description |
|------|-------------|
| `AGENT_RUN` | `AgentExecutor::run()` with `goal` |
| `CHAT` | `InferencePipeline::run()` with `goal` as user message |
| `REFLECT` | `MetaCognition::trigger_reflection()` |
| `TRAIN` | `SelfImprovementLoop::trigger_training()` |
| `SELF_IMPROVEMENT` | All three layers sequentially |
| `MAINTENANCE` | `ConsistencyChecker::run_maintenance()` |
| `EXPORT` | `TrainingExporter::export_to_file()` |
| `SHELL` | `ShellExecutor::run(shell_command)` |
| `WEBHOOK` | HTTP POST to `webhook_url` with run result |

---

## 12. Computer Use Subsystem (v1.5.0)

### 12.1 Overview

All computer use components are owned by `CardinalAPI` and wired into `ToolExecutor` via setters.

**Files:** `src/computer/`

### 12.2 DisplayDetector

Detects display server at startup via environment variables:
- `WAYLAND_DISPLAY` set → Wayland
- `DISPLAY` set → X11
- Neither → headless

Checks tool availability: `scrot`, `grim`, `xdotool`, `wmctrl`, `ydotool`, `wtype`, `swaymsg`.

### 12.3 ScreenReader

- X11: `scrot <path>` subprocess; Wayland: `grim <path>` subprocess
- `analyze(image_path, prompt)` → vision encode → description string
- `find_element(description)` → screenshot + vision coordinate extraction prompt → `std::optional<Point>`

### 12.4 InputController

**X11 (xdotool):** `mouse_click`, `type_text`, `send_key`, `mouse_scroll`
**Wayland (ydotool + wtype):** equivalent commands via uinput

### 12.5 AppController

| Method | X11 | Wayland |
|--------|-----|---------|
| `open_app(name)` | `gtk-launch` + exec fallback | same |
| `close_app(name)` | `wmctrl -c` + `pkill` | `swaymsg kill` |
| `focus_app(name)` | `wmctrl -a` | `swaymsg focus` |

### 12.6 BrowserController

Spawns a persistent Playwright Python helper process via the browser venv. Communicates via stdin/stdout JSON lines. Started lazily on first use.

**`BrowserActionType` enum:** `NAVIGATE`, `CLICK`, `CLICK_TEXT`, `TYPE`, `SCROLL`, `GET_CONTENT`, `SCREENSHOT`, `EXECUTE_JS`, `NEW_TAB`, `CLOSE_TAB`, `BACK`, `FORWARD`, `RELOAD`

### 12.7 ShellExecutor

`popen()` with configurable timeout (SIGKILL on expiry). Blocked commands checked before execution. Returns `ShellResult`: `success`, `exit_code`, `stdout_text`, `stderr_text`, `duration_ms`.

### 12.8 FileManager

All paths validated against `computer_use.safety.allowed_paths`. Operations: `list`, `move`, `copy`, `remove`, `mkdir`, `stat`, `exists`.

### 12.9 SystemController

| Method | Tool |
|--------|------|
| `set_volume(pct)` | `pactl set-sink-volume @DEFAULT_SINK@ pct%` |
| `set_brightness(pct)` | `brightnessctl set pct%` |
| `set_wifi(bool)` | `nmcli radio wifi on/off` |
| `set_bluetooth(bool)` | `bluetoothctl power on/off` |

### 12.10 EmailController

Two modes: `imap_smtp` (Python subprocess with `imaplib`/`smtplib`) and `gmail_api` (Google API Python subprocess). Password from `CARDINAL_EMAIL_PASS` env var.

### 12.11 AtSpiReader

Python subprocess using `pyatspi`. `get_tree(app_name)` → `AtSpiNode`. `find_nodes(app_name, role, name_contains)` → `vector<AtSpiNode>`. Used as primary element-location method before vision fallback.

### 12.12 Computer Use Tools

All registered when `computer_use.enabled=true`:

| Tool | Key arguments |
|------|---------------|
| `screenshot` | `analyze`, `prompt`, `region_x/y/w/h` |
| `click` | `description` OR `x`+`y`, `button`, `double_click` |
| `type_text` | `text` OR `key`, `delay_ms` |
| `open_app` | `app`, `focus` |
| `close_app` | `app` |
| `browser` | `action`, `url`, `selector`, `text`, `script` |
| `shell_run` | `command`, `timeout_seconds` |
| `file_ops` | `action`, `path`, `dest`, `recursive` |
| `system_control` | `action`, `value` |
| `email` | `action`, `folder`, `subject`, `from`, `to`, `body` |
| `watch_screen` | `wait_for`, `timeout_seconds`, `poll_seconds` |
| `schedule_task` | `action`, `description`, `task_id` |

---

## 13. Watch Subsystem (v1.5.0)

Three independent background watchers.

### 13.1 FileWatcher

inotify-based. Configured via `FileWatchConfig`: `path`, `recursive`, `events` (CREATE/MODIFY/DELETE/MOVE), `callback`.

### 13.2 ScreenWatcher

Periodic screenshot diff using ImageMagick `compare -metric PSNR`. The `watch_screen` tool blocks until PSNR drops below threshold or timeout expires.

### 13.3 ProcessWatcher

`/proc` polling. `ProcessWatchConfig`: `process_name`, `poll_interval_seconds`, `on_start_callback`, `on_stop_callback`.

---

## 14. Voice Subsystem (v1.6.0)

### 14.1 Overview

The voice subsystem provides full bidirectional audio: speech-to-text transcription of the user's microphone, text-to-speech synthesis of Cardinal's responses, wake-word passive listening, voice activity detection, and barge-in. It is owned by `CardinalAPI` as a single `VoiceLoop` `unique_ptr`.

**Files:** `src/voice/`

**Enabled at runtime:**
- Via `config.json` `voice.enabled=true` — starts with Cardinal
- Via `--voice` flag — starts after `api.init()`, before the interactive loop
- Via `/voice on [mode]` CLI command or `POST /api/voice/enable` HTTP endpoint

**All voice resources are freed** when `VoiceLoop::stop()` is called — PortAudio streams closed, whisper context freed, Piper terminated, PocketSphinx decoder freed.

### 14.2 VoiceLoop State Machine

```
IDLE
 ├─ (push_to_talk) → RECORDING (spacebar held)
 │                 → TRANSCRIBING
 │                 → INFERRING
 │                 → SPEAKING → IDLE
 │
 ├─ (vad)          → LISTENING (capture + VAD active)
 │                 → RECORDING (speech detected)
 │                 → TRANSCRIBING
 │                 → INFERRING
 │                 → SPEAKING → IDLE/LISTENING
 │
 └─ (wake_word)    → PASSIVE_LISTENING (PocketSphinx thread)
                   → (wake word detected) → LISTENING
                   → RECORDING → TRANSCRIBING → INFERRING → SPEAKING
                   → PASSIVE_LISTENING
```

**Barge-in:** `VADDetector::onset_callback` fires during `SPEAKING` state → `AudioDevice::stop_playback()` immediately → state transitions to `INTERRUPTED` → pending segment pushed to work queue → processes new utterance.

**Work queue:** `VoiceLoop` maintains a `std::queue<AudioChunk>` consumed by the main loop thread. The audio capture callback pushes completed segments from the VAD; PTT key-up pushes the accumulated buffer. The main thread processes segments serially: STT → inference → TTS.

### 14.3 AudioDevice

**File:** `src/voice/audio_device.h/.cpp`

PortAudio wrapper. Two independent streams:

**Capture stream:**
- Input: microphone, 16-bit signed PCM, mono, 16 kHz (configurable)
- Callback: `capture_callback()` — fires every `frames_per_buffer` samples, calls `capture_cb_` which routes to VAD and optionally PTT buffer and WakeWordDetector
- `start_capture(callback)` / `stop_capture()`

**Playback stream:**
- Output: speaker, 16-bit signed PCM, mono, device native rate
- Internal `PlaybackChunk` queue consumed by `playback_callback()`
- Linear interpolation resampler applied when TTS sample rate (22050 Hz) ≠ device rate
- `play(samples, sample_rate, on_done)` — enqueues chunk, starts stream if not running
- `stop_playback()` — clears queue immediately, fires all `on_done` callbacks
- `wait_until_done()` — blocks on condition variable until queue drained

**Device enumeration:** `list_devices()` returns all PortAudio devices with channel counts and default sample rates. `input_device=-1` and `output_device=-1` in config select system defaults.

### 14.4 VADDetector

**File:** `src/voice/vad_detector.h/.cpp`

Energy-based RMS threshold detector. No external model.

**State machine:** `IDLE` → `SPEECH` (on energy threshold crossing) → `IDLE` (after post-speech silence).

**Pre-roll buffer:** A `std::deque<int16_t>` ring buffer stores the last `pre_speech_ms` of audio at all times. When speech onset is detected, the pre-roll is prepended to the speech buffer — capturing any clipped leading audio.

**Segment emission rules:**
- `min_speech_ms`: discard segments shorter than this (noise suppression)
- `max_speech_ms`: hard-cap — emit immediately when reached
- `post_speech_ms`: silence window after which a segment is emitted

**Callbacks:**
- `set_segment_callback(SpeechSegmentCallback)` — called with complete `AudioChunk` when segment finalized
- `set_onset_callback(fn)` — called on speech onset (used for barge-in detection)
- `set_offset_callback(fn)` — called on speech offset

**Thread safety:** `push_frame()` is called from the PortAudio callback thread. All state is protected by `mutex_`. Callbacks are invoked with the lock released.

**`compute_rms(samples, n)` → float [0,1]:** static helper. Called per frame: `sqrt(sum(sample²/32768²) / n)`.

### 14.5 STTEngine

**File:** `src/voice/stt_engine.h/.cpp`

Whisper.cpp wrapper.

**Init:** `whisper_init_from_file_with_params()` with `use_gpu=true`, `gpu_device=0`. Uses `whisper_context_params` for CUDA configuration.

**Transcription (`transcribe(AudioChunk)`):**
1. Convert `int16_t` samples to `float32` normalised to [-1, 1]
2. Build `whisper_full_params` with `no_context=true` (each utterance independent), configured `language`, `n_threads`, `beam_size`, optional `initial_prompt`
3. `whisper_full()` — blocks on CUDA inference
4. Collect all segments via `whisper_full_n_segments()` / `whisper_full_get_segment_text()`
5. Trim whitespace, return `TranscriptResult`

**Thread safety:** `transcribe()` acquires `mutex_`. Only one transcription at a time.

**`TranscriptResult`:** `success`, `text`, `confidence`, `duration_ms`, `error_message`. `empty()` returns true for `""`, `"[BLANK_AUDIO]"`, `"(blank)"` — these are discarded by `VoiceLoop`.

### 14.6 TTSEngine

**File:** `src/voice/tts_engine.h/.cpp`

Piper TTS wrapper. `piper.cpp` is compiled directly into Cardinal — no separate piper library required.

**Init:** `piper::initialize(PiperConfig)`, `piper::loadVoice(config, model_path, config_path, voice, speaker_id, use_cuda=false)`. ONNX Runtime CPU inference.

**`synthesise(TTSRequest)` → `TTSResult`:**
1. Apply `length_scale`, `noise_scale`, `noise_w` to `voice.synthesisConfig`
2. `piper::textToAudio(config, voice, text, audio_buffer, result, callback)` — blocks until complete
3. Returns `TTSResult` with PCM samples at `voice.synthesisConfig.sampleRate` (typically 22050 Hz)

**`synthesise_streaming(text, request_template, on_sentence)`:**
1. `split_sentences(text)` → vector of sentence strings
2. For each sentence: `synthesise(req)` → `on_sentence(result)`
3. `VoiceLoop` feeds each result to `AudioDevice::play()` immediately

**`split_sentences(text)` — static:** Splits on `.`, `?`, `!`, `\n`. Minimum 3 words required before emitting. Remainder (partial sentence) emitted as-is. This prevents micro-utterances like "I" or "Hmm" from being synthesised and played as isolated clips.

**Per-request parameters:** `speaker_id`, `length_scale` (speed: <1 faster, >1 slower), `noise_scale`, `noise_w` — all configurable in `config.json` and overridable per-request.

**Thread safety:** `synthesise()` acquires `mutex_`. Only one synthesis at a time.

### 14.7 WakeWordDetector

**File:** `src/voice/wake_word_detector.h/.cpp`

PocketSphinx keyword spotting. Offline, no API key, no network.

**Init:** `ps_config_init()`, set `hmm` (acoustic model path), `dict` (dictionary path), `keyphrase` (phrase string), `kws_threshold` (log probability sensitivity), `samprate`. `ps_init()`.

**Recognition loop (runs on `recog_thread_`):**
1. `ps_start_utt()`
2. Pop frames from `frame_queue_` (bounded ring: max 2 seconds of audio)
3. `ps_process_raw()` — feed 512-sample chunks
4. `ps_get_hyp()` — check for hypothesis containing `config_.phrase`
5. On detection: fire `on_detected_` callback, `ps_end_utt()` / `ps_start_utt()` to reset

**`push_frame(samples, n)`:** called from audio capture callback thread. Acquires `queue_mutex_`, trims overflow, inserts, notifies `queue_cv_`. Non-blocking from caller's perspective.

**`sensitivity`:** Stored as `kws_threshold` log probability. Default `1e-20`. Lower = more sensitive (more false positives); higher = more selective (may miss detections). Tune based on ambient noise.

### 14.8 VoiceLoop Threading

`VoiceLoop` runs two threads:

**`loop_thread_`:** Main voice processing thread. Starts audio capture, enters initial state (IDLE/LISTENING/PASSIVE_LISTENING), then blocks on `work_cv_` waiting for completed `AudioChunk` segments. For each segment: calls `handle_speech_segment()` (STT + inference + TTS) then returns to listening state.

**`ptt_thread_`** (push-to-talk mode only): Reads raw stdin via `termios` raw mode. Space key down → `ptt_key_down()` (sets `ptt_recording_=true`, clears PTT buffer). Space key up → `ptt_key_up()` (moves accumulated PTT buffer to work queue).

**Audio capture callback thread** (PortAudio internal): Routes frames to VAD, WakeWordDetector, and PTT buffer simultaneously. Lock-free paths for low latency — VAD and wake-word use their own mutexes.

### 14.9 Sentence-Streaming TTS

In `sentence` TTS streaming mode, the LLM stream callback accumulates tokens into `sentence_buf`. `TTSEngine::split_sentences()` is called on each token append. When two or more sentences are detectable in the buffer, all but the last (potentially incomplete) are synthesised and played immediately. The incomplete sentence fragment is kept in the buffer and grows with subsequent tokens. On `is_final=true`, all remaining content is flushed through TTS.

This means synthesis of sentence 1 can overlap with LLM generation of sentence 2, achieving sub-400ms first-audio latency on the RTX 3050 for most sentence lengths.

### 14.10 Barge-In

When `is_barge_in_candidate()` returns true (state == `SPEAKING`), the VAD onset callback fires:
1. `audio_->stop_playback()` — immediately clears playback queue
2. `set_state(INTERRUPTED)` — signals the state machine
3. VAD continues to accumulate the new utterance
4. Completed segment pushed to work queue
5. Loop thread picks it up and begins new STT/inference/TTS cycle

Barge-in is transparent to the HTTP API — `VoiceStatus::current_state` will briefly show `"interrupted"` then `"recording"`.

### 14.11 Voice Tool — `voice_control`

**File:** `src/tools/builtin/voice/tool_voice_control.h/.cpp`

Registered when `voice.enabled=true`. Allows the LLM to control its own voice subsystem mid-conversation.

**Actions:**
| Action | Arguments | Effect |
|--------|-----------|--------|
| `set_mode` | `mode`: `"push_to_talk"` \| `"ptt"` \| `"vad"` \| `"wake_word"` | Calls `VoiceLoop::set_input_mode()` |
| `set_voice` | `voice`: model name string | Logs intent; requires restart to take effect |
| `set_volume` | `value`: `"0"`–`"100"` | Informational (system volume via OS) |
| `stop_speaking` | — | Calls `VoiceLoop::stop_speaking()` → `AudioDevice::stop_playback()` |

`VoiceLoop*` is passed to `execute_voice_control()` at dispatch time. If voice is not active, returns a `"Voice subsystem is not active"` error without crashing.

### 14.12 Voice HTTP API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/voice/status` | Returns `VoiceStatus` JSON |
| POST | `/api/voice/enable` | Enables voice; optional `{"input_mode":"vad"}` body |
| POST | `/api/voice/disable` | Disables voice, frees all audio resources |
| POST | `/api/voice/speak` | TTS test: `{"text":"hello"}` |
| POST | `/api/voice/transcribe` | Raw 16-bit PCM body; `X-Sample-Rate` header (default 16000) |

**`VoiceStatus` response:**
```json
{
  "active": true,
  "input_mode": "vad",
  "current_state": "listening",
  "stt_ready": true,
  "tts_ready": true,
  "wake_word_ready": false,
  "session_id": "voice_session",
  "transcriptions": 14,
  "utterances": 11
}
```

### 14.13 Voice Config

See §19.15 for full configuration reference.

**Key design decisions:**
- `voice.enabled=false` by default — audio hardware not touched unless explicitly requested
- `voice.session_id` is a dedicated session for voice inference, separate from any text CLI session — allows concurrent text and voice use without history contamination
- `voice.stt.gpu_layers=8` — Whisper medium.en on RTX 3050 uses 8 GPU layers by default; reduce if OOM with the LLM loaded
- `voice.tts.sample_rate=22050` — Piper's native output rate; PortAudio resamples to device rate
- `voice.audio.sample_rate=16000` — Whisper requires 16 kHz input; PortAudio captures at this rate directly

---

## 15. Agentic Pipeline

Unchanged from v1.2.0. `AgentExecutor::run(AgentGoal, TraceBuilder&, AgentProgressCallback)`.

```
AgentExecutor::run(goal, trace_builder, progress_cb)
  1. PLAN  — planner.decompose(goal) → vector<AgentStep>
  2. EXECUTE LOOP (max_iterations)
     a. THINK — LLM generates action
     b. ACT   — ToolExecutor::execute() → result → working_memory
     c. OBSERVE — check goal_achieved
  3. FINALIZE — synthesize response, finalize trace
```

**v1.5.0:** All computer use tools available within the agentic loop.
**v1.6.0:** `voice_control` tool available within the agentic loop when voice is enabled.

---

## 16. Tools System

### Registered tools (v1.6.0)

| Tool | Registration condition |
|------|------------------------|
| `web_search` | `tools.web_search.enabled` |
| `web_fetch` | `tools.web_fetch.enabled` |
| `calculator` | `tools.calculator.enabled` |
| `run_python` | `tools.run_python.enabled` |
| `file_read` | `tools.file_read.enabled` |
| `file_write` | `tools.file_write.enabled` |
| `knowledge_graph_query` | `tools.knowledge_graph.enabled` |
| `episodic_search` | `tools.episodic_search.enabled` |
| `analyze_image` | `vision.model_path` non-empty |
| `screenshot` through `watch_screen` | `computer_use.enabled` |
| `browser` | `computer_use.enabled` AND `browser.venv_path` non-empty |
| `shell_run` | `computer_use.enabled` AND `shell.enabled` |
| `email` | `computer_use.enabled` AND `email.enabled` |
| `schedule_task` | `scheduler.enabled` |
| `voice_control` | `voice.enabled` ← NEW v1.6.0 |

### ToolExecutor dispatch (v1.6.0 addition)

`voice_control` is dispatched by `ToolExecutor` with a pointer to the active `VoiceLoop`:

```cpp
tool_executor_->set_voice_loop(voice_loop_.get());
```

This setter is called from `CardinalAPI::enable_voice()` after `VoiceLoop::start()` succeeds, and cleared (set to `nullptr`) in `CardinalAPI::disable_voice()`.

---

## 17. Explainability Exports

Unchanged from v1.2.0. Every inference produces a signed, tamper-evident trace.

**v1.4.0 note:** Meta-correction rules appear in `rules_applied` with `reasoning_type = "meta_correction"`.
**v1.5.0 note:** Scheduled task inferences do not produce audit log entries. Computer use tool calls appear in the normal inference trace when invoked from chat or the agentic loop.
**v1.6.0 note:** Voice utterances flow through the normal `chat_stream()` path and therefore produce full audit log entries. The `session_id` is `voice.session_id` (default `"voice_session"`).

---

## 18. Training Export

`TrainingExporter` exports high-confidence episodes as Alpaca JSONL for external use. This is the manual export path. Layer 3 uses `DatasetCurator` directly.

**Filter parameters:** `min_confidence`, `domain`, `max_examples`, `include_rules`, `recent_first`.

---

## 19. Configuration Reference

### 19.1 `backend` (unchanged)

`type` (`"llama_cpp"` or `"tensorrt"`), `llama_cpp.*`, `tensorrt.*`.

### 19.2 `inference` (unchanged)

`temperature`, `top_p`, `max_tokens_feeling`, `max_tokens_response`.

### 19.3 `feeling_schema` (unchanged)

`type`, `grammar_path`, `fields`, `max_tokens`.

### 19.4 `memory` (unchanged)

`rule_store_path`, `knowledge_graph_path`, `episodic_log_path`, `max_rules`.

### 19.5 `verifier` (unchanged)

`mode`, `neural_model_path`, `contradiction_threshold`, `rule_confidence_decay`, `min_rule_confidence`.

### 19.6 `retriever` (unchanged)

`mode`, `keyword_weight`, `semantic_weight`, `max_results`, `min_score`.

### 19.7 `tools` (unchanged)

Per-tool `enabled`, `confirmation_required`, and tool-specific parameters.

### 19.8 `agent` (unchanged)

`enabled`, `max_iterations`, `self_correction_enabled`, `plan_before_execute`.

### 19.9 `explainability` (unchanged)

`enabled`, `audit_log_path`, `signing_enabled`, `private_key_path`, `public_key_path`.

### 19.10 `vision` (v1.3.0, unchanged)

`model_path`, `mmproj_path`, `gpu_layers`, `threads`, `max_tokens`, `cache_path`, `cache_ttl_hours`, `allowed_paths`.

### 19.11 `self_improvement` (v1.4.0, unchanged)

Three sub-blocks: `self_model`, `meta_cognition`, `training`. See v1.4.0 documentation for full reference.

### 19.12 `api` (unchanged)

`http_enabled`, `host`, `port`, `auth_enabled`, `api_key`, `stream_enabled`.

### 19.13 `scheduler` (v1.5.0)

```json
{
  "scheduler": {
    "enabled": true,
    "db_path": "data/scheduler/scheduler.db",
    "check_interval_seconds": 30,
    "max_concurrent_tasks": 1,
    "idle_threshold_minutes": 5,
    "task_session_prefix": "scheduler_",
    "run_history_max_entries": 1000,
    "max_task_duration_seconds": 300
  }
}
```

### 19.14 `computer_use` (v1.5.0)

```json
{
  "computer_use": {
    "enabled": true,
    "safety": {
      "whitelist_enabled": true,
      "allowed_apps": ["google-chrome", "firefox", "nautilus"],
      "allowed_domains": [],
      "allowed_paths": ["~/Documents", "~/Downloads", "data/"],
      "blocked_commands": ["rm -rf /", "mkfs", ":(){:|:&};:"],
      "confirmation_required": true,
      "full_autonomy": false,
      "allow_file_write": false
    },
    "screen": { "screenshot_tool": "auto", "vision_analysis": true },
    "browser": { "venv_path": "~/cardinal/cardinal-browser-venv", "headless": false },
    "shell": { "enabled": true, "timeout_seconds": 30 },
    "email": { "enabled": false, "mode": "imap_smtp" },
    "atspi": { "enabled": true, "fallback_to_vision": true }
  }
}
```

### 19.15 `voice` (v1.6.0) ← NEW

```json
{
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
      "beam_size": 5,
      "initial_prompt": ""
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
}
```

**`voice` top-level keys:**

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Auto-start voice on `CardinalAPI::init()`. Default `false`. |
| `input_mode` | string | `"vad"`, `"push_to_talk"` / `"ptt"`, `"wake_word"`. Default `"vad"`. |
| `tts_streaming` | string | `"sentence"` (low latency) or `"full"` (complete response). Default `"sentence"`. |
| `session_id` | string | Session used for voice inference. Separate from CLI sessions. |

**`stt` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `model_path` | string | Path to Whisper GGML model file |
| `language` | string | Language hint for Whisper (e.g. `"en"`) |
| `gpu_layers` | int | Number of Whisper model layers to offload to GPU |
| `threads` | int | CPU threads for Whisper pre/post processing |
| `beam_size` | int | Beam search width. 1 = greedy (fastest). 5 = default. |
| `initial_prompt` | string | Optional prompt to bias transcription style/vocabulary |

**`tts` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `model_path` | string | Path to Piper `.onnx` voice model |
| `config_path` | string | Path to Piper `.onnx.json` voice config |
| `speaker_id` | int | Multi-speaker model speaker index. 0 for single-speaker. |
| `length_scale` | float | Speed: <1.0 = faster, >1.0 = slower, 1.0 = normal |
| `noise_scale` | float | Prosody variation. 0.667 = default |
| `noise_w` | float | Phoneme duration noise. 0.8 = default |
| `sample_rate` | int | TTS output sample rate (Piper native = 22050) |

**`vad` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `energy_threshold` | float | RMS energy [0–1] above which speech is detected. Lower = more sensitive. |
| `pre_speech_ms` | int | Pre-roll buffer duration to prepend on onset |
| `post_speech_ms` | int | Silence duration after which segment is emitted |
| `min_speech_ms` | int | Minimum speech duration; shorter segments discarded |
| `max_speech_ms` | int | Hard cap; segment emitted immediately when reached |

**`audio` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `input_device` | int | PortAudio device index for mic. -1 = system default |
| `output_device` | int | PortAudio device index for speakers. -1 = system default |
| `sample_rate` | int | Capture sample rate. Must be 16000 for Whisper. |
| `channels` | int | Capture channels. Must be 1 (mono). |
| `frames_per_buffer` | int | PortAudio callback frame count. 512 default. |

---

## 20. CardinalAPI Reference

### 20.1 Lifecycle

```cpp
CardinalVoidResult init(const std::string& config_path = "config.json");
CardinalVoidResult shutdown();
```

### 20.2 Session Management (unchanged)

`create_session()`, `destroy_session()`, `reset_session()`, `get_session()`, `list_sessions()`.

### 20.3 Inference (unchanged)

```cpp
CardinalResult<ChatResponse> chat(const std::string& session_id, const std::string& message);
CardinalResult<ChatResponse> chat_stream(const std::string& session_id,
                                          const std::string& message,
                                          const ApiStreamCallback& stream_cb);
```

### 20.4 Agentic Inference (unchanged)

`agent(session_id, goal, max_iterations)`.

### 20.5 Memory & Stats (unchanged)

`get_stats()`, `get_rules()`, `get_episodes()`, `run_scan()`, `run_maintenance()`.

### 20.6 Self-Improvement (v1.4.0, unchanged)

`get_self_model_status()`, `reflect()`, `trigger_training(domain_hint)`, `on_session_boundary()`.

### 20.7 Scheduler API (v1.5.0, unchanged)

`get_scheduler_status()`, `list_tasks()`, `get_task()`, `create_task()`, `create_task_direct()`, `update_task()`, `delete_task()`, `enable_task()`, `disable_task()`, `run_task_now()`, `get_task_history()`, `get_recent_runs()`, `get_run_action_logs()`.

### 20.8 Computer Use API (v1.5.0, unchanged)

`get_computer_status()`, `take_screenshot()`, `computer_click()`, `computer_type()`, `computer_shell()`.

### 20.9 Voice API (v1.6.0) ← NEW

```cpp
// Enable the voice subsystem. input_mode overrides config if non-empty.
CardinalVoidResult               enable_voice(const std::string& input_mode = "");

// Disable voice subsystem and free all audio resources.
CardinalVoidResult               disable_voice();

// Returns true if VoiceLoop is running.
bool                             is_voice_active() const;

// Snapshot of current voice state.
VoiceStatus                      get_voice_status() const;

// Directly synthesise and play text via TTS (bypasses inference).
CardinalResult<TTSResult>        voice_speak(const std::string& text);

// Directly transcribe an AudioChunk (for HTTP multipart upload).
CardinalResult<TranscriptResult> voice_transcribe(const AudioChunk& audio);
```

`enable_voice()` constructs a `VoiceChatStreamFn` lambda that calls `chat_stream()` — this is the bridge between the voice loop's LLM calls and the full inference pipeline (two-pass, verifier, memory, self-improvement, all hooks). The voice loop has no direct dependency on `CardinalAPI`.

---

## 21. HTTP API Reference

Base URL: `http://127.0.0.1:8080`
Auth: `Authorization: Bearer <api_key>` (except `/api/health`)

### Existing endpoints (unchanged)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check (no auth) |
| POST | `/api/chat` | Chat with optional SSE stream |
| POST | `/api/sessions` | Create session |
| DELETE | `/api/sessions/:id` | Destroy session |
| POST | `/api/sessions/:id/reset` | Reset session history |
| GET | `/api/stats` | System statistics |
| GET | `/api/rules` | Rule store contents |
| GET | `/api/episodes` | Episode query |
| POST | `/api/scan` | Contradiction scan |
| POST | `/api/maintenance` | Maintenance cycle |
| GET | `/api/settings` | Get settings |
| POST | `/api/settings` | Update settings |
| POST | `/api/export` | Export training data |
| GET | `/api/self_model` | Self-model status |
| POST | `/api/reflect` | Trigger reflection |
| POST | `/api/train` | Trigger training |

### Scheduler endpoints (v1.5.0, unchanged)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scheduler/status` | Engine status |
| GET | `/api/scheduler/tasks` | List all tasks |
| POST | `/api/scheduler/tasks` | Create task |
| GET | `/api/scheduler/tasks/:id` | Get task |
| PUT | `/api/scheduler/tasks/:id` | Update task |
| DELETE | `/api/scheduler/tasks/:id` | Delete task |
| POST | `/api/scheduler/tasks/:id/run` | Run immediately |
| POST | `/api/scheduler/tasks/:id/enable` | Enable |
| POST | `/api/scheduler/tasks/:id/disable` | Disable |
| GET | `/api/scheduler/tasks/:id/history` | Run history |
| GET | `/api/scheduler/runs` | Recent runs |
| GET | `/api/scheduler/runs/:id/actions` | Action log |

### Computer Use endpoints (v1.5.0, unchanged)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/computer/status` | Display info |
| POST | `/api/computer/screenshot` | Capture screen |
| POST | `/api/computer/click` | Click |
| POST | `/api/computer/type` | Type |
| POST | `/api/computer/shell` | Shell command |

### Voice endpoints (v1.6.0) ← NEW

| Method | Endpoint | Body / Notes | Description |
|--------|----------|--------------|-------------|
| GET | `/api/voice/status` | — | Returns `VoiceStatus` |
| POST | `/api/voice/enable` | `{"input_mode":"vad"}` (optional) | Enable voice |
| POST | `/api/voice/disable` | — | Disable voice |
| POST | `/api/voice/speak` | `{"text":"hello"}` | TTS synthesis + playback |
| POST | `/api/voice/transcribe` | Raw 16-bit PCM body; `X-Sample-Rate` header | STT transcription |

**Enable example:**
```bash
curl -X POST http://127.0.0.1:8080/api/voice/enable \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"input_mode": "vad"}'
```

**Speak example:**
```bash
curl -X POST http://127.0.0.1:8080/api/voice/speak \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am Cardinal."}'
```

**Response:**
```json
{"ok": true, "duration_ms": 312, "sample_rate": 22050, "samples": 6878}
```

**Transcribe example (raw PCM file):**
```bash
# Record 5 seconds at 16kHz mono with arecord:
arecord -f S16_LE -r 16000 -c 1 -d 5 audio.raw

curl -X POST http://127.0.0.1:8080/api/voice/transcribe \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Sample-Rate: 16000" \
  --data-binary @audio.raw
```

**Response:**
```json
{"ok": true, "text": "what is the weather today", "success": true, "confidence": 1.0, "duration_ms": 843}
```

---

## 22. Settings Manager (unchanged)

Runtime-mutable settings propagate immediately. Manages retriever mode/weights, verifier mode, rule thresholds, sampling parameters, log level, agent limits.

---

## 23. Session Manager (unchanged)

`SessionManager` owns `ConversationSession` objects. Each tracks `turn_count`, `history`, `working_memory`, timestamps. `destroy_session()` calls `on_session_boundary()` for LoRA adapter application.

---

## 24. Type Reference

All public types in `src/api/cardinal_types.h`.

### New types (v1.6.0) — `src/voice/voice_types.h`

**`VoiceInputMode` enum:** `PUSH_TO_TALK`, `VAD`, `WAKE_WORD`

**`TTSStreamingMode` enum:** `SENTENCE`, `FULL`

**`VoiceLoopState` enum:** `IDLE`, `PASSIVE_LISTENING`, `LISTENING`, `RECORDING`, `TRANSCRIBING`, `INFERRING`, `SPEAKING`, `INTERRUPTED`, `STOPPING`

**`AudioChunk`:**
```cpp
struct AudioChunk {
    std::vector<int16_t> samples;
    int                  sample_rate = 16000;
    int                  channels    = 1;
    float duration_ms() const;
    bool  empty()       const;
};
```

**`TranscriptResult`:**
```cpp
struct TranscriptResult {
    bool        success        = false;
    std::string text;
    float       confidence     = 0.0f;
    int         duration_ms    = 0;
    std::string error_message;
    bool        empty()        const;  // true for blank/silence
};
```

**`TTSRequest`:**
```cpp
struct TTSRequest {
    std::string text;
    int         speaker_id   = 0;
    float       length_scale = 1.0f;
    float       noise_scale  = 0.667f;
    float       noise_w      = 0.8f;
};
```

**`TTSResult`:**
```cpp
struct TTSResult {
    bool                 success      = false;
    std::vector<int16_t> samples;
    int                  sample_rate  = 22050;
    int                  duration_ms  = 0;
    std::string          error_message;
};
```

**`VADResult`:**
```cpp
struct VADResult {
    bool  speech_detected = false;
    float rms             = 0.0f;
    float threshold       = 0.0f;
};
```

**`WakeWordResult`:**
```cpp
struct WakeWordResult {
    bool        detected = false;
    std::string phrase;
    float       score    = 0.0f;
};
```

**`VoiceStatus`:**
```cpp
struct VoiceStatus {
    bool        active          = false;
    std::string input_mode;
    std::string current_state;
    bool        stt_ready       = false;
    bool        tts_ready       = false;
    bool        wake_word_ready = false;
    std::string session_id;
    int         transcriptions  = 0;
    int         utterances      = 0;
};
```

**Config structs:** `VoiceSTTConfig`, `VoiceTTSConfig`, `VoiceVADConfig`, `VoicePTTConfig`, `VoiceWakeWordConfig`, `VoiceAudioConfig`, `VoiceConfig` — all in `voice_types.h`, included by `config_loader.h`.

**Callbacks:** `TranscriptCallback`, `WakeWordCallback`, `VoiceStateCallback`, `BargeInCallback`, `SpeechSegmentCallback`, `TTSSentenceCallback`, `VoiceChatStreamFn`.

### Types from v1.5.0 (unchanged)

`TriggerType`, `TriggerSpec`, `TaskActionType`, `OutputTarget`, `TaskAction`, `ScheduledTask`, `TaskRunStatus`, `TaskRun`, `TaskActionLog`, `TaskParseResult`, `SchedulerStatus`, `DisplayServer`, `Point`, `ScreenRegion`, `ScreenInfo`, `Screenshot`, `MouseButton`, `BrowserActionType`, `BrowserResult`, `ShellResult`, `FileOpResult`, `FileEntry`, `SystemState`, `EmailQuery`, `EmailMessage`, `EmailSendRequest`, `AppInfo`, `AtSpiNode`.

### Types from v1.4.0 and earlier (unchanged)

`CardinalStatus`, `CardinalResult<T>`, `CardinalVoidResult`, `ChatResponse`, `FeelingInfo`, `SessionInfo`, `RuleInfo`, `EpisodeInfo`, `SystemStats`, `ExportInfo`, `ScanResult`, `StreamToken`, `DomainStats`, `SelfModelSnapshot`, `ReflectionFinding`, `ReflectionResult`, `SelfImprovementStatus`, `TrainingExample`.

---

## 25. Error Handling

No C++ exceptions cross the `CardinalAPI` boundary. All internal exceptions caught and converted to `CardinalResult<T>` or `CardinalVoidResult`.

**Non-fatal paths:** retrieval failure, neural verifier failure, tool execution failure, reflection pass LLM failure, training subprocess non-zero exit, adapter load failure below threshold, screenshot tool missing, browser process crash, shell command blocked, file path not allowed, email auth failure, STT blank result, TTS synthesis error.

**Fatal paths** (cause `init()` to fail): missing config/model/grammar files, SQLite error on episode DB open, explainability key generation failure.

**v1.5.0 — Graceful degradation:**
- `computer_use.enabled=true` but display not detected → computer use not initialised; `check_computer_use()` returns `COMPUTER_USE_ERROR`
- `scheduler.enabled=true` but SQLite fails → scheduler not started
- Browser venv missing → `browser` tool not registered

**v1.6.0 — Voice graceful degradation:**
- `voice.enabled=true` in config but `STTEngine::init()` fails (model file missing) → warning logged, voice not started, system continues in text-only mode
- `voice.enabled=true` but `AudioDevice::init()` fails (no audio hardware) → same — non-fatal
- `--voice` flag set but voice fails to start → warning printed to stderr, falls back to text-only mode
- `VoiceLoop` already running when `enable_voice()` called → returns success immediately (idempotent)
- Voice active, `check_voice()` passes, but STT returns blank transcript → segment discarded silently, returns to LISTENING

**`CardinalStatus` codes (v1.6.0 addition):**

`VOICE_ERROR = 17` — returned by `check_voice()` when voice subsystem is not active, and by `voice_speak()` / `voice_transcribe()` on failure.

---

## 26. Offline Builds & Vendoring

All vendor dependencies cloned manually — no git submodules.

```
vendor/
    llama.cpp/          git clone https://github.com/ggerganov/llama.cpp
    nlohmann_json/      git clone https://github.com/nlohmann/json
    cpp-httplib/        git clone https://github.com/yhirose/cpp-httplib
    muparser/           git clone https://github.com/beltoforion/muparser
    tokenizers-cpp/     git clone https://github.com/mlc-ai/tokenizers-cpp
    whisper.cpp/        git clone https://github.com/ggerganov/whisper.cpp      ← v1.6.0
    piper/              git clone https://github.com/rhasspy/piper               ← v1.6.0
        install/        pre-built via cmake --target install
    portaudio/          git clone https://github.com/PortAudio/portaudio         ← v1.6.0
    pocketsphinx/       git clone https://github.com/cmusphinx/pocketsphinx      ← v1.6.0
        install/        pre-built via cmake --target install
```

**CMake integration strategy (v1.6.0):**

| Library | Strategy | Reason |
|---------|----------|--------|
| whisper.cpp | `add_subdirectory` | Has a proper library CMake target (`whisper`) |
| portaudio | `add_subdirectory` | Has a proper library CMake target (`portaudio`) |
| pocketsphinx | Pre-built into `install/` | `add_subdirectory` breaks: sources `#include <pocketsphinx.h>` which only exists after the build generates it |
| piper | Pre-built into `install/` + `piper.cpp` compiled directly into Cardinal | Builds an executable, not a library; uses `ExternalProject_Add` for its deps |

**Piper's flat install layout:** Piper installs all `.so` files directly into the install prefix root (not `lib/`). CMake `find_library` calls use `PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH` to handle this. Runtime `LD_LIBRARY_PATH` must include `vendor/piper/install/`.

---

## 27. Module Reference

### 27.1 VisionEncoder (v1.3.0)

```cpp
class VisionEncoder {
    void load();
    bool is_ready() const;
    void unload();
    VisionResult encode(const std::string& image_path,
                        const std::string& prompt,
                        const ImageMetadata& metadata) const;
};
```

### 27.2 SelfImprovementLoop (v1.4.0)

```cpp
class SelfImprovementLoop {
    void start();
    void stop();
    void on_inference(domain, reasoning_type, confidence,
                      contradiction, uncertainty, rule_committed);
    void on_session_boundary();
    ReflectionResult trigger_reflection();
    bool trigger_training(const std::string& domain_hint = "");
    std::string format_self_model_for_prompt() const;
    SelfImprovementStatus get_status() const;
};
```

### 27.3 SchedulerEngine (v1.5.0)

```cpp
class SchedulerEngine {
    void start();
    void stop();
    bool is_running() const;
    void on_inference();

    TaskParseResult create_task_from_nl(const std::string& nl,
                                         const std::string& session_id = "");
    std::string     create_task(ScheduledTask task);
    bool            update_task(const ScheduledTask& task);
    bool            delete_task(const std::string& task_id);
    std::optional<ScheduledTask> get_task(const std::string& task_id) const;
    std::vector<ScheduledTask>   list_tasks() const;
    bool            enable_task(const std::string& task_id);
    bool            disable_task(const std::string& task_id);
    std::string     run_task_now(const std::string& task_id);
    std::vector<TaskRun>       get_task_history(const std::string& task_id, int limit) const;
    std::vector<TaskRun>       get_recent_runs(int limit) const;
    std::vector<TaskActionLog> get_run_action_logs(const std::string& run_id) const;
    SchedulerStatus get_status() const;
};
```

### 27.4 SchedulerParser (v1.5.0)

```cpp
class SchedulerParser {
    explicit SchedulerParser(InferencePipeline* pipeline,
                              const CardinalConfig& config,
                              float min_confidence = 0.70f);
    TaskParseResult parse(const std::string& nl_description,
                          const std::string& session_id = "");
    TaskParseResult parse_json(const std::string& json_str,
                               const std::string& session_id = "");
    static TaskParseResult extract_from_json(const std::string& model_output,
                                              const std::string& session_id,
                                              float              min_confidence);
    static std::string build_system_prompt();
    static std::string build_user_message(const std::string& nl_description);
};
```

### 27.5 DisplayDetector (v1.5.0)

```cpp
class DisplayDetector {
    void detect();
    bool is_x11()      const;
    bool is_wayland()  const;
    bool is_headless() const;
    DisplayServer server() const;
    ScreenInfo    info()   const;
    bool has_scrot() const;
    bool has_grim()  const;
};
```

### 27.6 BrowserController (v1.5.0)

```cpp
class BrowserController {
    bool start();
    void stop();
    bool is_running() const;
    BrowserResult execute(const BrowserAction& action);
    BrowserResult navigate(const std::string& url);
    BrowserResult click(const std::string& selector);
    BrowserResult type(const std::string& selector, const std::string& text);
    BrowserResult get_content();
    BrowserResult screenshot(const std::string& save_path = "");
    BrowserResult execute_js(const std::string& js);
    bool is_domain_allowed(const std::string& url) const;
};
```

### 27.7 VoiceLoop (v1.6.0) ← NEW

```cpp
class VoiceLoop {
    VoiceLoop(const VoiceConfig& config, VoiceChatStreamFn chat_fn);

    bool start();  // init all components, start loop thread
    void stop();   // graceful shutdown
    bool is_running() const;

    // Runtime control
    void           set_input_mode(VoiceInputMode mode);
    VoiceInputMode input_mode() const;

    // Direct TTS (bypasses inference)
    TTSResult        speak(const std::string& text);

    // Direct STT (for HTTP API)
    TranscriptResult transcribe(const AudioChunk& audio);

    // Stop current playback immediately (barge-in or external call)
    void stop_speaking();

    // State query
    VoiceLoopState state()      const;
    VoiceStatus    get_status() const;

    // Callbacks (set before start())
    void set_state_callback(VoiceStateCallback cb);
    void set_transcript_callback(TranscriptCallback cb);
};
```

### 27.8 AudioDevice (v1.6.0) ← NEW

```cpp
class AudioDevice {
    explicit AudioDevice(const VoiceAudioConfig& config);

    bool init();
    void shutdown();
    bool is_ready() const;

    bool start_capture(std::function<void(const int16_t*, int)> callback);
    void stop_capture();
    bool is_capturing() const;

    void play(const std::vector<int16_t>& samples, int sample_rate,
              std::function<void()> on_done = nullptr);
    void stop_playback();
    bool is_playing() const;
    void wait_until_done();

    std::vector<DeviceInfo> list_devices() const;
    int default_input_device()  const;
    int default_output_device() const;
};
```

### 27.9 STTEngine (v1.6.0) ← NEW

```cpp
class STTEngine {
    explicit STTEngine(const VoiceSTTConfig& config);
    bool init();
    void shutdown();
    bool is_ready() const;
    TranscriptResult transcribe(const AudioChunk& audio);
};
```

### 27.10 TTSEngine (v1.6.0) ← NEW

```cpp
class TTSEngine {
    explicit TTSEngine(const VoiceTTSConfig& config);
    bool init();
    void shutdown();
    bool is_ready() const;
    TTSResult        synthesise(const TTSRequest& request);
    void             synthesise_streaming(const std::string& text,
                                          const TTSRequest& request_template,
                                          TTSSentenceCallback on_sentence);
    static std::vector<std::string> split_sentences(const std::string& text);
};
```

### 27.11 VADDetector (v1.6.0) ← NEW

```cpp
class VADDetector {
    explicit VADDetector(const VoiceVADConfig& config, int sample_rate);
    void push_frame(const int16_t* samples, int num_samples);
    void set_segment_callback(SpeechSegmentCallback cb);
    void set_onset_callback(std::function<void()> cb);
    void set_offset_callback(std::function<void()> cb);
    bool is_speech_active() const;
    static float compute_rms(const int16_t* samples, int num_samples);
    void reset();
};
```

### 27.12 WakeWordDetector (v1.6.0) ← NEW

```cpp
class WakeWordDetector {
    explicit WakeWordDetector(const VoiceWakeWordConfig& config, int sample_rate);
    bool init();
    void shutdown();
    bool is_ready() const;
    bool start(WakeWordCallback on_detected);
    void stop();
    bool is_listening() const;
    void push_frame(const int16_t* samples, int num_samples);
};
```

---

## 28. Threading Model

| Thread | Owner | Responsibility |
|--------|-------|----------------|
| Main / inference | CardinalAPI | `chat()`, `agent()`, all user-facing calls |
| HTTP server | HttpServer | Request handling, SSE streaming |
| Training | SelfImprovementLoop | Layer 3 training cycle (background) |
| Scheduler engine | SchedulerEngine | Trigger evaluation + task dispatch |
| Task execution | SchedulerEngine (detached) | Each task runs in a detached thread |
| Voice loop | VoiceLoop | STT → inference → TTS pipeline ← v1.6.0 |
| Push-to-talk | VoiceLoop | Stdin raw-mode keyboard reader ← v1.6.0 |
| Wake word | WakeWordDetector | PocketSphinx recognition loop ← v1.6.0 |
| Audio callbacks | PortAudio (internal) | Capture and playback callbacks ← v1.6.0 |

### Lock ordering

```
api_mutex_
  > voice_mutex_                          ← v1.6.0
  > session_mutex_
  > inference_mutex_
      > self_model.mutex_
      > meta_cognition.reflect_mutex_
      > training_mutex_ > trainer_mutex_ > pending_mutex_
  > scheduler cv_mutex_
      > scheduler idle_mutex_
      > scheduler status_mutex_
  > VoiceLoop state_mutex_               ← v1.6.0
      > VADDetector mutex_
      > WakeWordDetector queue_mutex_
      > AudioDevice playback_mutex_
```

### Voice thread safety notes

- `AudioDevice::capture_callback()` runs on PortAudio's audio thread. It must not block. All paths from the callback (VAD push_frame, PTT buffer insert, WakeWordDetector push_frame) use their own mutexes and return immediately.
- `VADDetector::push_frame()` acquires its own mutex and may call callbacks (onset, offset, segment) with the lock released to avoid deadlock.
- `WakeWordDetector::push_frame()` acquires `queue_mutex_` for a bounded insert and notifies the recognition thread. Non-blocking.
- `VoiceLoop::loop_thread_` blocks on `work_cv_` and processes segments serially. STT and TTS are both called on this thread under their respective internal mutexes.
- `CardinalAPI::voice_mutex_` guards `voice_loop_` pointer access. All public voice API methods acquire it briefly.

### Scheduler thread safety (v1.5.0, unchanged)

`SchedulerEngine::on_inference()` only acquires `idle_mutex_` briefly. Task dispatch goes to a detached thread, never blocking inference.

---

## 29. Lifecycle and Startup Sequence (v1.6.0)

```
main(argc, argv)
  |
  +-- Parse --voice / --voice=MODE flag
  +-- Logger::init()
  +-- CardinalAPI::init("config.json")
        |
        +-- ILLMBackend → load_model(), create 2 contexts
        +-- Memory subsystems (RuleStore, KG, Episodic, Retriever)
        +-- Verifier pipeline (SymbolicEngine, RuleExtractor, Checker)
        +-- InferencePipeline
        +-- Tools & Agent (ToolRegistry, ToolExecutor, AgentExecutor)
        +-- Vision (VisionCache, VisionEncoder)
        +-- Explainability (AuditLog, key generation)
        +-- API layer (TrainingExporter, SettingsManager, SessionManager)
        +-- Self-Improvement (SelfImprovementLoop::start())
        |
        +-- Scheduler (v1.5.0)
        |     +-- SchedulerEngine::start()
        |     +-- ToolRegistry registers schedule_task
        |
        +-- Computer Use (v1.5.0)
        |     +-- DisplayDetector::detect()
        |     +-- ScreenReader, InputController, AppController, ...
        |     +-- ToolRegistry registers screenshot, click, ...
        |     +-- ToolExecutor wired via setters
        |
        +-- Voice (v1.6.0)  ← NEW
              +-- if voice.enabled: enable_voice()
                    +-- VoiceLoop constructed with VoiceChatStreamFn lambda
                    +-- VoiceLoop::start()
                          +-- AudioDevice::init()  → Pa_Initialize()
                          +-- VADDetector constructed
                          +-- STTEngine::init()    → whisper_init_from_file_with_params()
                          +-- TTSEngine::init()    → piper::initialize() + loadVoice()
                          +-- WakeWordDetector::init() (if wake_word mode)
                          +-- loop_thread_.start()
                          +-- ptt_thread_.start()  (if push_to_talk mode)
                    +-- ToolRegistry registers voice_control
                    +-- ToolExecutor::set_voice_loop(voice_loop_.get())
        |
        +-- initialized_.store(true)

  +-- if --voice flag and !voice.enabled: enable_voice(mode_override)
  +-- HttpServer::start() (if http_enabled)
  +-- interactive loop
```

**Shutdown:**
```
main: api.disable_voice()      ← if --voice flag was used
main: http_server.stop()

CardinalAPI::shutdown()
  +-- voice_loop_->stop()      ← v1.6.0 — first, frees audio hardware
        +-- stop_requested_.store(true)
        +-- work_cv_.notify_all()
        +-- AudioDevice::stop_playback(), stop_capture()
        +-- WakeWordDetector::stop()
        +-- loop_thread_.join()
        +-- ptt_thread_.join()
        +-- STTEngine::shutdown()    → whisper_free()
        +-- TTSEngine::shutdown()    → piper::terminate()
        +-- AudioDevice::shutdown()  → Pa_Terminate()
  |
  +-- scheduler_->stop()
  +-- browser_controller_->stop()
  +-- self_improvement_->stop()
  +-- sessions_->destroy_all()
  +-- EpisodicStorage::close()
  +-- AuditLog::close()
  +-- RuleStore::save()
```

**Destruction order** is guaranteed by `unique_ptr` member declaration order in `CardinalAPI`. `voice_loop_` is declared after computer use controllers and before API layer members. `shutdown()` explicitly stops voice before all other subsystems to release audio hardware early.

---

*This documentation reflects Cardinal v1.6.0. The source of truth is always the source code.*
