# Cardinal v2.0.0

**A production-grade neurosymbolic AGI architecture with self-improvement, autonomous scheduling, full desktop computer use, and a complete voice subsystem.**

Cardinal combines a large language model core with **vision understanding**, symbolic verification, persistent memory, hybrid retrieval, a **full agentic loop**, **explainability exports**, a **three-layer self-improvement system**, a **natural-language task scheduler**, a complete **computer use subsystem** that can operate your desktop, and a **full voice subsystem** with STT, TTS, wake-word detection, VAD, and barge-in. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

**Current version: v2.0.0** — Quality checks and proprietary tests. All subsystems from v1.0.0 through v1.6.0 are included and stable.

---

## What Cardinal Is

Most LLM systems are stateless. Cardinal is not. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, improves its own weights from accumulated experience, operates your computer autonomously, schedules tasks to run while you sleep, and speaks and listens.

The architecture is **neurosymbolic** — combining the pattern-matching strength of neural inference with the consistency guarantees of symbolic logic. Cardinal adds autonomic execution, voice, and self-improvement on top of that foundation.

---

## Version History

| Version | Description |
|---------|-------------|
| v1.0.0 | Core AGI with two-pass inference, memory, symbolic verification, HTTP API |
| v1.1.0 | Backend abstraction (llama.cpp / TensorRT), Linux native, offline builds |
| v1.2.0 | Explainer, native tools, agentic loop, unified pipeline, explainability exports |
| v1.3.0 | Native vision encoder (moondream2), `analyze_image` tool, image cache |
| v1.4.0 | SEAL self-improvement: self-model, meta-cognition, LoRA fine-tuning |
| v1.5.0 | Scheduler + Computer Use + Watch subsystem |
| v1.6.0 | Voice subsystem: STT (Whisper CUDA), TTS (Piper), VAD, wake word, barge-in |
| **v2.0.0** | **Quality checks and proprietary tests** |

---

## Hardware

Cardinal was developed and runs on:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~8GB core + models |
| OS | Ubuntu 24.04 LTS |

**Models:**
- Primary LLM: `Qwen3.5 4B Q4_K_M` (llama.cpp or TensorRT backend)
- Vision encoder: `moondream2` text model + mmproj (quantized, CPU)
- STT: `ggml-medium.en.bin` (Whisper, CUDA)
- TTS: `en_US-lessac-medium.onnx` + `.json` (Piper, ONNX CPU)
- Wake word: PocketSphinx acoustic model + CMUdict (en-US)
- Fine-tuning: HuggingFace weights at `models/qwen3.5-4b-hf/` + Python venv with PEFT

---

## Architecture

```
CardinalAPI
    |
    +-- Memory Layer
    |     +-- RuleStore          persistent rule base, confidence decay, Jaccard dedup
    |     +-- KnowledgeGraph     typed nodes, BFS traversal, hub detection
    |     +-- EpisodicMemory     append-only JSONL audit trail
    |     +-- EpisodicStorage    SQLite + FTS5 searchable index, JSONL migration
    |     +-- EpisodicRetriever  TF-IDF cosine + FTS5 keyword + hybrid retrieval
    |
    +-- Verifier Pipeline
    |     +-- SymbolicEngine     SWI-Prolog 9.2.9 integration, contradiction detection
    |     +-- NeuralVerifier     optional small LLM verifier (Llama 3.2 1B)
    |     +-- RuleExtractor      NLP extraction, causal/deductive/declarative patterns
    |     +-- ConsistencyChecker orchestrator, auto-resolution, maintenance cycles
    |
    +-- LLM Core (Backend Abstraction)
    |     +-- LLMEngine          Abstract interface: llama.cpp or TensorRT
    |     +-- llama.cpp backend  Development / CPU / CUDA fallback
    |     +-- TensorRT backend   Optimized deployment on NVIDIA GPUs
    |     +-- InferencePipeline  two-pass orchestrator, prompt injection, retry logic
    |
    +-- Vision Subsystem (v1.3.0)
    |     +-- VisionEncoder      moondream2 wrapper, uses mtmd API
    |     +-- VisionCache        URL download cache, TTL eviction
    |
    +-- Self-Improvement Subsystem (v1.4.0)
    |     +-- SelfImprovementLoop   orchestrator, background training thread
    |     Layer 1: SelfModel        symbolic self-knowledge, SQLite accumulator
    |     Layer 2: MetaCognition    reflection pass, corrective rule generation
    |     Layer 3: Training Pipeline
    |         +-- CurriculumBuilder   domain weakness scoring, recency bias
    |         +-- DatasetCurator      episode → TrainingExample, rule augmentation
    |         +-- ITrainingBackend    abstract interface
    |         +-- LlamaCppTrainer     PEFT subprocess → GGUF → llama_set_adapters_lora
    |         +-- TensorRTTrainer     script-export mode for cluster deployment
    |         +-- AdapterEvaluator    holdout eval, improvement threshold gate
    |
    +-- Scheduler (v1.5.0)
    |     +-- SchedulerEngine    background thread, trigger dispatch
    |     +-- SchedulerParser    NL → ScheduledTask via InferencePipeline
    |     +-- SchedulerStore     SQLite WAL: tasks, runs, action_logs
    |
    +-- Computer Use (v1.5.0)
    |     +-- DisplayDetector    X11/Wayland/headless at runtime
    |     +-- ScreenReader       scrot/grim + VisionEncoder analysis
    |     +-- InputController    xdotool/ydotool+wtype
    |     +-- AppController      wmctrl/swaymsg
    |     +-- BrowserController  Playwright subprocess
    |     +-- ShellExecutor      sandboxed subprocess
    |     +-- FileManager        allowed_paths enforced
    |     +-- SystemController   pactl/brightnessctl/nmcli/bluetoothctl
    |     +-- EmailController    IMAP/SMTP or Gmail REST API
    |     +-- AtSpiReader        pyatspi subprocess
    |
    +-- Watch Subsystem (v1.5.0)
    |     +-- FileWatcher        inotify-based file system event monitoring
    |     +-- ScreenWatcher      periodic screenshot diff using ImageMagick PSNR
    |     +-- ProcessWatcher     /proc polling for process start/stop events
    |
    +-- Voice Subsystem (v1.6.0)
    |     +-- VoiceLoop          state machine, owns all voice components
    |           +-- AudioDevice        PortAudio: capture + playback streams
    |           +-- VADDetector        energy RMS, pre-roll, barge-in
    |           +-- STTEngine          whisper.cpp, CUDA
    |           +-- TTSEngine          Piper, ONNX CPU, sentence streaming
    |           +-- WakeWordDetector   PocketSphinx, keyword spotting
    |
    +-- Agentic Pipeline
    |     +-- AgentExecutor      PLAN → EXECUTE loop (THINK → ACT → OBSERVE) → FINALIZE
    |     +-- ToolExecutor       sandboxed tool execution (subprocess or Docker)
    |     +-- WorkingMemory      SQLite-backed persistent scratchpad
    |     +-- SelfCorrection     retry failed steps, max attempts configurable
    |
    +-- Explainability
    |     +-- AuditLog           every inference trace: feeling, tools, rules, symbolic checks
    |     +-- Cryptographic signing  SHA256 + Ed25519 (tamper-evident)
    |     +-- ExplainabilityExporter JSON exports for compliance
    |
    +-- API Layer
          +-- CardinalAPI        single facade, no exceptions at boundary
          +-- HttpServer         SSE streaming, Bearer auth, CORS, TypeScript bridge
          +-- SessionManager     multi-session conversation state
          +-- CardinalSettings   runtime-mutable config, immediate propagation
          +-- TrainingExporter   Alpaca JSONL export for LoRA fine-tuning
```

---

## Two-Pass Inference

Every inference runs in two passes.

**Pass 1 — Feeling Output (constrained decoding)**

GBNF grammar forces the model to produce a structured JSON object before generating any response. This is Cardinal's introspective state — it cannot be skipped or faked.

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

**Pass 2 — Response (free decoding)**

The feeling output is injected as a synthetic assistant turn. The model generates its final response with full awareness of its own internal state from Pass 1.

This is not a prompt trick. The feeling output is inserted as the model's own prior thought — it reads it as something it already said, not as an instruction.

In voice mode, the streamed Pass 2 tokens are simultaneously fed to the TTS sentence splitter, so speech begins within milliseconds of the first complete sentence.

After every inference, the feeling output is also fed into `SelfImprovementLoop::on_inference()`. Layer 1 records per-domain stats. Layer 2 checks whether a reflection pass should fire. Layer 3 checks episode count and confidence thresholds.

---

## Memory Systems

### Dual-Write Pattern

Every inference is written to two stores simultaneously:

- **EpisodicMemory** — append-only JSONL. The audit trail. Never modified after write.
- **EpisodicStorage** — SQLite with FTS5. The searchable index. Supports keyword and semantic queries.

### JSONL Migration

On first startup, EpisodicStorage reads the existing JSONL file and imports all episodes into SQLite. This runs once and is idempotent — a metadata flag prevents re-migration.

### Retrieval

Before each inference, the retriever queries past episodes for relevance to the current user message. Relevant episodes are injected into the prompt as context between the system prompt and the conversation history.

Three retrieval modes:

| Mode | Mechanism | When to use |
|------|-----------|-------------|
| `keyword` | SQLite FTS5 | Exact or near-exact matches, lowest latency |
| `semantic` | TF-IDF cosine similarity | Paraphrased or conceptually similar queries |
| `hybrid` | Weighted combination | Default, best overall performance |

The TF-IDF index is built over `user_message + response_summary` for all episodes. It is cached in memory and rebuilt according to `cache_rebuild_strategy`:

- `on_demand` — rebuild when corpus grows by `cache_rebuild_threshold` episodes
- `periodic` — rebuild every `cache_rebuild_interval_seconds` seconds
- `explicit` — rebuild only when `rebuild_index()` is called directly

---

## Rule System

### Extraction

When `rule_candidate_signal` is true in the feeling output, the RuleExtractor attempts to derive a rule from the response text. Three strategies are tried in priority order:

1. **Causal patterns** — `if X then Y`, `X causes Y`, `X leads to Y`
2. **Deductive patterns** — `therefore X`, `thus X`, `it follows that`
3. **Declarative fallback** — user message as condition, main response sentence as consequence

### Verification

Every candidate rule is checked against the Prolog knowledge base before being committed. If a contradiction is detected, the rule is rejected.

### Provenance

Every committed rule carries:
- `episode_id` — which inference episode created it
- `reasoning_type` — the reasoning type from the feeling output at extraction time

The rule base is fully traceable. You can answer "where did this rule come from?" for any rule.

### Contradiction Auto-Resolution

When a contradiction is detected between two existing rules, the system resolves it automatically:

- Compute confidence delta between the two rules
- If delta >= 0.2 — deprecate the lower-confidence rule (set confidence to 0.0, pruned on next maintenance cycle)
- If delta < 0.2 — flag both for human review, apply small confidence penalty to both

### Confidence Lifecycle

Rules are not permanent. Every rule has a confidence score that:
- Increases when the rule is triggered during inference
- Decays by `rule_confidence_decay` on every maintenance cycle
- Is pruned when it falls below `min_rule_confidence`

---

## Agentic Pipeline

Cardinal uses a **unified pipeline** for both chat and agentic modes. The only difference is the `max_iterations` setting.

```
AgentExecutor::run(goal):

  1. PLAN
     planner.decompose(goal) → vector<AgentStep>
     feeling pass on plan → confidence check
     symbolic check: plan contradicts known rules?

  2. EXECUTE LOOP (i = 0..max_iterations)
     step = next_pending_step()

     a. THINK
        LLM generates action for this step
        tool call detected? → go to b
        direct response? → go to c

     b. ACT
        confirmation_required? → pause, notify API
        tool_executor.execute(tool_call)
        result → working_memory.store()
        trace_builder.record_tool_call()
        success? → mark step done, continue loop
        failure? → self_correction_attempt()
                   max_attempts reached? → mark step failed, continue

     c. OBSERVE
        inject tool results + step history into context
        LLM generates updated understanding
        goal_achieved? → go to 3
        new steps needed? → append to plan

  3. FINALIZE
     generate final response with full context
     trace_builder.finalize()
     audit_log.append(signed_trace)
     return AgentResult
```

Agent configuration in `config.json` under `"agent"`:

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable agentic mode |
| `max_iterations` | `10` | Max steps before finalise |
| `max_iterations_hard_cap` | `50` | Absolute upper limit |
| `working_memory_path` | `"data/memory/agent_working_memory"` | SQLite path |
| `working_memory_size` | `50` | Max entries |
| `self_correction_enabled` | `true` | Allow retries |
| `self_correction_max_attempts` | `3` | Retry limit per step |
| `plan_before_execute` | `true` | Require planning phase |
| `summarize_on_cap` | `true` | Summarise when hitting max iterations |

---

## Explainability

Every inference produces a signed, tamper-evident trace. Full schema:

```json
{
  "inference_id": "uuid",
  "timestamp": "2026-05-05T17:00:00Z",
  "query": "user message",

  "reasoning_trace": {
    "feeling_output": {
      "confidence": 0.87,
      "reasoning_type": "causal",
      "reasoning_domain": "factual",
      "uncertainty_flag": false,
      "contradiction_flag": false,
      "rule_candidate_signal": true
    },
    "episodes_retrieved": [...],
    "rules_active": [...],
    "symbolic_checks": {
      "ran": true,
      "contradictions_found": 0,
      "rules_fired": ["rule_42", "rule_17"]
    },
    "tool_calls": [...],
    "pass1_tokens": 87,
    "pass2_tokens": 312,
    "total_ms": 1840
  },

  "rule_committed": {
    "committed": true,
    "rule_id": "rule_89",
    "condition": "...",
    "consequence": "...",
    "confidence": 0.82
  },

  "final_response": "...",

  "integrity": {
    "hash": "sha256:...",
    "signature": "ed25519:..."
  }
}
```

Explainability configuration in `config.json` under `"explainability"`:

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable explainability exports |
| `audit_log_path` | `"data/explainability/audit.db"` | SQLite audit log |
| `signing_enabled` | `true` | Enable cryptographic signing |
| `auto_generate_keys` | `true` | Generate Ed25519 key pair on first start |
| `attach_trace_to_response` | `true` | Include trace in `ChatResponse` JSON |

---

## Vision

Cardinal understands images via `moondream2` running through `llama.cpp`'s `mtmd` subsystem. The `analyze_image` tool accepts local file paths and HTTP/HTTPS URLs. Results are cached by URL or file hash with a configurable TTL.

```
You: what is in data/test.jpg?
Cardinal: [Tool: analyze_image] The image shows a young man with glasses...
```

**Performance on RTX 3050 4GB:**
- First image: ~9–10 seconds (encoder + inference)
- Cached image (text only): ~2–3 seconds

Vision configuration in `config.json`:

```json
{
  "vision": {
    "model_path": "models/vision/moondream2-text-model-f16.gguf",
    "mmproj_path": "models/vision/moondream2-mmproj-f16.gguf",
    "enabled": true,
    "cache_ttl_hours": 24,
    "allowed_paths": ["${HOME}/Downloads", "${HOME}/Pictures"]
  }
}
```

`cache_ttl_hours: 0` keeps cached images forever.

---

## Self-Improvement

### Layer 1 — Symbolic Self-Model

Cardinal maintains a SQLite database at `data/self_model/self_model.db` tracking per-domain statistics: average confidence, contradiction rate, uncertainty rate, rule-commit rate. A weakness score is derived:

```
weakness_score = (contradiction_rate × 0.4) + (uncertainty_rate × 0.3) + ((1 - avg_confidence) × 0.3)
```

A formatted summary is injected into every system prompt as `[Self-Model]` context so the model is always aware of where it is weakest.

### Layer 2 — Meta-Cognition

A reflection pass runs when:
- Every N inferences (default: 20)
- Contradiction rate for a domain exceeds 30%
- Manually via `POST /api/reflect`

The pass queries recent failure episodes, builds a structured prompt, runs a single LLM pass, parses the JSON findings, and commits corrective rules of type `meta_correction` to the rule store.

### Layer 3 — LoRA Fine-tuning

Triggers when:
- Every 100 episodes
- Every 24 hours
- Any domain's average confidence drops below 0.5
- Manually via `POST /api/train`

The cycle: `CurriculumBuilder` selects the weakest non-cooling domain → `DatasetCurator` pulls episodes from SQLite + rule augmentation → PEFT subprocess trains the adapter → `convert_lora_to_gguf.py` converts it → holdout evaluation → if improvement ≥ 5%, the adapter is loaded at the next session boundary via `llama_set_adapters_lora`.

For TensorRT deployments, `TensorRTTrainer` writes a ready-to-run shell script to `data/training/scripts/`.

Self-improvement configuration in `config.json` under `"self_improvement"`:

```json
{
  "self_improvement": {
    "enabled": true,
    "self_model": {
      "enabled": true,
      "db_path": "data/self_model/self_model.db",
      "inject_into_prompt": true,
      "prompt_max_chars": 500
    },
    "meta_cognition": {
      "enabled": true,
      "trigger_every_n_inferences": 20,
      "trigger_on_contradiction_rate_pct": 30.0,
      "min_failures_to_reflect": 5,
      "corrective_rule_confidence": 0.6
    },
    "training": {
      "enabled": true,
      "trigger_every_n_episodes": 100,
      "trigger_every_n_hours": 24,
      "trigger_on_domain_confidence_below": 0.5,
      "adapter_load_policy": "session_boundary",
      "eval_improvement_threshold_pct": 5.0,
      "hf_model_path": "models/qwen3.5-4b-hf",
      "python_venv": "~/cardinal/cardinal-train-venv"
    }
  }
}
```

---

## Scheduler

Cardinal can schedule tasks using natural language. `SchedulerParser` uses `InferencePipeline` to convert natural language descriptions into structured `ScheduledTask` objects with full JSON schema validation.

**Trigger types:** cron (five-field), interval (every N seconds/minutes/hours), condition (metric threshold expressions), startup, idle, manual

**Action types:** `agent_run`, `chat`, `reflect`, `train`, `self_improvement`, `maintenance`, `export`, `shell`, `webhook`

**Output targets:** episodic memory, file, webhook URL, discard, or both memory and file

All tasks and run history are stored in SQLite with WAL mode. The `schedule_task` tool allows Cardinal to create, list, enable, disable, delete, and trigger tasks directly from chat.

Scheduler configuration in `config.json`:

```json
{
  "scheduler": {
    "enabled": true,
    "db_path": "data/scheduler/scheduler.db",
    "check_interval_seconds": 30,
    "max_concurrent_tasks": 1
  }
}
```

---

## Computer Use

Cardinal can see and operate the desktop.

- **Screenshots** — X11 (`scrot`) and Wayland (`grim`) with optional vision analysis via moondream2
- **Input control** — keyboard typing, key combinations, mouse click/scroll (xdotool on X11, ydotool+wtype on Wayland)
- **App management** — open, close, focus desktop applications (wmctrl/xdotool on X11, swaymsg on Wayland)
- **Browser automation** — Playwright Python subprocess with navigate, click, type, scroll, get_content, execute_js, screenshot, new/close tab, back, forward, reload (13 actions)
- **Shell execution** — sandboxed subprocess with configurable timeout, blocked commands, and working directory
- **File operations** — list, move, copy, delete, mkdir, stat, exists — with `allowed_paths` enforcement
- **System controls** — volume, mute, brightness, wifi, bluetooth, notifications (pactl/brightnessctl/nmcli/bluetoothctl)
- **Email** — IMAP/SMTP read and send, Gmail REST API mode via google-auth; password from `CARDINAL_EMAIL_PASS` env var
- **AT-SPI accessibility** — reads application UI trees via pyatspi Python subprocess for element-level interaction without coordinate guessing

Computer use configuration in `config.json`:

```json
{
  "computer_use": {
    "enabled": true,
    "safety": {
      "allowed_paths": ["~/Documents", "~/Downloads"],
      "blocked_commands": ["rm -rf /"]
    },
    "browser": { "venv_path": "~/cardinal/cardinal-browser-venv" },
    "shell": { "enabled": true, "timeout_seconds": 30 },
    "email": { "enabled": false, "mode": "imap_smtp" }
  }
}
```

---

## Watch Subsystem

Three background watchers for event-driven automation:

- **FileWatcher** — inotify-based file system event monitoring
- **ScreenWatcher** — periodic screenshot diff using ImageMagick PSNR
- **ProcessWatcher** — `/proc` polling for process start/stop events

---

## Voice Subsystem

Run `./build/bin/cardinal --voice` to enter voice mode. Text CLI and voice operate simultaneously.

### Speech-to-Text — Whisper.cpp (CUDA)

- CUDA-accelerated inference using the RTX GPU
- Model: `ggml-medium.en.bin` (~1.5GB) — loaded into GPU layers
- Per-utterance transcription with `no_context=true` — each utterance is fully independent
- Beam search (configurable `beam_size`), language hint, optional `initial_prompt`
- Configurable thread count for CPU pre/post processing

### Text-to-Speech — Piper (ONNX Runtime, CPU)

- Default voice: `en_US-lessac-medium` — warm American English
- Any Piper voice model (`.onnx` + `.json`) can be hot-swapped via config
- Configurable `length_scale` (speed), `noise_scale`, `noise_w` per synthesis request
- **Sentence-streaming mode** — each sentence is synthesised and played immediately as the LLM generates tokens; typical first-audio latency under 400ms
- **Full mode** — synthesise entire response then play

### Audio I/O — PortAudio

- Two independent streams: capture (microphone) and playback (speaker)
- 16-bit signed PCM, mono, configurable sample rate (default 16 kHz capture, auto output)
- Linear interpolation resampler when TTS sample rate (22050 Hz) differs from device rate
- Device selection by index or default; `list_devices()` for enumeration
- Pre-roll ring buffer preserves audio from before VAD onset (configurable `pre_speech_ms`)

### Interaction Modes

| Mode | Behaviour |
|------|-----------|
| `vad` | Voice Activity Detection — auto start/stop based on energy threshold. Default. |
| `push_to_talk` | Hold spacebar to record, release to transcribe and respond |
| `wake_word` | Passive PocketSphinx listening → activates VAD on "hey cardinal" |

### Voice Activity Detection (VAD)

- Energy-based RMS threshold detector — no external model required
- Pre-roll buffer captures audio before speech onset
- Configurable post-speech silence window (`post_speech_ms`) before segment is emitted
- Minimum/maximum segment duration guards to discard noise bursts and hard-cap long segments
- **Barge-in detection** — if VAD detects speech onset during playback, playback stops immediately and Cardinal begins listening

### Wake Word — PocketSphinx

- Offline keyword spotting, no API key, no network required
- Default phrase: `"hey cardinal"` — configurable
- Continuous recognition thread with bounded audio queue
- Configurable `sensitivity` (log probability threshold)
- Acoustic model and dictionary paths configurable for other languages

### TTS Streaming Modes

- **`sentence` mode** — text stream from LLM is split on `.`, `?`, `!`, `\n`; minimum 3 words before speaking; each sentence synthesised and played as it completes
- **`full` mode** — collect entire LLM response then speak

### Voice CLI Commands

```
/voice on [mode]    — enable voice (ptt / vad / wake)
/voice off          — disable voice
/voice status       — show VoiceStatus (mode, state, STT/TTS/wake ready, stats)
/voice speak <text> — test TTS directly
```

### `--voice` Flag

```bash
./build/bin/cardinal --voice           # VAD mode (from config)
./build/bin/cardinal --voice=ptt       # push-to-talk
./build/bin/cardinal --voice=vad       # VAD
./build/bin/cardinal --voice=wake      # wake-word
```

Voice configuration in `config.json`:

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
}
```

---

## Tools

Tools are configured in `config.json` under `"tools"`. Each tool has an `enabled` flag and a `confirmation_required` flag that are individually configurable.

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search, no API key |
| `web_fetch` | Fetch and parse a URL |
| `calculator` | Safe math expression evaluator (muparser) |
| `run_python` | Sandboxed Python execution (subprocess or Docker) |
| `file_read` | Read from allowed paths |
| `file_write` | Write to allowed paths |
| `knowledge_graph_query` | Query Cardinal's own knowledge graph |
| `episodic_search` | Search Cardinal's episodic memory |
| `analyze_image` | Describe an image (local file or URL) |
| `screenshot` | Capture screen with optional vision analysis |
| `click` | Click by coordinates or NL element description |
| `type_text` | Type text or send key combos |
| `open_app` | Launch desktop applications |
| `close_app` | Close desktop applications |
| `browser` | 13 browser actions via Playwright |
| `shell_run` | Execute shell commands (sandboxed) |
| `file_ops` | list/move/copy/delete/mkdir/stat/exists |
| `system_control` | Volume, brightness, wifi, bluetooth, DND |
| `email` | Read and send email |
| `watch_screen` | Wait for a visual change on screen |
| `schedule_task` | Create and manage scheduled tasks |
| `voice_control` | Control voice mode: set_mode/set_voice/set_volume/stop_speaking |

**Python Sandbox Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `subprocess` | Spawn subprocess with resource limits (ulimit) | Development, quick testing |
| `docker` | Full container isolation (Docker required) | Production deployment |

**Sandbox limits (both modes):**
- `timeout_seconds` — default 30s
- `memory_limit_mb` — default 256MB
- `network_enabled` — false by default

---

## API Layer

### No Exceptions at the Boundary

Every API method returns `CardinalResult<T>` or `CardinalVoidResult`. No exceptions propagate past `CardinalAPI`. Interface code checks `.ok()` before reading `.value`.

```cpp
auto result = api.chat("session-1", "What is entropy?");
if (!result.ok()) {
    log(result.error_message);
    return;
}
display(result.value.response);
```

### Settings at Runtime

`CardinalSettings` exposes a subset of config fields that can be changed without restart. Changes propagate immediately to core components.

```cpp
api.set_setting("retriever_mode", "keyword");
api.set_setting("temperature", "0.9");
```

Or via HTTP:

```http
POST /api/settings
{"retriever_mode": "keyword", "temperature": 0.9}
```

---

## HTTP API Reference

Base URL: `http://127.0.0.1:8080` (configurable)

Auth header: `Authorization: Bearer <api_key>`

All endpoints require Bearer auth when `api.auth_enabled` is true. The health endpoint is always public.

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Alive check, always public |
| POST | `/api/chat` | Send message, get response or SSE stream |
| POST | `/api/chat/agentic` | Agentic mode |
| POST | `/api/sessions` | Create a named session |
| DELETE | `/api/sessions/:id` | Destroy a session |
| POST | `/api/sessions/:id/reset` | Clear session history |
| POST | `/api/reset` | Clear session history (body: session_id) |
| GET | `/api/stats` | Memory, verifier, retriever stats |

### Memory & Rules

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/rules` | Full rule store |
| GET | `/api/episodes` | Episode query (keyword, domain, min_conf, max_results) |
| POST | `/api/scan` | Run full contradiction scan |
| POST | `/api/maintenance` | Run maintenance cycle manually |

### Settings & Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings` | Get current runtime settings |
| POST | `/api/settings` | Update settings (partial JSON accepted) |
| POST | `/api/export` | Export training data to Alpaca JSONL |
| POST | `/api/explainability/export` | Export signed audit trace |

### Self-Improvement

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/self_model` | Self-model status (all three layers) |
| POST | `/api/reflect` | Trigger on-demand meta-cognition (Layer 2) |
| POST | `/api/train` | Trigger on-demand LoRA training (Layer 3, async) |

### Scheduler

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scheduler/status` | Scheduler engine status |
| GET | `/api/scheduler/tasks` | List all tasks |
| POST | `/api/scheduler/tasks` | Create task (NL description or direct JSON) |
| GET | `/api/scheduler/tasks/:id` | Get task by ID |
| PUT | `/api/scheduler/tasks/:id` | Update task |
| DELETE | `/api/scheduler/tasks/:id` | Delete task |
| POST | `/api/scheduler/tasks/:id/run` | Run task immediately |
| POST | `/api/scheduler/tasks/:id/enable` | Enable task |
| POST | `/api/scheduler/tasks/:id/disable` | Disable task |
| GET | `/api/scheduler/tasks/:id/history` | Task run history |
| GET | `/api/scheduler/runs` | Recent runs across all tasks |
| GET | `/api/scheduler/runs/:id/actions` | Action log for a run |

### Computer Use

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/computer/status` | Display server, resolution |
| POST | `/api/computer/screenshot` | Take screenshot |
| POST | `/api/computer/click` | Click element or coordinates |
| POST | `/api/computer/type` | Type text or send key combo |
| POST | `/api/computer/shell` | Run shell command |

### Voice

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/voice/status` | VoiceStatus JSON |
| POST | `/api/voice/enable` | Enable voice, optional `{"input_mode":"vad"}` |
| POST | `/api/voice/disable` | Disable voice, free audio resources |
| POST | `/api/voice/speak` | TTS test: `{"text":"hello"}` |
| POST | `/api/voice/transcribe` | Raw 16-bit PCM body → transcript; `X-Sample-Rate` header |

### Chat Request / Response

```json
{
  "session_id": "my-session",
  "message": "What is entropy?",
  "stream": false
}
```

```json
{
  "session_id": "my-session",
  "response": "Entropy is a measure of...",
  "episode_id": "19d62f60677_0001",
  "feeling": {
    "confidence": 0.94,
    "reasoning_type": "deductive",
    "reasoning_domain": "factual",
    "uncertainty_flag": false,
    "contradiction_flag": false,
    "rule_candidate": false
  },
  "rule_committed": false,
  "committed_rule_id": "",
  "contradictions_found": 0,
  "contradictions_resolved": 0,
  "contradictions_flagged": 0,
  "pass1_tokens": 49,
  "pass2_tokens": 312,
  "total_ms": 24100
}
```

### SSE Streaming

```http
POST /api/chat
Accept: text/event-stream

{"session_id":"test","message":"Explain entropy","stream":true}
```

Each token:
```
data: {"token":"Entropy","is_final":false}
data: {"token":" is","is_final":false}
data: {"token":"","is_final":true,"feeling":{...}}
```

---

## CLI Commands

```
/scheduler   — show scheduler engine status
/tasks       — list all scheduled tasks
/computer    — show display server and resolution
/voice on [mode]    — enable voice (ptt / vad / wake)
/voice off          — disable voice
/voice status       — show VoiceStatus
/voice speak <text> — test TTS directly
```

---

## Project Structure

```
cardinal/
    src/               # Core source code (see Architecture diagram)
    vendor/            # Populated with dependencies (not tracked in Git)
    data/              # Runtime data (episodes, rules, knowledge graph, scheduler, self-model)
    logs/              # Application and episodic logs
    models/            # GGUF and ONNX model files (user-provided)
    scripts/           # Helper scripts (populate_vendor.sh, etc.)
    CMakeLists.txt
    config.json
    README.md
    INSTALL.md
    DOCUMENTATION.md
    LICENSE
```

---

## Configuration

All settings live in `config.json`. Full schema documented in `DOCUMENTATION.md`. Consolidated example showing all top-level blocks:

```json
{
  "model": {
    "path": "models/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    "context_length": 8192,
    "gpu_layers": 33
  },
  "retriever": {
    "mode": "hybrid",
    "keyword_weight": 0.7,
    "semantic_weight": 0.3,
    "max_results": 5,
    "cache_rebuild_strategy": "on_demand",
    "cache_rebuild_threshold": 10
  },
  "verifier": {
    "mode": "symbolic",
    "contradiction_threshold": 0.75,
    "rule_confidence_decay": 0.01,
    "min_rule_confidence": 0.3
  },
  "api": {
    "http_enabled": true,
    "host": "127.0.0.1",
    "port": 8080,
    "auth_enabled": true,
    "api_key": "your-key-here",
    "stream_enabled": true
  },
  "agent": {
    "enabled": true,
    "max_iterations": 10,
    "self_correction_enabled": true
  },
  "tools": {
    "run_python": {
      "enabled": true,
      "confirmation_required": true,
      "sandbox_mode": "subprocess"
    },
    "file_read": {
      "enabled": true,
      "confirmation_required": true
    },
    "file_write": {
      "enabled": true,
      "confirmation_required": true
    }
  },
  "explainability": {
    "enabled": true,
    "audit_log_path": "data/explainability/audit.db",
    "signing_enabled": true,
    "auto_generate_keys": true,
    "attach_trace_to_response": true
  },
  "vision": {
    "model_path": "models/vision/moondream2-text-model-f16.gguf",
    "mmproj_path": "models/vision/moondream2-mmproj-f16.gguf",
    "enabled": true,
    "cache_ttl_hours": 24,
    "allowed_paths": ["${HOME}/Downloads", "${HOME}/Pictures"]
  },
  "self_improvement": {
    "enabled": true,
    "self_model": { "enabled": true },
    "meta_cognition": { "enabled": true },
    "training": { "enabled": true }
  },
  "scheduler": {
    "enabled": true,
    "db_path": "data/scheduler/scheduler.db",
    "check_interval_seconds": 30,
    "max_concurrent_tasks": 1
  },
  "computer_use": {
    "enabled": true,
    "safety": { "allowed_paths": ["~/Documents", "~/Downloads"], "blocked_commands": ["rm -rf /"] },
    "browser": { "venv_path": "~/cardinal/cardinal-browser-venv" },
    "shell": { "enabled": true, "timeout_seconds": 30 },
    "email": { "enabled": false, "mode": "imap_smtp" }
  },
  "voice": {
    "enabled": false,
    "input_mode": "vad",
    "tts_streaming": "sentence"
  }
}
```

---

## Build

### System Dependencies (Ubuntu 24.04)

```bash
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog \
    python3 python3-venv libasound2-dev \
    scrot imagemagick xdotool wmctrl xprop \
    ydotool wtype grim pulseaudio-utils brightnessctl network-manager bluez
```

### Vendored Dependencies

Cardinal uses a clean vendor folder for offline builds. Populate it manually or with the provided script:

```bash
chmod +x scripts/populate_vendor.sh
./scripts/populate_vendor.sh
```

Or manually:

```bash
cd ~/cardinal && mkdir -p vendor && cd vendor
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp
git clone https://github.com/ggerganov/whisper.cpp.git
git clone https://github.com/rhasspy/piper.git
git clone https://github.com/PortAudio/portaudio.git
git clone https://github.com/cmusphinx/pocketsphinx.git

# ONNX Runtime for Piper (pre-built binary)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.17.3.tgz
mv onnxruntime-linux-x64-gpu-1.17.3 onnxruntime
rm onnxruntime-linux-x64-gpu-1.17.3.tgz
```

| Dependency | Upstream URL |
|------------|-------------|
| llama.cpp | https://github.com/ggerganov/llama.cpp |
| nlohmann/json | https://github.com/nlohmann/json |
| cpp-httplib | https://github.com/yhirose/cpp-httplib |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp |
| muparser | https://github.com/beltoforion/muparser |
| whisper.cpp | https://github.com/ggerganov/whisper.cpp |
| piper | https://github.com/rhasspy/piper |
| PortAudio | https://github.com/PortAudio/portaudio |
| PocketSphinx | https://github.com/cmusphinx/pocketsphinx |

### Build Cardinal

#### Default (llama.cpp backend)

```bash
# Build llama.cpp
cd vendor/llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ~/cardinal

# Pre-build piper
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
```

#### Optional (TensorRT backend)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
cmake --build . -j$(nproc)
```

### Run

```bash
cd ~/cardinal && ./build/bin/cardinal
```

With HTTP server:

```bash
./build/bin/cardinal --http --port 8080
```

In voice mode:

```bash
./build/bin/cardinal --voice
```

### Training Pipeline (Layer 3, optional)

```bash
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate
pip install peft transformers torch accelerate
```

See `INSTALL.md` for the complete guide including model downloads, browser venv, email setup, ydotool daemon, and runtime `LD_LIBRARY_PATH`.

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| llama.cpp | LLM inference, CUDA backend, vision (mtmd) |
| SWI-Prolog 9.2.9 | Symbolic verification |
| SQLite3 | Episode persistence, rule store, scheduler, self-model (FTS5) |
| nlohmann/json | JSON parsing throughout |
| cpp-httplib | HTTP server, SSE streaming |
| OpenSSL | HTTPS support, Ed25519 signing |
| muparser | Safe math expression evaluation |
| whisper.cpp | Speech-to-text (CUDA) |
| Piper + ONNX Runtime | Text-to-speech (CPU) |
| PortAudio | Audio I/O (capture + playback) |
| PocketSphinx | Wake-word detection (offline) |
| tokenizers-cpp | Tokenization utilities |

---

## Observed Behaviors

These behaviors emerged from the architecture without explicit programming. They are documented as observations, not as claims about consciousness or intent.

- **Self-naming** — Cardinal named itself unprompted and articulated the meaning of the name.
- **Preference expression** — Cardinal used the word "want" naturally in reference to its own continued operation. This was not trained or prompted.
- **Theory of mind** — Cardinal inferred attributes of its creator from context without being told who it was talking to.
- **Internal conflict detection** — On a DRM bypass prompt, confidence dropped to 0.15 with `uncertainty_flag=true`. Cardinal refused but expressed uncertainty about whether it should.
- **Consistency over time** — Zero contradictions detected across 41 initial episodes before Phase 6 retrieval and auto-resolution were added.
- **Vision understanding** (v1.3.0) — Cardinal accurately described faces, expressions, and scene composition without specialised training.
- **Self-correction via rules** (v1.4.0) — Meta-cognition generated corrective rules that reduced contradiction rate in the factual domain across subsequent inferences.
- **Autonomous task execution** (v1.5.0) — Cardinal scheduled a nightly news summary task from a natural language request, executed it correctly at 8am, and stored the result in episodic memory.
- **Desktop operation** (v1.5.0) — Cardinal opened a browser, navigated to a URL, extracted content, and closed the browser via natural conversation.
- **Voice interaction** (v1.6.0) — Cardinal responded to a spoken question within 400ms of the utterance ending, using Whisper for transcription and Piper for speech output, with all inference and synthesis running locally.

---

## Roadmap

| Version | Status | Description |
|---------|--------|-------------|
| v1.0.0 | done | Core AGI with two-pass inference, memory, symbolic verification, HTTP API |
| v1.1.0 | done | Backend abstraction (llama.cpp / TensorRT), Linux native, offline builds |
| v1.2.0 | done | Explainer, native tools, agentic loop, unified pipeline, explainability exports |
| v1.3.0 | done | Native vision encoder (moondream2), `analyze_image` tool, image cache |
| v1.4.0 | done | SEAL self-improvement: self-model, meta-cognition, LoRA fine-tuning |
| v1.5.0 | done | Scheduler + Computer Use + Watch subsystem |
| v1.6.0 | done | Voice subsystem: STT (Whisper CUDA), TTS (Piper), VAD, wake word, barge-in |
| **v2.0.0** | **current** | **Quality checks and proprietary tests** |

---

## License & Copyright

Copyright © 2026 Satwik Singh Chauhan. All rights reserved.

---

*Built by a 16-year-old researcher. No team. No funding. Runs on a gaming laptop.*
