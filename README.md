# Cardinal v1.5.0

**A production-grade neurosymbolic AGI architecture with self-improvement, autonomous scheduling, and full desktop computer use.**

Cardinal combines a large language model core with **vision understanding**, symbolic verification, persistent memory, hybrid retrieval, a **full agentic loop**, **explainability exports**, a **three-layer self-improvement system**, a **natural-language task scheduler**, and a complete **computer use subsystem** that can operate your desktop. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

**Current version: v1.5.0** — Autonomic routine execution and computer use: NL-driven task scheduler, screen capture and analysis, keyboard/mouse control, browser automation (Playwright), shell execution, file management, system controls, email, and AT-SPI accessibility integration.

---

## What's New in v1.5.0

### Scheduler — Autonomous Routine Execution

Cardinal can now schedule tasks using natural language. Tell it "check the news every morning at 8am and save a summary to my desktop" and it parses the description, creates a `ScheduledTask`, and executes it autonomously at the right time — no cron syntax required.

- **NL parsing** — `SchedulerParser` uses `InferencePipeline` to convert natural language into structured `ScheduledTask` objects with full JSON schema validation
- **Trigger types** — cron (five-field), interval (every N seconds/minutes/hours), condition (metric threshold expressions), startup, idle, manual
- **Action types** — `agent_run`, `chat`, `reflect`, `train`, `self_improvement`, `maintenance`, `export`, `shell`, `webhook`
- **Output targets** — episodic memory, file, webhook URL, discard, or both memory and file
- **Persistent storage** — all tasks and run history stored in SQLite with WAL mode
- **`schedule_task` tool** — Cardinal can create, list, enable, disable, delete, and trigger tasks directly from chat

### Computer Use — Full Desktop Agent

Cardinal can now see and operate your desktop.

- **Screenshots** — X11 (`scrot`) and Wayland (`grim`) with optional vision analysis via moondream2
- **Input control** — keyboard typing, key combinations, mouse click/scroll (xdotool on X11, ydotool+wtype on Wayland)
- **App management** — open, close, focus desktop applications (wmctrl/xdotool on X11, swaymsg on Wayland)
- **Browser automation** — Playwright Python subprocess (persistent helper process) with navigate, click, type, scroll, get_content, execute_js, screenshot, new/close tab, back, forward, reload
- **Shell execution** — sandboxed subprocess with configurable timeout, blocked commands, and working directory
- **File operations** — list, move, copy, delete, mkdir, stat, exists — with allowed_paths enforcement
- **System controls** — volume, mute, brightness, wifi, bluetooth, notifications (pactl/brightnessctl/nmcli/bluetoothctl)
- **Email** — IMAP/SMTP read and send, Gmail REST API mode via google-auth; password from `CARDINAL_EMAIL_PASS` env var
- **AT-SPI accessibility** — reads application UI trees via pyatspi Python subprocess for element-level interaction without coordinate guessing

### Watch Subsystem

Three background watchers for event-driven automation:
- **FileWatcher** — inotify-based file system event monitoring
- **ScreenWatcher** — periodic screenshot diff using ImageMagick PSNR
- **ProcessWatcher** — `/proc` polling for process start/stop events

### New HTTP Endpoints (v1.5.0)

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
| GET | `/api/computer/status` | Display server, resolution |
| POST | `/api/computer/screenshot` | Take screenshot |
| POST | `/api/computer/click` | Click element or coordinates |
| POST | `/api/computer/type` | Type text or send key combo |
| POST | `/api/computer/shell` | Run shell command |

### New CLI Commands (v1.5.0)

```
/scheduler   — show scheduler engine status
/tasks       — list all scheduled tasks
/computer    — show display server and resolution
```

---

## What Was in v1.4.0

- **Layer 1 — Symbolic Self-Model** — Cardinal tracks its own reasoning patterns in SQLite. Per-domain statistics (confidence, contradiction rate, uncertainty rate, rule-commit rate) are recorded after every inference and injected into the system prompt.
- **Layer 2 — Meta-Cognition** — A scheduled reflection pass analyses recent failure episodes and generates corrective rules stored in the rule base. Available on-demand via `/api/reflect`.
- **Layer 3 — LoRA Fine-tuning Pipeline** — Cardinal curates its own training data from high-confidence episodes, runs HuggingFace PEFT via subprocess, converts to GGUF, evaluates, and hot-loads the adapter via `llama_set_adapters_lora`.

---

## What Cardinal Is

Most LLM systems are stateless. Cardinal is not. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, improves its own weights from accumulated experience, and now **operates your computer autonomously** and **schedules tasks to run while you sleep**.

The architecture is **neurosymbolic** — combining the pattern-matching strength of neural inference with the consistency guarantees of symbolic logic. With v1.5.0, Cardinal adds a third dimension: **autonomic execution** — the ability to act on the world on its own schedule.

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~5GB core + models |
| OS | Ubuntu 24.04 LTS |

**Models:**
- Primary LLM: `Qwen3.5 4B Q4_K_M` (llama.cpp or TensorRT)
- Vision encoder: `moondream2` text model + mmproj (quantized, CPU)
- Fine-tuning: HuggingFace weights at `models/qwen3.5-4b-hf/` + Python venv with PEFT

---

## Architecture (v1.5.0)

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
    +-- Scheduler (v1.5.0)  ← NEW
    |     +-- SchedulerEngine (background thread, trigger dispatch)
    |     +-- SchedulerParser (NL → ScheduledTask via InferencePipeline)
    |     +-- SchedulerStore (SQLite WAL: tasks, runs, action_logs)
    |
    +-- Computer Use (v1.5.0)  ← NEW
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
    +-- Watch Subsystem (v1.5.0)  ← NEW
    |     +-- FileWatcher (inotify)
    |     +-- ScreenWatcher (periodic PSNR diff)
    |     +-- ProcessWatcher (/proc polling)
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

## Tools (v1.5.0)

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

---

## Build

```bash
# System dependencies
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog \
    python3 python3-venv scrot imagemagick xdotool wmctrl xprop \
    ydotool wtype grim pulseaudio-utils brightnessctl network-manager bluez

# Install vendor dependencies manually (no submodules)
cd ~/cardinal && mkdir -p vendor && cd vendor
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp

# Build llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ~/cardinal

# Build Cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
cd ~/cardinal && ./build/bin/cardinal
```

See `INSTALL.md` for the complete guide including browser venv, email setup, ydotool daemon, and all models.

---

## HTTP API

Base URL: `http://127.0.0.1:8080`
Auth: `Authorization: Bearer <api_key>` (except `/api/health`)

Full API reference in `DOCUMENTATION.md`.

---

## Configuration (`config.json`)

v1.5.0 adds two top-level blocks:

```json
{
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
| **v1.5.0** | **done** | **Scheduler + Computer Use + Watch subsystem** |
| v1.6.0 | planned | Voice & Audio |
| v2.0.0 | planned | Production hardening + formal proof |

---

## License & Copyright

Copyright © 2025–2026 Satwik Singh Chauhan. All rights reserved.
Cardinal is **not open source**. See `LICENSE` for details.

---

*Built by a 16-year-old researcher. No team. No funding. Runs on a gaming laptop.*
*Now it acts.*
