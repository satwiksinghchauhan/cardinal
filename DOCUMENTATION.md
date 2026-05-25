# Cardinal v1.5.0 — Technical Documentation

**Architecture:** Neurosymbolic AGI Core + Agentic Loop + Explainability + Vision + Self-Improvement + Scheduler + Computer Use + Watch
**Language:** C++20
**Platform:** Linux (Ubuntu 24.04 LTS)
**GPU:** NVIDIA CUDA (TensorRT / llama.cpp)
**Vision:** moondream2 via `llama.cpp` `mtmd` subsystem
**Self-Improvement:** Three-layer SEAL system (self-model, meta-cognition, LoRA fine-tuning)
**v1.5.0 additions:** Natural-language scheduler, full desktop computer use, watch subsystem

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
14. [Agentic Pipeline](#14-agentic-pipeline)
15. [Tools System](#15-tools-system)
16. [Explainability Exports](#16-explainability-exports)
17. [Training Export](#17-training-export)
18. [Configuration Reference](#18-configuration-reference)
19. [CardinalAPI Reference](#19-cardinalapi-reference)
20. [HTTP API Reference](#20-http-api-reference)
21. [Settings Manager](#21-settings-manager)
22. [Session Manager](#22-session-manager)
23. [Type Reference](#23-type-reference)
24. [Error Handling](#24-error-handling)
25. [Offline Builds & Vendoring](#25-offline-builds--vendoring)
26. [Module Reference](#26-module-reference)
27. [Threading Model](#27-threading-model)
28. [Lifecycle and Startup Sequence](#28-lifecycle-and-startup-sequence)

---

## 1. Architecture Overview

Cardinal is structured in four layers. Each layer depends only on the layers below it.

```
Layer 4 — Interfaces
    CLI (interactive loop, /commands)
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
    ToolExecutor        sandboxed tool execution (all tools including computer use)
    ILLMBackend         abstract backend (llama.cpp / TensorRT)
    ConsistencyChecker  verifier orchestrator
    SymbolicEngine      SWI-Prolog integration
    SchedulerEngine     background thread, NL parsing, task dispatch  ← v1.5.0
    Computer Use Layer  display, input, browser, shell, file, email   ← v1.5.0
    Watch Subsystem     file, screen, process watchers                ← v1.5.0

Layer 1 — Foundation
    RuleStore           persistent rule base
    KnowledgeGraph      typed node graph
    EpisodicMemory      JSONL audit trail
    EpisodicStorage     SQLite + FTS5 searchable index
    EpisodicRetriever   TF-IDF + keyword + hybrid retrieval
    VisionEncoder       moondream2 via mtmd API                       ← v1.3.0
    VisionCache         URL download cache, TTL eviction              ← v1.3.0
    SelfImprovementLoop SEAL orchestrator                             ← v1.4.0
    SelfModel           per-domain statistics SQLite                  ← v1.4.0
    MetaCognition       reflection, corrective rules                  ← v1.4.0
    Training Pipeline   CurriculumBuilder, DatasetCurator, trainers  ← v1.4.0
    SchedulerStore      SQLite WAL: tasks, runs, action_logs          ← v1.5.0
    ConfigLoader        typed config, validated at startup
    Logger              thread-safe, 6 levels
```

### Component Ownership

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary — callers see only types defined in `cardinal_types.h`.

### Dependency Graph (v1.5.0)

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

### v1.4.0 Addition — Self-Improvement Hook

After every `run_post_inference()`, `SelfImprovementLoop::on_inference()` is called with feeling output fields:
- Layer 1: O(1) SQLite upsert to `domain_stats` and `reasoning_stats`
- Layer 2: checks inference counter and contradiction rate thresholds
- Layer 3: checks episode counter and domain confidence thresholds

### v1.5.0 Addition — Scheduler Hook

After every inference, `SchedulerEngine::on_inference()` updates the idle tracker timestamp. This allows idle-triggered tasks to measure actual idle time since the last inference.

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

**File:** `src/training/self_improvement_loop.h/.cpp`

**Hooks:**
- `on_inference(domain, reasoning_type, confidence, contradiction, uncertainty, rule_committed)` — after every inference. Fast path: updates atomics, may trigger reflection or post to training thread.
- `on_session_boundary()` — at `destroy_session()`. Applies pending LoRA adapter at a clean cut-point.

**Training thread:** sleeps on `condition_variable`, wakes on trigger or 60-second poll. One cycle at a time; overlapping triggers coalesced.

**Trigger conditions for Layer 3:**

| Trigger | Config key | Default |
|---------|-----------|---------|
| Episode count | `trigger_every_n_episodes` | 100 |
| Wall clock | `trigger_every_n_hours` | 24 |
| Domain confidence below | `trigger_on_domain_confidence_below` | 0.5 |
| Manual | `POST /api/train` | — |

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

**CurriculumBuilder** — decides target domain by weakness score + recency bonus. Adjusts LoRA hyperparameters per weakness severity.

**DatasetCurator** — converts episodes to alpaca format. Quality filters: confidence floor, minimum text length, FNV-1a dedup, holdout reservation. Rule augmentation adds corrective rules as training examples.

**LlamaCppTrainer** — PEFT subprocess → `convert_lora_to_gguf.py` → `llama_set_adapters_lora` on `ctx_pass2_` only.

**TensorRTTrainer** — writes ready-to-run shell script to `data/training/scripts/`. `evaluate()` and `load_adapter()` are no-ops.

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
    std::string name;             // short human-readable name
    std::string description;      // original NL description
    bool        enabled;

    TriggerSpec trigger;          // when to run
    TaskAction  action;           // what to do

    std::optional<bool> allow_file_write;
    std::optional<bool> allow_web_access;
    std::optional<bool> require_confirmation;
    std::optional<bool> full_autonomy;

    int    run_count;
    int    fail_count;
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
| `CONDITION` | Metric threshold expression | `condition_expr` |
| `STARTUP` | Once on Cardinal start | — |
| `IDLE` | After N idle minutes | `idle_minutes` |
| `MANUAL` | Only via `run_now` API | — |

**Cron format:** standard five-field: `minute hour dom month dow`. Supports `*/N` step, ranges, comma-separated values.

**Condition expression variables:**
`factual_confidence`, `total_contradictions`, `total_reflections`, `total_training_runs`, `last_improvement_pct`, `idle_minutes`, `hour_of_day`, `day_of_week`

**Condition operators:** `<`, `>`, `<=`, `>=`, `==`, `!=`, `AND`

Example: `"factual_confidence < 0.6 AND hour_of_day > 22"`

### 11.4 Action Types

| Type | Description |
|------|-------------|
| `AGENT_RUN` | Full agentic execution with tools |
| `CHAT` | Single inference via InferencePipeline |
| `REFLECT` | Force Layer 2 meta-cognition reflection |
| `TRAIN` | Post Layer 3 training request |
| `SELF_IMPROVEMENT` | Reflect then train |
| `MAINTENANCE` | ConsistencyChecker maintenance cycle |
| `EXPORT` | Training data export |
| `SHELL` | Run a shell command |
| `WEBHOOK` | POST result to a URL |

### 11.5 Output Targets

| Target | Description |
|--------|-------------|
| `MEMORY` (default) | Store result in episodic memory |
| `FILE` | Write to `output_file` path |
| `WEBHOOK` | POST JSON to `webhook_url` |
| `DISCARD` | Run silently, discard output |
| `BOTH` | MEMORY + FILE |

### 11.6 SchedulerParser

**File:** `src/scheduler/scheduler_parser.h/.cpp`

Takes an `InferencePipeline*` and converts natural language task descriptions to `ScheduledTask` structs.

**`parse(nl_description, session_id)` steps:**
1. Builds a structured system prompt with full JSON schema and selection rules
2. Calls `InferencePipeline::run(InferenceRequest)` with `tools_enabled=false`
3. Parses model JSON output via `extract_from_json()` (public static — callable directly)
4. Validates confidence threshold (default 0.70)
5. If confidence < threshold, sets `clarification_needed` and returns
6. Calls `finalise_task()` — assigns UUID, timestamps, session context

**`extract_from_json(model_output, session_id, min_confidence)` is public static** so `SchedulerEngine::create_task_from_nl()` can call it directly after its own LLM call.

**Note:** Raw string literals with `##` or `→` characters cannot be used in the prompt string — GCC treats `##` as a preprocessor token and `→` as a multi-byte identifier. The prompt is built with regular string concatenation.

### 11.7 SchedulerEngine

**File:** `src/scheduler/scheduler_engine.h/.cpp`

**Background thread lifecycle:**
1. On `start()`: opens SQLite store, fires STARTUP tasks immediately, starts `engine_loop()` thread
2. Engine loop: sleeps `check_interval_seconds` on `condition_variable`, calls `tick()` on wake
3. `tick()`: iterates enabled tasks, evaluates triggers via `should_fire()`, calls `dispatch_task()` for any that fire
4. On `stop()`: sets `stop_requested_=true`, notifies CV, joins thread, closes store

**Trigger evaluation:**
- `CRON` — compares current `localtime` against five cron fields, deduplicates by checking `last_run_at` within same minute
- `INTERVAL` — compares `now - last_run_at` against `interval_seconds`
- `CONDITION` — evaluates expression via `eval_condition_expr()` using metric variables from `SelfImprovementStatus`
- `IDLE` — checks `now - last_inference_at_` against `idle_minutes × 60`

**Task dispatch:**
- Maximum one concurrent task (`max_concurrent_tasks=1`)
- Safety whitelist check before dispatch
- Spawns a detached `std::thread` for execution
- Updates `task_runs` and `task_action_logs` in SQLite after completion
- Calls `store_result_to_episodic()` for MEMORY/BOTH output targets

**Action handlers:**
- `AGENT_RUN` — constructs `AgentGoal`, calls `AgentExecutor::run()` with a fresh `TraceBuilder`
- `CHAT` — builds `InferenceRequest`, calls `InferencePipeline::run()`
- `REFLECT` — calls `SelfImprovementLoop::trigger_reflection()`
- `TRAIN` — calls `SelfImprovementLoop::trigger_training(domain_hint)`
- `SHELL` — `popen()` with blocked_commands safety check
- `WEBHOOK` — `curl` subprocess POST with JSON payload

### 11.8 `schedule_task` Tool

**File:** `src/tools/builtin/computer/tool_schedule_task.h/.cpp`

Registered when `scheduler.enabled=true` (works headless — no display required).

**Actions:** `create`, `list`, `enable`, `disable`, `delete`, `run_now`

**Tool call examples:**
```json
{"name": "schedule_task", "arguments": {"action": "create", "description": "search for AI news every morning at 7am"}}
{"name": "schedule_task", "arguments": {"action": "list"}}
{"name": "schedule_task", "arguments": {"action": "run_now", "task_id": "abc12345"}}
```

---

## 12. Computer Use Subsystem (v1.5.0)

### 12.1 Overview

Cardinal can see and operate a desktop environment. The subsystem is initialised when `computer_use.enabled=true`. Display server (X11 / Wayland / headless) is detected at runtime by `DisplayDetector`.

All computer use controllers are owned by `CardinalAPI` and wired into `ToolExecutor` via setters after initialisation. The `ToolExecutor::dispatch()` function routes tool calls (`screenshot`, `click`, `type_text`, etc.) to their respective `execute_*()` implementations.

**Files:** `src/computer/`

### 12.2 DisplayDetector

**File:** `src/computer/display_detector.h/.cpp`

Detects the display server by checking environment variables and probing live connections:

1. Check `WAYLAND_DISPLAY` — if set and socket exists → Wayland
2. Check `DISPLAY` — if set and X server responds → X11
3. Otherwise → headless

Provides `DisplayDetector::info()` returning `ScreenInfo` with `server`, `display_var`, `width`, `height`, `scale_factor`.

Methods used by controllers:
- `is_x11()`, `is_wayland()`, `is_headless()`
- `has_scrot()`, `has_grim()` — tool availability checks

### 12.3 ScreenReader

**File:** `src/computer/screen_reader.h/.cpp`

**`capture(analyze)` → `Screenshot`:**
- X11: `scrot <path>` subprocess
- Wayland: `grim <path>` subprocess
- If `analyze=true` and `VisionEncoder` is ready: calls `VisionEncoder::encode(path, prompt, meta)` with `ImageMetadata{origin=path, cache_path=path, source=ImageSource::FILE}`. Result stored in `Screenshot::description`.

**`capture_region(ScreenRegion, analyze)`:** passes `-a x,y,w,h` to scrot or `-g "x,y WxH"` to grim.

**`analyze(image_path, prompt)` → `std::string`:** direct vision encode call. Returns `VisionResult::description`.

**`find_element(description)` → `std::optional<Point>`:** captures a screenshot, calls vision with a coordinate-extraction prompt: `"Find the UI element described as: '...' Respond ONLY with: x=NNN y=NNN"`. Parses `x=NNN y=NNN` from the response.

### 12.4 InputController

**File:** `src/computer/input_controller.h/.cpp`

**X11 (xdotool):**
- `mouse_click(x, y, button, clicks)` — `xdotool mousemove --sync x y click --clearmodifiers button` (repeated `clicks` times)
- `type_text(text)` — `xdotool type --clearmodifiers "text"`
- `send_key(key)` — `xdotool key --clearmodifiers "key"`
- `mouse_scroll(x, y, dx, dy)` — `xdotool scroll`

**Wayland (ydotool + wtype):**
- `wl_mouse_click(x, y, button, clicks)` — `ydotool mousemove abs x y` + `ydotool click button`
- `type_text(text)` — `wtype "text"`
- `send_key(key)` — `ydotool key keycode`

Display server chosen at call time based on `DisplayDetector::is_x11()`.

### 12.5 AppController

**File:** `src/computer/app_controller.h/.cpp`

| Method | X11 | Wayland |
|--------|-----|---------|
| `open_app(name)` | `gtk-launch name.desktop` then `exec name` fallback | same |
| `close_app(name)` | `wmctrl -c name` then `pkill name` | `swaymsg [app_id=name] kill` |
| `focus_app(name)` | `wmctrl -a name` | `swaymsg [app_id=name] focus` |
| `list_apps()` | `wmctrl -l` parsed | `swaymsg -t get_tree` parsed |
| `get_app(name)` | fuzzy match on list_apps | same |
| `get_focused_app()` | `xprop -root _NET_ACTIVE_WINDOW` | `swaymsg -t get_tree` |

`is_app_allowed(name)` checks against `computer_use.safety.allowed_apps`. If the list is empty, all apps are allowed.

### 12.6 BrowserController

**File:** `src/computer/browser_controller.h/.cpp`

Spawns a persistent Python helper process using the Playwright library from the browser venv. The helper communicates via stdin/stdout JSON lines and is kept alive for the session to avoid Playwright startup overhead on every action.

**Helper process lifecycle:**
- Started lazily on first `execute()` call via `ensure_started()`
- `start()` uses `venv_python(venv_path)` to find the interpreter
- `stop()` writes `{"action": "exit"}` then waits for process exit

**All actions go through `execute(BrowserAction)`.** Convenience wrappers (`navigate`, `click`, `click_text`, `type`, `get_content`, `screenshot`, `execute_js`) build `BrowserAction` and call `execute()`.

**`BrowserActionType` enum:** `NAVIGATE`, `CLICK`, `CLICK_TEXT`, `TYPE`, `SCROLL`, `GET_CONTENT`, `SCREENSHOT`, `EXECUTE_JS`, `NEW_TAB`, `CLOSE_TAB`, `BACK`, `FORWARD`, `RELOAD`

**Domain safety:** `is_domain_allowed(url)` checks against `allowed_domains` / `blocked_domains`. If `allowed_domains` is empty, all domains are permitted.

**Vision fallback:** if a selector-based click fails and `VisionEncoder` is available, falls back to screenshot + vision coordinate extraction.

### 12.7 ShellExecutor

**File:** `src/computer/shell_executor.h/.cpp`

Runs commands via `popen()` with a configurable timeout. The timeout is implemented with a background thread that calls `kill(pid, SIGKILL)` if `timeout_seconds > 0` and the process exceeds it.

**Safety checks before execution:**
1. `shell.enabled` must be true in config
2. Command must not contain any `blocked_commands` substring
3. Working directory changed to `working_directory` if non-empty

**Returns `ShellResult`:** `success`, `exit_code`, `stdout_text`, `stderr_text`, `duration_ms`, `command`.

### 12.8 FileManager

**File:** `src/computer/file_manager.h/.cpp`

All operations validate the resolved path against `computer_use.safety.allowed_paths`. Tilde expansion via `expand_home()`.

| Method | Description |
|--------|-------------|
| `list(path, recursive)` | Returns `FileOpResult` with `entries` vector |
| `move(src, dst)` | `std::filesystem::rename` |
| `copy(src, dst)` | `std::filesystem::copy` with `overwrite_existing` |
| `remove(path)` | `std::filesystem::remove_all` |
| `mkdir(path)` | `std::filesystem::create_directories` |
| `stat(path)` | Returns size, permissions, modified time |
| `exists(path)` | `std::filesystem::exists` |

**`FileOpResult`** contains: `success`, `error_message`, `entries` (vector of `FileEntry`), `dest_path`.

**`FileEntry`**: `name`, `path`, `is_dir`, `size_bytes`, `permissions`, `modified_at`.

File writes (`allow_file_write` config flag) are enforced via `is_write_allowed()`.

### 12.9 SystemController

**File:** `src/computer/system_controller.h/.cpp`

| Method | Tool used |
|--------|-----------|
| `get_state()` | `pactl get-sink-volume`, `/proc/net/wireless`, `rfkill list` |
| `set_volume(pct)` | `pactl set-sink-volume @DEFAULT_SINK@ pct%` |
| `set_mute(bool)` | `pactl set-sink-mute @DEFAULT_SINK@ 0/1` |
| `set_brightness(pct)` | `brightnessctl set pct%` |
| `set_wifi(bool)` | `nmcli radio wifi on/off` |
| `set_bluetooth(bool)` | `bluetoothctl power on/off` |
| `set_notifications(bool)` | `gsettings set org.gnome.desktop.notifications show-banners true/false` |

`get_state()` returns a `SystemState` struct: `volume_pct`, `muted`, `brightness_pct`, `wifi_enabled`, `wifi_ssid`, `bluetooth_enabled`, `battery_pct`, `battery_charging`.

### 12.10 EmailController

**File:** `src/computer/email_controller.h/.cpp`

Two modes, selected by `computer_use.email.mode`:

**`imap_smtp` mode:** Python subprocess using `imaplib` and `smtplib`. Password from `CARDINAL_EMAIL_PASS` environment variable — never in config.

**`gmail_api` mode:** Python subprocess using `google-api-python-client`. Credentials from `gmail_credentials_path`. OAuth token auto-refreshed.

**`read(EmailQuery)` → `vector<EmailMessage>`**

`EmailQuery` fields: `folder`, `subject_contains`, `from_contains`, `unread_only`, `max_results`.

`EmailMessage` fields: `id`, `message_id`, `from`, `to`, `subject`, `date`, `body_text`, `body_html`, `unread`.

**`send(EmailSendRequest)` → `bool`**

`EmailSendRequest` fields: `to` (vector), `cc` (vector), `subject`, `body`, `html_body`.

### 12.11 AtSpiReader

**File:** `src/computer/atspi_reader.h/.cpp`

Reads application accessibility trees via a Python subprocess using `pyatspi`. The Python script is built using `std::string +=` concatenation (not raw string literals with operator `+`, which causes pointer arithmetic errors with `const char[]`).

**`get_tree(app_name)` → `std::optional<AtSpiNode>`**

Walks the AT-SPI2 accessibility tree for the named application up to depth 8.

**`find_nodes(app_name, role, name_contains)` → `vector<AtSpiNode>`**

Searches the tree for nodes matching role (e.g. `"push button"`) and optional name substring.

**`AtSpiNode`:** `role`, `name`, `bounds` (x,y,width,height), `states` (vector), `children` (vector, recursive).

Used as the primary method for element location before falling back to vision-based coordinate finding.

### 12.12 Computer Use Tools (v1.5.0)

All registered when `computer_use.enabled=true`. Each has a `make_*_tool_def(const CardinalConfig&)` factory function.

| Tool | File | Key arguments |
|------|------|---------------|
| `screenshot` | `tool_screenshot.cpp` | `analyze`, `prompt`, `region_x/y/w/h` |
| `click` | `tool_click.cpp` | `description` OR `x`+`y`, `button`, `double_click`, `app` |
| `type_text` | `tool_type_text.cpp` | `text` OR `key`, `delay_ms` |
| `open_app` | `tool_open_app.cpp` | `app`, `focus` |
| `close_app` | `tool_close_app.cpp` | `app` |
| `browser` | `tool_browser.cpp` | `action`, `url`, `selector`, `text`, `script`, `scroll_y` |
| `shell_run` | `tool_shell_run.cpp` | `command`, `timeout_seconds`, `working_dir` |
| `file_ops` | `tool_file_ops.cpp` | `action`, `path`, `dest`, `recursive` |
| `system_control` | `tool_system_control.cpp` | `action`, `value` |
| `email` | `tool_email.cpp` | `action`, `folder`, `subject`, `from`, `to`, `body` |
| `watch_screen` | `tool_watch_screen.cpp` | `wait_for`, `timeout_seconds`, `poll_seconds`, `analyze` |
| `schedule_task` | `tool_schedule_task.cpp` | `action`, `description`, `task_id` |

**Browser and email are only registered if their respective sub-configs are non-empty.** Shell is only registered if `shell.enabled=true`.

---

## 13. Watch Subsystem (v1.5.0)

**Files:** `src/watch/`

Three independent watchers providing event-driven observation.

### 13.1 FileWatcher

**File:** `src/watch/file_watcher.h/.cpp`

inotify-based file system monitoring. Configured via `FileWatchConfig`: `path`, `recursive`, `events` (CREATE/MODIFY/DELETE/MOVE), `callback`.

### 13.2 ScreenWatcher

**File:** `src/watch/screen_watcher.h/.cpp`

Periodic screenshot diff using ImageMagick `compare -metric PSNR`. Configured via `ScreenWatchConfig`: `poll_interval_seconds`, `psnr_threshold`, `region` (optional), `callback`.

The `watch_screen` tool wraps this with a blocking wait — it polls until PSNR drops below threshold (visual change detected) or timeout expires.

### 13.3 ProcessWatcher

**File:** `src/watch/process_watcher.h/.cpp`

`/proc` polling for process start/stop events. Configured via `ProcessWatchConfig`: `process_name`, `poll_interval_seconds`, `on_start_callback`, `on_stop_callback`.

---

## 14. Agentic Pipeline

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

`AgentGoal` fields: `session_id`, `goal`, `max_iterations`, `stream`.

**v1.5.0:** All computer use tools are available within the agentic loop. When Cardinal runs an `AGENT_RUN` task from the scheduler, it constructs an `AgentGoal` and calls `AgentExecutor::run()` with a fresh `TraceBuilder("scheduler", "scheduler", "")`.

---

## 15. Tools System

### Registered tools (v1.5.0)

| Tool | Condition |
|------|-----------|
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
| `schedule_task` | `scheduler.enabled` (headless-safe) |

### ToolExecutor dispatch

`ToolExecutor::dispatch()` routes by tool name. Computer use controllers are injected via setters called from `CardinalAPI::init()` after the computer use block is initialised:

```cpp
tool_executor_->set_screen_reader(screen_reader_.get());
tool_executor_->set_input_controller(input_controller_.get());
tool_executor_->set_app_controller(app_controller_.get());
tool_executor_->set_browser_controller(browser_controller_.get());
tool_executor_->set_shell_executor(shell_executor_.get());
tool_executor_->set_file_manager(file_manager_.get());
tool_executor_->set_system_controller(system_controller_.get());
tool_executor_->set_email_controller(email_controller_.get());
tool_executor_->set_scheduler(scheduler_.get());
```

---

## 16. Explainability Exports

Unchanged from v1.2.0. Every inference produces a signed, tamper-evident trace:

```json
{
  "inference_id": "uuid",
  "reasoning_trace": {
    "feeling_output": {...},
    "tool_calls": [...],
    "rules_applied": [...],
    "contradictions": [...]
  },
  "integrity": {
    "hash": "sha256:...",
    "signature": "ed25519:..."
  }
}
```

**v1.4.0 note:** Meta-correction rules appear in `rules_applied` with `reasoning_type = "meta_correction"`.
**v1.5.0 note:** Scheduled task inferences do not produce audit log entries (no AuditLog pointer available in SchedulerEngine). Computer use tool calls appear in the normal inference trace when invoked from chat or the agentic loop.

---

## 17. Training Export

`TrainingExporter` exports high-confidence episodes as Alpaca JSONL for external use. This is the manual export path. Layer 3 uses `DatasetCurator` directly.

**Filter parameters:** `min_confidence`, `domain`, `max_examples`, `include_rules`, `recent_first`.

---

## 18. Configuration Reference

### 18.1 `backend` (unchanged)

`type` (`"llama_cpp"` or `"tensorrt"`), `llama_cpp.*`, `tensorrt.*`.

### 18.2 `inference` (unchanged)

`temperature`, `top_p`, `max_tokens_feeling`, `max_tokens_response`.

### 18.3 `feeling_schema` (unchanged)

`type`, `grammar_path`, `fields`, `max_tokens`.

### 18.4 `memory` (unchanged)

`rule_store_path`, `knowledge_graph_path`, `episodic_log_path`, `max_rules`.

### 18.5 `verifier` (unchanged)

`mode`, `neural_model_path`, `contradiction_threshold`, `rule_confidence_decay`, `min_rule_confidence`.

### 18.6 `retriever` (unchanged)

`mode`, `keyword_weight`, `semantic_weight`, `max_results`, `min_score`.

### 18.7 `tools` (unchanged)

Per-tool `enabled`, `confirmation_required`, and tool-specific parameters.

### 18.8 `agent` (unchanged)

`enabled`, `max_iterations`, `self_correction_enabled`, `plan_before_execute`.

### 18.9 `explainability` (unchanged)

`enabled`, `audit_log_path`, `signing_enabled`, `private_key_path`, `public_key_path`.

### 18.10 `vision` (v1.3.0, unchanged)

`model_path`, `mmproj_path`, `gpu_layers`, `threads`, `max_tokens`, `cache_path`, `cache_ttl_hours`, `allowed_paths`.

### 18.11 `self_improvement` (v1.4.0, unchanged)

See v1.4.0 documentation for full reference.

### 18.12 `api` (unchanged)

`http_enabled`, `host`, `port`, `auth_enabled`, `api_key`, `stream_enabled`.

### 18.13 `scheduler` (v1.5.0) ← NEW

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

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable the scheduler engine and `schedule_task` tool |
| `db_path` | string | SQLite database path |
| `check_interval_seconds` | int | How often the engine ticks (evaluates triggers) |
| `max_concurrent_tasks` | int | Maximum tasks running simultaneously (currently 1) |
| `idle_threshold_minutes` | int | Minutes without inference before idle trigger fires |
| `task_session_prefix` | string | Prefix for sessions created by scheduled tasks |
| `run_history_max_entries` | int | Maximum run history entries retained per task |
| `max_task_duration_seconds` | int | Hard timeout for any single task run |

### 18.14 `computer_use` (v1.5.0) ← NEW

```json
{
  "computer_use": {
    "enabled": true,
    "safety": {
      "whitelist_enabled": true,
      "allowed_apps": ["google-chrome", "firefox", "nautilus"],
      "allowed_domains": [],
      "allowed_paths": ["~/Documents", "~/Downloads", "~/Desktop", "data/"],
      "blocked_commands": ["rm -rf /", "mkfs", ":(){:|:&};:"],
      "confirmation_required": true,
      "confirmation_timeout_seconds": 30,
      "full_autonomy": false,
      "allow_file_write": false
    },
    "screen": {
      "screenshot_tool": "auto",
      "vision_analysis": true,
      "watch_interval_seconds": 2
    },
    "browser": {
      "executable": "google-chrome",
      "venv_path": "~/cardinal/cardinal-browser-venv",
      "playwright_timeout_ms": 10000,
      "headless": false,
      "user_data_dir": "data/browser_profile"
    },
    "shell": {
      "enabled": true,
      "shell": "/bin/bash",
      "timeout_seconds": 30,
      "working_directory": "~"
    },
    "email": {
      "enabled": false,
      "mode": "imap_smtp",
      "imap_host": "",
      "imap_port": 993,
      "smtp_host": "",
      "smtp_port": 587,
      "address": "",
      "gmail_api_enabled": false,
      "gmail_credentials_path": "data/gmail_credentials.json"
    },
    "atspi": {
      "enabled": true,
      "fallback_to_vision": true
    }
  }
}
```

**`safety` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `whitelist_enabled` | bool | Enforce `allowed_apps` whitelist |
| `allowed_apps` | array | App names/executables permitted. Empty = all allowed |
| `allowed_domains` | array | Browser domains permitted. Empty = all allowed |
| `allowed_paths` | array | File paths accessible to file_ops and shell |
| `blocked_commands` | array | Shell substrings that trigger immediate abort |
| `confirmation_required` | bool | Require human confirmation before executing |
| `confirmation_timeout_seconds` | int | Seconds to wait for confirmation |
| `full_autonomy` | bool | Override all confirmation requirements |
| `allow_file_write` | bool | Permit write operations via file_ops tool |

**`browser` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `venv_path` | string | Path to Python venv with Playwright installed |
| `playwright_timeout_ms` | int | Default action timeout in ms |
| `headless` | bool | Run browser without visible window |
| `user_data_dir` | string | Browser profile directory for session persistence |

---

## 19. CardinalAPI Reference

### 19.1 Lifecycle

```cpp
CardinalVoidResult init(const std::string& config_path = "config.json");
CardinalVoidResult shutdown();
```

### 19.2 Session Management (unchanged)

`create_session()`, `destroy_session()`, `reset_session()`, `get_session()`, `list_sessions()`.

### 19.3 Inference (unchanged signature)

```cpp
CardinalResult<ChatResponse> chat(const std::string& session_id,
                                   const std::string& message);
CardinalResult<ChatResponse> chat_stream(const std::string&       session_id,
                                          const std::string&       message,
                                          const ApiStreamCallback& stream_cb);
```

### 19.4 Agentic Inference (unchanged)

`agent(session_id, goal, max_iterations)`.

### 19.5 Memory & Stats (unchanged)

`get_stats()`, `get_rules()`, `get_episodes()`, `run_scan()`, `run_maintenance()`.

### 19.6 Self-Improvement (v1.4.0, unchanged)

`get_self_model_status()`, `reflect()`, `trigger_training(domain_hint)`, `on_session_boundary()`.

### 19.7 Scheduler API (v1.5.0) ← NEW

```cpp
CardinalResult<SchedulerStatus>              get_scheduler_status() const;
CardinalResult<std::vector<ScheduledTask>>   list_tasks() const;
CardinalResult<ScheduledTask>                get_task(const std::string& task_id) const;
CardinalResult<TaskParseResult>              create_task(const std::string& nl_description,
                                                          const std::string& session_id = "");
CardinalResult<std::string>                  create_task_direct(const ScheduledTask& task);
CardinalVoidResult                           update_task(const ScheduledTask& task);
CardinalVoidResult                           delete_task(const std::string& task_id);
CardinalVoidResult                           enable_task(const std::string& task_id);
CardinalVoidResult                           disable_task(const std::string& task_id);
CardinalResult<std::string>                  run_task_now(const std::string& task_id);
CardinalResult<std::vector<TaskRun>>         get_task_history(const std::string& task_id,
                                                               int limit = 50) const;
CardinalResult<std::vector<TaskRun>>         get_recent_runs(int limit = 100) const;
CardinalResult<std::vector<TaskActionLog>>   get_run_action_logs(const std::string& run_id) const;
```

### 19.8 Computer Use API (v1.5.0) ← NEW

```cpp
CardinalResult<ScreenInfo>     get_computer_status() const;
CardinalResult<Screenshot>     take_screenshot(bool analyze = true,
                                                const std::string& prompt = "");
CardinalResult<std::string>    computer_click(int x, int y,
                                               const std::string& description = "");
CardinalResult<std::string>    computer_type(const std::string& text,
                                              const std::string& key = "");
CardinalResult<ShellResult>    computer_shell(const std::string& command,
                                               int timeout_seconds = 0);
```

---

## 20. HTTP API Reference

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
| POST | `/api/export` | Export training data (Alpaca JSONL) |
| GET | `/api/self_model` | Self-model status (v1.4.0) |
| POST | `/api/reflect` | Trigger reflection (v1.4.0) |
| POST | `/api/train` | Trigger training (v1.4.0) |

### Scheduler endpoints (v1.5.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scheduler/status` | Engine status, run counts |
| GET | `/api/scheduler/tasks` | List all tasks |
| POST | `/api/scheduler/tasks` | Create task (NL `description` field OR direct JSON) |
| GET | `/api/scheduler/tasks/:id` | Get task by ID |
| PUT | `/api/scheduler/tasks/:id` | Update task |
| DELETE | `/api/scheduler/tasks/:id` | Delete task |
| POST | `/api/scheduler/tasks/:id/run` | Run task immediately |
| POST | `/api/scheduler/tasks/:id/enable` | Enable task |
| POST | `/api/scheduler/tasks/:id/disable` | Disable task |
| GET | `/api/scheduler/tasks/:id/history` | Run history (`?limit=N`) |
| GET | `/api/scheduler/runs` | Recent runs across all tasks (`?limit=N`) |
| GET | `/api/scheduler/runs/:id/actions` | Action log for a run |

**Create task from NL:**
```bash
curl -X POST http://127.0.0.1:8080/api/scheduler/tasks \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"description": "search for AI news every morning at 7am and save to a file"}'
```

**Response:**
```json
{
  "success": true,
  "confidence": 0.91,
  "task": {
    "id": "3f8a1b2c-...",
    "name": "Daily AI News Search",
    "enabled": true,
    "trigger": {"type": "cron", "cron_expression": "0 7 * * *"},
    "action": {"type": "agent_run", "goal": "search for AI news", "output_target": "file"}
  }
}
```

**Create task directly (no NL parsing):**
```bash
curl -X POST http://127.0.0.1:8080/api/scheduler/tasks \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hourly maintenance",
    "trigger": {"type": "interval", "interval_seconds": 3600},
    "action": {"type": "maintenance"}
  }'
```

### Computer Use endpoints (v1.5.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/computer/status` | Display server, resolution |
| POST | `/api/computer/screenshot` | Capture screen |
| POST | `/api/computer/click` | Click by coords or description |
| POST | `/api/computer/type` | Type text or key combo |
| POST | `/api/computer/shell` | Run shell command |

**Screenshot:**
```bash
curl -X POST http://127.0.0.1:8080/api/computer/screenshot \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"analyze": true, "prompt": "What applications are open?"}'
```

**Response:**
```json
{
  "path": "data/screenshots/screen_20260524_165312.png",
  "width": 1920, "height": 1080,
  "analyzed": true,
  "description": "The screen shows a terminal window with cardinal running..."
}
```

---

## 21. Settings Manager (unchanged)

Runtime-mutable settings propagate immediately. `SettingsManager` owns retriever mode/weights, verifier mode, rule thresholds, sampling parameters, log level, agent limits.

---

## 22. Session Manager (unchanged)

`SessionManager` owns `ConversationSession` objects. Each tracks `turn_count`, `history`, `working_memory`, timestamps. `destroy_session()` calls `on_session_boundary()` for LoRA adapter application.

---

## 23. Type Reference

All public types in `src/api/cardinal_types.h`.

### New types (v1.5.0) — `src/scheduler/scheduler_types.h`

**`TriggerType` enum:** `CRON`, `INTERVAL`, `CONDITION`, `STARTUP`, `IDLE`, `MANUAL`

**`TriggerSpec`:**
```cpp
struct TriggerSpec {
    TriggerType type = TriggerType::MANUAL;
    std::string cron_expression;
    int         interval_seconds = 0;
    std::string condition_expr;
    int         idle_minutes     = 30;
};
```

**`TaskActionType` enum:** `AGENT_RUN`, `CHAT`, `REFLECT`, `TRAIN`, `SELF_IMPROVEMENT`, `MAINTENANCE`, `EXPORT`, `SHELL`, `WEBHOOK`

**`OutputTarget` enum:** `MEMORY`, `FILE`, `WEBHOOK`, `DISCARD`, `BOTH`

**`TaskAction`:**
```cpp
struct TaskAction {
    TaskActionType type = TaskActionType::AGENT_RUN;
    std::string    goal;
    int            max_iterations = 0;
    std::string    domain_hint;
    std::string    shell_command;
    std::string    webhook_url;
    OutputTarget   output_target = OutputTarget::MEMORY;
    std::string    output_file;
};
```

**`ScheduledTask`:** (see §11.2)

**`TaskRunStatus` enum:** `PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `TIMEOUT`, `CANCELLED`

**`TaskRun`:**
```cpp
struct TaskRun {
    std::string    run_id;
    std::string    task_id;
    std::string    task_name;
    TaskRunStatus  status = TaskRunStatus::PENDING;
    std::string    started_at;
    std::string    finished_at;
    std::string    result_summary;
    std::string    error_message;
    int            duration_ms = 0;
    std::string    output_path;
    std::string    session_id;
    std::vector<TaskActionLog> action_log;
};
```

**`TaskActionLog`:**
```cpp
struct TaskActionLog {
    int         sequence = 0;
    std::string action_type;
    std::string description;
    std::string input_summary;
    std::string output_summary;
    bool        success               = false;
    bool        required_confirmation = false;
    bool        confirmation_granted  = false;
    int         duration_ms           = 0;
    std::string timestamp;
};
```

**`TaskParseResult`:**
```cpp
struct TaskParseResult {
    bool          success    = false;
    float         confidence = 0.0f;
    ScheduledTask task;
    std::string   error_message;
    std::string   clarification_needed;
};
```

**`SchedulerStatus`:**
```cpp
struct SchedulerStatus {
    bool        enabled;
    bool        running;
    int         total_tasks;
    int         enabled_tasks;
    int         total_runs;
    int         successful_runs;
    int         failed_runs;
    std::string current_task_id;
    std::string current_task_name;
    std::string last_run_at;
    std::string next_scheduled_at;
};
```

### New types (v1.5.0) — `src/computer/computer_types.h`

**`DisplayServer` enum:** `X11`, `WAYLAND`, `HEADLESS`

**`Point`:** `int x, y`

**`ScreenRegion`:** `int x, y, width, height`

**`ScreenInfo`:** `width`, `height`, `scale_factor`, `server`, `display_var`

**`Screenshot`:** `path`, `width`, `height`, `timestamp`, `analyzed`, `description`, `region`

**`ImageMetadata`:** `width`, `height`, `file_size`, `format`, `source`, `origin`, `cache_path`

**`MouseButton` enum:** `LEFT`, `RIGHT`, `MIDDLE`

**`BrowserActionType` enum:** `NAVIGATE`, `CLICK`, `CLICK_TEXT`, `TYPE`, `SCROLL`, `GET_CONTENT`, `SCREENSHOT`, `EXECUTE_JS`, `NEW_TAB`, `CLOSE_TAB`, `BACK`, `FORWARD`, `RELOAD`

**`BrowserResult`:** `success`, `error_message`, `url`, `content`, `screenshot_path`, `duration_ms`

**`ShellResult`:** `success`, `exit_code`, `stdout_text`, `stderr_text`, `duration_ms`, `command`

**`FileOpResult`:** `success`, `error_message`, `entries` (`vector<FileEntry>`), `dest_path`

**`FileEntry`:** `name`, `path`, `is_dir`, `size_bytes`, `permissions`, `modified_at`

**`SystemState`:** `volume_pct`, `muted`, `brightness_pct`, `wifi_enabled`, `wifi_ssid`, `bluetooth_enabled`, `battery_pct`, `battery_charging`

**`EmailQuery`:** `folder`, `subject_contains`, `from_contains`, `unread_only`, `max_results`

**`EmailMessage`:** `id`, `message_id`, `from`, `to`, `subject`, `date`, `body_text`, `body_html`, `unread`

**`EmailSendRequest`:** `to` (vector), `cc` (vector), `subject`, `body`, `html_body`

**`AppInfo`:** `window_id`, `name`, `title`, `pid`, `focused`

**`AtSpiNode`:** `role`, `name`, `bounds` (x,y,w,h), `states` (vector), `children` (vector)

### Existing types (unchanged from v1.4.0)

`CardinalStatus`, `CardinalResult<T>`, `CardinalVoidResult`, `ChatResponse`, `FeelingInfo`, `SessionInfo`, `RuleInfo`, `EpisodeInfo`, `SystemStats`, `ExportInfo`, `ScanResult`, `StreamToken`, `DomainStats`, `SelfModelSnapshot`, `ReflectionFinding`, `ReflectionResult`, `SelfImprovementStatus`, `TrainingExample`.

---

## 24. Error Handling

No C++ exceptions cross the `CardinalAPI` boundary. All internal exceptions caught and converted to `CardinalResult<T>` or `CardinalVoidResult`.

**Non-fatal paths:** retrieval failure, neural verifier failure, tool execution failure, reflection pass LLM failure, training subprocess non-zero exit, adapter load failure below threshold, screenshot tool missing, browser process crash, shell command blocked, file path not allowed, email auth failure.

**Fatal paths** (cause `init()` to fail): missing config/model/grammar files, SQLite error on episode DB open, explainability key generation failure.

**v1.5.0 — Graceful degradation:**
- `computer_use.enabled=true` but display not detected → computer use controllers not initialised; `check_computer_use()` returns `COMPUTER_USE_ERROR`; tools not registered; rest of system unaffected
- `scheduler.enabled=true` but SQLite fails to open → scheduler not started; `check_scheduler()` returns `SCHEDULER_ERROR`; rest of system unaffected
- Browser venv missing → `browser` tool not registered (guarded by `venv_path.empty()` check)
- Email disabled → `email` tool not registered

---

## 25. Offline Builds & Vendoring

All vendor dependencies cloned manually — no git submodules.

```
vendor/
    llama.cpp/          git clone https://github.com/ggerganov/llama.cpp
    nlohmann_json/      git clone https://github.com/nlohmann/json
    cpp-httplib/        git clone https://github.com/yhirose/cpp-httplib
    muparser/           git clone https://github.com/beltoforion/muparser
    tokenizers-cpp/     git clone https://github.com/mlc-ai/tokenizers-cpp
```

CMake auto-detects `mtmd` from `vendor/llama.cpp/tools/mtmd/` and sets `CARDINAL_MTMD_AVAILABLE`.

**v1.5.0:** No new vendored C++ dependencies. Playwright (Python) is installed into `cardinal-browser-venv`. Email dependencies (google-auth etc.) optionally installed into the same venv.

---

## 26. Module Reference

### 26.1 VisionEncoder (v1.3.0)

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

### 26.2 SelfImprovementLoop (v1.4.0)

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

### 26.3 SchedulerEngine (v1.5.0)

```cpp
class SchedulerEngine {
    void start();
    void stop();
    bool is_running() const;
    void on_inference();  // updates idle tracker

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

### 26.4 SchedulerParser (v1.5.0)

```cpp
class SchedulerParser {
    // pipeline may be nullptr — parse() fails gracefully
    explicit SchedulerParser(InferencePipeline* pipeline,
                              const CardinalConfig& config,
                              float min_confidence = 0.70f);

    TaskParseResult parse(const std::string& nl_description,
                          const std::string& session_id = "");

    TaskParseResult parse_json(const std::string& json_str,
                               const std::string& session_id = "");

    // Public static — called directly by SchedulerEngine::create_task_from_nl()
    static TaskParseResult extract_from_json(const std::string& model_output,
                                              const std::string& session_id,
                                              float              min_confidence);

    static std::string build_system_prompt();
    static std::string build_user_message(const std::string& nl_description);
};
```

### 26.5 DisplayDetector (v1.5.0)

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

### 26.6 BrowserController (v1.5.0)

```cpp
class BrowserController {
    bool start();
    void stop();
    bool is_running() const;
    BrowserResult execute(const BrowserAction& action);
    BrowserResult navigate(const std::string& url);
    BrowserResult click(const std::string& selector);
    BrowserResult click_text(const std::string& visible_text);
    BrowserResult type(const std::string& selector, const std::string& text);
    BrowserResult get_content();
    BrowserResult screenshot(const std::string& save_path = "");
    BrowserResult execute_js(const std::string& js);
    bool is_domain_allowed(const std::string& url) const;
};
```

---

## 27. Threading Model

| Thread | Owner | Responsibility |
|--------|-------|----------------|
| Main / inference | CardinalAPI | `chat()`, `agent()`, all user-facing calls |
| HTTP server | HttpServer | Request handling, SSE streaming |
| Training | SelfImprovementLoop | Layer 3 training cycle (background) |
| Scheduler engine | SchedulerEngine | Trigger evaluation + task dispatch |
| Task execution | SchedulerEngine (detached) | Each task runs in a detached thread |

### Lock ordering

```
api_mutex_
  > session_mutex_
  > inference_mutex_
      > self_model.mutex_
      > meta_cognition.reflect_mutex_
      > training_mutex_ > trainer_mutex_ > pending_mutex_
  > scheduler cv_mutex_
      > scheduler idle_mutex_
      > scheduler status_mutex_
      > scheduler sim_mutex_
```

### Scheduler thread safety

`SchedulerEngine::on_inference()` is called on the inference thread and only acquires `idle_mutex_` briefly to update `last_inference_at_`. The engine thread never blocks the inference thread — task dispatch goes to a detached thread.

`SchedulerEngine::status_mutex_` guards current task name/ID, run counts, and timing strings — read by HTTP handlers, written by dispatch threads.

---

## 28. Lifecycle and Startup Sequence (v1.5.0)

```
main()
  |
  +-- Logger::init()
  +-- ConfigLoader::load("config.json")
  |
  +-- CardinalAPI::init()
        |
        +-- ILLMBackend (LlamaCppBackend or TensorRTBackend)
        |     +-- load_model(), create_context() × 2 (pass1, pass2)
        |
        +-- Memory subsystems
        |     +-- RuleStore::load()
        |     +-- KnowledgeGraph::load()
        |     +-- EpisodicMemory::open()
        |     +-- EpisodicStorage::open()  → JSONL migration if needed
        |     +-- EpisodicRetriever::build_index()
        |
        +-- Verifier pipeline
        |     +-- SymbolicEngine::init()  → SWI-Prolog
        |     +-- RuleExtractor::init()
        |     +-- NeuralVerifier::init()  (if enabled)
        |     +-- ConsistencyChecker::init()
        |
        +-- InferencePipeline::init()
        |
        +-- Tools & Agent
        |     +-- ToolRegistry::init()    → registers all base tools
        |     +-- ToolExecutor::init()
        |     +-- AgentExecutor::init()
        |
        +-- Vision (v1.3.0)
        |     +-- VisionCache::init()
        |     +-- VisionEncoder::load()   → ready or disabled
        |
        +-- Explainability
        |     +-- AuditLog::open()
        |     +-- key generation if auto_generate_keys=true
        |
        +-- API layer
        |     +-- TrainingExporter, SettingsManager, SessionManager
        |
        +-- Self-Improvement (v1.4.0)
        |     +-- SelfImprovementLoop::start()
        |           +-- SelfModel::open()
        |           +-- MetaCognition::init()
        |           +-- TrainingFactory::create()
        |           +-- training_thread_.start()
        |
        +-- Scheduler (v1.5.0)
        |     +-- SchedulerEngine constructed with SchedulerDeps
        |           (agent_executor, pipeline, self_improvement,
        |            episodic, sessions, inference_busy_fn)
        |     +-- SchedulerEngine::start()
        |           +-- SchedulerStore::open()  → creates scheduler.db if needed
        |           +-- Fires STARTUP tasks immediately
        |           +-- engine_thread_.start()
        |     +-- ToolRegistry registers schedule_task tool
        |
        +-- Computer Use (v1.5.0)
        |     +-- DisplayDetector::detect()  → X11 / Wayland / headless
        |     +-- ScreenReader constructed with DisplayDetector + VisionEncoder
        |     +-- InputController, AppController
        |     +-- BrowserController (lazy start — first use spawns helper process)
        |     +-- ShellExecutor, FileManager, SystemController
        |     +-- EmailController, AtSpiReader
        |     +-- ToolRegistry registers screenshot, click, type_text, ...
        |     +-- ToolExecutor wired via set_screen_reader(), set_scheduler(), ...
        |
        +-- initialized_.store(true)
        +-- LOG_INFO("CardinalAPI initialized — N tools registered")

  +-- HttpServer::start() (if http_enabled)
  +-- interactive loop
```

**Shutdown:**
```
CardinalAPI::shutdown()
  |
  +-- scheduler_->stop()      ← stop before subsystems it depends on
  |     +-- notify CV with stop_requested_=true
  |     +-- engine_thread_.join()  ← waits for current tick/dispatch
  |     +-- SchedulerStore::close()
  |
  +-- browser_controller_->stop()  ← shut down Playwright helper process
  |
  +-- self_improvement_->stop()
  |     +-- training_thread_.join()
  |     +-- SelfModel::close()  ← WAL checkpoint
  |
  +-- sessions_->destroy_all()
  +-- EpisodicStorage::close()
  +-- AuditLog::close()
  +-- RuleStore::save()
  +-- VisionEncoder::unload()
  +-- ILLMBackend::unload()
```

**Destruction order is guaranteed by `unique_ptr` member declaration order in `CardinalAPI`.** `scheduler_` and computer use controllers are declared after `self_improvement_`, which is declared after all foundation subsystems. C++ destructs members in reverse declaration order, so foundation subsystems outlive everything that depends on them.

---

*This documentation reflects Cardinal v1.5.0. The source of truth is always the source code.*
