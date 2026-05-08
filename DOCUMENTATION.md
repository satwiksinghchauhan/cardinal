# Cardinal v1.3.0 — Technical Documentation

**Architecture:** Neurosymbolic AGI Core + Agentic Loop + Explainability + Vision  
**Language:** C++20  
**Platform:** Linux (Ubuntu 24.04 LTS)  
**GPU:** NVIDIA CUDA (TensorRT / llama.cpp)  
**Vision:** moondream2 via `llama.cpp` `mtmd` subsystem

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
10. [Agentic Pipeline (Unified)](#10-agentic-pipeline-unified)
11. [Tools System](#11-tools-system)
12. [Explainability Exports](#12-explainability-exports)
13. [Training Export](#13-training-export)
14. [Configuration Reference](#14-configuration-reference)
15. [CardinalAPI Reference](#15-cardinalapi-reference)
16. [HTTP API Reference](#16-http-api-reference)
17. [Settings Manager](#17-settings-manager)
18. [Session Manager](#18-session-manager)
19. [Type Reference](#19-type-reference)
20. [Error Handling](#20-error-handling)
21. [Offline Builds & Vendoring](#21-offline-builds--vendoring)
22. [Module Reference](#22-module-reference)
23. [Threading Model](#23-threading-model)
24. [Lifecycle and Startup Sequence](#24-lifecycle-and-startup-sequence)

---

## 1. Architecture Overview

Cardinal is structured in four layers. Each layer depends only on the layers below it. No layer reaches upward.

```
Layer 4 -- Interfaces (in development)
    Interface 1: Chat (C++)
    Interface 2: Agent (TypeScript via HTTP)
    Interface 3: SEAL (Python via pybind11)

Layer 3 -- API Layer
    CardinalAPI        single facade, owns all components
    HttpServer         TypeScript bridge, SSE streaming
    SettingsManager    runtime‑mutable config
    SessionManager     multi‑session conversation state
    TrainingExporter   Alpaca JSONL export
    Explainability     audit log, cryptographic signing, export API

Layer 2 -- Core Systems
    InferencePipeline  two‑pass orchestrator, prompt injection
    AgentExecutor      PLAN → EXECUTE loop (THINK → ACT → OBSERVE) → FINALIZE
    ToolExecutor       sandboxed tool execution (subprocess or Docker)
    LLMEngine          abstract backend (llama.cpp / TensorRT)
    ConsistencyChecker verifier orchestrator, auto‑resolution
    SymbolicEngine     SWI‑Prolog integration
    NeuralVerifier     optional small LLM verifier
    RuleExtractor      NLP rule extraction

Layer 1 -- Foundation
    RuleStore          persistent rule base
    KnowledgeGraph     typed node graph
    EpisodicMemory     JSONL audit trail
    EpisodicStorage    SQLite + FTS5 searchable index
    EpisodicRetriever  TF‑IDF + keyword + hybrid retrieval
    ConfigLoader       typed config, validated at startup
    Logger             thread‑safe, 6 levels
    JsonParser         serialization utilities

Layer 1.5 -- Vision Subsystem (v1.3.0)
    VisionCache        URL download cache, TTL eviction
    VisionEncoder      moondream2 wrapper via mtmd API
```

### Component Ownership

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary — callers see only the types defined in `cardinal_types.h`.

### Dependency Graph (v1.3.0)

```
CardinalAPI
    owns --> LLMEngine
    owns --> InferencePipeline --> LLMEngine
                               --> EpisodicRetriever
    owns --> AgentExecutor --> InferencePipeline
                           --> ToolExecutor
    owns --> ToolExecutor
    owns --> RuleStore
    owns --> KnowledgeGraph
    owns --> EpisodicMemory
    owns --> EpisodicStorage
    owns --> EpisodicRetriever --> EpisodicStorage
    owns --> SymbolicEngine
    owns --> NeuralVerifier
    owns --> RuleExtractor --> RuleStore
                           --> SymbolicEngine
    owns --> ConsistencyChecker --> RuleStore
                                --> EpisodicMemory
                                --> SymbolicEngine
                                --> RuleExtractor
                                --> NeuralVerifier
    owns --> VisionCache (v1.3.0)
    owns --> VisionEncoder (v1.3.0) --> mtmd library
    owns --> TrainingExporter --> EpisodicStorage
                              --> RuleStore
    owns --> SettingsManager --> EpisodicRetriever
                             --> InferencePipeline
                             --> AgentExecutor
    owns --> SessionManager
    owns --> HttpServer (separate, explicit start)
```

---

## 2. Two-Pass Inference

Every inference Cardinal performs consists of exactly two passes through the language model. The two passes use separate inference contexts to prevent grammar state contamination.

### Pass 1 — Constrained Decoding (Feeling Output)

The first pass uses GBNF grammar‑constrained decoding to force the model to produce a structured JSON object before generating any natural language response. This JSON object is called the **feeling output**.

The grammar is defined in `src/prompts/feeling_schema.gbnf`. The grammar constraint means the model cannot produce malformed output — every token it generates must be valid JSON conforming to the feeling schema.

The feeling output captures six fields:
- `confidence` — how confident the model is in its response (0.0 to 1.0)
- `reasoning_type` — what kind of reasoning this response requires
- `reasoning_domain` — what domain this reasoning operates in
- `uncertainty_flag` — whether the model is uncertain
- `contradiction_flag` — whether the model has detected a conflict
- `rule_candidate_signal` — whether a general rule might be extractable

### Synthetic Turn Injection

After Pass 1 completes, the feeling output JSON is injected into the message history as a synthetic assistant turn. This is the key architectural insight: the model reads its own feeling output as something it already said, not as an instruction from outside.

### Pass 2 — Free Decoding (Final Response)

Pass 2 runs with no grammar constraint. The model generates its final natural language response. The stream callback, if provided, is called for each token as it is generated.

### Retry Logic

If Pass 1 fails to produce a valid feeling output (the JSON does not parse, or fails schema validation), the pipeline retries up to `max_retries` times with a `retry_delay_ms` delay between attempts. If all retries are exhausted, the inference fails and `ChatResponse` returns an error.

Pass 2 does not retry — if the response generation fails, the inference fails.

### State Machine

The `FeelingContext` object tracks the state of a single inference cycle:

```
IDLE → PASS1_FEELING → PASS2_RESPONSE → COMPLETE
                      → FAILED (on retry exhaustion)
```

### Metrics

Every `ChatResponse` carries inference metrics:
- `pass1_tokens` — tokens produced in Pass 1
- `pass2_tokens` — tokens produced in Pass 2
- `total_ms` — wall clock time for the full inference cycle

---

## 3. Feeling Output Schema

(The schema, fields, validation rules remain unchanged from v1.2.0. See v1.2.0 documentation for complete reference.)

**Fields:** `confidence`, `reasoning_type` (causal/deductive/inductive/abductive/analogical/associative), `reasoning_domain` (factual/ethical/spatial/temporal/social/mathematical), `uncertainty_flag`, `contradiction_flag`, `rule_candidate_signal`.

**Validation:** `confidence > 0.8` with `uncertainty_flag = true` is rejected as contradictory.

---

## 4. Memory Systems

### 4.1 RuleStore

Persistent symbolic memory. Rules have `id`, `domain`, `condition`, `consequence`, `confidence`, `trigger_count`, timestamps, and provenance (`episode_id`, `reasoning_type`).  
Storage: `data/memory/rules.json` (atomic writes).

### 4.2 KnowledgeGraph

Typed nodes: `concept`, `fact`, `entity`, `relation`.  
Storage: `data/memory/knowledge.json`.

### 4.3 EpisodicMemory

Append‑only JSONL audit trail (`logs/episodic.log`). Never modified after write.

### 4.4 EpisodicStorage

SQLite + FTS5 searchable index (`data/memory/episodes.db`). Fully synchronised with JSONL via migration on first open.

---

## 5. Retrieval System

`EpisodicRetriever` provides three modes:
- **KEYWORD** – SQLite FTS5, BM25 ranking.
- **SEMANTIC** – TF‑IDF cosine similarity (in‑memory index).
- **HYBRID** – weighted combination (default: 0.7 keyword, 0.3 semantic).

The TF‑IDF index is rebuilt according to `cache_rebuild_strategy` (on_demand/periodic/explicit).

Retrieved episodes are injected as a structured `[MEMORY CONTEXT]` block before the feeling output. If retrieval fails, inference continues without memory (non‑fatal).

---

## 6. Verifier Pipeline

Runs after every inference to maintain rule base integrity.

- **Modes:** `symbolic` (default, SWI‑Prolog), `neural` (requires small LLM), `hybrid`.
- **Per‑inference sequence:**  
  1. Rule extraction (if `rule_candidate_signal` true).  
  2. Contradiction check (if `contradiction_flag` true OR a rule was committed).  
  3. Periodic maintenance (every 10 inferences): confidence decay, prune, save.  
  4. Build summary (returned in `ConsistencyCheckResult.summary`).

`SymbolicEngine` wraps SWI‑Prolog 9.2.9 via the C foreign language interface.  
`NeuralVerifier` is optional (disabled if `neural_model_path` empty).

---

## 7. Rule System

**Rule extraction strategies (priority order):**
1. Causal patterns (`if…then`, `because`, `causes`, `leads to`). Confidence: 0.6.
2. Deductive patterns (`therefore`, `thus`, `hence`). Confidence: 0.65.
3. Declarative fallback (user message as condition, longest sentence as consequence). Confidence: 0.5.

**Contradiction auto‑resolution:**
```cpp
delta = abs(confidence_a - confidence_b)
if delta >= 0.2 → deprecate(lower confidence rule)
else → penalize both (-0.05), flag for review
```

**Confidence lifecycle:**  
- Add: 0.5–0.65 (depending on strategy).  
- Record trigger: +0.01.  
- Decay: -0.01 per maintenance cycle.  
- Prune: removed if below `min_rule_confidence` (default 0.3).

---

## 8. Backend Abstraction

Abstract `LLMEngine` interface. Implementations: `LlamaCppEngine`, `TensorRTEngine`.  
Only one backend loaded at runtime, chosen at compile time.

- **llama.cpp** – default, development, CUDA/CPU fallback.
- **TensorRT** – optimised for deployment (fixed input/output shapes, no GBNF).

Switching backends:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release                       # llama.cpp
cmake .. -DCMAKE_BUILD_TYPE=Release -DCARDINAL_ENABLE_TENSORRT=ON   # TensorRT
```

---

## 9. Vision Subsystem (v1.3.0)

### 9.1 Overview

The vision subsystem uses **moondream2** (a small vision‑language model) via the `mtmd` (multimodal) API in `llama.cpp`. It runs on CPU (4 threads by default) and produces natural language descriptions of images.

### 9.2 Components

- **`VisionEncoder`** – loads the text GGUF and the mmproj (projector) file, wraps `mtmd` C API, returns `VisionResult`.  
- **`VisionCache`** – downloads images from URLs, caches them with a configurable TTL (24h default, 0 = keep forever). Cache location: `data/vision_cache/`.

### 9.3 Tool Integration

The `analyze_image` tool (built‑in) routes through `VisionCache` → `VisionEncoder` and returns a description with a metadata header:

```
[image: data/test.jpg | 3840x2160 | 1MB]
The image shows a young man with glasses, a nose ring, looking directly at the camera...
```

**Tool parameters:**
- `image` – local file path or URL.

**Confirmation:** configurable via `tools.analyze_image.confirmation_required` (default `false`).

### 9.4 Configuration

```json
"vision": {
  "enabled": true,
  "model_path": "models/vision/moondream2-text-model-f16.gguf",
  "mmproj_path": "models/vision/moondream2-mmproj-f16.gguf",
  "cache_ttl_hours": 24,
  "allowed_paths": ["/home/doctor/Downloads", "/home/doctor/Pictures"]
}
```

If `model_path` or `mmproj_path` is empty or the files do not exist, the vision subsystem stays disabled (graceful fallback).

### 9.5 Performance Notes (RTX 3050 4GB)

- First inference of an image: ~9 seconds (encoding + LLM pass).
- Subsequent same image (cached): ~2‑3 seconds (LLM pass only).  
- Images are encoded on CPU; the primary LLM still uses GPU.

---

## 10. Agentic Pipeline (Unified)

Unchanged from v1.2.0, but now with vision tools available.

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

Configuration in `config.json` under `"agent"`.  
Working memory: SQLite‑backed, persistent across session turns.

---

## 11. Tools System

### 11.1 Built‑in Tools

| Tool | Description | Confirmation Required | Sandbox |
|------|-------------|----------------------|---------|
| `web_search` | DuckDuckGo search | Configurable | – |
| `web_fetch` | Fetch and parse a URL | Configurable | – |
| `calculator` | Safe math expression evaluator | No | – |
| `run_python` | Sandboxed Python execution | Yes (default) | Subprocess / Docker |
| `file_read` | Read from allowed paths | Yes (default) | Path restrictions |
| `file_write` | Write to allowed paths | Yes (default) | Path restrictions |
| `knowledge_graph_query` | Query Cardinal's KG | No | – |
| `episodic_search` | Search Cardinal's memory | No | – |
| **`analyze_image` (v1.3.0)** | Describe an image (file or URL) | Configurable (default false) | – |

### 11.2 analyze_image Tool (v1.3.0)

**Implementation:** `src/tools/builtin/analyze_image.cpp`.  
It uses `VisionCache` (for URLs) and `VisionEncoder` to get the description. The output format follows the Option C pattern (metadata header + plain text description).

**Configuration snippet:**
```json
"tools": {
  "analyze_image": {
    "enabled": true,
    "confirmation_required": false
  }
}
```

**Allowed paths:** controlled by `vision.allowed_paths`.

### 11.3 Python Sandbox Modes

- **`subprocess`** – spawn subprocess with ulimit (development).  
- **`docker`** – full container isolation (production, DRDO/ISRO deployment).

Limits (both modes): `timeout_seconds` (30s), `memory_limit_mb` (256MB), `network_enabled` (false).

---

## 12. Explainability Exports

Every inference produces a signed, tamper‑evident trace.

**Export schema (summary):**
```json
{
  "inference_id": "uuid",
  "timestamp": "2026-05-08T17:00:00Z",
  "query": "...",
  "reasoning_trace": {
    "feeling_output": {...},
    "episodes_retrieved": [...],
    "rules_active": [...],
    "symbolic_checks": {...},
    "tool_calls": [...],
    "pass1_tokens": 87,
    "pass2_tokens": 312,
    "total_ms": 1840
  },
  "rule_committed": {...},
  "final_response": "...",
  "integrity": {
    "hash": "sha256:...",
    "signature": "ed25519:..."
  }
}
```

**Storage:** `data/explainability/audit.db` (SQLite).  
**Signing:** Ed25519; keys auto‑generated on first start if `auto_generate_keys: true`.  
**Export API:** `POST /api/explainability/export` (by session_id, time range, etc.).

---

## 13. Training Export

Alpaca‑format JSONL files for LoRA fine‑tuning.

- Output format: `{"instruction": "...", "input": "", "output": "..."}`.
- Response cleaning: strips `<think>`, `<feeling_state>`, normalises whitespace.
- Optional rule export `include_rules`.
- Filters: `min_confidence`, `domain`, `max_examples`.

---

## 14. Configuration Reference

All configuration lives in `config.json`. Every field is validated at startup.

### 14.1 `model` section (unchanged)

`path`, `context_length`, `gpu_layers`, `threads`.

### 14.2 `inference` section (unchanged)

`temperature`, `top_p`, `max_tokens_feeling`, `max_tokens_response`, `max_retries`, `retry_delay_ms`.

### 14.3 `feeling_schema` section (unchanged)

`grammar_path`.

### 14.4 `memory` section (unchanged)

`rule_store_path`, `knowledge_graph_path`, `episodic_log_path`, `max_rules`.

### 14.5 `verifier` section (unchanged)

`mode`, `neural_model_path`, `neural_gpu_layers`, `contradiction_threshold`, `rule_confidence_decay`, `min_rule_confidence`.

### 14.6 `retriever` section (unchanged)

`mode`, `keyword_weight`, `semantic_weight`, `max_results`, `min_score`, `cache_rebuild_strategy`, `cache_rebuild_threshold`, `cache_rebuild_interval_seconds`.

### 14.7 `tools` section

Adds `analyze_image` subsection:

```json
"analyze_image": {
  "enabled": true,
  "confirmation_required": false
}
```

### 14.8 `agent` section (unchanged)

`enabled`, `max_iterations`, `max_iterations_hard_cap`, `working_memory_path`, `working_memory_size`, `self_correction_enabled`, `self_correction_max_attempts`, `plan_before_execute`, `summarize_on_cap`.

### 14.9 `explainability` section (unchanged)

`enabled`, `audit_log_path`, `signing_enabled`, `private_key_path`, `public_key_path`, `auto_generate_keys`, `export_path`, `attach_trace_to_response`.

### 14.10 `vision` section (v1.3.0)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Master switch for vision subsystem. |
| `model_path` | string | – | Path to moondream2 text GGUF. |
| `mmproj_path` | string | – | Path to moondream2 mmproj (projector). |
| `cache_ttl_hours` | int | `24` | Hours to keep cached images (`0` = never delete). |
| `allowed_paths` | array | `[]` | Local directories that `analyze_image` may read. |

If `model_path` or `mmproj_path` is empty or the files are missing, the vision subsystem is disabled (warnings logged, `analyze_image` returns an error).

### 14.11 `api` section (unchanged)

`http_enabled`, `host`, `port`, `auth_enabled`, `api_key`, `stream_enabled`.

### 14.12 `logging` section (unchanged)

`level`, `path`.

---

## 15. CardinalAPI Reference

`CardinalAPI` is the single entry point for all interfaces.  
**File:** `src/api/cardinal_api.h`

### 15.1 Lifecycle

```cpp
CardinalAPI api;
CardinalVoidResult init(const std::string& config_path = "config.json");
CardinalVoidResult shutdown();
bool is_initialized() const;
```

### 15.2 Session Management (unchanged)

`create_session()`, `destroy_session()`, `reset_session()`, `get_session()`, `list_sessions()`.

### 15.3 Inference

```cpp
CardinalResult<ChatResponse> chat(
    const std::string& session_id,
    const std::string& message);

CardinalResult<ChatResponse> chat_stream(
    const std::string&       session_id,
    const std::string&       message,
    const ApiStreamCallback& stream_cb);
```

### 15.4 Agentic Inference (unchanged)

`chat_agentic(session_id, goal, max_iterations)`.

### 15.5 Memory & Stats (unchanged)

`get_stats()`, `get_rules()`, `get_episodes()`, `run_scan()`, `run_maintenance()`.

### 15.6 Training Export (unchanged)

`export_training_data()`, `export_dry_run()`.

### 15.7 Explainability (unchanged)

`export_explainability_traces()`.

### 15.8 Settings (unchanged)

`get_settings()`, `update_settings()`, `set_setting()`, `reset_settings()`.

---

## 16. HTTP API Reference

Base URL: `http://127.0.0.1:8080`. Auth: Bearer token (except `/api/health`).

All endpoints from v1.2.0 remain. Notable additions:

- `POST /api/chat` – now accepts `analyze_image` tool calls automatically. The tool is registered as any other built‑in tool.
- The vision cache can be cleared by deleting the `data/vision_cache/` directory (Cardinal recreates it).

---

## 17. Settings Manager (unchanged)

Runtime‑mutable settings propagate immediately.  
`SettingsManager` owns `EpisodicRetriever` mode/weights, verifier mode, rule thresholds, sampling parameters, log level, agent limits.

---

## 18. Session Manager (unchanged)

`SessionManager` owns `ConversationSession` objects. Each session tracks `turn_count`, `history`, `working_memory` (agentic sessions), timestamps, and provides `trim_to_token_budget()`.

---

## 19. Type Reference

All types in `src/api/cardinal_types.h`. Pybind11‑friendly.

### `VisionResult` (v1.3.0)

```cpp
struct VisionResult {
    bool success;
    std::string description;     // natural language description
    std::string error_message;
    int width = 0;               // metadata (if available)
    int height = 0;
    std::string mime_type;
    size_t file_size_bytes = 0;
    std::chrono::milliseconds encode_ms;
};
```

### `ChatResponse` (unchanged, but includes `tool_calls` with `analyze_image` when vision is used)

---

## 20. Error Handling (unchanged)

No C++ exceptions cross the `CardinalAPI` boundary. All internal exceptions are caught and converted to `CardinalResult<T>`.

Non‑fatal paths: retrieval failure, neural verifier failure, contradiction check failure, tool execution failure (agentic self‑correction handles it), JSONL migration parse errors.

Fatal paths (cause `init()` to fail): missing config/model/grammar files, SQLite error, explainability key generation failure, `mtmd` library missing (if vision is enabled).

---

## 21. Offline Builds & Vendoring (v1.1.0, continued)

Full offline builds with vendored dependencies.  
`vendor/` contains: `llama.cpp`, `nlohmann_json`, `cpp-httplib`, `tokenizers-cpp`.  
System dependencies installed via `apt`.  
CMake detects `mtmd` from `vendor/llama.cpp/tools/mtmd/` and sets `CARDINAL_MTMD_AVAILABLE` automatically.

If `mtmd` is not found, vision is disabled (build continues, `analyze_image` returns an error).

---

## 22. Module Reference

### 22.1 VisionEncoder (`src/vision/vision_encoder.h/.cpp`)

```cpp
class VisionEncoder {
public:
    Result<> load();
    bool is_ready() const;
    Result<VisionResult> encode(const std::string& image_path);
};
```

Uses `mtmd` API via `mtmd.h`. Loads moondream2 text GGUF and projector. Runs on CPU (configurable threads).

### 22.2 VisionCache (`src/vision/vision_cache.h/.cpp`)

```cpp
class VisionCache {
public:
    Result<> init();
    Result<std::string> get_or_download(const std::string& url);
    void set_ttl_hours(int hours);
};
```

### 22.3 analyze_image Tool (`src/tools/builtin/analyze_image.cpp`)

Implements `Tool` interface. Calls `VisionCache` then `VisionEncoder`. Returns `VisionResult` as tool output.

---

## 23. Threading Model (unchanged)

- `VisionEncoder` operations are serialized through the agent loop (same as any other tool). No extra threading inside vision.
- `VisionCache` uses separate mutex for its own state (concurrent downloads are serialised).
- The rest of the threading model (inference mutex, agent mutex, session mutex, etc.) remains identical to v1.2.0.

---

## 24. Lifecycle and Startup Sequence (v1.3.0)

The startup sequence now also includes the vision subsystem:

```
main()
  |
  +-- Logger, config, memory, verifier, LLM engine, etc.
  |
  +-- ToolRegistry::register_all() (includes analyze_image)
  |
  +-- VisionCache::init()
  +-- VisionEncoder::load()
  |      |
  |      +-- if both models exist and mtmd available → ready
  |      +-- otherwise → disabled (warn)
  |
  +-- CardinalAPI: inject vision encoder/cache into ToolExecutor
  |
  +-- HttpServer start
  +-- interactive loop
```

**Shutdown:**  
`CardinalAPI::shutdown()` → `VisionCache` flush (not needed) → `VisionEncoder` cleanup (`mtmd_free`) → remaining components shut down.

---

*This documentation reflects Cardinal v1.3.0. The source of truth is always the source code.*
