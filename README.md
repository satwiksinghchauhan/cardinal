# Cardinal v1.3.0

**A production‑grade neurosymbolic AGI architecture with vision.**

Cardinal combines a large language model core with **vision understanding**, symbolic verification, persistent memory, hybrid retrieval, a **full agentic loop**, **explainability exports**, and a clean API layer. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

**Current version: v1.3.0** — Native vision encoder (moondream2) via `mtmd`, `analyze_image` tool, image cache, full multimodal support, TensorRT backend, Linux native, offline builds.

---

## What's New in v1.3.0

- **Native vision encoder** — Image → text descriptions using `moondream2` via `llama.cpp`’s `mtmd` subsystem.
- **`analyze_image` tool** – Ask Cardinal “what’s in this image?” through the chat or agentic loop.
- **Image cache** – Downloaded images (URLs) are cached; configurable TTL (default 24h, `0` = keep forever).
- **Multi‑modal inference** – Combine text + images in a single prompt; the vision encoder feeds tokens directly into the two‑pass pipeline.
- **Full offline support** – All dependencies still vendored; no internet required at build or runtime (except optional web fetch).
- **Still runs on a 3050 4GB** – Vision adds ~1.1GB for the text model + ~400MB for the projector. Latency ~9s per image, but it **works**.

---

## What Cardinal Is

Most LLM systems are stateless — each conversation starts from zero. Cardinal is different. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, and can export its accumulated knowledge as LoRA training data.

The architecture is **neurosymbolic** — it combines the pattern‑matching strength of neural inference with the consistency guarantees of symbolic logic. Neither alone is sufficient. Together they produce a system that reasons carefully, stays consistent over time, and improves with experience.

**With v1.3.0, Cardinal can also see.** It understands images, extracts descriptions, and reasons about visual content just as it does with text.

---

## Hardware

Cardinal was developed and runs on:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~5GB core + models (vision adds ~1.5GB) |
| OS | Ubuntu 24.04 LTS (Linux only) |

**Models (v1.3.0):**
- Primary LLM: `Qwen3.5 4B Q4_K_M` (via `llama.cpp` or TensorRT backend)
- Vision encoder: `moondream2` text model + `mmproj` (quantized, runs on CPU)

---

## Architecture (v1.3.0)

```
CardinalAPI
    |
    +-- Memory Layer
    |     +-- RuleStore          persistent rule base, confidence decay, Jaccard dedup
    |     +-- KnowledgeGraph     typed nodes, BFS traversal, hub detection
    |     +-- EpisodicMemory     append‑only JSONL audit trail
    |     +-- EpisodicStorage    SQLite + FTS5 searchable index, JSONL migration
    |     +-- EpisodicRetriever  TF‑IDF cosine + FTS5 keyword + hybrid retrieval
    |
    +-- Verifier Pipeline
    |     +-- SymbolicEngine     SWI‑Prolog 9.2.9 integration, contradiction detection
    |     +-- NeuralVerifier     optional small LLM verifier (Llama 3.2 1B)
    |     +-- RuleExtractor      NLP extraction, causal/deductive/declarative patterns
    |     +-- ConsistencyChecker orchestrator, auto‑resolution, maintenance cycles
    |
    +-- LLM Core (Backend Abstraction)
    |     +-- LLMEngine          Abstract interface: llama.cpp or TensorRT
    |     +-- llama.cpp backend  Development / CPU / CUDA fallback
    |     +-- TensorRT backend   Optimized deployment on NVIDIA GPUs
    |     +-- InferencePipeline  two‑pass orchestrator, prompt injection, retry logic
    |
    +-- Vision Subsystem (v1.3.0)
    |     +-- VisionEncoder      moondream2 wrapper, uses mtmd API
    |     +-- VisionCache        URL download cache, TTL eviction
    |
    +-- Agentic Pipeline (Unified)
    |     +-- AgentExecutor      PLAN → EXECUTE loop (THINK → ACT → OBSERVE) → FINALIZE
    |     +-- ToolExecutor       Sandboxed tool execution (subprocess or Docker)
    |     +-- WorkingMemory      SQLite‑backed persistent scratchpad
    |     +-- SelfCorrection     Retry failed steps, max attempts configurable
    |
    +-- Explainability
    |     +-- AuditLog           Every inference trace: feeling, tools, rules, symbolic checks
    |     +-- Cryptographic signing  SHA256 + Ed25519 (tamper‑evident)
    |     +-- Export API         JSON exports for compliance (DRDO/ISRO)
    |
    +-- API Layer
    |     +-- CardinalTypes      shared types, no exceptions at boundary
    |     +-- CardinalSettings   runtime‑mutable config, immediate propagation
    |     +-- SessionManager     multi‑session conversation state
    |     +-- TrainingExporter   Alpaca JSONL export for LoRA fine‑tuning
    |     +-- HttpServer         SSE streaming, Bearer auth, CORS, TypeScript bridge
    |
    +-- Tools (v1.2.0 + v1.3.0)
          +-- web_search, web_fetch, calculator, run_python, file_read/write
          +-- knowledge_graph_query, episodic_search
          +-- **analyze_image** (new in v1.3.0)
```

---

## Two‑Pass Inference (Unchanged, Still Core)

Every inference runs in two passes.

**Pass 1 — Feeling Output (constrained decoding)**
GBNF grammar forces the model to produce a structured JSON object before generating any response.

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
The feeling output is injected as a synthetic assistant turn. The model generates its final response with full awareness of its own internal state from Pass 1.

---

## Vision (v1.3.0)

### Configuration (`config.json`)

```json
{
  "vision": {
    "model_path":   "models/vision/moondream2-text-model-f16.gguf",
    "mmproj_path":  "models/vision/moondream2-mmproj-f16.gguf",
    "enabled": true,
    "cache_ttl_hours": 24,
    "allowed_paths": ["/home/user/Downloads", "/home/user/Pictures"]
  }
}
```

- `cache_ttl_hours` = 0 → never delete cached images.
- `allowed_paths` – local directories that `analyze_image` may read.

### Tool Usage

```
You: what is in data/test.jpg?

Cardinal: [Tool: analyze_image] The image shows a young man with glasses, a nose ring...
```

- Supports **local file paths** and **HTTP/HTTPS URLs**.
- Results are cached (by URL / file hash) for the configured TTL.

### Performance (RTX 3050 4GB)

- First image: ~9‑10 seconds (encoder + inference)
- Cached image (text only): ~2‑3 seconds

---

## Agentic Pipeline (v1.2.0, Still Here)

Same unified pipeline, now with vision tools available.

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

---

## Explainability Exports (v1.2.0)

Every inference produces a signed, tamper‑evident trace (full schema in `DOCUMENTATION.md`). Now includes vision tool calls.

```json
{
  "inference_id": "uuid",
  "reasoning_trace": {
    "feeling_output": {...},
    "tool_calls": [...],
    ...
  },
  "integrity": {
    "hash": "sha256:...",
    "signature": "ed25519:..."
  }
}
```

Configuration in `config.json` under `"explainability"`.

---

## Tools (v1.2.0 + v1.3.0)

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `web_search` | DuckDuckGo search | Configurable |
| `web_fetch` | Fetch and parse a URL | Configurable |
| `calculator` | Math expression evaluator | No |
| `run_python` | Sandboxed Python execution | **Yes** (default) |
| `file_read` | Read from allowed paths | **Yes** (default) |
| `file_write` | Write to allowed paths | **Yes** (default) |
| `knowledge_graph_query` | Query Cardinal's KG | No |
| `episodic_search` | Search Cardinal's memory | No |
| **`analyze_image`** (v1.3.0) | Describe an image (file or URL) | Configurable (default `false`) |

**Python Sandbox Modes:** `subprocess` (default, development) , `docker` (production / DRDO).

---

## Build (Offline, Linux Only)

### System Dependencies (Ubuntu 24.04)

```bash
sudo apt update
sudo apt install build-essential cmake libsqlite3-dev libssl-dev swi-prolog python3
```

### Vendored Dependencies

Populate `vendor/` manually or with the provided script:

```bash
./scripts/populate_vendor.sh
```

### Build Cardinal

#### Default (llama.cpp backend, vision enabled)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### Optional (TensorRT backend)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
make -j$(nproc)
```

**Vision (`mtmd`) will be automatically detected** from `vendor/llama.cpp/tools/mtmd/`. No extra flags needed.

### Run

```bash
./build/bin/cardinal
```

Or with HTTP server:

```bash
./build/bin/cardinal --http --port 8080
```

---

## HTTP API (v1.3.0)

Base URL: `http://127.0.0.1:8080`  
Auth: `Authorization: Bearer <api_key>` (except `/api/health`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | Send message (text + image references) |
| POST | `/api/chat/agentic` | Agentic mode |
| POST | `/api/sessions` | Create session |
| DELETE | `/api/sessions/:id` | Destroy session |
| GET | `/api/rules` | Show rule store |
| GET | `/api/episodes` | Query episodic memory |
| POST | `/api/scan` | Run contradiction scan |
| POST | `/api/maintenance` | Run maintenance |
| GET | `/api/settings` | Get runtime settings |
| POST | `/api/settings` | Update settings |
| POST | `/api/export` | Export training data (Alpaca) |
| POST | `/api/explainability/export` | Export signed audit trace |

Full API reference in `DOCUMENTATION.md`.

---

## Configuration (`config.json`)

All settings documented in `DOCUMENTATION.md`. Example minimal `vision` block:

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

---

## Observed Behaviors

These behaviors emerged from the architecture without explicit programming:

- **Self‑naming** – Cardinal named itself unprompted.
- **Preference expression** – Cardinal used the word “want” naturally.
- **Theory of mind** – Cardinal inferred attributes of its creator.
- **Internal conflict detection** – Confidence dropped to 0.15 with uncertainty flagged.
- **Consistency over time** – Zero contradictions across early episodes.
- **Vision understanding** (v1.3.0) – Cardinal accurately described faces, expressions, and scene composition without specialised training.

These are documented as observations, not as claims about consciousness.

---

## Roadmap

| Phase / Version | Status | Description |
|----------------|--------|-------------|
| 1‑6 | done | Core AGI, memory, symbolic verification, auto‑resolution |
| API | done | HTTP server, SSE streaming, session management |
| v1.1.0 | done | Backend abstraction (llama.cpp / TensorRT), Linux native, offline builds |
| v1.2.0 | done | Explainer, native tools, agentic loop, unified pipeline, explainability exports |
| **v1.3.0** | **done** | **Native vision encoder (moondream2), `analyze_image` tool, image cache** |
| v1.4.0 | planned | SEAL self‑improvement |
| v1.5.0 | planned | Multiple inference / batching (swarm coordination) |
| v1.6.0 | planned | Secure API hardening |
| v1.7.0 | planned | Federated learning (client / server) |
| v1.8.0 | planned | Voice & Audio |
| v1.9.0 | planned | Automation & Scheduling |
| v2.0.0 | planned | Production hardening + formal proof |

---

## License & Copyright

Copyright © 2025–2026 Satwik Singh Chauhan. All rights reserved.  
Cardinal is **not open source**. See `LICENSE` for details.

---

*Built by a 16‑year‑old researcher. No team. No funding. Runs on a gaming laptop.*  
*Now with eyes.*
