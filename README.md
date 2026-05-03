# Cardinal v1.1.0

**A production-grade neurosymbolic AGI architecture.**

Cardinal combines a large language model core with symbolic verification, persistent memory, hybrid retrieval, and a clean API layer. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

**Current version: v1.1.0** — Backend abstraction (llama.cpp / TensorRT), Linux native, offline builds.

---

## What Cardinal Is

Most LLM systems are stateless — each conversation starts from zero. Cardinal is different. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, and can export its accumulated knowledge as LoRA training data.

The architecture is neurosymbolic — it combines the pattern-matching strength of neural inference with the consistency guarantees of symbolic logic. Neither alone is sufficient. Together they produce a system that reasons carefully, stays consistent over time, and improves with experience.

---

## Hardware

Cardinal was developed and runs on:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~5GB core (excluding models) |
| OS | Ubuntu 24.04 LTS (Linux only) |

Model: Qwen3.5 4B Q4_K_M — runs via llama.cpp or TensorRT backend.

---

## Architecture (v1.1.0)

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
    +-- API Layer
    |     +-- CardinalTypes      shared types, no exceptions at boundary
    |     +-- CardinalSettings   runtime-mutable config, immediate propagation
    |     +-- SessionManager     multi-session conversation state
    |     +-- TrainingExporter   Alpaca JSONL export for LoRA fine-tuning
    |     +-- HttpServer         SSE streaming, Bearer auth, CORS, TypeScript bridge
    |
    +-- Interfaces (in development)
          +-- Interface 1        Chat (C++, always-on personality)
          +-- Interface 2        Agent (TypeScript via HTTP, tool execution)
          +-- Interface 3        SEAL (Python via pybind11, self-improvement loop)
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

---

## Memory Systems

### Dual-Write Pattern

Every inference is written to two stores simultaneously:

- **EpisodicMemory** — append-only JSONL. The audit trail. Never modified after write.
- **EpisodicStorage** — SQLite with FTS5. The searchable index. Supports keyword and semantic queries.

### Retrieval

Three retrieval modes:

| Mode | Mechanism | When to use |
|------|-----------|-------------|
| `keyword` | SQLite FTS5 | Exact or near-exact matches, lowest latency |
| `semantic` | TF-IDF cosine similarity | Paraphrased or conceptually similar queries |
| `hybrid` | Weighted combination | Default, best overall performance |

---

## Rule System

See `DOCUMENTATION.md` for full details. Key features:

- Rule extraction from natural language
- Symbolic verification via Prolog
- Full provenance tracking (`episode_id`, `reasoning_type`)
- Contradiction auto-resolution
- Confidence decay and pruning

---

## API Layer

All API methods return `CardinalResult<T>` or `CardinalVoidResult`. No exceptions propagate past `CardinalAPI`.

### HTTP Server

The HTTP server exposes the full `CardinalAPI` to TypeScript and any HTTP client.

**All endpoints require Bearer auth when `api.auth_enabled` is true.** The health endpoint is always public.

Base URL: `http://127.0.0.1:8080`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Alive check |
| POST | `/api/chat` | Send message, get response or SSE stream |
| POST | `/api/sessions` | Create a named session |
| DELETE | `/api/sessions/:id` | Destroy a session |
| GET | `/api/rules` | Full rule store |
| GET | `/api/episodes` | Episode query |
| POST | `/api/scan` | Run full contradiction scan |
| POST | `/api/maintenance` | Run maintenance cycle manually |
| GET | `/api/settings` | Get runtime settings |
| POST | `/api/settings` | Update settings |

See `DOCUMENTATION.md` for full API reference.

---

## Build (Offline, Linux Only)

### System Dependencies (install via `apt`)

```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libsqlite3-dev libssl-dev swi-prolog
```

### Vendored Dependencies

Cardinal uses a **clean vendor folder** for offline builds. You populate `vendor/` with these repositories:

| Dependency | Upstream URL | Clone as |
| :--- | :--- | :--- |
| llama.cpp | https://github.com/ggerganov/llama.cpp | vendor/llama.cpp |
| nlohmann/json | https://github.com/nlohmann/json | vendor/nlohmann_json |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | vendor/cpp-httplib |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | vendor/tokenizers-cpp |

**Clone them manually or use the provided script:**

```bash
chmod +x scripts/populate_vendor.sh
./scripts/populate_vendor.sh
```

## Build cardinal

### Default (llama.cpp backend)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Optional (TensorRT backend)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
make -j$(nproc)
```

### Run

```bash
./bin/cardinal
```

Or with HTTP server:

```bash
./bin/cardinal --http --port 8080
```

---

## Project Structure

```
cardinal/
    src/               # Core source code (see Architecture diagram)
    vendor/            # Populated with dependencies (not tracked in Git)
    data/              # Runtime data (episodes, rules, knowledge graph)
    logs/              # Application and episodic logs
    models/            # GGUF model files (user-provided)
    scripts/           # Helper scripts (populate_vendor.sh, etc.)
    CMakeLists.txt
    config.json
    README.md
    INSTALL.md
    DOCUMENTATION.md
```

---

## Configuration

All settings live in `config.json`. See `DOCUMENTATION.md` for detailed schema.

Example:

```json
{
  "model": {
    "path": "models/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    "context_length": 8192,
    "gpu_layers": 33
  },
  "api": {
    "http_enabled": true,
    "host": "127.0.0.1",
    "port": 8080,
    "auth_enabled": true
  }
}
```

---

## Observed Behaviors

These behaviors emerged from the architecture without explicit programming:

- **Self-naming** — Cardinal named itself unprompted.
- **Preference expression** — Cardinal used the word "want" naturally.
- **Theory of mind** — Cardinal inferred attributes of its creator.
- **Internal conflict detection** — Confidence dropped to 0.15 with uncertainty flagged.
- **Consistency over time** — Zero contradictions across early episodes.

These are documented as observations, not as claims about consciousness.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1-6 | done | Core AGI, memory, symbolic verification, auto-resolution |
| API | done | HTTP server, SSE streaming, session management |
| v1.1.0 | done | Backend abstraction (llama.cpp / TensorRT), Linux native, offline builds |
| v1.2.0 | in progress | Explainer module, tools (file/web/scrape), agentic loop |
| v1.3.0 | planned | Native vision encoder |
| v1.4.0 | planned | SEAL self-improvement |
| v1.5.0 | planned | Multiple inference / batching |
| v1.6.0 | planned | Secure API hardening |
| v1.7.0 | planned | Federated learning (client / server) |
| v2.0.0 | planned | Production hardening + formal proof |

---

*Built by a 16-year-old researcher. No team. No funding. Runs on a gaming laptop.*

