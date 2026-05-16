# Cardinal v1.4.0

**A production‑grade neurosymbolic AGI architecture with self-improvement.**

Cardinal combines a large language model core with **vision understanding**, symbolic verification, persistent memory, hybrid retrieval, a **full agentic loop**, **explainability exports**, and a **three-layer self-improvement system**. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

**Current version: v1.4.0** — SEAL self-improvement: symbolic self-model (Layer 1), meta-cognitive reflection (Layer 2), LoRA fine-tuning pipeline (Layer 3). All three layers run automatically after every inference.

---

## What's New in v1.4.0

- **Layer 1 — Symbolic Self-Model** — Cardinal tracks its own reasoning patterns in a SQLite database. After every inference it records confidence, contradiction rate, uncertainty rate, and rule-commit rate per domain. This is injected into the system prompt as calibration context.
- **Layer 2 — Meta-Cognition** — A scheduled reflection pass analyses recent failure episodes and generates corrective rules (type `meta_correction`) stored in the rule base. Triggers automatically on contradiction rate threshold or every N inferences. Also available on-demand via `/api/reflect`.
- **Layer 3 — LoRA Fine-tuning Pipeline** — Cardinal curates its own training data from high-confidence episodes, builds a domain-prioritised curriculum, runs HuggingFace PEFT via subprocess, converts the adapter to GGUF, evaluates it against a holdout set, and hot-loads it via `llama_set_adapters_lora`. For TensorRT deployments, a ready-to-run shell script is generated instead.
- **Three new HTTP endpoints** — `GET /api/self_model`, `POST /api/reflect`, `POST /api/train`.
- **All layers individually togglable** — each layer has its own `enabled` flag in `config.json`.

---

## What Cardinal Is

Most LLM systems are stateless — each conversation starts from zero. Cardinal is different. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, and now **improves its own weights** from accumulated experience.

The architecture is **neurosymbolic** — it combines the pattern-matching strength of neural inference with the consistency guarantees of symbolic logic. Neither alone is sufficient. Together they produce a system that reasons carefully, stays consistent over time, and improves with experience.

**With v1.4.0, Cardinal can improve itself.** It reflects on its own failures, generates corrective rules, and fine-tunes its own weights — all autonomously, all on the same hardware it runs on.

---

## Hardware

Cardinal was developed and runs on:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~5GB core + models + adapters |
| OS | Ubuntu 24.04 LTS (Linux only) |

**Models (v1.4.0):**
- Primary LLM: `Qwen3.5 4B Q4_K_M` (llama.cpp or TensorRT backend)
- Vision encoder: `moondream2` text model + `mmproj` (quantized, CPU)
- Fine-tuning: HuggingFace weights at `models/qwen3.5-4b-hf/` + Python venv with PEFT

---

## Architecture (v1.4.0)

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
    +-- Self-Improvement Subsystem (v1.4.0)        ← NEW
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
    +-- Agentic Pipeline (Unified)
    |     +-- AgentExecutor      PLAN → EXECUTE loop (THINK → ACT → OBSERVE) → FINALIZE
    |     +-- ToolExecutor       Sandboxed tool execution (subprocess or Docker)
    |     +-- WorkingMemory      SQLite-backed persistent scratchpad
    |     +-- SelfCorrection     Retry failed steps, max attempts configurable
    |
    +-- Explainability
    |     +-- AuditLog           Every inference trace: feeling, tools, rules, symbolic checks
    |     +-- Cryptographic signing  SHA256 + Ed25519 (tamper-evident)
    |     +-- Export API         JSON exports for compliance
    |
    +-- API Layer
          +-- CardinalTypes      shared types, no exceptions at boundary
          +-- CardinalSettings   runtime-mutable config, immediate propagation
          +-- SessionManager     multi-session conversation state
          +-- TrainingExporter   Alpaca JSONL export
          +-- HttpServer         SSE streaming, Bearer auth, CORS, TypeScript bridge
```

---

## Two-Pass Inference (Unchanged, Still Core)

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
The feeling output is injected as a synthetic assistant turn. The model generates its final response with full awareness of its own internal state.

**v1.4.0 addition:** After every inference, the feeling output is fed into `SelfImprovementLoop::on_inference()`. Layer 1 records the stats. Layer 2 checks whether a reflection pass should fire. Layer 3 checks episode count and confidence thresholds.

---

## Self-Improvement (v1.4.0)

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

The pass queries recent failure episodes, builds a structured prompt, runs a single LLM pass, parses the JSON findings, and commits corrective rules of type `meta_correction` to the rule store. These rules are injected into future inferences just like any other rule.

### Layer 3 — LoRA Fine-tuning

Triggers when:
- Every 100 episodes
- Every 24 hours
- Any domain's average confidence drops below 0.5
- Manually via `POST /api/train`

The cycle: `CurriculumBuilder` selects the weakest non-cooling domain → `DatasetCurator` pulls episodes from SQLite + rule augmentation → PEFT subprocess trains the adapter → `convert_lora_to_gguf.py` converts it → holdout evaluation → if improvement ≥ 5%, the adapter is loaded at the next session boundary via `llama_set_adapters_lora`.

For TensorRT deployments, `TensorRTTrainer` instead writes a ready-to-run shell script to `data/training/scripts/`.

---

## Self-Improvement API

```bash
# Current self-model status (all three layers)
GET /api/self_model

# Trigger on-demand reflection (Layer 2)
POST /api/reflect

# Trigger on-demand training (Layer 3, async)
POST /api/train
{"domain_hint": "factual"}   # optional — omit to let CurriculumBuilder decide
```

---

## Vision (v1.3.0, unchanged)

Supports local file paths and HTTP/HTTPS URLs. Results cached by URL/hash with configurable TTL.

```
You: what is in data/test.jpg?
Cardinal: [Tool: analyze_image] The image shows a young man with glasses...
```

Performance on RTX 3050 4GB: ~9-10s first image, ~2-3s cached.

---

## Tools (v1.2.0 + v1.3.0, unchanged)

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `web_search` | DuckDuckGo search | Configurable |
| `web_fetch` | Fetch and parse a URL | Configurable |
| `calculator` | Math expression evaluator | No |
| `run_python` | Sandboxed Python execution | **Yes** (default) |
| `file_read` | Read from allowed paths | No |
| `file_write` | Write to allowed paths | **Yes** (default) |
| `knowledge_graph_query` | Query Cardinal's KG | No |
| `episodic_search` | Search Cardinal's memory | No |
| `analyze_image` (v1.3.0) | Describe an image (file or URL) | No (default) |

---

## Build

```bash
# System dependencies
sudo apt install build-essential cmake libsqlite3-dev libssl-dev swi-prolog python3

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
cd ~/cardinal
./build/bin/cardinal
```

For the training pipeline (Layer 3), also set up the Python venv:

```bash
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate
pip install peft transformers torch accelerate
```

See `INSTALL.md` for the complete guide.

---

## HTTP API

Base URL: `http://127.0.0.1:8080`
Auth: `Authorization: Bearer <api_key>` (except `/api/health`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | Send message |
| POST | `/api/sessions` | Create session |
| DELETE | `/api/sessions/:id` | Destroy session |
| GET | `/api/stats` | System stats |
| GET | `/api/rules` | Rule store |
| GET | `/api/episodes` | Episode query |
| POST | `/api/scan` | Contradiction scan |
| POST | `/api/maintenance` | Maintenance cycle |
| GET | `/api/settings` | Get settings |
| POST | `/api/settings` | Update settings |
| POST | `/api/export` | Export training data |
| **GET** | **`/api/self_model`** | **Self-model status (v1.4.0)** |
| **POST** | **`/api/reflect`** | **Trigger reflection (v1.4.0)** |
| **POST** | **`/api/train`** | **Trigger training (v1.4.0)** |

---

## Configuration (`config.json`)

New `self_improvement` block in v1.4.0:

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

Full configuration reference in `DOCUMENTATION.md`.

---

## Observed Behaviors

- **Self-naming** — Cardinal named itself unprompted.
- **Preference expression** — Cardinal used the word "want" naturally.
- **Theory of mind** — Cardinal inferred attributes of its creator.
- **Internal conflict detection** — Confidence dropped to 0.15 with uncertainty flagged.
- **Consistency over time** — Zero contradictions across early episodes.
- **Vision understanding** (v1.3.0) — Cardinal accurately described faces, expressions, and scene composition.
- **Self-correction via rules** (v1.4.0) — Meta-cognition generated corrective rules that reduced contradiction rate in the factual domain across subsequent inferences.

These are documented as observations, not as claims about consciousness.

---

## Roadmap

| Version | Status | Description |
|---------|--------|-------------|
| v1.0–1.2 | done | Core AGI, memory, symbolic verification, agentic loop, explainability |
| v1.3.0 | done | Native vision encoder (moondream2), `analyze_image` tool |
| **v1.4.0** | **done** | **SEAL self-improvement: self-model, meta-cognition, LoRA fine-tuning** |
| v1.5.0 | planned | Automation & Scheduling |
| v1.6.0 | planned | Voice & Audio |
| v1.7.0 | planned | Secure API hardening |
| v2.0.0 | planned | Production hardening + formal proof |

---

## License & Copyright

Copyright © 2025–2026 Satwik Singh Chauhan. All rights reserved.
Cardinal is **not open source**. See `LICENSE` for details.

---

*Built by a 16-year-old researcher. No team. No funding. Runs on a gaming laptop.*
*Now it learns.*
