# Cardinal v1.4.0 — Technical Documentation

**Architecture:** Neurosymbolic AGI Core + Agentic Loop + Explainability + Vision + Self-Improvement
**Language:** C++20
**Platform:** Linux (Ubuntu 24.04 LTS)
**GPU:** NVIDIA CUDA (TensorRT / llama.cpp)
**Vision:** moondream2 via `llama.cpp` `mtmd` subsystem
**Self-Improvement:** Three-layer SEAL system (self-model, meta-cognition, LoRA fine-tuning)

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
11. [Agentic Pipeline (Unified)](#11-agentic-pipeline-unified)
12. [Tools System](#12-tools-system)
13. [Explainability Exports](#13-explainability-exports)
14. [Training Export](#14-training-export)
15. [Configuration Reference](#15-configuration-reference)
16. [CardinalAPI Reference](#16-cardinalapi-reference)
17. [HTTP API Reference](#17-http-api-reference)
18. [Settings Manager](#18-settings-manager)
19. [Session Manager](#19-session-manager)
20. [Type Reference](#20-type-reference)
21. [Error Handling](#21-error-handling)
22. [Offline Builds & Vendoring](#22-offline-builds--vendoring)
23. [Module Reference](#23-module-reference)
24. [Threading Model](#24-threading-model)
25. [Lifecycle and Startup Sequence](#25-lifecycle-and-startup-sequence)

---

## 1. Architecture Overview

Cardinal is structured in four layers. Each layer depends only on the layers below it. No layer reaches upward.

```
Layer 4 -- Interfaces
    Interface 1: Chat (C++)
    Interface 2: Agent (TypeScript via HTTP)
    Interface 3: SEAL (Python via pybind11)

Layer 3 -- API Layer
    CardinalAPI        single facade, owns all components
    HttpServer         TypeScript bridge, SSE streaming
    SettingsManager    runtime-mutable config
    SessionManager     multi-session conversation state
    TrainingExporter   Alpaca JSONL export
    Explainability     audit log, cryptographic signing, export API

Layer 2 -- Core Systems
    InferencePipeline  two-pass orchestrator, prompt injection
    AgentExecutor      PLAN → EXECUTE loop
    ToolExecutor       sandboxed tool execution
    LLMEngine          abstract backend (llama.cpp / TensorRT)
    ConsistencyChecker verifier orchestrator, auto-resolution
    SymbolicEngine     SWI-Prolog integration
    NeuralVerifier     optional small LLM verifier
    RuleExtractor      NLP rule extraction

Layer 1 -- Foundation
    RuleStore          persistent rule base
    KnowledgeGraph     typed node graph
    EpisodicMemory     JSONL audit trail
    EpisodicStorage    SQLite + FTS5 searchable index
    EpisodicRetriever  TF-IDF + keyword + hybrid retrieval
    ConfigLoader       typed config, validated at startup
    Logger             thread-safe, 6 levels
    JsonParser         serialization utilities

Layer 1.5 -- Vision Subsystem (v1.3.0)
    VisionCache        URL download cache, TTL eviction
    VisionEncoder      moondream2 wrapper via mtmd API

Layer 1.6 -- Self-Improvement Subsystem (v1.4.0)      ← NEW
    SelfModel          symbolic self-knowledge (SQLite accumulator)
    MetaCognition      reflection pass, corrective rule generation
    CurriculumBuilder  domain weakness scoring, training plan
    DatasetCurator     episode → TrainingExample, rule augmentation
    ITrainingBackend   abstract training interface
    LlamaCppTrainer    PEFT subprocess → GGUF → adapter hot-load
    TensorRTTrainer    script-export mode for cluster deployment
    AdapterEvaluator   holdout evaluation, improvement threshold gate
    SelfImprovementLoop  orchestrator, background training thread
```

### Component Ownership

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary — callers see only the types defined in `cardinal_types.h`.

### Dependency Graph (v1.4.0)

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
    owns --> SelfImprovementLoop (v1.4.0)
                --> SelfModel        --> SQLite (self_model.db)
                --> MetaCognition    --> EpisodicStorage
                                    --> RuleStore
                                    --> SelfModel
                                    --> LLMEngine
                --> CurriculumBuilder --> SelfModel
                --> DatasetCurator  --> EpisodicStorage
                                    --> RuleStore
                --> ITrainingBackend (LlamaCppTrainer | TensorRTTrainer)
                                    --> LLMEngine (adapter hot-load)
                --> AdapterEvaluator --> ITrainingBackend
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

GBNF grammar-constrained decoding forces the model to produce a structured JSON object before generating any natural language response. The grammar is defined in `src/prompts/feeling_schema.gbnf`.

The feeling output captures six fields:
- `confidence` — how confident the model is (0.0–1.0)
- `reasoning_type` — causal/deductive/inductive/abductive/analogical/associative
- `reasoning_domain` — factual/ethical/spatial/temporal/social/mathematical
- `uncertainty_flag` — whether the model is uncertain
- `contradiction_flag` — whether the model detects a conflict
- `rule_candidate_signal` — whether a rule might be extractable

### Synthetic Turn Injection

After Pass 1, the feeling JSON is injected as a synthetic assistant turn. The model reads its own feeling output as something it already said, not as an instruction from outside.

### Pass 2 — Free Decoding (Final Response)

No grammar constraint. The model generates its final response. The stream callback is called per token.

### v1.4.0 Addition — Self-Improvement Hook

After Pass 2 completes and the episode is committed, `SelfImprovementLoop::on_inference()` is called with the feeling output fields. This is where all three layers receive their data:
- Layer 1 writes a SQLite UPSERT to `domain_stats` and `reasoning_stats`.
- Layer 2 checks whether the inference counter or contradiction rate threshold has been crossed.
- Layer 3 checks the episode counter and domain confidence thresholds.

The hook is synchronous and lock-free for the common case (no trigger fires).

### Prompt Injection (v1.4.0)

If `self_improvement.self_model.inject_into_prompt=true`, a `[Self-Model]` block is prepended to the system prompt on every inference. Example:

```
[Self-Model]
Weakest domain:   factual
Strongest domain: mathematical
Domain          Conf  Contradict  Uncertain  Inferences
factual         0.71      0.12       0.08          142
mathematical    0.93      0.01       0.02           38
ethical         0.84      0.04       0.05           27
Top reasoning: deductive(89) causal(63) inductive(41)
```

This gives the model calibrated self-awareness of where it is weakest, which measurably improves confidence output quality in those domains.

### Retry Logic

If Pass 1 fails to produce valid JSON, the pipeline retries up to `max_retries` times with `retry_delay_ms` delay. Pass 2 does not retry.

### Metrics

Every `ChatResponse` carries:
- `pass1_tokens`, `pass2_tokens`, `total_ms`

---

## 3. Feeling Output Schema

**Fields:** `confidence` (float 0–1), `reasoning_type` (causal/deductive/inductive/abductive/analogical/associative), `reasoning_domain` (factual/ethical/spatial/temporal/social/mathematical), `uncertainty_flag` (bool), `contradiction_flag` (bool), `rule_candidate_signal` (bool).

**Validation:** `confidence > 0.8` with `uncertainty_flag = true` is rejected as contradictory.

**v1.4.0 usage:** All six fields are forwarded to `SelfImprovementLoop::on_inference()` after every inference.

---

## 4. Memory Systems

### 4.1 RuleStore

Persistent symbolic memory. Rules: `id`, `domain`, `condition`, `consequence`, `confidence`, `trigger_count`, timestamps, provenance (`episode_id`, `reasoning_type`).
Storage: `data/memory/rules.json` (atomic writes).

**v1.4.0:** Corrective rules generated by MetaCognition are stored here with `reasoning_type = "meta_correction"`. They are retrieved and injected like any other rule.

### 4.2 KnowledgeGraph

Typed nodes: `concept`, `fact`, `entity`, `relation`.
Storage: `data/memory/knowledge.json`.

### 4.3 EpisodicMemory

Append-only JSONL audit trail (`logs/episodic.log`). Never modified after write.

### 4.4 EpisodicStorage

SQLite + FTS5 searchable index (`data/memory/episodes.db`). Synchronised with JSONL via migration on first open. Used by `DatasetCurator` (Layer 3) and `MetaCognition` (Layer 2) to query failure episodes.

### 4.5 SelfModel DB (v1.4.0)

SQLite database at `data/self_model/self_model.db` (separate from episodes.db). Two tables:

**`domain_stats`** — running accumulators per reasoning domain:
```sql
domain TEXT PRIMARY KEY
total_inferences     INTEGER
total_contradictions INTEGER
total_uncertainties  INTEGER
total_rule_commits   INTEGER
confidence_sum       REAL
last_updated         TEXT
```

**`reasoning_stats`** — running accumulators per reasoning type:
```sql
reasoning_type TEXT PRIMARY KEY
usage_count         INTEGER
confidence_sum      REAL
contradiction_count INTEGER
last_updated        TEXT
```

All writes are single-statement `INSERT ... ON CONFLICT DO UPDATE` upserts — no read-before-write, O(1) regardless of inference count.

---

## 5. Retrieval System

`EpisodicRetriever` provides three modes:
- **KEYWORD** — SQLite FTS5, BM25 ranking.
- **SEMANTIC** — TF-IDF cosine similarity (in-memory index).
- **HYBRID** — weighted combination (default: 0.7 keyword, 0.3 semantic).

Retrieved episodes inject as a `[MEMORY CONTEXT]` block before the feeling output. Retrieval failure is non-fatal.

---

## 6. Verifier Pipeline

Runs after every inference to maintain rule base integrity.

- **Modes:** `symbolic` (default, SWI-Prolog), `neural` (small LLM), `hybrid`.
- **Per-inference sequence:**
  1. Rule extraction (if `rule_candidate_signal` true).
  2. Consistency check against existing rules.
  3. Contradiction resolution (auto or flagged).
  4. Rule commit (if passes consistency check).
  5. `set_extracted_rule_id()` on the episode record.

---

## 7. Rule System

Rules have: `id`, `domain`, `condition`, `consequence`, `confidence`, `trigger_count`, `episode_id` (provenance), `reasoning_type`.

**Lifecycle:** extraction → consistency check → commit → confidence decay → pruning.

**v1.4.0 additions:**
- `meta_correction` reasoning type — rules generated by MetaCognition reflection passes.
- Rules are used as training examples by `DatasetCurator` (rule augmentation).

**Similarity dedup:** Jaccard overlap > 0.8 on condition text → merge rather than duplicate.

---

## 8. Backend Abstraction

`ILLMBackend` abstract interface. Two implementations:

| Method | LlamaCppBackend | TensorRTBackend |
|--------|-----------------|-----------------|
| `generate_feeling()` | GBNF constrained, ctx_pass1_ | GBNF constrained |
| `generate_response()` | Free decode, ctx_pass2_ | Free decode |
| `load_lora_adapter()` | `llama_adapter_lora_init` + `llama_set_adapters_lora` | no-op |
| `unload_lora_adapter()` | `llama_set_adapters_lora(0)` + `llama_adapter_lora_free` | no-op |

**v1.4.0 additions to `LlamaCppBackend`:**
- `get_llama_model()` — returns `model_*` (used by `LlamaCppTrainer` for adapter init)
- `get_llama_context()` — returns `ctx_pass2_*` (adapter applied to free-decode context only)

The adapter is applied only to `ctx_pass2_` and not `ctx_pass1_`. This is intentional — the constrained feeling pass should reflect the base model's intrinsic reasoning quality, not the adapter's influence.

---

## 9. Vision Subsystem (v1.3.0)

Unchanged from v1.3.0. See v1.3.0 documentation for full reference.

- `VisionEncoder` — moondream2 via `mtmd` API.
- `VisionCache` — URL download cache with TTL eviction.
- `analyze_image` tool — accepts local paths and HTTP/HTTPS URLs.

Performance on RTX 3050 4GB: ~9–10s first image, ~2–3s cached.

---

## 10. Self-Improvement Subsystem (v1.4.0)

This is the primary new addition in v1.4.0. All three layers are owned and orchestrated by `SelfImprovementLoop`.

### 10.1 SelfImprovementLoop

**File:** `src/training/self_improvement_loop.h/.cpp`

The orchestrator. Constructed by `CardinalAPI::init()` after all subsystems are ready. Owns all three layers. Runs a single background training thread.

**Two hooks into the inference path:**
- `on_inference(domain, reasoning_type, confidence, contradiction, uncertainty, rule_committed)` — called after every `run_post_inference()`. Fast path: updates atomics, may trigger reflection or post to training thread.
- `on_session_boundary()` — called by `destroy_session()`. Applies any pending LoRA adapter at a clean cut-point.

**Training thread:** Sleeps on `condition_variable`, wakes on trigger or 60-second poll (for interval trigger). One cycle at a time; overlapping triggers are coalesced.

**Trigger conditions for Layer 3:**
| Trigger | Config key | Default |
|---------|-----------|---------|
| Episode count | `trigger_every_n_episodes` | 100 |
| Wall clock | `trigger_every_n_hours` | 24 |
| Domain confidence below | `trigger_on_domain_confidence_below` | 0.5 |
| Manual API | `POST /api/train` | — |

### 10.2 Layer 1 — SelfModel

**File:** `src/self_model/self_model.h/.cpp`

Maintains `data/self_model/self_model.db`. Updated after every inference via single-statement SQLite upserts. Provides:
- `record_inference()` — O(1) upsert, called on hot path.
- `get_snapshot()` — builds `SelfModelSnapshot` with `DomainStats` and `ReasoningTypeStats`.
- `get_weakest_domains(n)` — sorted by `weakness_score()`.
- `format_for_prompt()` — compact text for system prompt injection.

**Weakness score formula:**
```
weakness_score = (contradiction_rate × 0.4)
               + (uncertainty_rate   × 0.3)
               + ((1 - avg_confidence) × 0.3)
```

Scores range 0–1. Higher = more room for improvement.

### 10.3 Layer 2 — MetaCognition

**File:** `src/self_model/meta_cognition.h/.cpp`

**Trigger conditions:**
- Every N inferences (counter resets after firing).
- Contradiction rate for a domain exceeds `trigger_on_contradiction_rate_pct`% in a rolling window.
- On-demand via `POST /api/reflect`.

**Reflection pass steps:**
1. Query `EpisodicStorage::get_recent()` for episodes with `contradiction=true` or `uncertainty=true`.
2. Check `min_failures_to_reflect` threshold — abort if insufficient failures.
3. Get `SelfModelSnapshot` for context.
4. Build a structured reflection prompt (failure episodes + self-model summary).
5. Run `ILLMBackend::generate_response()` — single LLM pass, no tools, no feeling.
6. Parse the response as a JSON array of findings: `[{domain, pattern, recommendation, confidence}, ...]`.
7. For each finding above `corrective_rule_confidence`:
   - `RuleStore::add_rule(domain, "[pattern] ...", "[corrective] ...", confidence, ..., "meta_correction")`
8. `RuleStore::save()` if any rules were committed.
9. Return `ReflectionResult`.

**Reflection uses `std::try_to_lock`** on `reflect_mutex_` — if a reflection is already running, the new trigger is silently dropped.

**Reflection prompt format:** instructs the model to reply with only a JSON array and nothing else. The parser strips Markdown fences defensively and skips malformed entries individually.

### 10.4 Layer 3 — Training Pipeline

#### CurriculumBuilder

**File:** `src/training/curriculum_builder.h/.cpp`

Decides what to train on. Inspects `SelfModel::get_weakest_domains()` and applies a priority score:

```
priority_score = weakness_score
               + recency_bonus  (up to +0.2 for domains not trained recently)
```

Domains in cooldown (1 hour default) are skipped. If no domain exceeds `weakness_threshold` (default 0.4), a general all-domain plan is returned.

**LoRA overrides by weakness:**
- `weakness_score > 0.6` — +1 epoch, learning rate × 0.5
- `weakness_score > 0.4` — learning rate × 0.75
- Below 0.4 — base config unchanged

#### DatasetCurator

**File:** `src/training/dataset_curator.h/.cpp`

Converts episodes to `TrainingExample` (alpaca format: `instruction`, `input`, `output`, `domain`, `confidence`, `episode_id`).

**Quality filters (all AND-combined):**
- `confidence >= plan.min_episode_confidence`
- `user_message` and `response_summary` non-empty, each ≥ 10 characters
- Dedup on FNV-1a hash of `user_message`
- Holdout set reserved (most-recent N episodes, never included in training)

**Rule augmentation:** For each committed rule above 0.5 confidence, adds a training example:
```
instruction: "Apply the following guideline in your response:"
input:       "<condition> → <consequence>"
output:      "I will keep in mind: <consequence> when reasoning about <domain> topics."
```

Final dataset is shuffled with `std::mt19937`.

#### ITrainingBackend

**File:** `src/training/i_training_backend.h`

Abstract interface with five virtual methods: `prepare()`, `train()`, `evaluate()`, `load_adapter()`, `unload_adapter()`. Default `run_full_cycle()` implementation chains them in order and gates `load_adapter()` on the improvement threshold.

#### LlamaCppTrainer

**File:** `src/training/llama_cpp_trainer.h/.cpp`

**`train()` subprocess sequence:**
1. Write JSONL dataset to `data/training/datasets/<run_id>.jsonl`.
2. Invoke `<python_venv>/bin/python -m cardinal_train` with LoRA hyperparameters. The script emits `STEP n/total LOSS f` lines parsed in real time by the progress callback.
3. Run `convert_lora_to_gguf.py` on the HF adapter directory.
4. Return `TrainingResult` with `adapter_path` pointing to the GGUF file.

**`evaluate()` scoring:**
- Baseline: average stored `ep.confidence` from holdout episodes (no inference needed, cached).
- Eval: after loading adapter, run `generate_feeling()` on each holdout episode and read `ctx.feeling().confidence`. Average across holdout set.
- `improvement_pct = (eval - baseline) / baseline × 100`.

**Adapter loading:**
```cpp
llama_adapter_lora* adapter = llama_adapter_lora_init(model, path.c_str());
float scale = 1.0f;
llama_set_adapters_lora(ctx_pass2_, &adapter, 1, &scale);
```

Applied to `ctx_pass2_` only (free-decode context). `ctx_pass1_` (constrained feeling pass) is left unmodified.

**Unloading:**
```cpp
llama_set_adapters_lora(ctx_pass2_, nullptr, 0, nullptr);
llama_adapter_lora_free(active_lora_handle_);
```

#### TensorRTTrainer

**File:** `src/training/tensorrt_trainer.h/.cpp`

Script-export mode. `train()` writes a self-contained shell script to `data/training/scripts/train_<run_id>.sh` containing:
1. PEFT training invocation
2. `convert_lora_to_gguf.py` invocation
3. Commented-out `trtllm-build` block for engine rebuild

Returns immediately. `evaluate()` and `load_adapter()` are no-ops with informational messages. Intended for production deployments where training runs on a separate orchestration cluster.

#### AdapterEvaluator

**File:** `src/training/adapter_evaluator.h/.cpp`

Applies the improvement threshold gate and load policy:

**`improvement_threshold_pct`** (default 5%): adapter rejected if improvement below this.

**Load policies:**
- `"immediate"` — calls `backend_.load_adapter()` immediately on the calling thread (training thread).
- `"session_boundary"` (default) — stores adapter path in `pending_adapter_path_`. Applied by `on_session_boundary()` → `apply_pending_adapter()` when no inference is active.

Session boundary policy prevents a mid-conversation weight swap causing inconsistency within a session.

---

## 11. Agentic Pipeline (Unified)

Unchanged from v1.2.0. The PLAN → EXECUTE (THINK → ACT → OBSERVE) → FINALIZE loop. Vision tools and self-improvement rules are both available within the agentic loop.

```
AgentExecutor::run(goal):

  1. PLAN
     planner.decompose(goal) → vector<AgentStep>
     feeling pass on plan → confidence check
     symbolic check: plan contradicts known rules?

  2. EXECUTE LOOP
     a. THINK — LLM generates action for this step
     b. ACT — tool_executor.execute(), result → working_memory
     c. OBSERVE — inject tool results, check goal_achieved?

  3. FINALIZE
     generate final response
     trace_builder.finalize()
     audit_log.append(signed_trace)
```

---

## 12. Tools System

| Tool | Description | v1.4.0 Note |
|------|-------------|-------------|
| `web_search` | DuckDuckGo search | Unchanged |
| `web_fetch` | Fetch and parse URL | Unchanged |
| `calculator` | Math expression (muparser) | Unchanged |
| `run_python` | Sandboxed Python | Unchanged |
| `file_read` | Read from allowed paths | Unchanged |
| `file_write` | Write to allowed paths | Unchanged |
| `knowledge_graph_query` | Query KG | Unchanged |
| `episodic_search` | Search episodes | Unchanged |
| `analyze_image` (v1.3.0) | Image description | Unchanged |

Self-improvement does not add new tools. Corrective rules generated by MetaCognition are injected as regular rules and surface in inference prompts automatically.

---

## 13. Explainability Exports

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

**v1.4.0 note:** Meta-correction rules that were applied during an inference appear in `rules_applied` with `reasoning_type = "meta_correction"`, making the self-improvement influence fully traceable and auditable.

---

## 14. Training Export

`TrainingExporter` exports high-confidence episodes and committed rules as Alpaca JSONL for external LoRA fine-tuning. This is the manual export path. The automatic Layer 3 pipeline uses `DatasetCurator` directly without going through `TrainingExporter`.

**Filter parameters:** `min_confidence`, `domain`, `max_examples`, `include_rules`, `recent_first`.

**Output format:**
```json
{"instruction": "<user_message>", "input": "", "output": "<response>"}
```

---

## 15. Configuration Reference

### 15.1 `backend` section (unchanged)

`type` ("llama_cpp" or "tensorrt"), `llama_cpp.*`, `tensorrt.*`.

### 15.2 `inference` section (unchanged)

`temperature`, `top_p`, `max_tokens_feeling`, `max_tokens_response`.

### 15.3 `feeling_schema` section (unchanged)

`type`, `grammar_path`, `fields`, `max_tokens`.

### 15.4 `memory` section (unchanged)

`rule_store_path`, `knowledge_graph_path`, `episodic_log_path`, `max_rules`.

### 15.5 `verifier` section (unchanged)

`mode`, `neural_model_path`, `neural_gpu_layers`, `neural_max_tokens`, `contradiction_threshold`, `rule_confidence_decay`, `min_rule_confidence`.

### 15.6 `feedback` section (unchanged)

`max_retries`, `retry_delay_ms`, `rule_injection_format`.

### 15.7 `retriever` section (unchanged)

`mode`, `keyword_weight`, `semantic_weight`, `max_results`, `min_score`, `cache_rebuild_strategy`, `cache_rebuild_threshold`, `cache_rebuild_interval_seconds`.

### 15.8 `tools` section (unchanged)

Per-tool `enabled`, `confirmation_required`, and tool-specific parameters.

### 15.9 `agent` section (unchanged)

`enabled`, `max_iterations`, `max_iterations_hard_cap`, `working_memory_path`, `working_memory_size`, `self_correction_enabled`, `self_correction_max_attempts`, `plan_before_execute`, `summarize_on_cap`.

### 15.10 `explainability` section (unchanged)

`enabled`, `audit_log_path`, `signing_enabled`, `private_key_path`, `public_key_path`, `auto_generate_keys`, `export_path`, `attach_trace_to_response`.

### 15.11 `vision` section (v1.3.0, unchanged)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | string | – | Path to moondream2 text GGUF |
| `mmproj_path` | string | – | Path to moondream2 mmproj |
| `cache_ttl_hours` | int | `24` | Image cache TTL (`0` = never delete) |
| `allowed_paths` | array | `[]` | Local directories `analyze_image` may read |

### 15.12 `self_improvement` section (v1.4.0) ← NEW

```json
{
  "self_improvement": {
    "enabled": true,

    "self_model": {
      "enabled": true,
      "db_path": "data/self_model/self_model.db",
      "inject_into_prompt": true,
      "prompt_max_chars": 500,
      "history_window": 100
    },

    "meta_cognition": {
      "enabled": true,
      "trigger_every_n_inferences": 20,
      "trigger_on_contradiction_rate_pct": 30.0,
      "on_demand_via_api": true,
      "min_failures_to_reflect": 5,
      "max_corrective_rules_per_session": 10,
      "corrective_rule_confidence": 0.6
    },

    "training": {
      "enabled": true,
      "lora_rank": 8,
      "lora_alpha": 16,
      "learning_rate": 0.0001,
      "epochs": 3,
      "batch_size": 4,
      "min_episodes_for_training": 50,
      "min_quality_confidence": 0.75,
      "max_examples": 0,
      "trigger_every_n_episodes": 100,
      "trigger_every_n_hours": 24,
      "trigger_on_domain_confidence_below": 0.5,
      "adapter_load_policy": "session_boundary",
      "eval_improvement_threshold_pct": 5.0,
      "eval_holdout_episodes": 20,
      "adapter_output_dir": "data/training/adapters",
      "dataset_output_dir": "data/training/datasets",
      "export_script_dir": "data/training/scripts",
      "hf_model_path": "models/qwen3.5-4b-hf",
      "python_venv": "~/cardinal/cardinal-train-venv",
      "convert_lora_script": "vendor/llama.cpp/convert_lora_to_gguf.py",
      "llama_finetune_binary": "vendor/llama.cpp/build/bin/llama-finetune"
    }
  }
}
```

**`self_improvement.enabled`** — master switch. If `false`, none of the three layers start.

**`self_model` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Layer 1 on/off |
| `db_path` | string | SQLite database path |
| `inject_into_prompt` | bool | Prepend `[Self-Model]` block to system prompt |
| `prompt_max_chars` | int | Maximum characters for the injected block |
| `history_window` | int | Reserved for future rolling-window averaging |

**`meta_cognition` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Layer 2 on/off |
| `trigger_every_n_inferences` | int | Fire reflection every N inferences |
| `trigger_on_contradiction_rate_pct` | float | Fire if contradiction rate exceeds this % |
| `on_demand_via_api` | bool | Allow `POST /api/reflect` |
| `min_failures_to_reflect` | int | Minimum failure episodes needed to run |
| `max_corrective_rules_per_session` | int | Cap on rules committed per reflection pass |
| `corrective_rule_confidence` | float | Minimum finding confidence to commit a rule |

**`training` sub-keys:**

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Layer 3 on/off |
| `lora_rank` | int | LoRA rank (r). Higher = more capacity, more VRAM |
| `lora_alpha` | int | LoRA alpha. Effective scale = alpha/rank |
| `learning_rate` | float | AdamW learning rate |
| `epochs` | int | Training epochs (CurriculumBuilder may +1 for weak domains) |
| `batch_size` | int | Training batch size |
| `min_episodes_for_training` | int | Minimum total episodes before any training runs |
| `min_quality_confidence` | float | Episode quality floor for dataset inclusion |
| `max_examples` | int | Hard cap on dataset size (0 = unlimited) |
| `trigger_every_n_episodes` | int | Fire training after N new episodes |
| `trigger_every_n_hours` | int | Fire training every N hours |
| `trigger_on_domain_confidence_below` | float | Fire training if any domain avg_confidence < this |
| `adapter_load_policy` | string | `"immediate"` or `"session_boundary"` |
| `eval_improvement_threshold_pct` | float | Minimum improvement % to load adapter |
| `eval_holdout_episodes` | int | Episodes reserved for evaluation (not trained on) |
| `adapter_output_dir` | string | GGUF adapter output directory |
| `dataset_output_dir` | string | JSONL dataset output directory |
| `export_script_dir` | string | TensorRT shell script output directory |
| `hf_model_path` | string | HuggingFace model weights for PEFT |
| `python_venv` | string | Python venv with PEFT installed |
| `convert_lora_script` | string | Path to `convert_lora_to_gguf.py` |

### 15.13 `api` section (unchanged)

`http_enabled`, `host`, `port`, `auth_enabled`, `api_key`, `stream_enabled`, `max_request_size_kb`, `request_timeout_seconds`.

### 15.14 `logging` section (unchanged)

`level` (debug/info/warn/error), `path`.

---

## 16. CardinalAPI Reference

`CardinalAPI` is the single entry point for all interfaces.
**File:** `src/api/cardinal_api.h`

### 16.1 Lifecycle

```cpp
CardinalAPI api;
CardinalVoidResult init(const std::string& config_path = "config.json");
CardinalVoidResult shutdown();
```

### 16.2 Session Management (unchanged)

`create_session()`, `destroy_session()`, `reset_session()`, `get_session()`, `list_sessions()`.

**v1.4.0:** `destroy_session()` now calls `on_session_boundary()` internally, which applies any pending LoRA adapter.

### 16.3 Inference (unchanged signature)

```cpp
CardinalResult<ChatResponse> chat(const std::string& session_id,
                                   const std::string& message);
CardinalResult<ChatResponse> chat_stream(const std::string&       session_id,
                                          const std::string&       message,
                                          const ApiStreamCallback& stream_cb);
```

### 16.4 Agentic Inference (unchanged)

`agent(session_id, goal, max_iterations)`.

### 16.5 Memory & Stats (unchanged)

`get_stats()`, `get_rules()`, `get_episodes()`, `run_scan()`, `run_maintenance()`.

### 16.6 Training Export (unchanged)

`export_training_data()`, `export_dry_run()`.

### 16.7 Explainability (unchanged)

`get_trace()`, `export_trace()`, `verify_trace()`, `get_public_key()`.

### 16.8 Settings (unchanged)

`get_settings()`, `update_settings()`, `set_setting()`, `reset_settings()`.

### 16.9 Self-Improvement (v1.4.0) ← NEW

```cpp
// Returns current state of all three layers.
CardinalResult<SelfImprovementStatus> get_self_model_status() const;

// Trigger on-demand Layer 2 reflection pass (synchronous).
CardinalResult<ReflectionResult> reflect();

// Post Layer 3 training request to background thread (async, returns immediately).
CardinalResult<bool> trigger_training(const std::string& domain_hint = "");

// Called at session boundaries to apply pending LoRA adapters.
void on_session_boundary();
```

---

## 17. HTTP API Reference

Base URL: `http://127.0.0.1:8080`
Auth: `Authorization: Bearer <api_key>` (except `/api/health`)

### Existing endpoints (unchanged)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check (no auth) |
| POST | `/api/chat` | Send message, get response or SSE stream |
| POST | `/api/reset` | Reset session history |
| GET | `/api/stats` | System stats |
| GET | `/api/rules` | Rule store contents |
| GET | `/api/episodes` | Episode query |
| POST | `/api/scan` | Run contradiction scan |
| POST | `/api/maintenance` | Run maintenance cycle |
| GET | `/api/settings` | Get current settings |
| POST | `/api/settings` | Update settings |
| POST | `/api/export` | Export training data (Alpaca JSONL) |
| POST | `/api/sessions` | Create session |
| DELETE | `/api/sessions/:id` | Destroy session |
| POST | `/api/sessions/:id/reset` | Reset session |

### New endpoints (v1.4.0)

#### `GET /api/self_model`

Returns the current `SelfImprovementStatus`.

**Response:**
```json
{
  "self_model_enabled": true,
  "weakest_domain": "factual",
  "strongest_domain": "mathematical",
  "total_domain_stats": 5,
  "meta_cognition_enabled": true,
  "total_reflections": 3,
  "total_corrective_rules": 7,
  "last_reflection_at": "2026-05-15T14:23:11Z",
  "training_enabled": true,
  "total_training_runs": 1,
  "last_training_at": "2026-05-15T12:00:00Z",
  "active_adapter_path": "data/training/adapters/run_1747310400000.gguf",
  "last_improvement_pct": 6.3
}
```

#### `POST /api/reflect`

Triggers an on-demand Layer 2 reflection pass. Runs synchronously — may take several seconds.

**Response:**
```json
{
  "ran": true,
  "trigger": "manual",
  "episodes_analyzed": 40,
  "failures_analyzed": 8,
  "rules_committed": 2,
  "duration_ms": 3142,
  "timestamp": "2026-05-15T14:23:11Z",
  "error_message": "",
  "findings": [
    {
      "domain": "factual",
      "pattern": "Hallucinating source citations when confidence is marginal",
      "recommendation": "Hedge claims about specific sources unless verified",
      "confidence": 0.84,
      "timestamp": "2026-05-15T14:23:11Z"
    }
  ]
}
```

If `ran=false`, `error_message` explains why (e.g. not enough failure episodes, reflection already in progress).

#### `POST /api/train`

Posts a Layer 3 training request to the background thread. Returns immediately.

**Request body (optional):**
```json
{"domain_hint": "factual"}
```

`domain_hint` is optional. If empty, `CurriculumBuilder` selects the target domain automatically.

**Response:**
```json
{
  "accepted": true,
  "domain_hint": "",
  "message": "Training request posted to background thread"
}
```

`accepted=false` means training is disabled in config or a cycle is already running.

---

## 18. Settings Manager (unchanged)

Runtime-mutable settings propagate immediately. `SettingsManager` owns `EpisodicRetriever` mode/weights, verifier mode, rule thresholds, sampling parameters, log level, agent limits.

---

## 19. Session Manager (unchanged)

`SessionManager` owns `ConversationSession` objects. Each session tracks `turn_count`, `history`, `working_memory` (agentic), timestamps, and provides `trim_to_token_budget()`.

**v1.4.0:** `destroy_session()` now calls `on_session_boundary()` before returning, which applies pending LoRA adapters.

---

## 20. Type Reference

All public types in `src/api/cardinal_types.h`. Pybind11-friendly.

### New types (v1.4.0) — `src/self_model/self_model_types.h`

#### `DomainStats`

```cpp
struct DomainStats {
    std::string domain;
    float avg_confidence;
    float contradiction_rate;
    float uncertainty_rate;
    float rule_commit_rate;
    int   total_inferences;
    int   total_contradictions;
    int   total_uncertainties;
    std::string last_updated;
    float weakness_score() const;  // derived
};
```

#### `SelfModelSnapshot`

```cpp
struct SelfModelSnapshot {
    std::vector<DomainStats>        domain_stats;
    std::vector<ReasoningTypeStats> reasoning_stats;
    std::vector<PerformanceTrend>   trends;
    std::string weakest_domain()   const;
    std::string strongest_domain() const;
    std::string format_for_prompt(int max_chars = 500) const;
};
```

#### `ReflectionFinding`

```cpp
struct ReflectionFinding {
    std::string domain;
    std::string pattern;
    std::string recommendation;
    float       confidence;
    std::string timestamp;
};
```

#### `ReflectionResult`

```cpp
struct ReflectionResult {
    bool        ran;
    std::string trigger;
    int         episodes_analyzed;
    int         failures_analyzed;
    std::vector<ReflectionFinding> findings;
    int         rules_committed;
    int         duration_ms;
    std::string timestamp;
    std::string error_message;
};
```

#### `SelfImprovementStatus`

```cpp
struct SelfImprovementStatus {
    bool        self_model_enabled;
    std::string weakest_domain;
    std::string strongest_domain;
    int         total_domain_stats;
    bool        meta_cognition_enabled;
    int         total_reflections;
    int         total_corrective_rules;
    std::string last_reflection_at;
    bool        training_enabled;
    int         total_training_runs;
    std::string last_training_at;
    std::string active_adapter_path;
    float       last_improvement_pct;
};
```

#### `TrainingExample` (merged, canonical definition)

```cpp
struct TrainingExample {
    std::string instruction;    // user message / task
    std::string input;          // additional context (often empty)
    std::string output;         // target response
    std::string domain;         // source reasoning domain
    float       confidence;     // source episode confidence
    std::string episode_id;     // provenance
    std::string reasoning_type; // e.g. "causal"
    std::string timestamp;      // ISO-8601
    std::string source;         // "episode" | "rule"
};
```

### Existing types (unchanged)

`CardinalStatus`, `CardinalResult<T>`, `CardinalVoidResult`, `ChatResponse`, `FeelingInfo`, `SessionInfo`, `RuleInfo`, `EpisodeInfo`, `SystemStats`, `ExportInfo`, `ScanResult`, `StreamToken`, `VisionResult`.

---

## 21. Error Handling

No C++ exceptions cross the `CardinalAPI` boundary. All internal exceptions are caught and converted to `CardinalResult<T>`.

**Non-fatal paths:** retrieval failure, neural verifier failure, contradiction check failure, tool execution failure, JSONL migration parse errors, reflection pass LLM failure, training subprocess non-zero exit, adapter load failure (improvement below threshold).

**Fatal paths** (cause `init()` to fail): missing config/model/grammar files, SQLite error on episode DB open, explainability key generation failure.

**Self-improvement errors are non-fatal by design.** If `SelfImprovementLoop::start()` encounters an error constructing Layer 3 (e.g. training disabled or venv missing), it logs the error and continues with Layers 1 and 2 only. If the SelfModel DB fails to open, Layer 1 is disabled and inference continues normally. The system degrades gracefully rather than failing to start.

---

## 22. Offline Builds & Vendoring

Full offline builds with vendored dependencies.
`vendor/` contains: `llama.cpp`, `nlohmann_json`, `cpp-httplib`, `muparser`.
System dependencies installed via `apt`.

CMake detects `mtmd` from `vendor/llama.cpp/tools/mtmd/` and sets `CARDINAL_MTMD_AVAILABLE` automatically.

**v1.4.0:** No new vendored dependencies. The training pipeline uses system Python + PEFT (installed in the venv). `convert_lora_to_gguf.py` is taken from `vendor/llama.cpp/` directly.

---

## 23. Module Reference

### 23.1 VisionEncoder (v1.3.0, unchanged)

```cpp
class VisionEncoder {
    Result<> load();
    bool is_ready() const;
    Result<VisionResult> encode(const std::string& image_path);
};
```

### 23.2 VisionCache (v1.3.0, unchanged)

```cpp
class VisionCache {
    Result<> init();
    Result<std::string> get_or_download(const std::string& url);
    void set_ttl_hours(int hours);
};
```

### 23.3 SelfModel (v1.4.0)

```cpp
class SelfModel {
    void open();
    void close();
    void record_inference(domain, reasoning_type, confidence,
                          contradiction, uncertainty, rule_committed);
    SelfModelSnapshot get_snapshot() const;
    std::vector<DomainStats> get_weakest_domains(int n = 3) const;
    std::vector<DomainStats> get_all_domain_stats() const;
    std::string format_for_prompt() const;
    int total_records() const;
};
```

### 23.4 MetaCognition (v1.4.0)

```cpp
class MetaCognition {
    ReflectionResult on_inference(domain, contradiction, uncertainty);
    ReflectionResult reflect(const std::string& trigger = "manual");
    int total_reflections() const;
    int total_corrective_rules() const;
    std::string last_reflection_at() const;
};
```

### 23.5 SelfImprovementLoop (v1.4.0)

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

### 23.6 ITrainingBackend (v1.4.0)

```cpp
class ITrainingBackend {
    virtual std::string name() const = 0;
    virtual bool can_train_locally() const = 0;
    virtual TrainingResult prepare(dataset, lora_cfg) = 0;
    virtual TrainingResult train(dataset, lora_cfg, progress_cb) = 0;
    virtual TrainingResult evaluate(adapter_path, eval_episodes) = 0;
    virtual TrainingResult load_adapter(adapter_path) = 0;
    virtual void unload_adapter() = 0;
    virtual bool has_adapter() const = 0;
    virtual std::string active_adapter_path() const = 0;
    TrainingResult run_full_cycle(...);  // default implementation
};
```

---

## 24. Threading Model

| Thread | Owner | Responsibility |
|--------|-------|----------------|
| Main / inference | CardinalAPI | `chat()`, `agent()`, all user-facing calls |
| HTTP server | HttpServer | Request handling, SSE streaming |
| Training | SelfImprovementLoop | Layer 3 training cycle (background) |

**Inference thread:**
- `on_inference()` is called synchronously after every `run_post_inference()`.
- Layer 1: lock-free SQLite upsert (own internal mutex in SelfModel).
- Layer 2: atomic counter increment + may run synchronous reflection pass if trigger fires (acquires `reflect_mutex_` with `try_to_lock` — aborts if already reflecting).
- Layer 3: atomic counter increment + may `post_training_request()` (acquires `training_mutex_` briefly via `lock_guard`).

**Training thread:**
- Sleeps on `condition_variable`, wakes on trigger or 60-second poll.
- All training operations (JSONL write, subprocess, adapter load) happen here.
- Never blocks the inference thread.

**Session boundary (inference thread):**
- `on_session_boundary()` calls `AdapterEvaluator::apply_pending_adapter()`.
- Acquires `pending_mutex_` briefly to read/clear `pending_adapter_path_`.
- Calls `backend_.load_adapter()` which acquires `trainer_mutex_` in `LlamaCppTrainer`.
- Designed to be fast — called between sessions, no inference is active.

**Lock ordering (no deadlocks):**
```
api_mutex_ > session_mutex_ > inference_mutex_
                                 > self_model.mutex_
                                 > meta_cognition.reflect_mutex_
                                 > meta_cognition.window_mutex_
                                 > training_mutex_ > trainer_mutex_ > pending_mutex_
```

---

## 25. Lifecycle and Startup Sequence (v1.4.0)

```
main()
  |
  +-- Logger::init()
  +-- ConfigLoader::load("config.json")
  |
  +-- CardinalAPI::init()
        |
        +-- ILLMBackend (LlamaCppBackend or TensorRTBackend)
        |     +-- load_model()
        |     +-- create_context() × 2 (pass1, pass2)
        |
        +-- Memory subsystems
        |     +-- RuleStore::load()
        |     +-- KnowledgeGraph::load()
        |     +-- EpisodicMemory::init()
        |     +-- EpisodicStorage::open() → migrate from JSONL if needed
        |     +-- EpisodicRetriever::build_index()
        |
        +-- Verifier pipeline
        |     +-- SymbolicEngine::init() → SWI-Prolog
        |     +-- RuleExtractor::init()
        |     +-- NeuralVerifier::init() (if enabled)
        |     +-- ConsistencyChecker::init()
        |
        +-- InferencePipeline::init()
        |
        +-- Tools & Agent
        |     +-- ToolRegistry::register_all()
        |     +-- ToolExecutor::init()
        |     +-- AgentExecutor::init()
        |
        +-- Vision (v1.3.0)
        |     +-- VisionCache::init()
        |     +-- VisionEncoder::load() → ready or disabled
        |
        +-- Explainability
        |     +-- AuditLog::open()
        |     +-- ExplainabilityExporter::init()
        |     +-- key generation if auto_generate_keys=true
        |
        +-- API layer
        |     +-- TrainingExporter::init()
        |     +-- SettingsManager::init()
        |     +-- SessionManager::init()
        |
        +-- Self-Improvement (v1.4.0)        ← NEW
              +-- SelfImprovementLoop::start()
                    +-- SelfModel::open()    → creates self_model.db if needed
                    +-- MetaCognition::init()
                    +-- TrainingFactory::create() → LlamaCppTrainer or TensorRTTrainer
                    +-- CurriculumBuilder::init()
                    +-- DatasetCurator::init()
                    +-- AdapterEvaluator::init()
                    +-- training_thread_.start()

  +-- HttpServer::start() (if http_enabled)
  +-- interactive loop
```

**Shutdown:**
```
CardinalAPI::shutdown()
  |
  +-- SelfImprovementLoop::stop()
  |     +-- training_cv_.notify() with stop_requested_=true
  |     +-- training_thread_.join()  ← waits for current cycle to finish
  |     +-- SelfModel::close()       ← WAL checkpoint + close
  |
  +-- HttpServer::stop()
  +-- AgentExecutor::shutdown()
  +-- ConsistencyChecker::shutdown()
  +-- EpisodicStorage::close()
  +-- RuleStore::save()
  +-- VisionEncoder::unload()
  +-- ILLMBackend::unload()
```

The training thread is always joined before any subsystem it depends on (storage, rule_store, backend) is shut down. This is guaranteed by the destruction order of `unique_ptr` members in `CardinalAPI` — `self_improvement_` is declared last among subsystems in the header, so it is destroyed first.

---

*This documentation reflects Cardinal v1.4.0. The source of truth is always the source code.*
