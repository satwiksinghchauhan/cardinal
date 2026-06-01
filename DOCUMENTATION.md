# Cardinal — Complete Technical Documentation

**Version:** 2.0.0
**Architecture:** Neurosymbolic AGI Core + Agentic Loop + Explainability + Vision + Self-Improvement + Scheduler + Computer Use + Watch + Voice
**Language:** C++20
**Platform:** Linux (Ubuntu 24.04 LTS)
**GPU:** NVIDIA CUDA (llama.cpp / TensorRT for LLM; CUDA for Whisper STT)
**Vision:** moondream2 via `llama.cpp` `mtmd` subsystem
**Self-Improvement:** Three-layer SEAL system (self-model, meta-cognition, LoRA fine-tuning)
**Voice:** Whisper.cpp STT (CUDA), Piper TTS (ONNX CPU), PortAudio I/O, PocketSphinx wake-word, VAD, push-to-talk, sentence-streaming TTS, barge-in

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
9. [Vision Subsystem](#9-vision-subsystem)
10. [Self-Improvement Subsystem](#10-self-improvement-subsystem)
11. [Scheduler Subsystem](#11-scheduler-subsystem)
12. [Computer Use Subsystem](#12-computer-use-subsystem)
13. [Watch Subsystem](#13-watch-subsystem)
14. [Voice Subsystem](#14-voice-subsystem)
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
29. [Data Formats](#29-data-formats)
30. [Lifecycle and Startup Sequence](#30-lifecycle-and-startup-sequence)
31. [Extending Cardinal](#31-extending-cardinal)

---

## 1. Architecture Overview

Cardinal is structured in four layers. Each layer depends only on the layers below it. No layer reaches upward.

```
Layer 4 — Interfaces
    CLI (interactive loop, /commands, --voice flag)
    HTTP API (SSE streaming, Bearer auth, CORS)

Layer 3 — API Layer
    CardinalAPI         single facade, owns all components
    HttpServer          SSE streaming, Bearer auth, CORS
    SettingsManager     runtime-mutable config
    SessionManager      multi-session conversation state
    TrainingExporter    Alpaca JSONL export
    Explainability      audit log, cryptographic signing, export API

Layer 2 — Core Systems
    InferencePipeline   two-pass orchestrator, prompt injection
    AgentExecutor       PLAN → EXECUTE loop (THINK → ACT → OBSERVE) → FINALIZE
    ToolExecutor        sandboxed tool execution (all tools including computer use and voice)
    ILLMBackend         abstract backend (llama.cpp / TensorRT)
    ConsistencyChecker  verifier orchestrator, auto-resolution
    SymbolicEngine      SWI-Prolog integration
    NeuralVerifier      optional small LLM verifier
    RuleExtractor       NLP rule extraction
    SchedulerEngine     background thread, NL parsing, task dispatch
    Computer Use Layer  display, input, browser, shell, file, email
    Watch Subsystem     file, screen, process watchers
    VoiceLoop           STT, TTS, VAD, wake-word, barge-in

Layer 1 — Foundation
    RuleStore           persistent rule base
    KnowledgeGraph      typed node graph
    EpisodicMemory      JSONL audit trail
    EpisodicStorage     SQLite + FTS5 searchable index
    EpisodicRetriever   TF-IDF + keyword + hybrid retrieval
    VisionEncoder       moondream2 via mtmd API
    VisionCache         URL download cache, TTL eviction
    SelfImprovementLoop SEAL orchestrator
    SelfModel           per-domain statistics SQLite
    MetaCognition       reflection, corrective rules
    Training Pipeline   CurriculumBuilder, DatasetCurator, trainers
    SchedulerStore      SQLite WAL: tasks, runs, action_logs
    AudioDevice         PortAudio capture + playback streams
    VADDetector         energy RMS, pre-roll, barge-in detection
    STTEngine           whisper.cpp, CUDA
    TTSEngine           Piper ONNX CPU, sentence streaming
    WakeWordDetector    PocketSphinx keyword spotting
    ConfigLoader        typed config, validated at startup
    Logger              thread-safe, 6 levels
    JsonParser          serialization utilities
```

### Component Ownership

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary — callers see only the types defined in `cardinal_types.h`.

The voice subsystem is owned differently from all other subsystems: `CardinalAPI` owns a single `VoiceLoop` which in turn owns all five voice components (`AudioDevice`, `VADDetector`, `STTEngine`, `TTSEngine`, `WakeWordDetector`). This containment means all audio resources are freed atomically when `VoiceLoop::stop()` is called.

### Dependency Graph

```
CardinalAPI
    owns --> ILLMBackend
    owns --> InferencePipeline --> ILLMBackend
                               --> EpisodicRetriever
    owns --> AgentExecutor --> InferencePipeline
                           --> ToolExecutor
    owns --> ToolExecutor --> [all controllers below via setters]
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
    owns --> VisionCache
    owns --> VisionEncoder --> mtmd library
    owns --> SelfImprovementLoop
                --> SelfModel        --> SQLite (self_model.db)
                --> MetaCognition    --> EpisodicStorage
                                    --> RuleStore
                                    --> SelfModel
                                    --> ILLMBackend
                --> CurriculumBuilder --> SelfModel
                --> DatasetCurator  --> EpisodicStorage
                                    --> RuleStore
                --> ITrainingBackend (LlamaCppTrainer | TensorRTTrainer)
                                    --> ILLMBackend (adapter hot-load)
                --> AdapterEvaluator --> ITrainingBackend
    owns --> SchedulerEngine
                --> SchedulerStore  --> scheduler.db
                --> SchedulerParser --> InferencePipeline
                --> AgentExecutor, SelfImprovementLoop, EpisodicStorage
    owns --> DisplayDetector
    owns --> ScreenReader    --> DisplayDetector, VisionEncoder
    owns --> InputController --> DisplayDetector
    owns --> AppController   --> DisplayDetector
    owns --> BrowserController --> VisionEncoder
    owns --> ShellExecutor
    owns --> FileManager
    owns --> SystemController
    owns --> EmailController
    owns --> AtSpiReader
    owns --> FileWatcher, ScreenWatcher, ProcessWatcher
    owns --> VoiceLoop
                owns --> AudioDevice   --> PortAudio
                owns --> VADDetector
                owns --> STTEngine     --> whisper.cpp (CUDA)
                owns --> TTSEngine     --> Piper (ONNX CPU)
                owns --> WakeWordDetector --> PocketSphinx
    owns --> AuditLog, ExplainabilityExporter
    owns --> TrainingExporter
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

The first pass uses GBNF grammar-constrained decoding to force the model to produce a structured JSON object before generating any natural language response. This JSON object is called the **feeling output**.

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

The message history at the start of Pass 2 looks like:

```
[system]    Cardinal is a neurosymbolic AI...
[user]      [MEMORY CONTEXT] ... [END MEMORY CONTEXT]   (if retrieval found results)
[assistant] I have reviewed the relevant past context...
[user]      What is entropy?
[assistant] {"confidence":0.94,"reasoning_type":"deductive",...}  <-- injected feeling
```

The model in Pass 2 sees the feeling output as its own prior thought and uses it to calibrate its response — higher uncertainty flags lead to more hedged language, contradiction flags trigger careful qualification, and so on.

### Pass 2 — Free Decoding (Final Response)

Pass 2 runs with no grammar constraint. The model generates its final natural language response. The stream callback, if provided, is called for each token as it is generated.

In sentence-streaming TTS mode (voice enabled), the Pass 2 stream callback simultaneously feeds tokens to the TTS sentence splitter. When a complete sentence boundary is detected (`.`, `?`, `!`, `\n`) and the buffer has ≥3 words, synthesis begins immediately — overlapping with the remaining LLM generation, achieving sub-400ms first-audio latency.

### Self-Improvement Hook

After every `run_post_inference()`, `SelfImprovementLoop::on_inference()` is called with feeling output fields:
- Layer 1: O(1) SQLite upsert to `domain_stats` and `reasoning_stats`
- Layer 2: checks inference counter and contradiction rate thresholds
- Layer 3: checks episode counter and domain confidence thresholds

The hook is synchronous and lock-free for the common case (no trigger fires).

### Scheduler Hook

After every inference, `SchedulerEngine::on_inference()` updates the idle tracker timestamp. This allows idle-triggered tasks to measure actual idle time since the last inference.

### Self-Model Prompt Injection

If `self_improvement.self_model.inject_into_prompt=true`, a `[Self-Model]` block is prepended to the system prompt on every inference:

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

If Pass 1 fails to produce a valid feeling output (the JSON does not parse, or fails schema validation), the pipeline retries up to `max_retries` times with a `retry_delay_ms` delay between attempts. If all retries are exhausted, the inference fails.

Pass 2 does not retry — if the response generation fails, the inference fails.

### State Machine

The `FeelingContext` object tracks the state of a single inference cycle:

```
IDLE → PASS1_FEELING → PASS2_RESPONSE → COMPLETE
                     → FAILED (on retry exhaustion)
```

### Metrics

Every `ChatResponse` carries:
- `pass1_tokens` — tokens produced in Pass 1
- `pass2_tokens` — tokens produced in Pass 2
- `total_ms` — wall clock time for the full inference cycle

---

## 3. Feeling Output Schema

The feeling output is the core introspective signal that drives everything downstream. It is produced by Pass 1 and consumed by the verifier pipeline, the retriever, the session history, the training exporter, and the self-improvement loop.

### Fields

**`confidence`** (float, 0.0 to 1.0)

How confident the model is in its response. This is not a post-hoc annotation — the model produces this value before generating its response. Values below 0.4 are considered low confidence. Values above 0.7 are considered high confidence.

The consistency checker uses high-confidence episodes as candidates for rule extraction. The training exporter uses `min_confidence` to filter training examples. All six fields are forwarded to `SelfImprovementLoop::on_inference()` after every inference.

**`reasoning_type`** (string, one of six values)

The kind of reasoning the model is applying:

| Value | Meaning |
|-------|---------|
| `causal` | Cause-and-effect reasoning |
| `deductive` | From general rules to specific conclusions |
| `inductive` | From specific observations to general rules |
| `abductive` | Inference to the best explanation |
| `analogical` | Reasoning by analogy to known cases |
| `associative` | Pattern matching and association |

The rule extractor prioritizes causal and deductive reasoning types for rule extraction, since these produce the most reliable rule candidates.

**`reasoning_domain`** (string, one of six values)

The domain the reasoning operates in:

| Value | Meaning |
|-------|---------|
| `factual` | Empirical facts about the world |
| `ethical` | Moral and ethical reasoning |
| `spatial` | Spatial and geometric reasoning |
| `temporal` | Time-based reasoning |
| `social` | Reasoning about people and relationships |
| `mathematical` | Formal mathematical reasoning |

The domain is used for rule store partitioning, retrieval filtering, and verifier routing. The symbolic verifier is weighted higher for factual and mathematical domains in hybrid mode.

**`uncertainty_flag`** (boolean)

True when the model is uncertain about its response. Logically inconsistent with high confidence — the validator will reject a feeling output with `confidence > 0.8` and `uncertainty_flag = true`. When true, the response will typically contain hedging language ("I believe", "it is likely") even though this is never explicitly instructed.

**`contradiction_flag`** (boolean)

True when the model has detected a conflict in its own reasoning or between the current query and prior knowledge. When this flag is true, the consistency checker runs a targeted contradiction scan against the rule base for the current domain.

**`rule_candidate_signal`** (boolean)

True when the model believes its response contains a general rule that could be extracted and stored. When this flag is true, the rule extractor attempts to derive a Rule from the response text.

### Validation

The `JsonParser::validate_feeling_output()` function enforces logical consistency:
- `confidence > 0.8` with `uncertainty_flag = true` is rejected as contradictory and triggers a retry
- `confidence < 0.3` without `uncertainty_flag = true` produces a warning (not a rejection)

### GBNF Grammar

The grammar file at `src/prompts/feeling_schema.gbnf` enforces the exact JSON structure. If the grammar file is missing or malformed, Cardinal will fail to start with a `ConfigError`.

---

## 4. Memory Systems

Cardinal uses six distinct memory components. Each serves a different purpose and has different access characteristics.

### 4.1 RuleStore

The rule store is the symbolic memory of Cardinal. It contains general rules derived from past inference cycles.

**Storage:** `data/memory/rules.json` (JSON array, atomic writes)

**Key operations:**
- `add_rule(domain, condition, consequence, confidence, episode_id, reasoning_type)` — add a new rule. If a semantically similar rule exists in the same domain (Jaccard similarity >= 0.7), the existing rule is merged instead of creating a duplicate.
- `get_rule(id)` — fetch a single rule by ID
- `query(RuleQuery)` — fetch rules matching domain, condition hint, confidence threshold
- `decay_confidence()` — reduce all rule confidences by `rule_confidence_decay`
- `prune()` — remove rules below `min_rule_confidence`
- `enforce_limit()` — remove lowest-confidence rules if count exceeds `max_rules`

**Confidence lifecycle:**
```
add_rule()          -- initial confidence (0.5–0.65 depending on extraction strategy)
record_trigger()    -- +0.01 per trigger (when rule is relevant to an inference)
decay_confidence()  -- -(rule_confidence_decay) per maintenance cycle (default every 10 inferences)
prune()             -- removed if below min_rule_confidence (default 0.3)
```

**Rule similarity:** Two rules are considered similar if they share the same domain and their condition texts have Jaccard word overlap >= 0.7. When a similar rule is found on insert, the existing rule's confidence is boosted toward the higher value and its consequence is updated if the incoming one is more detailed.

**Thread safety:** All operations protected by `std::mutex`.

**Dirty tracking:** `is_dirty()` returns true if unsaved changes exist. `save()` is a no-op if not dirty.

**Meta-correction rules:** Rules generated by MetaCognition reflection passes are stored here with `reasoning_type = "meta_correction"`. They are retrieved and injected like any other rule and appear in explainability traces with that reasoning type.

### 4.2 KnowledgeGraph

The knowledge graph stores typed factual nodes and their relationships.

**Storage:** `data/memory/knowledge.json` (JSON array, atomic writes)

**Node types:** `concept`, `fact`, `entity`, `relation`

**Key operations:**
- `add_node(KnowledgeNode)` — insert or update a node
- `get_node(id)` — fetch a node by ID
- `find_path(from_id, to_id)` — BFS path between two nodes
- `get_hub_nodes(n)` — returns the N nodes with the most connections

The knowledge graph is currently populated manually or by future interface code. The core architecture stores and retrieves it but does not yet auto-populate it from inference cycles.

### 4.3 EpisodicMemory

The episodic memory is the append-only JSONL audit trail. It is never modified after a write — it is the ground truth record of every inference Cardinal has ever made.

**Storage:** `logs/episodic.log` (JSONL, one JSON object per line)

**Key operations:**
- `log_episode(user_message, response, feeling, pass1_tokens, pass2_tokens, total_ms)` — append one episode, returns the generated episode ID
- `open()` — open the file for appending
- `close()` — flush and close

**Episode ID format:** `<hex_timestamp_ms>_<4digit_counter>` — for example `19d62f60677_0000`. This format is monotonically increasing and sortable.

**Why JSONL:** Append-only formats are crash-safe. If Cardinal crashes mid-write, the worst case is one malformed line at the end of the file. All prior lines are intact. The SQLite layer handles searchability.

### 4.4 EpisodicStorage

EpisodicStorage is the searchable SQLite index over the episode corpus. It is the query layer — the JSONL file is the truth, SQLite is the index.

**Storage:** `data/memory/episodes.db` (SQLite WAL mode)

**Schema:**
```sql
CREATE TABLE episodes (
    id                TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    user_message      TEXT NOT NULL,
    response_summary  TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 0.0,
    reasoning_type    TEXT NOT NULL DEFAULT '',
    reasoning_domain  TEXT NOT NULL DEFAULT '',
    contradiction     INTEGER NOT NULL DEFAULT 0,
    uncertainty       INTEGER NOT NULL DEFAULT 0,
    rule_candidate    INTEGER NOT NULL DEFAULT 0,
    extracted_rule_id TEXT NOT NULL DEFAULT '',
    pass1_tokens      INTEGER NOT NULL DEFAULT 0,
    pass2_tokens      INTEGER NOT NULL DEFAULT 0,
    total_ms          INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE episodes_fts USING fts5(
    id UNINDEXED,
    user_message,
    response_summary,
    content='episodes',
    content_rowid='rowid'
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

**Indexes:** `reasoning_domain`, `confidence`, `timestamp`, `rule_candidate` — all indexed for fast filtered queries.

**FTS5 triggers:** Three triggers keep the FTS5 index synchronized with the episodes table automatically: `episodes_fts_insert`, `episodes_fts_delete`, `episodes_fts_update`.

**JSONL Migration:** On first `open()`, EpisodicStorage checks the `metadata` table for the key `migration_v1_complete`. If absent, it reads the JSONL file line by line and imports all episodes into SQLite in a single transaction. After completion it sets `migration_v1_complete = 1`. This migration is idempotent — it will never run twice on the same database file.

**Dual-write pattern:** After every inference, the episode is written to both EpisodicMemory (JSONL) and EpisodicStorage (SQLite).

**WAL mode:** Runs with `PRAGMA synchronous=NORMAL`. Checkpointed with `SQLITE_CHECKPOINT_TRUNCATE` on `close()`.

**Key operations:**
- `insert_episode(EpisodeRecord)` — `INSERT OR IGNORE` — duplicate IDs silently skipped
- `get_episode(id)` — fetch by primary key
- `query(EpisodeQuery)` — filtered query with optional FTS5 keyword search
- `set_extracted_rule_id(episode_id, rule_id)` — link an episode back to the rule it produced
- `count()` — total episodes in database
- `stats()` — aggregate statistics

Used by `DatasetCurator` (Layer 3), `MetaCognition` (Layer 2), and `SchedulerEngine` to store scheduled task results.

### 4.5 SelfModel DB

SQLite database at `data/self_model/self_model.db` (separate from `episodes.db`). Two tables:

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

### 4.6 SchedulerStore

SQLite WAL at `data/scheduler/scheduler.db`. Three tables:

**`scheduled_tasks`** — task definitions with trigger spec, action, run stats.  
**`task_runs`** — every run execution record with status, timing, result summary.  
**`task_action_logs`** — per-step action log within a run.

WAL mode enabled for concurrent read during active run logging.

---

## 5. Retrieval System

The retrieval system finds past episodes relevant to the current user message and injects them into the inference prompt.

### 5.1 Architecture

```
EpisodicRetriever
    |
    +-- keyword_search()    --> EpisodicStorage FTS5
    +-- semantic_search()   --> TF-IDF cosine similarity (in-memory index)
    +-- merge_results()     --> weighted combination + deduplication
```

### 5.2 Retrieval Modes

**KEYWORD mode:** Uses SQLite FTS5 full-text search over `user_message` and `response_summary`. Results are ranked by FTS5's internal BM25 scoring. Lowest latency. Best for queries that use the same or similar words as past episodes.

**SEMANTIC mode:** Uses TF-IDF cosine similarity over `user_message + response_summary`. The index is cached in memory. Best for paraphrased queries.

**HYBRID mode (default):** Runs both keyword and semantic search, normalizes scores to [0,1], then merges:

```
combined_score = (keyword_weight * keyword_score) + (semantic_weight * semantic_score)
```

Default weights: `keyword_weight = 0.7`, `semantic_weight = 0.3`.

### 5.3 Index Lifecycle

The TF-IDF index is rebuilt according to `cache_rebuild_strategy`:
- **ON_DEMAND:** Rebuild when `cache_rebuild_threshold` new episodes have been added
- **PERIODIC:** Rebuild every `cache_rebuild_interval_seconds` seconds
- **EXPLICIT:** Never auto-rebuild — call `rebuild_index()` manually

### 5.4 Prompt Injection

Retrieved episodes are injected as a structured context block:

```
[user]:      [MEMORY CONTEXT]
             1. [factual | confidence: 94% | score: 87%]
                Q: What happens to gas molecules when temperature increases?
                A: Gas molecules move faster and collide more frequently...
             [END MEMORY CONTEXT]

[assistant]: I have reviewed the relevant past context...
```

If retrieval fails for any reason, inference continues without memory context (non-fatal).

---

## 6. Verifier Pipeline

The verifier pipeline runs after every inference to maintain rule base integrity.

### 6.1 Modes

| Mode | Description |
|------|-------------|
| `symbolic` | Only SWI-Prolog symbolic engine (default, most reliable) |
| `neural` | Only neural verifier (requires `neural_model_path`) |
| `hybrid` | Both engines, weighted consensus based on domain |

### 6.2 Per-Inference Check Sequence

1. **Rule extraction** (if `rule_candidate_signal` true) — the rule extractor attempts to derive a rule from the response text. See Section 7.
2. **Contradiction check** (if `contradiction_flag` true OR a rule was committed) — the symbolic engine checks for contradictions with existing rules. For each contradiction found, auto-resolution runs. See Section 7.4.
3. **Rule commit** — if the candidate passes consistency check, it is written to the rule store.
4. **`set_extracted_rule_id()`** on the episode record.
5. **Periodic maintenance** (every 10 inferences) — confidence decay, prune, save.
6. **Build summary** — a human-readable summary is returned in `ConsistencyCheckResult.summary`.

### 6.3 SymbolicEngine

Wraps SWI-Prolog 9.2.9 via the C foreign language interface.

**Initialization:** `symbolic.init("path/to/cardinal_kb.pl")`

**Rule format in Prolog:**
```prolog
cardinal_rule(domain, condition, consequence).
```

**Contradiction detection:**
```prolog
contradicts(RuleA, RuleB) :-
    cardinal_rule(D, C, E1),
    cardinal_rule(D, C, E2),
    E1 \= E2.
```

**Thread safety:** Single Prolog thread, all operations serialized via `ConsistencyChecker` mutex.

### 6.4 NeuralVerifier

Optional verifier using a small LLM (Llama 3.2 1B Q4_K_M by default).

**Input:** Domain, candidate rule condition/consequence, top-5 existing rules in domain.

**Output (JSON):**
```json
{
  "contradiction": false,
  "contradiction_score": 0.1,
  "rule_quality": 0.8,
  "reasoning": "The candidate rule is consistent..."
}
```

Disabled if `neural_model_path` is empty or the file does not exist.

---

## 7. Rule System

### 7.1 Rule Struct

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique ID (`hex_timestamp_counter`) |
| `domain` | string | Reasoning domain |
| `condition` | string | The "if" part |
| `consequence` | string | The "then" part |
| `confidence` | float | Current confidence (0.0 to 1.0) |
| `trigger_count` | int | How many times this rule has been used |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-update timestamp |
| `episode_id` | string | Provenance: which episode created this rule |
| `reasoning_type` | string | Provenance: reasoning type at extraction time |

### 7.2 Extraction Pipeline

Three strategies in priority order:

**Strategy 1 — Causal patterns** (applied when `reasoning_type` is `causal` or `deductive`):

Searches for causal connectives: `if...then`, `because`, `causes`, `leads to`, `results in`, `implies`, `therefore`, `thus`.

Initial confidence: **0.6**

**Strategy 2 — Deductive patterns** (applied when `reasoning_type` is `deductive` or `inductive`):

Looks for conclusion markers: `therefore`, `thus`, `hence`, `consequently`, `it follows that`, `we can conclude`, `this means`, `this shows`, `this demonstrates`, `in conclusion`, `as a result`.

When a conclusion marker is found, the preceding sentence becomes the condition and the current sentence (with the marker stripped) becomes the consequence.

Initial confidence: **0.65**

**Strategy 3 — Declarative fallback** (any reasoning type, when strategies 1 and 2 fail):

Uses the user message as the condition and the longest non-question, non-heading, non-list sentence from the response as the consequence.

Initial confidence: **0.5**

### 7.3 Candidate Validation

Before a candidate rule is checked for contradictions or committed, it must pass validation:
- Condition and consequence must both be non-empty
- Condition must be at least 5 characters
- Consequence must be at least 5 characters
- Condition and consequence must not be identical
- `confidence >= min_rule_confidence`

If validation fails, the rule is rejected with a logged reason.

### 7.4 Contradiction Auto-Resolution

When a contradiction is detected between two rules, the resolver runs:

```
delta = abs(confidence_a - confidence_b)

if delta >= 0.2:
    deprecate(lower_confidence_rule)  -- set confidence to 0.0
    log("resolved: deprecated rule X in favor of rule Y")
    return RESOLVED

else:
    penalize(rule_a, -0.05)
    penalize(rule_b, -0.05)
    log("flagged for review: rules X and Y too close")
    return FLAGGED
```

A deprecated rule has its confidence set to 0.0. It remains in the store until the next maintenance cycle's `prune()` call removes it.

`FLAGGED` means both rules are plausible and neither can be confidently discarded. Both are penalized slightly to surface the conflict in confidence metrics.

`SKIPPED` means one or both rule IDs were not found in the store. This can happen if a rule was pruned between detection and resolution.

### 7.5 Rule Injection into Prompts

When rules exist in the store, `InferencePipeline` formats and injects them into the system prompt as a `## Active Rules` section:

```
## Active Rules
The following rules have been derived from prior reasoning. Apply them when relevant:

1. [factual] IF gas temperature increases THEN molecules move faster (confidence: 87%)
2. [factual] IF pressure is constant THEN volume increases with temperature (confidence: 72%)
```

Rules are fetched via `InferenceRequest.active_rules`, which the caller populates before calling `pipeline.run()`.

---

## 8. Backend Abstraction

`ILLMBackend` is the abstract inference backend interface. Two implementations are supported. Only one backend is loaded at runtime — selected at CMake configuration time.

### 8.1 Supported Backends

| Backend | Purpose | When to Use |
|---------|---------|-------------|
| **LlamaCppBackend** | Development, fallback, CPU/CUDA | Default; works everywhere |
| **TensorRTBackend** | Optimized deployment | Production, fixed models, maximum performance |

### 8.2 Interface

```cpp
class ILLMBackend {
    virtual Result<FeelingOutput> generate_feeling(...) = 0;
    virtual Result<std::string>   generate_response(...) = 0;
    virtual void load_lora_adapter(const std::string& path) = 0;
    virtual void unload_lora_adapter() = 0;
};
```

| Method | LlamaCppBackend | TensorRTBackend |
|--------|-----------------|-----------------|
| `generate_feeling()` | GBNF constrained, ctx_pass1_ | GBNF constrained |
| `generate_response()` | Free decode, ctx_pass2_ | Free decode |
| `load_lora_adapter()` | `llama_adapter_lora_init` + `llama_set_adapters_lora` | no-op |
| `unload_lora_adapter()` | `llama_set_adapters_lora(0)` + free | no-op |

The adapter is applied only to `ctx_pass2_` (free-decode context) and not to `ctx_pass1_`. This is intentional — the constrained feeling pass should reflect the base model's intrinsic reasoning quality, not the adapter's influence.

`LlamaCppBackend` also exposes `get_llama_model()` and `get_llama_context()` — used by `LlamaCppTrainer` for adapter initialization.

### 8.3 Switching Backends

At CMake configuration time:

```bash
# Default: llama.cpp
cmake .. -DCMAKE_BUILD_TYPE=Release

# TensorRT
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON
```

The same `CardinalAPI` interface works regardless of backend.

### 8.4 GBNF and TensorRT

The GBNF grammar is **only used with llama.cpp**. TensorRT engines are built with fixed input/output shapes and do not require runtime grammar constraints.

---

## 9. Vision Subsystem

The vision subsystem provides image understanding via moondream2, a compact vision-language model running on CPU via the `mtmd` API from `llama.cpp`.

### 9.1 Components

**`VisionEncoder`** — moondream2 wrapper via `mtmd` API.

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

Uses the `mtmd` API via `mtmd.h`. Loads moondream2 text GGUF and projector. Runs on CPU (configurable threads).

**`VisionCache`** — URL download cache with TTL eviction.

```cpp
class VisionCache {
    Result<> init();
    Result<std::string> get_or_download(const std::string& url);
    void set_ttl_hours(int hours);
};
```

### 9.2 `analyze_image` Tool

Implements the `Tool` interface. Calls `VisionCache` (for URLs) then `VisionEncoder`. Accepts both local file paths and HTTP/HTTPS URLs. Returns `VisionResult` as tool output.

### 9.3 Computer Use Integration

`ScreenReader::analyze()` calls `VisionEncoder::encode()` with `ImageSource::FILE` to describe screenshots. Vision-based element finding uses the same encoder with a coordinate-extraction prompt: `"Find the UI element described as: '...' Respond ONLY with: x=NNN y=NNN"`.

### 9.4 VisionResult Type

```cpp
struct VisionResult {
    bool success;
    std::string description;
    std::string error_message;
    int width = 0;
    int height = 0;
    std::string mime_type;
    size_t file_size_bytes = 0;
    std::chrono::milliseconds encode_ms;
};
```

### 9.5 mtmd Availability

CMake auto-detects `mtmd` from `vendor/llama.cpp/tools/mtmd/` and sets `CARDINAL_MTMD_AVAILABLE`. If `mtmd` is not found, vision is disabled at compile time. If `model_path` or `mmproj_path` is empty or files are missing, vision is disabled at runtime (warnings logged, `analyze_image` returns an error).

**Performance on RTX 3050 4GB:** ~9–10s first image, ~2–3s cached.

---

## 10. Self-Improvement Subsystem

All three layers are owned and orchestrated by `SelfImprovementLoop`. This subsystem enables Cardinal to measure its own weaknesses, generate corrective rules, and autonomously fine-tune itself via LoRA.

### 10.1 SelfImprovementLoop

**File:** `src/training/self_improvement_loop.h/.cpp`

The orchestrator. Constructed by `CardinalAPI::init()` after all subsystems are ready. Owns all three layers. Runs a single background training thread.

**Two hooks into the inference path:**
- `on_inference(domain, reasoning_type, confidence, contradiction, uncertainty, rule_committed)` — called after every `run_post_inference()`. Fast path: updates atomics, may trigger reflection or post to training thread.
- `on_session_boundary()` — called by `destroy_session()`. Applies any pending LoRA adapter at a clean cut-point.

**Training thread:** Sleeps on `condition_variable`, wakes on trigger or 60-second poll. One cycle at a time; overlapping triggers are coalesced.

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
- `record_inference()` — O(1) upsert, called on hot path
- `get_snapshot()` — builds `SelfModelSnapshot` with `DomainStats` and `ReasoningTypeStats`
- `get_weakest_domains(n)` — sorted by `weakness_score()`
- `format_for_prompt()` — compact text for system prompt injection

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
- Every N inferences (counter resets after firing)
- Contradiction rate for a domain exceeds `trigger_on_contradiction_rate_pct`% in a rolling window
- On-demand via `POST /api/reflect`

**Reflection pass steps:**
1. Query `EpisodicStorage::get_recent()` for episodes with `contradiction=true` or `uncertainty=true`
2. Check `min_failures_to_reflect` threshold — abort if insufficient failures
3. Get `SelfModelSnapshot` for context
4. Build a structured reflection prompt (failure episodes + self-model summary)
5. Run `ILLMBackend::generate_response()` — single LLM pass, no tools, no feeling
6. Parse the response as a JSON array of findings: `[{domain, pattern, recommendation, confidence}, ...]`
7. For each finding above `corrective_rule_confidence`: `RuleStore::add_rule(..., "meta_correction")`
8. `RuleStore::save()` if any rules were committed

Uses `std::try_to_lock` on `reflect_mutex_` — if a reflection is already running, the new trigger is silently dropped.

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
1. Write JSONL dataset to `data/training/datasets/<run_id>.jsonl`
2. Invoke `<python_venv>/bin/python -m cardinal_train` with LoRA hyperparameters. The script emits `STEP n/total LOSS f` lines parsed in real time by the progress callback.
3. Run `convert_lora_to_gguf.py` on the HF adapter directory
4. Return `TrainingResult` with `adapter_path` pointing to the GGUF file

**`evaluate()` scoring:**
- Baseline: average stored `ep.confidence` from holdout episodes (no inference needed, cached)
- Eval: after loading adapter, run `generate_feeling()` on each holdout episode and average `confidence` values
- `improvement_pct = (eval - baseline) / baseline × 100`

**Adapter loading:**
```cpp
llama_adapter_lora* adapter = llama_adapter_lora_init(model, path.c_str());
float scale = 1.0f;
llama_set_adapters_lora(ctx_pass2_, &adapter, 1, &scale);
```

Applied to `ctx_pass2_` only. `ctx_pass1_` is left unmodified.

**Unloading:**
```cpp
llama_set_adapters_lora(ctx_pass2_, nullptr, 0, nullptr);
llama_adapter_lora_free(active_lora_handle_);
```

#### TensorRTTrainer

**File:** `src/training/tensorrt_trainer.h/.cpp`

Script-export mode. `train()` writes a self-contained shell script to `data/training/scripts/train_<run_id>.sh` containing a PEFT training invocation, `convert_lora_to_gguf.py` invocation, and a commented-out `trtllm-build` block for engine rebuild. Returns immediately. `evaluate()` and `load_adapter()` are no-ops. Intended for production deployments where training runs on a separate orchestration cluster.

#### AdapterEvaluator

**File:** `src/training/adapter_evaluator.h/.cpp`

Applies the improvement threshold gate and load policy.

**`improvement_threshold_pct`** (default 5%): adapter rejected if improvement is below this.

**Load policies:**
- `"immediate"` — calls `backend_.load_adapter()` immediately on the training thread
- `"session_boundary"` (default) — stores adapter path in `pending_adapter_path_`. Applied by `on_session_boundary()` when no inference is active

Session boundary policy prevents a mid-conversation weight swap causing inconsistency within a session.

---

## 11. Scheduler Subsystem

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
| `AGENT_RUN` | Full agentic execution via `AgentExecutor::run()` |
| `CHAT` | Single inference via `InferencePipeline::run()` |
| `REFLECT` | Force Layer 2 meta-cognition reflection |
| `TRAIN` | Post Layer 3 training request |
| `SELF_IMPROVEMENT` | All three layers sequentially |
| `MAINTENANCE` | `ConsistencyChecker` maintenance cycle |
| `EXPORT` | Training data export |
| `SHELL` | Run a shell command via `ShellExecutor` |
| `WEBHOOK` | HTTP POST result to a URL |

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

`extract_from_json()` is public static so `SchedulerEngine::create_task_from_nl()` can call it directly after its own LLM call.

### 11.7 SchedulerEngine

**File:** `src/scheduler/scheduler_engine.h/.cpp`

**Background thread lifecycle:**
1. On `start()`: opens SQLite store, fires STARTUP tasks immediately, starts `engine_loop()` thread
2. Engine loop: sleeps `check_interval_seconds` on `condition_variable`, calls `tick()` on wake
3. `tick()`: iterates enabled tasks, evaluates triggers via `should_fire()`, calls `dispatch_task()` for any that fire
4. On `stop()`: sets `stop_requested_=true`, notifies CV, joins thread, closes store

**Trigger evaluation:**
- `CRON` — compares current `localtime` against five cron fields, deduplicates by checking `last_run_at` within the same minute
- `INTERVAL` — compares `now - last_run_at` against `interval_seconds`
- `CONDITION` — evaluates expression via `eval_condition_expr()` using metric variables from `SelfImprovementStatus`
- `IDLE` — checks `now - last_inference_at_` against `idle_minutes × 60`

**Task dispatch:**
- Maximum one concurrent task (`max_concurrent_tasks=1`)
- Safety whitelist check before dispatch
- Spawns a detached `std::thread` for execution
- Updates `task_runs` and `task_action_logs` in SQLite after completion

### 11.8 `schedule_task` Tool

Registered when `scheduler.enabled=true` (works headless — no display required).

**Actions:** `create`, `list`, `enable`, `disable`, `delete`, `run_now`

**Tool call examples:**
```json
{"name": "schedule_task", "arguments": {"action": "create", "description": "search for AI news every morning at 7am"}}
{"name": "schedule_task", "arguments": {"action": "list"}}
{"name": "schedule_task", "arguments": {"action": "run_now", "task_id": "abc12345"}}
```

---

## 12. Computer Use Subsystem

### 12.1 Overview

Cardinal can see and operate a desktop environment. The subsystem is initialised when `computer_use.enabled=true`. Display server (X11 / Wayland / headless) is detected at runtime by `DisplayDetector`.

All computer use controllers are owned by `CardinalAPI` and wired into `ToolExecutor` via setters after initialisation.

**Files:** `src/computer/`

### 12.2 DisplayDetector

Detects the display server by checking environment variables and probing live connections:
1. Check `WAYLAND_DISPLAY` — if set and socket exists → Wayland
2. Check `DISPLAY` — if set and X server responds → X11
3. Otherwise → headless

Provides `info()` returning `ScreenInfo` with `server`, `display_var`, `width`, `height`, `scale_factor`.

Methods: `is_x11()`, `is_wayland()`, `is_headless()`, `has_scrot()`, `has_grim()`

### 12.3 ScreenReader

- X11: `scrot <path>` subprocess; Wayland: `grim <path>` subprocess
- `capture(analyze)` → `Screenshot`: if `analyze=true` and `VisionEncoder` is ready, calls `VisionEncoder::encode(path, prompt, meta)`. Result stored in `Screenshot::description`.
- `capture_region(ScreenRegion, analyze)` — passes `-a x,y,w,h` to scrot or `-g "x,y WxH"` to grim
- `analyze(image_path, prompt)` → `std::string`: direct vision encode call
- `find_element(description)` → `std::optional<Point>`: captures screenshot, calls vision with coordinate-extraction prompt, parses `x=NNN y=NNN` from the response

### 12.4 InputController

**X11 (xdotool):** `mouse_click`, `type_text`, `send_key`, `mouse_scroll`

**Wayland (ydotool + wtype):** equivalent commands via uinput

Display server chosen at call time based on `DisplayDetector::is_x11()`.

### 12.5 AppController

| Method | X11 | Wayland |
|--------|-----|---------|
| `open_app(name)` | `gtk-launch` + exec fallback | same |
| `close_app(name)` | `wmctrl -c` + `pkill` | `swaymsg [app_id=name] kill` |
| `focus_app(name)` | `wmctrl -a` | `swaymsg [app_id=name] focus` |
| `list_apps()` | `wmctrl -l` parsed | `swaymsg -t get_tree` parsed |

`is_app_allowed(name)` checks against `computer_use.safety.allowed_apps`. If the list is empty, all apps are allowed.

### 12.6 BrowserController

Spawns a persistent Python helper process using the Playwright library from the browser venv. Communicates via stdin/stdout JSON lines. Started lazily on first `execute()` call.

**`BrowserActionType` enum:** `NAVIGATE`, `CLICK`, `CLICK_TEXT`, `TYPE`, `SCROLL`, `GET_CONTENT`, `SCREENSHOT`, `EXECUTE_JS`, `NEW_TAB`, `CLOSE_TAB`, `BACK`, `FORWARD`, `RELOAD`

**Domain safety:** `is_domain_allowed(url)` checks against `allowed_domains` / `blocked_domains`. Empty `allowed_domains` = all permitted.

**Vision fallback:** if a selector-based click fails and `VisionEncoder` is available, falls back to screenshot + vision coordinate extraction.

### 12.7 ShellExecutor

`popen()` with configurable timeout (SIGKILL on expiry). Safety checks: `shell.enabled` must be true, command must not contain any `blocked_commands` substring.

Returns `ShellResult`: `success`, `exit_code`, `stdout_text`, `stderr_text`, `duration_ms`, `command`.

### 12.8 FileManager

All paths validated against `computer_use.safety.allowed_paths`. Tilde expansion via `expand_home()`.

Operations: `list`, `move`, `copy`, `remove`, `mkdir`, `stat`, `exists`. Write operations gated on `allow_file_write` config flag.

### 12.9 SystemController

| Method | Tool used |
|--------|-----------|
| `get_state()` | `pactl get-sink-volume`, `/proc/net/wireless`, `rfkill list` |
| `set_volume(pct)` | `pactl set-sink-volume @DEFAULT_SINK@ pct%` |
| `set_mute(bool)` | `pactl set-sink-mute @DEFAULT_SINK@ 0/1` |
| `set_brightness(pct)` | `brightnessctl set pct%` |
| `set_wifi(bool)` | `nmcli radio wifi on/off` |
| `set_bluetooth(bool)` | `bluetoothctl power on/off` |

Returns `SystemState`: `volume_pct`, `muted`, `brightness_pct`, `wifi_enabled`, `wifi_ssid`, `bluetooth_enabled`, `battery_pct`, `battery_charging`.

### 12.10 EmailController

Two modes, selected by `computer_use.email.mode`:

**`imap_smtp` mode:** Python subprocess using `imaplib` and `smtplib`. Password from `CARDINAL_EMAIL_PASS` environment variable — never in config.

**`gmail_api` mode:** Python subprocess using `google-api-python-client`. Credentials from `gmail_credentials_path`. OAuth token auto-refreshed.

### 12.11 AtSpiReader

Python subprocess using `pyatspi`. `get_tree(app_name)` → `AtSpiNode`. `find_nodes(app_name, role, name_contains)` → `vector<AtSpiNode>`. Used as the primary method for element location before falling back to vision-based coordinate finding.

**`AtSpiNode`:** `role`, `name`, `bounds` (x,y,width,height), `states` (vector), `children` (vector, recursive).

### 12.12 Computer Use Tools

All registered when `computer_use.enabled=true`:

| Tool | Key arguments |
|------|---------------|
| `screenshot` | `analyze`, `prompt`, `region_x/y/w/h` |
| `click` | `description` OR `x`+`y`, `button`, `double_click`, `app` |
| `type_text` | `text` OR `key`, `delay_ms` |
| `open_app` | `app`, `focus` |
| `close_app` | `app` |
| `browser` | `action`, `url`, `selector`, `text`, `script`, `scroll_y` |
| `shell_run` | `command`, `timeout_seconds`, `working_dir` |
| `file_ops` | `action`, `path`, `dest`, `recursive` |
| `system_control` | `action`, `value` |
| `email` | `action`, `folder`, `subject`, `from`, `to`, `body` |
| `watch_screen` | `wait_for`, `timeout_seconds`, `poll_seconds`, `analyze` |
| `schedule_task` | `action`, `description`, `task_id` |

Browser and email are only registered if their respective sub-configs are non-empty. Shell is only registered if `shell.enabled=true`.

---

## 13. Watch Subsystem

Three independent background watchers providing event-driven observation.

**Files:** `src/watch/`

### 13.1 FileWatcher

inotify-based file system monitoring. Configured via `FileWatchConfig`: `path`, `recursive`, `events` (CREATE/MODIFY/DELETE/MOVE), `callback`.

### 13.2 ScreenWatcher

Periodic screenshot diff using ImageMagick `compare -metric PSNR`. Configured via `ScreenWatchConfig`: `poll_interval_seconds`, `psnr_threshold`, `region` (optional), `callback`.

The `watch_screen` tool wraps this with a blocking wait — it polls until PSNR drops below threshold (visual change detected) or timeout expires.

### 13.3 ProcessWatcher

`/proc` polling for process start/stop events. Configured via `ProcessWatchConfig`: `process_name`, `poll_interval_seconds`, `on_start_callback`, `on_stop_callback`.

---

## 14. Voice Subsystem

### 14.1 Overview

The voice subsystem provides full bidirectional audio: speech-to-text transcription of the user's microphone, text-to-speech synthesis of Cardinal's responses, wake-word passive listening, voice activity detection, and barge-in. It is owned by `CardinalAPI` as a single `VoiceLoop` `unique_ptr`. All audio resources are freed atomically when `VoiceLoop::stop()` is called.

**Files:** `src/voice/`

**Enabled at runtime:**
- Via `config.json` `voice.enabled=true` — starts with Cardinal
- Via `--voice` flag — starts after `api.init()`, before the interactive loop
- Via `/voice on [mode]` CLI command or `POST /api/voice/enable` HTTP endpoint

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

**Barge-in:** `VADDetector::onset_callback` fires during `SPEAKING` state → `AudioDevice::stop_playback()` immediately → state transitions to `INTERRUPTED` → pending segment pushed to work queue.

**Work queue:** `VoiceLoop` maintains a `std::queue<AudioChunk>` consumed by the main loop thread. The main thread processes segments serially: STT → inference → TTS.

### 14.3 AudioDevice

PortAudio wrapper with two independent streams.

**Capture stream:** Input from microphone, 16-bit signed PCM, mono, 16 kHz. Callback routes to VAD, PTT buffer, and WakeWordDetector.

**Playback stream:** Output to speaker with linear interpolation resampling when TTS sample rate (22050 Hz) ≠ device rate.
- `play(samples, sample_rate, on_done)` — enqueues chunk
- `stop_playback()` — clears queue immediately, fires all `on_done` callbacks
- `wait_until_done()` — blocks on condition variable until queue drained

**Device enumeration:** `list_devices()` returns all PortAudio devices. `input_device=-1` and `output_device=-1` select system defaults.

### 14.4 VADDetector

Energy-based RMS threshold detector. No external model.

**State machine:** `IDLE` → `SPEECH` (on energy threshold crossing) → `IDLE` (after post-speech silence).

**Pre-roll buffer:** A `std::deque<int16_t>` ring buffer stores the last `pre_speech_ms` of audio at all times. When speech onset is detected, the pre-roll is prepended to the speech buffer to capture any clipped leading audio.

**Segment emission rules:**
- `min_speech_ms`: discard segments shorter than this (noise suppression)
- `max_speech_ms`: hard-cap — emit immediately when reached
- `post_speech_ms`: silence window after which a segment is emitted

**Callbacks:** `set_segment_callback`, `set_onset_callback`, `set_offset_callback`

**`compute_rms(samples, n)` → float [0,1]:** static helper, `sqrt(sum(sample²/32768²) / n)`.

### 14.5 STTEngine

Whisper.cpp wrapper.

**Init:** `whisper_init_from_file_with_params()` with `use_gpu=true`, `gpu_device=0`.

**Transcription (`transcribe(AudioChunk)`):**
1. Convert `int16_t` samples to `float32` normalised to [-1, 1]
2. Build `whisper_full_params` with `no_context=true`, configured `language`, `n_threads`, `beam_size`, optional `initial_prompt`
3. `whisper_full()` — blocks on CUDA inference
4. Collect all segments via `whisper_full_n_segments()` / `whisper_full_get_segment_text()`

`TranscriptResult::empty()` returns true for `""`, `"[BLANK_AUDIO]"`, `"(blank)"` — these are discarded by `VoiceLoop`.

### 14.6 TTSEngine

Piper TTS wrapper. `piper.cpp` is compiled directly into Cardinal.

**Init:** `piper::initialize(PiperConfig)`, `piper::loadVoice(config, model_path, config_path, voice, speaker_id, use_cuda=false)`. ONNX Runtime CPU inference.

**`synthesise(TTSRequest)` → `TTSResult`:** blocks until complete, returns PCM samples at `voice.synthesisConfig.sampleRate` (typically 22050 Hz).

**`synthesise_streaming(text, request_template, on_sentence)`:**
1. `split_sentences(text)` → vector of sentence strings
2. For each sentence: `synthesise(req)` → `on_sentence(result)`
3. `VoiceLoop` feeds each result to `AudioDevice::play()` immediately

**`split_sentences(text)` — static:** splits on `.`, `?`, `!`, `\n`. Minimum 3 words required before emitting. Prevents micro-utterances from being synthesised as isolated clips.

### 14.7 WakeWordDetector

PocketSphinx keyword spotting. Offline, no API key, no network.

**Init:** `ps_config_init()`, set acoustic model, dictionary, keyphrase, threshold, sample rate.

**Recognition loop:** pops frames from bounded `frame_queue_`, feeds to `ps_process_raw()`, checks `ps_get_hyp()` for phrase match, fires `on_detected_` callback on detection.

**`push_frame(samples, n)`:** called from audio capture callback thread. Acquires `queue_mutex_`, trims overflow, inserts, notifies `queue_cv_`. Non-blocking from caller's perspective.

**`sensitivity`:** stored as `kws_threshold` log probability. Default `1e-20`. Lower = more sensitive.

### 14.8 Sentence-Streaming TTS

In `sentence` TTS streaming mode, the LLM stream callback accumulates tokens. `TTSEngine::split_sentences()` is called on each token append. When two or more sentences are detectable, all but the last (potentially incomplete) are synthesised and played immediately. The incomplete fragment is kept in the buffer and grows with subsequent tokens. On `is_final=true`, all remaining content is flushed through TTS.

This means synthesis of sentence 1 can overlap with LLM generation of sentence 2, achieving sub-400ms first-audio latency on the RTX 3050.

### 14.9 Barge-In

When state is `SPEAKING`, the VAD onset callback fires:
1. `audio_->stop_playback()` — immediately clears playback queue
2. `set_state(INTERRUPTED)` — signals the state machine
3. VAD continues to accumulate the new utterance
4. Completed segment pushed to work queue
5. Loop thread picks it up and begins new STT/inference/TTS cycle

### 14.10 Voice Tool — `voice_control`

**File:** `src/tools/builtin/voice/tool_voice_control.h/.cpp`

Registered when `voice.enabled=true`. Allows the LLM to control its own voice subsystem mid-conversation.

| Action | Arguments | Effect |
|--------|-----------|--------|
| `set_mode` | `mode`: `"push_to_talk"` / `"vad"` / `"wake_word"` | Calls `VoiceLoop::set_input_mode()` |
| `set_voice` | `voice`: model name string | Logs intent; requires restart to take effect |
| `set_volume` | `value`: `"0"`–`"100"` | Informational (system volume via OS) |
| `stop_speaking` | — | Calls `VoiceLoop::stop_speaking()` → `AudioDevice::stop_playback()` |

If voice is not active, returns a `"Voice subsystem is not active"` error without crashing.

### 14.11 Voice HTTP API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/voice/status` | Returns `VoiceStatus` JSON |
| POST | `/api/voice/enable` | Enables voice; optional `{"input_mode":"vad"}` body |
| POST | `/api/voice/disable` | Disables voice, frees all audio resources |
| POST | `/api/voice/speak` | TTS test: `{"text":"hello"}` |
| POST | `/api/voice/transcribe` | Raw 16-bit PCM body; `X-Sample-Rate` header (default 16000) |

---

## 15. Agentic Pipeline

Cardinal uses a **unified pipeline** for both chat and agentic modes. The only difference is the `max_iterations` setting. One code path, one audit trail format.

### 15.1 Agent Executor Loop

```
AgentExecutor::run(goal, trace_builder, progress_cb):

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

### 15.2 Tool Execution Loop (Integrated)

The same two-pass inference is used, but Pass 2 loops:

```
Pass 1: feeling output (unchanged)

Pass 2 loop:
  ┌─────────────────────────────────┐
  │ generate response               │
  │ detect tool call in output?     │
  │   yes → execute tool            │
  │       → inject result to context│
  │       → iteration < max?        │
  │           yes → loop back       │
  │           no  → force final     │
  │   no  → done, return response   │
  └─────────────────────────────────┘
```

### 15.3 Working Memory

Persistent SQLite-backed scratchpad at `data/memory/agent_working_memory`. Enables task resumption after interruption.

### 15.4 Self-Correction

When a tool call fails, the agent can:
1. Analyse the error
2. Adjust the plan
3. Retry the step (up to `self_correction_max_attempts`)

### 15.5 Notes

All computer use tools and the `voice_control` tool are available within the agentic loop. When Cardinal runs an `AGENT_RUN` task from the scheduler, it constructs an `AgentGoal` and calls `AgentExecutor::run()` with a fresh `TraceBuilder("scheduler", "scheduler", "")`.

---

## 16. Tools System

### 16.1 Registered Tools

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
| `schedule_task` | `scheduler.enabled` (headless-safe) |
| `voice_control` | `voice.enabled` |

### 16.2 Built-in Tools (non-computer-use)

| Tool | Description | Confirmation Required | Sandbox |
|------|-------------|----------------------|---------|
| `web_search` | DuckDuckGo search, no API key | Configurable | None |
| `web_fetch` | Fetch and parse a URL | Configurable | None |
| `calculator` | Safe math expression evaluator (muparser) | No | None |
| `run_python` | Python code execution | Yes (default) | Subprocess or Docker |
| `file_read` | Read from allowed paths | Yes (default) | Path restrictions |
| `file_write` | Write to allowed paths | Yes (default) | Path restrictions |
| `knowledge_graph_query` | Query Cardinal's own KG | No | None |
| `episodic_search` | Search Cardinal's memory | No | None |

### 16.3 Python Sandbox Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `subprocess` | Spawn subprocess with resource limits (ulimit) | Development, quick testing |
| `docker` | Full container isolation | Production deployment |

**Sandbox limits (both modes):** `timeout_seconds` (default 30s), `memory_limit_mb` (default 256MB), `network_enabled` (false by default).

### 16.4 ToolExecutor Dispatch

`ToolExecutor::dispatch()` routes by tool name. Computer use controllers are injected via setters called from `CardinalAPI::init()`:

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
tool_executor_->set_voice_loop(voice_loop_.get());
```

The `voice_loop_` setter is called from `CardinalAPI::enable_voice()` after `VoiceLoop::start()` succeeds, and cleared (set to `nullptr`) in `CardinalAPI::disable_voice()`.

---

## 17. Explainability Exports

Every inference produces a signed, tamper-evident trace.

### 17.1 Export Schema

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
    "episodes_retrieved": [
      {
        "id": "ep_123",
        "score": 0.85,
        "user_message": "...",
        "response_summary": "..."
      }
    ],
    "rules_active": [
      {
        "id": "rule_42",
        "condition": "...",
        "consequence": "...",
        "confidence": 0.87
      }
    ],
    "symbolic_checks": {
      "ran": true,
      "contradictions_found": 0,
      "rules_fired": ["rule_42", "rule_17"]
    },
    "tool_calls": [
      {
        "tool": "web_search",
        "input": {"query": "..."},
        "output": "...",
        "duration_ms": 342,
        "success": true
      }
    ],
    "pass1_tokens": 87,
    "pass2_tokens": 312,
    "total_ms": 1840
  },

  "rule_committed": {
    "committed": true,
    "rule_id": "rule_89",
    "condition": "...",
    "consequence": "...",
    "confidence": 0.82,
    "reasoning_type": "inductive"
  },

  "final_response": "...",

  "integrity": {
    "hash": "sha256:...",
    "signature": "ed25519:..."
  }
}
```

### 17.2 Audit Log Storage

SQLite database at `data/explainability/audit.db`:

```sql
CREATE TABLE traces (
    id            TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    inference_id  TEXT NOT NULL UNIQUE,
    session_id    TEXT NOT NULL,
    user_message  TEXT NOT NULL,
    trace_json    TEXT NOT NULL,
    hash          TEXT NOT NULL,
    signature     TEXT NOT NULL
);

CREATE INDEX idx_timestamp ON traces(timestamp);
CREATE INDEX idx_session ON traces(session_id);
```

### 17.3 Cryptographic Signing

- **Hash:** SHA256 of the trace JSON
- **Signature:** Ed25519 (private key in `data/explainability/cardinal_private.pem`)
- **Auto-generation:** Keys created on first start if `auto_generate_keys: true`

### 17.4 Export API

**POST `/api/explainability/export`**

Request:
```json
{
  "session_id": "my-session",
  "start_time": "2026-05-01T00:00:00Z",
  "end_time": "2026-05-05T23:59:59Z",
  "format": "json"
}
```

### 17.5 Notes

- Meta-correction rules that were applied during an inference appear in `rules_applied` with `reasoning_type = "meta_correction"`, making the self-improvement influence fully traceable and auditable.
- Scheduled task inferences do not produce audit log entries (no `AuditLog` pointer available in `SchedulerEngine`). Computer use tool calls appear in the normal inference trace when invoked from chat or the agentic loop.
- Voice utterances flow through the normal `chat_stream()` path and therefore produce full audit log entries. The `session_id` is `voice.session_id` (default `"voice_session"`).

---

## 18. Training Export

`TrainingExporter` exports high-confidence episodes and committed rules as Alpaca-format JSONL files suitable for LoRA fine-tuning with Axolotl, LLaMA-Factory, unsloth, and similar tools. This is the **manual** export path. The automatic Layer 3 pipeline uses `DatasetCurator` directly without going through `TrainingExporter`.

### 18.1 Output Format

Each line in the output file is one JSON object:

```json
{"instruction": "What happens to gas molecules when temperature increases?", "input": "", "output": "When temperature increases, gas molecules gain kinetic energy and move faster..."}
```

- `instruction` — the user message
- `input` — always empty for Cardinal exports
- `output` — the cleaned final response

### 18.2 Response Cleaning

Before writing to the export file, responses are cleaned:
1. `<think>...</think>` blocks are stripped — Qwen3's chain-of-thought traces are not useful as training targets
2. `<feeling_state>...</feeling_state>` blocks are stripped
3. Consecutive newlines are collapsed to a maximum of two
4. Leading and trailing whitespace is trimmed

If a tag is unclosed (rare, only on truncated responses), everything from the opening tag to the end of the string is removed.

### 18.3 Rule Export (optional)

When `include_rules = true`, committed rules are also exported as training examples:

```json
{"instruction": "What rule applies when: gas temperature increases?", "input": "", "output": "Gas molecules move faster and collide more frequently, increasing pressure."}
```

### 18.4 Filtering

| Field | Default | Description |
|-------|---------|-------------|
| `min_confidence` | 0.7 | Minimum episode confidence |
| `domain` | "" (all) | Filter by reasoning domain |
| `max_examples` | 0 (no limit) | Maximum examples to export |
| `include_rules` | false | Also export rules |
| `recent_first` | true | Most recent episodes first |

### 18.5 Dry Run

`export_dry_run()` returns an `ExportInfo` showing how many examples would be exported without writing any files.

---

## 19. Configuration Reference

All configuration lives in `config.json`. Every field is validated at startup. Cardinal will not start if any required field is missing or invalid.

### 19.1 `backend` section

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"llama_cpp"` or `"tensorrt"` |
| `llama_cpp.path` | string | Absolute path to the primary GGUF model file. Must exist. |
| `llama_cpp.chat_template` | string | Chat template format: `qwen3`, `llama3`, `mistral` |
| `llama_cpp.context_length` | int | Maximum context window in tokens (>= 512) |
| `llama_cpp.gpu_layers` | int | Number of layers to offload to GPU (0 = CPU only) |
| `llama_cpp.threads` | int | CPU threads for non-GPU work (>= 1) |
| `tensorrt.*` | various | TensorRT-specific engine paths and config |

### 19.2 `inference` section

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | float | Sampling temperature (0.0 to 2.0) |
| `top_p` | float | Nucleus sampling (0.0 to 1.0) |
| `max_tokens_feeling` | int | Max tokens for Pass 1 (>= 32) |
| `max_tokens_response` | int | Max tokens for Pass 2 (>= 64) |
| `max_retries` | int | Maximum Pass 1 retry attempts (>= 0) |
| `retry_delay_ms` | int | Milliseconds between retries (>= 0) |

### 19.3 `feeling_schema` section

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"structured_json"` |
| `grammar_path` | string | Absolute path to the GBNF grammar file. Must exist. |
| `fields` | object | Field definitions (informational, not parsed at runtime) |
| `max_tokens` | int | Alias for `inference.max_tokens_feeling` |

### 19.4 `memory` section

| Field | Type | Description |
|-------|------|-------------|
| `rule_store_path` | string | Path to `rules.json`. Created on first save. |
| `knowledge_graph_path` | string | Path to `knowledge.json`. Created on first save. |
| `episodic_log_path` | string | Path to the JSONL episode log. Created on first episode. |
| `max_rules` | int | Maximum number of rules in the store (>= 1) |

### 19.5 `verifier` section

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `"symbolic"`, `"neural"`, or `"hybrid"` |
| `neural_model_path` | string | Path to neural verifier GGUF. Empty string disables neural verifier. |
| `neural_gpu_layers` | int | GPU layers for neural verifier (0 = CPU only) |
| `neural_max_tokens` | int | Max tokens for neural verifier output (default 512) |
| `contradiction_threshold` | float | Score above which a contradiction is flagged (0.0 to 1.0) |
| `rule_confidence_decay` | float | Confidence reduction per maintenance cycle (>= 0) |
| `min_rule_confidence` | float | Rules below this are pruned (0.0 to 1.0) |

### 19.6 `retriever` section

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `"keyword"`, `"semantic"`, or `"hybrid"` |
| `keyword_weight` | float | Weight for keyword score in hybrid (0.0 to 1.0) |
| `semantic_weight` | float | Weight for semantic score in hybrid (0.0 to 1.0) |
| `max_results` | int | Maximum episodes returned per retrieval (>= 1) |
| `min_score` | float | Minimum combined score to include in results (0.0 to 1.0) |
| `cache_rebuild_strategy` | string | `"on_demand"`, `"periodic"`, or `"explicit"` |
| `cache_rebuild_threshold` | int | Episodes added before on_demand rebuild (>= 1) |
| `cache_rebuild_interval_seconds` | int | Seconds between periodic rebuilds (>= 1) |

### 19.7 `tools` section

```json
"tools": {
  "home_access": true,
  "web_search": {
    "enabled": true,
    "confirmation_required": false,
    "max_results": 5,
    "timeout_seconds": 10
  },
  "web_fetch": {
    "enabled": true,
    "confirmation_required": false,
    "allowed_domains": [],
    "blocked_domains": [],
    "timeout_seconds": 15,
    "max_content_kb": 512
  },
  "calculator": {
    "enabled": true,
    "confirmation_required": false
  },
  "run_python": {
    "enabled": true,
    "confirmation_required": true,
    "sandbox_mode": "subprocess",
    "docker_image": "python:3.12-slim",
    "timeout_seconds": 30,
    "memory_limit_mb": 256,
    "network_enabled": false
  },
  "file_read": {
    "enabled": true,
    "confirmation_required": true,
    "allowed_paths": ["data/", "logs/", "~/Downloads"]
  },
  "file_write": {
    "enabled": true,
    "confirmation_required": true,
    "allowed_paths": ["data/output/", "~/Downloads"]
  }
}
```

### 19.8 `agent` section

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | bool | Enable agentic mode |
| `max_iterations` | int | Default max iterations per agentic run |
| `max_iterations_hard_cap` | int | Hard cap regardless of request |
| `working_memory_path` | string | Path to agent working memory SQLite |
| `working_memory_size` | int | Maximum entries in working memory |
| `self_correction_enabled` | bool | Allow the agent to retry failed steps |
| `self_correction_max_attempts` | int | Maximum retry attempts per step |
| `plan_before_execute` | bool | Run PLAN phase before EXECUTE loop |
| `summarize_on_cap` | bool | Summarize result when max_iterations hit |

### 19.9 `explainability` section

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | bool | Enable explainability system |
| `audit_log_path` | string | Path to audit SQLite database |
| `signing_enabled` | bool | Enable Ed25519 cryptographic signing |
| `private_key_path` | string | Path to Ed25519 private key PEM |
| `public_key_path` | string | Path to Ed25519 public key PEM |
| `auto_generate_keys` | bool | Generate keys on first start if missing |
| `export_path` | string | Directory for explainability export files |
| `attach_trace_to_response` | bool | Include trace in `ChatResponse` |

### 19.10 `vision` section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Master switch for vision subsystem |
| `model_path` | string | — | Path to moondream2 text GGUF |
| `mmproj_path` | string | — | Path to moondream2 mmproj (projector) |
| `gpu_layers` | int | 0 | GPU layers for vision model |
| `threads` | int | 4 | CPU threads |
| `max_tokens` | int | 512 | Max tokens for vision response |
| `cache_path` | string | `data/vision_cache/` | Directory for cached downloaded images |
| `cache_ttl_hours` | int | `24` | Hours to keep cached images (`0` = never delete) |
| `allowed_paths` | array | `[]` | Local directories that `analyze_image` may read |

### 19.11 `self_improvement` section

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

### 19.12 `api` section

| Field | Type | Description |
|-------|------|-------------|
| `http_enabled` | bool | Enable HTTP server on startup |
| `host` | string | Bind address (`"127.0.0.1"` for localhost only) |
| `port` | int | Listen port (1 to 65535) |
| `auth_enabled` | bool | Require Bearer token on all requests (except `/api/health`) |
| `api_key` | string | The Bearer token. Required if `auth_enabled` is true. |
| `stream_enabled` | bool | Allow SSE streaming responses |
| `max_request_size_kb` | int | Maximum request body size in kilobytes (>= 1) |
| `request_timeout_seconds` | int | Per-request timeout (>= 1) |

### 19.13 `scheduler` section

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

### 19.14 `computer_use` section

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

### 19.15 `voice` section

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

**Key design decisions:**
- `voice.enabled=false` by default — audio hardware not touched unless explicitly requested
- `voice.session_id` is a dedicated session for voice inference, separate from any text CLI session — allows concurrent text and voice use without history contamination
- `voice.stt.gpu_layers=8` — Whisper medium.en on RTX 3050 uses 8 GPU layers by default; reduce if OOM with the LLM loaded
- `voice.tts.sample_rate=22050` — Piper's native output rate; PortAudio resamples to device rate
- `voice.audio.sample_rate=16000` — Whisper requires 16 kHz input

### 19.16 `benchmark` section

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | Path to benchmark dataset file |
| `metrics` | array of strings | Metrics to evaluate: `consistency`, `parseability`, `accuracy` |
| `eval_frequency_seconds` | int | Seconds between auto-evaluations (>= 1) |

### 19.17 `logging` section

| Field | Type | Description |
|-------|------|-------------|
| `level` | string | Minimum log level: `trace`, `debug`, `info`, `warn`, `error`, `fatal` |
| `path` | string | Absolute path to the log file |

---

## 20. CardinalAPI Reference

`CardinalAPI` is the single entry point for all interfaces. It owns all core components and exposes a clean, exception-free interface.

**File:** `src/api/cardinal_api.h`

### 20.1 Lifecycle

```cpp
CardinalAPI api;

CardinalVoidResult init(const std::string& config_path = "config.json");
CardinalVoidResult shutdown();
bool is_initialized() const;
CardinalVoidResult health_check() const;
std::string        uptime_string() const;
```

`init()` performs the complete startup sequence. If any component fails to initialize, `init()` returns a failure result with a descriptive message. No partial state is left — if `init()` fails, the API is in a clean uninitialized state.

`shutdown()` runs final maintenance, saves rule store, destroys all sessions, and closes the SQLite database. Safe to call multiple times — subsequent calls are no-ops.

The destructor calls `shutdown()` automatically if the API was initialized and not yet shut down.

`health_check()` returns `OK` if initialized and not shutting down.

`uptime_string()` returns a formatted string like `"02h 14m 37s"`.

### 20.2 Session Management

```cpp
CardinalResult<std::string> create_session(const std::string& session_id = "");
CardinalVoidResult destroy_session(const std::string& session_id);
CardinalVoidResult reset_session(const std::string& session_id);
CardinalResult<SessionInfo> get_session(const std::string& session_id) const;
CardinalResult<std::vector<std::string>> list_sessions() const;
```

Sessions are created automatically on the first `chat()` call if the session does not exist. The default session is `"default"`. `destroy_session()` calls `on_session_boundary()` internally, which applies any pending LoRA adapter.

### 20.3 Inference

```cpp
CardinalResult<ChatResponse> chat(
    const std::string& session_id,
    const std::string& message);

CardinalResult<ChatResponse> chat_stream(
    const std::string&       session_id,
    const std::string&       message,
    const ApiStreamCallback& stream_cb);
```

Both methods run the full inference pipeline: retrieve episodes, inject memory context, Pass 1 (feeling), Pass 2 (response), dual-write to JSONL + SQLite, consistency check, rule extraction, contradiction resolution, link rule to episode, save rule store if dirty, update session history.

`chat()` calls `chat_stream()` with a null callback.

If `stream_cb` returns false, generation is aborted.

### 20.4 Agentic Inference

```cpp
CardinalResult<AgentResult> agent(
    const std::string& session_id,
    const std::string& goal,
    int max_iterations = 0);
```

### 20.5 Memory & Stats

```cpp
CardinalResult<SystemStats>              get_stats() const;
CardinalResult<std::vector<RuleInfo>>    get_rules() const;
CardinalResult<std::vector<EpisodeInfo>> get_episodes(
    const std::string& keyword = "",
    const std::string& domain  = "",
    float min_conf             = 0.0f,
    int   max_results          = 20) const;
CardinalResult<ScanResult>               run_scan();
CardinalVoidResult                       run_maintenance();
```

`run_scan()` runs `ConsistencyChecker::run_full_scan()` — checks every rule against every other rule in the Prolog engine. O(n²) for large rule bases.

`run_maintenance()` manually triggers decay/prune/save. Also runs automatically every 10 inferences.

### 20.6 Training Export

```cpp
CardinalResult<ExportInfo> export_training_data(const ExportRequest& request);
CardinalResult<ExportInfo> export_dry_run(const ExportRequest& request) const;
```

### 20.7 Explainability

```cpp
CardinalResult<std::string> get_trace(const std::string& inference_id) const;
CardinalResult<std::string> export_trace(
    const std::string& session_id,
    const std::string& start_time,
    const std::string& end_time) const;
CardinalResult<bool> verify_trace(const std::string& trace_json) const;
std::string          get_public_key() const;
```

### 20.8 Settings

```cpp
CardinalResult<CardinalSettings> get_settings() const;
CardinalVoidResult update_settings(const CardinalSettings& settings);
CardinalVoidResult set_setting(const std::string& key, const std::string& value);
CardinalVoidResult reset_settings();
```

Valid keys for `set_setting()`: `retriever_mode`, `keyword_weight`, `semantic_weight`, `max_retrieval_results`, `min_retrieval_score`, `verifier_mode`, `min_rule_confidence`, `contradiction_threshold`, `temperature`, `top_p`, `stream_responses`, `log_level`.

### 20.9 Self-Improvement

```cpp
CardinalResult<SelfImprovementStatus> get_self_model_status() const;
CardinalResult<ReflectionResult>      reflect();
CardinalResult<bool>                  trigger_training(const std::string& domain_hint = "");
void                                  on_session_boundary();
```

`reflect()` triggers an on-demand Layer 2 reflection pass (synchronous — may take several seconds).

`trigger_training()` posts a Layer 3 training request to the background thread (async, returns immediately).

### 20.10 Scheduler API

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

### 20.11 Computer Use API

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

### 20.12 Voice API

```cpp
CardinalVoidResult               enable_voice(const std::string& input_mode = "");
CardinalVoidResult               disable_voice();
bool                             is_voice_active() const;
VoiceStatus                      get_voice_status() const;
CardinalResult<TTSResult>        voice_speak(const std::string& text);
CardinalResult<TranscriptResult> voice_transcribe(const AudioChunk& audio);
```

`enable_voice()` constructs a `VoiceChatStreamFn` lambda that calls `chat_stream()` — this is the bridge between the voice loop's LLM calls and the full inference pipeline. The voice loop has no direct dependency on `CardinalAPI`.

---

## 21. HTTP API Reference

**Base URL:** `http://127.0.0.1:8080` (configurable via `api.host` and `api.port`)

**Authentication:** All endpoints except `/api/health` require:
```
Authorization: Bearer <api_key>
```

When `auth_enabled` is false in config, the auth header is not required.

**CORS:** All endpoints include CORS headers. `Access-Control-Allow-Origin: *` is set on all responses. OPTIONS preflight requests return 204.

**Error format:**
```json
{
  "error": "Human-readable error message",
  "status": 6,
  "code": "INVALID_INPUT"
}
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check (no auth). Returns `{"status":"ok","uptime":"...","version":"2.0.0"}` |
| POST | `/api/chat` | Chat. Body: `{"session_id":"...","message":"...","stream":false}`. Supports SSE. |
| POST | `/api/sessions` | Create session. Body: `{"session_id":"..."}` (optional). |
| DELETE | `/api/sessions/:id` | Destroy session. |
| POST | `/api/sessions/:id/reset` | Reset session history. |
| GET | `/api/stats` | Full system statistics (`SystemStats`). |
| GET | `/api/rules` | Rule store contents. |
| GET | `/api/episodes` | Episode query. Params: `keyword`, `domain`, `min_conf`, `max_results`. |
| POST | `/api/scan` | Full contradiction scan. Response: `{"total_contradictions":2,"resolved":1,"flagged":1,"skipped":0}`. |
| POST | `/api/maintenance` | Manual maintenance cycle. |
| GET | `/api/settings` | Get current runtime settings. |
| POST | `/api/settings` | Update runtime settings. Partial JSON accepted. |
| POST | `/api/export` | Export training data (Alpaca JSONL). |

### Self-Improvement Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/self_model` | Returns `SelfImprovementStatus` |
| POST | `/api/reflect` | Trigger on-demand Layer 2 reflection (synchronous) |
| POST | `/api/train` | Post Layer 3 training request (async). Body: `{"domain_hint":"factual"}` (optional). |

### Explainability Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/explainability/export` | Export signed traces for a session/time range |
| GET | `/api/explainability/download/:id` | Download exported trace file |

### Scheduler Endpoints

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

### Computer Use Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/computer/status` | Display info |
| POST | `/api/computer/screenshot` | Capture screen |
| POST | `/api/computer/click` | Click at coordinates |
| POST | `/api/computer/type` | Type text or send key |
| POST | `/api/computer/shell` | Run shell command |

### Voice Endpoints

| Method | Endpoint | Body / Notes | Description |
|--------|----------|--------------|-------------|
| GET | `/api/voice/status` | — | Returns `VoiceStatus` |
| POST | `/api/voice/enable` | `{"input_mode":"vad"}` (optional) | Enable voice |
| POST | `/api/voice/disable` | — | Disable voice |
| POST | `/api/voice/speak` | `{"text":"hello"}` | TTS synthesis + playback |
| POST | `/api/voice/transcribe` | Raw 16-bit PCM body; `X-Sample-Rate` header | STT transcription |

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

**Transcribe example (raw PCM):**
```bash
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

### TypeScript Quick Start

```typescript
const BASE_URL = "http://127.0.0.1:8080";
const API_KEY  = "cardinal-dev-key-change-in-production";

const headers = {
  "Content-Type":  "application/json",
  "Authorization": `Bearer ${API_KEY}`
};

// Non-streaming chat
async function chat(message: string, sessionId = "default") {
  const res = await fetch(`${BASE_URL}/api/chat`, {
    method:  "POST",
    headers,
    body: JSON.stringify({ session_id: sessionId, message })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();  // ChatResponse
}

// Streaming chat
async function streamChat(
  message: string,
  onToken: (token: string) => void,
  sessionId = "default"
) {
  const res = await fetch(`${BASE_URL}/api/chat`, {
    method:  "POST",
    headers: { ...headers, Accept: "text/event-stream" },
    body: JSON.stringify({ session_id: sessionId, message, stream: true })
  });

  const text = await res.text();
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = JSON.parse(line.slice(6));
    if (data.is_final) return data.feeling;
    onToken(data.token);
  }
}

// Get system stats
async function getStats() {
  const res = await fetch(`${BASE_URL}/api/stats`, { headers });
  return res.json();  // SystemStats
}
```

---

## 22. Settings Manager

`SettingsManager` manages runtime-mutable settings. Changes propagate immediately to core components without restart.

**File:** `src/api/cardinal_settings.h/cpp`

### 22.1 Mutable Settings

| Setting | Type | Propagates to |
|---------|------|---------------|
| `retriever_mode` | string | `EpisodicRetriever::set_mode()` |
| `keyword_weight` | float | `EpisodicRetriever::set_weights()` |
| `semantic_weight` | float | `EpisodicRetriever::set_weights()` |
| `max_retrieval_results` | int | Used on next retrieval call |
| `min_retrieval_score` | float | Used on next retrieval call |
| `verifier_mode` | string | Validated, stored |
| `min_rule_confidence` | float | Validated, stored |
| `contradiction_threshold` | float | Validated, stored |
| `temperature` | float | Validated, stored |
| `top_p` | float | Validated, stored |
| `stream_responses` | bool | Used on next chat call |
| `log_level` | string | Logger level |

### 22.2 Validation

All settings are validated before any change is applied. If validation fails, no settings are changed — the update is atomic.

### 22.3 Partial JSON Updates

`from_json()` starts from the current settings and applies only the fields present in the incoming JSON. A POST to `/api/settings` with one field only changes that one field — all others remain unchanged.

### 22.4 Reset

`reset()` restores all settings to the values originally loaded from `config.json` at startup. It does not reload the config file.

---

## 23. Session Manager

`SessionManager` owns all active `ConversationSession` objects.

**File:** `src/api/session.h/cpp`

### 23.1 ConversationSession

Each session tracks:
- `session_id` — unique identifier
- `turn_count` — number of user/assistant pairs (incremented on user turn only)
- `history` — `std::vector<ChatMessage>` for direct pipeline use
- `timestamps_` — parallel vector of ISO 8601 timestamps for each history entry
- `working_memory` — agentic session scratchpad
- `created_at` — ISO 8601 timestamp when session was created
- `last_active_at` — ISO 8601 timestamp of last activity

### 23.2 History Management

`add_user_turn(content)` appends a user message and increments `turn_count_`.

`add_assistant_turn(content)` appends an assistant message but does NOT increment `turn_count_` — one turn is one user+assistant pair.

`get_history()` returns the raw `std::vector<ChatMessage>` used by `InferenceRequest`.

`get_chat_turns()` returns `std::vector<ChatTurn>` with role, content, and timestamp — this is the API-boundary type used in `SessionInfo`.

### 23.3 Context Window Management

`estimated_token_count()` provides a rough estimate (1 token per 4 characters).

`trim_to_token_budget(max_tokens, min_turns_to_keep)` removes the oldest history entries until the estimated token count is within budget. It always keeps at least `min_turns_to_keep` user/assistant pairs.

### 23.4 Thread Safety

`SessionManager` protects its session map with a `std::mutex` on all operations except `get()`. The `get()` method has no lock — it is called from `CardinalAPI` which holds its own `session_mutex_` for the duration of any session access.

---

## 24. Type Reference

All types defined in `src/api/cardinal_types.h`. All types are pybind11-compatible: copyable, no raw pointers, `std::string` strings, `std::vector` collections.

### Core Result Types

```cpp
enum class CardinalStatus : int {
    OK                   = 0,
    NOT_INITIALIZED      = 1,
    ALREADY_INITIALIZED  = 2,
    INFERENCE_FAILED     = 3,
    STORAGE_ERROR        = 4,
    CONFIG_ERROR         = 5,
    INVALID_INPUT        = 6,
    EXPORT_FAILED        = 7,
    SESSION_NOT_FOUND    = 8,
    AUTH_FAILED          = 9,
    TIMEOUT              = 10,
    SHUTDOWN             = 11,
    COMPUTER_USE_ERROR   = 15,
    SCHEDULER_ERROR      = 16,
    VOICE_ERROR          = 17,
    INTERNAL_ERROR       = 99
};

template<typename T>
struct CardinalResult {
    CardinalStatus status;
    std::string    error_message;
    T              value;
    bool ok() const;
    static CardinalResult<T> success(T val);
    static CardinalResult<T> failure(CardinalStatus s, const std::string& msg);
};

struct CardinalVoidResult {
    CardinalStatus status;
    std::string    error_message;
    bool ok() const;
    static CardinalVoidResult success();
    static CardinalVoidResult failure(CardinalStatus s, const std::string& msg);
};
```

### Inference Types

```cpp
struct FeelingInfo {
    float       confidence;
    std::string reasoning_type;
    std::string reasoning_domain;
    bool        uncertainty_flag;
    bool        contradiction_flag;
    bool        rule_candidate;
};

struct ChatResponse {
    std::string session_id;
    std::string response;
    FeelingInfo feeling;
    std::string episode_id;
    bool        rule_committed;
    std::string committed_rule_id;
    int         contradictions_found;
    int         contradictions_resolved;
    int         contradictions_flagged;
    int         pass1_tokens;
    int         pass2_tokens;
    int         total_ms;
};

struct StreamToken {
    std::string session_id;
    std::string token;
    bool        is_final;
    FeelingInfo feeling;   // only populated when is_final is true
};

using ApiStreamCallback = std::function<bool(const StreamToken&)>;
```

Returning `false` from `ApiStreamCallback` aborts generation early.

### Session Types

```cpp
struct ChatTurn {
    std::string role;       // "user" or "assistant"
    std::string content;
    std::string timestamp;
};

struct SessionInfo {
    std::string           session_id;
    int                   turn_count;
    std::vector<ChatTurn> history;
    std::string           created_at;
    std::string           last_active_at;
};
```

### Memory Types

```cpp
struct RuleInfo {
    std::string id;
    std::string domain;
    std::string condition;
    std::string consequence;
    float       confidence;
    int         trigger_count;
    std::string episode_id;
    std::string reasoning_type;
    std::string created_at;
    std::string updated_at;
    bool        has_provenance;
};

struct EpisodeInfo {
    std::string id;
    std::string timestamp;
    std::string user_message;
    std::string response_summary;
    float       confidence;
    std::string reasoning_type;
    std::string reasoning_domain;
    bool        contradiction;
    bool        uncertainty;
    bool        rule_candidate;
    std::string extracted_rule_id;
    int         pass1_tokens;
    int         pass2_tokens;
    int         total_ms;
};

struct MemoryStats {
    int   total_episodes;
    int   migrated_episodes;
    int   high_conf_episodes;
    int   rule_candidate_count;
    float avg_episode_confidence;
    int   total_rules;
    int   active_rules;
    float avg_rule_confidence;
    int   index_size;
    int   vocabulary_size;
    bool  index_ready;
};

struct VerifierStats {
    int total_checks;
    int total_rules_extracted;
    int total_contradictions;
    int total_resolved;
    int total_flagged;
    int total_maintenance_runs;
};

struct SystemStats {
    MemoryStats   memory;
    VerifierStats verifier;
    std::string   uptime_seconds;
    std::string   version;
    bool          initialized;
};
```

### Export Types

```cpp
struct ExportRequest {
    std::string output_path;
    float       min_confidence;   // default 0.7
    std::string domain;           // empty = all
    int         max_examples;     // 0 = no limit
    bool        include_rules;    // default true
};

struct ExportInfo {
    int         episodes_exported;
    int         rules_exported;
    int         total_exported;
    float       avg_confidence;
    std::string output_path;
    std::string timestamp;
};

struct ScanResult {
    int total_contradictions;
    int resolved;
    int flagged;
    int skipped;
};

struct TrainingExample {
    std::string instruction;
    std::string input;
    std::string output;
    std::string domain;
    float       confidence;
    std::string episode_id;
    std::string reasoning_type;
    std::string timestamp;
    std::string source;         // "episode" | "rule"
};
```

### Settings Type

```cpp
struct CardinalSettings {
    std::string retriever_mode;
    float       keyword_weight;
    float       semantic_weight;
    int         max_retrieval_results;
    float       min_retrieval_score;
    std::string verifier_mode;
    float       min_rule_confidence;
    float       contradiction_threshold;
    float       temperature;
    float       top_p;
    bool        stream_responses;
    std::string log_level;
};
```

### Vision Type

```cpp
struct VisionResult {
    bool success;
    std::string description;
    std::string error_message;
    int width = 0;
    int height = 0;
    std::string mime_type;
    size_t file_size_bytes = 0;
    std::chrono::milliseconds encode_ms;
};
```

### Self-Improvement Types (`src/self_model/self_model_types.h`)

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

struct SelfModelSnapshot {
    std::vector<DomainStats>        domain_stats;
    std::vector<ReasoningTypeStats> reasoning_stats;
    std::vector<PerformanceTrend>   trends;
    std::string weakest_domain()   const;
    std::string strongest_domain() const;
    std::string format_for_prompt(int max_chars = 500) const;
};

struct ReflectionFinding {
    std::string domain;
    std::string pattern;
    std::string recommendation;
    float       confidence;
    std::string timestamp;
};

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

### Scheduler Types (`src/scheduler/scheduler_types.h`)

`TriggerType`, `TriggerSpec`, `TaskActionType`, `OutputTarget`, `TaskAction`, `ScheduledTask`, `TaskRunStatus`, `TaskRun`, `TaskActionLog`, `TaskParseResult`, `SchedulerStatus` — see §11.2 for struct definitions.

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

### Computer Use Types (`src/computer/computer_types.h`)

`DisplayServer` enum: `X11`, `WAYLAND`, `HEADLESS`

`Point`: `int x, y`  
`ScreenRegion`: `int x, y, width, height`  
`ScreenInfo`: `width`, `height`, `scale_factor`, `server`, `display_var`  
`Screenshot`: `path`, `width`, `height`, `timestamp`, `analyzed`, `description`, `region`  
`MouseButton` enum: `LEFT`, `RIGHT`, `MIDDLE`  
`BrowserActionType` enum: `NAVIGATE`, `CLICK`, `CLICK_TEXT`, `TYPE`, `SCROLL`, `GET_CONTENT`, `SCREENSHOT`, `EXECUTE_JS`, `NEW_TAB`, `CLOSE_TAB`, `BACK`, `FORWARD`, `RELOAD`  
`BrowserResult`: `success`, `error_message`, `url`, `content`, `screenshot_path`, `duration_ms`  
`ShellResult`: `success`, `exit_code`, `stdout_text`, `stderr_text`, `duration_ms`, `command`  
`FileOpResult`: `success`, `error_message`, `entries` (vector of `FileEntry`), `dest_path`  
`FileEntry`: `name`, `path`, `is_dir`, `size_bytes`, `permissions`, `modified_at`  
`SystemState`: `volume_pct`, `muted`, `brightness_pct`, `wifi_enabled`, `wifi_ssid`, `bluetooth_enabled`, `battery_pct`, `battery_charging`  
`EmailQuery`: `folder`, `subject_contains`, `from_contains`, `unread_only`, `max_results`  
`EmailMessage`: `id`, `message_id`, `from`, `to`, `subject`, `date`, `body_text`, `body_html`, `unread`  
`EmailSendRequest`: `to` (vector), `cc` (vector), `subject`, `body`, `html_body`  
`AppInfo`: `window_id`, `name`, `title`, `pid`, `focused`  
`AtSpiNode`: `role`, `name`, `bounds` (x,y,w,h), `states` (vector), `children` (vector)  
`ImageMetadata`: `width`, `height`, `file_size`, `format`, `source`, `origin`, `cache_path`

### Voice Types (`src/voice/voice_types.h`)

`VoiceInputMode` enum: `PUSH_TO_TALK`, `VAD`, `WAKE_WORD`  
`TTSStreamingMode` enum: `SENTENCE`, `FULL`  
`VoiceLoopState` enum: `IDLE`, `PASSIVE_LISTENING`, `LISTENING`, `RECORDING`, `TRANSCRIBING`, `INFERRING`, `SPEAKING`, `INTERRUPTED`, `STOPPING`

```cpp
struct AudioChunk {
    std::vector<int16_t> samples;
    int                  sample_rate = 16000;
    int                  channels    = 1;
    float duration_ms() const;
    bool  empty()       const;
};

struct TranscriptResult {
    bool        success        = false;
    std::string text;
    float       confidence     = 0.0f;
    int         duration_ms    = 0;
    std::string error_message;
    bool        empty()        const;
};

struct TTSRequest {
    std::string text;
    int         speaker_id   = 0;
    float       length_scale = 1.0f;
    float       noise_scale  = 0.667f;
    float       noise_w      = 0.8f;
};

struct TTSResult {
    bool                 success      = false;
    std::vector<int16_t> samples;
    int                  sample_rate  = 22050;
    int                  duration_ms  = 0;
    std::string          error_message;
};

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

---

## 25. Error Handling

### 25.1 API Boundary

No C++ exceptions cross the `CardinalAPI` boundary. Every method either returns `CardinalResult<T>` or `CardinalVoidResult`. All internal exceptions are caught in `CardinalAPI` methods and converted to failure results.

```cpp
try {
    // ... operation
    return CardinalResult<T>::success(value);
}
catch (const std::exception& e) {
    LOG_WARN("context: " + std::string(e.what()));
    return CardinalResult<T>::failure(
        CardinalStatus::INTERNAL_ERROR,
        std::string("context: ") + e.what());
}
```

### 25.2 Internal Exception Types

Within the core (below the API boundary), components throw typed exceptions:

| Exception | Thrown by |
|-----------|-----------|
| `ParseError` | `JsonParser` |
| `ConfigError` | `ConfigLoader` |
| `RuleStoreError` | `RuleStore` |
| `EpisodicStorageError` | `EpisodicStorage` |
| `EpisodicRetrieverError` | `EpisodicRetriever` |
| `InferenceError` | `InferencePipeline` |
| `ConsistencyCheckError` | `ConsistencyChecker` |
| `RuleExtractorError` | `RuleExtractor` |
| `TrainingExporterError` | `TrainingExporter` |
| `HttpServerError` | `HttpServer` |

All are derived from `std::runtime_error`.

### 25.3 Non-Fatal Paths

These failures are intentionally non-fatal:
- Retrieval failure — inference continues without memory context
- Neural verifier failure — symbolic verifier result used alone
- Contradiction check failure — logged, candidate rule proceeds without verification
- JSONL migration parse errors — malformed lines skipped with a warning
- Rule store duplicate IDs — `INSERT OR IGNORE` silently skips duplicates
- Tool execution failure — agentic self-correction handles it
- Reflection pass LLM failure — logged, Layer 2 skipped
- Training subprocess non-zero exit — logged, no adapter loaded
- Adapter load failure below improvement threshold — rejected gracefully
- STT blank result — segment discarded silently, returns to LISTENING
- TTS synthesis error — logged, no audio played

### 25.4 Fatal Paths

These cause `init()` to return failure:
- Config file not found or invalid JSON
- Model file not found
- Grammar file not found
- SQLite database cannot be opened or created
- ILLMBackend model fails to load
- Explainability key generation failure (if `auto_generate_keys=true` and key generation fails)

### 25.5 Graceful Degradation

- `computer_use.enabled=true` but display not detected → computer use controllers not initialised; tools not registered; rest of system unaffected
- `scheduler.enabled=true` but SQLite fails to open → scheduler not started; rest of system unaffected
- Browser venv missing → `browser` tool not registered
- Email disabled → `email` tool not registered
- `mtmd` library missing at compile time → vision disabled; `analyze_image` returns error
- `vision.model_path` or `mmproj_path` missing → vision disabled at runtime
- `self_improvement.enabled=true` but SelfModel DB fails to open → Layer 1 disabled; inference continues normally
- `voice.enabled=true` but `STTEngine::init()` fails → warning logged, voice not started, system continues in text-only mode
- `voice.enabled=true` but `AudioDevice::init()` fails (no audio hardware) → same — non-fatal
- `--voice` flag set but voice fails to start → warning printed to stderr, falls back to text-only mode

---

## 26. Offline Builds & Vendoring

All vendor dependencies are cloned manually. No git submodules.

```
vendor/
    llama.cpp/          git clone https://github.com/ggerganov/llama.cpp
    nlohmann_json/      git clone https://github.com/nlohmann/json
    cpp-httplib/        git clone https://github.com/yhirose/cpp-httplib
    muparser/           git clone https://github.com/beltoforion/muparser
    tokenizers-cpp/     git clone https://github.com/mlc-ai/tokenizers-cpp
    whisper.cpp/        git clone https://github.com/ggerganov/whisper.cpp
    piper/              git clone https://github.com/rhasspy/piper
        install/        pre-built via cmake --target install
    portaudio/          git clone https://github.com/PortAudio/portaudio
    pocketsphinx/       git clone https://github.com/cmusphinx/pocketsphinx
        install/        pre-built via cmake --target install
```

Only source and headers are tracked in Git. No `.git`, `.github`, `tests/`, `examples/`, `docs/`, or build artifacts.

**System dependencies** installed via `apt` (not vendored): `build-essential`, `cmake`, `libsqlite3-dev`, `libssl-dev`, `swi-prolog`

**CMake integration strategy:**

| Library | Strategy | Reason |
|---------|----------|--------|
| whisper.cpp | `add_subdirectory` | Has a proper library CMake target (`whisper`) |
| portaudio | `add_subdirectory` | Has a proper library CMake target (`portaudio`) |
| pocketsphinx | Pre-built into `install/` | `add_subdirectory` breaks: sources `#include <pocketsphinx.h>` which only exists after the build generates it |
| piper | Pre-built into `install/` + `piper.cpp` compiled directly into Cardinal | Builds an executable, not a library |

**Piper's flat install layout:** Piper installs all `.so` files directly into the install prefix root (not `lib/`). CMake `find_library` calls use `PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH`. Runtime `LD_LIBRARY_PATH` must include `vendor/piper/install/`.

CMake auto-detects `mtmd` from `vendor/llama.cpp/tools/mtmd/` and sets `CARDINAL_MTMD_AVAILABLE`. If `mtmd` is not found, vision is disabled (build continues).

Playwright (Python) is installed into `cardinal-browser-venv`. Email dependencies optionally installed into the same venv.

**Offline build command:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

No internet required after vendored dependencies are in place.

---

## 27. Module Reference

### 27.1 Logger (`src/utils/logger.h/cpp`)

Thread-safe singleton. Levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL.

Output format: `[TIMESTAMP] [LEVEL] [file:line] message`

ANSI color codes applied to console output only. Both file and console output happen on every log call.

```cpp
Logger::instance().init("path/to/cardinal.log");
LOG_TRACE("message");
LOG_DEBUG("message");
LOG_INFO("message");
LOG_WARN("message");
LOG_ERROR("Failed to load model: {}", path);
LOG_FATAL("message");
```

### 27.2 ConfigLoader (`src/utils/config_loader.h/cpp`)

Static class. No instances. Load once at startup.

```cpp
auto config = ConfigLoader::load("config.json");      // throws ConfigError on failure
auto config = ConfigLoader::reload("config.json");    // same, for hot-reload
ConfigLoader::validate(config);                        // throws ConfigError if invalid
std::string json = ConfigLoader::to_json_string(config);
```

### 27.3 JsonParser (`src/utils/json_parser.h/cpp`)

Static utility class for all JSON serialization.

```cpp
// Feeling output
FeelingOutput feeling = JsonParser::parse_feeling_output(json_str);
bool valid = JsonParser::validate_feeling_output(feeling, error_msg);
std::string json = JsonParser::serialize_feeling_output(feeling);

// Rules
std::vector<Rule> rules = JsonParser::load_rules(path);
JsonParser::save_rules(path, rules);      // atomic write via .tmp + rename
nlohmann::json j = JsonParser::rule_to_json(rule);
Rule rule = JsonParser::rule_from_json(j);

// Knowledge nodes
std::vector<KnowledgeNode> nodes = JsonParser::load_knowledge(path);
JsonParser::save_knowledge(path, nodes);  // atomic write

// Utilities
std::string id = JsonParser::generate_id();           // hex_timestamp_counter
std::string ts = JsonParser::current_timestamp();      // ISO 8601
bool valid = JsonParser::is_valid_json(str);
std::string pretty = JsonParser::pretty_print(str, 2);
```

**Atomic writes:** `save_rules()` and `save_knowledge()` write to a `.tmp` file then rename atomically. A crash during write leaves the old file intact.

**ID generation:** Uses `std::chrono::system_clock` for the timestamp component and a `std::atomic<uint32_t>` counter for uniqueness within the same millisecond. IDs are hex-encoded.

### 27.4 ILLMBackend (`src/core/llm_engine.h`)

Abstract interface. Implementations: `LlamaCppBackend`, `TensorRTBackend`.

Manages two inference contexts (one for Pass 1, one for Pass 2) to prevent grammar state contamination.

```cpp
ILLMBackend* backend = ...;
backend->load_model();

// Pass 1 (grammar-constrained)
auto result = backend->generate_feeling(ctx, messages);

// Pass 2 (free decoding with streaming)
auto result = backend->generate_response(ctx, messages, token_cb);
```

### 27.5 InferencePipeline (`src/core/inference.h/cpp`)

Orchestrates the two-pass inference cycle.

```cpp
InferencePipeline pipeline(config, backend);
pipeline.set_system_prompt("...");
pipeline.set_retriever(&retriever);

InferenceRequest req;
req.user_message    = "What is entropy?";
req.history         = session.get_history();
req.active_rules    = rule_store.get_top_rules("factual", 5);
req.stream_response = true;
req.max_iterations  = is_agentic ? config.max_iterations : 0;

InferenceResponse resp = pipeline.run(req, stream_callback);
```

If `max_iterations > 0`, the pipeline runs in agentic mode.

### 27.6 EpisodicRetriever (`src/memory/episodic_retriever.h/cpp`)

```cpp
EpisodicRetriever retriever(config, storage);
retriever.init();

auto results = retriever.retrieve("What is entropy?");
auto results = retriever.retrieve("...", RetrievalMode::KEYWORD);

retriever.notify_new_episode(ep_id);
retriever.rebuild_index();
retriever.set_mode(RetrievalMode::HYBRID);
retriever.set_weights(0.7f, 0.3f);
```

### 27.7 EpisodicStorage (`src/memory/episodic_storage.h/cpp`)

```cpp
EpisodicStorage storage(config);
storage.open();

bool inserted = storage.insert_episode(record);
bool updated  = storage.set_extracted_rule_id(ep_id, rule_id);
auto ep       = storage.get_episode(id);
auto results  = storage.query(q);
auto recent   = storage.get_recent(10);
int count     = storage.count();
auto stats    = storage.stats();

storage.close();    // WAL checkpoint + close
```

### 27.8 RuleExtractor (`src/verifier/rule_extractor.h/cpp`)

```cpp
RuleExtractor extractor(config, rule_store, symbolic_engine);

ExtractionInput input;
input.feeling       = resp.feeling;
input.user_message  = req.user_message;
input.response_text = resp.response;
input.episode_id    = ep_id;

ExtractionResult result = extractor.extract(input);
// result.extracted           -- was a candidate found?
// result.committed           -- was it written to rule store?
// result.contradiction_found -- did it contradict existing rules?
// result.committed_rule_id   -- ID if committed
// result.rejection_reason    -- why it was rejected
```

### 27.9 ConsistencyChecker (`src/verifier/consistency_check.h/cpp`)

```cpp
ConsistencyChecker checker(config, rule_store, episodic,
                            symbolic, extractor, neural_verifier);
checker.init();

ConsistencyCheckResult result = checker.check(input);
auto contradictions = checker.run_full_scan();
int removed = checker.run_maintenance();
```

### 27.10 TrainingExporter (`src/learning/training_exporter.h/cpp`)

```cpp
TrainingExporter exporter(config, storage, rule_store);

ExportFilter filter;
filter.min_confidence = 0.7f;
filter.include_rules  = true;

auto stats = exporter.export_to_file("output.jsonl", filter);
auto dry   = exporter.dry_run(filter);
auto examples = exporter.collect(filter);   // in-memory, no file write

int ep_count   = exporter.available_episode_count(0.7f);
int rule_count = exporter.available_rule_count();
```

### 27.11 HttpServer (`src/api/http_server.h/cpp`)

```cpp
HttpServer server(api, config);

std::thread t([&]{ server.start(); });  // start blocks; run in thread

server.stop();   // safe to call from any thread
bool running = server.is_running();
```

`cpp-httplib` handles one request at a time. Read-only endpoints do not hold `inference_mutex_`. Chat endpoints hold `inference_mutex_` for the entire inference. Agentic endpoints hold `agent_mutex_`.

### 27.12 VisionEncoder (`src/vision/vision_encoder.h/cpp`)

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

### 27.13 SelfImprovementLoop (`src/training/self_improvement_loop.h/cpp`)

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

### 27.14 SchedulerEngine (`src/scheduler/scheduler_engine.h/cpp`)

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

### 27.15 SchedulerParser (`src/scheduler/scheduler_parser.h/cpp`)

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

### 27.16 DisplayDetector (`src/computer/display_detector.h/cpp`)

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

### 27.17 BrowserController (`src/computer/browser_controller.h/cpp`)

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

### 27.18 VoiceLoop (`src/voice/voice_loop.h/cpp`)

```cpp
class VoiceLoop {
    VoiceLoop(const VoiceConfig& config, VoiceChatStreamFn chat_fn);

    bool start();
    void stop();
    bool is_running() const;

    void           set_input_mode(VoiceInputMode mode);
    VoiceInputMode input_mode() const;

    TTSResult        speak(const std::string& text);
    TranscriptResult transcribe(const AudioChunk& audio);
    void             stop_speaking();

    VoiceLoopState state()      const;
    VoiceStatus    get_status() const;

    void set_state_callback(VoiceStateCallback cb);
    void set_transcript_callback(TranscriptCallback cb);
};
```

### 27.19 AudioDevice (`src/voice/audio_device.h/cpp`)

```cpp
class AudioDevice {
    explicit AudioDevice(const VoiceAudioConfig& config);
    bool init();
    void shutdown();
    bool is_ready() const;
    bool start_capture(std::function<void(const int16_t*, int)> callback);
    void stop_capture();
    void play(const std::vector<int16_t>& samples, int sample_rate,
              std::function<void()> on_done = nullptr);
    void stop_playback();
    bool is_playing() const;
    void wait_until_done();
    std::vector<DeviceInfo> list_devices() const;
};
```

### 27.20 STTEngine (`src/voice/stt_engine.h/cpp`)

```cpp
class STTEngine {
    explicit STTEngine(const VoiceSTTConfig& config);
    bool init();
    void shutdown();
    bool is_ready() const;
    TranscriptResult transcribe(const AudioChunk& audio);
};
```

### 27.21 TTSEngine (`src/voice/tts_engine.h/cpp`)

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

### 27.22 VADDetector (`src/voice/vad_detector.h/cpp`)

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

### 27.23 WakeWordDetector (`src/voice/wake_word_detector.h/cpp`)

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

### 28.1 Thread Inventory

| Thread | Owner | Responsibility |
|--------|-------|----------------|
| Main / inference | CardinalAPI | `chat()`, `agent()`, all user-facing calls |
| HTTP server | HttpServer | Request handling, SSE streaming |
| Training | SelfImprovementLoop | Layer 3 training cycle (background) |
| Scheduler engine | SchedulerEngine | Trigger evaluation + task dispatch |
| Task execution | SchedulerEngine (detached) | Each task runs in a detached thread |
| Voice loop | VoiceLoop | STT → inference → TTS pipeline |
| Push-to-talk | VoiceLoop | Stdin raw-mode keyboard reader |
| Wake word | WakeWordDetector | PocketSphinx recognition loop |
| Audio callbacks | PortAudio (internal) | Capture and playback callbacks |

### 28.2 Component Thread Safety

| Component | Mechanism |
|-----------|-----------|
| `Logger` | Singleton with internal mutex |
| `RuleStore` | `std::mutex` on all operations |
| `EpisodicStorage` | `std::mutex` + SQLite `FULLMUTEX` |
| `EpisodicRetriever` | `std::shared_mutex` (shared reads, exclusive writes) |
| `SettingsManager` | `std::shared_mutex` |
| `SessionManager` | `std::mutex` on map operations |
| `ConsistencyChecker` | `std::mutex` on `check()` |
| `ToolExecutor` | `std::mutex` (state changes only) |
| `ExplainabilityManager` | `std::mutex` on audit log writes |
| `SelfModel` | Internal mutex (SQLite upserts) |
| `MetaCognition` | `reflect_mutex_` (try_to_lock) + `window_mutex_` |
| `SchedulerEngine` | `cv_mutex_`, `idle_mutex_`, `status_mutex_`, `sim_mutex_` |
| `VADDetector` | `mutex_` (lock released during callbacks) |
| `WakeWordDetector` | `queue_mutex_` |
| `AudioDevice` | `playback_mutex_` |
| `VoiceLoop` | `state_mutex_` |
| `CardinalAPI` | Multiple mutexes (see below) |

### 28.3 CardinalAPI Mutexes

- **`inference_mutex_`** (`std::mutex`) — Serializes all inference. Only one inference runs at a time across all sessions.
- **`agent_mutex_`** (`std::mutex`) — Serializes agentic loops (separate from chat inference to allow concurrent non-agentic requests).
- **`session_mutex_`** (`std::shared_mutex`) — Protects session map.
- **`api_mutex_`** (`std::mutex`) — Protects init/shutdown.
- **`voice_mutex_`** (`std::mutex`) — Guards `voice_loop_` pointer access. All public voice API methods acquire it briefly.

### 28.4 Lock Ordering (no deadlocks)

```
api_mutex_
  > voice_mutex_
  > session_mutex_
  > inference_mutex_
      > self_model.mutex_
      > meta_cognition.reflect_mutex_
      > meta_cognition.window_mutex_
      > training_mutex_ > trainer_mutex_ > pending_mutex_
  > scheduler cv_mutex_
      > scheduler idle_mutex_
      > scheduler status_mutex_
      > scheduler sim_mutex_
  > VoiceLoop state_mutex_
      > VADDetector mutex_
      > WakeWordDetector queue_mutex_
      > AudioDevice playback_mutex_
```

### 28.5 Voice Thread Safety Notes

- `AudioDevice::capture_callback()` runs on PortAudio's audio thread. It must not block. All paths from the callback use their own mutexes and return immediately.
- `VADDetector::push_frame()` acquires its own mutex and calls callbacks with the lock released to avoid deadlock.
- `WakeWordDetector::push_frame()` acquires `queue_mutex_` for a bounded insert and notifies the recognition thread. Non-blocking.
- `VoiceLoop::loop_thread_` blocks on `work_cv_` and processes segments serially. STT and TTS are both called on this thread under their respective internal mutexes.

### 28.6 Training Thread

- Sleeps on `condition_variable`, wakes on trigger or 60-second poll.
- All training operations (JSONL write, subprocess, adapter load) happen on this thread.
- Never blocks the inference thread.
- `on_session_boundary()` (inference thread) acquires `pending_mutex_` briefly to read/clear `pending_adapter_path_`, then calls `backend_.load_adapter()` which acquires `trainer_mutex_` in `LlamaCppTrainer`. Designed to be fast.

### 28.7 HTTP Server Threading

`cpp-httplib` handles one request at a time. Chat endpoints hold `inference_mutex_` for the entire inference. Agentic endpoints hold `agent_mutex_`. Read-only endpoints (`/api/stats`, `/api/rules`, `/api/episodes`, `/api/settings`) do not hold `inference_mutex_` and can be served concurrently with an in-progress inference.

---

## 29. Data Formats

### 29.1 JSONL Episode Format

Each line in `logs/episodic.log`:

```json
{
  "id": "19d62f60677_0000",
  "timestamp": "2026-04-06T18:53:03",
  "user_message": "What is entropy?",
  "response_summary": "<think>\n\n</think>\n\nEntropy is a measure of disorder...",
  "confidence": 0.95,
  "reasoning_type": "deductive",
  "reasoning_domain": "factual",
  "contradiction": false,
  "uncertainty": false,
  "rule_candidate": false,
  "extracted_rule_id": "",
  "pass1_tokens": 49,
  "pass2_tokens": 195,
  "total_ms": 19568
}
```

Note that `response_summary` in the JSONL contains the raw model output including `<think>` blocks. The training exporter strips these blocks before writing to training data.

### 29.2 Rule Store Format

`data/memory/rules.json` is a JSON array:

```json
[
  {
    "id": "19d62f60677_0001",
    "domain": "factual",
    "condition": "gas temperature increases",
    "consequence": "gas molecules move faster and pressure increases",
    "confidence": 0.87,
    "trigger_count": 3,
    "created_at": "2026-04-06T18:53:03",
    "updated_at": "2026-04-07T10:22:41",
    "episode_id": "19d62f60677_0000",
    "reasoning_type": "causal"
  }
]
```

Rules created before the provenance feature was added have empty `episode_id` and `reasoning_type`. These fields are written with empty string defaults when loading and resaving legacy rules.

### 29.3 Knowledge Graph Format

`data/memory/knowledge.json` is a JSON array:

```json
[
  {
    "id": "node_001",
    "label": "Entropy",
    "type": "concept",
    "content": "A measure of disorder or randomness in a thermodynamic system",
    "related_ids": ["node_002", "node_003"],
    "confidence": 0.95,
    "source": "inference",
    "created_at": "2026-04-06T18:53:03",
    "updated_at": "2026-04-06T18:53:03"
  }
]
```

### 29.4 Training Export Format

`data/training_export.jsonl` — Alpaca format, one JSON object per line:

```json
{"instruction": "What happens to gas molecules when temperature increases?", "input": "", "output": "When temperature increases, gas molecules gain kinetic energy and move faster, increasing pressure and collision frequency."}
{"instruction": "What rule applies when: gas temperature increases?", "input": "", "output": "Gas molecules move faster and pressure increases."}
```

### 29.5 SQLite Schema Summary

**`episodes.db`** — See §4.4 for the full schema. `metadata` table keys:

| Key | Value | Description |
|-----|-------|-------------|
| `migration_v1_complete` | `"1"` | Set after JSONL migration completes |
| `migration_v1_count` | `"41"` | Number of episodes migrated |

**`self_model.db`** — Two tables: `domain_stats`, `reasoning_stats`. See §4.5.

**`scheduler.db`** — Three tables: `scheduled_tasks`, `task_runs`, `task_action_logs`. See §4.6.

**`audit.db`** — One table: `traces`. See §17.2.

---

## 30. Lifecycle and Startup Sequence

### 30.1 Full Startup Sequence

```
main(argc, argv)
  |
  +-- Parse --voice / --voice=MODE flag
  +-- Logger::instance().init()
  +-- CardinalAPI::init("config.json")
        |
        +-- ConfigLoader::load()  -- validates all fields
        |
        +-- ILLMBackend (LlamaCppBackend or TensorRTBackend)
        |     +-- load_model()
        |     +-- create_context() × 2 (pass1, pass2)
        |
        +-- Memory subsystems
        |     +-- RuleStore::load()         -- reads rules.json
        |     +-- KnowledgeGraph::load()    -- reads knowledge.json
        |     +-- EpisodicMemory::open()    -- opens JSONL for appending
        |     +-- EpisodicStorage::open()   -- opens SQLite, runs JSONL migration if needed
        |     +-- EpisodicRetriever::init() -- builds TF-IDF index from all episodes
        |
        +-- Verifier pipeline
        |     +-- SymbolicEngine::init(kb_path)  -- loads cardinal_kb.pl
        |     +-- RuleExtractor::init()
        |     +-- NeuralVerifier::load()         -- no-op if neural_model_path empty
        |     +-- ConsistencyChecker::init()     -- sync_rules_to_prolog()
        |
        +-- InferencePipeline::init()
        |
        +-- Tools & Agent
        |     +-- ToolRegistry::register_all()   -- registers all base tools
        |     +-- ToolExecutor::init()
        |     +-- AgentExecutor::init()
        |
        +-- Vision
        |     +-- VisionCache::init()
        |     +-- VisionEncoder::load()    -- ready or disabled
        |
        +-- Explainability
        |     +-- AuditLog::open()
        |     +-- ExplainabilityExporter::init()
        |     +-- key generation if auto_generate_keys=true
        |
        +-- API layer
        |     +-- TrainingExporter::init()
        |     +-- SettingsManager::init()
        |     +-- SessionManager::init()   -- creates initial default session
        |
        +-- Self-Improvement
        |     +-- SelfImprovementLoop::start()
        |           +-- SelfModel::open()      -- creates self_model.db if needed
        |           +-- MetaCognition::init()
        |           +-- TrainingFactory::create() -- LlamaCppTrainer or TensorRTTrainer
        |           +-- CurriculumBuilder::init()
        |           +-- DatasetCurator::init()
        |           +-- AdapterEvaluator::init()
        |           +-- training_thread_.start()
        |
        +-- Scheduler
        |     +-- SchedulerEngine constructed with SchedulerDeps
        |     +-- SchedulerEngine::start()
        |           +-- SchedulerStore::open()  -- creates scheduler.db if needed
        |           +-- Fires STARTUP tasks immediately
        |           +-- engine_thread_.start()
        |     +-- ToolRegistry registers schedule_task tool
        |
        +-- Computer Use
        |     +-- DisplayDetector::detect()  -- X11 / Wayland / headless
        |     +-- ScreenReader constructed with DisplayDetector + VisionEncoder
        |     +-- InputController, AppController
        |     +-- BrowserController (lazy start — first use spawns helper process)
        |     +-- ShellExecutor, FileManager, SystemController
        |     +-- EmailController, AtSpiReader
        |     +-- ToolRegistry registers screenshot, click, type_text, ...
        |     +-- ToolExecutor wired via set_screen_reader(), set_scheduler(), ...
        |
        +-- Voice (if voice.enabled in config)
        |     +-- VoiceLoop constructed with VoiceChatStreamFn lambda
        |     +-- VoiceLoop::start()
        |           +-- AudioDevice::init()         -- Pa_Initialize()
        |           +-- VADDetector constructed
        |           +-- STTEngine::init()            -- whisper_init_from_file_with_params()
        |           +-- TTSEngine::init()            -- piper::initialize() + loadVoice()
        |           +-- WakeWordDetector::init()     -- if wake_word mode
        |           +-- loop_thread_.start()
        |           +-- ptt_thread_.start()          -- if push_to_talk mode
        |     +-- ToolRegistry registers voice_control
        |     +-- ToolExecutor::set_voice_loop(voice_loop_.get())
        |
        +-- initialized_.store(true)
        +-- LOG_INFO("CardinalAPI initialized — N tools registered")

  +-- if --voice flag and !voice.enabled: enable_voice(mode_override)
  +-- HttpServer::start() (if http_enabled)
  +-- interactive loop
```

### 30.2 Per-Inference Sequence (Chat Mode)

```
chat_stream(session_id, message, stream_cb)
  |
  +-- check_initialized()
  +-- check message not empty
  +-- ensure session exists (create if not)
  +-- acquire inference_mutex_
  +-- get session history
  +-- build InferenceRequest
  +-- pipeline.run(req, core_cb)
        |
        +-- build_messages()
        |     +-- retriever.retrieve(user_message)  -- if retriever set
        |     +-- format_episodes()                  -- inject memory context
        |     +-- format_rules()                     -- inject active rules
        |     +-- inject [Self-Model] block           -- if inject_into_prompt=true
        |
        +-- run_pass1() with retry
        |     +-- backend.generate_feeling()
        |     +-- JsonParser::parse_feeling_output()
        |     +-- validate_feeling_output()
        |
        +-- run_pass2()
              +-- backend.generate_response()        -- calls stream_cb per token
  |
  +-- run_post_inference()
        |
        +-- episodic.log_episode()                  -- JSONL write
        +-- storage.insert_episode()                -- SQLite write
        +-- retriever.notify_new_episode()          -- rebuild check
        +-- checker.check()
              |
              +-- handle_rule_extraction()           -- if rule_candidate_signal
              +-- handle_contradiction_check()       -- if contradiction_flag
              +-- resolve_contradiction()            -- per contradiction found
              +-- run_periodic_maintenance()         -- every 10 inferences
        |
        +-- if explainability enabled: build, sign, save trace
        +-- storage.set_extracted_rule_id()         -- if rule committed
        +-- rule_store.save()                       -- if dirty
        +-- self_improvement.on_inference()         -- Layer 1/2/3 hooks
        +-- scheduler.on_inference()                -- idle tracker update
  |
  +-- release inference_mutex_
  +-- update session history
  +-- return ChatResponse
```

### 30.3 Per-Inference Sequence (Agentic Mode)

```
agent(session_id, goal, max_iterations)
  |
  +-- acquire agent_mutex_
  +-- ensure session exists
  +-- AgentExecutor::run(goal, trace_builder, progress_cb)
        |
        +-- PLAN: decompose goal, feeling check, symbolic check
        +-- EXECUTE LOOP (i = 0..max_iterations)
        |     |
        |     +-- THINK: InferencePipeline::run() (single step, no agent loop inside)
        |     +-- ACT: ToolExecutor::execute()
        |     +-- OBSERVE: inject results, update understanding
        +-- FINALIZE: generate final response
  +-- if explainability enabled: trace saved to audit log
  +-- release agent_mutex_
```

### 30.4 Shutdown Sequence

```
main: api.disable_voice()       -- if --voice flag was used
main: http_server.stop()

CardinalAPI::shutdown()
  |
  +-- acquire api_mutex_
  +-- voice_loop_->stop()       -- first, frees audio hardware
        +-- stop_requested_.store(true)
        +-- work_cv_.notify_all()
        +-- AudioDevice::stop_playback(), stop_capture()
        +-- WakeWordDetector::stop()
        +-- loop_thread_.join()
        +-- ptt_thread_.join()
        +-- STTEngine::shutdown()     -- whisper_free()
        +-- TTSEngine::shutdown()     -- piper::terminate()
        +-- AudioDevice::shutdown()   -- Pa_Terminate()
  |
  +-- scheduler_->stop()
        +-- notify CV with stop_requested_=true
        +-- engine_thread_.join()
        +-- SchedulerStore::close()
  |
  +-- browser_controller_->stop()   -- shuts down Playwright helper process
  |
  +-- self_improvement_->stop()
        +-- training_thread_.join()
        +-- SelfModel::close()       -- WAL checkpoint
  |
  +-- checker.run_maintenance()     -- final decay/prune/save
  +-- sessions->destroy_all()       -- clear all session state
  +-- EpisodicStorage::close()      -- WAL checkpoint + close SQLite
  +-- AuditLog::close()
  +-- rule_store.save()             -- save any unsaved rules
  +-- ILLMBackend::unload()
  +-- VisionEncoder::unload()       -- mtmd_free
  +-- initialized = false
```

**Destruction order** is guaranteed by `unique_ptr` member declaration order in `CardinalAPI`. The training thread is always joined before any subsystem it depends on (storage, rule_store, backend) is shut down. `voice_loop_` is stopped explicitly first to release audio hardware early.

---

## 31. Extending Cardinal

### 31.1 Adding a New Reasoning Domain

1. Add the domain string to the GBNF grammar's enum constraint in `feeling_schema.gbnf`
2. Add the domain to the `valid_domains` vector in `JsonParser::parse_feeling_output()`
3. Add the domain to the `validate()` function in `ConfigLoader` (if you add a config field for it)
4. Add the domain to `RuleStoreStats.rules_by_domain` array and the domain names array in `RuleStore::stats()`

### 31.2 Adding a New Reasoning Type

1. Add the type string to the GBNF grammar's enum constraint
2. Add it to the `valid_reasoning_types` vector in `JsonParser::parse_feeling_output()`
3. Add extraction handling in `RuleExtractor::extract_candidate()` if the new type warrants a specialized extraction strategy

### 31.3 Adding a New API Endpoint

1. Add a handler declaration in `http_server.h`
2. Implement the handler in `http_server.cpp`
3. Register the route in `register_routes()`
4. Add the corresponding method to `CardinalAPI` if new functionality is needed

### 31.4 Adding a New Setting

1. Add the field to `CardinalSettings` in `cardinal_settings.h`
2. Add initialization from config in `SettingsManager` constructor
3. Add the field to `validate()` in `cardinal_settings.cpp`
4. Add the field to `set()` (single-key update) in `cardinal_settings.cpp`
5. Add the field to `to_json()` and `from_json()` in `cardinal_settings.cpp`
6. Add propagation logic in `propagate()` in `cardinal_settings.cpp`
7. Add the field to `settings_to_json()` in `http_server.cpp`

### 31.5 Adding a Python Binding

All types in `cardinal_types.h` are pybind11-compatible by design. A minimal pybind11 module:

```cpp
// src/api/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "api/cardinal_api.h"

namespace py = pybind11;

PYBIND11_MODULE(cardinal, m) {
    py::class_<CardinalAPI>(m, "CardinalAPI")
        .def(py::init<>())
        .def("init",     &CardinalAPI::init)
        .def("shutdown", &CardinalAPI::shutdown)
        .def("chat",     &CardinalAPI::chat)
        .def("get_stats", &CardinalAPI::get_stats)
        .def("get_rules", &CardinalAPI::get_rules);

    py::class_<ChatResponse>(m, "ChatResponse")
        .def_readonly("response",     &ChatResponse::response)
        .def_readonly("episode_id",   &ChatResponse::episode_id)
        .def_readonly("feeling",      &ChatResponse::feeling)
        .def_readonly("total_ms",     &ChatResponse::total_ms);

    // ... other types
}
```

`ApiStreamCallback` wraps `std::function<bool(const StreamToken&)>` which pybind11 can wrap directly as a Python callable.

### 31.6 Interface 3 (SEAL Python) Overview

Layer 4 Interface 3 (SEAL) uses pybind11 to expose `CardinalAPI` directly to Python. The design principle is that pybind11 bindings are thin wrappers only — no logic lives in the Python layer. All types in `cardinal_types.h` are pybind11-friendly by design.

---

*This documentation reflects Cardinal v2.0.0. The source of truth for any discrepancy is always the source code.*
