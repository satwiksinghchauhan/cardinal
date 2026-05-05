# Cardinal v1.2.0 — Technical Documentation

**Architecture:** Neurosymbolic AGI Core + Agentic Loop + Explainability  
**Language:** C++20  
**Platform:** Linux (Ubuntu 24.04 LTS)  
**GPU:** NVIDIA CUDA (TensorRT / llama.cpp)

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
9. [Agentic Pipeline (Unified)](#9-agentic-pipeline-unified)
10. [Tools System](#10-tools-system)
11. [Explainability Exports](#11-explainability-exports)
12. [Training Export](#12-training-export)
13. [Configuration Reference](#13-configuration-reference)
14. [CardinalAPI Reference](#14-cardinalapi-reference)
15. [HTTP API Reference](#15-http-api-reference)
16. [Settings Manager](#16-settings-manager)
17. [Session Manager](#17-session-manager)
18. [Type Reference](#18-type-reference)
19. [Error Handling](#19-error-handling)
20. [Offline Builds & Vendoring](#20-offline-builds--vendoring)
21. [Module Reference](#21-module-reference)
22. [Threading Model](#22-threading-model)
23. [Lifecycle and Startup Sequence](#23-lifecycle-and-startup-sequence)

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
    SettingsManager    runtime-mutable config
    SessionManager     multi-session conversation state
    TrainingExporter   Alpaca JSONL export
    Explainability     audit log, cryptographic signing, export API

Layer 2 -- Core Systems
    InferencePipeline  two-pass orchestrator, prompt injection
    AgentExecutor      PLAN → EXECUTE loop (THINK → ACT → OBSERVE) → FINALIZE
    ToolExecutor       sandboxed tool execution (subprocess or Docker)
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
```

### Component Ownership

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary — callers see only the types defined in `cardinal_types.h`.

### Dependency Graph (v1.2.0)

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

### Retry Logic

If Pass 1 fails to produce a valid feeling output (the JSON does not parse, or fails schema validation), the pipeline retries up to `max_retries` times with a `retry_delay_ms` delay between attempts. If all retries are exhausted, the inference fails and `ChatResponse` returns an error.

Pass 2 does not retry — if the response generation fails, the inference fails.

### State Machine

The `FeelingContext` object tracks the state of a single inference cycle:

```
IDLE -> PASS1_FEELING -> PASS2_RESPONSE -> COMPLETE
                      -> FAILED (on retry exhaustion)
```

### Metrics

Every `ChatResponse` carries inference metrics:
- `pass1_tokens` — tokens produced in Pass 1
- `pass2_tokens` — tokens produced in Pass 2
- `total_ms` — wall clock time for the full inference cycle

---

## 3. Feeling Output Schema

The feeling output is the core introspective signal that drives everything downstream. It is produced by Pass 1 and consumed by the verifier pipeline, the retriever, the session history, and the training exporter.

### Fields

**`confidence`** (float, 0.0 to 1.0)

How confident the model is in its response. This is not a post-hoc annotation — the model produces this value before generating its response. Values below 0.4 are considered low confidence. Values above 0.7 are considered high confidence.

The consistency checker uses high-confidence episodes as candidates for rule extraction. The training exporter uses `min_confidence` to filter training examples.

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

**`uncertainty_flag`** (boolean)

True when the model is uncertain about its response. Logically inconsistent with high confidence — the validator will reject a feeling output with `confidence > 0.8` and `uncertainty_flag = true`.

**`contradiction_flag`** (boolean)

True when the model has detected a conflict in its own reasoning or between the current query and prior knowledge. When this flag is true, the consistency checker runs a targeted contradiction scan against the rule base for the current domain.

**`rule_candidate_signal`** (boolean)

True when the model believes its response contains a general rule that could be extracted and stored. When this flag is true, the rule extractor attempts to derive a Rule from the response text.

### Validation

The `JsonParser::validate_feeling_output()` function enforces logical consistency:
- `confidence > 0.8` with `uncertainty_flag = true` is rejected as contradictory
- `confidence < 0.3` without `uncertainty_flag = true` produces a warning (not a rejection)

### GBNF Grammar

The grammar file at `src/prompts/feeling_schema.gbnf` enforces the exact JSON structure. If the grammar file is missing or malformed, Cardinal will fail to start with a `ConfigError`.

---

## 4. Memory Systems

Cardinal uses four distinct memory components. Each serves a different purpose and has different access characteristics.

### 4.1 RuleStore

The rule store is the symbolic memory of Cardinal. It contains general rules derived from past inference cycles.

**Storage:** `data/memory/rules.json` (JSON array, atomic writes)

**Key operations:**
- `add_rule(domain, condition, consequence, confidence, episode_id, reasoning_type)` — add a new rule. If a semantically similar rule exists (Jaccard similarity >= 0.7), the existing rule is merged instead of creating a duplicate.
- `query(RuleQuery)` — fetch rules matching domain, condition hint, confidence threshold
- `decay_confidence()` — reduce all rule confidences by `verifier.rule_confidence_decay`
- `prune()` — remove rules below `verifier.min_rule_confidence`

**Confidence lifecycle:**
```
add_rule()          -- initial confidence (0.5-0.65 depending on extraction strategy)
record_trigger()    -- +0.01 per trigger (when rule is relevant to an inference)
decay_confidence()  -- -0.01 per maintenance cycle (default every 10 inferences)
prune()             -- removed if below min_rule_confidence (default 0.3)
```

**Thread safety:** All operations protected by `std::mutex`.

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

**Episode ID format:** `<hex_timestamp_ms>_<4digit_counter>` — for example `19d62f60677_0000`. This format is monotonically increasing and sortable.

**Why JSONL:** Append-only formats are crash-safe. If Cardinal crashes mid-write, the worst case is one malformed line at the end of the file. All prior lines are intact.

### 4.4 EpisodicStorage

EpisodicStorage is the searchable SQLite index over the episode corpus.

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
    user_message,
    response_summary,
    content='episodes'
);
```

**JSONL Migration:** On first `open()`, EpisodicStorage reads the JSONL file and imports all episodes into SQLite. This migration is idempotent — it runs once and never again.

**Dual-write pattern:** After every inference, the episode is written to both EpisodicMemory (JSONL) and EpisodicStorage (SQLite).

---

## 5. Retrieval System

The retrieval system finds past episodes relevant to the current user message and injects them into the inference prompt. This gives Cardinal context-aware memory.

### 5.1 Architecture

```
EpisodicRetriever
    |
    +-- keyword_search()    --> EpisodicStorage FTS5
    +-- semantic_search()   --> TF-IDF cosine similarity (in-memory index)
    +-- merge_results()     --> weighted combination + deduplication
```

### 5.2 Retrieval Modes

**KEYWORD mode:** Uses SQLite FTS5 full-text search. Results ranked by BM25 scoring. Lowest latency.

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

**Step 1 — Rule extraction (if `rule_candidate_signal` true):** The rule extractor attempts to derive a rule from the response text. See Section 7 for the full extraction pipeline.

**Step 2 — Contradiction check (if `contradiction_flag` true OR a rule was committed):** The symbolic engine checks for contradictions with existing rules. For each contradiction found, auto-resolution runs (see Section 7.4).

**Step 3 — Periodic maintenance (every 10 inferences):** Confidence decay, prune, save.

**Step 4 — Build summary:** A human-readable summary is returned in `ConsistencyCheckResult.summary`.

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

Disabled if `neural_model_path` is empty or file does not exist.

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

**Strategy 1 — Causal patterns (applied when `reasoning_type` is `causal` or `deductive`):**

Searches for causal connectives: `if...then`, `because`, `causes`, `leads to`, `results in`, `implies`, `therefore`, `thus`. Initial confidence: 0.6

**Strategy 2 — Deductive patterns (applied when `reasoning_type` is `deductive` or `inductive`):**

Looks for conclusion markers: `therefore`, `thus`, `hence`, `consequently`, `it follows that`, `this means`. Initial confidence: 0.65

**Strategy 3 — Declarative fallback (any reasoning type):**

Uses user message as condition and the longest non-question sentence from response as consequence. Initial confidence: 0.5

### 7.3 Candidate Validation

A candidate must pass:
- Condition and consequence non-empty, length >= 5 characters
- Condition != consequence
- `confidence >= min_rule_confidence`

### 7.4 Contradiction Auto-Resolution

```
delta = abs(confidence_a - confidence_b)

if delta >= 0.2:
    deprecate(lower_confidence_rule)  -- set confidence to 0.0
    return RESOLVED
else:
    penalize(rule_a, -0.05)
    penalize(rule_b, -0.05)
    return FLAGGED
```

Deprecated rules remain in the store until the next maintenance cycle's `prune()` call.

### 7.5 Rule Injection into Prompts

Rules are formatted as a `## Active Rules` section in the system prompt:

```
## Active Rules
1. [factual] IF gas temperature increases THEN molecules move faster (confidence: 87%)
2. [factual] IF pressure is constant THEN volume increases with temperature (confidence: 72%)
```

---

## 8. Backend Abstraction

**New in v1.1.0.** Cardinal now supports multiple inference backends via an abstraction layer. Only one backend is loaded at runtime — you choose which one to compile.

### 8.1 Supported Backends

| Backend | Purpose | When to Use |
|---------|---------|-------------|
| **llama.cpp** | Development, fallback, CPU/CUDA | Default; works everywhere |
| **TensorRT** | Optimized deployment | Production, fixed models, maximum performance |

### 8.2 Architecture

```cpp
class LLMEngine {
    // Abstract interface
    virtual Result<FeelingOutput> generate_feeling(...) = 0;
    virtual Result<std::string> generate_response(...) = 0;
};

class LlamaCppEngine : public LLMEngine { ... };
class TensorRTEngine : public LLMEngine { ... };
```

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

## 9. Agentic Pipeline (Unified)

**New in v1.2.0.** Cardinal uses a **unified pipeline** for both chat and agentic modes. The only difference is the `max_iterations` setting. One code path, one audit trail format.

### 9.1 Agent Executor Loop

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

### 9.2 Tool Execution Loop (Integrated)

The same two-pass inference is used, but Pass 2 now loops:

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

### 9.3 Working Memory

Persistent SQLite-backed scratchpad for multi-step tasks. Stored at `data/memory/agent_working_memory`. Enables task resumption after interruption.

### 9.4 Self-Correction

When a tool call fails, the agent can:
1. Analyse the error
2. Adjust the plan
3. Retry the step (up to `self_correction_max_attempts`)

### 9.5 Configuration

```json
"agent": {
  "enabled": true,
  "max_iterations": 10,
  "max_iterations_hard_cap": 50,
  "working_memory_path": "data/memory/agent_working_memory",
  "working_memory_size": 50,
  "self_correction_enabled": true,
  "self_correction_max_attempts": 3,
  "plan_before_execute": true,
  "summarize_on_cap": true
}
```

---

## 10. Tools System

**New in v1.2.0.** Tools are configurable, sandboxed, and support confirmation prompts.

### 10.1 Built-in Tools

| Tool | Description | Confirmation Required | Sandbox |
|------|-------------|----------------------|---------|
| `web_search` | DuckDuckGo search, no API key | Configurable | None |
| `web_fetch` | Fetch and parse a URL | Configurable | None |
| `calculator` | Safe math expression evaluator | No | None |
| `run_python` | Python code execution | Yes (default) | Subprocess or Docker |
| `file_read` | Read from allowed paths | Yes (default) | Path restrictions |
| `file_write` | Write to allowed paths | Yes (default) | Path restrictions |
| `knowledge_graph_query` | Query Cardinal's own KG | No | None |
| `episodic_search` | Search Cardinal's memory | No | None |

### 10.2 Python Sandbox Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `subprocess` | Spawn subprocess with resource limits (ulimit) | Development, quick testing |
| `docker` | Full container isolation | Production, DRDO/ISRO deployment |

**Sandbox limits (both modes):**
- `timeout_seconds` — default 30s
- `memory_limit_mb` — default 256MB
- `network_enabled` — false by default

### 10.3 Configuration

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

### 10.4 User-Defined Tools (via API)

Callers can register custom tools via the HTTP API per request, following Claude's tools parameter format. Each tool has:
- `name` — unique identifier
- `description` — what the tool does (used by LLM for tool selection)
- `parameters` — JSON schema defining expected arguments

---

## 11. Explainability Exports

**New in v1.2.0.** Every inference produces a signed, tamper-evident trace.

### 11.1 Export Schema

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
        "input": {"query": "DRDO latest news"},
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

### 11.2 Audit Log Storage

SQLite database at `data/explainability/audit.db` with schema:

```sql
CREATE TABLE traces (
    id            TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    inference_id  TEXT NOT NULL UNIQUE,
    session_id    TEXT NOT NULL,
    user_message  TEXT NOT NULL,
    trace_json    TEXT NOT NULL,  -- full trace as JSON
    hash          TEXT NOT NULL,
    signature     TEXT NOT NULL
);

CREATE INDEX idx_timestamp ON traces(timestamp);
CREATE INDEX idx_session ON traces(session_id);
```

### 11.3 Cryptographic Signing

- **Hash:** SHA256 of the trace JSON
- **Signature:** Ed25519 (private key in `data/explainability/cardinal_private.pem`)
- **Auto-generation:** Keys created on first start if `auto_generate_keys: true`

### 11.4 Export API

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

Response:
```json
{
  "export_id": "exp_123",
  "trace_count": 42,
  "download_url": "/api/explainability/download/exp_123.json"
}
```

### 11.5 Configuration

```json
"explainability": {
  "enabled": true,
  "audit_log_path": "data/explainability/audit.db",
  "signing_enabled": true,
  "private_key_path": "data/explainability/cardinal_private.pem",
  "public_key_path": "data/explainability/cardinal_public.pem",
  "auto_generate_keys": true,
  "export_path": "data/explainability/exports",
  "attach_trace_to_response": true
}
```

---

## 12. Training Export

The training exporter produces Alpaca-format JSONL files suitable for LoRA fine-tuning.

### 12.1 Output Format

```json
{"instruction": "What happens to gas molecules when temperature increases?", "input": "", "output": "When temperature increases, gas molecules gain kinetic energy and move faster..."}
```

### 12.2 Response Cleaning

Before export:
- `<think>...</think>` blocks stripped
- `<feeling_state>...</feeling_state>` blocks stripped
- Consecutive newlines collapsed
- Leading/trailing whitespace trimmed

### 12.3 Rule Export (optional)

When `include_rules = true`, rules are exported as:

```json
{"instruction": "What rule applies when: gas temperature increases?", "input": "", "output": "Gas molecules move faster and pressure increases."}
```

### 12.4 Filtering

| Field | Default | Description |
|-------|---------|-------------|
| `min_confidence` | 0.7 | Minimum episode confidence |
| `domain` | "" (all) | Filter by reasoning domain |
| `max_examples` | 0 (no limit) | Maximum examples to export |
| `include_rules` | false | Also export rules |

---

## 13. Configuration Reference

All configuration lives in `config.json`. Every field is validated at startup.

### 13.1 `model` section

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Path to the primary GGUF model file |
| `context_length` | int | Maximum context window in tokens (>= 512) |
| `gpu_layers` | int | Number of layers to offload to GPU (0 = CPU only) |
| `threads` | int | CPU threads for non-GPU work (>= 1) |

### 13.2 `inference` section

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | float | Sampling temperature (0.0 to 2.0) |
| `top_p` | float | Nucleus sampling (0.0 to 1.0) |
| `max_tokens_feeling` | int | Max tokens for Pass 1 (>= 32) |
| `max_tokens_response` | int | Max tokens for Pass 2 (>= 64) |
| `max_retries` | int | Maximum Pass 1 retry attempts (>= 0) |
| `retry_delay_ms` | int | Milliseconds between retries (>= 0) |

### 13.3 `feeling_schema` section

| Field | Type | Description |
|-------|------|-------------|
| `grammar_path` | string | Path to the GBNF grammar file |

### 13.4 `memory` section

| Field | Type | Description |
|-------|------|-------------|
| `rule_store_path` | string | Path to `rules.json` |
| `knowledge_graph_path` | string | Path to `knowledge.json` |
| `episodic_log_path` | string | Path to the JSONL episode log |
| `max_rules` | int | Maximum number of rules in the store (>= 1) |

### 13.5 `verifier` section

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `symbolic`, `neural`, or `hybrid` |
| `neural_model_path` | string | Path to neural verifier GGUF (empty = disabled) |
| `neural_gpu_layers` | int | GPU layers for neural verifier (0 = CPU only) |
| `contradiction_threshold` | float | Score above which a contradiction is flagged (0.0 to 1.0) |
| `rule_confidence_decay` | float | Confidence reduction per maintenance cycle (>= 0) |
| `min_rule_confidence` | float | Rules below this are pruned (0.0 to 1.0) |

### 13.6 `retriever` section

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `keyword`, `semantic`, or `hybrid` |
| `keyword_weight` | float | Weight for keyword score in hybrid (0.0 to 1.0) |
| `semantic_weight` | float | Weight for semantic score in hybrid (0.0 to 1.0) |
| `max_results` | int | Maximum episodes returned per retrieval (>= 1) |
| `min_score` | float | Minimum combined score to include in results (0.0 to 1.0) |
| `cache_rebuild_strategy` | string | `on_demand`, `periodic`, or `explicit` |
| `cache_rebuild_threshold` | int | Episodes added before on_demand rebuild (>= 1) |
| `cache_rebuild_interval_seconds` | int | Seconds between periodic rebuilds (>= 1) |

### 13.7 `tools` section

See Section 10.3 for full schema.

### 13.8 `agent` section

See Section 9.5 for full schema.

### 13.9 `explainability` section

See Section 11.5 for full schema.

### 13.10 `api` section

| Field | Type | Description |
|-------|------|-------------|
| `http_enabled` | bool | Enable HTTP server on startup |
| `host` | string | Bind address (`127.0.0.1` for localhost) |
| `port` | int | Listen port (1-65535) |
| `auth_enabled` | bool | Require Bearer token on all requests (except `/api/health`) |
| `api_key` | string | The Bearer token (required if `auth_enabled` true) |
| `stream_enabled` | bool | Allow SSE streaming responses |

### 13.11 `logging` section

| Field | Type | Description |
|-------|------|-------------|
| `level` | string | Minimum log level: `trace`, `debug`, `info`, `warn`, `error`, `fatal` |
| `path` | string | Path to the log file |

---

## 14. CardinalAPI Reference

`CardinalAPI` is the single entry point for all interfaces.

**File:** `src/api/cardinal_api.h`

### 14.1 Lifecycle

```cpp
CardinalAPI api;
CardinalVoidResult init(const std::string& config_path = "config.json");
CardinalVoidResult shutdown();
bool is_initialized() const;
```

### 14.2 Session Management

```cpp
CardinalResult<std::string> create_session(const std::string& session_id = "");
CardinalVoidResult destroy_session(const std::string& session_id);
CardinalVoidResult reset_session(const std::string& session_id);
CardinalResult<SessionInfo> get_session(const std::string& session_id) const;
CardinalResult<std::vector<std::string>> list_sessions() const;
```

Sessions are created automatically on first `chat()` call if not exist.

### 14.3 Inference

```cpp
CardinalResult<ChatResponse> chat(
    const std::string& session_id,
    const std::string& message);

CardinalResult<ChatResponse> chat_stream(
    const std::string&       session_id,
    const std::string&       message,
    const ApiStreamCallback& stream_cb);
```

### 14.4 Agentic Inference (v1.2.0)

```cpp
CardinalResult<ChatResponse> chat_agentic(
    const std::string& session_id,
    const std::string& goal,
    int max_iterations = 10);
```

The `max_iterations` parameter overrides the config setting for this request only.

### 14.5 Memory & Stats

```cpp
CardinalResult<SystemStats> get_stats() const;
CardinalResult<std::vector<RuleInfo>> get_rules() const;
CardinalResult<std::vector<EpisodeInfo>> get_episodes(
    const std::string& keyword = "",
    const std::string& domain = "",
    float min_conf = 0.0f,
    int max_results = 20) const;
CardinalResult<ScanResult> run_scan();
CardinalVoidResult run_maintenance();
```

### 14.6 Training Export

```cpp
CardinalResult<ExportInfo> export_training_data(const ExportRequest& request);
CardinalResult<ExportInfo> export_dry_run(const ExportRequest& request) const;
```

### 14.7 Explainability (v1.2.0)

```cpp
CardinalResult<ExportInfo> export_explainability_traces(
    const std::string& session_id = "",
    const std::string& start_time = "",
    const std::string& end_time = "");
```

### 14.8 Settings

```cpp
CardinalResult<CardinalSettings> get_settings() const;
CardinalVoidResult update_settings(const CardinalSettings& settings);
CardinalVoidResult set_setting(const std::string& key, const std::string& value);
CardinalVoidResult reset_settings();
```

---

## 15. HTTP API Reference

Base URL: `http://127.0.0.1:8080` (configurable)

Authentication: `Authorization: Bearer <api_key>` (except `/api/health`)

### 15.1 GET `/api/health`

Always public. Returns 200 if server is running.

**Response:**
```json
{"status": "ok", "uptime": "00h 12m 34s", "version": "1.2.0"}
```

### 15.2 POST `/api/chat`

**Request (non-agentic):**
```json
{
  "session_id": "my-session",
  "message": "What is entropy?",
  "stream": false
}
```

**Request (agentic, v1.2.0):**
```json
{
  "session_id": "agent-session",
  "message": "Search the web for DRDO news and save to ~/Downloads/drdo_news.txt",
  "max_iterations": 5,
  "stream": true
}
```

**Non-streaming response:** See `ChatResponse` type (includes `reasoning_trace` if explainability is enabled).

**Streaming:** Set `"stream": true` or `Accept: text/event-stream`. Each token delivered as SSE event. Final event contains full `reasoning_trace` and `integrity` fields.

### 15.3 POST `/api/chat/agentic` (v1.2.0, convenience)

Same as `/api/chat` with `max_iterations` implicitly set (uses config default if not provided).

### 15.4 POST `/api/sessions`

**Request:** `{"session_id": "agent-session-1"}` (optional)

**Response:** `{"status": "ok", "session_id": "agent-session-1"}`

### 15.5 DELETE `/api/sessions/:id`

**Response:** `{"status": "ok", "session_id": "agent-session-1", "message": "Session destroyed"}`

### 15.6 GET `/api/stats`

Returns complete system statistics (memory, verifier, uptime).

### 15.7 GET `/api/rules`

Returns all rules in the rule store.

### 15.8 GET `/api/episodes`

Query parameters: `keyword`, `domain`, `min_conf`, `max_results`.

### 15.9 POST `/api/scan`

Run full contradiction scan. **Response:** `{"total_contradictions": 2, "resolved": 1, "flagged": 1, "skipped": 0}`

### 15.10 POST `/api/maintenance`

Manually trigger maintenance cycle.

### 15.11 GET `/api/settings` / POST `/api/settings`

Get or update runtime settings. Partial JSON accepted.

### 15.12 POST `/api/export`

Export training data.

**Request:** `{"min_confidence": 0.7, "include_rules": true, "dry_run": false}`

### 15.13 POST `/api/explainability/export` (v1.2.0)

Export signed traces.

**Request:**
```json
{
  "session_id": "my-session",
  "start_time": "2026-05-01T00:00:00Z",
  "end_time": "2026-05-05T23:59:59Z"
}
```

**Response:**
```json
{
  "export_id": "exp_123",
  "trace_count": 42,
  "download_url": "/api/explainability/download/exp_123.json"
}
```

### 15.14 GET `/api/explainability/download/:export_id` (v1.2.0)

Download a previously generated export.

---

## 16. Settings Manager

`SettingsManager` manages runtime-mutable settings. Changes propagate immediately.

### 16.1 Mutable Settings

| Setting | Type | Propagates to |
|---------|------|---------------|
| `retriever_mode` | string | `EpisodicRetriever::set_mode()` |
| `keyword_weight` | float | `EpisodicRetriever::set_weights()` |
| `semantic_weight` | float | `EpisodicRetriever::set_weights()` |
| `max_retrieval_results` | int | Next retrieval call |
| `min_retrieval_score` | float | Next retrieval call |
| `verifier_mode` | string | Validated, stored |
| `min_rule_confidence` | float | Validated, stored |
| `contradiction_threshold` | float | Validated, stored |
| `temperature` | float | Validated, stored |
| `top_p` | float | Validated, stored |
| `log_level` | string | Logger level |
| `agent_max_iterations` | int | Next agentic call (default for new sessions) |
| `agent_self_correction_enabled` | bool | Agent executor |

### 16.2 Validation

All settings validated before changes. Updates are atomic.

---

## 17. Session Manager

`SessionManager` owns all active `ConversationSession` objects.

### 17.1 ConversationSession

Each session tracks:
- `session_id`
- `turn_count` (number of user/assistant pairs)
- `history` (`std::vector<ChatMessage>`)
- `working_memory` (SQLite-backed, for agentic sessions)
- `timestamps` (ISO 8601 for each history entry)
- `created_at`, `last_active_at`

### 17.2 Context Window Management

`estimated_token_count()` provides rough estimate (1 token per 4 characters).

`trim_to_token_budget(max_tokens, min_turns_to_keep)` removes oldest history entries.

---

## 18. Type Reference

All types in `src/api/cardinal_types.h`. Pybind11-compatible.

### CardinalStatus

```cpp
enum class CardinalStatus : int {
    OK = 0, NOT_INITIALIZED = 1, ALREADY_INITIALIZED = 2,
    INFERENCE_FAILED = 3, STORAGE_ERROR = 4, CONFIG_ERROR = 5,
    INVALID_INPUT = 6, EXPORT_FAILED = 7, SESSION_NOT_FOUND = 8,
    AUTH_FAILED = 9, TIMEOUT = 10, SHUTDOWN = 11, TOOL_EXECUTION_FAILED = 12,
    AGENT_MAX_ITERATIONS_REACHED = 13, INTERNAL_ERROR = 99
};
```

### CardinalResult\<T\>

```cpp
template<typename T>
struct CardinalResult {
    CardinalStatus status;
    std::string error_message;
    T value;
    bool ok() const;
};
```

### FeelingInfo, ChatResponse, SessionInfo, RuleInfo, EpisodeInfo, SystemStats, ExportRequest, ExportInfo, CardinalSettings, ToolCall, AgentStep, ReasoningTrace, ExplainabilityExport

(Full definitions in `cardinal_types.h`)

---

## 19. Error Handling

### 19.1 API Boundary

No C++ exceptions cross the `CardinalAPI` boundary. All internal exceptions caught and converted to `CardinalResult<T>`.

### 19.2 Non-Fatal Paths

- Retrieval failure → warning, continue without memory context
- Neural verifier failure → fall back to symbolic only
- Contradiction check failure → log, proceed without verification
- Tool execution failure → mark step failed, continue agent loop (if self-correction enabled)
- JSONL migration parse errors → skip line, continue

### 19.3 Fatal Paths

These cause `init()` to fail:
- Config file missing/invalid
- Model file missing
- Grammar file missing
- SQLite database cannot be opened
- llama.cpp/TensorRT engine fails to load
- Explainability key generation fails (if `auto_generate_keys: true`)

---

## 20. Offline Builds & Vendoring

**New in v1.1.0.** Cardinal now supports fully offline builds with vendored dependencies.

### 20.1 Vendored Dependencies

Users populate `vendor/` with:

| Dependency | Clone as |
| :--- | :--- |
| https://github.com/ggerganov/llama.cpp | `vendor/llama.cpp` |
| https://github.com/nlohmann/json | `vendor/nlohmann_json` |
| https://github.com/yhirose/cpp-httplib | `vendor/cpp-httplib` |
| https://github.com/mlc-ai/tokenizers-cpp | `vendor/tokenizers-cpp` |

### 20.2 Minimal Vendor Structure

Only source + headers are tracked in Git. No `.git`, `.github`, `tests/`, `examples/`, `docs/`, or build artifacts.

### 20.3 System Dependencies

Installed via `apt` (not vendored):
- `build-essential`, `cmake`
- `libsqlite3-dev`, `libssl-dev`
- `swi-prolog`
- `python3` (for `run_python` tool)

### 20.4 Offline Build Command

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

No internet required after vendored dependencies are in place.

---

## 21. Module Reference

### 21.1 Logger (`src/utils/logger.h`)

Thread-safe singleton. Levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL.

```cpp
LOG_INFO("message");
LOG_ERROR("Failed to load model: {}", path);
```

### 21.2 ConfigLoader (`src/utils/config_loader.h`)

```cpp
auto config = ConfigLoader::load("config.json");
ConfigLoader::validate(config);
```

### 21.3 JsonParser (`src/utils/json_parser.h`)

```cpp
FeelingOutput feeling = JsonParser::parse_feeling_output(json_str);
std::vector<Rule> rules = JsonParser::load_rules(path);
JsonParser::save_rules(path, rules);
std::string id = JsonParser::generate_id();
```

### 21.4 LLMEngine (`src/core/llm_engine.h`)

Abstract interface. Implementations: `LlamaCppEngine`, `TensorRTEngine`.

### 21.5 InferencePipeline (`src/core/inference.h`)

```cpp
InferenceRequest req;
req.user_message = message;
req.history = session.get_history();
req.active_rules = rule_store.query(...);
req.max_iterations = is_agentic ? config.max_iterations : 0;

InferenceResponse resp = pipeline.run(req, stream_callback);
```

If `max_iterations > 0`, the pipeline runs in agentic mode.

### 21.6 AgentExecutor (`src/agent/agent_executor.h`)

**New in v1.2.0.**

```cpp
AgentResult result = executor.run(goal, max_iterations);
```

### 21.7 ToolExecutor (`src/tools/tool_executor.h`)

**New in v1.2.0.**

```cpp
ToolResult result = executor.execute(tool_call);
```

### 21.8 EpisodicRetriever (`src/memory/episodic_retriever.h`)

```cpp
auto results = retriever.retrieve("What is entropy?");
retriever.notify_new_episode(ep_id);
retriever.set_mode(RetrievalMode::HYBRID);
```

### 21.9 RuleExtractor (`src/verifier/rule_extractor.h`)

```cpp
ExtractionResult result = extractor.extract(input);
```

### 21.10 ConsistencyChecker (`src/verifier/consistency_check.h`)

```cpp
ConsistencyCheckResult result = checker.check(input);
auto contradictions = checker.run_full_scan();
int removed = checker.run_maintenance();
```

### 21.11 TrainingExporter (`src/learning/training_exporter.h`)

```cpp
ExportFilter filter{.min_confidence = 0.7f, .include_rules = true};
auto stats = exporter.export_to_file("output.jsonl", filter);
```

### 21.12 ExplainabilityManager (`src/explainability/manager.h`)

**New in v1.2.0.**

```cpp
auto trace = manager.build_trace(inference_id, request, response, reasoning_trace);
manager.save_trace(trace);
manager.sign_trace(trace);
auto export_id = manager.export_traces(session_id, start_time, end_time);
```

---

## 22. Threading Model

### 22.1 Component Thread Safety

| Component | Mechanism |
|-----------|-----------|
| `Logger` | Singleton with internal mutex |
| `RuleStore` | `std::mutex` on all operations |
| `EpisodicStorage` | `std::mutex` + SQLite `FULLMUTEX` |
| `EpisodicRetriever` | `std::shared_mutex` (shared reads, exclusive writes) |
| `SettingsManager` | `std::shared_mutex` |
| `SessionManager` | `std::mutex` on map operations |
| `ConsistencyChecker` | `std::mutex` on `check()` |
| `ToolExecutor` | `std::mutex` (state changes only, execution is parallel across tools) |
| `ExplainabilityManager` | `std::mutex` on audit log writes |
| `CardinalAPI` | Multiple mutexes (see below) |

### 22.2 CardinalAPI Mutexes

- **`inference_mutex_`** (`std::mutex`) — Serializes all inference. Only one at a time.
- **`agent_mutex_`** (`std::mutex`) — Serialises agentic loops (separate from chat inference to allow concurrent non-agentic requests).
- **`session_mutex_`** (`std::shared_mutex`) — Protects session map.
- **`api_mutex_`** (`std::mutex`) — Protects init/shutdown.

### 22.3 HTTP Server Threading

`cpp-httplib` handles one request at a time. Read-only endpoints (`/api/stats`, `/api/rules`, etc.) do not hold `inference_mutex_`. Chat endpoints hold `inference_mutex_` for the entire inference. Agentic endpoints hold `agent_mutex_`.

### 22.4 Tool Execution Threading

Tool calls are executed on the main inference thread (agent loop owns the execution). This is intentional — no parallelism within a single agent loop, simplifying state management.

---

## 23. Lifecycle and Startup Sequence

### 23.1 Full Startup Sequence

```
main()
  |
  +-- Logger::instance().init()
  +-- CardinalAPI::init("config.json")
        |
        +-- ConfigLoader::load()
        +-- RuleStore, KnowledgeGraph, EpisodicMemory, EpisodicStorage init
        +-- rule_store.load(), knowledge_graph.load()
        +-- episodic.open(), storage.open() + migration
        +-- EpisodicRetriever::init() (TF-IDF index)
        +-- SymbolicEngine::init(kb_path)
        +-- RuleExtractor init
        +-- NeuralVerifier::load() (if path provided)
        +-- ConsistencyChecker::init()
        +-- LLMEngine::load_model() (llama.cpp or TensorRT)
        +-- InferencePipeline init
        +-- AgentExecutor init
        +-- ToolExecutor init
        +-- ExplainabilityManager::init() (if enabled)
        +-- TrainingExporter init
        +-- SettingsManager init
        +-- SessionManager init, default session created
        +-- initialized = true
  |
  +-- HttpServer start (if http_enabled)
  |
  +-- interactive loop
```

### 23.2 Per-Inference Sequence (Chat Mode)

```
chat_stream()
  |
  +-- acquire inference_mutex_
  +-- ensure session exists
  +-- retriever.retrieve() → inject memory context
  +-- Pass 1: constrained generation (feeling output)
  +-- Pass 2: free generation (response)
  +-- dual-write: JSONL + SQLite
  +-- retriever.notify_new_episode()
  +-- consistency check + rule extraction + contradiction resolution
  +-- if rule committed: storage.set_extracted_rule_id()
  +-- if explainability enabled: build, sign, save trace
  +-- if rule store dirty: rule_store.save()
  +-- release inference_mutex_
  +-- update session history
```

### 23.3 Per-Inference Sequence (Agentic Mode)

```
chat_agentic()
  |
  +-- acquire agent_mutex_
  +-- ensure session exists
  +-- AgentExecutor::run(goal)
        |
        +-- PLAN: decompose goal, feeling check, symbolic check
        +-- EXECUTE LOOP (i = 0..max_iterations)
        |     |
        |     +-- THINK: InferencePipeline (single step, no agent loop inside)
        |     +-- ACT: ToolExecutor.execute()
        |     +-- OBSERVE: inject results, update understanding
        +-- FINALIZE: generate final response
  +-- trace saved to audit log (if explainability enabled)
  +-- release agent_mutex_
```

### 23.4 Shutdown Sequence

```
CardinalAPI::shutdown()
  |
  +-- acquire api_mutex_
  +-- checker.run_maintenance()
  +-- rule_store.save()
  +-- agent_working_memory close (flush)
  +-- explainability_audit close
  +-- sessions.destroy_all()
  +-- storage.close() (WAL checkpoint)
  +-- initialized = false
```

---

*This documentation reflects Cardinal v1.2.0. The source of truth is always the source code.*
