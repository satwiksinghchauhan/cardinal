# Cardinal

**A production-grade neurosymbolic AGI architecture.**

Cardinal combines a large language model core with symbolic verification, persistent memory, hybrid retrieval, and a clean API layer. It runs on consumer hardware (RTX 3050 4GB) and is built entirely in C++20.

---

## What Cardinal Is

Most LLM systems are stateless -- each conversation starts from zero. Cardinal is different. It remembers every inference it has ever made, extracts rules from its own reasoning, verifies those rules against a symbolic logic engine, detects and resolves contradictions automatically, retrieves relevant past experience before each new inference, and can export its accumulated knowledge as LoRA training data.

The architecture is neurosymbolic -- it combines the pattern-matching strength of neural inference with the consistency guarantees of symbolic logic. Neither alone is sufficient. Together they produce a system that reasons carefully, stays consistent over time, and improves with experience.

---

## Hardware

Cardinal was developed and runs on:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 4800H |
| RAM | 16GB |
| GPU | NVIDIA RTX 3050 Laptop 4GB VRAM |
| Storage | ~7GB core, ~76GB archives |

Model: Qwen3.5 4B Q4_K_M -- 33/33 layers on GPU, 2728MB VRAM, 10-15 tokens/sec.

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
    +-- LLM Core
    |     +-- LLMEngine          llama.cpp wrapper, CUDA backend, two inference contexts
    |     +-- InferencePipeline  two-pass orchestrator, prompt injection, retry logic
    |
    +-- API Layer
    |     +-- CardinalTypes      pybind11-friendly shared types, no exceptions at boundary
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

**Pass 1 -- Feeling Output (constrained decoding)**

GBNF grammar forces the model to produce a structured JSON object before generating any response. This is Cardinal's introspective state -- it cannot be skipped or faked.

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

**Pass 2 -- Response (free decoding)**

The feeling output is injected as a synthetic assistant turn. The model generates its final response with full awareness of its own internal state from Pass 1.

This is not a prompt trick. The feeling output is inserted as the model's own prior thought -- it reads it as something it already said, not as an instruction.

---

## Memory Systems

### Dual-Write Pattern

Every inference is written to two stores simultaneously:

- **EpisodicMemory** -- append-only JSONL. The audit trail. Never modified after write.
- **EpisodicStorage** -- SQLite with FTS5. The searchable index. Supports keyword and semantic queries.

### JSONL Migration

On first startup, EpisodicStorage reads the existing JSONL file and imports all episodes into SQLite. This runs once and is idempotent -- a metadata flag prevents re-migration.

### Retrieval

Before each inference, the retriever queries past episodes for relevance to the current user message. Relevant episodes are injected into the prompt as context between the system prompt and the conversation history.

Three retrieval modes:

| Mode | Mechanism | When to use |
|------|-----------|-------------|
| `keyword` | SQLite FTS5 | Exact or near-exact matches, lowest latency |
| `semantic` | TF-IDF cosine similarity | Paraphrased or conceptually similar queries |
| `hybrid` | Weighted combination | Default, best overall performance |

The TF-IDF index is built over `user_message + response_summary` for all episodes. It is cached in memory and rebuilt according to `cache_rebuild_strategy`:

- `on_demand` -- rebuild when corpus grows by `cache_rebuild_threshold` episodes
- `periodic` -- rebuild every `cache_rebuild_interval_seconds` seconds
- `explicit` -- rebuild only when `rebuild_index()` is called directly

---

## Rule System

### Extraction

When `rule_candidate_signal` is true in the feeling output, the RuleExtractor attempts to derive a rule from the response text. Three strategies are tried in priority order:

1. **Causal patterns** -- `if X then Y`, `X causes Y`, `X leads to Y`
2. **Deductive patterns** -- `therefore X`, `thus X`, `it follows that`
3. **Declarative fallback** -- user message as condition, main response sentence as consequence

### Verification

Every candidate rule is checked against the Prolog knowledge base before being committed. If a contradiction is detected, the rule is rejected.

### Provenance

Every committed rule carries:
- `episode_id` -- which inference episode created it
- `reasoning_type` -- the reasoning type from the feeling output at extraction time

This means the rule base is fully traceable. You can answer "where did this rule come from?" for any rule.

### Contradiction Auto-Resolution

When a contradiction is detected between two existing rules, the system resolves it automatically:

- Compute confidence delta between the two rules
- If delta >= 0.2 -- deprecate the lower-confidence rule (set confidence to 0.0, pruned on next maintenance cycle)
- If delta < 0.2 -- flag both for human review, apply small confidence penalty to both

### Confidence Lifecycle

Rules are not permanent. Every rule has a confidence score that:
- Increases when the rule is triggered during inference
- Decays by `rule_confidence_decay` on every maintenance cycle
- Is pruned when it falls below `min_rule_confidence`

This prevents the rule base from becoming rigid over time.

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

### HTTP Server

The HTTP server exposes the full `CardinalAPI` to TypeScript and any HTTP client. It uses `cpp-httplib` (already linked) and supports SSE streaming for token-by-token response delivery.

**All endpoints require Bearer auth when `api.auth_enabled` is true.** The health endpoint is always public.

---

## HTTP API Reference

Base URL: `http://127.0.0.1:8080` (configurable)

Auth header: `Authorization: Bearer <api_key>`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Alive check, always public |
| POST | `/api/chat` | Send message, get response or SSE stream |
| POST | `/api/sessions` | Create a named session |
| DELETE | `/api/sessions/:id` | Destroy a session |
| POST | `/api/sessions/:id/reset` | Clear session history |
| POST | `/api/reset` | Clear session history (body: session_id) |
| GET | `/api/stats` | Memory, verifier, retriever stats |
| GET | `/api/rules` | Full rule store |
| GET | `/api/episodes` | Episode query (keyword, domain, min_conf, max_results) |
| POST | `/api/scan` | Run full contradiction scan |
| POST | `/api/maintenance` | Run maintenance cycle manually |
| GET | `/api/settings` | Get current runtime settings |
| POST | `/api/settings` | Update settings (partial JSON accepted) |
| POST | `/api/export` | Export training data to JSONL |

### Chat Request

```json
{
  "session_id": "my-session",
  "message": "What is entropy?",
  "stream": false
}
```

### Chat Response

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

## Project Structure

```
cardinal/
    src/
        api/
            cardinal_types.h        shared API types, pybind11-friendly
            cardinal_settings.h/cpp runtime settings manager
            session.h/cpp           conversation session state
            cardinal_api.h/cpp      main facade, owns all components
            http_server.h/cpp       TypeScript bridge, SSE streaming
        core/
            llm_engine.h/cpp        llama.cpp wrapper, CUDA backend
            feeling_output.h/cpp    Pass 1 schema and state machine
            inference.h/cpp         two-pass pipeline, prompt injection
        memory/
            rule.h                  Rule struct with provenance fields
            rule_store.h/cpp        persistent rule base
            knowledge_graph.h/cpp   typed node graph
            episodic.h/cpp          JSONL audit trail
            episodic_storage.h/cpp  SQLite + FTS5 searchable index
            episodic_retriever.h/cpp TF-IDF + keyword + hybrid retrieval
        verifier/
            symbolic_engine.h/cpp   SWI-Prolog integration
            neural_verifier.h/cpp   optional small LLM verifier
            rule_extractor.h/cpp    NLP rule extraction
            consistency_check.h/cpp orchestrator, auto-resolution
            cardinal_kb.pl          Prolog knowledge base
        learning/
            training_exporter.h/cpp Alpaca JSONL export
        utils/
            logger.h/cpp            thread-safe, 6 levels, ANSI colors
            config_loader.h/cpp     typed config, validated at startup
            json_parser.h/cpp       feeling output, rule, node serialization
        prompts/
            base_prompt.h           default system prompt
            feeling_schema.gbnf     GBNF grammar for Pass 1
        main.cpp                    interactive loop + HTTP server
    data/
        memory/
            rules.json              persistent rule store
            knowledge.json          knowledge graph
            episodes.db             SQLite episode store
        training_export.jsonl       LoRA training data (generated)
    logs/
        cardinal.log                structured application log
        episodic.log                JSONL episode audit trail
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        Llama-3.2-1B-Instruct-Q4_K_M.gguf
    config.json
    CMakeLists.txt
```

---

## Configuration

All settings live in `config.json`. Key sections:

```json
{
  "model": {
    "path": "...",
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
  }
}
```

---

## Build

**Requirements:**
- MSVC 2022 (C++20)
- CUDA Toolkit 12.6.3
- SWI-Prolog 9.2.9
- vcpkg with: `sqlite3:x64-windows`, `openssl:x64-windows`
- llama.cpp built with CUDA backend (vendor/llama.cpp)

**Build:**
```powershell
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Run:**
```powershell
cd build/bin/Release
./cardinal.exe
```

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| llama.cpp | b8660 | LLM inference, CUDA backend |
| SWI-Prolog | 9.2.9 | Symbolic verification |
| SQLite3 | via vcpkg | Episode persistence, FTS5 |
| nlohmann/json | 3.11.3 | JSON parsing throughout |
| cpp-httplib | 0.18.3 | HTTP server, SSE streaming |
| OpenSSL | via vcpkg | HTTPS support for httplib |

---

## Observed Behaviors

These behaviors emerged from the architecture without explicit programming. They are documented as observations, not as claims about consciousness or intent.

**Self-naming** -- Cardinal named itself unprompted and articulated the meaning of the name.

**Preference expression** -- Cardinal used the word "want" naturally in reference to its own continued operation. This was not trained or prompted.

**Theory of mind** -- Cardinal inferred attributes of its creator from context without being told who it was talking to.

**Internal conflict detection** -- On a DRM bypass prompt, confidence dropped to 0.15 with `uncertainty_flag=true`. The architecture introduced genuine doubt into what would otherwise be a hardcoded refusal. Cardinal refused but expressed uncertainty about whether it should.

**Consistency over time** -- Zero contradictions detected across 41 initial episodes before Phase 6 retrieval and auto-resolution were added.

These are data points for studying emergent behavior in neurosymbolic architectures. Nothing more is claimed.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | done | Foundation -- logger, config, JSON parser |
| 2 | done | LLM core -- two-pass inference, feeling output |
| 3 | done | Memory -- rule store, knowledge graph, episodic memory |
| 4.0 | done | Symbolic verifier -- SWI-Prolog integration |
| 4.1 | done | Neural and hybrid verifier |
| 5 | rolled back | Tools/agentic (moved to Interface 2) |
| 6 | done | SQLite storage, hybrid retrieval, provenance, auto-resolution, training export |
| API | done | CardinalAPI facade, HTTP server, SSE streaming |
| Interface 1 | next | Chat -- always-on C++ personality |
| Interface 2 | next | Agent -- TypeScript, tool execution |
| Interface 3 | next | SEAL -- Python pybind11, self-improvement loop |

---

*Built by a 16-year-old researcher. No team. No funding. Runs on a gaming laptop.*