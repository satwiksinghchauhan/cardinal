# Cardinal -- Complete Technical Documentation

**Version:** 1.0.0
**Architecture:** Neurosymbolic AGI Core
**Language:** C++20
**Platform:** Windows x64, NVIDIA CUDA

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Two-Pass Inference](#2-two-pass-inference)
3. [Feeling Output Schema](#3-feeling-output-schema)
4. [Memory Systems](#4-memory-systems)
5. [Retrieval System](#5-retrieval-system)
6. [Verifier Pipeline](#6-verifier-pipeline)
7. [Rule System](#7-rule-system)
8. [Training Export](#8-training-export)
9. [Configuration Reference](#9-configuration-reference)
10. [CardinalAPI Reference](#10-cardinalapi-reference)
11. [HTTP API Reference](#11-http-api-reference)
12. [Settings Manager](#12-settings-manager)
13. [Session Manager](#13-session-manager)
14. [Type Reference](#14-type-reference)
15. [Error Handling](#15-error-handling)
16. [Module Reference](#16-module-reference)
17. [Data Formats](#17-data-formats)
18. [Threading Model](#18-threading-model)
19. [Lifecycle and Startup Sequence](#19-lifecycle-and-startup-sequence)
20. [Extending Cardinal](#20-extending-cardinal)

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

Layer 2 -- Core Systems
    InferencePipeline  two-pass orchestrator, prompt injection
    LLMEngine          llama.cpp wrapper, CUDA backend
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

`CardinalAPI` owns every component via `std::unique_ptr`. Components are constructed in `init()` and destroyed in `shutdown()`. No component is accessible from outside the API boundary -- callers see only the types defined in `cardinal_types.h`.

### Dependency Graph

```
CardinalAPI
    owns --> LLMEngine
    owns --> InferencePipeline --> LLMEngine
                               --> EpisodicRetriever
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
    owns --> SessionManager
    owns --> HttpServer (separate, explicit start)
```

---

## 2. Two-Pass Inference

Every inference Cardinal performs consists of exactly two passes through the language model. The two passes use separate inference contexts to prevent grammar state contamination.

### Pass 1 -- Constrained Decoding (Feeling Output)

The first pass uses GBNF grammar-constrained decoding to force the model to produce a structured JSON object before generating any natural language response. This JSON object is called the **feeling output**.

The grammar is defined in `src/prompts/feeling_schema.gbnf`. The grammar constraint means the model cannot produce malformed output -- every token it generates must be valid JSON conforming to the feeling schema.

The feeling output captures six fields:
- `confidence` -- how confident the model is in its response (0.0 to 1.0)
- `reasoning_type` -- what kind of reasoning this response requires
- `reasoning_domain` -- what domain this reasoning operates in
- `uncertainty_flag` -- whether the model is uncertain
- `contradiction_flag` -- whether the model has detected a conflict
- `rule_candidate_signal` -- whether a general rule might be extractable

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

The model in Pass 2 sees the feeling output as its own prior thought and uses it to calibrate its response -- higher uncertainty flags lead to more hedged language, contradiction flags trigger careful qualification, and so on.

### Pass 2 -- Free Decoding (Final Response)

Pass 2 runs with no grammar constraint. The model generates its final natural language response. The stream callback, if provided, is called for each token as it is generated.

### Retry Logic

If Pass 1 fails to produce a valid feeling output (the JSON does not parse, or fails schema validation), the pipeline retries up to `feedback.max_retries` times with a `feedback.retry_delay_ms` delay between attempts. If all retries are exhausted, the inference fails and `InferenceResponse.success` is false.

Pass 2 does not retry -- if the response generation fails, the inference fails.

### State Machine

The `FeelingContext` object tracks the state of a single inference cycle:

```
IDLE -> PASS1_FEELING -> PASS2_RESPONSE -> COMPLETE
                      -> FAILED (on retry exhaustion)
```

### Metrics

Every `InferenceResponse` carries an `InferenceMetrics` object with:
- `pass1_tokens_generated` -- tokens produced in Pass 1
- `pass2_tokens_generated` -- tokens produced in Pass 2
- `total_duration` -- wall clock time for the full inference cycle
- `retry_count` -- how many Pass 1 retries were needed

---

## 3. Feeling Output Schema

The feeling output is the core introspective signal that drives everything downstream. It is produced by Pass 1 and consumed by the verifier pipeline, the retriever, the session history, and the training exporter.

### Fields

**`confidence`** (float, 0.0 to 1.0)

How confident the model is in its response. This is not a post-hoc annotation -- the model produces this value before generating its response. Values below 0.4 are considered low confidence. Values above 0.7 are considered high confidence.

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

The domain is used for rule store partitioning, retrieval filtering, and verifier routing (the symbolic verifier is weighted higher for factual and mathematical domains in hybrid mode).

**`uncertainty_flag`** (boolean)

True when the model is uncertain about its response. Logically inconsistent with high confidence -- the validator will reject a feeling output with `confidence > 0.8` and `uncertainty_flag = true`.

When `uncertainty_flag` is true, the response will typically contain hedging language ("I believe", "it is likely", "I am not certain") even though this is never explicitly instructed.

**`contradiction_flag`** (boolean)

True when the model has detected a conflict in its own reasoning or between the current query and prior knowledge. When this flag is true, the consistency checker runs a targeted contradiction scan against the rule base for the current domain.

**`rule_candidate_signal`** (boolean)

True when the model believes its response contains a general rule that could be extracted and stored. When this flag is true, the rule extractor attempts to derive a Rule from the response text.

### Validation

The `JsonParser::validate_feeling_output()` function enforces logical consistency beyond field-level type checking:
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
- `add_rule(domain, condition, consequence, confidence, episode_id, reasoning_type)` -- add a new rule. If a semantically similar rule exists in the same domain (Jaccard similarity >= 0.7), the existing rule is merged instead of creating a duplicate.
- `get_rule(id)` -- fetch a single rule by ID
- `query(RuleQuery)` -- fetch rules matching domain, condition hint, confidence threshold
- `decay_confidence()` -- reduce all rule confidences by `rule_confidence_decay`
- `prune()` -- remove rules below `min_rule_confidence`
- `enforce_limit()` -- remove lowest-confidence rules if count exceeds `max_rules`

**Rule similarity:** Two rules are considered similar if they share the same domain and their condition texts have Jaccard word overlap >= 0.7. When a similar rule is found on insert, the existing rule's confidence is boosted toward the higher value and its consequence is updated if the incoming one is more detailed.

**Confidence lifecycle:**
```
add_rule()          -- initial confidence (default 0.5)
record_trigger()    -- +0.01 per trigger
decay_confidence()  -- -(rule_confidence_decay) per maintenance cycle
prune()             -- removed if below min_rule_confidence
```

**Thread safety:** All operations protected by `std::mutex`.

**Dirty tracking:** The rule store tracks whether it has been modified since the last save. `is_dirty()` returns true if unsaved changes exist. `save()` is a no-op if not dirty.

### 4.2 KnowledgeGraph

The knowledge graph stores typed factual nodes and their relationships.

**Storage:** `data/memory/knowledge.json` (JSON array, atomic writes)

**Node types:** `concept`, `fact`, `entity`, `relation`

**Key operations:**
- `add_node(KnowledgeNode)` -- insert or update a node
- `get_node(id)` -- fetch a node by ID
- `find_path(from_id, to_id)` -- BFS path between two nodes
- `get_hub_nodes(n)` -- returns the N nodes with the most connections

The knowledge graph is currently populated manually or by future interface code. The core architecture stores and retrieves it but does not yet auto-populate it from inference cycles.

### 4.3 EpisodicMemory

The episodic memory is the append-only JSONL audit trail. It is never modified after a write -- it is the ground truth record of every inference Cardinal has ever made.

**Storage:** `logs/episodic.log` (JSONL, one JSON object per line)

**Key operations:**
- `log_episode(user_message, response, feeling, pass1_tokens, pass2_tokens, total_ms)` -- append one episode, returns the generated episode ID
- `open()` -- open the file for appending
- `close()` -- flush and close

**Episode ID format:** `<hex_timestamp_ms>_<4digit_counter>` -- for example `19d62f60677_0000`. This format is monotonically increasing and sortable.

**Why JSONL:** Append-only formats are crash-safe. If Cardinal crashes mid-write, the worst case is one malformed line at the end of the file. All prior lines are intact. The SQLite layer handles searchability.

### 4.4 EpisodicStorage

EpisodicStorage is the searchable SQLite index over the episode corpus. It is the query layer -- the JSONL file is the truth, SQLite is the index.

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

**Indexes:** `reasoning_domain`, `confidence`, `timestamp`, `rule_candidate` -- all indexed for fast filtered queries.

**FTS5 triggers:** Three triggers keep the FTS5 index synchronized with the episodes table automatically: `episodes_fts_insert`, `episodes_fts_delete`, `episodes_fts_update`. Any write to the episodes table automatically updates the full-text search index.

**JSONL Migration:** On first `open()`, EpisodicStorage checks the `metadata` table for the key `migration_v1_complete`. If absent, it reads the JSONL file line by line and imports all episodes into SQLite in a single transaction. After completion, it sets `migration_v1_complete = 1` in metadata. This migration is idempotent -- it will never run twice on the same database file.

**Dual-write pattern:** After every inference, the episode is written to both EpisodicMemory (JSONL) and EpisodicStorage (SQLite). Both writes happen in `CardinalAPI::run_post_inference()`.

**Key operations:**
- `insert_episode(EpisodeRecord)` -- `INSERT OR IGNORE` -- duplicate IDs are silently skipped
- `get_episode(id)` -- fetch by primary key
- `query(EpisodeQuery)` -- filtered query with optional FTS5 keyword search
- `set_extracted_rule_id(episode_id, rule_id)` -- link an episode back to the rule it produced
- `count()` -- total episodes in database
- `stats()` -- aggregate statistics

**WAL mode:** The database runs in WAL (Write-Ahead Logging) mode with `PRAGMA synchronous=NORMAL`. This provides good crash safety without the performance penalty of `PRAGMA synchronous=FULL`. WAL is checkpointed with `SQLITE_CHECKPOINT_TRUNCATE` on `close()`.

---

## 5. Retrieval System

The retrieval system finds past episodes relevant to the current user message and injects them into the inference prompt. This gives Cardinal context-aware memory -- it can recall and apply past reasoning to new queries.

### 5.1 Architecture

```
EpisodicRetriever
    |
    +-- keyword_search()    --> EpisodicStorage FTS5
    +-- semantic_search()   --> TF-IDF cosine similarity (in-memory index)
    +-- merge_results()     --> weighted combination + deduplication
```

### 5.2 Retrieval Modes

**KEYWORD mode**

Uses SQLite FTS5 full-text search over `user_message` and `response_summary`. Results are ranked by FTS5's internal BM25 scoring. This mode is always available and has the lowest latency since it requires no index computation.

Best for: queries that use the same or similar words as past episodes.

**SEMANTIC mode**

Uses TF-IDF (Term Frequency-Inverse Document Frequency) cosine similarity. The TF-IDF index is built over the concatenated `user_message + response_summary` text of all episodes and cached in memory.

At query time:
1. Tokenize the user message into lowercase unigrams
2. Strip stopwords (see stopword list in `episodic_retriever.cpp`)
3. Compute TF-IDF vector for the query
4. Compute cosine similarity between query vector and each indexed episode vector
5. Return top-N by cosine similarity score

Best for: queries that are paraphrased versions of past queries, or conceptually similar without sharing exact words.

**HYBRID mode (default)**

Runs both keyword and semantic search independently, normalizes both result sets to [0,1], then merges with configurable weights:

```
combined_score = (keyword_weight * keyword_score) + (semantic_weight * semantic_score)
```

Default weights: `keyword_weight = 0.7`, `semantic_weight = 0.3`. These can be changed at runtime via `SettingsManager` or `POST /api/settings`.

Episodes that appear in both result sets get contributions from both components. Episodes that appear in only one result set get a zero contribution from the other.

After merging, results below `min_score` are filtered out. The final result set is capped at `max_results`.

### 5.3 TF-IDF Implementation

**Vocabulary construction:**

For a corpus of N documents, the vocabulary is built by:
1. Tokenizing all documents into lowercase unigrams
2. Stripping stopwords and single-character tokens
3. Computing document frequency (DF) for each term
4. Assigning a term index to each unique term
5. Computing IDF weight: `idf = log((N+1) / (df+1)) + 1`

The +1 smoothing prevents division by zero and keeps IDF positive for terms that appear in every document.

**TF-IDF vector computation:**

For a single document with tokens T:
```
tf(term) = count(term in T) / len(T)
tfidf(term) = tf(term) * idf(term)
```

Vectors are sparse -- only terms that appear in the document have entries.

**Cosine similarity:**

```
similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Where `norm(v) = sqrt(sum(v[i]^2))`. The L2 norm of each indexed episode vector is precomputed and cached. At query time only the query vector's norm needs to be computed, making queries O(vocabulary_size) not O(vocabulary_size * corpus_size).

**Stopword list:**

The following words are removed from all texts before indexing and querying: a, an, the, and, or, but, in, on, at, to, for, of, with, by, from, is, are, was, were, be, been, being, have, has, had, do, does, did, will, would, could, should, may, might, shall, it, its, this, that, these, those, i, you, he, she, we, they, what, which, who, how, when, where, why, not, no, so, if, as, up, out, about, into, than, then, there, can, my, your.

### 5.4 Index Lifecycle

The TF-IDF index is held entirely in memory. It is built once at startup via `init()` and maintained according to the `cache_rebuild_strategy`:

**ON_DEMAND:** After each new episode is added via `notify_new_episode()`, the counter `episodes_since_rebuild_` is incremented. When it reaches `cache_rebuild_threshold`, the index is rebuilt from scratch from all episodes in storage.

**PERIODIC:** A timestamp is checked on every `notify_new_episode()` call. If more than `cache_rebuild_interval_seconds` seconds have elapsed since the last rebuild, the index is rebuilt.

**EXPLICIT:** The index is never auto-rebuilt. Call `rebuild_index()` directly when needed. This is useful for batch-processing scenarios where you want full control over when the expensive rebuild happens.

**Rebuild cost:** Rebuilding reads all episodes from SQLite, tokenizes all texts, recomputes the full vocabulary and IDF weights, and recomputes TF-IDF vectors for every episode. For a corpus of 1000 episodes this takes approximately 50-100ms on the development hardware.

### 5.5 Prompt Injection

Retrieved episodes are injected into the inference prompt as a structured context block. The injection happens in `InferencePipeline::build_messages()`, between the system prompt and the conversation history.

The injected block is formatted as a user/assistant exchange:

```
[user]:      [MEMORY CONTEXT]
             1. [factual | confidence: 94% | score: 87%]
                Q: What happens to gas molecules when temperature increases?
                A: Gas molecules move faster and collide more frequently...

             2. [factual | confidence: 89% | score: 71%]
                Q: How does pressure relate to temperature?
                A: According to Gay-Lussac's law...
             [END MEMORY CONTEXT]

[assistant]: I have reviewed the relevant past context and will use it
             to inform my response.
```

This positions the memory context as reference material the model has already acknowledged, not as an instruction. The model can then refer back to it during Pass 2.

Response summaries are truncated to 300 characters in the injected context to control prompt token budget. The full response is always available in the episode store.

If retrieval fails for any reason (index not ready, storage error), the failure is logged as a warning and inference continues without memory context. Retrieval failure is never fatal.

---

## 6. Verifier Pipeline

The verifier pipeline runs after every inference to maintain rule base integrity. It is orchestrated by `ConsistencyChecker`.

### 6.1 Modes

The verifier operates in one of three modes, configured via `verifier.mode`:

**SYMBOLIC:** Only the SWI-Prolog symbolic engine is used. This is the default and most reliable mode. Rules are verified against formal logic predicates.

**NEURAL:** Only the neural verifier is used. Requires `neural_model_path` to be set. Uses a small LLM (Llama 3.2 1B) to score rule quality and detect contradictions. Slower but handles fuzzy/semantic contradictions that formal logic misses.

**HYBRID:** Both engines run simultaneously. Results are merged with weighted consensus -- symbolic takes priority for factual and mathematical domains, neural takes priority for ethical and social domains. In hybrid mode, a committed rule's confidence is adjusted by the neural quality score: `adjustment = (quality_score - 0.5) * 0.1`, giving a range of -0.05 to +0.05.

### 6.2 Per-Inference Check Sequence

The consistency checker runs four steps after every inference:

**Step 1 -- Rule extraction (if signaled)**

If `feeling.rule_candidate_signal` is true, the rule extractor attempts to derive a rule from the response text. See Section 7 for the full extraction pipeline.

**Step 2 -- Contradiction check (if flagged or rule committed)**

If `feeling.contradiction_flag` is true OR a new rule was just committed, the symbolic engine checks for contradictions:
- Fetches the top 20 rules for the current domain
- Runs `check_contradiction(domain, condition, consequence)` for each
- For each contradiction found, calls `resolve_contradiction()` (see Section 7.4)

The neural verifier also runs in this step if the mode is `neural` or `hybrid`, verifying any newly committed rule against the top 5 rules in the domain.

**Step 3 -- Periodic maintenance (every 10 inferences)**

`inferences_since_maintenance_` is incremented on every check. When it reaches `MAINTENANCE_INTERVAL_INFERENCES` (10), a maintenance cycle runs:
- `decay_confidence()` -- all rules decay by `rule_confidence_decay`
- `prune()` -- rules below `min_rule_confidence` are removed
- `enforce_limit()` -- rules above `max_rules` are removed (lowest confidence first)
- `save()` -- rule store is saved to disk
- `sync_rules_to_prolog()` -- Prolog engine is re-synced if any rules were removed

**Step 4 -- Build summary**

A human-readable summary string is built and returned in `ConsistencyCheckResult.summary`. This is what you see in the log after each inference.

### 6.3 SymbolicEngine

The symbolic engine wraps SWI-Prolog 9.2.9 via the C foreign language interface.

**Initialization:** `symbolic.init("path/to/cardinal_kb.pl")` loads the knowledge base file into the Prolog engine.

**Rule format in Prolog:**

Rules are asserted as:
```prolog
cardinal_rule(domain, condition, consequence).
```

**Contradiction detection:**

```prolog
contradicts(rule_a, rule_b) :-
    cardinal_rule(D, C1, E1),
    cardinal_rule(D, C2, E2),
    C1 = C2,   % same condition
    E1 \= E2.  % different consequence
```

This catches cases where two rules in the same domain share a condition but have different consequences -- a direct logical contradiction.

**Bulk operations:**
- `assert_rules(rules)` -- asserts all rules at once, returns count
- `retract_all_rules()` -- removes all asserted rules (used before re-sync)
- `check_all_contradictions()` -- scans entire rule base for contradictions

**Thread safety:** SWI-Prolog is initialized with `PL_initialise()` once per process. The symbolic engine uses a single Prolog thread and all operations are serialized through `ConsistencyChecker`'s mutex.

### 6.4 NeuralVerifier

The neural verifier is optional and requires a separate model (`neural_model_path`). It uses Llama 3.2 1B Q4_K_M by default.

**Input:** Domain, candidate rule condition and consequence, top-5 existing rules in domain.

**Output:**
- `contradiction_detected` (bool) -- whether a semantic contradiction was found
- `contradiction_score` (float, 0-1) -- confidence in the contradiction
- `rule_quality_score` (float, 0-1) -- how good the rule candidate is
- `reasoning` (string) -- explanation of the verdict

**JSON output format** (the neural verifier produces structured JSON):
```json
{
  "contradiction": false,
  "contradiction_score": 0.1,
  "rule_quality": 0.8,
  "reasoning": "The candidate rule is consistent with existing rules..."
}
```

The neural verifier is disabled if `neural_model_path` is empty or if the model file does not exist. `is_available()` returns false in this case. All neural verifier code paths check `is_available()` before running.

---

## 7. Rule System

### 7.1 Rule Struct

Every rule has the following fields:

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

Rules created before Phase 6 (before provenance was added) have empty `episode_id` and `reasoning_type`. `has_provenance()` returns false for these.

### 7.2 Extraction Pipeline

Rule extraction runs when `feeling.rule_candidate_signal` is true. The extractor tries three strategies in priority order:

**Strategy 1: Causal patterns**

Applied when `reasoning_type` is `causal` or `deductive`.

Searches for sentences containing causal connectives:
- `if ... then ...`
- `when ..., ...`
- `whenever ..., ...`
- `because ..., ...`
- `X causes Y`
- `X leads to Y`
- `X results in Y`
- `X produces Y`
- `X implies Y`
- `therefore X`
- `thus X`

When a pattern is found, the text before the connective becomes the condition and the text after becomes the consequence. Both must be at least 10 characters to be accepted.

Initial confidence: **0.6**

**Strategy 2: Deductive patterns**

Applied when `reasoning_type` is `deductive` or `inductive`.

Looks for conclusion markers in sentences: `therefore`, `thus`, `hence`, `consequently`, `it follows that`, `we can conclude`, `this means`, `this shows`, `this demonstrates`, `in conclusion`, `as a result`.

When a conclusion marker is found, the preceding sentence becomes the condition and the current sentence (with the marker stripped) becomes the consequence.

Initial confidence: **0.65**

**Strategy 3: Declarative fallback**

Applied for any reasoning type when strategies 1 and 2 fail.

Uses the user message as the condition and the longest non-question, non-heading, non-list sentence from the response as the consequence. This is a weaker extraction method.

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

A deprecated rule has its confidence set to 0.0. It remains in the store until the next maintenance cycle's `prune()` call removes it. This soft-delete approach means a deprecated rule can be recovered before the next maintenance cycle if needed.

`FLAGGED` means both rules are plausible and neither can be confidently discarded. Both are penalized slightly to surface the conflict in confidence metrics. A future human review pass can inspect flagged pairs.

`SKIPPED` means one or both rule IDs were not found in the store. This can happen if a rule was pruned between detection and resolution.

### 7.5 Rule Injection into Prompts

When rules exist in the store, `InferencePipeline` formats and injects them into the system prompt as a `## Active Rules` section:

```
## Active Rules
The following rules have been derived from prior reasoning. Apply them when relevant:

1. [factual] IF gas temperature increases THEN molecules move faster (confidence: 87%)
2. [factual] IF pressure is constant THEN volume increases with temperature (confidence: 72%)
```

Rules are fetched via `InferenceRequest.active_rules`, which the caller populates before calling `pipeline.run()`. In `CardinalAPI`, this is done by querying the top rules for the current domain before each inference.

---

## 8. Training Export

The training exporter produces Alpaca-format JSONL files suitable for LoRA fine-tuning with Axolotl, LLaMA-Factory, unsloth, and similar tools.

### 8.1 Output Format

Each line in the output file is one JSON object:

```json
{"instruction": "What happens to gas molecules when temperature increases?", "input": "", "output": "When temperature increases, gas molecules gain kinetic energy and move faster..."}
```

- `instruction` -- the user message
- `input` -- always empty for Cardinal exports (Alpaca format supports a context field but Cardinal does not use it)
- `output` -- the cleaned final response

### 8.2 Response Cleaning

Before writing to the export file, responses are cleaned:

1. `<think>...</think>` blocks are stripped -- these are Qwen3's chain-of-thought traces and are not useful as training targets
2. `<feeling_state>...</feeling_state>` blocks are stripped -- these are Cardinal's synthetic turn injections
3. Consecutive newlines are collapsed to a maximum of two
4. Leading and trailing whitespace is trimmed

If a tag is unclosed (rare, only on truncated responses), everything from the opening tag to the end of the string is removed.

### 8.3 Rule Export (optional)

When `include_rules = true`, committed rules are also exported as training examples:

```json
{"instruction": "What rule applies when: gas temperature increases?", "input": "", "output": "Gas molecules move faster and collide more frequently, increasing pressure."}
```

This frames rules as question-answer pairs, teaching the model to apply known rules when queried about their conditions.

### 8.4 Filtering

The `ExportFilter` controls what gets exported:

| Field | Default | Description |
|-------|---------|-------------|
| `min_confidence` | 0.7 | Minimum episode confidence |
| `domain` | "" (all) | Filter by reasoning domain |
| `max_examples` | 0 (no limit) | Maximum examples to export |
| `include_rules` | false | Also export rules |
| `recent_first` | true | Most recent episodes first |

Only episodes with `confidence >= min_confidence` are exported. This ensures training data quality -- low-confidence responses where the model was uncertain are excluded.

### 8.5 Dry Run

`export_dry_run()` returns an `ExportInfo` showing how many examples would be exported without writing any files. Use this to preview before committing to disk.

---

## 9. Configuration Reference

All configuration lives in `config.json`. Every field is validated at startup. Cardinal will not start if any required field is missing or invalid.

### 9.1 `model` section

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Absolute path to the primary GGUF model file. Must exist. |
| `chat_template` | string | Chat template format: `qwen3`, `llama3`, `mistral` |
| `context_length` | int | Maximum context window in tokens. Must be >= 512. |
| `gpu_layers` | int | Number of layers to offload to GPU. 0 = CPU only. Must be >= 0. |
| `threads` | int | CPU threads for non-GPU work. Must be >= 1. |

### 9.2 `inference` section

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | float | Sampling temperature. Range: 0.0 to 2.0. |
| `top_p` | float | Nucleus sampling. Range: 0.0 to 1.0 (exclusive of 0). |
| `max_tokens_feeling` | int | Max tokens for Pass 1. Must be >= 32. |
| `max_tokens_response` | int | Max tokens for Pass 2. Must be >= 64. |

### 9.3 `feeling_schema` section

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"structured_json"`. |
| `grammar_path` | string | Absolute path to the GBNF grammar file. Must exist. |
| `fields` | object | Field definitions (informational, not parsed at runtime). |
| `max_tokens` | int | Alias for `inference.max_tokens_feeling`. |

### 9.4 `memory` section

| Field | Type | Description |
|-------|------|-------------|
| `rule_store_path` | string | Path to `rules.json`. Created on first save. |
| `knowledge_graph_path` | string | Path to `knowledge.json`. Created on first save. |
| `episodic_log_path` | string | Path to the JSONL episode log. Created on first episode. |
| `max_rules` | int | Maximum number of rules in the store. Must be >= 1. |

### 9.5 `verifier` section

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `"symbolic"`, `"neural"`, or `"hybrid"`. |
| `neural_model_path` | string | Path to neural verifier GGUF. Empty string disables neural verifier. |
| `neural_gpu_layers` | int | GPU layers for neural verifier. 0 = CPU only. |
| `neural_max_tokens` | int | Max tokens for neural verifier output. Default 512. |
| `contradiction_threshold` | float | Score above which a contradiction is flagged. Range: 0.0 to 1.0. |
| `rule_confidence_decay` | float | Confidence reduction per maintenance cycle. Must be >= 0. |
| `min_rule_confidence` | float | Rules below this are pruned. Range: 0.0 to 1.0. |

### 9.6 `feedback` section

| Field | Type | Description |
|-------|------|-------------|
| `max_retries` | int | Maximum Pass 1 retry attempts. Must be >= 0. |
| `retry_delay_ms` | int | Milliseconds between retries. Must be >= 0. |
| `rule_injection_format` | string | Currently `"prepend_to_context"`. Reserved for future formats. |

### 9.7 `retriever` section

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `"keyword"`, `"semantic"`, or `"hybrid"`. |
| `keyword_weight` | float | Weight for keyword score in hybrid. Range: 0.0 to 1.0. |
| `semantic_weight` | float | Weight for semantic score in hybrid. Range: 0.0 to 1.0. |
| `max_results` | int | Maximum episodes returned per retrieval. Must be >= 1. |
| `min_score` | float | Minimum combined score to include in results. Range: 0.0 to 1.0. |
| `cache_rebuild_strategy` | string | `"on_demand"`, `"periodic"`, or `"explicit"`. |
| `cache_rebuild_threshold` | int | Episodes added before on_demand rebuild. Must be >= 1. |
| `cache_rebuild_interval_seconds` | int | Seconds between periodic rebuilds. Must be >= 1. |

### 9.8 `api` section

| Field | Type | Description |
|-------|------|-------------|
| `http_enabled` | bool | Enable HTTP server on startup. |
| `host` | string | Bind address. `"127.0.0.1"` for localhost only. Cannot be empty. |
| `port` | int | Listen port. Range: 1 to 65535. |
| `auth_enabled` | bool | Require Bearer token on all requests (except `/api/health`). |
| `api_key` | string | The Bearer token. Required if `auth_enabled` is true. |
| `stream_enabled` | bool | Allow SSE streaming responses. |
| `max_request_size_kb` | int | Maximum request body size in kilobytes. Must be >= 1. |
| `request_timeout_seconds` | int | Per-request timeout. Must be >= 1. |

### 9.9 `tools` section

| Field | Type | Description |
|-------|------|-------------|
| `browser_enabled` | bool | Reserved for Interface 2 tool use. |
| `max_browse_depth` | int | Reserved. Must be >= 1. |
| `search_results_limit` | int | Reserved. Must be >= 1. |

### 9.10 `benchmark` section

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | Path to benchmark dataset file. |
| `metrics` | array of strings | Metrics to evaluate: `consistency`, `parseability`, `accuracy`. |
| `eval_frequency_seconds` | int | Seconds between auto-evaluations. Must be >= 1. |

### 9.11 `logging` section

| Field | Type | Description |
|-------|------|-------------|
| `level` | string | Minimum log level: `trace`, `debug`, `info`, `warn`, `error`, `fatal`. |
| `path` | string | Absolute path to the log file. |

---

## 10. CardinalAPI Reference

`CardinalAPI` is the single entry point for all interfaces. It owns all core components and exposes a clean, exception-free interface.

**File:** `src/api/cardinal_api.h`

### 10.1 Lifecycle

```cpp
CardinalAPI api;

// Initialize -- must be called first
CardinalVoidResult init(const std::string& config_path = "config.json");

// Clean shutdown
CardinalVoidResult shutdown();

// Check initialization state
bool is_initialized() const;
```

`init()` performs the complete startup sequence: config load, memory layer, retriever, verifier pipeline, LLM engine, and API layer components. If any component fails to initialize, `init()` returns a failure result with a descriptive message. No partial state is left -- if `init()` fails, the API is in a clean uninitialized state.

`shutdown()` runs final maintenance, saves rule store, destroys all sessions, and closes the SQLite database. It is safe to call multiple times -- subsequent calls are no-ops.

The destructor calls `shutdown()` automatically if the API was initialized and not yet shut down.

### 10.2 Session Management

```cpp
// Create a session -- returns session ID
CardinalResult<std::string> create_session(const std::string& session_id = "");

// Destroy a session
CardinalVoidResult destroy_session(const std::string& session_id);

// Reset session history without destroying
CardinalVoidResult reset_session(const std::string& session_id);

// Get session state snapshot
CardinalResult<SessionInfo> get_session(const std::string& session_id) const;

// List all active session IDs
CardinalResult<std::vector<std::string>> list_sessions() const;
```

Sessions are created automatically on the first `chat()` call if the session does not exist. You do not need to call `create_session()` explicitly unless you want a specific session ID.

The default session used by the interactive loop is `"default"`. This session is created automatically when needed.

### 10.3 Inference

```cpp
// Non-streaming chat
CardinalResult<ChatResponse> chat(
    const std::string& session_id,
    const std::string& message);

// Streaming chat with token callback
CardinalResult<ChatResponse> chat_stream(
    const std::string&       session_id,
    const std::string&       message,
    const ApiStreamCallback& stream_cb);
```

Both methods run the full inference pipeline:
1. Retrieve relevant episodes and inject into prompt
2. Pass 1: constrained feeling output decoding
3. Pass 2: free response decoding
4. Dual-write episode to JSONL and SQLite
5. Notify retriever
6. Consistency check (rule extraction, contradiction detection/resolution)
7. Link rule back to episode if committed
8. Save rule store if dirty
9. Update session history

`chat()` calls `chat_stream()` with a null callback.

`chat_stream()` calls the `stream_cb` for each token as it is generated. On the final token, `is_final` is true and `feeling` is populated. If `stream_cb` returns false, generation is aborted.

Both methods serialize inference -- only one inference runs at a time across all sessions. This is enforced by `inference_mutex_`.

### 10.4 Memory

```cpp
// System stats
CardinalResult<SystemStats> get_stats() const;

// All rules
CardinalResult<std::vector<RuleInfo>> get_rules() const;

// Episode query
CardinalResult<std::vector<EpisodeInfo>> get_episodes(
    const std::string& keyword    = "",
    const std::string& domain     = "",
    float              min_conf   = 0.0f,
    int                max_results = 20) const;

// Full contradiction scan
CardinalResult<ScanResult> run_scan();

// Manual maintenance cycle
CardinalVoidResult run_maintenance();
```

`run_scan()` runs `ConsistencyChecker::run_full_scan()` which checks every rule against every other rule in the Prolog engine. For large rule bases this is O(n^2). After the scan, if any rules were deprecated, the Prolog engine is re-synced and the rule store is saved.

`run_maintenance()` manually triggers the decay/prune/save cycle. This cycle also runs automatically every 10 inferences.

### 10.5 Training Export

```cpp
// Export to file
CardinalResult<ExportInfo> export_training_data(const ExportRequest& request);

// Preview without writing
CardinalResult<ExportInfo> export_dry_run(const ExportRequest& request) const;
```

### 10.6 Settings

```cpp
// Get current settings
CardinalResult<CardinalSettings> get_settings() const;

// Update settings (full struct)
CardinalVoidResult update_settings(const CardinalSettings& settings);

// Update single setting by key/value string
CardinalVoidResult set_setting(const std::string& key, const std::string& value);

// Reset to config file defaults
CardinalVoidResult reset_settings();
```

Valid keys for `set_setting()`: `retriever_mode`, `keyword_weight`, `semantic_weight`, `max_retrieval_results`, `min_retrieval_score`, `verifier_mode`, `min_rule_confidence`, `contradiction_threshold`, `temperature`, `top_p`, `stream_responses`, `log_level`.

### 10.7 Health

```cpp
CardinalVoidResult health_check() const;
std::string        uptime_string() const;
```

`health_check()` returns `OK` if initialized and not shutting down. Returns `NOT_INITIALIZED` or `SHUTDOWN` otherwise.

`uptime_string()` returns a formatted string like `"02h 14m 37s"`.

---

## 11. HTTP API Reference

The HTTP server exposes the full `CardinalAPI` to TypeScript and any HTTP client. All responses are JSON. Streaming uses Server-Sent Events.

**Base URL:** `http://127.0.0.1:8080` (configurable via `api.host` and `api.port`)

**Authentication:** All endpoints except `/api/health` require:
```
Authorization: Bearer <api_key>
```

When `auth_enabled` is false in config, the auth header is not required on any endpoint.

**CORS:** All endpoints include CORS headers. `Access-Control-Allow-Origin: *` is set on all responses. OPTIONS preflight requests return 204 with the appropriate CORS headers.

**Error format:** All errors return JSON:
```json
{
  "error": "Human-readable error message",
  "status": 6,
  "code": "INVALID_INPUT"
}
```

### 11.1 GET /api/health

Always public. Returns 200 if the server is running.

**Response:**
```json
{
  "status": "ok",
  "uptime": "00h 12m 34s",
  "version": "1.0.0"
}
```

Returns 503 if the API is not initialized or is shutting down.

---

### 11.2 POST /api/chat

Send a message and receive a response.

**Request body:**
```json
{
  "session_id": "my-session",
  "message": "What is entropy?",
  "stream": false
}
```

`session_id` defaults to `"default"` if omitted. Session is created automatically if it does not exist. `stream` defaults to false.

**Non-streaming response (200):**
```json
{
  "session_id": "my-session",
  "response": "Entropy is a measure of disorder...",
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

**Streaming response:**

To receive a streaming response, set `"stream": true` in the request body OR set `Accept: text/event-stream` in the request headers.

The response Content-Type is `text/event-stream`. Each token is delivered as an SSE event:

```
data: {"token":"Entropy","is_final":false}

data: {"token":" is","is_final":false}

data: {"token":" a","is_final":false}

data: {"token":"","is_final":true,"feeling":{"confidence":0.94,"reasoning_type":"deductive","reasoning_domain":"factual","uncertainty_flag":false,"contradiction_flag":false,"rule_candidate":false}}

```

The final event has `"is_final": true`, an empty token, and the full feeling object. After the final event the SSE stream ends.

On error during streaming:
```
data: {"error":"Inference failed: ..."}

```

**Error responses:**
- 400 -- missing or empty `message` field, or invalid JSON body
- 401 -- missing or invalid auth header
- 500 -- inference failed

---

### 11.3 POST /api/sessions

Create a named session.

**Request body:**
```json
{
  "session_id": "agent-session-1"
}
```

`session_id` is optional. If omitted, a session ID is generated automatically.

If the requested `session_id` already exists, returns the existing session ID without error.

**Response (200):**
```json
{
  "status": "ok",
  "session_id": "agent-session-1"
}
```

---

### 11.4 DELETE /api/sessions/:id

Destroy a session and its entire history.

**Response (200):**
```json
{
  "status": "ok",
  "session_id": "agent-session-1",
  "message": "Session destroyed"
}
```

**Error responses:**
- 404 -- session not found

---

### 11.5 POST /api/sessions/:id/reset

Clear a session's conversation history without destroying the session.

**Response (200):**
```json
{
  "status": "ok",
  "session_id": "agent-session-1",
  "message": "Session history cleared"
}
```

**Error responses:**
- 404 -- session not found

---

### 11.6 POST /api/reset

Alternative session reset endpoint. Accepts session ID in the body.

**Request body:**
```json
{
  "session_id": "my-session"
}
```

`session_id` defaults to `"default"` if omitted.

**Response (200):**
```json
{
  "status": "ok",
  "session_id": "my-session"
}
```

---

### 11.7 GET /api/stats

Returns a complete system statistics snapshot.

**Response (200):**
```json
{
  "memory": {
    "total_episodes": 87,
    "migrated_episodes": 41,
    "high_conf_episodes": 62,
    "rule_candidate_count": 14,
    "avg_episode_confidence": 0.91,
    "total_rules": 12,
    "active_rules": 10,
    "avg_rule_confidence": 0.67,
    "index_size": 87,
    "vocabulary_size": 2341,
    "index_ready": true
  },
  "verifier": {
    "total_checks": 46,
    "total_rules_extracted": 12,
    "total_contradictions": 2,
    "total_resolved": 1,
    "total_flagged": 1,
    "total_maintenance_runs": 4
  },
  "uptime": "00h 34m 12s",
  "version": "1.0.0",
  "initialized": true
}
```

---

### 11.8 GET /api/rules

Returns all rules in the rule store.

**Response (200):**
```json
{
  "rules": [
    {
      "id": "19d62f60677_0001",
      "domain": "factual",
      "condition": "gas temperature increases",
      "consequence": "gas molecules move faster and pressure increases",
      "confidence": 0.87,
      "trigger_count": 3,
      "episode_id": "19d62f60677_0000",
      "reasoning_type": "causal",
      "created_at": "2026-04-06T18:53:03",
      "updated_at": "2026-04-07T10:22:41",
      "has_provenance": true
    }
  ],
  "count": 1
}
```

---

### 11.9 GET /api/episodes

Query the episode store.

**Query parameters (all optional):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | string | FTS5 keyword search over user_message and response_summary |
| `domain` | string | Filter by reasoning_domain |
| `min_conf` | float | Minimum confidence threshold (default 0.0) |
| `max_results` | int | Maximum results to return (default 20) |

**Examples:**
```
GET /api/episodes
GET /api/episodes?keyword=entropy
GET /api/episodes?domain=factual&min_conf=0.8&max_results=5
GET /api/episodes?keyword=gas+molecules&domain=factual
```

**Response (200):**
```json
{
  "episodes": [
    {
      "id": "19d62f60677_0000",
      "timestamp": "2026-04-06T18:53:03",
      "user_message": "What happens to gas molecules when temperature increases?",
      "response_summary": "When temperature increases, gas molecules gain kinetic energy...",
      "confidence": 0.95,
      "reasoning_type": "deductive",
      "reasoning_domain": "factual",
      "contradiction": false,
      "uncertainty": false,
      "rule_candidate": true,
      "extracted_rule_id": "19d62f60677_0001",
      "pass1_tokens": 49,
      "pass2_tokens": 195,
      "total_ms": 19568
    }
  ],
  "count": 1
}
```

---

### 11.10 POST /api/scan

Run a full contradiction scan across all rules.

**Request body:** empty (no body required)

**Response (200):**
```json
{
  "total_contradictions": 2,
  "resolved": 1,
  "flagged": 1,
  "skipped": 0
}
```

`resolved` -- contradictions where one rule was deprecated (confidence set to 0.0).
`flagged` -- contradictions where the confidence delta was too small to auto-resolve.
`skipped` -- contradictions where one or both rule IDs were not found.

---

### 11.11 POST /api/maintenance

Manually trigger a maintenance cycle (decay, prune, save).

**Request body:** empty

**Response (200):**
```json
{
  "status": "ok",
  "message": "Maintenance cycle completed"
}
```

---

### 11.12 GET /api/settings

Get current runtime settings.

**Response (200):**
```json
{
  "retriever_mode": "hybrid",
  "keyword_weight": 0.7,
  "semantic_weight": 0.3,
  "max_retrieval_results": 5,
  "min_retrieval_score": 0.1,
  "verifier_mode": "symbolic",
  "min_rule_confidence": 0.3,
  "contradiction_threshold": 0.75,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream_responses": true,
  "log_level": "info"
}
```

---

### 11.13 POST /api/settings

Update runtime settings. Partial JSON accepted -- only fields you include are changed.

**Request body (any subset of settings fields):**
```json
{
  "retriever_mode": "keyword",
  "temperature": 0.8,
  "log_level": "debug"
}
```

**Response (200):** Full updated settings object (same format as GET /api/settings).

**Error responses:**
- 400 -- invalid field value (e.g. temperature out of range, unknown key)

---

### 11.14 POST /api/export

Export training data to a JSONL file.

**Request body:**
```json
{
  "output_path": "D:/cardinal/data/training_export.jsonl",
  "min_confidence": 0.7,
  "domain": "",
  "max_examples": 0,
  "include_rules": true,
  "dry_run": false
}
```

All fields are optional. Defaults: `output_path = "D:/cardinal/data/training_export.jsonl"`, `min_confidence = 0.7`, `include_rules = true`, `dry_run = false`.

Set `"dry_run": true` to preview without writing to disk.

**Response (200):**
```json
{
  "episodes_exported": 45,
  "rules_exported": 8,
  "total_exported": 53,
  "avg_confidence": 0.89,
  "output_path": "D:/cardinal/data/training_export.jsonl",
  "timestamp": "2026-04-11T14:22:03"
}
```

For dry runs, `output_path` is `"(dry run)"`.

---

## 12. Settings Manager

`SettingsManager` manages runtime-mutable settings. Changes propagate immediately to core components without restart.

**File:** `src/api/cardinal_settings.h/cpp`

### 12.1 Mutable Settings

| Setting | Type | Propagates to |
|---------|------|---------------|
| `retriever_mode` | string | `EpisodicRetriever::set_mode()` |
| `keyword_weight` | float | `EpisodicRetriever::set_weights()` |
| `semantic_weight` | float | `EpisodicRetriever::set_weights()` |
| `max_retrieval_results` | int | Used on next retrieval call |
| `min_retrieval_score` | float | Used on next retrieval call |
| `verifier_mode` | string | Validated, stored (full propagation in future) |
| `min_rule_confidence` | float | Validated, stored |
| `contradiction_threshold` | float | Validated, stored |
| `temperature` | float | Validated, stored (full propagation requires LLMEngine setter) |
| `top_p` | float | Validated, stored |
| `stream_responses` | bool | Used on next chat call |
| `log_level` | string | Validated, stored |

### 12.2 Validation

All settings are validated before any change is applied. If validation fails, no settings are changed -- the update is atomic.

### 12.3 Partial JSON Updates

`from_json()` starts from the current settings and applies only the fields present in the incoming JSON. This means a POST to `/api/settings` with one field only changes that one field -- all others remain unchanged.

### 12.4 Reset

`reset()` restores all settings to the values originally loaded from `config.json` at startup. This does not reload the config file -- it uses the values that were captured in the `SettingsManager` constructor.

---

## 13. Session Manager

`SessionManager` owns all active `ConversationSession` objects.

**File:** `src/api/session.h/cpp`

### 13.1 ConversationSession

Each session tracks:
- `session_id` -- unique identifier
- `turn_count` -- number of user/assistant pairs (incremented on user turn only)
- `history` -- `std::vector<ChatMessage>` for direct pipeline use
- `timestamps_` -- parallel vector of ISO 8601 timestamps for each history entry
- `created_at` -- ISO 8601 timestamp when session was created
- `last_active_at` -- ISO 8601 timestamp of last activity (updated on every add)

### 13.2 History Management

`add_user_turn(content)` appends a user message and increments `turn_count_`.

`add_assistant_turn(content)` appends an assistant message but does NOT increment `turn_count_` -- one turn is one user+assistant pair.

`get_history()` returns the raw `std::vector<ChatMessage>` used by `InferenceRequest`.

`get_chat_turns()` returns `std::vector<ChatTurn>` with role, content, and timestamp for each entry -- this is the API-boundary type used in `SessionInfo`.

### 13.3 Context Window Management

`estimated_token_count()` provides a rough estimate of how many tokens the history occupies (1 token per 4 characters).

`trim_to_token_budget(max_tokens, min_turns_to_keep)` removes the oldest history entries until the estimated token count is within budget. It always keeps at least `min_turns_to_keep` user/assistant pairs regardless of token count.

This is not currently called automatically -- it is available for interfaces to call when the context window is approaching its limit.

### 13.4 Thread Safety

`SessionManager` protects its session map with a `std::mutex` on all operations except `get()`. The `get()` method intentionally has no lock -- it is called from `CardinalAPI` which holds its own `session_mutex_` for the duration of any session access.

---

## 14. Type Reference

All types defined in `src/api/cardinal_types.h`. All types are pybind11-compatible: copyable, no raw pointers, `std::string` strings, `std::vector` collections.

### CardinalStatus

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
    INTERNAL_ERROR       = 99
};
```

`status_to_string(CardinalStatus)` converts to a human-readable string.

### CardinalResult\<T\>

```cpp
template<typename T>
struct CardinalResult {
    CardinalStatus status;
    std::string    error_message;
    T              value;

    bool ok() const;
    static CardinalResult<T> success(T val);
    static CardinalResult<T> failure(CardinalStatus s, const std::string& msg);
};
```

### CardinalVoidResult

```cpp
struct CardinalVoidResult {
    CardinalStatus status;
    std::string    error_message;

    bool ok() const;
    static CardinalVoidResult success();
    static CardinalVoidResult failure(CardinalStatus s, const std::string& msg);
};
```

### FeelingInfo

```cpp
struct FeelingInfo {
    float       confidence;
    std::string reasoning_type;
    std::string reasoning_domain;
    bool        uncertainty_flag;
    bool        contradiction_flag;
    bool        rule_candidate;
};
```

### ChatResponse

```cpp
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
```

### ChatTurn

```cpp
struct ChatTurn {
    std::string role;       // "user" or "assistant"
    std::string content;
    std::string timestamp;
};
```

### SessionInfo

```cpp
struct SessionInfo {
    std::string           session_id;
    int                   turn_count;
    std::vector<ChatTurn> history;
    std::string           created_at;
    std::string           last_active_at;
};
```

### RuleInfo

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
```

### EpisodeInfo

```cpp
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
```

### MemoryStats

```cpp
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
```

### VerifierStats

```cpp
struct VerifierStats {
    int total_checks;
    int total_rules_extracted;
    int total_contradictions;
    int total_resolved;
    int total_flagged;
    int total_maintenance_runs;
};
```

### SystemStats

```cpp
struct SystemStats {
    MemoryStats   memory;
    VerifierStats verifier;
    std::string   uptime_seconds;
    std::string   version;
    bool          initialized;
};
```

### ExportRequest

```cpp
struct ExportRequest {
    std::string output_path;
    float       min_confidence;   // default 0.7
    std::string domain;           // empty = all
    int         max_examples;     // 0 = no limit
    bool        include_rules;    // default true
};
```

### ExportInfo

```cpp
struct ExportInfo {
    int         episodes_exported;
    int         rules_exported;
    int         total_exported;
    float       avg_confidence;
    std::string output_path;
    std::string timestamp;
};
```

### ScanResult

```cpp
struct ScanResult {
    int total_contradictions;
    int resolved;
    int flagged;
    int skipped;
};
```

### StreamToken

```cpp
struct StreamToken {
    std::string session_id;
    std::string token;
    bool        is_final;
    FeelingInfo feeling;   // only populated when is_final is true
};
```

### ApiStreamCallback

```cpp
using ApiStreamCallback = std::function<bool(const StreamToken&)>;
```

Return `false` to abort generation early. Returning `false` causes the pipeline to stop generating tokens and return whatever was produced so far.

### CardinalSettings

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

---

## 15. Error Handling

### 15.1 API Boundary

No C++ exceptions cross the `CardinalAPI` boundary. Every method either returns `CardinalResult<T>` or `CardinalVoidResult`. All internal exceptions are caught in `CardinalAPI` methods and converted to failure results.

The pattern is:
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

### 15.2 Internal Error Types

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

### 15.3 Non-Fatal Paths

Some failures are intentionally non-fatal:

- **Retrieval failure** -- if the retriever throws during prompt injection, a warning is logged and inference continues without memory context
- **Neural verifier failure** -- if the neural verifier is unavailable or throws, the symbolic verifier result is used alone
- **Contradiction check failure** -- if the symbolic engine throws during contradiction checking, the failure is logged and the candidate rule proceeds without verification
- **JSONL migration parse errors** -- malformed lines in the JSONL file are skipped with a warning, not fatal
- **Rule store duplicate IDs** -- `INSERT OR IGNORE` in SQLite means duplicate episode IDs are silently skipped

### 15.4 Fatal Paths

These cause `init()` to return failure:
- Config file not found or invalid JSON
- Model file not found
- Grammar file not found
- SQLite database cannot be opened or created
- llama.cpp model fails to load

---

## 16. Module Reference

### 16.1 Logger

**File:** `src/utils/logger.h/cpp`

Thread-safe singleton logger with 6 levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL.

```cpp
// Initialize (called once at startup)
Logger::instance().init("path/to/cardinal.log");

// Macro API (preferred -- captures file and line)
LOG_TRACE("message");
LOG_DEBUG("message");
LOG_INFO("message");
LOG_WARN("message");
LOG_ERROR("message");
LOG_FATAL("message");
```

Output format: `[TIMESTAMP] [LEVEL] [file:line] message`

ANSI color codes are applied to console output. Color codes are not written to the log file.

Both file and console output happen on every log call. The minimum level is set by `logging.level` in config.

### 16.2 ConfigLoader

**File:** `src/utils/config_loader.h/cpp`

Static class. No instances. Load once at startup.

```cpp
auto config = ConfigLoader::load("config.json");     // throws ConfigError on failure
auto config = ConfigLoader::reload("config.json");   // same as load, for hot-reload
ConfigLoader::validate(config);                       // throws ConfigError if invalid
std::string json = ConfigLoader::to_json_string(config); // for debug logging
```

All validation runs in `validate()` which is called automatically by `load()`. The validator checks type correctness, range constraints, file existence (for paths that must exist at startup), and logical consistency.

### 16.3 JsonParser

**File:** `src/utils/json_parser.h/cpp`

Static utility class for all JSON serialization in the system.

```cpp
// Feeling output
FeelingOutput feeling = JsonParser::parse_feeling_output(json_str);
bool valid = JsonParser::validate_feeling_output(feeling, error_msg);
std::string json = JsonParser::serialize_feeling_output(feeling);

// Rules
std::vector<Rule> rules = JsonParser::load_rules(path);
JsonParser::save_rules(path, rules);
nlohmann::json j = JsonParser::rule_to_json(rule);
Rule rule = JsonParser::rule_from_json(j);

// Knowledge nodes
std::vector<KnowledgeNode> nodes = JsonParser::load_knowledge(path);
JsonParser::save_knowledge(path, nodes);

// Utilities
std::string id = JsonParser::generate_id();           // hex_timestamp_counter
std::string ts = JsonParser::current_timestamp();      // ISO 8601
bool valid = JsonParser::is_valid_json(str);
std::string pretty = JsonParser::pretty_print(str, 2);
```

**Atomic writes:** `save_rules()` and `save_knowledge()` write to a `.tmp` file then rename atomically. A crash during write leaves the old file intact.

**ID generation:** `generate_id()` uses `std::chrono::system_clock` for the timestamp component and a `std::atomic<uint32_t>` counter for uniqueness within the same millisecond. IDs are hex-encoded for compactness.

### 16.4 LLMEngine

**File:** `src/core/llm_engine.h/cpp`

Wraps llama.cpp. Manages two inference contexts (one for Pass 1, one for Pass 2) to prevent grammar state contamination between passes.

```cpp
LLMEngine engine(config);
engine.load_model();

// Pass 1 (grammar-constrained)
auto result = engine.generate_feeling(ctx, messages);

// Pass 2 (free decoding with streaming)
auto result = engine.generate_response(ctx, messages, token_cb);

engine.clear_kv_cache();
bool near_limit = engine.context_near_limit();
```

### 16.5 InferencePipeline

**File:** `src/core/inference.h/cpp`

Orchestrates the two-pass inference cycle. Manages retries, prompt construction, and memory context injection.

```cpp
InferencePipeline pipeline(config, engine);
pipeline.set_system_prompt("...");
pipeline.set_retriever(&retriever);     // optional

InferenceRequest req;
req.user_message    = "What is entropy?";
req.history         = session.get_history();
req.active_rules    = rule_store.get_top_rules("factual", 5);
req.stream_response = true;

InferenceResponse resp = pipeline.run(req, stream_callback);
```

`InferenceRequest.active_rules` contains rules that will be injected into the system prompt. These are formatted as a `## Active Rules` section.

If a retriever is set via `set_retriever()`, relevant past episodes are fetched and injected as memory context before each inference.

### 16.6 EpisodicRetriever

**File:** `src/memory/episodic_retriever.h/cpp`

```cpp
EpisodicRetriever retriever(config, storage);
retriever.init();                                        // builds initial TF-IDF index

auto results = retriever.retrieve("What is entropy?");  // uses config mode
auto results = retriever.retrieve("...", RetrievalMode::KEYWORD);  // mode override

retriever.notify_new_episode(ep_id);    // after each inference
retriever.rebuild_index();              // force rebuild

retriever.set_mode(RetrievalMode::HYBRID);
retriever.set_weights(0.7f, 0.3f);

int n = retriever.index_size();
int v = retriever.vocabulary_size();
bool ready = retriever.index_ready();
```

### 16.7 EpisodicStorage

**File:** `src/memory/episodic_storage.h/cpp`

```cpp
EpisodicStorage storage(config);
storage.open();     // creates schema, runs migration if needed

bool inserted = storage.insert_episode(record);      // INSERT OR IGNORE
bool updated  = storage.set_extracted_rule_id(ep_id, rule_id);
auto ep       = storage.get_episode(id);             // returns optional<EpisodeRecord>
auto results  = storage.query(q);                    // filtered query
auto recent   = storage.get_recent(10);
auto by_domain = storage.get_by_domain("factual", 20);
auto high_conf = storage.get_high_confidence(0.8f, 50);
int count     = storage.count();
auto stats    = storage.stats();
int migrated  = storage.migrate_from_jsonl(path);   // idempotent
bool done     = storage.migration_complete();

storage.close();    // WAL checkpoint + close
```

### 16.8 RuleExtractor

**File:** `src/verifier/rule_extractor.h/cpp`

```cpp
RuleExtractor extractor(config, rule_store, symbolic_engine);

ExtractionInput input;
input.feeling      = resp.feeling;
input.user_message = req.user_message;
input.response_text = resp.response;
input.episode_id   = ep_id;

ExtractionResult result = extractor.extract(input);
// result.extracted          -- was a candidate found?
// result.committed          -- was it written to rule store?
// result.contradiction_found -- did it contradict existing rules?
// result.committed_rule_id  -- ID if committed
// result.rejection_reason   -- why it was rejected

extractor.sync_rules_to_prolog();   // re-sync on startup
```

### 16.9 ConsistencyChecker

**File:** `src/verifier/consistency_check.h/cpp`

```cpp
ConsistencyChecker checker(config, rule_store, episodic,
                            symbolic, extractor, neural_verifier);
checker.init();     // syncs rules to Prolog

ConsistencyCheckInput input;
input.feeling      = resp.feeling;
input.user_message = req.user_message;
input.response_text = resp.response;
input.episode_id   = ep_id;

ConsistencyCheckResult result = checker.check(input);
// result.rule_extracted          -- was a rule candidate found?
// result.rule_committed          -- was it committed?
// result.committed_rule_id       -- ID if committed
// result.contradiction_found     -- was a contradiction detected?
// result.contradictions          -- vector of ContradictionResult
// result.contradictions_resolved -- how many were auto-resolved
// result.contradictions_flagged  -- how many were flagged for review
// result.periodic_sweep_ran      -- did maintenance run this cycle?
// result.summary                 -- human-readable summary string

auto contradictions = checker.run_full_scan();
int removed = checker.run_maintenance();

checker.total_checks();
checker.total_rules_extracted();
checker.total_contradictions();
checker.total_resolved();
checker.total_flagged();
checker.total_maintenance_runs();
```

### 16.10 TrainingExporter

**File:** `src/learning/training_exporter.h/cpp`

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

### 16.11 HttpServer

**File:** `src/api/http_server.h/cpp`

```cpp
HttpServer server(api, config);

// Start -- blocks until stop() is called
// Run in a thread for non-blocking behavior:
std::thread t([&]{ server.start(); });

server.stop();   // safe to call from any thread
bool running = server.is_running();

server.host();   // "127.0.0.1"
server.port();   // 8080
```

---

## 17. Data Formats

### 17.1 JSONL Episode Format

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

Note that `response_summary` in the JSONL contains the raw model output including `<think>` blocks. The SQLite layer stores the same raw text. The training exporter strips these blocks before writing to training data.

### 17.2 Rule Store Format

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

Rules created before Phase 6 have empty `episode_id` and `reasoning_type`. These fields are written with empty string defaults when loading and resaving legacy rules.

### 17.3 Knowledge Graph Format

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

### 17.4 Training Export Format

`data/training_export.jsonl` -- Alpaca format, one JSON object per line:

```json
{"instruction": "What happens to gas molecules when temperature increases?", "input": "", "output": "When temperature increases, gas molecules gain kinetic energy and move faster, increasing pressure and collision frequency."}
{"instruction": "What rule applies when: gas temperature increases?", "input": "", "output": "Gas molecules move faster and pressure increases."}
```

### 17.5 SQLite Schema

See Section 4.4 for the full SQLite schema.

The `metadata` table contains:

| Key | Value | Description |
|-----|-------|-------------|
| `migration_v1_complete` | `"1"` | Set after JSONL migration completes |
| `migration_v1_count` | `"41"` | Number of episodes migrated |

---

## 18. Threading Model

### 18.1 Component Thread Safety

| Component | Thread safety mechanism |
|-----------|------------------------|
| `Logger` | Singleton with internal mutex |
| `RuleStore` | `std::mutex` on all operations |
| `EpisodicStorage` | `std::mutex` via `SQLITE_OPEN_FULLMUTEX` + member mutex |
| `EpisodicRetriever` | `std::shared_mutex` (shared reads, exclusive writes) |
| `SettingsManager` | `std::shared_mutex` (shared reads, exclusive writes) |
| `SessionManager` | `std::mutex` on map operations |
| `ConsistencyChecker` | `std::mutex` on check() |
| `CardinalAPI` | Multiple mutexes (see below) |

### 18.2 CardinalAPI Mutexes

`CardinalAPI` uses three mutexes:

**`inference_mutex_`** (`std::mutex`) -- Serializes all inference operations. Only one inference runs at a time across all sessions. This prevents GPU memory exhaustion and KV cache corruption.

**`session_mutex_`** (`std::shared_mutex`) -- Protects the session map. Reads (get, list) use shared lock. Writes (create, destroy) use exclusive lock.

**`api_mutex_`** (`std::mutex`) -- Protects init/shutdown. Prevents concurrent initialization or double-shutdown.

### 18.3 HTTP Server Threading

`cpp-httplib` handles one request at a time by default (synchronous mode). Each incoming HTTP request is processed on the httplib server thread. The request handler calls into `CardinalAPI` which acquires the appropriate mutex.

For `/api/chat`, the inference mutex is held for the entire duration of the inference cycle. Concurrent HTTP chat requests will queue behind the mutex. This is intentional -- the GPU cannot handle concurrent inference.

For read-only endpoints (`/api/stats`, `/api/rules`, `/api/episodes`, `/api/settings`), the inference mutex is not held. These can be served concurrently with an in-progress inference.

### 18.4 HTTP Server Lifecycle

The HTTP server runs in its own thread, started explicitly via `server.start()` (which blocks) in a `std::thread`. The server is stopped via `server.stop()` which signals httplib to stop accepting new connections and finish current requests.

The interactive loop and HTTP server run concurrently -- the loop is on the main thread, the HTTP server is on a `std::thread`.

---

## 19. Lifecycle and Startup Sequence

### 19.1 Full Startup Sequence

```
main()
  |
  +-- Logger::instance().init()
  +-- CardinalAPI::init("config.json")
        |
        +-- ConfigLoader::load()          -- validates all fields
        +-- RuleStore(config)
        +-- KnowledgeGraph(config)
        +-- EpisodicMemory(config)
        +-- EpisodicStorage(config)
        +-- rule_store.load()             -- reads rules.json
        +-- knowledge_graph.load()        -- reads knowledge.json
        +-- episodic.open()               -- opens JSONL for appending
        +-- storage.open()                -- opens SQLite, runs migration if needed
        +-- EpisodicRetriever(config, storage)
        +-- retriever.init()              -- builds TF-IDF index from all episodes
        +-- SymbolicEngine(config)
        +-- symbolic.init(kb_path)        -- loads cardinal_kb.pl
        +-- RuleExtractor(config, rule_store, symbolic)
        +-- NeuralVerifier(config)
        +-- neural_verifier.load()        -- no-op if neural_model_path empty
        +-- ConsistencyChecker(config, ...)
        +-- checker.init()                -- sync_rules_to_prolog()
        +-- LLMEngine(config)
        +-- engine.load_model()           -- loads GGUF, offloads to GPU
        +-- InferencePipeline(config, engine)
        +-- pipeline.set_retriever()
        +-- TrainingExporter(config, storage, rule_store)
        +-- SettingsManager(config, retriever, pipeline)
        +-- SessionManager()
        +-- sessions.create()             -- creates initial default session
        +-- initialized = true
  |
  +-- HttpServer(api, config)
  +-- http_thread = thread(server.start)  -- if http_enabled
  |
  +-- interactive loop
```

### 19.2 Per-Inference Sequence

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
        |     +-- retriever.retrieve(user_message)   -- if retriever set
        |     +-- format_episodes()                   -- inject memory context
        |     +-- format_rules()                      -- inject active rules
        |
        +-- run_pass1() with retry
        |     +-- engine.generate_feeling()
        |     +-- JsonParser::parse_feeling_output()
        |     +-- validate_feeling_output()
        |
        +-- run_pass2()
              +-- engine.generate_response()         -- calls stream_cb per token
  |
  +-- run_post_inference()
        |
        +-- episodic.log_episode()                   -- JSONL write
        +-- storage.insert_episode()                 -- SQLite write
        +-- retriever.notify_new_episode()           -- rebuild check
        +-- checker.check()
              |
              +-- handle_rule_extraction()            -- if rule_candidate_signal
              +-- handle_contradiction_check()        -- if contradiction_flag
              +-- resolve_contradiction()             -- per contradiction found
              +-- run_periodic_maintenance()          -- every 10 inferences
        |
        +-- storage.set_extracted_rule_id()          -- if rule committed
        +-- rule_store.save()                        -- if dirty
  |
  +-- release inference_mutex_
  +-- update session history
  +-- return ChatResponse
```

### 19.3 Shutdown Sequence

```
CardinalAPI::shutdown()
  |
  +-- acquire api_mutex_
  +-- shutting_down = true
  +-- checker.run_maintenance()     -- final decay/prune/save
  +-- rule_store.save()             -- save any unsaved rules
  +-- sessions.destroy_all()        -- clear all session state
  +-- storage.close()               -- WAL checkpoint + close SQLite
  +-- initialized = false
  +-- shutting_down = false
```

---

## 20. Extending Cardinal

### 20.1 Adding a New Reasoning Domain

1. Add the domain string to the GBNF grammar's enum constraint in `feeling_schema.gbnf`
2. Add the domain to the `valid_domains` vector in `JsonParser::parse_feeling_output()`
3. Add the domain to the `validate()` function in `ConfigLoader` (if you add a config field for it)
4. Add the domain to `RuleStoreStats.rules_by_domain` array and the domain names array in `RuleStore::stats()`

### 20.2 Adding a New Reasoning Type

1. Add the type string to the GBNF grammar's enum constraint
2. Add it to the `valid_reasoning_types` vector in `JsonParser::parse_feeling_output()`
3. Add extraction handling in `RuleExtractor::extract_candidate()` if the new type warrants a specialized extraction strategy

### 20.3 Adding a New API Endpoint

1. Add a handler declaration in `http_server.h`
2. Implement the handler in `http_server.cpp`
3. Register the route in `register_routes()`
4. Add the corresponding method to `CardinalAPI` if new functionality is needed

### 20.4 Adding a New Setting

1. Add the field to `CardinalSettings` in `cardinal_settings.h`
2. Add initialization from config in `SettingsManager` constructor
3. Add the field to `validate()` in `cardinal_settings.cpp`
4. Add the field to `set()` (single-key update) in `cardinal_settings.cpp`
5. Add the field to `to_json()` and `from_json()` in `cardinal_settings.cpp`
6. Add propagation logic in `propagate()` in `cardinal_settings.cpp`
7. Add the field to `settings_to_json()` in `http_server.cpp`

### 20.5 Adding a Python Binding

All types in `cardinal_types.h` are pybind11-compatible by design. A minimal pybind11 module would look like:

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

### 20.6 Interface 2 (TypeScript Agent) Quick Start

The HTTP API is immediately usable from TypeScript:

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

*This documentation reflects Cardinal v1.0.0. The source of truth for any discrepancy is always the source code.*
