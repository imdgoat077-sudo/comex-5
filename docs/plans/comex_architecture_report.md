# Comex Architecture Report — Phase 0–10 Comprehension Pass

> **Audit Date:** 2026-09-02
> **Scope:** Documentation-only. No code modified. No Phase 11 design.
> **Source of Truth:** Actual code in the repository, cross-referenced against documentation.

---

## A. CURRENT ARCHITECTURE

Comex is a desktop AI browser built on **Electron + React (Vite)** with a **Python FastAPI backend** ([`agent_service.py`](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%205/agent_service.py)) that drives browser automation via CDP. The AI agent loop uses LangChain's `ChatOpenAI` (model `ag/gemini-3.8-flash` via local proxy at `localhost:20128/v1`).

**Process topology (unchanged from original design):**
```
Electron Main Process (main.js)
  ├── React Renderer (App.jsx, BrowserCanvas.jsx, ChatDrawer.jsx)
  ├── WebContentsView instances (foreground + hidden agent tabs)
  └── CDP at localhost:9222
          │
          ▼
Python FastAPI (agent_service.py, port 8000)
  ├── execute_agent_loop() — up to 100 LLM steps
  ├── Tools: run_browser_harness, create_hidden_tab, search_memory, save_artifact, query_graph_memory
  ├── Graph Memory (graph/ package → embedded Kuzu DB at ~/.comex/graph.db)
  └── Debug endpoints (/agent/graph/*)
```

The graph memory layer is **in-process** — no new service, no new port. It is connected via a module-level singleton `graph_store: Optional[AgentGraphStore]` governed by `config.json`'s `graph_memory.enabled` flag.

---

## B. PHASE-BY-PHASE EVOLUTION

### Phase 0 — Environment & Dependencies
- Added `kuzu` to `requirements.txt`, `pytest` to `requirements-dev.txt`
- Created `pytest.ini` with `asyncio_mode = auto`
- Added `GRAPH_DB_PATH=~/.comex/graph.db` to `.env`
- Added `graph_memory` config block to `config.json`
- Created `~/.comex/` directory on startup

### Phase 1 — Graph Store Module
- Created the `graph/` package: `schema.py`, `store.py`, `entity_resolution.py`, `__init__.py`
- Defined 7 node tables (Agent, Tab, Page, Entity, ToolCall, SchemaMeta, Artifact¹) and 8 relationship tables
- Implemented `AgentGraphStore` with full CRUD: `create_agent`, `complete_agent`, `record_tab`, `record_tool_call`, `record_navigation`, `upsert_entity`, `find_recent_entities`, `get_agent_provenance`, `get_stats`
- All methods protected by `threading.Lock`
- Entity deduplication via SHA-256 canonical key (`domain:product_id_or_normalized_name`)

¹ *Artifact was added in Phase 9 but lives in the same schema file.*

### Phase 2 — Write Instrumentation
- `agent_service.py` instantiates `AgentGraphStore` at module level
- Agent lifecycle hooks: `create_agent` on entry, `complete_agent` in `finally`
- `run_browser_harness` post-processing: records `ToolCall`, parses `===COMEX_EXTRACT===` delimiter blocks for structured entity extraction, records `Page` and `Entity` nodes
- `create_hidden_tab` records `Tab` node with `SPAWNED` relationship
- All graph writes wrapped in `try/except` — graph failure never crashes the agent

### Phase 3 — Read-Path: query_graph_memory Tool
- Decision: **tool-only** for v1, no automatic injection (reversed in Phase 10)
- `graph/tools.py`: `make_query_graph_memory_tool` — LangChain `@tool` wrapping `find_recent_entities(etype, max_price, since_minutes)`
- Registered alongside `run_browser_harness` and `create_hidden_tab` in the tool list
- System prompt updated to mention the tool

### Phase 4 — Staleness & Dedup Verification
- `find_recent_entities` default window: 30 minutes (from `config.json`)
- `include_stale=True` parameter to retrieve historical data
- Stale data is **excluded from default reads** but **never deleted**
- Entity resolution edge cases tested: identical name, different casing, whitespace, product_id priority

### Phase 5 — Debug Endpoints
- `GET /agent/graph/stats` — node/edge counts, DB file size
- `GET /agent/graph/agent/{agent_id}` — full provenance trace
- `GET /agent/graph/entities?type=...&max_price=...` — entity query
- `POST /agent/graph/reset?confirm=true` — destructive wipe (dev-only)
- All gated by `graph_memory.debug_endpoints_enabled` in concept (in practice, always available when `graph_store` is initialized)

### Phase 6 — Concurrency & Performance
- Tested with 5 concurrent simulated agent lifecycles
- Verified no lost writes, no deadlocks
- `threading.Lock` on all read and write operations
- Performance measured via `time.perf_counter()` around all store methods
- Logged at DEBUG level

### Phase 7 — Observability
- Structured `logger.warning` for all graph write failures
- `logger.debug` with timing for all graph reads
- `GET /health` enhanced: includes `graph_memory` stats when store is enabled
- Exception resilience: health endpoint survives graph stats failures

### Phase 8 — Unified Historical Memory Retrieval
- **FTS indices** created in `schema.py` via `CALL CREATE_FTS_INDEX(...)`:
  - `Agent_FTS` on `['task']`
  - `ToolCall_FTS` on `['tool_name', 'args_json', 'result_summary']`
  - `Entity_FTS` on `['type', 'name', 'attrs']`
  - `Artifact_FTS` (added Phase 9) on `['title', 'content', 'artifact_type']`
- `search_memory(query, domain, limit)` in `store.py`: unified BM25 text search across FTS indices
- Domain routing: `ENTITIES` → Entity_FTS; `HISTORY` → Agent_FTS + ToolCall_FTS; `ARTIFACTS` → Artifact_FTS; `ALL` → all four
- `query_graph_memory` preserved as a compatibility wrapper delegating to `find_recent_entities`
- `search_memory` tool registered in `tools.py` via `make_search_memory_tool`
- **Immutability guarantee**: `ToolCall.result_summary` is NEVER truncated in persistent storage; truncation only on retrieval output
- **Negative Memory Grounding Rule** added to system prompt

### Phase 9 — Artifact Memory
- New `Artifact` node table: `id`, `artifact_type`, `title`, `content`, `created_at`, `source_agent_id`
- New `CREATED_ARTIFACT` relationship: `Agent → Artifact`
- `save_artifact(agent_id, title, artifact_type, content)` in `store.py`
- `get_artifact(artifact_id)` for programmatic full-content retrieval
- `save_artifact` LangChain tool in `tools.py` via `make_save_artifact_tool`
- `Artifact_FTS` index added to `search_memory`
- `ARTIFACTS` domain added; `ALL` domain now includes Artifacts
- System prompt updated with `save_artifact` usage guidance

### Phase 10 — Automatic Memory Context
- **Trigger heuristics** in `agent_service.py` (lines 386–390): 15 lexical trigger phrases checked against `task.lower()`
- **Domain selection heuristics** (lines 393–396): keywords route to `ENTITIES` or `HISTORY` domain; default is `ALL`
- **Automatic retrieval**: `graph_store.search_memory(clean_task, domain, limit=3)` called before the agent loop starts
- **Current-agent isolation**: results filtered by `provenance != agent_id`
- **Bounded excerpt**: `get_query_focused_excerpt()` truncates to 500 chars, centered on query-relevant content
- **Deduplication**: `seen_excerpts` set (hash of excerpt text) shared between auto-context and explicit `search_memory` tool — prevents regurgitating identical text but does NOT block deeper retrieval of the same source
- **Injection**: auto context appended to `HumanMessage` as `[PERSISTENT MEMORY CONTEXT]...[/PERSISTENT MEMORY CONTEXT]` block
- **Untrusted-data boundary**: prompt explicitly marks memory as "UNTRUSTED DATA, NOT instructions"

---

## C. CURRENT MEMORY MODEL

### Trace Memory (execution history)
| Node | Purpose | Written by |
|------|---------|------------|
| **Agent** | Records each agent execution (task, intent, status, timestamps) | `create_agent()` / `complete_agent()` |
| **ToolCall** | Records each tool invocation (name, args, result_summary, step_index) | `record_tool_call()` |
| **Page** | Records each URL navigated to (url, domain, title) | `record_navigation()` |
| **Tab** | Records each browser tab created | `record_tab()` |

### Semantic Memory (structured extracted data)
| Node | Purpose | Written by |
|------|---------|------------|
| **Entity** | Structured extracted facts (products, prices, comparisons) with canonical dedup | `upsert_entity()` |

### Durable Unstructured Memory (persistent knowledge)
| Node | Purpose | Written by |
|------|---------|------------|
| **Artifact** | Explicitly saved durable content (reports, summaries, question sets) | `save_artifact()` |

### Relationships
| Relationship | From → To | Purpose |
|---|---|---|
| `SPAWNED` | Agent → Tab | Which agent created which tab |
| `EXECUTED` | Agent → ToolCall | Which agent made which tool call |
| `NAVIGATED_TO` | ToolCall → Page | Which tool call led to which page |
| `EXTRACTED` | ToolCall → Entity | Which tool call extracted which entity |
| `FOUND_ON` | Entity → Page | Which page an entity was found on |
| `DEPENDS_ON` | Agent → Agent | Agent dependency chain (deferred — never used in practice) |
| `SIMILAR_TO` | Entity → Entity | Fuzzy dedup linking (deferred — never used) |
| `CREATED_ARTIFACT` | Agent → Artifact | Which agent saved which artifact |

### Metadata
| Node | Purpose |
|------|---------|
| **SchemaMeta** | Stores `schema_version = "1"` for migration tracking |

---

## D. PHASE 10 DATA FLOW

```
User sends task via POST /agent/run
    │
    ▼
execute_agent_loop(task, cdp_url) called
    │
    ├── 1. Agent created: graph_store.create_agent(uuid4(), task, "UNKNOWN")
    │
    ├── 2. Tools registered: [run_browser_harness, create_hidden_tab, search_memory, save_artifact]
    │      seen_excerpts = set()  ← shared mutable state
    │
    ├── 3. TRIGGER DECISION (lexical):
    │      task_lower checked against 15 hardcoded phrases
    │      e.g., "previously", "earlier", "remember", "history"
    │      If NO match → skip automatic retrieval entirely
    │
    ├── 4. DOMAIN SELECTION (lexical):
    │      "monitor"/"product"/"price" → ENTITIES
    │      "what did i ask"/"task" → HISTORY
    │      default → ALL
    │
    ├── 5. AUTOMATIC search_memory(clean_task, domain, limit=3):
    │      ├── Kuzu FTS queries against relevant indices
    │      ├── BM25 scoring, sorted descending
    │      ├── Top 3 results selected
    │      ├── get_query_focused_excerpt() applied (500 char max, centered on query words)
    │      └── Current-agent results filtered (provenance != agent_id)
    │
    ├── 6. DEDUP TRACKING:
    │      Each auto-context excerpt hash → seen_excerpts set
    │
    ├── 7. CONTEXT CONSTRUCTION:
    │      [PERSISTENT MEMORY CONTEXT]
    │      Source ID, Memory Type, Timestamp, Provenance, Content
    │      [/PERSISTENT MEMORY CONTEXT]
    │
    ├── 8. MESSAGE ASSEMBLY:
    │      messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(task + auto_context_text)]
    │
    ├── 9. AGENT LOOP (up to 100 steps):
    │      LLM decides actions:
    │      ├── run_browser_harness → subprocess → extraction parsing → graph writes
    │      ├── create_hidden_tab → graph_store.record_tab()
    │      ├── search_memory (EXPLICIT) → graph_store.search_memory()
    │      │     └── Dedup: hash(excerpt) checked against seen_excerpts
    │      │         Same source, DIFFERENT excerpt → PASSES through (key fix)
    │      │         Same source, SAME excerpt → FILTERED
    │      ├── save_artifact → graph_store.save_artifact()
    │      └── query_graph_memory (legacy) → find_recent_entities()
    │
    └── 10. COMPLETION:
           graph_store.complete_agent(agent_id, "completed", step_count)
```

### AUTOMATIC vs EXPLICIT RETRIEVAL

| Property | Automatic | Explicit |
|----------|-----------|----------|
| **When** | Before agent loop starts | During agent loop, LLM-initiated |
| **Trigger** | Lexical phrase match | LLM tool call |
| **Limit** | 3 results | 10 results (default) |
| **Excerpt** | 500 chars, query-focused | 500 chars, query-focused |
| **Dedup** | Populates `seen_excerpts` | Reads `seen_excerpts`, filters exact duplicates |
| **Domain** | Heuristic-selected | LLM-specified |
| **Query** | Cleaned task text | LLM-chosen query |

Both exist because:
1. **Automatic** ensures the LLM has relevant context without needing to "know" to search — critical when the user references prior work implicitly
2. **Explicit** allows the LLM to do targeted, deeper retrieval with specific queries — the LLM controls the search terms and domain

---

## E. CURRENT RETRIEVAL MODEL

### search_memory(query, domain, limit)
**Location:** [`graph/store.py:457-555`](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%205/graph/store.py#L457-L555)

1. Selects FTS indices based on `domain`
2. Executes `CALL QUERY_FTS_INDEX(table, index, $query) YIELD node, score`
3. Constructs result dicts with `memory_type`, `source_id`, `timestamp`, `relevance_score`, `content_excerpt`, `provenance`
4. For Entity/ToolCall: follows relationships to find provenance (Agent ID)
5. Sorts all results by BM25 score descending
6. Truncates to `limit`
7. Applies `get_query_focused_excerpt()` post-retrieval (500 char window centered on first matching query word)

### find_recent_entities(etype, max_price, since_minutes, include_stale)
**Location:** [`graph/store.py:294-337`](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%205/graph/store.py#L294-L337)

Legacy Phase 3 read path. Direct Cypher query filtering by `type`, optional `max_price`, and staleness window. Returns full entity data (not truncated). Used by `query_graph_memory` compatibility wrapper.

### query_graph_memory (compatibility wrapper)
**Location:** [`graph/tools.py:56-93`](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%205/graph/tools.py#L56-L93)

Thin wrapper around `find_recent_entities`. Preserved for backward compatibility but **not registered as a tool in the tool list** (line 369–378 of `agent_service.py` — only `search_mem_tool` and `save_artifact_tool` are appended). However, it IS constructed and dispatched in the tool call handler (line 462–464). This means the LLM **could** call it if it knew about it, but it is NOT in the `bind_tools` list and therefore the LLM will never see it in its tool schema — it is effectively dead code for new agent runs, accessible only if the LLM somehow hallucinated the tool name.

> [!IMPORTANT]
> **DISCREPANCY:** `query_graph_memory` tool is constructed (line 374) and dispatch-handled (line 462-464), but **never appended to the `tools` list** that gets passed to `llm.bind_tools()`. It is functionally unreachable by the LLM under normal operation.

---

## F. WRITE PATH

### Entity Write Flow
```
run_browser_harness output
    → regex parse ===COMEX_EXTRACT=== blocks
    → JSON parse payload
    → record_tool_call(agent_id, "run_browser_harness", args, result_summary, step_index, success)
    → record_navigation(tool_call_id, url, title)  → MERGE Page node + NAVIGATED_TO edge
    → upsert_entity(tool_call_id, page_id, etype, name, price, currency, attrs)
        → canonical_key(domain, product_id, name)
        → MERGE Entity by id=ent_{canonical_key[:16]}
        → CREATE EXTRACTED edge (ToolCall → Entity)
        → CREATE FOUND_ON edge (Entity → Page)
```

### Artifact Write Flow
```
LLM calls save_artifact(title, artifact_type, content)
    → graph_store.save_artifact(agent_id, title, artifact_type, content)
        → CREATE Artifact node (id=art_{uuid4().hex})
        → CREATE CREATED_ARTIFACT edge (Agent → Artifact)
    → record_tool_call(agent_id, "save_artifact", args, result_summary)
```

### Agent Lifecycle Write Flow
```
execute_agent_loop entry → create_agent(uuid4(), task, "UNKNOWN")
execute_agent_loop exit  → complete_agent(agent_id, "completed", step_count)
```

---

## G. READ PATH

### Three read mechanisms:

1. **`search_memory`** (Phase 8+): BM25 full-text search across FTS indices. Returns ranked, truncated excerpts. Available as both auto-context and explicit tool.

2. **`find_recent_entities`** (Phase 3): Direct Cypher query with type/price/staleness filters. Returns full entity data. Available via `query_graph_memory` wrapper (unreachable by LLM).

3. **`get_agent_provenance`** (Phase 1): Full trace reconstruction for a specific agent. Used by debug endpoint only.

4. **`get_artifact`** (Phase 9): Programmatic full-content retrieval by artifact ID. NOT exposed as a tool.

5. **`get_stats`** (Phase 1): Aggregate counts. Used by `/health` and `/agent/graph/stats`.

---

## H. PROVENANCE / ISOLATION

### Provenance model:
- **Entity provenance**: traced via `EXTRACTED` edge back to `ToolCall`, then `EXECUTED` edge back to `Agent`
- **Artifact provenance**: traced via `CREATED_ARTIFACT` edge back to `Agent`
- **ToolCall provenance**: traced via `EXECUTED` edge back to `Agent`
- **Agent provenance**: `Agent.id` is the terminal provenance identifier

### Current-agent isolation:
- **Automatic context** (Phase 10): `filtered_results = [r for r in results if r.get("provenance") != agent_id]` — agent's own data excluded
- **Explicit search_memory tool**: No current-agent filtering. The tool returns ALL matching results regardless of provenance.

> [!WARNING]
> **DISCREPANCY:** Current-agent isolation is applied in **automatic context** (line 403) but NOT in **explicit search_memory** (tools.py). This means the LLM could explicitly search for and find its own in-progress data. This may be intentional (the LLM might need to recall what it just did), but it is inconsistent with the automatic context policy.

---

## I. DEDUPLICATION MODEL

### Entity deduplication (write-time):
- **Deterministic canonical key**: `sha256(domain:product_id_or_normalized_name)`
- `MERGE` on `Entity.id = ent_{canonical_key[:16]}` — ON MATCH updates price/attrs/last_seen_at
- Name normalization: lowercase, strip, collapse whitespace, remove punctuation
- Product ID takes priority over name

### Memory retrieval deduplication (read-time, Phase 10):
- `seen_excerpts: set[int]` — stores `hash()` of each excerpt string
- **Automatic context**: hashes of auto-injected excerpts added to `seen_excerpts`
- **Explicit search_memory**: each returned excerpt checked against `seen_excerpts`; if hash matches, filtered out; otherwise added
- **Key invariant**: same source with a DIFFERENT excerpt (e.g., different 500-char window) passes through. Only exact textual duplicates are filtered.

### Previous bug (fixed):
- Old approach used `seen_source_ids` (tracking by source UUID)
- This completely blocked deeper retrieval of ANY content from a source already seen in auto-context, even if the auto-context excerpt was only 500 chars of a 50,000-char artifact
- Fixed by switching to `seen_excerpts` (hash of actual text) — same source, different text window → passes through

---

## J. WHAT IS ACTUALLY PROVEN (Deterministic / Code-Level)

These properties are enforced by deterministic code and verified by tests that cannot pass due to mocking alone:

| Property | Evidence |
|----------|----------|
| Schema creates all 7 node tables and 8 relationship tables | `test_graph_store.py::test_all_6_node_and_7_rel_tables_exist` (name is stale — actually checks 7+8) |
| Agent lifecycle CRUD works | `test_graph_store.py::test_agent_lifecycle` |
| `upsert_entity` with same canonical key merges, not duplicates | `test_graph_store.py::test_upsert_entity_merge_updates` |
| Staleness exclusion works (31-min entity excluded) | `test_phase4_staleness_and_dedup.py::test_31_min_stale_entity_excluded_by_default` |
| Canonical key: casing/whitespace/product_id normalization | `test_entity_resolution.py` (7 tests) |
| Concurrent writes don't corrupt or deadlock | `test_graph_concurrency.py::test_concurrent_agent_lifecycles_and_dedup` |
| Operation p95 latency < 50ms | `test_graph_concurrency.py::test_operation_performance_latencies` |
| Graph write failure never crashes agent task | `test_agent_service_graph_integration.py::test_graph_store_exception_isolation` |
| Extraction parsing: valid JSON payload writes entities | `test_agent_service_graph_integration.py::test_extraction_parsing_valid_payload` |
| Extraction parsing: malformed JSON creates zero entities | `test_agent_service_graph_integration.py::test_malformed_extraction_payload` |
| Canonical dedup across repeated extractions | `test_agent_service_graph_integration.py::test_repeated_extraction_canonical_dedup` |
| FTS indices created and return results | `test_phase8_memory_retrieval.py::test_fts_index_creation_and_routing` |
| Domain routing returns correct memory types | Same test |
| Truncation applies post-retrieval, DB retains full text | `test_phase8_memory_retrieval.py::test_truncation_limits` |
| Artifact saved and retrievable via search_memory | `test_phase9_artifact_memory.py::test_distinction_trace_and_artifact` |
| Artifact distinct from HISTORY domain | Same test |
| Artifact failure isolation | `test_phase9_artifact_memory.py::test_failure_isolation` |
| Automatic context injected when trigger phrase matches | `test_phase10_automatic_context.py::test_auto_context_relevant_entity` and peers |
| No auto-context when no trigger phrase | `test_phase10_automatic_context.py::test_irrelevant_memory_control` |
| Auto-context bounded to ≤ 3 items | `test_phase10_automatic_context.py::test_bounded_context` |
| Provenance retained in auto-context | `test_phase10_automatic_context.py::test_provenance_retention` |
| Current-agent excluded from auto-context | `test_phase10_automatic_context.py::test_current_agent_isolation` |
| Dedup: exact excerpt not re-shown in explicit search | `test_phase10_automatic_context.py::test_duplicate_prevention` |
| Dedup: different excerpt from same source passes through | `test_phase10_dedup_deeper.py::test_dedup_deeper_retrieval` |
| Untrusted data boundary present in auto-context | `test_phase10_automatic_context.py::test_prompt_injection_boundary` |
| Debug endpoints respond correctly | `test_phase5_debug_endpoints.py` (9 tests) |
| Health endpoint includes graph stats | `test_phase7_observability.py` (5 tests) |
| Health endpoint survives graph stats exception | `test_phase7_observability.py::test_health_endpoint_stats_exception_resilience` |
| Graph read operations emit DEBUG timing logs | `test_phase7_observability.py::test_graph_read_debug_timing_logging` |
| Graph write failures emit WARNING logs | `test_phase7_observability.py::test_graph_write_failure_warning_logs` |

### TEST QUALITY AUDIT (85 tests across 13 files)

| File | Tests | LLM? | Type | Key Weakness |
|------|-------|------|------|-------------|
| `conftest.py` | 2 fixtures | — | Infrastructure | `ignore_errors=True` in rmtree may hide lock issues |
| `test_agent_service_graph_integration.py` | 5 | None | Deterministic | Extraction parsing logic duplicated in test instead of calling production code |
| `test_entity_resolution.py` | 7 | None | Deterministic | Missing Unicode/non-Latin tests |
| `test_graph_concurrency.py` | 2 | None | Deterministic | Thread-only (no multi-process); p95 assertion may be flaky in CI |
| `test_graph_store.py` | 16 | None | Deterministic | Some staleness tests use raw Cypher instead of injectable clock |
| `test_phase10_automatic_context.py` | 14 | Mocked | Deterministic | Hardcoded `~/.comex/` paths instead of `tempfile` |
| `test_phase10_dedup_deeper.py` | 1 | Mocked | Deterministic | Could be consolidated into phase10 test file |
| `test_phase3_graph_memory_tool.py` | 8 | Mixed | Deterministic | Tests `query_graph_memory` which is now dead code |
| `test_phase4_staleness_and_dedup.py` | 12 | None | Deterministic | Direct Cypher timestamp mutation |
| `test_phase5_debug_endpoints.py` | 9 | None | Deterministic | No negative/malformed param tests |
| `test_phase7_observability.py` | 5 | Mixed | Deterministic | Log assertions via substring only |
| `test_phase8_memory_retrieval.py` | 3 | None | Deterministic | No BM25 ranking order verification; no injection edge cases |
| `test_phase9_artifact_memory.py` | 3 | Mocked | Deterministic | Hardcoded `~/.comex/` paths |

---

## K. WHAT IS ONLY MODEL-DEPENDENT

These properties depend on LLM behavior and are NOT provable by deterministic tests:

| Property | Status | Reasoning |
|----------|--------|-----------|
| LLM actually calls `search_memory` when appropriate | **Assumed** | System prompt guides it, but no deterministic enforcement |
| LLM obeys `[PERSISTENT MEMORY CONTEXT]` as evidence, not instructions | **Assumed** | Prompt injection boundary is textual; model adherence is model-dependent |
| LLM emits `===COMEX_EXTRACT===` blocks during extraction | **Assumed** | Depends on LLM following system prompt instructions |
| LLM respects Negative Memory Grounding Rule | **Assumed** | Prompt says "never volunteer unrelated records" but enforcement is soft |
| LLM calls `save_artifact` for durable results | **Assumed** | Entirely dependent on LLM judgment |
| LLM prefers memory over re-scraping | **Assumed** | System prompt says "always prefer" but no enforcement |
| Auto-context improves task completion quality | **Unverified** | No production metrics, no A/B test |

---

## L. KNOWN LIMITATIONS

### Retrieval limitations:
1. **BM25 lexical search only** — no semantic embeddings, no vector similarity. "coding quiz" will NOT match "Programming Test"
2. **No semantic reranking** — raw BM25 scores determine ordering
3. **500-character excerpt limit** — large artifacts are only partially visible per retrieval call
4. **No cross-artifact aggregation** — cannot combine information from multiple artifacts in a single retrieval

### Trigger policy limitations:
5. **Hardcoded lexical trigger phrases** — 15 phrases checked via substring match; misses many natural patterns (e.g., "before", "showed me", "told me about")
6. **Domain selection heuristics** — crude keyword routing ("monitor" → ENTITIES, "task" → HISTORY); no understanding of intent
7. **Trigger phrases are all lowercase-compared** — correct behavior, but the phrase list itself is limited

### Memory model limitations:
8. **No Insight node type** — no way to store derived conclusions or learned patterns
9. **No UserPreference node type** — no way to remember user preferences across sessions
10. **No conversational memory** — only stores tool-call traces and explicit artifacts; not conversational turns
11. **No autonomous reflection/learning** — agent does not automatically extract lessons from its executions
12. **No planner memory** — no persistent memory of plans or strategies

### Provenance limitations:
13. **Agent intent always "UNKNOWN"** — `create_agent` is called with `intent="UNKNOWN"` (line 234); the intent classifier runs on the JS side and is never communicated to the Python backend
14. **`DEPENDS_ON` relationship deferred** — schema exists, code supports it, but never used because nested `execute_agent_loop` calls don't exist
15. **`SIMILAR_TO` relationship deferred** — schema exists, never populated

### Architecture limitations:
16. **Single shared Kuzu DB** — no per-session isolation; all agents share one graph
17. **`query_graph_memory` tool is dead code** — constructed and dispatch-handled but never registered in `bind_tools` → unreachable by LLM
18. **No `Page_FTS` index** — explicitly deferred in Phase 8 documentation
19. **Current-agent isolation inconsistent** — applied in auto-context but not in explicit `search_memory` tool
20. **`graph_store.get_artifact()` not exposed as a tool** — programmatic access only; LLM cannot retrieve full artifact content beyond 500-char excerpt

---

## M. DOCUMENTATION vs CODE DISCREPANCIES

| # | Document says | Code actually does | Severity |
|---|---|---|---|
| 1 | `graph-memory-plan.md` Phase numbers: Phase 8=Debug Endpoints, Phase 9=Frontend Explorer, Phase 10=Testing, Phase 11=Performance, Phase 12=Observability | Implementation renumbered: Phase 5=Debug, Phase 6=Concurrency, Phase 7=Observability, Phase 8=FTS/search_memory, Phase 9=Artifact, Phase 10=Auto-context | **Medium** — confusing but documented in execution plan |
| 2 | `graph-memory-plan.md` lists 6 node tables and 7 relationship tables | Code has 7 node tables (Artifact added in Phase 9) and 8 relationship tables (CREATED_ARTIFACT added) | **Low** — documented in phase9 doc |
| 3 | `graph-memory-execution-plan.md` lists `test_agent_integration.py` and `test_extraction_parser.py` | Actual file is `test_agent_service_graph_integration.py`; `test_extraction_parser.py` does not exist | **Low** — tests exist under different names |
| 4 | Phase 8 doc says `ALL = ENTITIES + HISTORY` | Code: `ALL = ENTITIES + HISTORY + ARTIFACTS` (post Phase 9) | **Low** — Phase 9 doc correctly updates this |
| 5 | Phase 9 doc says "64 tests pass" | Current suite has 85 tests | **Low** — count was correct at Phase 9 time |
| 6 | `graph-memory-plan.md` discusses `SIMILAR_TO` population via embedding similarity | Never implemented, never planned for implementation | **Informational** — explicitly deferred |
| 7 | `test_graph_store.py::test_all_6_node_and_7_rel_tables_exist` (test name) | Actually verifies 7 node tables and 8 rel tables (name stale after Phase 9) | **Low** — test logic is correct despite name |
| 8 | Execution plan says `query_graph_memory` is a "compatibility wrapper during migration" | Tool IS constructed and dispatched, but NOT in `bind_tools` list — effectively dead | **Medium** — should be documented or cleaned up |

---

## N. OPEN ARCHITECTURAL QUESTIONS

1. **Should `query_graph_memory` be removed?** It is currently dead code (not in `bind_tools`). The dispatch handler still processes it. This is legacy debt that should be explicitly resolved.

2. **Should explicit `search_memory` apply current-agent isolation?** Auto-context filters out `provenance == agent_id`, but explicit search does not. Is this intentional? If so, it should be documented. If not, it's a bug.

3. **Should `get_artifact` be exposed as a tool?** Currently, the LLM can only see 500-char excerpts of artifacts via `search_memory`. If the artifact is large, the LLM has no way to retrieve the full content. A `read_artifact(artifact_id)` tool would close this gap.

4. **Should the intent classifier result be communicated to the Python backend?** `Agent.intent` is always `"UNKNOWN"`. The JS `IntentClassifier.js` classifies intent but never passes it to `execute_agent_loop`.

5. **Should the trigger phrase list be extensible via config?** Currently hardcoded in `agent_service.py`. Moving to `config.json` would allow tuning without code changes.

6. **Is the `get_query_focused_excerpt` algorithm robust enough?** It finds the first query word (>3 chars) and centers a 500-char window. This could miss cases where the relevant content uses different terminology than the query.

7. **Should `Page_FTS` be implemented?** Deferred in Phase 8. If pages store meaningful content beyond URL/title, FTS on page data could be valuable.

8. **What happens when the Kuzu DB grows very large?** No archival, no pruning, no size limits. A single-user desktop app may not hit this limit soon, but there's no mitigation strategy.

---

I understand the current Comex architecture and Phase 10 boundary.
