# Comex Browser — Agent Graph Memory System
## Full Implementation Plan (for execution via antigravity cli)

---

## 0. How to Use This Document

This is a complete engineering plan, not a summary. It is written so that an
agentic coding CLI (antigravity cli or equivalent) can:

1. Read it end-to-end for full context.
2. Re-plan/refine the phase breakdown if it sees a better sequencing.
3. Execute phase by phase, checking off tasks by ID (`GM-XXX`).
4. Use the schema, function signatures, and file paths as ground truth —
   these are meant to be implemented as specified, not re-derived from scratch.

**Suggested first prompt to paste into antigravity cli**, after this file is
in the repo (e.g. at `docs/plans/graph-memory-plan.md`):

> Read docs/plans/graph-memory-plan.md in full. Re-validate the phase
> ordering and task breakdown against the current state of the repo
> (some files may have changed since this plan was written — check before
> assuming paths/signatures are accurate). Then execute Phase 0 and Phase 1,
> writing tests as you go. Stop and summarize before starting Phase 2.

Do not let the CLI skip straight to Phase 3+ without 0–2 done — later phases
assume the store and schema already exist and are tested.

---

## 1. Objective

Give every agent in Comex Browser (foreground `VISIBLE_BROWSER_TASK` runs and
background `RESEARCH` runs in `hiddenAgentViews`) a **shared, persistent,
queryable memory** of what has already been navigated, extracted, and
concluded — modeled as a graph — so that:

- Agents can reuse prior findings instead of re-scraping.
- Multi-hop questions ("everything found under ₹20k across any hidden tab
  this session") are graph traversals, not ad-hoc log parsing.
- Every extracted fact has provenance back to the exact agent step, tool
  call, and page it came from.
- Future features (agent-to-agent task dependencies, cross-session recall,
  a debug/graph-explorer UI) have a real data layer to build on instead of
  scraping the chat transcript.

### Success criteria (Definition of Done for the whole project)

- [ ] Every `run_browser_harness` call that extracts structured data writes
      it to the graph, not just to the LLM's returned Markdown.
- [ ] A second agent run on an overlapping task (same entity type, similar
      constraints, within a configurable staleness window) demonstrably
      reuses graph data — provable via a test that asserts zero additional
      `create_hidden_tab` calls for a repeated query.
- [ ] The LLM has a working `query_graph_memory` tool it actually calls
      (verified via `ToolCall` log inspection, not just "it's wired up").
- [ ] Graph writes/reads do not visibly slow down the agent loop (<50ms
      p95 per graph operation, measured).
- [ ] Concurrent hidden-tab agents can write to the graph without
      corruption or dropped writes (tested with ≥3 parallel agent runs).
- [ ] There's a way for a human to inspect the graph without writing Cypher
      (at minimum a debug endpoint; ideally a small panel).

---

## 2. Current System Context (do not re-derive, use as-is)

Recap of the existing architecture this plan integrates into:

- **Electron main process** (`main.js`): owns Chromium with CDP on
  `--remote-debugging-port=9222`; maintains `views` (visible tabs) and
  `hiddenAgentViews` (background agent tabs) as separate `WebContentsView`
  maps.
- **React renderer** (`App.jsx`, `BrowserCanvas.jsx`, `ChatDrawer.jsx`):
  tab state, bounds syncing, chat UI. Talks to the agent backend via
  `POST /agent/run` and `POST /agent/stop`.
- **Intent classifier** (`IntentClassifier.js`): routes each prompt to
  `RESEARCH` (background/hidden tab, doesn't steal the visible tab) or
  `VISIBLE_BROWSER_TASK` (foreground tab).
- **Python agent backend** (`agent_service.py`): FastAPI on `localhost:8000`.
  Runs `execute_agent_loop(task, default_cdp_url)` — a LangChain
  `ChatOpenAI` loop (model `ag/gemini-3.8-flash`, via local proxy at
  `localhost:20128/v1`), up to 100 steps, with two existing tools:
  - `run_browser_harness(code, target_tab_ws)` — executes LLM-generated
    Python against a CDP target via `browser-harness`.
  - `create_hidden_tab(url, tab_id)` — asks Electron (via CDP eval into the
    React UI) to spin up a background `WebContentsView`, resolves its
    WebSocket URL via `GET http://127.0.0.1:9222/json/list`.
- **`config.json`**: proxy endpoints, scoring, timeouts, blacklist rules,
  rate limits.
- **`.env`**: `OPENAI_BASE_URL`, `MODEL_NAME`, `CDP_URL`, `ROUTER_API_KEY`.

This plan adds a graph persistence layer **inside `agent_service.py`'s
process** — no new service, no new port, so it doesn't disturb the existing
process topology.

---

## 3. Scope

**In scope:**
- Embedded graph DB (Kuzu) inside the Python agent process.
- Schema for Agent / Tab / Page / Entity / ToolCall + relationships.
- Write-path instrumentation in `agent_service.py` and inside
  `run_browser_harness` extraction code.
- Read-path: pre-loop context retrieval + an explicit LangChain tool for the
  LLM to query memory mid-loop.
- Entity resolution/dedup, staleness handling.
- Debug/inspection endpoints.
- Tests (unit + integration + a basic concurrency test).

**Out of scope for this plan** (flag as future work, don't build now):
- Cross-device/cloud sync of the graph.
- Migrating to Neo4j/Memgraph (architecture should allow it later; not
  doing it now).
- A full graph-visualization UI panel (Phase 9 below is optional/stretch —
  do it last, only if time remains).
- Any change to the Electron/React bounds/view logic — untouched by this
  plan.

---

## 4. Technology Decision

**Chosen: [Kuzu](https://kuzudb.com/)** — embedded, in-process graph DB,
Cypher-like query language, single `pip install kuzu`, no server to run or
operate, file-backed at rest (survives app restarts).

Why not the alternatives:
- **Neo4j/Memgraph**: require running a separate server process — adds an
  op burden (another thing to start/monitor/crash-recover) disproportionate
  to a single-user desktop app. Revisit only if Comex ever becomes
  multi-user/cloud.
- **Plain Postgres/SQLite with JSON columns**: would work, but multi-hop
  queries ("entities found via tool calls made by agents whose task
  mentioned 'monitor', on pages visited in the last 30 min") become
  recursive CTEs or app-side joins. Not worth it when Kuzu gives us this
  natively and still ships as a single Python package.

Kuzu's query language is Cypher-compatible enough that migrating to Neo4j
later, if ever needed, is a connector swap, not a schema rewrite.

**Task GM-001**: Confirm Kuzu supports the target OS/Python version in this
repo's environment before committing (check Python version in
`agent_service.py`'s venv/requirements; Kuzu wheels currently target
CPython 3.8–3.12 on Linux/macOS/Windows — verify against what Comex ships).

---

## 5. Data Model

### 5.1 Node tables

```
Agent(
  id            STRING,       -- uuid4, PRIMARY KEY
  task          STRING,       -- raw user prompt
  intent        STRING,       -- 'RESEARCH' | 'VISIBLE_BROWSER_TASK'
  status        STRING,       -- 'running' | 'completed' | 'failed' | 'stopped'
  created_at    TIMESTAMP,
  completed_at  TIMESTAMP,    -- nullable
  step_count    INT64,        -- final step count when done
  parent_agent_id STRING      -- nullable, for DEPENDS_ON chains
)

Tab(
  id            STRING,       -- PRIMARY KEY, matches Electron's tab_id
  is_hidden     BOOLEAN,
  cdp_ws_url    STRING,
  created_at    TIMESTAMP,
  closed_at     TIMESTAMP     -- nullable
)

Page(
  id            STRING,       -- PRIMARY KEY, hash(url) or uuid4
  url           STRING,
  domain        STRING,       -- extracted host, indexed access pattern
  title         STRING,
  visited_at    TIMESTAMP,
  screenshot_hash STRING      -- nullable, optional dedup/debug aid
)

Entity(
  id            STRING,       -- PRIMARY KEY, see 5.3 Entity Resolution
  type          STRING,       -- 'product' | 'comparison' | 'fact' | ...
  name          STRING,
  price         DOUBLE,       -- nullable, NaN-safe handling required
  currency      STRING,       -- default 'INR', nullable
  attrs         STRING,       -- JSON-encoded blob for type-specific fields
  canonical_key STRING,       -- normalized dedup key, see 5.3
  first_seen_at TIMESTAMP,
  last_seen_at  TIMESTAMP
)

ToolCall(
  id            STRING,       -- PRIMARY KEY, uuid4
  tool_name     STRING,       -- 'run_browser_harness' | 'create_hidden_tab' | 'query_graph_memory'
  args_json     STRING,       -- JSON-encoded args (truncate large code blobs, see GM-014)
  result_summary STRING,      -- short summary, not full stdout
  step_index    INT64,
  timestamp     TIMESTAMP,
  success       BOOLEAN
)

SchemaMeta(
  key           STRING,       -- PRIMARY KEY, e.g. 'schema_version'
  value         STRING
)
```

### 5.2 Relationship tables

```
(Agent)  -[SPAWNED]->        (Tab)
(Agent)  -[EXECUTED]->       (ToolCall)
(ToolCall)-[NAVIGATED_TO]->  (Page)
(ToolCall)-[EXTRACTED]->     (Entity)
(Entity) -[FOUND_ON]->       (Page)
(Agent)  -[DEPENDS_ON]->     (Agent)      -- research chains / follow-ups
(Entity) -[SIMILAR_TO]->     (Entity)     -- post-hoc dedup links, not required at write time
```

Rationale for keeping `ToolCall` as its own node (vs. a property blob on
`Agent`): it's the unit of provenance. Every fact in the graph should be
traceable to "which tool call, at which step, in which agent run" — this is
what makes the debug endpoint (Phase 12) actually useful when the LLM
hallucinates a price and you need to know whether it came from a real
extraction or was fabricated in the final Markdown synthesis step.

### 5.3 Entity resolution (dedup key)

Do **not** dedup by fuzzy name matching to start. Use a deterministic
canonical key:

```
canonical_key = sha256(f"{domain}:{product_id_or_normalized_name}").hexdigest()
```

Where `product_id_or_normalized_name` is, in priority order:
1. A site-specific product ID if `run_browser_harness` extraction code can
   pull one (e.g. Amazon ASIN from the URL or DOM) — most reliable.
2. Else, a normalized name: lowercase, strip whitespace/punctuation,
   collapse multiple spaces.

`MERGE` on `canonical_key` when upserting an `Entity` — update `price`,
`attrs`, `last_seen_at` on match; create fresh with `first_seen_at` set on
miss. Fuzzy `SIMILAR_TO` linking (e.g. via embedding similarity) is a
Phase-13-and-beyond stretch goal — explicitly not required for v1.

### 5.4 Staleness

- `Page.visited_at` and `Entity.last_seen_at` are the staleness signals.
- Default staleness window: **30 minutes**, configurable via
  `config.json` (`graph_memory.staleness_minutes`).
- Any read-path query (Phase 4/5) must filter
  `last_seen_at > now() - staleness_window` unless the caller explicitly
  asks for historical data.

---

## 6. New Files & Directory Layout

```
agent-backend/                          (wherever agent_service.py lives)
├── agent_service.py                    [MODIFIED]
├── graph/
│   ├── __init__.py                     [NEW]
│   ├── store.py                        [NEW]  -- AgentGraphStore class
│   ├── schema.py                       [NEW]  -- DDL + migrations
│   ├── entity_resolution.py            [NEW]  -- canonical_key logic
│   └── tools.py                        [NEW]  -- LangChain tool: query_graph_memory
├── tests/
│   ├── test_graph_store.py             [NEW]
│   ├── test_entity_resolution.py       [NEW]
│   ├── test_graph_concurrency.py       [NEW]
│   └── test_agent_loop_graph_integration.py  [NEW]
├── config.json                         [MODIFIED — add graph_memory block]
└── .env                                [MODIFIED — add GRAPH_DB_PATH]

src/components/                         (React side, Phase 9 only, optional)
└── GraphExplorer/
    └── GraphExplorerPanel.jsx          [NEW, stretch]
```

---

## 7. Phased Implementation Plan

Each phase lists: goal, tasks (with IDs), and a Definition of Done. Execute
in order — later phases assume earlier ones are done and tested.

### Phase 0 — Environment & Dependencies

**Goal:** Kuzu installed and importable in the same environment
`agent_service.py` runs in.

- `GM-001` Verify Kuzu wheel compatibility with the repo's Python version
  (see §4).
- `GM-002` Add `kuzu` to the backend's dependency file (`requirements.txt`
  / `pyproject.toml` — check which this repo uses).
- `GM-003` Add `GRAPH_DB_PATH` to `.env` (default:
  `~/.comex/graph.db`, expanded via `os.path.expanduser`).
- `GM-004` Add a `graph_memory` config block to `config.json`:
  ```json
  "graph_memory": {
    "enabled": true,
    "db_path_env": "GRAPH_DB_PATH",
    "staleness_minutes": 30,
    "max_args_json_bytes": 4096,
    "debug_endpoints_enabled": true
  }
  ```
- `GM-005` Confirm `~/.comex/` directory creation is handled (create on
  startup if missing, same pattern as any existing app-data directory
  Comex already uses — check `main.js`/`agent_service.py` for existing
  convention before inventing a new one).

**DoD:** `python -c "import kuzu; print(kuzu.__version__)"` succeeds in the
backend's environment; `.env`/`config.json` changes committed.

---

### Phase 1 — Graph Store Module

**Goal:** `graph/store.py` with a tested `AgentGraphStore` class, no wiring
into `agent_service.py` yet.

- `GM-006` Implement `graph/schema.py`:
  - DDL for all node/relationship tables from §5.1/§5.2 using
    `CREATE NODE TABLE IF NOT EXISTS` / `CREATE REL TABLE IF NOT EXISTS`.
  - A `SchemaMeta` row written on init with `schema_version = "1"`.
  - A `run_migrations(conn)` function — even if it's a no-op today beyond
    v1, establish the pattern now (check `SchemaMeta.schema_version`,
    apply DDL diffs if behind). This avoids a painful retrofit later.

- `GM-007` Implement `graph/store.py`:

  ```python
  import kuzu
  import threading
  from datetime import datetime, timedelta

  class AgentGraphStore:
      def __init__(self, db_path: str):
          self.db = kuzu.Database(db_path)
          self.conn = kuzu.Connection(self.db)
          self._lock = threading.Lock()   # see GM-008 concurrency note
          from .schema import run_migrations
          run_migrations(self.conn)

      # --- Agent lifecycle ---
      def create_agent(self, agent_id: str, task: str, intent: str,
                        parent_agent_id: str | None = None) -> None: ...
      def complete_agent(self, agent_id: str, status: str,
                          step_count: int) -> None: ...

      # --- Tab ---
      def record_tab(self, tab_id: str, is_hidden: bool,
                      cdp_ws_url: str, spawned_by_agent_id: str) -> None: ...

      # --- ToolCall / Page ---
      def record_tool_call(self, agent_id: str, tool_name: str,
                            args_json: str, result_summary: str,
                            step_index: int, success: bool) -> str:
          """Returns the new ToolCall id."""
          ...
      def record_navigation(self, tool_call_id: str, url: str,
                             title: str) -> str:
          """Upserts a Page, links ToolCall -[NAVIGATED_TO]-> Page.
          Returns Page id."""
          ...

      # --- Entity ---
      def upsert_entity(self, tool_call_id: str, page_id: str,
                         etype: str, name: str, price: float | None,
                         currency: str, attrs: dict) -> str:
          """Canonical-key MERGE per §5.3. Links EXTRACTED + FOUND_ON.
          Returns Entity id."""
          ...

      # --- Read paths ---
      def find_recent_entities(self, etype: str, max_price: float | None,
                                since_minutes: int) -> list[dict]: ...
      def get_agent_provenance(self, agent_id: str) -> dict:
          """Full trace: tool calls -> pages -> entities, for debugging."""
          ...
  ```

  Fill in real Cypher-ish Kuzu queries per method — use parameterized
  queries (`self.conn.execute(query, params_dict)`), never string-format
  user/LLM-supplied values into the query text (injection risk, see
  `GM-018`).

- `GM-008` **Concurrency**: Kuzu's Python `Connection` is not guaranteed
  thread/async-safe for concurrent writers. Since `agent_service.py` is
  `async` FastAPI and multiple agent loops (visible + several hidden-tab
  RESEARCH agents) may run concurrently:
  - Wrap all write methods in `self._lock` (a plain `threading.Lock`,
    acquired via `asyncio.to_thread` if calling from async code, or a
    dedicated single writer thread with a queue if lock contention proves
    to be a bottleneck).
  - Reads can be less strict, but start with the same lock for correctness
    first, optimize later if profiling (Phase 11) shows it's a real
    bottleneck.
  - Write `test_graph_concurrency.py` (Phase 10) before considering this
    done — don't just assume the lock works.

- `GM-009` Implement `graph/entity_resolution.py`:
  ```python
  def canonical_key(domain: str, product_id: str | None,
                     name: str | None) -> str: ...
  def normalize_name(name: str) -> str: ...
  ```

**DoD:** `graph/store.py` fully implemented, `tests/test_graph_store.py`
passes covering: create_agent, record_tool_call, record_navigation,
upsert_entity (including a MERGE-updates-not-duplicates case), and
find_recent_entities (including a staleness-window exclusion case).

---

### Phase 2 — Wiring Writes into `agent_service.py`

**Goal:** Every real agent run produces graph data, without changing agent
behavior yet (pure instrumentation — no read-path reuse this phase).

- `GM-010` Instantiate a single module-level `AgentGraphStore` in
  `agent_service.py` at startup (respecting `graph_memory.enabled` from
  config — if false, use a no-op stub implementing the same interface, so
  call sites don't need `if` checks scattered everywhere).

- `GM-011` In `execute_agent_loop`, on entry:
  `agent_id = graph_store.create_agent(uuid4(), task, intent, parent_agent_id=None)`.
  On loop exit (success/failure/stop), call `complete_agent`.

- `GM-012` Wrap the existing `run_browser_harness` tool function: after the
  subprocess call returns, if the harness's stdout contains structured
  extraction output (check how `browser-harness` currently returns
  extracted data — likely JSON on stdout or a return value; **inspect the
  actual current implementation before assuming the format**), parse it
  and call `graph_store.record_tool_call` + `record_navigation` +
  `upsert_entity` for each extracted item.
  - If `browser-harness` currently does *not* return structured
    JSON-friendly output (e.g. it's raw scraped text/Markdown), that's a
    prerequisite gap — file it as `GM-012a`: extend the harness's
    extraction contract to emit a structured
    `{"url":..., "title":..., "entities": [...]}` shape, since without
    this the graph has nothing reliable to write. Do not attempt to
    regex-parse Markdown for prices — too fragile.

- `GM-013` Wrap `create_hidden_tab`: call `graph_store.record_tab` when a
  hidden tab is created, linking `Agent -[SPAWNED]-> Tab`.

- `GM-014` Guard against oversized writes: truncate `args_json` (the LLM's
  generated Python code can be long) to `graph_memory.max_args_json_bytes`
  before storing — store a hash of the full code separately if full
  reproducibility is wanted later (stretch, not required for v1).

- `GM-015` Every graph write call must be wrapped in try/except that logs
  and continues — **a graph write failure must never fail the user's agent
  task**. This is memory/observability infrastructure, not the critical
  path.

**DoD:** Run one real end-to-end agent task manually (e.g. the "27-inch
monitor under ₹20,000" example from the existing docs) and confirm via a
debug query (Phase 12) that `Agent`, `ToolCall`, `Page`, and `Entity` nodes
were created with correct relationships.

---

### Phase 3 — Read-Path: Pre-Loop Context Injection

**Goal:** Before an agent starts working, give it a chance to know what's
already known — without requiring the LLM to think to ask.

- `GM-016` At the start of `execute_agent_loop`, after intent
  classification, run a lightweight heuristic (not a full LLM call) to
  extract a candidate `entity_type` + constraints from the task string —
  or, simpler for v1: skip heuristic extraction and let this be purely
  tool-driven (Phase 4's `query_graph_memory` tool) rather than
  automatic. **Decision needed from you** — see §12, Open Decision #1.
  Recommended default: **skip automatic injection for v1**, rely on the
  explicit tool (Phase 4) so the LLM's tool-use is observable/debuggable,
  and revisit automatic injection once you see how often the LLM actually
  calls the tool unprompted.
- `GM-017` If automatic injection is chosen instead: format retrieved
  entities as a `SystemMessage` addendum, capped at N items (avoid
  blowing the context window), clearly labeled as "prior findings, may be
  stale" so the LLM doesn't treat it as ground truth without verification.

**DoD:** Documented decision (§12 Decision #1 resolved) and, if automatic
injection chosen, a test showing a second run's message history contains
the injected context block.

---

### Phase 4 — LangChain Tool: `query_graph_memory`

**Goal:** Give the LLM an explicit, observable way to check memory
mid-loop.

- `GM-018` Implement `graph/tools.py`:
  ```python
  from langchain.tools import StructuredTool
  from pydantic import BaseModel, Field

  class QueryGraphMemoryArgs(BaseModel):
      entity_type: str = Field(..., description="e.g. 'product'")
      max_price: float | None = Field(None)
      since_minutes: int = Field(30, description="staleness window")

  def make_query_graph_memory_tool(graph_store, agent_id: str, step_ref: dict):
      def _run(entity_type: str, max_price: float | None = None,
               since_minutes: int = 30) -> str:
          results = graph_store.find_recent_entities(
              entity_type, max_price, since_minutes)
          graph_store.record_tool_call(
              agent_id, "query_graph_memory",
              args_json=json.dumps({"entity_type": entity_type,
                                     "max_price": max_price,
                                     "since_minutes": since_minutes}),
              result_summary=f"{len(results)} entities found",
              step_index=step_ref["value"], success=True)
          return json.dumps(results)  # structured, LLM parses/quotes as needed
      return StructuredTool.from_function(
          func=_run, name="query_graph_memory",
          description=(
              "Check prior findings from this or earlier agent runs before "
              "navigating/scraping again. Returns entities (e.g. products) "
              "matching a type and optional max price, seen within the "
              "staleness window. Always prefer this before re-scraping a "
              "site you may have already visited this session."
          ),
          args_schema=QueryGraphMemoryArgs)
  ```
  Note: all query params flow through Kuzu's parameterized execute — never
  string-concatenate `entity_type` etc. into the Cypher text (injection
  risk if the LLM is ever prompted or tricked into passing crafted
  strings).

- `GM-019` Register this tool alongside the two existing tools in
  `execute_agent_loop`'s tool list.
- `GM-020` Update the agent's system prompt to mention the tool exists and
  when to prefer it (the docstring above is a start, but the system prompt
  is what actually shapes behavior — check the current system prompt
  content in `agent_service.py` and add a line).

**DoD:** A test (`test_agent_loop_graph_integration.py`) that runs two
sequential tasks with overlapping constraints and asserts the second run's
`ToolCall` log contains a `query_graph_memory` call before any
`create_hidden_tab` call. (This test may require mocking the LLM to force
tool selection deterministically — that's fine, the point is proving the
plumbing works, not testing LLM judgment.)

---

### Phase 5 — Dedup Verification Under Load

**Goal:** Confirm `canonical_key` MERGE logic actually prevents duplicate
entities across realistic variation (different capitalization, trailing
whitespace, minor DOM text differences).

- `GM-021` Write `test_entity_resolution.py` cases: same product scraped
  twice with (a) identical name, (b) different casing, (c) extra
  whitespace, (d) same product ID but different name text (site A/B
  variant) — assert single `Entity` node in all cases where a product ID
  is available, and correct fallback behavior when it isn't.

**DoD:** All cases pass; document any known false-negative dedup cases
(e.g. no product ID + genuinely different name text) as a known limitation
rather than silently failing.

---

### Phase 6 — Staleness & TTL

**Goal:** Old data doesn't silently poison new agent decisions.

- `GM-022` Confirm `find_recent_entities` (already built in Phase 1)
  correctly excludes stale entities by default.
- `GM-023` Add an optional `include_stale: bool` param for cases where the
  LLM (or a human via debug endpoint) explicitly wants historical data.
- `GM-024` Add a background cleanup consideration: **do not hard-delete**
  stale data by default (provenance/debug value), just exclude from
  default reads. If DB size becomes a concern later, add an opt-in
  archival/prune job — not required for v1.

**DoD:** Test proving a 31-minute-old entity (with default 30-min window)
is excluded from `find_recent_entities` but present when
`include_stale=True`.

---

### Phase 7 — Multi-Agent / Hidden Tab Wiring

**Goal:** `DEPENDS_ON` and multi-hidden-tab research chains are properly
represented, not just flat independent Agent nodes.

- `GM-025` When `create_hidden_tab` is called *from within* an agent that
  itself was spawned to follow up on another agent's findings (i.e. a
  RESEARCH task that decomposes into sub-research), support an optional
  `parent_agent_id` passed into `create_agent`, linking
  `Agent -[DEPENDS_ON]-> Agent`.
- `GM-026` This likely requires a small change to how `agent_service.py`
  decides to spawn a *new* `execute_agent_loop` vs. just a new hidden tab
  within the *same* loop — inspect current behavior first. If Comex
  currently never spawns nested agent loops (only nested tabs within one
  loop), this phase may reduce to: skip `DEPENDS_ON` for v1, revisit if/when
  nested agent loops are added. **Check before building — don't add
  unused plumbing.**

**DoD:** Either `DEPENDS_ON` is wired and tested, or explicitly deferred
with a one-line note in the repo (not silently dropped).

---

### Phase 8 — Debug/Inspection Endpoints

**Goal:** A human can look inside the graph without writing Kuzu queries by
hand.

- `GM-027` Add to `agent_service.py` (guarded by
  `config.json`'s `graph_memory.debug_endpoints_enabled`):
  - `GET /agent/graph/stats` → node/edge counts per table, DB file size.
  - `GET /agent/graph/agent/{agent_id}` → full provenance trace via
    `get_agent_provenance` (Phase 1).
  - `GET /agent/graph/entities?type=product&max_price=20000` → thin wrapper
    over `find_recent_entities`, for manual spot-checking.
  - `POST /agent/graph/reset` → wipes the DB file, dev-only, should refuse
    to run unless an explicit `?confirm=true` query param is passed, and
    should probably be disabled entirely outside a dev/debug build flag.

**DoD:** All four endpoints return correct data against the manual test
run from Phase 2's DoD.

---

### Phase 9 — Frontend Graph Explorer (Stretch, do last)

**Goal:** Optional. Only build if Phases 0–8 are solid and tested.

- `GM-028` A minimal `GraphExplorerPanel.jsx` — a dev-only panel (e.g.
  toggled via a keyboard shortcut or hidden settings flag) that hits the
  Phase 8 debug endpoints and renders a simple list/tree view (not a full
  force-directed graph viz — that's a bigger investment than this project
  needs right now). Table/list rendering of "recent entities" +
  "agent provenance trace" covers 90% of the debugging value.
- `GM-029` If a visual graph is genuinely wanted later, that's a separate
  follow-up project (e.g. embedding a lightweight graph-viz lib) — flag it
  as future work, don't scope-creep it into this plan.

**DoD:** N/A unless undertaken — explicitly optional.

---

### Phase 10 — Testing Summary

Consolidate what's referenced throughout — this is the checklist to run
before calling the project done:

- `GM-030` `test_graph_store.py` — CRUD + MERGE correctness (Phase 1).
- `GM-031` `test_entity_resolution.py` — dedup edge cases (Phase 5).
- `GM-032` `test_graph_concurrency.py` — ≥3 concurrent simulated agent
  loops writing simultaneously; assert no lost writes, no corruption, no
  deadlock (Phase 1/GM-008).
- `GM-033` `test_agent_loop_graph_integration.py` — end-to-end with a
  mocked LLM: agent run → graph populated → second overlapping run reuses
  via `query_graph_memory` (Phase 4).
- `GM-034` Manual smoke test: real (non-mocked) run of the monitor-search
  example from the original architecture doc, followed by a repeat query,
  observed via Phase 8 debug endpoints.

**DoD:** All of the above green in CI (or local test run if no CI exists
yet — flag that gap if so).

---

### Phase 11 — Performance Pass

**Goal:** Confirm graph ops aren't a hidden bottleneck.

- `GM-035` Add lightweight timing around `AgentGraphStore` write/read
  methods (log at debug level, don't spam production logs).
- `GM-036` Run the concurrency test (`GM-032`) with timing captured;
  confirm p95 per-operation latency is acceptable (<50ms target from §1).
  If the global lock (`GM-008`) is the bottleneck under realistic
  concurrency (e.g. 5+ simultaneous hidden-tab agents), consider a
  single-writer-thread-with-queue pattern instead of a blocking lock — but
  only if measurement shows it's actually needed.

**DoD:** Numbers captured and documented; no action needed unless target
missed.

---

### Phase 12 — Observability

**Goal:** When something looks wrong in production use, it's diagnosable.

- `GM-037` Log (structured, not print) every graph write failure with
  enough context (agent_id, tool_call context) to debug without needing to
  reproduce.
- `GM-038` Surface `GET /agent/graph/stats` (already built in Phase 8) as
  something checked periodically during development — not automated
  monitoring, just make sure it's easy to glance at.

**DoD:** Deliberately light — this is "make sure future debugging is
possible," not a full observability stack.

---

## 8. File-by-File Change List

| File | Status | Change |
|---|---|---|
| `graph/__init__.py` | NEW | Package init |
| `graph/schema.py` | NEW | DDL + migration runner |
| `graph/store.py` | NEW | `AgentGraphStore` class |
| `graph/entity_resolution.py` | NEW | Canonical key logic |
| `graph/tools.py` | NEW | `query_graph_memory` LangChain tool |
| `agent_service.py` | MODIFIED | Instantiate store, wrap tool calls, add debug endpoints, register new tool, agent lifecycle hooks |
| `config.json` | MODIFIED | `graph_memory` block |
| `.env` | MODIFIED | `GRAPH_DB_PATH` |
| `tests/test_graph_store.py` | NEW | Store unit tests |
| `tests/test_entity_resolution.py` | NEW | Dedup tests |
| `tests/test_graph_concurrency.py` | NEW | Concurrency tests |
| `tests/test_agent_loop_graph_integration.py` | NEW | End-to-end tests |
| `src/components/GraphExplorer/GraphExplorerPanel.jsx` | NEW, stretch | Debug UI (Phase 9 only) |

---

## 9. Config Additions (reference)

`config.json`:
```json
{
  "graph_memory": {
    "enabled": true,
    "db_path_env": "GRAPH_DB_PATH",
    "staleness_minutes": 30,
    "max_args_json_bytes": 4096,
    "debug_endpoints_enabled": true
  }
}
```

`.env`:
```
GRAPH_DB_PATH=~/.comex/graph.db
```

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kuzu embedded DB concurrency issues under multiple simultaneous hidden-tab agents | Global write lock in v1 (`GM-008`), measured in Phase 11, escalate to single-writer-thread pattern only if measurement demands it |
| `browser-harness`'s current output format isn't structured enough to extract entities reliably | Flagged explicitly as `GM-012a` — extend the harness's output contract before Phase 2 can fully complete; don't regex-parse Markdown |
| Graph writes silently fail and nobody notices data isn't accumulating | `GM-015` (never fail the task) + `GM-037` (structured logging) + Phase 8 debug endpoints for periodic manual spot-checks |
| LLM never actually calls `query_graph_memory` even though it exists | System prompt update (`GM-020`); if usage stays low after real-world testing, revisit Phase 3's "automatic injection" path as the primary mechanism instead of relying on tool-choice |
| Entity dedup false negatives (same product, no ID, different name text) | Documented known limitation (`GM-021`); acceptable for v1, revisit with embedding-similarity `SIMILAR_TO` linking later if it matters in practice |
| Schema needs to change later and there's data already on disk | `run_migrations` pattern established from day one (`GM-006`), even though v1 has nothing to migrate from |

---

## 11. Open Decisions Needed From You

Before or during execution, these need a call — antigravity cli should
either ask you directly or pick the stated recommended default and flag it
clearly in its output so you can override:

1. **Automatic context injection vs. tool-only (Phase 3)** — Recommended
   default: tool-only for v1 (simpler, more observable). Revisit once you
   see real usage.
2. **Where does `~/.comex/` (or equivalent) already get created?** — Reuse
   the existing app-data-directory convention rather than inventing a new
   one; antigravity cli should grep the repo for this before Phase 0.
3. **What does `browser-harness`'s current extraction output actually look
   like?** — This blocks `GM-012` from being implemented correctly. Needs
   inspection of the real current output format before Phase 2 can be
   trusted as "done," not just "code written."
4. **Does Comex ever spawn nested `execute_agent_loop` calls today, or only
   nested hidden tabs within one loop?** — Determines whether Phase 7
   (`DEPENDS_ON`) is meaningful now or should be deferred.
5. **Is there an existing test runner/CI setup?** — If not, Phase 10 should
   include setting one up minimally (e.g. `pytest` + a GitHub Action or
   equivalent), not just writing test files nobody runs.

---

## 12. Suggested Execution Order (summary)

```
Phase 0  → environment/deps
Phase 1  → graph store, fully unit-tested, nothing wired in yet
Phase 2  → wire writes into agent_service.py (needs Decision #3 resolved)
Phase 5  → verify dedup under realistic variation (can run right after Phase 2)
Phase 6  → staleness/TTL correctness
Phase 3  → decide + implement context injection strategy (Decision #1)
Phase 4  → query_graph_memory tool
Phase 7  → multi-agent DEPENDS_ON (needs Decision #4 resolved, may be deferred)
Phase 8  → debug endpoints
Phase 10 → full test suite consolidation
Phase 11 → performance pass
Phase 12 → observability polish
Phase 9  → stretch, optional, last
```

Stop and report back to the user after Phase 2 and after Phase 4 — these
are the two points where "does this actually work as intended" should be
eyeballed on a real task before continuing, not just where tests pass.
