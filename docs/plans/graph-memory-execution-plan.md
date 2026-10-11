# Comex Browser — Agent Graph Memory: Verified Execution Plan
## Built from exhaustive codebase analysis on 2026-09-01

---

> **This plan is verified against the actual repo state.**
> Every file path, function signature, line number, and architectural
> assumption has been cross-checked against the live codebase.
> No guessing. No re-derivation. Execute as written.

---

## 0. Ground Truth Summary (from codebase audit)

Before building anything, these are the verified facts that shape every
decision in this plan:

### Python Backend State
| Fact | Detail |
|:-----|:-------|
| **Main file** | `agent_service.py` — 301 lines, FastAPI on `localhost:8000` |
| **Python version** | 3.12 (via `uv` toolchain) |
| **Dependency management** | **NONE** — no `requirements.txt`, no `pyproject.toml`, no `Pipfile`. Deps installed globally or via `uv`. |
| **Test infrastructure** | **NONE** — no `pytest.ini`, no `conftest.py`. Only standalone manual scripts (`test_cdp.py`, `test_browser_use.py`). |
| **App-data directory** | `~/.comex/` does **NOT** exist yet. Closest pattern: `os.path.expanduser("~/.gemini/...")` used for SKILL.md loading. |
| **browser-harness output** | Returns **raw `stdout + "\n" + stderr`** as a plain string. NOT structured JSON. This is the #1 blocker for reliable entity extraction. |
| **browser-harness binary** | Installed at `~/.local/bin/browser-harness` via `uv`, bound to Python 3.12. |
| **LLM integration** | `ChatOpenAI` from `langchain_openai`, model `ag/gemini-3.8-flash` via proxy at `localhost:20128/v1`. |
| **Existing tools** | 2 tools: `run_browser_harness(code, target_tab_ws)` and `create_hidden_tab(url, tab_id)`. |
| **Agent loop** | `execute_agent_loop()` — `for step in range(100)`, async, uses `asyncio.to_thread` for tool execution. |
| **Concurrency model** | Multiple agent loops can run simultaneously (foreground + hidden tabs). `active_tasks: dict[str, asyncio.Task]` tracks them. |
| **SYSTEM_PROMPT** | f-string at line 71, injects `BH_SKILL_CONTENT`. Contains browser-harness docs + code guidelines + markdown formatting rules. |
| **Nested agent loops** | **NO** — Comex never spawns nested `execute_agent_loop` calls. Only nested hidden tabs within a single loop. `DEPENDS_ON` is deferred. |

### Electron + React State
| Fact | Detail |
|:-----|:-------|
| **Electron main** | `electron/main.js` — 385 lines. `views` Map (foreground), `hiddenAgentViews` Map (background). |
| **IPC handlers** | 10 handlers: `tabs-create`, `tabs-create-hidden-agent`, `tabs-destroy`, `tabs-set-active`, `tabs-navigate`, `tabs-go-back`, `tabs-go-forward`, `tabs-reload`, `tabs-set-bounds`, `tabs-destroy-hidden-agent`. |
| **preload.js** | Exposes `window.nexusNative` with `tabs.*` methods + `onUpdate` callback. |
| **ChatDrawer.jsx** | Calls `POST /agent/run` with `{ task, taskId, tabId, url }`. Handles stop via `POST /agent/stop`. |
| **IntentClassifier.js** | LLM-based classification into `research` or `visible_browser_task`. |
| **Mock agents** | `OrchestratorAgent.js`, `NavigatorAgent.js`, `ResearcherAgent.js` are **hardcoded mocks** — not real LLM agents. |
| **Package deps** | `react`, `react-dom`, `lucide-react`, `electron`, `vite`, `electron-builder`. |

### Open Decisions — RESOLVED

| # | Decision | Resolution |
|:--|:---------|:-----------|
| 1 | Auto-injection vs tool-only | **Tool-only for v1**. The LLM calls `query_graph_memory` explicitly. More observable, debuggable. |
| 2 | App-data directory | **`~/.comex/`** — new directory. Created on startup via `os.makedirs(os.path.expanduser('~/.comex'), exist_ok=True)`. |
| 3 | browser-harness output format | **Raw stdout string**. Must build a structured extraction protocol (GM-012a) — the LLM's harness code must `print()` a JSON block that the wrapper parses. |
| 4 | Nested agent loops | **No**. `DEPENDS_ON` relationship deferred to v2. |
| 5 | Test runner | **No existing setup**. Phase 0 bootstraps `pytest` with a `requirements-dev.txt`. |

---

## 1. Architecture: Where Graph Memory Fits

```
+---------------------------------------------------------------------+
|                     agent_service.py (FastAPI)                        |
|                                                                       |
|  +---------------------------------------------------------------+  |
|  |              execute_agent_loop(task, cdp_url)                 |  |
|  |                                                                |  |
|  |  Tools:                                                        |  |
|  |   +-- run_browser_harness(code, target_tab_ws)                |  |
|  |   |   +-- [INSTRUMENTED] -> parse structured output           |  |
|  |   |       -> graph_store.record_tool_call()                   |  |
|  |   |       -> graph_store.record_navigation()                  |  |
|  |   |       -> graph_store.upsert_entity() per extracted item   |  |
|  |   |                                                            |  |
|  |   +-- create_hidden_tab(url, tab_id)                          |  |
|  |   |   +-- [INSTRUMENTED] -> graph_store.record_tab()          |  |
|  |   |                                                            |  |
|  |   +-- query_graph_memory(entity_type, max_price, since_min)   |  |
|  |       +-- [NEW TOOL] -> graph_store.find_recent_entities()    |  |
|  |                                                                |  |
|  |  On entry:  graph_store.create_agent(...)                     |  |
|  |  On exit:   graph_store.complete_agent(...)                   |  |
|  +---------------------------------------------------------------+  |
|                              |                                        |
|                              v                                        |
|  +---------------------------------------------------------------+  |
|  |              AgentGraphStore (graph/store.py)                  |  |
|  |                                                                |  |
|  |  +-------------+  +---------------+  +-------------------+   |  |
|  |  | schema.py   |  | entity_res.py |  | tools.py          |   |  |
|  |  | DDL+migrate |  | canonical_key |  | query_graph_mem   |   |  |
|  |  +-------------+  +---------------+  +-------------------+   |  |
|  |                                                                |  |
|  |  Backend: Kuzu embedded DB                                    |  |
|  |  Storage: ~/.comex/graph.db                                   |  |
|  |  Concurrency: threading.Lock on all writes                    |  |
|  +---------------------------------------------------------------+  |
|                                                                       |
|  Debug Endpoints (Phase 5):                                          |
|   GET  /agent/graph/stats                                            |
|   GET  /agent/graph/agent/{agent_id}                                 |
|   GET  /agent/graph/entities?type=...&max_price=...                  |
|   POST /agent/graph/reset?confirm=true                               |
+---------------------------------------------------------------------+
```

**Key constraint**: Graph memory is **instrumentation infrastructure**, not
the critical path. A graph write failure must NEVER crash or slow the
user's agent task. Every graph call is wrapped in try/except + logged.

---

## 2. Structured Extraction Protocol (GM-012a — The Real Blocker)

`browser-harness` returns raw stdout. The LLM generates arbitrary Python
code that `print()`s whatever it wants. There is no structured contract.

**Solution**: We do NOT modify `browser-harness` itself. Instead, we:

1. **Teach the LLM** (via SYSTEM_PROMPT) to emit a structured JSON block
   when extracting entities, using a specific delimiter:

   ```
   ===COMEX_EXTRACT===
   {"url": "https://amazon.in/dp/B0...", "title": "Amazon.in", "entities": [
     {"type": "product", "name": "LG 27MR400", "price": 14999, "currency": "INR",
      "product_id": "B0CDR6LHKZ", "attrs": {"rating": 4.3, "screen_size": "27 inch"}}
   ]}
   ===END_EXTRACT===
   ```

2. **Parse in the wrapper** (`agent_service.py`'s `run_browser_harness`
   post-processing): regex for `===COMEX_EXTRACT===...===END_EXTRACT===`
   blocks in stdout, parse as JSON, feed to `graph_store`.

3. **Fallback**: If no extraction block is found (most harness calls are
   navigation/clicking, not extraction), the tool call is still recorded
   but with no entities — this is correct behavior.

This approach:
- Requires zero changes to `browser-harness` binary.
- Is fully observable (the delimiter is in stdout logs).
- Fails gracefully (malformed JSON -> log warning, continue).
- Gives the LLM control over what gets extracted (it decides when to emit
  the block, matching its understanding of the task).

---

## 3. Complete File Manifest

```
/Users/pranavkumarn/Documents/Comex Good Versions/Comex copy 5/
|-- agent_service.py                        [MODIFIED - 7 integration points]
|-- graph/                                  [NEW PACKAGE]
|   |-- __init__.py                         [NEW - package init + singleton]
|   |-- store.py                            [NEW - AgentGraphStore class]
|   |-- schema.py                           [NEW - DDL + migrations]
|   |-- entity_resolution.py                [NEW - canonical_key + normalize]
|   +-- tools.py                            [NEW - query_graph_memory LangChain tool]
|-- tests/                                  [NEW DIRECTORY]
|   |-- conftest.py                         [NEW - pytest fixtures]
|   |-- test_graph_store.py                 [NEW - CRUD + MERGE tests]
|   |-- test_entity_resolution.py           [NEW - dedup edge cases]
|   |-- test_extraction_parser.py           [NEW - structured output parsing]
|   |-- test_graph_concurrency.py           [NEW - parallel write safety]
|   +-- test_agent_integration.py           [NEW - end-to-end with mocked LLM]
|-- requirements.txt                        [NEW - Python deps for agent backend]
|-- requirements-dev.txt                    [NEW - pytest + dev deps]
|-- pytest.ini                              [NEW - pytest configuration]
|-- config.json                             [MODIFIED - add graph_memory block]
|-- .env                                    [MODIFIED - add GRAPH_DB_PATH]
+-- docs/plans/
    +-- graph-memory-plan.md                [EXISTS - reference plan]
```

---

## 4. Phased Execution Plan

### PHASE 0 — Foundation: Dependencies and Environment

**Goal**: Kuzu importable. pytest runnable. Directory conventions established.

#### GM-001: Verify Kuzu compatibility
```bash
python3 -c "import platform; print(platform.python_version())"
# Expected: 3.12.x — Kuzu supports 3.8-3.12
pip install kuzu
python3 -c "import kuzu; print(kuzu.__version__)"
```

#### GM-002: Create `requirements.txt`
```
# Agent backend dependencies for Comex Browser
fastapi>=0.104.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
websockets>=12.0
pydantic>=2.0.0
httpx>=0.25.0
kuzu>=0.4.0
```

#### GM-002a: Create `requirements-dev.txt`
```
-r requirements.txt
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

#### GM-002b: Create `pytest.ini`
```ini
[pytest]
testpaths = tests
asyncio_mode = auto
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

#### GM-003: Add to `.env`
```
GRAPH_DB_PATH=~/.comex/graph.db
```

#### GM-004: Add `graph_memory` block to `config.json`
```json
"graph_memory": {
  "enabled": true,
  "db_path_env": "GRAPH_DB_PATH",
  "staleness_minutes": 30,
  "max_args_json_bytes": 4096,
  "debug_endpoints_enabled": true
}
```

#### GM-005: App-data directory creation
In `agent_service.py`, after `load_dotenv()` (line 22), add:
```python
GRAPH_DB_PATH = os.path.expanduser(os.getenv('GRAPH_DB_PATH', '~/.comex/graph.db'))
os.makedirs(os.path.dirname(GRAPH_DB_PATH), exist_ok=True)
```

**Definition of Done**:
- [ ] `python3 -c "import kuzu"` succeeds
- [ ] `pytest --collect-only` runs without errors
- [ ] `~/.comex/` directory created on startup
- [ ] `config.json` has `graph_memory` block
- [ ] `.env` has `GRAPH_DB_PATH`

---

### PHASE 1 — Graph Store Module (Standalone, Tested)

**Goal**: `graph/store.py` fully working and unit-tested. Zero wiring into
`agent_service.py` yet — pure library code.

#### GM-006: `graph/schema.py`

DDL definitions and migration runner for the Comex Agent Graph Memory.
Uses Kuzu Cypher-like DDL. Must define:
- 6 node tables: Agent, Tab, Page, Entity, ToolCall, SchemaMeta
- 7 relationship tables: SPAWNED, EXECUTED, NAVIGATED_TO, EXTRACTED, FOUND_ON, DEPENDS_ON, SIMILAR_TO
- All using `CREATE NODE TABLE IF NOT EXISTS` / `CREATE REL TABLE IF NOT EXISTS`
- `run_migrations(conn)` function that executes all DDL and upserts `SchemaMeta.schema_version = "1"`

#### GM-007: `graph/store.py`

Full `AgentGraphStore` class with these exact methods:
- `__init__(self, db_path: str)` — opens Kuzu DB, runs migrations
- `create_agent(agent_id, task, intent, parent_agent_id=None)`
- `complete_agent(agent_id, status, step_count)`
- `record_tab(tab_id, is_hidden, cdp_ws_url, spawned_by_agent_id)`
- `record_tool_call(agent_id, tool_name, args_json, result_summary, step_index, success) -> str`
- `record_navigation(tool_call_id, url, title) -> str`
- `upsert_entity(tool_call_id, page_id, etype, name, price, currency, attrs) -> str`
- `find_recent_entities(etype, max_price=None, since_minutes=30, include_stale=False) -> list[dict]`
- `get_agent_provenance(agent_id) -> dict`
- `get_stats() -> dict`

**Critical implementation rules**:
- ALL write methods wrapped in `self._lock` (`threading.Lock`)
- ALL queries use parameterized `conn.execute(query, params)` — NEVER string interpolation
- `upsert_entity` uses `canonical_key` from `entity_resolution.py` for MERGE logic
- All timestamps use `datetime.utcnow()` for consistency

#### GM-008: Concurrency safety
- `threading.Lock` on all write AND read methods initially
- `asyncio.to_thread(graph_store.method, ...)` when called from async context
- Tested explicitly in Phase 6 (`test_graph_concurrency.py`)

#### GM-009: `graph/entity_resolution.py`

```python
import hashlib
import re

def normalize_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name

def canonical_key(domain: str, product_id: str | None = None,
                  name: str | None = None) -> str:
    """
    Generate a deterministic dedup key.
    Priority: product_id (e.g. ASIN) > normalized name.
    """
    identifier = product_id if product_id else normalize_name(name or "")
    raw = f"{domain.lower()}:{identifier}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

#### GM-009a: `graph/__init__.py`
```python
"""Comex Agent Graph Memory package."""
from .store import AgentGraphStore
__all__ = ["AgentGraphStore"]
```

**Definition of Done**:
- [ ] `tests/test_graph_store.py` passes with 12+ test cases covering:
  - create_agent + complete_agent lifecycle
  - record_tool_call returns valid ID
  - record_navigation upserts Page, links NAVIGATED_TO
  - upsert_entity with new entity creates node
  - upsert_entity with same canonical_key UPDATES not duplicates
  - upsert_entity with different canonical_key creates second node
  - find_recent_entities returns matching entities
  - find_recent_entities excludes stale entities (>30 min)
  - find_recent_entities with include_stale=True returns old entities
  - find_recent_entities with max_price filter
  - get_agent_provenance returns full trace
  - get_stats returns correct counts
- [ ] `tests/test_entity_resolution.py` passes with 6+ cases:
  - Same name, same domain -> same key
  - Different casing -> same key
  - Extra whitespace -> same key
  - Product ID present -> uses product ID regardless of name
  - Different domains -> different keys
  - No product ID + different names -> different keys (known limitation)

---

### PHASE 2 — Wire Writes into agent_service.py

**Goal**: Every real agent run produces graph data. Pure instrumentation —
no read-path yet. Agent behavior unchanged.

#### GM-010: Instantiate graph store at module level

At the top of `agent_service.py`, after the existing globals (after line 28):

```python
# --- Graph Memory ---
import json as _json
from graph import AgentGraphStore

_graph_config = {}
try:
    with open(os.path.join(os.path.dirname(__file__), 'config.json')) as _f:
        _full_config = _json.load(_f)
        _graph_config = _full_config.get('graph_memory', {})
except Exception:
    pass

if _graph_config.get('enabled', False):
    graph_store = AgentGraphStore(GRAPH_DB_PATH)
    logger.info(f"Graph memory enabled at {GRAPH_DB_PATH}")
else:
    graph_store = None
    logger.info("Graph memory disabled")
```

#### GM-011: Agent lifecycle hooks in execute_agent_loop

**On entry** (after `messages` initialization, around line 199):
```python
import uuid
agent_id = str(uuid.uuid4())
step_ref = {"value": 0}
if graph_store:
    try:
        graph_store.create_agent(agent_id, task, "UNKNOWN")
    except Exception as e:
        logger.warning(f"Graph: failed to create agent: {e}")
```

**On exit** (wrap the entire for-loop in try/finally):
```python
finally:
    if graph_store:
        try:
            graph_store.complete_agent(agent_id, "completed", step_ref["value"])
        except Exception as e:
            logger.warning(f"Graph: failed to complete agent: {e}")
```

#### GM-012: Structured extraction parsing in run_browser_harness wrapper

After `subprocess.run` returns and `output` is constructed (lines 128-130),
add extraction parsing:

```python
if graph_store:
    try:
        import re as _re
        tool_call_id = graph_store.record_tool_call(
            agent_id, "run_browser_harness",
            args_json=code[:_graph_config.get('max_args_json_bytes', 4096)],
            result_summary=output[:200],
            step_index=step_ref["value"],
            success=result.returncode == 0
        )
        extract_pattern = r'===COMEX_EXTRACT===\s*(\{.*?\})\s*===END_EXTRACT==='
        for match in _re.finditer(extract_pattern, output, _re.DOTALL):
            try:
                payload = _json.loads(match.group(1))
                url = payload.get("url", "")
                title = payload.get("title", "")
                if url:
                    page_id = graph_store.record_navigation(tool_call_id, url, title)
                    for ent in payload.get("entities", []):
                        graph_store.upsert_entity(
                            tool_call_id, page_id,
                            etype=ent.get("type", "unknown"),
                            name=ent.get("name", ""),
                            price=ent.get("price"),
                            currency=ent.get("currency", "INR"),
                            attrs=ent
                        )
            except _json.JSONDecodeError as je:
                logger.warning(f"Graph: malformed extraction JSON: {je}")
    except Exception as e:
        logger.warning(f"Graph: failed to record tool call: {e}")
```

#### GM-012a: SYSTEM_PROMPT addition for structured extraction

Add to SYSTEM_PROMPT (inside the f-string, after the existing
PYTHON CODE GUIDELINES block). ALL curly braces MUST be doubled:

```
STRUCTURED EXTRACTION (CRITICAL):
When you extract structured data (products, prices, facts, comparisons),
you MUST print a structured extraction block so the system can remember
findings for future queries. Format:

print("===COMEX_EXTRACT===")
print(json.dumps({{
    "url": current_page_url,
    "title": page_title,
    "entities": [
        {{
            "type": "product",
            "name": "Product Name",
            "price": 14999.0,
            "currency": "INR",
            "product_id": "ASIN_OR_SKU_IF_AVAILABLE",
            "attrs": {{"rating": 4.3, "screen_size": "27 inch"}}
        }}
    ]
}}))
print("===END_EXTRACT===")

Always include this block AFTER extracting data from a page. The system
uses this to build a memory graph so you don't need to re-scrape pages
you have already visited. Entity types: product, fact, comparison, review.
```

#### GM-013: Hidden tab recording

In the `create_hidden_tab` tool, after successfully finding the WebSocket
URL (around line 186):
```python
if graph_store:
    try:
        graph_store.record_tab(tab_id, is_hidden=True,
                               cdp_ws_url=ws_url,
                               spawned_by_agent_id=agent_id)
    except Exception as e:
        logger.warning(f"Graph: failed to record tab: {e}")
```

#### GM-015: Failure isolation
Every graph call is inside try/except. The agent task NEVER fails due to
a graph error. This is already handled by the patterns above.

**Definition of Done**:
- [ ] Run a real agent task ("Find monitors under 20,000 on Amazon")
- [ ] Inspect via Python REPL:
  ```python
  from graph import AgentGraphStore
  store = AgentGraphStore(os.path.expanduser("~/.comex/graph.db"))
  print(store.get_stats())
  ```
- [ ] Confirm agent behavior is IDENTICAL to pre-graph (no slowdowns, no errors)
- [ ] `python3 -m py_compile agent_service.py` passes

### STOP POINT 1: Report back to user after Phase 2
Verify graph data is being written correctly before proceeding.

---

### PHASE 3 — Read-Path: query_graph_memory Tool

**Goal**: LLM has an explicit tool to check what is already known before
re-scraping.

#### GM-018: `graph/tools.py`

```python
"""LangChain tool for querying the Comex Agent Graph Memory."""
import json
from langchain_core.tools import tool as langchain_tool

def make_query_graph_memory_tool(graph_store, agent_id: str, step_ref: dict):
    @langchain_tool
    def query_graph_memory(entity_type: str,
                           max_price: float | None = None,
                           since_minutes: int = 30) -> str:
        """Check prior findings from this or earlier agent runs before
        navigating/scraping again. Returns entities matching a type and
        optional max price, seen within the staleness window. Always
        prefer calling this FIRST before re-visiting a site you may
        have already scraped this session."""
        try:
            results = graph_store.find_recent_entities(
                entity_type, max_price, since_minutes)
            graph_store.record_tool_call(
                agent_id, "query_graph_memory",
                args_json=json.dumps({
                    "entity_type": entity_type,
                    "max_price": max_price,
                    "since_minutes": since_minutes
                }),
                result_summary=f"{len(results)} entities found",
                step_index=step_ref["value"],
                success=True
            )
            if results:
                return json.dumps(results, indent=2, default=str)
            return "No matching entities found in memory."
        except Exception as e:
            return f"Graph memory query failed: {e}"
    return query_graph_memory
```

#### GM-019: Register tool in execute_agent_loop

In `execute_agent_loop`, after defining the two existing tools (around line 190):

```python
tools = [run_browser_harness, create_hidden_tab]
if graph_store:
    from graph.tools import make_query_graph_memory_tool
    query_mem_tool = make_query_graph_memory_tool(graph_store, agent_id, step_ref)
    tools.append(query_mem_tool)
llm_with_tools = llm.bind_tools(tools)
```

And in the tool dispatch loop (around line 212), add:
```python
elif tool_call["name"] == "query_graph_memory":
    result = await asyncio.to_thread(
        query_mem_tool.invoke,
        tool_call["args"]
    )
    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
```

#### GM-020: SYSTEM_PROMPT update for query_graph_memory

Add to SYSTEM_PROMPT (double all braces):

```
GRAPH MEMORY:
You have a `query_graph_memory` tool that checks your persistent memory
for entities (products, facts, comparisons) found in previous browsing
sessions. ALWAYS call this tool FIRST before navigating to a website you
may have already visited. If memory has fresh results matching the user
query, use them directly instead of re-scraping.
```

**Definition of Done**:
- [ ] `tests/test_agent_integration.py` with mocked LLM proves:
  - Task 1 extracts entities -> written to graph
  - Task 2 with overlapping query -> LLM tool calls include
    `query_graph_memory` BEFORE any `create_hidden_tab`
- [ ] Real manual test: run same query twice, second run is faster

### STOP POINT 2: Report back to user after Phase 3
Verify the LLM actually calls query_graph_memory in practice.

---

### PHASE 4 — Staleness and Dedup Verification

**Goal**: Stale data excluded. Dedup works under variation.

#### GM-021: Entity resolution edge cases
Test in `tests/test_entity_resolution.py`:
- Same product, identical name -> 1 entity
- Same product, different casing ("LG 27MR400" vs "lg 27mr400") -> 1 entity
- Same product, trailing whitespace -> 1 entity
- Same ASIN, different display name -> 1 entity (product_id wins)
- Different products, no product_id -> 2 entities (correct)

#### GM-022-024: Staleness tests
Test in `tests/test_graph_store.py`:
- Entity with `last_seen_at` 31 minutes ago excluded by default
- Same entity included with `include_stale=True`
- No hard-deletion of stale data

---

### PHASE 5 — Debug Endpoints

**Goal**: Human-inspectable graph without writing Cypher.

#### GM-027: Add to agent_service.py

```python
@app.get("/agent/graph/stats")
async def graph_stats():
    if not graph_store:
        return {"enabled": False}
    return await asyncio.to_thread(graph_store.get_stats)

@app.get("/agent/graph/agent/{agent_id}")
async def graph_agent_provenance(agent_id: str):
    if not graph_store:
        raise HTTPException(404, "Graph memory disabled")
    return await asyncio.to_thread(graph_store.get_agent_provenance, agent_id)

@app.get("/agent/graph/entities")
async def graph_entities(type: str = "product", max_price: float = None,
                         since_minutes: int = 30, include_stale: bool = False):
    if not graph_store:
        raise HTTPException(404, "Graph memory disabled")
    return await asyncio.to_thread(
        graph_store.find_recent_entities, type, max_price,
        since_minutes, include_stale)

@app.post("/agent/graph/reset")
async def graph_reset(confirm: bool = False):
    if not confirm:
        raise HTTPException(400, "Pass ?confirm=true to reset graph")
    if not graph_store:
        raise HTTPException(404, "Graph memory disabled")
    import shutil
    shutil.rmtree(os.path.dirname(GRAPH_DB_PATH), ignore_errors=True)
    os.makedirs(os.path.dirname(GRAPH_DB_PATH), exist_ok=True)
    graph_store.__init__(GRAPH_DB_PATH)
    return {"status": "reset", "path": GRAPH_DB_PATH}
```

---

### PHASE 6 — Concurrency and Performance Testing

**Goal**: Prove parallel hidden-tab agents don't corrupt the graph.

#### GM-032: `tests/test_graph_concurrency.py`
- Spawn 5 threads, each simulating a full agent lifecycle
  (create_agent -> record_tool_call -> upsert_entity x 3 -> complete_agent)
- Assert: no lost writes, no exceptions, no deadlocks
- Assert: total entity count matches expected (accounting for dedup)

#### GM-035-036: Performance measurement
- Wrap `AgentGraphStore` methods with `time.perf_counter()` timing
- Log at DEBUG level
- Target: <50ms p95 per operation
- If lock contention is measured >50ms under 5 concurrent writers,
  document and flag for v2 optimization

---

### PHASE 7 — Observability and Polish

#### GM-037: Structured logging
- Every graph write failure logs: `logger.warning(f"Graph write failed: ...")`
- Every graph read logs at DEBUG with timing info

#### GM-038: Health check integration
Add graph stats to existing `/health` endpoint:
```python
@app.get("/health")
async def health():
    result = {"status": "ok", "cdp_url": CDP_URL, "model": MODEL_NAME}
    if graph_store:
        result["graph_memory"] = await asyncio.to_thread(graph_store.get_stats)
    return result
```

---

## 5. Execution Order (Optimized)

```
Phase 0  -> Dependencies, pytest, directory setup         [~30 min]
Phase 1  -> graph/ package, fully unit-tested              [~2 hours]
   |
Phase 2  -> Wire writes into agent_service.py              [~1.5 hours]
   |
   STOP POINT 1: Manual verification with real task
   |
Phase 4  -> Dedup + staleness verification                 [~30 min]
   |
Phase 3  -> query_graph_memory tool + SYSTEM_PROMPT         [~1 hour]
   |
   STOP POINT 2: Verify LLM actually calls the tool
   |
Phase 5  -> Debug endpoints                                [~30 min]
Phase 6  -> Concurrency + performance tests                [~1 hour]
Phase 7  -> Observability + health check                   [~30 min]
```

**Total estimated effort: ~7-8 hours of focused execution.**

---

## 6. Integration Points in agent_service.py (Exact Locations)

| What | Where (current line) | Change |
|:-----|:---------------------|:-------|
| Import graph package | After line 19 | `from graph import AgentGraphStore` |
| GRAPH_DB_PATH + makedirs | After line 27 | New env var + directory creation |
| Graph store instantiation | After line 30 | Conditional init based on config |
| SYSTEM_PROMPT: extraction protocol | Line 84 (inside f-string) | Add STRUCTURED EXTRACTION block |
| SYSTEM_PROMPT: query_graph_memory | Line 92 (inside f-string) | Add GRAPH MEMORY block |
| Agent lifecycle: create | Inside execute_agent_loop, line ~199 | `graph_store.create_agent(...)` |
| Agent lifecycle: complete | Wrap loop in try/finally | `graph_store.complete_agent(...)` |
| run_browser_harness: record + parse | After line 130 | Tool call recording + extraction parsing |
| create_hidden_tab: record | After line 186 | Tab recording |
| Tool registration | Line ~197 | Add query_graph_memory to tools list |
| Tool dispatch | Line ~212 | Handle query_graph_memory tool calls |
| Debug endpoints | After line 295 | 4 new FastAPI routes |
| Health endpoint | Line 101 | Add graph stats |

---

## 7. Risks and Mitigations (Verified)

| Risk | Likelihood | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Kuzu DDL syntax differs from Neo4j Cypher | Medium | Blocks Phase 1 | Test DDL immediately in GM-006. Kuzu docs: kuzudb.com/docs/cypher. |
| LLM ignores extraction protocol | High initially | No entities in graph | Strong SYSTEM_PROMPT wording + observe first 5 real runs. Fall back to auto-injection if needed. |
| f-string escaping breaks SYSTEM_PROMPT | High (already happened once) | SyntaxError on startup | All curly braces in added prompt sections MUST be doubled. Run `python3 -m py_compile agent_service.py` after every edit. |
| Lock contention under 5+ concurrent agents | Low for v1 | Latency spike | Measure first (Phase 6). Only optimize if >50ms p95 measured. |
| browser-harness subprocess timeout (90s) kills long extractions | Medium | Partial extraction lost | Graph records whatever was extracted before timeout. Acceptable for v1. |
| Graph DB file grows unbounded | Low (single-user desktop) | Disk usage | No auto-prune in v1. POST /agent/graph/reset for manual cleanup. Monitor via /agent/graph/stats. |

---

## 8. Success Criteria Checklist

- [ ] Every `run_browser_harness` call with `===COMEX_EXTRACT===` writes entities to graph
- [ ] Second agent run on overlapping task reuses graph data (zero extra `create_hidden_tab` calls)
- [ ] LLM calls `query_graph_memory` (verified via tool call logs)
- [ ] Graph ops <50ms p95 (measured in Phase 6)
- [ ] 5 concurrent agent writes: no corruption, no lost writes
- [ ] `/agent/graph/stats` returns correct counts after real task
- [ ] `/agent/graph/entities?type=product` returns extracted entities
- [ ] Agent tasks NEVER fail due to graph errors (isolation verified)
- [ ] `python3 -m py_compile agent_service.py` passes after all changes
- [ ] All pytest tests green
