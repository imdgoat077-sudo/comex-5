# Comex Browser — Complete Architecture, Subsystems & Operational Context

> **Antigravity Agent Canonical Context Document**  
> This document serves as the authoritative, comprehensive context file for Antigravity and AI coding agents operating on the Comex project. It details the runtime topology, IPC contracts, coordinate system math, graph memory implementation, agent loop dynamics, empirical benchmark findings, known pitfalls, and identified code discrepancies.

---

## 1. Project Overview & System Topology

**Comex** is an AI-augmented desktop browser that combines:
1. An **Electron** native desktop shell managing Chromium windows and child views.
2. A **React 19 (Vite)** single-page application rendering the browser chrome, tabs, navigation bar, launchpad, and AI sidebar.
3. A **Python FastAPI** agent daemon (`agent_service.py`) executing autonomous web interactions over the **Chrome DevTools Protocol (CDP)** using `browser-harness` and an embedded **Kuzu Graph Database**.

### Process & Port Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Electron Main Process (electron/main.js)                            │
│   ├── CDP Remote Debugging Port: localhost:9222                        │
│   ├── Base WebContents: Hosts React UI (localhost:5173 / dist)         │
│   └── Native Child Views: WebContentsView (Active View & Hidden Tabs)  │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ IPC Bridge (nexusNative via preload.js)
┌───────────────────────────────────┴────────────────────────────────────┐
│ 2. React Renderer (src/)                                               │
│   ├── App.jsx: Master tab state, hotkeys, theme, window listeners      │
│   ├── BrowserCanvas.jsx: Dynamic WebContentsView bounds coordinator    │
│   ├── ChatDrawer.jsx: Dual-mode agent chat & "Brain" memory dashboard  │
│   └── IntentClassifier.js: LLM-based intent routing                    │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ HTTP / SSE (http://127.0.0.1:8000)
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. Python Agent Service (agent_service.py)                             │
│   ├── FastAPI running on port 8000                                     │
│   ├── LLM: ChatOpenAI (ag/gemini-3.8-flash via 9Router proxy)          │
│   ├── Subprocess: browser-harness (~/.local/bin/browser-harness)       │
│   └── Graph Memory: Embedded Kuzu DB (~/.comex/graph.db)               │
└────────────────────────────────────────────────────────────────────────┘
```

* **CDP Remote Debugging**: `http://127.0.0.1:9222` (launched via `app.commandLine.appendSwitch('remote-debugging-port', '9222')`).
* **Python Agent API**: `http://127.0.0.1:8000`.
* **LLM Proxy (9Router)**: `http://localhost:20128/v1` (OpenAI-compatible API format, model `ag/gemini-3.8-flash`).
* **React Dev Server**: `http://localhost:5173` (Vite).
* **Graph Database**: Embedded Kuzu DB files at `~/.comex/graph.db`.

---

## 2. Directory Structure & Key Files

```
├── electron/
│   ├── main.js                  # Electron main process; window & WebContentsView management, CDP switch
│   └── preload.js               # Context bridge exposing window.nexusNative to the React renderer
├── src/
│   ├── App.jsx                  # Master tab state, active tab sync, hotkeys, theme, landing prompts
│   ├── App.css                  # Layout styling
│   ├── main.jsx                 # React root entry point
│   ├── index.css                # Global CSS variables, reset, design tokens
│   ├── agents/
│   │   ├── IntentClassifier.js  # Semantic LLM intent classifier ('research' vs 'visible_browser_task')
│   │   ├── OrchestratorAgent.js # Legacy simulated multi-agent swarm orchestrator
│   │   ├── NavigatorAgent.js    # Sub-agent navigation simulation
│   │   ├── ResearcherAgent.js   # Sub-agent DOM extraction simulation
│   │   └── types.js             # Agent roles, status constants, log helpers
│   ├── components/
│   │   ├── Canvas/
│   │   │   └── BrowserCanvas.jsx# Layout bounds calculator for WebContentsView positioning
│   │   ├── Chat/
│   │   │   ├── ChatDrawer.jsx   # AI sidebar: SSE streaming agent & "Brain" memory manager
│   │   │   ├── ChatDrawer.css   # Chat drawer styling
│   │   │   └── MarkdownRenderer.jsx # Markdown viewer with code highlighting & copy buttons
│   │   ├── Navbar/
│   │   │   └── BrowserNavbar.jsx# Address bar, tab strip, window controls, navigation buttons
│   │   ├── Bookmarks/
│   │   │   └── BookmarksBar.jsx # Quick access bookmarks
│   │   ├── Apps/                # Fallback non-electron mock apps (Launchpad, Google, GitHub, HN)
│   │   └── Modals/
│   │       └── ShortcutsModal.jsx # Keyboard shortcuts cheat sheet
├── graph/                       # Embedded Kuzu Graph Memory Package
│   ├── __init__.py              # Package exports (AgentGraphStore)
│   ├── schema.py                # 11 Node tables, 12 Rel tables, 7 FTS indices, migrations
│   ├── store.py                 # Thread-safe CRUD operations wrapped in threading.Lock
│   ├── tools.py                 # LangChain tools for search, artifacts, skills, and user memory
│   └── entity_resolution.py     # Deterministic SHA-256 canonical entity keys
├── agent_service.py             # FastAPI backend running LangChain agent loop (up to 100 steps)
├── induction.py                 # Offline episodic procedure induction worker
├── config.json                  # Master configuration: graph_memory, timeouts, scoring, routing
├── requirements.txt             # Python runtime dependencies (fastapi, kuzu, langchain, websockets)
├── package.json                 # Node.js dependencies (electron, react 19, vite, oxlint)
├── CLAUDE.md                    # Reference file for Claude Code
└── AGENTS.md                    # Canonical reference file for Antigravity (this document)
```

---

## 3. Electron Native Shell & Native View Composition

### 3.1 Why WebContentsView Replaced `<webview>`
Initially, the application used Electron's `<webview>` tag. However, Playwright and CDP tools connecting over `localhost:9222` could not reliably inspect DOM elements or run scripts inside `<webview>` out-of-process frames. The architecture was transitioned to Electron's **`WebContentsView`**, which attaches directly as native Chromium child views to `mainWindow.contentView`.

### 3.2 The Critical Compositing Rule
In Electron, `WebContentsView` child views are **hardware-composited above the parent window's base WebContents**. 
- CSS `z-index` **cannot** place React HTML above a `WebContentsView`.
- The only way to keep React chrome (URL bar, tabs, Chat Drawer) visible is to calculate WebContentsView bounds so they never overlap the chrome coordinates.

### 3.3 Coordinate System Math & Titlebar Offsets
[BrowserCanvas.jsx](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/src/components/Canvas/BrowserCanvas.jsx) coordinates layout bounds with [main.js](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/electron/main.js):

1. **macOS HiddenInset Titlebar Offset**:
   The macOS `hiddenInset` titlebar places the traffic lights in the top-left, shifting native content coordinates downward by `28px` (`MACOS_TITLEBAR_OFFSET = 28`).
2. **Top Offset Calculation**:
   ```javascript
   const MACOS_TITLEBAR_OFFSET = 28;
   const bodyRect = bodyLayoutEl.getBoundingClientRect();
   const topOffset = bodyRect.top > 0 ? bodyRect.top : 80;
   const viewStartY = Math.max(MACOS_TITLEBAR_OFFSET, Math.round(topOffset + MACOS_TITLEBAR_OFFSET));
   ```
3. **Chat Drawer Width Deduction**:
   When the AI drawer is open, its width is subtracted from the container width:
   ```javascript
   const width = Math.max(0, Math.round(containerWidth - (isChatOpen ? drawerWidth : 0)));
   ```
4. **Height Calculation**:
   ```javascript
   const height = Math.max(0, Math.min(Math.round(bodyRect.height), window.innerHeight - Math.round(bodyRect.top)));
   ```
5. **Clamping in Main Process**:
   In [electron/main.js](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/electron/main.js#L120-L137), bounds are clamped against window dimensions to prevent overflow. **Critical trap**: If child view bounds exceed window bounds, Chromium's compositor snaps the view to `(0, 0)`, blanketing the React chrome.
6. **Launchpad / Internal Page Detachment**:
   When viewing internal pages (`comex://launchpad`, `nexus://launchpad`, `about:blank`), [syncActiveView()](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/electron/main.js#L88-L106) detaches all child views and calls `setVisible(false)`. This exposes the underlying React launchpad cleanly.

---

## 4. IPC Bridge (`window.nexusNative`)

Exposed in [electron/preload.js](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/electron/preload.js):

| Method | Description |
|---|---|
| `tabs.create(tabId, url)` | Creates a new foreground `WebContentsView` and begins loading `url`. |
| `tabs.destroy(tabId)` | Destroys the WebContentsView and closes the tab. |
| `tabs.setActive(tabId)` | Changes the currently focused and visible foreground view. |
| `tabs.navigate(tabId, url)` | Navigates the specified tab to `url`. |
| `tabs.goBack(tabId)` | Navigates backward in history. |
| `tabs.goForward(tabId)` | Navigates forward in history. |
| `tabs.reload(tabId)` | Reloads the current page. |
| `tabs.setBounds(bounds)` | Updates native view bounds: `{ x, y, width, height, isHidden }`. |
| `tabs.createHiddenAgentTab(tabId, url)` | Spawns a headless background view with URL hash `#<tabId>` for background CDP tasks. |
| `tabs.destroyHiddenAgentTab(tabId)` | Destroys a background hidden view. |
| `tabs.onUpdate(callback)` | Listens for title, favicon, loading state, audible status, and `__new_tab__` events. |

---

## 5. React UI Subsystems (`src/`)

### 5.1 Master State Management ([App.jsx](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/src/App.jsx))
- Manages the `tabs` array: `[{ id, title, url, history, historyIndex, isLoading, isBookmarked, canGoBack, canGoForward }]`.
- Listens to global hotkeys:
  - `Cmd/Ctrl + T`: Open new tab (defaults to `comex://launchpad`).
  - `Cmd/Ctrl + W`: Close active tab (resets to launchpad if closing last tab).
  - `Cmd/Ctrl + R`: Reload active tab.
  - `Cmd/Ctrl + J`: Toggle AI chat drawer.
- Listens for `__new_tab__` window open requests from native web pages.

### 5.2 Intent Routing ([IntentClassifier.js](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/src/agents/IntentClassifier.js))
When the user submits a prompt on the launchpad:
- Dispatches prompt to 9Router LLM endpoint.
- Classifies into:
  - `research`: User wants information, summaries, or comparisons; stays on the current launchpad/chat flow without disrupting tabs.
  - `visible_browser_task`: User explicitly wants to open, view, or interact with a page; creates a new visible tab, switches active workspace, pre-adjusts bounds, and launches automation.

### 5.3 AI Chat Drawer ([ChatDrawer.jsx](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/src/components/Chat/ChatDrawer.jsx))
- **Live Streaming Single Agent**:
  Posts to `http://127.0.0.1:8000/agent/run/stream` via SSE. Receives:
  - `perception`: Initial tab connection and URL.
  - `thinking`: Agent planning state.
  - `step`: Individual tool invocation (`run_browser_harness`, `search_memory`, etc.).
  - `step_done`: Tool execution success/failure.
  - `done`: Final structured Markdown response and execution duration.
- **"Brain" Knowledge Dashboard**:
  Allows the user to inspect, search, add, delete, and consolidate:
  - User Memories (preferences, profile facts, constraints).
  - Domain Skills (distilled automation recipes).
  - Reflections (post-task critiques and lessons learned).
  - Graph Memory Statistics (node counts, relationship counts, database file size).

---

## 6. Python Agent Service ([agent_service.py](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/agent_service.py))

### 6.1 Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check reporting CDP URL, model name, and graph stats. |
| `POST` | `/agent/run` | Non-streaming agent execution returning `AgentResponse`. |
| `POST` | `/agent/run/stream` | Streaming agent execution emitting SSE events. |
| `POST` | `/agent/stop` | Cancels an active task by `taskId` via `asyncio.Task.cancel()`. |
| `GET` | `/cdp/targets` | Debug endpoint returning all active CDP targets from port 9222. |
| `GET` | `/agent/graph/stats` | Returns node, edge counts, and storage metrics from Kuzu. |
| `GET` | `/agent/graph/agent/{id}`| Returns provenance trace for an agent run. |
| `GET` | `/agent/graph/entities` | Queries recent entities with staleness window & max price filter. |
| `GET/POST/DELETE` | `/agent/user/memory` | CRUD for user preferences and facts. |
| `GET/DELETE` | `/agent/skills` | Queries and removes distilled domain skills. |
| `POST` | `/agent/skills/consolidate`| Merges near-identical skills within a domain using Jaccard token overlap. |
| `GET` | `/agent/reflections` | Returns recent post-task self-reflections. |
| `POST` | `/agent/graph/reset` | Resets/wipes the graph database (requires `?confirm=true`). |

### 6.2 Agent Tools Suite
1. `run_browser_harness(code: str, target_tab_ws: Optional[str] = None)`:
   Runs automation Python code via `~/.local/bin/browser-harness`. Automatically intercepts `===COMEX_EXTRACT===` delimiter blocks to persist `Page` and `Entity` nodes into Kuzu.
2. `create_hidden_tab(url: str, tab_id: str)`:
   Evaluates JavaScript in the React UI via CDP evaluate to create a hidden background tab, discovers its WebSocket URL via `/json/list`, and links it with `SPAWNED` relationship in Kuzu.
3. `search_memory(query: str, domain: str = "ALL")`:
   BM25 full-text search across `ENTITIES`, `HISTORY`, `ARTIFACTS`, `USER`, `SKILLS`, or `ALL`. Deduplicates against already-injected excerpts.
4. `save_artifact(title: str, artifact_type: str, content: str)`:
   Persists durable unstructured documents, tables, or findings.
5. `read_artifact(artifact_id: str, chunk_index: int = 0)`:
   Reads stored artifacts in sequential 4,000-character chunks wrapped in `[UNTRUSTED DATA START]` and `[UNTRUSTED DATA END]` delimiters.
6. `save_user_memory(key: str, value: str, category: str = "preference")`:
   Persists facts, preferences, or constraints about the user.
7. `save_skill(name: str, domain: str, description: str, code_snippet: str)`:
   Persists reusable web interaction recipes for a domain.

### 6.3 Structured Extraction Contract
Agents scraping pages output structured extraction blocks:
```text
===COMEX_EXTRACT===
{
  "url": "https://...",
  "title": "Page Title",
  "entities": [
    {
      "type": "product",
      "name": "Product Name",
      "price": 14999.0,
      "currency": "INR",
      "product_id": "SKU_OR_ASIN",
      "attrs": {"rating": 4.5}
    }
  ]
}
===END_EXTRACT===
```

### 6.4 Automatic Memory Context Injection (Phase 10)
Before the agent loop starts, [agent_service.py](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/agent_service.py#L703-L751) scans the user prompt for 14 trigger phrases (`"previously"`, `"earlier"`, `"remember"`, `"what did you find"`, etc.). If matched, it automatically searches Kuzu and injects up to 3 relevant records into the prompt inside `[PERSISTENT MEMORY CONTEXT]` delimiters, marking the data as untrusted evidence.

---

## 7. Embedded Graph Memory Architecture ([graph/](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/graph/))

Stored at `~/.comex/graph.db` using embedded Kuzu.

### 7.1 Schema Definitions ([graph/schema.py](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/graph/schema.py))
- **Node Tables (11)**:
  - `Agent`: `id`, `task`, `intent`, `status`, `created_at`, `completed_at`, `step_count`, `parent_agent_id`
  - `Tab`: `id`, `is_hidden`, `cdp_ws_url`, `created_at`, `closed_at`
  - `Page`: `id`, `url`, `domain`, `title`, `visited_at`, `screenshot_hash`
  - `Entity`: `id`, `type`, `name`, `price`, `currency`, `attrs`, `canonical_key`, `first_seen_at`, `last_seen_at`
  - `ToolCall`: `id`, `tool_name`, `args_json`, `result_summary`, `step_index`, `timestamp`, `success`
  - `Artifact`: `id`, `artifact_type`, `title`, `content`, `created_at`, `source_agent_id`
  - `Episode`: `id`, `agent_id`, `outcome_status`, `self_reported`, `constraint_check`, `created_at`
  - `UserMemory`: `id`, `category`, `key`, `value`, `confidence`, `source_agent_id`, `created_at`, `updated_at`
  - `Skill`: `id`, `name`, `domain`, `description`, `code_snippet`, `usage_count`, `success_count`, `created_at`, `updated_at`, `source_agent_id`
  - `Reflection`: `id`, `agent_id`, `is_novel`, `novelty_reason`, `critique`, `lesson_learned`, `suggested_skill_name`, `created_at`
  - `SchemaMeta`: `key`, `value`
- **Relationship Tables (12)**:
  - `SPAWNED(Agent -> Tab)`
  - `EXECUTED(Agent -> ToolCall)`
  - `NAVIGATED_TO(ToolCall -> Page)`
  - `EXTRACTED(ToolCall -> Entity)`
  - `FOUND_ON(Entity -> Page)`
  - `CREATED_ARTIFACT(Agent -> Artifact)`
  - `FOR(Episode -> Agent)`
  - `REFLECTED_ON(Reflection -> Agent)`
  - `DISTILLED_FROM(Skill -> Agent)`
  - `LEARNED_FROM(UserMemory -> Agent)`
  - `DEPENDS_ON(Agent -> Agent)`
  - `SIMILAR_TO(Entity -> Entity)`
- **FTS Full-Text Search Indices (7)**:
  - `Agent_FTS` on `Agent(task)`
  - `ToolCall_FTS` on `ToolCall(tool_name, args_json, result_summary)`
  - `Entity_FTS` on `Entity(type, name, attrs)`
  - `Artifact_FTS` on `Artifact(title, content, artifact_type)`
  - `UserMemory_FTS` on `UserMemory(category, key, value)`
  - `Skill_FTS` on `Skill(name, domain, description, code_snippet)`
  - `Reflection_FTS` on `Reflection(novelty_reason, critique, lesson_learned)`

### 7.2 Entity Resolution ([graph/entity_resolution.py](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/graph/entity_resolution.py))
Deduplicates scraped products and entities without fuzzy drift:
$$\text{canonical\_key} = \text{SHA256}(\text{domain} : \text{product\_id} \lor \text{normalize\_name}(\text{name}))$$
Entity ID is prefixed as `ent_<canonical_key[:16]>`.

### 7.3 Thread Safety
All Kuzu calls in [AgentGraphStore](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/graph/store.py) execute under `with self._lock:`. Unsynchronized concurrent queries or writes will cause Kuzu connection crashes.

---

## 8. Empirical Research & Historical Phases

- **Phases 0–7**: Core graph schema, write hooks, read path (`find_recent_entities` with 30-min staleness window), debug endpoints, 5-agent concurrency verification, and observability.
- **Phases 8–10**: Unified BM25 text search across history and entities (`search_memory`), durable unstructured artifacts (`save_artifact`), and automatic memory context injection via lexical triggers.
- **Phase 11**: Sequential chunked reading of large artifacts (`read_artifact`).
- **Phases 12–14 (Procedural Gating & Induction Findings)**:
  - Phase 14 conducted a 54-run replication study testing procedural memory transfer across three tasks: Overlay Bypass, Inner Container Scroll, and Hover Reveal.
  - **Key Finding**: While LLM gating prevented 100% of mismatched injections, procedural injection yielded **no reproducible benefit** over zero-shot baselines. The latency overhead of retrieval and gating (~5 seconds) outweighed any minor tool-call reductions.
- **Phase 2.1D (ComexToolService Migration, Authority Hardening & Scope Bounds)**:
  - Migrated browser tools (click, type, navigate, extract, inspect) and graph tools (`search_memory`, `save_artifact`, `read_artifact`) to authoritative `ComexToolService` with native async execution.
  - Preflight fail-closed propagation: `/agent/run` and `/agent/run/stream` explicitly return/emit `success=False` with error payload on preflight halts.
  - Authority invariant: Single-agent context trust derives strictly from verified `trusted_app_context`; missing worker credentials never grant implicit trust.
  - Graph agent lifecycle: `create_agent` deferred until after service and context preflight checks succeed; early halts never leave orphaned records.
  - Memory search completeness: preserved evidence (`truncated`, `omitted_count`, `partial_failure`, `failed_domains`) and user-facing warning notices in tool return.
  - Telemetry error classification: authoritatively detects all uppercase bracketed error codes (`r"^\[[A-Z0-9_]+\]"`) and memory write failure prefixes (`Failed to save user memory:`, `Failed to save skill:`).
  - Cursor notification hardening: `/agent/cursor/notify` scopes context tokens to matching `target_tab_id` and permitted actions; CDP target discovery uses strict origin/path checks.
  - Sync artifact write timeout: surfaces explicit `[OPERATION_TIMEOUT]` with outcome unknown notice.
- **Phase 2.2 (Performance Benchmarking Baseline)**:
  - Established empirical baseline across Graph Memory, Tool Execution Service, Behavioral Nervous System, BrowserGym Multi-Turn Battery, and End-to-End Agent Loop.
  - **Latency Distribution**: LLM reasoning turns account for **94.7%** of end-to-end task duration (~3,600ms P50 per turn, ~11.8s for 3-turn task). CDP navigation takes ~515ms P50, Method E click dispatch takes ~105ms P50 (including 85ms quiescence window), raw DOM inspect/extract takes 1–2ms, and graph memory queries take 3–5ms per domain (0.24ms for entity lookups).
  - **Graph Search Bottleneck**: `search_memory(ALL)` takes ~185ms P50 due to 5 sequential Cypher queries across each node table's FTS index; domain-specific queries execute in 3–5ms.
  - **Nervous System Overhead**: Autonomic overlay matching adds only **+2.08ms P50** net overhead over baseline clean clicks (9.26ms vs 7.18ms), with 100% barrier integrity (zero unexpected physical dispatches across 650 live operations).
  - **BrowserGym Accuracy**: Achieved **86.7% (13/15)** accuracy on MiniWoB++ benchmark battery inside live Electron shell (P50 duration 7.39s).
  - Benchmark reports and raw trial summaries saved to `benchmarks/phase2_2_performance_baseline_report.md` and `benchmarks/phase2_2_benchmark_summary.json`.

---

## 9. Identified Bugs & Code Discrepancies

When working on this codebase, be aware of the following known issues:

1. **Test Failure in `tests/test_dispatch_validation.py`**:
   - `test_save_artifact_missing_title_does_not_crash_loop` fails.
   - **Cause**: In [graph/tools.py:57](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/graph/tools.py#L57), `save_artifact` declares a default `title: str = "Untitled Artifact"`. When the test emits a call without `title`, Pydantic does not raise a `ValidationError`; it defaults the title and succeeds. The test expects a JSON error payload, causing a `JSONDecodeError`.
2. **Missing Function in `src/agents/OrchestratorAgent.js`**:
   - Line 37 calls `parseUserIntent(userPrompt)`.
   - **Cause**: `parseUserIntent` was removed when intent classification was migrated to [IntentClassifier.js](file:///Users/pranavkumarn/Documents/Comex%20Good%20Versions/Comex%20copy%2013/src/agents/IntentClassifier.js). Triggering the legacy simulated multi-agent swarm in `isMultiAgentMode` throws a runtime `ReferenceError`.
3. **Hidden Tab Listener in `electron/main.js`**:
   - In `tabs-create-hidden-agent` (line 333), `attachTabListeners(tabId, view)` is called instead of `attachHiddenTabListeners(tabId, view)`. `attachHiddenTabListeners` is defined at line 229 specifically to avoid triggering React tab updates, but is currently unused.

---

## 10. Development Commands

```bash
# Python Environment
.venv/bin/python agent_service.py    # Start backend daemon on port 8000
.venv/bin/pytest tests/              # Run test suite (120 of 121 currently pass)
.venv/bin/pytest tests/test_graph_store.py # Run specific test module

# Frontend & Electron
npm run dev                          # Start Vite dev server (port 5173)
npm run electron:dev                 # Start Vite and Electron concurrently
npx electron .                       # Launch Electron shell
```
