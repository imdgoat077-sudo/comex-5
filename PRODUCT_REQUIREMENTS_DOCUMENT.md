# Comex Browser — In-Depth Product Requirement Document (PRD) & Technical Architecture Specification

**Product Name:** Comex Browser (AI-Native Autonomous Web Browser)  
**Document Version:** 1.0.0 (Comprehensive Architecture & System Audit)  
**System Status:** Active Development / Hybrid Production Prototype  
**Scope:** Core Shell, React Frontend, Python Agent Daemon, CDP Nervous Subsystem, Embedded Graph Memory, and Operational Tooling  

---

## 1. Executive Summary & Product Vision

### 1.1 Product Statement
**Comex** is an AI-augmented desktop browser that merges traditional Chromium web browsing with autonomous agentic execution. Instead of embedding standard web pages inside synthetic DOM iframes or sandboxed webviews, Comex orchestrates **native Chromium child views (`WebContentsView`)** bound to an **Electron shell**, exposing direct **Chrome DevTools Protocol (CDP)** sockets to a local Python daemon.

### 1.2 Core Value Proposition
- **Dual-Plane Operation**: Users browse normally on the foreground canvas while an AI agent works alongside in real time via an interactive drawer or autonomously inside invisible headless tabs.
- **Visual Cursor Telemetry**: Real-time pointer overlay tracing the AI's physical mouse movements, smooth cubic-bezier transitions, and click ripple effects directly over the rendered native page without breaking DOM isolation.
- **Embedded Graph Memory ("The Brain")**: Long-term state tracking using an embedded Kuzu graph database (`~/.comex/graph.db`) retaining browsed pages, extracted product entities, execution episodes, user preferences, and synthesized domain skills.
- **Deterministic Nervous System (Stage A–E)**: A low-latency pre-execution gate intercepting CDP events to detect DOM churn, race conditions, floating modals/cookie overlays, and element moving anomalies before physical mouse dispatch occurs.

---

## 2. Technical Stack & Subsystems Matrix

| Layer | Component | Technologies & Libraries | Version / Specification |
|---|---|---|---|
| **Native Shell** | Desktop Wrapper | Electron | `43.4.0` |
| | Window Compositor | Electron `WebContentsView` | Native hardware-composited child views |
| | Packaging & Build | `electron-builder`, `concurrently`, `wait-on` | Builder: `26.15.3` |
| **Renderer UI** | Frontend Framework | React 19, ReactDOM | `19.2.8` |
| | Build Tooling | Vite, `@vitejs/plugin-react` | Vite: `8.2.0` |
| | Linter / Static Analysis | Oxlint | `1.75.0` |
| | UI Icons & Components | `lucide-react` | `1.32.0` |
| | Styling | CSS Modules, Design Tokens, Hardware Canvas Maths | Vanilla CSS3 + PostCSS |
| **Agent Daemon** | API Server | FastAPI, Uvicorn | FastAPI `^0.104.0`, Uvicorn `^0.24.0` |
| | Concurrency & Async | `asyncio`, `websockets`, `httpx` | websockets `^12.0`, httpx `^0.25.0` |
| | Schema Validation | Pydantic v2 | `^2.0.0` |
| | LLM Orchestration | LangChain Core, LangChain OpenAI | `langchain-core ^0.2.0`, `langchain-openai ^0.1.0` |
| **AI Model & Proxy** | LLM Gateway | 9Router Proxy | Port `20128` (`http://localhost:20128/v1`) |
| | Foundation Model | Google Gemini via Router (`ag/gemini-3.8-flash`) | Default LLM runtime |
| | Intent Classifier | Fast OpenAI-compatible JSON endpoint | Temperature `0.1`, max tokens `1024` |
| **Memory & Storage** | Graph Database | Embedded Kuzu Database | `kuzu >= 0.4.0` (`~/.comex/graph.db`) |
| | Storage Topology | Embedded C++ Columnar Graph Engine | Multi-thread synchronized (`threading.Lock`) |
| | Search Engine | Built-in BM25 Full-Text Search (FTS) | 9 Specialized FTS Indexes across tables |
| **Browser Harness** | CDP Automation | Browser-Harness CLI | Custom tool at `~/.local/bin/browser-harness` |
| | CDP Remote Port | Chromium DevTools Protocol | Port `9222` |

---

## 3. Environment Variables, Endpoints & API Keys

### 3.1 Network Port Map
- **`http://localhost:5173`**: React 19 Frontend dev server running on Vite.
- **`http://127.0.0.1:8000`**: Python FastAPI agent backend service.
- **`http://127.0.0.1:9222`**: Chromium Remote Debugging CDP interface launched by Electron.
- **`http://localhost:20128/v1`**: Local 9Router LLM reverse proxy exposing an OpenAI-compatible API.

### 3.2 Environment Variables

```bash
# Python Agent Backend (.env or process environment)
ROUTER_API_KEY=""                                      # Bearer token for local 9Router proxy
OPENAI_BASE_URL="http://localhost:20128/v1"            # Target LLM endpoint
MODEL_NAME="ag/gemini-3.8-flash"                # Primary model identifier
CDP_URL="http://127.0.0.1:9222"                        # Electron Chromium CDP entrypoint
GRAPH_DB_PATH="~/.comex/graph.db"                      # Persistent Kuzu graph storage directory
VISUAL_CURSOR_ENABLED="true"                           # Toggles native cursor overlay rendering

# Frontend (.env.local / Vite build time)
VITE_ROUTER_API_KEY=""                                 # Fallback client token
VITE_OPENAI_BASE_URL="http://localhost:20128/v1"       # Intent classifier endpoint
VITE_MODEL_NAME="ag/gemini-3.8-flash"           # Client classifier model
```

### 3.3 Backend API Endpoint Directory (agent_service.py)

| Method | Endpoint | Description & Payload |
|---|---|---|
| `GET` | `/health` | Reports status of daemon, active CDP connection, model configuration, and database record metrics. |
| `POST` | `/agent/run` | Synchronous agent execution blocking until final resolution or timeout. |
| `POST` | `/agent/run/stream` | Server-Sent Events (SSE) streaming endpoint emitting `perception`, `thinking`, `step`, and `done` events. |
| `POST` | `/agent/stop` | Cancels an ongoing task via `asyncio.Task.cancel()`. Expects `{"taskId": "..."}`. |
| `GET` | `/cdp/targets` | Introspects port 9222 and returns active tabs, target IDs, and websocket URLs. |
| `POST` | `/agent/cursor/notify`| Dispatches visual cursor position updates to the overlay window. |
| `GET` | `/agent/graph/stats` | Retrieves node counts, relationship counts, and physical database file footprint. |
| `GET` | `/agent/graph/entities`| Returns structured scraped entities with staleness window filtering (default: 30 minutes). |
| `GET/POST/DELETE` | `/agent/user/memory` | CRUD for persisted user preferences, constraints, and identity markers. |
| `GET/DELETE` | `/agent/skills` | Retains and prunes domain skills (distilled web interaction scripts). |
| `POST` | `/agent/skills/consolidate` | Merges duplicate skills within identical domains using Jaccard token overlap metrics. |
| `GET` | `/agent/reflections` | Returns post-task self-critiques, novelty indicators, and failure analyses. |
| `GET` | `/agent/adapters` | Fetches site-specific compiled automation macros and action templates. |
| `POST` | `/agent/graph/reset` | Destructive graph wipe requiring explicit confirmation (`?confirm=true`). |

---

## 4. System Architecture & Component Interaction

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ELECTRON MAIN PROCESS                                  │
│  [--remote-debugging-port=9222]                                                        │
│  ├── MainWindow (Base WebContents: React UI on localhost:5173)                          │
│  ├── CursorWindow (Transparent always-on-top click-through overlay window)              │
│  └── Native Child Views (WebContentsView for Active Tab + Background Hidden Views)     │
└────────────┬─────────────────────────────▲──────────────────────────────┬──────────────┘
             │                             │                              │
             │ ContextBridge (IPC)         │ CDP Protocol                 │ CDP Protocol
             │ [window.nexusNative]        │ (Port 9222)                  │ (Port 9222)
┌────────────▼─────────────────────────────┴──────┐         ┌─────────────▼──────────────┐
│             REACT 19 RENDERER UI                │         │       NERVOUS SYSTEM       │
│  ├── App.jsx (Tab manager, hotkeys, chrome)     │         │  ├── WorldStateTracker     │
│  ├── BrowserCanvas.jsx (Bounds sync coordinator)│         │  ├── ReflexEngine          │
│  ├── ChatDrawer.jsx (AI stream & Brain UI)      │         │  ├── ActuatorGate          │
│  └── IntentClassifier.js (LLM prompt routing)   │         │  └── Method E Validator    │
└────────────────────┬────────────────────────────┘         └─────────────┬──────────────┘
                     │ HTTP / SSE Requests                                │
                     │ (Port 8000)                                        │ Local In-Process
┌────────────────────▼────────────────────────────────────────────────────▼──────────────┐
│                                  FASTAPI AGENT DAEMON                                  │
│  ├── LangChain Core ReAct Agent Loop (Up to 100 steps)                                  │
│  ├── Tool Suite: run_browser_harness, search_memory, save_artifact, read_artifact       │
│  ├── Lexical Context Injector (Automatic trigger phrases: "previously", "remember")    │
│  └── Embedded Kuzu Graph Memory (14 Node Tables, 15 Rel Tables, 9 FTS Indexes)          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Subsystems Deep-Dive

### 5.1 Native Compositing & Geometric Coordinate Engine
In standard Electron setups, embedding web pages inside `<webview>` breaks deep CDP control. Comex utilizes native `WebContentsView`. 
- **Compositor Snapping Trap**: If a child view overflows the parent window bounds by even 1 pixel, Chromium's hardware compositor resets the view to `(0, 0)`, obliterating the React navbar, tab bar, and AI drawer.
- **Offset Formula**: Handled inside `BrowserCanvas.jsx` and `main.js`:
  - `Y_start = max(MACOS_TITLEBAR_OFFSET, bodyRect.top + 28)`
  - `Width = max(0, containerWidth - (isChatOpen ? drawerWidth : 0))`
  - `Height = max(0, min(bodyRect.height, window.innerHeight - bodyRect.top))`
- **Internal Route Decoupling**: When the tab loads internal schemes (`comex://launchpad`, `nexus://launchpad`, or `about:blank`), all native child views are detached from `mainWindow.contentView` and set to `setVisible(false)`, allowing the React launchpad to render without native view interference.

### 5.2 Deterministic Nervous System (`nervous/`)
To prevent agent race conditions on dynamic modern websites, the nervous subsystem executes a low-overhead evaluation before any tool click:
1. **WorldStateTracker**: Listens to raw CDP events (`Network.requestWillBeSent`, `DOM.documentUpdated`, `Page.frameNavigated`) to track blocking requests and mutation watermarks without full DOM serialization.
2. **Method E**: Validates bounding boxes, visibility styles, occlusion by z-index elements, and pointer-events capability.
3. **ActuatorGate**: A single-flight gate ensuring:
   - `ALLOW`: Exact single logical action execution.
   - `WAIT` / `INVALIDATE`: Zero executions when DOM settles or mutations occur.
   - `ROUTE_TO_KNOWN_HANDLER`: Diverts clicks to registered cookie banner or floating dialog dismissers before touching target elements.

### 5.3 Embedded Graph Memory Schema (`graph/schema.py`)
Persisted in Kuzu with thread-safe locking:
- **Node Entities**: `Agent`, `Tab`, `Page`, `Entity`, `ToolCall`, `Artifact`, `Episode`, `UserMemory`, `Skill`, `Reflection`, `SiteAdapter`, `DOMFingerprint`, `AntiPattern`, `SchemaMeta`.
- **Relational Edges**: `SPAWNED`, `EXECUTED`, `NAVIGATED_TO`, `EXTRACTED`, `FOUND_ON`, `DEPENDS_ON`, `SIMILAR_TO`, `CREATED_ARTIFACT`, `FOR`, `REFLECTED_ON`, `DISTILLED_FROM`, `LEARNED_FROM`, `COMPILED_FROM`, `MATCHES_DOM`, `HAS_ANTIPATTERN`.
- **Entity Resolution Contract**: Scraped elements are uniquely identified by:
  `canonical_key = SHA256(domain : product_id OR normalize_name(name))`
  Preventing entity duplication across sessions.

### 5.4 Visual Cursor Telemetry Subsystem
A dedicated frameless transparent window (`CursorWindow`) sits directly on top of the browser viewport with `setIgnoreMouseEvents(true, { forward: true })`. When agent tools run, CDP coordinates are transmitted via `/agent/cursor/notify` or direct WebSocket to render an SVG mouse cursor with dynamic click rings and acceleration physics.

---

## 6. Functional & Non-Functional Requirements (PRD)

### 6.1 Functional Requirements

1. **Omnibox & Navigation**:
   - The browser must accept valid URLs (`https://...`, `http://...`) and navigate the active native child view.
   - Search strings must automatically resolve to search engine queries (default: Google).
   - Internal URLs (`comex://launchpad`) must instantly hide native views to show the React launchpad.

2. **Semantic Intent Classification**:
   - When a user inputs natural language into the launchpad or omnibox, the system must classify the prompt via `IntentClassifier.js` into `research` or `visible_browser_task`.
   - `research` tasks must execute in the background without stealing user focus.
   - `visible_browser_task` must open a foreground tab, configure layout coordinates, and stream agent actions.

3. **Autonomous Agent Execution Loop**:
   - The agent service must support up to 100 execution steps with tool binding (`run_browser_harness`, `search_memory`, `save_artifact`, `read_artifact`, `create_hidden_tab`).
   - Every DOM extraction matching `===COMEX_EXTRACT===` must be parsed and inserted into Kuzu as `Entity` and `Page` nodes.

4. **Brain Dashboard**:
   - The user must be able to view, search, manually inject, and delete memories and distilled skills from the chat drawer.
   - Skill consolidation must merge overlapping skills using token similarity.

### 6.2 Non-Functional Requirements

1. **Latency & Execution Budget**:
   - The ActuatorGate evaluation must complete within `<= 15ms` to prevent lagging agent interaction.
   - Bounds updates sent from React to Electron over IPC must trigger within a single animation frame (`requestAnimationFrame`).

2. **Crash Resilience & Thread Safety**:
   - All Kuzu database reads and writes must be protected under a mutex (`threading.Lock`) to prevent native C++ engine segmentation faults during concurrent tool calls.
   - If child views fail to load, the main window must not freeze.

3. **Security & Data Isolation**:
   - Context isolation must be enabled (`contextIsolation: true`) in all Electron WebPreferences.
   - External web pages must never have access to Node.js primitives or native electron APIs.
   - Retrieved long-term artifacts injected into prompts must be enclosed inside `[UNTRUSTED DATA START]` delimiters to mitigate prompt injection.

---

## 7. Known Defects & Technical Debt (Audit Findings)

1. **Validation Error Discrepancy (`graph/tools.py:57`)**:
   - `save_artifact` declares a default `title: str = "Untitled Artifact"`. When unit tests send payloads without a title expecting a Pydantic validation failure, the tool succeeds with the default, causing test assertion failures in `tests/test_dispatch_validation.py`.
2. **Missing Function in Simulated Swarm (`src/agents/OrchestratorAgent.js:37`)**:
   - Calls `parseUserIntent(userPrompt)`, which was deleted during the migration to `IntentClassifier.js`. Activating legacy multi-agent mode throws an unhandled `ReferenceError`.
3. **Unused Hidden Tab Listener in Main Process (`electron/main.js:333`)**:
   - `tabs-create-hidden-agent` invokes `attachTabListeners` instead of `attachHiddenTabListeners`, causing background hidden tabs to erroneously emit title and URL update events to the visible tab strip.

---

## 8. Run & Verification Guide

```bash
# 1. Start Python Agent Daemon (Port 8000)
source venv/bin/activate
python agent_service.py

# 2. Run Python Backend Test Suite
pytest tests/ -v

# 3. Launch React & Electron Shell (Concurrent)
npm run electron:dev

# 4. Standalone React Browser UI (Port 5173)
npm run dev
```
