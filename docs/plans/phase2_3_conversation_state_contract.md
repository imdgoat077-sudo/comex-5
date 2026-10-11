# Comex Phase 2.3 — Conversation-State Contract & Architecture

## 1. Overview & Purpose
This document specifies the authoritative conversation-state contract for Comex Phase 2.3. It defines how multi-turn conversational continuity is maintained between the Electron chat UI, FastAPI endpoints (`agent_service.py`), and the underlying LangChain agent loop without introducing heavyweight memory architectures or extra LLM inference rounds.

---

## 2. Distinct Identity Model

| Identifier | Type | Scope | Semantics |
|---|---|---|---|
| `conversation_id` | `string` | Chat thread | Identifies a continuous user discussion thread across multiple turns. Stable across turns in the same chat. |
| `taskId` | `string` | Execution run | Identifies a single agent operation or invocation. Used for `/agent/stop` cancellation and telemetry. |
| `tabId` | `string` | Browser view | Identifies the Chromium native child view (`WebContentsView`) targeted by browser operations. |
| `contextToken` | `string` | Security token | Application-level secret (`COMEX_INTERNAL_APP_TOKEN`) used for IPC and lease validation. Never used as conversation identity. |
| `task_context` | `object` | Compact outcome | Compact, verified outcome of previous browser work (e.g. product name, action performed, status, target URL). |

---

## 3. Data Flow & Single Source of Truth

The Electron/React chat drawer (`ChatDrawer.jsx`) acts as the single source of truth for the user's active discussion history:
1. `ChatDrawer.jsx` initializes a `conversation_id` for each chat session.
2. When the user sends Turn $N$:
   - The frontend formats the recent conversation turns (at most 10 messages, filtered to user and assistant roles).
   - If the previous turn completed a browser task, compact structured metadata (e.g. entity name, action, status) is attached as `task_context`.
   - The payload `{ conversation_id, messages, task_context, task, taskId, tabId, url, intent }` is passed via IPC bridge (`nexusNative.agent.stream`) to the main process and on to FastAPI `POST /agent/run/stream`.
3. FastAPI (`agent_service.py`):
   - Validates the `AgentRequest` containing the optional `conversation_id`, `messages`, and `task_context`.
   - Forwards them into `execute_agent_loop`.
4. `execute_agent_loop`:
   - Builds a clean LangChain message sequence:
     1. `SystemMessage(SYSTEM_PROMPT)`
     2. Preceding `HumanMessage` and `AIMessage` turns (bounded).
     3. Compact verified task context block (if available from previous turns).
     4. Current `HumanMessage(task + ...)`.
   - LLM sees the complete conversation context naturally in its prompt window.
5. Result Delivery:
   - When the agent completes Turn $N$, it returns the response text along with structured `verified_result` metadata.
   - The frontend updates the chat history and records `verified_result` in the message's task outcome, ready to feed into Turn $N+1$.

---

## 4. Contract Schema

### 4.1 FastAPI Request Schema (`AgentRequest`)
```python
class ConversationMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    task_outcome: Optional[dict] = None

class AgentRequest(BaseModel):
    task: str
    taskId: str
    tabId: str
    url: Optional[str] = None
    intent: str = "UNKNOWN"
    conversation_id: Optional[str] = None
    messages: Optional[list[ConversationMessage]] = None
    task_context: Optional[dict] = None
    use_procedures: bool = False
    procedure_mode: str = "none"
    use_graph: bool = True
    workerId: Optional[str] = None
    workerToken: Optional[str] = None
    contextToken: Optional[str] = None
```

### 4.2 Bounded Constraints
- **Max History Turns**: At most 10 recent messages (5 user + 5 assistant turns).
- **Max Message Length**: Truncate individual historical messages to 2,000 characters to prevent prompt context bloat.
- **Max Task Context**: Structured dict containing only:
  - `entity_name` or `product`
  - `action` (e.g. `add_to_cart`, `navigate`, `search`)
  - `status` (`success`, `failed`, `unknown`)
  - `url` (target URL)
  - `details` (brief summary string, max 200 chars)

---

## 5. Operational Scenarios & Safety Rules
1. **New Conversation**: Fresh `conversation_id` created, `messages` empty. Zero overhead.
2. **Follow-Up (e.g. "Do you think it is the best?")**:
   - Prior message: User asked to add "Introduction to Algorithms, 4th Edition".
   - Prior response: Assistant confirmed it was added to cart.
   - Task context: `{ product: "Introduction to Algorithms, 4th Edition", action: "add_to_cart", status: "success" }`.
   - The agent resolves "it" directly to the book without asking for clarification.
3. **Action Safety**: A question about a prior result ("Do you think it is the best?", "What was the price?") MUST NOT re-trigger the original physical browser action (e.g. do NOT click add-to-cart again).
4. **Conversation Switching**: Switching tabs or chats provides a different `conversation_id` and message array. No history leaks across threads.
5. **Backwards Compatibility**: All parameters default to `None` / empty. Existing single-turn tests and API callers execute without modification.
