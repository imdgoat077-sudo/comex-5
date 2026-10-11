# Phase 2.1C — Graph Memory Integration and Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a high-performance, deterministic caching and memory enrichment subsystem on top of the embedded Kùzu graph store in `ComexToolService` and `graph/`, eliminating lock contention and keeping LLM calls completely out of the critical retrieval path.

**Architecture:** A partitioned, in-memory LRU/TTL cache (`GraphMemoryCache`) sits in front of the single-threaded embedded Kùzu engine (`AgentGraphStore`). High-frequency reads (`search_memory`, `get_context`, `read_artifact`, `get_all_user_memory`, `find_skills_for_domain`, `find_matching_adapter`) hit the cache in sub-millisecond time without acquiring the Kùzu lock. Graph writes deterministically invalidate their respective cache partitions. `ComexToolService` exposes cache telemetry and guarantees strict context deadline/cancellation checking even on cache hits.

**Tech Stack:** Python 3.11+, embedded Kùzu Graph DB (`kuzu`), `asyncio`, `threading.RLock`, Pytest.

**Spec:** PRD Section 2/3 & Comex Architecture Report Section D/E; constraints from Phase 2.1B review.

## Global Constraints

- Keep the existing Kùzu graph store (`~/.comex/graph.db`); do not introduce another database (no SQLite, Redis, Chroma, DuckDB).
- Keep LLM calls completely out of the critical retrieval path; memory search must remain deterministic (BM25 FTS / exact domain / key lookups).
- Do not reopen Phase 1.5 (actuator gates, method E, coordinate validation).
- Do not touch future multi-agent swarm orchestration.
- Preserve thread safety across embedded Kùzu's `self._lock` and all cache reads/writes.
- All existing 481 repository tests must continue to pass without regression.

---

### File Structure Map

| File | Purpose | Responsibility |
|---|---|---|
| `graph/cache.py` (New) | In-Memory Partitioned Cache | Thread-safe LRU/TTL cache with partition invalidation, hit/miss stats |
| `graph/store.py` (Modify) | Kùzu Graph Store | Connect `GraphMemoryCache` to `AgentGraphStore` reads and write-invalidation hooks |
| `services/tool_service.py` (Modify) | Tool Service Layer | Integrate cache into `get_context`, `search_memory`, `read_artifact`, cache stats |
| `agent_service.py` (Modify) | Backend Daemon | Expose cache stats on `/agent/graph/stats`; ensure 0 LLM calls in memory path |
| `tests/test_graph_cache.py` (New) | Cache Unit Tests | Unit tests for TTL expiration, LRU eviction, invalidation, thread safety |
| `tests/test_graph_store_caching.py` (New) | Store Cache Tests | Integration tests for cached queries and write invalidation in `AgentGraphStore` |
| `tests/test_tool_service_caching.py` (New) | Service Cache Tests | Tests for `ComexToolService` fast-path hits, deadline safety, invalidation |

---

### Task 1: Thread-Safe Partitioned Graph Memory Cache (`graph/cache.py`)

**Files:**
- Create: `graph/cache.py`
- Test: `tests/test_graph_cache.py`

**Interfaces:**
- Produces:
  - `class GraphMemoryCache`:
    - `get(partition: str, key: str) -> Optional[Any]`
    - `set(partition: str, key: str, value: Any, ttl: Optional[float] = None) -> None`
    - `invalidate_partition(partition: str) -> int`
    - `invalidate_key(partition: str, key: str) -> bool`
    - `clear() -> None`
    - `get_stats() -> Dict[str, Any]`

- [ ] **Step 1: Write the failing tests for `GraphMemoryCache`**

Create `tests/test_graph_cache.py` with tests for TTL expiration, LRU eviction, partition-level invalidation, and hit/miss stats:

```python
import time
import threading
import pytest
from graph.cache import GraphMemoryCache

def test_cache_set_and_get_hit():
    cache = GraphMemoryCache(default_ttl=10.0, max_entries_per_partition=10)
    cache.set("search", "query:laptops", [{"id": "ent_1", "name": "Laptop"}])
    result = cache.get("search", "query:laptops")
    assert result == [{"id": "ent_1", "name": "Laptop"}]
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0

def test_cache_miss_returns_none():
    cache = GraphMemoryCache()
    result = cache.get("search", "nonexistent")
    assert result is None
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1

def test_cache_ttl_expiration():
    cache = GraphMemoryCache(default_ttl=0.05)
    cache.set("artifacts", "art_123", {"title": "Test", "content": "Hello"})
    assert cache.get("artifacts", "art_123") is not None
    time.sleep(0.06)
    assert cache.get("artifacts", "art_123") is None
    stats = cache.get_stats()
    assert stats["misses"] == 1

def test_cache_lru_eviction():
    cache = GraphMemoryCache(default_ttl=10.0, max_entries_per_partition=2)
    cache.set("skills", "k1", "v1")
    cache.set("skills", "k2", "v2")
    # Access k1 to make it most recently used
    _ = cache.get("skills", "k1")
    # Insert k3, should evict k2 (least recently used)
    cache.set("skills", "k3", "v3")
    assert cache.get("skills", "k1") == "v1"
    assert cache.get("skills", "k3") == "v3"
    assert cache.get("skills", "k2") is None
    stats = cache.get_stats()
    assert stats["evictions"] == 1

def test_cache_partition_invalidation():
    cache = GraphMemoryCache()
    cache.set("search", "q1", [1])
    cache.set("search", "q2", [2])
    cache.set("artifacts", "a1", "doc")
    
    count = cache.invalidate_partition("search")
    assert count == 2
    assert cache.get("search", "q1") is None
    assert cache.get("search", "q2") is None
    assert cache.get("artifacts", "a1") == "doc"

def test_cache_thread_safety():
    cache = GraphMemoryCache(default_ttl=5.0, max_entries_per_partition=50)
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(100):
                key = f"key_{i % 10}"
                cache.set("search", key, f"val_{worker_id}_{i}")
                _ = cache.get("search", key)
                if i % 25 == 0:
                    cache.invalidate_partition("search")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph_cache.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'graph.cache'`

- [ ] **Step 3: Implement `graph/cache.py`**

Create `graph/cache.py` with:
- `CacheEntry` dataclass storing `value`, `expires_at: float`, and `last_accessed: float`.
- `CachePartition` managing an `OrderedDict` under `threading.RLock`.
- `GraphMemoryCache` managing partitions (`search`, `artifacts`, `entities`, `skills`, `user_memory`, `adapters`).
- Thread-safe metric collection (`hits`, `misses`, `evictions`, `invalidations`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph_cache.py -v`  
Expected: PASS (all 6 tests passing).

- [ ] **Step 5: Commit task deliverable**

Run:
```bash
git add graph/cache.py tests/test_graph_cache.py
git commit -m "feat(graph): add thread-safe partitioned GraphMemoryCache with LRU and TTL"
```

---

### Task 2: Graph Store Write-Invalidation Hooks (`graph/store.py`)

**Files:**
- Modify: `graph/store.py`
- Test: `tests/test_graph_store_caching.py`

**Interfaces:**
- Consumes: `from graph.cache import GraphMemoryCache`
- Modifies: `AgentGraphStore.__init__(self, db_path: str, cache: Optional[GraphMemoryCache] = None)`
- Produces:
  - Cache-backed `search_memory`, `get_artifact`, `find_recent_entities`, `get_all_user_memory`, `find_skills_for_domain`, `find_matching_adapter`.
  - Automatic partition invalidation on writes:
    - `save_artifact`, `delete_artifact` -> invalidates `artifacts`, `search`
    - `upsert_entity` -> invalidates `entities`, `search`
    - `set_user_memory`, `delete_user_memory` -> invalidates `user_memory`, `search`
    - `save_skill`, `delete_skill`, `consolidate_skills` -> invalidates `skills`, `search`
    - `save_site_adapter`, `delete_adapter` -> invalidates `adapters`

- [ ] **Step 1: Write the failing tests for store caching and invalidation**

Create `tests/test_graph_store_caching.py`:

```python
import os
import shutil
import tempfile
import pytest
from graph.store import AgentGraphStore
from graph.cache import GraphMemoryCache

@pytest.fixture
def temp_graph_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_cache.db")
    cache = GraphMemoryCache(default_ttl=60.0)
    store = AgentGraphStore(db_path=db_path, cache=cache)
    yield store, cache
    store.close() if hasattr(store, "close") else None
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_store_search_memory_cached(temp_graph_store):
    store, cache = temp_graph_store
    store.create_agent("agent_1", "Search for laptops", "test")
    store.save_artifact("agent_1", "Laptops List", "notes", "MacBook Pro M3")

    # First search -> Miss, then caches
    res1 = store.search_memory("MacBook", domain="ARTIFACTS", limit=5)
    assert len(res1) > 0
    assert cache.get_stats()["misses"] >= 1

    # Second search -> Hit from cache without executing Kùzu FTS query
    res2 = store.search_memory("MacBook", domain="ARTIFACTS", limit=5)
    assert res2 == res1
    assert cache.get_stats()["hits"] >= 1

def test_store_save_artifact_invalidates_cache(temp_graph_store):
    store, cache = temp_graph_store
    store.create_agent("agent_1", "Write notes", "test")
    art_id = store.save_artifact("agent_1", "Notes 1", "notes", "Initial text")

    # Read artifact -> Caches it
    doc1 = store.get_artifact(art_id)
    assert doc1["content"] == "Initial text"
    assert cache.get("artifacts", f"art:{art_id}") is not None

    # Save a new artifact -> Should invalidate search and artifacts partitions
    store.save_artifact("agent_1", "Notes 2", "notes", "Second text")
    assert cache.get("artifacts", f"art:{art_id}") is None

def test_store_user_memory_cached_and_invalidated(temp_graph_store):
    store, cache = temp_graph_store
    store.set_user_memory("budget", "1500 USD", category="preference")
    
    # First get -> caches
    mem1 = store.get_all_user_memory()
    assert any(m["key"] == "budget" for m in mem1)
    
    # Second get -> hits cache
    mem2 = store.get_all_user_memory()
    assert mem2 == mem1

    # Update memory -> invalidates cache
    store.set_user_memory("budget", "2000 USD", category="preference")
    mem3 = store.get_all_user_memory()
    assert any(m["value"] == "2000 USD" for m in mem3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph_store_caching.py -v`  
Expected: FAIL (`TypeError: AgentGraphStore.__init__() got an unexpected keyword argument 'cache'`)

- [ ] **Step 3: Modify `graph/store.py` to integrate `GraphMemoryCache`**

Update `AgentGraphStore`:
1. In `__init__`: accept `cache: Optional[GraphMemoryCache] = None`. If None, instantiate a default `GraphMemoryCache`.
2. In `search_memory`: check `self.cache.get("search", cache_key)`. If hit, return cached list. If miss, run query under `with self._lock:`, set in cache, return.
3. In `get_artifact`: check `self.cache.get("artifacts", f"art:{artifact_id}")`. If hit, return. If miss, query and cache.
4. In `get_all_user_memory`: check `self.cache.get("user_memory", f"cat:{category}")`. If hit, return. If miss, query and cache.
5. In `find_skills_for_domain`: check `self.cache.get("skills", f"dom:{domain}:{limit}")`. If hit, return. If miss, query and cache.
6. In `find_matching_adapter`: check `self.cache.get("adapters", f"match:{domain}:{intent_query}")`. If hit, return. If miss, query and cache.
7. In all mutation methods (`save_artifact`, `delete_artifact`, `upsert_entity`, `set_user_memory`, `delete_user_memory`, `save_skill`, `delete_skill`, `save_site_adapter`): call `self.cache.invalidate_partition(...)` corresponding to the mutated domain.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph_store_caching.py -v`  
Expected: PASS (all 3 tests pass).

- [ ] **Step 5: Run existing graph store tests to ensure zero regressions**

Run: `.venv/bin/pytest tests/test_graph_store.py -q`  
Expected: PASS (all existing tests pass).

- [ ] **Step 6: Commit task deliverable**

Run:
```bash
git add graph/store.py tests/test_graph_store_caching.py
git commit -m "feat(graph): integrate cache into AgentGraphStore with automatic write invalidation"
```

---

### Task 3: ComexToolService Caching Integration (`services/tool_service.py`)

**Files:**
- Modify: `services/tool_service.py`
- Test: `tests/test_tool_service_caching.py`

**Interfaces:**
- Consumes: `GraphMemoryCache`, `AgentGraphStore`
- Produces:
  - `ComexToolService.get_context()` fast-path on cache hit (`cache_hit: True` in evidence)
  - `ComexToolService.search_memory()` fast-path on cache hit (`cache_hit: True` in evidence)
  - `ComexToolService.read_artifact()` fast-path on cache hit
  - `ComexToolService.get_cache_stats()` returning telemetry
  - Strict preservation of `context.check_active()`, cancellation, deadline expiration, and lease verification.

- [ ] **Step 1: Write the failing tests for `ComexToolService` caching**

Create `tests/test_tool_service_caching.py`:

```python
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from nervous.action_policy import ServerTabLeaseRegistry
from services.tool_service import ComexToolService, create_tool_execution_context, ToolErrorCode
from graph.cache import GraphMemoryCache

@pytest.fixture
def lease_registry():
    return ServerTabLeaseRegistry()

@pytest.fixture
def mock_graph_store():
    store = MagicMock()
    store.search_memory.return_value = [
        {"source_id": "art_1", "memory_type": "artifact", "content_excerpt": "MacBook details", "provenance": "other_agent"}
    ]
    store.get_artifact.return_value = {
        "artifact_id": "art_1",
        "title": "Specs",
        "artifact_type": "notes",
        "content": "A" * 8000,
    }
    return store

@pytest.mark.asyncio
async def test_tool_service_search_memory_cache_hit_and_evidence(lease_registry, mock_graph_store):
    cache = GraphMemoryCache()
    service = ComexToolService(
        lease_registry=lease_registry,
        graph_store=mock_graph_store,
        cache=cache,
    )
    lease = lease_registry.acquire_lease("w_cache_1", "tab_1", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-cache-1",
        worker_id="w_cache_1",
        worker_token=lease.lease_token,
        target_tab="tab_1",
        lease_registry=lease_registry,
    )

    # First call: cache miss, calls store
    res1 = await service.search_memory(ctx, query="MacBook", domain="ALL", limit=5)
    assert res1.success is True
    assert res1.evidence["cache_hit"] is False
    assert mock_graph_store.search_memory.call_count == 1

    # Second call: cache hit, skips store call!
    res2 = await service.search_memory(ctx, query="MacBook", domain="ALL", limit=5)
    assert res2.success is True
    assert res2.evidence["cache_hit"] is True
    assert mock_graph_store.search_memory.call_count == 1  # Unchanged!

@pytest.mark.asyncio
async def test_tool_service_cache_hit_respects_cancelled_context(lease_registry, mock_graph_store):
    cache = GraphMemoryCache()
    service = ComexToolService(
        lease_registry=lease_registry,
        graph_store=mock_graph_store,
        cache=cache,
    )
    cancel_evt = asyncio.Event()
    lease = lease_registry.acquire_lease("w_cache_2", "tab_2", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-cache-2",
        worker_id="w_cache_2",
        worker_token=lease.lease_token,
        target_tab="tab_2",
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    # Populate cache first
    await service.search_memory(ctx, query="MacBook", domain="ALL", limit=5)

    # Cancel context
    cancel_evt.set()

    # Even though data is in cache, cancelled context must fail closed!
    res = await service.search_memory(ctx, query="MacBook", domain="ALL", limit=5)
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value

@pytest.mark.asyncio
async def test_tool_service_save_artifact_invalidates_service_cache(lease_registry, mock_graph_store):
    cache = GraphMemoryCache()
    mock_graph_store.save_artifact.return_value = "art_new_1"
    service = ComexToolService(
        lease_registry=lease_registry,
        graph_store=mock_graph_store,
        cache=cache,
    )
    lease = lease_registry.acquire_lease("w_cache_3", "tab_3", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-cache-3",
        worker_id="w_cache_3",
        worker_token=lease.lease_token,
        target_tab="tab_3",
        lease_registry=lease_registry,
    )

    # Prime cache
    await service.search_memory(ctx, query="MacBook", domain="ALL", limit=5)
    assert cache.get_stats()["items_count"] > 0

    # Save artifact
    res_save = await service.save_artifact(ctx, title="New Note", content="Fresh data")
    assert res_save.success is True

    # Cache for search must have been invalidated!
    assert cache.get("search", "search:MacBook:ALL:5") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tool_service_caching.py -v`  
Expected: FAIL (`TypeError: ComexToolService.__init__() got an unexpected keyword argument 'cache'`)

- [ ] **Step 3: Update `services/tool_service.py` to integrate caching**

1. In `ComexToolService.__init__`: accept `cache: Optional[GraphMemoryCache] = None`. If None and `graph_store` has a `.cache` attribute, bind to `graph_store.cache`; otherwise instantiate a `GraphMemoryCache()`.
2. In `get_context`:
   - Revalidate claims & lease.
   - Check context active.
   - Key: `f"context:{query.strip()}:{domain}:{eff_limit}"`.
   - Check cache under `context.check_active()`.
   - If cache hit: return immediately with `evidence={"cache_hit": True, ...}`.
   - If cache miss: run `asyncio.to_thread(_do_get)`, populate cache, return with `evidence={"cache_hit": False, ...}`.
3. In `search_memory`:
   - Same fast-path with `f"search:{query.strip()}:{domain}:{eff_limit}"`.
4. In `read_artifact`:
   - Same fast-path with `f"artifact:{artifact_id}:{chunk_index}:{eff_chunk_size}"`.
5. In `save_artifact`:
   - On successful write, call `self._cache.invalidate_partition("search")` and `self._cache.invalidate_partition("artifacts")`.
6. Add `get_cache_stats()` method on `ComexToolService`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tool_service_caching.py -v`  
Expected: PASS (all 3 tests pass).

- [ ] **Step 5: Run contract test suite to verify zero regressions**

Run: `.venv/bin/pytest tests/test_tool_service_contract.py -v`  
Expected: PASS (all 103 tests pass).

- [ ] **Step 6: Commit task deliverable**

Run:
```bash
git add services/tool_service.py tests/test_tool_service_caching.py
git commit -m "feat(service): wire GraphMemoryCache into ComexToolService memory methods"
```

---

### Task 4: Fast Deterministic Background Context Enrichment & Telemetry (`agent_service.py`)

**Files:**
- Modify: `agent_service.py`
- Test: `tests/test_memory_enrichment_and_stats.py`

**Interfaces:**
- Consumes: `ComexToolService`, `GraphMemoryCache`
- Produces:
  - Endpoint `GET /agent/graph/stats` enriched with cache telemetry:
    `{"stats": ..., "cache": {"hits": ..., "misses": ..., "hit_ratio": ..., "size": ...}}`
  - Fully deterministic automatic context retrieval: zero LLM gating calls, pure BM25/FTS cache retrieval.

- [ ] **Step 1: Write the failing tests for memory stats and deterministic retrieval**

Create `tests/test_memory_enrichment_and_stats.py`:

```python
import pytest
from fastapi.testclient import TestClient
from agent_service import app, graph_store

@pytest.fixture
def client():
    return TestClient(app)

def test_graph_stats_endpoint_includes_cache_metrics(client):
    response = client.get("/agent/graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert "cache" in data
    cache_data = data["cache"]
    assert "hits" in cache_data
    assert "misses" in cache_data
    assert "hit_ratio" in cache_data
    assert "items_count" in cache_data

def test_deterministic_retrieval_submillisecond_on_cache_hit():
    from graph.cache import GraphMemoryCache
    cache = GraphMemoryCache()
    # Populate cache
    cache.set("search", "search:test:ALL:3", [{"id": "ent_1", "text": "cached item"}])
    
    import time
    t0 = time.perf_counter()
    res = cache.get("search", "search:test:ALL:3")
    dur_ms = (time.perf_counter() - t0) * 1000.0

    assert res is not None
    assert dur_ms < 1.0  # Sub-millisecond!
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_memory_enrichment_and_stats.py -v`  
Expected: FAIL (missing `cache` key in `/agent/graph/stats`)

- [ ] **Step 3: Update `agent_service.py`**

1. In `/agent/graph/stats` endpoint: call `graph_store.cache.get_stats()` if available and embed into returned JSON payload.
2. Ensure automatic context retrieval in `execute_agent_loop` uses `search_memory` (which now hits the cache on repeat/similar queries) without initiating any LLM calls in the retrieval path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_memory_enrichment_and_stats.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit task deliverable**

Run:
```bash
git add agent_service.py tests/test_memory_enrichment_and_stats.py
git commit -m "feat(api): expose cache telemetry on /agent/graph/stats and guarantee deterministic retrieval"
```

---

### Task 5: Full Repository Verification & Performance Benchmark

**Files:**
- Verification only

- [ ] **Step 1: Run Python compilation across all modified files**

Run:
```bash
.venv/bin/python -m py_compile graph/cache.py graph/store.py services/tool_service.py agent_service.py
```
Expected: Clean compilation, return code 0.

- [ ] **Step 2: Run focused service contract suite**

Run:
```bash
.venv/bin/pytest tests/test_tool_service_contract.py tests/test_tool_service_caching.py tests/test_graph_cache.py tests/test_graph_store_caching.py -v
```
Expected: All tests pass.

- [ ] **Step 3: Run security regression suites**

Run:
```bash
.venv/bin/pytest tests/test_phase1_5_action_hardening.py tests/test_phase1_tab_safety.py tests/test_stream_agent_endpoint.py -v
```
Expected: All 91 tests pass.

- [ ] **Step 4: Run full repository test suite**

Run:
```bash
.venv/bin/pytest tests/ -q
```
Expected: 490+ passed, 0 failures.
