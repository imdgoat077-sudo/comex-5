"""
Phase 2.1C Comprehensive Tests: Memory Integration, Bounded Retrieval, Provenance & Concurrency.

Covers:
1. Valid and invalid memory domains (ToolErrorCode.VALIDATION_ERROR on unsupported domains).
2. Search results across all supported domains (ALL, ENTITIES, HISTORY, ARTIFACTS, USER, SKILLS).
3. Store that over-returns records or oversized fields with arbitrary metadata.
4. Aggregate output budget enforcement (MAX_AGGREGATE_MEMORY_OUTPUT_CHARS = 16,000) and truncation metadata.
5. get_context returns only the records represented in context_text (items alignment, content consistency).
6. Correct same-run and same-task memory exclusion using actual provenance.
7. Real Agent-linked artifact with graph relationship traversal.
8. Orphan run artifact: run/task identities preserved while missing source_agent_id remains empty.
9. Correct LEARNED_FROM and DISTILLED_FROM graph relationship traversal.
10. Entity observation timestamps (first_seen_at, last_seen_at) and historical-memory instructions.
11. Multi-chunk artifact reading with source_run_id and source_task_id roundtripping.
12. Write failure, cancellation, deadline, and compensating-deletion semantics preserved.
13. Concurrent reads and writes through a shared AgentGraphStore with content verification.
14. Expired/revoked authorization blocks memory operations before store execution.
15. In-flight read cancellation and deadline expiration fail closed with proper error codes.
16. Malformed top-level backend results (None, dict, string) return explicit OPERATION_FAILED.
17. Store exception and total domain failure fail explicitly, while partial domain failure is observable.
18. Malformed entity attributes handled defensively without dropping remaining results.
19. Missing timestamps do not become literal string "None".
20. Schema migration from old database without new columns proves backward compatibility.
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from unittest.mock import MagicMock

import kuzu
import pytest

from graph.schema import run_migrations, _safe_add_column, NODE_TABLES, REL_TABLES
from graph.store import AgentGraphStore
from nervous.action_policy import ServerTabLeaseRegistry
from services.tool_service import ServerContextAuthorityRegistry
from services import (
    ComexToolService,
    MAX_AGGREGATE_MEMORY_OUTPUT_CHARS,
    MAX_MEMORY_QUERY_LENGTH,
    MAX_MEMORY_RESULTS,
    MEMORY_TRUNCATION_MARKER,
    ToolErrorCode,
    ToolExecutionContext,
    VALID_MEMORY_DOMAINS,
    create_tool_execution_context,
)


@pytest.fixture
def lease_registry():
    return ServerTabLeaseRegistry()


@pytest.fixture
def auth_registry():
    return ServerContextAuthorityRegistry()


@pytest.fixture
def temp_kuzu_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_p21c.db")
    store = AgentGraphStore(db_path)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 1. Valid and Invalid Memory Domains
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_domain_validation_valid_domains(lease_registry):
    mock_store = MagicMock()
    mock_store.search_memory.return_value = []
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_dom", "tab_dom", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-dom",
        worker_id="w_dom",
        worker_token=lease.lease_token,
        target_tab="tab_dom",
        lease_registry=lease_registry,
    )

    for domain in ["ALL", "ENTITIES", "HISTORY", "ARTIFACTS", "USER", "SKILLS"]:
        # Test exact case
        res = await service.search_memory(ctx, query="laptop", domain=domain)
        assert res.success is True, f"Failed for valid domain {domain}: {res.error_message}"
        assert res.evidence["domain"] == domain

        # Test case-insensitivity and whitespace trimming
        res_lower = await service.search_memory(ctx, query="laptop", domain=f"  {domain.lower()}  ")
        assert res_lower.success is True
        assert res_lower.evidence["domain"] == domain


@pytest.mark.asyncio
async def test_domain_validation_invalid_domains_rejected(lease_registry):
    mock_store = MagicMock()
    mock_store.search_memory.return_value = []
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_inv", "tab_inv", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-inv",
        worker_id="w_inv",
        worker_token=lease.lease_token,
        target_tab="tab_inv",
        lease_registry=lease_registry,
    )

    # Empty, whitespace, wrong strings, wrong types, booleans
    invalid_domains = ["", "   ", "UNKNOWN", "PROCEDURES", "SYSTEM", None, 123, True, False]
    for inv in invalid_domains:
        res = await service.search_memory(ctx, query="laptop", domain=inv)
        assert res.success is False
        assert res.error_code == ToolErrorCode.VALIDATION_ERROR.value
        mock_store.search_memory.assert_not_called()

        res_ctx = await service.get_context(ctx, query="laptop", domain=inv)
        assert res_ctx.success is False
        assert res_ctx.error_code == ToolErrorCode.VALIDATION_ERROR.value


@pytest.mark.asyncio
async def test_query_validation_bounds(lease_registry):
    mock_store = MagicMock()
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_query", "tab_query", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-query",
        worker_id="w_query",
        worker_token=lease.lease_token,
        target_tab="tab_query",
        lease_registry=lease_registry,
    )

    # Empty query
    for bad_q in ["", "   ", None, 123]:
        res = await service.search_memory(ctx, query=bad_q)
        assert res.success is False
        assert res.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # Oversized query (> 1,000 characters)
    oversized_q = "a" * (MAX_MEMORY_QUERY_LENGTH + 1)
    res_over = await service.search_memory(ctx, query=oversized_q)
    assert res_over.success is False
    assert res_over.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "exceeds maximum length" in res_over.error_message


# -----------------------------------------------------------------------------
# 2. Search Memory across all supported domains with real Kùzu
# -----------------------------------------------------------------------------

def test_kuzu_search_all_supported_domains(temp_kuzu_store):
    store = temp_kuzu_store

    # 1. Entity
    store.upsert_entity("tc_1", "p_1", "product", "Dell XPS 15", 1999.0, "USD", {"ram": "32GB"})
    ag_id = "agent_pricing_1"
    store.create_agent(ag_id, "Investigate pricing", "research")
    # 3. ToolCall
    store.record_tool_call(ag_id, "browser_click", '{"selector": "#buy"}', "Success click", step_index=1, success=True)
    # 4. Artifact
    store.save_artifact(ag_id, "Laptops Summary", "report", "Dell XPS 15 was found to be the best overall.")
    # 5. UserMemory
    store.set_user_memory("preference", "preferred_os", "Linux")
    # 6. Skill
    store.save_skill("login_flow", "auth", "Authenticates user", "def login(): pass")

    # Domain specific queries
    assert len(store.search_memory("Dell", domain="ENTITIES")) > 0
    assert len(store.search_memory("Investigate", domain="HISTORY")) > 0
    assert len(store.search_memory("Laptops", domain="ARTIFACTS")) > 0
    assert len(store.search_memory("preferred_os", domain="USER")) > 0
    assert len(store.search_memory("login_flow", domain="SKILLS")) > 0

    # ALL domain queries
    all_res = store.search_memory("Dell", domain="ALL")
    assert len(all_res) >= 2  # Found in both Entity and Artifact


# -----------------------------------------------------------------------------
# 3. Aggregate Output Budget Enforcement on search_memory (with Extra Metadata)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_memory_aggregate_output_budget(lease_registry):
    mock_store = MagicMock()
    # 20 records with oversized content AND oversized extra metadata fields
    oversized_records = [
        {
            "source_id": f"art_{i}",
            "memory_type": "Artifact",
            "timestamp": "2026-10-10 12:00:00",
            "provenance": "other_agent",
            "content_excerpt": f"Record {i}: " + ("X" * 1500),
            "relevance_score": 1.0 - (i * 0.01),
            "extra_unbounded_metadata": "M" * 500,  # All retained fields must be counted in budget!
            "extra_nested_dict": {"field": "D" * 200},
        }
        for i in range(20)
    ]
    mock_store.search_memory.return_value = oversized_records

    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)
    lease = lease_registry.acquire_lease("w_budget", "tab_budget", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-budget",
        worker_id="w_budget",
        worker_token=lease.lease_token,
        target_tab="tab_budget",
        lease_registry=lease_registry,
    )

    res = await service.search_memory(ctx, query="test", domain="ARTIFACTS", limit=20)
    assert res.success is True
    bounded = res.data
    evidence = res.evidence

    assert evidence["truncated"] is True
    assert evidence["omitted_count"] > 0
    assert len(bounded) < 20
    assert evidence["total_chars"] <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert evidence["budget_chars"] == MAX_AGGREGATE_MEMORY_OUTPUT_CHARS

    # The last bounded item must contain the truncation marker
    last_item = bounded[-1]
    assert MEMORY_TRUNCATION_MARKER in last_item["content_excerpt"]


# -----------------------------------------------------------------------------
# 4. get_context Items Alignment & Content Consistency with Context Text
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_context_items_aligned_with_context_text(lease_registry):
    mock_store = MagicMock()
    mock_items = [
        {
            "source_id": f"art_align_{i}",
            "memory_type": "Artifact",
            "timestamp": "2026-10-10 10:00:00",
            "provenance": f"agent_align_{i}",
            "content_excerpt": f"Large content chunk {i}: " + ("A" * 6000),
            "relevance_score": 0.9,
        }
        for i in range(5)
    ]
    mock_store.search_memory.return_value = mock_items

    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)
    lease = lease_registry.acquire_lease("w_align", "tab_align", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-align",
        worker_id="w_align",
        worker_token=lease.lease_token,
        target_tab="tab_align",
        lease_registry=lease_registry,
    )

    res = await service.get_context(ctx, query="alignment test", domain="ALL", limit=5)
    assert res.success is True
    data = res.data
    context_text = data["context_text"]
    items = data["items"]

    # Budget strictly enforced
    assert len(context_text) <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert res.evidence["truncated"] is True
    assert res.evidence["omitted_count"] > 0

    assert len(items) > 0
    for item in items:
        assert item["source_id"] in context_text

    # Content consistency guarantee:
    # If context_text truncated an item, that item's content_excerpt in `items` MUST match the truncated excerpt!
    last_item = items[-1]
    if MEMORY_TRUNCATION_MARKER in context_text:
        assert MEMORY_TRUNCATION_MARKER in last_item["content_excerpt"]


# -----------------------------------------------------------------------------
# 5. Same-Run and Same-Task Exclusion
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_context_same_run_and_task_exclusion(lease_registry):
    mock_store = MagicMock()
    mock_store.search_memory.return_value = [
        {"source_id": "art_same_run", "memory_type": "Artifact", "provenance": "run_current_123", "content_excerpt": "Same run note"},
        {"source_id": "art_same_task", "memory_type": "Artifact", "source_task_id": "t-excl", "content_excerpt": "Same task note"},
        {"source_id": "art_diff", "memory_type": "Artifact", "provenance": "run_other_456", "source_task_id": "t-other", "content_excerpt": "Other note"},
    ]

    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)
    lease = lease_registry.acquire_lease("w_excl", "tab_excl", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-excl",
        run_id="run_current_123",
        worker_id="w_excl",
        worker_token=lease.lease_token,
        target_tab="tab_excl",
        lease_registry=lease_registry,
    )

    res = await service.get_context(ctx, query="notes")
    assert res.success is True
    items = res.data["items"]
    assert len(items) == 1
    assert items[0]["source_id"] == "art_diff"
    assert "art_same_run" not in res.data["context_text"]
    assert "art_same_task" not in res.data["context_text"]


# -----------------------------------------------------------------------------
# 6. Real Agent-Linked Artifact & Correct Relationship Traversal
# -----------------------------------------------------------------------------

def test_real_agent_linked_artifact_and_relationships(temp_kuzu_store):
    store = temp_kuzu_store

    # 1. Create a real Agent node in Kùzu
    ag_id = "agent_real_1"
    store.create_agent(ag_id, "Real research task", "research")

    # 2. Save artifact linked to this real agent
    art_id = store.save_artifact(
        ag_id,
        "Verified Report",
        "report",
        "Content from real agent.",
        source_url="https://example.com/verified",
        source_run_id="run_v1",
        source_task_id="task_v1",
    )
    assert art_id.startswith("art_")

    # 3. get_artifact returns agent ID, run ID, and task ID
    doc = store.get_artifact(art_id)
    assert doc["source_agent_id"] == ag_id
    assert doc["source_run_id"] == "run_v1"
    assert doc["source_task_id"] == "task_v1"
    assert doc["source_url"] == "https://example.com/verified"

    # 4. search_memory finds artifact and traverses CREATED_ARTIFACT relationship to get provenance
    res = store.search_memory("Verified Report", domain="ARTIFACTS")
    assert len(res) > 0
    found = next((r for r in res if r["source_id"] == art_id), None)
    assert found is not None
    assert found["provenance"] == ag_id
    assert found["source_run_id"] == "run_v1"
    assert found["source_task_id"] == "task_v1"

    # 5. UserMemory and Skill with real Agent: verify LEARNED_FROM and DISTILLED_FROM traversal
    store.set_user_memory("pref", "theme", "dark", agent_id=ag_id)
    um_res = store.search_memory("theme", domain="USER")
    assert len(um_res) > 0
    assert um_res[0]["provenance"] == ag_id  # Verified via -[:LEARNED_FROM]->(ag)

    store.save_skill("scroll_helper", "dom", "Scrolls view", "def scroll(): pass", agent_id=ag_id)
    sk_res = store.search_memory("scroll_helper", domain="SKILLS")
    assert len(sk_res) > 0
    assert sk_res[0]["provenance"] == ag_id  # Verified via -[:DISTILLED_FROM]->(ag)


# -----------------------------------------------------------------------------
# 7. Orphan Run Artifact & Missing Provenance Remains Missing
# -----------------------------------------------------------------------------

def test_orphan_run_artifact_and_missing_provenance_remains_missing(temp_kuzu_store):
    store = temp_kuzu_store

    # Save artifact with NO agent_id, but with source_run_id and source_task_id
    art_id = store.save_artifact(
        agent_id="",
        title="Unlinked Note",
        artifact_type="note",
        content="Note saved without a graph Agent node.",
        source_run_id="run_orphan_99",
        source_task_id="task_orphan_99",
    )
    doc = store.get_artifact(art_id)
    # source_agent_id MUST remain empty, NOT fabricated from run or task!
    assert doc["source_agent_id"] == ""
    assert doc["source_run_id"] == "run_orphan_99"
    assert doc["source_task_id"] == "task_orphan_99"

    # search_memory must preserve provenance as empty string, NOT invent an agent!
    res = store.search_memory("Unlinked Note", domain="ARTIFACTS")
    assert len(res) > 0
    found = next((r for r in res if r["source_id"] == art_id), None)
    assert found is not None
    assert found["provenance"] == ""
    assert found["source_run_id"] == "run_orphan_99"
    assert found["source_task_id"] == "task_orphan_99"

    # Also test nonempty nonexistent agent_id: MUST NOT be stored as valid provenance!
    art_ghost = store.save_artifact(
        agent_id="agent_ghost_nonexistent_999",
        title="Ghost Note",
        artifact_type="note",
        content="Note with nonexistent agent ID.",
        source_run_id="run_ghost_1",
        source_task_id="task_ghost_1",
    )
    doc_ghost = store.get_artifact(art_ghost)
    assert doc_ghost["source_agent_id"] == ""  # Nonexistent agent rejected from source_agent_id!
    assert doc_ghost["source_run_id"] == "run_ghost_1"
    assert doc_ghost["source_task_id"] == "task_ghost_1"

    # Confirm no CREATED_ARTIFACT relationship was created
    rel_check = store.conn.execute(
        "MATCH (ag:Agent)-[:CREATED_ARTIFACT]->(a:Artifact {id: $id}) RETURN count(ag)",
        {"id": art_ghost}
    )
    assert rel_check.has_next()
    assert rel_check.get_next()[0] == 0


@pytest.mark.asyncio
async def test_tool_service_artifact_provenance_and_source_url_roundtrip(lease_registry, temp_kuzu_store):
    service = ComexToolService(lease_registry=lease_registry, graph_store=temp_kuzu_store)
    lease = lease_registry.acquire_lease("w_art", "tab_art", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-art-888",
        run_id="run_art_888",
        worker_id="w_art",
        worker_token=lease.lease_token,
        target_tab="tab_art",
        lease_registry=lease_registry,
    )

    # Save artifact via service
    res_save = await service.save_artifact(
        ctx,
        title="Benchmark Summary",
        content="Results: 99% accuracy across test suites.",
        artifact_type="report",
        source_url="https://benchmark.internal/run/888",
    )
    assert res_save.success is True
    art_id = res_save.data["artifact_id"]
    assert res_save.data["source_agent_id"] == ""
    assert res_save.data["source_run_id"] == "run_art_888"
    assert res_save.data["source_task_id"] == "t-art-888"
    assert res_save.data["source_url"] == "https://benchmark.internal/run/888"

    # Read artifact via service
    res_read = await service.read_artifact(ctx, artifact_id=art_id)
    assert res_read.success is True
    data = res_read.data
    assert data["artifact_id"] == art_id
    assert data["title"] == "Benchmark Summary"
    assert data["source_agent_id"] == ""
    assert data["source_run_id"] == "run_art_888"
    assert data["source_task_id"] == "t-art-888"
    assert data["source_url"] == "https://benchmark.internal/run/888"
    assert data["created_at"] != ""

    # Now create a REAL agent in Kùzu and pass agent_id to ctx
    temp_kuzu_store.create_agent("agent_verified_999", "Benchmark Task", "research")
    ctx_real = create_tool_execution_context(
        task_id="t-art-real",
        run_id="run_art_real",
        worker_id="w_art",
        worker_token=lease.lease_token,
        target_tab="tab_art",
        lease_registry=lease_registry,
        agent_id="agent_verified_999",
    )
    res_save_real = await service.save_artifact(
        ctx_real,
        title="Verified Real Summary",
        content="Real agent report content.",
    )
    assert res_save_real.success is True
    assert res_save_real.data["source_agent_id"] == "agent_verified_999"

    res_read_real = await service.read_artifact(ctx_real, artifact_id=res_save_real.data["artifact_id"])
    assert res_read_real.success is True
    assert res_read_real.data["source_agent_id"] == "agent_verified_999"

    # Verify real graph relationship was created
    rel_real = temp_kuzu_store.conn.execute(
        "MATCH (ag:Agent {id: 'agent_verified_999'})-[:CREATED_ARTIFACT]->(a:Artifact {id: $id}) RETURN count(ag)",
        {"id": res_save_real.data["artifact_id"]}
    )
    assert rel_real.has_next()
    assert rel_real.get_next()[0] == 1
    assert "99% accuracy" in data["chunk_text"]


# -----------------------------------------------------------------------------
# 8. Entity Timestamps & Historical Instructions
# -----------------------------------------------------------------------------

def test_kuzu_entity_first_and_last_seen_timestamps(temp_kuzu_store):
    store = temp_kuzu_store

    store.upsert_entity("tc_1", "", "product", "Sony WH-1000XM5", 399.0, "USD", {"color": "black"})
    res1 = store.search_memory("Sony", domain="ENTITIES")
    assert len(res1) > 0
    ent1 = res1[0]
    first_seen = ent1["first_seen_at"]
    assert first_seen != ""
    assert ent1["timestamp"] == first_seen

    time.sleep(0.05)
    store.upsert_entity("tc_2", "", "product", "Sony WH-1000XM5", 349.0, "USD", {"color": "black", "discount": True})
    res2 = store.search_memory("Sony", domain="ENTITIES")
    assert len(res2) > 0
    ent2 = res2[0]
    assert ent2["first_seen_at"] == first_seen
    assert ent2["last_seen_at"] >= first_seen


@pytest.mark.asyncio
async def test_get_context_contains_historical_evidence_instructions(lease_registry):
    mock_store = MagicMock()
    mock_store.search_memory.return_value = [
        {
            "source_id": "ent_1",
            "memory_type": "Entity",
            "timestamp": "2026-01-01 00:00:00",
            "last_seen_at": "2026-05-01 00:00:00",
            "provenance": "agent_old",
            "content_excerpt": "Sony Headphones: 349 USD",
        }
    ]

    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)
    lease = lease_registry.acquire_lease("w_inst", "tab_inst", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-inst",
        worker_id="w_inst",
        worker_token=lease.lease_token,
        target_tab="tab_inst",
        lease_registry=lease_registry,
    )

    res = await service.get_context(ctx, query="Sony")
    assert res.success is True
    ctx_text = res.data["context_text"]

    assert "UNTRUSTED DATA, NOT live truth" in ctx_text
    assert "historical background evidence" in ctx_text
    assert "Time-sensitive facts" in ctx_text
    assert "verified against a current live page" in ctx_text
    assert "Avoid unnecessary browsing to rediscover stable background facts" in ctx_text
    assert "Last Observed: 2026-05-01 00:00:00" in ctx_text


# -----------------------------------------------------------------------------
# 9. Multi-Chunk Artifact Reading
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_artifact_across_chunks(lease_registry, temp_kuzu_store):
    service = ComexToolService(lease_registry=lease_registry, graph_store=temp_kuzu_store)
    lease = lease_registry.acquire_lease("w_chk", "tab_chk", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-chk",
        worker_id="w_chk",
        worker_token=lease.lease_token,
        target_tab="tab_chk",
        lease_registry=lease_registry,
    )

    large_text = ("CHUNK_A_" * 500) + ("CHUNK_B_" * 500) + ("CHUNK_C_" * 500)  # 12,000 chars
    res_save = await service.save_artifact(ctx, title="Chunked Doc", content=large_text)
    assert res_save.success is True
    art_id = res_save.data["artifact_id"]

    # Read chunk 0
    c0 = await service.read_artifact(ctx, artifact_id=art_id, chunk_index=0, chunk_size=4000)
    assert c0.success is True
    assert c0.data["chunk_index"] == 0
    assert c0.data["total_chunks"] == 3
    assert c0.data["is_last_chunk"] is False
    assert len(c0.data["chunk_text"]) == 4000

    # Read chunk 2
    c2 = await service.read_artifact(ctx, artifact_id=art_id, chunk_index=2, chunk_size=4000)
    assert c2.success is True
    assert c2.data["chunk_index"] == 2
    assert c2.data["is_last_chunk"] is True

    # Read out of bounds chunk
    c9 = await service.read_artifact(ctx, artifact_id=art_id, chunk_index=9, chunk_size=4000)
    assert c9.success is True
    assert c9.data["chunk_text"] == ""


# -----------------------------------------------------------------------------
# 10. Artifact Write Cancellation & Compensation Preservation
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_artifact_compensation_on_timeout(lease_registry):
    store_entered = threading.Event()
    allow_finish = threading.Event()
    mock_store = MagicMock()

    def slow_save(agent_id, title, atype, content, source_url="", source_run_id="", source_task_id=""):
        store_entered.set()
        allow_finish.wait(timeout=5.0)
        return "art_zombie_999"

    mock_store.save_artifact.side_effect = slow_save
    mock_store.delete_artifact.return_value = True

    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)
    lease = lease_registry.acquire_lease("w_comp", "tab_comp", trusted_issuer=True)
    cancel_evt = asyncio.Event()

    ctx = create_tool_execution_context(
        task_id="t-comp",
        worker_id="w_comp",
        worker_token=lease.lease_token,
        target_tab="tab_comp",
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Zombie Doc", content="Roll me back"))

    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    cancel_evt.set()
    allow_finish.set()

    res = await task
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "ROLLED_BACK_CLEAN"
    assert res.evidence.get("persisted") is False
    assert res.evidence.get("compensated_artifact_id") == "art_zombie_999"
    mock_store.delete_artifact.assert_called_once_with("art_zombie_999")


# -----------------------------------------------------------------------------
# 11. Shared Store Concurrency with Payload Verification
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_reads_and_writes_shared_store(lease_registry, temp_kuzu_store):
    service = ComexToolService(lease_registry=lease_registry, graph_store=temp_kuzu_store)

    lease1 = lease_registry.acquire_lease("w_conc_1", "tab_conc_1", trusted_issuer=True)
    lease2 = lease_registry.acquire_lease("w_conc_2", "tab_conc_2", trusted_issuer=True)

    ctx1 = create_tool_execution_context(
        task_id="t-conc-1",
        worker_id="w_conc_1",
        worker_token=lease1.lease_token,
        target_tab="tab_conc_1",
        lease_registry=lease_registry,
    )
    ctx2 = create_tool_execution_context(
        task_id="t-conc-2",
        worker_id="w_conc_2",
        worker_token=lease2.lease_token,
        target_tab="tab_conc_2",
        lease_registry=lease_registry,
    )

    async def writer(worker_ctx, prefix: str):
        ids = []
        for i in range(5):
            res = await service.save_artifact(
                worker_ctx,
                title=f"{prefix} Note {i}",
                content=f"Payload from {prefix} {i}",
            )
            assert res.success is True
            ids.append(res.data["artifact_id"])
        return ids

    async def reader(worker_ctx):
        reads = []
        for i in range(5):
            res = await service.search_memory(worker_ctx, query="Note", domain="ARTIFACTS")
            assert res.success is True
            reads.append(res.data)
        return reads

    results = await asyncio.gather(
        writer(ctx1, "Alpha"),
        writer(ctx2, "Beta"),
        reader(ctx1),
        reader(ctx2),
    )

    alpha_ids, beta_ids, reads1, reads2 = results
    assert len(alpha_ids) == 5
    assert len(beta_ids) == 5
    assert len(set(alpha_ids + beta_ids)) == 10

    # Verify content of all written artifacts
    for art_id in alpha_ids + beta_ids:
        doc = temp_kuzu_store.get_artifact(art_id)
        assert doc["content"] != ""
        assert "Payload" in doc["content"]


# -----------------------------------------------------------------------------
# 12. Authorization, Cancellation, and Deadline Checks
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_revoked_authorization_blocks_memory_operations(lease_registry):
    mock_store = MagicMock()
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_revoked", "tab_revoked", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-revoked",
        worker_id="w_revoked",
        worker_token=lease.lease_token,
        target_tab="tab_revoked",
        lease_registry=lease_registry,
    )

    lease_registry.revoke_lease("w_revoked", trusted_caller=True)

    res_search = await service.search_memory(ctx, query="test")
    assert res_search.success is False
    assert res_search.error_code == ToolErrorCode.UNAUTHORIZED.value
    mock_store.search_memory.assert_not_called()

    res_get = await service.get_context(ctx, query="test")
    assert res_get.success is False
    assert res_get.error_code == ToolErrorCode.UNAUTHORIZED.value
    mock_store.search_memory.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_context_read_fails_closed(lease_registry):
    mock_store = MagicMock()
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    cancel_evt = asyncio.Event()
    cancel_evt.set()

    lease = lease_registry.acquire_lease("w_cancel", "tab_cancel", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-cancel",
        worker_id="w_cancel",
        worker_token=lease.lease_token,
        target_tab="tab_cancel",
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    res = await service.search_memory(ctx, query="test")
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    mock_store.search_memory.assert_not_called()


@pytest.mark.asyncio
async def test_in_flight_read_deadline_expiration(lease_registry):
    mock_store = MagicMock()

    def slow_search(*args, **kwargs):
        time.sleep(0.2)
        return []

    mock_store.search_memory.side_effect = slow_search
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_slow", "tab_slow", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-slow",
        worker_id="w_slow",
        worker_token=lease.lease_token,
        target_tab="tab_slow",
        timeout_seconds=0.05,  # Deadline will expire while search is in flight
        lease_registry=lease_registry,
    )

    res = await service.search_memory(ctx, query="slow")
    assert res.success is False
    assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value


# -----------------------------------------------------------------------------
# 13. Backend Failure Semantics & Malformed Store Results
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_graph_store_explicit_failure(lease_registry):
    service_no_db = ComexToolService(lease_registry=lease_registry, graph_store=None)

    lease = lease_registry.acquire_lease("w_nodb", "tab_nodb", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-nodb",
        worker_id="w_nodb",
        worker_token=lease.lease_token,
        target_tab="tab_nodb",
        lease_registry=lease_registry,
    )

    res1 = await service_no_db.search_memory(ctx, query="test")
    assert res1.success is False
    assert res1.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "Graph store not configured" in res1.error_message

    res2 = await service_no_db.get_context(ctx, query="test")
    assert res2.success is False
    assert res2.error_code == ToolErrorCode.OPERATION_FAILED.value


@pytest.mark.asyncio
async def test_malformed_top_level_backend_results_fail_explicitly(lease_registry):
    mock_store = MagicMock()
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_top", "tab_top", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-top",
        worker_id="w_top",
        worker_token=lease.lease_token,
        target_tab="tab_top",
        lease_registry=lease_registry,
    )

    # When store returns None, dict, or string instead of list, it MUST fail explicitly
    for malformed_val in [None, {"status": "error"}, "unexpected string"]:
        mock_store.search_memory.return_value = malformed_val
        res = await service.search_memory(ctx, query="test")
        assert res.success is False
        assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
        assert "expected list" in res.error_message

        res_ctx = await service.get_context(ctx, query="test")
        assert res_ctx.success is False
        assert res_ctx.error_code == ToolErrorCode.OPERATION_FAILED.value


@pytest.mark.asyncio
async def test_store_search_exception_and_total_domain_failure(lease_registry):
    mock_store = MagicMock()
    mock_store.search_memory.side_effect = RuntimeError("Kùzu disk read error")
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_err", "tab_err", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-err",
        worker_id="w_err",
        worker_token=lease.lease_token,
        target_tab="tab_err",
        lease_registry=lease_registry,
    )

    res = await service.search_memory(ctx, query="test")
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "Kùzu disk read error" in res.error_message


@pytest.mark.asyncio
async def test_partial_domain_failure_observable(lease_registry):
    mock_store = MagicMock()
    from graph.store import AgentGraphStore

    # Mock results with failed_domains attribute
    class MockResultList(list):
        failed_domains = ["Skill"]

    mock_res = MockResultList([
        {"source_id": "art_1", "memory_type": "Artifact", "content_excerpt": "Good artifact"}
    ])
    mock_store.search_memory.return_value = mock_res
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_part", "tab_part", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-part",
        worker_id="w_part",
        worker_token=lease.lease_token,
        target_tab="tab_part",
        lease_registry=lease_registry,
    )

    res = await service.search_memory(ctx, query="test", domain="ALL")
    assert res.success is True
    assert res.evidence.get("partial_failure") is True
    assert "Skill" in res.evidence.get("failed_domains", [])


def test_malformed_stored_entity_attributes_defensive(temp_kuzu_store):
    store = temp_kuzu_store

    # Insert entity with malformed non-JSON attrs directly
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with store._lock:
        store.conn.execute("""
        CREATE (e:Entity {id: 'ent_corrupt'})
        SET e.type = 'product',
            e.name = 'Corrupted Phone',
            e.price = 100.0,
            e.currency = 'USD',
            e.attrs = '{corrupted: non-json,',
            e.canonical_key = 'corrupt_key',
            e.first_seen_at = timestamp($now),
            e.last_seen_at = timestamp($now)
        """, {"now": now})

    # Search should NOT raise an unhandled JSONDecodeError and should return the entity
    res = store.search_memory("Corrupted", domain="ENTITIES")
    assert len(res) == 1
    assert res[0]["source_id"] == "ent_corrupt"


def test_missing_timestamps_do_not_become_string_none(temp_kuzu_store):
    store = temp_kuzu_store

    # Insert artifact without created_at timestamp
    with store._lock:
        store.conn.execute("""
        CREATE (a:Artifact {id: 'art_no_ts'})
        SET a.title = 'Timeless Artifact',
            a.artifact_type = 'note',
            a.content = 'No timestamp content.'
        """)

    doc = store.get_artifact("art_no_ts")
    assert doc["created_at"] == ""  # Must NOT be "None"!

    res = store.search_memory("Timeless", domain="ARTIFACTS")
    assert len(res) == 1
    assert res[0]["timestamp"] == ""  # Must NOT be "None"!


# -----------------------------------------------------------------------------
# 14. Real Database Schema Upgrade Migration Test
# -----------------------------------------------------------------------------

def test_schema_migration_from_old_database():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "old_schema.db")
    try:
        # 1. Create a database using the OLD schema without source_url, source_run_id, source_task_id
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)

        old_artifact_ddl = """
        CREATE NODE TABLE IF NOT EXISTS Artifact(
            id STRING,
            artifact_type STRING,
            title STRING,
            content STRING,
            created_at TIMESTAMP,
            source_agent_id STRING,
            PRIMARY KEY(id)
        )
        """
        conn.execute(old_artifact_ddl)

        # Insert an artifact using old schema
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
        CREATE (a:Artifact {id: 'art_legacy_1'})
        SET a.artifact_type = 'report',
            a.title = 'Legacy Report',
            a.content = 'Created under old schema.',
            a.created_at = timestamp($now),
            a.source_agent_id = 'agent_old_1'
        """, {"now": now})

        # 2. Run real migrations on this database
        run_migrations(conn)

        # Verify schema version was updated to 4
        res_ver = conn.execute("MATCH (s:SchemaMeta {key: 'schema_version'}) RETURN s.value")
        assert res_ver.has_next()
        assert res_ver.get_next()[0] == "4"

        # 3. Verify old artifact is readable and new columns defaulted to ""
        res = conn.execute("MATCH (a:Artifact {id: 'art_legacy_1'}) RETURN a.title, a.source_url, a.source_run_id, a.source_task_id")
        assert res.has_next()
        row = res.get_next()
        assert row[0] == "Legacy Report"
        assert row[1] == ""  # source_url defaulted
        assert row[2] == ""  # source_run_id defaulted
        assert row[3] == ""  # source_task_id defaulted

        # 4. Now instantiate AgentGraphStore on this upgraded database
        store = AgentGraphStore(db_path)
        legacy_doc = store.get_artifact("art_legacy_1")
        assert legacy_doc["title"] == "Legacy Report"
        assert legacy_doc["source_agent_id"] == "agent_old_1"
        assert legacy_doc["source_url"] == ""

        # 5. Insert new artifact with all new provenance columns
        store.create_agent("agent_new_2", "New Task", "new_intent")
        new_id = store.save_artifact(
            "agent_new_2",
            "New Schema Report",
            "report",
            "Created under upgraded schema.",
            source_url="https://example.com/upgraded",
            source_run_id="run_upgraded_1",
            source_task_id="task_upgraded_1",
        )
        new_doc = store.get_artifact(new_id)
        assert new_doc["source_agent_id"] == "agent_new_2"
        assert new_doc["source_url"] == "https://example.com/upgraded"
        assert new_doc["source_run_id"] == "run_upgraded_1"
        assert new_doc["source_task_id"] == "task_upgraded_1"

        # 6. Verify that _safe_add_column raises on genuine errors (e.g. non-existent table)
        with pytest.raises(RuntimeError):
            _safe_add_column(conn, "NonExistentTable", "col_x", "STRING")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 15. Agent ID Claim Binding and Tampering Rejection
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_id_claim_tampering_rejected(lease_registry):
    mock_store = MagicMock()
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_agent_claim", "tab_agent_claim", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-claim",
        worker_id="w_agent_claim",
        worker_token=lease.lease_token,
        target_tab="tab_agent_claim",
        lease_registry=lease_registry,
        agent_id="agent_legit_1",
    )

    # Context with tampered agent_id
    tampered_ctx = ToolExecutionContext(
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        worker_id=ctx.worker_id,
        target_tab_id=ctx.target_tab_id,
        lease=ctx.lease,
        permitted_operations=ctx.permitted_operations,
        deadline_mono=ctx.deadline_mono,
        cancellation_event=ctx.cancellation_event,
        is_single_agent=ctx.is_single_agent,
        context_token=ctx.context_token,
        agent_id="agent_forged_999",  # Tampered!
    )

    res = await service.search_memory(tampered_ctx, query="test")
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Context agent_id does not match authoritative record" in res.error_message


# -----------------------------------------------------------------------------
# 16. get_context Partial Domain Failure Propagation
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_context_partial_domain_failure_observable(lease_registry):
    mock_store = MagicMock()

    class MockResultList(list):
        failed_domains = ["Skill", "UserMemory"]

    mock_res = MockResultList([
        {"source_id": "art_part_1", "memory_type": "Artifact", "content_excerpt": "Good artifact text"}
    ])
    mock_store.search_memory.return_value = mock_res
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_part_ctx", "tab_part_ctx", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-part-ctx",
        worker_id="w_part_ctx",
        worker_token=lease.lease_token,
        target_tab="tab_part_ctx",
        lease_registry=lease_registry,
    )

    res = await service.get_context(ctx, query="test", domain="ALL")
    assert res.success is True
    assert "[PERSISTENT MEMORY CONTEXT]" in res.data["context_text"]
    assert res.evidence.get("partial_failure") is True
    assert "Skill" in res.evidence.get("failed_domains", [])
    assert "UserMemory" in res.evidence.get("failed_domains", [])


# -----------------------------------------------------------------------------
# 17. save_artifact Does Not Retry on Store Internal TypeError
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_artifact_does_not_retry_on_internal_type_error(lease_registry):
    mock_store = MagicMock()
    call_count = 0

    def fail_with_type_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise TypeError("Simulated internal TypeError in store business logic")

    mock_store.save_artifact = fail_with_type_error
    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)

    lease = lease_registry.acquire_lease("w_no_retry", "tab_no_retry", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-no-retry",
        worker_id="w_no_retry",
        worker_token=lease.lease_token,
        target_tab="tab_no_retry",
        lease_registry=lease_registry,
    )

    res = await service.save_artifact(ctx, title="Title", content="Content")
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "Simulated internal TypeError" in res.error_message
    assert call_count == 1  # Exactly ONE call, no retries!


# -----------------------------------------------------------------------------
# 18. Complete Memory-Output Budget with Oversized Metadata
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_payload_budget_with_oversized_metadata(lease_registry):
    mock_store = MagicMock()
    # 5 records with long excerpts (5000 chars each) + extremely long metadata keys (1500 chars)
    # + massive extra metadata (10KB) that doesn't appear in context_text.
    # Note: 2 records of 5000 chars would yield context_text (~10.3KB) and items (~10.2KB)
    # which individually fit under 16,000 chars, but together exceed 20,500 chars!
    extremely_long_key = "K" * 1500
    mock_items = [
        {
            "source_id": f"art_huge_meta_{i}",
            "memory_type": "Artifact",
            "timestamp": "2026-10-10 10:00:00",
            "provenance": f"agent_{i}",
            "content_excerpt": f"Chunk {i}: " + ("A" * 5000),
            extremely_long_key: "huge_key_value",
            "massive_unrendered_metadata": "Z" * 10000,  # 10KB extra metadata!
            "nested_unrendered_data": {"k": "Y" * 5000},
        }
        for i in range(5)
    ]
    mock_store.search_memory.return_value = mock_items

    service = ComexToolService(lease_registry=lease_registry, graph_store=mock_store)
    lease = lease_registry.acquire_lease("w_meta_budget", "tab_meta_budget", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="t-meta-budget",
        worker_id="w_meta_budget",
        worker_token=lease.lease_token,
        target_tab="tab_meta_budget",
        lease_registry=lease_registry,
    )

    # Test get_context: complete returned data payload MUST be <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    res_ctx = await service.get_context(ctx, query="meta test", domain="ALL", limit=5)
    assert res_ctx.success is True
    data = res_ctx.data
    context_text = data["context_text"]
    items = data["items"]

    # Shared ceiling: complete serialized data payload must fit within budget
    complete_payload_chars = len(json.dumps(res_ctx.data, default=str))
    assert complete_payload_chars <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert res_ctx.evidence["total_chars"] == complete_payload_chars
    assert res_ctx.evidence["budget_chars"] == MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert res_ctx.evidence["truncated"] is True
    assert res_ctx.evidence["omitted_count"] > 0

    # Components individually also fit within the ceiling
    assert len(context_text) <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert res_ctx.evidence["context_text_chars"] == len(context_text)
    items_serialized_len = len(json.dumps(items, default=str))
    assert items_serialized_len <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert res_ctx.evidence["items_total_chars"] == items_serialized_len

    # Truncated excerpts stay synchronized between context_text and items
    last_item = items[-1]
    assert MEMORY_TRUNCATION_MARKER in last_item["content_excerpt"]
    assert MEMORY_TRUNCATION_MARKER in context_text
    assert last_item["content_excerpt"] in context_text

    # Extremely long metadata keys and values are clamped
    for it in items:
        for k in it.keys():
            assert len(k) <= 150  # Clamped to <= 100 + marker
        assert len(it.get("massive_unrendered_metadata", "")) < 300

    # Test search_memory: total serialized JSON is strictly bounded
    res_search = await service.search_memory(ctx, query="meta test", domain="ALL", limit=5)
    assert res_search.success is True
    search_data = res_search.data
    search_serialized_len = len(json.dumps(search_data, default=str))
    assert search_serialized_len <= MAX_AGGREGATE_MEMORY_OUTPUT_CHARS
    assert res_search.evidence["total_chars"] == search_serialized_len

    # Empty search accounting consistency: empty results report serialized length len('[]') == 2
    mock_store.search_memory.return_value = []
    res_empty = await service.search_memory(ctx, query="empty test", domain="ALL")
    assert res_empty.success is True
    assert res_empty.data == []
    empty_chars = len(json.dumps([], default=str))
    assert res_empty.evidence["total_chars"] == empty_chars
    assert res_empty.evidence["total_chars"] == 2

