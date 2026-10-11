"""
Comprehensive Deterministic Integration Tests for Comex Multi-Agent Orchestrator MVP.

Tests verify:
1. Two independent browser tasks created and executed.
2. Separate authorized tabs and distinct worker leases.
3. Genuine concurrent overlap (mathematical interval overlap & synchronization barrier).
4. Structured worker results with source evidence.
5. Correct result aggregation and summary synthesis.
6. Failure containment: Worker A failure does NOT corrupt Worker B's result.
7. Dependency ordering: Dependent task waits for prerequisites.
8. Cancellation cleanup: Tasks cancelled and leases reclaimed immediately.
9. Cross-tab authorization rejection: Worker A cannot access Worker B's tab.
10. Sensitive action gating: Unapproved sensitive actions are blocked.
11. Cycle and invalid dependency rejection.
12. API endpoint (/agent/orchestrate) integration.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import agent_service
from nervous.action_policy import (
    ServerTabLeaseRegistry,
    ServerTabTargetRegistry,
    TabLease,
)
from services.orchestrator import (
    AggregatedResult,
    MultiAgentOrchestrator,
    OrchestrationRun,
    OrchestratorTask,
    RunStatus,
    TaskStatus,
    WorkerResult,
    decompose_request_into_tasks,
    validate_dependency_graph,
)
from services.tool_service import (
    ComexToolService,
    ToolErrorCode,
    ToolExecutionContext,
    ToolResult,
    create_tool_execution_context,
)


@pytest.fixture
def lease_registry():
    return ServerTabLeaseRegistry()


@pytest.fixture
def target_registry():
    return ServerTabTargetRegistry()


@pytest.fixture
def mock_tool_service(lease_registry, target_registry):
    svc = MagicMock(spec=ComexToolService)
    svc._lease_registry = lease_registry
    svc._target_registry = target_registry

    # Standard tool mocks returning successful ToolResult
    async def mock_navigate(ctx, url):
        return ToolResult(success=True, data={"navigated": True, "url": url, "title": f"Page for {url}"})

    async def mock_inspect(ctx, max_elements=50):
        return ToolResult(success=True, data={"returned": 5, "elements": [{"tag": "h1"}, {"tag": "p"}]})

    async def mock_extract(ctx, selector="", fields=None):
        return ToolResult(success=True, data={"text": "Sample specifications and review details", "url": "https://example.com"})

    svc.navigate = AsyncMock(side_effect=mock_navigate)
    svc.inspect = AsyncMock(side_effect=mock_inspect)
    svc.extract = AsyncMock(side_effect=mock_extract)
    return svc


@pytest.fixture
def orchestrator(mock_tool_service, lease_registry, target_registry):
    return MultiAgentOrchestrator(
        tool_service=mock_tool_service,
        lease_registry=lease_registry,
        target_registry=target_registry,
        max_concurrency=2,
    )


# ---------------------------------------------------------------------------
# Test 1 & 2: Two independent browser tasks with distinct tabs and leases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_workers_separate_tabs_and_leases(orchestrator, lease_registry):
    """
    Req 1 & 2: A run creates two independent browser tasks, each receiving
    a distinct worker ID, separate authorized tab, and exclusive tab lease.
    """
    assigned_leases = {}

    async def task_a_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        assigned_leases["worker_a"] = {
            "worker_id": ctx.worker_id,
            "tab_id": ctx.target_tab_id,
            "lease": ctx.lease,
        }
        await svc.navigate(ctx, "https://specs.example.com/laptop")
        return WorkerResult(
            findings={"cpu": "Intel Core i7-13700H", "ram": "16GB LPDDR5", "battery": "72Wh"},
            source_urls=["https://specs.example.com/laptop"],
            success=True,
            tool_calls_count=1,
        )

    async def task_b_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        assigned_leases["worker_b"] = {
            "worker_id": ctx.worker_id,
            "tab_id": ctx.target_tab_id,
            "lease": ctx.lease,
        }
        await svc.navigate(ctx, "https://reviews.example.com/laptop")
        return WorkerResult(
            findings={"price": "₹76,990", "rating": 4.6, "verdict": "Great thermals, solid keyboard"},
            source_urls=["https://reviews.example.com/laptop"],
            success=True,
            tool_calls_count=1,
        )

    task_a = OrchestratorTask(id="task_a", description="Research specifications", task_fn=task_a_fn)
    task_b = OrchestratorTask(id="task_b", description="Research pricing & reviews", task_fn=task_b_fn)

    run = orchestrator.create_run(
        user_request="Find best programming laptop under 80000",
        tasks=[task_a, task_b],
    )
    result = await orchestrator.execute_run(run)

    assert result.has_failures is False
    assert run.status == RunStatus.COMPLETED

    # Verify both workers executed with distinct identities and separate tabs
    w_a = assigned_leases["worker_a"]
    w_b = assigned_leases["worker_b"]
    assert w_a["worker_id"] != w_b["worker_id"]
    assert w_a["tab_id"] != w_b["tab_id"]
    # Tab ids are derived from the task id and made unique per run
    assert w_a["tab_id"].startswith("tab_task_a_")
    assert w_b["tab_id"].startswith("tab_task_b_")

    # Verify leases were active during execution and released upon completion
    assert lease_registry.get_active_lease(w_a["worker_id"]) is None
    assert lease_registry.get_active_lease(w_b["worker_id"]) is None


# ---------------------------------------------------------------------------
# Test 3: Concurrency Proof (Barrier Synchronization & Time Overlap)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_genuine_concurrent_execution_overlap(orchestrator):
    """
    Req 3: Proves tasks execute concurrently rather than sequentially.
    Uses an asyncio.Barrier(2) to ensure both tasks are actively running
    at the exact same physical moment, plus interval overlap verification.
    """
    barrier = asyncio.Barrier(2)
    task_times = {}

    async def worker_1_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        t_start = time.monotonic()
        # Wait at barrier for worker 2 to arrive simultaneously
        await asyncio.wait_for(barrier.wait(), timeout=2.0)
        await asyncio.sleep(0.08)  # Active work
        t_end = time.monotonic()
        task_times["w1"] = (t_start, t_end)
        return WorkerResult(findings="Worker 1 findings", success=True)

    async def worker_2_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        t_start = time.monotonic()
        # Wait at barrier for worker 1 to arrive simultaneously
        await asyncio.wait_for(barrier.wait(), timeout=2.0)
        await asyncio.sleep(0.08)  # Active work
        t_end = time.monotonic()
        task_times["w2"] = (t_start, t_end)
        return WorkerResult(findings="Worker 2 findings", success=True)

    task_1 = OrchestratorTask(id="task_1", description="Worker 1 concurrent task", task_fn=worker_1_fn)
    task_2 = OrchestratorTask(id="task_2", description="Worker 2 concurrent task", task_fn=worker_2_fn)

    run = orchestrator.create_run(user_request="Concurrent test", tasks=[task_1, task_2])
    agg = await orchestrator.execute_run(run)

    assert agg.has_failures is False
    s1, e1 = task_times["w1"]
    s2, e2 = task_times["w2"]

    # Mathematical interval overlap assertion: max(start) < min(end)
    overlap = min(e1, e2) - max(s1, s2)
    assert overlap > 0.0, f"Expected positive time overlap, got {overlap}s"
    print(f"\n[CONCURRENCY PROOF] Overlap duration: {overlap*1000:.1f}ms between Worker 1 and Worker 2.")


# ---------------------------------------------------------------------------
# Test 4 & 5: Structured Results and Aggregator Synthesis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_aggregation_and_sources(orchestrator):
    """
    Req 4 & 5: Structured findings and source URLs are consolidated cleanly.
    """
    async def task_specs_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        return WorkerResult(
            findings="M3 chip with 16GB unified memory",
            source_urls=["https://apple.com/macbook-air"],
            success=True,
        )

    async def task_reviews_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        return WorkerResult(
            findings="Rated 4.8/5 on Wirecutter",
            source_urls=["https://nytimes.com/wirecutter/reviews"],
            success=True,
        )

    task_1 = OrchestratorTask(id="task_specs", description="Specs", task_fn=task_specs_fn)
    task_2 = OrchestratorTask(id="task_reviews", description="Reviews", task_fn=task_reviews_fn)

    run = orchestrator.create_run(user_request="MacBook review", tasks=[task_1, task_2])
    agg = await orchestrator.execute_run(run)

    assert "M3 chip" in agg.summary
    assert "Wirecutter" in agg.summary
    assert len(agg.sources) == 2
    assert "https://apple.com/macbook-air" in agg.sources
    assert "https://nytimes.com/wirecutter/reviews" in agg.sources
    assert len(agg.missing_information) == 0


# ---------------------------------------------------------------------------
# Test 6: Failure Containment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_failure_containment(orchestrator):
    """
    Req 6: Failure in Worker A does not corrupt or halt independent Worker B.
    """
    async def failing_worker(ctx: ToolExecutionContext, svc: ComexToolService):
        raise RuntimeError("Network timeout on retail search")

    async def healthy_worker(ctx: ToolExecutionContext, svc: ComexToolService):
        return WorkerResult(
            findings="Benchmark score: 14,250",
            source_urls=["https://geekbench.com/scores"],
            success=True,
        )

    task_fail = OrchestratorTask(id="t_fail", description="Failing task", task_fn=failing_worker)
    task_ok = OrchestratorTask(id="t_ok", description="Healthy task", task_fn=healthy_worker)

    run = orchestrator.create_run(user_request="Resilience test", tasks=[task_fail, task_ok])
    agg = await orchestrator.execute_run(run)

    assert agg.has_failures is True
    assert run.tasks["t_fail"].status == TaskStatus.FAILED
    assert "Network timeout" in run.tasks["t_fail"].error
    assert run.tasks["t_ok"].status == TaskStatus.COMPLETED
    assert run.tasks["t_ok"].result.findings == "Benchmark score: 14,250"
    assert "t_fail" in agg.missing_information[0]


# ---------------------------------------------------------------------------
# Test 7: Dependency Sequencing (Prerequisites & Blocked State)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependent_task_sequencing_and_blocking(orchestrator):
    """
    Req 7: Dependent task C strictly waits for prerequisite task A and B.
    If prerequisite fails, dependent task transitions to BLOCKED.
    """
    execution_order = []

    async def task_1_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        execution_order.append("task_1_start")
        await asyncio.sleep(0.05)
        execution_order.append("task_1_end")
        return WorkerResult(findings="Candidate models identified", success=True)

    async def task_2_dep_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        execution_order.append("task_2_start")
        return WorkerResult(findings="Detailed comparison completed", success=True)

    t1 = OrchestratorTask(id="t1", description="Find candidates", task_fn=task_1_fn)
    t2 = OrchestratorTask(id="t2", description="Compare candidates", dependencies=["t1"], task_fn=task_2_dep_fn)

    run = orchestrator.create_run(user_request="Sequential test", tasks=[t1, t2])
    agg = await orchestrator.execute_run(run)

    assert agg.has_failures is False
    assert execution_order == ["task_1_start", "task_1_end", "task_2_start"]

    # Now verify blocking if prerequisite fails
    async def broken_t1(ctx, svc):
        return WorkerResult(findings=None, success=False, error="Candidate fetch failed")

    t1_fail = OrchestratorTask(id="t1_fail", description="Failing prerequisite", task_fn=broken_t1)
    t2_blocked = OrchestratorTask(id="t2_blocked", description="Dependent task", dependencies=["t1_fail"], task_fn=task_2_dep_fn)

    run2 = orchestrator.create_run(user_request="Block test", tasks=[t1_fail, t2_blocked])
    agg2 = await orchestrator.execute_run(run2)

    assert run2.tasks["t2_blocked"].status == TaskStatus.BLOCKED
    assert "Prerequisite task 't1_fail' was not completed" in run2.tasks["t2_blocked"].error


# ---------------------------------------------------------------------------
# Test 8: Cooperative Cancellation & Lease Reclamation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancellation_and_lease_cleanup(orchestrator, lease_registry):
    """
    Req 8: Cancelling a run propagates to tasks and reclaims all allocated leases.
    """
    worker_started = asyncio.Event()

    async def long_worker(ctx: ToolExecutionContext, svc: ComexToolService):
        worker_started.set()
        while not ctx.cancellation_event.is_set():
            await asyncio.sleep(0.05)
        return WorkerResult(findings="Should not finish", success=False)

    t_long = OrchestratorTask(id="t_long", description="Long task", task_fn=long_worker)
    run = orchestrator.create_run(user_request="Cancel test", tasks=[t_long])

    exec_task = asyncio.create_task(orchestrator.execute_run(run))
    await worker_started.wait()

    # Cancel the run while worker is running
    orchestrator.cancel_run(run.id)
    await exec_task

    assert run.status == RunStatus.CANCELLED
    assert run.tasks["t_long"].status == TaskStatus.CANCELLED
    # Verify no leaked leases remain
    assert len([l for l in lease_registry._leases.values() if l.is_active]) == 0


# ---------------------------------------------------------------------------
# Test 9: Cross-Tab Authorization Rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_cross_tab_access_rejected(lease_registry):
    """
    Req 9: Worker A holding a lease for Tab A cannot construct a valid context
    or perform operations on Tab B.
    """
    # Issue lease for worker_a on tab_a
    lease_a = lease_registry.acquire_lease(
        worker_id="worker_a",
        tab_id="tab_a",
        trusted_issuer=True,
    )

    # Worker A attempts to create context targeting tab_b
    with pytest.raises(PermissionError) as exc_info:
        create_tool_execution_context(
            task_id="task_exploit",
            run_id="run_1",
            worker_id="worker_a",
            target_tab="tab_b",  # Tampered tab target!
            worker_token=lease_a.lease_token,
            lease_registry=lease_registry,
        )

    assert "Worker 'worker_a' is leased tab 'tab_a', not 'tab_b'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 10: Sensitive Action Gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sensitive_action_gating(orchestrator):
    """
    External sensitive actions (purchases, payments) are rejected if unapproved.
    """
    async def purchase_fn(ctx, svc):
        return WorkerResult(findings="Book purchased", success=True)

    task_sensitive = OrchestratorTask(
        id="task_checkout",
        description="Checkout and buy item",
        sensitive_action=True,
        approved=False,  # Unapproved!
        task_fn=purchase_fn,
    )

    run = orchestrator.create_run(user_request="Buy book", tasks=[task_sensitive])
    agg = await orchestrator.execute_run(run)

    assert run.tasks["task_checkout"].status == TaskStatus.FAILED
    assert "requires explicit user approval" in run.tasks["task_checkout"].error


# ---------------------------------------------------------------------------
# Test 11: Dependency Cycle Rejection
# ---------------------------------------------------------------------------

def test_dependency_cycle_detection(orchestrator):
    """
    Cycle detection rejects circular task graphs before execution.
    """
    t_a = OrchestratorTask(id="a", description="A", dependencies=["b"])
    t_b = OrchestratorTask(id="b", description="B", dependencies=["a"])

    with pytest.raises(ValueError) as exc:
        orchestrator.create_run(user_request="Cycle test", tasks=[t_a, t_b])
    assert "Cyclic dependency detected" in str(exc.value)

    # Self-dependency
    t_self = OrchestratorTask(id="self_dep", description="Self", dependencies=["self_dep"])
    with pytest.raises(ValueError) as exc_self:
        orchestrator.create_run(user_request="Self cycle", tasks=[t_self])
    assert "cannot depend on itself" in str(exc_self.value)


# ---------------------------------------------------------------------------
# Test 12: API Integration Endpoint (/agent/orchestrate)
# ---------------------------------------------------------------------------

def test_api_orchestrate_endpoint():
    """
    Verify /agent/orchestrate executes via FastAPI HTTP test client.
    """
    client = TestClient(agent_service.app)

    async def mock_execute(run):
        run.status = RunStatus.COMPLETED
        return AggregatedResult(
            summary="Two workers finished laptop comparison.",
            task_results={
                "task_1": WorkerResult(findings="Specs: 16GB RAM", source_urls=["https://specs.com"], success=True, worker_id="w1", tab_id="tab1"),
                "task_2": WorkerResult(findings="Reviews: 4.7/5", source_urls=["https://reviews.com"], success=True, worker_id="w2", tab_id="tab2"),
            },
            sources=["https://specs.com", "https://reviews.com"],
            missing_information=[],
            has_failures=False,
            execution_duration_ms=150.0,
        )

    with patch.object(MultiAgentOrchestrator, "execute_run", side_effect=mock_execute):
        res = client.post("/agent/orchestrate", json={
            "user_request": "Compare best gaming laptops under 100k",
            "tasks": [
                {"id": "task_1", "description": "Check specs"},
                {"id": "task_2", "description": "Check reviews"}
            ],
            "contextToken": agent_service._INTERNAL_APP_TOKEN,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "Two workers finished" in data["summary"]
        assert len(data["sources"]) == 2
        assert "task_1" in data["task_results"]
        assert "task_2" in data["task_results"]
