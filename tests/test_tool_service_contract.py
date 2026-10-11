"""
Unit tests for ComexToolService contracts, execution context, and minimal skeleton.
Direct in-process Python unit tests using controlled doubles without live browser or LLM.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from nervous.action_policy import (
    ServerTabLeaseRegistry,
    ServerTabTargetRegistry,
    TabLease,
)
from nervous.actuator_gate import GateResult
from nervous.reflex_engine import DecisionType, Priority, ReflexDecision
from nervous.target_binding import TargetBinding
from services.tool_service import (
    ComexToolService,
    DEFAULT_PERMITTED_OPERATIONS,
    MAX_ARTIFACT_CHUNK_SIZE,
    MAX_ARTIFACT_CONTENT_SIZE,
    MAX_EXTRACT_CHARS,
    MAX_INSPECT_ELEMENTS,
    MAX_MEMORY_RESULTS,
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
def mock_tab_session():
    session = MagicMock()
    session.tab_id = "worker-tab-1"
    session.target_id = "target-123"
    session.action_lock = asyncio.Lock()

    # World state tracker mock
    tracker = MagicMock()
    tracker.state.is_target_subtree_quiet = True
    tracker.state.current_url = "https://example.com"
    tracker.state.current_title = "Example"
    session.tracker = tracker

    # Target binding mock
    binding = MagicMock(spec=TargetBinding)
    session.resolve_and_bind = AsyncMock(return_value=binding)

    # ActuatorGate mock
    decision = ReflexDecision(
        decision=DecisionType.ALLOW,
        priority=Priority.P4_METHOD_E,
        rule_name="ALLOW",
        reason="Method E verified actionable",
    )
    gate_res = GateResult(
        decision=decision,
        executed=True,
        execution_result={"x": 100.0, "y": 200.0},
    )
    session.gate = MagicMock()
    session.gate.async_attempt = AsyncMock(return_value=gate_res)

    # Browser capability mocks
    session.navigate = AsyncMock(return_value={"navigated": True, "url": "https://example.com"})
    session.inspect_elements = AsyncMock(return_value={"returned": 3, "elements": [{"tag": "button"}]})
    session.extract_read_only = AsyncMock(return_value={"text": "Page content", "title": "Example", "url": "https://example.com"})

    return session


@pytest.fixture
def mock_graph_store():
    store = MagicMock()
    store.search_memory.return_value = [
        {
            "source_id": "art_abc123",
            "memory_type": "artifact",
            "content_excerpt": "Prior research on target domain.",
            "provenance": "agent_prior_42",
            "timestamp": "2026-10-10T12:00:00Z",
        }
    ]
    store.save_artifact.return_value = "art_new_98765432"
    store.get_artifact.return_value = {
        "id": "art_new_98765432",
        "title": "Test Artifact",
        "artifact_type": "notes",
        "content": "A" * 9000,
    }
    return store


@pytest.fixture
def tool_service(lease_registry, target_registry, mock_tab_session, mock_graph_store):
    async def fake_session_resolver(tab_id, ws_url, worker_id=None):
        return mock_tab_session

    def fake_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    return ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        target_registry=target_registry,
        session_resolver=fake_session_resolver,
        target_resolver=fake_target_resolver,
    )


# =============================================================================
# 1. Construction and validation of execution context
# =============================================================================


def test_context_construction_worker_requires_valid_lease_and_token(lease_registry):
    # Acquire legitimate lease
    lease = lease_registry.acquire_lease(
        worker_id="worker-alpha",
        tab_id="tab-worker-alpha",
        trusted_issuer=True,
    )

    # Missing/wrong worker token fails closed
    with pytest.raises(PermissionError, match="invalid worker token"):
        create_tool_execution_context(
            task_id="task-1",
            worker_id="worker-alpha",
            worker_token="wrong_token",
            target_tab="tab-worker-alpha",
            lease_registry=lease_registry,
        )

    # Valid worker token succeeds
    ctx = create_tool_execution_context(
        task_id="task-1",
        run_id="run-1",
        worker_id="worker-alpha",
        worker_token=lease.lease_token,
        target_tab="tab-worker-alpha",
        timeout_seconds=30.0,
        lease_registry=lease_registry,
    )
    assert ctx.task_id == "task-1"
    assert ctx.run_id == "run-1"
    assert ctx.worker_id == "worker-alpha"
    assert ctx.target_tab_id == "tab-worker-alpha"
    assert ctx.lease is not None
    assert ctx.lease.lease_token == lease.lease_token
    assert not ctx.is_single_agent
    assert ctx.deadline_mono is not None
    assert ctx.time_remaining() > 0


def test_context_construction_prohibits_worker_foreground_aliases(lease_registry):
    lease_registry.acquire_lease(
        worker_id="worker-beta",
        tab_id="isolated-tab-2",
        trusted_issuer=True,
    )
    with pytest.raises(PermissionError, match="prohibited from leasing foreground"):
        create_tool_execution_context(
            task_id="task-2",
            worker_id="worker-beta",
            worker_token="any",
            target_tab="active_tab",
            lease_registry=lease_registry,
        )


def test_context_construction_single_agent_requires_trusted_server_authority(lease_registry):
    # Untrusted single agent attempt fails
    with pytest.raises(PermissionError, match="requires trusted server authority"):
        create_tool_execution_context(
            task_id="task-3",
            target_tab="active_tab",
            trusted_server_context=False,
            lease_registry=lease_registry,
        )

    # Trusted single agent succeeds
    ctx = create_tool_execution_context(
        task_id="task-3",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )
    assert ctx.is_single_agent
    assert ctx.worker_id is None
    assert ctx.target_tab_id == "active_tab"


# =============================================================================
# 2. Successful service-result construction
# =============================================================================


def test_tool_result_ok_construction():
    res = ToolResult.ok(
        data={"items_found": 3},
        evidence={"query": "test"},
        duration_ms=42,
        metadata={"step": 1},
    )
    assert res.success is True
    assert res.data == {"items_found": 3}
    assert res.error_code is None
    assert res.error_message is None
    assert res.evidence == {"query": "test"}
    assert res.duration_ms == 42
    assert res.retryable is False
    assert res.metadata == {"step": 1}

    d = res.to_dict()
    assert d["success"] is True
    assert d["duration_ms"] == 42


# =============================================================================
# 3. Consistent error-result construction
# =============================================================================


def test_tool_result_fail_enforces_non_retryable_security_errors():
    # Security errors must never be retryable even if caller passes retryable=True
    res_unauth = ToolResult.fail(
        error_code=ToolErrorCode.UNAUTHORIZED,
        error_message="Unauthorized access",
        retryable=True,
    )
    assert res_unauth.success is False
    assert res_unauth.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert res_unauth.retryable is False

    res_target = ToolResult.fail(
        error_code=ToolErrorCode.CANONICAL_TARGET_MISMATCH,
        error_message="Target mismatch",
        retryable=True,
    )
    assert res_target.retryable is False

    # Transient operation failures can be retryable if specified
    res_transient = ToolResult.fail(
        error_code=ToolErrorCode.OPERATION_FAILED,
        error_message="Network glitch",
        retryable=True,
    )
    assert res_transient.retryable is True


# =============================================================================
# 4. Authorization failures and invalid contexts
# =============================================================================


@pytest.mark.asyncio
async def test_authorization_failure_on_invalid_context_type(tool_service):
    # Passing a non-ToolExecutionContext object
    res = await tool_service.click("not_a_context", selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Invalid execution context" in res.error_message


@pytest.mark.asyncio
async def test_authorization_failure_on_unpermitted_operation(tool_service, lease_registry):
    lease = lease_registry.acquire_lease("w1", "tab1", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-restricted",
        worker_id="w1",
        worker_token=lease.lease_token,
        target_tab="tab1",
        permitted_operations=["inspect", "extract"],  # 'click' not permitted
        lease_registry=lease_registry,
    )

    res = await tool_service.click(ctx, selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "not permitted by execution context" in res.error_message


@pytest.mark.asyncio
async def test_authorization_failure_on_tampered_worker_context(tool_service, lease_registry):
    lease = lease_registry.acquire_lease("w1", "tab1", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-tamper",
        worker_id="w1",
        worker_token=lease.lease_token,
        target_tab="tab1",
        lease_registry=lease_registry,
    )

    # Tamper with context object to point to another tab
    tampered_ctx = ToolExecutionContext(
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        worker_id=ctx.worker_id,
        target_tab_id="tab-unauthorized",  # Tampered target tab!
        lease=ctx.lease,
        permitted_operations=ctx.permitted_operations,
        deadline_mono=ctx.deadline_mono,
    )

    res = await tool_service.click(tampered_ctx, selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Worker authorization mismatch" in res.error_message or "Context capability verification failed" in res.error_message


# =============================================================================
# 5. Missing, expired, or invalid authorization
# =============================================================================


@pytest.mark.asyncio
async def test_expired_lease_fails_closed_in_revalidation(tool_service, lease_registry):
    lease = lease_registry.acquire_lease("w_exp", "tab_exp", ttl_seconds=0.01, trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-exp",
        worker_id="w_exp",
        worker_token=lease.lease_token,
        target_tab="tab_exp",
        lease_registry=lease_registry,
    )

    # Simulate lease expiry
    lease.expires_at_mono = time.monotonic() - 10.0

    res = await tool_service.click(ctx, selector="#btn")
    assert res.success is False
    assert res.error_code in (ToolErrorCode.DEADLINE_EXCEEDED.value, ToolErrorCode.UNAUTHORIZED.value)


@pytest.mark.asyncio
async def test_canonical_target_mismatch_fails_closed(lease_registry, target_registry, mock_tab_session):
    lease = lease_registry.acquire_lease(
        "w_mismatch",
        "tab_mismatch",
        canonical_target_id="target-expected-aaa",
        trusted_issuer=True,
    )
    # Register target pointing to different physical ID
    target_registry.register_tab_target("tab_mismatch", "target-actual-bbb", "ws://127.0.0.1:9222/page/target-actual-bbb")

    service = ComexToolService(
        lease_registry=lease_registry,
        target_registry=target_registry,
        session_resolver=AsyncMock(return_value=mock_tab_session),
    )

    ctx = create_tool_execution_context(
        task_id="task-mismatch",
        worker_id="w_mismatch",
        worker_token=lease.lease_token,
        target_tab="tab_mismatch",
        lease_registry=lease_registry,
    )

    res = await service.click(ctx, selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value
    assert "mismatch" in res.error_message.lower()


# =============================================================================
# 6. Cancellation and deadline propagation
# =============================================================================


@pytest.mark.asyncio
async def test_cancellation_handle_propagates(tool_service, lease_registry):
    lease = lease_registry.acquire_lease("w_cancel", "tab_cancel", trusted_issuer=True)
    cancel_event = asyncio.Event()
    cancel_event.set()  # Already cancelled

    ctx = create_tool_execution_context(
        task_id="task-cancel",
        worker_id="w_cancel",
        worker_token=lease.lease_token,
        target_tab="tab_cancel",
        cancellation_event=cancel_event,
        lease_registry=lease_registry,
    )

    res = await tool_service.click(ctx, selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert "cancelled" in res.error_message.lower()


@pytest.mark.asyncio
async def test_deadline_exceeded_propagates(tool_service, lease_registry):
    lease = lease_registry.acquire_lease("w_dl", "tab_dl", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-dl",
        worker_id="w_dl",
        worker_token=lease.lease_token,
        target_tab="tab_dl",
        timeout_seconds=0.001,
        lease_registry=lease_registry,
    )
    # Sleep to expire deadline
    await asyncio.sleep(0.01)

    res = await tool_service.click(ctx, selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
    assert "deadline exceeded" in res.error_message.lower()


# =============================================================================
# 7. Correct delegation to injected authoritative components
# =============================================================================


@pytest.mark.asyncio
async def test_browser_click_delegates_to_method_e_and_gate(tool_service, lease_registry, mock_tab_session):
    lease = lease_registry.acquire_lease("w_click", "tab_click", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-click",
        worker_id="w_click",
        worker_token=lease.lease_token,
        target_tab="tab_click",
        lease_registry=lease_registry,
    )

    res = await tool_service.click(ctx, selector="#submit-btn")
    assert res.success is True
    assert res.data == {"x": 100.0, "y": 200.0}
    mock_tab_session.resolve_and_bind.assert_awaited_once_with("#submit-btn")
    mock_tab_session.gate.async_attempt.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_type_delegates_to_gate(tool_service, lease_registry, mock_tab_session):
    lease = lease_registry.acquire_lease("w_type", "tab_type", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-type",
        worker_id="w_type",
        worker_token=lease.lease_token,
        target_tab="tab_type",
        lease_registry=lease_registry,
    )

    res = await tool_service.type(ctx, selector="#input-email", text="test@example.com")
    assert res.success is True
    mock_tab_session.resolve_and_bind.assert_awaited_once_with("#input-email")
    mock_tab_session.gate.async_attempt.assert_awaited_once_with(
        "type", binding=mock_tab_session.resolve_and_bind.return_value, text="test@example.com", clear_first=True
    )


@pytest.mark.asyncio
async def test_browser_navigate_validates_url_and_delegates(tool_service, lease_registry, mock_tab_session):
    lease = lease_registry.acquire_lease("w_nav", "tab_nav", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-nav",
        worker_id="w_nav",
        worker_token=lease.lease_token,
        target_tab="tab_nav",
        lease_registry=lease_registry,
    )

    # Valid navigation
    res = await tool_service.navigate(ctx, url="https://example.com/checkout")
    assert res.success is True
    mock_tab_session.navigate.assert_awaited_once_with("https://example.com/checkout")

    # Prohibited scheme navigation fails validation
    res_bad = await tool_service.navigate(ctx, url="javascript:alert(1)")
    assert res_bad.success is False
    assert res_bad.error_code == ToolErrorCode.VALIDATION_ERROR.value


@pytest.mark.asyncio
async def test_browser_inspect_and_extract_read_only(tool_service, lease_registry, mock_tab_session):
    lease = lease_registry.acquire_lease("w_ro", "tab_ro", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-ro",
        worker_id="w_ro",
        worker_token=lease.lease_token,
        target_tab="tab_ro",
        lease_registry=lease_registry,
    )

    res_inspect = await tool_service.inspect(ctx, max_elements=25)
    assert res_inspect.success is True
    assert res_inspect.data["returned"] == 3
    mock_tab_session.inspect_elements.assert_awaited_once_with(max_elements=25)

    res_extract = await tool_service.extract(ctx, selector="#article", max_chars=4000)
    assert res_extract.success is True
    assert res_extract.data["title"] == "Example"
    mock_tab_session.extract_read_only.assert_awaited_once_with(selector="#article", max_chars=4000)


@pytest.mark.asyncio
async def test_memory_operations_delegate_to_graph_store(tool_service, lease_registry, mock_graph_store):
    lease = lease_registry.acquire_lease("w_mem", "tab_mem", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-mem",
        worker_id="w_mem",
        worker_token=lease.lease_token,
        target_tab="tab_mem",
        lease_registry=lease_registry,
    )

    # 1. Search memory
    res_search = await tool_service.search_memory(ctx, query="laptop price", domain="ENTITIES", limit=5)
    assert res_search.success is True
    assert len(res_search.data) == 1
    mock_graph_store.search_memory.assert_called_with("laptop price", domain="ENTITIES", limit=5)

    # 2. Save artifact
    res_save = await tool_service.save_artifact(ctx, title="Summary Report", content="Report text")
    assert res_save.success is True
    assert res_save.data["artifact_id"] == "art_new_98765432"
    mock_graph_store.save_artifact.assert_called_with("", "Summary Report", "notes", "Report text", source_url="", source_run_id=ctx.run_id, source_task_id="task-mem")

    # 3. Read artifact
    res_read = await tool_service.read_artifact(ctx, artifact_id="art_new_98765432", chunk_index=0, chunk_size=4000)
    assert res_read.success is True
    assert res_read.data["artifact_id"] == "art_new_98765432"
    assert res_read.data["chunk_index"] == 0
    assert len(res_read.data["chunk_text"]) == 4000
    mock_graph_store.get_artifact.assert_called_with("art_new_98765432")

    # 4. Get context
    res_ctx = await tool_service.get_context(ctx, query="laptop price", domain="ALL", limit=3)
    assert res_ctx.success is True
    assert "[PERSISTENT MEMORY CONTEXT]" in res_ctx.data["context_text"]


# =============================================================================
# 8. Exception handling without false success
# =============================================================================


@pytest.mark.asyncio
async def test_exception_handling_without_false_success(tool_service, lease_registry, mock_tab_session):
    lease = lease_registry.acquire_lease("w_err", "tab_err", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-err",
        worker_id="w_err",
        worker_token=lease.lease_token,
        target_tab="tab_err",
        lease_registry=lease_registry,
    )

    # Simulate unexpected CDP failure in resolve_and_bind
    mock_tab_session.resolve_and_bind.side_effect = RuntimeError("CDP WebSocket connection severed")

    res = await tool_service.click(ctx, selector="#btn")
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "CDP WebSocket connection severed" in res.error_message


# =============================================================================
# 9. Explicit failure for unsupported, not-yet-implemented capabilities
# =============================================================================


@pytest.mark.asyncio
async def test_perform_sequence_executes_successfully_in_phase2_1b(tool_service, lease_registry):
    lease = lease_registry.acquire_lease("w_seq", "tab_seq", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-seq",
        worker_id="w_seq",
        worker_token=lease.lease_token,
        target_tab="tab_seq",
        lease_registry=lease_registry,
    )

    res = await tool_service.perform_sequence(ctx, steps=[{"action": "click", "selector": "#btn"}])
    assert res.success is True
    assert res.data["step_count"] == 1
    assert len(res.data["completed_steps"]) == 1


# =============================================================================
# 10. Prevention of arbitrary operations outside the typed contract
# =============================================================================


def test_prevention_of_arbitrary_operations(tool_service):
    # The service does NOT have evaluate_js, run_script, or raw cdp methods
    assert not hasattr(tool_service, "evaluate_js")
    assert not hasattr(tool_service, "run_browser_harness")
    assert not hasattr(tool_service, "execute_script")
    assert not hasattr(tool_service, "dispatch_raw_cdp")

    # Calling nonexistent arbitrary methods raises AttributeError
    with pytest.raises(AttributeError):
        tool_service.execute_arbitrary_code("import os; os.system('ls')")


# =============================================================================
# Additional Edge Cases & Single-Agent Workflows
# =============================================================================


@pytest.mark.asyncio
async def test_single_agent_execution_workflow(tool_service, lease_registry, mock_tab_session):
    ctx = create_tool_execution_context(
        task_id="task-single-agent",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )
    assert ctx.is_single_agent is True

    res = await tool_service.click(ctx, selector="#cta")
    assert res.success is True
    assert res.data == {"x": 100.0, "y": 200.0}


@pytest.mark.asyncio
async def test_memory_operations_fail_when_graph_store_not_configured(lease_registry):
    service_no_db = ComexToolService(
        graph_store=None,
        lease_registry=lease_registry,
    )
    ctx = create_tool_execution_context(
        task_id="task-no-db",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    res_search = await service_no_db.search_memory(ctx, query="test")
    assert res_search.success is False
    assert res_search.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "Graph store not configured" in res_search.error_message

    res_save = await service_no_db.save_artifact(ctx, title="Title", content="Content")
    assert res_save.success is False
    assert res_save.error_code == ToolErrorCode.OPERATION_FAILED.value


@pytest.mark.asyncio
async def test_memory_input_validation_failures(tool_service, lease_registry):
    ctx = create_tool_execution_context(
        task_id="task-val",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # Empty title or content
    res_save_bad = await tool_service.save_artifact(ctx, title="", content="Content")
    assert res_save_bad.success is False
    assert res_save_bad.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # Negative chunk index
    res_read_bad = await tool_service.read_artifact(ctx, artifact_id="art_123", chunk_index=-1)
    assert res_read_bad.success is False
    assert res_read_bad.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # Malformed artifact id
    res_read_malformed = await tool_service.read_artifact(ctx, artifact_id="invalid_prefix_123")
    assert res_read_malformed.success is False
    assert res_read_malformed.error_code == ToolErrorCode.VALIDATION_ERROR.value


@pytest.mark.asyncio
async def test_browser_action_blocked_by_nervous_system(tool_service, lease_registry, mock_tab_session):
    ctx = create_tool_execution_context(
        task_id="task-blocked",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # Simulate blocked decision from ReflexEngine / ActuatorGate
    blocked_decision = ReflexDecision(
        decision=DecisionType.REPORT_BLOCKED,
        priority=Priority.P3_OBSTRUCTION,
        rule_name="OBSTRUCTED_BY_MODAL",
        reason="Element obscured by unhandled modal dialog",
    )
    blocked_gate_res = GateResult(
        decision=blocked_decision,
        executed=False,
        handler_invoked=False,
    )
    mock_tab_session.gate.async_attempt.return_value = blocked_gate_res

    res = await tool_service.click(ctx, selector="#covered-button")
    assert res.success is False
    assert res.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "OBSTRUCTED_BY_MODAL" in res.error_message
    assert res.retryable is False


@pytest.mark.asyncio
async def test_browser_input_validation_failures(tool_service, lease_registry):
    ctx = create_tool_execution_context(
        task_id="task-browser-val",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # Empty selector for click
    res_click = await tool_service.click(ctx, selector="   ")
    assert res_click.success is False
    assert res_click.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # Empty selector for type
    res_type = await tool_service.type(ctx, selector="", text="hello")
    assert res_type.success is False
    assert res_type.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # Empty url for navigate
    res_nav = await tool_service.navigate(ctx, url="")
    assert res_nav.success is False
    assert res_nav.error_code == ToolErrorCode.VALIDATION_ERROR.value


# =============================================================================
# Phase 2.1A Targeted Corrections: Central Canonical-Target Enforcement,
# In-Flight Cancellation/Deadlines, and Resource Bounds Tests
# =============================================================================


@pytest.mark.asyncio
async def test_injected_resolver_different_target_id_fails_closed(lease_registry, mock_tab_session, mock_graph_store):
    """
    1. An injected resolver returns a different target ID with no error: the action fails.
    """
    # Lease is bound to 'target-legit'
    lease = lease_registry.acquire_lease(
        worker_id="worker-resolver-test",
        tab_id="tab-worker-res",
        canonical_target_id="target-legit",
        trusted_issuer=True,
    )

    # Injected resolver returns a mismatched target ID
    def mismatched_target_resolver(l, tab, trusted):
        return ("target-rogue", "ws://127.0.0.1:9222/devtools/page/target-rogue", None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=mismatched_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-diff-target",
        worker_id="worker-resolver-test",
        worker_token=lease.lease_token,
        target_tab="tab-worker-res",
        lease_registry=lease_registry,
    )

    # Invocation through public ComexToolService method
    res = await service.click(ctx, selector="#submit")
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value
    assert "target-legit" in res.error_message and "target-rogue" in res.error_message
    assert res.retryable is False
    # Verify session was NOT created and no browser mutation occurred
    mock_session_resolver.assert_not_called()
    mock_tab_session.gate.async_attempt.assert_not_called()


@pytest.mark.asyncio
async def test_injected_resolver_missing_target_id_fails_closed(lease_registry, mock_tab_session, mock_graph_store):
    """
    2. An injected resolver returns no target ID when the lease has a canonical target: the action fails.
    """
    lease = lease_registry.acquire_lease(
        worker_id="worker-missing-id",
        tab_id="tab-worker-missing",
        canonical_target_id="target-legit",
        trusted_issuer=True,
    )

    # Injected resolver returns None as target ID
    def missing_target_resolver(l, tab, trusted):
        return (None, "ws://127.0.0.1:9222/devtools/page/target-legit", None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=missing_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-missing-target-id",
        worker_id="worker-missing-id",
        worker_token=lease.lease_token,
        target_tab="tab-worker-missing",
        lease_registry=lease_registry,
    )

    # Public method test with None target_id
    res_none = await service.inspect(ctx)
    assert res_none.success is False
    assert res_none.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value
    assert "missing" in res_none.error_message.lower()
    mock_session_resolver.assert_not_called()

    # Injected resolver returns empty string as target ID
    def empty_target_resolver(l, tab, trusted):
        return ("", "ws://127.0.0.1:9222/devtools/page/target-legit", None)

    service._target_resolver = empty_target_resolver
    res_empty = await service.extract(ctx)
    assert res_empty.success is False
    assert res_empty.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value
    mock_session_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_injected_resolver_forged_id_with_mismatched_ws_endpoint_fails_closed(
    lease_registry, mock_tab_session, mock_graph_store
):
    """
    3. An injected resolver claims the expected target ID but supplies a WebSocket endpoint
       identifying a different page: the action fails.
    """
    lease = lease_registry.acquire_lease(
        worker_id="worker-forged-endpoint",
        tab_id="tab-worker-forged",
        canonical_target_id="target-legit",
        trusted_issuer=True,
    )

    # Resolver claims target-legit but WebSocket endpoint actually targets target-other
    def forged_endpoint_resolver(l, tab, trusted):
        return ("target-legit", "ws://127.0.0.1:9222/devtools/page/target-other", None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=forged_endpoint_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-forged-endpoint",
        worker_id="worker-forged-endpoint",
        worker_token=lease.lease_token,
        target_tab="tab-worker-forged",
        lease_registry=lease_registry,
    )

    res = await service.navigate(ctx, url="https://example.com")
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value
    assert "target-other" in res.error_message
    assert res.retryable is False
    mock_session_resolver.assert_not_called()
    mock_tab_session.navigate.assert_not_called()


@pytest.mark.asyncio
async def test_correctly_bound_target_and_endpoint_succeeds(
    lease_registry, mock_tab_session, mock_graph_store
):
    """
    4. A correctly bound target ID and endpoint succeed.
    """
    lease = lease_registry.acquire_lease(
        worker_id="worker-bound-legit",
        tab_id="tab-worker-bound",
        canonical_target_id="target-verified-100",
        trusted_issuer=True,
    )

    valid_ws = "ws://127.0.0.1:9222/devtools/page/target-verified-100"

    def valid_resolver(l, tab, trusted):
        return ("target-verified-100", valid_ws, None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=valid_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-valid-binding",
        worker_id="worker-bound-legit",
        worker_token=lease.lease_token,
        target_tab="tab-worker-bound",
        lease_registry=lease_registry,
    )

    res = await service.click(ctx, selector="#good-button")
    assert res.success is True
    assert res.data == {"x": 100.0, "y": 200.0}
    mock_session_resolver.assert_called_once_with(
        "tab-worker-bound", valid_ws, worker_id="worker-bound-legit"
    )
    mock_tab_session.gate.async_attempt.assert_called_once()


@pytest.mark.asyncio
async def test_mismatch_prevents_session_creation_and_browser_mutation(
    lease_registry, mock_tab_session, mock_graph_store
):
    """
    5. A mismatch prevents session creation and prevents any browser mutation.
    """
    lease = lease_registry.acquire_lease(
        worker_id="worker-mutation-guard",
        tab_id="tab-guard",
        canonical_target_id="target-guard-alpha",
        trusted_issuer=True,
    )

    def mismatch_resolver(l, tab, trusted):
        return ("target-guard-beta", "ws://127.0.0.1:9222/devtools/page/target-guard-beta", None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=mismatch_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-guard-mismatch",
        worker_id="worker-mutation-guard",
        worker_token=lease.lease_token,
        target_tab="tab-guard",
        lease_registry=lease_registry,
    )

    # Test across type mutation
    res_type = await service.type(ctx, selector="#input", text="secret")
    assert res_type.success is False
    assert res_type.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value

    # Test across navigate mutation
    res_nav = await service.navigate(ctx, url="https://malicious-target.com")
    assert res_nav.success is False
    assert res_nav.error_code == ToolErrorCode.CANONICAL_TARGET_MISMATCH.value

    # Zero sessions created, zero mutations performed
    assert mock_session_resolver.call_count == 0
    assert mock_tab_session.gate.async_attempt.call_count == 0
    assert mock_tab_session.navigate.call_count == 0


@pytest.mark.asyncio
async def test_injected_resolver_http_discovery_url_fails_closed(
    lease_registry, mock_tab_session, mock_graph_store
):
    """
    HTTP CDP discovery URLs (e.g. http://localhost:9222/json) must never be passed
    to a WebSocket client and must fail closed with TARGET_RESOLUTION_FAILED.
    """
    lease = lease_registry.acquire_lease(
        worker_id="worker-http-test",
        tab_id="tab-http",
        canonical_target_id="target-http-123",
        trusted_issuer=True,
    )

    def http_resolver(l, tab, trusted):
        return ("target-http-123", "http://127.0.0.1:9222/json", None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=http_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-http-endpoint",
        worker_id="worker-http-test",
        worker_token=lease.lease_token,
        target_tab="tab-http",
        lease_registry=lease_registry,
    )

    res = await service.inspect(ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.TARGET_RESOLUTION_FAILED.value
    assert "HTTP discovery endpoints are prohibited" in res.error_message
    mock_session_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_default_foreground_resolution_without_resolver_fails_closed(lease_registry):
    """
    Default foreground path with default HTTP CDP url and no resolver or registered
    target must fail closed and never pass an HTTP URL as WebSocket endpoint.
    """
    service_default = ComexToolService(
        default_cdp_url="http://localhost:9222",
        lease_registry=lease_registry,
        target_resolver=None,
    )

    ctx = create_tool_execution_context(
        task_id="task-default-foreground",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    res = await service_default.inspect(ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.TARGET_RESOLUTION_FAILED.value
    assert "could not be resolved" in res.error_message


@pytest.mark.asyncio
async def test_in_flight_deadline_expiration_and_lock_release(lease_registry, mock_tab_session, mock_graph_store):
    """
    In-flight deadline expiration must cancel pending operation, return DEADLINE_EXCEEDED,
    and release the per-tab action lock.
    """
    async def slow_gate_attempt(*args, **kwargs):
        await asyncio.sleep(0.5)
        return MagicMock()

    mock_tab_session.gate.async_attempt = slow_gate_attempt

    async def fake_session_resolver(tab_id, ws_url, worker_id=None):
        return mock_tab_session

    def fake_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=fake_session_resolver,
        target_resolver=fake_target_resolver,
    )

    # 40ms deadline
    ctx = create_tool_execution_context(
        task_id="task-deadline-inflight",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=0.04,
        lease_registry=lease_registry,
    )

    res = await service.click(ctx, selector="#slow-button")
    assert res.success is False
    assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
    # Action lock must be released
    assert not mock_tab_session.action_lock.locked()


@pytest.mark.asyncio
async def test_in_flight_cancellation_and_lock_release(lease_registry, mock_tab_session, mock_graph_store):
    """
    Setting cancellation_event while an operation is pending must cancel the in-flight
    operation, return CANCELLED, and release the per-tab action lock.
    """
    cancel_evt = asyncio.Event()

    async def slow_navigate(url):
        # Trigger cancellation while in flight
        await asyncio.sleep(0.05)
        cancel_evt.set()
        await asyncio.sleep(0.5)
        return {"navigated": True}

    mock_tab_session.navigate = slow_navigate

    async def fake_session_resolver(tab_id, ws_url, worker_id=None):
        return mock_tab_session

    def fake_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=fake_session_resolver,
        target_resolver=fake_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-cancel-inflight",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    res = await service.navigate(ctx, url="https://example.com/slow")
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    # Action lock must be released
    assert not mock_tab_session.action_lock.locked()


@pytest.mark.asyncio
async def test_resource_bounds_clamping_and_validation(tool_service, lease_registry, mock_tab_session, mock_graph_store):
    """
    Verify clamping and validation for caller-controlled inspection count, extraction size,
    memory limit, artifact chunk size, and saved artifact content size.
    """
    ctx = create_tool_execution_context(
        task_id="task-resource-bounds",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # 1. inspect max_elements clamping and validation
    res_insp_over = await tool_service.inspect(ctx, max_elements=MAX_INSPECT_ELEMENTS + 500)
    assert res_insp_over.success is True
    mock_tab_session.inspect_elements.assert_called_with(max_elements=MAX_INSPECT_ELEMENTS)

    res_insp_invalid = await tool_service.inspect(ctx, max_elements=-1)
    assert res_insp_invalid.success is False
    assert res_insp_invalid.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # 2. extract max_chars clamping and validation
    res_ext_over = await tool_service.extract(ctx, max_chars=MAX_EXTRACT_CHARS + 50_000)
    assert res_ext_over.success is True
    mock_tab_session.extract_read_only.assert_called_with(selector=None, max_chars=MAX_EXTRACT_CHARS)

    res_ext_invalid = await tool_service.extract(ctx, max_chars=0)
    assert res_ext_invalid.success is False
    assert res_ext_invalid.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # 3. search_memory limit clamping and validation
    res_search_over = await tool_service.search_memory(ctx, query="test", limit=MAX_MEMORY_RESULTS + 200)
    assert res_search_over.success is True
    mock_graph_store.search_memory.assert_called_with("test", domain="ALL", limit=MAX_MEMORY_RESULTS)

    res_search_invalid = await tool_service.search_memory(ctx, query="test", limit=0)
    assert res_search_invalid.success is False
    assert res_search_invalid.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # 4. save_artifact content size rejection
    huge_content = "X" * (MAX_ARTIFACT_CONTENT_SIZE + 100)
    res_save_huge = await tool_service.save_artifact(ctx, title="Huge Artifact", content=huge_content)
    assert res_save_huge.success is False
    assert res_save_huge.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "exceeds maximum size" in res_save_huge.error_message

    # 5. read_artifact chunk_size clamping
    res_read_over = await tool_service.read_artifact(
        ctx, artifact_id="art_new_98765432", chunk_size=MAX_ARTIFACT_CHUNK_SIZE + 10_000
    )
    assert res_read_over.success is True
    assert len(res_read_over.data["chunk_text"]) <= MAX_ARTIFACT_CHUNK_SIZE


@pytest.mark.asyncio
async def test_external_asyncio_cancellation_cleans_up_and_propagates(lease_registry, mock_tab_session, mock_graph_store):
    """
    Verify that external asyncio task cancellation (task.cancel()):
    1. Cancels the underlying in-flight operation coroutine.
    2. Releases session.action_lock immediately.
    3. Propagates asyncio.CancelledError to the caller rather than swallowing it into ToolResult.
    4. Prevents delayed mock mutations.
    """
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()
    mutation_occurred = False

    async def slow_click(*args, **kwargs):
        nonlocal mutation_occurred
        operation_started.set()
        try:
            await asyncio.sleep(2.0)
            mutation_occurred = True
        except asyncio.CancelledError:
            operation_cancelled.set()
            raise

    # Replace gate async_attempt with slow_click
    mock_tab_session.gate.async_attempt = slow_click

    async def fake_session_resolver(tab_id, ws_url, worker_id=None):
        return mock_tab_session

    def fake_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=fake_session_resolver,
        target_resolver=fake_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-external-cancel",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # Spawn task running the browser operation
    task = asyncio.create_task(service.click(ctx, selector="#button"))

    # Wait until operation has begun and acquired action_lock
    await operation_started.wait()
    assert mock_tab_session.action_lock.locked()

    # Externally cancel the enclosing task
    task.cancel()

    # Must propagate CancelledError to caller
    with pytest.raises(asyncio.CancelledError):
        await task

    # Verify underlying operation received cancellation and completed cleanup
    assert operation_cancelled.is_set()
    assert not mock_tab_session.action_lock.locked()
    assert mutation_occurred is False


@pytest.mark.asyncio
async def test_untrusted_websocket_endpoint_host_rejected(lease_registry, mock_tab_session):
    """
    Verify that an untrusted WebSocket endpoint host is rejected BEFORE session creation,
    even if the resolved target ID matches the canonical ID.
    """
    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    # Target resolver returns matching target ID but untrusted external host
    def untrusted_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-legit", "ws://untrusted-host.example:9222/devtools/page/target-legit", None)

    service = ComexToolService(
        default_cdp_url="http://127.0.0.1:9222",
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=untrusted_target_resolver,
    )

    lease = lease_registry.acquire_lease("w_trust", "tab_trust", canonical_target_id="target-legit", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-untrusted-endpoint",
        worker_id="w_trust",
        worker_token=lease.lease_token,
        target_tab="tab_trust",
        lease_registry=lease_registry,
    )

    res = await service.navigate(ctx, url="https://example.com")
    assert res.success is False
    assert res.error_code == ToolErrorCode.TARGET_RESOLUTION_FAILED.value
    assert "untrusted" in res.error_message.lower()

    # Session resolver and browser operations must NOT have been called
    mock_session_resolver.assert_not_called()
    mock_tab_session.navigate.assert_not_called()


@pytest.mark.asyncio
async def test_target_resolution_and_session_acquisition_respect_deadlines(lease_registry, mock_tab_session):
    """
    Verify that the context deadline covers target resolution and session acquisition,
    not merely the final browser operation.
    """
    # 1. Slow target resolver exceeding deadline
    async def slow_target_resolver(lease, target_tab, trusted_app_context):
        await asyncio.sleep(1.0)
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    service_slow_target = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=slow_target_resolver,
    )

    ctx_short_target = create_tool_execution_context(
        task_id="task-slow-target",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=0.05,
        lease_registry=lease_registry,
    )

    res_target = await service_slow_target.inspect(ctx_short_target)
    assert res_target.success is False
    assert res_target.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
    mock_session_resolver.assert_not_called()

    # 2. Slow session acquisition exceeding deadline
    def fast_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    async def slow_session_resolver(tab_id, ws_url, worker_id=None):
        await asyncio.sleep(1.0)
        return mock_tab_session

    service_slow_session = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=slow_session_resolver,
        target_resolver=fast_target_resolver,
    )

    ctx_short_session = create_tool_execution_context(
        task_id="task-slow-session",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=0.05,
        lease_registry=lease_registry,
    )

    res_session = await service_slow_session.inspect(ctx_short_session)
    assert res_session.success is False
    assert res_session.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
    mock_tab_session.inspect_elements.assert_not_called()


@pytest.mark.asyncio
async def test_slow_graph_memory_store_respects_deadline(lease_registry):
    """
    Verify that slow graph-store operations run off the main event loop and respect context deadlines.
    """
    mock_store = MagicMock()

    def slow_search(*args, **kwargs):
        time.sleep(0.3)
        return [{"source_id": "art_1", "content_excerpt": "Slow"}]

    mock_store.search_memory = slow_search

    service = ComexToolService(graph_store=mock_store, lease_registry=lease_registry)
    ctx = create_tool_execution_context(
        task_id="task-slow-memory",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=0.05,
        lease_registry=lease_registry,
    )

    res = await service.search_memory(ctx, query="test")
    assert res.success is False
    assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value


@pytest.mark.asyncio
async def test_revoked_lease_blocks_memory_operations(lease_registry, mock_graph_store):
    """
    Verify that revoking a worker's lease immediately blocks subsequent graph-memory operations
    (get_context, search_memory, save_artifact, read_artifact) from reaching the store.
    """
    service = ComexToolService(graph_store=mock_graph_store, lease_registry=lease_registry)

    lease = lease_registry.acquire_lease("w_mem_rev", "tab_mem_rev", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-mem-revoked",
        worker_id="w_mem_rev",
        worker_token=lease.lease_token,
        target_tab="tab_mem_rev",
        lease_registry=lease_registry,
    )

    # Initial search succeeds with active lease
    res_ok = await service.search_memory(ctx, query="initial")
    assert res_ok.success is True

    # Now revoke lease in server registry
    assert lease_registry.revoke_lease("w_mem_rev", trusted_caller=True) is True

    # Reset mock call counters
    mock_graph_store.search_memory.reset_mock()
    mock_graph_store.save_artifact.reset_mock()
    mock_graph_store.get_artifact.reset_mock()

    # Subsequent memory operations must fail closed as UNAUTHORIZED
    res_search = await service.search_memory(ctx, query="test")
    assert res_search.success is False
    assert res_search.error_code == ToolErrorCode.UNAUTHORIZED.value
    mock_graph_store.search_memory.assert_not_called()

    res_save = await service.save_artifact(ctx, title="Title", content="Content")
    assert res_save.success is False
    assert res_save.error_code == ToolErrorCode.UNAUTHORIZED.value
    mock_graph_store.save_artifact.assert_not_called()

    res_read = await service.read_artifact(ctx, artifact_id="art_12345678")
    assert res_read.success is False
    assert res_read.error_code == ToolErrorCode.UNAUTHORIZED.value
    mock_graph_store.get_artifact.assert_not_called()

    res_ctx = await service.get_context(ctx, query="test")
    assert res_ctx.success is False
    assert res_ctx.error_code == ToolErrorCode.UNAUTHORIZED.value


@pytest.mark.asyncio
async def test_resolver_exceptions_and_malformed_returns_handled(lease_registry, mock_tab_session):
    """
    Verify that target resolver exceptions and malformed return tuples are converted
    into structured ToolResult.fail values without leaking exceptions or invoking sessions.
    """
    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    # 1. Resolver raising exception
    def crashing_target_resolver(lease, target_tab, trusted_app_context):
        raise RuntimeError("CDP daemon socket connection refused: 9222")

    service_crash = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=crashing_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-crash-resolver",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    res_crash = await service_crash.inspect(ctx)
    assert res_crash.success is False
    assert res_crash.error_code == ToolErrorCode.TARGET_RESOLUTION_FAILED.value
    assert "raised exception" in res_crash.error_message
    mock_session_resolver.assert_not_called()

    # 2. Resolver returning malformed shape (not 3-tuple)
    def malformed_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222")  # Missing 3rd element

    service_malformed = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=malformed_target_resolver,
    )

    res_malformed = await service_malformed.inspect(ctx)
    assert res_malformed.success is False
    assert res_malformed.error_code == ToolErrorCode.TARGET_RESOLUTION_FAILED.value
    assert "malformed" in res_malformed.error_message
    mock_session_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_type_validation_rejects_invalid_types_as_validation_error(tool_service, lease_registry):
    """
    Verify runtime type checking returns VALIDATION_ERROR instead of leaking TypeError for non-int / bool values.
    """
    ctx = create_tool_execution_context(
        task_id="task-type-validation",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # inspect: string and bool for max_elements
    r1 = await tool_service.inspect(ctx, max_elements="50")  # type: ignore
    assert r1.success is False and r1.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r2 = await tool_service.inspect(ctx, max_elements=True)  # type: ignore
    assert r2.success is False and r2.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # extract: string, bool, and invalid selector
    r3 = await tool_service.extract(ctx, max_chars="8000")  # type: ignore
    assert r3.success is False and r3.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r4 = await tool_service.extract(ctx, max_chars=False)  # type: ignore
    assert r4.success is False and r4.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r5 = await tool_service.extract(ctx, selector=123)  # type: ignore
    assert r5.success is False and r5.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # type: clear_first non-bool
    r6 = await tool_service.type(ctx, selector="#in", text="hi", clear_first="yes")  # type: ignore
    assert r6.success is False and r6.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # search_memory: limit invalid types
    r7 = await tool_service.search_memory(ctx, query="test", limit="ten")  # type: ignore
    assert r7.success is False and r7.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r8 = await tool_service.search_memory(ctx, query="test", limit=True)  # type: ignore
    assert r8.success is False and r8.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # get_context: limit invalid types
    r9 = await tool_service.get_context(ctx, query="test", limit="three")  # type: ignore
    assert r9.success is False and r9.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # read_artifact: chunk_index and chunk_size invalid types
    r10 = await tool_service.read_artifact(ctx, artifact_id="art_12345", chunk_index="zero")  # type: ignore
    assert r10.success is False and r10.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r11 = await tool_service.read_artifact(ctx, artifact_id="art_12345", chunk_index=True)  # type: ignore
    assert r11.success is False and r11.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r12 = await tool_service.read_artifact(ctx, artifact_id="art_12345", chunk_size="4000")  # type: ignore
    assert r12.success is False and r12.error_code == ToolErrorCode.VALIDATION_ERROR.value

    r13 = await tool_service.read_artifact(ctx, artifact_id="art_12345", chunk_size=False)  # type: ignore
    assert r13.success is False and r13.error_code == ToolErrorCode.VALIDATION_ERROR.value


@pytest.mark.asyncio
async def test_service_level_output_ceilings_enforce_hard_limits(lease_registry, mock_tab_session, mock_graph_store):
    """
    Verify that even if an underlying dependency over-returns data, the service hard-caps
    the returned results to the requested ceiling.
    """
    # 1. Session inspect returns 500 items when requested max is 10
    mock_tab_session.inspect_elements.return_value = {
        "returned": 500,
        "elements": [{"tag": f"item_{i}"} for i in range(500)],
    }

    # 2. Session extract returns 100,000 characters when requested max is 50
    mock_tab_session.extract_read_only.return_value = {
        "title": "Huge Page",
        "text": "A" * 100_000,
    }

    # 3. Graph store search returns 300 results when requested max is 5
    mock_graph_store.search_memory.return_value = [
        {"source_id": f"art_{i}", "content_excerpt": f"text {i}"} for i in range(300)
    ]

    async def fake_session_resolver(tab_id, ws_url, worker_id=None):
        return mock_tab_session

    def fake_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    service = ComexToolService(
        graph_store=mock_graph_store,
        lease_registry=lease_registry,
        session_resolver=fake_session_resolver,
        target_resolver=fake_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-over-returns",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    # Test inspect hard cap
    res_insp = await service.inspect(ctx, max_elements=10)
    assert res_insp.success is True
    assert len(res_insp.data["elements"]) == 10
    assert res_insp.data["returned"] == 10

    # Test extract hard cap
    res_ext = await service.extract(ctx, max_chars=50)
    assert res_ext.success is True
    assert len(res_ext.data["text"]) == 50

    # Test search_memory hard cap
    res_search = await service.search_memory(ctx, query="test", limit=5)
    assert res_search.success is True
    assert len(res_search.data) == 5


@pytest.mark.asyncio
async def test_kuzu_store_cancellation_and_side_effect_compensation(lease_registry):
    """
    Verify that if a store save_artifact operation is in flight when context cancellation occurs:
    1. The operation waits until write and compensating deletion are settled before returning a terminal failure.
    2. The caller receives ToolErrorCode.CANCELLED with evidence proving persisted=False and clean rollback.
    3. Compensating deletion was executed BEFORE the caller received the failure result.
    """
    import threading
    store_entered = threading.Event()
    allow_store_finish = threading.Event()
    mock_store = MagicMock()

    def slow_save(agent_id, title, atype, content):
        store_entered.set()
        allow_store_finish.wait(timeout=5.0)
        return "art_test_zombie_123"

    mock_store.save_artifact = slow_save
    mock_store.delete_artifact = MagicMock()

    service = ComexToolService(graph_store=mock_store, lease_registry=lease_registry)
    cancel_evt = asyncio.Event()

    ctx = create_tool_execution_context(
        task_id="task-cancel-store-write",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Zombie Doc", content="Should not persist"))

    # Wait until the thread has entered the store operation
    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    # Trigger context cancellation while store write is in flight
    cancel_evt.set()

    # While task is awaiting settlement, allow the thread to finish its write and execute compensation
    allow_store_finish.set()

    # The caller task receives CANCELLED result AFTER settlement completes
    res = await task
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "ROLLED_BACK_CLEAN"
    assert res.evidence.get("persisted") is False
    assert res.evidence.get("compensated_artifact_id") == "art_test_zombie_123"

    # Verify that compensating cleanup was already executed BEFORE res was returned
    mock_store.delete_artifact.assert_called_once_with("art_test_zombie_123")


@pytest.mark.asyncio
async def test_kuzu_store_production_cleanup_fallback_path(lease_registry):
    """
    Verify the actual cleanup fallback path used by production AgentGraphStore:
    AgentGraphStore does not expose a delete_artifact method, but instead uses
    store.conn.execute("MATCH (a:Artifact {id: $id}) DETACH DELETE a", ...) under store._lock.
    """
    import threading
    store_entered = threading.Event()
    allow_store_finish = threading.Event()

    class ProductionLikeGraphStore:
        def __init__(self):
            self.conn = MagicMock()
            self._lock = threading.Lock()

        def save_artifact(self, agent_id, title, atype, content):
            store_entered.set()
            allow_store_finish.wait(timeout=5.0)
            return "art_prod_fallback_456"

    prod_store = ProductionLikeGraphStore()
    service = ComexToolService(graph_store=prod_store, lease_registry=lease_registry)
    cancel_evt = asyncio.Event()

    ctx = create_tool_execution_context(
        task_id="task-prod-cleanup",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Prod Doc", content="Content"))

    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    cancel_evt.set()
    allow_store_finish.set()

    res = await task
    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "ROLLED_BACK_CLEAN"
    assert res.evidence.get("persisted") is False

    # Verify that conn.execute was called with the production DETACH DELETE query under _lock
    prod_store.conn.execute.assert_called_once_with(
        "MATCH (a:Artifact {id: $id}) DETACH DELETE a",
        {"id": "art_prod_fallback_456"},
    )


@pytest.mark.asyncio
async def test_kuzu_store_cleanup_failure_reports_uncompensated_side_effect(lease_registry):
    """
    Verify that if compensating deletion fails (e.g. database error during DETACH DELETE):
    1. The service does NOT falsely report clean cancellation.
    2. The service returns ToolErrorCode.OPERATION_FAILED.
    3. The evidence explicitly reports settlement="COMPENSATION_FAILED", persisted=True, and side_effect_possible=True.
    """
    import threading
    store_entered = threading.Event()
    allow_store_finish = threading.Event()

    class FailingCleanupGraphStore:
        def __init__(self):
            self.conn = MagicMock()
            self.conn.execute.side_effect = RuntimeError("Kùzu disk write error on delete")
            self._lock = threading.Lock()

        def save_artifact(self, agent_id, title, atype, content):
            store_entered.set()
            allow_store_finish.wait(timeout=5.0)
            return "art_fail_cleanup_789"

    fail_store = FailingCleanupGraphStore()
    service = ComexToolService(graph_store=fail_store, lease_registry=lease_registry)
    cancel_evt = asyncio.Event()

    ctx = create_tool_execution_context(
        task_id="task-fail-cleanup",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Fail Doc", content="Content"))

    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    cancel_evt.set()
    allow_store_finish.set()

    res = await task
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "COMPENSATION_FAILED"
    assert res.evidence.get("persisted") is True
    assert res.evidence.get("side_effect_possible") is True
    assert "Kùzu disk write error" in str(res.evidence.get("cleanup_error"))


@pytest.mark.asyncio
async def test_kuzu_store_cancellation_unsettled_grace_period_timeout(lease_registry, monkeypatch):
    """
    Verify that if the underlying store write hangs indefinitely and exceeds the settlement grace period:
    1. The service times out after the grace period and returns failure.
    2. The result does NOT claim clean cancellation or safe rollback.
    3. The evidence explicitly reports settlement="UNSETTLED_IN_FLIGHT", persisted=None, and side_effect_possible=True.
    """
    import threading
    import services.tool_service as ts_mod

    # Set grace period to 0.1s for fast testing
    monkeypatch.setattr(ts_mod, "SETTLEMENT_GRACE_PERIOD_S", 0.1)

    store_entered = threading.Event()
    hang_event = threading.Event()
    mock_store = MagicMock()

    def hanging_save(agent_id, title, atype, content):
        store_entered.set()
        hang_event.wait(timeout=5.0)
        return "art_hung_999"

    mock_store.save_artifact = hanging_save
    service = ComexToolService(graph_store=mock_store, lease_registry=lease_registry)
    cancel_evt = asyncio.Event()

    ctx = create_tool_execution_context(
        task_id="task-hung-write",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Hung Doc", content="Content"))

    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    # Cancel while write is hung
    cancel_evt.set()

    # Wait for service to time out its settlement wait
    res = await task
    # Unblock hanging thread so background pool remains clean
    hang_event.set()

    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "UNSETTLED_IN_FLIGHT"
    assert res.evidence.get("persisted") is None
    assert res.evidence.get("side_effect_possible") is True


@pytest.mark.asyncio
async def test_blocking_synchronous_target_resolver_does_not_stall_event_loop(lease_registry, mock_tab_session):
    """
    Verify that a blocking synchronous injected target resolver is offloaded to a worker thread
    and cannot freeze the asyncio event loop or bypass context deadline bounds.
    Also verifies that ToolResult evidence explicitly documents that the underlying thread was not terminated.
    """
    def blocking_sync_target_resolver(lease, target_tab, trusted_app_context):
        # Blocks synchronous thread for 400ms
        time.sleep(0.4)
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=mock_tab_session),
        target_resolver=blocking_sync_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-blocking-sync-resolver",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=0.05,
        lease_registry=lease_registry,
    )

    # Track that the event loop can still tick concurrently
    ticks = 0
    stop_ticker = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop_ticker.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())

    res = await service.inspect(ctx)
    stop_ticker.set()
    await ticker_task

    # Must return DEADLINE_EXCEEDED without stalling the loop
    assert res.success is False
    assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
    assert res.evidence is not None
    assert res.evidence.get("resolver_type") == "synchronous"
    assert res.evidence.get("terminated") is False
    assert res.evidence.get("thread_state") == "in_flight_timeout"
    # Confirm event loop ticked while resolver was blocked in thread
    assert ticks >= 2


@pytest.mark.asyncio
async def test_synchronous_target_resolver_cancellation_reports_unterminated_thread(lease_registry, mock_tab_session):
    """
    Verify that cancelling target resolution with a synchronous resolver unblocks the caller
    with ToolErrorCode.CANCELLED, while explicitly reporting terminated=False and thread_state="in_flight_cancelled".
    """
    import threading
    resolver_entered = threading.Event()
    hang_evt = threading.Event()

    def hanging_sync_target_resolver(lease, target_tab, trusted_app_context):
        resolver_entered.set()
        hang_evt.wait(timeout=5.0)
        return ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=mock_tab_session),
        target_resolver=hanging_sync_target_resolver,
    )

    cancel_evt = asyncio.Event()
    ctx = create_tool_execution_context(
        task_id="task-cancel-sync-resolver",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.inspect(ctx))

    while not resolver_entered.is_set():
        await asyncio.sleep(0.01)

    cancel_evt.set()
    res = await task
    hang_evt.set()

    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("resolver_type") == "synchronous"
    assert res.evidence.get("terminated") is False
    assert res.evidence.get("thread_state") == "in_flight_cancelled"


@pytest.mark.asyncio
async def test_manually_constructed_single_agent_context_rejected_without_server_authority(
    tool_service, lease_registry
):
    """
    Verify that manually constructing a ToolExecutionContext with is_single_agent=True
    cannot, by itself, authorize a caller that lacks trusted server authority:
    1. Context with lease=None is rejected as UNAUTHORIZED.
    2. Context with a fabricated TabLease / forged lease_token is rejected as UNAUTHORIZED.
    """
    # 1. Raw dataclass with lease=None
    forged_ctx_none = ToolExecutionContext(
        run_id="fake-run",
        task_id="fake-task",
        worker_id=None,
        target_tab_id="active_tab",
        lease=None,
        permitted_operations=frozenset({"inspect", "navigate", "search_memory", "save_artifact"}),
        deadline_mono=None,
        is_single_agent=True,
    )

    res_insp = await tool_service.inspect(forged_ctx_none)
    assert res_insp.success is False
    assert res_insp.error_code == ToolErrorCode.UNAUTHORIZED.value

    res_nav = await tool_service.navigate(forged_ctx_none, "https://example.com")
    assert res_nav.success is False
    assert res_nav.error_code == ToolErrorCode.UNAUTHORIZED.value

    res_search = await tool_service.search_memory(forged_ctx_none, query="test")
    assert res_search.success is False
    assert res_search.error_code == ToolErrorCode.UNAUTHORIZED.value

    res_save = await tool_service.save_artifact(forged_ctx_none, title="T", content="C")
    assert res_save.success is False
    assert res_save.error_code == ToolErrorCode.UNAUTHORIZED.value

    # 2. Raw dataclass with fabricated TabLease holding forged token
    fake_lease = TabLease(
        worker_id="__trusted_single_agent__",
        tab_id="active_tab",
        lease_token="forged_token_not_in_server_authority_registry",
        is_active=True,
    )
    forged_ctx_lease = ToolExecutionContext(
        run_id="fake-run-2",
        task_id="fake-task-2",
        worker_id=None,
        target_tab_id="active_tab",
        lease=fake_lease,
        permitted_operations=frozenset({"inspect", "navigate", "search_memory", "save_artifact"}),
        deadline_mono=None,
        is_single_agent=True,
    )

    res_insp2 = await tool_service.inspect(forged_ctx_lease)
    assert res_insp2.success is False
    assert res_insp2.error_code == ToolErrorCode.UNAUTHORIZED.value

    res_nav2 = await tool_service.navigate(forged_ctx_lease, "https://example.com")
    assert res_nav2.success is False
    assert res_nav2.error_code == ToolErrorCode.UNAUTHORIZED.value

    res_search2 = await tool_service.search_memory(forged_ctx_lease, query="test")
    assert res_search2.success is False
    assert res_search2.error_code == ToolErrorCode.UNAUTHORIZED.value

    res_save2 = await tool_service.save_artifact(forged_ctx_lease, title="T", content="C")
    assert res_save2.success is False
    assert res_save2.error_code == ToolErrorCode.UNAUTHORIZED.value


# =============================================================================
# 16. Settlement and external cancellation of in-flight artifact writes
# =============================================================================


@pytest.mark.asyncio
async def test_external_task_cancellation_settles_and_compensates_store_write(lease_registry):
    """
    Verify that when an external asyncio Task running save_artifact() is cancelled via task.cancel():
    1. The method preserves standard asyncio cancellation semantics by re-raising CancelledError.
    2. An independent abort signal is triggered for the worker thread.
    3. The service awaits settlement within the grace period, ensuring compensating deletion is executed.
    4. The artifact is deleted from the store and cannot remain orphaned.
    """
    import threading
    store_entered = threading.Event()
    allow_finish = threading.Event()
    mock_store = MagicMock()

    def slow_save(agent_id, title, artifact_type, content):
        store_entered.set()
        allow_finish.wait(timeout=5.0)
        return "art_ext_cancel_999"

    mock_store.save_artifact = slow_save
    mock_store.delete_artifact = MagicMock(return_value=True)

    service = ComexToolService(graph_store=mock_store, lease_registry=lease_registry)
    ctx = create_tool_execution_context(
        task_id="task-ext-cancel",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=5.0,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Title", content="Content"))

    # Wait until worker thread is executing the synchronous store write
    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    # Externally cancel the task
    task.cancel()

    # Allow store write to finish; worker will see abort_event and trigger compensation
    allow_finish.set()

    # Caller must receive CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task

    # Compensating deletion must have been executed for the persisted artifact ID
    mock_store.delete_artifact.assert_called_once_with("art_ext_cancel_999")


# =============================================================================
# 17. Partial writes with empty artifact IDs and rollback integrity
# =============================================================================


def test_store_partial_write_relationship_failure_triggers_rollback(graph_store):
    """
    Verify that in AgentGraphStore.save_artifact, if the Artifact node is created
    but the subsequent CREATED_ARTIFACT relationship creation raises an exception:
    1. The store performs best-effort rollback (DETACH DELETE the created Artifact node).
    2. The store returns an empty string without leaving orphaned Artifact nodes in Kùzu.
    """
    # Create the agent first
    graph_store.create_agent("agent_partial_1", task="Test Task", intent="RESEARCH")

    original_execute = graph_store.conn.execute

    def fail_on_relationship(query, params=None):
        if "CREATED_ARTIFACT" in query:
            raise RuntimeError("Simulated Kùzu relationship query crash")
        if params is not None:
            return original_execute(query, params)
        return original_execute(query)

    with graph_store._lock:
        graph_store.conn.execute = fail_on_relationship

    try:
        res_id = graph_store.save_artifact("agent_partial_1", "Title", "notes", "Content")
        assert res_id == ""
    finally:
        with graph_store._lock:
            graph_store.conn.execute = original_execute

    # Verify that the Artifact node was rolled back and does not exist in the database
    with graph_store._lock:
        res = graph_store.conn.execute("MATCH (a:Artifact) RETURN count(a)")
        assert res.has_next()
        count = res.get_next()[0]
        assert count == 0


def test_store_partial_write_rollback_failure_raises_runtime_error(graph_store):
    """
    Verify that if both relationship creation AND compensating rollback fail in AgentGraphStore.save_artifact,
    the store raises a RuntimeError and does not silently swallow the uncompensated side effect.
    """
    graph_store.create_agent("agent_partial_2", task="Test Task", intent="RESEARCH")
    original_execute = graph_store.conn.execute

    def fail_relationship_and_rollback(query, params=None):
        if "CREATED_ARTIFACT" in query:
            raise RuntimeError("Simulated relationship failure")
        if "DETACH DELETE" in query:
            raise RuntimeError("Simulated rollback failure")
        if params is not None:
            return original_execute(query, params)
        return original_execute(query)

    with graph_store._lock:
        graph_store.conn.execute = fail_relationship_and_rollback

    try:
        with pytest.raises(RuntimeError, match="Partial artifact write failed and compensating rollback failed"):
            graph_store.save_artifact("agent_partial_2", "Title", "notes", "Content")
    finally:
        with graph_store._lock:
            graph_store.conn.execute = original_execute


def test_cleanup_stored_artifact_fails_closed_on_missing_or_empty_id(mock_graph_store):
    """
    Verify that _cleanup_stored_artifact fails closed and returns (False, ...)
    when artifact_id is missing, empty, or whitespace, and does NOT claim success.
    """
    from services.tool_service import _cleanup_stored_artifact

    ok, err = _cleanup_stored_artifact(mock_graph_store, "")
    assert ok is False
    assert "missing or empty artifact_id" in err

    ok2, err2 = _cleanup_stored_artifact(mock_graph_store, None)
    assert ok2 is False
    assert "missing or empty artifact_id" in err2

    ok3, err3 = _cleanup_stored_artifact(mock_graph_store, "   ")
    assert ok3 is False
    assert "missing or empty artifact_id" in err3

    ok4, err4 = _cleanup_stored_artifact(None, "art_123")
    assert ok4 is False
    assert "graph store is not configured" in err4


@pytest.mark.asyncio
async def test_save_artifact_empty_id_during_cancellation_reports_compensation_failed(lease_registry):
    """
    Verify that when save_artifact() returns an empty ID while context is cancelled,
    the service returns COMPENSATION_FAILED / OPERATION_FAILED and NEVER reports ROLLED_BACK_CLEAN with persisted=False.
    """
    import threading
    store_entered = threading.Event()
    allow_finish = threading.Event()
    cancel_evt = asyncio.Event()
    mock_store = MagicMock()

    def failing_save(agent_id, title, artifact_type, content):
        store_entered.set()
        allow_finish.wait(timeout=5.0)
        # Return empty ID (simulating store partial write failure)
        return ""

    mock_store.save_artifact = failing_save
    mock_store.delete_artifact = MagicMock(return_value=False)

    service = ComexToolService(graph_store=mock_store, lease_registry=lease_registry)
    ctx = create_tool_execution_context(
        task_id="task-cancel-empty-id",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.save_artifact(ctx, title="Title", content="Content"))

    while not store_entered.is_set():
        await asyncio.sleep(0.01)

    cancel_evt.set()
    allow_finish.set()
    res = await task

    assert res.success is False
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "COMPENSATION_FAILED"
    assert res.evidence.get("persisted") is True
    assert res.evidence.get("side_effect_possible") is True


# =============================================================================
# 18. Synchronous session resolver offloading and non-blocking event loop
# =============================================================================


@pytest.mark.asyncio
async def test_blocking_synchronous_session_resolver_does_not_stall_event_loop(lease_registry, mock_tab_session):
    """
    Verify that a blocking synchronous session resolver is offloaded to a worker thread
    and does not freeze the asyncio event loop or bypass context deadline bounds.
    Also verifies evidence reports resolver_type="synchronous", terminated=False, thread_state="in_flight_timeout".
    """
    def blocking_sync_session_resolver(tab_id, ws_url, worker_id=None):
        time.sleep(0.4)
        return mock_tab_session

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=blocking_sync_session_resolver,
        target_resolver=lambda lease, tab, trusted: ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None),
    )

    ctx = create_tool_execution_context(
        task_id="task-blocking-sync-session",
        target_tab="active_tab",
        trusted_server_context=True,
        timeout_seconds=0.05,
        lease_registry=lease_registry,
    )

    ticks = 0
    stop_ticker = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop_ticker.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())

    res = await service.inspect(ctx)
    stop_ticker.set()
    await ticker_task

    assert res.success is False
    assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
    assert res.evidence is not None
    assert res.evidence.get("resolver_type") == "synchronous"
    assert res.evidence.get("terminated") is False
    assert res.evidence.get("thread_state") == "in_flight_timeout"
    assert ticks >= 2


@pytest.mark.asyncio
async def test_synchronous_session_resolver_cancellation_reports_unterminated_thread(lease_registry, mock_tab_session):
    """
    Verify that cancelling session acquisition with a synchronous resolver unblocks the caller
    with ToolErrorCode.CANCELLED, while reporting terminated=False and thread_state="in_flight_cancelled".
    """
    import threading
    resolver_entered = threading.Event()
    hang_evt = threading.Event()

    def hanging_sync_session_resolver(tab_id, ws_url, worker_id=None):
        resolver_entered.set()
        hang_evt.wait(timeout=5.0)
        return mock_tab_session

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=hanging_sync_session_resolver,
        target_resolver=lambda lease, tab, trusted: ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None),
    )

    cancel_evt = asyncio.Event()
    ctx = create_tool_execution_context(
        task_id="task-cancel-sync-session",
        target_tab="active_tab",
        trusted_server_context=True,
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(service.inspect(ctx))

    while not resolver_entered.is_set():
        await asyncio.sleep(0.01)

    cancel_evt.set()
    res = await task
    hang_evt.set()

    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("resolver_type") == "synchronous"
    assert res.evidence.get("terminated") is False
    assert res.evidence.get("thread_state") == "in_flight_cancelled"


# =============================================================================
# 19. Context capability and deadline integrity with legitimate lease tokens
# =============================================================================


@pytest.mark.asyncio
async def test_context_reconstruction_with_widened_permissions_rejected(tool_service, lease_registry):
    """
    Verify that a caller possessing a legitimate worker lease and valid token cannot widen permissions
    by reconstructing a ToolExecutionContext with extra operations.
    """
    lease = lease_registry.acquire_lease("w_cap", "tab_cap", trusted_issuer=True)
    # Issue context legitimately permitting ONLY "inspect"
    legit_ctx = create_tool_execution_context(
        task_id="task-cap-widened",
        worker_id="w_cap",
        worker_token=lease.lease_token,
        target_tab="tab_cap",
        permitted_operations=frozenset({"inspect"}),
        lease_registry=lease_registry,
    )

    # Legitimate inspect call succeeds
    res_insp = await tool_service.inspect(legit_ctx)
    assert res_insp.success is True

    # Reconstruct context using the legitimate lease and token, but widening permitted_operations
    widened_ctx = ToolExecutionContext(
        run_id=legit_ctx.run_id,
        task_id=legit_ctx.task_id,
        worker_id=legit_ctx.worker_id,
        target_tab_id=legit_ctx.target_tab_id,
        lease=legit_ctx.lease,
        permitted_operations=frozenset({"inspect", "click", "save_artifact"}),  # Widened!
        deadline_mono=legit_ctx.deadline_mono,
        context_token=legit_ctx.context_token,
    )

    # Invoking click must fail closed as UNAUTHORIZED
    res_click = await tool_service.click(widened_ctx, selector="#submit")
    assert res_click.success is False
    assert res_click.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Permission check failed" in res_click.error_message

    # Invoking save_artifact must fail closed as UNAUTHORIZED
    res_save = await tool_service.save_artifact(widened_ctx, title="Title", content="Content")
    assert res_save.success is False
    assert res_save.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Permission check failed" in res_save.error_message


@pytest.mark.asyncio
async def test_context_reconstruction_with_extended_deadline_rejected(tool_service, lease_registry):
    """
    Verify that a caller possessing a legitimate worker lease and valid token cannot extend its deadline
    by reconstructing a ToolExecutionContext with a later deadline_mono.
    """
    lease = lease_registry.acquire_lease("w_dl", "tab_dl", ttl_seconds=5.0, trusted_issuer=True)
    legit_ctx = create_tool_execution_context(
        task_id="task-dl-extended",
        worker_id="w_dl",
        worker_token=lease.lease_token,
        target_tab="tab_dl",
        timeout_seconds=2.0,
        lease_registry=lease_registry,
    )

    # Reconstruct context extending deadline far into future
    extended_ctx = ToolExecutionContext(
        run_id=legit_ctx.run_id,
        task_id=legit_ctx.task_id,
        worker_id=legit_ctx.worker_id,
        target_tab_id=legit_ctx.target_tab_id,
        lease=legit_ctx.lease,
        permitted_operations=legit_ctx.permitted_operations,
        deadline_mono=time.monotonic() + 3600.0,  # Extended by 1 hour!
        context_token=legit_ctx.context_token,
    )

    res = await tool_service.inspect(extended_ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Deadline extension rejected" in res.error_message


@pytest.mark.asyncio
async def test_context_reconstruction_with_altered_target_tab_rejected(tool_service, lease_registry):
    """
    Verify that reconstructing a context with a different target_tab_id using a legitimate token is rejected.
    """
    lease = lease_registry.acquire_lease("w_tab", "tab_alpha", trusted_issuer=True)
    legit_ctx = create_tool_execution_context(
        task_id="task-tab-swap",
        worker_id="w_tab",
        worker_token=lease.lease_token,
        target_tab="tab_alpha",
        lease_registry=lease_registry,
    )

    tampered_ctx = ToolExecutionContext(
        run_id=legit_ctx.run_id,
        task_id=legit_ctx.task_id,
        worker_id=legit_ctx.worker_id,
        target_tab_id="tab_beta",  # Swapped tab!
        lease=legit_ctx.lease,
        permitted_operations=legit_ctx.permitted_operations,
        deadline_mono=legit_ctx.deadline_mono,
        context_token=legit_ctx.context_token,
    )

    res = await tool_service.inspect(tampered_ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Context target tab mismatch" in res.error_message


# =============================================================================
# 20. Input validation for artifact_type
# =============================================================================


@pytest.mark.asyncio
async def test_save_artifact_validates_artifact_type_type(tool_service, lease_registry):
    """
    Verify that save_artifact validates the type of artifact_type and returns VALIDATION_ERROR
    instead of raising AttributeError when a truthy non-string is passed.
    """
    ctx = create_tool_execution_context(
        task_id="task-val-art-type",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    res = await tool_service.save_artifact(ctx, title="Title", content="Content", artifact_type=123)
    assert res.success is False
    assert res.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "artifact_type must be a string" in res.error_message

    res2 = await tool_service.save_artifact(ctx, title="Title", content="Content", artifact_type=["notes"])
    assert res2.success is False
    assert res2.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "artifact_type must be a string" in res2.error_message


# =============================================================================
# 21. Context authority binding, rollback error propagation, and endpoint parsing
# =============================================================================


@pytest.mark.asyncio
async def test_reissuing_context_with_same_lease_does_not_widen_first_context(tool_service, lease_registry):
    """
    Verify that creating a second context (Context B with wider permissions) under the same worker lease
    does NOT overwrite the authority record of the first context (Context A with narrow permissions).
    """
    lease = lease_registry.acquire_lease("w_shared", "tab_shared", trusted_issuer=True)

    # 1. Issue Context A with permission to inspect only
    ctx_a = create_tool_execution_context(
        task_id="task-a",
        worker_id="w_shared",
        worker_token=lease.lease_token,
        target_tab="tab_shared",
        permitted_operations=frozenset({"inspect"}),
        lease_registry=lease_registry,
    )

    # 2. Issue Context B under the SAME lease with wider permissions
    ctx_b = create_tool_execution_context(
        task_id="task-b",
        worker_id="w_shared",
        worker_token=lease.lease_token,
        target_tab="tab_shared",
        permitted_operations=frozenset({"inspect", "click", "save_artifact"}),
        lease_registry=lease_registry,
    )

    # Context A and Context B must have distinct context tokens
    assert ctx_a.context_token != ctx_b.context_token

    # Context A can inspect
    res_a_insp = await tool_service.inspect(ctx_a)
    assert res_a_insp.success is True

    # Context A CANNOT click (fails at execution context permitted operations check)
    res_a_click = await tool_service.click(ctx_a, selector="#btn")
    assert res_a_click.success is False
    assert res_a_click.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "not permitted by execution context" in res_a_click.error_message

    # Reconstructing Context A with click using Context A's token fails because
    # Context A's authoritative record was NOT overwritten by Context B
    tampered_ctx_a = ToolExecutionContext(
        run_id=ctx_a.run_id,
        task_id=ctx_a.task_id,
        worker_id=ctx_a.worker_id,
        target_tab_id=ctx_a.target_tab_id,
        lease=ctx_a.lease,
        permitted_operations=frozenset({"inspect", "click"}),
        deadline_mono=ctx_a.deadline_mono,
        context_token=ctx_a.context_token,
    )
    res_tampered = await tool_service.click(tampered_ctx_a, selector="#btn")
    assert res_tampered.success is False
    assert res_tampered.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Permission check failed" in res_tampered.error_message

    # Context B CAN click because its own authoritative record permits it
    res_b_click = await tool_service.click(ctx_b, selector="#btn")
    assert res_b_click.success is True


@pytest.mark.asyncio
async def test_missing_worker_authority_record_fails_closed(tool_service, lease_registry):
    """
    Verify that a worker context missing an authoritative server issuance record
    fails closed (UNAUTHORIZED) rather than failing open.
    """
    lease = lease_registry.acquire_lease("w_fail_closed", "tab_fc", trusted_issuer=True)

    # Worker context with fabricated / missing context token
    fabricated_worker_ctx = ToolExecutionContext(
        run_id="run-fc",
        task_id="task-fc",
        worker_id="w_fail_closed",
        target_tab_id="tab_fc",
        lease=lease,
        permitted_operations=frozenset({"inspect"}),
        deadline_mono=None,
        context_token="fabricated_ctx_token_999",
    )

    res = await tool_service.inspect(fabricated_worker_ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "authoritative server issuance record" in res.error_message


@pytest.mark.asyncio
async def test_context_reconstruction_removing_cancellation_event_rejected(tool_service, lease_registry):
    """
    Verify that reconstructing a context to remove or alter its cancellation event is rejected.
    """
    cancel_evt = asyncio.Event()
    lease = lease_registry.acquire_lease("w_cancel_bind", "tab_cb", trusted_issuer=True)
    legit_ctx = create_tool_execution_context(
        task_id="task-cb",
        worker_id="w_cancel_bind",
        worker_token=lease.lease_token,
        target_tab="tab_cb",
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    # Tampered context removes cancellation event
    tampered_ctx = ToolExecutionContext(
        run_id=legit_ctx.run_id,
        task_id=legit_ctx.task_id,
        worker_id=legit_ctx.worker_id,
        target_tab_id=legit_ctx.target_tab_id,
        lease=legit_ctx.lease,
        permitted_operations=legit_ctx.permitted_operations,
        deadline_mono=legit_ctx.deadline_mono,
        cancellation_event=None,  # Removed!
        context_token=legit_ctx.context_token,
    )

    res = await tool_service.inspect(tampered_ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Cancellation handle tampering rejected" in res.error_message


@pytest.mark.asyncio
async def test_context_reconstruction_removing_deadline_rejected(tool_service, lease_registry):
    """
    Verify that reconstructing a context with deadline_mono=None when an authoritative finite
    deadline was issued is rejected, preventing indefinite execution.
    """
    lease = lease_registry.acquire_lease("w_dl_bind", "tab_db", trusted_issuer=True)
    legit_ctx = create_tool_execution_context(
        task_id="task-db",
        worker_id="w_dl_bind",
        worker_token=lease.lease_token,
        target_tab="tab_db",
        timeout_seconds=5.0,
        lease_registry=lease_registry,
    )

    # Tampered context replaces finite deadline with None
    tampered_ctx = ToolExecutionContext(
        run_id=legit_ctx.run_id,
        task_id=legit_ctx.task_id,
        worker_id=legit_ctx.worker_id,
        target_tab_id=legit_ctx.target_tab_id,
        lease=legit_ctx.lease,
        permitted_operations=legit_ctx.permitted_operations,
        deadline_mono=None,  # Stripped!
        context_token=legit_ctx.context_token,
    )

    res = await tool_service.inspect(tampered_ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "Deadline bypass rejected" in res.error_message


@pytest.mark.asyncio
async def test_save_artifact_rollback_failure_propagates_uncertainty_and_artifact_id(tool_service, lease_registry):
    """
    Verify that when store save_artifact suffers a partial-write rollback failure:
    1. The exception conveys the artifact ID via PartialWriteRollbackError or error message.
    2. ComexToolService.save_artifact() returns ToolErrorCode.OPERATION_FAILED.
    3. The evidence explicitly reports settlement="COMPENSATION_FAILED", persisted=True,
       side_effect_possible=True, and includes the affected artifact_id.
    4. The service NEVER claims persisted=False.
    """
    from graph.store import PartialWriteRollbackError

    class RollbackFailingStore:
        def __init__(self):
            self.search_memory = MagicMock(return_value=[])
            self.get_artifact = MagicMock(return_value=None)

        def save_artifact(self, agent_id, title, artifact_type, content):
            raise PartialWriteRollbackError(
                "art_failed_rollback_123",
                "Partial artifact write failed and compensating rollback failed for 'art_failed_rollback_123': simulated rollback lock error",
            )

    store = RollbackFailingStore()
    service = ComexToolService(graph_store=store, lease_registry=lease_registry)
    ctx = create_tool_execution_context(
        task_id="task-rollback-fail",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    res = await service.save_artifact(ctx, title="Title", content="Content")

    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert res.evidence is not None
    assert res.evidence.get("settlement") == "COMPENSATION_FAILED"
    assert res.evidence.get("persisted") is True
    assert res.evidence.get("side_effect_possible") is True
    assert res.evidence.get("artifact_id") == "art_failed_rollback_123"


@pytest.mark.asyncio
async def test_save_artifact_end_to_end_store_rollback_failure(graph_store, lease_registry):
    """
    End-to-end service test against real AgentGraphStore verifying that if both relationship
    creation and compensating rollback fail, the service returns COMPENSATION_FAILED with persisted=True.
    """
    graph_store.create_agent("agent_e2e_rollback", task="Test", intent="RESEARCH")
    original_execute = graph_store.conn.execute

    def fail_relationship_and_rollback(query, params=None):
        if "CREATED_ARTIFACT" in query:
            raise RuntimeError("Simulated relationship failure")
        if "DETACH DELETE" in query:
            raise RuntimeError("Simulated rollback failure")
        if params is not None:
            return original_execute(query, params)
        return original_execute(query)

    with graph_store._lock:
        graph_store.conn.execute = fail_relationship_and_rollback

    try:
        service = ComexToolService(graph_store=graph_store, lease_registry=lease_registry)
        ctx = create_tool_execution_context(
            task_id="task-e2e-rollback-fail",
            target_tab="active_tab",
            trusted_server_context=True,
            lease_registry=lease_registry,
            agent_id="agent_e2e_rollback",
        )

        res = await service.save_artifact(ctx, title="Doc", content="Content")

        assert res.success is False
        assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
        assert res.evidence is not None
        assert res.evidence.get("settlement") == "COMPENSATION_FAILED"
        assert res.evidence.get("persisted") is True
        assert res.evidence.get("side_effect_possible") is True
        assert res.evidence.get("artifact_id", "").startswith("art_")
    finally:
        with graph_store._lock:
            graph_store.conn.execute = original_execute


@pytest.mark.asyncio
async def test_malformed_port_in_websocket_url_returns_structured_failure(lease_registry, mock_tab_session):
    """
    Verify that a malformed port in the WebSocket URL (e.g. non-numeric port) returns a structured
    TARGET_RESOLUTION_FAILED ToolResult without raising an unhandled ValueError.
    """
    mock_session_resolver = AsyncMock(return_value=mock_tab_session)

    def malformed_port_target_resolver(lease, target_tab, trusted_app_context):
        return ("target-123", "ws://127.0.0.1:invalid-port/devtools/page/target-123", None)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=mock_session_resolver,
        target_resolver=malformed_port_target_resolver,
    )

    ctx = create_tool_execution_context(
        task_id="task-malformed-port",
        target_tab="active_tab",
        trusted_server_context=True,
        lease_registry=lease_registry,
    )

    res = await service.inspect(ctx)
    assert res.success is False
    assert res.error_code == ToolErrorCode.TARGET_RESOLUTION_FAILED.value
    assert "Malformed WebSocket URL" in res.error_message
    mock_session_resolver.assert_not_called()


# =============================================================================
# 22. Phase 2.1B: Bounded Browser Action Sequences & Per-Tab Concurrency
# =============================================================================


@pytest.mark.asyncio
async def test_sequence_execution_success_in_order(tool_service, lease_registry, mock_tab_session):
    """
    Verify valid multi-step sequence executes in order, evaluating preconditions and postconditions,
    and returning a complete list of completed steps with durations and evidence.
    """
    lease = lease_registry.acquire_lease("w_seq_ok", "tab_seq_ok", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-seq-ok",
        worker_id="w_seq_ok",
        worker_token=lease.lease_token,
        target_tab="tab_seq_ok",
        lease_registry=lease_registry,
    )

    steps = [
        {
            "action": "navigate",
            "url": "https://example.com/start",
            "preconditions": [],
            "postconditions": [{"type": "url_contains", "expected": "example.com"}],
        },
        {
            "action": "click",
            "selector": "#login-btn",
            "preconditions": [{"type": "element_exists", "selector": "#login-btn"}],
            "postconditions": [],
        },
        {
            "action": "type",
            "selector": "#username",
            "text": "alice",
            "clear_first": True,
            "preconditions": [{"type": "element_exists", "selector": "#username"}],
            "postconditions": [],
        },
        {
            "action": "extract",
            "selector": "#welcome-msg",
            "max_chars": 500,
            "preconditions": [],
            "postconditions": [],
        },
        {
            "action": "inspect",
            "max_elements": 10,
            "preconditions": [],
            "postconditions": [],
        },
    ]

    res = await tool_service.perform_sequence(ctx, steps=steps)
    assert res.success is True
    assert res.data["step_count"] == 5
    completed = res.data["completed_steps"]
    assert len(completed) == 5
    assert [c["action"] for c in completed] == ["navigate", "click", "type", "extract", "inspect"]
    assert completed[0]["step_index"] == 0
    assert completed[4]["step_index"] == 4


@pytest.mark.asyncio
async def test_sequence_validation_rejects_unsupported_actions_and_bounds_violations(tool_service, lease_registry):
    """
    Verify that sequence input validation rejects non-lists, empty sequences, overlong sequences,
    unsupported operations, overlong arguments, malformed conditions, and non-int booleans.
    """
    lease = lease_registry.acquire_lease("w_val", "tab_val", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-seq-val",
        worker_id="w_val",
        worker_token=lease.lease_token,
        target_tab="tab_val",
        lease_registry=lease_registry,
    )

    # 1. Non-list steps
    r1 = await tool_service.perform_sequence(ctx, steps="not_a_list")  # type: ignore
    assert r1.success is False and r1.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "steps must be a list" in r1.error_message

    # 2. Empty list
    r2 = await tool_service.perform_sequence(ctx, steps=[])
    assert r2.success is False and r2.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "Sequence cannot be empty" in r2.error_message

    # 3. Exceeds 15 steps
    sixteen_steps = [{"action": "click", "selector": f"#btn-{i}"} for i in range(16)]
    r3 = await tool_service.perform_sequence(ctx, steps=sixteen_steps)
    assert r3.success is False and r3.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "exceeds maximum allowed 15" in r3.error_message

    # 4. Unknown fields on step
    r4 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "click", "selector": "#btn", "unauthorized_prop": 123}]
    )
    assert r4.success is False and r4.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "contains unknown fields" in r4.error_message

    # 5. Unsupported action (e.g. eval)
    r5 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "eval", "code": "alert(1)"}]
    )
    assert r5.success is False and r5.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "action 'eval' is invalid or unsupported" in r5.error_message

    # 6. Non-int boolean for max_elements / max_chars
    r6 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "inspect", "max_elements": True}]
    )
    assert r6.success is False and r6.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "max_elements must be a positive integer" in r6.error_message

    r7 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "extract", "max_chars": False}]
    )
    assert r7.success is False and r7.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "max_chars must be a positive integer" in r7.error_message

    # 7. Overlong selector (> 500 chars)
    r8 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "click", "selector": "#" + "a" * 501}]
    )
    assert r8.success is False and r8.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "selector exceeds 500 characters" in r8.error_message

    # 8. Overlong text (> 10000 chars)
    r9 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "type", "selector": "#input", "text": "x" * 10001}]
    )
    assert r9.success is False and r9.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "text exceeds 10000 characters" in r9.error_message

    # 9. Overlong URL (> 2000 chars)
    r10 = await tool_service.perform_sequence(
        ctx, steps=[{"action": "navigate", "url": "https://example.com/" + "u" * 2001}]
    )
    assert r10.success is False and r10.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "url exceeds 2000 characters" in r10.error_message

    # 10. Unknown condition field
    r11 = await tool_service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#btn",
            "preconditions": [{"type": "element_exists", "selector": "#btn", "extra_bad": True}],
        }],
    )
    assert r11.success is False and r11.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "contains unknown fields" in r11.error_message

    # 11. Invalid condition type
    r12 = await tool_service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#btn",
            "preconditions": [{"type": "magic_condition"}],
        }],
    )
    assert r12.success is False and r12.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "condition type 'magic_condition' is invalid" in r12.error_message

    # 12. More than 5 conditions per step
    six_conds = [{"type": "url_contains", "expected": f"part_{i}"} for i in range(6)]
    r13 = await tool_service.perform_sequence(
        ctx,
        steps=[{"action": "click", "selector": "#btn", "preconditions": six_conds}],
    )
    assert r13.success is False and r13.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "exceeds 5" in r13.error_message


@pytest.mark.asyncio
async def test_sequence_revalidates_authoritative_claims_per_step(tool_service, lease_registry):
    """
    Verify that each step in a sequence is authoritatively checked against the server issuance record.
    If context authority allows only 'inspect' and 'perform_sequence', attempting a 'click' step
    aborts the sequence as UNAUTHORIZED at that step without executing it.
    """
    lease = lease_registry.acquire_lease("w_auth_seq", "tab_auth_seq", trusted_issuer=True)
    # Legitimately authorized ONLY for perform_sequence and inspect
    ctx = create_tool_execution_context(
        task_id="task-seq-auth",
        worker_id="w_auth_seq",
        worker_token=lease.lease_token,
        target_tab="tab_auth_seq",
        permitted_operations=frozenset({"perform_sequence", "inspect"}),
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "inspect", "max_elements": 10},
        {"action": "click", "selector": "#unauthorized-btn"},  # Not permitted!
    ]

    res = await tool_service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "authorization check failed" in res.error_message
    assert res.evidence is not None
    assert res.evidence.get("failed_step_index") == 1
    assert res.evidence.get("failed_action") == "click"
    assert res.evidence.get("condition_stage") == "authorization"
    # Step 0 succeeded and is recorded in completed_steps
    assert len(res.evidence.get("completed_steps", [])) == 1
    assert res.evidence["completed_steps"][0]["action"] == "inspect"


@pytest.mark.asyncio
async def test_sequence_precondition_failure_aborts_before_action(tool_service, lease_registry, mock_tab_session):
    """
    Verify that a failing precondition aborts the sequence before the action is executed.
    The action is NOT called and evidence documents condition_stage='precondition'.
    """
    mock_tab_session.gate.async_attempt.reset_mock()
    mock_tab_session.extract_read_only.return_value = {"url": "https://example.com/home", "title": "Home"}

    lease = lease_registry.acquire_lease("w_pre_fail", "tab_pre_fail", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-pre-fail",
        worker_id="w_pre_fail",
        worker_token=lease.lease_token,
        target_tab="tab_pre_fail",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "inspect", "max_elements": 5},
        {
            "action": "click",
            "selector": "#btn-must-fail",
            "preconditions": [{"type": "url_contains", "expected": "dashboard_secret"}],
        },
    ]

    res = await tool_service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "precondition failed" in res.error_message
    assert res.evidence is not None
    assert res.evidence.get("failed_step_index") == 1
    assert res.evidence.get("condition_stage") == "precondition"
    assert res.evidence["condition"]["type"] == "url_contains"
    assert len(res.evidence.get("completed_steps", [])) == 1

    # The click action was NEVER executed!
    mock_tab_session.gate.async_attempt.assert_not_called()


@pytest.mark.asyncio
async def test_sequence_postcondition_failure_stops_sequence(tool_service, lease_registry, mock_tab_session):
    """
    Verify that a failing postcondition halts the sequence immediately after the action executes.
    Subsequent steps are NOT executed. Evidence documents condition_stage='postcondition'
    and side_effect_possible=True.
    """
    mock_tab_session.gate.async_attempt.reset_mock()
    mock_tab_session.extract_read_only.return_value = {"url": "https://example.com/checkout", "title": "Wrong Title"}

    lease = lease_registry.acquire_lease("w_post_fail", "tab_post_fail", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-post-fail",
        worker_id="w_post_fail",
        worker_token=lease.lease_token,
        target_tab="tab_post_fail",
        lease_registry=lease_registry,
    )

    steps = [
        {
            "action": "click",
            "selector": "#submit-btn",
            "postconditions": [{"type": "title_contains", "expected": "Order Confirmed"}],
        },
        {"action": "extract", "selector": "#confirmation-code"},  # Must NOT be reached
    ]

    res = await tool_service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "postcondition failed" in res.error_message
    assert res.evidence is not None
    assert res.evidence.get("failed_step_index") == 0
    assert res.evidence.get("condition_stage") == "postcondition"
    assert res.evidence.get("side_effect_possible") is True
    # Step 0 executed and is in completed_steps
    assert len(res.evidence.get("completed_steps", [])) == 1
    assert res.evidence["completed_steps"][0]["action"] == "click"
    mock_tab_session.gate.async_attempt.assert_called_once()


@pytest.mark.asyncio
async def test_sequence_element_visibility_condition_checks(tool_service, lease_registry, mock_tab_session):
    """
    Verify element_visible condition evaluates the specific selected element's bounding box
    and actionability, failing when dimensions are non-positive.
    """
    lease = lease_registry.acquire_lease("w_vis", "tab_vis", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-vis",
        worker_id="w_vis",
        worker_token=lease.lease_token,
        target_tab="tab_vis",
        lease_registry=lease_registry,
    )

    # 1. Zero bounding box -> fails
    invisible_binding = MagicMock(spec=TargetBinding)
    invisible_binding.bounding_box = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    mock_tab_session.resolve_and_bind.return_value = invisible_binding

    res_fail = await tool_service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#hidden-elem",
            "preconditions": [{"type": "element_visible", "selector": "#hidden-elem"}],
        }],
    )
    assert res_fail.success is False
    assert res_fail.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "non-positive dimensions" in res_fail.error_message

    # 2. Positive bounding box -> passes
    visible_binding = MagicMock(spec=TargetBinding)
    visible_binding.bounding_box = {"x": 10.0, "y": 20.0, "width": 100.0, "height": 40.0}
    mock_tab_session.resolve_and_bind.return_value = visible_binding

    res_ok = await tool_service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#visible-elem",
            "preconditions": [{"type": "element_visible", "selector": "#visible-elem"}],
        }],
    )
    assert res_ok.success is True


@pytest.mark.asyncio
async def test_sequence_value_matches_condition_checks(tool_service, lease_registry, mock_tab_session):
    """
    Verify value_matches condition tests live input value against expected value.
    """
    lease = lease_registry.acquire_lease("w_val_m", "tab_val_m", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-val-m",
        worker_id="w_val_m",
        worker_token=lease.lease_token,
        target_tab="tab_val_m",
        lease_registry=lease_registry,
    )

    # Mock extract_read_only returning text or value
    mock_tab_session.extract_read_only.return_value = {"value": "expected_token_123", "text": ""}

    # 1. Matching value passes
    res_ok = await tool_service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#submit",
            "preconditions": [{"type": "value_matches", "selector": "#token-input", "expected": "expected_token_123"}],
        }],
    )
    assert res_ok.success is True

    # 2. Mismatched value fails
    res_fail = await tool_service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#submit",
            "preconditions": [{"type": "value_matches", "selector": "#token-input", "expected": "different_value"}],
        }],
    )
    assert res_fail.success is False
    assert res_fail.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "expected 'different_value'" in res_fail.error_message


@pytest.mark.asyncio
async def test_sequence_in_flight_cancellation_reports_side_effect_uncertainty(lease_registry, mock_tab_session):
    """
    Verify that if cancellation occurs while an action is in flight:
    1. The sequence aborts immediately and unblocks caller with CANCELLED.
    2. evidence reports failed_step_index and in_flight=True.
    3. evidence reports side_effect_possible=True.
    4. Action lock is cleanly released.
    """
    action_started = asyncio.Event()

    async def slow_gate_attempt(*args, **kwargs):
        action_started.set()
        await asyncio.sleep(2.0)
        return GateResult(decision=None, executed=True)

    mock_tab_session.gate.async_attempt = slow_gate_attempt

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=mock_tab_session),
        target_resolver=lambda lease, tab, trusted: ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None),
    )

    cancel_evt = asyncio.Event()
    lease = lease_registry.acquire_lease("w_inflight", "tab_inflight", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-seq-cancel",
        worker_id="w_inflight",
        worker_token=lease.lease_token,
        target_tab="tab_inflight",
        cancellation_event=cancel_evt,
        lease_registry=lease_registry,
    )

    task = asyncio.create_task(
        service.perform_sequence(ctx, steps=[{"action": "click", "selector": "#slow-btn"}])
    )

    await action_started.wait()
    assert mock_tab_session.action_lock.locked()

    # Cancel while action is in flight
    cancel_evt.set()
    res = await task

    assert res.success is False
    assert res.error_code == ToolErrorCode.CANCELLED.value
    assert res.evidence is not None
    assert res.evidence.get("failed_step_index") == 0
    assert res.evidence.get("in_flight") is True
    assert res.evidence.get("side_effect_possible") is True
    # Action lock must be unlocked
    assert not mock_tab_session.action_lock.locked()


@pytest.mark.asyncio
async def test_sequence_lock_acquisition_respects_deadline(lease_registry, mock_tab_session):
    """
    Verify that waiting to acquire the tab lock is bounded by the context deadline.
    If the tab is busy and the timeout expires before acquiring the lock:
    1. Returns DEADLINE_EXCEEDED.
    2. Reports side_effect_possible=False because no steps ever executed.
    3. Releases all resources cleanly.
    """
    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=mock_tab_session),
        target_resolver=lambda lease, tab, trusted: ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None),
    )

    # Hold the tab lock externally to simulate busy tab
    await mock_tab_session.action_lock.acquire()

    lease = lease_registry.acquire_lease("w_lock_dl", "tab_lock_dl", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-lock-dl",
        worker_id="w_lock_dl",
        worker_token=lease.lease_token,
        target_tab="tab_lock_dl",
        timeout_seconds=0.05,  # Short timeout
        lease_registry=lease_registry,
    )

    try:
        res = await service.perform_sequence(ctx, steps=[{"action": "click", "selector": "#btn"}])
        assert res.success is False
        assert res.error_code == ToolErrorCode.DEADLINE_EXCEEDED.value
        assert res.evidence is not None
        assert res.evidence.get("side_effect_possible") is False
        assert res.evidence.get("in_flight") is False
        assert len(res.evidence.get("completed_steps", [])) == 0
    finally:
        mock_tab_session.action_lock.release()


@pytest.mark.asyncio
async def test_same_tab_operations_are_strictly_serialized(lease_registry, mock_tab_session):
    """
    Verify that a sequence and concurrent operations on the same tab are strictly serialized
    and cannot interleave their actions.
    """
    sequence_execution_order = []

    async def logging_click(*args, **kwargs):
        sequence_execution_order.append("seq_click_start")
        await asyncio.sleep(0.05)
        sequence_execution_order.append("seq_click_end")
        return GateResult(decision=None, executed=True)

    async def logging_type(*args, **kwargs):
        sequence_execution_order.append("seq_type_start")
        await asyncio.sleep(0.05)
        sequence_execution_order.append("seq_type_end")
        return GateResult(decision=None, executed=True)

    async def logging_inspect(*args, **kwargs):
        sequence_execution_order.append("single_inspect")
        return {"returned": 1, "elements": []}

    mock_tab_session.gate.async_attempt = AsyncMock(side_effect=[
        await logging_click(),
        await logging_type(),
    ]) if False else logging_click  # Handled dynamically below

    # Setup dispatch on gate
    call_count = 0
    async def dynamic_gate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await logging_click(*args, **kwargs)
        else:
            return await logging_type(*args, **kwargs)

    mock_tab_session.gate.async_attempt = dynamic_gate
    mock_tab_session.inspect_elements = logging_inspect

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=mock_tab_session),
        target_resolver=lambda lease, tab, trusted: ("target-123", "ws://127.0.0.1:9222/devtools/page/target-123", None),
    )

    lease_seq = lease_registry.acquire_lease("w_seq_ser", "tab_ser", trusted_issuer=True)
    ctx_seq = create_tool_execution_context(
        task_id="task-seq-ser",
        worker_id="w_seq_ser",
        worker_token=lease_seq.lease_token,
        target_tab="tab_ser",
        lease_registry=lease_registry,
    )

    ctx_single = create_tool_execution_context(
        task_id="task-single-ser",
        worker_id="w_seq_ser",
        worker_token=lease_seq.lease_token,
        target_tab="tab_ser",
        lease_registry=lease_registry,
    )

    # Launch sequence (click then type) and single inspect and extract concurrently
    seq_task = asyncio.create_task(
        service.perform_sequence(ctx_seq, steps=[
            {"action": "click", "selector": "#btn"},
            {"action": "type", "selector": "#in", "text": "hello"},
        ])
    )
    # Brief yield so sequence starts and acquires lock first
    await asyncio.sleep(0.01)

    single_inspect_task = asyncio.create_task(service.inspect(ctx_single))
    single_extract_task = asyncio.create_task(service.extract(ctx_single))

    res_seq, res_inspect, res_extract = await asyncio.gather(seq_task, single_inspect_task, single_extract_task)
    assert res_seq.success is True
    assert res_inspect.success is True
    assert res_extract.success is True

    # Inspect must NOT have run between seq_click_start and seq_type_end
    click_start_idx = sequence_execution_order.index("seq_click_start")
    type_end_idx = sequence_execution_order.index("seq_type_end")
    inspect_idx = sequence_execution_order.index("single_inspect")

    assert inspect_idx > type_end_idx or inspect_idx < click_start_idx


@pytest.mark.asyncio
async def test_independent_tabs_operate_concurrently(lease_registry):
    """
    Verify that sequences on independent tabs (distinct canonical target IDs) run concurrently
    without blocking each other.
    """
    lock_a = asyncio.Lock()
    lock_b = asyncio.Lock()

    binding_a = MagicMock(spec=TargetBinding)
    binding_b = MagicMock(spec=TargetBinding)

    async def fake_gate_a(*args, **kwargs):
        await asyncio.sleep(0.1)
        return GateResult(decision=None, executed=True)

    async def fake_gate_b(*args, **kwargs):
        await asyncio.sleep(0.1)
        return GateResult(decision=None, executed=True)

    session_a = MagicMock()
    session_a.action_lock = lock_a
    session_a.tracker.state.is_target_subtree_quiet = True
    session_a.resolve_and_bind = AsyncMock(return_value=binding_a)
    session_a.gate = MagicMock()
    session_a.gate.async_attempt = AsyncMock(side_effect=fake_gate_a)

    session_b = MagicMock()
    session_b.action_lock = lock_b
    session_b.tracker.state.is_target_subtree_quiet = True
    session_b.resolve_and_bind = AsyncMock(return_value=binding_b)
    session_b.gate = MagicMock()
    session_b.gate.async_attempt = AsyncMock(side_effect=fake_gate_b)

    async def multi_session_resolver(tab_id, ws_url, worker_id=None):
        if "tab_a" in tab_id:
            return session_a
        return session_b

    def multi_target_resolver(lease, target_tab, trusted):
        if "tab_a" in target_tab:
            return ("target-A", "ws://127.0.0.1:9222/devtools/page/target-A", None)
        return ("target-B", "ws://127.0.0.1:9222/devtools/page/target-B", None)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=multi_session_resolver,
        target_resolver=multi_target_resolver,
    )

    lease_a = lease_registry.acquire_lease("w_a", "tab_a", trusted_issuer=True)
    ctx_a = create_tool_execution_context(
        task_id="task-a",
        worker_id="w_a",
        worker_token=lease_a.lease_token,
        target_tab="tab_a",
        lease_registry=lease_registry,
    )

    lease_b = lease_registry.acquire_lease("w_b", "tab_b", trusted_issuer=True)
    ctx_b = create_tool_execution_context(
        task_id="task-b",
        worker_id="w_b",
        worker_token=lease_b.lease_token,
        target_tab="tab_b",
        lease_registry=lease_registry,
    )

    t0 = time.monotonic()
    task_a = asyncio.create_task(
        service.perform_sequence(ctx_a, steps=[{"action": "click", "selector": "#btn-a"}])
    )
    task_b = asyncio.create_task(
        service.perform_sequence(ctx_b, steps=[{"action": "click", "selector": "#btn-b"}])
    )

    res_a, res_b = await asyncio.gather(task_a, task_b)
    dur = time.monotonic() - t0

    assert res_a.success is True
    assert res_b.success is True
    # Two 100ms operations running concurrently should take ~0.1s, NOT 0.2s!
    assert dur < 0.18


@pytest.mark.asyncio
async def test_independently_created_sessions_share_canonical_lock_and_serialize_work(lease_registry):
    """
    Verify that TabSessionContext instances created for the same physical canonical target ID
    (via distinct worker sessions) share the identical asyncio.Lock object, and that public
    inspect and extract operations cannot interleave with a sequence on that tab.
    Constructs real TabSessionContext instances rather than manually injecting locks to mocks.
    """
    from nervous.cdp_session import TabSessionContext, get_canonical_tab_lock, extract_canonical_target_id

    cdp_client_1 = MagicMock()
    cdp_client_1.ws_url = "ws://127.0.0.1:9222/devtools/page/target-canonical-shared"
    cdp_client_2 = MagicMock()
    cdp_client_2.ws_url = "ws://127.0.0.1:9222/devtools/page/target-canonical-shared"

    # Construct separate real TabSessionContext instances with no manual lock assignment
    session_1 = TabSessionContext(
        tab_id="tab_canonical_1",
        target_id="target-canonical-shared",
        cdp_client=cdp_client_1,
        worker_id="w_worker1",
    )
    session_2 = TabSessionContext(
        tab_id="tab_canonical_2",
        target_id="target-canonical-shared",
        cdp_client=cdp_client_2,
        worker_id="w_worker2",
    )

    # 1. Real TabSessionContext instances constructed independently share the identical action_lock
    assert session_1.action_lock is session_2.action_lock

    # 2. A session constructed for a different canonical target has an independent lock
    cdp_client_diff = MagicMock()
    cdp_client_diff.ws_url = "ws://127.0.0.1:9222/devtools/page/target-canonical-different"
    session_diff = TabSessionContext(
        tab_id="tab_diff",
        target_id="target-canonical-different",
        cdp_client=cdp_client_diff,
        worker_id="w_worker_diff",
    )
    assert session_diff.action_lock is not session_1.action_lock

    # 3. Simulate two distinct sessions created independently for the same canonical target
    order = []

    async def slow_click(*args, **kwargs):
        order.append("seq_step_start")
        await asyncio.sleep(0.06)
        order.append("seq_step_end")
        return GateResult(decision=None, executed=True)

    session_1.gate.async_attempt = AsyncMock(side_effect=slow_click)
    session_1.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))

    async def session2_extract(*args, **kwargs):
        order.append("session2_extract")
        return {"text": "session 2 content"}

    session_2.extract_read_only = AsyncMock(side_effect=session2_extract)

    # Session resolver returns session_1 for worker_1, session_2 for worker_2
    async def resolving_session(tab_id, ws_url, worker_id=None):
        if worker_id == "w_worker1":
            return session_1
        return session_2

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=resolving_session,
        target_resolver=lambda lease, tab, trusted: (
            "target-canonical-shared",
            "ws://127.0.0.1:9222/devtools/page/target-canonical-shared",
            None,
        ),
    )

    lease_1 = lease_registry.acquire_lease("w_worker1", "tab_canonical_1", trusted_issuer=True)
    ctx_1 = create_tool_execution_context(
        task_id="task-w1",
        worker_id="w_worker1",
        worker_token=lease_1.lease_token,
        target_tab="tab_canonical_1",
        lease_registry=lease_registry,
    )

    # Acquire lease on same target tab for worker_2 after release/transfer or simulate concurrent caller
    lease_registry._leases["w_worker2"] = TabLease(
        worker_id="w_worker2",
        tab_id="tab_canonical_2",
        lease_token="tok-w2",
        expires_at_mono=time.monotonic() + 300.0,
    )
    ctx_2 = create_tool_execution_context(
        task_id="task-w2",
        worker_id="w_worker2",
        worker_token="tok-w2",
        target_tab="tab_canonical_2",
        lease_registry=lease_registry,
    )

    # Task 1: Sequence with click on session 1
    t1 = asyncio.create_task(
        service.perform_sequence(ctx_1, steps=[{"action": "click", "selector": "#btn"}])
    )
    await asyncio.sleep(0.01)

    # Task 2: Extract on session 2
    t2 = asyncio.create_task(service.extract(ctx_2))

    res1, res2 = await asyncio.gather(t1, t2)
    assert res1.success is True
    assert res2.success is True

    # Because they share real action_lock, session2_extract MUST execute strictly after seq_step_end
    assert order == ["seq_step_start", "seq_step_end", "session2_extract"]


@pytest.mark.asyncio
async def test_sequence_title_condition_fails_on_empty_live_title_without_stale_tracker_fallback(lease_registry):
    """
    Verify that an empty live document title ("") is distinguished from an unread title and does NOT
    fall back to a stale cached title in tracker.state. The title_contains precondition must fail closed.
    """
    session = MagicMock()
    session.tab_id = "tab_title_stale"
    session.target_id = "target_title_stale"
    session.action_lock = asyncio.Lock()

    # Cached tracker state has an old stale title "Secret Admin Dashboard"
    session.tracker.state.is_target_subtree_quiet = True
    session.tracker.state.current_title = "Secret Admin Dashboard"
    session.tracker.state.current_url = "https://example.com/admin"

    # Live CDP Runtime.evaluate returns an empty title ""
    async def mock_cdp_send(method, params=None):
        if method == "Runtime.evaluate":
            expr = params.get("expression", "") if params else ""
            if "document.title" in expr:
                return {"result": {"value": ""}}
            if "window.location.href" in expr:
                return {"result": {"value": "https://example.com/blank"}}
        return {}

    session.cdp = MagicMock()
    session.cdp.send = AsyncMock(side_effect=mock_cdp_send)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_title_stale",
            "ws://127.0.0.1:9222/devtools/page/target_title_stale",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_stale_title", "tab_title_stale", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-title-stale",
        worker_id="w_stale_title",
        worker_token=lease.lease_token,
        target_tab="tab_title_stale",
        lease_registry=lease_registry,
    )

    steps = [
        {
            "action": "click",
            "selector": "#btn",
            "preconditions": [
                {"type": "title_contains", "expected": "Admin"}
            ],
        }
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    # Must fail because live title is "" which does NOT contain "Admin", despite tracker having "Admin"
    assert res.success is False
    assert res.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "Title '' does not contain expected 'Admin'" in res.error_message


@pytest.mark.asyncio
async def test_sequence_fails_closed_when_live_condition_state_unavailable(lease_registry):
    """
    Verify that when live browser state cannot be read (e.g. CDP evaluate raises error and extract_read_only fails),
    URL and title conditions fail closed rather than falling back to stale tracker values.
    """
    session = MagicMock()
    session.tab_id = "tab_fail_closed"
    session.target_id = "target_fail_closed"
    session.action_lock = asyncio.Lock()

    # Cached tracker state has URL and title that would match
    session.tracker.state.is_target_subtree_quiet = True
    session.tracker.state.current_url = "https://example.com/dashboard"
    session.tracker.state.current_title = "Dashboard"

    # CDP send raises an error
    session.cdp = MagicMock()
    session.cdp.send = AsyncMock(side_effect=RuntimeError("CDP connection broken"))
    session.extract_read_only = AsyncMock(side_effect=RuntimeError("DOM extraction unavailable"))

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_fail_closed",
            "ws://127.0.0.1:9222/devtools/page/target_fail_closed",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_fail_closed", "tab_fail_closed", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-fail-closed",
        worker_id="w_fail_closed",
        worker_token=lease.lease_token,
        target_tab="tab_fail_closed",
        lease_registry=lease_registry,
    )

    # 1. URL condition fails closed
    res_url = await service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#btn",
            "preconditions": [{"type": "url_contains", "expected": "dashboard"}],
        }],
    )
    assert res_url.success is False
    assert res_url.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "Live URL condition check failed" in res_url.error_message

    # 2. Title condition fails closed
    res_title = await service.perform_sequence(
        ctx,
        steps=[{
            "action": "click",
            "selector": "#btn",
            "preconditions": [{"type": "title_contains", "expected": "Dashboard"}],
        }],
    )
    assert res_title.success is False
    assert res_title.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "Live title condition check failed" in res_title.error_message


@pytest.mark.asyncio
async def test_sequence_rejects_irrelevant_action_fields_and_strictly_bounds_all_fields(lease_registry):
    """
    Verify that fields irrelevant to a specific action (e.g. text on click, selector on navigate)
    are strictly rejected, and bounds on all fields (selector <= 500, text <= 10000, url <= 2000,
    clear_first bool, max_elements/max_chars positive int) cannot be bypassed.
    """
    lease = lease_registry.acquire_lease("w_fields", "tab_fields", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-fields",
        worker_id="w_fields",
        worker_token=lease.lease_token,
        target_tab="tab_fields",
        lease_registry=lease_registry,
    )
    service = ComexToolService(lease_registry=lease_registry)

    # 1. Click step carrying text field (even normal text) is rejected as irrelevant
    r1 = await service.perform_sequence(ctx, steps=[{"action": "click", "selector": "#btn", "text": "hello"}])
    assert r1.success is False
    assert r1.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "contains unknown fields: ['text']" in r1.error_message

    # 2. Click step carrying oversized text field cannot bypass text bounds
    r2 = await service.perform_sequence(ctx, steps=[{"action": "click", "selector": "#btn", "text": "x" * 15000}])
    assert r2.success is False
    assert r2.error_code == ToolErrorCode.VALIDATION_ERROR.value

    # 3. Navigate step carrying selector or text is rejected
    r3 = await service.perform_sequence(ctx, steps=[{"action": "navigate", "url": "https://example.com", "selector": "#btn"}])
    assert r3.success is False
    assert r3.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "contains unknown fields: ['selector']" in r3.error_message

    # 4. Inspect step carrying selector is rejected
    r4 = await service.perform_sequence(ctx, steps=[{"action": "inspect", "max_elements": 10, "selector": "#box"}])
    assert r4.success is False
    assert r4.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "contains unknown fields: ['selector']" in r4.error_message

    # 5. Type step with boolean max_elements or clear_first as int
    r5 = await service.perform_sequence(ctx, steps=[{"action": "type", "selector": "#input", "text": "abc", "clear_first": 123}])
    assert r5.success is False
    assert r5.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "clear_first' must be a boolean" in r5.error_message

    # 6. Type step with oversized text (> 10000 chars)
    r6 = await service.perform_sequence(ctx, steps=[{"action": "type", "selector": "#input", "text": "a" * 10001}])
    assert r6.success is False
    assert r6.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "text exceeds 10000 characters" in r6.error_message

    # 7. Extract step with oversized max_chars or bool
    r7 = await service.perform_sequence(ctx, steps=[{"action": "extract", "max_chars": True}])
    assert r7.success is False
    assert r7.error_code == ToolErrorCode.VALIDATION_ERROR.value
    assert "max_chars must be a positive integer" in r7.error_message


@pytest.mark.asyncio
async def test_sequence_aborts_when_worker_lease_revoked_mid_execution(lease_registry):
    """
    Verify that if a worker lease is revoked while a sequence is executing,
    the active lease revalidation check immediately halts the sequence prior to the next action.
    """
    session = MagicMock()
    session.tab_id = "tab_revoked"
    session.target_id = "target_revoked"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True
    session.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))

    step_count = 0

    async def click_and_revoke(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        # Revoke the lease after step 0 executes!
        lease_registry.revoke_lease("w_revoked", trusted_caller=True)
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=click_and_revoke)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_revoked",
            "ws://127.0.0.1:9222/devtools/page/target_revoked",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_revoked", "tab_revoked", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-revoked",
        worker_id="w_revoked",
        worker_token=lease.lease_token,
        target_tab="tab_revoked",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "click", "selector": "#btn-1"},
        {"action": "click", "selector": "#btn-2"},
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    # Step 0 succeeded, but before step 1 could execute, lease was revoked and caught!
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "lease validation failed" in res.error_message or "lease revoked" in res.error_message
    assert res.evidence["failed_step_index"] == 1
    assert len(res.evidence["completed_steps"]) == 1
    assert step_count == 1  # Step 2 never ran!


@pytest.mark.asyncio
async def test_cdp_live_evaluations_for_visibility_and_value(lease_registry):
    """
    Exercise realistic CDP Runtime.evaluate responses for element visibility, disabled state,
    and value comparisons, verifying that condition checks accurately interpret CDP data
    and report evaluation errors.
    """
    session = MagicMock()
    session.tab_id = "tab_cdp_eval"
    session.target_id = "target_cdp_eval"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    eval_responses = {}

    async def mock_cdp_send(method, params=None):
        if method == "Runtime.evaluate":
            expr = params.get("expression", "") if params else ""
            for pattern, resp in eval_responses.items():
                if pattern in expr:
                    return resp
        return {}

    session.cdp = MagicMock()
    session.cdp.send = AsyncMock(side_effect=mock_cdp_send)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_cdp_eval",
            "ws://127.0.0.1:9222/devtools/page/target_cdp_eval",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_cdp_eval", "tab_cdp_eval", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-cdp-eval",
        worker_id="w_cdp_eval",
        worker_token=lease.lease_token,
        target_tab="tab_cdp_eval",
        lease_registry=lease_registry,
    )

    # 1. Element hidden via display: none
    eval_responses = {
        "#hidden-el": {
            "result": {
                "value": {"exists": True, "visible": False, "reason": "display is none"}
            }
        }
    }
    r1 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#hidden-el",
        "preconditions": [{"type": "element_visible", "selector": "#hidden-el"}],
    }])
    assert r1.success is False
    assert "is not visible: display is none" in r1.error_message

    # 2. Element disabled
    eval_responses = {
        "#disabled-btn": {
            "result": {
                "value": {"exists": True, "visible": True, "disabled": True, "width": 50, "height": 20}
            }
        }
    }
    r2 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#disabled-btn",
        "preconditions": [{"type": "element_visible", "selector": "#disabled-btn"}],
    }])
    assert r2.success is False
    assert "is disabled" in r2.error_message

    # 3. Value matches successfully
    eval_responses = {
        "#email-input": {
            "result": {"value": "test@example.com"}
        }
    }
    session.gate.async_attempt = AsyncMock(return_value=GateResult(decision=None, executed=True))
    session.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))
    r3 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#submit-btn",
        "preconditions": [{"type": "value_matches", "selector": "#email-input", "expected": "test@example.com"}],
    }])
    assert r3.success is True

    # 4. Value matches failure when actual differs
    eval_responses = {
        "#email-input": {
            "result": {"value": "wrong@example.com"}
        }
    }
    r4 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#submit-btn",
        "preconditions": [{"type": "value_matches", "selector": "#email-input", "expected": "test@example.com"}],
    }])
    assert r4.success is False
    assert "value is 'wrong@example.com', expected 'test@example.com'" in r4.error_message

    # 5. Value matches failure when element is not found / evaluate returns null
    eval_responses = {
        "#email-input": {
            "result": {"value": None}
        }
    }
    session.extract_read_only = AsyncMock(return_value=None)
    r5 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#submit-btn",
        "preconditions": [{"type": "value_matches", "selector": "#email-input", "expected": "test@example.com"}],
    }])
    assert r5.success is False
    assert "value could not be confirmed" in r5.error_message


@pytest.mark.asyncio
async def test_sequence_aborts_when_context_authority_revoked_during_slow_precondition(lease_registry):
    """
    Verify that if context authority is revoked while a slow precondition is running
    (even if the worker lease in ServerTabLeaseRegistry remains active), perform_sequence
    revalidates context claims immediately prior to executing the action, aborts, and
    never invokes the underlying action.
    """
    session = MagicMock()
    session.tab_id = "tab_slow_pre"
    session.target_id = "target_slow_pre"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False

    async def fake_click(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_slow_pre",
            "ws://127.0.0.1:9222/devtools/page/target_slow_pre",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_slow_pre", "tab_slow_pre", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-slow-pre",
        worker_id="w_slow_pre",
        worker_token=lease.lease_token,
        target_tab="tab_slow_pre",
        lease_registry=lease_registry,
    )

    # Slow resolve_and_bind that revokes context authority mid-flight
    async def slow_resolve(selector):
        await asyncio.sleep(0.04)
        # Revoke the context authority while precondition is evaluating!
        service._context_authority.revoke_authority(ctx.context_token)
        return MagicMock(spec=TargetBinding)

    session.resolve_and_bind = AsyncMock(side_effect=slow_resolve)

    steps = [
        {
            "action": "click",
            "selector": "#btn-critical",
            "preconditions": [{"type": "element_exists", "selector": "#btn-critical"}],
        }
    ]

    res = await service.perform_sequence(ctx, steps=steps)

    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "authorization revoked prior to action execution" in res.error_message
    assert res.evidence["condition_stage"] == "pre_execution_authorization"
    assert action_invoked is False


@pytest.mark.asyncio
async def test_single_action_waiting_for_lock_aborts_when_lease_or_authority_revoked(lease_registry):
    """
    Verify that if a single public action (click, type, navigate, inspect, extract) is queued
    behind a held action lock and its lease is revoked while waiting, it revalidates authority
    immediately upon acquiring the lock, aborts with UNAUTHORIZED, and does not execute.
    """
    session = MagicMock()
    session.tab_id = "tab_lock_race"
    session.target_id = "target_lock_race"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False

    async def fake_click(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)
    session.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_lock_race",
            "ws://127.0.0.1:9222/devtools/page/target_lock_race",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_lock_race", "tab_lock_race", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-lock-race",
        worker_id="w_lock_race",
        worker_token=lease.lease_token,
        target_tab="tab_lock_race",
        lease_registry=lease_registry,
    )

    # 1. Acquire the tab lock externally to block incoming actions
    await session.action_lock.acquire()

    # 2. Launch public click in background task; it will pass _authorize_and_resolve_session and block on action_lock
    task_click = asyncio.create_task(service.click(ctx, selector="#race-btn"))
    await asyncio.sleep(0.02)

    # 3. Revoke the worker lease while the action is waiting for the lock
    lease_registry.revoke_lease("w_lock_race", trusted_caller=True)

    # 4. Release the lock to allow the action task to acquire it
    session.action_lock.release()

    res = await task_click

    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "lease revoked while waiting for lock" in res.error_message
    assert action_invoked is False


@pytest.mark.asyncio
async def test_value_matches_rejects_ordinary_text_and_unconfirmed_values(lease_registry):
    """
    Verify that value_matches requires an explicit confirmed value and strictly rejects
    substituting extract_read_only text (including empty string or ordinary text) for an unreadable value.
    """
    session = MagicMock()
    session.tab_id = "tab_val_strict"
    session.target_id = "target_val_strict"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    # 1. CDP evaluate returns an element without value property (e.g. div with text)
    async def mock_cdp_send(method, params=None):
        if method == "Runtime.evaluate":
            return {
                "result": {
                    "value": {"exists": True, "has_value": False, "value": None}
                }
            }
        return {}

    session.cdp = MagicMock()
    session.cdp.send = AsyncMock(side_effect=mock_cdp_send)

    # extract_read_only returns ordinary text matching what the condition expects
    session.extract_read_only = AsyncMock(return_value={"text": "Ordinary Page Text"})

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_val_strict",
            "ws://127.0.0.1:9222/devtools/page/target_val_strict",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_val_strict", "tab_val_strict", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-val-strict",
        worker_id="w_val_strict",
        worker_token=lease.lease_token,
        target_tab="tab_val_strict",
        lease_registry=lease_registry,
    )

    # Condition expects "Ordinary Page Text" — must FAIL because ordinary text is NOT an input value!
    r1 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#btn",
        "preconditions": [{"type": "value_matches", "selector": "#heading", "expected": "Ordinary Page Text"}],
    }])
    assert r1.success is False
    assert r1.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "does not have an input value property" in r1.error_message or "value could not be confirmed" in r1.error_message

    # Test with empty string text in extract_read_only
    session.extract_read_only = AsyncMock(return_value={"text": ""})
    r2 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#btn",
        "preconditions": [{"type": "value_matches", "selector": "#heading", "expected": ""}],
    }])
    assert r2.success is False
    assert r2.error_code == ToolErrorCode.ACTION_BLOCKED.value


@pytest.mark.asyncio
async def test_text_contains_fails_closed_on_none_and_malformed_extraction(lease_registry):
    """
    Verify that text_contains fails closed when extraction returns None or invalid objects,
    and never stringifies None to 'None' to satisfy expected: 'None'.
    """
    session = MagicMock()
    session.tab_id = "tab_text_strict"
    session.target_id = "target_text_strict"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    # extract_read_only returns None
    session.extract_read_only = AsyncMock(return_value=None)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_text_strict",
            "ws://127.0.0.1:9222/devtools/page/target_text_strict",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_text_strict", "tab_text_strict", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-text-strict",
        worker_id="w_text_strict",
        worker_token=lease.lease_token,
        target_tab="tab_text_strict",
        lease_registry=lease_registry,
    )

    # Condition expecting literal "None" must fail closed, NOT pass by stringifying None
    r1 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#btn",
        "preconditions": [{"type": "text_contains", "selector": "#el", "expected": "None"}],
    }])
    assert r1.success is False
    assert r1.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "text could not be extracted" in r1.error_message

    # Extraction raises an exception
    session.extract_read_only = AsyncMock(side_effect=RuntimeError("Extraction crashed"))
    r2 = await service.perform_sequence(ctx, steps=[{
        "action": "click",
        "selector": "#btn",
        "preconditions": [{"type": "text_contains", "selector": "#el", "expected": "hello"}],
    }])
    assert r2.success is False
    assert r2.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert "text extraction failed" in r2.error_message


@pytest.mark.asyncio
async def test_standalone_click_mutation_boundary_revocation_prevents_dispatch(lease_registry):
    """
    Deterministic regression test:
    Hold target resolution open via synchronization events.
    Revoke context authority while the worker lease remains valid in ServerTabLeaseRegistry.
    Release target resolution and verify the actuator gate is NEVER called, and the
    public result is UNAUTHORIZED.
    """
    session = MagicMock()
    session.tab_id = "tab_click_bound"
    session.target_id = "target_click_bound"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False

    async def fake_click(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)

    resolve_entered = asyncio.Event()
    allow_resolve = asyncio.Event()

    async def controlled_resolve(selector):
        resolve_entered.set()
        await allow_resolve.wait()
        return MagicMock(spec=TargetBinding)

    session.resolve_and_bind = AsyncMock(side_effect=controlled_resolve)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_click_bound",
            "ws://127.0.0.1:9222/devtools/page/target_click_bound",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_click_bound", "tab_click_bound", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-click-bound",
        worker_id="w_click_bound",
        worker_token=lease.lease_token,
        target_tab="tab_click_bound",
        lease_registry=lease_registry,
    )

    click_task = asyncio.create_task(service.click(ctx, selector="#crit-btn"))
    await resolve_entered.wait()

    assert lease_registry.get_active_lease("w_click_bound") is not None
    service._context_authority.revoke_authority(ctx.context_token)

    allow_resolve.set()
    res = await click_task

    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "prior to mutation dispatch" in res.error_message
    assert res.evidence.get("action_dispatched") is False
    assert action_invoked is False


@pytest.mark.asyncio
async def test_standalone_type_mutation_boundary_revocation_prevents_dispatch(lease_registry):
    """
    Deterministic regression test:
    Hold target resolution open for typing via synchronization events.
    Revoke context authority while the worker lease remains valid.
    Release target resolution and verify the actuator gate is NEVER called, and the
    public result is UNAUTHORIZED.
    """
    session = MagicMock()
    session.tab_id = "tab_type_bound"
    session.target_id = "target_type_bound"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False

    async def fake_type(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_type)

    resolve_entered = asyncio.Event()
    allow_resolve = asyncio.Event()

    async def controlled_resolve(selector):
        resolve_entered.set()
        await allow_resolve.wait()
        return MagicMock(spec=TargetBinding)

    session.resolve_and_bind = AsyncMock(side_effect=controlled_resolve)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_type_bound",
            "ws://127.0.0.1:9222/devtools/page/target_type_bound",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_type_bound", "tab_type_bound", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-type-bound",
        worker_id="w_type_bound",
        worker_token=lease.lease_token,
        target_tab="tab_type_bound",
        lease_registry=lease_registry,
    )

    type_task = asyncio.create_task(service.type(ctx, selector="#user-input", text="sensitive_payload"))
    await resolve_entered.wait()

    assert lease_registry.get_active_lease("w_type_bound") is not None
    service._context_authority.revoke_authority(ctx.context_token)

    allow_resolve.set()
    res = await type_task

    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "prior to mutation dispatch" in res.error_message
    assert res.evidence.get("action_dispatched") is False
    assert action_invoked is False


@pytest.mark.asyncio
async def test_sequence_mutation_boundary_revocation_prevents_dispatch_and_reports_partial_state(lease_registry):
    """
    Deterministic regression test:
    Execute a sequence where step 0 (navigate) succeeds.
    For step 1 (click), hold target resolution open via synchronization events.
    Revoke context authority while worker lease remains active.
    Release resolution and verify:
    - Step 1 mutation is NOT dispatched to actuator gate.
    - Sequence reports UNAUTHORIZED, failed_step_index=1, failed_action='click'.
    - Step 0 history is preserved in completed_steps.
    - side_effect_possible is True (from earlier step 0) and action_dispatched is False.
    """
    session = MagicMock()
    session.tab_id = "tab_seq_bound"
    session.target_id = "target_seq_bound"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False

    async def fake_click(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)
    session.navigate = AsyncMock(return_value={"navigated": True, "url": "https://example.com/checkout"})

    resolve_entered = asyncio.Event()
    allow_resolve = asyncio.Event()

    call_count = 0
    async def controlled_resolve(selector):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            # First call was the precondition check (which passes);
            # second call is inside _execute_click_unlocked during action execution.
            resolve_entered.set()
            await allow_resolve.wait()
        return MagicMock(spec=TargetBinding)

    session.resolve_and_bind = AsyncMock(side_effect=controlled_resolve)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_seq_bound",
            "ws://127.0.0.1:9222/devtools/page/target_seq_bound",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_seq_bound", "tab_seq_bound", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-seq-bound",
        worker_id="w_seq_bound",
        worker_token=lease.lease_token,
        target_tab="tab_seq_bound",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "navigate", "url": "https://example.com/checkout"},
        {
            "action": "click",
            "selector": "#btn-confirm-pay",
            "preconditions": [{"type": "element_exists", "selector": "#btn-confirm-pay"}],
        },
    ]

    seq_task = asyncio.create_task(service.perform_sequence(ctx, steps=steps))
    await resolve_entered.wait()

    assert lease_registry.get_active_lease("w_seq_bound") is not None
    service._context_authority.revoke_authority(ctx.context_token)

    allow_resolve.set()
    res = await seq_task

    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "prior to mutation dispatch" in res.error_message
    assert res.evidence["failed_step_index"] == 1
    assert res.evidence["failed_action"] == "click"
    assert res.evidence["condition_stage"] == "mutation_boundary"
    assert res.evidence["action_dispatched"] is False
    assert len(res.evidence["completed_steps"]) == 1
    assert res.evidence["completed_steps"][0]["action"] == "navigate"
    assert res.evidence["side_effect_possible"] is True
    assert action_invoked is False


@pytest.mark.asyncio
async def test_authority_only_revocation_during_lock_wait_aborts_action(lease_registry):
    """
    Deterministic regression test:
    Verify that revoking context authority ALONE while the worker lease remains
    completely active in ServerTabLeaseRegistry prevents a queued action from executing.
    Uses synchronization to guarantee the operation is blocked on the lock before revocation.
    """
    session = MagicMock()
    session.tab_id = "tab_auth_only_lock"
    session.target_id = "target_auth_only_lock"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False

    async def fake_click(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)
    session.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_auth_only_lock",
            "ws://127.0.0.1:9222/devtools/page/target_auth_only_lock",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_auth_lock", "tab_auth_only_lock", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-auth-lock",
        worker_id="w_auth_lock",
        worker_token=lease.lease_token,
        target_tab="tab_auth_only_lock",
        lease_registry=lease_registry,
    )

    # 1. Acquire lock externally to force queuing
    await session.action_lock.acquire()

    # 2. Launch public click in background task
    click_task = asyncio.create_task(service.click(ctx, selector="#lock-btn"))

    # Wait deterministically until click_task is waiting on action_lock
    while not getattr(session.action_lock, "_waiters", None):
        await asyncio.sleep(0.001)

    # 3. Revoke ONLY context authority; worker lease in registry remains ACTIVE!
    service._context_authority.revoke_authority(ctx.context_token)
    assert lease_registry.get_active_lease("w_auth_lock") is not None

    # 4. Release lock to allow task to acquire it
    session.action_lock.release()

    res = await click_task

    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert "authorization revoked while waiting for lock" in res.error_message
    assert action_invoked is False
    assert lease_registry.get_active_lease("w_auth_lock") is not None


@pytest.mark.asyncio
async def test_result_contract_all_operations_propagate_unauthorized_on_lock_wait_revocation(lease_registry):
    """
    Verify result-contract consistency:
    For all 5 operations (inspect, extract, navigate, click, type), when authority is revoked
    while waiting for the lock, the returned result is strictly success=False with
    error_code=UNAUTHORIZED, and the underlying operation is never dispatched.
    """
    ops = [
        ("inspect", lambda svc, ctx: svc.inspect(ctx)),
        ("extract", lambda svc, ctx: svc.extract(ctx, selector="#txt")),
        ("navigate", lambda svc, ctx: svc.navigate(ctx, url="https://example.com/page")),
        ("click", lambda svc, ctx: svc.click(ctx, selector="#btn")),
        ("type", lambda svc, ctx: svc.type(ctx, selector="#inp", text="val")),
    ]

    for op_name, op_callable in ops:
        session = MagicMock()
        session.tab_id = f"tab_contract_{op_name}"
        session.target_id = f"target_contract_{op_name}"
        session.action_lock = asyncio.Lock()
        session.tracker.state.is_target_subtree_quiet = True

        underlying_called = False

        async def record_call(*args, **kwargs):
            nonlocal underlying_called
            underlying_called = True
            return {}

        session.inspect_elements = AsyncMock(side_effect=record_call)
        session.extract_read_only = AsyncMock(side_effect=record_call)
        session.navigate = AsyncMock(side_effect=record_call)
        session.gate.async_attempt = AsyncMock(side_effect=record_call)
        session.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))

        service = ComexToolService(
            lease_registry=lease_registry,
            session_resolver=AsyncMock(return_value=session),
            target_resolver=lambda lease, tab, trusted: (
                f"target_contract_{op_name}",
                f"ws://127.0.0.1:9222/devtools/page/target_contract_{op_name}",
                None,
            ),
        )

        worker_id = f"w_contract_{op_name}"
        lease = lease_registry.acquire_lease(worker_id, f"tab_contract_{op_name}", trusted_issuer=True)
        ctx = create_tool_execution_context(
            task_id=f"task-contract-{op_name}",
            worker_id=worker_id,
            worker_token=lease.lease_token,
            target_tab=f"tab_contract_{op_name}",
            lease_registry=lease_registry,
        )

        await session.action_lock.acquire()
        task = asyncio.create_task(op_callable(service, ctx))

        while not getattr(session.action_lock, "_waiters", None):
            await asyncio.sleep(0.001)

        service._context_authority.revoke_authority(ctx.context_token)
        session.action_lock.release()

        res = await task
        assert res.success is False, f"Operation {op_name} must fail on revoked authority"
        assert res.error_code == ToolErrorCode.UNAUTHORIZED.value, f"Operation {op_name} must return UNAUTHORIZED"
        assert underlying_called is False, f"Operation {op_name} must never dispatch underlying action"


@pytest.mark.asyncio
async def test_canonical_target_lock_invariant_across_session_instantiations():
    """
    Verify the canonical per-tab lock invariant:
    Two independently created sessions targeting the same canonical Chromium target
    (whether via direct TabSessionContext instantiation or get_or_create_tab_session)
    CANNOT receive separate locks. They must share the identical asyncio.Lock instance.
    """
    from nervous.cdp_session import (
        TabSessionContext,
        get_canonical_tab_lock,
        extract_canonical_target_id,
        reset_tab_locks,
    )

    reset_tab_locks()

    cdp_1 = MagicMock()
    cdp_1.ws_url = "ws://127.0.0.1:9222/devtools/page/CANONICAL_TARGET_XYZ"
    cdp_2 = MagicMock()
    cdp_2.ws_url = "ws://127.0.0.1:9222/devtools/page/CANONICAL_TARGET_XYZ"

    # 1. Independent direct instantiations without explicit action_lock
    session_a = TabSessionContext(tab_id="tab_A", target_id="CANONICAL_TARGET_XYZ", cdp_client=cdp_1)
    session_b = TabSessionContext(tab_id="tab_B", target_id="CANONICAL_TARGET_XYZ", cdp_client=cdp_2)

    assert session_a.action_lock is session_b.action_lock
    assert session_a.action_lock is get_canonical_tab_lock("CANONICAL_TARGET_XYZ")

    # 2. Verify independent session targeting a DIFFERENT canonical target receives a distinct lock
    cdp_diff = MagicMock()
    cdp_diff.ws_url = "ws://127.0.0.1:9222/devtools/page/CANONICAL_TARGET_OTHER"
    session_diff = TabSessionContext(tab_id="tab_diff", target_id="CANONICAL_TARGET_OTHER", cdp_client=cdp_diff)

    assert session_diff.action_lock is not session_a.action_lock

    # 3. Clean up
    reset_tab_locks()


# =============================================================================
# Phase 2.1B: Final Side-Effect Evidence Regression Tests
# =============================================================================

@pytest.mark.asyncio
async def test_mutation_boundary_auth_rejection_reports_action_not_dispatched_and_no_side_effects(lease_registry):
    """
    Regression Test 1:
    A mutation-boundary authorization rejection returns action_dispatched=False,
    and when no prior mutating steps ran, reports side_effect_possible=False.
    """
    session = MagicMock()
    session.tab_id = "tab_auth_reject_nomut"
    session.target_id = "target_auth_reject_nomut"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    action_invoked = False
    async def fake_click(*args, **kwargs):
        nonlocal action_invoked
        action_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)

    resolve_entered = asyncio.Event()
    allow_resolve = asyncio.Event()

    async def controlled_resolve(selector):
        resolve_entered.set()
        await allow_resolve.wait()
        return MagicMock(spec=TargetBinding)

    session.resolve_and_bind = AsyncMock(side_effect=controlled_resolve)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_auth_reject_nomut",
            "ws://127.0.0.1:9222/devtools/page/target_auth_reject_nomut",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_auth_reject", "tab_auth_reject_nomut", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-auth-reject-nomut",
        worker_id="w_auth_reject",
        worker_token=lease.lease_token,
        target_tab="tab_auth_reject_nomut",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "click", "selector": "#btn-no-prior"},
    ]

    seq_task = asyncio.create_task(service.perform_sequence(ctx, steps=steps))
    await resolve_entered.wait()

    # Revoke authority at the mutation boundary
    service._context_authority.revoke_authority(ctx.context_token)
    allow_resolve.set()

    res = await seq_task
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert res.evidence["failed_step_index"] == 0
    assert res.evidence["failed_action"] == "click"
    assert res.evidence["condition_stage"] == "mutation_boundary"
    assert res.evidence["action_dispatched"] is False
    assert res.evidence["side_effect_possible"] is False
    assert len(res.evidence["completed_steps"]) == 0
    assert action_invoked is False


@pytest.mark.asyncio
async def test_click_raising_after_entering_dispatch_path_reports_side_effect_possible(lease_registry):
    """
    Regression Test 2:
    A click or type operation that raises after entering its dispatch path
    reports possible side effects (side_effect_possible=True, action_dispatched=True).
    The failed step is NOT marked as completed.
    """
    session = MagicMock()
    session.tab_id = "tab_click_raise"
    session.target_id = "target_click_raise"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    dispatch_started = asyncio.Event()

    async def exploding_click(*args, **kwargs):
        dispatch_started.set()
        raise RuntimeError("CDP connection lost after physical click dispatch attempt")

    session.gate.async_attempt = AsyncMock(side_effect=exploding_click)
    session.resolve_and_bind = AsyncMock(return_value=MagicMock(spec=TargetBinding))

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_click_raise",
            "ws://127.0.0.1:9222/devtools/page/target_click_raise",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_click_raise", "tab_click_raise", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-click-raise",
        worker_id="w_click_raise",
        worker_token=lease.lease_token,
        target_tab="tab_click_raise",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "click", "selector": "#exploding-btn"},
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert dispatch_started.is_set() is True
    assert res.evidence["failed_step_index"] == 0
    assert res.evidence["failed_action"] == "click"
    assert res.evidence["condition_stage"] == "execution"
    assert res.evidence["action_dispatched"] is True
    assert res.evidence["side_effect_possible"] is True
    assert len(res.evidence["completed_steps"]) == 0

    # Also verify type operation behaves identically when raising after entering dispatch path
    type_dispatch_started = asyncio.Event()

    async def exploding_type(*args, **kwargs):
        type_dispatch_started.set()
        raise RuntimeError("CDP Input.dispatchKeyEvent timed out after keystroke dispatch")

    session.gate.async_attempt = AsyncMock(side_effect=exploding_type)
    type_steps = [
        {"action": "type", "selector": "#exploding-input", "text": "partial_keystrokes"},
    ]
    res_type = await service.perform_sequence(ctx, steps=type_steps)
    assert res_type.success is False
    assert type_dispatch_started.is_set() is True
    assert res_type.evidence["failed_step_index"] == 0
    assert res_type.evidence["failed_action"] == "type"
    assert res_type.evidence["condition_stage"] == "execution"
    assert res_type.evidence["action_dispatched"] is True
    assert res_type.evidence["side_effect_possible"] is True
    assert len(res_type.evidence["completed_steps"]) == 0


@pytest.mark.asyncio
async def test_navigation_error_after_navigation_dispatch_does_not_claim_no_side_effects(lease_registry):
    """
    Regression Test 3:
    A navigation error after navigation dispatch does not incorrectly claim
    no side effects (side_effect_possible=True, action_dispatched=True).
    """
    session = MagicMock()
    session.tab_id = "tab_nav_err"
    session.target_id = "target_nav_err"
    session.action_lock = asyncio.Lock()

    nav_dispatched = asyncio.Event()

    async def exploding_navigate(url):
        nav_dispatched.set()
        raise RuntimeError("Navigation error: net::ERR_CONNECTION_RESET after Page.navigate dispatched")

    session.navigate = AsyncMock(side_effect=exploding_navigate)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_nav_err",
            "ws://127.0.0.1:9222/devtools/page/target_nav_err",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_nav_err", "tab_nav_err", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-nav-err",
        worker_id="w_nav_err",
        worker_token=lease.lease_token,
        target_tab="tab_nav_err",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "navigate", "url": "https://example.com/payment"},
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert nav_dispatched.is_set() is True
    assert res.evidence["failed_step_index"] == 0
    assert res.evidence["failed_action"] == "navigate"
    assert res.evidence["condition_stage"] == "execution"
    assert res.evidence["action_dispatched"] is True
    assert res.evidence["side_effect_possible"] is True
    assert len(res.evidence["completed_steps"]) == 0


@pytest.mark.asyncio
async def test_pre_dispatch_rejection_without_preceding_mutations_reports_no_side_effects(lease_registry):
    """
    Regression Test 4:
    A pre-dispatch rejection (e.g. failing precondition) without preceding mutations
    reports side_effect_possible=False and action_dispatched=False,
    even when preceded by successful purely read-only steps (inspect/extract).
    """
    session = MagicMock()
    session.tab_id = "tab_pre_reject"
    session.target_id = "target_pre_reject"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    # Step 0: inspect elements succeeds
    session.inspect_elements = AsyncMock(return_value={"elements": [{"selector": "#item1"}], "returned": 1})
    # Step 1: precondition fails on resolve_and_bind returning None
    session.resolve_and_bind = AsyncMock(return_value=None)

    gate_called = False
    async def fake_click(*args, **kwargs):
        nonlocal gate_called
        gate_called = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_click)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_pre_reject",
            "ws://127.0.0.1:9222/devtools/page/target_pre_reject",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_pre_reject", "tab_pre_reject", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-pre-reject",
        worker_id="w_pre_reject",
        worker_token=lease.lease_token,
        target_tab="tab_pre_reject",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "inspect", "max_elements": 10},
        {
            "action": "click",
            "selector": "#submit-btn",
            "preconditions": [{"type": "element_exists", "selector": "#submit-btn"}],
        },
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.ACTION_BLOCKED.value
    assert res.evidence["failed_step_index"] == 1
    assert res.evidence["failed_action"] == "click"
    assert res.evidence["condition_stage"] == "precondition"
    assert res.evidence["action_dispatched"] is False
    # Purely read-only inspect step in completed_steps must NOT trigger side_effect_possible
    assert len(res.evidence["completed_steps"]) == 1
    assert res.evidence["completed_steps"][0]["action"] == "inspect"
    assert res.evidence["side_effect_possible"] is False
    assert gate_called is False


@pytest.mark.asyncio
async def test_sequence_with_prior_mutation_and_later_auth_rejection_preserves_history_and_reports_side_effects(lease_registry):
    """
    Regression Test 5:
    A sequence containing an earlier successful mutation and a later authorization
    rejection preserves completed-step history, reports action_dispatched=False for the
    rejected step, and reports that earlier side effects may exist (side_effect_possible=True).
    """
    session = MagicMock()
    session.tab_id = "tab_prior_mut"
    session.target_id = "target_prior_mut"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    session.navigate = AsyncMock(return_value={"navigated": True, "url": "https://example.com/form"})

    type_invoked = False
    async def fake_type(*args, **kwargs):
        nonlocal type_invoked
        type_invoked = True
        return GateResult(decision=None, executed=True)

    session.gate.async_attempt = AsyncMock(side_effect=fake_type)

    resolve_entered = asyncio.Event()
    allow_resolve = asyncio.Event()

    async def controlled_resolve(selector):
        resolve_entered.set()
        await allow_resolve.wait()
        return MagicMock(spec=TargetBinding)

    session.resolve_and_bind = AsyncMock(side_effect=controlled_resolve)

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_prior_mut",
            "ws://127.0.0.1:9222/devtools/page/target_prior_mut",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_prior_mut", "tab_prior_mut", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-prior-mut",
        worker_id="w_prior_mut",
        worker_token=lease.lease_token,
        target_tab="tab_prior_mut",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "navigate", "url": "https://example.com/form"},
        {"action": "type", "selector": "#name-field", "text": "Alice"},
    ]

    seq_task = asyncio.create_task(service.perform_sequence(ctx, steps=steps))
    await resolve_entered.wait()

    # Revoke authority during type preparation (mutation boundary)
    service._context_authority.revoke_authority(ctx.context_token)
    allow_resolve.set()

    res = await seq_task
    assert res.success is False
    assert res.error_code == ToolErrorCode.UNAUTHORIZED.value
    assert res.evidence["failed_step_index"] == 1
    assert res.evidence["failed_action"] == "type"
    assert res.evidence["condition_stage"] == "mutation_boundary"
    assert res.evidence["action_dispatched"] is False
    # Step 0 (navigate) is preserved in completed_steps
    assert len(res.evidence["completed_steps"]) == 1
    assert res.evidence["completed_steps"][0]["action"] == "navigate"
    # Earlier mutating step 0 means side effects may exist
    assert res.evidence["side_effect_possible"] is True
    assert type_invoked is False


@pytest.mark.asyncio
async def test_inspect_failing_postcondition_reports_dispatched_but_no_side_effects(lease_registry):
    """
    Regression Test:
    Read-only inspect step completes successfully, but its postcondition fails.
    Since inspect is purely read-only and no prior mutating steps ran:
    - action_dispatched is True (inspect was executed)
    - side_effect_possible is False (read-only inspect produces no browser side effects)
    - condition_stage is 'postcondition'
    - completed_steps contains the executed inspect step
    """
    session = MagicMock()
    session.tab_id = "tab_inspect_post_fail"
    session.target_id = "target_inspect_post_fail"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    session.inspect_elements = AsyncMock(return_value={"elements": [{"selector": "#item1"}], "returned": 1})
    session.extract_read_only = AsyncMock(return_value={"url": "https://example.com/page", "title": "Actual Page Title"})

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_inspect_post_fail",
            "ws://127.0.0.1:9222/devtools/page/target_inspect_post_fail",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_insp_post", "tab_inspect_post_fail", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-insp-post",
        worker_id="w_insp_post",
        worker_token=lease.lease_token,
        target_tab="tab_inspect_post_fail",
        lease_registry=lease_registry,
    )

    steps = [
        {
            "action": "inspect",
            "max_elements": 10,
            "postconditions": [{"type": "title_contains", "expected": "Nonexistent Title"}],
        },
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "postcondition failed" in res.error_message
    assert res.evidence["failed_step_index"] == 0
    assert res.evidence["failed_action"] == "inspect"
    assert res.evidence["condition_stage"] == "postcondition"
    assert res.evidence["action_dispatched"] is True
    assert res.evidence["side_effect_possible"] is False
    assert len(res.evidence["completed_steps"]) == 1
    assert res.evidence["completed_steps"][0]["action"] == "inspect"


@pytest.mark.asyncio
async def test_extract_failing_postcondition_reports_dispatched_but_no_side_effects(lease_registry):
    """
    Regression Test:
    Read-only extract step completes successfully, but its postcondition fails.
    Since extract is purely read-only and no prior mutating steps ran:
    - action_dispatched is True (extract was executed)
    - side_effect_possible is False (read-only extract produces no browser side effects)
    - condition_stage is 'postcondition'
    - completed_steps contains the executed extract step
    """
    session = MagicMock()
    session.tab_id = "tab_extract_post_fail"
    session.target_id = "target_extract_post_fail"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    session.extract_read_only = AsyncMock(return_value={"text": "Hello world", "title": "Actual Title", "url": "https://example.com"})

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_extract_post_fail",
            "ws://127.0.0.1:9222/devtools/page/target_extract_post_fail",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_ext_post", "tab_extract_post_fail", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-ext-post",
        worker_id="w_ext_post",
        worker_token=lease.lease_token,
        target_tab="tab_extract_post_fail",
        lease_registry=lease_registry,
    )

    steps = [
        {
            "action": "extract",
            "selector": "#content",
            "postconditions": [{"type": "url_contains", "expected": "https://other.com"}],
        },
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "postcondition failed" in res.error_message
    assert res.evidence["failed_step_index"] == 0
    assert res.evidence["failed_action"] == "extract"
    assert res.evidence["condition_stage"] == "postcondition"
    assert res.evidence["action_dispatched"] is True
    assert res.evidence["side_effect_possible"] is False
    assert len(res.evidence["completed_steps"]) == 1
    assert res.evidence["completed_steps"][0]["action"] == "extract"


@pytest.mark.asyncio
async def test_completed_mutation_followed_by_readonly_failing_postcondition_reports_side_effects(lease_registry):
    """
    Regression Test:
    A sequence containing an earlier successful mutation (navigate) followed by a
    read-only step (inspect or extract) whose postcondition fails.
    Since the earlier mutating step completed, side_effect_possible must be True.
    """
    session = MagicMock()
    session.tab_id = "tab_mut_then_post_fail"
    session.target_id = "target_mut_then_post_fail"
    session.action_lock = asyncio.Lock()
    session.tracker.state.is_target_subtree_quiet = True

    session.navigate = AsyncMock(return_value={"navigated": True, "url": "https://example.com/checkout"})
    session.inspect_elements = AsyncMock(return_value={"elements": [{"selector": "#item1"}], "returned": 1})
    session.extract_read_only = AsyncMock(return_value={"url": "https://example.com/checkout", "title": "Checkout Page"})

    service = ComexToolService(
        lease_registry=lease_registry,
        session_resolver=AsyncMock(return_value=session),
        target_resolver=lambda lease, tab, trusted: (
            "target_mut_then_post_fail",
            "ws://127.0.0.1:9222/devtools/page/target_mut_then_post_fail",
            None,
        ),
    )

    lease = lease_registry.acquire_lease("w_mut_post", "tab_mut_then_post_fail", trusted_issuer=True)
    ctx = create_tool_execution_context(
        task_id="task-mut-post",
        worker_id="w_mut_post",
        worker_token=lease.lease_token,
        target_tab="tab_mut_then_post_fail",
        lease_registry=lease_registry,
    )

    steps = [
        {"action": "navigate", "url": "https://example.com/checkout"},
        {
            "action": "inspect",
            "max_elements": 10,
            "postconditions": [{"type": "title_contains", "expected": "Confirmation Page"}],
        },
    ]

    res = await service.perform_sequence(ctx, steps=steps)
    assert res.success is False
    assert res.error_code == ToolErrorCode.OPERATION_FAILED.value
    assert "postcondition failed" in res.error_message
    assert res.evidence["failed_step_index"] == 1
    assert res.evidence["failed_action"] == "inspect"
    assert res.evidence["condition_stage"] == "postcondition"
    assert res.evidence["action_dispatched"] is True
    # Step 0 (navigate) was mutating, so side effects may exist
    assert res.evidence["side_effect_possible"] is True
    assert len(res.evidence["completed_steps"]) == 2
    assert res.evidence["completed_steps"][0]["action"] == "navigate"
    assert res.evidence["completed_steps"][1]["action"] == "inspect"







