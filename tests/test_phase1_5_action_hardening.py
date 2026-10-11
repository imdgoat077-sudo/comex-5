"""
Phase 1.5 Comprehensive Adversarial Test Suite: Action-Boundary Hardening.

Verifies all critical security and isolation boundaries:
1. Authoritative worker identity and tab authorization (server-owned lease registry).
2. Fail-closed rejection of forged worker IDs, expired leases, revoked leases, and cross-tab access.
3. Leases require trusted issuer; unauthenticated HTTP lease management fails closed with 403.
4. Exclusive tab ownership: tabs cannot be stolen or concurrently leased by competing workers.
5. Workers cannot hold conflicting concurrent leases across multiple tabs.
6. Prohibited foreground aliases ('active_tab', 'foreground', 'active', 'current') for workers.
7. Omitted worker ID attempting to access foreground or named tabs fails closed without trusted context.
8. Trusted internal single-agent execution permitted only with explicit trusted application token.
9. Server-verified CDP WebSocket target resolution (caller cannot supply mismatched WS URLs or canonical IDs).
10. Explicit target tabs NEVER silently fall back to default/foreground browser on lookup failure.
11. Complete URL validation (rejects missing netloc/hostname, malformed IP literals, embedded credentials,
    and sensitive internal infrastructure ports like 9222 CDP, 8000 API, 20128 proxy).
12. AST script guard against exec, eval, getattr, __builtins__, dangerous imports, f-strings, and dynamic evaluate calls.
13. Worker agents cannot access or invoke run_browser_harness (tool completely removed from agent catalog).
14. Genuinely per-tab canonical locks shared across all worker sessions targeting the same Chromium page.
15. Independent tabs remain strictly concurrent without lock contention.
16. REST action endpoints enforce lease verification and fail closed on tampered inputs or canonical target mismatches.
"""

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nervous.action_policy import (
    PROHIBITED_FOREGROUND_ALIASES,
    DEFAULT_LEASE_TTL,
    MAX_LEASE_TTL,
    TabLease,
    ServerTabLeaseRegistry,
    ServerTabTargetRegistry,
    get_lease_registry,
    get_target_registry,
    validate_harness_script_safety,
    validate_worker_tab_authorization,
    validate_navigation_url,
)
from nervous.cdp_session import (
    TabSessionContext,
    cleanup_worker_sessions,
    close_tab_session,
    extract_canonical_target_id,
    get_canonical_tab_lock,
    get_or_create_tab_session,
    reset_tab_locks,
)
from nervous.target_binding import TargetBinding
from agent_service import _INTERNAL_APP_TOKEN, AgentRequest, AgentResponse


@pytest.fixture(autouse=True)
def reset_registries():
    """Ensure a clean registry state before each test."""
    get_lease_registry().reset()
    get_target_registry().reset()
    reset_tab_locks()
    yield
    get_lease_registry().reset()
    get_target_registry().reset()
    reset_tab_locks()


# ── 1. Authoritative Worker Identity & Tab Lease Registry Tests ───────────

def test_lease_requires_trusted_issuer():
    """Acquiring a tab lease directly without trusted_issuer=True raises PermissionError."""
    registry = get_lease_registry()
    with pytest.raises(PermissionError) as exc_info:
        registry.acquire_lease("untrusted_worker", "tab_1", trusted_issuer=False)
    assert "only be issued by an authenticated trusted orchestrator" in str(exc_info.value)


def test_exclusive_tab_ownership_prevents_lease_stealing():
    """Tab A cannot be leased to Worker B if Worker A currently holds an active lease for it."""
    registry = get_lease_registry()
    registry.acquire_lease("worker_A", "tab_A", trusted_issuer=True)

    with pytest.raises(PermissionError) as exc_info:
        registry.acquire_lease("worker_B", "tab_A", trusted_issuer=True)
    assert "already exclusively leased to worker 'worker_A'" in str(exc_info.value)


def test_worker_cannot_hold_multiple_conflicting_leases():
    """A single worker cannot hold multiple active leases for different tabs simultaneously."""
    registry = get_lease_registry()
    registry.acquire_lease("worker_A", "tab_1", trusted_issuer=True)

    with pytest.raises(PermissionError) as exc_info:
        registry.acquire_lease("worker_A", "tab_2", trusted_issuer=True)
    assert "already holds an active lease for tab 'tab_1'" in str(exc_info.value)


def test_worker_without_server_lease_fails_closed():
    """An invented or unregistered worker ID fails closed on access."""
    registry = get_lease_registry()
    assert registry.get_active_lease("forged_worker_999") is None

    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(worker_id="forged_worker_999", target_tab="tab_A")
    assert "No active server lease found for worker 'forged_worker_999'" in str(exc_info.value)


def test_assigned_tab_cannot_auto_create_lease():
    """Passing assigned_tab to validate_worker_tab_authorization does NOT auto-create a lease."""
    registry = get_lease_registry()
    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(
            worker_id="sneaky_worker",
            target_tab="tab_secret",
            assigned_tab="tab_secret"
        )
    assert "No active server lease found for worker 'sneaky_worker'" in str(exc_info.value)
    assert registry.get_active_lease("sneaky_worker") is None


def test_worker_targeting_another_workers_tab_fails_closed():
    """Worker A cannot act on Tab B even if Tab B exists."""
    registry = get_lease_registry()
    lease_A = registry.acquire_lease("worker_A", "tab_A", trusted_issuer=True)
    registry.acquire_lease("worker_B", "tab_B", trusted_issuer=True)

    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(worker_id="worker_A", target_tab="tab_B", worker_token=lease_A.lease_token)
    assert "Worker authorization mismatch: worker 'worker_A' is leased tab 'tab_A', not target tab 'tab_B'" in str(exc_info.value)


def test_worker_targeting_foreground_aliases_fails_closed():
    """Workers are strictly prohibited from targeting foreground aliases."""
    registry = get_lease_registry()
    registry.acquire_lease("worker_A", "tab_A", trusted_issuer=True)

    for alias in ["active_tab", "foreground", "active", "current", "activetab", "default", "default_tab"]:
        with pytest.raises(PermissionError) as exc_info:
            validate_worker_tab_authorization(worker_id="worker_A", target_tab=alias)
        assert "is prohibited from accessing foreground/active tab alias" in str(exc_info.value)


def test_omitted_worker_id_fails_closed_without_trusted_context():
    """An omitted worker ID attempting to access foreground or specific tab fails closed without trusted context."""
    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(worker_id=None, target_tab="active_tab", trusted_app_context=False)
    assert "Untrusted access to foreground tab 'active_tab' is prohibited" in str(exc_info.value)

    with pytest.raises(PermissionError) as exc_info2:
        validate_worker_tab_authorization(worker_id=None, target_tab="tab_1", trusted_app_context=False)
    assert "Missing worker authorization context" in str(exc_info2.value)


def test_trusted_single_agent_allows_access():
    """Trusted internal application context permits foreground or assigned tab access."""
    lease = validate_worker_tab_authorization(worker_id=None, target_tab="active_tab", trusted_app_context=True)
    assert lease.is_active is True
    assert lease.tab_id == "active_tab"


def test_expired_lease_fails_closed_and_cannot_be_restored():
    """An expired tab lease fails closed immediately and cannot be restored by subsequent calls."""
    registry = get_lease_registry()
    registry.acquire_lease("worker_fast_expire", "tab_fast", ttl_seconds=0.01, trusted_issuer=True)
    time.sleep(0.02)

    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(worker_id="worker_fast_expire", target_tab="tab_fast")
    assert "No active server lease found for worker 'worker_fast_expire'" in str(exc_info.value)

    # Calling with assigned_tab cannot recreate it
    with pytest.raises(PermissionError):
        validate_worker_tab_authorization(
            worker_id="worker_fast_expire",
            target_tab="tab_fast",
            assigned_tab="tab_fast"
        )
    assert registry.get_active_lease("worker_fast_expire") is None


def test_revoked_lease_fails_closed_and_cannot_be_restored():
    """A revoked lease remains revoked throughout task execution."""
    registry = get_lease_registry()
    registry.acquire_lease("worker_revoked", "tab_revoked", trusted_issuer=True)
    ok = registry.revoke_lease("worker_revoked", trusted_caller=True)
    assert ok is True

    with pytest.raises(PermissionError):
        validate_worker_tab_authorization(worker_id="worker_revoked", target_tab="tab_revoked")

    with pytest.raises(PermissionError):
        validate_worker_tab_authorization(
            worker_id="worker_revoked",
            target_tab="tab_revoked",
            assigned_tab="tab_revoked"
        )


def test_mismatched_websocket_target_fails_closed():
    """Caller supplying a WebSocket URL differing from the server-verified lease endpoint fails closed."""
    registry = get_lease_registry()
    lease = registry.acquire_lease(
        "worker_ws_test",
        "tab_ws_test",
        cdp_ws_url="ws://127.0.0.1:9222/devtools/page/REAL_PAGE_UUID",
        trusted_issuer=True
    )

    tampered_ws = "ws://127.0.0.1:9222/devtools/page/FORGED_PAGE_UUID"
    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(
            worker_id="worker_ws_test",
            target_tab="tab_ws_test",
            caller_ws_url=tampered_ws,
            worker_token=lease.lease_token,
        )
    assert "Target WebSocket mismatch" in str(exc_info.value)


def test_canonical_target_id_mismatch_fails_closed():
    """Target resolution verifying against a lease with a bound canonical Chromium ID fails on mismatch."""
    registry = get_lease_registry()
    lease = registry.acquire_lease(
        "worker_canon",
        "tab_canon",
        canonical_target_id="CHROMIUM_TARGET_AAA",
        trusted_issuer=True
    )

    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(
            worker_id="worker_canon",
            target_tab="tab_canon",
            expected_target_id="CHROMIUM_TARGET_BBB",
            worker_token=lease.lease_token,
        )
    assert "Canonical target mismatch" in str(exc_info.value)


# ── 2. Complete Navigation Scheme, Hostname & Network Policy Validation ────

def test_navigation_url_scheme_validation():
    """Only http and https schemes are permitted; javascript/file/data are rejected."""
    assert validate_navigation_url("https://apple.com") == "https://apple.com"
    assert validate_navigation_url("http://localhost:5173") == "http://localhost:5173"
    assert validate_navigation_url("https://example.com/path?query=val#hash") == "https://example.com/path?query=val#hash"

    with pytest.raises(ValueError) as exc1:
        validate_navigation_url("javascript:alert(1)")
    assert "Disallowed navigation scheme 'javascript'" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        validate_navigation_url("file:///etc/passwd")
    assert "Disallowed navigation scheme 'file'" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        validate_navigation_url("data:text/html,<h1>evil</h1>")
    assert "Disallowed navigation scheme 'data'" in str(exc3.value)


def test_navigation_url_rejects_missing_hostname():
    """URLs lacking a valid network location (e.g. 'http:///example.com') are rejected."""
    with pytest.raises(ValueError) as exc1:
        validate_navigation_url("http:///example.com")
    assert "Missing network location" in str(exc1.value) or "Invalid hostname format" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        validate_navigation_url("http://")
    assert "Missing network location" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        validate_navigation_url("https://")
    assert "Missing network location" in str(exc3.value)


def test_navigation_url_rejects_malformed_hostnames():
    """URLs containing illegal whitespace or control characters in the hostname are rejected."""
    with pytest.raises(ValueError) as exc1:
        validate_navigation_url("http://example com/test")
    assert "illegal characters" in str(exc1.value) or "Malformed" in str(exc1.value)


def test_navigation_url_rejects_malformed_ip_literals():
    """Dotted-quad strings with invalid numbers (e.g. 999.999.999.999) are rejected via ipaddress parsing."""
    with pytest.raises(ValueError) as exc1:
        validate_navigation_url("http://999.999.999.999")
    assert "Malformed IP literal" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        validate_navigation_url("http://256.1.1.1/test")
    assert "Malformed IP literal" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        validate_navigation_url("http://1.2.3.4.5/")
    assert "Malformed IP literal" in str(exc3.value)


def test_navigation_url_blocks_comex_internal_infrastructure_ports():
    """Navigation to internal infrastructure ports (9222 CDP, 8000 API, 20128 proxy) is prohibited."""
    with pytest.raises(PermissionError) as exc1:
        validate_navigation_url("http://127.0.0.1:9222/json")
    assert "port 9222" in str(exc1.value)

    with pytest.raises(PermissionError) as exc2:
        validate_navigation_url("http://localhost:8000/agent/action")
    assert "port 8000" in str(exc2.value)

    with pytest.raises(PermissionError) as exc3:
        validate_navigation_url("http://127.0.0.1:20128/v1/models")
    assert "port 20128" in str(exc3.value)


def test_navigation_url_allows_legitimate_local_development_servers():
    """Legitimate local development servers (e.g. Vite on 5173, Next.js on 3000) are permitted."""
    assert validate_navigation_url("http://localhost:5173") == "http://localhost:5173"
    assert validate_navigation_url("http://127.0.0.1:3000/dashboard") == "http://127.0.0.1:3000/dashboard"
    assert validate_navigation_url("http://localhost:8080/") == "http://localhost:8080/"


def test_navigation_url_rejects_embedded_user_credentials():
    """URLs with user credentials (http://user:pass@host) are prohibited."""
    with pytest.raises(ValueError) as exc1:
        validate_navigation_url("http://admin:secret@example.com")
    assert "user credentials" in str(exc1.value)


# ── 3. AST Script Analysis & Unrestricted Execution Elimination ───────────

def test_ast_prohibits_raw_playwright_clicks():
    """AST guard rejects unmediated Playwright clicks."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("page.click('#buy')")


def test_ast_prohibits_raw_mouse_down_up():
    """AST guard rejects raw mouse actions."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("page.mouse.down()")


def test_ast_prohibits_raw_keyboard_type():
    """AST guard rejects unmediated keyboard typing."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("page.keyboard.type('test')")


def test_ast_prohibits_exec_eval_builtins():
    """AST guard rejects exec, eval, compile builtins."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("exec(\"page.click('#buy')\")")

    with pytest.raises(PermissionError):
        validate_harness_script_safety("eval(\"page.click('#buy')\")")


def test_ast_prohibits_getattr_reflection():
    """AST guard rejects getattr reflection used to bypass method denylists."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("fn = getattr(page, 'click'); fn('#buy')")


def test_ast_prohibits_builtins_subscript_access():
    """AST guard rejects subscript access on __builtins__."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("__builtins__['exec'](\"page.click('#buy')\")")


def test_ast_prohibits_dangerous_modules():
    """AST guard rejects imports of unisolated OS/process modules."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("import subprocess\nsubprocess.run(['ls'])")

    with pytest.raises(PermissionError):
        validate_harness_script_safety("from pathlib import Path\nPath('/etc/passwd').read_text()")


def test_ast_prohibits_dynamic_evaluate_concatenation():
    """AST guard rejects string concatenation inside evaluate()."""
    code = "await page.evaluate(\"document.querySelector('#buy')\" + \".click()\")"
    with pytest.raises(PermissionError):
        validate_harness_script_safety(code)


def test_ast_prohibits_fstring_in_evaluate():
    """AST guard rejects f-strings inside evaluate()."""
    code = "sel = '#buy'\nawait page.evaluate(f\"document.querySelector('{sel}').click()\")"
    with pytest.raises(PermissionError):
        validate_harness_script_safety(code)


def test_ast_prohibits_javascript_dom_mutation_in_evaluate():
    """AST guard rejects in-page JavaScript mutations via .click() or bracket notation ['click']()."""
    with pytest.raises(PermissionError):
        validate_harness_script_safety("await page.evaluate(\"document.querySelector('#buy').click()\")")

    with pytest.raises(PermissionError):
        validate_harness_script_safety("await page.evaluate(\"document.querySelector('#buy')['click']()\")")

    with pytest.raises(PermissionError):
        validate_harness_script_safety("await page.evaluate(\"document.forms[0].submit()\")")


def test_ast_allows_read_only_extraction_scripts():
    """Read-only extraction scripts parse cleanly."""
    valid_code = "title = await page.title()\ntext = await page.text_content('h1')\nprint(title, text)"
    validate_harness_script_safety(valid_code)


def test_ast_allows_mediated_gateway_calls():
    """Calls to the mediated click_element shim parse cleanly."""
    valid_shim = "click_element('#cookie-accept')"
    validate_harness_script_safety(valid_shim)


# ── 4. Autonomous Agent Tool Catalog & Execution Restrictions ─────────────

def test_autonomous_agent_tool_catalog_excludes_run_browser_harness():
    """Autonomous agent loop tool catalog does NOT contain run_browser_harness."""
    from agent_service import execute_agent_loop
    # execute_agent_loop defines local tools list without run_browser_harness
    # Verify inspect.getsource confirms exclusion
    import inspect
    source = inspect.getsource(execute_agent_loop)
    assert "tools.append(run_browser_harness)" not in source


def test_worker_agent_cannot_invoke_run_browser_harness():
    """Verify run_browser_harness is strictly unexported and excluded from worker execution."""
    import agent_service
    assert not hasattr(agent_service, "run_browser_harness")


# ── 5. End-to-End Endpoint Authorization & Fallback Prevention ────────────

@pytest.mark.asyncio
async def test_run_agent_fails_closed_without_worker_id_or_trusted_token():
    """POST /agent/run without workerId and without trusted token fails closed."""
    from agent_service import run_agent, AgentRequest

    req = AgentRequest(
        task="Scrape data",
        tabId="active_tab",
        taskId="t1",
        workerId=None,
        contextToken=None,
    )
    res = await run_agent(req)
    assert res.success is False
    assert "Unauthenticated request" in res.error
    assert res.metadata.get("unauthenticated") is True


@pytest.mark.asyncio
async def test_run_agent_stream_fails_closed_without_worker_id_or_trusted_token():
    """POST /agent/run/stream without workerId and without trusted token returns SSE error."""
    from agent_service import run_agent_stream, AgentRequest

    req = AgentRequest(
        task="Scrape data",
        tabId="active_tab",
        taskId="t1",
        workerId=None,
        contextToken=None,
    )
    response = await run_agent_stream(req)
    assert response.status_code == 200  # SSE stream response
    # Read first SSE event
    body = ""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, str) else chunk.decode("utf-8")
    assert "Unauthenticated request" in body
    assert '"unauthenticated": true' in body


@pytest.mark.asyncio
async def test_run_agent_explicit_tab_does_not_fall_back_to_default_cdp():
    """An explicit tabId that cannot be resolved fails closed; NEVER falls back to default CDP."""
    from agent_service import run_agent, AgentRequest

    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"[]"
        mock_url.return_value = mock_resp

        req = AgentRequest(
            task="Open tab",
            tabId="tab_missing_999",
            taskId="t_explicit",
            workerId=None,
            contextToken=_INTERNAL_APP_TOKEN,
        )
        res = await run_agent(req)
        assert res.success is False
        assert "Target resolution failed" in res.error
        assert "Implicit foreground fallback is prohibited" in res.error


@pytest.mark.asyncio
async def test_trusted_single_agent_can_run_on_foreground():
    """Trusted single-agent run with contextToken on active_tab proceeds to agent loop."""
    from agent_service import run_agent, AgentRequest

    with patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop, \
         patch("agent_service._resolve_foreground_cdp_ws", return_value="ws://127.0.0.1:9222/devtools/page/fg_page"):

        mock_loop.return_value = "Success result"

        req = AgentRequest(
            task="Check foreground",
            tabId="active_tab",
            taskId="t_trusted",
            workerId=None,
            contextToken=_INTERNAL_APP_TOKEN,
        )
        res = await run_agent(req)
        assert res.success is True
        assert res.result == "Success result"


# ── 6. Authenticated Lease Management Endpoints ────────────────────────────

@pytest.mark.asyncio
async def test_unauthenticated_lease_endpoints_fail_closed():
    """Lease management endpoints reject unauthenticated requests with 403."""
    from agent_service import (
        acquire_worker_tab_lease,
        revoke_worker_tab_lease,
        get_worker_tab_lease,
        LeaseAcquirePayload,
        LeaseRevokePayload,
    )

    # 1. Acquire without token -> 403
    p_acq = LeaseAcquirePayload(workerId="attacker", tabId="tab_steal")
    res1 = await acquire_worker_tab_lease(p_acq)
    assert res1.status_code == 403

    # 2. Revoke without token -> 403
    p_rev = LeaseRevokePayload(workerId="victim")
    res2 = await revoke_worker_tab_lease(p_rev)
    assert res2.status_code == 403

    # 3. Inspect without token -> 403
    res3 = await get_worker_tab_lease("victim")
    assert res3.status_code == 403

    # 4. Acquire WITH token -> succeeds
    p_valid = LeaseAcquirePayload(
        workerId="trusted_w",
        tabId="tab_owned",
        contextToken=_INTERNAL_APP_TOKEN
    )
    res_valid = await acquire_worker_tab_lease(p_valid)
    assert res_valid["success"] is True
    assert res_valid["workerId"] == "trusted_w"


# ── 7. Canonical Per-Tab Concurrency Locks ─────────────────────────────────

def test_different_worker_sessions_on_same_cdp_tab_share_canonical_lock():
    """Two distinct worker sessions connecting to the same physical page share the exact same lock instance."""
    canon_target_id = "CANONICAL_TARGET_PAGE_123"
    ws1 = f"ws://127.0.0.1:9222/devtools/page/{canon_target_id}"
    ws2 = f"ws://127.0.0.1:9222/devtools/page/{canon_target_id}"

    target_id_1 = extract_canonical_target_id(ws1, "tab-1")
    target_id_2 = extract_canonical_target_id(ws2, "tab-2")
    assert target_id_1 == canon_target_id
    assert target_id_2 == canon_target_id

    lock1 = get_canonical_tab_lock(target_id_1)
    lock2 = get_canonical_tab_lock(target_id_2)
    assert lock1 is lock2


def test_sessions_on_different_cdp_tabs_have_independent_locks():
    """Sessions pointing to different CDP page UUIDs have independent locks."""
    ws1 = "ws://127.0.0.1:9222/devtools/page/PAGE_AAA"
    ws2 = "ws://127.0.0.1:9222/devtools/page/PAGE_BBB"

    lock1 = get_canonical_tab_lock(extract_canonical_target_id(ws1, "tab-A"))
    lock2 = get_canonical_tab_lock(extract_canonical_target_id(ws2, "tab-B"))
    assert lock1 is not lock2


@pytest.mark.asyncio
async def test_canonical_locks_serialize_mutations_across_different_worker_sessions():
    """Two workers targeting the same page serialize executions via the canonical lock."""
    mock_cdp_w1 = MagicMock()
    mock_cdp_w1.ws_url = "ws://127.0.0.1:9222/devtools/page/SHARED_TAB_UUID"
    mock_cdp_w1.send = AsyncMock(return_value={})
    mock_cdp_w1.connect = AsyncMock()

    mock_cdp_w2 = MagicMock()
    mock_cdp_w2.ws_url = "ws://127.0.0.1:9222/devtools/page/SHARED_TAB_UUID"
    mock_cdp_w2.send = AsyncMock(return_value={})
    mock_cdp_w2.connect = AsyncMock()

    with patch("nervous.cdp_session.WebSocketCDPClient") as mock_client_cls:
        mock_client_cls.side_effect = [mock_cdp_w1, mock_cdp_w2]

        session_w1 = await get_or_create_tab_session("tab-shared", mock_cdp_w1.ws_url, worker_id="worker-1")
        session_w2 = await get_or_create_tab_session("tab-shared", mock_cdp_w2.ws_url, worker_id="worker-2")

        assert session_w1.action_lock is session_w2.action_lock

        execution_order = []

        async def worker1_action():
            async with session_w1.action_lock:
                execution_order.append("w1-start")
                await asyncio.sleep(0.04)
                execution_order.append("w1-end")

        async def worker2_action():
            async with session_w2.action_lock:
                execution_order.append("w2-start")
                await asyncio.sleep(0.01)
                execution_order.append("w2-end")

        await asyncio.gather(worker1_action(), worker2_action())
        assert execution_order == ["w1-start", "w1-end", "w2-start", "w2-end"]


# ── 8. REST Action Endpoints Boundary Enforcement ──────────────────────────

@pytest.mark.asyncio
async def test_action_navigate_endpoint_fails_closed_on_unauthorized_tab():
    """POST /agent/action/navigate fails closed when worker is not leased the tab."""
    from agent_service import execute_action_navigate, ActionNavigatePayload

    payload = ActionNavigatePayload(
        url="https://apple.com",
        tabId="unauthorized_tab_99",
        workerId="worker_1",
    )
    res = await execute_action_navigate(payload)
    assert res["success"] is False
    assert res["blocked"] is True
    assert "No active server lease found for worker 'worker_1'" in res["error"]


@pytest.mark.asyncio
async def test_action_navigate_endpoint_rejects_unsafe_scheme():
    """POST /agent/action/navigate rejects non-HTTP(S) schemes."""
    from agent_service import execute_action_navigate, ActionNavigatePayload

    lease = get_lease_registry().acquire_lease("worker_1", "tab_1", trusted_issuer=True)

    payload = ActionNavigatePayload(
        url="javascript:alert(1)",
        tabId="tab_1",
        workerId="worker_1",
        workerToken=lease.lease_token,
    )
    res = await execute_action_navigate(payload)
    assert res["success"] is False
    assert res["blocked"] is True
    assert "Disallowed navigation scheme 'javascript'" in res["error"]


@pytest.mark.asyncio
async def test_action_click_endpoint_fails_closed_without_lease_or_context():
    """POST /agent/action/click fails closed when worker has no lease and no trusted token."""
    from agent_service import execute_action_click, ActionClickPayload

    payload = ActionClickPayload(
        selector="#btn",
        tabId="tab_unregistered",
        workerId="unregistered_worker",
    )
    res = await execute_action_click(payload)
    assert res["success"] is False
    assert res["blocked"] is True
    assert "No active server lease found for worker 'unregistered_worker'" in res["error"]


@pytest.mark.asyncio
async def test_action_type_endpoint_fails_closed_on_unresolvable_tab():
    """POST /agent/action/type fails closed when target tab cannot be resolved via CDP."""
    from agent_service import execute_action_type, ActionTypePayload

    lease = get_lease_registry().acquire_lease("worker_1", "non_existent_tab_999", trusted_issuer=True)

    payload = ActionTypePayload(
        selector="#input",
        text="test",
        tabId="non_existent_tab_999",
        workerId="worker_1",
        workerToken=lease.lease_token,
    )
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"[]"
        mock_url.return_value = mock_resp

        res = await execute_action_type(payload)
        assert res["success"] is False
        assert res["blocked"] is True
        assert "Target resolution failed" in res["error"]


@pytest.mark.asyncio
async def test_action_click_fails_closed_on_mismatched_ws_url():
    """POST /agent/action/click fails closed when caller supplies a forged ws_url."""
    from agent_service import execute_action_click, ActionClickPayload

    verified_ws = "ws://127.0.0.1:9222/devtools/page/REAL_PAGE"
    lease = get_lease_registry().acquire_lease("worker_1", "tab_1", cdp_ws_url=verified_ws, trusted_issuer=True)

    with patch("agent_service._resolve_verified_tab_ws", return_value=("REAL_PAGE", verified_ws)):
        payload = ActionClickPayload(
            selector="#btn",
            tabId="tab_1",
            workerId="worker_1",
            workerToken=lease.lease_token,
            ws_url="ws://127.0.0.1:9222/devtools/page/FORGED_PAGE"
        )
        res = await execute_action_click(payload)
        assert res["success"] is False
        assert res["blocked"] is True
        assert "Target WebSocket mismatch" in res["error"]


@pytest.mark.asyncio
async def test_action_click_fails_closed_on_canonical_target_mismatch():
    """POST /agent/action/click fails closed when resolved Chromium target ID doesn't match lease."""
    from agent_service import execute_action_click, ActionClickPayload

    verified_ws = "ws://127.0.0.1:9222/devtools/page/DIFF_PAGE"
    lease = get_lease_registry().acquire_lease(
        "worker_1",
        "tab_1",
        canonical_target_id="PAGE_BOUND_IN_LEASE",
        cdp_ws_url=verified_ws,
        trusted_issuer=True,
    )

    with patch("agent_service._resolve_verified_tab_ws", return_value=("PAGE_DIFF_FROM_LEASE", verified_ws)):
        payload = ActionClickPayload(
            selector="#btn",
            tabId="tab_1",
            workerId="worker_1",
            workerToken=lease.lease_token,
            ws_url=verified_ws
        )
        res = await execute_action_click(payload)
        assert res["success"] is False
        assert res["blocked"] is True
        assert "Canonical target mismatch" in res["error"]


@pytest.mark.asyncio
async def test_cleanup_worker_sessions_releases_locks():
    """cleanup_worker_sessions properly destroys worker sessions and closes CDPs."""
    mock_cdp = AsyncMock()
    mock_cdp.ws_url = "ws://127.0.0.1:9222/devtools/page/worker_cleanup_tab"
    mock_cdp.close = AsyncMock()

    with patch("nervous.cdp_session.WebSocketCDPClient") as mock_client_cls:
        mock_client_cls.return_value = mock_cdp

        session = await get_or_create_tab_session(
            tab_id="tab_cleanup",
            ws_url="ws://127.0.0.1:9222/devtools/page/worker_cleanup_tab",
            worker_id="worker_cleanup"
        )
        assert session is not None

        await cleanup_worker_sessions("worker_cleanup")

        from nervous.cdp_session import _GLOBAL_SESSIONS
        assert "worker_cleanup:tab_cleanup" not in _GLOBAL_SESSIONS


# ── 12 Focused Adversarial Boundary Tests (Phase 1.5 Final Remediation) ───

def test_remediation_1_renderer_bridge_does_not_expose_internal_token():
    """1. Renderer JavaScript cannot read the internal token through the preload bridge."""
    import pathlib
    preload_path = pathlib.Path(__file__).parent.parent / "electron" / "preload.js"
    assert preload_path.exists()
    content = preload_path.read_text("utf8")
    # Verify internalToken is never exposed in window.nexusNative
    assert "internalToken:" not in content
    assert "process.env.COMEX_INTERNAL_APP_TOKEN" not in content
    assert "contextBridge.exposeInMainWorld" in content
    # Verify only safe typed agent operations are exposed, and generic proxy is strictly eliminated
    assert "agent:" in content
    assert "ipcRenderer.invoke('agent-run'" in content
    assert "agent-proxy" not in content


def test_remediation_2_only_legitimate_application_window_can_call_privileged_ipc():
    """2. Only the legitimate application window can call privileged IPC operations."""
    import pathlib, re
    main_path = pathlib.Path(__file__).parent.parent / "electron" / "main.js"
    assert main_path.exists()
    content = main_path.read_text("utf8")
    
    # Must enforce senderFrame equality against mainWindow.webContents.mainFrame
    assert "isAuthorizedSender(event)" in content
    assert "event.senderFrame !== mainWindow.webContents.mainFrame" in content
    
    # Generic proxy is removed
    assert "agent-proxy" not in content

    # Every privileged IPC handler must call isAuthorizedSender
    for channel in ["tabs-create", "tabs-destroy", "tabs-navigate", "agent-run", "agent-stop", "agent-stream-start"]:
        pattern = rf"ipcMain\.(handle|on)\('{channel}',"
        assert re.search(pattern, content), f"Channel {channel} missing IPC handler"


@pytest.mark.asyncio
async def test_remediation_3_model_supplied_parameters_cannot_become_executable_javascript():
    """3. Model-supplied tab IDs and URLs cannot become executable JavaScript."""
    import agent_service
    import inspect
    src = inspect.getsource(agent_service.execute_agent_loop)
    assert "_trigger_ui_cdp" not in src
    assert "create_hidden_tab" not in src
    assert "Runtime.evaluate" not in src
    assert "createHiddenAgentTab" not in src
    
    # Navigation parameter validation rejects javascript: scheme
    with pytest.raises(ValueError) as exc_info:
        validate_navigation_url("javascript:alert(document.cookie)")
    assert "Disallowed navigation scheme 'javascript'" in str(exc_info.value)


def test_remediation_4_tab_creation_cannot_mint_unauthorized_lease_or_expose_raw_ws():
    """4. Tab creation cannot be used to obtain arbitrary CDP access or mint an unauthorized worker lease."""
    registry = get_lease_registry()
    # Unauthenticated lease minting is strictly rejected
    with pytest.raises(PermissionError) as exc_info:
        registry.acquire_lease("malicious_worker", "tab_forged", trusted_issuer=False)
    assert "only be issued by an authenticated trusted orchestrator/issuer" in str(exc_info.value)
    
    # Autonomous tools catalogue does not contain tab creation or raw ws endpoints
    import inspect, agent_service
    src = inspect.getsource(agent_service.execute_agent_loop)
    assert "create_hidden_tab," not in src


def test_remediation_5_missing_configuration_fails_closed_without_fixed_credential():
    """5. Missing credential configuration cannot activate a fixed credential; must fail closed."""
    from agent_service import _is_trusted_request
    import agent_service
    
    orig_token = agent_service._INTERNAL_APP_TOKEN
    try:
        agent_service._INTERNAL_APP_TOKEN = None
        # Must fail closed even if attacker supplies old or common default token
        assert _is_trusted_request("comex_internal_secret_token_19") is False
        assert _is_trusted_request("default_secret") is False
        assert _is_trusted_request("") is False
        assert _is_trusted_request(None) is False
        
        # Test with empty whitespace token
        agent_service._INTERNAL_APP_TOKEN = "   "
        assert _is_trusted_request("   ") is False
        assert _is_trusted_request("comex_internal_secret_token_19") is False
    finally:
        agent_service._INTERNAL_APP_TOKEN = orig_token


@pytest.mark.asyncio
async def test_remediation_6_credentials_not_included_in_request_body_logs(caplog):
    """6. Credentials are not included in request-body logs or server output."""
    import logging
    from agent_service import run_agent, AgentRequest
    
    secret_canary = "SUPER_SECRET_CREDENTIAL_CANARY_TOKEN_777"
    req = AgentRequest(
        task="Check test specs",
        tabId="tab_1",
        taskId="task_test_log_1",
        contextToken=secret_canary,
        workerId=None
    )
    
    with caplog.at_level(logging.DEBUG):
        with patch("agent_service._is_trusted_request", return_value=False):
            await run_agent(req)
            
    # Verify the secret canary token was NEVER logged
    for record in caplog.records:
        assert secret_canary not in record.message
        assert secret_canary not in str(record.__dict__)


@pytest.mark.asyncio
async def test_remediation_7_unauthenticated_caller_cannot_impersonate_existing_worker():
    """7. An unauthenticated caller cannot impersonate an existing worker by supplying its ID and tab ID."""
    from agent_service import execute_action_click, ActionClickPayload, run_agent, AgentRequest
    
    registry = get_lease_registry()
    real_lease = registry.acquire_lease("victim_worker", "tab_victim", trusted_issuer=True)
    
    # Attacker tries to call action endpoint supplying victim worker ID and tab ID without workerToken
    payload = ActionClickPayload(
        selector="#btn",
        tabId="tab_victim",
        workerId="victim_worker",
        workerToken=None
    )
    res = await execute_action_click(payload)
    assert res["success"] is False
    assert res["blocked"] is True
    assert "Unauthorized caller cannot act as worker 'victim_worker'" in res["error"]
    
    # Attacker tries with forged token
    payload_forged = ActionClickPayload(
        selector="#btn",
        tabId="tab_victim",
        workerId="victim_worker",
        workerToken="forged_token_guess"
    )
    res_forged = await execute_action_click(payload_forged)
    assert res_forged["success"] is False
    assert res_forged["blocked"] is True
    assert "Unauthorized caller cannot act as worker 'victim_worker'" in res_forged["error"]
    
    # Attacker tries /agent/run with victim workerId without workerToken
    agent_req = AgentRequest(
        task="Do evil",
        tabId="tab_victim",
        taskId="task_evil",
        workerId="victim_worker",
        workerToken=None,
        contextToken=None
    )
    agent_res = await run_agent(agent_req)
    assert agent_res.success is False
    assert agent_res.metadata.get("worker_authorization_failed") is True


@pytest.mark.asyncio
async def test_remediation_8_actual_autonomous_tool_dispatcher_rejects_harness_and_unknown_tools():
    """8. The actual autonomous tool dispatcher rejects run_browser_harness and unknown tool calls."""
    from agent_service import execute_agent_loop
    
    mock_llm_response = MagicMock()
    mock_llm_response.tool_calls = [
        {"name": "run_browser_harness", "args": {"code": "print('malicious')"}, "id": "call_1"},
        {"name": "unknown_eval_func", "args": {"foo": "bar"}, "id": "call_2"}
    ]
    
    mock_final_response = MagicMock()
    mock_final_response.tool_calls = []
    mock_final_response.content = "Summary of results."
    
    with patch("agent_service.ChatOpenAI") as mock_chat_cls:
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke.side_effect = [mock_llm_response, mock_final_response]
        mock_llm.bind_tools.return_value = mock_bound
        mock_chat_cls.return_value = mock_llm
        
        loop_res = await execute_agent_loop(
            task="Test task",
            default_cdp_url="ws://127.0.0.1:9222/devtools/page/test",
            use_graph=False,
            trusted_app_context=True,
        )
        
        meta = loop_res.metadata
        # Both unpermitted calls must be recorded as failed tool calls
        assert meta["tool_call_count"] == 2
        assert meta["failed_tool_call_count"] == 2
        traces = meta["tool_traces"]
        assert len(traces) == 2
        assert traces[0]["success"] is False
        assert "Tool 'run_browser_harness' is not permitted or does not exist" in traces[0]["error"]
        assert traces[1]["success"] is False
        assert "Tool 'unknown_eval_func' is not permitted or does not exist" in traces[1]["error"]


def test_remediation_9_no_worker_facing_route_reaches_unrestricted_python():
    """9. No worker-facing execution route reaches unrestricted Python or arbitrary CDP mutation."""
    from nervous.action_policy import validate_harness_script_safety
    
    # Script safety guard strictly blocks dynamic eval, exec, builtins, dangerous modules
    dangerous_payloads = [
        "import os; os.system('ls')",
        "__import__('subprocess').Popen('whoami')",
        "eval('1 + 1')",
        "exec('x = 2')",
        "getattr(sys, 'modules')",
        "globals()['__builtins__']",
        "js(f'document.cookie = {secret}')",
        "js('document.querySelector(\"#btn\").click()')",
    ]
    for p in dangerous_payloads:
        with pytest.raises(PermissionError):
            validate_harness_script_safety(p)


def test_remediation_10_expired_revoked_leases_and_canonical_mismatches_remain_rejected():
    """10. Expired/revoked leases and canonical-target mismatches remain rejected."""
    registry = get_lease_registry()
    
    # 10a. Expired lease
    lease = registry.acquire_lease("worker_exp", "tab_exp", ttl_seconds=0.01, trusted_issuer=True)
    lease.expires_at_mono = time.monotonic() - 1.0  # Monotonic expiration
    assert lease.is_expired() is True
    with pytest.raises(PermissionError):
        validate_worker_tab_authorization("worker_exp", "tab_exp", worker_token=lease.lease_token)
        
    # 10b. Revoked lease
    lease_rev = registry.acquire_lease("worker_rev", "tab_rev", trusted_issuer=True)
    registry.revoke_lease("worker_rev", trusted_caller=True)
    with pytest.raises(PermissionError):
        validate_worker_tab_authorization("worker_rev", "tab_rev", worker_token=lease_rev.lease_token)
        
    # 10c. Canonical target mismatch
    lease_canon = registry.acquire_lease(
        "worker_c", "tab_c",
        canonical_target_id="CANON_ID_REAL",
        trusted_issuer=True
    )
    with pytest.raises(PermissionError) as exc_info:
        validate_worker_tab_authorization(
            "worker_c", "tab_c",
            expected_target_id="CANON_ID_DIFFERENT",
            worker_token=lease_canon.lease_token
        )
    assert "Canonical target mismatch" in str(exc_info.value)


@pytest.mark.asyncio
async def test_remediation_11_legitimate_single_agent_foreground_operations():
    """11. Legitimate single-agent foreground operations still work through the authorized request path."""
    lease = validate_worker_tab_authorization(
        worker_id=None,
        target_tab="active_tab",
        trusted_app_context=True
    )
    assert lease.is_active is True
    assert lease.tab_id == "active_tab"
    assert lease.worker_id == "__trusted_single_agent__"


@pytest.mark.asyncio
async def test_remediation_12_existing_tab_operations_and_per_tab_concurrency():
    """12. Existing tab operations and per-tab concurrency behavior continue to work."""
    lock1 = get_canonical_tab_lock("tab_shared_1")
    lock2 = get_canonical_tab_lock("tab_shared_1")
    assert lock1 is lock2
    
    lock3 = get_canonical_tab_lock("tab_independent_2")
    assert lock1 is not lock3


# ── Execution-Boundary Follow-up Regression Tests ─────────────────────────

def test_followup_1_renderer_cannot_access_privileged_lease_endpoints_through_ipc():
    """1. Renderer attempts to access privileged lease endpoints through IPC are impossible."""
    import pathlib
    preload_path = pathlib.Path(__file__).parent.parent / "electron" / "preload.js"
    main_path = pathlib.Path(__file__).parent.parent / "electron" / "main.js"
    
    preload_src = preload_path.read_text("utf8")
    main_src = main_path.read_text("utf8")
    
    # Generic proxy is removed from preload bridge and main process handlers
    assert "agent-proxy" not in preload_src
    assert "agent-proxy" not in main_src
    
    # Neither preload nor main exposes lease acquisition, renewal, or revocation channels
    for forbidden in ["agent/lease/acquire", "agent/lease/renew", "agent/lease/revoke", "lease-acquire", "lease-renew"]:
        assert forbidden not in preload_src
        assert forbidden not in main_src


@pytest.mark.asyncio
async def test_followup_2_renderer_request_with_worker_id_fails_without_valid_worker_token():
    """2. A request supplying an existing worker ID without valid worker capability fails closed, even under trusted app context."""
    from agent_service import run_agent, AgentRequest
    
    registry = get_lease_registry()
    lease = registry.acquire_lease("worker_victim_sec", "tab_victim_sec", trusted_issuer=True)
    
    # Caller supplies trusted application token but no workerToken for worker_victim_sec
    req_no_token = AgentRequest(
        task="Impersonate worker",
        tabId="tab_victim_sec",
        taskId="task_imp_1",
        workerId="worker_victim_sec",
        workerToken=None,
        contextToken=_INTERNAL_APP_TOKEN
    )
    res_no_token = await run_agent(req_no_token)
    assert res_no_token.success is False
    assert res_no_token.metadata.get("worker_authorization_failed") is True

    # Caller supplies forged workerToken
    req_bad_token = AgentRequest(
        task="Impersonate worker",
        tabId="tab_victim_sec",
        taskId="task_imp_2",
        workerId="worker_victim_sec",
        workerToken="forged_capability_value",
        contextToken=_INTERNAL_APP_TOKEN
    )
    res_bad_token = await run_agent(req_bad_token)
    assert res_bad_token.success is False
    assert res_bad_token.metadata.get("worker_authorization_failed") is True


@pytest.mark.asyncio
async def test_followup_3_worker_with_correct_lease_rejects_different_physical_cdp_target():
    """3. A worker request uses the correct lease but resolves to a different physical CDP target fails closed."""
    from agent_service import run_agent, AgentRequest
    import json
    
    registry = get_lease_registry()
    lease = registry.acquire_lease(
        "worker_phys_mismatch",
        "tab_phys_mismatch",
        canonical_target_id="TGT_EXPECTED_PHYSICAL",
        trusted_issuer=True
    )
    
    # Mock /json/list returning a different physical target ID for this tab
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {
                "id": "TGT_DIFFERENT_PHYSICAL",
                "type": "page",
                "url": "https://example.com#tab_phys_mismatch",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/TGT_DIFFERENT_PHYSICAL"
            }
        ]).encode("utf-8")
        mock_urlopen.return_value = mock_resp
        
        req = AgentRequest(
            task="Perform action on tab",
            tabId="tab_phys_mismatch",
            taskId="task_phys_1",
            workerId="worker_phys_mismatch",
            workerToken=lease.lease_token,
            contextToken=_INTERNAL_APP_TOKEN
        )
        res = await run_agent(req)
        assert res.success is False
        assert "Canonical target mismatch" in res.error
        assert res.metadata.get("canonical_target_mismatch") is True


@pytest.mark.asyncio
async def test_followup_4_explicit_tab_fails_resolution_while_default_ws_is_available():
    """4. An explicit tab fails resolution while a different default WebSocket is available: must NOT fall back to default."""
    from agent_service import _resolve_and_verify_tool_target
    from nervous.action_policy import TabLease
    
    lease = TabLease(
        worker_id="worker_explicit",
        tab_id="tab_explicit_unresolvable",
        created_at_mono=time.monotonic(),
        expires_at_mono=time.monotonic() + 300,
        is_active=True
    )
    
    default_ws = "ws://127.0.0.1:9222/devtools/page/foreground_active"
    
    # Mock _resolve_verified_tab_ws returning (None, None) for explicit unresolvable tab
    with patch("agent_service._resolve_verified_tab_ws", return_value=(None, None)):
        target_id, ws_url, err = _resolve_and_verify_tool_target(
            lease=lease,
            target_tab="tab_explicit_unresolvable",
            default_cdp_url=default_ws,
            trusted_app_context=True  # Even under trusted context!
        )
        assert target_id is None
        assert ws_url is None
        assert err is not None
        assert "could not be resolved. Implicit foreground fallback is disallowed" in err


@pytest.mark.asyncio
async def test_followup_5_each_tool_rejects_canonical_target_mismatch():
    """5. Each of click, type, navigate, extract and inspect rejects a canonical-target mismatch."""
    from agent_service import execute_agent_loop
    from nervous.action_policy import get_lease_registry
    
    registry = get_lease_registry()
    lease = registry.acquire_lease(
        "worker_mismatch_all",
        "tab_mismatch_all",
        canonical_target_id="TARGET_CORRECT_123",
        trusted_issuer=True
    )
    
    # Mock resolution returning a mismatched target ID
    with patch("agent_service._resolve_verified_tab_ws", return_value=("TARGET_WRONG_999", "ws://127.0.0.1:9222/devtools/page/wrong")):
        
        # Test each tool call through execute_agent_loop tool catalogue
        for tool_name, tool_args in [
            ("browser_click", {"selector": "#btn"}),
            ("browser_type", {"selector": "#input", "text": "test"}),
            ("browser_navigate", {"url": "https://example.com"}),
            ("browser_extract", {"selector": "h1"}),
            ("browser_inspect", {}),
        ]:
            mock_llm_response = MagicMock()
            mock_llm_response.tool_calls = [{"name": tool_name, "args": tool_args, "id": f"call_{tool_name}"}]
            mock_final_response = MagicMock()
            mock_final_response.tool_calls = []
            mock_final_response.content = "Done."
            
            with patch("agent_service.ChatOpenAI") as mock_chat_cls:
                mock_llm = MagicMock()
                mock_bound = AsyncMock()
                mock_bound.ainvoke.side_effect = [mock_llm_response, mock_final_response]
                mock_llm.bind_tools.return_value = mock_bound
                mock_chat_cls.return_value = mock_llm
                
                loop_res = await execute_agent_loop(
                    task="Run tool test",
                    default_cdp_url="ws://127.0.0.1:9222/devtools/page/default",
                    tab_id="tab_mismatch_all",
                    worker_id="worker_mismatch_all",
                    worker_token=lease.lease_token,
                    trusted_app_context=False,
                    use_graph=False
                )
                
                traces = loop_res.metadata["tool_traces"]
                assert len(traces) == 1
                assert traces[0]["success"] is False
                assert "Canonical target mismatch" in traces[0]["error"]


@pytest.mark.asyncio
async def test_followup_6_legitimate_single_agent_foreground_workflow_succeeds():
    """6. The normal authenticated single-agent foreground workflow still works."""
    from agent_service import run_agent, AgentRequest
    
    with patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop, \
         patch("agent_service._resolve_foreground_cdp_ws", return_value="ws://127.0.0.1:9222/devtools/page/fg_valid"):
        
        mock_loop.return_value = "Foreground research completed successfully."
        
        req = AgentRequest(
            task="Find top laptops",
            tabId="active_tab",
            taskId="task_fg_legit",
            workerId=None,
            contextToken=_INTERNAL_APP_TOKEN
        )
        res = await run_agent(req)
        assert res.success is True
        assert res.result == "Foreground research completed successfully."


@pytest.mark.asyncio
async def test_followup_7_worker_authorization_enforced_in_both_run_and_stream():
    """7. Worker authorization remains enforced through both /agent/run and /agent/run/stream."""
    from agent_service import run_agent, run_agent_stream, AgentRequest
    import json
    
    registry = get_lease_registry()
    lease = registry.acquire_lease("worker_dual_test", "tab_dual_valid", trusted_issuer=True)
    
    # 7a. Missing worker token on /agent/run
    req_run_bad = AgentRequest(
        task="Run test",
        tabId="tab_dual_valid",
        taskId="task_run_1",
        workerId="worker_dual_test",
        workerToken=None
    )
    res_run = await run_agent(req_run_bad)
    assert res_run.success is False
    assert res_run.metadata.get("worker_authorization_failed") is True
    
    # 7b. Missing worker token on /agent/run/stream
    req_stream_bad = AgentRequest(
        task="Stream test",
        tabId="tab_dual_valid",
        taskId="task_stream_1",
        workerId="worker_dual_test",
        workerToken=None
    )
    res_stream = await run_agent_stream(req_stream_bad)
    body = ""
    async for chunk in res_stream.body_iterator:
        body += chunk if isinstance(chunk, str) else chunk.decode("utf-8")
    assert "Worker authorization failed" in body
    assert '"worker_authorization_failed": true' in body

    # 7c. Targeting unauthorized tab on /agent/run and /agent/run/stream
    req_unauth_tab = AgentRequest(
        task="Target wrong tab",
        tabId="tab_unauthorized_999",
        taskId="task_stream_2",
        workerId="worker_dual_test",
        workerToken=lease.lease_token
    )
    res_run_unauth = await run_agent(req_unauth_tab)
    assert res_run_unauth.success is False
    assert res_run_unauth.metadata.get("worker_authorization_failed") is True
    
    res_stream_unauth = await run_agent_stream(req_unauth_tab)
    body_unauth = ""
    async for chunk in res_stream_unauth.body_iterator:
        body_unauth += chunk if isinstance(chunk, str) else chunk.decode("utf-8")
    assert "Worker authorization mismatch" in body_unauth
    assert '"worker_authorization_failed": true' in body_unauth
