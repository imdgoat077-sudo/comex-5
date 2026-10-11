"""
Phase 1 Foundation Tests:
1. Fast memory retrieval without procedural LLM calls by default.
2. Preserved SiteAdapter and ordinary memory functionality.
3. Hidden-tab events not leaking into visible UI.
4. Two distinct worker/tab IDs creating distinct session contexts.
5. Missing target IDs failing closed instead of falling back to foreground.
6. Concurrent actions on the same tab being serialized via action_lock.
7. Actions on different tabs retaining separate browser targets and independent locks.
8. Cancellation and closed targets cleaning up sessions without stale leaks.
9. Browser-harness actions not bypassing the enforced action policy.
"""

import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

import agent_service
from agent_service import app, execute_agent_loop, AgentRequest
from nervous.cdp_session import (
    TabSessionContext,
    _GLOBAL_SESSIONS,
    get_or_create_tab_session,
    close_tab_session,
    cleanup_worker_sessions,
)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure session dictionary is cleared before and after each test."""
    _GLOBAL_SESSIONS.clear()
    yield
    _GLOBAL_SESSIONS.clear()


@pytest.mark.asyncio
async def test_fast_memory_retrieval_has_no_procedural_llm_calls_by_default():
    """Verify that in the default configuration, procedural retrieval is disabled,
    eliminating upstream LLM roundtrips during normal task startup."""
    mock_llm_response = MagicMock()
    mock_llm_response.tool_calls = []
    mock_llm_response.content = "Direct answer without procedure."

    with patch("agent_service.ChatOpenAI") as mock_chat_cls, \
         patch("agent_service.generate_procedure_route", new_callable=AsyncMock) as mock_route:
        
        mock_chat_inst = MagicMock()
        mock_chat_inst.bind_tools.return_value = mock_chat_inst
        mock_chat_inst.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_chat_cls.return_value = mock_chat_inst

        # execute_agent_loop with default use_procedures=False and default config
        result = await execute_agent_loop("What is the capital of France?", "http://fake", trusted_app_context=True)
        
        # Verify generate_procedure_route was NOT called
        mock_route.assert_not_called()
        
        # Verify timing instrumentation exists and captures metrics
        meta = result.metadata
        assert "timing" in meta
        timing = meta["timing"]
        assert "memory_prep_ms" in timing
        assert "db_query_ms" in timing
        assert "llm_roundtrips_ms" in timing
        assert "total_llm_ms" in timing
        assert "time_to_first_token_ms" in timing
        assert "total_duration_ms" in timing
        assert timing["memory_prep_ms"] >= 0.0
        assert len(timing["llm_roundtrips_ms"]) >= 1


@pytest.mark.asyncio
async def test_distinct_worker_tab_sessions():
    """Verify that distinct worker IDs and tab IDs produce distinct session contexts."""
    mock_cdp_a = MagicMock()
    mock_cdp_a.ws_url = "ws://127.0.0.1:9222/devtools/page/worker1_tab1"
    mock_cdp_a.send = AsyncMock(return_value={})
    mock_cdp_a.connect = AsyncMock()

    mock_cdp_b = MagicMock()
    mock_cdp_b.ws_url = "ws://127.0.0.1:9222/devtools/page/worker2_tab2"
    mock_cdp_b.send = AsyncMock(return_value={})
    mock_cdp_b.connect = AsyncMock()

    with patch("nervous.cdp_session.WebSocketCDPClient") as mock_client_cls:
        mock_client_cls.side_effect = [mock_cdp_a, mock_cdp_b]

        session_1 = await get_or_create_tab_session("tab-1", "ws://127.0.0.1:9222/devtools/page/worker1_tab1", worker_id="worker-1")
        session_2 = await get_or_create_tab_session("tab-2", "ws://127.0.0.1:9222/devtools/page/worker2_tab2", worker_id="worker-2")

        assert session_1 is not session_2
        assert session_1.worker_id == "worker-1"
        assert session_2.worker_id == "worker-2"
        assert session_1.tab_id == "tab-1"
        assert session_2.tab_id == "tab-2"
        assert "worker-1:tab-1" in _GLOBAL_SESSIONS
        assert "worker-2:tab-2" in _GLOBAL_SESSIONS
        assert session_1.action_lock is not session_2.action_lock


@pytest.mark.asyncio
async def test_missing_target_id_fails_closed_in_run_agent():
    """Verify that when a specific worker tabId is requested but cannot be found in /json/list,
    the endpoint fails closed with a structured error rather than falling back to the foreground tab."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("urllib.request.urlopen") as mock_urlopen:
            # Mock /json/list returning only the foreground tab
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps([
                {"id": "fg-tab", "type": "page", "url": "https://google.com", "webSocketDebuggerUrl": "ws://dummy/fg"}
            ]).encode("utf-8")
            mock_urlopen.return_value = mock_resp

            from nervous.action_policy import get_lease_registry
            lease = get_lease_registry().acquire_lease("worker-99", "worker-tab-missing", trusted_issuer=True)

            payload = {
                "task": "Scrape in background",
                "tabId": "worker-tab-missing",
                "taskId": "task-abc",
                "workerId": "worker-99",
                "workerToken": lease.lease_token,
            }
            res = await client.post("/agent/run", json=payload)
            data = res.json()

            assert data["success"] is False
            assert "Target resolution failed" in data["error"]
            assert "Implicit foreground fallback is prohibited" in data["error"]
            assert data["metadata"].get("target_resolution_failed") is True


@pytest.mark.asyncio
async def test_missing_target_id_fails_closed_in_browser_click():
    """Verify browser_click tool fails closed if a specified worker tabId cannot be resolved."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {"id": "fg-tab", "type": "page", "url": "https://google.com", "webSocketDebuggerUrl": "ws://dummy/fg"}
        ]).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        mock_llm_response = MagicMock()
        mock_llm_response.tool_calls = [{"name": "browser_click", "args": {"selector": "#submit"}, "id": "tc1"}]
        mock_llm_response.content = ""

        mock_final_response = MagicMock()
        mock_final_response.tool_calls = []
        mock_final_response.content = "Failed as expected."

        with patch("agent_service.ChatOpenAI") as mock_chat_cls:
            mock_chat_inst = MagicMock()
            mock_chat_inst.bind_tools.return_value = mock_chat_inst
            mock_chat_inst.ainvoke = AsyncMock(side_effect=[mock_llm_response, mock_final_response])
            mock_chat_cls.return_value = mock_chat_inst

            result = await execute_agent_loop(
                task="Click missing tab button",
                default_cdp_url="http://127.0.0.1:9222",
                tab_id="tab-nonexistent-worker",
                worker_id="worker-w1",
                legacy_mode=True,
            )

            traces = result.metadata["tool_traces"]
            assert len(traces) >= 1
            assert traces[0]["tool"] == "browser_click"
            assert traces[0]["success"] is False


@pytest.mark.asyncio
async def test_concurrent_actions_on_same_tab_are_serialized():
    """Verify that multiple concurrent actions on the same tab are serialized via action_lock."""
    mock_cdp = MagicMock()
    mock_cdp.ws_url = "ws://127.0.0.1:9222/devtools/page/test1"
    mock_cdp.send = AsyncMock(return_value={})
    mock_cdp.connect = AsyncMock()

    with patch("nervous.cdp_session.WebSocketCDPClient", return_value=mock_cdp):
        session = await get_or_create_tab_session("tab-1", "ws://127.0.0.1:9222/devtools/page/test1", worker_id="worker-1")

        execution_order = []

        async def worker_action(action_id: int, sleep_duration: float):
            async with session.action_lock:
                execution_order.append(f"start-{action_id}")
                await asyncio.sleep(sleep_duration)
                execution_order.append(f"end-{action_id}")

        # Start two concurrent tasks attempting actions on the same tab
        task1 = asyncio.create_task(worker_action(1, 0.05))
        task2 = asyncio.create_task(worker_action(2, 0.01))
        await asyncio.gather(task1, task2)

        # Confirm strict serialization: task 1 finished before task 2 started
        assert execution_order == ["start-1", "end-1", "start-2", "end-2"]


@pytest.mark.asyncio
async def test_actions_on_different_tabs_retain_separate_targets_and_locks():
    """Verify that actions on distinct tabs do not block each other."""
    mock_cdp_1 = MagicMock()
    mock_cdp_1.ws_url = "ws://127.0.0.1:9222/devtools/page/tab1"
    mock_cdp_1.send = AsyncMock(return_value={})
    mock_cdp_1.connect = AsyncMock()

    mock_cdp_2 = MagicMock()
    mock_cdp_2.ws_url = "ws://127.0.0.1:9222/devtools/page/tab2"
    mock_cdp_2.send = AsyncMock(return_value={})
    mock_cdp_2.connect = AsyncMock()

    with patch("nervous.cdp_session.WebSocketCDPClient") as mock_client_cls:
        mock_client_cls.side_effect = [mock_cdp_1, mock_cdp_2]

        session_1 = await get_or_create_tab_session("tab-1", "ws://127.0.0.1:9222/devtools/page/tab1", worker_id="worker-1")
        session_2 = await get_or_create_tab_session("tab-2", "ws://127.0.0.1:9222/devtools/page/tab2", worker_id="worker-2")

        execution_order = []

        async def action_tab1():
            async with session_1.action_lock:
                execution_order.append("start-tab1")
                await asyncio.sleep(0.05)
                execution_order.append("end-tab1")

        async def action_tab2():
            async with session_2.action_lock:
                execution_order.append("start-tab2")
                await asyncio.sleep(0.01)
                execution_order.append("end-tab2")

        # Tab 2 should be able to start and finish while Tab 1 is still sleeping
        await asyncio.gather(action_tab1(), action_tab2())
        assert "start-tab2" in execution_order
        assert execution_order[0] == "start-tab1"
        assert execution_order[1] == "start-tab2"
        assert execution_order[2] == "end-tab2"
        assert execution_order[3] == "end-tab1"


@pytest.mark.asyncio
async def test_cancellation_and_cleanup_worker_sessions():
    """Verify that cleaning up a worker's sessions cleanly closes and deletes all associated tab sessions."""
    mock_cdp = MagicMock()
    mock_cdp.ws_url = "ws://127.0.0.1:9222/devtools/page/tab1"
    mock_cdp.send = AsyncMock(return_value={})
    mock_cdp.connect = AsyncMock()
    mock_cdp.close = AsyncMock()

    with patch("nervous.cdp_session.WebSocketCDPClient", return_value=mock_cdp):
        session = await get_or_create_tab_session("tab-worker-1", "ws://127.0.0.1:9222/devtools/page/tab1", worker_id="worker-to-clean")
        assert "worker-to-clean:tab-worker-1" in _GLOBAL_SESSIONS

        await cleanup_worker_sessions("worker-to-clean")

        assert "worker-to-clean:tab-worker-1" not in _GLOBAL_SESSIONS
        mock_cdp.close.assert_awaited()


@pytest.mark.asyncio
async def test_action_click_endpoint_enforces_gate():
    """Verify /agent/action/click endpoint evaluates ActuatorGate and returns structured allow or block."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        mock_cdp = MagicMock()
        mock_cdp.ws_url = "ws://127.0.0.1:9222/devtools/page/action_tab"
        mock_cdp.send = AsyncMock(return_value={})
        mock_cdp.connect = AsyncMock()

        with patch("nervous.cdp_session.WebSocketCDPClient", return_value=mock_cdp), \
             patch("nervous.cdp_session.get_or_create_tab_session") as mock_get_session:

            mock_session = MagicMock()
            mock_session.action_lock = asyncio.Lock()
            mock_session.tracker.state.is_target_subtree_quiet = True
            mock_session.policy_registry.get_policy.return_value = True
            mock_get_session.return_value = mock_session

            # Mock binding and gate result
            mock_binding = MagicMock()
            mock_session.resolve_and_bind = AsyncMock(return_value=mock_binding)

            mock_gate_res = MagicMock()
            mock_gate_res.executed = True
            mock_gate_res.handler_invoked = False
            mock_gate_res.decision.rule_name = "ALLOW"
            mock_gate_res.execution_result = {"x": 100.0, "y": 200.0}
            mock_session.gate.async_attempt = AsyncMock(return_value=mock_gate_res)

            from agent_service import _INTERNAL_APP_TOKEN
            with patch("agent_service._resolve_verified_tab_ws", return_value=("action_tab", "ws://127.0.0.1:9222/devtools/page/action_tab")):
                res = await client.post("/agent/action/click", json={
                    "selector": "#buy-now",
                    "tabId": "active_tab",
                    "ws_url": "ws://127.0.0.1:9222/devtools/page/action_tab",
                    "contextToken": _INTERNAL_APP_TOKEN,
                })
                data = res.json()

            assert data["success"] is True
            assert data["blocked"] is False
            assert data["result"]["x"] == 100.0


def test_electron_hidden_tab_listeners_contract():
    """Verify in electron/main.js that hiddenAgentViews explicitly attaches attachHiddenTabListeners,
    denies window.open, and has no notifyRenderer calls."""
    electron_main_path = os.path.join(os.path.dirname(__file__), "..", "electron", "main.js")
    with open(electron_main_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. hiddenAgentViews calls attachHiddenTabListeners, NOT attachTabListeners
    assert "attachHiddenTabListeners(tabId, view)" in content
    
    # 2. attachHiddenTabListeners explicitly denies window.open
    assert "function attachHiddenTabListeners(tabId, view)" in content
    # Find attachHiddenTabListeners function body
    fn_start = content.index("function attachHiddenTabListeners(tabId, view)")
    fn_end = content.index("function loadTabUrl", fn_start)
    fn_body = content[fn_start:fn_end]

    assert "webContents.setWindowOpenHandler" in fn_body
    assert "return { action: 'deny' };" in fn_body
    # Ensure hidden tab listeners contain no notifyRenderer calls (call invocations)
    assert "notifyRenderer(" not in fn_body


@pytest.mark.asyncio
async def test_browser_harness_shim_prohibits_unmediated_clicks():
    """Verify that browser-harness execution code shim prohibits unmediated coordinate clicks (click_at_xy)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        
        mock_llm_response = MagicMock()
        mock_llm_response.tool_calls = [{
            "name": "run_browser_harness",
            "args": {"code": "print('hello')"},
            "id": "tc-bh"
        }]
        mock_llm_response.content = ""

        mock_final = MagicMock()
        mock_final.tool_calls = []
        mock_final.content = "Done."

        with patch("agent_service.ChatOpenAI") as mock_chat_cls:
            mock_chat_inst = MagicMock()
            mock_chat_inst.bind_tools.return_value = mock_chat_inst
            mock_chat_inst.ainvoke = AsyncMock(side_effect=[mock_llm_response, mock_final])
            mock_chat_cls.return_value = mock_chat_inst

            await execute_agent_loop(
                task="Run script",
                default_cdp_url="http://127.0.0.1:9222",
                tab_id="tab-worker-x",
                worker_id=None
            )

            # Verify tool allowlist rejects run_browser_harness so subprocess is never called
            assert not mock_run.called

        from nervous.action_policy import validate_harness_script_safety
        with pytest.raises(PermissionError):
            validate_harness_script_safety("click_at_xy(100, 200)")


@pytest.mark.asyncio
async def test_preserved_site_adapter_fast_path():
    """Verify that compiled SiteAdapter fast path executes directly and is preserved."""
    mock_adapter = {
        "id": "adap_test_1",
        "name": "search_books",
        "domain": "books.example.com",
        "match_score": 0.95,
        "dom_fingerprint": {"#search-input": True},
        "parameter_schema": {},
        "code_snippet": "print('fast adapter')"
    }

    with patch("agent_service.graph_store") as mock_store, \
         patch("graph.adapter_runner.verify_dom_fingerprint", return_value=(True, None)), \
         patch("graph.adapter_runner.execute_fast_path_adapter") as mock_exec:
        
        mock_store.find_matching_adapter.return_value = [mock_adapter]
        mock_exec.return_value = {"durationMs": 42, "output": "Found 5 books"}

        res = await execute_agent_loop("search books on books.example.com", "https://books.example.com", trusted_app_context=True)
        
        assert res.metadata.get("fast_path_executed") is True
        assert res.metadata.get("adapter_id") == "adap_test_1"
        assert "Executed via Fast-Path Adapter" in str(res)
        mock_exec.assert_called_once()

