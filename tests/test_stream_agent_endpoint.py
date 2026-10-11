import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from nervous import get_target_registry
import agent_service

client = TestClient(agent_service.app)

def test_stream_endpoint_contract():
    # Test POST /agent/run/stream with mocked execute_agent_loop
    with patch("agent_service._resolve_verified_tab_ws", return_value=("target-tab-1", "ws://127.0.0.1:9222/devtools/page/target-tab-1")), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
        class FakeResult:
            def __str__(self):
                return "Successfully interacted with page."
            metadata = {"tool_call_count": 1}

        async def fake_loop(task, default_cdp_url, **kwargs):
            callback = kwargs.get("event_callback")
            if callback:
                await callback("thinking", {"detail": "Thinking about the task..."})
                await callback("step", {"step": 1, "tool": "run_browser_harness", "label": "Executing browser code", "status": "running"})
                await callback("step_done", {"step": 1, "tool": "run_browser_harness", "label": "Executing browser code", "success": True})
            return FakeResult()

        mock_loop.side_effect = fake_loop

        response = client.post("/agent/run/stream", json={
            "task": "Draw rocket on excalidraw",
            "taskId": "stream-test-1",
            "tabId": "tab-1",
            "url": "https://excalidraw.com",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = response.text
        assert "event: perception" in text
        assert "event: thinking" in text
        assert "event: step" in text
        assert "event: step_done" in text
        assert "event: done" in text
        assert "Draw rocket" in text or "excalidraw" in text


def test_stream_regression_test_a_default_foreground():
    """Test A: Mandatory regression test - default foreground streaming with tabId='active_tab'.
    Must fail against the broken implementation and pass after the correction."""
    fake_fg_ws = "ws://127.0.0.1:9222/devtools/page/active-fg-page"
    with patch("agent_service._resolve_verified_tab_ws", return_value=("active_tab_target", fake_fg_ws)), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
        class FakeResult:
            def __str__(self):
                return "Foreground streaming task completed successfully."
            metadata = {"is_conversational": False}

        mock_loop.return_value = FakeResult()

        response = client.post("/agent/run/stream", json={
            "task": "Perform foreground browser task",
            "taskId": "stream-fg-1",
            "tabId": "active_tab",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = response.text

        # Verify no UnboundLocalError occurred
        assert "cannot access free variable" not in text
        assert "UnboundLocalError" not in text

        # Verify agent loop was reached with valid endpoint
        mock_loop.assert_called_once()
        call_args = mock_loop.call_args
        passed_cdp_url = call_args[0][1]
        assert passed_cdp_url == fake_fg_ws or (passed_cdp_url and passed_cdp_url.startswith("http"))

        # Verify stream completed with success
        assert "Foreground streaming task completed successfully." in text
        assert '"success": true' in text or '"success":true' in text.lower()


def test_stream_test_b_foreground_alias_coverage():
    """Test B: Verify all supported foreground aliases and missing/empty tabId establish valid target."""
    from nervous.action_policy import PROHIBITED_FOREGROUND_ALIASES

    test_aliases = list(PROHIBITED_FOREGROUND_ALIASES) + [""]
    fake_fg_ws = "ws://127.0.0.1:9222/devtools/page/fg-alias-page"

    for alias in test_aliases:
        with patch("agent_service._resolve_verified_tab_ws", return_value=("fg_target", fake_fg_ws)), \
             patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
            class FakeResult:
                def __str__(self):
                    return f"Completed for alias {alias}"
                metadata = {}

            mock_loop.return_value = FakeResult()

            payload = {
                "task": f"Test alias {alias}",
                "taskId": f"stream-alias-{alias or 'empty'}",
                "tabId": alias,
                "contextToken": agent_service._INTERNAL_APP_TOKEN
            }

            response = client.post("/agent/run/stream", json=payload)
            assert response.status_code == 200
            text = response.text

            assert "cannot access free variable" not in text
            assert "UnboundLocalError" not in text
            mock_loop.assert_called_once()
            passed_cdp_url = mock_loop.call_args[0][1]
            assert passed_cdp_url == fake_fg_ws or (passed_cdp_url and passed_cdp_url.startswith("http"))
            assert "Completed for alias" in text


def test_stream_test_c_untrusted_foreground_request():
    """Test C: Untrusted foreground requests must fail closed and never invoke execute_agent_loop."""
    with patch("agent_service._resolve_verified_tab_ws", return_value=("fg_target", "ws://127.0.0.1:9222/page/1")), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:

        response = client.post("/agent/run/stream", json={
            "task": "Untrusted request to foreground",
            "taskId": "stream-untrusted-1",
            "tabId": "active_tab",
            "contextToken": "invalid_or_missing_token"
        })

        assert response.status_code == 200
        text = response.text
        assert any(term in text for term in ["Unauthenticated request", "Untrusted access", "unauthenticated", "blocked"])
        mock_loop.assert_not_called()


def test_stream_test_d_explicit_tab_unresolved_fails_closed():
    """Test D: Explicit tab that cannot be resolved fails closed; default endpoint is never used."""
    with patch("agent_service._resolve_verified_tab_ws", return_value=(None, None)), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:

        response = client.post("/agent/run/stream", json={
            "task": "Interact with nonexistent tab",
            "taskId": "stream-explicit-unresolved",
            "tabId": "tab-nonexistent-999",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "Target resolution failed: tabId 'tab-nonexistent-999' not found." in text
        assert "Implicit foreground fallback is prohibited." in text
        mock_loop.assert_not_called()


def test_stream_test_e_canonical_target_mismatch_fails_closed():
    """Test E: Canonical target mismatch fails closed and execute_agent_loop is never called."""
    from nervous.action_policy import get_lease_registry
    registry = get_lease_registry()
    lease = registry.acquire_lease("worker_mismatch", "tab_mismatch", canonical_target_id="TARGET_CORRECT", trusted_issuer=True)

    with patch("agent_service._resolve_verified_tab_ws", return_value=("TARGET_WRONG", "ws://127.0.0.1:9222/devtools/page/TARGET_WRONG")), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:

        response = client.post("/agent/run/stream", json={
            "task": "Task with target mismatch",
            "taskId": "stream-canon-mismatch",
            "tabId": "tab_mismatch",
            "workerId": "worker_mismatch",
            "workerToken": lease.lease_token,
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "Canonical target mismatch" in text
        mock_loop.assert_not_called()


def test_stream_test_f_worker_capability_enforcement():
    """Test F: Worker asserting identity without valid lease token remains unauthorized, even with trusted token."""
    from nervous.action_policy import get_lease_registry
    registry = get_lease_registry()
    lease = registry.acquire_lease("worker_victim", "tab_victim", trusted_issuer=True)

    with patch("agent_service._resolve_verified_tab_ws", return_value=("target_victim", "ws://127.0.0.1:9222/page/victim")), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:

        response = client.post("/agent/run/stream", json={
            "task": "Impersonate worker without lease token",
            "taskId": "stream-worker-impersonate",
            "tabId": "tab_victim",
            "workerId": "worker_victim",
            "workerToken": "wrong_token_forged",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "Worker authorization failed" in text
        mock_loop.assert_not_called()


def test_stream_test_g_existing_explicit_tab_workflow():
    """Test G: Existing explicit tab streaming workflow succeeds as expected."""
    with patch("agent_service._resolve_verified_tab_ws", return_value=("target_explicit", "ws://127.0.0.1:9222/devtools/page/target_explicit")), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
        class FakeResult:
            def __str__(self):
                return "Explicit tab streaming finished."
            metadata = {}

        mock_loop.return_value = FakeResult()

        response = client.post("/agent/run/stream", json={
            "task": "Explicit tab task",
            "taskId": "stream-explicit-ok",
            "tabId": "tab-explicit-123",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        mock_loop.assert_called_once()
        assert "Explicit tab streaming finished." in text


def test_stream_regression_conversational_response():
    """Regression Test 1: Normal conversational message returns visible assistant response."""
    fake_fg_ws = "ws://127.0.0.1:9222/devtools/page/active-fg-page"
    with patch("agent_service._resolve_verified_tab_ws", return_value=("active_target", fake_fg_ws)), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
        class FakeChatResult:
            def __str__(self):
                return "COMEX_STREAM_TEST_OK"
            metadata = {"is_conversational": True, "tool_call_count": 0}

        async def fake_chat_loop(task, default_cdp_url, **kwargs):
            callback = kwargs.get("event_callback")
            if callback:
                await callback("completed", {
                    "type": "chat",
                    "is_conversational": True,
                    "content": "COMEX_STREAM_TEST_OK",
                    "text": "COMEX_STREAM_TEST_OK",
                    "result": "COMEX_STREAM_TEST_OK",
                    "done": True,
                })
            return FakeChatResult()

        mock_loop.side_effect = fake_chat_loop

        response = client.post("/agent/run/stream", json={
            "task": "Reply with exactly COMEX_STREAM_TEST_OK",
            "taskId": "stream-chat-1",
            "tabId": "active_tab",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "event: completed" in text
        assert "event: done" in text
        assert "COMEX_STREAM_TEST_OK" in text
        assert '"is_conversational": true' in text.lower()
        mock_loop.assert_called_once()


def test_stream_regression_extract_page_title():
    """Regression Test 2: Read-only task extracts page title over active_tab."""
    fake_fg_ws = "ws://127.0.0.1:9222/devtools/page/active-fg-page"
    with patch("agent_service._resolve_verified_tab_ws", return_value=("active_target", fake_fg_ws)), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
        class FakeExtractResult:
            def __str__(self):
                return "**Title:** Example Domain"
            metadata = {"is_conversational": False, "tool_call_count": 1}

        async def fake_extract_loop(task, default_cdp_url, **kwargs):
            callback = kwargs.get("event_callback")
            if callback:
                await callback("step", {"step": 1, "tool": "browser_extract", "label": "Extracting content", "status": "running"})
                await callback("step_done", {"step": 1, "tool": "browser_extract", "label": "Extracting content", "success": True})
                await callback("completed", {
                    "type": "result",
                    "is_conversational": False,
                    "content": "**Title:** Example Domain",
                    "done": True,
                })
            return FakeExtractResult()

        mock_loop.side_effect = fake_extract_loop

        response = client.post("/agent/run/stream", json={
            "task": "Tell me the title of the currently open page.",
            "taskId": "stream-title-1",
            "tabId": "active_tab",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "event: step" in text
        assert "browser_extract" in text
        assert "**Title:** Example Domain" in text
        mock_loop.assert_called_once()
        passed_cdp_url = mock_loop.call_args[0][1]
        assert passed_cdp_url == fake_fg_ws


def test_stream_regression_navigate_youtube_task():
    """Regression Test 3: Browser task navigates to YouTube search results."""
    fake_fg_ws = "ws://127.0.0.1:9222/devtools/page/active-fg-page"
    with patch("agent_service._resolve_verified_tab_ws", return_value=("active_target", fake_fg_ws)), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:
        class FakeNavResult:
            def __str__(self):
                return "YouTube search executed for Lo-Fi Beats."
            metadata = {"is_conversational": False, "tool_call_count": 2}

        async def fake_nav_loop(task, default_cdp_url, **kwargs):
            callback = kwargs.get("event_callback")
            if callback:
                await callback("step", {"step": 1, "tool": "browser_navigate", "label": "Navigating to YouTube", "status": "running"})
                await callback("step_done", {"step": 1, "tool": "browser_navigate", "label": "Navigating to YouTube", "success": True})
                await callback("step", {"step": 2, "tool": "browser_extract", "label": "Extracting results", "status": "running"})
                await callback("step_done", {"step": 2, "tool": "browser_extract", "label": "Extracting results", "success": True})
            return FakeNavResult()

        mock_loop.side_effect = fake_nav_loop

        response = client.post("/agent/run/stream", json={
            "task": "Open YouTube and search for Lo-Fi Beats",
            "taskId": "stream-yt-1",
            "tabId": "active_tab",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "browser_navigate" in text
        assert "browser_extract" in text
        assert "YouTube search executed for Lo-Fi Beats." in text
        mock_loop.assert_called_once()
        passed_cdp_url = mock_loop.call_args[0][1]
        assert passed_cdp_url == fake_fg_ws


def test_stream_regression_error_payload_emits_failure():
    """Regression Test 4: Target resolution error emits success=False and explicit error payload, never masking as Task completed."""
    with patch("agent_service._resolve_verified_tab_ws", return_value=(None, None)), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:

        response = client.post("/agent/run/stream", json={
            "task": "Task with unresolved explicit tab",
            "taskId": "stream-err-payload-1",
            "tabId": "unregistered-tab-404",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert response.status_code == 200
        text = response.text
        assert "event: done" in text
        assert '"success": false' in text.lower() or '"success":false' in text.lower()
        assert "Target resolution failed: tabId 'unregistered-tab-404' not found." in text
        assert "Task completed." not in text
        mock_loop.assert_not_called()

