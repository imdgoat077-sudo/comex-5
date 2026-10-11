"""
Tests for services/tab_provisioning.py: visible agent tabs opened through the Comex UI.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services import tab_provisioning
from services.tab_provisioning import is_trusted_ui_target_url

UI_TARGET = {
    "type": "page",
    "id": "ui-target-id",
    "url": "http://localhost:5173/",
    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/ui-target-id",
}
AGENT_TARGET = {
    "type": "page",
    "id": "agent-target-id",
    "url": "https://news.ycombinator.com/",
    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/agent-target-id",
}


def _fake_ws(eval_value=None, exception=None):
    """Mock websocket connection whose Runtime.evaluate reply returns eval_value (or an exception)."""
    ws = MagicMock()
    ws.send = AsyncMock()
    result = {"exceptionDetails": {"text": exception}} if exception else {"result": {"value": eval_value}}
    ws.recv = AsyncMock(return_value=json.dumps({"id": 1, "result": result}))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, ws


def _sent_expression(ws) -> str:
    return json.loads(ws.send.call_args[0][0])["params"]["expression"]


async def test_open_agent_tab_returns_exact_target_and_escapes_arguments():
    cm, ws = _fake_ws(eval_value="agent-target-id")
    with patch.object(tab_provisioning, "_list_cdp_targets", return_value=[UI_TARGET, AGENT_TARGET]), \
         patch.object(tab_provisioning.websockets, "connect", return_value=cm):
        target_id, ws_url = await tab_provisioning.open_agent_tab(
            "tab_hn_1", "https://news.ycombinator.com/", 'Find "top" story', "http://127.0.0.1:9222"
        )

    assert (target_id, ws_url) == ("agent-target-id", AGENT_TARGET["webSocketDebuggerUrl"])
    expr = _sent_expression(ws)
    assert expr.startswith("window.nexusNative.tabs.openAgentTab(")
    # Arguments are JSON-encoded, never raw string interpolation
    assert json.dumps('Find "top" story') in expr
    assert json.dumps("https://news.ycombinator.com/") in expr


async def test_open_agent_tab_defaults_to_real_http_start_page():
    cm, ws = _fake_ws(eval_value="agent-target-id")
    with patch.object(tab_provisioning, "_list_cdp_targets", return_value=[UI_TARGET, AGENT_TARGET]), \
         patch.object(tab_provisioning.websockets, "connect", return_value=cm):
        await tab_provisioning.open_agent_tab("tab_x", None, "label", "http://127.0.0.1:9222")
    assert json.dumps(tab_provisioning.DEFAULT_START_URL) in _sent_expression(ws)


async def test_open_agent_tab_raises_when_ui_missing():
    with patch.object(tab_provisioning, "_list_cdp_targets", return_value=[AGENT_TARGET]):
        with pytest.raises(RuntimeError, match="Comex UI"):
            await tab_provisioning.open_agent_tab("tab_x", None, "", "http://127.0.0.1:9222")


async def test_open_agent_tab_raises_on_ui_exception():
    cm, _ws = _fake_ws(exception="Tab tab_x already exists")
    with patch.object(tab_provisioning, "_list_cdp_targets", return_value=[UI_TARGET]), \
         patch.object(tab_provisioning.websockets, "connect", return_value=cm):
        with pytest.raises(RuntimeError, match="already exists"):
            await tab_provisioning.open_agent_tab("tab_x", None, "", "http://127.0.0.1:9222")


async def test_open_agent_tab_raises_when_target_never_appears():
    cm, _ws = _fake_ws(eval_value="missing-target")
    with patch.object(tab_provisioning, "_list_cdp_targets", return_value=[UI_TARGET]), \
         patch.object(tab_provisioning.websockets, "connect", return_value=cm), \
         patch.object(tab_provisioning, "DISCOVERY_POLL_INTERVAL_S", 0.0):
        with pytest.raises(RuntimeError, match="did not appear"):
            await tab_provisioning.open_agent_tab("tab_x", None, "", "http://127.0.0.1:9222")


async def test_set_agent_tab_status_ignores_unknown_status_and_never_raises():
    with patch.object(tab_provisioning, "_list_cdp_targets", side_effect=AssertionError("must not be called")):
        await tab_provisioning.set_agent_tab_status("tab_x", "bogus", "http://127.0.0.1:9222")
    with patch.object(tab_provisioning, "_list_cdp_targets", side_effect=OSError("CDP down")):
        await tab_provisioning.set_agent_tab_status("tab_x", "completed", "http://127.0.0.1:9222")


def test_ui_target_validator_rejects_websites_with_index_html():
    assert is_trusted_ui_target_url("http://localhost:5173/")
    assert not is_trusted_ui_target_url("https://example.com/index.html")
    assert not is_trusted_ui_target_url("https://www.google.com/")
