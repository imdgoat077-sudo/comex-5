"""
Visible browser tabs for multi-agent orchestrator workers.

Each worker gets a normal, visible Comex tab (it appears in the tab bar and the user can watch
the agent work in it). The backend asks the React UI's own WebContents, over CDP, to call
window.nexusNative.tabs.openAgentTab(); Electron's main process creates the tab, adds it to the
tab bar, and returns the tab's exact Chromium target id, which we resolve to its CDP WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

import websockets

logger = logging.getLogger("comex.services.tab_provisioning")

DISCOVERY_POLL_ATTEMPTS = 10
DISCOVERY_POLL_INTERVAL_S = 0.2
DEFAULT_START_URL = "https://www.google.com/"
AGENT_TAB_STATUSES = frozenset({"working", "completed", "failed", "cancelled", "blocked"})


def is_trusted_ui_target_url(url: str) -> bool:
    """Strict origin/path validator for the Comex React UI CDP target (never loose substring matching)."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            return parsed.netloc in ("localhost:5173", "127.0.0.1:5173") and parsed.path in ("", "/", "/index.html")
        if parsed.scheme == "file":
            norm = os.path.normpath(parsed.path)
            return norm.endswith(os.path.join("dist", "index.html")) or (
                norm.endswith("index.html") and "dist" in norm.split(os.sep)
            )
    except Exception:
        return False
    return False


def _list_cdp_targets(cdp_base_url: str) -> list[dict]:
    with urllib.request.urlopen(f"{cdp_base_url}/json/list", timeout=3) as resp:
        return json.loads(resp.read().decode())


def _find_ui_target_ws(cdp_base_url: str) -> Optional[str]:
    for target in _list_cdp_targets(cdp_base_url):
        if target.get("type") == "page" and is_trusted_ui_target_url(target.get("url", "")):
            return target.get("webSocketDebuggerUrl")
    return None


async def _evaluate_in_ui(ui_ws_url: str, expression: str) -> Any:
    """Evaluate an expression in the React UI (awaiting promises) and return its value."""
    async with websockets.connect(ui_ws_url) as ws:
        await ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "awaitPromise": True, "returnByValue": True},
        }))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1:
                break
    result = msg.get("result", {})
    if "exceptionDetails" in result:
        details = result["exceptionDetails"]
        text = (details.get("exception") or {}).get("description") or details.get("text") or "unknown error"
        raise RuntimeError(f"UI evaluation failed: {text}")
    return (result.get("result") or {}).get("value")


async def open_agent_tab(tab_id: str, url: Optional[str], label: str, cdp_base_url: str) -> Tuple[str, str]:
    """
    Open a visible tab for a worker and return (canonical_target_id, websocket_debugger_url).
    Raises RuntimeError if the UI or the new tab's CDP target cannot be found.
    """
    ui_ws = _find_ui_target_ws(cdp_base_url)
    if not ui_ws:
        raise RuntimeError("Could not find the Comex UI's CDP target to open an agent tab.")

    expression = "window.nexusNative.tabs.openAgentTab({}, {}, {})".format(
        json.dumps(tab_id), json.dumps(url or DEFAULT_START_URL), json.dumps(label or "")
    )
    target_id = await _evaluate_in_ui(ui_ws, expression)
    if not target_id or not isinstance(target_id, str):
        raise RuntimeError(f"Electron did not return a target id for agent tab '{tab_id}'.")

    for _ in range(DISCOVERY_POLL_ATTEMPTS):
        for target in _list_cdp_targets(cdp_base_url):
            if target.get("id") == target_id and target.get("webSocketDebuggerUrl"):
                logger.info(f"Opened agent tab '{tab_id}' -> {target_id}")
                return target_id, target["webSocketDebuggerUrl"]
        await asyncio.sleep(DISCOVERY_POLL_INTERVAL_S)

    raise RuntimeError(f"Agent tab '{tab_id}' (target {target_id}) did not appear in the CDP target list.")


async def set_agent_tab_status(tab_id: str, status: str, cdp_base_url: str) -> None:
    """Update the tab bar's agent indicator for a worker tab. Best-effort; never raises."""
    if status not in AGENT_TAB_STATUSES:
        return
    try:
        ui_ws = _find_ui_target_ws(cdp_base_url)
        if not ui_ws:
            return
        expression = "window.nexusNative.tabs.setAgentStatus({}, {})".format(json.dumps(tab_id), json.dumps(status))
        await _evaluate_in_ui(ui_ws, expression)
    except Exception as e:
        logger.warning(f"Could not update agent status for tab '{tab_id}': {e}")
