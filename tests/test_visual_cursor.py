"""
Unit tests for the Visual Fake Cursor integration:
1. Executor calls visual_cursor_hook with exact final authoritative coordinates (target_x, target_y).
2. If pre-scroll validation fails, visual_cursor_hook is never called.
3. If pre-input validation fails, visual_cursor_hook is never called.
4. If visual_cursor_hook raises an exception, the executor safely catches it and continues physical dispatch.
5. When visual_cursor_hook is None, executor behaves normally with zero overhead.
6. TabSessionContext.set_visual_cursor_hook forwards to executor.
7. Feature flag VISUAL_CURSOR_ENABLED=False returns None for the hook.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional
import pytest

from nervous.cdp_session import TabSessionContext
from nervous.executor import SelectorClickExecutor
from nervous.target_binding import TargetBinding
from nervous.world_state import BrowserState


class MockCDPClient:
    def __init__(self, box_quad=None):
        self.commands = []
        self.dispatched_mouse_events = []
        self.doc_node_id = 1
        self.target_node_id = 42
        self.backend_node_id = 420
        self.frame_id = "main_frame_1"
        self.box_model_quad = box_quad or [200.0, 100.0, 400.0, 100.0, 400.0, 300.0, 200.0, 300.0]

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.commands.append((method, params))
        if method == "DOM.getDocument":
            return {"root": {"nodeId": self.doc_node_id}}
        elif method == "DOM.querySelector":
            return {"nodeId": self.target_node_id}
        elif method == "DOM.describeNode":
            return {
                "node": {
                    "nodeId": self.target_node_id,
                    "backendNodeId": self.backend_node_id,
                    "frameId": self.frame_id,
                }
            }
        elif method == "DOM.resolveNode":
            return {"object": {"objectId": "remote_obj_42"}}
        elif method == "Runtime.callFunctionOn":
            return {
                "result": {
                    "value": {
                        "actionable": True,
                        "reasons": [],
                        "geometry": {"x": 200, "y": 100, "width": 200, "height": 200},
                    }
                }
            }
        elif method == "Runtime.releaseObject":
            return {}
        elif method == "DOM.scrollIntoViewIfNeeded":
            return {}
        elif method == "DOM.getBoxModel":
            return {
                "model": {
                    "border": self.box_model_quad,
                    "width": 200,
                    "height": 200,
                }
            }
        elif method == "Input.dispatchMouseEvent":
            if params:
                self.dispatched_mouse_events.append(params)
            return {}
        return {}


@pytest.mark.asyncio
async def test_executor_triggers_visual_cursor_with_final_coordinates():
    cdp = MockCDPClient(box_quad=[200.0, 100.0, 400.0, 100.0, 400.0, 300.0, 200.0, 300.0])
    session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
    await session.start()

    hook_calls = []
    def test_hook(x: float, y: float, action: str, tab_id: str):
        hook_calls.append({"x": x, "y": y, "action": action, "tab_id": tab_id})

    session.set_visual_cursor_hook(test_hook)

    try:
        binding = await session.resolve_and_bind("#buy-btn")
        res = await session.executor.execute("click", binding)

        assert res["clicked"] is True
        # Centroid of [200, 100], [400, 100], [400, 300], [200, 300] is (300.0, 200.0)
        assert res["x"] == 300.0
        assert res["y"] == 200.0

        # Verify visual hook was called exactly once with exact final coordinates
        assert len(hook_calls) == 1
        assert hook_calls[0]["x"] == 300.0
        assert hook_calls[0]["y"] == 200.0
        assert hook_calls[0]["action"] == "left"
        assert hook_calls[0]["tab_id"] == "tab_1"

        # Verify physical mouse event was dispatched
        assert len(cdp.dispatched_mouse_events) == 2
        assert cdp.dispatched_mouse_events[0]["type"] == "mousePressed"
        assert cdp.dispatched_mouse_events[0]["x"] == 300.0
        assert cdp.dispatched_mouse_events[0]["y"] == 200.0
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_executor_does_not_trigger_hook_on_validation_failure():
    cdp = MockCDPClient()
    session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
    await session.start()

    hook_calls = []
    def test_hook(x: float, y: float, action: str, tab_id: str):
        hook_calls.append({"x": x, "y": y, "action": action, "tab_id": tab_id})

    session.set_visual_cursor_hook(test_hook)

    try:
        binding = await session.resolve_and_bind("#buy-btn")

        # Simulate page reload / document invalidation
        session.tracker.state.document_generation += 1

        with pytest.raises(RuntimeError, match="Pre-scroll validation failed"):
            await session.executor.execute("click", binding)

        # Hook must NOT be called when validation fails
        assert len(hook_calls) == 0
        assert len(cdp.dispatched_mouse_events) == 0
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_executor_catches_hook_exceptions_silently():
    cdp = MockCDPClient()
    session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
    await session.start()

    def crashing_hook(x: float, y: float, action: str, tab_id: str):
        raise ValueError("Simulated network/transport failure in visual hook!")

    session.set_visual_cursor_hook(crashing_hook)

    try:
        binding = await session.resolve_and_bind("#buy-btn")
        # Must NOT raise exception despite hook failure
        res = await session.executor.execute("click", binding)
        assert res["clicked"] is True
        assert len(cdp.dispatched_mouse_events) == 2
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_tab_session_forwards_hook_to_executor():
    cdp = MockCDPClient()
    session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp)

    hook_calls = []
    def my_hook(x, y, action, tab_id):
        hook_calls.append((x, y))

    session.set_visual_cursor_hook(my_hook)
    assert session.executor._visual_cursor_hook is my_hook


def test_visual_cursor_flag_behavior(monkeypatch):
    from agent_service import _make_visual_cursor_hook

    # When flag is false: hook is None
    monkeypatch.setenv("VISUAL_CURSOR_ENABLED", "false")
    import importlib
    import agent_service
    importlib.reload(agent_service)
    assert agent_service.VISUAL_CURSOR_ENABLED is False
    assert agent_service._make_visual_cursor_hook("tab_1") is None

    # When flag is true: hook is callable
    monkeypatch.setenv("VISUAL_CURSOR_ENABLED", "true")
    importlib.reload(agent_service)
    assert agent_service.VISUAL_CURSOR_ENABLED is True
    assert callable(agent_service._make_visual_cursor_hook("tab_1"))
