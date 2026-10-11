"""
Stage C Milestone 1 Verification Suite: tests/test_milestone1_click.py

Comprehensive tests for the Controlled Mutation Plane and browser_click(selector):
1. Coherent object resolution (Method E on bound node)
2. Ordered event-processing barrier (wait_until_processed)
3. Happy path execution (exactly 1 logical click, 2 physical mouse events)
4. Post-scroll subtree mutation abort (zero physical mouse events)
5. Document generation reload invalidation (zero physical mouse events)
6. Frame safety rejection (iframe targets safely rejected)
7. Session/tab mismatch abort
8. Async ActuatorGate compatibility and zero-divergence with sync attempt
9. Latency segments A, B, C recording
10. Node detachment / identity mismatch fail-closed
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from nervous.actuator_gate import ActuatorGate, DecisionType, GateResult, LatencyTracker
from nervous.cdp_session import TabSessionContext
from nervous.executor import SelectorClickExecutor
from nervous.reflex_engine import ReflexEngine
from nervous.target_binding import BindingFailureReason, TargetBinding, validate_binding_integrity
from nervous.world_state import BrowserState, TargetState, WorldStateTracker


class MockCDPClient:
    """Mock CDP client for simulating Chromium CDP responses and recording dispatches."""

    def __init__(self) -> None:
        self.commands: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self.dispatched_mouse_events: List[Dict[str, Any]] = []
        self.doc_node_id: int = 1
        self.target_node_id: int = 42
        self.backend_node_id: int = 420
        self.frame_id: str = "main_frame_1"
        self.is_method_e_actionable: bool = True
        self.box_model_quad = [100.0, 100.0, 200.0, 100.0, 200.0, 150.0, 100.0, 150.0]
        self.on_scroll_hook: Optional[Any] = None

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.commands.append((method, params))

        if method == "DOM.getDocument":
            return {"root": {"nodeId": self.doc_node_id}}
        elif method == "DOM.querySelector":
            sel = params.get("selector") if params else ""
            if sel == "#missing":
                return {"nodeId": 0}
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
            return {"object": {"objectId": f"remote_obj_{self.target_node_id}"}}
        elif method == "Runtime.callFunctionOn":
            return {
                "result": {
                    "value": {
                        "actionable": self.is_method_e_actionable,
                        "box": {"x": 100.0, "y": 100.0, "width": 100.0, "height": 50.0},
                    }
                }
            }
        elif method == "Runtime.releaseObject":
            return {}
        elif method == "DOM.scrollIntoViewIfNeeded":
            if self.on_scroll_hook:
                await self.on_scroll_hook()
            return {}
        elif method == "DOM.getBoxModel":
            return {
                "model": {
                    "border": self.box_model_quad,
                    "width": 100,
                    "height": 50,
                }
            }
        elif method == "Input.dispatchMouseEvent":
            if params:
                self.dispatched_mouse_events.append(params)
            return {}

        return {}


# ══════════════════════════════════════════════════════════════════════
#  1. Coherent Object Resolution
# ══════════════════════════════════════════════════════════════════════

class TestCoherentResolution:

    @pytest.mark.asyncio
    async def test_coherent_binding_on_same_node_handle(self):
        """Method E and backendNodeId are resolved and bound from the same node handle."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            binding = await session.resolve_and_bind("#buy-btn")
            assert binding.selector == "#buy-btn"
            assert binding.node_id == 42
            assert binding.backend_node_id == 420
            assert binding.frame_id == "main_frame_1"
            assert binding.bounding_box == {"x": 100.0, "y": 100.0, "width": 100.0, "height": 50.0}

            # Verify CDP calls sequence
            methods = [c[0] for c in cdp.commands]
            assert "DOM.getDocument" in methods
            assert "DOM.querySelector" in methods
            assert "DOM.describeNode" in methods
            assert "DOM.resolveNode" in methods
            assert "Runtime.callFunctionOn" in methods

            # Verify tracker was updated
            assert session.tracker.state.current_target is not None
            assert session.tracker.state.current_target.is_actionable is True
            assert session.tracker.state.current_target.backend_node_id == 420
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  2. Ordered Event-Processing Barrier
# ══════════════════════════════════════════════════════════════════════

class TestEventProcessingBarrier:

    @pytest.mark.asyncio
    async def test_wait_until_processed_drains_queued_events(self):
        """Barrier ensures events enqueued before the watermark are applied to WorldStateTracker."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp)
        await session.start()

        try:
            # Enqueue 3 events
            seq1 = session.on_cdp_notification("Network.requestWillBeSent", {"requestId": "req_1", "type": "Fetch"})
            seq2 = session.on_cdp_notification("DOM.attributeModified", {"nodeId": 10, "name": "class", "value": "active"})
            seq3 = session.on_cdp_notification("Network.loadingFinished", {"requestId": "req_1"})

            # Wait on the barrier for seq3
            await session.wait_until_processed(seq3)

            assert session._processed_watermark >= seq3
            assert session.tracker.state.dom_revision == 1
            assert session.tracker.state.network_in_flight == 0
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  3. Happy Path: Main-Frame Selector Click
# ══════════════════════════════════════════════════════════════════════

class TestHappyPathClick:

    @pytest.mark.asyncio
    async def test_successful_click_exact_once(self):
        """Stable element -> ALLOW -> exactly 1 logical executor call -> mousePressed + mouseReleased."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            binding = await session.resolve_and_bind("#btn")
            # State is made fully quiet/stable for ALLOW
            session.tracker.state.tracking_started_at_mono = time.monotonic() - 0.200
            session.tracker.state.last_dom_mutation_at_mono = time.monotonic() - 0.200
            session.tracker.state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

            gate_result = await session.gate.async_attempt("click", binding=binding)

            assert gate_result.executed is True
            assert gate_result.decision.decision == DecisionType.ALLOW
            assert gate_result.eval_to_dispatch_ns > 0

            # Verify physical mouse events: exactly 2 CDP events (1 press, 1 release)
            assert len(cdp.dispatched_mouse_events) == 2
            press = cdp.dispatched_mouse_events[0]
            release = cdp.dispatched_mouse_events[1]

            assert press["type"] == "mousePressed"
            assert release["type"] == "mouseReleased"
            assert press["x"] == 150.0  # Center of [100, 200]
            assert press["y"] == 125.0  # Center of [100, 150]
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  4. Post-Scroll Subtree Mutation Abort
# ══════════════════════════════════════════════════════════════════════

class TestPostScrollMutationAbort:

    @pytest.mark.asyncio
    async def test_mutation_after_scroll_aborts_dispatch(self):
        """Subtree mutation injected after scroll causes post-scroll re-validation failure with 0 mouse events."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            binding = await session.resolve_and_bind("#churn-btn")
            session.tracker.state.tracking_started_at_mono = time.monotonic() - 0.200
            session.tracker.state.last_dom_mutation_at_mono = time.monotonic() - 0.200
            session.tracker.state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

            # Hook scroll to mutate target subtree
            async def inject_mutation():
                session.tracker.state.target_subtree_mutation_generation += 1

            cdp.on_scroll_hook = inject_mutation

            with pytest.raises(RuntimeError, match="Post-scroll validation failed: SUBTREE_MUTATION_AFTER_BINDING"):
                await session.gate.async_attempt("click", binding=binding)

            # Zero physical mouse events dispatched
            assert len(cdp.dispatched_mouse_events) == 0
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  5. Document Generation Reload Invalidation
# ══════════════════════════════════════════════════════════════════════

class TestDocumentReloadInvalidation:

    @pytest.mark.asyncio
    async def test_document_updated_aborts_execution(self):
        """DOM.documentUpdated after binding causes DOCUMENT_GENERATION_MISMATCH with 0 mouse events."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            binding = await session.resolve_and_bind("#btn")
            session.tracker.state.tracking_started_at_mono = time.monotonic() - 0.200
            session.tracker.state.last_dom_mutation_at_mono = time.monotonic() - 0.200
            session.tracker.state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

            # Simulate document reload
            session.tracker.handle_document_updated()

            reason = validate_binding_integrity(binding, session.tracker.state)
            assert reason == BindingFailureReason.DOCUMENT_GENERATION_MISMATCH

            # Gate evaluate
            gate_res = await session.gate.async_attempt("click", binding=binding)
            assert gate_res.executed is False
            assert gate_res.decision.decision.name == "INVALIDATE"
            assert len(cdp.dispatched_mouse_events) == 0
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  6. Frame Safety Rejection
# ══════════════════════════════════════════════════════════════════════

class TestFrameSafety:

    @pytest.mark.asyncio
    async def test_cross_frame_target_rejected(self):
        """Target in an iframe is rejected before binding in Milestone 1."""
        cdp = MockCDPClient()
        cdp.frame_id = "iframe_sub_99"
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            with pytest.raises(RuntimeError, match="Cross-frame targets are explicitly out of scope for Milestone 1"):
                await session.resolve_and_bind("#iframe-input")

            assert len(cdp.dispatched_mouse_events) == 0
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  7. Session Target Mismatch Abort
# ══════════════════════════════════════════════════════════════════════

class TestSessionMismatch:

    @pytest.mark.asyncio
    async def test_session_mismatch_aborts(self):
        """If tab/target context changes before execution, execution aborts."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_A", target_id="target_A", cdp_client=cdp)
        await session.start()

        try:
            binding = await session.resolve_and_bind("#btn")
            reason = validate_binding_integrity(binding, session.tracker.state, current_tab_id="tab_B")
            assert reason == BindingFailureReason.SESSION_TARGET_MISMATCH
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  8. Async ActuatorGate & Backward Compatibility
# ══════════════════════════════════════════════════════════════════════

class TestAsyncActuatorGate:

    def test_sync_attempt_still_works_unmodified(self):
        """Existing synchronous attempt() remains 100% backward compatible."""
        state = BrowserState()
        state.tracking_started_at_mono = time.monotonic() - 0.200
        state.last_dom_mutation_at_mono = time.monotonic() - 0.200
        target = state.assign_target(locator="#btn", node_id=10, is_actionable=True, is_stable=True)
        target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

        mock_exec = MagicMock()
        mock_exec.execute.return_value = "sync_ok"

        gate = ActuatorGate(state, mock_exec)
        res = gate.attempt("click")
        assert res.executed is True
        assert res.execution_result == "sync_ok"
        mock_exec.execute.assert_called_once_with("click")

    @pytest.mark.asyncio
    async def test_async_attempt_with_async_executor(self):
        """async_attempt awaits async executor and records latency."""
        state = BrowserState()
        state.tracking_started_at_mono = time.monotonic() - 0.200
        state.last_dom_mutation_at_mono = time.monotonic() - 0.200
        target = state.assign_target(locator="#btn", node_id=10, is_actionable=True, is_stable=True)
        target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

        mock_async_exec = MagicMock()
        mock_async_exec.execute = AsyncMock(return_value={"validation_to_dispatch_ns": 4500})

        gate = ActuatorGate(state, mock_async_exec)
        res = await gate.async_attempt("click")

        assert res.executed is True
        assert res.validation_to_dispatch_ns == 4500
        assert res.eval_to_dispatch_ns > 0
        mock_async_exec.execute.assert_awaited_once_with("click")


# ══════════════════════════════════════════════════════════════════════
#  9. Latency Segment Recording
# ══════════════════════════════════════════════════════════════════════

class TestLatencySegments:

    @pytest.mark.asyncio
    async def test_latency_segments_recorded(self):
        """Segment A (eval to dispatch) and Segment B (validation to dispatch) are populated."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            binding = await session.resolve_and_bind("#btn")
            session.tracker.state.tracking_started_at_mono = time.monotonic() - 0.200
            session.tracker.state.last_dom_mutation_at_mono = time.monotonic() - 0.200
            session.tracker.state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

            gate_result = await session.gate.async_attempt("click", binding=binding)

            assert gate_result.eval_to_dispatch_ns > 0  # Segment A
            assert gate_result.validation_to_dispatch_ns >= 0  # Segment B

            percentiles = session.gate.latency.percentiles()
            assert "p50_us" in percentiles
            assert "p95_us" in percentiles
            assert "p99_us" in percentiles
        finally:
            await session.stop()


# ══════════════════════════════════════════════════════════════════════
#  10. Node Detachment / Identity Mismatch Fail-Closed
# ══════════════════════════════════════════════════════════════════════

class TestNodeDetachment:

    @pytest.mark.asyncio
    async def test_node_identity_mismatch_fails_closed(self):
        """If node identity changes after binding, validation fails closed with 0 dispatches."""
        cdp = MockCDPClient()
        session = TabSessionContext(tab_id="tab_1", target_id="target_1", cdp_client=cdp, main_frame_id="main_frame_1")
        await session.start()

        try:
            binding = await session.resolve_and_bind("#btn")
            # Replace target with new node ID
            session.tracker.state.current_target.backend_node_id = 999

            reason = validate_binding_integrity(binding, session.tracker.state)
            assert reason == BindingFailureReason.NODE_IDENTITY_MISMATCH

            with pytest.raises(RuntimeError, match="NODE_IDENTITY_MISMATCH"):
                await session.executor.execute("click", binding=binding)

            assert len(cdp.dispatched_mouse_events) == 0
        finally:
            await session.stop()
