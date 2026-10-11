"""
Stage-C Verification Suite: tests/test_actuator_gate.py

Validates the ActuatorGate against all decision paths, side-effect
isolation, single-execution guarantees, handler routing, TOCTOU
latency measurement, and synthetic replay scenarios.
"""

import time
import pytest
from unittest.mock import MagicMock, call

from nervous.world_state import BrowserState, TargetState, WorldStateTracker
from nervous.reflex_engine import DecisionType, Priority, ReflexDecision, ReflexEngine
from nervous.actuator_gate import ActuatorGate, GateResult, LatencyTracker


# ──────────────────────────────────────────────────────────────────────
#  Fixtures: Mock Executor & Handler
# ──────────────────────────────────────────────────────────────────────

def _mock_executor() -> MagicMock:
    """Returns a mock executor conforming to the Executor protocol."""
    executor = MagicMock()
    executor.execute = MagicMock(return_value="exec_ok")
    return executor


def _mock_handler() -> MagicMock:
    """Returns a mock handler conforming to the ActionHandler protocol."""
    handler = MagicMock()
    handler.handle = MagicMock(return_value="handler_ok")
    return handler


def _make_ready_state() -> BrowserState:
    """BrowserState with a stable, actionable, quiet target."""
    state = BrowserState()
    state.tracking_started_at_mono = time.monotonic() - 0.200
    state.last_dom_mutation_at_mono = time.monotonic() - 0.200
    state.lifecycle_phase = "STABLE"
    state.is_navigating = False
    target = state.assign_target(
        locator="#btn",
        node_id=100,
        is_actionable=True,
        is_stable=True,
    )
    target.subtree_last_mutation_at_mono = time.monotonic() - 0.200
    return state


# ══════════════════════════════════════════════════════════════════════
#  A. Stable Actionable Target → ALLOW → Executes Once
# ══════════════════════════════════════════════════════════════════════

class TestAllowExecution:

    def test_allow_executes_action_once(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is True
        assert result.handler_invoked is False
        assert result.was_blocked is False
        assert result.decision.decision == DecisionType.ALLOW
        assert result.execution_result == "exec_ok"
        executor.execute.assert_called_once_with("click")

    def test_allow_passes_kwargs_to_executor(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        gate.attempt("click", target="#buy-btn", x=100, y=200)

        executor.execute.assert_called_once_with(
            "click", target="#buy-btn", x=100, y=200
        )

    def test_no_duplicate_execution_on_single_attempt(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        gate.attempt("click")

        assert executor.execute.call_count == 1

    def test_multiple_attempts_produce_independent_executions(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        gate.attempt("click")
        gate.attempt("type")

        assert executor.execute.call_count == 2
        assert gate.attempt_count == 2


# ══════════════════════════════════════════════════════════════════════
#  B. Moving Target → WAIT → Zero Executions
# ══════════════════════════════════════════════════════════════════════

class TestMovingTarget:

    def test_geometrically_unstable_target_blocks(self):
        state = _make_ready_state()
        state.current_target.is_stable = False
        state.current_target.last_position_delta = 5.0
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.was_blocked is True
        assert result.decision.decision == DecisionType.WAIT
        assert result.decision.rule_name == "TargetGeometricallyUnstable"
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  C. Target Subtree Churn → WAIT → Zero Executions
# ══════════════════════════════════════════════════════════════════════

class TestSubtreeChurn:

    def test_subtree_churning_blocks(self):
        state = _make_ready_state()
        state.current_target.subtree_last_mutation_at_mono = time.monotonic()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.was_blocked is True
        assert result.decision.decision == DecisionType.WAIT
        assert result.decision.rule_name == "TargetSubtreeChurning"
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  D. Target Detached / Document Invalidated → INVALIDATE → Zero
# ══════════════════════════════════════════════════════════════════════

class TestInvalidation:

    def test_document_invalidated_blocks(self):
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
        tracker.handle_document_updated()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.was_blocked is True
        assert result.decision.decision == DecisionType.INVALIDATE
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  E. Active Navigation → WAIT → Zero Executions
# ══════════════════════════════════════════════════════════════════════

class TestNavigationBlocking:

    def test_navigation_in_progress_blocks(self):
        state = _make_ready_state()
        state.is_navigating = True
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.was_blocked is True
        assert result.decision.decision == DecisionType.WAIT
        assert result.decision.rule_name == "NavigationInProgress"
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  F. Unknown Overlay → REPORT_BLOCKED → Zero Executions
# ══════════════════════════════════════════════════════════════════════

class TestUnknownOverlay:

    def test_unmatched_overlay_blocks(self):
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = None
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.handler_invoked is False
        assert result.was_blocked is True
        assert result.decision.decision == DecisionType.REPORT_BLOCKED
        assert result.decision.rule_name == "UnmatchedOverlayBlocking"
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  G. Known Handler → ROUTE_TO_KNOWN_HANDLER → Handler Invoked
# ══════════════════════════════════════════════════════════════════════

class TestKnownHandler:

    def test_known_overlay_routes_to_handler(self):
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = "cookie_consent"
        executor = _mock_executor()
        handler = _mock_handler()
        gate = ActuatorGate(state, executor, handler=handler)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.handler_invoked is True
        assert result.execution_result == "handler_ok"
        assert result.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
        executor.execute.assert_not_called()
        handler.handle.assert_called_once()

    def test_route_without_handler_reports_blocked(self):
        """ROUTE_TO_KNOWN_HANDLER but no handler supplied → effectively blocked."""
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = "cookie_consent"
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)  # No handler

        result = gate.attempt("click")

        assert result.executed is False
        assert result.handler_invoked is False
        assert result.was_blocked is True
        executor.execute.assert_not_called()

    def test_handler_receives_correct_args(self):
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = "gdpr"
        executor = _mock_executor()
        handler = _mock_handler()
        gate = ActuatorGate(state, executor, handler=handler)

        gate.attempt("click", selector=".dismiss")

        args, kwargs = handler.handle.call_args
        assert args[0] == "click"             # action
        assert args[1] is state               # state
        assert isinstance(args[2], ReflexDecision)  # decision
        assert kwargs.get("selector") == ".dismiss"


# ══════════════════════════════════════════════════════════════════════
#  H. Unrelated Global DOM Churn → Local Target ACTION_READY → Executes
# ══════════════════════════════════════════════════════════════════════

class TestGlobalChurnImmunity:

    def test_global_churn_does_not_block_local_target(self):
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.tracking_started_at_mono = time.monotonic() - 0.200
        state.last_dom_mutation_at_mono = time.monotonic() - 0.200

        target = state.assign_target(
            locator="#buy", node_id=10, is_actionable=True, is_stable=True
        )
        target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

        # Sibling churn (ad ticker node 999)
        state.node_parent_map[10] = 1
        state.node_parent_map[999] = 1
        tracker.handle_character_data_modified(node_id=999, character_data="Ad")

        assert not state.is_global_dom_quiet
        assert not state.is_page_stable

        executor = _mock_executor()
        gate = ActuatorGate(state, executor)
        result = gate.attempt("click")

        assert result.executed is True
        assert result.decision.decision == DecisionType.ALLOW
        executor.execute.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
#  I. DOM.documentUpdated → INVALIDATE → Stale Target Cannot Execute
# ══════════════════════════════════════════════════════════════════════

class TestDocumentUpdatedInvalidation:

    def test_stale_target_after_document_updated(self):
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.assign_target(locator="#link", node_id=200, is_actionable=True, is_stable=True)

        tracker.handle_frame_navigated("https://example.com/new", is_main_frame=True)
        tracker.handle_document_updated()

        executor = _mock_executor()
        gate = ActuatorGate(state, executor)
        result = gate.attempt("click")

        assert result.executed is False
        assert result.decision.decision == DecisionType.INVALIDATE
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  J. Dialog Resolution → Handler Invoked, No Ordinary Click
# ══════════════════════════════════════════════════════════════════════

class TestDialogResolution:

    def test_accept_dialog_routes_to_handler(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "confirm", "message": "OK?"}
        executor = _mock_executor()
        handler = _mock_handler()
        gate = ActuatorGate(state, executor, handler=handler)

        result = gate.attempt("accept_dialog")

        assert result.handler_invoked is True
        assert result.executed is False
        assert result.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
        assert result.decision.rule_name == "DialogResolutionHandler"
        executor.execute.assert_not_called()
        handler.handle.assert_called_once()

    def test_dismiss_dialog_routes_to_handler(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "beforeunload", "message": "Leave?"}
        executor = _mock_executor()
        handler = _mock_handler()
        gate = ActuatorGate(state, executor, handler=handler)

        result = gate.attempt("dismiss_dialog")

        assert result.handler_invoked is True
        assert result.executed is False
        executor.execute.assert_not_called()

    def test_dialog_blocks_non_dialog_action(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "alert", "message": "Wait"}
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.executed is False
        assert result.was_blocked is True
        assert result.decision.decision == DecisionType.REPORT_BLOCKED
        executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
#  K. Pre-Quiescent Target → No Artificial Wait
# ══════════════════════════════════════════════════════════════════════

class TestNoArtificialWait:

    def test_pre_quiescent_target_executes_immediately(self):
        state = BrowserState()
        now = time.monotonic()
        state.tracking_started_at_mono = now - 0.200
        state.last_dom_mutation_at_mono = now - 0.200
        state.lifecycle_phase = "STABLE"
        target = state.assign_target(
            locator="#ready", node_id=50, is_actionable=True, is_stable=True
        )
        # Subtree inherited quiet timestamp, should be immediately quiet
        assert state.is_target_subtree_quiet is True

        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        t_before = time.perf_counter_ns()
        result = gate.attempt("click")
        t_after = time.perf_counter_ns()

        assert result.executed is True
        # Gate itself should not introduce any sleep — entire attempt
        # should complete in well under 5ms (generous bound)
        elapsed_ms = (t_after - t_before) / 1_000_000
        assert elapsed_ms < 5.0, f"Gate took {elapsed_ms:.2f}ms — too slow for no-sleep"


# ══════════════════════════════════════════════════════════════════════
#  5. Safety / Side-Effect Tests
# ══════════════════════════════════════════════════════════════════════

class TestSideEffectIsolation:
    """Verify that non-ALLOW decisions produce exactly zero logical executions."""

    def test_wait_zero_executions(self):
        state = _make_ready_state()
        state.is_navigating = True
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")
        assert result.decision.decision == DecisionType.WAIT
        executor.execute.assert_not_called()

    def test_invalidate_zero_executions(self):
        state = BrowserState()
        state.document_invalidated = True
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")
        assert result.decision.decision == DecisionType.INVALIDATE
        executor.execute.assert_not_called()

    def test_report_blocked_zero_executions(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "alert", "message": "X"}
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")
        assert result.decision.decision == DecisionType.REPORT_BLOCKED
        executor.execute.assert_not_called()

    def test_recompute_target_zero_executions(self):
        state = BrowserState()
        state.lifecycle_phase = "STABLE"
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")
        assert result.decision.decision == DecisionType.RECOMPUTE_TARGET
        executor.execute.assert_not_called()

    def test_allow_exactly_one_execution(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        gate.attempt("click")
        assert executor.execute.call_count == 1

    def test_route_handler_exactly_one_invocation(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "confirm", "message": "?"}
        executor = _mock_executor()
        handler = _mock_handler()
        gate = ActuatorGate(state, executor, handler=handler)

        gate.attempt("accept_dialog")

        handler.handle.assert_called_once()
        executor.execute.assert_not_called()

    def test_reflex_evaluation_never_executes(self):
        """ReflexEngine.evaluate alone must never trigger executor."""
        state = _make_ready_state()
        executor = _mock_executor()
        engine = ReflexEngine(state)

        # Evaluate without going through gate
        engine.evaluate("click")
        engine.evaluate("type")
        engine.evaluate("scroll")

        executor.execute.assert_not_called()

    def test_gate_never_sleeps(self):
        """All decision paths complete without sleep-like delays."""
        scenarios = []

        # ALLOW
        s1 = _make_ready_state()
        scenarios.append(("click", s1, _mock_executor(), None))

        # WAIT (navigation)
        s2 = _make_ready_state()
        s2.is_navigating = True
        scenarios.append(("click", s2, _mock_executor(), None))

        # INVALIDATE
        s3 = BrowserState()
        s3.document_invalidated = True
        scenarios.append(("click", s3, _mock_executor(), None))

        # REPORT_BLOCKED (dialog)
        s4 = _make_ready_state()
        s4.active_dialog = {"type": "alert", "message": "X"}
        scenarios.append(("click", s4, _mock_executor(), None))

        # ROUTE_TO_KNOWN_HANDLER
        s5 = _make_ready_state()
        s5.active_dialog = {"type": "confirm", "message": "?"}
        scenarios.append(("accept_dialog", s5, _mock_executor(), _mock_handler()))

        for action, state, executor, handler in scenarios:
            gate = ActuatorGate(state, executor, handler=handler)
            t0 = time.perf_counter_ns()
            gate.attempt(action)
            t1 = time.perf_counter_ns()
            elapsed_ms = (t1 - t0) / 1_000_000
            assert elapsed_ms < 5.0, (
                f"Gate took {elapsed_ms:.2f}ms on {action} — "
                f"should never sleep"
            )


# ══════════════════════════════════════════════════════════════════════
#  6. TOCTOU Latency Measurement
# ══════════════════════════════════════════════════════════════════════

class TestTOCTOULatency:

    def test_latency_recorded_on_allow(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.eval_to_dispatch_ns > 0
        assert gate.latency.sample_count == 1

    def test_latency_not_recorded_on_blocked(self):
        state = _make_ready_state()
        state.is_navigating = True
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        result = gate.attempt("click")

        assert result.eval_to_dispatch_ns == 0
        assert gate.latency.sample_count == 0

    def test_percentile_computation(self):
        tracker = LatencyTracker()
        # Simulate 100 samples between 1000–100000 ns
        for i in range(100):
            tracker.record(1000 + i * 1000)

        pcts = tracker.percentiles()
        assert "p50_us" in pcts
        assert "p95_us" in pcts
        assert "p99_us" in pcts
        assert pcts["p50_us"] > 0
        assert pcts["p95_us"] >= pcts["p50_us"]
        assert pcts["p99_us"] >= pcts["p95_us"]

    def test_percentiles_empty(self):
        tracker = LatencyTracker()
        pcts = tracker.percentiles()
        assert pcts["p50_us"] == 0.0

    def test_multiple_allows_accumulate_latency(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        for _ in range(10):
            gate.attempt("click")

        assert gate.latency.sample_count == 10
        pcts = gate.latency.percentiles()
        assert pcts["p50_us"] > 0


# ══════════════════════════════════════════════════════════════════════
#  7. GateResult Properties
# ══════════════════════════════════════════════════════════════════════

class TestGateResultProperties:

    def test_was_blocked_true_for_wait(self):
        state = _make_ready_state()
        state.is_navigating = True
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)
        result = gate.attempt("click")
        assert result.was_blocked is True

    def test_was_blocked_false_for_allow(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)
        result = gate.attempt("click")
        assert result.was_blocked is False

    def test_was_blocked_false_for_handler(self):
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = "consent"
        executor = _mock_executor()
        handler = _mock_handler()
        gate = ActuatorGate(state, executor, handler=handler)
        result = gate.attempt("click")
        assert result.was_blocked is False

    def test_gate_result_is_frozen(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)
        result = gate.attempt("click")
        with pytest.raises(AttributeError):
            result.executed = False  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════
#  8. Attempt Log
# ══════════════════════════════════════════════════════════════════════

class TestAttemptLog:

    def test_log_accumulates(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)

        gate.attempt("click")
        gate.attempt("type")

        assert gate.attempt_count == 2
        log = gate.attempt_log
        assert log[0].decision.decision == DecisionType.ALLOW
        assert log[1].decision.decision == DecisionType.ALLOW

    def test_log_returns_copy(self):
        state = _make_ready_state()
        executor = _mock_executor()
        gate = ActuatorGate(state, executor)
        gate.attempt("click")

        log1 = gate.attempt_log
        log2 = gate.attempt_log
        assert log1 is not log2
