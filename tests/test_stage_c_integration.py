"""
Stage-C Integration Verification Harness: tests/test_stage_c_integration.py

Instruments the live-equivalent control path:
    request → WorldState → ReflexEngine → ActuatorGate → executor → CDP

Proves three required scenarios without modifying production code.
The mock executor stands in for subprocess.run(browser-harness).
"""

import time
import pytest
from unittest.mock import MagicMock, call

from nervous.world_state import BrowserState, WorldStateTracker
from nervous.reflex_engine import DecisionType, ReflexEngine
from nervous.actuator_gate import ActuatorGate, GateResult


# ──────────────────────────────────────────────────────────────────────
#  Mock executor: stands in for subprocess.run(browser-harness) → CDP
# ──────────────────────────────────────────────────────────────────────

class MockBrowserHarnessExecutor:
    """
    Simulates the existing actuator: the run_browser_harness subprocess call.

    In production this is agent_service.py:593:
        subprocess.run(["~/.local/bin/browser-harness"], input=code, ...)

    Records every logical dispatch so tests can assert exact call counts.
    """

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def execute(self, action: str, **kwargs) -> str:
        self.call_count += 1
        self.calls.append((action, kwargs))
        return f"cdp_ok:{action}"


class MockActionHandler:
    """Simulates a dialog/overlay handler supplied by the caller."""

    def __init__(self):
        self.call_count = 0

    def handle(self, action, state, decision, **kwargs):
        self.call_count += 1
        return f"handler_ok:{action}"


def _make_ready_state() -> BrowserState:
    """BrowserState satisfying all ACTION_READY conditions."""
    state = BrowserState()
    state.tracking_started_at_mono = time.monotonic() - 0.200
    state.last_dom_mutation_at_mono = time.monotonic() - 0.200
    state.lifecycle_phase = "STABLE"
    state.is_navigating = False
    target = state.assign_target(
        locator="#btn", node_id=100, is_actionable=True, is_stable=True
    )
    target.subtree_last_mutation_at_mono = time.monotonic() - 0.200
    return state


# ══════════════════════════════════════════════════════════════════════
#  Scenario A: Ready target → exactly one logical executor invocation
# ══════════════════════════════════════════════════════════════════════

def test_scenario_a_ready_target_one_execution():
    """
    Pipeline trace:
        request "click"
            → WorldState[STABLE, target=#btn actionable+stable+quiet]
            → ReflexEngine → ALLOW @ P5
            → ActuatorGate → executor.execute("click") called once
            → CDP (simulated)
    """
    state = _make_ready_state()
    executor = MockBrowserHarnessExecutor()
    gate = ActuatorGate(state, executor)

    result = gate.attempt("click", target="#btn")

    # Pipeline traversal proof
    assert result.decision.decision == DecisionType.ALLOW,       "ReflexEngine must emit ALLOW"
    assert result.executed is True,                              "ActuatorGate must set executed=True"
    assert executor.call_count == 1,                             "Executor must be called exactly once"
    assert executor.calls[0][0] == "click",                     "Action must reach executor"
    assert result.execution_result == "cdp_ok:click",           "Executor return value must propagate"

    # No duplicate dispatch
    assert gate.attempt_count == 1
    assert len(executor.calls) == 1


# ══════════════════════════════════════════════════════════════════════
#  Scenario B: Not-ready target → zero executor invocations
# ══════════════════════════════════════════════════════════════════════

def test_scenario_b_not_ready_zero_executions():
    """
    Pipeline trace:
        request "click"
            → WorldState[target subtree mutated 5ms ago]
            → ReflexEngine → WAIT @ P2 (TargetSubtreeChurning)
            → ActuatorGate → executor NOT called
            → CDP: 0 calls
    """
    state = _make_ready_state()
    # Force subtree to appear actively churning
    state.current_target.subtree_last_mutation_at_mono = time.monotonic()

    executor = MockBrowserHarnessExecutor()
    gate = ActuatorGate(state, executor)

    result = gate.attempt("click")

    assert result.decision.decision == DecisionType.WAIT,        "ReflexEngine must emit WAIT"
    assert result.decision.rule_name == "TargetSubtreeChurning", "Correct rule must fire"
    assert result.executed is False,                             "Gate must not execute"
    assert result.was_blocked is True,                           "Gate must report blocked"
    assert executor.call_count == 0,                             "Executor must not be called"


# ══════════════════════════════════════════════════════════════════════
#  Scenario C: Dialog/overlay → zero page actions, handler routed
# ══════════════════════════════════════════════════════════════════════

def test_scenario_c_dialog_zero_page_actions():
    """
    Pipeline trace (two sub-cases):

    C1 — ordinary click while dialog present:
        request "click"
            → WorldState[active_dialog=confirm]
            → ReflexEngine → REPORT_BLOCKED @ P0 (ActiveDialogRule)
            → ActuatorGate → executor NOT called
            → CDP: 0 calls

    C2 — accept_dialog resolution:
        request "accept_dialog"
            → WorldState[active_dialog=confirm]
            → ReflexEngine → ROUTE_TO_KNOWN_HANDLER @ P0 (DialogResolutionHandler)
            → ActuatorGate → handler.handle called once
            → executor NOT called
            → CDP: 0 page actions
    """
    state = _make_ready_state()
    state.active_dialog = {"type": "confirm", "message": "Are you sure?"}

    executor = MockBrowserHarnessExecutor()
    handler = MockActionHandler()
    gate = ActuatorGate(state, executor, handler=handler)

    # C1: click blocked
    r1 = gate.attempt("click")
    assert r1.decision.decision == DecisionType.REPORT_BLOCKED,  "Dialog must block click"
    assert r1.decision.rule_name == "ActiveDialogRule"
    assert r1.executed is False
    assert executor.call_count == 0,                             "No CDP page action"

    # C2: accept_dialog routes to handler
    r2 = gate.attempt("accept_dialog")
    assert r2.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER, "Must route to handler"
    assert r2.decision.rule_name == "DialogResolutionHandler"
    assert r2.handler_invoked is True,                           "Handler must be invoked"
    assert r2.executed is False,                                 "Executor must not fire"
    assert executor.call_count == 0,                             "Still zero page CDP actions"
    assert handler.call_count == 1,                              "Handler invoked exactly once"


# ══════════════════════════════════════════════════════════════════════
#  Pipeline trace order verification
# ══════════════════════════════════════════════════════════════════════

def test_pipeline_trace_order():
    """
    Verify the pipeline is traversed in the correct documented order:
        WorldState → ReflexEngine → ActuatorGate → executor

    The gate must not call the executor before the reflex decision,
    and reflex evaluation must not itself call the executor.
    """
    execution_order = []

    class TracingState(BrowserState):
        pass  # real BrowserState, just aliased for clarity

    class TracingExecutor:
        def execute(self, action, **kw):
            execution_order.append("executor")
            return "ok"

    # Patch ReflexEngine.evaluate to record its position
    original_evaluate = ReflexEngine.evaluate

    def tracing_evaluate(self, action="click"):
        execution_order.append("reflex_engine")
        return original_evaluate(self, action)

    ReflexEngine.evaluate = tracing_evaluate
    try:
        state = _make_ready_state()
        executor = TracingExecutor()
        gate = ActuatorGate(state, executor)
        execution_order.append("gate_attempt_start")
        gate.attempt("click")
        execution_order.append("gate_attempt_end")
    finally:
        ReflexEngine.evaluate = original_evaluate

    assert execution_order == [
        "gate_attempt_start",
        "reflex_engine",   # WorldState read + ReflexEngine.evaluate
        "executor",        # ActuatorGate dispatches to executor
        "gate_attempt_end",
    ], f"Pipeline order violated: {execution_order}"
