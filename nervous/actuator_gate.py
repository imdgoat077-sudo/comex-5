"""
Comex Browser Nervous System — Stage C: Actuator Gate.

First authoritative gate in the control pipeline.  Sits between the
ReflexEngine and the existing actuator/executor.

Pipeline position:
    requested action → WorldState → ReflexEngine → ActuatorGate → executor → CDP

Guarantees:
    • ALLOW  → exactly one logical action execution.
    • WAIT / INVALIDATE / RECOMPUTE_TARGET / REPORT_BLOCKED → zero executions.
    • ROUTE_TO_KNOWN_HANDLER → exactly one handler invocation, zero page actions.
    • No gate-generated sleep.
    • No hidden retries or duplicate dispatches.
    • No new actionability semantics (Method E unchanged).
"""

from __future__ import annotations

import inspect
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .reflex_engine import DecisionType, ReflexDecision, ReflexEngine
from .world_state import BrowserState

logger = logging.getLogger("comex.nervous.actuator_gate")


# ──────────────────────────────────────────────────────────────────────
#  Executor & Handler Protocols
# ──────────────────────────────────────────────────────────────────────

@runtime_checkable
class Executor(Protocol):
    """
    Interface for the existing actuator/executor.

    A single call to `execute` represents one *logical* browser action
    (e.g. click, type, scroll).  The executor may internally perform
    multiple CDP operations (scroll-into-view, coordinate resolution,
    mousePressed + mouseReleased, etc.) — that is its concern, not the
    gate's.
    """

    def execute(self, action: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class ActionHandler(Protocol):
    """
    Interface for overlay / dialog handlers supplied by the caller.

    The gate does NOT invent handlers — it invokes whatever handler is
    supplied when ROUTE_TO_KNOWN_HANDLER fires.
    """

    def handle(self, action: str, state: BrowserState, decision: ReflexDecision, **kwargs: Any) -> Any: ...


# ──────────────────────────────────────────────────────────────────────
#  Gate Result
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateResult:
    """Immutable result of a single gate attempt."""

    decision: ReflexDecision
    executed: bool = False
    handler_invoked: bool = False
    execution_result: Any = None
    eval_to_dispatch_ns: int = 0       # Segment A: reflex eval → executor call
    validation_to_dispatch_ns: int = 0 # Segment B: authoritative validation → physical dispatch

    @property
    def was_blocked(self) -> bool:
        return not self.executed and not self.handler_invoked


# ──────────────────────────────────────────────────────────────────────
#  TOCTOU Latency Tracker
# ──────────────────────────────────────────────────────────────────────

class LatencyTracker:
    """Accumulates eval-to-dispatch latency samples and reports percentiles."""

    def __init__(self) -> None:
        self._samples_ns: List[int] = []

    def record(self, latency_ns: int) -> None:
        self._samples_ns.append(latency_ns)

    @property
    def sample_count(self) -> int:
        return len(self._samples_ns)

    def percentiles(self) -> Dict[str, float]:
        """Return P50, P95, P99 in microseconds.  Empty if no samples."""
        if not self._samples_ns:
            return {"p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0}
        sorted_ns = sorted(self._samples_ns)
        n = len(sorted_ns)

        def _pct(p: float) -> float:
            idx = int(p / 100.0 * (n - 1))
            return sorted_ns[idx] / 1000.0  # ns → µs

        return {
            "p50_us": _pct(50),
            "p95_us": _pct(95),
            "p99_us": _pct(99),
        }

    def clear(self) -> None:
        self._samples_ns.clear()


# ──────────────────────────────────────────────────────────────────────
#  Actuator Gate
# ──────────────────────────────────────────────────────────────────────

class ActuatorGate:
    """
    Authoritative action gate.

    Usage:
        gate = ActuatorGate(state, executor)
        result = gate.attempt("click", target="#buy-btn")

        # or in async context:
        result = await gate.async_attempt("click", binding=binding)
    """

    def __init__(
        self,
        state: BrowserState,
        executor: Executor,
        *,
        handler: Optional[ActionHandler] = None,
    ) -> None:
        self._state = state
        self._engine = ReflexEngine(state)
        self._executor = executor
        self._handler = handler
        self._latency = LatencyTracker()
        self._attempt_log: List[GateResult] = []

    # ── public API ────────────────────────────────────────────────────

    @property
    def latency(self) -> LatencyTracker:
        return self._latency

    @property
    def attempt_log(self) -> List[GateResult]:
        return list(self._attempt_log)

    @property
    def attempt_count(self) -> int:
        return len(self._attempt_log)

    def _evaluate_reflex(self, action: str) -> tuple[ReflexDecision, int]:
        t_eval_ns = time.perf_counter_ns()
        decision = self._engine.evaluate(action)
        return decision, t_eval_ns

    def _record_result(self, result: GateResult, action: str) -> GateResult:
        self._attempt_log.append(result)
        logger.info(
            "GATE %s action=%s executed=%s handler=%s latency=%dns",
            result.decision.decision.name,
            action,
            result.executed,
            result.handler_invoked,
            result.eval_to_dispatch_ns,
        )
        return result

    def attempt(self, action: str, **kwargs: Any) -> GateResult:
        """
        Evaluate the reflex engine and, if authorized, dispatch exactly
        one logical action through the executor (synchronous path).

        Never sleeps.  Never retries.  Never dispatches more than once.
        """
        decision, t_eval_ns = self._evaluate_reflex(action)

        result: GateResult
        if decision.decision == DecisionType.ALLOW:
            t_dispatch_ns = time.perf_counter_ns()
            latency_ns = t_dispatch_ns - t_eval_ns
            exec_result = self._executor.execute(action, **kwargs)
            self._latency.record(latency_ns)
            result = GateResult(
                decision=decision,
                executed=True,
                execution_result=exec_result,
                eval_to_dispatch_ns=latency_ns,
            )
        elif decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER:
            if self._handler is not None:
                handler_result = self._handler.handle(
                    action, self._state, decision, **kwargs
                )
                result = GateResult(
                    decision=decision,
                    handler_invoked=True,
                    execution_result=handler_result,
                )
            else:
                logger.warning(
                    "ROUTE_TO_KNOWN_HANDLER but no handler supplied for %s",
                    decision.rule_name,
                )
                result = GateResult(decision=decision)
        else:
            result = GateResult(decision=decision)

        return self._record_result(result, action)

    async def async_attempt(self, action: str, **kwargs: Any) -> GateResult:
        """
        Asynchronous attempt path.
        Shares the identical reflex evaluation and routing rules as attempt(),
        but natively awaits an asynchronous executor if one is provided.
        """
        decision, t_eval_ns = self._evaluate_reflex(action)

        result: GateResult
        if decision.decision == DecisionType.ALLOW:
            t_dispatch_ns = time.perf_counter_ns()
            latency_ns = t_dispatch_ns - t_eval_ns
            exec_result = self._executor.execute(action, **kwargs)
            if inspect.isawaitable(exec_result):
                exec_result = await exec_result
            self._latency.record(latency_ns)

            val_ns = 0
            if isinstance(exec_result, dict) and "validation_to_dispatch_ns" in exec_result:
                val_ns = exec_result["validation_to_dispatch_ns"]

            result = GateResult(
                decision=decision,
                executed=True,
                execution_result=exec_result,
                eval_to_dispatch_ns=latency_ns,
                validation_to_dispatch_ns=val_ns,
            )
        elif decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER:
            if self._handler is not None:
                handler_result = self._handler.handle(
                    action, self._state, decision, **kwargs
                )
                if inspect.isawaitable(handler_result):
                    handler_result = await handler_result
                result = GateResult(
                    decision=decision,
                    handler_invoked=True,
                    execution_result=handler_result,
                )
            else:
                logger.warning(
                    "ROUTE_TO_KNOWN_HANDLER but no handler supplied for %s",
                    decision.rule_name,
                )
                result = GateResult(decision=decision)
        else:
            result = GateResult(decision=decision)

        return self._record_result(result, action)
