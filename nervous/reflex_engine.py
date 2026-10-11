"""
Comex Browser Nervous System — Stage B: Deterministic Reflex Engine (Shadow Mode).

Consumes a live BrowserState snapshot from the WorldStateTracker and produces
prioritized ReflexDecision objects.  In shadow mode the engine ONLY logs
decisions — it does NOT intercept, delay, modify, or cancel real browser
actions.

Priority Hierarchy (Section F of Frozen Spec):
    P0  Safety & Invalidation            (Highest)
    P1  Lifecycle & Navigation
    P2  Target Integrity & Temporal Stability
    P3  Known Obstruction (Policy Registry)
    P4  Frozen Method E Actionability     (Authorization)
    P5  Lossy Telemetry                   (Lowest)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .world_state import BrowserState

logger = logging.getLogger("comex.nervous.reflex")


# ──────────────────────────────────────────────────────────────────────
#  Decision Types & Priority
# ──────────────────────────────────────────────────────────────────────

class DecisionType(Enum):
    """The six deterministic decision types emitted by the Reflex Engine."""
    ALLOW = auto()
    WAIT = auto()
    INVALIDATE = auto()
    RECOMPUTE_TARGET = auto()
    ROUTE_TO_KNOWN_HANDLER = auto()
    REPORT_BLOCKED = auto()


class Priority(Enum):
    """Priority tiers — lower numeric value = higher priority."""
    P0_SAFETY = 0
    P1_LIFECYCLE = 1
    P2_TARGET_INTEGRITY = 2
    P3_OBSTRUCTION = 3
    P4_METHOD_E = 4
    P5_TELEMETRY = 5


@dataclass(frozen=True)
class ReflexDecision:
    """Immutable decision record produced by a single evaluation pass."""
    decision: DecisionType
    priority: Priority
    rule_name: str
    reason: str
    timestamp_mono: float = field(default_factory=time.monotonic)

    def __str__(self) -> str:
        return (
            f"[{self.priority.name}] {self.decision.name} "
            f"({self.rule_name}): {self.reason}"
        )


# ──────────────────────────────────────────────────────────────────────
#  Individual Deterministic Rules
# ──────────────────────────────────────────────────────────────────────
#  Each rule is a pure function:
#      (BrowserState, requested_action) → Optional[ReflexDecision]
#
#  Returning None means the rule has no opinion; the engine proceeds to
#  the next lower-priority rule.
# ──────────────────────────────────────────────────────────────────────

def _rule_document_invalidated(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P0 — If DOM.documentUpdated has fired and the document is in invalidated state."""
    if state.document_invalidated:
        return ReflexDecision(
            decision=DecisionType.INVALIDATE,
            priority=Priority.P0_SAFETY,
            rule_name="DocumentInvalidated",
            reason="DOM.documentUpdated has fired; all node handles are stale.",
        )
    return None


def _rule_active_dialog(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """
    P0 — Active JS dialog blocks all non-dialog actions.
    Dialog-handling actions (accept_dialog, dismiss_dialog) are explicitly
    exempt; the dialog IS their target.
    """
    if state.active_dialog is None:
        return None

    # Dialog resolution commands bypass the blocker
    if action in ("accept_dialog", "dismiss_dialog"):
        return ReflexDecision(
            decision=DecisionType.ROUTE_TO_KNOWN_HANDLER,
            priority=Priority.P0_SAFETY,
            rule_name="DialogResolutionHandler",
            reason=f"Dialog present; routing '{action}' to dialog handler.",
        )

    return ReflexDecision(
        decision=DecisionType.REPORT_BLOCKED,
        priority=Priority.P0_SAFETY,
        rule_name="ActiveDialogRule",
        reason=f"JS {state.active_dialog.get('type', 'unknown')} dialog is blocking input pipeline.",
    )


def _rule_navigation_in_progress(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P1 — Page is mid-navigation; action should wait."""
    if state.is_navigating:
        return ReflexDecision(
            decision=DecisionType.WAIT,
            priority=Priority.P1_LIFECYCLE,
            rule_name="NavigationInProgress",
            reason="Main frame is navigating; action deferred until load settles.",
        )
    return None


def _rule_no_target_assigned(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P2 — No target is assigned; action cannot proceed."""
    if state.current_target is None:
        return ReflexDecision(
            decision=DecisionType.RECOMPUTE_TARGET,
            priority=Priority.P2_TARGET_INTEGRITY,
            rule_name="NoTargetAssigned",
            reason="No current target; target must be resolved before action.",
        )
    return None


def _rule_target_not_actionable(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P4 — Target exists but Method E predicate evaluates to False."""
    if state.current_target and not state.current_target.is_actionable:
        return ReflexDecision(
            decision=DecisionType.WAIT,
            priority=Priority.P4_METHOD_E,
            rule_name="MethodEFailed",
            reason="Target does not satisfy frozen Method E actionability predicate.",
        )
    return None


def _rule_target_geometrically_unstable(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P2 — Target bounding box is still moving (Δ ≥ 1.0 px)."""
    if state.current_target and not state.current_target.is_stable:
        return ReflexDecision(
            decision=DecisionType.WAIT,
            priority=Priority.P2_TARGET_INTEGRITY,
            rule_name="TargetGeometricallyUnstable",
            reason=(
                f"Target position delta {state.current_target.last_position_delta:.2f}px "
                f"≥ 1.0px threshold."
            ),
        )
    return None


def _rule_target_subtree_churning(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P2 — Target subtree has not been quiet for ≥ 80 ms."""
    if not state.is_target_subtree_quiet:
        return ReflexDecision(
            decision=DecisionType.WAIT,
            priority=Priority.P2_TARGET_INTEGRITY,
            rule_name="TargetSubtreeChurning",
            reason="Target subtree mutations detected within 80ms quiescence window.",
        )
    return None


def _rule_known_overlay_blocking(state: BrowserState, action: str) -> Optional[ReflexDecision]:
    """P3 — A known blocking overlay has been detected by the world state."""
    if not state.blocking_overlay_detected:
        return None

    if state.known_overlay_match is not None:
        return ReflexDecision(
            decision=DecisionType.ROUTE_TO_KNOWN_HANDLER,
            priority=Priority.P3_OBSTRUCTION,
            rule_name="KnownOverlayBlocking",
            reason=f"Blocking overlay detected; matched handler: '{state.known_overlay_match}'.",
        )

    # Overlay detected but no verified handler registered
    return ReflexDecision(
        decision=DecisionType.REPORT_BLOCKED,
        priority=Priority.P3_OBSTRUCTION,
        rule_name="UnmatchedOverlayBlocking",
        reason="Blocking overlay detected but no registered handler matches.",
    )


# Ordered rule chain — evaluated top-to-bottom; first non-None wins.
_RULE_CHAIN: list = [
    # P0 Safety
    _rule_document_invalidated,
    _rule_active_dialog,
    # P1 Lifecycle
    _rule_navigation_in_progress,
    # P2 Target Integrity
    _rule_no_target_assigned,
    _rule_target_geometrically_unstable,
    _rule_target_subtree_churning,
    # P3 Obstruction
    _rule_known_overlay_blocking,
    # P4 Method E
    _rule_target_not_actionable,
]


# ──────────────────────────────────────────────────────────────────────
#  Reflex Engine (Shadow Mode)
# ──────────────────────────────────────────────────────────────────────

class ReflexEngine:
    """
    Deterministic reflex evaluator running in **shadow mode**.

    Usage:
        engine = ReflexEngine(tracker.state)
        decision = engine.evaluate("click")
        # decision is logged; it does NOT gate the actual action.

    The engine evaluates the ordered rule chain against the current
    BrowserState snapshot and returns the highest-priority decision.
    If no rule fires, ALLOW is returned at P5 (telemetry tier).
    """

    def __init__(self, state: BrowserState) -> None:
        self._state = state
        self._decision_log: List[ReflexDecision] = []

    @property
    def state(self) -> BrowserState:
        return self._state

    @property
    def decision_log(self) -> List[ReflexDecision]:
        """Read-only access to the complete decision history."""
        return list(self._decision_log)

    @property
    def decision_count(self) -> int:
        return len(self._decision_log)

    def evaluate(self, action: str = "click") -> ReflexDecision:
        """
        Run the deterministic rule chain against current state.

        Returns the highest-priority ReflexDecision.  In shadow mode the
        decision is only logged — never enforced.
        """
        for rule_fn in _RULE_CHAIN:
            decision = rule_fn(self._state, action)
            if decision is not None:
                self._record(decision)
                return decision

        # All rules passed — action is allowed
        allow = ReflexDecision(
            decision=DecisionType.ALLOW,
            priority=Priority.P5_TELEMETRY,
            rule_name="AllRulesPassed",
            reason="All P0–P4 rules passed; action authorized.",
        )
        self._record(allow)
        return allow

    def evaluate_batch(self, actions: List[str]) -> List[ReflexDecision]:
        """Evaluate a sequence of actions and return their decisions."""
        return [self.evaluate(action) for action in actions]

    def clear_log(self) -> None:
        """Clear the decision history."""
        self._decision_log.clear()

    # ── internal ──────────────────────────────────────────────────────

    def _record(self, decision: ReflexDecision) -> None:
        """Append to log and emit a shadow-mode log line."""
        self._decision_log.append(decision)
        logger.info("SHADOW %s", decision)
