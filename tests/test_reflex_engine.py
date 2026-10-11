"""
Stage-B Verification Suite: tests/test_reflex_engine.py

Validates the deterministic Reflex Engine in shadow mode against
the frozen Priority Hierarchy (P0–P5), Dialog Arbitration Exception,
document invalidation, target integrity rules, overlay routing,
Method E gating, and synthetic replay timelines.
"""

import time
import pytest
from nervous.world_state import BrowserState, TargetState, WorldStateTracker
from nervous.reflex_engine import (
    DecisionType,
    Priority,
    ReflexDecision,
    ReflexEngine,
)


# ──────────────────────────────────────────────────────────────────────
#  Helper: build a "ready" state with a stable, actionable, quiet target
# ──────────────────────────────────────────────────────────────────────

def _make_ready_state() -> BrowserState:
    """Return a BrowserState that satisfies all ACTION_READY conditions."""
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
    # Ensure subtree is quiet by backdating the mutation timestamp
    target.subtree_last_mutation_at_mono = time.monotonic() - 0.200
    return state


# ══════════════════════════════════════════════════════════════════════
#  1. ALLOW — Happy Path
# ══════════════════════════════════════════════════════════════════════

class TestAllowHappyPath:
    """When all conditions are met the engine must produce ALLOW at P5."""

    def test_allow_on_ready_state(self):
        state = _make_ready_state()
        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.ALLOW
        assert d.priority == Priority.P5_TELEMETRY
        assert d.rule_name == "AllRulesPassed"

    def test_allow_is_logged(self):
        state = _make_ready_state()
        engine = ReflexEngine(state)
        engine.evaluate("click")

        assert engine.decision_count == 1
        assert engine.decision_log[0].decision == DecisionType.ALLOW

    def test_multiple_evaluations_accumulate(self):
        state = _make_ready_state()
        engine = ReflexEngine(state)
        engine.evaluate("click")
        engine.evaluate("type")
        engine.evaluate("scroll")

        assert engine.decision_count == 3

    def test_clear_log(self):
        state = _make_ready_state()
        engine = ReflexEngine(state)
        engine.evaluate("click")
        engine.clear_log()
        assert engine.decision_count == 0


# ══════════════════════════════════════════════════════════════════════
#  2. P0 — Document Invalidated
# ══════════════════════════════════════════════════════════════════════

class TestDocumentInvalidation:
    """DOM.documentUpdated → INVALIDATE at P0."""

    def test_invalidate_after_document_updated(self):
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
        tracker.handle_document_updated()

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.INVALIDATE
        assert d.priority == Priority.P0_SAFETY
        assert d.rule_name == "DocumentInvalidated"
        assert state.document_invalidated is True

    def test_invalidation_takes_precedence_over_lower_rules(self):
        """Even if navigation is also active, P0 invalidation wins."""
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
        tracker.handle_document_updated()
        state.is_navigating = True  # P1 condition also true

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.priority == Priority.P0_SAFETY
        assert d.decision == DecisionType.INVALIDATE

    def test_invalidation_clears_on_target_assignment(self):
        """After document invalidation, assigning a new target clears the flag."""
        tracker = WorldStateTracker("tab")
        state = tracker.state
        tracker.handle_document_updated()
        assert state.document_invalidated is True

        state.assign_target(locator="#new-btn", node_id=200, is_actionable=True, is_stable=True)
        assert state.document_invalidated is False

        # No longer invalidated → should not produce INVALIDATE
        engine = ReflexEngine(state)
        state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 0.200
        d = engine.evaluate("click")
        assert d.decision != DecisionType.INVALIDATE

    def test_invalidation_clears_on_load_event(self):
        """Load event also clears the invalidation flag."""
        tracker = WorldStateTracker("tab")
        state = tracker.state
        tracker.handle_document_updated()
        assert state.document_invalidated is True

        tracker.handle_load_event_fired()
        assert state.document_invalidated is False


# ══════════════════════════════════════════════════════════════════════
#  3. P0 — Active Dialog & Dialog Arbitration Exception
# ══════════════════════════════════════════════════════════════════════

class TestDialogArbitration:
    """JS dialog blocks normal actions but permits dialog resolution commands."""

    def test_dialog_blocks_click(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "alert", "message": "Are you sure?"}

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.REPORT_BLOCKED
        assert d.priority == Priority.P0_SAFETY
        assert d.rule_name == "ActiveDialogRule"

    def test_dialog_blocks_type_action(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "prompt", "message": "Enter name"}

        engine = ReflexEngine(state)
        d = engine.evaluate("type")

        assert d.decision == DecisionType.REPORT_BLOCKED
        assert d.priority == Priority.P0_SAFETY

    def test_accept_dialog_is_allowed(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "confirm", "message": "OK?"}

        engine = ReflexEngine(state)
        d = engine.evaluate("accept_dialog")

        assert d.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
        assert d.priority == Priority.P0_SAFETY
        assert d.rule_name == "DialogResolutionHandler"

    def test_dismiss_dialog_is_allowed(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "beforeunload", "message": "Unsaved changes"}

        engine = ReflexEngine(state)
        d = engine.evaluate("dismiss_dialog")

        assert d.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
        assert d.priority == Priority.P0_SAFETY

    def test_no_dialog_does_not_fire(self):
        state = _make_ready_state()
        assert state.active_dialog is None

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        # Should fall through to ALLOW, not REPORT_BLOCKED
        assert d.decision == DecisionType.ALLOW


# ══════════════════════════════════════════════════════════════════════
#  4. P1 — Navigation In Progress
# ══════════════════════════════════════════════════════════════════════

class TestNavigationInProgress:
    """Mid-navigation produces WAIT at P1."""

    def test_wait_during_navigation(self):
        state = _make_ready_state()
        state.is_navigating = True

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.WAIT
        assert d.priority == Priority.P1_LIFECYCLE
        assert d.rule_name == "NavigationInProgress"

    def test_navigation_has_lower_priority_than_dialog(self):
        """P0 dialog should still win over P1 navigation."""
        state = _make_ready_state()
        state.is_navigating = True
        state.active_dialog = {"type": "alert", "message": "Wait"}

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.priority == Priority.P0_SAFETY
        assert d.decision == DecisionType.REPORT_BLOCKED


# ══════════════════════════════════════════════════════════════════════
#  5. P2 — Target Integrity (no target / unstable / churning)
# ══════════════════════════════════════════════════════════════════════

class TestTargetIntegrity:
    """Target-local integrity checks at P2."""

    def test_no_target_assigned(self):
        state = BrowserState()
        state.lifecycle_phase = "STABLE"

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.RECOMPUTE_TARGET
        assert d.priority == Priority.P2_TARGET_INTEGRITY
        assert d.rule_name == "NoTargetAssigned"

    def test_target_geometrically_unstable(self):
        state = _make_ready_state()
        state.current_target.is_stable = False
        state.current_target.last_position_delta = 5.3

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.WAIT
        assert d.priority == Priority.P2_TARGET_INTEGRITY
        assert d.rule_name == "TargetGeometricallyUnstable"
        assert "5.30" in d.reason

    def test_target_subtree_churning(self):
        state = _make_ready_state()
        # Force subtree to appear actively mutating
        state.current_target.subtree_last_mutation_at_mono = time.monotonic()

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.WAIT
        assert d.priority == Priority.P2_TARGET_INTEGRITY
        assert d.rule_name == "TargetSubtreeChurning"

    def test_geometric_instability_before_subtree_churn(self):
        """Geometric check precedes subtree check in P2 chain."""
        state = _make_ready_state()
        state.current_target.is_stable = False
        state.current_target.last_position_delta = 2.0
        state.current_target.subtree_last_mutation_at_mono = time.monotonic()

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        # Geometric instability rule fires first
        assert d.rule_name == "TargetGeometricallyUnstable"


# ══════════════════════════════════════════════════════════════════════
#  6. P3 — Known Overlay Blocking
# ══════════════════════════════════════════════════════════════════════

class TestOverlayBlocking:
    """Detected overlay routes to a known handler at P3."""

    def test_overlay_routes_to_handler(self):
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = "cookie_consent"

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
        assert d.priority == Priority.P3_OBSTRUCTION
        assert d.rule_name == "KnownOverlayBlocking"
        assert "cookie_consent" in d.reason

    def test_overlay_without_match(self):
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = None

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.REPORT_BLOCKED
        assert d.priority == Priority.P3_OBSTRUCTION
        assert d.rule_name == "UnmatchedOverlayBlocking"

    def test_overlay_lower_than_navigation(self):
        """P1 navigation should fire before P3 overlay."""
        state = _make_ready_state()
        state.is_navigating = True
        state.blocking_overlay_detected = True

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.priority == Priority.P1_LIFECYCLE


# ══════════════════════════════════════════════════════════════════════
#  7. P4 — Method E Actionability
# ══════════════════════════════════════════════════════════════════════

class TestMethodEActionability:
    """Target that fails Method E produces WAIT at P4."""

    def test_method_e_failed(self):
        state = _make_ready_state()
        state.current_target.is_actionable = False

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.decision == DecisionType.WAIT
        assert d.priority == Priority.P4_METHOD_E
        assert d.rule_name == "MethodEFailed"

    def test_method_e_lower_than_target_integrity(self):
        """P2 geometric instability wins over P4 Method E."""
        state = _make_ready_state()
        state.current_target.is_actionable = False
        state.current_target.is_stable = False
        state.current_target.last_position_delta = 3.0

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        assert d.priority == Priority.P2_TARGET_INTEGRITY


# ══════════════════════════════════════════════════════════════════════
#  8. Decision Immutability & Metadata
# ══════════════════════════════════════════════════════════════════════

class TestDecisionProperties:
    """ReflexDecision objects are immutable and carry valid metadata."""

    def test_decision_is_frozen(self):
        d = ReflexDecision(
            decision=DecisionType.ALLOW,
            priority=Priority.P5_TELEMETRY,
            rule_name="Test",
            reason="test",
        )
        with pytest.raises(AttributeError):
            d.decision = DecisionType.WAIT  # type: ignore[misc]

    def test_decision_has_monotonic_timestamp(self):
        before = time.monotonic()
        d = ReflexDecision(
            decision=DecisionType.ALLOW,
            priority=Priority.P5_TELEMETRY,
            rule_name="Test",
            reason="test",
        )
        after = time.monotonic()
        assert before <= d.timestamp_mono <= after

    def test_str_representation(self):
        d = ReflexDecision(
            decision=DecisionType.WAIT,
            priority=Priority.P1_LIFECYCLE,
            rule_name="NavigationInProgress",
            reason="Main frame is navigating.",
        )
        s = str(d)
        assert "P1_LIFECYCLE" in s
        assert "WAIT" in s
        assert "NavigationInProgress" in s


# ══════════════════════════════════════════════════════════════════════
#  9. Evaluate Batch
# ══════════════════════════════════════════════════════════════════════

class TestEvaluateBatch:
    """evaluate_batch processes a list of actions sequentially."""

    def test_batch_all_allow(self):
        state = _make_ready_state()
        engine = ReflexEngine(state)
        decisions = engine.evaluate_batch(["click", "type", "scroll"])

        assert len(decisions) == 3
        assert all(d.decision == DecisionType.ALLOW for d in decisions)
        assert engine.decision_count == 3

    def test_batch_mixed(self):
        state = _make_ready_state()
        state.active_dialog = {"type": "alert", "message": "X"}
        engine = ReflexEngine(state)
        decisions = engine.evaluate_batch(["click", "accept_dialog"])

        assert decisions[0].decision == DecisionType.REPORT_BLOCKED
        assert decisions[1].decision == DecisionType.ROUTE_TO_KNOWN_HANDLER


# ══════════════════════════════════════════════════════════════════════
#  10. Synthetic Replay Timelines
# ══════════════════════════════════════════════════════════════════════

class TestReplayTimelines:
    """Replay synthetic CDP event sequences through WorldStateTracker → ReflexEngine."""

    def test_page_load_lifecycle(self):
        """
        Timeline: navigate → requests arrive → load fires → DOM settles → ALLOW.
        """
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.tracking_started_at_mono = time.monotonic() - 0.200

        # 1. Navigation starts → P1 WAIT
        tracker.handle_frame_navigated("https://example.com", is_main_frame=True)
        engine = ReflexEngine(state)
        d = engine.evaluate("click")
        assert d.decision == DecisionType.WAIT
        assert d.priority == Priority.P1_LIFECYCLE

        # 2. Load event fires, assign target → P2 (subtree churning because just assigned)
        tracker.handle_load_event_fired()
        state.last_dom_mutation_at_mono = time.monotonic() - 0.010  # Recent mutation
        state.assign_target(locator="#btn", node_id=50, is_actionable=True, is_stable=True)

        engine2 = ReflexEngine(state)
        d2 = engine2.evaluate("click")
        # Target just assigned on a recently-mutating page → subtree not quiet yet
        assert d2.decision == DecisionType.WAIT
        assert d2.priority == Priority.P2_TARGET_INTEGRITY

        # 3. After quiet period → ALLOW
        state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 0.200
        engine3 = ReflexEngine(state)
        d3 = engine3.evaluate("click")
        assert d3.decision == DecisionType.ALLOW

    def test_dialog_interrupt_and_recovery(self):
        """
        Timeline: ready → dialog opens → blocked → dialog closed → re-ready → ALLOW.
        """
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.tracking_started_at_mono = time.monotonic() - 0.200
        state.last_dom_mutation_at_mono = time.monotonic() - 0.200
        target = state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
        target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

        # 1. Confirm ready
        engine = ReflexEngine(state)
        assert engine.evaluate("click").decision == DecisionType.ALLOW

        # 2. Dialog opens → REPORT_BLOCKED
        tracker.handle_javascript_dialog_opening("confirm", "Delete?")
        d = engine.evaluate("click")
        assert d.decision == DecisionType.REPORT_BLOCKED

        # 3. accept_dialog → ROUTE_TO_KNOWN_HANDLER
        d = engine.evaluate("accept_dialog")
        assert d.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER

        # 4. Dialog dismissed → re-assign target, settle → ALLOW
        tracker.handle_javascript_dialog_closed()
        state.lifecycle_phase = "STABLE"
        target2 = state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
        target2.subtree_last_mutation_at_mono = time.monotonic() - 0.200
        d = engine.evaluate("click")
        assert d.decision == DecisionType.ALLOW

    def test_dom_churn_to_stable_transition(self):
        """
        Timeline: global DOM churn (ad ticker) → target subtree quiet → ALLOW despite global noise.
        """
        tracker = WorldStateTracker("tab")
        state = tracker.state
        state.tracking_started_at_mono = time.monotonic() - 0.200
        state.last_dom_mutation_at_mono = time.monotonic() - 0.200

        target = state.assign_target(locator="#buy", node_id=10, is_actionable=True, is_stable=True)
        target.subtree_last_mutation_at_mono = time.monotonic() - 0.200

        # Sibling churn (unrelated ad node 999)
        state.node_parent_map[10] = 1
        state.node_parent_map[999] = 1
        tracker.handle_character_data_modified(node_id=999, character_data="Ad refresh")

        # Global DOM is NOT quiet
        assert not state.is_global_dom_quiet
        assert not state.is_page_stable

        # But target subtree IS quiet → ALLOW
        engine = ReflexEngine(state)
        d = engine.evaluate("click")
        assert d.decision == DecisionType.ALLOW

    def test_navigation_to_document_invalidation(self):
        """
        Timeline: navigate → document updated → no target → INVALIDATE.
        """
        tracker = WorldStateTracker("tab")
        state = tracker.state

        state.assign_target(locator="#link", node_id=200, is_actionable=True, is_stable=True)

        tracker.handle_frame_navigated("https://new-page.com", is_main_frame=True)
        tracker.handle_document_updated()

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        # Document invalidation fires at P0
        assert d.decision == DecisionType.INVALIDATE
        assert d.priority == Priority.P0_SAFETY

    def test_overlay_then_resolved(self):
        """
        Timeline: overlay detected → ROUTE_TO_KNOWN_HANDLER → overlay clears → ALLOW.
        """
        state = _make_ready_state()
        state.blocking_overlay_detected = True
        state.known_overlay_match = "gdpr_consent"

        engine = ReflexEngine(state)
        d = engine.evaluate("click")
        assert d.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
        assert "gdpr_consent" in d.reason

        # Overlay cleared
        state.blocking_overlay_detected = False
        state.known_overlay_match = None
        d2 = engine.evaluate("click")
        assert d2.decision == DecisionType.ALLOW


# ══════════════════════════════════════════════════════════════════════
#  11. Priority Ordering Exhaustive Verification
# ══════════════════════════════════════════════════════════════════════

class TestPriorityOrdering:
    """Verify strict priority ordering among all tiers."""

    def test_p0_beats_p1_beats_p2_beats_p3_beats_p4(self):
        """Set up a state where ALL rules would fire; only P0 should win."""
        state = BrowserState()
        state.document_invalidated = True
        state.is_navigating = True
        state.blocking_overlay_detected = True
        # current_target is None

        engine = ReflexEngine(state)
        d = engine.evaluate("click")

        # P0 DocumentInvalidated fires first
        assert d.priority == Priority.P0_SAFETY
        assert d.decision == DecisionType.INVALIDATE

    def test_priority_enum_ordering(self):
        """Priority values should be numerically ordered."""
        assert Priority.P0_SAFETY.value < Priority.P1_LIFECYCLE.value
        assert Priority.P1_LIFECYCLE.value < Priority.P2_TARGET_INTEGRITY.value
        assert Priority.P2_TARGET_INTEGRITY.value < Priority.P3_OBSTRUCTION.value
        assert Priority.P3_OBSTRUCTION.value < Priority.P4_METHOD_E.value
        assert Priority.P4_METHOD_E.value < Priority.P5_TELEMETRY.value
