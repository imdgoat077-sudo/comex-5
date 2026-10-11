"""
Tests for Stage D: Safe Policy Registry, Presence-First Matcher,
Segregated Action Router, and Non-Recursive SafeOverlayHandler.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from nervous.actuator_gate import ActionHandler, ActuatorGate, GateResult
from nervous.policy_registry import (
    OverlayPolicyMatcher,
    PolicyRegistry,
    ReflexActionRouter,
    SafeOverlayHandler,
    SafeOverlayPolicy,
    matches_domain,
    normalize_hostname,
)
from nervous.reflex_engine import DecisionType, Priority, ReflexDecision
from nervous.target_binding import TargetBinding
from nervous.world_state import BrowserState, TargetState


# ──────────────────────────────────────────────────────────────────────
#  1. Domain Matching Tests
# ──────────────────────────────────────────────────────────────────────

def test_domain_matching_universal_wildcard():
    assert matches_domain("*", "https://example.com") is True
    assert matches_domain("*", "https://foo.bar.com/path?query=1") is True
    assert matches_domain("*", "http://localhost:8000") is True
    assert matches_domain("*", "") is False


def test_domain_matching_exact():
    pattern = "example.com"
    assert matches_domain(pattern, "https://example.com") is True
    assert matches_domain(pattern, "https://example.com/some/path") is True
    assert matches_domain(pattern, "http://EXAMPLE.COM") is True
    # Subdomains should not match exact domain pattern
    assert matches_domain(pattern, "https://sub.example.com") is False
    # Suffix attacks should be rejected
    assert matches_domain(pattern, "https://notexample.com") is False
    assert matches_domain(pattern, "https://example.com.evil.org") is False


def test_domain_matching_subdomain_wildcard():
    pattern = "*.example.com"
    # Matches root domain and subdomains
    assert matches_domain(pattern, "https://example.com") is True
    assert matches_domain(pattern, "https://foo.example.com") is True
    assert matches_domain(pattern, "https://a.b.c.example.com/test") is True
    # Rejects sibling/malicious domains
    assert matches_domain(pattern, "https://notexample.com") is False
    assert matches_domain(pattern, "https://evil-example.com") is False
    assert matches_domain(pattern, "https://example.org") is False


def test_domain_matching_empty_or_malformed():
    assert matches_domain("", "https://example.com") is False
    assert matches_domain("example.com", "") is False
    assert matches_domain("example.com", "not_a_valid_url") is False
    assert normalize_hostname("") == ""


# ──────────────────────────────────────────────────────────────────────
#  2. Presence-Over-Eligibility Tests (OverlayPolicyMatcher)
# ──────────────────────────────────────────────────────────────────────

class MockCDPForMatcher:
    def __init__(self, existing_selectors: List[str]) -> None:
        self.existing_selectors = set(existing_selectors)

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        if method == "DOM.getDocument":
            return {"root": {"nodeId": 1}}
        if method == "DOM.querySelector":
            selector = params.get("selector", "")
            if selector in self.existing_selectors:
                return {"nodeId": 100 + abs(hash(selector)) % 1000}
            return {"nodeId": 0}
        return {}


@pytest.mark.asyncio
async def test_matcher_no_overlays_present():
    policy = SafeOverlayPolicy(
        policy_id="cookie_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#cookie-banner",
        dismiss_selector="#cookie-accept",
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=[])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is False
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_single_eligible_policy_present():
    policy = SafeOverlayPolicy(
        policy_id="cookie_v1",
        version="1.0.0",
        domain_pattern="example.com",
        overlay_selector="#cookie-banner",
        dismiss_selector="#cookie-accept",
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=["#cookie-banner"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match == "cookie_v1"


@pytest.mark.asyncio
async def test_matcher_single_disabled_policy_present():
    policy = SafeOverlayPolicy(
        policy_id="cookie_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#cookie-banner",
        dismiss_selector="#cookie-accept",
        enabled=False,
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=["#cookie-banner"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    # Physically present, but disabled -> must set blocking=True, match=None (fail closed)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_single_wrong_domain_present():
    policy = SafeOverlayPolicy(
        policy_id="cookie_v1",
        version="1.0.0",
        domain_pattern="otherdomain.com",
        overlay_selector="#cookie-banner",
        dismiss_selector="#cookie-accept",
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=["#cookie-banner"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_single_distilled_skill_present():
    policy = SafeOverlayPolicy(
        policy_id="skill_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#modal",
        dismiss_selector="#close",
        provenance="distilled_skill",
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=["#modal"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_single_unsafe_tier_present():
    policy = SafeOverlayPolicy(
        policy_id="unsafe_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#modal",
        dismiss_selector="#close",
        safety_tier="UNSAFE",
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=["#modal"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_single_non_click_action_present():
    policy = SafeOverlayPolicy(
        policy_id="type_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#modal",
        dismiss_selector="#close",
        action="type",
    )
    registry = PolicyRegistry([policy])
    cdp = MockCDPForMatcher(existing_selectors=["#modal"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_multiple_overlays_present_ambiguity():
    p1 = SafeOverlayPolicy(
        policy_id="cookie_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#cookie-banner",
        dismiss_selector="#cookie-accept",
    )
    p2 = SafeOverlayPolicy(
        policy_id="newsletter_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#newsletter-popup",
        dismiss_selector="#newsletter-close",
    )
    registry = PolicyRegistry([p1, p2])
    cdp = MockCDPForMatcher(existing_selectors=["#cookie-banner", "#newsletter-popup"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    # >1 present -> fail closed
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


@pytest.mark.asyncio
async def test_matcher_safe_plus_disabled_both_present():
    # Presence overrides eligibility! Even though one is eligible and one is disabled,
    # both are in DOM -> >1 present -> match must be None (REPORT_BLOCKED).
    p_safe = SafeOverlayPolicy(
        policy_id="safe_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#safe-banner",
        dismiss_selector="#dismiss",
        enabled=True,
    )
    p_disabled = SafeOverlayPolicy(
        policy_id="disabled_v1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#disabled-banner",
        dismiss_selector="#close",
        enabled=False,
    )
    registry = PolicyRegistry([p_safe, p_disabled])
    cdp = MockCDPForMatcher(existing_selectors=["#safe-banner", "#disabled-banner"])
    matcher = OverlayPolicyMatcher(registry, cdp)
    state = BrowserState(tab_id="tab-1", current_url="https://example.com")

    await matcher.evaluate_overlay_state(state)
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match is None


# ──────────────────────────────────────────────────────────────────────
#  3. ReflexActionRouter Segregation Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_dispatches_dialog_to_dialog_handler():
    dialog_handler = MagicMock()
    dialog_handler.handle.return_value = {"dialog": "accepted"}
    overlay_handler = MagicMock()

    router = ReflexActionRouter(dialog_handler=dialog_handler, overlay_handler=overlay_handler)
    state = BrowserState(tab_id="tab-1")
    decision = ReflexDecision(
        decision=DecisionType.ROUTE_TO_KNOWN_HANDLER,
        priority=Priority.P0_SAFETY,
        rule_name="DialogResolutionHandler",
        reason="Active dialog",
    )

    res = await router.handle("click", state, decision)
    assert res == {"dialog": "accepted"}
    dialog_handler.handle.assert_called_once_with("click", state, decision)
    overlay_handler.handle.assert_not_called()


@pytest.mark.asyncio
async def test_router_dispatches_overlay_to_overlay_handler():
    dialog_handler = MagicMock()
    overlay_handler = AsyncMock()
    overlay_handler.handle.return_value = {"status": "DISMISS_DISPATCHED"}

    router = ReflexActionRouter(dialog_handler=dialog_handler, overlay_handler=overlay_handler)
    state = BrowserState(tab_id="tab-1")
    decision = ReflexDecision(
        decision=DecisionType.ROUTE_TO_KNOWN_HANDLER,
        priority=Priority.P3_OBSTRUCTION,
        rule_name="KnownOverlayBlocking",
        reason="Known overlay",
    )

    res = await router.handle("click", state, decision)
    assert res == {"status": "DISMISS_DISPATCHED"}
    overlay_handler.handle.assert_awaited_once_with("click", state, decision)
    dialog_handler.handle.assert_not_called()


@pytest.mark.asyncio
async def test_router_unmapped_rule_returns_none():
    router = ReflexActionRouter()
    state = BrowserState(tab_id="tab-1")
    decision = ReflexDecision(
        decision=DecisionType.ROUTE_TO_KNOWN_HANDLER,
        priority=Priority.P5_TELEMETRY,
        rule_name="UnknownRule",
        reason="Unknown",
    )
    res = await router.handle("click", state, decision)
    assert res is None


# ──────────────────────────────────────────────────────────────────────
#  4. SafeOverlayHandler Defense-in-Depth & Readiness Tests
# ──────────────────────────────────────────────────────────────────────

class FakeSessionForHandler:
    def __init__(self, tab_id: str = "tab-1") -> None:
        self.tab_id = tab_id
        self.executor = MagicMock()
        self.executor.execute = AsyncMock(return_value={"executed": True, "cdp_status": "ok"})
        self.resolve_and_bind_result: Optional[TargetBinding] = None

    async def resolve_and_bind(self, selector: str) -> TargetBinding:
        if self.resolve_and_bind_result:
            return self.resolve_and_bind_result
        raise RuntimeError("No binding configured")


def _make_valid_binding(tab_id: str = "tab-1", node_id: int = 10, backend_node_id: int = 20, document_generation: int = 0) -> TargetBinding:
    return TargetBinding(
        tab_id=tab_id,
        target_id=tab_id,
        session_id=None,
        frame_id=None,
        selector="#dismiss-btn",
        node_id=node_id,
        backend_node_id=backend_node_id,
        document_generation=document_generation,
        target_subtree_mutation_generation=0,
        state_watermark=1,
        bounding_box={"x": 10.0, "y": 10.0, "width": 50.0, "height": 30.0},
        resolved_at_mono=100.0,
    )


@pytest.mark.asyncio
async def test_handler_missing_known_overlay_match_aborts():
    registry = PolicyRegistry()
    session = FakeSessionForHandler()
    handler = SafeOverlayHandler(registry, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match=None)
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    with pytest.raises(RuntimeError, match="missing state.known_overlay_match"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_p0_p1_guards():
    policy = SafeOverlayPolicy(
        policy_id="pol_1",
        version="1.0.0",
        domain_pattern="*",
        overlay_selector="#ov",
        dismiss_selector="#close",
    )
    registry = PolicyRegistry([policy])
    session = FakeSessionForHandler()
    handler = SafeOverlayHandler(registry, session)
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    # 1. document_invalidated
    state = BrowserState(tab_id="tab-1", known_overlay_match="pol_1", document_invalidated=True)
    with pytest.raises(RuntimeError, match="document invalidation"):
        await handler.handle("click", state, decision)

    # 2. active_dialog
    state = BrowserState(tab_id="tab-1", known_overlay_match="pol_1", active_dialog={"type": "alert"})
    with pytest.raises(RuntimeError, match="active JavaScript dialog"):
        await handler.handle("click", state, decision)

    # 3. is_navigating
    state = BrowserState(tab_id="tab-1", known_overlay_match="pol_1", is_navigating=True)
    with pytest.raises(RuntimeError, match="active navigation"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_policy_revalidation_guards():
    session = FakeSessionForHandler()
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    # Disabled policy
    p_dis = SafeOverlayPolicy("p_dis", "1.0", "*", "#ov", "#cl", enabled=False)
    reg = PolicyRegistry([p_dis])
    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p_dis", current_url="https://example.com")
    with pytest.raises(RuntimeError, match="is disabled"):
        await handler.handle("click", state, decision)

    # Wrong domain at execution time
    p_dom = SafeOverlayPolicy("p_dom", "1.0", "trusted.com", "#ov", "#cl")
    reg = PolicyRegistry([p_dom])
    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p_dom", current_url="https://evil.com")
    with pytest.raises(RuntimeError, match="does not match URL"):
        await handler.handle("click", state, decision)

    # Non-builtin
    p_skill = SafeOverlayPolicy("p_skill", "1.0", "*", "#ov", "#cl", provenance="distilled_skill")
    reg = PolicyRegistry([p_skill])
    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p_skill", current_url="https://example.com")
    with pytest.raises(RuntimeError, match="provenance is not 'builtin'"):
        await handler.handle("click", state, decision)

    # Non-SAFE tier
    p_unsafe = SafeOverlayPolicy("p_unsafe", "1.0", "*", "#ov", "#cl", safety_tier="UNSAFE")
    reg = PolicyRegistry([p_unsafe])
    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p_unsafe", current_url="https://example.com")
    with pytest.raises(RuntimeError, match="tier is not 'SAFE_AUTONOMIC'"):
        await handler.handle("click", state, decision)

    # Action != click
    p_type = SafeOverlayPolicy("p_type", "1.0", "*", "#ov", "#cl", action="type")
    reg = PolicyRegistry([p_type])
    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p_type", current_url="https://example.com")
    with pytest.raises(RuntimeError, match="action is not 'click'"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_binding_integrity_failure_aborts():
    policy = SafeOverlayPolicy("p1", "1.0", "*", "#ov", "#cl")
    reg = PolicyRegistry([policy])
    session = FakeSessionForHandler(tab_id="tab-1")
    # Binding tab_id is tab-2, but session is tab-1 -> integrity failure
    binding = _make_valid_binding(tab_id="tab-2")
    session.resolve_and_bind_result = binding
    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p1", current_url="https://example.com")
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    with pytest.raises(RuntimeError, match="target binding integrity failed"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_identity_mismatch_aborts():
    policy = SafeOverlayPolicy("p1", "1.0", "*", "#ov", "#cl")
    reg = PolicyRegistry([policy])
    session = FakeSessionForHandler(tab_id="tab-1")
    binding = _make_valid_binding(tab_id="tab-1", node_id=10, backend_node_id=20, document_generation=0)
    session.resolve_and_bind_result = binding

    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p1", current_url="https://example.com")
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    # Subtest 1: target is None
    state.current_target = None
    with pytest.raises(RuntimeError, match="no current_target assigned"):
        await handler.handle("click", state, decision)

    # Subtest 2: Target assigned with matching node_id/backend_node_id, but target.document_generation drifted
    state.assign_target(
        locator="#cl",
        node_id=10,
        backend_node_id=20,
        is_actionable=True,
        is_stable=True,
        bounding_box={"x": 0, "y": 0, "width": 10, "height": 10},
    )
    state.current_target.document_generation = 99  # Drifted target generation
    with pytest.raises(RuntimeError, match="TargetBinding identity mismatch"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_geometric_instability_aborts():
    policy = SafeOverlayPolicy("p1", "1.0", "*", "#ov", "#cl")
    reg = PolicyRegistry([policy])
    session = FakeSessionForHandler(tab_id="tab-1")
    binding = _make_valid_binding(tab_id="tab-1", node_id=10, backend_node_id=20)
    session.resolve_and_bind_result = binding

    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p1", current_url="https://example.com")
    state.assign_target(
        locator="#cl",
        node_id=10,
        backend_node_id=20,
        is_actionable=True,
        is_stable=False,  # Unstable / moving target!
        bounding_box={"x": 10, "y": 10, "width": 50, "height": 30},
    )
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    with pytest.raises(RuntimeError, match="geometrically unstable"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_subtree_churn_aborts():
    policy = SafeOverlayPolicy("p1", "1.0", "*", "#ov", "#cl")
    reg = PolicyRegistry([policy])
    session = FakeSessionForHandler(tab_id="tab-1")
    binding = _make_valid_binding(tab_id="tab-1", node_id=10, backend_node_id=20)
    session.resolve_and_bind_result = binding

    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(tab_id="tab-1", known_overlay_match="p1", current_url="https://example.com")
    state.assign_target(
        locator="#cl",
        node_id=10,
        backend_node_id=20,
        is_actionable=True,
        is_stable=True,
        bounding_box={"x": 10, "y": 10, "width": 50, "height": 30},
    )
    # Simulate subtree mutation within quietness window (< 80ms)
    import time
    state.current_target.subtree_last_mutation_at_mono = time.monotonic()
    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    with pytest.raises(RuntimeError, match="subtree is actively churning"):
        await handler.handle("click", state, decision)


@pytest.mark.asyncio
async def test_handler_happy_path_executes_click():
    import time
    policy = SafeOverlayPolicy("p1", "1.0", "*", "#ov", "#cl")
    reg = PolicyRegistry([policy])
    session = FakeSessionForHandler(tab_id="tab-1")
    binding = _make_valid_binding(tab_id="tab-1", node_id=10, backend_node_id=20)
    session.resolve_and_bind_result = binding

    handler = SafeOverlayHandler(reg, session)
    state = BrowserState(
        tab_id="tab-1",
        known_overlay_match="p1",
        blocking_overlay_detected=True,
        current_url="https://example.com",
    )
    state.assign_target(
        locator="#cl",
        node_id=10,
        backend_node_id=20,
        is_actionable=True,
        is_stable=True,
        bounding_box={"x": 10, "y": 10, "width": 50, "height": 30},
    )
    # Target subtree is quiet (> 80ms)
    state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 1.0

    decision = ReflexDecision(DecisionType.ROUTE_TO_KNOWN_HANDLER, Priority.P3_OBSTRUCTION, "KnownOverlayBlocking", "")

    res = await handler.handle("click", state, decision)
    assert res["status"] == "DISMISS_DISPATCHED"
    assert res["policy_id"] == "p1"
    assert res["action"] == "click"
    session.executor.execute.assert_awaited_once_with("click", binding=binding)

    # CRITICAL: Handler must NOT clear overlay state; removal verified on next evaluation
    assert state.blocking_overlay_detected is True
    assert state.known_overlay_match == "p1"


# ──────────────────────────────────────────────────────────────────────
#  5. ActuatorGate Integration with Stage D
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_actuator_gate_known_overlay_routes_to_router():
    import time
    state = BrowserState(
        tab_id="tab-1",
        blocking_overlay_detected=True,
        known_overlay_match="cookie_v1",
    )
    state.assign_target("#buy-btn", node_id=1, backend_node_id=1, is_actionable=False, is_stable=True)
    state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 1.0

    fake_handler = MagicMock()
    fake_handler.handle.return_value = {"status": "DISMISS_DISPATCHED"}
    mock_executor = MagicMock()

    gate = ActuatorGate(state, mock_executor, handler=fake_handler)
    result = gate.attempt("click")

    assert result.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
    assert result.decision.rule_name == "KnownOverlayBlocking"
    assert result.decision.priority == Priority.P3_OBSTRUCTION
    assert result.executed is False  # Zero direct dispatches!
    assert result.handler_invoked is True
    assert result.execution_result == {"status": "DISMISS_DISPATCHED"}
    mock_executor.execute.assert_not_called()
    fake_handler.handle.assert_called_once()


@pytest.mark.asyncio
async def test_actuator_gate_unmatched_overlay_blocks():
    import time
    state = BrowserState(
        tab_id="tab-1",
        blocking_overlay_detected=True,
        known_overlay_match=None,  # Unmatched or multiple overlays present
    )
    state.assign_target("#buy-btn", node_id=1, backend_node_id=1, is_actionable=False, is_stable=True)
    state.current_target.subtree_last_mutation_at_mono = time.monotonic() - 1.0

    fake_handler = MagicMock()
    mock_executor = MagicMock()

    gate = ActuatorGate(state, mock_executor, handler=fake_handler)
    result = gate.attempt("click")

    assert result.decision.decision == DecisionType.REPORT_BLOCKED
    assert result.decision.rule_name == "UnmatchedOverlayBlocking"
    assert result.decision.priority == Priority.P3_OBSTRUCTION
    assert result.executed is False
    assert result.handler_invoked is False
    assert result.was_blocked is True
    mock_executor.execute.assert_not_called()
    fake_handler.handle.assert_not_called()

    assert result.was_blocked is True
    mock_executor.execute.assert_not_called()
    fake_handler.handle.assert_not_called()
