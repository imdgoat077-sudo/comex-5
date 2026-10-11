"""
Comex Browser Nervous System — Stage D: Safe Policy Registry & Autonomic Overlay Subsystem.

Provides declarative, pre-approved SafeOverlayPolicy management, domain matching,
presence-first overlay detection, segregated reflex action routing, and non-recursive,
defense-in-depth overlay dismissal execution.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
from urllib.parse import urlparse

from .actuator_gate import ActionHandler
from .reflex_engine import DecisionType, ReflexDecision
from .target_binding import BindingFailureReason, TargetBinding, validate_binding_integrity
from .world_state import BrowserState

logger = logging.getLogger("comex.nervous.policy_registry")


# ──────────────────────────────────────────────────────────────────────
#  Domain Matching Utilities
# ──────────────────────────────────────────────────────────────────────

def normalize_hostname(url: str) -> str:
    """Extract and normalize lowercase hostname from URL without port."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower().strip()
    except Exception:
        return ""


def matches_domain(domain_pattern: str, url: str) -> bool:
    """
    Deterministic domain pattern matching.
    Supports:
      - '*' : universal wildcard matching any valid HTTP/HTTPS host.
      - Exact hostname matching: 'example.com' matches 'https://example.com/'.
      - Subdomain wildcard '*.example.com': matches 'example.com', 'foo.example.com',
        'a.b.example.com'; strictly rejects 'notexample.com' or 'evil-example.com'.
    """
    if not domain_pattern or not url:
        return False

    pattern = domain_pattern.strip().lower()
    host = normalize_hostname(url)
    if not host:
        return False

    if pattern == "*":
        return True

    if pattern == host:
        return True

    if pattern.startswith("*."):
        suffix = pattern[2:]
        if host == suffix:
            return True
        if host.endswith("." + suffix):
            return True

    return False


# ──────────────────────────────────────────────────────────────────────
#  Declarative SafeOverlayPolicy Schema
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SafeOverlayPolicy:
    """
    Declarative specification for pre-approved autonomic overlay dismissal.
    Strictly contains no executable code or arbitrary callbacks.
    """
    policy_id: str
    version: str
    domain_pattern: str
    overlay_selector: str
    dismiss_selector: str
    action: str = "click"
    enabled: bool = True
    safety_tier: str = "SAFE_AUTONOMIC"
    provenance: str = "builtin"  # "builtin" | "distilled_skill"


# ──────────────────────────────────────────────────────────────────────
#  Static / Pre-Configured Policy Registry
# ──────────────────────────────────────────────────────────────────────

class PolicyRegistry:
    """
    Static catalog of pre-approved SafeOverlayPolicy specifications.
    Precludes arbitrary runtime additions by agents or prompts.
    """

    def __init__(self, initial_policies: Optional[List[SafeOverlayPolicy]] = None) -> None:
        self._policies: Dict[str, SafeOverlayPolicy] = {}
        if initial_policies:
            for policy in initial_policies:
                self._policies[policy.policy_id] = policy

    def register_policy(self, policy: SafeOverlayPolicy) -> None:
        """Register a pre-approved policy specification."""
        self._policies[policy.policy_id] = policy

    def get_policy(self, policy_id: str) -> Optional[SafeOverlayPolicy]:
        return self._policies.get(policy_id)

    def get_all_policies(self) -> List[SafeOverlayPolicy]:
        return list(self._policies.values())

    def set_policy_enabled(self, policy_id: str, enabled: bool) -> None:
        """Explicit toggle for policy enablement."""
        existing = self._policies.get(policy_id)
        if existing:
            updated = SafeOverlayPolicy(
                policy_id=existing.policy_id,
                version=existing.version,
                domain_pattern=existing.domain_pattern,
                overlay_selector=existing.overlay_selector,
                dismiss_selector=existing.dismiss_selector,
                action=existing.action,
                enabled=enabled,
                safety_tier=existing.safety_tier,
                provenance=existing.provenance,
            )
            self._policies[policy_id] = updated


# ──────────────────────────────────────────────────────────────────────
#  Presence-First Overlay Policy Matcher
# ──────────────────────────────────────────────────────────────────────

class OverlayPolicyMatcher:
    """
    Evaluates overlay presence and policy eligibility against the live DOM.
    Enforces presence-over-eligibility: ALL registered overlay signatures
    are scanned in the DOM first. Multiple overlays or an ineligible overlay
    unconditionally produce REPORT_BLOCKED.
    """

    def __init__(self, registry: PolicyRegistry, cdp_client: Any) -> None:
        self.registry = registry
        self.cdp = cdp_client

    async def _check_selector_exists(self, selector: str) -> bool:
        """Probes DOM existence of selector via CDP."""
        try:
            doc_res = await self.cdp.send("DOM.getDocument", {"depth": 0})
            root_node_id = doc_res.get("root", {}).get("nodeId", 0)
            if not root_node_id:
                return False

            query_res = await self.cdp.send("DOM.querySelector", {
                "nodeId": root_node_id,
                "selector": selector,
            })
            node_id = query_res.get("nodeId", 0)
            return node_id > 0
        except Exception as e:
            logger.debug("Overlay selector existence probe error for '%s': %s", selector, e)
            return False

    async def evaluate_overlay_state(self, state: BrowserState) -> None:
        """
        Runs per-action immediately after the event barrier.
        Updates state.blocking_overlay_detected and state.known_overlay_match ephemerally.
        """
        all_policies = self.registry.get_all_policies()
        present_policies: List[SafeOverlayPolicy] = []

        # Step 1: Scan for ALL physically present overlays in the DOM regardless of eligibility
        for policy in all_policies:
            if await self._check_selector_exists(policy.overlay_selector):
                present_policies.append(policy)

        if len(present_policies) == 0:
            state.blocking_overlay_detected = False
            state.known_overlay_match = None
            return

        # An overlay is physically present!
        state.blocking_overlay_detected = True

        if len(present_policies) > 1:
            # Ambiguity: multiple overlays present -> fail closed
            state.known_overlay_match = None
            return

        # Exactly 1 overlay is present: check full eligibility criteria
        single_policy = present_policies[0]
        if (
            single_policy.enabled
            and single_policy.safety_tier == "SAFE_AUTONOMIC"
            and single_policy.provenance == "builtin"
            and single_policy.action == "click"
            and matches_domain(single_policy.domain_pattern, state.current_url)
        ):
            state.known_overlay_match = single_policy.policy_id
        else:
            # Present but ineligible (e.g. wrong domain, disabled, distilled, or unsafe)
            state.known_overlay_match = None


# ──────────────────────────────────────────────────────────────────────
#  Segregated ReflexActionRouter
# ──────────────────────────────────────────────────────────────────────

class ReflexActionRouter:
    """
    Implements the ActionHandler protocol for ActuatorGate.
    Segregates P0 dialog handlers from P3 safe overlay policy handlers.
    """

    def __init__(
        self,
        dialog_handler: Optional[ActionHandler] = None,
        overlay_handler: Optional[ActionHandler] = None,
    ) -> None:
        self.dialog_handler = dialog_handler
        self.overlay_handler = overlay_handler

    async def _invoke(
        self,
        handler: ActionHandler,
        action: str,
        state: BrowserState,
        decision: ReflexDecision,
        **kwargs: Any,
    ) -> Any:
        res = handler.handle(action, state, decision, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    async def handle(
        self,
        action: str,
        state: BrowserState,
        decision: ReflexDecision,
        **kwargs: Any,
    ) -> Any:
        if decision.rule_name == "DialogResolutionHandler":
            if self.dialog_handler is not None:
                return await self._invoke(self.dialog_handler, action, state, decision, **kwargs)
            logger.warning("ReflexActionRouter: DialogResolutionHandler fired but no dialog_handler configured")
            return None

        if decision.rule_name == "KnownOverlayBlocking":
            if self.overlay_handler is not None:
                return await self._invoke(self.overlay_handler, action, state, decision, **kwargs)
            logger.warning("ReflexActionRouter: KnownOverlayBlocking fired but no overlay_handler configured")
            return None

        logger.warning("ReflexActionRouter received unmapped rule: %s", decision.rule_name)
        return None


# ──────────────────────────────────────────────────────────────────────
#  Non-Recursive SafeOverlayHandler (Defense-in-Depth)
# ──────────────────────────────────────────────────────────────────────

class SafeOverlayHandler:
    """
    Non-recursive handler executing pre-approved overlay dismissal clicks.
    Enforces complete policy authorization, P0/P1 preconditions, TargetBinding
    integrity, target-local stability, and subtree quiescence before direct dispatch.
    """

    def __init__(self, registry: PolicyRegistry, session: Any) -> None:
        self.registry = registry
        self.session = session

    async def handle(
        self,
        action: str,
        state: BrowserState,
        decision: ReflexDecision,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        policy_id = state.known_overlay_match
        if not policy_id:
            raise RuntimeError("SafeOverlayHandler: missing state.known_overlay_match")

        # 1. State Precondition Guard (P0/P1 Check)
        if state.document_invalidated:
            raise RuntimeError("SafeOverlayHandler: aborted due to document invalidation (P0)")
        if state.active_dialog is not None:
            raise RuntimeError("SafeOverlayHandler: aborted due to active JavaScript dialog (P0)")
        if state.is_navigating:
            raise RuntimeError("SafeOverlayHandler: aborted due to active navigation (P1)")

        # 2. Complete Policy Authorization Guard
        policy = self.registry.get_policy(policy_id)
        if not policy:
            raise RuntimeError(f"SafeOverlayHandler: policy '{policy_id}' not found in registry")
        if not policy.enabled:
            raise RuntimeError(f"SafeOverlayHandler: policy '{policy_id}' is disabled")
        if policy.provenance != "builtin":
            raise RuntimeError(f"SafeOverlayHandler: policy '{policy_id}' provenance is not 'builtin'")
        if policy.safety_tier != "SAFE_AUTONOMIC":
            raise RuntimeError(f"SafeOverlayHandler: policy '{policy_id}' tier is not 'SAFE_AUTONOMIC'")
        if policy.action != "click":
            raise RuntimeError(f"SafeOverlayHandler: policy '{policy_id}' action is not 'click'")
        if not matches_domain(policy.domain_pattern, state.current_url):
            raise RuntimeError(f"SafeOverlayHandler: policy '{policy_id}' does not match URL '{state.current_url}'")

        # 3. Coherent Target Binding & Method E Evaluation
        binding = await self.session.resolve_and_bind(policy.dismiss_selector)

        # 4. TargetBinding Integrity Verification
        integrity_failure = validate_binding_integrity(
            binding, state, current_tab_id=getattr(self.session, "tab_id", state.tab_id)
        )
        if integrity_failure is not None:
            raise RuntimeError(f"SafeOverlayHandler: target binding integrity failed: {integrity_failure.name}")

        # 5. Exact Binding-to-Readiness Identity Enforcement
        target = state.current_target
        if target is None:
            raise RuntimeError("SafeOverlayHandler: no current_target assigned in state")

        if (
            binding.backend_node_id != target.backend_node_id
            or binding.node_id != target.node_id
            or binding.document_generation != target.document_generation
            or binding.tab_id != getattr(self.session, "tab_id", state.tab_id)
        ):
            raise RuntimeError(
                f"SafeOverlayHandler: TargetBinding identity mismatch! "
                f"Binding(backend={binding.backend_node_id}, doc_gen={binding.document_generation}) != "
                f"CurrentTarget(backend={target.backend_node_id}, doc_gen={target.document_generation})"
            )

        # Ensure observation window has elapsed since binding to observe live mutations/movement
        elapsed = time.monotonic() - binding.resolved_at_mono
        min_observation = 0.035  # 35ms covers 15ms DOM churn and 25ms geometric movement
        if elapsed < min_observation:
            await asyncio.sleep(min_observation - elapsed)

        # Flush CDP event queue to process all pending events
        if hasattr(self.session, "flush_and_sync"):
            res = self.session.flush_and_sync()
            if inspect.isawaitable(res):
                await res

        # Sample consecutive position observation before final stability decision
        if hasattr(self.session, "update_target_position"):
            res = self.session.update_target_position()
            if inspect.isawaitable(res):
                await res

        # 6. Target-Local Geometric Stability Check
        if not target.is_stable:
            raise RuntimeError("SafeOverlayHandler: dismiss target is geometrically unstable (moving)")

        # 7. Target-Local Subtree Quiescence Check
        if not state.is_target_subtree_quiet:
            raise RuntimeError("SafeOverlayHandler: dismiss target subtree is actively churning")

        # 8. Direct Physical Dispatch via Executor
        exec_res = await self.session.executor.execute(policy.action, binding=binding)

        # 9. Return execution summary without assuming overlay is gone
        return {
            "status": "DISMISS_DISPATCHED",
            "policy_id": policy.policy_id,
            "action": policy.action,
            "execution_result": exec_res,
        }


__all__ = [
    "normalize_hostname",
    "matches_domain",
    "SafeOverlayPolicy",
    "PolicyRegistry",
    "OverlayPolicyMatcher",
    "ReflexActionRouter",
    "SafeOverlayHandler",
]
