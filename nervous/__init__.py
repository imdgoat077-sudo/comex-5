"""
Comex Browser Nervous System - World-State Tracker & Reflex Subsystem.
"""

from .world_state import BrowserState, TargetState, WorldStateTracker
from .reflex_engine import DecisionType, Priority, ReflexDecision, ReflexEngine
from .actuator_gate import ActuatorGate, GateResult, LatencyTracker
from .target_binding import BindingFailureReason, TargetBinding, validate_binding_integrity
from .executor import SelectorClickExecutor
from .cdp_session import (
    TabSessionContext,
    WebSocketCDPClient,
    get_or_create_tab_session,
    close_tab_session,
    cleanup_worker_sessions,
    extract_canonical_target_id,
    get_canonical_tab_lock,
    reset_tab_locks,
)
from .method_e import METHOD_E_CHECK_FUNC, METHOD_E_JS
from .policy_registry import (
    OverlayPolicyMatcher,
    PolicyRegistry,
    ReflexActionRouter,
    SafeOverlayHandler,
    SafeOverlayPolicy,
    matches_domain,
    normalize_hostname,
)
from .action_policy import (
    TabLease,
    ServerTabLeaseRegistry,
    ServerTabTargetRegistry,
    GLOBAL_LEASE_REGISTRY,
    GLOBAL_TARGET_REGISTRY,
    get_lease_registry,
    get_target_registry,
    DEFAULT_LEASE_TTL,
    MAX_LEASE_TTL,
    PROHIBITED_FOREGROUND_ALIASES,
    validate_worker_tab_authorization,
    validate_harness_script_safety,
    validate_navigation_url,
)

__all__ = [
    "BrowserState",
    "TargetState",
    "WorldStateTracker",
    "DecisionType",
    "Priority",
    "ReflexDecision",
    "ReflexEngine",
    "ActuatorGate",
    "GateResult",
    "LatencyTracker",
    "BindingFailureReason",
    "TargetBinding",
    "validate_binding_integrity",
    "SelectorClickExecutor",
    "TabSessionContext",
    "WebSocketCDPClient",
    "get_or_create_tab_session",
    "close_tab_session",
    "cleanup_worker_sessions",
    "extract_canonical_target_id",
    "get_canonical_tab_lock",
    "reset_tab_locks",
    "validate_worker_tab_authorization",
    "TabLease",
    "ServerTabLeaseRegistry",
    "ServerTabTargetRegistry",
    "GLOBAL_LEASE_REGISTRY",
    "GLOBAL_TARGET_REGISTRY",
    "get_lease_registry",
    "get_target_registry",
    "DEFAULT_LEASE_TTL",
    "MAX_LEASE_TTL",
    "PROHIBITED_FOREGROUND_ALIASES",
    "METHOD_E_CHECK_FUNC",
    "METHOD_E_JS",
    "SafeOverlayPolicy",
    "PolicyRegistry",
    "OverlayPolicyMatcher",
    "ReflexActionRouter",
    "SafeOverlayHandler",
    "matches_domain",
    "normalize_hostname",
    "validate_harness_script_safety",
    "validate_navigation_url",
]

