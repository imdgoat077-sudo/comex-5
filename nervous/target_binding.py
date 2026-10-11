"""
Comex Browser Nervous System — Stage C: Target Binding & Invalidation.

Defines the immutable TargetBinding data structure and fail-closed
invalidation predicates for the Controlled Mutation Plane.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .world_state import BrowserState


class BindingFailureReason(str, Enum):
    """Explicit fail-closed invalidation classifications."""
    DOCUMENT_GENERATION_MISMATCH = "DOCUMENT_GENERATION_MISMATCH"
    SUBTREE_MUTATION_AFTER_BINDING = "SUBTREE_MUTATION_AFTER_BINDING"
    SESSION_TARGET_MISMATCH = "SESSION_TARGET_MISMATCH"
    NODE_DETACHED = "NODE_DETACHED"
    NODE_IDENTITY_MISMATCH = "NODE_IDENTITY_MISMATCH"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True)
class TargetBinding:
    """
    Immutable snapshot of the resolved target at authorization time.
    Binds the pure Method E evaluation to the exact DOM object and
    backend_node_id identity used by the actuator.
    """
    tab_id: str
    target_id: str
    session_id: Optional[str]
    frame_id: Optional[str]
    selector: str
    node_id: int                                # CDP DOM NodeId (Stage A event/ancestry tracking)
    backend_node_id: int                        # CDP DOM BackendNodeId (Actuator operations)
    document_generation: int                    # Page navigation/document invalidation counter
    target_subtree_mutation_generation: int     # Target-local mutation counter
    state_watermark: int                        # Event processing sequence watermark
    bounding_box: Dict[str, float]              # {'x': float, 'y': float, 'width': float, 'height': float}
    resolved_at_mono: float                     # time.monotonic() at resolution completion


def validate_binding_integrity(
    binding: TargetBinding,
    state: BrowserState,
    current_tab_id: Optional[str] = None,
    current_target_id: Optional[str] = None,
) -> Optional[BindingFailureReason]:
    """
    Evaluate target-local invalidation rules against the current BrowserState.
    Returns None if the binding is intact and valid, or a BindingFailureReason
    indicating why execution must abort.
    """
    # 1. Session / Target ID isolation
    if current_tab_id is not None and current_tab_id != binding.tab_id:
        return BindingFailureReason.SESSION_TARGET_MISMATCH
    if current_target_id is not None and current_target_id != binding.target_id:
        return BindingFailureReason.SESSION_TARGET_MISMATCH

    # 2. Document Generation (reloads, navigations, documentUpdated)
    if state.document_generation != binding.document_generation or state.document_invalidated:
        return BindingFailureReason.DOCUMENT_GENERATION_MISMATCH

    # 3. Target Subtree Mutations after binding point
    if state.target_subtree_mutation_generation != binding.target_subtree_mutation_generation:
        return BindingFailureReason.SUBTREE_MUTATION_AFTER_BINDING

    # 4. Target Identity verification
    if state.current_target is not None:
        if (
            state.current_target.backend_node_id is not None
            and state.current_target.backend_node_id != binding.backend_node_id
        ):
            return BindingFailureReason.NODE_IDENTITY_MISMATCH
        if (
            state.current_target.node_id is not None
            and state.current_target.node_id != binding.node_id
        ):
            return BindingFailureReason.NODE_IDENTITY_MISMATCH

    return None
