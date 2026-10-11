"""
Comex Browser Nervous System - World-State Tracker (Stage A).
Event-driven BrowserState and WorldStateTracker modeling real-time CDP page,
DOM, and network lifecycles without whole-page DOM snapshot overhead.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Set


@dataclass
class TargetState:
    locator: Optional[str] = None
    node_id: Optional[int] = None                      # CDP DOM NodeId
    backend_node_id: Optional[int] = None              # CDP DOM BackendNodeId
    bounding_box: Optional[Dict[str, float]] = None    # {'x': float, 'y': float, 'width': float, 'height': float}
    is_actionable: bool = False                        # Evaluated via Frozen Method E
    is_stable: bool = False                            # Temporal/geometric stability evaluation
    last_position_delta: float = 0.0
    last_stable_at_mono: float = 0.0                   # time.monotonic()
    
    # Target Subtree Baselines
    assigned_at_mono: float = field(default_factory=time.monotonic)
    subtree_last_mutation_at_mono: float = field(default_factory=time.monotonic)
    ancestor_node_ids: Set[int] = field(default_factory=set)
    document_generation: int = 0
    target_subtree_mutation_generation: int = 0
    state_watermark: int = 0


@dataclass
class BrowserState:
    # 1. Identity & Routing
    tab_id: str = "default_tab"
    target_ws_url: str = "ws://localhost:9222/devtools/page/default"
    current_url: str = "about:blank"
    page_title: str = ""
    frame_tree_depth: int = 1
    
    # 2. Lifecycle & Navigation Phase
    lifecycle_phase: str = "UNKNOWN"                   # UNKNOWN, NAVIGATING, LOADING, DOM_SETTLING, STABLE, ACTIONABLE, BLOCKED
    is_navigating: bool = False
    document_invalidated: bool = False                  # Explicit flag set by DOM.documentUpdated
    document_generation: int = 0                       # Monotonic counter incremented on DOM.documentUpdated
    navigation_started_at_mono: float = 0.0
    tracking_started_at_mono: float = field(default_factory=time.monotonic)
    
    # 3. Robust Network Bookkeeping
    active_requests: Set[str] = field(default_factory=set)
    blocking_requests: Set[str] = field(default_factory=set)
    last_network_activity_at_mono: float = 0.0
    
    # 4. DOM Dynamics, Ancestry & Scoped Quiescence
    dom_revision: int = 0
    target_subtree_mutation_generation: int = 0        # Monotonic counter incremented on target subtree mutations
    state_watermark: int = 0                           # Monotonic event sequence applied
    last_dom_mutation_at_mono: float = 0.0
    mutation_types_recent: Set[str] = field(default_factory=set)
    node_parent_map: Dict[int, int] = field(default_factory=dict)
    
    # 5. Viewport & Geometry
    viewport_width: int = 1280
    viewport_height: int = 800
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    device_pixel_ratio: float = 1.0
    
    # 6. Environmental Blockers & Dialogs
    active_dialog: Optional[Dict[str, Any]] = None
    blocking_overlay_detected: bool = False
    known_overlay_match: Optional[str] = None
    
    # 7. Action Target Tracking
    current_target: Optional[TargetState] = None

    # ------------------------------------------------------------------
    # Network Handlers
    # ------------------------------------------------------------------
    def on_request_will_be_sent(self, request_id: str, resource_type: str = "Other") -> None:
        now = time.monotonic()
        self.active_requests.add(request_id)
        self.last_network_activity_at_mono = now
        # Blocking resource types: Document, Script, XHR, Fetch, Stylesheet
        if resource_type in {"Document", "Script", "XHR", "Fetch", "Stylesheet"}:
            self.blocking_requests.add(request_id)

    def on_request_finished_or_failed(self, request_id: str) -> None:
        now = time.monotonic()
        self.active_requests.discard(request_id)
        self.blocking_requests.discard(request_id)
        self.last_network_activity_at_mono = now

    @property
    def network_in_flight(self) -> int:
        return len(self.active_requests)

    @property
    def blocking_network_in_flight(self) -> int:
        return len(self.blocking_requests)

    # ------------------------------------------------------------------
    # Independent Sensory Predicates
    # ------------------------------------------------------------------
    @property
    def has_trustworthy_dom_history(self) -> bool:
        """True if the state tracker has observed the DOM event stream for >= 80ms."""
        return (time.monotonic() - self.tracking_started_at_mono) >= 0.080

    @property
    def has_no_blocking_requests(self) -> bool:
        """True if zero core network requests (Document, Script, XHR, Fetch) are in flight."""
        return len(self.blocking_requests) == 0

    @property
    def is_network_activity_quiet(self) -> bool:
        """True if no network event has occurred within the last 150ms window."""
        return (time.monotonic() - self.last_network_activity_at_mono) >= 0.150

    @property
    def is_global_dom_quiet(self) -> bool:
        """True if no global DOM mutations have occurred within the last 100ms window."""
        return (time.monotonic() - self.last_dom_mutation_at_mono) >= 0.100

    @property
    def is_target_subtree_quiet(self) -> bool:
        """
        True if the target subtree has experienced zero mutations for >= 80ms.
        Uses observed stability rather than an artificial post-assignment sleep.
        """
        if not self.current_target:
            return True
        return (time.monotonic() - self.current_target.subtree_last_mutation_at_mono) >= 0.080

    @property
    def is_target_stable(self) -> bool:
        """True if target exists, is actionable, and has stopped moving across consecutive frames."""
        if not self.current_target:
            return False
        return self.current_target.is_actionable and self.current_target.is_stable

    @property
    def is_page_stable(self) -> bool:
        """Informational global page stability: global DOM quiet, no blocking requests, not navigating."""
        return self.is_global_dom_quiet and self.has_no_blocking_requests and not self.is_navigating

    @property
    def is_action_ready(self) -> bool:
        """
        Target-local readiness gate:
        Target passes Method E, is temporally stable, subtree is quiet, no active dialog, not navigating.
        Does NOT require global page stability.
        """
        if not self.current_target:
            return False
        return (
            self.current_target.is_actionable
            and self.current_target.is_stable
            and self.is_target_subtree_quiet
            and (self.active_dialog is None)
            and not self.is_navigating
        )

    # ------------------------------------------------------------------
    # Target Lifecycle & Ancestry
    # ------------------------------------------------------------------
    def assign_target(
        self,
        locator: str,
        node_id: Optional[int] = None,
        backend_node_id: Optional[int] = None,
        ancestor_ids: Optional[List[int]] = None,
        is_actionable: bool = False,
        is_stable: bool = False,
        bounding_box: Optional[Dict[str, float]] = None,
    ) -> TargetState:
        now = time.monotonic()
        ancestor_set: Set[int] = set(ancestor_ids or [])
        
        # Populate parent map for known ancestors in hierarchy order if provided
        if ancestor_ids and len(ancestor_ids) > 1:
            for i in range(len(ancestor_ids) - 1):
                child_id = ancestor_ids[i]
                parent_id = ancestor_ids[i + 1]
                self.node_parent_map[child_id] = parent_id
        if node_id and ancestor_ids:
            self.node_parent_map[node_id] = ancestor_ids[0]

        # Observed stability inheritance (Zero artificial delay)
        # If tracker has trustworthy history and DOM has been quiet globally for >= 80ms:
        if self.has_trustworthy_dom_history and (now - self.last_dom_mutation_at_mono >= 0.080):
            subtree_last_mutation = self.last_dom_mutation_at_mono
        else:
            subtree_last_mutation = now

        target = TargetState(
            locator=locator,
            node_id=node_id,
            backend_node_id=backend_node_id,
            bounding_box=bounding_box,
            is_actionable=is_actionable,
            is_stable=is_stable,
            assigned_at_mono=now,
            subtree_last_mutation_at_mono=subtree_last_mutation,
            ancestor_node_ids=ancestor_set,
            document_generation=self.document_generation,
            target_subtree_mutation_generation=self.target_subtree_mutation_generation,
            state_watermark=self.state_watermark,
        )
        self.current_target = target
        # Target successfully resolved → document is no longer in invalidated state
        self.document_invalidated = False
        return target

    def update_target_position(self, x: float, y: float, width: float, height: float) -> None:
        """Update geometric coordinates of the target and calculate positional stability."""
        if not self.current_target:
            return
        
        now = time.monotonic()
        prev_box = self.current_target.bounding_box
        if prev_box is not None:
            dx = x - prev_box.get("x", x)
            dy = y - prev_box.get("y", y)
            delta = math.sqrt(dx * dx + dy * dy)
            self.current_target.last_position_delta = delta
            # If delta < 1.0px, target is temporally/geometrically stable
            if delta < 1.0:
                self.current_target.is_stable = True
                if self.current_target.last_stable_at_mono == 0.0:
                    self.current_target.last_stable_at_mono = now
            else:
                self.current_target.is_stable = False
                self.current_target.last_stable_at_mono = 0.0
        else:
            self.current_target.last_position_delta = 0.0
            self.current_target.is_stable = True
            self.current_target.last_stable_at_mono = now

        self.current_target.bounding_box = {"x": x, "y": y, "width": width, "height": height}

    def invalidate_target(self) -> None:
        """Clear current target state."""
        self.current_target = None

    def is_node_in_target_subtree(self, start_node_id: int) -> bool:
        """
        Deterministically traverse upward via node_parent_map to check if start_node_id
        resolves to the target node or is within the target's descendant subtree.
        """
        if not self.current_target or self.current_target.node_id is None:
            return False
        
        target_id = self.current_target.node_id
        curr: Optional[int] = start_node_id
        visited: Set[int] = set()
        
        while curr is not None and curr not in visited:
            if curr == target_id:
                return True
            visited.add(curr)
            curr = self.node_parent_map.get(curr)
        return False

    def _mark_target_subtree_mutated(self) -> None:
        self.target_subtree_mutation_generation += 1
        if self.current_target:
            self.current_target.subtree_last_mutation_at_mono = time.monotonic()
            self.current_target.target_subtree_mutation_generation = self.target_subtree_mutation_generation


class WorldStateTracker:
    """
    Event-driven observer handling CDP event subscriptions and maintaining
    the live BrowserState mirror.
    """

    def __init__(self, tab_id: str = "default_tab", target_ws_url: str = ""):
        self.state = BrowserState(tab_id=tab_id, target_ws_url=target_ws_url)

    # ------------------------------------------------------------------
    # Page Domain Events
    # ------------------------------------------------------------------
    def handle_frame_navigated(self, url: str, is_main_frame: bool = True) -> None:
        self.state.state_watermark += 1
        now = time.monotonic()
        if is_main_frame:
            self.state.current_url = url
            self.state.is_navigating = True
            self.state.navigation_started_at_mono = now
            self.state.lifecycle_phase = "NAVIGATING"
            # Mid-action navigation strictly invalidates active target
            self.state.invalidate_target()

    def handle_load_event_fired(self) -> None:
        self.state.state_watermark += 1
        self.state.is_navigating = False
        self.state.lifecycle_phase = "LOADING"
        self.state.document_invalidated = False

    def handle_javascript_dialog_opening(self, dialog_type: str, message: str, default_prompt: str = "") -> None:
        self.state.state_watermark += 1
        self.state.active_dialog = {
            "type": dialog_type,
            "message": message,
            "defaultPrompt": default_prompt,
            "opened_at_mono": time.monotonic(),
        }
        self.state.lifecycle_phase = "BLOCKED"

    def handle_javascript_dialog_closed(self) -> None:
        self.state.state_watermark += 1
        self.state.active_dialog = None
        self.state.lifecycle_phase = "DOM_SETTLING"

    # ------------------------------------------------------------------
    # Network Domain Events
    # ------------------------------------------------------------------
    def handle_request_will_be_sent(self, request_id: str, resource_type: str = "Other") -> None:
        self.state.state_watermark += 1
        self.state.on_request_will_be_sent(request_id, resource_type)

    def handle_loading_finished(self, request_id: str) -> None:
        self.state.state_watermark += 1
        self.state.on_request_finished_or_failed(request_id)

    def handle_loading_failed(self, request_id: str) -> None:
        self.state.state_watermark += 1
        self.state.on_request_finished_or_failed(request_id)

    # ------------------------------------------------------------------
    # DOM Domain Events
    # ------------------------------------------------------------------
    def handle_document_updated(self) -> None:
        """
        On DOM.documentUpdated:
        All node IDs in Chromium's DOM agent become invalid.
        Strict Invalidation: Clear target, clear parent map, reset revision.
        """
        self.state.state_watermark += 1
        self.state.document_generation += 1
        now = time.monotonic()
        self.state.dom_revision = 0
        self.state.node_parent_map.clear()
        self.state.invalidate_target()
        self.state.last_dom_mutation_at_mono = now
        self.state.lifecycle_phase = "DOM_SETTLING"
        self.state.document_invalidated = True

    def populate_tree_hierarchy(self, root_node: Dict[str, Any], parent_id: Optional[int] = None) -> None:
        """Populate node_parent_map from a DOM.getDocument or DOM.setChildNodes tree."""
        if not root_node or not isinstance(root_node, dict):
            return
        nid = root_node.get("nodeId")
        if nid:
            if parent_id is not None:
                self.state.node_parent_map[nid] = parent_id
            for child in root_node.get("children", []):
                self.populate_tree_hierarchy(child, nid)
            for sr in root_node.get("shadowRoots", []):
                self.populate_tree_hierarchy(sr, nid)
            cd = root_node.get("contentDocument")
            if cd:
                self.populate_tree_hierarchy(cd, nid)

    def handle_set_child_nodes(self, parent_node_id: int, nodes: List[Dict[str, Any]]) -> None:
        self.state.state_watermark += 1

        for node in nodes:
            self.populate_tree_hierarchy(node, parent_node_id)

    def handle_child_node_inserted(self, parent_node_id: int, node_id: int) -> None:
        self.state.state_watermark += 1
        now = time.monotonic()
        self.state.dom_revision += 1
        self.state.mutation_types_recent.add("childList")
        self.state.last_dom_mutation_at_mono = now

        # Add to parent map first
        self.state.node_parent_map[node_id] = parent_node_id

        # Direct hit or ancestry evaluation
        if self.state.current_target and self.state.current_target.node_id is not None:
            if parent_node_id == self.state.current_target.node_id or self.state.is_node_in_target_subtree(parent_node_id):
                self.state._mark_target_subtree_mutated()

    def handle_child_node_removed(self, parent_node_id: int, node_id: int) -> None:
        self.state.state_watermark += 1
        now = time.monotonic()
        self.state.dom_revision += 1
        self.state.mutation_types_recent.add("childList")
        self.state.last_dom_mutation_at_mono = now

        # Direct hit or ancestry evaluation starting from parentNodeId
        if self.state.current_target and self.state.current_target.node_id is not None:
            if parent_node_id == self.state.current_target.node_id or self.state.is_node_in_target_subtree(parent_node_id):
                self.state._mark_target_subtree_mutated()

        # Only then remove node from parent map
        self.state.node_parent_map.pop(node_id, None)

    def handle_attribute_modified(self, node_id: int, name: str, value: str) -> None:
        self.state.state_watermark += 1
        now = time.monotonic()
        self.state.dom_revision += 1
        self.state.mutation_types_recent.add("attributes")
        self.state.last_dom_mutation_at_mono = now

        if self.state.current_target and self.state.current_target.node_id is not None:
            if node_id == self.state.current_target.node_id or self.state.is_node_in_target_subtree(node_id):
                self.state._mark_target_subtree_mutated()

    def handle_attribute_removed(self, node_id: int, name: str) -> None:
        self.state.state_watermark += 1
        now = time.monotonic()
        self.state.dom_revision += 1
        self.state.mutation_types_recent.add("attributes")
        self.state.last_dom_mutation_at_mono = now

        if self.state.current_target and self.state.current_target.node_id is not None:
            if node_id == self.state.current_target.node_id or self.state.is_node_in_target_subtree(node_id):
                self.state._mark_target_subtree_mutated()

    def handle_character_data_modified(self, node_id: int, character_data: str) -> None:
        self.state.state_watermark += 1
        now = time.monotonic()
        self.state.dom_revision += 1
        self.state.mutation_types_recent.add("characterData")
        self.state.last_dom_mutation_at_mono = now

        if self.state.current_target and self.state.current_target.node_id is not None:
            if node_id == self.state.current_target.node_id or self.state.is_node_in_target_subtree(node_id):
                self.state._mark_target_subtree_mutated()
