"""
Stage-A Verification Suite: tests/test_world_state.py
Validates BrowserState, WorldStateTracker, CDP event dispatch, target subtree
ancestry tracking, document invalidation, and independent sensory predicates.
"""

import time
import pytest
from nervous.world_state import BrowserState, TargetState, WorldStateTracker


def test_network_accounting_and_blocking_categorization():
    """Verify request addition, completion, and blocking resource classification."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Initial state
    assert state.network_in_flight == 0
    assert state.blocking_network_in_flight == 0
    assert state.has_no_blocking_requests is True

    # Send 1 blocking request (Fetch) and 1 non-blocking (WebSocket)
    tracker.handle_request_will_be_sent("req_1", "Fetch")
    tracker.handle_request_will_be_sent("req_2", "WebSocket")

    assert state.network_in_flight == 2
    assert state.blocking_network_in_flight == 1
    assert state.has_no_blocking_requests is False

    # req_1 finishes -> no blocking requests remain, but 1 non-blocking remains
    tracker.handle_loading_finished("req_1")
    assert state.network_in_flight == 1
    assert state.blocking_network_in_flight == 0
    assert state.has_no_blocking_requests is True

    # req_2 fails -> zero active requests
    tracker.handle_loading_failed("req_2")
    assert state.network_in_flight == 0
    assert state.blocking_network_in_flight == 0


def test_target_itself_mutates():
    """Case 1: Target itself mutates (attribute/characterData) -> resets subtree timestamp."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Assign target with node_id=100
    target = state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    # Attribute modified on target itself
    tracker.handle_attribute_modified(node_id=100, name="class", value="active")
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert "attributes" in state.mutation_types_recent


def test_direct_child_mutates():
    """Case 2: Mutation on direct child of target resets subtree timestamp."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Target: node 100. Direct child: node 101.
    target = state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[101] = 100
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    tracker.handle_character_data_modified(node_id=101, character_data="Click Me")
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert "characterData" in state.mutation_types_recent


def test_deep_descendant_mutates():
    """Case 3: Mutation on node 3+ levels deep inside target resets subtree timestamp."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Target: 100 -> Child: 101 -> Grandchild: 102 -> Great-grandchild: 103
    target = state.assign_target(locator="#container", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[101] = 100
    state.node_parent_map[102] = 101
    state.node_parent_map[103] = 102

    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    tracker.handle_attribute_modified(node_id=103, name="style", value="color: red")
    assert target.subtree_last_mutation_at_mono > baseline_time


def test_unrelated_sibling_mutates():
    """Case 4: Mutation on sibling does NOT touch target subtree timestamp."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Target: 100. Sibling: 200 (both children of body 50)
    target = state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[100] = 50
    state.node_parent_map[200] = 50

    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    tracker.handle_attribute_modified(node_id=200, name="data-test", value="sibling")
    # Global DOM mutation revision incremented, but target subtree timestamp untouched
    assert state.dom_revision == 1
    assert target.subtree_last_mutation_at_mono == baseline_time


def test_direct_insertion_under_target():
    """Case 5: Insertion directly under target marks subtree mutated and registers parent map."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#list", node_id=100, is_actionable=True, is_stable=True)
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    # Insert node 105 under node 100
    tracker.handle_child_node_inserted(parent_node_id=100, node_id=105)
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert state.node_parent_map.get(105) == 100


def test_deep_insertion_under_target():
    """Case 6: Insertion where parent is deep descendant marks subtree mutated."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Target 100 -> Child 101
    target = state.assign_target(locator="#card", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[101] = 100
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    # Insert 102 under 101
    tracker.handle_child_node_inserted(parent_node_id=101, node_id=102)
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert state.node_parent_map.get(102) == 101


def test_direct_removal_under_target():
    """Case 7: Removal directly under target marks subtree mutated and removes node from map."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#list", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[105] = 100
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    tracker.handle_child_node_removed(parent_node_id=100, node_id=105)
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert 105 not in state.node_parent_map


def test_deep_removal_under_target():
    """Case 8: Removal where parent is a descendant marks subtree mutated."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#wrapper", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[101] = 100
    state.node_parent_map[102] = 101
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    tracker.handle_child_node_removed(parent_node_id=101, node_id=102)
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert 102 not in state.node_parent_map
    assert state.node_parent_map.get(101) == 100


def test_attribute_removed_affects_target_subtree():
    """Case 9: DOM.attributeRemoved on target descendant properly updates timestamps."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#input", node_id=100, is_actionable=True, is_stable=True)
    baseline_time = target.subtree_last_mutation_at_mono
    time.sleep(0.01)

    tracker.handle_attribute_removed(node_id=100, name="disabled")
    assert target.subtree_last_mutation_at_mono > baseline_time
    assert "attributes" in state.mutation_types_recent


def test_document_updated_strictly_invalidates_state():
    """Case 10: DOM.documentUpdated strictly invalidates current target, clears parent map, resets revision."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#button", node_id=100, is_actionable=True, is_stable=True)
    state.node_parent_map[100] = 10
    state.dom_revision = 42
    assert state.current_target is not None

    tracker.handle_document_updated()

    assert state.current_target is None
    assert len(state.node_parent_map) == 0
    assert state.dom_revision == 0
    assert state.lifecycle_phase == "DOM_SETTLING"
    assert state.is_action_ready is False


def test_global_churn_immunity_and_target_local_action_ready():
    """Case 11: Global DOM churn outside target allows ACTION_READY while PAGE_STABLE is False."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Make tracker observe for >= 80ms so history is trustworthy
    state.tracking_started_at_mono = time.monotonic() - 0.100

    target = state.assign_target(locator="#isolated-btn", node_id=100, is_actionable=True, is_stable=True)
    # Sibling node 999
    state.node_parent_map[100] = 1
    state.node_parent_map[999] = 1

    # Simulate ongoing global churn on node 999 (e.g. ad ticker)
    tracker.handle_character_data_modified(node_id=999, character_data="Ticker update...")

    # Global DOM quiet is False -> PAGE_STABLE is False
    assert state.is_global_dom_quiet is False
    assert state.is_page_stable is False

    # But target subtree is quiet -> ACTION_READY is True!
    assert state.is_target_subtree_quiet is True
    assert state.is_action_ready is True


def test_zero_artificial_delay_on_pre_quiescent_page():
    """Case 12: Target assigned on already-quiet page evaluates is_target_subtree_quiet immediately."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Simulate established observation history (100ms) with zero recent mutations
    now = time.monotonic()
    state.tracking_started_at_mono = now - 0.200
    state.last_dom_mutation_at_mono = now - 0.150

    target = state.assign_target(locator="#ready-btn", node_id=100, is_actionable=True, is_stable=True)

    # Immediately quiet with zero added sleep!
    assert target.subtree_last_mutation_at_mono == state.last_dom_mutation_at_mono
    assert state.is_target_subtree_quiet is True
    assert state.is_action_ready is True


def test_quiescence_window_enforced_during_churn():
    """Case 13: Target assigned during active churn requires the full 80ms window to elapse."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    # Recent mutation happened 10ms ago
    now = time.monotonic()
    state.tracking_started_at_mono = now - 0.100
    state.last_dom_mutation_at_mono = now - 0.010

    target = state.assign_target(locator="#churning-btn", node_id=100, is_actionable=True, is_stable=True)

    # Subtree baseline set to current time, so is_target_subtree_quiet is False initially
    assert state.is_target_subtree_quiet is False
    assert state.is_action_ready is False


def test_target_geometric_stability_delta():
    """Verify position update calculates delta and stability accurately."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#box", node_id=100, is_actionable=True, is_stable=False)

    # Initial position
    state.update_target_position(x=100.0, y=200.0, width=50.0, height=20.0)
    assert target.is_stable is True

    # Small jitter (< 1.0px) -> remains stable
    state.update_target_position(x=100.4, y=200.3, width=50.0, height=20.0)
    assert target.last_position_delta < 1.0
    assert target.is_stable is True

    # Large movement (> 1.0px) -> becomes unstable
    state.update_target_position(x=120.0, y=200.0, width=50.0, height=20.0)
    assert target.last_position_delta >= 1.0
    assert target.is_stable is False


def test_javascript_dialog_and_navigation_interruption():
    """Verify dialog and navigation interrupt target readiness."""
    tracker = WorldStateTracker("test_tab")
    state = tracker.state

    target = state.assign_target(locator="#btn", node_id=100, is_actionable=True, is_stable=True)
    # Fake quiet history
    target.subtree_last_mutation_at_mono = time.monotonic() - 0.100
    assert state.is_action_ready is True

    # Dialog opens -> BLOCKED, is_action_ready becomes False
    tracker.handle_javascript_dialog_opening("alert", "Confirm action?")
    assert state.lifecycle_phase == "BLOCKED"
    assert state.is_action_ready is False

    # Dialog closed -> DOM_SETTLING
    tracker.handle_javascript_dialog_closed()
    assert state.lifecycle_phase == "DOM_SETTLING"
    assert state.active_dialog is None

    # Frame navigated -> NAVIGATING, target invalidated
    tracker.handle_frame_navigated("https://example.com/checkout", is_main_frame=True)
    assert state.lifecycle_phase == "NAVIGATING"
    assert state.is_navigating is True
    assert state.current_target is None
    assert state.is_action_ready is False
