"""
Empirical Coordinate Validation Matrix:
Verifies that:
1. Viewport coordinates from executor map 1:1 to window coordinates with currentBounds.
2. Top-left edge target (e.g. x=10, y=10) translates cleanly without clipping or negative offsets.
3. Bottom-right edge target (e.g. x=1350, y=850) translates cleanly without overflow.
4. Resized window and maximized window currentBounds are handled properly.
5. Invariant: visual_cursor_tip == executor_final_x,y translated by view offset.
"""

import math
import pytest


def js_round(val: float) -> int:
    return math.floor(val + 0.5)


def translate_coordinates(executor_x: float, executor_y: float, bounds: dict) -> tuple[int, int]:
    """Mirror exact Electron main.js coordinate translation logic."""
    view_x = int(bounds.get("x", 0) or 0)
    view_y = int(bounds.get("y", 0) or 0)
    target_x = js_round(view_x + executor_x)
    target_y = js_round(view_y + executor_y)
    return target_x, target_y


class TestCoordinateMatrix:

    def test_normal_window(self):
        # Default normal window: navbar/tabs top offset = 78, x = 0
        bounds = {"x": 0, "y": 78, "width": 1400, "height": 822}
        exec_x, exec_y = 520.5, 340.25

        cursor_x, cursor_y = translate_coordinates(exec_x, exec_y, bounds)

        assert cursor_x == 521  # js_round(0 + 520.5)
        assert cursor_y == 418  # js_round(78 + 340.25)
        assert cursor_x - bounds["x"] == js_round(exec_x)
        assert cursor_y - bounds["y"] == js_round(exec_y)

    def test_resized_window(self):
        # Window resized narrower with ChatDrawer open: x = 0, y = 82
        bounds = {"x": 0, "y": 82, "width": 960, "height": 718}
        exec_x, exec_y = 450.0, 210.0

        cursor_x, cursor_y = translate_coordinates(exec_x, exec_y, bounds)

        assert cursor_x == 450
        assert cursor_y == 292
        assert cursor_x - bounds["x"] == round(exec_x)
        assert cursor_y - bounds["y"] == round(exec_y)

    def test_maximized_window(self):
        # Maximized display: 1920x1080, top offset = 78
        bounds = {"x": 0, "y": 78, "width": 1920, "height": 1002}
        exec_x, exec_y = 960.0, 500.0

        cursor_x, cursor_y = translate_coordinates(exec_x, exec_y, bounds)

        assert cursor_x == 960
        assert cursor_y == 578
        assert cursor_x - bounds["x"] == round(exec_x)
        assert cursor_y - bounds["y"] == round(exec_y)

    def test_top_left_edge_target(self):
        # Edge case: Element in very top-left corner of web page (e.g. hamburger menu at x=12, y=14)
        bounds = {"x": 0, "y": 78, "width": 1400, "height": 822}
        exec_x, exec_y = 12.0, 14.0

        cursor_x, cursor_y = translate_coordinates(exec_x, exec_y, bounds)

        assert cursor_x == 12
        assert cursor_y == 92  # 78 + 14
        assert cursor_y >= bounds["y"], "Cursor must never be placed inside the native chrome header"

    def test_bottom_right_edge_target(self):
        # Edge case: Element in very bottom-right corner of web page (e.g. cookie accept at bottom right)
        bounds = {"x": 0, "y": 78, "width": 1400, "height": 822}
        exec_x, exec_y = 1380.0, 810.0

        cursor_x, cursor_y = translate_coordinates(exec_x, exec_y, bounds)

        assert cursor_x == 1380
        assert cursor_y == 888  # 78 + 810
        assert cursor_x <= bounds["x"] + bounds["width"]
        assert cursor_y <= bounds["y"] + bounds["height"]
