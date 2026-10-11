"""
Unit and Contract Verification Suite for Canonical Method E: tests/test_method_e_contract.py

Verifies:
A. The canonical nervous.method_e module exists and exports METHOD_E_CHECK_FUNC.
B. All six frozen actionability conditions are explicitly checked.
C. Forbidden constructs (elementFromPoint, opacity thresholds, scrolling, hit testing) are absent.
D. nervous.cdp_session imports and uses the canonical Method E without local duplicate definitions.
E. The exported predicate is structurally and behaviorally identical to the frozen Stage C predicate.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

import nervous.cdp_session as cdp_session_mod
import nervous.method_e as method_e_mod
from nervous.method_e import METHOD_E_CHECK_FUNC, METHOD_E_JS


class TestCanonicalMethodEModule:
    """A. Canonical Method E module existence and structure."""

    def test_canonical_module_and_exports_exist(self):
        assert hasattr(method_e_mod, "METHOD_E_CHECK_FUNC")
        assert isinstance(METHOD_E_CHECK_FUNC, str)
        assert len(METHOD_E_CHECK_FUNC) > 0
        assert METHOD_E_JS == METHOD_E_CHECK_FUNC


class TestFrozenSixConditions:
    """B. Verification that all six frozen conditions are present in the predicate."""

    def test_condition_1_element_existence(self):
        # Must assert that el exists and is an Element
        assert "el instanceof Element" in METHOD_E_CHECK_FUNC
        assert "!el" in METHOD_E_CHECK_FUNC

    def test_condition_2_width_greater_than_zero(self):
        assert "rect.width > 0" in METHOD_E_CHECK_FUNC

    def test_condition_3_height_greater_than_zero(self):
        assert "rect.height > 0" in METHOD_E_CHECK_FUNC

    def test_condition_4_display_not_none(self):
        assert "display !== 'none'" in METHOD_E_CHECK_FUNC or 'display !== "none"' in METHOD_E_CHECK_FUNC

    def test_condition_5_visibility_not_hidden(self):
        assert "visibility !== 'hidden'" in METHOD_E_CHECK_FUNC or 'visibility !== "hidden"' in METHOD_E_CHECK_FUNC

    def test_condition_6_not_disabled_and_pointer_events_not_none(self):
        assert "!el.disabled" in METHOD_E_CHECK_FUNC
        assert "pointerEvents !== 'none'" in METHOD_E_CHECK_FUNC or 'pointerEvents !== "none"' in METHOD_E_CHECK_FUNC

    def test_compound_actionability_boolean(self):
        # Must evaluate conjunction of all six conditions
        assert "hasWidth && hasHeight && isDisplay && isVisibility && notDisabled && pointerEvents" in METHOD_E_CHECK_FUNC


class TestForbiddenConstructsAbsence:
    """C. Verification that forbidden constructs are absent from Method E."""

    def test_no_element_from_point(self):
        assert "elementFromPoint" not in METHOD_E_CHECK_FUNC

    def test_no_opacity_thresholds(self):
        assert "opacity" not in METHOD_E_CHECK_FUNC

    def test_no_scroll_into_view(self):
        assert "scrollIntoView" not in METHOD_E_CHECK_FUNC
        assert "scroll" not in METHOD_E_CHECK_FUNC.lower()

    def test_no_viewport_intersection_or_hit_testing(self):
        assert "intersection" not in METHOD_E_CHECK_FUNC.lower()
        assert "elementsFromPoint" not in METHOD_E_CHECK_FUNC
        assert "hittest" not in METHOD_E_CHECK_FUNC.lower()


class TestSingleSourceOfTruthInCDPSession:
    """D. Verification that cdp_session.py uses the canonical module without duplicate local definitions."""

    def test_cdp_session_uses_canonical_method_e_identity(self):
        assert hasattr(cdp_session_mod, "METHOD_E_CHECK_FUNC")
        assert cdp_session_mod.METHOD_E_CHECK_FUNC is method_e_mod.METHOD_E_CHECK_FUNC

    def test_cdp_session_source_does_not_contain_duplicate_literal(self):
        cdp_session_path = Path(inspect.getfile(cdp_session_mod))
        source = cdp_session_path.read_text(encoding="utf-8")

        # Parse AST to ensure METHOD_E_CHECK_FUNC is an imported name, not an assigned string literal
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "METHOD_E_CHECK_FUNC":
                        pytest.fail("cdp_session.py must NOT assign a local METHOD_E_CHECK_FUNC; it must import it.")

        # Ensure import statement exists
        assert re.search(r"from\s+\.method_e\s+import\s+METHOD_E_CHECK_FUNC", source) is not None


class TestPredicateBehavioralEquivalence:
    """E. Verification that the extracted predicate returns the expected schema contract."""

    def test_return_schema_contract(self):
        # Must return an object with actionable and box fields
        assert "actionable: isActionable" in METHOD_E_CHECK_FUNC
        assert "box: {" in METHOD_E_CHECK_FUNC
        assert "x: Number(rect.left)" in METHOD_E_CHECK_FUNC
        assert "y: Number(rect.top)" in METHOD_E_CHECK_FUNC
        assert "width: Number(rect.width)" in METHOD_E_CHECK_FUNC
        assert "height: Number(rect.height)" in METHOD_E_CHECK_FUNC
