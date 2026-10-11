"""
Comex Browser Nervous System — Canonical Method E Actionability Module.

Defines the authoritative, frozen Method E browser-native actionability predicate.
Preserves the exact six-condition behavioral contract established in the Comex
architecture:
  1. Element existence / resolved DOM element: !el || !(el instanceof Element) -> false
  2. Non-zero width: rect.width > 0
  3. Non-zero height: rect.height > 0
  4. Visible display: style.display !== 'none'
  5. Visible visibility: style.visibility !== 'hidden'
  6. Interactive and enabled: !el.disabled && style.pointerEvents !== 'none'

Explicit Invariants & Forbidden Constructs:
  - NO elementFromPoint
  - NO viewport intersection testing
  - NO opacity thresholds
  - NO scrolling (handled exclusively by actuator/executor)
  - NO coordinate translation (handled exclusively by actuator/executor)
  - NO hit-testing
  - NO semantic postcondition inference
  - Offscreen elements are explicitly allowed if the six conditions pass.
"""

from __future__ import annotations

# Canonical JavaScript function evaluated against a resolved remote DOM object
# via CDP Runtime.callFunctionOn (with objectId binding).
METHOD_E_CHECK_FUNC = """
function() {
    const el = this;
    if (!el || !(el instanceof Element)) {
        return { actionable: false, reason: "not_an_element" };
    }
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const hasWidth = rect.width > 0;
    const hasHeight = rect.height > 0;
    const isDisplay = style.display !== 'none';
    const isVisibility = style.visibility !== 'hidden';
    const notDisabled = !el.disabled;
    const pointerEvents = style.pointerEvents !== 'none';

    const isActionable = Boolean(
        hasWidth && hasHeight && isDisplay && isVisibility && notDisabled && pointerEvents
    );

    return {
        actionable: isActionable,
        box: {
            x: Number(rect.left),
            y: Number(rect.top),
            width: Number(rect.width),
            height: Number(rect.height)
        }
    };
}
""".strip()

# Alias for canonical JS string
METHOD_E_JS = METHOD_E_CHECK_FUNC

__all__ = [
    "METHOD_E_CHECK_FUNC",
    "METHOD_E_JS",
]
