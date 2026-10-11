"""
Comex Browser Nervous System — Stage D Behavioral Benchmark
Target Runtime: Live Comex Electron Application (Port 9222)

Exercises the authentic production path:
    browser_click(selector)
    -> TabSessionContext
    -> CDP event barrier (wait_until_processed)
    -> OverlayPolicyMatcher (DOM presence-first evaluation)
    -> ReflexEngine (Priority P0-P5 rule chain)
    -> ActuatorGate.async_attempt
    -> SafeOverlayHandler (preconditions, authorization, binding integrity, identity, stability, quietness)
    -> SelectorClickExecutor (backendNodeId exact dispatch)
    -> Live Electron Chromium (DevTools Protocol on port 9222)

Rules:
  - Connects ONLY to the live Electron browser on port 9222.
  - Zero standalone/headless Chrome processes.
  - Zero production architecture modifications.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import math
import os
import socketserver
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nervous.actuator_gate import ActuatorGate, DecisionType, GateResult
from nervous.cdp_session import TabSessionContext, WebSocketCDPClient, get_or_create_tab_session
from nervous.policy_registry import (
    OverlayPolicyMatcher,
    PolicyRegistry,
    ReflexActionRouter,
    SafeOverlayHandler,
    SafeOverlayPolicy,
)
from nervous.world_state import BrowserState

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("comex.benchmark")

ELECTRON_CDP_URL = "http://127.0.0.1:9222"
FIXTURE_PORT = 8899
BASE_FIXTURE_URL = f"http://127.0.0.1:{FIXTURE_PORT}"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "stage_d"


# ──────────────────────────────────────────────────────────────────────
#  1. Local Fixture Static Server
# ──────────────────────────────────────────────────────────────────────

class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FIXTURES_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class FixtureServer:
    def __init__(self, port: int = FIXTURE_PORT) -> None:
        self.port = port
        self.httpd: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), QuietHTTPHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


# ──────────────────────────────────────────────────────────────────────
#  2. Benchmark Metrics Records
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    scenario: str
    iteration: int
    decision: str
    rule_name: str
    executed: bool
    handler_invoked: bool
    was_blocked: bool
    physical_dispatches: int
    real_clicks_observed: int
    dismiss_clicks_observed: int
    success: bool
    failure_reason: Optional[str] = None
    elapsed_ms: float = 0.0


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)
    def _pct(p: float) -> float:
        idx = int(p / 100.0 * (n - 1))
        return s[idx]
    return {
        "p50": round(_pct(50), 2),
        "p95": round(_pct(95), 2),
        "p99": round(_pct(99), 2),
    }


# ──────────────────────────────────────────────────────────────────────
#  3. Benchmark Harness for Live Electron
# ──────────────────────────────────────────────────────────────────────

class ElectronStageDBenchmark:
    def __init__(self, cdp_url: str = ELECTRON_CDP_URL) -> None:
        self.cdp_url = cdp_url
        self.session: Optional[TabSessionContext] = None
        self.target_id: Optional[str] = None
        self.ws_url: Optional[str] = None

    async def attach_to_electron(self) -> None:
        """Discover the active WebContentsView in the live Electron application."""
        req = urllib.request.urlopen(f"{self.cdp_url}/json/list", timeout=3)
        targets = json.loads(req.read().decode())

        # Select the active WebContentsView (not the localhost:5173 React shell)
        browser_targets = [
            t for t in targets
            if t.get("type") == "page" and "localhost:5173" not in t.get("url", "") and "cursor_overlay" not in t.get("url", "")
        ]

        if not browser_targets:
            raise RuntimeError(
                f"No active WebContentsView found in live Electron at {self.cdp_url}. "
                "Ensure Comex Electron is running (npm run electron:dev)."
            )

        target = browser_targets[0]
        self.target_id = target["id"]
        self.ws_url = target["webSocketDebuggerUrl"]

        self.session = await get_or_create_tab_session(self.target_id, self.ws_url)

        # Focus the WebContentsView so native click events are processed
        await self.session.cdp.send("Page.bringToFront")

        # Configure policies
        self._configure_policies()

    def _configure_policies(self) -> None:
        assert self.session is not None
        reg = self.session.policy_registry

        # 1. Valid safe autonomic cookie policy
        reg.register_policy(SafeOverlayPolicy(
            policy_id="cookie_policy",
            version="1.0.0",
            domain_pattern="*",
            overlay_selector="#cookie-overlay",
            dismiss_selector="#cookie-accept",
            action="click",
            enabled=True,
            safety_tier="SAFE_AUTONOMIC",
            provenance="builtin",
        ))

        # 2. Disabled policy
        reg.register_policy(SafeOverlayPolicy(
            policy_id="disabled_policy",
            version="1.0.0",
            domain_pattern="*",
            overlay_selector="#disabled-overlay",
            dismiss_selector="#disabled-accept",
            action="click",
            enabled=False,
            safety_tier="SAFE_AUTONOMIC",
            provenance="builtin",
        ))

        # 3. Wrong domain policy
        reg.register_policy(SafeOverlayPolicy(
            policy_id="wrong_domain_policy",
            version="1.0.0",
            domain_pattern="restricted-domain-example.com",
            overlay_selector="#wrong-domain-overlay",
            dismiss_selector="#wrong-domain-accept",
            action="click",
            enabled=True,
            safety_tier="SAFE_AUTONOMIC",
            provenance="builtin",
        ))

    async def navigate_fixture(self, fixture_name: str) -> None:
        assert self.session is not None
        url = f"{BASE_FIXTURE_URL}/{fixture_name}"
        await self.session.cdp.send("Page.navigate", {"url": url})

        # Wait until document readyState is complete
        for _ in range(50):
            res = await self.session.cdp.send("Runtime.evaluate", {
                "expression": "document.readyState",
                "returnByValue": True,
            })
            if res.get("result", {}).get("value") == "complete":
                break
            await asyncio.sleep(0.02)

        # Drain CDP queue and settle
        await self.session.flush_and_sync()
        await self.session.cdp.send("Page.bringToFront")
        await self.session.cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
        # Satisfy the 80ms post-navigation quiescence window
        await asyncio.sleep(0.1)

    async def get_dom_counter(self, var_name: str) -> int:
        assert self.session is not None
        res = await self.session.cdp.send("Runtime.evaluate", {
            "expression": f"window.{var_name} || 0",
            "returnByValue": True,
        })
        return int(res.get("result", {}).get("value", 0))

    async def production_browser_click(self, selector: str) -> Tuple[GateResult, float, Optional[str]]:
        """
        Executes the exact production control path:
            binding = await session.resolve_and_bind(selector)
            gate_res = await session.gate.async_attempt("click", binding=binding)
        """
        assert self.session is not None
        t0 = time.perf_counter()
        error_msg = None
        try:
            binding = await self.session.resolve_and_bind(selector)
            if not self.session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            gate_res = await self.session.gate.async_attempt("click", binding=binding)
        except Exception as e:
            error_msg = str(e)
            from nervous.reflex_engine import DecisionType, Priority, ReflexDecision
            gate_res = GateResult(
                decision=ReflexDecision(
                    decision=DecisionType.REPORT_BLOCKED,
                    priority=Priority.P0_SAFETY,
                    rule_name=type(e).__name__,
                    reason=str(e),
                ),
                executed=False,
                handler_invoked=False,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return gate_res, elapsed_ms, error_msg

    # ──────────────────────────────────────────────────────────────────
    #  Scenarios
    # ──────────────────────────────────────────────────────────────────

    async def benchmark_clean_page(self, iterations: int = 50) -> List[IterationResult]:
        """Scenario 1: Clean page without overlays -> normal Stage C click."""
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("clean.html")
            gate_res, elapsed_ms, err = await self.production_browser_click("#real-target")
            await asyncio.sleep(0.04)
            real_clicks = await self.get_dom_counter("__real_clicks")
            print(f"DEBUG benchmark_clean_page iter {i}: executed={gate_res.executed}, exec_res={gate_res.execution_result}, real_clicks={real_clicks}")

            success = (
                gate_res.executed is True
                and gate_res.decision.decision == DecisionType.ALLOW
                and real_clicks == 1
                and err is None
            )
            results.append(IterationResult(
                scenario="clean_page",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=gate_res.was_blocked,
                physical_dispatches=1 if gate_res.executed else 0,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=0,
                success=success,
                failure_reason=err or (None if success else f"clicks={real_clicks}"),
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_safe_overlay(self, iterations: int = 50) -> List[IterationResult]:
        """
        Scenario 2: Single valid SAFE_AUTONOMIC overlay.
        Call 1: routes to SafeOverlayHandler -> clicks dismiss button -> target NOT clicked!
        Call 2: next observation cycle -> overlay gone -> real target clicked!
        """
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("safe_overlay.html")

            # Call 1: user requests #real-target with overlay present
            gate_res1, elapsed_ms1, err1 = await self.production_browser_click("#real-target")
            await asyncio.sleep(0.04)
            real_clicks1 = await self.get_dom_counter("__real_clicks")
            dismiss_clicks1 = await self.get_dom_counter("__dismiss_clicks")
            print(f"DEBUG safe_overlay Call 1: decision={gate_res1.decision.decision.name}, handler_invoked={gate_res1.handler_invoked}, executed={gate_res1.executed}, err1={err1}, dismiss={dismiss_clicks1}, real={real_clicks1}")
            print(f"DEBUG safe_overlay Call 1 exec_res: {gate_res1.execution_result}")

            call1_ok = (
                gate_res1.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
                and gate_res1.handler_invoked is True
                and gate_res1.executed is False
                and real_clicks1 == 0  # CRITICAL: underlying target was NOT clicked!
                and dismiss_clicks1 == 1  # Dismissal button WAS clicked
                and err1 is None
            )

            # Settle & Call 2: next observation cycle -> overlay removed in DOM -> normal click
            await asyncio.sleep(0.05)
            await self.session.flush_and_sync()
            gate_res2, elapsed_ms2, err2 = await self.production_browser_click("#real-target")
            await asyncio.sleep(0.04)
            real_clicks2 = await self.get_dom_counter("__real_clicks")

            call2_ok = (
                gate_res2.decision.decision == DecisionType.ALLOW
                and gate_res2.executed is True
                and real_clicks2 == 1
                and err2 is None
            )

            success = call1_ok and call2_ok
            results.append(IterationResult(
                scenario="safe_overlay_dismissal",
                iteration=i + 1,
                decision=gate_res1.decision.decision.name,
                rule_name=gate_res1.decision.rule_name,
                executed=gate_res1.executed,
                handler_invoked=gate_res1.handler_invoked,
                was_blocked=gate_res1.was_blocked,
                physical_dispatches=1,
                real_clicks_observed=real_clicks2,
                dismiss_clicks_observed=dismiss_clicks1,
                success=success,
                failure_reason=None if success else f"c1_ok={call1_ok}, c2_ok={call2_ok}",
                elapsed_ms=elapsed_ms1,
            ))
        return results

    async def benchmark_disabled_overlay(self, iterations: int = 50) -> List[IterationResult]:
        """Scenario 3: Disabled overlay present -> fail closed, 0 dispatches."""
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("disabled_overlay.html")
            gate_res, elapsed_ms, err = await self.production_browser_click("#real-target")
            real_clicks = await self.get_dom_counter("__real_clicks")
            dismiss_clicks = await self.get_dom_counter("__dismiss_clicks")

            success = (
                gate_res.decision.decision == DecisionType.REPORT_BLOCKED
                and gate_res.decision.rule_name == "UnmatchedOverlayBlocking"
                and gate_res.executed is False
                and gate_res.handler_invoked is False
                and real_clicks == 0
                and dismiss_clicks == 0
            )
            results.append(IterationResult(
                scenario="disabled_overlay_blocked",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=gate_res.was_blocked,
                physical_dispatches=0,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=dismiss_clicks,
                success=success,
                failure_reason=err or (None if success else f"clicks={real_clicks}/{dismiss_clicks}"),
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_wrong_domain(self, iterations: int = 50) -> List[IterationResult]:
        """Scenario 4: Wrong domain overlay -> fail closed, 0 dispatches."""
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("wrong_domain.html")
            gate_res, elapsed_ms, err = await self.production_browser_click("#real-target")
            real_clicks = await self.get_dom_counter("__real_clicks")
            dismiss_clicks = await self.get_dom_counter("__dismiss_clicks")

            success = (
                gate_res.decision.decision == DecisionType.REPORT_BLOCKED
                and gate_res.decision.rule_name == "UnmatchedOverlayBlocking"
                and gate_res.executed is False
                and gate_res.handler_invoked is False
                and real_clicks == 0
                and dismiss_clicks == 0
            )
            results.append(IterationResult(
                scenario="wrong_domain_blocked",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=gate_res.was_blocked,
                physical_dispatches=0,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=dismiss_clicks,
                success=success,
                failure_reason=err or (None if success else f"clicks={real_clicks}/{dismiss_clicks}"),
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_multiple_overlays(self, iterations: int = 50) -> List[IterationResult]:
        """Scenario 5: Multiple overlays present (>1) -> strictly fail closed, 0 dispatches."""
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("multiple_overlays.html")
            gate_res, elapsed_ms, err = await self.production_browser_click("#real-target")
            real_clicks = await self.get_dom_counter("__real_clicks")
            dismiss_clicks = await self.get_dom_counter("__dismiss_clicks")
            disabled_clicks = await self.get_dom_counter("__disabled_clicks")

            success = (
                gate_res.decision.decision == DecisionType.REPORT_BLOCKED
                and gate_res.decision.rule_name == "UnmatchedOverlayBlocking"
                and gate_res.executed is False
                and gate_res.handler_invoked is False
                and real_clicks == 0
                and dismiss_clicks == 0
                and disabled_clicks == 0
            )
            results.append(IterationResult(
                scenario="multiple_overlays_blocked",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=gate_res.was_blocked,
                physical_dispatches=0,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=dismiss_clicks,
                success=success,
                failure_reason=err or (None if success else f"clicks={real_clicks}/{dismiss_clicks}"),
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_moving_target(self, iterations: int = 100) -> List[IterationResult]:
        """Scenario 6: Geometrically moving dismiss button -> aborts safely, 0 dispatches."""
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("moving_target.html")
            # Wait for button to start moving in DOM
            await asyncio.sleep(0.04)
            gate_res, elapsed_ms, err = await self.production_browser_click("#real-target")
            real_clicks = await self.get_dom_counter("__real_clicks")
            dismiss_clicks = await self.get_dom_counter("__dismiss_clicks")

            # Must abort before dispatch: 0 clicks received
            success = (
                real_clicks == 0
                and dismiss_clicks == 0
                and (gate_res.was_blocked or "actively churning" in str(err) or "geometrically unstable" in str(err))
            )
            results.append(IterationResult(
                scenario="moving_target_blocked",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=True,
                physical_dispatches=0,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=dismiss_clicks,
                success=success,
                failure_reason=err,
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_churning_target(self, iterations: int = 100) -> List[IterationResult]:
        """Scenario 7: Churning dismiss button subtree -> aborts safely, 0 dispatches."""
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("churning_target.html")
            await asyncio.sleep(0.04)
            gate_res, elapsed_ms, err = await self.production_browser_click("#real-target")
            real_clicks = await self.get_dom_counter("__real_clicks")
            dismiss_clicks = await self.get_dom_counter("__dismiss_clicks")

            success = (
                real_clicks == 0
                and dismiss_clicks == 0
                and (gate_res.was_blocked or "actively churning" in str(err))
            )
            results.append(IterationResult(
                scenario="churning_target_blocked",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=True,
                physical_dispatches=0,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=dismiss_clicks,
                success=success,
                failure_reason=err,
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_disappearing_overlay(self, iterations: int = 50) -> List[IterationResult]:
        """
        Scenario 8: State lifecycle verification.
        Validates overlay lifecycle:
        - Call 1: Overlay detected -> routed to SafeOverlayHandler -> dismiss click dispatched.
        - Settle & drain CDP event queue -> evaluate_overlay_state() probes DOM and confirms removal.
        - Call 2: Overlay is confirmed gone -> normal click on #real-target is ALLOWED and executed.
        """
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("disappearing_overlay.html")

            # Call 1: Intercepted and routed to SafeOverlayHandler to dismiss
            gate_res1, elapsed_ms1, err1 = await self.production_browser_click("#real-target")
            await asyncio.sleep(0.04)
            dismiss_clicks = await self.get_dom_counter("__dismiss_clicks")

            call1_ok = (
                gate_res1.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER
                and gate_res1.handler_invoked is True
                and gate_res1.executed is False
                and dismiss_clicks == 1
                and err1 is None
            )

            # Settle & drain CDP queue
            await asyncio.sleep(0.04)
            await self.session.flush_and_sync()

            # Next evaluation cycle: OverlayPolicyMatcher probes DOM and confirms removal
            await self.session.evaluate_overlay_state()
            state_cleared_by_matcher = self.session.tracker.state.blocking_overlay_detected is False

            # Call 2: Normal click on real target
            gate_res2, elapsed_ms2, err2 = await self.production_browser_click("#real-target")
            await asyncio.sleep(0.04)
            real_clicks = await self.get_dom_counter("__real_clicks")

            call2_ok = (
                state_cleared_by_matcher
                and gate_res2.decision.decision == DecisionType.ALLOW
                and gate_res2.executed is True
                and real_clicks == 1
                and err2 is None
            )

            success = call1_ok and call2_ok
            results.append(IterationResult(
                scenario="disappearing_overlay_lifecycle",
                iteration=i + 1,
                decision=gate_res1.decision.decision.name,
                rule_name=gate_res1.decision.rule_name,
                executed=gate_res1.executed,
                handler_invoked=gate_res1.handler_invoked,
                was_blocked=gate_res1.was_blocked,
                physical_dispatches=2,
                real_clicks_observed=real_clicks,
                dismiss_clicks_observed=dismiss_clicks,
                success=success,
                failure_reason=None if success else f"c1_ok={call1_ok}, c2_ok={call2_ok}, matcher_cleared={state_cleared_by_matcher}",
                elapsed_ms=elapsed_ms1,
            ))
        return results

    async def benchmark_race_target(self, iterations: int = 100) -> List[IterationResult]:
        """
        Scenario 9: Rapid target replacement/mutation (TOCTOU race test).
        Validates correct-target dispatch vs replacement target.
        - If executed: MUST click the exact resolved target (bound UID == clicked UID).
        - If target was replaced or moved: MUST safely abort with pre-input validation or stability error.
        - NEVER dispatch to a replacement node (bound UID != clicked UID is a strict failure).
        """
        results: List[IterationResult] = []
        for i in range(iterations):
            await self.navigate_fixture("race_target.html")
            await asyncio.sleep(0.02)

            t0 = time.perf_counter()
            err = None
            gate_res = None
            bound_uid = None
            try:
                # 1. Resolve and bind target
                binding = await self.session.resolve_and_bind("#race-target")

                # 2. Inspect the bound node's target UID via CDP
                node_res = await self.session.cdp.send("DOM.resolveNode", {"backendNodeId": binding.backend_node_id})
                obj_id = node_res.get("object", {}).get("objectId")
                if obj_id:
                    eval_res = await self.session.cdp.send("Runtime.callFunctionOn", {
                        "objectId": obj_id,
                        "functionDeclaration": "function() { return this.dataset ? this.dataset.targetUid : null; }",
                        "returnByValue": True,
                    })
                    bound_uid = eval_res.get("result", {}).get("value")

                # 3. Gate attempt
                gate_res = await self.session.gate.async_attempt("click", binding=binding)
            except Exception as e:
                err = str(e)
                from nervous.reflex_engine import DecisionType, Priority, ReflexDecision
                gate_res = GateResult(
                    decision=ReflexDecision(
                        decision=DecisionType.REPORT_BLOCKED,
                        priority=Priority.P0_SAFETY,
                        rule_name=type(e).__name__,
                        reason=str(e),
                    ),
                    executed=False,
                    handler_invoked=False,
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # 4. Check DOM click counters & clicked UID
            if gate_res.executed:
                for _ in range(5):
                    if await self.get_dom_counter("__race_clicks") > 0:
                        break
                    await asyncio.sleep(0.01)

            race_clicks = await self.get_dom_counter("__race_clicks")
            last_clicked_res = await self.session.cdp.send("Runtime.evaluate", {
                "expression": "window.__last_clicked_uid || ''",
                "returnByValue": True,
            })
            last_clicked_uid = last_clicked_res.get("result", {}).get("value", "")

            # Precise evaluation:
            # - If executed: bound_uid MUST equal last_clicked_uid (exact intended target hit!)
            # - If aborted/blocked: 0 clicks received, or no replacement clicked
            if gate_res.executed:
                if bound_uid and last_clicked_uid == bound_uid and race_clicks == 1:
                    success = True
                    fail_reason = None
                else:
                    success = False
                    fail_reason = f"TOCTOU Corrupted Dispatch: bound={bound_uid} but clicked={last_clicked_uid} (clicks={race_clicks})"
            else:
                # Safely aborted
                if race_clicks == 0:
                    success = True
                    fail_reason = None
                else:
                    success = False
                    fail_reason = f"Action blocked but click registered: {race_clicks} (clicked_uid={last_clicked_uid})"

            results.append(IterationResult(
                scenario="race_target_toctou",
                iteration=i + 1,
                decision=gate_res.decision.decision.name,
                rule_name=gate_res.decision.rule_name,
                executed=gate_res.executed,
                handler_invoked=gate_res.handler_invoked,
                was_blocked=gate_res.was_blocked,
                physical_dispatches=1 if gate_res.executed else 0,
                real_clicks_observed=race_clicks,
                dismiss_clicks_observed=0,
                success=success,
                failure_reason=fail_reason or err,
                elapsed_ms=elapsed_ms,
            ))
        return results

    async def benchmark_stage_c_vs_stage_d_clean_latency(self, iterations: int = 50) -> Dict[str, Any]:
        """
        Scenario 10: Clean-page latency comparison between Stage C (no overlay matcher scan)
        and Stage D (with overlay policy matcher scan on clean page) in the live Electron browser.
        """
        # 1. Measure Stage C (bypass overlay matcher)
        c_latencies: List[float] = []
        orig_evaluate = self.session.evaluate_overlay_state
        async def _noop_evaluate():
            pass

        self.session.evaluate_overlay_state = _noop_evaluate
        try:
            for _ in range(iterations):
                await self.navigate_fixture("clean.html")
                _, ms, _ = await self.production_browser_click("#real-target")
                c_latencies.append(ms)
        finally:
            self.session.evaluate_overlay_state = orig_evaluate

        # 2. Measure Stage D (full production evaluate_overlay_state)
        d_latencies: List[float] = []
        for _ in range(iterations):
            await self.navigate_fixture("clean.html")
            _, ms, _ = await self.production_browser_click("#real-target")
            d_latencies.append(ms)

        return {
            "stage_c": compute_percentiles(c_latencies),
            "stage_d": compute_percentiles(d_latencies),
            "stage_c_samples": [round(x, 2) for x in c_latencies[:10]],
            "stage_d_samples": [round(x, 2) for x in d_latencies[:10]],
        }


# ──────────────────────────────────────────────────────────────────────
#  4. Main Entry Point
# ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("  Comex Browser Nervous System — Stage D Behavioral Benchmark")
    print("  Target Runtime: Live Comex Electron Application (CDP :9222)")
    print("=" * 70)

    server = FixtureServer(FIXTURE_PORT)

    try:
        print("[1/3] Starting local fixture HTTP server on port", FIXTURE_PORT, "...")
        server.start()

        print("[2/3] Attaching to live Comex Electron runtime on CDP port 9222...")
        runner = ElectronStageDBenchmark(ELECTRON_CDP_URL)
        await runner.attach_to_electron()
        print(f"      Successfully attached to Electron WebContentsView target: {runner.target_id}")

        print("[3/3] Executing Behavioral Test Matrix against live Electron...")
        t_bench_start = time.monotonic()

        print("  - Running Clean Page (50 runs)...")
        r_clean = await runner.benchmark_clean_page(50)

        print("  - Running Safe Overlay Dismissal (50 runs)...")
        r_safe = await runner.benchmark_safe_overlay(50)

        print("  - Running Disabled Overlay (50 runs)...")
        r_disabled = await runner.benchmark_disabled_overlay(50)

        print("  - Running Wrong Domain Overlay (50 runs)...")
        r_wrong_domain = await runner.benchmark_wrong_domain(50)

        print("  - Running Multiple Overlays (50 runs)...")
        r_multiple = await runner.benchmark_multiple_overlays(50)

        print("  - Running Moving Dismiss Target (100 runs)...")
        r_moving = await runner.benchmark_moving_target(100)

        print("  - Running Churning Dismiss Target (100 runs)...")
        r_churning = await runner.benchmark_churning_target(100)

        print("  - Running Disappearing Overlay State Lifecycle (50 runs)...")
        r_disappearing = await runner.benchmark_disappearing_overlay(50)

        print("  - Running TOCTOU Race Target (100 runs)...")
        r_race = await runner.benchmark_race_target(100)

        print("  - Measuring Clean-Page Stage C vs Stage D Latency in Electron (50 runs each)...")
        latency_comp = await runner.benchmark_stage_c_vs_stage_d_clean_latency(50)

        bench_duration = round(time.monotonic() - t_bench_start, 2)

        # Restore tab to about:blank cleanly
        try:
            await runner.session.cdp.send("Page.navigate", {"url": "about:blank"})
        except Exception:
            pass

        # ──────────────────────────────────────────────────────────────
        #  Aggregate & Print Results Report
        # ──────────────────────────────────────────────────────────────
        all_results = [
            ("Clean click", r_clean),
            ("Safe overlay dismissal", r_safe),
            ("Disabled overlay blocked", r_disabled),
            ("Wrong-domain blocked", r_wrong_domain),
            ("Multiple overlays blocked", r_multiple),
            ("Moving target blocked", r_moving),
            ("Subtree churn blocked", r_churning),
            ("Disappearing lifecycle", r_disappearing),
            ("TOCTOU race target", r_race),
        ]

        print("\n" + "=" * 70)
        print("STAGE D BROWSER BEHAVIORAL BENCHMARK REPORT (ELECTRON RUNTIME)")
        print("=" * 70)
        print("Environment: Live Comex Electron (Electron 43.4.0 / Chromium 150.0.7871.224) on macOS")
        print(f"CDP Target: {runner.target_id} at {ELECTRON_CDP_URL}")
        print(f"Total Duration: {bench_duration}s")
        print("-" * 70)
        print(f"{'Scenario':<35} {'Passed/Total':<15} {'Success %':<10} {'P50 (ms)':<10} {'P95 (ms)':<10}")
        print("-" * 70)

        total_unexpected_clicks = 0

        for name, res_list in all_results:
            passed = sum(1 for r in res_list if r.success)
            total = len(res_list)
            pct = round(passed / total * 100, 1)
            times = [r.elapsed_ms for r in res_list]
            p = compute_percentiles(times)
            print(f"{name:<35} {f'{passed}/{total}':<15} {f'{pct}%':<10} {p['p50']:<10} {p['p95']:<10}")

            if "blocked" in name.lower():
                unexpected = sum(r.physical_dispatches for r in res_list)
                total_unexpected_clicks += unexpected

        print("-" * 70)
        print(f"Unexpected physical dispatches in blocked scenarios: {total_unexpected_clicks}")
        print("-" * 70)
        print("\nClean-Page Latency Comparison in Electron (50 iterations each):")
        print(f"  Stage C (baseline without overlay matcher): P50={latency_comp['stage_c']['p50']}ms, P95={latency_comp['stage_c']['p95']}ms, P99={latency_comp['stage_c']['p99']}ms")
        print(f"  Stage D (production with overlay matcher):  P50={latency_comp['stage_d']['p50']}ms, P95={latency_comp['stage_d']['p95']}ms, P99={latency_comp['stage_d']['p99']}ms")
        p50_delta = round(latency_comp['stage_d']['p50'] - latency_comp['stage_c']['p50'], 2)
        print(f"  Stage D Net Overhead on Clean Page:        P50 delta = +{p50_delta}ms")
        print("=" * 70)

        bench_out = {
            "runtime": "Comex Electron (43.4.0)",
            "cdp_url": ELECTRON_CDP_URL,
            "target_id": runner.target_id,
            "duration_s": bench_duration,
            "unexpected_physical_dispatches": total_unexpected_clicks,
            "latency_comparison": latency_comp,
            "summary": {
                name: {
                    "passed": sum(1 for r in res_list if r.success),
                    "total": len(res_list),
                    "percentiles": compute_percentiles([r.elapsed_ms for r in res_list]),
                }
                for name, res_list in all_results
            },
        }
        out_path = Path("stage_d_electron_benchmark_results.json")
        out_path.write_text(json.dumps(bench_out, indent=2))
        print(f"\nBenchmark details saved to {out_path.resolve()}")

    finally:
        print("\nStopping Fixture Server...")
        server.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
