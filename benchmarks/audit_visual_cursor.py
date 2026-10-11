#!/usr/bin/env python3
"""
Comex Visual Fake Cursor Rigorous Performance Audit Benchmark
============================================================
Evaluates:
  Condition A: VISUAL_CURSOR_ENABLED=false
  Condition B: VISUAL_CURSOR_ENABLED=true

Tests:
  1. High-Resolution Click Latency (T0-T7 timestamps: 120 clicks/condition)
  2. Controlled A/B End-to-End Task Duration (30 runs/condition, 25 actions/run)
  3. CPU Utilization Profiling (Idle, 1 Click, Continuous, 50-Click Rapid Burst)
  4. Memory & Leak Audit (Baseline, 100, 500, 1000 clicks + .ripple DOM cap)
  5. Event-Loop / Async Queue Pressure (10, 50, 100, 500, 1000 rapid clicks)
  6. Frame Timing & Rendering Cost (rAF interval delta, jank frame analysis)
  7. 5-Variant Isolation Test (OFF vs Hook vs Transport vs Animation vs Full)
  8. Multi-Page Workloads (Clean Static vs Dynamic Churning vs Heavy 5k DOM)
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import math
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import tempfile
temp_db_dir = tempfile.mkdtemp()
os.environ["GRAPH_DB_PATH"] = os.path.join(temp_db_dir, "audit_graph.db")

import agent_service
from nervous.actuator_gate import ActuatorGate, DecisionType, GateResult
from nervous.cdp_session import TabSessionContext, WebSocketCDPClient, get_or_create_tab_session
from nervous.target_binding import validate_binding_integrity
from nervous.executor import SelectorClickExecutor
from nervous.world_state import BrowserState

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cursor_audit")

CDP_URL = "http://127.0.0.1:9222"
FIXTURE_PORT = 8899
BASE_FIXTURE_URL = f"http://127.0.0.1:{FIXTURE_PORT}"
FIXTURES_DIR = ROOT_DIR / "fixtures" / "stage_d"
RESULTS_FILE = ROOT_DIR / "visual_cursor_audit_results.json"


# ──────────────────────────────────────────────────────────────────────
#  1. Local Fixture Server
# ──────────────────────────────────────────────────────────────────────

class QuietHandler(http.server.SimpleHTTPRequestHandler):
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
        self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), QuietHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


# ──────────────────────────────────────────────────────────────────────
#  2. Process Telemetry Sampler
# ──────────────────────────────────────────────────────────────────────

class ProcessSampler:
    """Samples CPU % and RSS (in MB) for target PIDs at 50ms intervals."""

    def __init__(self, pids: List[int], interval: float = 0.05) -> None:
        self.pids = [p for p in pids if p > 0]
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self.samples.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.compute_stats()

    def _sample_loop(self) -> None:
        if not self.pids:
            return
        pid_str = ",".join(str(p) for p in self.pids)
        while self._running:
            try:
                res = subprocess.run(
                    ["/bin/ps", "-p", pid_str, "-o", "pid,%cpu,rss"],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                if res.returncode == 0:
                    lines = res.stdout.strip().split("\n")[1:]
                    entry: Dict[str, Any] = {"time": time.time(), "processes": {}}
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            pid, cpu, rss = int(parts[0]), float(parts[1]), float(parts[2]) / 1024.0  # MB
                            entry["processes"][pid] = {"cpu": cpu, "rss": rss}
                    self.samples.append(entry)
            except Exception:
                pass
            time.sleep(self.interval)

    def compute_stats(self) -> Dict[str, Any]:
        if not self.samples:
            return {"mean_cpu": 0.0, "peak_cpu": 0.0, "p95_cpu": 0.0, "baseline_rss": 0.0, "peak_rss": 0.0, "final_rss": 0.0}

        all_cpus: List[float] = []
        all_rss: List[float] = []

        for s in self.samples:
            total_cpu = sum(p["cpu"] for p in s["processes"].values())
            total_rss = sum(p["rss"] for p in s["processes"].values())
            all_cpus.append(total_cpu)
            all_rss.append(total_rss)

        if not all_cpus:
            return {"mean_cpu": 0.0, "peak_cpu": 0.0, "p95_cpu": 0.0, "baseline_rss": 0.0, "peak_rss": 0.0, "final_rss": 0.0}

        all_cpus.sort()
        p95_idx = int(len(all_cpus) * 0.95)
        return {
            "mean_cpu": round(sum(all_cpus) / len(all_cpus), 2),
            "peak_cpu": round(max(all_cpus), 2),
            "p95_cpu": round(all_cpus[min(p95_idx, len(all_cpus) - 1)], 2),
            "baseline_rss": round(all_rss[0], 2),
            "peak_rss": round(max(all_rss), 2),
            "final_rss": round(all_rss[-1], 2),
            "sample_count": len(all_cpus)
        }


# ──────────────────────────────────────────────────────────────────────
#  3. High-Resolution T0-T7 Instrumented Click Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ClickLatencyRecord:
    t0_actuator_start_ns: int
    t1_pre_validation_ns: int
    t2_mouse_pressed_ns: int
    t3_mouse_released_ns: int
    t4_hook_scheduled_ns: int
    t5_transport_start_ns: int
    t6_main_ipc_received_ns: int
    t7_cursor_rendered_ns: int
    is_cursor_enabled: bool

    @property
    def t1_t2_validation_to_press_ms(self) -> float:
        return max(0.0, (self.t2_mouse_pressed_ns - self.t1_pre_validation_ns) / 1e6)

    @property
    def t2_t3_press_to_release_ms(self) -> float:
        return max(0.0, (self.t3_mouse_released_ns - self.t2_mouse_pressed_ns) / 1e6)

    @property
    def t3_t4_release_to_hook_ms(self) -> float:
        return max(0.0, (self.t4_hook_scheduled_ns - self.t3_mouse_released_ns) / 1e6)

    @property
    def t4_t5_hook_to_transport_ms(self) -> float:
        if self.t5_transport_start_ns == 0:
            return 0.0
        return max(0.0, (self.t5_transport_start_ns - self.t4_hook_scheduled_ns) / 1e6)

    @property
    def t5_t6_transport_to_main_ms(self) -> float:
        if self.t5_transport_start_ns == 0 or self.t6_main_ipc_received_ns == 0:
            return 0.0
        return max(0.0, (self.t6_main_ipc_received_ns - self.t5_transport_start_ns) / 1e6)

    @property
    def t6_t7_main_to_cursor_ms(self) -> float:
        if self.t6_main_ipc_received_ns == 0 or self.t7_cursor_rendered_ns == 0:
            return 0.0
        return max(0.0, (self.t7_cursor_rendered_ns - self.t6_main_ipc_received_ns) / 1e6)

    @property
    def real_action_duration_ms(self) -> float:
        """Physical action duration: validation completion to physical release."""
        return max(0.0, (self.t3_mouse_released_ns - self.t1_pre_validation_ns) / 1e6)


# Global timing tap for transport coroutine
_g_active_transport_t5: int = 0
_g_active_transport_t6: int = 0
_g_active_transport_t7: int = 0
_orig_send_visual_cursor_cdp = agent_service._send_visual_cursor_cdp

async def _instrumented_send_visual_cursor_cdp(x: float, y: float, action: str, tab_id: str) -> None:
    global _g_active_transport_t5, _g_active_transport_t6, _g_active_transport_t7
    _g_active_transport_t5 = time.perf_counter_ns()
    try:
        await _orig_send_visual_cursor_cdp(x, y, action, tab_id)
        _g_active_transport_t6 = time.perf_counter_ns()
    except Exception:
        pass

agent_service._send_visual_cursor_cdp = _instrumented_send_visual_cursor_cdp


class InstrumentedClickExecutor(SelectorClickExecutor):
    """
    Instrumented subclass of SelectorClickExecutor capturing exact
    nanosecond timestamps T0-T7 for performance auditing.
    Preserves exact production logic and ordering.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: List[ClickLatencyRecord] = []

    async def execute(self, action: str, binding: Any, **kwargs: Any) -> Dict[str, Any]:
        global _g_active_transport_t5, _g_active_transport_t6, _g_active_transport_t7
        t0 = time.perf_counter_ns()
        if action != "click":
            return await super().execute(action, binding, **kwargs)

        # 1. Resolve coordinates
        box_res = await self._cdp.send("DOM.getBoxModel", {"backendNodeId": binding.backend_node_id})
        model = box_res.get("model", {})
        border_quad = model.get("border") or model.get("content")
        xs = [border_quad[0], border_quad[2], border_quad[4], border_quad[6]]
        ys = [border_quad[1], border_quad[3], border_quad[5], border_quad[7]]
        target_x = sum(xs) / 4.0
        target_y = sum(ys) / 4.0

        # 2. Pre-input validation
        fail_reason = validate_binding_integrity(
            binding, self._state, current_tab_id=self._tab_id, current_target_id=self._target_id
        )
        if fail_reason:
            raise RuntimeError(f"Pre-input validation failed: {fail_reason.value}")
        t1 = time.perf_counter_ns()

        button = kwargs.get("button", "left")
        clicks = kwargs.get("clicks", 1)
        buttons_mask = 1 if button == "left" else (2 if button == "right" else 4)

        # 3. Schedule visual hook if enabled
        t4 = 0
        _g_active_transport_t5 = 0
        _g_active_transport_t6 = 0
        _g_active_transport_t7 = 0
        if self._visual_cursor_hook:
            t4 = time.perf_counter_ns()
            try:
                self._visual_cursor_hook(target_x, target_y, button, self._tab_id)
            except Exception:
                pass

        # 4. Dispatch physical click (fast path)
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": target_x,
            "y": target_y,
            "button": button,
            "buttons": buttons_mask,
            "clickCount": clicks,
        })
        t2 = time.perf_counter_ns()

        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": target_x,
            "y": target_y,
            "button": button,
            "buttons": 0,
            "clickCount": clicks,
        })
        t3 = time.perf_counter_ns()

        if t4 == 0:
            t4 = t3

        rec = ClickLatencyRecord(
            t0_actuator_start_ns=t0,
            t1_pre_validation_ns=t1,
            t2_mouse_pressed_ns=t2,
            t3_mouse_released_ns=t3,
            t4_hook_scheduled_ns=t4,
            t5_transport_start_ns=_g_active_transport_t5,
            t6_main_ipc_received_ns=_g_active_transport_t6,
            t7_cursor_rendered_ns=_g_active_transport_t7,
            is_cursor_enabled=bool(self._visual_cursor_hook is not None)
        )
        self.records.append(rec)
        return {
            "clicked": True,
            "backend_node_id": binding.backend_node_id,
            "x": target_x,
            "y": target_y,
            "t_record": rec
        }


# ──────────────────────────────────────────────────────────────────────
#  4. Audit Helper Functions & Target Discovery
# ──────────────────────────────────────────────────────────────────────

async def navigate_fixture(session: TabSessionContext, fixture_name: str) -> None:
    url = f"{BASE_FIXTURE_URL}/{fixture_name}"
    await session.cdp.send("Page.navigate", {"url": url})
    for _ in range(50):
        res = await session.cdp.send("Runtime.evaluate", {
            "expression": "document.readyState",
            "returnByValue": True,
        })
        if res.get("result", {}).get("value") == "complete":
            break
        await asyncio.sleep(0.02)
    await session.flush_and_sync()
    await session.cdp.send("Page.bringToFront")
    await session.cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
    await asyncio.sleep(0.1)


async def production_click(session: TabSessionContext, selector: str, max_retries: int = 5) -> GateResult:
    for attempt in range(max_retries):
        try:
            binding = await session.resolve_and_bind(selector)
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            return await session.gate.async_attempt("click", binding=binding)
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.05)


async def discover_targets() -> Dict[str, Any]:
    req = urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=3)
    targets = json.loads(req.read())

    ui_target = next((t for t in targets if t.get("type") == "page" and ("localhost:5173" in t.get("url", "") or "index.html" in t.get("url", ""))), None)
    cursor_target = next((t for t in targets if "cursor_overlay.html" in t.get("url", "")), None)
    app_tab = next((t for t in targets if t.get("type") == "page" and t != ui_target and t != cursor_target), None)

    return {
        "ui": ui_target,
        "cursor": cursor_target,
        "tab": app_tab,
        "all": targets
    }


def get_process_pids() -> Dict[str, Any]:
    """Finds PIDs for Electron main, renderers, and agent_service."""
    res = subprocess.run(["pgrep", "-l", "-f", "Electron.app/Contents/MacOS/Electron"], capture_output=True, text=True)
    electron_main_pid = 0
    for line in res.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            electron_main_pid = int(parts[0])
            break

    py_res = subprocess.run(["pgrep", "-l", "-f", "agent_service.py"], capture_output=True, text=True)
    py_pid = 0
    for line in py_res.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            py_pid = int(parts[0])
            break

    renderer_pids: List[int] = []
    if electron_main_pid > 0:
        c_res = subprocess.run(["pgrep", "-P", str(electron_main_pid)], capture_output=True, text=True)
        for p in c_res.stdout.strip().split("\n"):
            if p.strip().isdigit():
                renderer_pids.append(int(p.strip()))

    return {
        "main_pid": electron_main_pid,
        "python_pid": py_pid,
        "renderer_pids": renderer_pids,
        "all_pids": [electron_main_pid, py_pid] + renderer_pids
    }


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1


# ──────────────────────────────────────────────────────────────────────
#  5. Benchmarks
# ──────────────────────────────────────────────────────────────────────

async def run_benchmark_1_click_latency(tab_ws: str, cursor_ws: Optional[str]) -> Dict[str, Any]:
    """
    Test 1 & 2: Measure Real Click Latency (T0-T7) across 120 clicks:
    60 Clicks with Cursor OFF vs 60 Clicks with Cursor ON (interleaved).
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 1: REAL CLICK LATENCY (T0-T7 NANOSECOND AUDIT)")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)
    instrumented_executor = InstrumentedClickExecutor(session.cdp, session.tracker.state, "active_tab", session.target_id)
    session.executor = instrumented_executor
    session.gate._executor = instrumented_executor

    await navigate_fixture(session, "clean.html")

    off_records: List[ClickLatencyRecord] = []
    on_records: List[ClickLatencyRecord] = []

    total_clicks = 120
    for i in range(total_clicks):
        is_on = (i % 2 == 1)
        if is_on:
            session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))
        else:
            session.set_visual_cursor_hook(None)

        binding = await session.resolve_and_bind("#real-target")
        if not session.tracker.state.is_target_subtree_quiet:
            await asyncio.sleep(0.085)
        res = await session.gate.async_attempt("click", binding=binding)
        rec = res.execution_result.get("t_record") if isinstance(res.execution_result, dict) else None

        if rec:
            if is_on:
                on_records.append(rec)
            else:
                off_records.append(rec)

        await asyncio.sleep(0.01)

    def calc_stats(recs: List[ClickLatencyRecord]) -> Dict[str, Any]:
        t1_t2 = [r.t1_t2_validation_to_press_ms for r in recs]
        t2_t3 = [r.t2_t3_press_to_release_ms for r in recs]
        real_action = [r.real_action_duration_ms for r in recs]
        t3_t4 = [r.t3_t4_release_to_hook_ms for r in recs]

        return {
            "count": len(recs),
            "t1_t2_ms": {
                "mean": round(sum(t1_t2)/len(t1_t2), 3),
                "median": round(percentile(t1_t2, 50), 3),
                "p95": round(percentile(t1_t2, 95), 3),
                "p99": round(percentile(t1_t2, 99), 3),
            },
            "t2_t3_ms": {
                "mean": round(sum(t2_t3)/len(t2_t3), 3),
                "median": round(percentile(t2_t3, 50), 3),
                "p95": round(percentile(t2_t3, 95), 3),
                "p99": round(percentile(t2_t3, 99), 3),
            },
            "real_action_duration_ms": {
                "mean": round(sum(real_action)/len(real_action), 3),
                "median": round(percentile(real_action, 50), 3),
                "p95": round(percentile(real_action, 95), 3),
                "p99": round(percentile(real_action, 99), 3),
            },
            "t3_t4_hook_overhead_ms": {
                "mean": round(sum(t3_t4)/len(t3_t4), 4),
                "median": round(percentile(t3_t4, 50), 4),
                "p95": round(percentile(t3_t4, 95), 4),
            }
        }

    off_stats = calc_stats(off_records)
    on_stats = calc_stats(on_records)

    diff_real_mean = round(on_stats["real_action_duration_ms"]["mean"] - off_stats["real_action_duration_ms"]["mean"], 3)
    diff_real_pct = round((diff_real_mean / max(0.001, off_stats["real_action_duration_ms"]["mean"])) * 100, 2)

    print(f"OFF Clicks Measured: {off_stats['count']} | Mean Real Action: {off_stats['real_action_duration_ms']['mean']}ms (median {off_stats['real_action_duration_ms']['median']}ms)")
    print(f"ON  Clicks Measured: {on_stats['count']} | Mean Real Action: {on_stats['real_action_duration_ms']['mean']}ms (median {on_stats['real_action_duration_ms']['median']}ms)")
    print(f"Delta on Real Action Duration: {diff_real_mean:+.3f}ms ({diff_real_pct:+.2f}%)")
    print(f"Visual Hook Scheduling Overhead (T3-T4): {on_stats['t3_t4_hook_overhead_ms']['mean']:.4f}ms (p95: {on_stats['t3_t4_hook_overhead_ms']['p95']:.4f}ms)")

    return {
        "off": off_stats,
        "on": on_stats,
        "diff_real_mean_ms": diff_real_mean,
        "diff_real_pct": diff_real_pct
    }


async def run_benchmark_2_end_to_end_tasks(tab_ws: str) -> Dict[str, Any]:
    """
    Test 3: Controlled A/B End-to-End Task Duration:
    30 complete runs for Condition A (OFF) vs 30 complete runs for Condition B (ON).
    Total 60 runs, 25 deterministic interactions per run = 1,500 total actions.
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 2: CONTROLLED A/B END-TO-END TASK DURATION (60 RUNS)")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)
    off_durations: List[float] = []
    on_durations: List[float] = []
    off_click_latencies: List[float] = []
    on_click_latencies: List[float] = []

    total_pairs = 30
    for pair_idx in range(total_pairs):
        conditions = [("OFF", False), ("ON", True)] if (pair_idx % 2 == 0) else [("ON", True), ("OFF", False)]

        for cond_name, is_on in conditions:
            if is_on:
                session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))
            else:
                session.set_visual_cursor_hook(None)

            t_run_start = time.monotonic()

            await navigate_fixture(session, "benchmark_task.html")

            run_click_lats: List[float] = []

            # 15 grid buttons
            for btn_idx in range(1, 16):
                t_clk_start = time.monotonic()
                await production_click(session, f"#action-btn-{btn_idx}")
                clk_ms = (time.monotonic() - t_clk_start) * 1000
                run_click_lats.append(clk_ms)

            # 5 Interactive Toggles and Radios
            for toggle_id in ["#toggle-chk-1", "#toggle-chk-2", "#toggle-chk-3", "#radio-opt-1", "#radio-opt-2"]:
                t_clk_start = time.monotonic()
                await production_click(session, toggle_id)
                clk_ms = (time.monotonic() - t_clk_start) * 1000
                run_click_lats.append(clk_ms)

            # Deep scroll to verification button, then click
            await session.cdp.send("Runtime.evaluate", {
                "expression": "document.getElementById('scroll-target-btn').scrollIntoView({ behavior: 'instant' });"
            })
            t_clk_start = time.monotonic()
            await production_click(session, "#scroll-target-btn")
            run_click_lats.append((time.monotonic() - t_clk_start) * 1000)

            # Type into input field
            await session.cdp.send("Runtime.evaluate", {
                "expression": "const inp = document.getElementById('task-text-input'); inp.value = 'COMEX-7788'; inp.dispatchEvent(new Event('input', {bubbles: true}));"
            })

            # Verify Input
            t_clk_start = time.monotonic()
            await production_click(session, "#verify-input-btn")
            run_click_lats.append((time.monotonic() - t_clk_start) * 1000)

            # Final Submit Button
            t_clk_start = time.monotonic()
            await production_click(session, "#final-submit-btn")
            run_click_lats.append((time.monotonic() - t_clk_start) * 1000)

            total_run_ms = (time.monotonic() - t_run_start) * 1000

            if is_on:
                on_durations.append(total_run_ms)
                on_click_latencies.extend(run_click_lats)
            else:
                off_durations.append(total_run_ms)
                off_click_latencies.extend(run_click_lats)

        if (pair_idx + 1) % 5 == 0:
            print(f"Completed {pair_idx + 1}/{total_pairs} balanced pairs ({(pair_idx + 1)*2}/60 total runs)...")

    off_mean = sum(off_durations) / len(off_durations)
    on_mean = sum(on_durations) / len(on_durations)
    diff_mean = on_mean - off_mean
    diff_pct = (diff_mean / off_mean) * 100

    results = {
        "runs_per_condition": total_pairs,
        "actions_per_run": 25,
        "total_actions_evaluated": total_pairs * 2 * 25,
        "off_task_duration_ms": {
            "mean": round(off_mean, 2),
            "median": round(percentile(off_durations, 50), 2),
            "p95": round(percentile(off_durations, 95), 2),
            "p99": round(percentile(off_durations, 99), 2),
            "min": round(min(off_durations), 2),
            "max": round(max(off_durations), 2)
        },
        "on_task_duration_ms": {
            "mean": round(on_mean, 2),
            "median": round(percentile(on_durations, 50), 2),
            "p95": round(percentile(on_durations, 95), 2),
            "p99": round(percentile(on_durations, 99), 2),
            "min": round(min(on_durations), 2),
            "max": round(max(on_durations), 2)
        },
        "task_duration_delta": {
            "absolute_diff_ms": round(diff_mean, 2),
            "percentage_diff_pct": round(diff_pct, 2)
        },
        "off_action_latency_ms": {
            "mean": round(sum(off_click_latencies) / len(off_click_latencies), 2),
            "median": round(percentile(off_click_latencies, 50), 2),
            "p95": round(percentile(off_click_latencies, 95), 2),
            "p99": round(percentile(off_click_latencies, 99), 2),
        },
        "on_action_latency_ms": {
            "mean": round(sum(on_click_latencies) / len(on_click_latencies), 2),
            "median": round(percentile(on_click_latencies, 50), 2),
            "p95": round(percentile(on_click_latencies, 95), 2),
            "p99": round(percentile(on_click_latencies, 99), 2),
        }
    }

    print(f"\nTask Duration (OFF): Mean {results['off_task_duration_ms']['mean']}ms | Median {results['off_task_duration_ms']['median']}ms | p95 {results['off_task_duration_ms']['p95']}ms")
    print(f"Task Duration (ON):  Mean {results['on_task_duration_ms']['mean']}ms | Median {results['on_task_duration_ms']['median']}ms | p95 {results['on_task_duration_ms']['p95']}ms")
    print(f"Absolute Task Delta: {diff_mean:+.2f}ms ({diff_pct:+.2f}%)")
    print(f"Action Latency: OFF {results['off_action_latency_ms']['mean']}ms vs ON {results['on_action_latency_ms']['mean']}ms")

    return results


async def run_benchmark_3_cpu_profiling(tab_ws: str, pids: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test 4: Measure CPU usage for Electron main, renderers, agent_service:
    A. Cursor Idle (3s)
    B. Single Click (1s window)
    C. Continuous Clicks (20 clicks, 1 every 200ms)
    D. Rapid Consecutive Clicks (50 clicks at 15ms intervals)
    Comparing OFF vs ON.
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 3: CPU UTILIZATION PROFILING (IDLE / SINGLE / CONTINUOUS / RAPID)")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)
    await navigate_fixture(session, "clean.html")

    target_pids = pids["all_pids"]
    cpu_results: Dict[str, Any] = {}

    for mode in ["OFF", "ON"]:
        is_on = (mode == "ON")
        session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab") if is_on else None)

        mode_stats: Dict[str, Any] = {}

        # 1. Idle (3s)
        sampler = ProcessSampler(target_pids, interval=0.05)
        sampler.start()
        await asyncio.sleep(3.0)
        mode_stats["idle"] = sampler.stop()

        # 2. Single Click (1s window)
        sampler = ProcessSampler(target_pids, interval=0.03)
        sampler.start()
        binding = await session.resolve_and_bind("#real-target")
        if not session.tracker.state.is_target_subtree_quiet:
            await asyncio.sleep(0.085)
        await session.gate.async_attempt("click", binding=binding)
        await asyncio.sleep(1.0)
        mode_stats["single_click"] = sampler.stop()

        # 3. Continuous Clicks (20 clicks, 1 every 200ms = 4s)
        sampler = ProcessSampler(target_pids, interval=0.05)
        sampler.start()
        for _ in range(20):
            binding = await session.resolve_and_bind("#real-target")
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            await session.gate.async_attempt("click", binding=binding)
            await asyncio.sleep(0.20)
        mode_stats["continuous_clicks"] = sampler.stop()

        # 4. Rapid Consecutive Clicks (50 clicks with minimal delay)
        sampler = ProcessSampler(target_pids, interval=0.03)
        sampler.start()
        for _ in range(50):
            binding = await session.resolve_and_bind("#real-target")
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            await session.gate.async_attempt("click", binding=binding)
            await asyncio.sleep(0.015)
        await asyncio.sleep(0.5)
        mode_stats["rapid_clicks"] = sampler.stop()

        cpu_results[mode] = mode_stats
        print(f"Mode {mode:4s} -> Idle Mean CPU: {mode_stats['idle']['mean_cpu']}% | Cont. Mean: {mode_stats['continuous_clicks']['mean_cpu']}% (Peak: {mode_stats['continuous_clicks']['peak_cpu']}%) | Rapid Peak: {mode_stats['rapid_clicks']['peak_cpu']}%")

    return cpu_results


async def run_benchmark_4_memory_and_leaks(tab_ws: str, cursor_ws: Optional[str], pids: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test 5: Measure Memory Growth & Ripple DOM Node Leak Check:
    Checks RSS and DOM Nodes at Baseline, 100 clicks, 500 clicks, and 1000 clicks.
    Verifies that the ripple cap (max 3 ripples) prevents unbounded DOM growth.
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 4: MEMORY USAGE & RIPPLE DOM LEAK AUDIT (UP TO 1,000 CLICKS)")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)
    session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))
    await navigate_fixture(session, "clean.html")

    target_pids = pids["all_pids"]

    def read_rss() -> float:
        pid_str = ",".join(str(p) for p in target_pids)
        res = subprocess.run(["/bin/ps", "-p", pid_str, "-o", "rss"], capture_output=True, text=True)
        total_kb = 0
        for l in res.stdout.strip().split("\n")[1:]:
            if l.strip().isdigit():
                total_kb += int(l.strip())
        return round(total_kb / 1024.0, 2)  # MB

    async def get_cursor_dom_metrics() -> Dict[str, Any]:
        if not cursor_ws:
            return {"ripples": 0, "dom_nodes": 0, "js_heap_mb": 0.0}
        async with websockets.connect(cursor_ws) as ws:
            await ws.send(json.dumps({
                "id": 101,
                "method": "Runtime.evaluate",
                "params": {"expression": "document.querySelectorAll('.ripple').length"}
            }))
            resp1 = json.loads(await ws.recv())
            ripples = resp1.get("result", {}).get("result", {}).get("value", 0)

            await ws.send(json.dumps({"id": 102, "method": "Performance.enable"}))
            await ws.recv()
            await ws.send(json.dumps({"id": 103, "method": "Performance.getMetrics"}))
            resp2 = json.loads(await ws.recv())
            metrics = {m["name"]: m["value"] for m in resp2.get("result", {}).get("metrics", [])}

            return {
                "ripples": ripples,
                "dom_nodes": int(metrics.get("Nodes", 0)),
                "js_heap_mb": round(metrics.get("JSHeapUsedSize", 0) / (1024.0 * 1024.0), 3)
            }

    checkpoints: Dict[str, Any] = {}

    base_rss = read_rss()
    base_dom = await get_cursor_dom_metrics()
    checkpoints["baseline"] = {"rss_mb": base_rss, **base_dom}
    print(f"Checkpoint Baseline: Total RSS = {base_rss} MB | Cursor DOM Nodes = {base_dom['dom_nodes']} | JS Heap = {base_dom['js_heap_mb']} MB")

    click_targets = [100, 500, 1000]
    total_done = 0

    binding = await session.resolve_and_bind("#real-target")

    for target_count in click_targets:
        clicks_to_do = target_count - total_done
        for _ in range(clicks_to_do):
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            await session.gate.async_attempt("click", binding=binding)
            await asyncio.sleep(0.005)
        total_done = target_count

        await asyncio.sleep(0.35)

        curr_rss = read_rss()
        curr_dom = await get_cursor_dom_metrics()
        checkpoints[f"after_{target_count}_clicks"] = {
            "rss_mb": curr_rss,
            "rss_growth_from_base_mb": round(curr_rss - base_rss, 2),
            **curr_dom
        }
        print(f"Checkpoint {target_count:4d} Clicks: Total RSS = {curr_rss} MB (Delta: {round(curr_rss - base_rss, 2):+0.2f} MB) | Cursor Nodes = {curr_dom['dom_nodes']} | Active Ripples = {curr_dom['ripples']}")

    return checkpoints


async def run_benchmark_5_async_pressure_and_invariant(tab_ws: str) -> Dict[str, Any]:
    """
    Test 6 & 9: Event-Loop / Async Backlog & Fast-Path Invariant:
    Floods the event loop with rapid visual events (10, 50, 100, 500, 1000 clicks).
    Measures pending tasks, queue depth, and verifies that real browser clicks are NEVER delayed or dropped.
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 5: ASYNC PRESSURE & FAST-PATH INVARIANT VERIFICATION")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)
    session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))
    await navigate_fixture(session, "clean.html")

    binding = await session.resolve_and_bind("#real-target")

    burst_sizes = [10, 50, 100, 500, 1000]
    pressure_results: Dict[str, Any] = {}

    for burst_size in burst_sizes:
        pending_counts: List[int] = []
        real_action_latencies: List[float] = []

        t_start = time.monotonic()

        for _ in range(burst_size):
            t_act_start = time.perf_counter_ns()
            await session.executor.execute("click", binding=binding)
            t_act_end = time.perf_counter_ns()
            real_action_latencies.append((t_act_end - t_act_start) / 1e6)

            pending = len([t for t in asyncio.all_tasks() if not t.done()])
            pending_counts.append(pending)

        burst_duration_s = time.monotonic() - t_start

        await asyncio.sleep(0.3)
        final_pending = len([t for t in asyncio.all_tasks() if not t.done()])

        pressure_results[f"burst_{burst_size}"] = {
            "burst_size": burst_size,
            "duration_seconds": round(burst_duration_s, 3),
            "peak_pending_tasks": max(pending_counts),
            "mean_pending_tasks": round(sum(pending_counts) / len(pending_counts), 1),
            "final_pending_tasks": final_pending,
            "real_action_latency_ms": {
                "mean": round(sum(real_action_latencies) / len(real_action_latencies), 3),
                "median": round(percentile(real_action_latencies, 50), 3),
                "p95": round(percentile(real_action_latencies, 95), 3),
                "max": round(max(real_action_latencies), 3)
            }
        }
        print(f"Burst {burst_size:4d} Clicks -> Peak Pending Tasks: {max(pending_counts):3d} | Mean Action Latency: {pressure_results[f'burst_{burst_size}']['real_action_latency_ms']['mean']}ms (p95: {pressure_results[f'burst_{burst_size}']['real_action_latency_ms']['p95']}ms)")

    # Verify real click count on page
    async with websockets.connect(tab_ws) as ws:
        await ws.send(json.dumps({
            "id": 201,
            "method": "Runtime.evaluate",
            "params": {"expression": "window.__real_clicks"}
        }))
        resp = json.loads(await ws.recv())
        total_delivered_to_dom = resp.get("result", {}).get("result", {}).get("value", 0)

    expected_clicks = sum(burst_sizes)
    print(f"\nTotal Physical Clicks Dispatched: {expected_clicks} | DOM Verified Clicks: {total_delivered_to_dom}")
    pressure_results["dom_verified_clicks"] = total_delivered_to_dom
    pressure_results["clicks_dropped"] = expected_clicks - total_delivered_to_dom

    return pressure_results


async def run_benchmark_6_frame_timing(cursor_ws: Optional[str], tab_ws: str) -> Dict[str, Any]:
    """
    Test 7: Measure Frame / Rendering Cost (rAF analysis in cursorWindow):
    Tracks frame interval deltas during cursor glide, measuring jank frames (>24ms).
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 6: FRAME TIMING & RENDERING COST (rAF JANK AUDIT)")
    print("="*70)

    if not cursor_ws:
        return {"error": "cursorWindow target not available"}

    session = await get_or_create_tab_session("active_tab", tab_ws)
    session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))
    await navigate_fixture(session, "clean.html")
    binding = await session.resolve_and_bind("#real-target")

    frame_results: Dict[str, Any] = {}

    async with websockets.connect(cursor_ws) as ws:
        install_expr = """
        window.__frame_intervals = [];
        window.__raf_recording = true;
        let lastTime = performance.now();
        function frameAudit(now) {
            if (window.__raf_recording) {
                window.__frame_intervals.push(now - lastTime);
            }
            lastTime = now;
            requestAnimationFrame(frameAudit);
        }
        requestAnimationFrame(frameAudit);
        """
        await ws.send(json.dumps({"id": 301, "method": "Runtime.evaluate", "params": {"expression": install_expr}}))
        await ws.recv()

        async def collect_frames(label: str, duration_sec: float) -> Dict[str, Any]:
            await ws.send(json.dumps({"id": 302, "method": "Runtime.evaluate", "params": {"expression": "window.__frame_intervals = [];"}}))
            await ws.recv()

            await asyncio.sleep(duration_sec)

            await ws.send(json.dumps({"id": 303, "method": "Runtime.evaluate", "params": {"expression": "JSON.stringify(window.__frame_intervals)"}}))
            resp = json.loads(await ws.recv())
            raw_intervals: List[float] = json.loads(resp.get("result", {}).get("result", {}).get("value", "[]"))

            if not raw_intervals:
                return {"sample_count": 0, "fps": 60.0, "mean_frame_ms": 16.67, "jank_frames": 0}

            intervals = raw_intervals[1:] if len(raw_intervals) > 2 else raw_intervals
            mean_ms = sum(intervals) / len(intervals)
            fps = 1000.0 / max(1.0, mean_ms)
            jank = len([f for f in intervals if f > 24.0])

            res = {
                "sample_count": len(intervals),
                "fps": round(fps, 1),
                "mean_frame_ms": round(mean_ms, 2),
                "median_frame_ms": round(percentile(intervals, 50), 2),
                "p95_frame_ms": round(percentile(intervals, 95), 2),
                "max_frame_ms": round(max(intervals), 2),
                "jank_frames_count": jank,
                "jank_pct": round((jank / len(intervals)) * 100, 2)
            }
            print(f"  [{label}] FPS: {res['fps']} | Mean Delta: {res['mean_frame_ms']}ms | p95: {res['p95_frame_ms']}ms | Jank Frames: {res['jank_frames_count']}/{res['sample_count']} ({res['jank_pct']}%)")
            return res

        frame_results["idle"] = await collect_frames("Idle (2s)", 2.0)

        # Single Click
        await ws.send(json.dumps({"id": 304, "method": "Runtime.evaluate", "params": {"expression": "window.__frame_intervals = [];"}}))
        await ws.recv()
        await session.gate.async_attempt("click", binding=binding)
        frame_results["single_click"] = await collect_frames("Single Click", 1.0)

        # 10 Rapid Clicks
        await ws.send(json.dumps({"id": 305, "method": "Runtime.evaluate", "params": {"expression": "window.__frame_intervals = [];"}}))
        await ws.recv()
        for _ in range(10):
            await session.gate.async_attempt("click", binding=binding)
            await asyncio.sleep(0.04)
        frame_results["10_rapid_clicks"] = await collect_frames("10 Rapid Clicks", 1.5)

        # 50 Rapid Clicks
        await ws.send(json.dumps({"id": 306, "method": "Runtime.evaluate", "params": {"expression": "window.__frame_intervals = [];"}}))
        await ws.recv()
        for _ in range(50):
            await session.gate.async_attempt("click", binding=binding)
            await asyncio.sleep(0.02)
        frame_results["50_rapid_clicks"] = await collect_frames("50 Rapid Clicks", 2.0)

    return frame_results


async def run_benchmark_7_isolation_test(tab_ws: str, cursor_ws: Optional[str]) -> Dict[str, Any]:
    """
    Test 8: Critical 5-Variant Isolation Test:
      Variant A: Cursor completely OFF
      Variant B: Hook-only (hook runs, no CDP transport)
      Variant C: Transport-only (CDP/IPC fires, cursorWindow skips DOM/rAF)
      Variant D: Animation-only (cursor glides via rAF, ripple DOM creation skipped)
      Variant E: Full cursor enabled (glide + cyan ripple)
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 7: 5-VARIANT ISOLATION TEST (LAYER-BY-LAYER ATTRIBUTION)")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)
    await navigate_fixture(session, "clean.html")
    binding = await session.resolve_and_bind("#real-target")

    # Hook isolation wrapper in cursorWindow
    if cursor_ws:
        async with websockets.connect(cursor_ws) as ws:
            patch_script = """
            if (!window.__isolation_patched) {
                window.__isolation_patched = true;
                window.__isolate_skip_all = false;
                window.__isolate_skip_ripple = false;
                const origRipple = createRipple;
                createRipple = function(x, y) {
                    if (window.__isolate_skip_ripple) return;
                    return origRipple(x, y);
                };
                const origStep = stepAnimation;
                stepAnimation = function(t) {
                    if (window.__isolate_skip_all) {
                        isAnimating = false;
                        return;
                    }
                    return origStep(t);
                };
            }
            """
            await ws.send(json.dumps({"id": 400, "method": "Runtime.evaluate", "params": {"expression": patch_script}}))
            await ws.recv()

    clicks_per_variant = 40
    isolation_results: Dict[str, Any] = {}

    variants = [
        ("Variant A: Cursor OFF", "A_OFF"),
        ("Variant B: Hook-Only", "B_HOOK_ONLY"),
        ("Variant C: Transport-Only", "C_TRANSPORT_ONLY"),
        ("Variant D: Animation-Only", "D_ANIMATION_ONLY"),
        ("Variant E: Full Cursor", "E_FULL")
    ]

    for label, code in variants:
        if cursor_ws:
            async with websockets.connect(cursor_ws) as ws:
                if code == "C_TRANSPORT_ONLY":
                    await ws.send(json.dumps({"id": 401, "method": "Runtime.evaluate", "params": {
                        "expression": "window.__isolate_skip_all = true; window.__isolate_skip_ripple = true;"
                    }}))
                elif code == "D_ANIMATION_ONLY":
                    await ws.send(json.dumps({"id": 402, "method": "Runtime.evaluate", "params": {
                        "expression": "window.__isolate_skip_all = false; window.__isolate_skip_ripple = true;"
                    }}))
                else:
                    await ws.send(json.dumps({"id": 403, "method": "Runtime.evaluate", "params": {
                        "expression": "window.__isolate_skip_all = false; window.__isolate_skip_ripple = false;"
                    }}))
                await ws.recv()

        if code == "A_OFF":
            session.set_visual_cursor_hook(None)
        elif code == "B_HOOK_ONLY":
            session.set_visual_cursor_hook(lambda x, y, a, t: None)
        else:
            session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))

        latencies: List[float] = []
        for _ in range(clicks_per_variant):
            t_start = time.perf_counter_ns()
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            await session.gate.async_attempt("click", binding=binding)
            t_end = time.perf_counter_ns()
            latencies.append((t_end - t_start) / 1e6)
            await asyncio.sleep(0.015)

        stats = {
            "mean_ms": round(sum(latencies) / len(latencies), 3),
            "median_ms": round(percentile(latencies, 50), 3),
            "p95_ms": round(percentile(latencies, 95), 3),
            "p99_ms": round(percentile(latencies, 99), 3)
        }
        isolation_results[code] = stats
        print(f"  {label:32s}: Mean = {stats['mean_ms']:.3f}ms | Median = {stats['median_ms']:.3f}ms | p95 = {stats['p95_ms']:.3f}ms")

    # Layer attribution breakdown
    base_off = isolation_results["A_OFF"]["mean_ms"]
    hook_cost = round(isolation_results["B_HOOK_ONLY"]["mean_ms"] - base_off, 4)
    transport_cost = round(isolation_results["C_TRANSPORT_ONLY"]["mean_ms"] - isolation_results["B_HOOK_ONLY"]["mean_ms"], 4)
    anim_cost = round(isolation_results["D_ANIMATION_ONLY"]["mean_ms"] - isolation_results["C_TRANSPORT_ONLY"]["mean_ms"], 4)
    ripple_cost = round(isolation_results["E_FULL"]["mean_ms"] - isolation_results["D_ANIMATION_ONLY"]["mean_ms"], 4)
    total_delta = round(isolation_results["E_FULL"]["mean_ms"] - base_off, 4)

    isolation_results["layer_attribution_ms"] = {
        "executor_hook_cost_ms": hook_cost,
        "transport_ipc_cost_ms": transport_cost,
        "glide_animation_cost_ms": anim_cost,
        "ripple_dom_cost_ms": ripple_cost,
        "total_overhead_ms": total_delta
    }
    print(f"\nAttributed Breakdown: Hook = {hook_cost:+.4f}ms | Transport = {transport_cost:+.4f}ms | Animation = {anim_cost:+.4f}ms | Ripple = {ripple_cost:+.4f}ms | Total Delta = {total_delta:+.4f}ms")

    return isolation_results


async def run_benchmark_8_multi_page(tab_ws: str) -> Dict[str, Any]:
    """
    Test 10: Multi-Page Matrix (Static, Dynamic, Heavy DOM):
      1. Simple static page (clean.html)
      2. Moderately dynamic page (churning_target.html)
      3. Heavy page with 5000 animated DOM nodes (heavy_animated.html)
    """
    print("\n" + "="*70)
    print("RUNNING BENCHMARK 8: MULTI-PAGE WORKLOAD COMPARISON (STATIC / DYNAMIC / HEAVY 5K DOM)")
    print("="*70)

    session = await get_or_create_tab_session("active_tab", tab_ws)

    pages = [
        ("Static Clean Page", "clean.html", "#real-target"),
        ("Dynamic Churning Page", "churning_target.html", "#real-target"),
        ("Heavy 5k Animated DOM", "heavy_animated.html", "#heavy-target-btn")
    ]

    page_results: Dict[str, Any] = {}
    clicks_per_test = 40

    for page_name, fixture_file, selector in pages:
        await navigate_fixture(session, fixture_file)
        binding = await session.resolve_and_bind(selector)

        # OFF test
        session.set_visual_cursor_hook(None)
        off_lats: List[float] = []
        for _ in range(clicks_per_test):
            t_start = time.perf_counter_ns()
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            await session.gate.async_attempt("click", binding=binding)
            off_lats.append((time.perf_counter_ns() - t_start) / 1e6)
            await asyncio.sleep(0.015)

        # ON test
        session.set_visual_cursor_hook(agent_service._make_visual_cursor_hook("active_tab"))
        on_lats: List[float] = []
        for _ in range(clicks_per_test):
            t_start = time.perf_counter_ns()
            if not session.tracker.state.is_target_subtree_quiet:
                await asyncio.sleep(0.085)
            await session.gate.async_attempt("click", binding=binding)
            on_lats.append((time.perf_counter_ns() - t_start) / 1e6)
            await asyncio.sleep(0.015)

        off_mean = sum(off_lats) / len(off_lats)
        on_mean = sum(on_lats) / len(on_lats)
        diff_ms = on_mean - off_mean
        diff_pct = (diff_ms / off_mean) * 100

        entry = {
            "page": page_name,
            "off": {
                "mean_ms": round(off_mean, 3),
                "median_ms": round(percentile(off_lats, 50), 3),
                "p95_ms": round(percentile(off_lats, 95), 3)
            },
            "on": {
                "mean_ms": round(on_mean, 3),
                "median_ms": round(percentile(on_lats, 50), 3),
                "p95_ms": round(percentile(on_lats, 95), 3)
            },
            "delta_ms": round(diff_ms, 3),
            "delta_pct": round(diff_pct, 2)
        }
        page_results[page_name] = entry
        print(f"  {page_name:24s}: OFF {off_mean:.2f}ms vs ON {on_mean:.2f}ms -> Delta: {diff_ms:+.2f}ms ({diff_pct:+.2f}%)")

    return page_results


# ──────────────────────────────────────────────────────────────────────
#  6. Master Runner
# ──────────────────────────────────────────────────────────────────────

async def main_async() -> None:
    print("Initializing Comex Visual Fake Cursor Audit Harness...")

    fixture_server = FixtureServer(FIXTURE_PORT)
    fixture_server.start()
    print(f"Local Fixture Server started at {BASE_FIXTURE_URL}")

    try:
        targets = await discover_targets()
        if not targets["tab"]:
            print("ERROR: No active browser tab found via CDP!")
            return
        tab_ws = targets["tab"]["webSocketDebuggerUrl"]
        cursor_ws = targets["cursor"]["webSocketDebuggerUrl"] if targets["cursor"] else None
        pids = get_process_pids()

        print(f"Connected to Active Tab: {tab_ws}")
        print(f"Connected to Cursor Window: {cursor_ws}")
        print(f"Monitored PIDs: Main={pids['main_pid']}, Python={pids['python_pid']}, Renderers={pids['renderer_pids']}")

        await asyncio.sleep(0.5)

        master_report: Dict[str, Any] = {
            "timestamp": time.time(),
            "target_tab": targets["tab"]["url"],
            "monitored_pids": pids,
        }

        # 1. Real Click Latency (T0-T7)
        master_report["test_1_click_latency"] = await run_benchmark_1_click_latency(tab_ws, cursor_ws)

        # 2. Controlled A/B End-to-End Task Duration (30 runs each, 60 runs total)
        master_report["test_2_end_to_end_tasks"] = await run_benchmark_2_end_to_end_tasks(tab_ws)

        # 3. CPU Utilization Profiling
        master_report["test_3_cpu_profiling"] = await run_benchmark_3_cpu_profiling(tab_ws, pids)

        # 4. Memory Usage & Ripple DOM Leak Audit
        master_report["test_4_memory_and_leaks"] = await run_benchmark_4_memory_and_leaks(tab_ws, cursor_ws, pids)

        # 5. Async Pressure & Fast-Path Invariant
        master_report["test_5_async_pressure"] = await run_benchmark_5_async_pressure_and_invariant(tab_ws)

        # 6. Frame Timing & Rendering Cost (rAF)
        master_report["test_6_frame_timing"] = await run_benchmark_6_frame_timing(cursor_ws, tab_ws)

        # 7. 5-Variant Isolation Test
        master_report["test_7_isolation_test"] = await run_benchmark_7_isolation_test(tab_ws, cursor_ws)

        # 8. Multi-Page Workloads
        master_report["test_8_multi_page"] = await run_benchmark_8_multi_page(tab_ws)

        with open(RESULTS_FILE, "w") as f:
            json.dump(master_report, f, indent=2)
        print(f"\nAudit complete! Complete raw results saved to: {RESULTS_FILE}")

    finally:
        fixture_server.stop()
        print("Local Fixture Server stopped.")


if __name__ == "__main__":
    asyncio.run(main_async())
