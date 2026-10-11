#!/usr/bin/env python3
"""
Sequential vs. Parallel Multi-Agent Performance Benchmark
==========================================================
Compares sequential single-agent execution against concurrent two-worker
orchestration on a representative two-part research task:
1. Technical specifications research (CPU, RAM, battery life, display).
2. Pricing and community review evaluation (price, rating, pros & cons).

Both modes execute against:
- Live Electron Chromium runtime via Chrome DevTools Protocol (CDP port 9222).
- Authoritative ComexToolService with separate server-issued tab leases.
- Live LLM reasoning turns via 9Router proxy (ag/gemini-3.8-flash).
- Local deterministic fixtures served on http://127.0.0.1:8899/.
"""

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from nervous.action_policy import (
    ServerTabLeaseRegistry,
    ServerTabTargetRegistry,
    get_lease_registry,
    get_target_registry,
)
from services.orchestrator import (
    AggregatedResult,
    MultiAgentOrchestrator,
    OrchestratorTask,
    RunStatus,
    TaskStatus,
    WorkerResult,
)
from services.tool_service import (
    ComexToolService,
    ToolExecutionContext,
    create_tool_execution_context,
    get_context_authority,
)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark.seq_vs_par")

CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "") or "router-local-key"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "ag/gemini-3.8-flash")
FIXTURE_PORT = 8899

URL_SPECS = f"http://127.0.0.1:{FIXTURE_PORT}/laptop_specs.html"
URL_REVIEWS = f"http://127.0.0.1:{FIXTURE_PORT}/laptop_reviews.html"


def compute_stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    def pct(p: float) -> float:
        idx = int(math.ceil(p * n)) - 1
        return s[max(0, min(n - 1, idx))]
    return {
        "count": n,
        "mean": round(mean, 2),
        "p50": round(pct(0.50), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
    }


def start_fixture_server() -> socketserver.TCPServer:
    fixtures_dir = ROOT_DIR / "fixtures" / "stage_d"
    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(fixtures_dir), **kwargs)
        def log_message(self, format, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", FIXTURE_PORT), _QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


def discover_electron_tabs() -> List[Dict[str, str]]:
    req = urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=3)
    targets = json.loads(req.read().decode())
    browser_targets = [
        t for t in targets
        if t.get("type") == "page"
        and "localhost:5173" not in t.get("url", "")
        and "cursor_overlay" not in t.get("url", "")
    ]
    if len(browser_targets) < 2:
        raise RuntimeError(
            f"Expected at least 2 distinct browser tabs on port 9222, found {len(browser_targets)}. "
            "Please open a second tab in Comex."
        )
    return browser_targets[:2]


async def run_sequential_trial(
    llm: ChatOpenAI,
    tool_service: ComexToolService,
    tab_id: str,
    lease_registry: ServerTabLeaseRegistry,
) -> Dict[str, Any]:
    """
    Executes Subtask 1 then Subtask 2 sequentially in a single agent workflow.
    """
    t_start = time.perf_counter()
    metrics = {
        "mode": "sequential",
        "total_wall_ms": 0.0,
        "llm_time_ms": 0.0,
        "llm_calls": 0,
        "browser_time_ms": 0.0,
        "browser_calls": 0,
        "navigation_count": 0,
        "success": False,
        "extracted_specs": None,
        "extracted_reviews": None,
        "summary": "",
    }

    ctx_seq = create_tool_execution_context(
        task_id=f"seq_{int(time.time()*1000)}",
        run_id=f"seq_run_{int(time.time()*1000)}",
        target_tab=tab_id,
        trusted_server_context=True,
        timeout_seconds=60.0,
        lease_registry=lease_registry,
    )

    # --- Step 1: Research Specifications ---
    t_b0 = time.perf_counter()
    nav1 = await tool_service.navigate(ctx_seq, URL_SPECS)
    metrics["browser_calls"] += 1
    metrics["navigation_count"] += 1
    if not nav1.success:
        metrics["total_wall_ms"] = (time.perf_counter() - t_start) * 1000.0
        return metrics

    ext1 = await tool_service.extract(ctx_seq, selector="table")
    metrics["browser_calls"] += 1
    metrics["browser_time_ms"] += (time.perf_counter() - t_b0) * 1000.0

    # LLM Step 1: Synthesize Specs
    t_l0 = time.perf_counter()
    prompt1 = f"Extract and summarize the hardware specs (CPU, RAM, Battery) from this DOM text:\n{ext1.data.get('text', '')}"
    res1 = await llm.ainvoke([HumanMessage(content=prompt1)])
    metrics["llm_time_ms"] += (time.perf_counter() - t_l0) * 1000.0
    metrics["llm_calls"] += 1
    specs_text = res1.content.strip()
    metrics["extracted_specs"] = specs_text

    # --- Step 2: Research Pricing & Reviews ---
    t_b1 = time.perf_counter()
    nav2 = await tool_service.navigate(ctx_seq, URL_REVIEWS)
    metrics["browser_calls"] += 1
    metrics["navigation_count"] += 1
    if not nav2.success:
        metrics["total_wall_ms"] = (time.perf_counter() - t_start) * 1000.0
        return metrics

    ext2 = await tool_service.extract(ctx_seq, selector=".price-box, .verdict-box")
    metrics["browser_calls"] += 1
    metrics["browser_time_ms"] += (time.perf_counter() - t_b1) * 1000.0

    # LLM Step 2: Synthesize Reviews & Price
    t_l1 = time.perf_counter()
    prompt2 = f"Extract and summarize the deal price and overall rating/verdict from this DOM text:\n{ext2.data.get('text', '')}"
    res2 = await llm.ainvoke([HumanMessage(content=prompt2)])
    metrics["llm_time_ms"] += (time.perf_counter() - t_l1) * 1000.0
    metrics["llm_calls"] += 1
    reviews_text = res2.content.strip()
    metrics["extracted_reviews"] = reviews_text

    # --- Step 3: LLM Synthesis ---
    t_l2 = time.perf_counter()
    synth_prompt = (
        f"Combine these findings into a laptop recommendation:\n"
        f"Technical Specifications:\n{specs_text}\n\n"
        f"Pricing and Reviews:\n{reviews_text}\n"
    )
    res_synth = await llm.ainvoke([HumanMessage(content=synth_prompt)])
    metrics["llm_time_ms"] += (time.perf_counter() - t_l2) * 1000.0
    metrics["llm_calls"] += 1
    metrics["summary"] = res_synth.content.strip()

    metrics["total_wall_ms"] = (time.perf_counter() - t_start) * 1000.0

    # Verify correctness
    has_cpu = "13700h" in specs_text.lower() or "intel" in specs_text.lower() or "i7" in specs_text.lower()
    has_price = "76,499" in reviews_text or "76499" in reviews_text or "4.7" in reviews_text
    metrics["success"] = bool(has_cpu and has_price)

    return metrics


async def run_parallel_trial(
    llm: ChatOpenAI,
    orchestrator: MultiAgentOrchestrator,
    tab_specs_id: str,
    tab_reviews_id: str,
) -> Dict[str, Any]:
    """
    Executes Subtask 1 and Subtask 2 concurrently using the two-worker MultiAgentOrchestrator.
    """
    t_start = time.perf_counter()
    worker_metrics = {
        "w_specs": {"llm_time_ms": 0.0, "llm_calls": 0, "browser_time_ms": 0.0, "browser_calls": 0, "nav_count": 0},
        "w_reviews": {"llm_time_ms": 0.0, "llm_calls": 0, "browser_time_ms": 0.0, "browser_calls": 0, "nav_count": 0},
    }

    async def worker_specs_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        t_b0 = time.perf_counter()
        nav = await svc.navigate(ctx, URL_SPECS)
        worker_metrics["w_specs"]["browser_calls"] += 1
        worker_metrics["w_specs"]["nav_count"] += 1
        if not nav.success:
            return WorkerResult(findings=None, success=False, error=nav.error)

        ext = await svc.extract(ctx, selector="table")
        worker_metrics["w_specs"]["browser_calls"] += 1
        worker_metrics["w_specs"]["browser_time_ms"] += (time.perf_counter() - t_b0) * 1000.0

        t_l0 = time.perf_counter()
        prompt = f"Extract and summarize the hardware specs (CPU, RAM, Battery) from this DOM text:\n{ext.data.get('text', '')}"
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        worker_metrics["w_specs"]["llm_time_ms"] += (time.perf_counter() - t_l0) * 1000.0
        worker_metrics["w_specs"]["llm_calls"] += 1

        return WorkerResult(
            findings=res.content.strip(),
            source_urls=[URL_SPECS],
            success=True,
            tool_calls_count=2,
        )

    async def worker_reviews_fn(ctx: ToolExecutionContext, svc: ComexToolService):
        t_b0 = time.perf_counter()
        nav = await svc.navigate(ctx, URL_REVIEWS)
        worker_metrics["w_reviews"]["browser_calls"] += 1
        worker_metrics["w_reviews"]["nav_count"] += 1
        if not nav.success:
            return WorkerResult(findings=None, success=False, error=nav.error)

        ext = await svc.extract(ctx, selector=".price-box, .verdict-box")
        worker_metrics["w_reviews"]["browser_calls"] += 1
        worker_metrics["w_reviews"]["browser_time_ms"] += (time.perf_counter() - t_b0) * 1000.0

        t_l0 = time.perf_counter()
        prompt = f"Extract and summarize the deal price and overall rating/verdict from this DOM text:\n{ext.data.get('text', '')}"
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        worker_metrics["w_reviews"]["llm_time_ms"] += (time.perf_counter() - t_l0) * 1000.0
        worker_metrics["w_reviews"]["llm_calls"] += 1

        return WorkerResult(
            findings=res.content.strip(),
            source_urls=[URL_REVIEWS],
            success=True,
            tool_calls_count=2,
        )

    task_1 = OrchestratorTask(
        id="specs_task",
        description="Specifications research",
        assigned_tab_id=tab_specs_id,
        task_fn=worker_specs_fn,
    )
    task_2 = OrchestratorTask(
        id="reviews_task",
        description="Pricing and reviews research",
        assigned_tab_id=tab_reviews_id,
        task_fn=worker_reviews_fn,
    )

    run = orchestrator.create_run(
        user_request="Evaluate ProBook Ultra 14 laptop",
        tasks=[task_1, task_2],
    )

    t_orch_start = time.perf_counter()
    agg = await orchestrator.execute_run(run)
    orch_time_ms = (time.perf_counter() - t_orch_start) * 1000.0

    # LLM Step 3: Synthesis of aggregated findings
    t_synth0 = time.perf_counter()
    specs_res = run.tasks["specs_task"].result.findings if run.tasks["specs_task"].result else ""
    reviews_res = run.tasks["reviews_task"].result.findings if run.tasks["reviews_task"].result else ""
    synth_prompt = (
        f"Combine these findings into a laptop recommendation:\n"
        f"Technical Specifications:\n{specs_res}\n\n"
        f"Pricing and Reviews:\n{reviews_res}\n"
    )
    synth_res = await llm.ainvoke([HumanMessage(content=synth_prompt)])
    synth_llm_time = (time.perf_counter() - t_synth0) * 1000.0

    total_wall_ms = (time.perf_counter() - t_start) * 1000.0

    total_llm_time = (
        worker_metrics["w_specs"]["llm_time_ms"]
        + worker_metrics["w_reviews"]["llm_time_ms"]
        + synth_llm_time
    )
    total_llm_calls = (
        worker_metrics["w_specs"]["llm_calls"]
        + worker_metrics["w_reviews"]["llm_calls"]
        + 1
    )
    total_browser_time = (
        worker_metrics["w_specs"]["browser_time_ms"]
        + worker_metrics["w_reviews"]["browser_time_ms"]
    )
    total_browser_calls = (
        worker_metrics["w_specs"]["browser_calls"]
        + worker_metrics["w_reviews"]["browser_calls"]
    )
    total_nav_count = (
        worker_metrics["w_specs"]["nav_count"]
        + worker_metrics["w_reviews"]["nav_count"]
    )

    # Check correctness
    has_cpu = "13700h" in specs_res.lower() or "intel" in specs_res.lower() or "i7" in specs_res.lower()
    has_price = "76,499" in reviews_res or "76499" in reviews_res or "4.7" in reviews_res
    success = bool(has_cpu and has_price and not agg.has_failures)

    return {
        "mode": "parallel",
        "total_wall_ms": total_wall_ms,
        "llm_time_ms": total_llm_time,
        "llm_calls": total_llm_calls,
        "browser_time_ms": total_browser_time,
        "browser_calls": total_browser_calls,
        "navigation_count": total_nav_count,
        "orchestration_overhead_ms": max(0.0, total_wall_ms - (max(worker_metrics["w_specs"]["llm_time_ms"] + worker_metrics["w_specs"]["browser_time_ms"], worker_metrics["w_reviews"]["llm_time_ms"] + worker_metrics["w_reviews"]["browser_time_ms"]) + synth_llm_time)),
        "success": success,
        "extracted_specs": specs_res,
        "extracted_reviews": reviews_res,
        "summary": synth_res.content.strip(),
    }


async def main():
    print("=" * 70)
    print("Comex Benchmark: Sequential Single-Agent vs. Parallel Multi-Agent")
    print("=" * 70)

    server = start_fixture_server()
    print(f"✓ Fixture server running on http://127.0.0.1:{FIXTURE_PORT}")

    tabs = discover_electron_tabs()
    tab1 = tabs[0]
    tab2 = tabs[1]
    tab1_id = tab1["id"]
    tab2_id = tab2["id"]
    ws1 = tab1["webSocketDebuggerUrl"]
    ws2 = tab2["webSocketDebuggerUrl"]

    print(f"✓ Target Tab 1: {tab1_id} ({tab1.get('title', '')[:30]})")
    print(f"✓ Target Tab 2: {tab2_id} ({tab2.get('title', '')[:30]})")

    lease_reg = get_lease_registry()
    target_reg = get_target_registry()
    target_reg.register_tab_target(tab1_id, tab1_id, ws1)
    target_reg.register_tab_target("tab_specs", tab1_id, ws1)
    target_reg.register_tab_target(tab2_id, tab2_id, ws2)
    target_reg.register_tab_target("tab_reviews", tab2_id, ws2)

    tool_service = ComexToolService(
        lease_registry=lease_reg,
        target_registry=target_reg,
        default_cdp_url=ws1,
    )

    orchestrator = MultiAgentOrchestrator(
        tool_service=tool_service,
        lease_registry=lease_reg,
        target_registry=target_reg,
        max_concurrency=2,
    )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=ROUTER_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.0,
    )

    num_trials = 5
    sequential_runs: List[Dict[str, Any]] = []
    parallel_runs: List[Dict[str, Any]] = []

    print("\n[Warm-up] Executing 1 warm-up trial per mode...")
    await run_sequential_trial(llm, tool_service, tab1_id, lease_reg)
    await run_parallel_trial(llm, orchestrator, "tab_specs", "tab_reviews")
    print("✓ Warm-up complete.\n")

    print(f"Running {num_trials} balanced trials (alternating Sequential and Parallel)...")
    for i in range(1, num_trials + 1):
        if i % 2 == 1:
            print(f"\n--- Trial {i}/{num_trials}: Sequential first, then Parallel ---")
            t_s = await run_sequential_trial(llm, tool_service, tab1_id, lease_reg)
            sequential_runs.append(t_s)
            print(f"  Sequential: {t_s['total_wall_ms']:.1f}ms (LLM: {t_s['llm_time_ms']:.1f}ms, Browser: {t_s['browser_time_ms']:.1f}ms)")

            t_p = await run_parallel_trial(llm, orchestrator, "tab_specs", "tab_reviews")
            parallel_runs.append(t_p)
            print(f"  Parallel:   {t_p['total_wall_ms']:.1f}ms (LLM: {t_p['llm_time_ms']:.1f}ms, Browser: {t_p['browser_time_ms']:.1f}ms)")
        else:
            print(f"\n--- Trial {i}/{num_trials}: Parallel first, then Sequential ---")
            t_p = await run_parallel_trial(llm, orchestrator, "tab_specs", "tab_reviews")
            parallel_runs.append(t_p)
            print(f"  Parallel:   {t_p['total_wall_ms']:.1f}ms (LLM: {t_p['llm_time_ms']:.1f}ms, Browser: {t_p['browser_time_ms']:.1f}ms)")

            t_s = await run_sequential_trial(llm, tool_service, tab1_id, lease_reg)
            sequential_runs.append(t_s)
            print(f"  Sequential: {t_s['total_wall_ms']:.1f}ms (LLM: {t_s['llm_time_ms']:.1f}ms, Browser: {t_s['browser_time_ms']:.1f}ms)")

    # Aggregate Statistics
    seq_walls = [r["total_wall_ms"] for r in sequential_runs]
    par_walls = [r["total_wall_ms"] for r in parallel_runs]

    seq_stats = compute_stats(seq_walls)
    par_stats = compute_stats(par_walls)

    abs_saved = round(seq_stats["p50"] - par_stats["p50"], 2)
    pct_reduction = round(((seq_stats["p50"] - par_stats["p50"]) / seq_stats["p50"]) * 100.0, 2)
    speedup_ratio = round(seq_stats["p50"] / par_stats["p50"], 2)

    summary_data = {
        "environment": {
            "model": MODEL_NAME,
            "llm_proxy": OPENAI_BASE_URL,
            "cdp_port": 9222,
            "tabs_used": [tab1_id, tab2_id],
            "trial_count": num_trials,
        },
        "trials": {
            "sequential": sequential_runs,
            "parallel": parallel_runs,
        },
        "stats": {
            "sequential_wall_ms": seq_stats,
            "parallel_wall_ms": par_stats,
            "absolute_saved_ms": abs_saved,
            "percentage_reduction_pct": pct_reduction,
            "speedup_ratio": speedup_ratio,
        },
    }

    # Save JSON summary
    summary_path = ROOT_DIR / "benchmarks" / "sequential_vs_parallel_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n✓ Saved raw benchmark summary to: {summary_path}")

    # Display comparison table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY (Medians & Ratios)")
    print("=" * 70)
    print(f"Sequential P50 Total Wall Time: {seq_stats['p50']}ms (Min: {seq_stats['min']}ms, Max: {seq_stats['max']}ms)")
    print(f"Parallel   P50 Total Wall Time: {par_stats['p50']}ms (Min: {par_stats['min']}ms, Max: {par_stats['max']}ms)")
    print(f"Absolute Time Saved:            {abs_saved}ms")
    print(f"Percentage Time Reduction:      {pct_reduction}%")
    print(f"Speedup Ratio:                  {speedup_ratio}x")
    print("=" * 70)

    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
