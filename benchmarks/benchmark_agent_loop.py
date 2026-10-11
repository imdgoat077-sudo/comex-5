#!/usr/bin/env python3
"""
Comex End-to-End Agent Loop Benchmark
======================================
Measures live execution of execute_agent_loop() against a local deterministic fixture:
- Total end-to-end task duration (ms)
- Time to first token (TTFT ms)
- Total LLM inference duration (ms) and roundtrip count
- Total Tool execution duration (ms) and individual tool calls
- Graph memory prep and DB query latency (ms)
- Task success and correctness
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
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import agent_service
from services.tool_service import ComexToolService

logging.basicConfig(level=logging.WARNING)


def compute_stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
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
        "p95": round(pct(0.95), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
    }


async def main() -> None:
    fixtures_dir = ROOT_DIR / "fixtures" / "stage_d"
    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(fixtures_dir), **kwargs)
        def log_message(self, format, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", 8899), _QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Discover active Electron tab
        req = urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=3)
        targets = json.loads(req.read().decode())
        browser_targets = [
            t for t in targets
            if t.get("type") == "page"
            and "localhost:5173" not in t.get("url", "")
            and "cursor_overlay" not in t.get("url", "")
        ]
        if not browser_targets:
            raise RuntimeError("No active WebContentsView tab found.")

        target = browser_targets[0]
        target_id = target["id"]
        ws_url = target["webSocketDebuggerUrl"]

        from nervous.action_policy import get_target_registry
        get_target_registry().register_tab_target(target_id, target_id, ws_url)
        get_target_registry().register_tab_target("active_tab", target_id, ws_url)

        tool_service = ComexToolService()

        trials = 5
        trial_results = []
        prompt = "Navigate to http://127.0.0.1:8899/clean.html, click the button with selector '#real-target', and read the text of '#status'."

        print(f"\n--- Benchmarking End-to-End Agent Loop ({trials} trials) ---")
        print(f"Target Tab: {target_id}")
        print(f"Prompt: {prompt}\n")

        for i in range(1, trials + 1):
            print(f"Trial {i}/{trials}...", end=" ", flush=True)
            t0 = time.perf_counter()
            res = await agent_service.execute_agent_loop(
                user_prompt=prompt,
                default_cdp_url=ws_url,
                trusted_app_context=True,
                tab_id=target_id,
                tool_service=tool_service,
            )
            dur = (time.perf_counter() - t0) * 1000.0
            
            meta = res.metadata
            timing = meta.get("timing", {})
            outcome = meta.get("outcome")
            success = "Clicked" in res.response or "1 times" in res.response or outcome == "completed"

            print(f"Done ({dur:.1f}ms) | Outcome: {outcome} | Tools: {meta.get('tool_call_count')} | LLM: {len(timing.get('llm_roundtrips_ms', []))} turns")
            trial_results.append({
                "trial": i,
                "total_duration_ms": dur,
                "timing": timing,
                "outcome": outcome,
                "tool_call_count": meta.get("tool_call_count"),
                "failed_tool_call_count": meta.get("failed_tool_call_count"),
                "tool_traces": meta.get("tool_traces", []),
                "success": success,
                "response_snippet": res.response[:150],
            })

        # Summary statistics
        total_durs = [r["total_duration_ms"] for r in trial_results]
        llm_durs = [r["timing"].get("total_llm_ms", 0.0) for r in trial_results]
        tool_durs = [r["timing"].get("total_tool_ms", 0.0) for r in trial_results]
        ttfts = [r["timing"].get("time_to_first_token_ms", 0.0) for r in trial_results if r["timing"].get("time_to_first_token_ms") is not None]

        stats_total = compute_stats(total_durs)
        stats_llm = compute_stats(llm_durs)
        stats_tool = compute_stats(tool_durs)
        stats_ttft = compute_stats(ttfts)

        print("\n" + "=" * 60)
        print("AGENT LOOP E2E PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Duration (ms): p50={stats_total['p50']}ms, p95={stats_total['p95']}ms, mean={stats_total['mean']}ms")
        print(f"TTFT (ms)          : p50={stats_ttft['p50']}ms, p95={stats_ttft['p95']}ms, mean={stats_ttft['mean']}ms")
        print(f"LLM Inference (ms) : p50={stats_llm['p50']}ms, p95={stats_llm['p95']}ms, mean={stats_llm['mean']}ms")
        print(f"Tool Execution (ms): p50={stats_tool['p50']}ms, p95={stats_tool['p95']}ms, mean={stats_tool['mean']}ms")
        print(f"Success Rate       : {sum(1 for r in trial_results if r['success'])}/{len(trial_results)} ({sum(1 for r in trial_results if r['success'])/len(trial_results)*100:.1f}%)")
        print("=" * 60)

        out_data = {
            "prompt": prompt,
            "target_tab": target_id,
            "trials_count": trials,
            "summary": {
                "total_duration_ms": stats_total,
                "ttft_ms": stats_ttft,
                "llm_inference_ms": stats_llm,
                "tool_execution_ms": stats_tool,
                "success_rate": sum(1 for r in trial_results if r['success']) / len(trial_results),
            },
            "trials": trial_results,
        }
        out_file = ROOT_DIR / "benchmarks" / "agent_e2e_benchmark_results.json"
        out_file.write_text(json.dumps(out_data, indent=2))
        print(f"Saved E2E benchmark data to {out_file}")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    asyncio.run(main())
