#!/usr/bin/env python3
"""
Comex Subsystem Performance Benchmark
======================================
Measures:
1. Graph Memory subsystem:
   - BM25 full-text search across domains (ALL, ENTITIES, ARTIFACTS, SKILLS, USER)
   - Entity query (find_recent_entities with staleness filtering)
   - Artifact operations (save_artifact, read_artifact chunked)
   - User memory & skill persistence (save_user_memory, save_skill)
2. Tool Execution Service (ComexToolService):
   - browser_inspect (DOM tree extraction & accessibility node computation)
   - browser_click (selector resolution, binding, physical dispatch)
   - browser_type (text entry dispatch)
   - browser_extract (text extraction)
   - browser_navigate (page load and readyState settlement)

Runs on live Comex environment:
- Embedded Kuzu DB at ~/.comex/graph.db
- Live Electron CDP at http://127.0.0.1:9222
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from graph.store import AgentGraphStore
from services.tool_service import ComexToolService, ToolExecutionContext

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("comex.bench_subsystems")


def compute_stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
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
        "p99": round(pct(0.99), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
    }


async def benchmark_graph_memory(db_path: str, trials: int = 25) -> Dict[str, Any]:
    import shutil
    import tempfile
    print("\n--- [1/2] Benchmarking Graph Memory (Kuzu DB Clone) ---")
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, "graph.db")
    print(f"  Cloning production DB ({db_path}) to isolated benchmark sandbox...")
    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            shutil.copytree(db_path, temp_db)
        else:
            shutil.copy2(db_path, temp_db)
    else:
        temp_db = os.path.join(temp_dir, "test.db")
    
    try:
        store = AgentGraphStore(temp_db)
    
        # 1. Warm-up
        try:
            store.create_agent("bench_agent", "Benchmark task", "benchmark")
        except Exception:
            pass
        _ = store.search_memory("monitor", domain="ALL", limit=5)
        _ = store.find_recent_entities(etype="product", since_minutes=60)

        # 2. BM25 Search across domains
        search_domains = ["ALL", "ENTITIES", "ARTIFACTS", "SKILLS", "USER"]
        search_results = {}
        for domain in search_domains:
            samples = []
            for _ in range(trials):
                t0 = time.perf_counter()
                _ = store.search_memory("laptop computer code test review", domain=domain, limit=10)
                samples.append((time.perf_counter() - t0) * 1000.0)
            search_results[domain] = compute_stats(samples)
            print(f"  search_memory ({domain:10s}): p50={search_results[domain]['p50']}ms, p95={search_results[domain]['p95']}ms, mean={search_results[domain]['mean']}ms")

        # 3. Entity staleness query
        entity_samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = store.find_recent_entities(etype="product", since_minutes=60)
            entity_samples.append((time.perf_counter() - t0) * 1000.0)
        entity_stats = compute_stats(entity_samples)
        print(f"  find_recent_entities     : p50={entity_stats['p50']}ms, p95={entity_stats['p95']}ms, mean={entity_stats['mean']}ms")

        # 4. Write latency: save_artifact
        save_art_samples = []
        sample_content = "Benchmark test artifact content payload. " * 50  # ~2KB
        art_ids = []
        for i in range(trials):
            t0 = time.perf_counter()
            art_id = store.save_artifact("bench_agent", f"Benchmark Art {i}", "benchmark_doc", sample_content)
            save_art_samples.append((time.perf_counter() - t0) * 1000.0)
            art_ids.append(art_id)
        save_art_stats = compute_stats(save_art_samples)
        print(f"  save_artifact (~2KB)     : p50={save_art_stats['p50']}ms, p95={save_art_stats['p95']}ms, mean={save_art_stats['mean']}ms")

        # 5. Read latency: get_artifact (chunked reading)
        read_art_samples = []
        for art_id in art_ids:
            t0 = time.perf_counter()
            _ = store.get_artifact(art_id)
            read_art_samples.append((time.perf_counter() - t0) * 1000.0)
        read_art_stats = compute_stats(read_art_samples)
        print(f"  get_artifact (~2KB)      : p50={read_art_stats['p50']}ms, p95={read_art_stats['p95']}ms, mean={read_art_stats['mean']}ms")

        # 6. Write latency: save_user_memory
        user_mem_samples = []
        for i in range(trials):
            t0 = time.perf_counter()
            _ = store.set_user_memory(f"bench_key_{i}", f"value_{i}", category="preference", confidence=0.9, agent_id="bench_agent")
            user_mem_samples.append((time.perf_counter() - t0) * 1000.0)
        user_mem_stats = compute_stats(user_mem_samples)
        print(f"  save_user_memory         : p50={user_mem_stats['p50']}ms, p95={user_mem_stats['p95']}ms, mean={user_mem_stats['mean']}ms")

        # 7. Write latency: save_skill
        skill_samples = []
        for i in range(trials):
            t0 = time.perf_counter()
            _ = store.save_skill(f"bench_skill_{i}", "benchmark.com", "Test skill description", "def run(): pass", agent_id="bench_agent")
            skill_samples.append((time.perf_counter() - t0) * 1000.0)
        skill_stats = compute_stats(skill_samples)
        print(f"  save_skill               : p50={skill_stats['p50']}ms, p95={skill_stats['p95']}ms, mean={skill_stats['mean']}ms")

        return {
            "search_memory": search_results,
            "find_recent_entities": entity_stats,
            "save_artifact": save_art_stats,
            "get_artifact": read_art_stats,
            "save_user_memory": user_mem_stats,
            "save_skill": skill_stats,
        }
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


async def benchmark_tool_service(cdp_url: str = "http://127.0.0.1:9222", trials: int = 25) -> Dict[str, Any]:
    print("\n--- [2/2] Benchmarking ComexToolService (Live Electron CDP) ---")
    req = urllib.request.urlopen(f"{cdp_url}/json/list", timeout=3)
    targets = json.loads(req.read().decode())
    browser_targets = [
        t for t in targets
        if t.get("type") == "page"
        and "localhost:5173" not in t.get("url", "")
        and "cursor_overlay" not in t.get("url", "")
    ]
    if not browser_targets:
        raise RuntimeError("No active WebContentsView tab found for ToolService benchmark.")

    target = browser_targets[0]
    target_id = target["id"]
    ws_url = target["webSocketDebuggerUrl"]
    print(f"  Attaching to tab: {target.get('title', '')} ({target_id})")
    from nervous.action_policy import get_target_registry
    get_target_registry().register_tab_target(target_id, target_id, ws_url)

    svc = ComexToolService()
    from services.tool_service import create_tool_execution_context
    ctx = create_tool_execution_context(
        task_id="bench_task",
        run_id="bench_run",
        target_tab=target_id,
        trusted_server_context=True,
    )

    import http.server
    import socketserver
    import threading

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
        fixture_url = "http://127.0.0.1:8899/benchmark_task.html"
        nav_samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            nav_res = await svc.navigate(ctx, fixture_url)
            nav_samples.append((time.perf_counter() - t0) * 1000.0)
        nav_stats = compute_stats(nav_samples)
        print(f"  browser_navigate         : p50={nav_stats['p50']}ms, p95={nav_stats['p95']}ms, mean={nav_stats['mean']}ms (success={nav_res.success})")

        # 2. Inspect latency
        inspect_samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            insp_res = await svc.inspect(ctx)
            inspect_samples.append((time.perf_counter() - t0) * 1000.0)
        inspect_stats = compute_stats(inspect_samples)
        print(f"  browser_inspect          : p50={inspect_stats['p50']}ms, p95={inspect_stats['p95']}ms, mean={inspect_stats['mean']}ms (success={insp_res.success})")

        # 3. Click latency
        click_samples = []
        for i in range(trials):
            btn_idx = (i % 15) + 1
            t0 = time.perf_counter()
            clk_res = await svc.click(ctx, selector=f"#action-btn-{btn_idx}")
            click_samples.append((time.perf_counter() - t0) * 1000.0)
        click_stats = compute_stats(click_samples)
        print(f"  browser_click            : p50={click_stats['p50']}ms, p95={click_stats['p95']}ms, mean={click_stats['mean']}ms (success={clk_res.success})")

        # 4. Type latency
        type_samples = []
        for i in range(trials):
            t0 = time.perf_counter()
            typ_res = await svc.type(ctx, selector="#task-text-input", text=f"code_{i}")
            type_samples.append((time.perf_counter() - t0) * 1000.0)
        type_stats = compute_stats(type_samples)
        print(f"  browser_type             : p50={type_stats['p50']}ms, p95={type_stats['p95']}ms, mean={type_stats['mean']}ms (success={typ_res.success})")

        # 5. Extract latency
        extract_samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            ext_res = await svc.extract(ctx)
            extract_samples.append((time.perf_counter() - t0) * 1000.0)
        extract_stats = compute_stats(extract_samples)
        print(f"  browser_extract          : p50={extract_stats['p50']}ms, p95={extract_stats['p95']}ms, mean={extract_stats['mean']}ms (success={ext_res.success})")
    finally:
        server.shutdown()
        server.server_close()

    return {
        "target_id": target_id,
        "browser_navigate": nav_stats,
        "browser_inspect": inspect_stats,
        "browser_click": click_stats,
        "browser_type": type_stats,
        "browser_extract": extract_stats,
    }


async def main() -> None:
    db_path = os.path.expanduser(os.getenv("GRAPH_DB_PATH", "~/.comex/graph.db"))
    cdp_url = os.getenv("CDP_URL", "http://127.0.0.1:9222")

    print("=" * 70)
    print("  COMEX SUBSYSTEM PERFORMANCE BENCHMARK (PHASE 2.2)")
    print("=" * 70)
    print(f"Graph DB Path: {db_path}")
    print(f"CDP URL: {cdp_url}")

    graph_res = await benchmark_graph_memory(db_path, trials=30)
    tool_res = await benchmark_tool_service(cdp_url, trials=30)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "platform": sys.platform,
            "python_version": sys.version,
            "graph_db_path": db_path,
            "cdp_url": cdp_url,
        },
        "graph_memory": graph_res,
        "tool_service": tool_res,
    }

    out_file = ROOT_DIR / "benchmarks" / "subsystem_benchmark_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved raw subsystem benchmark metrics to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
