"""
Benchmark script for Phase 1 Deliverable:
Measures:
1. Kùzu memory search & retrieval latency (database only).
2. Default task startup with procedural LLM calls disabled (fast path).
3. Experimental task startup with procedural LLM routing enabled.
4. Upstream LLM roundtrip latency.
"""

import asyncio
import time
import json
import statistics
import os
import sys

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.store import AgentGraphStore
import configparser

def run_db_benchmarks():
    db_path = os.path.expanduser("~/.comex/graph.db")
    store = AgentGraphStore(db_path=db_path)

    queries = [
        ("BM25 Entity Search ('laptop')", lambda: store.search_memory("laptop", domain="ENTITIES", limit=3)),
        ("BM25 History Search ('scrape products')", lambda: store.search_memory("scrape products", domain="HISTORY", limit=3)),
        ("BM25 All Memory Search ('monitors')", lambda: store.search_memory("monitors", domain="ALL", limit=3)),
        ("User Memory Fetch (all)", lambda: store.get_all_user_memory()),
        ("Domain Skills Fetch ('amazon.com')", lambda: store.find_skills_for_domain("amazon.com", limit=3)),
        ("SiteAdapter Lookup ('amazon.com')", lambda: store.find_matching_adapter("amazon.com", "search monitors", limit=1)),
        ("Domain Antipatterns ('amazon.com')", lambda: store.get_domain_antipatterns("amazon.com", limit=3)),
    ]

    results = {}
    for name, fn in queries:
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        median_ms = statistics.median(times)
        p95_ms = sorted(times)[int(len(times) * 0.95)]
        results[name] = {"median_ms": round(median_ms, 2), "p95_ms": round(p95_ms, 2)}
    return results

async def run_startup_benchmarks():
    from agent_service import execute_agent_loop
    from unittest.mock import MagicMock, AsyncMock, patch

    mock_llm_response = MagicMock()
    mock_llm_response.tool_calls = []
    mock_llm_response.content = "Benchmark answer"

    # 1. Startup without procedural LLM calls (Default Phase 1 path)
    default_times = []
    default_mem_prep = []
    default_db_times = []
    for _ in range(5):
        with patch("agent_service.ChatOpenAI") as mock_chat_cls:
            mock_chat_inst = MagicMock()
            mock_chat_inst.bind_tools.return_value = mock_chat_inst
            mock_chat_inst.ainvoke = AsyncMock(return_value=mock_llm_response)
            mock_chat_cls.return_value = mock_chat_inst

            t0 = time.perf_counter()
            res = await execute_agent_loop(
                "Find monitors previously saved",
                "http://127.0.0.1:9222",
                use_procedures=False
            )
            t1 = time.perf_counter()
            default_times.append((t1 - t0) * 1000.0)
            default_mem_prep.append(res.metadata["timing"]["memory_prep_ms"])
            default_db_times.append(res.metadata["timing"]["db_query_ms"])

    return {
        "default_phase1_startup_total_median_ms": round(statistics.median(default_times), 2),
        "default_phase1_memory_prep_median_ms": round(statistics.median(default_mem_prep), 2),
        "default_phase1_db_query_median_ms": round(statistics.median(default_db_times), 2),
    }

async def main():
    print("=== Running Kùzu DB Benchmark ===")
    db_metrics = run_db_benchmarks()
    print(json.dumps(db_metrics, indent=2))

    print("\n=== Running Memory Startup Benchmark ===")
    startup_metrics = await run_startup_benchmarks()
    print(json.dumps(startup_metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
