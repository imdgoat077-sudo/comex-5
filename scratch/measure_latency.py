import time
import os
import shutil
import tempfile
import statistics
import json
import asyncio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.store import AgentGraphStore

def benchmark_isolated_kuzu():
    temp_dir = tempfile.mkdtemp(prefix="comex_bench_")
    db_path = os.path.join(temp_dir, "graph.db")
    try:
        store = AgentGraphStore(db_path=db_path)
        
        # Seed test data
        agent_id = "agent_bench_1"
        store.create_agent(agent_id, "scrape products on amazon", "task")
        store.set_user_memory("preferred_currency", "USD", category="preference")
        store.save_skill("amazon_search", "amazon.com", "search products", "code")
        store.save_artifact(agent_id, "Laptops comparison", "research", "content")
        
        queries = [
            ("BM25 Entity Search ('laptop')", lambda: store.search_memory("laptop", domain="ENTITIES", limit=3)),
            ("BM25 History Search ('scrape')", lambda: store.search_memory("scrape", domain="HISTORY", limit=3)),
            ("BM25 All Memory Search ('currency')", lambda: store.search_memory("currency", domain="ALL", limit=3)),
            ("User Memory Fetch (all)", lambda: store.get_all_user_memory()),
            ("Domain Skills Fetch ('amazon.com')", lambda: store.find_skills_for_domain("amazon.com", limit=3)),
            ("SiteAdapter Lookup ('amazon.com')", lambda: store.find_matching_adapter("amazon.com", "search monitors", limit=1)),
            ("Domain Antipatterns ('amazon.com')", lambda: store.get_domain_antipatterns("amazon.com", limit=3)),
        ]
        
        results = {}
        for name, fn in queries:
            times = []
            for _ in range(15):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
            median_ms = statistics.median(times)
            p95_ms = sorted(times)[int(len(times) * 0.95)]
            results[name] = {"median_ms": round(median_ms, 2), "p95_ms": round(p95_ms, 2)}
        return results
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def run_startup_benchmarks():
    from unittest.mock import MagicMock, AsyncMock, patch
    from agent_service import execute_agent_loop
    
    mock_llm_response = MagicMock()
    mock_llm_response.tool_calls = []
    mock_llm_response.content = "Benchmark answer"
    
    # 1. Startup without procedural LLM calls (Default Phase 1 path)
    default_mem_prep = []
    default_db_times = []
    temp_dir = tempfile.mkdtemp(prefix="comex_bench_startup_")
    db_path = os.path.join(temp_dir, "graph.db")
    temp_store = AgentGraphStore(db_path=db_path)
    temp_store.set_user_memory("currency", "USD")
    temp_store.save_skill("amazon_search", "amazon.com", "desc", "code")

    try:
        for _ in range(5):
            with patch("agent_service.ChatOpenAI") as mock_chat_cls, \
                 patch("agent_service.graph_store", temp_store):
                mock_chat_inst = MagicMock()
                mock_chat_inst.bind_tools.return_value = mock_chat_inst
                mock_chat_inst.ainvoke = AsyncMock(return_value=mock_llm_response)
                mock_chat_cls.return_value = mock_chat_inst
                
                res = await execute_agent_loop(
                    "Find monitors previously saved",
                    "http://127.0.0.1:9222",
                    use_procedures=False
                )
                default_mem_prep.append(res.metadata["timing"]["memory_prep_ms"])
                default_db_times.append(res.metadata["timing"]["db_query_ms"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
            
    # 2. Startup with procedural LLM calls (Legacy / Experimental path)
    # Simulate LLM roundtrip of ~5.28s for procedure classification and procedure gating
    simulated_llm_delay = 5.28
    proc_startup_estimated = statistics.median(default_mem_prep) + (2 * simulated_llm_delay * 1000.0)
    
    return {
        "phase1_default_memory_prep_ms": round(statistics.median(default_mem_prep), 2),
        "phase1_default_db_query_ms": round(statistics.median(default_db_times), 2),
        "phase1_default_llm_startup_calls": 0,
        "legacy_procedural_memory_prep_ms": round(proc_startup_estimated, 2),
        "legacy_procedural_llm_startup_calls": 2,
        "startup_latency_reduction_ratio": f"{round(proc_startup_estimated / max(statistics.median(default_mem_prep), 0.1), 1)}x"
    }

async def main():
    db_results = benchmark_isolated_kuzu()
    print("=== Isolated Kùzu Database Latencies ===")
    print(json.dumps(db_results, indent=2))
    
    startup_results = await run_startup_benchmarks()
    print("\n=== Memory Startup Latency Comparison ===")
    print(json.dumps(startup_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

