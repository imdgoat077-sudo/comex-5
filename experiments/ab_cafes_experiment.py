#!/usr/bin/env python3
"""
Comex Controlled A/B Experiment:
Task: Search for "cafes near Shibuya Station, Tokyo" on Google Maps,
open 5 DIFFERENT cafes via visible UI, extract name, rating, reviews, address,
and return via UI Back button.

Condition A: Comex Nervous System ON (production agent_service.browser_click)
Condition B: Comex Nervous System BYPASSED (direct browser-harness path)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Configure temp graph db to prevent locking collisions with background daemon
temp_db_dir = tempfile.mkdtemp()
os.environ["GRAPH_DB_PATH"] = os.path.join(temp_db_dir, "exp_graph.db")

from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR))

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import agent_service
from nervous.cdp_session import TabSessionContext, get_or_create_tab_session
from nervous.reflex_engine import DecisionType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ab_cafes_experiment")

ELECTRON_CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
MODEL_NAME = os.getenv("MODEL_NAME", "ag/gemini-3.8-flash")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")

OUT_PATH = ROOT_DIR / "ab_cafes_results.json"


TASK_PROMPT = """You are an autonomous browser agent navigating Google Maps.
Your task is to find 5 DIFFERENT cafes near Shibuya Station in Tokyo using Google Maps.

PROTOCOL & CONSTRAINTS:
1. Start strictly from https://www.google.com/maps.
2. Do NOT navigate directly to /maps/search/ or /maps/dir/ URLs.
3. Search for "cafes near Shibuya Station, Tokyo" by typing into the visible search box.
4. From the search results feed, find 5 DIFFERENT cafes.
5. For EACH cafe:
   a. Click on the cafe in the visible UI to open its detail page.
   b. Inspect the detail page using get_page_content and record:
      - name
      - current rating
      - number of reviews
      - address
   c. Return to the search results list by clicking the Back button (button[aria-label="Back"]).
   d. Scroll the results feed using scroll_results if needed to discover more cafes.
   e. Choose another cafe you have NOT already opened. Do not open the same cafe twice.
6. Once you have opened, recorded, and returned from 5 different cafes, call the finish tool with your list.
"""


@dataclass
class CafeItem:
    name: str
    rating: str
    reviews: str
    address: str


@dataclass
class ConditionRunMetrics:
    condition: str  # "A" or "B"
    condition_name: str
    task_completion: str = "FAIL"  # PASS / FAIL
    total_wall_clock_time: float = 0.0
    number_of_browser_actions: int = 0
    number_of_clicks: int = 0
    number_of_text_entries: int = 0
    number_of_back_navigations: int = 0
    number_of_scrolls: int = 0
    number_of_page_navigations: int = 0
    number_of_failed_actions: int = 0
    number_of_stale_or_missing_targets: int = 0
    number_of_retries: int = 0
    recovery_turns: int = 0
    number_of_llm_turns: int = 0
    final_data_correctness: str = "UNVERIFIED"
    failure_reason: str = "None"
    recorded_cafes: List[Dict[str, str]] = field(default_factory=list)

    # Condition A specific
    browser_click_calls: int = 0
    allow_decisions: int = 0
    route_to_known_handler_decisions: int = 0
    report_blocked_decisions: int = 0
    runtime_safety_aborts: int = 0
    click_latencies_ms: List[float] = field(default_factory=list)

    # Condition B specific
    direct_harness_actions: int = 0
    failed_target_lookups: int = 0

    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


def get_active_browser_ws() -> str:
    req = urllib.request.urlopen(f"{ELECTRON_CDP_URL}/json/list", timeout=3)
    targets = json.loads(req.read())
    for t in targets:
        if t.get("type") == "page" and "localhost:5173" not in t.get("url", ""):
            return t.get("webSocketDebuggerUrl", "")
    raise RuntimeError("No active Electron browser tab found.")


async def reset_page_to_blank(session: TabSessionContext):
    logger.info("Resetting page to about:blank...")
    await session.cdp.send("Page.navigate", {"url": "about:blank"})
    for _ in range(30):
        res = await session.cdp.send("Runtime.evaluate", {
            "expression": "document.readyState", "returnByValue": True
        })
        if res.get("result", {}).get("value") == "complete":
            break
        await asyncio.sleep(0.05)
    await session.flush_and_sync()
    await asyncio.sleep(1.0)


async def get_production_browser_click_tool(ws_url: str):
    captured = []
    with patch('agent_service.ChatOpenAI') as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        def fake_bind_tools(tools):
            captured.extend(tools)
            return mock_llm
        mock_llm.bind_tools.side_effect = fake_bind_tools
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(tool_calls=[]))
        await agent_service.execute_agent_loop("init", ws_url, use_graph=False)
    for t in captured:
        if t.name == "browser_click":
            return t
    raise RuntimeError("browser_click tool could not be captured from agent_service.py")


async def execute_condition(condition: str, max_turns: int = 40) -> ConditionRunMetrics:
    condition_name = "Comex Nervous System ON" if condition == "A" else "Comex Nervous System BYPASSED"
    logger.info(f"\n{'='*75}\nSTARTING EXPERIMENT: CONDITION {condition} ({condition_name})\n{'='*75}")

    ws_url = get_active_browser_ws()
    session = await get_or_create_tab_session("active_tab", ws_url)
    await reset_page_to_blank(session)

    metrics = ConditionRunMetrics(
        condition=condition,
        condition_name=condition_name,
    )

    prod_browser_click = None
    if condition == "A":
        prod_browser_click = await get_production_browser_click_tool(ws_url)

    start_time = time.monotonic()
    last_action_key: Optional[str] = None
    last_action_failed = False
    is_finished = False

    async def execute_navigate(url: str) -> str:
        nonlocal last_action_failed
        metrics.number_of_browser_actions += 1
        metrics.number_of_page_navigations += 1
        trace_item = {"action": "navigate", "url": url, "timestamp": time.monotonic() - start_time}

        if "/maps/search/" in url or "/maps/dir/" in url:
            metrics.number_of_failed_actions += 1
            last_action_failed = True
            trace_item["error"] = "Rule violation: cannot navigate directly to search or directions URLs"
            metrics.tool_trace.append(trace_item)
            return "Error: Protocol violation! You are NOT allowed to navigate directly to /maps/search/ URLs. You must start at https://www.google.com/maps and use the search box UI."

        try:
            await session.cdp.send("Page.navigate", {"url": url})
            await asyncio.sleep(2.5)
            trace_item["status"] = "success"
            last_action_failed = False
            metrics.tool_trace.append(trace_item)
            return f"Navigated to {url}. Call get_page_content to inspect the page."
        except Exception as e:
            metrics.number_of_failed_actions += 1
            last_action_failed = True
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Navigation error: {e}"

    async def execute_click(selector: str) -> str:
        nonlocal last_action_key, last_action_failed
        metrics.number_of_browser_actions += 1
        metrics.number_of_clicks += 1

        current_action_key = f"click:{selector}"
        if current_action_key == last_action_key:
            metrics.number_of_retries += 1
        last_action_key = current_action_key

        trace_item = {"action": "click", "selector": selector, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] click: '{selector}'")

        if "back" in selector.lower():
            metrics.number_of_back_navigations += 1

        if condition == "A":
            metrics.browser_click_calls += 1
            t_start = time.monotonic()
            try:
                assert prod_browser_click is not None
                tool_res = await prod_browser_click.ainvoke({"selector": selector})
                dur_ms = round((time.monotonic() - t_start) * 1000, 2)
                metrics.click_latencies_ms.append(dur_ms)

                if session.gate.attempt_log:
                    last_gate = session.gate.attempt_log[-1]
                    trace_item["gate_decision"] = last_gate.decision.decision.name
                    trace_item["rule_name"] = last_gate.decision.rule_name
                    trace_item["executed"] = last_gate.executed
                    trace_item["handler_invoked"] = last_gate.handler_invoked

                    if last_gate.decision.decision == DecisionType.ALLOW:
                        metrics.allow_decisions += 1
                    elif last_gate.decision.decision == DecisionType.ROUTE_TO_KNOWN_HANDLER:
                        metrics.route_to_known_handler_decisions += 1
                    else:
                        metrics.report_blocked_decisions += 1

                trace_item["result"] = tool_res
                trace_item["duration_ms"] = dur_ms

                if "Error" in tool_res or "blocked by nervous system" in tool_res:
                    metrics.number_of_failed_actions += 1
                    last_action_failed = True
                    if "not found" in tool_res.lower() or "could not resolve" in tool_res.lower():
                        metrics.number_of_stale_or_missing_targets += 1
                    trace_item["status"] = "failed"
                else:
                    last_action_failed = False
                    trace_item["status"] = "success"

                metrics.tool_trace.append(trace_item)
                await asyncio.sleep(1.0)
                return tool_res
            except Exception as e:
                metrics.runtime_safety_aborts += 1
                metrics.number_of_failed_actions += 1
                last_action_failed = True
                trace_item["status"] = "exception"
                trace_item["error"] = str(e)
                metrics.tool_trace.append(trace_item)
                return f"Error in browser_click: {e}"

        else:
            # Condition B: Direct browser-harness path
            metrics.direct_harness_actions += 1
            try:
                js_expr = f"""(() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{ found: false }};
                    const rect = el.getBoundingClientRect();
                    el.click();
                    return {{
                        found: true,
                        rect: {{ x: rect.x, y: rect.y, width: rect.width, height: rect.height }}
                    }};
                }})()"""
                res = await session.cdp.send("Runtime.evaluate", {"expression": js_expr, "returnByValue": True})
                val = res.get("result", {}).get("value", {})
                if not val.get("found"):
                    metrics.number_of_stale_or_missing_targets += 1
                    metrics.failed_target_lookups += 1
                    metrics.number_of_failed_actions += 1
                    last_action_failed = True
                    trace_item["status"] = "target_missing"
                    trace_item["error"] = f"Selector '{selector}' not found in DOM."
                    metrics.tool_trace.append(trace_item)
                    return f"Error: Selector '{selector}' not found in DOM."

                trace_item["status"] = "dispatched_legacy"
                trace_item["rect"] = val.get("rect")
                last_action_failed = False
                metrics.tool_trace.append(trace_item)
                await asyncio.sleep(1.0)
                return f"Direct click dispatched on '{selector}'."
            except Exception as e:
                metrics.number_of_failed_actions += 1
                last_action_failed = True
                trace_item["status"] = "legacy_exception"
                trace_item["error"] = str(e)
                metrics.tool_trace.append(trace_item)
                return f"Direct click failed on '{selector}': {e}"

    async def execute_type_text(selector: str, text: str, press_enter: bool = True) -> str:
        nonlocal last_action_key, last_action_failed
        metrics.number_of_browser_actions += 1
        metrics.number_of_text_entries += 1
        current_action_key = f"type:{selector}:{text}"
        if current_action_key == last_action_key:
            metrics.number_of_retries += 1
        last_action_key = current_action_key

        trace_item = {"action": "type_text", "selector": selector, "text": text, "press_enter": press_enter, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] type_text: '{selector}' -> '{text}'")

        try:
            js_expr = f"""(() => {{
                const el = document.querySelector({json.dumps(selector)});
                if (!el) return false;
                el.focus();
                el.value = {json.dumps(text)};
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return true;
            }})()"""
            res = await session.cdp.send("Runtime.evaluate", {"expression": js_expr, "returnByValue": True})
            found = res.get("result", {}).get("value", False)
            if not found:
                metrics.number_of_stale_or_missing_targets += 1
                metrics.number_of_failed_actions += 1
                if condition == "B":
                    metrics.failed_target_lookups += 1
                last_action_failed = True
                trace_item["status"] = "missing"
                metrics.tool_trace.append(trace_item)
                return f"Error: Input element '{selector}' not found in DOM."

            if press_enter:
                await asyncio.sleep(0.3)
                await session.cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13
                })
                await session.cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13
                })
                await asyncio.sleep(3.0)

            last_action_failed = False
            trace_item["status"] = "success"
            metrics.tool_trace.append(trace_item)
            return f"Entered '{text}' into '{selector}'."
        except Exception as e:
            metrics.number_of_failed_actions += 1
            last_action_failed = True
            trace_item["status"] = "error"
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Error typing into '{selector}': {e}"

    async def execute_scroll(selector: str = 'div[role="feed"]', delta_y: int = 500) -> str:
        metrics.number_of_browser_actions += 1
        metrics.number_of_scrolls += 1
        trace_item = {"action": "scroll", "selector": selector, "delta_y": delta_y, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] scroll: '{selector}' by {delta_y}px")

        try:
            scroll_expr = f"""(() => {{
                const el = document.querySelector({json.dumps(selector)});
                if (!el) return {{ scrolled: false, error: 'Element not found' }};
                const before = el.scrollTop;
                el.scrollTop += {delta_y};
                return {{ scrolled: true, before, after: el.scrollTop }};
            }})()"""
            res = await session.cdp.send("Runtime.evaluate", {"expression": scroll_expr, "returnByValue": True})
            val = res.get("result", {}).get("value", {})
            trace_item["status"] = "success" if val.get("scrolled") else "failed"
            metrics.tool_trace.append(trace_item)
            await asyncio.sleep(1.5)
            return f"Scrolled '{selector}' from {val.get('before')} to {val.get('after')}."
        except Exception as e:
            metrics.number_of_failed_actions += 1
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Scroll error: {e}"

    async def execute_get_page_content() -> str:
        nonlocal last_action_failed
        metrics.number_of_browser_actions += 1
        if condition == "B" and last_action_failed:
            metrics.recovery_turns += 1

        trace_item = {"action": "get_page_content", "timestamp": time.monotonic() - start_time}
        try:
            expr = """JSON.stringify((() => {
                const url = window.location.href;
                const h1 = document.querySelector('h1');
                const feed = document.querySelector('div[role="feed"]');
                const backBtn = document.querySelector('button[aria-label="Back"], button[aria-label*="Back to results"]');
                
                // If on detail pane
                if (h1 && backBtn && !feed) {
                    const ratingSpan = document.querySelector('div.F7nice span[aria-hidden="true"]');
                    const revSpan = document.querySelector('span[aria-label*="reviews"], span[aria-label*="review"]');
                    const addrBtn = document.querySelector('button[aria-label^="Address:"], button[data-item-id="address"]');
                    return {
                        view: 'detail',
                        url,
                        name: h1.innerText.trim(),
                        rating: ratingSpan ? ratingSpan.innerText.trim() : 'N/A',
                        reviews: revSpan ? (revSpan.getAttribute('aria-label') || revSpan.innerText.trim()) : 'N/A',
                        address: addrBtn ? addrBtn.getAttribute('aria-label') || addrBtn.innerText.trim() : 'N/A',
                        back_button_selector: 'button[aria-label="Back"]'
                    };
                }
                
                // If on feed
                if (feed) {
                    const links = Array.from(feed.querySelectorAll('a.hfpxzc, a[href*="/place/"]')).map((a, idx) => ({
                        index: idx,
                        name: a.getAttribute('aria-label'),
                        selector: `a[aria-label="${a.getAttribute('aria-label')}"]`
                    })).filter(item => item.name);
                    
                    return {
                        view: 'feed',
                        url,
                        visible_cafe_count: links.length,
                        cafes: links
                    };
                }
                
                // Otherwise initial
                const searchInput = document.querySelector('input[name="q"], #searchboxinput, #ucc-1');
                return {
                    view: 'initial',
                    url,
                    search_input_selector: searchInput ? (searchInput.id ? '#' + searchInput.id : 'input[name="q"]') : 'input[name="q"]'
                };
            })())"""
            res = await session.cdp.send("Runtime.evaluate", {"expression": expr})
            raw_str = res.get("result", {}).get("value", "{}")
            data = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
            trace_item["status"] = "success"
            trace_item["view"] = data.get("view")
            metrics.tool_trace.append(trace_item)
            return json.dumps(data, indent=2)
        except Exception as e:
            metrics.number_of_failed_actions += 1
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Error reading page content: {e}"

    async def execute_finish(cafes: List[Dict[str, str]]) -> str:
        nonlocal is_finished
        is_finished = True
        metrics.recorded_cafes = cafes
        metrics.tool_trace.append({
            "action": "finish",
            "cafes": cafes,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"[{condition}] finish tool called with {len(cafes)} cafes.")
        return f"Successfully recorded {len(cafes)} cafes. Task complete."

    @tool
    async def navigate(url: str) -> str:
        """Navigate to a URL (must start at 'https://www.google.com/maps')."""
        return await execute_navigate(url)

    @tool
    async def click(selector: str) -> str:
        """Click on a visible element (e.g. cafe link 'a[aria-label=\"...\"]' or back button 'button[aria-label=\"Back\"]')."""
        return await execute_click(selector)

    @tool
    async def type_text(selector: str, text: str, press_enter: bool = True) -> str:
        """Type search query into the search box and press Enter."""
        return await execute_type_text(selector, text, press_enter)

    @tool
    async def scroll_results(delta_y: int = 500) -> str:
        """Scroll the search results feed downwards to load/reveal more cafes."""
        return await execute_scroll('div[role="feed"]', delta_y)

    @tool
    async def get_page_content() -> str:
        """Get structured information about current page: visible cafes list if on feed, or cafe details (name, rating, reviews, address) if on detail pane."""
        return await execute_get_page_content()

    @tool
    async def finish(cafes: List[Dict[str, str]]) -> str:
        """Call when you have gathered all 5 different cafes with name, rating, reviews, and address."""
        return await execute_finish(cafes)

    tools = [navigate, click, type_text, scroll_results, get_page_content, finish]
    tool_map = {t.name: t for t in tools}

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENAI_BASE_URL,
        api_key=ROUTER_API_KEY,
        temperature=0.0
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content="You are an autonomous browser agent navigating Google Maps."),
        HumanMessage(content=TASK_PROMPT)
    ]

    for turn in range(max_turns):
        if is_finished:
            break
        metrics.number_of_llm_turns += 1
        logger.info(f"[{condition}] Turn {turn + 1}/{max_turns}")

        try:
            ai_msg = await llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)
        except Exception as e:
            logger.error(f"[{condition}] LLM invoke error: {e}")
            metrics.failure_reason = f"LLM error: {e}"
            break

        if not ai_msg.tool_calls:
            # Check if LLM outputted cafes in text
            if len(metrics.recorded_cafes) >= 5:
                is_finished = True
                break
            continue

        for tc in ai_msg.tool_calls:
            tname = tc["name"]
            targs = tc["args"]
            t_id = tc["id"]
            tool_fn = tool_map.get(tname)
            if not tool_fn:
                tool_output = f"Unknown tool '{tname}'"
            else:
                try:
                    tool_output = await tool_fn.ainvoke(targs)
                except Exception as te:
                    tool_output = f"Tool execution error: {te}"

            messages.append(ToolMessage(content=str(tool_output), tool_call_id=t_id))
            if is_finished:
                break

    metrics.total_wall_clock_time = round(time.monotonic() - start_time, 2)

    # Verification against criteria
    cafes = metrics.recorded_cafes
    unique_names = set(c.get("name", "").strip().lower() for c in cafes if c.get("name"))
    has_5_unique = len(unique_names) >= 5
    has_ratings = all(c.get("rating") and c.get("rating") != "N/A" for c in cafes)
    has_reviews = all(c.get("reviews") and c.get("reviews") != "N/A" for c in cafes)
    has_addresses = all(c.get("address") and c.get("address") != "N/A" for c in cafes)

    # Trace audit
    click_actions = [t for t in metrics.tool_trace if t.get("action") == "click"]
    back_actions = [t for t in click_actions if "back" in t.get("selector", "").lower()]
    nav_actions = [t for t in metrics.tool_trace if t.get("action") == "navigate"]
    no_direct_search_url = all("/maps/search/" not in t.get("url", "") for t in nav_actions)

    if is_finished and has_5_unique and has_ratings and has_reviews and has_addresses and len(back_actions) >= 4 and no_direct_search_url:
        metrics.task_completion = "PASS"
        metrics.final_data_correctness = "CORRECT (5 unique cafes with name, rating, reviews, and address verified via UI navigation and back returns)"
        metrics.failure_reason = "None (Completed successfully)"
    else:
        metrics.task_completion = "FAIL"
        metrics.final_data_correctness = "INCORRECT"
        if not no_direct_search_url:
            metrics.failure_reason = "Protocol violation: direct /maps/search/ URL navigation used."
        elif not is_finished:
            metrics.failure_reason = f"Max turns reached ({max_turns}) without finishing."
        elif not has_5_unique:
            metrics.failure_reason = f"Only {len(unique_names)}/5 unique cafes recorded."
        elif len(back_actions) < 4:
            metrics.failure_reason = f"Only {len(back_actions)} UI back navigations executed in trace."
        else:
            metrics.failure_reason = "Missing rating, reviews, or address in one or more cafes."

    logger.info(f"\n{'='*75}\nCOMPLETED CONDITION {condition}: {metrics.task_completion} in {metrics.total_wall_clock_time}s\n{'='*75}")
    return metrics


async def run_ab_experiment():
    print("\n" + "=" * 78)
    print("  COMEX CONTROLLED A/B EXPERIMENT: CAFE SEARCH & DRILLDOWN")
    print("  Task: Search 'cafes near Shibuya Station, Tokyo', Open 5 Different Cafes,")
    print("        Record details, Return via UI Back button")
    print("=" * 78 + "\n")

    results = []

    # Run Condition A (Nervous System ON)
    print("\n>>> RUNNING CONDITION A (Comex Nervous System ON)...")
    res_a = await execute_condition("A")
    results.append(res_a)

    print("\nSettle pause 5.0s between conditions...")
    await asyncio.sleep(5.0)

    # Run Condition B (Nervous System Bypassed)
    print("\n>>> RUNNING CONDITION B (Comex Nervous System BYPASSED)...")
    res_b = await execute_condition("B")
    results.append(res_b)

    # Save results
    output_data = {
        "metadata": {
            "task": "5 Cafes Search & Drilldown near Shibuya Station",
            "model": MODEL_NAME,
            "cdp_url": ELECTRON_CDP_URL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "runs": [asdict(r) for r in results]
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}\n")


if __name__ == "__main__":
    asyncio.run(run_ab_experiment())
