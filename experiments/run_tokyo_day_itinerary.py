#!/usr/bin/env python3
"""
Comex Controlled High-Difficulty Real-World Task:
Build a Constrained Tokyo Day Itinerary on Live Google Maps

Start: Shibuya Station
End: Shinjuku Station
Places: 3 Attractions (rating >= 4.3), 2 Cafes (rating >= 4.2), 1 Restaurant (rating >= 4.2)
Constraints:
- Walking only
- Under 7 km total
- Avoid geographic backtracking (Shibuya -> Harajuku/Meiji -> Shinjuku)
- Reject at least TWO candidates with reasons
- At least ONE search must require scrolling the results feed
- Every candidate inspected via detail page, then returned via UI Back button
- Final route constructed in Google Maps UI and every leg inspected
- Uses production Comex Nervous System (agent_service.browser_click -> resolve_and_bind -> ActuatorGate -> ReflexEngine -> physical CDP click)
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
logger = logging.getLogger("tokyo_day_itinerary")

ELECTRON_CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
MODEL_NAME = os.getenv("MODEL_NAME", "ag/gemini-3.8-flash")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")

OUT_PATH = ROOT_DIR / "tokyo_day_itinerary_results.json"


SYSTEM_PROMPT = """You are an expert autonomous browser agent operating inside an Electron browser on live Google Maps.
You must construct a high-difficulty, real-world Tokyo walking itinerary that strictly satisfies all user constraints.

CRITICAL RULES:
1. You must interact with Google Maps strictly through its visible UI.
2. Do NOT navigate directly to /maps/search/ or /maps/dir/ URLs.
3. Every search must be typed into the search box UI.
4. Every candidate place must be clicked and opened in the UI, inspected on its detail page, and then returned from via the UI Back button (button[aria-label="Back"]).
5. You must reject at least TWO candidate places before finalizing the 6 stops (log them using reject_candidate with the exact reason, e.g. rating below threshold, closed, or geographic backtracking).
6. At least ONE search must require calling scroll_results on the feed to discover candidates.
7. Accept exactly:
   - 3 Attractions (rating >= 4.3)
   - 2 Cafes (rating >= 4.2)
   - 1 Restaurant (rating >= 4.2)
8. After selecting the 6 places, order them geographically from Shibuya Station to Shinjuku Station to keep total walking under 7 km without backtracking.
9. Then open Directions in Google Maps, select Walking mode, construct the route between the stops, and inspect the walking distance and time for every leg.
10. Call finish with the complete structured itinerary and execution details.
"""


@dataclass
class ItineraryMetrics:
    task_completion: str = "FAIL"
    total_wall_clock_time: float = 0.0
    number_of_browser_actions: int = 0
    number_of_clicks: int = 0
    number_of_text_entries: int = 0
    number_of_back_navigations: int = 0
    number_of_scrolls: int = 0
    number_of_page_navigations: int = 0
    number_of_retries: int = 0
    number_of_failed_actions: int = 0
    number_of_stale_or_missing_targets: int = 0
    number_of_llm_turns: int = 0
    final_data_correctness: str = "UNVERIFIED"
    failure_reason: str = "None"

    # Nervous System specifics
    browser_click_calls: int = 0
    allow_decisions: int = 0
    route_to_known_handler_decisions: int = 0
    report_blocked_decisions: int = 0
    runtime_safety_aborts: int = 0
    click_latencies_ms: List[float] = field(default_factory=list)

    # Domain details
    selected_places: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    legs: List[Dict[str, Any]] = field(default_factory=list)
    total_walking_distance: str = ""
    total_walking_time: str = ""
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


async def run_tokyo_day_itinerary():
    print("\n" + "=" * 78)
    print("  COMEX HIGH-DIFFICULTY REAL-WORLD BROWSER-AGENT TASK")
    print("  Task: Constrained Tokyo Day Itinerary on Live Google Maps")
    print("  Start: Shibuya Station -> 3 Attractions, 2 Cafes, 1 Restaurant -> Shinjuku Station")
    print("  Constraints: Walking only (<7km), reject >=2 candidates, scroll >=1 feed, inspect all")
    print("=" * 78 + "\n")

    ws_url = get_active_browser_ws()
    session = await get_or_create_tab_session("active_tab", ws_url)
    await reset_page_to_blank(session)

    prod_browser_click = await get_production_browser_click_tool(ws_url)

    metrics = ItineraryMetrics()
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
            return "Error: Protocol violation! You are NOT allowed to navigate directly to /maps/search/ or /maps/dir/ URLs. Start at https://www.google.com/maps and use the visible UI."

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
        metrics.browser_click_calls += 1

        current_action_key = f"click:{selector}"
        if current_action_key == last_action_key:
            metrics.number_of_retries += 1
        last_action_key = current_action_key

        trace_item = {"action": "click", "selector": selector, "timestamp": time.monotonic() - start_time}
        logger.info(f"click: '{selector}'")

        if "back" in selector.lower():
            metrics.number_of_back_navigations += 1

        t_start = time.monotonic()
        try:
            tool_res = await prod_browser_click.ainvoke({"selector": selector})
            dur_ms = round((time.monotonic() - t_start) * 1000, 2)
            metrics.click_latencies_ms.append(dur_ms)

            if session.gate.attempt_log:
                last_gate = session.gate.attempt_log[-1]
                trace_item["gate_decision"] = last_gate.decision.decision.name
                trace_item["rule_name"] = last_gate.decision.rule_name
                trace_item["executed"] = last_gate.executed

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

    async def execute_type_text(selector: str, text: str, press_enter: bool = True) -> str:
        nonlocal last_action_key, last_action_failed
        metrics.number_of_browser_actions += 1
        metrics.number_of_text_entries += 1
        current_action_key = f"type:{selector}:{text}"
        if current_action_key == last_action_key:
            metrics.number_of_retries += 1
        last_action_key = current_action_key

        trace_item = {"action": "type_text", "selector": selector, "text": text, "press_enter": press_enter, "timestamp": time.monotonic() - start_time}
        logger.info(f"type_text: '{selector}' -> '{text}'")

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

    async def execute_scroll(selector: str = 'div[role="feed"]', delta_y: int = 600) -> str:
        metrics.number_of_browser_actions += 1
        metrics.number_of_scrolls += 1
        trace_item = {"action": "scroll", "selector": selector, "delta_y": delta_y, "timestamp": time.monotonic() - start_time}
        logger.info(f"scroll: '{selector}' by {delta_y}px")

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
        metrics.number_of_browser_actions += 1
        trace_item = {"action": "get_page_content", "timestamp": time.monotonic() - start_time}
        try:
            expr = """JSON.stringify((() => {
                const url = window.location.href;
                const h1 = document.querySelector('h1');
                const feed = document.querySelector('div[role="feed"]');
                const backBtn = document.querySelector('button[aria-label="Back"], button[aria-label*="Back to results"]');
                const dirInputs = Array.from(document.querySelectorAll('div[id*="sb_ifc"] input'));
                
                // If on directions pane
                if (dirInputs.length >= 2) {
                    const modes = Array.from(document.querySelectorAll('button[data-tooltip], button[aria-label*="Walking"], button[aria-label*="Driving"]')).map(b => ({
                        label: b.getAttribute('data-tooltip') || b.getAttribute('aria-label'),
                        selected: b.getAttribute('aria-selected') === 'true' || b.classList.contains('selected')
                    }));
                    const pane = document.querySelector('#pane, [role="main"]');
                    const paneText = pane ? pane.innerText : document.body.innerText;
                    return {
                        view: 'directions',
                        url,
                        inputs: dirInputs.map(inp => ({ id: inp.closest('div[id*="sb_ifc"]')?.id, val: inp.value, aria: inp.getAttribute('aria-label') })),
                        modes: modes.slice(0, 5),
                        summary: paneText.slice(0, 1500)
                    };
                }
                
                // If on detail pane
                if (h1 && backBtn && !feed) {
                    const ratingSpan = document.querySelector('div.F7nice span[aria-hidden="true"]');
                    const revSpan = document.querySelector('span[aria-label*="reviews"], span[aria-label*="review"]');
                    const addrBtn = document.querySelector('button[aria-label^="Address:"], button[data-item-id="address"]');
                    const hoursEl = document.querySelector('[data-item-id*="oh"], [aria-label*="Open"], [aria-label*="Closed"]');
                    return {
                        view: 'detail',
                        url,
                        name: h1.innerText.trim(),
                        rating: ratingSpan ? ratingSpan.innerText.trim() : 'N/A',
                        reviews: revSpan ? (revSpan.getAttribute('aria-label') || revSpan.innerText.trim()) : 'N/A',
                        address: addrBtn ? addrBtn.getAttribute('aria-label') || addrBtn.innerText.trim() : 'N/A',
                        hours_status: hoursEl ? (hoursEl.getAttribute('aria-label') || hoursEl.innerText.slice(0, 50).replace(/\\n/g, ' ')) : 'Unknown',
                        back_button_selector: 'button[aria-label="Back"]'
                    };
                }
                
                // If on search feed
                if (feed) {
                    const links = Array.from(feed.querySelectorAll('a.hfpxzc, a[href*="/place/"]')).map((a, idx) => ({
                        index: idx,
                        name: a.getAttribute('aria-label'),
                        selector: `a[aria-label="${a.getAttribute('aria-label')}"]`
                    })).filter(item => item.name);
                    
                    return {
                        view: 'feed',
                        url,
                        visible_place_count: links.length,
                        places: links.slice(0, 12)
                    };
                }
                
                // Otherwise initial/base map
                const searchInput = document.querySelector('input[name="q"], #searchboxinput, #ucc-1');
                const dirBtn = document.querySelector('button[aria-label*="Directions"], button#hArJGc');
                return {
                    view: 'base_map',
                    url,
                    search_input_selector: searchInput ? (searchInput.id ? '#' + searchInput.id : 'input[name="q"]') : 'input[name="q"]',
                    directions_button_selector: dirBtn ? 'button[aria-label="Directions"]' : null
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

    @tool
    async def navigate(url: str) -> str:
        """Navigate to a URL (must start at 'https://www.google.com/maps')."""
        return await execute_navigate(url)

    @tool
    async def click(selector: str) -> str:
        """Click on a visible element via Comex Nervous System (e.g. place link 'a[aria-label=\"...\"]', 'button[aria-label=\"Back\"]', 'button[aria-label=\"Directions\"]')."""
        return await execute_click(selector)

    @tool
    async def type_text(selector: str, text: str, press_enter: bool = True) -> str:
        """Type text into an input box and optionally press Enter."""
        return await execute_type_text(selector, text, press_enter)

    @tool
    async def scroll_results(delta_y: int = 600) -> str:
        """Scroll the results feed downwards to load and reveal more places."""
        return await execute_scroll('div[role="feed"]', delta_y)

    @tool
    async def get_page_content() -> str:
        """Get structured information about current page: base map, search feed, place detail pane, or directions pane."""
        return await execute_get_page_content()

    @tool
    async def reject_candidate(name: str, category: str, rating: str, reason: str) -> str:
        """Log an unsuitable candidate place that was inspected on its detail page and rejected. (At least 2 required)."""
        metrics.rejected_candidates.append({
            "name": name,
            "category": category,
            "rating": rating,
            "reason": reason,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"REJECTED CANDIDATE: '{name}' ({category}, rating {rating}): {reason}")
        return f"Logged rejection of candidate '{name}'. Total rejections logged: {len(metrics.rejected_candidates)}. Now click the Back button and continue."

    @tool
    async def record_selected_place(category: str, name: str, rating: float, reviews: str, address: str) -> str:
        """Record an inspected candidate place that satisfies all criteria (Attraction rating >= 4.3, Cafe rating >= 4.2, Restaurant rating >= 4.2)."""
        metrics.selected_places.append({
            "category": category,
            "name": name,
            "rating": rating,
            "reviews": reviews,
            "address": address,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"ACCEPTED PLACE [{category}]: '{name}' (rating {rating}, reviews: {reviews})")
        return f"Recorded '{name}' as accepted {category}. Total selected: {len(metrics.selected_places)}/6. Now click Back to return to search."

    @tool
    async def record_leg(from_stop: str, to_stop: str, distance: str, walking_time: str) -> str:
        """Record distance and walking time for a route leg directly obtained from Google Maps Directions."""
        metrics.legs.append({
            "from": from_stop,
            "to": to_stop,
            "distance": distance,
            "walking_time": walking_time
        })
        logger.info(f"LEG RECORDED: {from_stop} -> {to_stop} ({distance}, {walking_time})")
        return f"Recorded leg {len(metrics.legs)}: {from_stop} -> {to_stop} ({distance}, {walking_time})."

    @tool
    async def finish(total_distance: str, total_time: str, itinerary_summary: str) -> str:
        """Call when all 6 places have been selected, at least 2 candidates rejected, and the full walking route has been constructed and inspected."""
        nonlocal is_finished
        is_finished = True
        metrics.total_walking_distance = total_distance
        metrics.total_walking_time = total_time
        metrics.tool_trace.append({
            "action": "finish",
            "total_distance": total_distance,
            "total_time": total_time,
            "itinerary_summary": itinerary_summary,
            "timestamp": time.monotonic() - start_time
        })
        logger.info("TASK FINISH CALLED.")
        return "Itinerary finalized and submitted. Task complete."

    tools = [
        navigate, click, type_text, scroll_results, get_page_content,
        reject_candidate, record_selected_place, record_leg, finish
    ]
    tool_map = {t.name: t for t in tools}

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENAI_BASE_URL,
        api_key=ROUTER_API_KEY,
        temperature=0.0
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Begin the task now. Start at https://www.google.com/maps and construct the constrained Tokyo Day Itinerary.")
    ]

    max_turns = 100
    for turn in range(max_turns):
        if is_finished:
            break
        metrics.number_of_llm_turns += 1
        logger.info(f"Turn {turn + 1}/{max_turns}")

        try:
            ai_msg = await llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)
        except Exception as e:
            logger.error(f"LLM invoke error: {e}")
            metrics.failure_reason = f"LLM error: {e}"
            break

        if not ai_msg.tool_calls:
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
    has_3_attractions = sum(1 for p in metrics.selected_places if p.get("category", "").lower() == "attraction") >= 3
    has_2_cafes = sum(1 for p in metrics.selected_places if p.get("category", "").lower() == "cafe") >= 2
    has_1_restaurant = sum(1 for p in metrics.selected_places if p.get("category", "").lower() in ["restaurant", "food"]) >= 1
    has_6_places = len(metrics.selected_places) >= 6
    has_2_rejections = len(metrics.rejected_candidates) >= 2
    has_scroll = metrics.number_of_scrolls >= 1
    has_legs = len(metrics.legs) >= 6

    click_actions = [t for t in metrics.tool_trace if t.get("action") == "click"]
    back_actions = [t for t in click_actions if "back" in t.get("selector", "").lower()]

    if is_finished and has_6_places and has_3_attractions and has_2_cafes and has_1_restaurant and has_2_rejections and has_scroll and has_legs and len(back_actions) >= 5:
        metrics.task_completion = "PASS"
        metrics.final_data_correctness = "CORRECT (All 6 places, >=2 rejections, feed scroll, UI back returns, and walking route verified)"
        metrics.failure_reason = "None (Completed successfully)"
    else:
        metrics.task_completion = "FAIL"
        metrics.final_data_correctness = "INCOMPLETE / UNVERIFIED"
        reasons = []
        if not is_finished:
            reasons.append("Max turns reached without finish")
        if not has_6_places:
            reasons.append(f"Only {len(metrics.selected_places)}/6 places selected")
        if not has_2_rejections:
            reasons.append(f"Only {len(metrics.rejected_candidates)}/2 rejections logged")
        if not has_scroll:
            reasons.append("Feed scroll requirement not met")
        if not has_legs:
            reasons.append(f"Only {len(metrics.legs)} route legs inspected")
        metrics.failure_reason = "; ".join(reasons)

    # Save output
    output_data = {
        "metadata": {
            "task": "Tokyo Day Itinerary",
            "model": MODEL_NAME,
            "cdp_url": ELECTRON_CDP_URL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "metrics": asdict(metrics)
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}\n")
    return metrics


if __name__ == "__main__":
    asyncio.run(run_tokyo_day_itinerary())
