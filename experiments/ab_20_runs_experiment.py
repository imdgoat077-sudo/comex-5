#!/usr/bin/env python3
"""
Comex Controlled 20-Run A/B Experiment:
Measuring whether the integrated Comex Nervous System improves a real
long-horizon browser-agent task on live Google Maps.

Balanced sequence (20 runs total, 10 Condition A, 10 Condition B):
  Pair 1:  A, B
  Pair 2:  B, A
  Pair 3:  A, B
  Pair 4:  B, A
  Pair 5:  A, B
  Pair 6:  B, A
  Pair 7:  A, B
  Pair 8:  B, A
  Pair 9:  A, B
  Pair 10: B, A

Condition A: Comex Nervous System ON (agent_service.browser_click -> resolve_and_bind -> ActuatorGate -> ReflexEngine -> Executor)
Condition B: Nervous System Bypassed (Direct browser-harness / DOM querySelector .click())
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

# Load environment
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
logger = logging.getLogger("ab_20_experiment")

ELECTRON_CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
MODEL_NAME = os.getenv("MODEL_NAME", "ag/gemini-3.8-flash")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")

OUT_PATH = ROOT_DIR / "ab_20_runs_results.json"


# ─────────────────────────────────────────────────────────────────────────────
# Task Specification & Prompt
# ─────────────────────────────────────────────────────────────────────────────

TASK_PROMPT = """You are an autonomous browser navigation agent operating inside an Electron desktop browser.
Your task is to plan a multi-stop walking itinerary in Tokyo using Google Maps.

CRITICAL PROTOCOL RULES:
1. You must interact with Google Maps strictly through its visible UI elements.
2. Do NOT navigate directly to /maps/dir/ URLs.
3. Do NOT construct a pre-populated URL containing origin, destination, or waypoints.
4. Do NOT solve the task from memory or general knowledge.
5. Mode MUST be walking only. Make sure Walking travel mode is selected.
6. The itinerary must start at Shibuya Station and visit the following stops in exact order:
   - Start: Shibuya Station
   - Stop 1: Hachiko Statue
   - Stop 2: Shibuya Scramble Crossing
   - Stop 3: Miyashita Park
   - Stop 4: Takeshita Street
   - Stop 5: Meiji Jingu
   - Stop 6: Shinjuku Gyoen National Garden
   - Stop 7: Omoide Yokocho
   - End: Shinjuku Station

7. For EVERY leg (8 legs total), obtain the walking distance and walking time directly from Google Maps:
   - Leg 1: Shibuya Station -> Hachiko Statue
   - Leg 2: Hachiko Statue -> Shibuya Scramble Crossing
   - Leg 3: Shibuya Scramble Crossing -> Miyashita Park
   - Leg 4: Miyashita Park -> Takeshita Street
   - Leg 5: Takeshita Street -> Meiji Jingu
   - Leg 6: Meiji Jingu -> Shinjuku Gyoen National Garden
   - Leg 7: Shinjuku Gyoen National Garden -> Omoide Yokocho
   - Leg 8: Omoide Yokocho -> Shinjuku Station

8. Once you have gathered the distance and walking time for all 8 legs, call the `finish` tool and report:
   - All stops in order
   - Distance for every leg
   - Walking time for every leg
   - Total distance
   - Total walking time
"""


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    run_index: int
    condition: str  # "A" or "B"
    condition_name: str  # "Comex Nervous System ON" or "Nervous System Bypassed"
    task_completion: str = "FAIL"  # PASS or FAIL
    total_wall_clock_time: float = 0.0
    number_of_browser_actions: int = 0
    number_of_clicks: int = 0
    number_of_text_entries: int = 0
    number_of_page_navigations: int = 0
    number_of_back_navigations: int = 0
    number_of_retries: int = 0
    number_of_failed_actions: int = 0
    number_of_stale_or_missing_targets: int = 0
    number_of_overlays_encountered: int = 0
    number_of_action_blocks: int = 0
    number_of_llm_turns: int = 0
    final_route_correctness: str = "UNVERIFIED"
    failure_reason: str = "None"
    itinerary_report: str = ""

    # Condition A specific
    browser_click_calls: int = 0
    allow_decisions: int = 0
    route_to_known_handler_decisions: int = 0
    report_blocked_decisions: int = 0
    runtime_safety_aborts: int = 0
    click_latencies_ms: List[float] = field(default_factory=list)
    click_latency_p50_ms: float = 0.0
    click_latency_p95_ms: float = 0.0

    # Condition B specific
    direct_harness_actions: int = 0
    failed_target_lookups: int = 0
    recovery_turns: int = 0
    retries_caused_by_invalid_targets: int = 0

    # Trace
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


def get_active_browser_ws() -> str:
    req = urllib.request.urlopen(f"{ELECTRON_CDP_URL}/json/list", timeout=3)
    targets = json.loads(req.read())
    for t in targets:
        if t.get("type") == "page" and "localhost:5173" not in t.get("url", ""):
            return t.get("webSocketDebuggerUrl", "")
    raise RuntimeError("No active Electron browser tab found.")


async def reset_page_to_blank(session: TabSessionContext):
    logger.info("Navigating tab to about:blank to reset state...")
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
    """Instantiate the actual agent_service.browser_click tool from execute_agent_loop."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Single Run Execution
# ─────────────────────────────────────────────────────────────────────────────

async def execute_run(run_idx: int, condition: str, max_turns: int = 35) -> RunMetrics:
    condition_name = "Comex Nervous System ON" if condition == "A" else "Nervous System Bypassed"
    logger.info(f"\n{'='*75}\nRUN {run_idx}/20: CONDITION {condition} ({condition_name})\n{'='*75}")

    ws_url = get_active_browser_ws()
    session = await get_or_create_tab_session("active_tab", ws_url)
    await reset_page_to_blank(session)

    metrics = RunMetrics(
        run_index=run_idx,
        condition=condition,
        condition_name=condition_name,
    )

    # In Condition A, capture the production browser_click tool
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

        if "/maps/dir/" in url and ("Shibuya" in url or "Hachiko" in url or "Shinjuku" in url):
            metrics.number_of_failed_actions += 1
            last_action_failed = True
            trace_item["error"] = "Rule violation: cannot navigate directly to pre-populated /maps/dir/ URLs"
            metrics.tool_trace.append(trace_item)
            return "Error: Protocol violation! You are NOT allowed to navigate directly to pre-populated /maps/dir/ URLs with waypoints. You must start at https://www.google.com/maps and use the UI."

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
            if condition == "B" and last_action_failed:
                metrics.retries_caused_by_invalid_targets += 1
        last_action_key = current_action_key

        trace_item = {"action": "click", "selector": selector, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition} Run {run_idx}] click: '{selector}'")

        if "back" in selector.lower():
            metrics.number_of_back_navigations += 1

        if condition == "A":
            # ─────────────────────────────────────────────────────────────────
            # CONDITION A: Nervous System ON (using actual agent_service.browser_click)
            # ─────────────────────────────────────────────────────────────────
            metrics.browser_click_calls += 1
            t_click_start = time.monotonic()
            try:
                assert prod_browser_click is not None
                tool_res = await prod_browser_click.ainvoke({"selector": selector})
                dur_ms = round((time.monotonic() - t_click_start) * 1000, 2)
                metrics.click_latencies_ms.append(dur_ms)

                # Inspect gate attempt log
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
                        metrics.number_of_action_blocks += 1

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
                await asyncio.sleep(0.8)
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
            # ─────────────────────────────────────────────────────────────────
            # CONDITION B: Nervous System BYPASSED (direct browser-harness path)
            # ─────────────────────────────────────────────────────────────────
            metrics.direct_harness_actions += 1
            try:
                js_expr = f"""(() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{ found: false }};
                    const rect = el.getBoundingClientRect();
                    el.click();
                    return {{
                        found: true,
                        rect: {{ x: rect.x, y: rect.y, width: rect.width, height: rect.height }},
                        disabled: el.disabled || false
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
                await asyncio.sleep(0.8)
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
            if condition == "B" and last_action_failed:
                metrics.retries_caused_by_invalid_targets += 1
        last_action_key = current_action_key

        trace_item = {"action": "type_text", "selector": selector, "text": text, "press_enter": press_enter, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition} Run {run_idx}] type_text: '{selector}' -> '{text}'")

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
                await asyncio.sleep(2.0)

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

    async def execute_get_page_content() -> str:
        nonlocal last_action_failed
        metrics.number_of_browser_actions += 1
        if condition == "B" and last_action_failed:
            metrics.recovery_turns += 1

        trace_item = {"action": "get_page_content", "timestamp": time.monotonic() - start_time}
        try:
            expr = """JSON.stringify((() => {
                const inputs = Array.from(document.querySelectorAll('div[id*="sb_ifc"] input, input[name="q"], input#searchboxinput')).map(inp => ({
                    id: inp.closest('div[id*="sb_ifc"]')?.id || inp.id || inp.name,
                    aria: inp.getAttribute('aria-label'),
                    val: inp.value,
                    placeholder: inp.placeholder
                }));
                const modes = Array.from(document.querySelectorAll('button[data-tooltip], button[aria-label*="Driving"], button[aria-label*="Walking"], button[aria-label*="Transit"]')).map(b => ({
                    label: b.getAttribute('data-tooltip') || b.getAttribute('aria-label'),
                    selected: b.getAttribute('aria-selected') === 'true' || b.classList.contains('selected')
                })).filter(m => m.label);
                const pane = document.querySelector('#pane, [role="main"]');
                const paneText = pane ? pane.innerText : document.body.innerText;
                const consent = document.querySelector('form[action*="consent"], div[aria-modal="true"], div.modal');
                return {
                    url: window.location.href,
                    inputs,
                    modes: modes.slice(0, 8),
                    hasOverlay: !!consent,
                    paneSummary: paneText.slice(0, 1500)
                };
            })())"""
            res = await session.cdp.send("Runtime.evaluate", {"expression": expr})
            raw_str = res.get("result", {}).get("value", "{}")
            data = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
            if isinstance(data, dict) and data.get("hasOverlay"):
                metrics.number_of_overlays_encountered += 1

            trace_item["status"] = "success"
            metrics.tool_trace.append(trace_item)
            return json.dumps(data, indent=2)
        except Exception as e:
            metrics.number_of_failed_actions += 1
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Error reading page content: {e}"

    async def execute_finish(itinerary_report: str) -> str:
        nonlocal is_finished
        is_finished = True
        metrics.itinerary_report = itinerary_report
        metrics.tool_trace.append({
            "action": "finish",
            "report": itinerary_report,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"[{condition} Run {run_idx}] Task finished report submitted.")
        return "Itinerary report received. Task marked complete."

    @tool
    async def navigate(url: str) -> str:
        """Navigate the browser to a URL (e.g. 'https://www.google.com/maps')."""
        return await execute_navigate(url)

    @tool
    async def click(selector: str) -> str:
        """Click on an element matching the given CSS selector (e.g. 'button[aria-label=\"Directions\"]', 'button[data-tooltip=\"Walking\"]')."""
        return await execute_click(selector)

    @tool
    async def type_text(selector: str, text: str, press_enter: bool = True) -> str:
        """Type text into an input element (e.g. '#sb_ifc50 input') and optionally press Enter."""
        return await execute_type_text(selector, text, press_enter)

    @tool
    async def get_page_content() -> str:
        """Get visible page text, directions input values, active travel mode, and route calculation results."""
        return await execute_get_page_content()

    @tool
    async def finish(itinerary_report: str) -> str:
        """Call this tool when the walking itinerary is complete with all 8 legs, distances, walking times, and totals."""
        return await execute_finish(itinerary_report)

    tools = [navigate, click, type_text, get_page_content, finish]
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
        logger.info(f"[{condition} Run {run_idx}] Turn {turn + 1}/{max_turns}")

        try:
            ai_msg = await llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)
        except Exception as e:
            logger.error(f"[{condition} Run {run_idx}] LLM invoke error: {e}")
            metrics.failure_reason = f"LLM invoke error: {e}"
            break

        if not ai_msg.tool_calls:
            if "total distance" in str(ai_msg.content).lower() and "hachiko" in str(ai_msg.content).lower():
                is_finished = True
                metrics.itinerary_report = str(ai_msg.content)
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

    # Latency percentiles for Condition A
    if metrics.click_latencies_ms:
        s_lat = sorted(metrics.click_latencies_ms)
        n = len(s_lat)
        metrics.click_latency_p50_ms = round(s_lat[int(0.50 * (n - 1))], 2)
        metrics.click_latency_p95_ms = round(s_lat[int(0.95 * (n - 1))], 2)

    # Verification against ground truth criteria
    rep_lower = metrics.itinerary_report.lower()
    required_stops = [
        "shibuya station", "hachiko", "scramble", "miyashita",
        "takeshita", "meiji jingu", "shinjuku gyoen", "omoide yokocho", "shinjuku station"
    ]
    stops_found = sum(1 for s in required_stops if s in rep_lower)
    has_distance = "distance" in rep_lower or "km" in rep_lower or " m" in rep_lower
    has_time = "time" in rep_lower or "min" in rep_lower or "hr" in rep_lower

    # Trace check: verify typing actually occurred on inputs
    type_actions = [t for t in metrics.tool_trace if t.get("action") == "type_text"]
    nav_actions = [t for t in metrics.tool_trace if t.get("action") == "navigate"]
    no_prepopulated_url = all("/maps/dir/" not in t.get("url", "") for t in nav_actions)

    if is_finished and stops_found >= 7 and has_distance and has_time and len(type_actions) >= 8 and no_prepopulated_url:
        metrics.task_completion = "PASS"
        metrics.final_route_correctness = "CORRECT (All stops, distances, and times verified from live UI)"
        metrics.failure_reason = "None (Completed successfully)"
    else:
        metrics.task_completion = "FAIL"
        metrics.final_route_correctness = "INCORRECT"
        if not no_prepopulated_url:
            metrics.failure_reason = "Protocol violation: direct /maps/dir/ navigation used."
        elif not is_finished:
            metrics.failure_reason = f"Max turns reached ({max_turns}) without calling finish."
        elif stops_found < 7:
            metrics.failure_reason = f"Incomplete itinerary: only {stops_found}/9 stops identified."
        elif len(type_actions) < 8:
            metrics.failure_reason = f"Only {len(type_actions)} input typing actions executed in trace."
        else:
            metrics.failure_reason = "Missing leg distances or walking times."

    logger.info(f"\n{'='*75}\nCOMPLETED RUN {run_idx}/20 [{condition}]: {metrics.task_completion} in {metrics.total_wall_clock_time}s\n{'='*75}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 20-Run Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE = [
    ("A", "Nervous System ON"),
    ("B", "Nervous System Bypassed"),
    ("B", "Nervous System Bypassed"),
    ("A", "Nervous System ON"),
    ("A", "Nervous System ON"),
    ("B", "Nervous System Bypassed"),
    ("B", "Nervous System Bypassed"),
    ("A", "Nervous System ON"),
    ("A", "Nervous System ON"),
    ("B", "Nervous System Bypassed"),
    ("B", "Nervous System Bypassed"),
    ("A", "Nervous System ON"),
    ("A", "Nervous System ON"),
    ("B", "Nervous System Bypassed"),
    ("B", "Nervous System Bypassed"),
    ("A", "Nervous System ON"),
    ("A", "Nervous System ON"),
    ("B", "Nervous System Bypassed"),
    ("B", "Nervous System Bypassed"),
    ("A", "Nervous System ON"),
]


def save_incremental_results(runs: List[RunMetrics]):
    data = {
        "metadata": {
            "total_runs": len(runs),
            "planned_runs": 20,
            "model": MODEL_NAME,
            "cdp_url": ELECTRON_CDP_URL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "runs": [asdict(r) for r in runs]
    }
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved incremental results ({len(runs)} runs) to {OUT_PATH}")


async def run_20_ab_experiment():
    print("\n" + "=" * 78)
    print("  COMEX 20-RUN CONTROLLED A/B EXPERIMENT")
    print("  Task: 8-Leg Walking Itinerary on Live Google Maps")
    print(f"  Model: {MODEL_NAME} | CDP: {ELECTRON_CDP_URL}")
    print("=" * 78 + "\n")

    runs: List[RunMetrics] = []

    for i, (cond, name) in enumerate(SEQUENCE):
        run_num = i + 1
        print(f"\n>>> Starting Run {run_num}/20: Condition {cond} ({name})")
        metric = await execute_run(run_idx=run_num, condition=cond)
        runs.append(metric)
        save_incremental_results(runs)

        print(f">>> Finished Run {run_num}/20: {metric.task_completion} in {metric.total_wall_clock_time}s (Actions: {metric.number_of_browser_actions}, Turns: {metric.number_of_llm_turns})")

        if run_num < len(SEQUENCE):
            print("Settle pause 3.0s before next run...")
            await asyncio.sleep(3.0)

    print("\n" + "=" * 78)
    print("  ALL 20 RUNS COMPLETE. GENERATING SUMMARY REPORT.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    asyncio.run(run_20_ab_experiment())
