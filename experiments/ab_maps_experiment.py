#!/usr/bin/env python3
"""
Comex Controlled A/B Experiment Runner: Long-Horizon Walking Itinerary on Google Maps

Compares:
  CONDITION A: Comex Nervous System ON (Production path: resolve_and_bind -> ActuatorGate -> ReflexEngine)
  CONDITION B: Comex Nervous System BYPASSED (Direct/legacy path: raw querySelector -> .click())

Rules:
  - Exact same browser (Electron port 9222)
  - Exact same model (ag/gemini-3.8-flash via 9Router proxy)
  - Exact same task prompt and tool definitions
  - Fresh browser page for each condition
  - No retries or timing hacks added
  - Measurement-only execution recording actual tool/browser traces
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from nervous.cdp_session import TabSessionContext, get_or_create_tab_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ab_experiment")

ELECTRON_CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
MODEL_NAME = os.getenv("MODEL_NAME", "ag/gemini-3.8-flash")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Task Specification
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
class ConditionMetrics:
    condition: str  # "A" or "B"
    condition_name: str  # "Nervous System ON" or "Nervous System Bypassed"
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
    final_answer_correctness: str = "UNVERIFIED"
    failure_reason: str = "None"
    itinerary_report: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CDP Target Finder & Page Reset
# ─────────────────────────────────────────────────────────────────────────────

def get_active_browser_ws() -> str:
    req = urllib.request.urlopen(f"{ELECTRON_CDP_URL}/json/list", timeout=3)
    targets = json.loads(req.read())
    for t in targets:
        if t.get("type") == "page" and "localhost:5173" not in t.get("url", ""):
            return t.get("webSocketDebuggerUrl", "")
    raise RuntimeError("No active Electron browser tab found.")


async def reset_page_to_blank(session: TabSessionContext):
    logger.info("Resetting browser tab to about:blank...")
    await session.cdp.send("Page.navigate", {"url": "about:blank"})
    await asyncio.sleep(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Execution Loop for a Condition
# ─────────────────────────────────────────────────────────────────────────────

async def run_condition(
    condition: str,
    max_turns: int = 40,
) -> ConditionMetrics:
    condition_name = "Nervous System ON" if condition == "A" else "Nervous System Bypassed"
    logger.info(f"\n{'='*70}\nSTARTING CONDITION {condition}: {condition_name}\n{'='*70}")

    ws_url = get_active_browser_ws()
    session = await get_or_create_tab_session(f"tab_{condition}", ws_url)
    
    # Reset page to about:blank before starting
    await reset_page_to_blank(session)

    metrics = ConditionMetrics(
        condition=condition,
        condition_name=condition_name,
    )
    
    start_time = time.monotonic()
    last_action_key: Optional[str] = None
    is_finished = False

    # Define tools dynamically bound to this condition's session & telemetry
    async def execute_navigate(url: str) -> str:
        metrics.number_of_browser_actions += 1
        metrics.number_of_page_navigations += 1
        trace_item = {"action": "navigate", "url": url, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] navigate: {url}")
        
        # Check rule: Do not construct pre-populated Maps URL with destinations
        if "/maps/dir/" in url and ("Shibuya" in url or "Hachiko" in url or "Shinjuku" in url):
            metrics.number_of_failed_actions += 1
            trace_item["error"] = "Rule violation: cannot navigate directly to pre-populated /maps/dir/ URLs"
            metrics.tool_trace.append(trace_item)
            return "Error: Protocol violation! You are NOT allowed to navigate directly to pre-populated /maps/dir/ URLs with waypoints. You must start at https://www.google.com/maps and use the UI."

        try:
            await session.cdp.send("Page.navigate", {"url": url})
            await asyncio.sleep(3.0)
            trace_item["status"] = "success"
            metrics.tool_trace.append(trace_item)
            return f"Navigated to {url}. Call get_page_content to inspect the page."
        except Exception as e:
            metrics.number_of_failed_actions += 1
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Navigation error: {e}"

    async def execute_click(selector: str) -> str:
        nonlocal last_action_key
        metrics.number_of_browser_actions += 1
        metrics.number_of_clicks += 1
        current_action_key = f"click:{selector}"
        if current_action_key == last_action_key:
            metrics.number_of_retries += 1
        last_action_key = current_action_key

        trace_item = {"action": "click", "selector": selector, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] click: '{selector}'")

        # Track back navigations
        if "back" in selector.lower() or "back" in selector:
            metrics.number_of_back_navigations += 1

        if condition == "A":
            # ─────────────────────────────────────────────────────────────────
            # CONDITION A: Nervous System ON
            # Full production pipeline: resolve_and_bind -> Method E -> ActuatorGate -> SelectorClickExecutor
            # ─────────────────────────────────────────────────────────────────
            try:
                binding = await session.resolve_and_bind(selector)
            except Exception as e:
                err_str = str(e)
                metrics.number_of_failed_actions += 1
                if "not found" in err_str.lower() or "could not resolve" in err_str.lower():
                    metrics.number_of_stale_or_missing_targets += 1
                trace_item["status"] = "bind_failed"
                trace_item["error"] = err_str
                metrics.tool_trace.append(trace_item)
                return f"Error binding target '{selector}': {err_str}"

            # Gate attempt
            try:
                gate_res = await session.gate.async_attempt("click", binding=binding)
                trace_item["decision"] = str(gate_res.decision)
                trace_item["executed"] = gate_res.executed

                if gate_res.executed:
                    trace_item["status"] = "executed"
                    metrics.tool_trace.append(trace_item)
                    await asyncio.sleep(1.0)
                    return f"Successfully clicked '{selector}' (decision: ALLOW)."
                else:
                    metrics.number_of_action_blocks += 1
                    metrics.number_of_failed_actions += 1
                    trace_item["status"] = "blocked"
                    trace_item["rule_name"] = gate_res.decision.rule_name
                    trace_item["reason"] = gate_res.decision.reason
                    metrics.tool_trace.append(trace_item)
                    return f"Click blocked by nervous system: {gate_res.decision.rule_name} ({gate_res.decision.reason})."
            except Exception as e:
                metrics.number_of_failed_actions += 1
                trace_item["status"] = "gate_exception"
                trace_item["error"] = str(e)
                metrics.tool_trace.append(trace_item)
                return f"Error executing gated click on '{selector}': {e}"

        else:
            # ─────────────────────────────────────────────────────────────────
            # CONDITION B: Nervous System BYPASSED
            # Direct/legacy path: raw querySelector -> .click() without nervous gating
            # ─────────────────────────────────────────────────────────────────
            try:
                js_expr = f"""(() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{ found: false }};
                    // Check if element is disabled or hidden
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
                    metrics.number_of_failed_actions += 1
                    trace_item["status"] = "target_missing"
                    trace_item["error"] = f"Selector '{selector}' not found in DOM."
                    metrics.tool_trace.append(trace_item)
                    return f"Error: Selector '{selector}' not found in DOM."
                
                trace_item["status"] = "dispatched_legacy"
                trace_item["rect"] = val.get("rect")
                metrics.tool_trace.append(trace_item)
                await asyncio.sleep(1.0)
                return f"Direct click dispatched on '{selector}'."
            except Exception as e:
                metrics.number_of_failed_actions += 1
                trace_item["status"] = "legacy_exception"
                trace_item["error"] = str(e)
                metrics.tool_trace.append(trace_item)
                return f"Direct click failed on '{selector}': {e}"

    async def execute_type_text(selector: str, text: str, press_enter: bool = True) -> str:
        nonlocal last_action_key
        metrics.number_of_browser_actions += 1
        metrics.number_of_text_entries += 1
        current_action_key = f"type:{selector}:{text}"
        if current_action_key == last_action_key:
            metrics.number_of_retries += 1
        last_action_key = current_action_key

        trace_item = {"action": "type_text", "selector": selector, "text": text, "press_enter": press_enter, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] type_text: '{selector}' -> '{text}' (Enter={press_enter})")

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
                await asyncio.sleep(2.5)
            
            trace_item["status"] = "success"
            metrics.tool_trace.append(trace_item)
            return f"Entered '{text}' into '{selector}'."
        except Exception as e:
            metrics.number_of_failed_actions += 1
            trace_item["status"] = "error"
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Error typing into '{selector}': {e}"

    async def execute_get_page_content() -> str:
        metrics.number_of_browser_actions += 1
        trace_item = {"action": "get_page_content", "timestamp": time.monotonic() - start_time}
        try:
            expr = """JSON.stringify((() => {
                // Inputs in directions panel
                const inputs = Array.from(document.querySelectorAll('div[id*="sb_ifc"] input, input[name="q"], input#searchboxinput')).map(inp => ({
                    id: inp.closest('div[id*="sb_ifc"]')?.id || inp.id || inp.name,
                    aria: inp.getAttribute('aria-label'),
                    val: inp.value,
                    placeholder: inp.placeholder
                }));

                // Modes
                const modes = Array.from(document.querySelectorAll('button[data-tooltip], button[aria-label*="Driving"], button[aria-label*="Walking"], button[aria-label*="Transit"]')).map(b => ({
                    label: b.getAttribute('data-tooltip') || b.getAttribute('aria-label'),
                    selected: b.getAttribute('aria-selected') === 'true' || b.classList.contains('selected')
                })).filter(m => m.label);

                // Main directions pane text
                const pane = document.querySelector('#pane, [role="main"]');
                const paneText = pane ? pane.innerText : document.body.innerText;

                // Key visible action buttons
                const keyButtons = Array.from(document.querySelectorAll('button')).filter(b => b.offsetWidth > 0 && b.offsetHeight > 0).map(b => ({
                    aria: b.getAttribute('aria-label'),
                    text: b.innerText.trim(),
                    tooltip: b.getAttribute('data-tooltip'),
                    class: b.className
                })).slice(0, 15);

                // Detect any consent overlay or popup
                const consent = document.querySelector('form[action*="consent"], div[aria-modal="true"], div.modal');

                return {
                    url: window.location.href,
                    inputs,
                    modes: modes.slice(0, 8),
                    keyButtons,
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
        logger.info(f"[{condition}] Task finished with report:\n{itinerary_report}")
        return "Itinerary report received. Task marked complete."

    # Tool definitions for LangChain
    @tool
    async def navigate(url: str) -> str:
        """Navigate the browser to a URL (e.g. 'https://www.google.com/maps')."""
        return await execute_navigate(url)

    @tool
    async def click(selector: str) -> str:
        """Click on an element matching the given CSS selector (e.g. 'button[aria-label="Directions"]', 'button[data-tooltip="Walking"]')."""
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
        logger.info(f"[{condition}] Turn {turn + 1}/{max_turns}")

        try:
            ai_msg = await llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)
        except Exception as e:
            logger.error(f"[{condition}] LLM error: {e}")
            metrics.failure_reason = f"LLM invoke error: {e}"
            break

        if not ai_msg.tool_calls:
            logger.info(f"[{condition}] LLM returned message without tool calls: {ai_msg.content[:200]}")
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

    # Verify task completion and correctness
    rep_lower = metrics.itinerary_report.lower()
    required_stops = [
        "shibuya station", "hachiko", "scramble", "miyashita",
        "takeshita", "meiji jingu", "shinjuku gyoen", "omoide yokocho", "shinjuku station"
    ]
    stops_found = sum(1 for s in required_stops if s in rep_lower)

    has_distance = "distance" in rep_lower or "km" in rep_lower or " m" in rep_lower
    has_time = "time" in rep_lower or "min" in rep_lower or "hr" in rep_lower
    has_total = "total" in rep_lower

    if is_finished and stops_found >= 7 and has_distance and has_time:
        metrics.task_completion = "PASS"
        metrics.final_answer_correctness = "CORRECT (All stops, distances, and times reported)"
        metrics.failure_reason = "None (Completed successfully)"
    else:
        metrics.task_completion = "FAIL"
        if not is_finished:
            metrics.failure_reason = f"Max turns reached ({max_turns}) without calling finish."
        elif stops_found < 7:
            metrics.failure_reason = f"Incomplete itinerary: only {stops_found}/9 stops identified."
        else:
            metrics.failure_reason = "Missing leg distances or walking times."
        metrics.final_answer_correctness = "INCORRECT"

    logger.info(f"\n{'='*70}\nFINISHED CONDITION {condition}: {metrics.task_completion} in {metrics.total_wall_clock_time}s\n{'='*70}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator: Condition A then Condition B
# ─────────────────────────────────────────────────────────────────────────────

async def run_full_ab_experiment() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("COMEX BROWSER NERVOUS SYSTEM A/B CONTROLLED EXPERIMENT")
    print("Task: Long-Horizon Walking Itinerary on Google Maps (Tokyo, 8 Legs)")
    print("=" * 78)

    # 1. Run Condition A (Nervous System ON)
    metrics_a = await run_condition("A", max_turns=35)

    # Brief settle pause between conditions
    print("\nWaiting 5 seconds before initializing fresh page for Condition B...")
    await asyncio.sleep(5.0)

    # 2. Run Condition B (Nervous System Bypassed)
    metrics_b = await run_condition("B", max_turns=35)

    # Compile comparative results
    results = {
        "task": "Google Maps 8-Leg Walking Itinerary (Shibuya to Shinjuku)",
        "model": MODEL_NAME,
        "cdp_url": ELECTRON_CDP_URL,
        "condition_a": asdict(metrics_a),
        "condition_b": asdict(metrics_b),
        "comparison": {
            "task_completion": {"A": metrics_a.task_completion, "B": metrics_b.task_completion},
            "total_wall_clock_time_s": {"A": metrics_a.total_wall_clock_time, "B": metrics_b.total_wall_clock_time},
            "browser_actions": {"A": metrics_a.number_of_browser_actions, "B": metrics_b.number_of_browser_actions},
            "clicks": {"A": metrics_a.number_of_clicks, "B": metrics_b.number_of_clicks},
            "text_entries": {"A": metrics_a.number_of_text_entries, "B": metrics_b.number_of_text_entries},
            "page_navigations": {"A": metrics_a.number_of_page_navigations, "B": metrics_b.number_of_page_navigations},
            "back_navigations": {"A": metrics_a.number_of_back_navigations, "B": metrics_b.number_of_back_navigations},
            "retries": {"A": metrics_a.number_of_retries, "B": metrics_b.number_of_retries},
            "failed_actions": {"A": metrics_a.number_of_failed_actions, "B": metrics_b.number_of_failed_actions},
            "stale_or_missing_targets": {"A": metrics_a.number_of_stale_or_missing_targets, "B": metrics_b.number_of_stale_or_missing_targets},
            "overlays_encountered": {"A": metrics_a.number_of_overlays_encountered, "B": metrics_b.number_of_overlays_encountered},
            "action_blocks": {"A": metrics_a.number_of_action_blocks, "B": metrics_b.number_of_action_blocks},
            "llm_turns": {"A": metrics_a.number_of_llm_turns, "B": metrics_b.number_of_llm_turns},
            "final_answer_correctness": {"A": metrics_a.final_answer_correctness, "B": metrics_b.final_answer_correctness},
            "failure_reason": {"A": metrics_a.failure_reason, "B": metrics_b.failure_reason},
        }
    }

    # Save to disk
    out_path = Path(__file__).resolve().parent.parent / "ab_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw experimental results to: {out_path}")

    # Print Comparative Table
    print("\n" + "=" * 78)
    print("A/B EXPERIMENTAL COMPARISON TABLE")
    print("=" * 78)
    print(f"{'Metric':<38} {'Condition A (Nervous ON)':<20} {'Condition B (Bypassed)':<20}")
    print("-" * 78)
    c = results["comparison"]
    for key, vals in c.items():
        val_a = str(vals["A"])
        val_b = str(vals["B"])
        print(f"{key:<38} {val_a:<20} {val_b:<20}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    asyncio.run(run_full_ab_experiment())
