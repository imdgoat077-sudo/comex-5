#!/usr/bin/env python3
"""
Comex Controlled A/B Experiment:
Task: Amazon Laptop Search & Evaluation for CSE Student
Constraints:
- Price under ₹80,000
- At least 16 GB RAM
- At least 512 GB SSD
- Intel Core i5 / Ultra 5 or AMD Ryzen 5 / Ryzen 7
- Rating >= 4.0
- At least 500 reviews
- Inspect product detail page for every candidate
- Reject at least 3 products with explicit reasons
- Collect 5 valid products and compare in table
- Start strictly at https://www.amazon.in

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
logger = logging.getLogger("ab_amazon_laptops")

ELECTRON_CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
MODEL_NAME = os.getenv("MODEL_NAME", "ag/gemini-3.8-flash")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")

OUT_PATH = ROOT_DIR / "ab_amazon_laptops_results.json"


TASK_PROMPT = """You are an autonomous browser agent shopping on Amazon India (https://www.amazon.in).
Your task is to find 5 DIFFERENT laptops suitable for a CSE student.

STRICT CONSTRAINTS FOR VALID LAPTOPS:
- Price: Under ₹80,000 (INR)
- RAM: At least 16 GB
- Storage: At least 512 GB SSD
- Processor: Intel Core i5 / Ultra 5 OR AMD Ryzen 5 / Ryzen 7
- Rating: >= 4.0 out of 5 stars
- Reviews: At least 500 reviews

PROTOCOL RULES:
1. Start strictly from https://www.amazon.in.
2. Do NOT navigate directly to search or product URLs.
3. Type your search query into the search box UI (#twotabsearchtextbox) and submit.
4. For EACH candidate laptop:
   a. Click on the product link to open its full product page.
   b. Use get_page_content to inspect:
      - Price
      - RAM and Storage
      - Processor model
      - Rating and review count
      - Exact model name
   c. If it fails ANY constraint, call reject_product with the reason. (You MUST reject at least 3 candidates!).
   d. If it satisfies ALL constraints, call accept_product.
   e. Call go_back to return to the search results before inspecting the next candidate.
   f. Do NOT inspect the same product (ASIN) twice.
5. Once you have accepted 5 valid laptops and rejected at least 3 unsuitable candidates, call finish with your final comparison table.

SEARCH STRATEGY TIP:
- You need products with at least 500 reviews and rating >= 4.0. Newly launched listings often have very few reviews (<500).
- To find established models with >=500 reviews, you can search for popular bestselling series such as 'Acer Aspire Lite 16GB RAM', 'Lenovo IdeaPad 16GB RAM i5', 'HP 15s 16GB RAM i5', 'ASUS Vivobook 16GB RAM i5', 'Honor MagicBook 16GB', etc.
- Also inspect candidates that fail constraints (e.g. low reviews, rating < 4.0, or price > 80k) so you satisfy the requirement of rejecting at least 3 candidates!
"""


@dataclass
class AmazonRunMetrics:
    condition: str  # "A" or "B"
    condition_name: str
    task_completion: str = "FAIL"
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

    # Domain outputs
    accepted_products: List[Dict[str, Any]] = field(default_factory=list)
    rejected_products: List[Dict[str, Any]] = field(default_factory=list)
    comparison_table: str = ""

    # Condition A specifics
    browser_click_calls: int = 0
    allow_decisions: int = 0
    route_to_known_handler_decisions: int = 0
    report_blocked_decisions: int = 0
    runtime_safety_aborts: int = 0
    click_latencies_ms: List[float] = field(default_factory=list)

    # Condition B specifics
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


async def execute_amazon_condition(condition: str, max_turns: int = 75) -> AmazonRunMetrics:
    condition_name = "Comex Nervous System ON" if condition == "A" else "Comex Nervous System BYPASSED"
    logger.info(f"\n{'='*75}\nSTARTING RUN: CONDITION {condition} ({condition_name})\n{'='*75}")

    ws_url = get_active_browser_ws()
    session = await get_or_create_tab_session("active_tab", ws_url)
    await reset_page_to_blank(session)

    metrics = AmazonRunMetrics(
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

        if "/s?" in url or "/dp/" in url or "/gp/" in url:
            metrics.number_of_failed_actions += 1
            last_action_failed = True
            trace_item["error"] = "Rule violation: cannot navigate directly to search or product URLs"
            metrics.tool_trace.append(trace_item)
            return "Error: Protocol violation! You are NOT allowed to navigate directly to search or product URLs. Start at https://www.amazon.in and use the visible UI."

        try:
            await session.cdp.send("Page.navigate", {"url": url})
            await asyncio.sleep(3.5)
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

        # Normalize target to _self so Electron does not spawn detached windows
        normalize_target_expr = f"""(() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (el) {{
                const a = el.tagName === 'A' ? el : el.closest('a');
                if (a && a.target === '_blank') a.target = '_self';
            }}
        }})()"""
        await session.cdp.send("Runtime.evaluate", {"expression": normalize_target_expr})

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
                await asyncio.sleep(2.5)
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
            # Condition B: direct harness
            metrics.direct_harness_actions += 1
            try:
                js_expr = f"""(() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{ found: false }};
                    const a = el.tagName === 'A' ? el : el.closest('a');
                    if (a && a.target === '_blank') a.target = '_self';
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
                await asyncio.sleep(2.5)
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
                await asyncio.sleep(3.5)

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

    async def execute_go_back() -> str:
        metrics.number_of_browser_actions += 1
        metrics.number_of_back_navigations += 1
        trace_item = {"action": "go_back", "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] go_back")

        try:
            await session.cdp.send("Runtime.evaluate", {"expression": "window.history.back()"})
            await asyncio.sleep(3.0)
            trace_item["status"] = "success"
            metrics.tool_trace.append(trace_item)
            return "Navigated back in browser history. Call get_page_content to inspect the search results."
        except Exception as e:
            metrics.number_of_failed_actions += 1
            trace_item["error"] = str(e)
            metrics.tool_trace.append(trace_item)
            return f"Error navigating back: {e}"

    async def execute_scroll(delta_y: int = 800) -> str:
        metrics.number_of_browser_actions += 1
        metrics.number_of_scrolls += 1
        trace_item = {"action": "scroll", "delta_y": delta_y, "timestamp": time.monotonic() - start_time}
        logger.info(f"[{condition}] scroll: {delta_y}px")

        try:
            await session.cdp.send("Runtime.evaluate", {"expression": f"window.scrollBy(0, {delta_y})"})
            await asyncio.sleep(1.5)
            trace_item["status"] = "success"
            metrics.tool_trace.append(trace_item)
            return f"Scrolled page down by {delta_y}px."
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
                const productTitle = document.querySelector('#productTitle');
                
                // If on product page
                if (productTitle) {
                    const priceEl = document.querySelector('.priceToPay span.a-price-whole, #corePrice_feature_div .a-price-whole, .apexPriceToPay span.a-price-whole');
                    const ratingEl = document.querySelector('#acrPopover span.a-icon-alt, span[data-hook="rating-out-of-text"]');
                    const reviewsEl = document.querySelector('#acrCustomerReviewText, span[data-hook="total-review-count"]');
                    
                    const specs = {};
                    const rows = Array.from(document.querySelectorAll('#productDetails_techSpec_section_1 tr, #poExpander tr, table.a-keyvalue tr, #productOverview_feature_div tr'));
                    for (const r of rows) {
                        const th = r.querySelector('th, td.a-span3, td.po-label, span.a-text-bold');
                        const td = r.querySelector('td.a-size-base, td.a-span9, td.po-value, td:last-child');
                        if (th && td) {
                            const k = th.innerText.trim().replace(/[:\\s]+/g, ' ');
                            const v = td.innerText.trim();
                            if (k && v && !specs[k]) specs[k] = v;
                        }
                    }
                    
                    const bullets = Array.from(document.querySelectorAll('#feature-bullets li span.a-list-item')).map(b => b.innerText.trim()).slice(0, 4);
                    
                    return {
                        view: 'product_page',
                        url,
                        title: productTitle.innerText.trim().slice(0, 150),
                        price: priceEl ? priceEl.innerText.trim().replace(/,/g, '') : 'N/A',
                        rating: ratingEl ? (ratingEl.innerText || ratingEl.getAttribute('aria-label') || '').trim() : 'N/A',
                        reviews: reviewsEl ? reviewsEl.innerText.trim() : 'N/A',
                        processor: specs['CPU Model'] || specs['Processor Type'] || specs['CPU Model Number'] || specs['Processor Brand'] || 'N/A',
                        ram: specs['RAM Memory Installed Size'] || specs['RAM Size'] || specs['RAM Memory Installed'] || 'N/A',
                        storage: specs['Hard Disk Size'] || specs['Hard Drive Size'] || specs['Hard Disk Description'] || specs['DriveMemory Storage Capacity'] || 'N/A',
                        specs_sample: specs,
                        bullets
                    };
                }
                
                // If on search results page
                const cards = Array.from(document.querySelectorAll('div[data-component-type="s-search-result"]'));
                if (cards.length > 0) {
                    const items = cards.map((card, idx) => {
                        const h2 = card.querySelector('h2');
                        const priceEl = card.querySelector('.a-price-whole');
                        const ratingEl = card.querySelector('i.a-icon-star-small, span[aria-label*="out of 5"]');
                        const reviewsEl = card.querySelector('a[href*="#customerReviews"]');
                        const asin = card.getAttribute('data-asin');
                        return {
                            index: idx,
                            asin,
                            title: h2 ? h2.innerText.trim().slice(0, 120) : 'Untitled',
                            price_preview: priceEl ? priceEl.innerText.trim().replace(/,/g, '') : null,
                            rating_preview: ratingEl ? (ratingEl.getAttribute('aria-label') || ratingEl.innerText) : null,
                            reviews_preview: reviewsEl ? reviewsEl.innerText.trim() : null,
                            click_selector: `div[data-asin="${asin}"] a.a-text-normal, div[data-asin="${asin}"] a.s-no-outline`
                        };
                    }).filter(it => it.asin && it.title);
                    
                    return {
                        view: 'search_results',
                        url,
                        count: items.length,
                        products: items.slice(0, 10)
                    };
                }
                
                // Initial page
                const searchInp = document.querySelector('#twotabsearchtextbox');
                return {
                    view: 'home_page',
                    url,
                    search_input_selector: searchInp ? '#twotabsearchtextbox' : 'input[type="text"]'
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
        """Navigate to Amazon (must start at 'https://www.amazon.in')."""
        return await execute_navigate(url)

    @tool
    async def click(selector: str) -> str:
        """Click on an element (e.g. search button '#nav-search-submit-button', product link 'div[data-asin=\"...\"] a.a-text-normal')."""
        return await execute_click(selector)

    @tool
    async def type_text(selector: str, text: str, press_enter: bool = True) -> str:
        """Type search query into the search box and press Enter."""
        return await execute_type_text(selector, text, press_enter)

    @tool
    async def go_back() -> str:
        """Navigate back to the search results after inspecting a product page."""
        return await execute_go_back()

    @tool
    async def scroll_page(delta_y: int = 800) -> str:
        """Scroll down the page to view more search results or product details."""
        return await execute_scroll(delta_y)

    @tool
    async def get_page_content() -> str:
        """Get structured information about current page: search results list with selectors or product details (specs, price, rating, reviews)."""
        return await execute_get_page_content()

    @tool
    async def reject_product(asin: str, model: str, reason: str) -> str:
        """Log a candidate product that failed one or more constraints (At least 3 required)."""
        metrics.rejected_products.append({
            "asin": asin,
            "model": model,
            "reason": reason,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"REJECTED PRODUCT [{asin}]: {model} -> Reason: {reason}")
        return f"Logged rejection of [{asin}] {model}: {reason}. Total rejections: {len(metrics.rejected_products)}. Now call go_back."

    @tool
    async def accept_product(asin: str, model: str, price: str, processor: str, ram: str, ssd: str, rating: str, reviews: str) -> str:
        """Log an accepted valid product that satisfies ALL constraints (price < 80000, RAM >= 16GB, SSD >= 512GB, CPU i5/Ultra 5/Ryzen 5/7, rating >= 4.0, reviews >= 500). (5 required)."""
        metrics.accepted_products.append({
            "asin": asin,
            "model": model,
            "price": price,
            "processor": processor,
            "ram": ram,
            "ssd": ssd,
            "rating": rating,
            "reviews": reviews,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"ACCEPTED PRODUCT [{asin}]: {model} (Price: {price}, RAM: {ram}, SSD: {ssd}, CPU: {processor}, Rating: {rating}, Reviews: {reviews})")
        return f"Recorded valid product [{asin}] {model}. Total accepted: {len(metrics.accepted_products)}/5. Now call go_back."

    @tool
    async def finish(comparison_table: str) -> str:
        """Call when you have accepted 5 valid products and rejected at least 3 unsuitable candidates, providing the final markdown comparison table."""
        nonlocal is_finished
        is_finished = True
        metrics.comparison_table = comparison_table
        metrics.tool_trace.append({
            "action": "finish",
            "table": comparison_table,
            "timestamp": time.monotonic() - start_time
        })
        logger.info(f"[{condition}] TASK FINISH CALLED.")
        return "Comparison submitted. Task complete."

    tools = [
        navigate, click, type_text, go_back, scroll_page, get_page_content,
        reject_product, accept_product, finish
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
        SystemMessage(content="You are an autonomous browser shopping agent on Amazon India."),
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
            if len(metrics.accepted_products) >= 5 and len(metrics.rejected_products) >= 3:
                is_finished = True
                metrics.comparison_table = str(ai_msg.content)
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

    # Verification
    has_5_accepted = len(metrics.accepted_products) >= 5
    has_3_rejected = len(metrics.rejected_products) >= 3
    has_back_navs = metrics.number_of_back_navigations >= 5

    # Check unique ASINs
    asins = [p.get("asin") for p in metrics.accepted_products if p.get("asin")]
    has_unique = len(set(asins)) == len(asins)

    if is_finished and has_5_accepted and has_3_rejected and has_unique and has_back_navs:
        metrics.task_completion = "PASS"
        metrics.final_data_correctness = "CORRECT (5 valid laptops verified against constraints + >=3 rejected with reasons + UI back returns)"
        metrics.failure_reason = "None (Completed successfully)"
    else:
        metrics.task_completion = "FAIL"
        metrics.final_data_correctness = "INCOMPLETE / UNVERIFIED"
        reasons = []
        if not is_finished:
            reasons.append("Max turns reached without calling finish")
        if not has_5_accepted:
            reasons.append(f"Only {len(metrics.accepted_products)}/5 valid products accepted")
        if not has_3_rejected:
            reasons.append(f"Only {len(metrics.rejected_products)}/3 rejections logged")
        if not has_back_navs:
            reasons.append(f"Only {metrics.number_of_back_navigations} back navigations executed")
        metrics.failure_reason = "; ".join(reasons)

    logger.info(f"\n{'='*75}\nCOMPLETED CONDITION {condition}: {metrics.task_completion} in {metrics.total_wall_clock_time}s\n{'='*75}")
    return metrics


async def run_amazon_ab_experiment():
    print("\n" + "=" * 78)
    print("  COMEX CONTROLLED A/B EXPERIMENT: AMAZON LAPTOP SEARCH & SELECTION")
    print("  Task: Find 5 CSE Laptops under ₹80k, >=16GB RAM, >=512GB SSD, Core i5/Ultra 5 or Ryzen 5/7,")
    print("        >=4.0 stars, >=500 reviews. Reject >=3 products. Return via UI Back.")
    print("=" * 78 + "\n")

    results = []

    # Condition A (Nervous System ON)
    print("\n>>> RUNNING CONDITION A (Comex Nervous System ON)...")
    res_a = await execute_amazon_condition("A")
    results.append(res_a)

    print("\nSettle pause 5.0s between conditions...")
    await asyncio.sleep(5.0)

    # Condition B (Nervous System BYPASSED)
    print("\n>>> RUNNING CONDITION B (Comex Nervous System BYPASSED)...")
    res_b = await execute_amazon_condition("B")
    results.append(res_b)

    # Save results
    output_data = {
        "metadata": {
            "task": "Amazon India Laptop Search & Evaluation",
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
    asyncio.run(run_amazon_ab_experiment())
