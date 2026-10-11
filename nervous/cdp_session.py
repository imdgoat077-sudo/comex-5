"""
Comex Browser Nervous System — Stage C: Per-Tab CDP Session & Controlled Mutation Plane.

Encapsulates per-tab WebSocket CDP sessions, ordered event queues with sequence
barriers, explicit DOM object resolution, Method E verification on bound remote
objects, and closed-loop execution through ActuatorGate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .actuator_gate import ActuatorGate, GateResult
from .executor import SelectorClickExecutor
from .method_e import METHOD_E_CHECK_FUNC
from .policy_registry import OverlayPolicyMatcher, PolicyRegistry, ReflexActionRouter, SafeOverlayHandler
from .target_binding import BindingFailureReason, TargetBinding
from .world_state import BrowserState, WorldStateTracker

logger = logging.getLogger("comex.nervous.cdp_session")

# Bounds the links returned by extract_read_only (each link is capped at ~400 chars)
MAX_EXTRACT_LINKS = 60

# Attribute used to give page elements stable numeric references ("refs") agents can act on.
REF_ATTRIBUTE = "data-comex-ref"


def ref_selector(ref: Any) -> str:
    """CSS selector for an element ref assigned by inspect_elements/find_by_text."""
    return f'[{REF_ATTRIBUTE}="{int(ref)}"]'


# Shared in-page helpers for numbered element inspection. Refs are kept on the element and
# numbered from a per-document counter, so re-inspecting a page keeps existing numbers stable.
_REF_HELPERS_JS = (
    "const INTERACTIVE_SELECTOR = 'a[href], button, input:not([type=hidden]), select, textarea, summary,"
    " [role=button], [role=link], [role=tab], [role=menuitem], [role=checkbox], [role=radio], [role=option],"
    " [role=switch], [role=combobox], [role=searchbox], [role=textbox], [contenteditable=true], [tabindex=\"0\"]';"
    "const isVisible = (el) => {"
    "  const r = el.getBoundingClientRect();"
    "  if (r.width < 1 || r.height < 1) return false;"
    "  const cs = window.getComputedStyle(el);"
    "  return cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';"
    "};"
    "const clean = (s, n) => (s || '').replace(/\\s+/g, ' ').trim().slice(0, n);"
    "const labelOf = (el) => {"
    "  const byId = el.id ? document.querySelector(`label[for=\"${CSS.escape(el.id)}\"]`) : null;"
    "  const img = el.querySelector ? el.querySelector('img[alt]') : null;"
    "  const titled = el.querySelector ? el.querySelector('[title], [aria-label]') : null;"
    "  return clean(el.getAttribute('aria-label') || el.innerText || (byId && byId.innerText) ||"
    "    el.getAttribute('placeholder') || el.getAttribute('title') || (img && img.alt) ||"
    "    (titled && (titled.getAttribute('aria-label') || titled.getAttribute('title'))) || el.value || el.name || '', 100);"
    "};"
    "const kindOf = (el) => {"
    "  const role = el.getAttribute('role'); if (role) return role;"
    "  const tag = el.tagName.toLowerCase();"
    "  if (tag === 'a') return 'link';"
    "  if (tag === 'select') return 'dropdown';"
    "  if (tag === 'textarea' || el.isContentEditable) return 'textbox';"
    "  if (tag === 'input') {"
    "    const t = (el.type || 'text').toLowerCase();"
    "    if (['submit', 'button', 'reset', 'image'].includes(t)) return 'button';"
    "    if (['checkbox', 'radio'].includes(t)) return t;"
    "    return t === 'search' ? 'searchbox' : (t === 'text' ? 'textbox' : t + ' input');"
    "  }"
    "  return tag;"
    "};"
    f"const REF_ATTR = '{REF_ATTRIBUTE}';"
    "const refOf = (el) => {"
    "  let ref = el.getAttribute(REF_ATTR);"
    "  if (!ref) { window.__comexRefSeq = (window.__comexRefSeq || 0) + 1; ref = String(window.__comexRefSeq); el.setAttribute(REF_ATTR, ref); }"
    "  return Number(ref);"
    "};"
    "const describe = (el) => {"
    "  const item = { ref: refOf(el), kind: kindOf(el), label: labelOf(el), tag: el.tagName.toLowerCase() };"
    "  item.selector = `[${REF_ATTR}=\"${item.ref}\"]`;"
    "  if (el.href) item.href = clean(el.href, 200);"
    "  if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {"
    "    if (el.value) item.value = clean(el.value, 60);"
    "    if (el.placeholder) item.placeholder = clean(el.placeholder, 60);"
    "  }"
    "  if (el.type === 'checkbox' || el.type === 'radio') item.checked = !!el.checked;"
    "  if (el.disabled || el.getAttribute('aria-disabled') === 'true') item.disabled = true;"
    "  return item;"
    "};"
)



class TabSessionContext:
    """
    Per-tab session context holding the target's dedicated WorldStateTracker,
    ordered event-processing queue, ActuatorGate, and SelectorClickExecutor.
    """

    def __init__(
        self,
        tab_id: str,
        target_id: str,
        cdp_client: Any,
        main_frame_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        action_lock: Optional[asyncio.Lock] = None,
    ) -> None:
        self.tab_id = tab_id
        self.target_id = target_id
        self.worker_id = worker_id
        self.cdp = cdp_client
        self.main_frame_id = main_frame_id

        # Canonical per-tab lock: shared across all worker sessions targeting this tab.
        # Production invariant: all sessions targeting the same canonical Chromium target share the same lock.
        # Explicit lock injection is strictly reserved for controlled testing and isolation fixtures;
        # in production (and by default), the shared lock is always derived from the canonical target registry.
        canonical_id = extract_canonical_target_id(getattr(cdp_client, "ws_url", ""), target_id or tab_id)
        if action_lock is not None:
            self.action_lock = action_lock
        else:
            self.action_lock = get_canonical_tab_lock(canonical_id)

        # Wire notification handler if client supports it
        if hasattr(cdp_client, "set_notification_handler"):
            handler_res = cdp_client.set_notification_handler(self.on_cdp_notification)
            if asyncio.iscoroutine(handler_res):
                handler_res.close()

        # Dedicated per-tab nervous system components
        self.tracker = WorldStateTracker(tab_id=tab_id, target_ws_url=getattr(cdp_client, "ws_url", ""))
        self.executor = SelectorClickExecutor(self.cdp, self.tracker.state, tab_id=tab_id, target_id=target_id)
        self.policy_registry = PolicyRegistry()
        self.overlay_handler = SafeOverlayHandler(self.policy_registry, self)
        self.action_router = ReflexActionRouter(overlay_handler=self.overlay_handler)
        self.gate = ActuatorGate(self.tracker.state, self.executor, handler=self.action_router)
        self.matcher = OverlayPolicyMatcher(self.policy_registry, self.cdp)

        # Monotonic event sequence & barrier primitives
        self._rx_seq: int = 0
        self._processed_watermark: int = 0
        self._event_queue: asyncio.Queue[Tuple[int, str, Dict[str, Any]]] = asyncio.Queue()
        self._watermark_cond = asyncio.Condition()
        self._pump_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Auto-answered JavaScript dialogs (see _auto_answer_dialog)
        self.last_dialog: Optional[Dict[str, Any]] = None
        self._dialog_tasks: set = set()

    async def start(self) -> None:
        """Start the sequential event processing loop."""
        if self._running:
            return
        self._running = True
        self._pump_task = asyncio.create_task(self._process_events_loop())

    async def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        if self._pump_task:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass

    def set_visual_cursor_hook(self, hook: Optional[Callable[[float, float, str, str], None]]) -> None:
        """Set or update the non-blocking visual cursor hook on the executor."""
        self.executor.set_visual_cursor_hook(hook)

    def on_cdp_notification(self, method: str, params: Dict[str, Any]) -> int:
        """
        Called synchronously by the WebSocket receive loop upon receiving a CDP event.
        Assigns a monotonically increasing sequence number and enqueues the event.
        Returns the assigned sequence number.
        """
        self._rx_seq += 1
        seq = self._rx_seq
        self._event_queue.put_nowait((seq, method, params))
        if method == "Page.javascriptDialogOpening":
            self._auto_answer_dialog(params)
        return seq

    def _auto_answer_dialog(self, params: Dict[str, Any]) -> None:
        """
        A JavaScript dialog blocks the page, so every later agent action on the tab would hang.
        Answer it immediately: alerts and leave-page prompts are accepted (they only inform or let the
        agent's own navigation proceed); confirm/prompt are dismissed, so an agent can never silently
        confirm something like a deletion. The dialog is recorded for the agent to see.
        """
        dialog_type = params.get("type", "alert")
        accept = dialog_type in ("alert", "beforeunload")
        self.last_dialog = {
            "type": dialog_type,
            "message": str(params.get("message", ""))[:300],
            "action": "accepted" if accept else "dismissed",
        }
        logger.info("Auto-%s %s dialog on tab %s: %s", self.last_dialog["action"], dialog_type,
                    self.tab_id, self.last_dialog["message"][:120])
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.cdp.send("Page.handleJavaScriptDialog", {"accept": accept}))
        # Keep a reference so the task isn't garbage-collected; surface failures in the log only.
        self._dialog_tasks.add(task)
        task.add_done_callback(self._dialog_tasks.discard)
        task.add_done_callback(lambda t: t.cancelled() or (t.exception() and logger.warning(
            "Could not answer dialog on tab %s: %s", self.tab_id, t.exception())))

    def take_last_dialog(self) -> Optional[Dict[str, Any]]:
        """Return and clear the most recent auto-answered dialog, if any."""
        dialog, self.last_dialog = self.last_dialog, None
        return dialog

    async def _process_events_loop(self) -> None:
        """Sequential consumer applying CDP events in FIFO order to WorldStateTracker."""
        while self._running:
            try:
                seq, method, params = await self._event_queue.get()
                self._apply_event_to_tracker(method, params)
                async with self._watermark_cond:
                    self._processed_watermark = seq
                    self._watermark_cond.notify_all()
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error processing CDP event in TabSessionContext: %s", e)

    def _apply_event_to_tracker(self, method: str, params: Dict[str, Any]) -> None:
        """Route CDP event to WorldStateTracker."""
        if method == "DOM.documentUpdated":
            self.tracker.handle_document_updated()
        elif method == "DOM.childNodeInserted":
            self.tracker.handle_child_node_inserted(
                params.get("parentNodeId", 0),
                params.get("node", {}).get("nodeId", 0),
            )
        elif method == "DOM.childNodeRemoved":
            self.tracker.handle_child_node_removed(
                params.get("parentNodeId", 0),
                params.get("nodeId", 0),
            )
        elif method == "DOM.attributeModified":
            self.tracker.handle_attribute_modified(
                params.get("nodeId", 0),
                params.get("name", ""),
                params.get("value", ""),
            )
        elif method == "DOM.attributeRemoved":
            self.tracker.handle_attribute_removed(
                params.get("nodeId", 0),
                params.get("name", ""),
            )
        elif method == "DOM.characterDataModified":
            self.tracker.handle_character_data_modified(
                params.get("nodeId", 0),
                params.get("characterData", ""),
            )
        elif method == "DOM.setChildNodes":
            self.tracker.handle_set_child_nodes(
                params.get("parentId", 0),
                params.get("nodes", []),
            )
        elif method == "Page.frameNavigated":
            frame = params.get("frame", {})
            is_main = not frame.get("parentId")
            if is_main and frame.get("id"):
                self.main_frame_id = frame.get("id")
            self.tracker.handle_frame_navigated(frame.get("url", ""), is_main_frame=is_main)
        elif method == "Page.loadEventFired":
            self.tracker.handle_load_event_fired()
        elif method == "Page.javascriptDialogOpening":
            self.tracker.handle_javascript_dialog_opening(
                params.get("type", "alert"),
                params.get("message", ""),
                params.get("defaultPrompt", ""),
            )
        elif method == "Page.javascriptDialogClosed":
            self.tracker.handle_javascript_dialog_closed()
        elif method == "Network.requestWillBeSent":
            self.tracker.handle_request_will_be_sent(
                params.get("requestId", ""),
                params.get("type", "Other"),
            )
        elif method == "Network.loadingFinished":
            self.tracker.handle_loading_finished(params.get("requestId", ""))
        elif method == "Network.loadingFailed":
            self.tracker.handle_loading_failed(params.get("requestId", ""))

    async def wait_until_processed(self, sequence_watermark: int, timeout: float = 3.0) -> None:
        """
        Ordered event barrier: Wait until all CDP events received up to sequence_watermark
        have been applied to WorldStateTracker.
        """
        t_deadline = time.monotonic() + timeout
        async with self._watermark_cond:
            while self._processed_watermark < sequence_watermark:
                remaining = t_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for watermark {sequence_watermark} "
                        f"(current: {self._processed_watermark})"
                    )
                try:
                    await asyncio.wait_for(self._watermark_cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Timed out waiting for watermark {sequence_watermark} "
                        f"(current: {self._processed_watermark})"
                    )

    async def flush_and_sync(self) -> int:
        """
        Send a round-trip probe over CDP and await processing of all preceding events.
        Guarantees that WorldStateTracker reflects all events dispatched prior to the probe.
        """
        # 1. Send round-trip probe
        doc_res = await self.cdp.send("DOM.getDocument", {"depth": 0})
        # 2. Capture the sequence number at this point
        watermark = self._rx_seq
        # 3. Wait until event processing loop drains through watermark
        await self.wait_until_processed(watermark)
        return watermark

    async def evaluate_overlay_state(self) -> None:
        """Evaluate overlay presence and policy match against current DOM state."""
        await self.matcher.evaluate_overlay_state(self.tracker.state)

    async def resolve_and_bind(self, selector: str) -> TargetBinding:
        """
        Resolve the selector via CDP, evaluate Method E against the resolved remote object,
        and construct an immutable TargetBinding.
        """
        # Step 1: Flush event queue so tracker is fully up-to-date
        watermark = await self.flush_and_sync()
        await self.evaluate_overlay_state()

        # Step 2: Resolve selector to DOM nodeId with full descendant tree population
        doc = await self.cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})
        root = doc.get("root", {})
        root_node_id = root.get("nodeId", 1)
        self.tracker.populate_tree_hierarchy(root)

        q_res = await self.cdp.send("DOM.querySelector", {
            "nodeId": root_node_id,
            "selector": selector,
        })
        node_id = q_res.get("nodeId", 0)
        if not node_id:
            raise RuntimeError(f"Selector '{selector}' not found in document")

        # Step 2b: Ensure target subtree descendants are requested and monitored in CDP DOM agent
        try:
            await self.cdp.send("DOM.requestChildNodes", {"nodeId": node_id, "depth": -1})
        except Exception:
            pass

        # Step 3: Describe node to extract backend_node_id and frame_id
        desc_res = await self.cdp.send("DOM.describeNode", {"nodeId": node_id})
        node_desc = desc_res.get("node", {})
        backend_node_id = node_desc.get("backendNodeId")
        frame_id = node_desc.get("frameId")

        if backend_node_id is None:
            raise RuntimeError(f"Could not extract backendNodeId for selector '{selector}'")

        # Step 4: Main-frame safety assertion (Milestone 1)
        if self.main_frame_id and frame_id and frame_id != self.main_frame_id:
            raise RuntimeError(
                f"Selector '{selector}' resolves inside iframe ({frame_id}). "
                "Cross-frame targets are explicitly out of scope for Milestone 1."
            )

        # Step 5: Resolve remote object to evaluate Method E on the identical node handle
        obj_res = await self.cdp.send("DOM.resolveNode", {"nodeId": node_id})
        remote_obj = obj_res.get("object", {})
        object_id = remote_obj.get("objectId")
        if not object_id:
            raise RuntimeError(f"Could not resolve remote object handle for selector '{selector}'")

        # Step 6: Evaluate frozen Method E on the bound remote object
        call_res = await self.cdp.send("Runtime.callFunctionOn", {
            "objectId": object_id,
            "functionDeclaration": METHOD_E_CHECK_FUNC,
            "returnByValue": True,
        })
        eval_val = call_res.get("result", {}).get("value", {})
        is_actionable = bool(eval_val.get("actionable", False))
        box = eval_val.get("box", {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0})

        # Release the remote object handle
        try:
            await self.cdp.send("Runtime.releaseObject", {"objectId": object_id})
        except Exception:
            pass

        now = time.monotonic()

        # Step 7: Create immutable TargetBinding
        binding = TargetBinding(
            tab_id=self.tab_id,
            target_id=self.target_id,
            session_id=None,
            frame_id=frame_id,
            selector=selector,
            node_id=node_id,
            backend_node_id=backend_node_id,
            document_generation=self.tracker.state.document_generation,
            target_subtree_mutation_generation=self.tracker.state.target_subtree_mutation_generation,
            state_watermark=watermark,
            bounding_box=box,
            resolved_at_mono=now,
        )

        # Step 8: Update WorldStateTracker observed state (is_stable=False requires consecutive observation)
        self.tracker.state.assign_target(
            locator=selector,
            node_id=node_id,
            backend_node_id=backend_node_id,
            is_actionable=is_actionable,
            is_stable=False,
            bounding_box=box,
        )

        # Step 9: Sample consecutive observation to establish initial geometric stability
        await self.update_target_position()

        return binding

    async def update_target_position(self) -> None:
        """
        Sample target position via CDP and update WorldStateTracker to establish
        geometric stability across consecutive observations.
        """
        target = self.tracker.state.current_target
        if not target:
            return
        try:
            if target.backend_node_id is not None:
                box_res = await self.cdp.send(
                    "DOM.getBoxModel",
                    {"backendNodeId": target.backend_node_id},
                )
            elif target.node_id is not None:
                box_res = await self.cdp.send(
                    "DOM.getBoxModel",
                    {"nodeId": target.node_id},
                )
            else:
                return
            model = box_res.get("model", {})
            border_quad = model.get("border") or model.get("content")
            if border_quad and len(border_quad) >= 8:
                xs = [border_quad[0], border_quad[2], border_quad[4], border_quad[6]]
                ys = [border_quad[1], border_quad[3], border_quad[5], border_quad[7]]
                min_x = min(xs)
                min_y = min(ys)
                width = float(model.get("width", max(xs) - min_x))
                height = float(model.get("height", max(ys) - min_y))
                self.tracker.state.update_target_position(min_x, min_y, width, height)
        except Exception as e:
            logger.debug("Failed to sample target position: %s", e)

    async def navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate tab to url with scheme validation and lifecycle settlement.
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"Disallowed navigation scheme: '{parsed.scheme}'. Only http:// and https:// are permitted.")

        t0 = time.monotonic()
        nav_res = await self.cdp.send("Page.navigate", {"url": url})
        frame_id = nav_res.get("frameId")
        error_text = nav_res.get("errorText")
        if error_text:
            raise RuntimeError(f"Navigation error: {error_text}")

        # Wait briefly for document settlement
        await asyncio.sleep(0.5)
        try:
            await self.flush_and_sync()
        except Exception:
            pass
        dur_ms = int((time.monotonic() - t0) * 1000)
        return {
            "navigated": True,
            "url": url,
            "frameId": frame_id,
            "duration_ms": dur_ms,
        }

    async def extract_read_only(self, selector: Optional[str] = None, max_chars: int = 8000) -> Dict[str, Any]:
        """
        Pure read-only extraction of document or element content.
        Does not mutate DOM or dispatch physical inputs.

        A selector matches ALL elements (e.g. every product card), not just the first, and the
        result includes the links inside the extracted content. Without both, agents asked for
        "a link for each product" looped through selector guesses, since innerText drops hrefs.
        """
        js_code = (
            "(function() {"
            f"  const sel = {json.dumps(selector)};"
            f"  const maxLen = {int(max_chars)};"
            f"  const maxLinks = {int(MAX_EXTRACT_LINKS)};"
            "  let els = sel ? Array.from(document.querySelectorAll(sel)) : [document.body || document.documentElement];"
            "  if (sel && els.length === 0) return { error: `Selector '${sel}' not found` };"
            "  const text = els.map(e => (e.innerText || e.textContent || '').trim()).filter(Boolean).join('\\n---\\n');"
            "  const links = []; const seen = new Set();"
            "  for (const root of els) {"
            "    const anchors = root.tagName === 'A' ? [root] : root.querySelectorAll('a[href]');"
            "    for (const a of anchors) {"
            "      if (links.length >= maxLinks) break;"
            "      if (!/^https?:/i.test(a.href || '')) continue;"
            "      if (a.closest('header, nav, footer, [role=\"navigation\"], [role=\"banner\"]')) continue;"
            # Unwrap redirect/tracking links (e.g. Amazon sponsored /sspa/click?url=/item/dp/...)
            "      let u = new URL(a.href);"
            "      const target = u.searchParams.get('url') || u.searchParams.get('u');"
            "      if (target && (target.startsWith('/') || /^https?:/i.test(target))) {"
            "        try { u = new URL(target, u.origin); } catch (e) {}"
            "      }"
            # Long paths identify the page (/dp/ASIN, /p/itm...): drop tracking queries and collapse
            # duplicate links to the same item (rating count, colour variants). Short paths keep the query (?q=).
            "      const href = u.pathname.length > 20 ? u.origin + u.pathname : u.origin + u.pathname + u.search;"
            "      if (seen.has(href)) continue;"
            "      const label = (a.innerText || a.getAttribute('aria-label') || a.title || '').trim().replace(/\\s+/g, ' ');"
            "      if (!label) continue;"
            "      seen.add(href);"
            "      links.push({ text: label.slice(0, 100), href: href.slice(0, 300) });"
            "    }"
            "  }"
            "  return {"
            "    title: document.title,"
            "    url: window.location.href,"
            "    matched: els.length,"
            "    text: text.slice(0, maxLen),"
            "    truncated: text.length > maxLen,"
            "    length: text.length,"
            "    links: links"
            "  };"
            "})()"
        )
        res = await self.cdp.send("Runtime.evaluate", {
            "expression": js_code,
            "returnByValue": True,
            "awaitPromise": False,
        })
        val = res.get("result", {}).get("value", {})
        if isinstance(val, dict) and "error" in val:
            raise RuntimeError(val["error"])
        return val

    async def inspect_elements(self, max_elements: int = 50) -> Dict[str, Any]:
        """
        Numbered inspection of visible interactive elements.

        Each element is tagged in the page with a stable `data-comex-ref` number, so an agent can act on
        "ref 12" instead of guessing CSS selectors; `[data-comex-ref="12"]` is then a precise selector that
        still goes through the full actionability/quiescence pipeline. Elements in the viewport come first.
        The attribute is the only DOM change made.
        """
        js_code = (
            "(function() {"
            f"  const limit = {int(max_elements)};"
            f"  {_REF_HELPERS_JS}"
            "  const all = Array.from(document.querySelectorAll(INTERACTIVE_SELECTOR)).filter(isVisible);"
            "  const vh = window.innerHeight;"
            "  const inView = (el) => { const r = el.getBoundingClientRect(); return r.bottom > 0 && r.top < vh; };"
            "  const ordered = all.filter(inView).concat(all.filter(el => !inView(el)));"
            "  const items = ordered.slice(0, limit).map(describe);"
            "  return {"
            "    title: document.title,"
            "    url: window.location.href,"
            "    elements: items,"
            "    total_found: all.length,"
            "    returned: items.length"
            "  };"
            "})()"
        )
        res = await self.cdp.send("Runtime.evaluate", {
            "expression": js_code,
            "returnByValue": True,
            "awaitPromise": False,
        })
        return res.get("result", {}).get("value", {})

    async def find_by_text(self, text: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Find visible elements whose label matches `text` (exact, then prefix, then substring; case-insensitive),
        preferring interactive elements. Matches are tagged with refs like inspect_elements.
        """
        js_code = (
            "(function() {"
            f"  const query = {json.dumps(text)}.trim().toLowerCase().replace(/\\s+/g, ' ');"
            f"  const limit = {int(max_results)};"
            f"  {_REF_HELPERS_JS}"
            "  if (!query) return { title: document.title, url: location.href, elements: [], returned: 0 };"
            "  const interactive = Array.from(document.querySelectorAll(INTERACTIVE_SELECTOR)).filter(isVisible);"
            "  const textual = Array.from(document.querySelectorAll('label, h1, h2, h3, h4, li, span, div, p, td'))"
            "    .filter(el => el.children.length === 0 && isVisible(el));"
            "  const scored = [];"
            "  const consider = (el, bonus, label) => {"
            "    const l = label.toLowerCase().replace(/\\s+/g, ' ');"
            "    if (!l) return;"
            "    let score = l === query ? 3 : l.startsWith(query) ? 2 : l.includes(query) ? 1 : 0;"
            "    if (score) scored.push({ el, score: score + bonus, len: l.length });"
            "  };"
            "  interactive.forEach(el => consider(el, 3, labelOf(el)));"
            # Text in a plain span/div inside a link or button: the clickable thing is the enclosing control
            "  textual.forEach(el => { const ctl = el.closest(INTERACTIVE_SELECTOR); consider(ctl || el, ctl ? 2 : 0, labelOf(el)); });"
            "  scored.sort((a, b) => b.score - a.score || a.len - b.len);"
            "  const seen = new Set(); const items = [];"
            "  for (const s of scored) {"
            "    if (seen.has(s.el)) continue; seen.add(s.el);"
            "    items.push(describe(s.el));"
            "    if (items.length >= limit) break;"
            "  }"
            "  return { title: document.title, url: location.href, elements: items, returned: items.length };"
            "})()"
        )
        res = await self.cdp.send("Runtime.evaluate", {
            "expression": js_code,
            "returnByValue": True,
            "awaitPromise": False,
        })
        return res.get("result", {}).get("value", {})


class WebSocketCDPClient:
    """Async WebSocket CDP client managing request/response routing and notifications."""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self._ws: Optional[Any] = None
        self._msg_id: int = 0
        self._pending: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._notification_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        if self._ws:
            return
        import websockets
        self._ws = await websockets.connect(self.ws_url, max_size=20 * 1024 * 1024)
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def close(self) -> None:
        if self._listen_task:
            self._listen_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None

    def set_notification_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        self._notification_handler = handler

    async def _listen_loop(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                msg_id = data.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        if "error" in data:
                            fut.set_exception(RuntimeError(data["error"].get("message", "CDP Error")))
                        else:
                            fut.set_result(data.get("result", {}))
                elif "method" in data and self._notification_handler:
                    self._notification_handler(data["method"], data.get("params", {}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("CDP listen loop error on %s: %s", self.ws_url, e)

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._ws:
            await self.connect()
        self._msg_id += 1
        call_id = self._msg_id
        fut: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = fut
        payload = {"id": call_id, "method": method, "params": params or {}}
        await self._ws.send(json.dumps(payload))
        return await fut


_GLOBAL_SESSIONS: Dict[str, TabSessionContext] = {}
_CANONICAL_TAB_LOCKS: Dict[str, asyncio.Lock] = {}


def extract_canonical_target_id(ws_url: Optional[str], fallback_tab_id: str) -> str:
    """Extract stable canonical Chromium target ID from CDP WebSocket URL or tab identity."""
    if ws_url:
        if "/devtools/page/" in ws_url:
            return ws_url.split("/devtools/page/")[-1].split("?")[0].strip()
        if "/page/" in ws_url:
            return ws_url.split("/page/")[-1].split("?")[0].strip()
    return fallback_tab_id.strip() if fallback_tab_id else "default_tab"


def get_canonical_tab_lock(canonical_target_id: str) -> asyncio.Lock:
    """
    Returns the shared serialization lock for the specified canonical tab target.
    All sessions and workers targeting the same underlying Chromium page share this lock.
    """
    if not canonical_target_id:
        canonical_target_id = "default_tab"
    if canonical_target_id not in _CANONICAL_TAB_LOCKS:
        _CANONICAL_TAB_LOCKS[canonical_target_id] = asyncio.Lock()
    return _CANONICAL_TAB_LOCKS[canonical_target_id]


def reset_tab_locks() -> None:
    """Reset canonical locks and global sessions (for test isolation)."""
    _CANONICAL_TAB_LOCKS.clear()
    _GLOBAL_SESSIONS.clear()


async def get_or_create_tab_session(tab_id: str, ws_url: str, worker_id: Optional[str] = None) -> TabSessionContext:
    """Retrieve or initialize a persistent TabSessionContext for tab_id and optional worker_id."""
    if not ws_url or not ws_url.startswith("ws"):
        raise ValueError(f"Invalid WebSocket CDP URL: {ws_url!r}")

    session_key = f"{worker_id}:{tab_id}" if worker_id else tab_id
    session = _GLOBAL_SESSIONS.get(session_key)

    if session and getattr(session.cdp, "ws_url", "") == ws_url:
        return session

    # If session exists for this key but ws_url changed, gracefully close the stale session first
    if session:
        try:
            await session.stop()
            if hasattr(session.cdp, "close"):
                await session.cdp.close()
        except Exception as e:
            logger.warning("Error stopping stale session %s: %s", session_key, e)
        _GLOBAL_SESSIONS.pop(session_key, None)

    client = WebSocketCDPClient(ws_url)
    await client.connect()
    # Enable core domains
    await asyncio.gather(
        client.send("Page.enable"),
        client.send("DOM.enable"),
        client.send("Runtime.enable"),
        client.send("Network.enable"),
        client.send("Emulation.setFocusEmulationEnabled", {"enabled": True}),
    )

    canonical_id = extract_canonical_target_id(ws_url, tab_id)
    shared_lock = get_canonical_tab_lock(canonical_id)

    new_session = TabSessionContext(
        tab_id=tab_id,
        target_id=canonical_id,
        cdp_client=client,
        worker_id=worker_id,
        action_lock=shared_lock,
    )
    if hasattr(client, "set_notification_handler"):
        notif_res = client.set_notification_handler(new_session.on_cdp_notification)
        if asyncio.iscoroutine(notif_res):
            notif_res.close()

    await new_session.start()
    _GLOBAL_SESSIONS[session_key] = new_session
    return new_session


async def close_tab_session(session_key: str) -> None:
    """Closes and unregisters a tab session by its session key."""
    session = _GLOBAL_SESSIONS.pop(session_key, None)
    if session:
        try:
            await session.stop()
            if hasattr(session.cdp, "close"):
                await session.cdp.close()
        except Exception as e:
            logger.warning("Error closing tab session %s: %s", session_key, e)


async def cleanup_worker_sessions(worker_id: str) -> None:
    """Clean up all sessions associated with a worker_id."""
    prefix = f"{worker_id}:"
    keys_to_clean = [k for k in list(_GLOBAL_SESSIONS.keys()) if k.startswith(prefix) or k == worker_id]
    for k in keys_to_clean:
        await close_tab_session(k)


def validate_worker_tab_authorization(
    worker_id: Optional[str],
    target_tab: str,
    assigned_tab: Optional[str] = None
) -> None:
    """
    Enforce server-level worker-to-tab authorization.
    Fail closed on missing, ambiguous, or mismatched context.
    Never permits worker fallback to 'active_tab'.
    """
    if not worker_id:
        return  # Single-agent foreground session without worker context

    if not target_tab or target_tab == "active_tab":
        raise PermissionError(
            f"Worker '{worker_id}' attempted to access unassigned/foreground tab '{target_tab}'. "
            "Workers must operate exclusively on their explicitly authorized tab."
        )

    if assigned_tab and target_tab != assigned_tab:
        raise PermissionError(
            f"Worker authorization mismatch: worker '{worker_id}' is assigned to tab '{assigned_tab}', "
            f"not target tab '{target_tab}'."
        )



