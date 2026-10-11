"""
Tests for the agent efficiency work: element refs, find-by-text, combined actions (type+submit,
sequences), plain-language step labels, dialog auto-answering, LLM timeouts, and helper agents.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent_service as _as
from nervous.action_policy import ServerTabLeaseRegistry, ServerTabTargetRegistry
from nervous.cdp_session import TabSessionContext, ref_selector
from services.orchestrator import MultiAgentOrchestrator, WorkerResult
from services.tool_service import ComexToolService, ToolResult, _parse_and_validate_sequence


@pytest.fixture(autouse=True)
def _isolate_service_singletons():
    """These tests run the agent loop without a graph store. The shared ComexToolService (and
    orchestrator) deliberately keep the store they were created with, so don't let one created
    here leak into later tests."""
    saved = (_as._comex_tool_service, _as._orchestrator_instance)
    yield
    _as._comex_tool_service, _as._orchestrator_instance = saved


def _tools(svc):
    page_refs = {}
    tools = _as._build_browser_tools(worker_id="w1", tab_id="tab-1", _svc=svc, _svc_ctx=MagicMock(), page_refs=page_refs)
    return {t.name: t for t in tools}, page_refs


def _svc():
    svc = MagicMock()
    svc.click = AsyncMock(return_value=ToolResult.ok(data={"clicked": True}))
    svc.type = AsyncMock(return_value=ToolResult.ok(data={"typed": True}))
    svc.inspect = AsyncMock(return_value=ToolResult.ok(data={
        "title": "Amazon.in", "url": "https://www.amazon.in/s?k=x", "returned": 2, "total_found": 40,
        "elements": [
            {"ref": 7, "kind": "searchbox", "label": "Search Amazon.in", "placeholder": "Search Amazon.in"},
            {"ref": 12, "kind": "button", "label": "Add to Cart"},
        ],
    }))
    svc.find_by_text = AsyncMock(return_value=ToolResult.ok(data={
        "title": "T", "url": "u", "elements": [{"ref": 31, "kind": "button", "label": "Buy Now", "selector": ref_selector(31)}],
    }))
    svc.perform_sequence = AsyncMock(return_value=ToolResult.ok(data={"completed_steps": 2}))
    return svc


# --- 1. Element refs ------------------------------------------------------------------------

async def test_inspect_returns_compact_numbered_listing_and_click_by_ref_uses_its_label():
    svc = _svc()
    tools, page_refs = _tools(svc)

    listing = await tools["browser_inspect"].ainvoke({})
    assert '[12] button "Add to Cart"' in listing
    assert '[7] searchbox "Search Amazon.in"' in listing
    assert "2 of 40 interactive elements returned" in listing
    assert page_refs[12] == "Add to Cart"

    out = await tools["browser_click"].ainvoke({"ref": 12})
    svc.click.assert_awaited_once()
    assert svc.click.await_args.args[1] == '[data-comex-ref="12"]'
    assert out.startswith("Clicked “Add to Cart”.")


async def test_click_by_visible_text_resolves_through_find():
    svc = _svc()
    tools, page_refs = _tools(svc)
    out = await tools["browser_click"].ainvoke({"text": "Buy Now"})
    svc.find_by_text.assert_awaited_once()
    assert svc.click.await_args.args[1] == ref_selector(31)
    assert "Buy Now" in out and page_refs[31] == "Buy Now"


async def test_click_requires_exactly_one_target():
    svc = _svc()
    tools, _ = _tools(svc)
    out = await tools["browser_click"].ainvoke({"ref": 1, "selector": "#x"})
    assert "exactly one" in out
    svc.click.assert_not_awaited()


async def test_stale_ref_error_tells_the_agent_to_inspect_again():
    svc = _svc()
    svc.click = AsyncMock(return_value=ToolResult.fail("OPERATION_FAILED", "Selector '[data-comex-ref=\"9\"]' not found in document"))
    tools, _ = _tools(svc)
    out = await tools["browser_click"].ainvoke({"ref": 9})
    assert "call browser_inspect again" in out


async def test_find_reports_no_match_helpfully():
    svc = _svc()
    svc.find_by_text = AsyncMock(return_value=ToolResult.ok(data={"url": "https://x", "elements": []}))
    tools, _ = _tools(svc)
    out = await tools["browser_find"].ainvoke({"text": "Checkout"})
    assert "No visible element matches" in out


# --- 2. Combined actions ----------------------------------------------------------------------

async def test_type_with_submit_presses_enter_in_one_call():
    svc = _svc()
    tools, _ = _tools(svc)
    out = await tools["browser_type"].ainvoke({"ref": 7, "text": "headphones", "submit": True})
    args, kwargs = svc.type.await_args
    assert args[1:] == ('[data-comex-ref="7"]', "headphones", True) and kwargs == {"submit": True}
    assert "pressed Enter" in out


async def test_sequence_compiles_refs_to_selectors():
    svc = _svc()
    tools, _ = _tools(svc)
    await tools["browser_sequence"].ainvoke({"steps": [
        {"action": "type", "ref": 7, "text": "headphones", "submit": True},
        {"action": "click", "target_text": "Add to Cart"},
    ]})
    steps = svc.perform_sequence.await_args.args[1]
    assert steps[0] == {"action": "type", "selector": '[data-comex-ref="7"]', "text": "headphones", "submit": True}
    assert steps[1] == {"action": "click", "target_text": "Add to Cart"}


def test_sequence_validation_accepts_text_targets_and_submit():
    steps, err = _parse_and_validate_sequence([
        {"action": "type", "target_text": "Search", "text": "x", "submit": True},
        {"action": "click", "selector": "#go"},
    ])
    assert err is None
    assert steps[0].target_text == "Search" and steps[0].submit is True and steps[0].selector is None

    _, err = _parse_and_validate_sequence([{"action": "click", "selector": "#a", "target_text": "A"}])
    assert "exactly one" in err
    _, err = _parse_and_validate_sequence([{"action": "click"}])
    assert "exactly one" in err
    _, err = _parse_and_validate_sequence([{"action": "type", "selector": "#a", "text": "x", "submit": "yes"}])
    assert "submit" in err


async def test_sequence_step_resolves_text_target_when_it_runs():
    svc = ComexToolService(lease_registry=ServerTabLeaseRegistry(), target_registry=ServerTabTargetRegistry())
    session = MagicMock()
    session.find_by_text = AsyncMock(return_value={"elements": [{"ref": 5, "selector": ref_selector(5)}]})
    gate = MagicMock(executed=True, handler_invoked=False, execution_result={"clicked": True})
    svc._revalidate_locked_authority = MagicMock(return_value=None)
    svc._execute_click_unlocked = AsyncMock(return_value=gate)
    steps, _ = _parse_and_validate_sequence([{"action": "click", "target_text": "Next"}])
    res = await svc._execute_step_action(session, steps[0], MagicMock())
    assert res.success
    session.find_by_text.assert_awaited_once_with("Next", max_results=1)
    assert svc._execute_click_unlocked.await_args.args[1] == ref_selector(5)


async def test_executor_type_clears_with_select_all_command_and_can_submit():
    from nervous.executor import SelectorClickExecutor

    async def send(method, params=None):
        if method == "DOM.getBoxModel":
            return {"model": {"border": [0, 0, 10, 0, 10, 10, 0, 10]}}
        return {}

    cdp = MagicMock()
    cdp.send = AsyncMock(side_effect=send)
    executor = SelectorClickExecutor(cdp, MagicMock(), tab_id="t", target_id="t")
    binding = MagicMock(backend_node_id=1, resolved_at_mono=0.0)
    with patch("nervous.executor.validate_binding_integrity", return_value=None):
        result = await executor.execute("type", binding, text="hi", clear_first=True, press_enter=True)
    sent = [c.args for c in cdp.send.await_args_list]
    assert any(m == "Input.dispatchKeyEvent" and p.get("commands") == ["selectAll"] for m, p in sent)
    assert any(m == "Input.dispatchKeyEvent" and p.get("key") == "Enter" for m, p in sent)
    assert result["submitted"] is True


# --- 3. Plain-language labels ---------------------------------------------------------------

def test_step_labels_are_plain_language():
    refs = {12: "Add to Cart"}
    assert _as.describe_step("browser_navigate", {"url": "https://www.flipkart.com/search?q=x"}) == "Opening flipkart.com"
    assert _as.describe_step("browser_click", {"ref": 12}, refs) == "Clicking “Add to Cart”"
    assert _as.describe_step("browser_type", {"text": "headphones", "submit": True}) == "Searching for “headphones”"
    assert _as.describe_step("browser_extract", {}) == "Reading the page"
    assert _as.describe_step("browser_inspect", {}) == "Looking at the page"


# --- 4. Reliability guards -------------------------------------------------------------------

async def test_dialogs_are_answered_safely_and_reported():
    cdp = MagicMock()
    cdp.ws_url = "ws://127.0.0.1:9222/devtools/page/T1"
    cdp.send = AsyncMock(return_value={})
    session = TabSessionContext("tab-1", "T1", cdp)

    session.on_cdp_notification("Page.javascriptDialogOpening", {"type": "confirm", "message": "Delete account?"})
    await asyncio.sleep(0)
    cdp.send.assert_awaited_with("Page.handleJavaScriptDialog", {"accept": False})
    assert session.take_last_dialog() == {"type": "confirm", "message": "Delete account?", "action": "dismissed"}
    assert session.take_last_dialog() is None

    session.on_cdp_notification("Page.javascriptDialogOpening", {"type": "alert", "message": "Saved"})
    await asyncio.sleep(0)
    cdp.send.assert_awaited_with("Page.handleJavaScriptDialog", {"accept": True})


def test_model_clients_have_a_timeout_and_retry():
    llm = _as._make_llm()
    assert llm.request_timeout == _as.LLM_TIMEOUT_S
    assert llm.max_retries == _as.LLM_MAX_RETRIES


# --- 3b/5. Progress events and helper agents ------------------------------------------------

def _orchestrator():
    return MultiAgentOrchestrator(
        tool_service=MagicMock(spec=ComexToolService),
        lease_registry=ServerTabLeaseRegistry(),
        target_registry=ServerTabTargetRegistry(),
    )


async def test_worker_step_labels_reach_the_run_as_task_progress():
    from services.orchestrator import OrchestratorTask

    async def worker(ctx, svc):
        await ctx.metadata["report_progress"]("Opening amazon.in")
        return WorkerResult(findings="ok", success=True)

    orch = _orchestrator()
    run = orch.create_run("q", [OrchestratorTask(id="a", description="A", task_fn=worker)])
    events = []

    async def on_event(kind, data):
        events.append((kind, data.get("label")))

    await orch.execute_run(run, on_event=on_event)
    assert ("task_progress", "Opening amazon.in") in events


async def test_worker_task_fn_forwards_loop_steps():
    reports = []

    async def report(label):
        reports.append(label)

    async def fake_loop(**kwargs):
        await kwargs["event_callback"]("step", {"label": "Reading the page"})
        await kwargs["event_callback"]("thinking", {"detail": "x"})
        return _as.AgentLoopResult("Found it", {"tool_traces": []})

    ctx = MagicMock(worker_id="w", target_tab_id="t", lease=None, metadata={"report_progress": report})
    with patch.object(_as, "execute_agent_loop", side_effect=fake_loop):
        result = await _as.make_worker_task_fn("brief")(ctx, MagicMock())
    assert reports == ["Reading the page"]
    assert result.success and result.findings == "Found it"


def _helper_env(behaviour):
    """Run helpers on a real orchestrator with a fake worker loop (no browser)."""
    orch = _orchestrator()

    def fake_factory(brief, inbox=None):
        async def fn(ctx, svc):
            return await behaviour(brief, inbox, ctx)
        return fn

    return orch, patch.object(_as, "_get_orchestrator", return_value=orch), patch.object(_as, "make_worker_task_fn", fake_factory)


async def test_helpers_start_report_progress_and_return_findings():
    async def behaviour(brief, inbox, ctx):
        await ctx.metadata["report_progress"]("Opening flipkart.com")
        return WorkerResult(findings="Sony WH-CH520 ₹2,990 https://flipkart.com/p/1", success=True)

    events = []

    async def emit(kind, data):
        events.append((kind, data))

    orch, p1, p2 = _helper_env(behaviour)
    with p1, p2:
        helpers = _as.HelperAgentManager(emit)
        started = await helpers.start("Find the best headphones on Flipkart", "https://www.flipkart.com")
        assert started.startswith("Started helper-1")
        result = await helpers.wait(timeout_s=5)

    assert "helper-1 (completed)" in result and "Sony WH-CH520" in result
    assert result.startswith("[UNTRUSTED DATA START]")
    kinds = [k for k, _ in events]
    assert kinds[0] == "helper_started" and "helper_progress" in kinds and kinds[-1] == "helper_finished"
    assert next(d for k, d in events if k == "helper_progress")["label"] == "Opening flipkart.com"


async def test_helper_receives_guidance_and_cap_is_enforced():
    release = asyncio.Event()
    seen = []

    async def behaviour(brief, inbox, ctx):
        await release.wait()
        while not inbox.empty():
            seen.append(inbox.get_nowait())
        return WorkerResult(findings="done", success=True)

    orch, p1, p2 = _helper_env(behaviour)
    with p1, p2:
        helpers = _as.HelperAgentManager(AsyncMock())
        for i in range(_as.MAX_HELPER_AGENTS):
            await helpers.start(f"task {i}")
        assert "already running" in await helpers.start("one too many")
        assert "next step" in await helpers.message("helper-1", "also check delivery dates")
        assert "no helper agent" in await helpers.message("helper-99", "x")
        release.set()
        await helpers.wait(timeout_s=5)
    assert "also check delivery dates" in seen


async def test_stop_and_shutdown_cancel_running_helpers():
    async def behaviour(brief, inbox, ctx):
        await asyncio.sleep(30)
        return WorkerResult(findings="late", success=True)

    orch, p1, p2 = _helper_env(behaviour)
    with p1, p2:
        helpers = _as.HelperAgentManager(AsyncMock())
        await helpers.start("slow 1")
        await helpers.start("slow 2")
        assert helpers.stop("helper-1") == "Stopping helper-1."
        await helpers.shutdown()
        assert "cancelled" in helpers.describe()
        assert "running" not in helpers.describe()


async def test_inbox_messages_are_delivered_into_the_agent_loop():
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Noted.", tool_calls=[]))
    llm.bind_tools = MagicMock(return_value=llm)
    inbox: asyncio.Queue = asyncio.Queue()
    await inbox.put("Focus on wireless models only.")
    with patch.object(_as, "ChatOpenAI", return_value=llm), \
         patch.object(_as, "_make_agent_loop_context", return_value=MagicMock()), \
         patch.object(_as, "_build_browser_tools", return_value=[]):
        await _as.execute_agent_loop(task="find headphones", default_cdp_url="http://127.0.0.1:9222", use_graph=False, inbox=inbox)
    sent = llm.ainvoke.await_args.args[0]
    assert any("Focus on wireless models only." in str(m.content) for m in sent)


async def test_delegation_tools_only_for_top_level_agents():
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok", tool_calls=[]))
    llm.bind_tools = MagicMock(return_value=llm)
    with patch.object(_as, "ChatOpenAI", return_value=llm), \
         patch.object(_as, "_make_agent_loop_context", return_value=MagicMock()), \
         patch.object(_as, "_build_browser_tools", side_effect=lambda **kw: []):
        await _as.execute_agent_loop(task="t", default_cdp_url="http://x", use_graph=False, allow_delegation=True)
        top_level = {t.name for t in llm.bind_tools.call_args.args[0]}
        await _as.execute_agent_loop(task="t", default_cdp_url="http://x", use_graph=False, allow_delegation=False)
        plain = {t.name for t in llm.bind_tools.call_args.args[0]}
    assert {"start_agent", "wait_for_agents", "message_agent", "list_agents", "stop_agent"} <= top_level
    assert not plain & {"start_agent", "wait_for_agents"}
