"""
Tests for the orchestrator's shared team context, final-answer synthesis, visible-tab hooks,
and Stop interrupting running workers.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nervous.action_policy import ServerTabLeaseRegistry, ServerTabTargetRegistry
from services.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorTask,
    RunStatus,
    TaskStatus,
    WorkerResult,
    build_worker_brief,
)
from services.tool_service import ComexToolService


def _orchestrator(**kwargs) -> MultiAgentOrchestrator:
    return MultiAgentOrchestrator(
        tool_service=MagicMock(spec=ComexToolService),
        lease_registry=ServerTabLeaseRegistry(),
        target_registry=ServerTabTargetRegistry(),
        **kwargs,
    )


def _llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=reply))
    return llm


def _task(task_id: str, findings=None, error=None, sources=None) -> OrchestratorTask:
    async def fn(ctx, svc):
        if error:
            return WorkerResult(findings=None, success=False, error=error)
        return WorkerResult(findings=findings, source_urls=sources or [], success=True)
    return OrchestratorTask(id=task_id, description=f"Research {task_id}", task_fn=fn)


def test_worker_brief_contains_goal_own_subtask_and_siblings():
    a = OrchestratorTask(id="a", description="Price of the Pixel 8", url="https://store.google.com")
    b = OrchestratorTask(id="b", description="Price of the iPhone 15")
    brief = build_worker_brief("Compare Pixel 8 and iPhone 15 prices", a, [a, b])

    assert "Compare Pixel 8 and iPhone 15 prices" in brief
    assert "YOUR SUBTASK: Price of the Pixel 8" in brief
    assert "already open at https://store.google.com" in brief
    assert "- Price of the iPhone 15" in brief
    assert "one of 2 browser agents" in brief


async def test_final_answer_compares_all_findings_with_one_llm_call():
    llm = _llm("Pixel 8 is cheaper (₹59,999 vs ₹69,900).")
    orch = _orchestrator(llm=llm)
    run = orch.create_run("Compare prices", [
        _task("pixel", findings="Pixel 8: ₹59,999", sources=["https://store.google.com"]),
        _task("iphone", findings="iPhone 15: ₹69,900"),
    ])
    events = []

    async def on_event(kind, data):
        events.append(kind)

    result = await orch.execute_run(run, on_event=on_event)

    assert result.final_answer == "Pixel 8 is cheaper (₹59,999 vs ₹69,900)."
    llm.ainvoke.assert_awaited_once()
    prompt = llm.ainvoke.await_args.args[0][1]["content"]
    assert "Pixel 8: ₹59,999" in prompt and "iPhone 15: ₹69,900" in prompt
    assert "https://store.google.com" in prompt
    assert "[UNTRUSTED DATA START]" in prompt
    assert events[-1] == "synthesis_started"


async def test_final_answer_reports_failed_agents_to_synthesis():
    llm = _llm("Only the Pixel price was found.")
    orch = _orchestrator(llm=llm)
    run = orch.create_run("Compare prices", [
        _task("pixel", findings="Pixel 8: ₹59,999"),
        _task("iphone", error="Page blocked by captcha"),
    ])
    result = await orch.execute_run(run)

    prompt = llm.ainvoke.await_args.args[0][1]["content"]
    assert "Page blocked by captcha" in prompt
    assert result.final_answer == "Only the Pixel price was found."


async def test_single_agent_run_skips_synthesis_llm_call():
    llm = _llm("unused")
    orch = _orchestrator(llm=llm)
    run = orch.create_run("Top HN story", [_task("hn", findings="Top story: X")])
    result = await orch.execute_run(run)

    assert result.final_answer == "Top story: X"
    llm.ainvoke.assert_not_awaited()


async def test_synthesis_failure_falls_back_to_per_agent_summary():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("proxy down"))
    orch = _orchestrator(llm=llm)
    run = orch.create_run("Compare", [_task("a", findings="A"), _task("b", findings="B")])
    result = await orch.execute_run(run)
    assert result.final_answer == result.summary


async def test_each_worker_gets_its_own_tab_which_stays_open_and_is_marked():
    opened, released = [], []

    async def open_tab(tab_id, url, label):
        opened.append((tab_id, url, label))
        return f"target-{tab_id}", f"ws://127.0.0.1:9222/devtools/page/target-{tab_id}"

    async def release_tab(tab_id, status):
        released.append((tab_id, status))

    orch = _orchestrator(tab_provisioner=open_tab, tab_release=release_tab)
    task_a = _task("a", findings="A")
    task_a.url = "https://example.com"
    run = orch.create_run("Compare", [task_a, _task("b", error="boom")])
    await orch.execute_run(run)

    assert len({t[0] for t in opened}) == 2
    assert ("https://example.com") in [t[1] for t in opened]
    assert sorted(s for _, s in released) == ["completed", "failed"]
    # Agent binding is removed once the worker is done
    for tab_id, _, _ in opened:
        assert orch.target_registry.lookup_tab_target(tab_id) is None


async def test_stop_interrupts_a_running_worker_loop():
    started = asyncio.Event()
    finished_naturally = False

    async def slow_worker(ctx, svc):
        nonlocal finished_naturally
        started.set()
        await asyncio.sleep(30)
        finished_naturally = True
        return WorkerResult(findings="late", success=True)

    released = []

    async def release_tab(tab_id, status):
        released.append(status)

    async def open_tab(tab_id, url, label):
        return "t", "ws://127.0.0.1:9222/devtools/page/t"

    orch = _orchestrator(tab_provisioner=open_tab, tab_release=release_tab)
    run = orch.create_run("Slow task", [OrchestratorTask(id="slow", description="slow", task_fn=slow_worker)])
    execution = asyncio.create_task(orch.execute_run(run))
    await asyncio.wait_for(started.wait(), timeout=2)

    orch.cancel_run(run.id)
    result = await asyncio.wait_for(execution, timeout=2)

    assert finished_naturally is False
    assert run.status == RunStatus.CANCELLED
    assert run.tasks["slow"].status == TaskStatus.CANCELLED
    assert released == ["cancelled"]
    assert "stopped" in result.final_answer


# ---------------------------------------------------------------------------
# Follow-up action (e.g. "add the best one to cart"): only with user approval
# ---------------------------------------------------------------------------

def _factory_recording(briefs):
    def factory(brief):
        briefs.append(brief)
        async def fn(ctx, svc):
            if "YOUR TASK (approved by the user)" in brief:
                return WorkerResult(findings="Added Sony WH-CH520 to cart (cart: 1 item, ₹2,990).", success=True)
            return WorkerResult(findings="Sony WH-CH520 ₹2,990 4.4★ https://amazon.in/dp/X", success=True)
        return fn
    return factory


async def test_orchestrator_builds_worker_briefs_via_factory():
    briefs = []
    orch = _orchestrator(worker_factory=_factory_recording(briefs))
    run = orch.create_run("Best headphones under 3k", [
        OrchestratorTask(id="amazon", description="Search Amazon"),
        OrchestratorTask(id="flipkart", description="Search Flipkart"),
    ])
    await orch.execute_run(run)
    assert len(briefs) == 2
    assert all("Best headphones under 3k" in b and "browser_extract" in b for b in briefs)


async def test_follow_up_runs_only_after_approval_and_reports_outcome():
    briefs, asked = [], []

    async def approve(run_id, action):
        asked.append(action)
        return True

    orch = _orchestrator(worker_factory=_factory_recording(briefs), llm=_llm("Best: Sony WH-CH520 (₹2,990)."))
    run = orch.create_run("Best headphones, add best to cart",
                          [OrchestratorTask(id="a", description="Amazon"), OrchestratorTask(id="b", description="Flipkart")],
                          follow_up_action="Add the best headphone to the cart")
    events = []

    async def on_event(kind, data):
        events.append((kind, data))

    result = await orch.execute_run(run, on_event=on_event, request_approval=approve)

    assert asked == ["Add the best headphone to the cart"]
    approval = next(d for k, d in events if k == "approval_required")
    assert approval["proposal"].startswith("Best: Sony WH-CH520")
    action_brief = briefs[-1]
    assert "YOUR TASK (approved by the user): Add the best headphone to the cart" in action_brief
    assert "Best: Sony WH-CH520" in action_brief and "never proceed to checkout" in action_brief
    assert result.follow_up["status"] == "completed"
    assert "Added Sony WH-CH520 to cart" in result.final_answer


async def test_follow_up_is_not_performed_when_declined():
    briefs = []

    async def decline(run_id, action):
        return False

    orch = _orchestrator(worker_factory=_factory_recording(briefs), llm=_llm("Best: X"))
    run = orch.create_run("q", [OrchestratorTask(id="a", description="A"), OrchestratorTask(id="b", description="B")],
                          follow_up_action="Add X to cart")
    result = await orch.execute_run(run, request_approval=decline)

    assert len(briefs) == 2  # no action agent was created
    assert result.follow_up["status"] == "declined"
    assert "Not performed" in result.final_answer


async def test_follow_up_is_never_performed_without_an_approval_channel():
    briefs = []
    orch = _orchestrator(worker_factory=_factory_recording(briefs), llm=_llm("Best: X"))
    run = orch.create_run("q", [OrchestratorTask(id="a", description="A"), OrchestratorTask(id="b", description="B")],
                          follow_up_action="Add X to cart")
    result = await orch.execute_run(run)
    assert len(briefs) == 2
    assert result.follow_up["status"] == "skipped"


async def test_stop_while_waiting_for_approval_cancels_follow_up():
    waiting = asyncio.Event()

    async def never_answers(run_id, action):
        waiting.set()
        await asyncio.sleep(60)
        return True

    briefs = []
    orch = _orchestrator(worker_factory=_factory_recording(briefs), llm=_llm("Best: X"))
    run = orch.create_run("q", [OrchestratorTask(id="a", description="A"), OrchestratorTask(id="b", description="B")],
                          follow_up_action="Add X to cart")
    execution = asyncio.create_task(orch.execute_run(run, request_approval=never_answers))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    orch.cancel_run(run.id)
    result = await asyncio.wait_for(execution, timeout=2)

    assert len(briefs) == 2
    assert result.follow_up["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Empty final replies are no longer silently accepted
# ---------------------------------------------------------------------------

async def _run_loop_with_replies(replies):
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    import agent_service as _as

    @_as.tool
    async def browser_extract(selector: str) -> str:
        """Extract page text."""
        return "Sony WH-CH520 ₹2,990 4.4 out of 5"

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[{"name": "browser_extract", "args": {"selector": "main"}, "id": "c1"}]),
        *[AIMessage(content=r, tool_calls=[]) for r in replies],
    ])
    llm.bind_tools = MagicMock(return_value=llm)
    with patch.object(_as, "ChatOpenAI", return_value=llm), \
         patch.object(_as, "_make_agent_loop_context", return_value=MagicMock()), \
         patch.object(_as, "_build_browser_tools", return_value=[browser_extract]):
        res = await _as.execute_agent_loop(task="find headphones", default_cdp_url="http://127.0.0.1:9222", use_graph=False)
    return res, llm


async def test_empty_final_reply_gets_one_nudge_and_keeps_findings():
    res, llm = await _run_loop_with_replies(["", "Sony WH-CH520 at ₹2,990 (4.4★)."])
    assert str(res) == "Sony WH-CH520 at ₹2,990 (4.4★)."
    assert llm.ainvoke.await_count == 3
    sent = llm.ainvoke.await_args_list[2].args[0]
    assert any("stopped without writing a final answer" in str(m.content) for m in sent)
    assert not res.metadata.get("empty_final_answer")


async def test_still_empty_after_nudge_is_flagged_and_fails_the_worker():
    from unittest.mock import patch
    import agent_service as _as

    res, _ = await _run_loop_with_replies(["", ""])
    assert res.metadata["empty_final_answer"] is True

    ctx = MagicMock(worker_id="w", target_tab_id="t", lease=None)
    with patch.object(_as, "execute_agent_loop", AsyncMock(return_value=res)):
        worker_result = await _as.make_worker_task_fn("brief")(ctx, MagicMock())
    assert worker_result.success is False
    assert "without reporting any findings" in worker_result.error
