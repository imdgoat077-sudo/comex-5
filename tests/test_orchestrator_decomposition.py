"""
Tests for LLM-driven task decomposition in services/orchestrator.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.orchestrator import decompose_request_via_llm


def _mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.content = response_text
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


@pytest.mark.asyncio
async def test_decomposition_parses_valid_json_array():
    llm = _mock_llm(
        '[{"id": "t1", "description": "Research specs of X", "url": null},'
        ' {"id": "t2", "description": "Research price of X", "url": "https://example.com"},'
        ' {"id": "t3", "description": "Research reviews of X"}]'
    )
    tasks = (await decompose_request_via_llm("Compare X on specs, price, and reviews", llm, max_tasks=4)).tasks
    assert [t.id for t in tasks] == ["t1", "t2", "t3"]
    assert tasks[1].url == "https://example.com"
    assert tasks[0].url is None


@pytest.mark.asyncio
async def test_decomposition_strips_markdown_fences():
    llm = _mock_llm('```json\n[{"id": "t1", "description": "Do the single task"}]\n```')
    tasks = (await decompose_request_via_llm("Do the thing", llm, max_tasks=4)).tasks
    assert len(tasks) == 1
    assert tasks[0].id == "t1"


@pytest.mark.asyncio
async def test_decomposition_caps_at_max_tasks():
    items = [{"id": f"t{i}", "description": f"Subtask {i}"} for i in range(6)]
    import json
    llm = _mock_llm(json.dumps(items))
    tasks = (await decompose_request_via_llm("Split into many parts", llm, max_tasks=3)).tasks
    assert len(tasks) == 3


@pytest.mark.asyncio
async def test_decomposition_falls_back_on_malformed_json():
    llm = _mock_llm("this is not valid json at all")
    tasks = (await decompose_request_via_llm("Some request", llm, max_tasks=4)).tasks
    assert len(tasks) == 1
    assert tasks[0].id == "task_main"
    assert tasks[0].description == "Some request"


@pytest.mark.asyncio
async def test_decomposition_falls_back_on_llm_exception():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("proxy unavailable"))
    tasks = (await decompose_request_via_llm("Some other request", llm, max_tasks=4)).tasks
    assert len(tasks) == 1
    assert tasks[0].id == "task_main"


@pytest.mark.asyncio
async def test_decomposition_falls_back_on_empty_request():
    llm = _mock_llm("[]")
    tasks = (await decompose_request_via_llm("   ", llm, max_tasks=4)).tasks
    assert len(tasks) == 1
    assert tasks[0].description == ""


async def test_decomposition_extracts_follow_up_action_from_object_format():
    llm = _mock_llm(
        '{"subtasks": [{"id": "amazon", "description": "Find top rated headphones on Amazon"},'
        ' {"id": "flipkart", "description": "Find top rated headphones on Flipkart"}],'
        ' "follow_up_action": "Add the best headphone to the cart"}'
    )
    plan = await decompose_request_via_llm("best headphones on amazon and flipkart, add best to cart", llm)
    assert [t.id for t in plan.tasks] == ["amazon", "flipkart"]
    assert plan.follow_up_action == "Add the best headphone to the cart"


async def test_decomposition_null_follow_up_and_unsafe_ids():
    llm = _mock_llm('{"subtasks": [{"id": "hn top/story!", "description": "Read HN"}], "follow_up_action": null}')
    plan = await decompose_request_via_llm("What is on HN?", llm)
    assert plan.follow_up_action is None
    assert plan.tasks[0].id == "hn_top_story_"
