"""
Phase 11 — Dispatch Validation Guard Regression Tests

Proves that pydantic.ValidationError from .invoke() is caught at the
execute_agent_loop dispatch boundary and returned as a structured
ToolMessage rather than terminating the entire agent loop.

Covers: save_artifact (the real failure), search_memory, read_artifact.
"""
import json
import os
import pytest
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from graph.store import AgentGraphStore
from agent_service import execute_agent_loop


@pytest.fixture
def fresh_store():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_dispatch_validation.db")
    store = AgentGraphStore(db_path)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def _make_mock_llm_class(mock_llm_class, *responses):
    """Wire mock_llm_class → mock_llm with a sequence of ainvoke responses."""
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock()
    mock_llm.ainvoke.side_effect = list(responses)
    return mock_llm


# ─────────────────────────────────────────────────────────────────────────────
# save_artifact — missing required title field
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_save_artifact_missing_title_does_not_crash_loop(mock_llm_class, fresh_store):
    """Real failure reproduction: LLM calls save_artifact without 'title'.
    Loop must survive, return a ToolMessage with field-level error detail,
    and the agent must identify 'title' as the missing field.
    """
    # Step 1: LLM emits save_artifact call with NO title argument
    bad_call = AIMessage(content="", tool_calls=[{
        "name": "save_artifact",
        "args": {"artifact_type": "research", "content": "Some findings."},
        "id": "tc_bad"
    }])
    # Step 2: After receiving the error, LLM produces a final answer
    final = AIMessage(content="I need to retry with a title.", tool_calls=[])

    _make_mock_llm_class(mock_llm_class, bad_call, final)

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Do research.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    # Loop did not crash — it returned the LLM's final content
    assert result == "I need to retry with a title."

    # Inspect the ToolMessage the LLM received for the bad call
    mock_llm = mock_llm_class.return_value
    calls = mock_llm.ainvoke.call_args_list
    # The second ainvoke receives the conversation including the ToolMessage
    second_call_messages = calls[1][0][0]
    tool_msg = next((m for m in second_call_messages
                     if getattr(m, 'tool_call_id', None) == "tc_bad"), None)

    assert tool_msg is not None, "ToolMessage for the bad call must be present"

    payload = json.loads(tool_msg.content)
    assert payload["error"] == "validation_error"
    assert payload["tool"] == "save_artifact"

    # Actionable: 'title' must be named as the offending field
    field_names = [f["field"] for f in payload["fields"]]
    assert "title" in field_names, (
        f"Expected 'title' in validation fields, got: {field_names}"
    )

    # Confirm 'Field required' is reported
    title_error = next(f for f in payload["fields"] if f["field"] == "title")
    assert "required" in title_error["msg"].lower()


@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_save_artifact_valid_call_still_succeeds(mock_llm_class, fresh_store):
    """Valid save_artifact call must still persist the artifact normally."""
    good_call = AIMessage(content="", tool_calls=[{
        "name": "save_artifact",
        "args": {
            "title": "Research Summary",
            "artifact_type": "report",
            "content": "Findings about widgets."
        },
        "id": "tc_good"
    }])
    final = AIMessage(content="Artifact saved.", tool_calls=[])

    _make_mock_llm_class(mock_llm_class, good_call, final)

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Save my research.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    assert result == "Artifact saved."

    # Verify artifact was actually persisted in the store
    mock_llm = mock_llm_class.return_value
    calls = mock_llm.ainvoke.call_args_list
    second_call_messages = calls[1][0][0]
    tool_msg = next((m for m in second_call_messages
                     if getattr(m, 'tool_call_id', None) == "tc_good"), None)

    assert tool_msg is not None
    # A success response contains no "error" key
    assert "validation_error" not in tool_msg.content
    assert "Successfully saved artifact" in tool_msg.content
    assert "Research Summary" in tool_msg.content


# ─────────────────────────────────────────────────────────────────────────────
# search_memory — missing required query field
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_search_memory_missing_query_does_not_crash_loop(mock_llm_class, fresh_store):
    """LLM calls search_memory without required 'query' arg.
    Loop must survive; ToolMessage must identify 'query' as missing.
    """
    bad_call = AIMessage(content="", tool_calls=[{
        "name": "search_memory",
        "args": {"domain": "ALL"},   # missing: query
        "id": "tc_sm_bad"
    }])
    final = AIMessage(content="I'll retry search.", tool_calls=[])

    _make_mock_llm_class(mock_llm_class, bad_call, final)

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Search memory.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    assert result == "I'll retry search."

    mock_llm = mock_llm_class.return_value
    calls = mock_llm.ainvoke.call_args_list
    second_call_messages = calls[1][0][0]
    tool_msg = next((m for m in second_call_messages
                     if getattr(m, 'tool_call_id', None) == "tc_sm_bad"), None)

    assert tool_msg is not None
    payload = json.loads(tool_msg.content)
    assert payload["error"] == "validation_error"
    assert payload["tool"] == "search_memory"

    field_names = [f["field"] for f in payload["fields"]]
    assert "query" in field_names, (
        f"Expected 'query' in validation fields, got: {field_names}"
    )
    query_error = next(f for f in payload["fields"] if f["field"] == "query")
    assert "required" in query_error["msg"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# read_artifact — missing required artifact_id field
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_read_artifact_missing_artifact_id_does_not_crash_loop(mock_llm_class, fresh_store):
    """LLM calls read_artifact without required 'artifact_id' arg.
    Loop must survive; ToolMessage must identify 'artifact_id' as missing.
    """
    bad_call = AIMessage(content="", tool_calls=[{
        "name": "read_artifact",
        "args": {"chunk_index": 0},  # missing: artifact_id
        "id": "tc_ra_bad"
    }])
    final = AIMessage(content="Need the artifact ID.", tool_calls=[])

    _make_mock_llm_class(mock_llm_class, bad_call, final)

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Read artifact.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    assert result == "Need the artifact ID."

    mock_llm = mock_llm_class.return_value
    calls = mock_llm.ainvoke.call_args_list
    second_call_messages = calls[1][0][0]
    tool_msg = next((m for m in second_call_messages
                     if getattr(m, 'tool_call_id', None) == "tc_ra_bad"), None)

    assert tool_msg is not None
    payload = json.loads(tool_msg.content)
    assert payload["error"] == "validation_error"
    assert payload["tool"] == "read_artifact"

    field_names = [f["field"] for f in payload["fields"]]
    assert "artifact_id" in field_names, (
        f"Expected 'artifact_id' in validation fields, got: {field_names}"
    )
    ra_error = next(f for f in payload["fields"] if f["field"] == "artifact_id")
    assert "required" in ra_error["msg"].lower()
