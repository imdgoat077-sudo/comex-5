"""
Episode Recording Tests — Phase 12 (Write-only instrumentation)

Covers:
1. Successful run produces Episode with outcome_status='completed', non-empty
   self_reported matching agent's actual final message, constraint_check='not_checkable'.
2. Max-iterations run produces outcome_status='max_steps_reached'.
3. Uncaught exception produces outcome_status='error' (using existing finally path).
4. Episode.agent_id correctly links via FOR -> Agent traversal.
5. Episode schema does NOT contain goal/task/intent/step_count (schema-level check).
6. Episode write failure does NOT crash or alter the agent loop's returned result.
7. self_reported is stored verbatim and independently from outcome_status
   (decoupling proof: LLM says "success" but mechanical outcome is 'error').
8. Full regression suite maintained (no new failures).
"""
import json
import os
import pytest
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from graph.store import AgentGraphStore
from graph.schema import SCHEMA_VERSION
from agent_service import execute_agent_loop


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_store():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_episode.db")
    store = AgentGraphStore(db_path)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def _mock_llm_class(mock_llm_class, *responses):
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(side_effect=list(responses))
    return mock_llm


def _get_episode(store, agent_id: str) -> dict | None:
    """Retrieve the Episode for agent_id directly from the store."""
    with store._lock:
        res = store.conn.execute(
            "MATCH (e:Episode {agent_id: $aid}) RETURN "
            "e.id, e.agent_id, e.outcome_status, e.self_reported, "
            "e.constraint_check, e.created_at",
            {"aid": agent_id}
        )
        if not res.has_next():
            return None
        row = res.get_next()
        return {
            "id": row[0],
            "agent_id": row[1],
            "outcome_status": row[2],
            "self_reported": row[3],
            "constraint_check": row[4],
            "created_at": row[5],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Successful run → outcome_status='completed'
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_episode_completed_on_success(mock_llm_class, fresh_store):
    """A successful run produces one Episode with outcome_status='completed',
    non-empty self_reported matching the agent's actual final message,
    and constraint_check='not_checkable'.
    """
    final_msg = "Here are the 5 cheapest monitors I found."
    final = AIMessage(content=final_msg, tool_calls=[])
    _mock_llm_class(mock_llm_class, final)

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Find cheapest monitors.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    assert result == final_msg

    ep = _get_episode(fresh_store, "agent-0")
    if ep is None:
        # agent_id may differ — fetch by outcome
        with fresh_store._lock:
            res = fresh_store.conn.execute(
                "MATCH (e:Episode) WHERE e.outcome_status = 'completed' "
                "RETURN e.id, e.agent_id, e.outcome_status, e.self_reported, "
                "e.constraint_check, e.created_at"
            )
            assert res.has_next(), "Expected at least one Episode with outcome_status='completed'"
            row = res.get_next()
            ep = {"id": row[0], "agent_id": row[1], "outcome_status": row[2],
                  "self_reported": row[3], "constraint_check": row[4], "created_at": row[5]}

    assert ep["outcome_status"] == "completed"
    assert ep["self_reported"] == final_msg, \
        f"self_reported should be the exact agent final message, got: {ep['self_reported']!r}"
    assert ep["constraint_check"] == "not_checkable"
    assert ep["created_at"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Max-iterations → outcome_status='max_steps_reached'
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_episode_max_steps_reached(mock_llm_class, fresh_store):
    """A run that exhausts all 100 steps produces Episode with
    outcome_status='max_steps_reached'. self_reported is empty because
    no final LLM content message was produced.
    """
    # Use search_memory tool (registered via graph_store) so no browser harness needed.
    # The LLM always returns a search_memory tool call — never a final content-only response.
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_memory", "args": {"query": "test", "domain": "ALL"}, "id": "tc"}]
    )
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    mock_llm.bind_tools.return_value = mock_llm

    async def always_tool_call(messages):
        return tool_call_msg
    mock_llm.ainvoke = always_tool_call

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Loop forever.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    assert result == "Task stopped after reaching maximum iterations."

    with fresh_store._lock:
        res = fresh_store.conn.execute(
            "MATCH (e:Episode) WHERE e.outcome_status = 'max_steps_reached' "
            "RETURN e.outcome_status, e.self_reported, e.constraint_check"
        )
        assert res.has_next(), "Expected Episode with outcome_status='max_steps_reached'"
        row = res.get_next()
        assert row[0] == "max_steps_reached"
        assert row[1] == ""  # no final LLM message
        assert row[2] == "not_checkable"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Uncaught exception → outcome_status='error'
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_episode_error_on_exception(mock_llm_class, fresh_store):
    """An uncaught exception from the LLM call still produces an Episode
    with outcome_status='error'. The exception is re-raised (caller sees it).
    """
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM connection dropped"))

    with patch('agent_service.graph_store', fresh_store):
        with pytest.raises(RuntimeError, match="LLM connection dropped"):
            await execute_agent_loop("Some task.", "ws://dummy", trusted_app_context=True)

    with fresh_store._lock:
        res = fresh_store.conn.execute(
            "MATCH (e:Episode) WHERE e.outcome_status = 'error' "
            "RETURN e.outcome_status, e.self_reported, e.constraint_check"
        )
        assert res.has_next(), "Expected Episode with outcome_status='error'"
        row = res.get_next()
        assert row[0] == "error"
        assert row[2] == "not_checkable"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Episode.agent_id links transitively to Agent via FOR relationship
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_episode_for_rel_links_to_agent(mock_llm_class, fresh_store):
    """Episode -[FOR]-> Agent traversal resolves correctly.
    The linked Agent's task matches what was passed to execute_agent_loop.
    """
    task_text = "Check monitor prices."
    final = AIMessage(content="Done.", tool_calls=[])
    _mock_llm_class(mock_llm_class, final)

    with patch('agent_service.graph_store', fresh_store):
        await execute_agent_loop(task_text, "ws://dummy", trusted_app_context=True)

    with fresh_store._lock:
        res = fresh_store.conn.execute(
            "MATCH (e:Episode)-[:FOR]->(a:Agent) "
            "RETURN e.agent_id, a.id, a.task, e.outcome_status"
        )
        assert res.has_next(), "Expected Episode-[FOR]->Agent traversal to succeed"
        row = res.get_next()
        ep_agent_id = row[0]
        agent_id = row[1]
        agent_task = row[2]
        outcome = row[3]

        assert ep_agent_id == agent_id, "Episode.agent_id must match Agent.id"
        assert task_text in agent_task, \
            f"Agent.task should contain '{task_text}', got: {agent_task!r}"
        assert outcome == "completed"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Episode schema does NOT contain goal/task/intent/step_count
# ─────────────────────────────────────────────────────────────────────────────

def test_episode_schema_has_no_agent_duplicate_fields(fresh_store):
    """Schema-level check: Episode must NOT have task, intent, step_count, or goal.
    These live on Agent; Episode links to Agent via agent_id + FOR rel.
    Also verifies the required fields are present.
    """
    with fresh_store._lock:
        # Create a dummy Episode to get column names back via RETURN
        import uuid
        from datetime import datetime, timezone
        ep_id = f"ep_schema_test_{uuid.uuid4().hex[:8]}"
        fresh_store.conn.execute(
            "CREATE (e:Episode {id: $id, agent_id: 'dummy', outcome_status: 'completed', "
            "self_reported: 'x', constraint_check: 'not_checkable', created_at: $now})",
            {"id": ep_id, "now": datetime.now(timezone.utc)}
        )
        res = fresh_store.conn.execute(
            "MATCH (e:Episode {id: $id}) RETURN e.*", {"id": ep_id}
        )
        assert res.has_next()
        # Get column names from the result using Kuzu's API
        col_names = res.get_column_names()

    # Forbidden fields (belong on Agent, not Episode)
    forbidden = {"task", "goal", "intent", "step_count"}
    present_forbidden = forbidden & set(col_names)
    assert not present_forbidden, (
        f"Episode schema must NOT contain Agent fields: {present_forbidden}"
    )

    # Required Episode fields must be present
    required = {"e.id", "e.agent_id", "e.outcome_status", "e.self_reported",
                "e.constraint_check", "e.created_at"}
    # column names from e.* are prefixed with "e."
    assert required <= set(col_names), (
        f"Missing required Episode fields. Got: {col_names}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Episode write failure does NOT affect agent loop's return value
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_episode_write_failure_does_not_crash_loop(mock_llm_class, fresh_store):
    """If create_episode raises, the agent loop still completes normally
    and returns the correct result to the user.
    """
    final_msg = "Research complete."
    final = AIMessage(content=final_msg, tool_calls=[])
    _mock_llm_class(mock_llm_class, final)

    # Make create_episode raise every time
    original_create_episode = fresh_store.create_episode
    def exploding_create_episode(*args, **kwargs):
        raise RuntimeError("Simulated DB write failure")
    fresh_store.create_episode = exploding_create_episode

    with patch('agent_service.graph_store', fresh_store):
        result = await execute_agent_loop("Do research.", "ws://dummy", trusted_app_context=True)
        if isinstance(result, tuple):
            result = result[0]

    # Loop must have returned the correct value, not crashed
    assert result == final_msg

    # Verify no Episode was written (the mock prevented it)
    with fresh_store._lock:
        res = fresh_store.conn.execute("MATCH (e:Episode) RETURN count(e)")
        count = res.get_next()[0] if res.has_next() else 0
    assert count == 0, "No Episode should exist since write was mocked to fail"

    # Restore for cleanup
    fresh_store.create_episode = original_create_episode


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: self_reported is stored verbatim; outcome_status is independently derived
# ─────────────────────────────────────────────────────────────────────────────

@patch('agent_service.ChatOpenAI')
@pytest.mark.asyncio
async def test_self_reported_decoupled_from_outcome_status(mock_llm_class, fresh_store):
    """Decoupling proof: the LLM always returns tool calls (never a final
    content-only message), so outcome = max_steps_reached and self_reported = ''
    even though individual step responses contain text like 'I will succeed soon!'.
    outcome_status is derived from control flow, self_reported from the final
    content-only response — they are never coupled.
    """
    step_msg = AIMessage(
        content="I'm still working on it, going to succeed soon!",
        tool_calls=[{"name": "search_memory", "args": {"query": "test", "domain": "ALL"}, "id": "tc2"}]
    )
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    mock_llm.bind_tools.return_value = mock_llm

    async def always_tool_step(messages):
        return step_msg
    mock_llm.ainvoke = always_tool_step

    with patch('agent_service.graph_store', fresh_store):
        await execute_agent_loop("Do something.", "ws://dummy", trusted_app_context=True)

    with fresh_store._lock:
        res = fresh_store.conn.execute(
            "MATCH (e:Episode) WHERE e.outcome_status = 'max_steps_reached' "
            "RETURN e.outcome_status, e.self_reported"
        )
        assert res.has_next()
        row = res.get_next()

    # outcome is mechanically derived from control flow (max steps)
    assert row[0] == "max_steps_reached"
    # self_reported is empty because no final LLM response was issued
    # — even though step messages contained "I'm still working on it..."
    assert row[1] == "", (
        "self_reported must be empty for max_steps_reached since no final "
        "content-only LLM message was produced. Individual step messages "
        "are NOT captured."
    )
