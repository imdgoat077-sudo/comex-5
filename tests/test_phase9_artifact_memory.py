import pytest
from unittest import mock
import os
import shutil
import asyncio

import agent_service
from graph.store import AgentGraphStore
from langchain_core.messages import AIMessage

@pytest.fixture
def fresh_store(temp_db_dir):
    store = AgentGraphStore(temp_db_dir)
    yield store

def test_distinction_trace_and_artifact(fresh_store):
    agent_id = "agent_trace_artifact_distinction"
    fresh_store.create_agent(agent_id, "unrelated objective", "UNKNOWN")
    
    # Save an artifact without logging any tool call containing its text
    artifact_id = fresh_store.save_artifact(
        agent_id=agent_id,
        title="Programming Test — Exact Questions",
        artifact_type="document",
        content="Question 1: What is 2+2?\nQuestion 2: What is AI?"
    )
    
    # Now try to search for it using search_memory with ALL
    res_all = fresh_store.search_memory("Programming Test", "ALL")
    assert len(res_all) > 0
    assert res_all[0]["memory_type"] == "Artifact"
    assert "Artifact [document]: Programming Test — Exact Questions" in res_all[0]["content_excerpt"]
    assert res_all[0]["provenance"] == agent_id

    # Check search with HISTORY domain (should not find the artifact, since artifact is not history trace)
    # Actually, in Phase 8, HISTORY was Agent+ToolCall. Let's make sure it doesn't return the Artifact.
    res_history = fresh_store.search_memory("Programming Test", "HISTORY")
    assert len(res_history) == 0
    
    # Check that the full content is preserved in the database and accessible via get_artifact
    artifact = fresh_store.get_artifact(artifact_id)
    assert artifact["content"] == "Question 1: What is 2+2?\nQuestion 2: What is AI?"

def test_failure_isolation(fresh_store):
    # If the DB connection raises an exception during save_artifact, it should not crash
    with mock.patch.object(fresh_store.conn, 'execute', side_effect=Exception("Simulated DB failure")):
        res = fresh_store.save_artifact("agent_fail", "Title", "type", "content")
        assert res == ""  # should return empty string on failure

@pytest.mark.asyncio
async def test_e2e_artifact_memory(temp_db_dir):
    # We explicitly patch agent_service.graph_store so execute_agent_loop uses our fresh DB
    store = AgentGraphStore(temp_db_dir)
    
    with mock.patch("agent_service.graph_store", store):
        # --- TASK A ---
        mock_run_a = mock.MagicMock()
        mock_run_a.returncode = 0
        mock_run_a.stdout = ""
        mock_run_a.stderr = ""
        
        # We will mock the LLM for Task A to output a save_artifact tool call, then finish.
        mock_llm_a = mock.MagicMock()
        mock_llm_a.bind_tools.return_value = mock_llm_a
        
        call_count_a = 0
        async def side_effect_a(messages, **kwargs):
            nonlocal call_count_a
            call_count_a += 1
            if call_count_a == 1:
                # Agent decides to save artifact
                return AIMessage(content="", tool_calls=[{
                    "name": "save_artifact",
                    "args": {
                        "title": "Programming Test Questions",
                        "artifact_type": "report",
                        "content": "SENTINEL_Q1: What is Python?\nSENTINEL_Q2: What is Java?"
                    },
                    "id": "call_1"
                }])
            else:
                return AIMessage(content="I have saved the artifact.", tool_calls=[])
                
        mock_llm_a.ainvoke.side_effect = side_effect_a
        
        with mock.patch("agent_service.ChatOpenAI", return_value=mock_llm_a), \
             mock.patch("subprocess.run", return_value=mock_run_a):
            res_a = await agent_service.execute_agent_loop("Compile the test questions into a durable artifact.", "http://127.0.0.1:9222", legacy_mode=True)
            assert "I have saved the artifact" in res_a

        # Verify artifact is stored
        artifacts = store.conn.execute("MATCH (a:Artifact) RETURN count(a)").get_next()[0]
        assert artifacts == 1
        
        # Verify provenance
        has_edge = store.conn.execute("MATCH (ag:Agent)-[:CREATED_ARTIFACT]->(a:Artifact) RETURN count(a)").get_next()[0]
        assert has_edge == 1

        # --- TASK B ---
        mock_llm_b = mock.MagicMock()
        mock_llm_b.bind_tools.return_value = mock_llm_b
        
        call_count_b = 0
        async def side_effect_b(messages, **kwargs):
            nonlocal call_count_b
            call_count_b += 1
            if call_count_b == 1:
                # Agent searches memory
                return AIMessage(content="", tool_calls=[{
                    "name": "search_memory",
                    "args": {"query": "Programming Test Questions", "domain": "ARTIFACTS"},
                    "id": "call_2"
                }])
            else:
                last_msg = messages[-1]
                assert last_msg.type == "tool", f"Expected tool message, got {last_msg.type}"
                try:
                    import json
                    results = json.loads(last_msg.content)
                except Exception:
                    results = []
                
                if not results:
                    return AIMessage(content="Memory is empty or corrupted.", tool_calls=[])
                
                excerpt = results[0].get("content_excerpt", "")
                if "SENTINEL_Q1" not in excerpt:
                    return AIMessage(content="Sentinel missing.", tool_calls=[])
                
                return AIMessage(content=f"The retrieved questions are: {excerpt}", tool_calls=[])
                
        mock_llm_b.ainvoke.side_effect = side_effect_b
        
        mock_run_b = mock.MagicMock()
        mock_run_b.side_effect = Exception("Browser must not be called in Task B!")
        
        with mock.patch("agent_service.ChatOpenAI", return_value=mock_llm_b), \
             mock.patch("subprocess.run", return_value=mock_run_b):
            res_b = await agent_service.execute_agent_loop("Retrieve the exact questions from the recent test.", "http://127.0.0.1:9222", legacy_mode=True)
            assert "SENTINEL_Q1" in res_b and "SENTINEL_Q2" in res_b

        # --- TASK C (Negative Control) ---
        mock_llm_c = mock.MagicMock()
        mock_llm_c.bind_tools.return_value = mock_llm_c
        
        call_count_c = 0
        async def side_effect_c(messages, **kwargs):
            nonlocal call_count_c
            call_count_c += 1
            if call_count_c == 1:
                # Agent searches memory
                return AIMessage(content="", tool_calls=[{
                    "name": "search_memory",
                    "args": {"query": "Chemistry Test Questions", "domain": "ARTIFACTS"},
                    "id": "call_3"
                }])
            else:
                last_msg = messages[-1]
                assert last_msg.type == "tool"
                try:
                    import json
                    results = json.loads(last_msg.content)
                except Exception:
                    results = []
                
                if not results:
                    return AIMessage(content="The requested artifact is unavailable.", tool_calls=[])
                
                excerpt = results[0].get("content_excerpt", "")
                if "Chemistry" not in excerpt:
                    # LLM determines the retrieved document is irrelevant to the request
                    return AIMessage(content="The requested artifact is unavailable.", tool_calls=[])
                else:
                    return AIMessage(content="Wait, I found the chemistry test!", tool_calls=[])
                
        mock_llm_c.ainvoke.side_effect = side_effect_c
        
        with mock.patch("agent_service.ChatOpenAI", return_value=mock_llm_c), \
             mock.patch("subprocess.run", return_value=mock_run_b):
            res_c = await agent_service.execute_agent_loop("Retrieve the chemistry questions.", "http://127.0.0.1:9222", legacy_mode=True)
            assert "unavailable" in res_c.lower()
    pass
