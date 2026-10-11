import os
import uuid
import pytest
import kuzu
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from graph.store import AgentGraphStore
from agent_service import (
    generate_procedure_route,
    retrieve_procedures,
    assemble_procedure_context,
    ProcedureRoute,
    execute_agent_loop,
    app
)

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / f"test_graph_{uuid.uuid4().hex}.db")

@pytest.fixture
def store(temp_db_path):
    s = AgentGraphStore(temp_db_path)
    yield s
    # No need to rmtree, tmp_path handles it

@pytest.mark.asyncio
async def test_router_parsing():
    from unittest.mock import AsyncMock
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ProcedureRoute(
        capability="list_record_extraction",
        abstract_query="extract stuff",
        confidence=0.9
    ))
    route = await generate_procedure_route("Get top 3 quotes", mock_llm)
    assert route.capability == "list_record_extraction"
    assert route.abstract_query == "extract stuff"
    assert route.confidence == 0.9

@pytest.mark.asyncio
async def test_router_capability_none():
    from unittest.mock import AsyncMock
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ProcedureRoute(
        capability="none",
        abstract_query="",
        confidence=0.1
    ))
    route = await generate_procedure_route("Tell me a joke", mock_llm)
    assert route.capability == "none"

def test_procedure_only_retrieval_and_unrelated_exclusion(store):
    with patch("agent_service.graph_store", store):
        # 1. Unrelated Artifact
        store.save_artifact("agent1", "Unrelated Report", "report", "Some unrelated extraction task report.")
        # 2. Procedure Artifact
        store.save_artifact("agent1", "Proc", "procedure", "Extract fields from repeated records.")
        # 3. Another unrelated Artifact
        store.save_artifact("agent1", "Another Proc", "document", "Extract fields document.")
        
        candidates = retrieve_procedures("Extract fields", "list_record_extraction")
        assert len(candidates) == 1
        assert candidates[0]["title"] == "Proc"
        assert candidates[0]["score"] > 0.0

def test_assemble_procedure_context_bounds():
    cands = [{"title": "TestProc", "content": "TestContent"}]
    ctx = assemble_procedure_context(cands)
    assert "[AVAILABLE PROCEDURES]" in ctx
    assert "[/AVAILABLE PROCEDURES]" in ctx
    assert "TestContent" in ctx
    assert "DO NOT use the exact CSS selectors" in ctx

def test_context_ignored_if_no_candidates():
    assert assemble_procedure_context([]) == ""

@pytest.mark.asyncio
async def test_full_agent_loop_procedural_retrieval(store):
    # This tests requirements 5, 6, 8, 9, 10, 11
    # We will patch `ChatOpenAI.ainvoke` and `generate_procedure_route` and `graph_store`
    with patch("agent_service.graph_store", store), \
         patch("agent_service.ChatOpenAI") as MockLLM:
        
        # Setup DB with procedure
        store.save_artifact("ag1", "Target Procedure", "procedure", "Magic pattern to extract items.")

        # Setup router mock
        mock_router_resp = ProcedureRoute(capability="list", abstract_query="Magic pattern", confidence=0.9)
        with patch("agent_service.generate_procedure_route", return_value=mock_router_resp):
            
            # Setup agent LLM mock
            from unittest.mock import AsyncMock
            mock_agent = MagicMock()
            MockLLM.return_value.bind_tools.return_value = mock_agent
            from langchain_core.messages import AIMessage
            # The agent ignores the procedure and just completes
            mock_agent.ainvoke = AsyncMock(return_value=AIMessage(content="Task completed, ignoring procedure."))

            # Normal new task WITHOUT memory triggers
            result = await execute_agent_loop("Extract magic items", "http://fake", trusted_app_context=True)

            assert "Task completed" in result

            # Verify that the procedure was injected into the prompt
            call_args = mock_agent.ainvoke.call_args[0][0]
            # [0] is system, [1] is human
            human_msg = call_args[1].content
            
            # 8. Automatic retrieval occurs on a normal new task
            # 9. old trigger NOT required
            assert "[AVAILABLE PROCEDURES]" in human_msg
            assert "Target Procedure" in human_msg
            # 11. generic memory behavior remains unchanged (auto_context empty here)
            assert "[PERSISTENT MEMORY CONTEXT]" not in human_msg

@pytest.mark.asyncio
async def test_low_confidence_rejected(store):
    with patch("agent_service.graph_store", store), \
         patch("agent_service.ChatOpenAI") as MockLLM:
        
        store.save_artifact("ag1", "Target Procedure", "procedure", "Magic pattern to extract items.")
        # Confidence < 0.5
        mock_router_resp = ProcedureRoute(capability="list", abstract_query="Magic pattern", confidence=0.4)
        with patch("agent_service.generate_procedure_route", return_value=mock_router_resp):
            from unittest.mock import AsyncMock
            mock_agent = MagicMock()
            MockLLM.return_value.bind_tools.return_value = mock_agent
            from langchain_core.messages import AIMessage
            mock_agent.ainvoke = AsyncMock(return_value=AIMessage(content="Task completed."))

            await execute_agent_loop("Extract magic items", "http://fake", trusted_app_context=True)
            human_msg = mock_agent.ainvoke.call_args[0][0][1].content
            
            # 5. low-confidence candidate rejected
            assert "[AVAILABLE PROCEDURES]" not in human_msg

@pytest.mark.asyncio
async def test_graph_disabled_behavior():
    # 12. graph-disabled behavior remains graceful
    with patch("agent_service.graph_store", None), \
         patch("agent_service.ChatOpenAI") as MockLLM:
        
        from unittest.mock import AsyncMock
        mock_agent = MagicMock()
        MockLLM.return_value.bind_tools.return_value = mock_agent
        from langchain_core.messages import AIMessage
        mock_agent.ainvoke = AsyncMock(return_value=AIMessage(content="No graph."))

        await execute_agent_loop("Extract items", "http://fake", trusted_app_context=True)
        human_msg = mock_agent.ainvoke.call_args[0][0][1].content
        
        assert "[AVAILABLE PROCEDURES]" not in human_msg
