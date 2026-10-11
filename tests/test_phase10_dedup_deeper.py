import pytest
import os
import shutil
from unittest import mock
import agent_service
from langchain_core.messages import AIMessage
from graph.store import AgentGraphStore

@pytest.mark.asyncio
async def test_dedup_deeper_retrieval(temp_db_dir):
    store = AgentGraphStore(temp_db_dir)
    
    # 500+ chars of filler + secret
    filler = "Filler content. " * 40
    content = filler + "\nTHERE_IS_A_UNIQUE_SECRET_SENTINEL_AT_THE_END_984731"
    
    store.save_artifact("agA", "Project Specification", "report", content)

    with mock.patch("agent_service.graph_store", store):
        # Task triggers on "previously" and mentions "project specification"
        mock_llm = mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        
        call_count = 0
        tool_response = ""
        
        async def side_effect(messages, **kwargs):
            nonlocal call_count, tool_response
            call_count += 1
            
            if call_count == 1:
                # Assert auto context is injected and bounded
                assert "PERSISTENT MEMORY CONTEXT" in messages[1].content
                assert "Project Specification" in messages[1].content
                assert "THERE_IS_A_UNIQUE_SECRET_SENTINEL" not in messages[1].content # because 500 limit
                
                # Explicitly call search_memory
                return AIMessage(content="", tool_calls=[{
                    "name": "search_memory",
                    "args": {"query": "THERE_IS_A_UNIQUE_SECRET_SENTINEL", "domain": "ARTIFACTS"},
                    "id": "call_1"
                }])
            elif call_count == 2:
                # Check tool response
                last_msg = messages[-1]
                assert last_msg.type == "tool"
                tool_response = last_msg.content
                return AIMessage(content="Done", tool_calls=[])

        mock_llm.ainvoke.side_effect = side_effect
        mock_run = mock.MagicMock()
        mock_run.side_effect = Exception("Browser not allowed")
        
        with mock.patch("agent_service.ChatOpenAI", return_value=mock_llm), \
             mock.patch("subprocess.run", return_value=mock_run):
            await agent_service.execute_agent_loop("previously, find the project specification", "http://127.0.0.1:9222", legacy_mode=True)
            
            print("Tool Response:", tool_response)
            assert "THERE_IS_A_UNIQUE_SECRET_SENTINEL" in tool_response

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dedup_deeper_retrieval())
