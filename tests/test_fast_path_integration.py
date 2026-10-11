import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from agent_service import app, execute_agent_loop

@pytest.mark.asyncio
async def test_fast_path_execution_success(graph_store):
    # Seed an active adapter in graph store
    adapter_id = graph_store.save_site_adapter(
        name="test-quick-search",
        domain="fastsite.com",
        action_intent="search items on fastsite",
        code_template="print('Fast-Path success output')"
    )

    with patch("agent_service.graph_store", graph_store), \
         patch("graph.adapter_runner.verify_dom_fingerprint", return_value=(True, None)), \
         patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout="Found 5 items successfully.",
            stderr=""
        )

        res = await execute_agent_loop("search items on fastsite", "https://fastsite.com", trusted_app_context=True)
        assert "Fast-Path Adapter (`test-quick-search`)" in str(res)
        assert res.metadata["fast_path_executed"] is True
        assert res.metadata["adapter_id"] == adapter_id


@pytest.mark.asyncio
async def test_fast_path_self_healing_fallback(graph_store):
    # Seed an adapter that will fail
    graph_store.save_site_adapter(
        name="test-broken-search",
        domain="brokensite.com",
        action_intent="search on brokensite",
        code_template="js('document.querySelector(\"#old-btn\").click()')"
    )

    mock_llm_inst = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(return_value=AIMessage(content="I repaired the selector and completed the search.", tool_calls=[]))
    mock_llm_inst.bind_tools = MagicMock(return_value=mock_bound)
    mock_llm_inst.ainvoke = AsyncMock(return_value=AIMessage(content="I repaired the selector and completed the search.", tool_calls=[]))
    mock_llm_class = MagicMock(return_value=mock_llm_inst)

    with patch("agent_service.graph_store", graph_store), \
         patch("agent_service.ChatOpenAI", mock_llm_class), \
         patch("graph.adapter_runner.verify_dom_fingerprint", return_value=(True, None)), \
         patch("subprocess.run") as mock_sub:
        # Fast path fails
        mock_sub.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="querySelector('#old-btn') returned null"
        )

        res = await execute_agent_loop("search on brokensite", "https://brokensite.com", trusted_app_context=True)
        assert "I repaired the selector" in str(res)

        # Verify the self-healing notice was passed into HumanMessage
        sent_messages = mock_bound.ainvoke.call_args[0][0]
        human_msg = next(m for m in sent_messages if m.type == "human")
        assert "[SELF-HEALING ADAPTER FAILURE NOTICE]" in human_msg.content


def test_adapter_api_endpoints(graph_store):
    with patch("agent_service.graph_store", graph_store):
        client = TestClient(app)

        # 1. Initially empty
        res = client.get("/agent/adapters")
        assert res.status_code == 200
        assert res.json() == []

        # 2. Save an adapter
        aid = graph_store.save_site_adapter(
            name="api-adapter",
            domain="api.com",
            action_intent="run api task",
            code_template="print('ok')"
        )

        res = client.get("/agent/adapters")
        assert res.status_code == 200
        items = res.json()
        assert len(items) == 1
        assert items[0]["id"] == aid

        # 3. Antipatterns
        graph_store.record_antipattern("api.com", "#bad", "stale_element", "wait 1s")
        res_anti = client.get("/agent/antipatterns?domain=api.com")
        assert res_anti.status_code == 200
        assert len(res_anti.json()) == 1

        # 4. Delete adapter
        res_del = client.delete(f"/agent/adapters/{aid}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"
