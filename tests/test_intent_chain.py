import pytest
from pydantic import ValidationError
from agent_service import AgentRequest, execute_agent_loop, graph_store, app, _INTERNAL_APP_TOKEN
from nervous import get_target_registry
from fastapi.testclient import TestClient
from unittest import mock

client = TestClient(app)

def test_agent_request_parsing_with_intent():
    # Valid payload with intent
    payload = {
        "task": "Find monitors",
        "tabId": "tab1",
        "taskId": "task1",
        "intent": "RESEARCH"
    }
    req = AgentRequest(**payload)
    assert req.intent == "RESEARCH"

def test_agent_request_parsing_fallback():
    # Valid payload without intent should fallback to UNKNOWN
    payload = {
        "task": "Find monitors",
        "tabId": "tab1",
        "taskId": "task1"
    }
    req = AgentRequest(**payload)
    assert req.intent == "UNKNOWN"

@pytest.mark.asyncio
async def test_execute_agent_loop_with_intent(monkeypatch):
    mock_store = mock.MagicMock()
    monkeypatch.setattr("agent_service.graph_store", mock_store)
    
    recorded_intent = None
    def mock_create_agent(agent_id, task, intent, parent_agent_id=None):
        nonlocal recorded_intent
        recorded_intent = intent
    
    mock_store.create_agent.side_effect = mock_create_agent

    # Prevent actual LLM/browser execution by mocking ChatOpenAI
    class MockLLM:
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, messages):
            class MockResponse:
                tool_calls = []
                content = "Task finished."
            return MockResponse()
    monkeypatch.setattr("agent_service.ChatOpenAI", lambda **kwargs: MockLLM())

    # Invoke execute_agent_loop with an explicit classifier value
    await execute_agent_loop("Find some monitors", "http://fake", intent="research", trusted_app_context=True)

    # Assert graph_store.create_agent received the exact intent
    assert recorded_intent == "research"

def test_post_agent_run_endpoint_mocked(monkeypatch):
    recorded_intent = None
    async def mock_execute_agent_loop(task, default_cdp_url, intent="UNKNOWN"):
        nonlocal recorded_intent
        recorded_intent = intent
        return "mock result"
        
    monkeypatch.setattr("agent_service.execute_agent_loop", mock_execute_agent_loop)
    monkeypatch.setattr("agent_service._resolve_verified_tab_ws", lambda tab, **kwargs: ("target-tab1", "ws://127.0.0.1:9222/devtools/page/target-tab1"))
    
    # Send with intent
    resp = client.post("/agent/run", json={
        "task": "Find something",
        "tabId": "tab1",
        "taskId": "task1",
        "intent": "VISIBLE_BROWSER_TASK",
        "contextToken": _INTERNAL_APP_TOKEN
    })
    assert resp.status_code == 200
    assert recorded_intent == "VISIBLE_BROWSER_TASK"
    
    # Send without intent (fallback)
    resp2 = client.post("/agent/run", json={
        "task": "Find something else",
        "tabId": "tab1",
        "taskId": "task2",
        "contextToken": _INTERNAL_APP_TOKEN
    })
    assert resp2.status_code == 200
    assert recorded_intent == "UNKNOWN"

def test_legacy_unknown_agent_provenance(graph_store):
    # Directly inject an UNKNOWN agent to test backwards compatibility
    agent_id = "test_legacy_agent"
    graph_store.create_agent(agent_id, "Find legacy monitors", "UNKNOWN")
    
    prov = graph_store.get_agent_provenance(agent_id)
    assert prov.get("id") == agent_id
    assert prov.get("task") == "Find legacy monitors"
    
    # Verify search memory HISTORY domain doesn't break
    results = graph_store.search_memory("legacy monitors", domain="HISTORY")
    assert any(r.get("source_id") == agent_id for r in results)
