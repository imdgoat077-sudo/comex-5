import os
import shutil
import tempfile
import logging
import unittest.mock as mock
import pytest
from fastapi.testclient import TestClient

import agent_service
from graph.store import AgentGraphStore

client = TestClient(agent_service.app)


@pytest.fixture
def temp_graph_store():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_p7_obs.db")
    store = AgentGraphStore(db_path)
    yield store, db_path
    shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# GM-038: Health Check Integration Tests
# =====================================================================

def test_health_endpoint_with_graph_enabled(temp_graph_store):
    store, _ = temp_graph_store
    store.create_agent("agent_health_1", "Health task", "RESEARCH")

    with mock.patch("agent_service.graph_store", store):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "cdp_url" in data
        assert "model" in data
        assert "graph_memory" in data
        assert data["graph_memory"]["agent_count"] == 1


def test_health_endpoint_with_graph_disabled():
    with mock.patch("agent_service.graph_store", None):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "graph_memory" not in data


def test_health_endpoint_stats_exception_resilience():
    failing_store = mock.MagicMock(spec=AgentGraphStore)
    failing_store.get_stats.side_effect = Exception("Stats locked")

    with mock.patch("agent_service.graph_store", failing_store):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["graph_memory"]["enabled"] is True
        assert "Stats locked" in data["graph_memory"]["error"]


# =====================================================================
# GM-037: Structured Read-Timing Logging Tests
# =====================================================================

def test_graph_read_debug_timing_logging(temp_graph_store, caplog):
    store, _ = temp_graph_store
    agent_id = "agent_log_1"
    store.create_agent(agent_id, "Log read test", "RESEARCH")

    with caplog.at_level(logging.DEBUG):
        store.find_recent_entities("product")
        store.get_agent_provenance(agent_id)
        store.get_stats()

    debug_logs = [rec.message for rec in caplog.records if rec.levelname == "DEBUG"]
    assert any("Graph read find_recent_entities" in msg for msg in debug_logs)
    assert any("Graph read get_agent_provenance" in msg for msg in debug_logs)
    assert any("Graph read get_stats" in msg for msg in debug_logs)


# =====================================================================
# GM-037: Structured Write-Failure Logging Tests
# =====================================================================

@pytest.mark.asyncio
async def test_graph_write_failure_warning_logs(caplog):
    failing_store = mock.MagicMock(spec=AgentGraphStore)
    failing_store.create_agent.side_effect = Exception("Create fail")
    failing_store.complete_agent.side_effect = Exception("Complete fail")

    mock_llm = mock.MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = mock.AsyncMock(return_value=mock.MagicMock(content="Finished.", tool_calls=[]))

    with caplog.at_level(logging.WARNING), \
         mock.patch("agent_service.ChatOpenAI", return_value=mock_llm), \
         mock.patch("agent_service.graph_store", failing_store):
        
        # Run agent loop, expect write failure warning logs without throwing
        res = await agent_service.execute_agent_loop("Write fail test", "http://127.0.0.1:9222", trusted_app_context=True)
        assert res == "Finished."

        warning_logs = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
        assert any("Graph: failed to create agent" in msg for msg in warning_logs)
        assert any("Graph: failed to complete agent" in msg for msg in warning_logs)
