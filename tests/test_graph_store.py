import time
import pytest
from datetime import datetime, timedelta, timezone
from graph.schema import run_migrations


def test_schema_meta_version(graph_store):
    with graph_store._lock:
        res = graph_store.conn.execute("MATCH (s:SchemaMeta {key: 'schema_version'}) RETURN s.value")
        assert res.has_next()
        assert res.get_next()[0] == "4"

    # Verify idempotency of run_migrations
    with graph_store._lock:
        run_migrations(graph_store.conn)
        res = graph_store.conn.execute("MATCH (s:SchemaMeta {key: 'schema_version'}) RETURN s.value")
        assert res.has_next()
        assert res.get_next()[0] == "4"


def test_all_6_node_and_7_rel_tables_exist(graph_store):
    # Schema v3: 11 node tables, 12 rel tables
    node_tables = ["Agent", "Tab", "Page", "Entity", "ToolCall", "Artifact", "Episode", "UserMemory", "Skill", "Reflection", "SchemaMeta"]
    rel_tables = ["SPAWNED", "EXECUTED", "NAVIGATED_TO", "EXTRACTED", "FOUND_ON",
                  "DEPENDS_ON", "SIMILAR_TO", "CREATED_ARTIFACT", "FOR",
                  "REFLECTED_ON", "DISTILLED_FROM", "LEARNED_FROM"]

    with graph_store._lock:
        for node in node_tables:
            res = graph_store.conn.execute(f"MATCH (n:{node}) RETURN count(n)")
            assert res.has_next()
            count = res.get_next()[0]
            assert isinstance(count, int)

        for rel in rel_tables:
            res = graph_store.conn.execute(f"MATCH ()-[r:{rel}]->() RETURN count(r)")
            assert res.has_next()
            count = res.get_next()[0]
            assert isinstance(count, int)


def test_agent_lifecycle(graph_store):
    agent_id = "agent_test_001"
    graph_store.create_agent(agent_id, task="Search monitors", intent="RESEARCH")
    
    stats = graph_store.get_stats()
    assert stats["agent_count"] == 1

    graph_store.complete_agent(agent_id, status="completed", step_count=5)
    prov = graph_store.get_agent_provenance(agent_id)
    assert prov["id"] == agent_id
    assert prov["status"] == "completed"
    assert prov["task"] == "Search monitors"


def test_parent_agent_depends_on(graph_store):
    parent_id = "agent_parent_001"
    child_id = "agent_child_001"
    graph_store.create_agent(parent_id, task="Parent Task", intent="RESEARCH")
    graph_store.create_agent(child_id, task="Child Task", intent="RESEARCH", parent_agent_id=parent_id)

    with graph_store._lock:
        res = graph_store.conn.execute(
            "MATCH (c:Agent {id: $child})-[:DEPENDS_ON]->(p:Agent {id: $parent}) RETURN count(*)",
            {"child": child_id, "parent": parent_id}
        )
        assert res.has_next()
        assert res.get_next()[0] == 1


def test_record_tool_call(graph_store):
    agent_id = "agent_test_002"
    graph_store.create_agent(agent_id, task="Test tool call", intent="VISIBLE_BROWSER_TASK")
    
    tc_id = graph_store.record_tool_call(
        agent_id, tool_name="run_browser_harness",
        args_json='{"code": "print(1)"}', result_summary="1\n",
        step_index=0, success=True
    )
    assert tc_id is not None
    assert len(tc_id) > 0

    stats = graph_store.get_stats()
    assert stats["toolcall_count"] == 1

    prov = graph_store.get_agent_provenance(agent_id)
    assert len(prov["tool_calls"]) == 1
    assert prov["tool_calls"][0]["tool_name"] == "run_browser_harness"


def test_record_navigation(graph_store):
    agent_id = "agent_test_003"
    graph_store.create_agent(agent_id, task="Navigate test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)

    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon Product")
    assert page_id is not None
    assert page_id.startswith("page_")

    stats = graph_store.get_stats()
    assert stats["page_count"] == 1


def test_upsert_entity_create_new(graph_store):
    agent_id = "agent_test_004"
    graph_store.create_agent(agent_id, task="Upsert entity", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon")

    ent_id = graph_store.upsert_entity(
        tc_id, page_id, etype="product", name="LG 27MR400",
        price=14999.0, currency="INR", attrs={"rating": 4.3}
    )
    assert ent_id is not None

    stats = graph_store.get_stats()
    assert stats["entity_count"] == 1

    entities = graph_store.find_recent_entities("product")
    assert len(entities) == 1
    assert entities[0]["name"] == "LG 27MR400"
    assert entities[0]["price"] == 14999.0


def test_upsert_entity_merge_updates(graph_store):
    agent_id = "agent_test_005"
    graph_store.create_agent(agent_id, task="Merge entity test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon")

    ent_id1 = graph_store.upsert_entity(
        tc_id, page_id, etype="product", name="LG 27MR400",
        price=15000.0, currency="INR", attrs={"rating": 4.2}
    )

    # Upsert same item with updated price and rating
    ent_id2 = graph_store.upsert_entity(
        tc_id, page_id, etype="product", name="LG 27MR400",
        price=14499.0, currency="INR", attrs={"rating": 4.4}
    )

    assert ent_id1 == ent_id2

    stats = graph_store.get_stats()
    assert stats["entity_count"] == 1

    entities = graph_store.find_recent_entities("product")
    assert len(entities) == 1
    assert entities[0]["price"] == 14499.0
    assert entities[0]["attrs"]["rating"] == 4.4


def test_upsert_entity_different_canonical_keys(graph_store):
    agent_id = "agent_test_006"
    graph_store.create_agent(agent_id, task="Multi entity test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon")

    e1 = graph_store.upsert_entity(tc_id, page_id, "product", "LG 27MR400", 14999.0, "INR", {})
    e2 = graph_store.upsert_entity(tc_id, page_id, "product", "Samsung Odyssey G5", 19999.0, "INR", {})

    assert e1 != e2
    stats = graph_store.get_stats()
    assert stats["entity_count"] == 2


def test_upsert_entity_null_nan_price(graph_store):
    agent_id = "agent_test_007_price"
    graph_store.create_agent(agent_id, task="Null price test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B02", title="Amazon")

    e1 = graph_store.upsert_entity(tc_id, page_id, "product", "No Price Item", None, "INR", {})
    e2 = graph_store.upsert_entity(tc_id, page_id, "product", "NaN Price Item", float('nan'), "INR", {})

    entities = graph_store.find_recent_entities("product")
    for e in entities:
        if e["name"] in ["No Price Item", "NaN Price Item"]:
            assert e["price"] is None


def test_find_recent_entities_max_price_filter(graph_store):
    agent_id = "agent_test_007"
    graph_store.create_agent(agent_id, task="Price filter test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon")

    graph_store.upsert_entity(tc_id, page_id, "product", "Cheap Monitor", 9999.0, "INR", {})
    graph_store.upsert_entity(tc_id, page_id, "product", "Mid Monitor", 14999.0, "INR", {})
    graph_store.upsert_entity(tc_id, page_id, "product", "Expensive Monitor", 29999.0, "INR", {})

    under_20k = graph_store.find_recent_entities("product", max_price=20000.0)
    names = [e["name"] for e in under_20k]
    assert len(under_20k) == 2
    assert "Cheap Monitor" in names
    assert "Mid Monitor" in names
    assert "Expensive Monitor" not in names


def test_find_recent_entities_staleness_exclusion(graph_store):
    agent_id = "agent_test_008"
    graph_store.create_agent(agent_id, task="Staleness test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon")

    ent_id = graph_store.upsert_entity(tc_id, page_id, "product", "Old Product", 5000.0, "INR", {})

    # Artificially set last_seen_at to 40 minutes ago
    old_time = datetime.now(timezone.utc) - timedelta(minutes=40)
    with graph_store._lock:
        graph_store.conn.execute(
            "MATCH (e:Entity {id: $id}) SET e.last_seen_at = $old_time",
            {"id": ent_id, "old_time": old_time}
        )

    # By default (since_minutes=30), old item should be excluded
    recent = graph_store.find_recent_entities("product", since_minutes=30)
    assert len(recent) == 0

    # With include_stale=True, it should be present
    stale_included = graph_store.find_recent_entities("product", since_minutes=30, include_stale=True)
    assert len(stale_included) == 1
    assert stale_included[0]["name"] == "Old Product"


def test_get_agent_provenance(graph_store):
    agent_id = "agent_test_009"
    graph_store.create_agent(agent_id, task="Provenance test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", '{"code": "test"}', "OK", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://example.com", title="Example Domain")
    graph_store.upsert_entity(tc_id, page_id, "product", "Example Product", 100.0, "USD", {})

    prov = graph_store.get_agent_provenance(agent_id)
    assert prov["id"] == agent_id
    assert len(prov["tool_calls"]) == 1
    tc = prov["tool_calls"][0]
    assert tc["tool_name"] == "run_browser_harness"
    assert len(tc["pages"]) == 1
    assert tc["pages"][0]["url"] == "https://example.com"
    assert len(tc["entities"]) == 1
    assert tc["entities"][0]["name"] == "Example Product"


def test_get_agent_provenance_not_found(graph_store):
    prov = graph_store.get_agent_provenance("non_existent_agent")
    assert prov == {"error": "Agent not found"}


def test_record_tab(graph_store):
    agent_id = "agent_test_010"
    graph_store.create_agent(agent_id, task="Tab test", intent="RESEARCH")
    graph_store.record_tab("tab-agent-123", is_hidden=True, cdp_ws_url="ws://127.0.0.1:9222/devtools/page/1", spawned_by_agent_id=agent_id)

    stats = graph_store.get_stats()
    assert stats["tab_count"] == 1


def test_similar_to_relationship_creation(graph_store):
    agent_id = "agent_test_011"
    graph_store.create_agent(agent_id, task="Similar to test", intent="RESEARCH")
    tc_id = graph_store.record_tool_call(agent_id, "run_browser_harness", "{}", "ok", 0, True)
    page_id = graph_store.record_navigation(tc_id, url="https://amazon.in/dp/B01", title="Amazon")

    e1 = graph_store.upsert_entity(tc_id, page_id, "product", "Product A", 100.0, "INR", {})
    e2 = graph_store.upsert_entity(tc_id, page_id, "product", "Product B", 120.0, "INR", {})

    with graph_store._lock:
        graph_store.conn.execute(
            "MATCH (e1:Entity {id: $e1}), (e2:Entity {id: $e2}) MERGE (e1)-[:SIMILAR_TO]->(e2)",
            {"e1": e1, "e2": e2}
        )
        res = graph_store.conn.execute(
            "MATCH (e1:Entity {id: $e1})-[:SIMILAR_TO]->(e2:Entity {id: $e2}) RETURN count(*)",
            {"e1": e1, "e2": e2}
        )
        assert res.has_next()
        assert res.get_next()[0] == 1
