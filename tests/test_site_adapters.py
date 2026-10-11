import pytest
from graph.store import AgentGraphStore

def test_site_adapter_crud(graph_store: AgentGraphStore):
    # 1. Save adapter
    template = """
# Search for product on Amazon
js("const input = document.querySelector('#twotabsearchtextbox'); if (input) { input.value = '{{query}}'; input.form.submit(); }")
"""
    adapter_id = graph_store.save_site_adapter(
        name="amazon-search",
        domain="amazon.in",
        action_intent="search product on amazon",
        code_template=template,
        parameter_schema={"query": "string"},
        dom_fingerprint={"root_form": "#nav-searchbar-form"}
    )
    assert adapter_id.startswith("adp_")

    # 2. Retrieve all adapters
    all_adapters = graph_store.get_all_adapters()
    assert len(all_adapters) == 1
    assert all_adapters[0]["name"] == "amazon-search"
    assert all_adapters[0]["domain"] == "amazon.in"
    assert all_adapters[0]["parameter_schema"] == {"query": "string"}

    # 3. Find matching adapter
    matched = graph_store.find_matching_adapter("amazon.in", "search iphone on amazon")
    assert len(matched) == 1
    assert matched[0]["id"] == adapter_id

    # 4. Record successful outcome
    graph_store.record_adapter_outcome(adapter_id, success=True)
    updated = graph_store.get_all_adapters(domain="amazon.in")
    assert updated[0]["success_count"] == 2
    assert updated[0]["status"] == "active"

    # 5. Record failures and check degradation
    graph_store.record_adapter_outcome(adapter_id, success=False)
    graph_store.record_adapter_outcome(adapter_id, success=False)
    graph_store.record_adapter_outcome(adapter_id, success=False)
    degraded = graph_store.get_all_adapters(domain="amazon.in")
    assert degraded[0]["failure_count"] == 3
    assert degraded[0]["status"] == "broken"

    # 6. Delete adapter
    assert graph_store.delete_adapter(adapter_id) is True
    assert len(graph_store.get_all_adapters()) == 0


def test_antipattern_tracking(graph_store: AgentGraphStore):
    # Record failed selector
    ap_id = graph_store.record_antipattern(
        domain="amazon.in",
        failed_selector="#broken-nav-btn",
        trap_type="dynamic_id",
        mitigation_hint="Use [name='field-keywords'] instead"
    )
    assert ap_id.startswith("ap_")

    # Record occurrence again
    graph_store.record_antipattern(
        domain="amazon.in",
        failed_selector="#broken-nav-btn",
        trap_type="dynamic_id",
        mitigation_hint="Use [name='field-keywords'] instead"
    )

    # Fetch domain antipatterns
    patterns = graph_store.get_domain_antipatterns("amazon.in")
    assert len(patterns) == 1
    assert patterns[0]["failed_selector"] == "#broken-nav-btn"
    assert patterns[0]["occurrence_count"] == 2
    assert patterns[0]["trap_type"] == "dynamic_id"
