import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import AIMessage
from graph.adapter_compiler import interpolate_template, compile_trajectory_to_adapter, extract_parameters_from_task
from graph.adapter_runner import execute_fast_path_adapter, AdapterExecutionError

def test_interpolate_template():
    tmpl = "js(\"input.value = '{{query}}'; button.click();\")"
    res = interpolate_template(tmpl, {"query": "gaming mouse"})
    assert res == "js(\"input.value = 'gaming mouse'; button.click();\")"

    # Multiple placeholders
    multi = "search('{{category}}', max_price={{price}})"
    res_multi = interpolate_template(multi, {"category": "laptops", "price": 50000})
    assert res_multi == "search('laptops', max_price=50000)"


@pytest.mark.asyncio
async def test_compile_trajectory_mock():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="""
    {
      "name": "youtube-play",
      "action_intent": "play video on youtube",
      "code_template": "js(\\\"document.querySelector('#search').value = '{{query}}';\\\")",
      "parameter_schema": {"query": "string"},
      "dom_fingerprint": {"required_selectors": ["#search"]}
    }
    """))

    traces = [
        {"tool": "run_browser_harness", "args": {"code": "js(\"document.querySelector('#search').value = 'lofi';\")"}}
    ]

    adapter = await compile_trajectory_to_adapter("play lofi on youtube", "youtube.com", traces, mock_llm)
    assert adapter is not None
    assert adapter["name"] == "youtube-play"
    assert adapter["domain"] == "youtube.com"
    assert "{{query}}" in adapter["code_template"]


def test_execute_fast_path_error_handling(graph_store):
    adapter = {
        "id": "adp_test123",
        "name": "test-fail",
        "domain": "test.com",
        "code_template": "js(\"document.querySelector('#non-existent').click()\")"
    }

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: querySelector('#non-existent') returned null"
        )
        with pytest.raises(AdapterExecutionError) as exc_info:
            execute_fast_path_adapter(adapter, {}, "http://127.0.0.1:9222", graph_store=graph_store)

        assert "adp_test123" in str(exc_info.value)
        # Verify antipattern was automatically recorded
        anti = graph_store.get_domain_antipatterns("test.com")
        assert len(anti) == 1
        assert anti[0]["failed_selector"] == "#non-existent"
