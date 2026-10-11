"""
Autonomous Trajectory Compiler for Comex Site Adapters.
Transforms successful multi-step browser-harness agent traces into parameterized,
deterministic Fast-Path code templates stored in the Kùzu graph database.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> str:
    """Extract clean JSON object from potentially noisy model response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def interpolate_template(template: str, params: Dict[str, Any]) -> str:
    """Replace {{placeholder}} with sanitized parameter values."""
    res = template
    for k, v in params.items():
        val_str = str(v).replace("\\", "\\\\").replace("'", "\\'")
        res = re.sub(r'\{\{\s*' + re.escape(k) + r'\s*\}\}', val_str, res)
    return res


async def compile_trajectory_to_adapter(
    task: str,
    domain: str,
    tool_traces: List[Dict[str, Any]],
    llm: Any
) -> Optional[Dict[str, Any]]:
    """Synthesize a deterministic, parameterized SiteAdapter from a successful agent run."""
    if not tool_traces:
        return None

    # Collect all executed browser-harness code blocks
    executed_code_blocks = []
    for t in tool_traces:
        if t.get("tool") == "run_browser_harness":
            code = t.get("args", {}).get("code") if isinstance(t.get("args"), dict) else str(t.get("args") or "")
            if code:
                executed_code_blocks.append(code)

    if not executed_code_blocks:
        return None

    prompt = f"""You are an expert browser automation compiler for Comex.
An autonomous agent completed the following task on {domain}:
Task: "{task}"

Here are the code snippets executed during the session:
{"--- Next Action ---".join(executed_code_blocks)}

Your goal:
Compile these separate exploration steps into a single, clean, parameterizable Python script for `browser-harness`.
Rules:
1. Extract variable user inputs (e.g. search keywords, target items, form values) into placeholders format: `{{variable_name}}`.
2. Synthesize resilient fallback selectors (e.g. `document.querySelector('#nav-search-submit-button') || document.querySelector('input[type="submit"]')`).
3. Add a check verifying the DOM element exists before acting.
4. Include structured extraction block (`===COMEX_EXTRACT===`) if extracting data.
5. Do NOT include markdown blocks in your response. Output raw JSON only.

Schema:
{{
  "name": "kebab-case-name",
  "action_intent": "normalized user intent like 'search product on amazon'",
  "code_template": "single python script using js(...) with {{placeholders}}",
  "parameter_schema": {{"placeholder_name": "string"}},
  "dom_fingerprint": {{"required_selectors": ["selector_1", "selector_2"]}}
}}
"""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="You are an autonomous compiler synthesizing deterministic web adapters. Output JSON only."),
            HumanMessage(content=prompt)
        ])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        parsed = json.loads(extract_json_object(content))

        if not parsed.get("code_template") or not parsed.get("name"):
            return None

        return {
            "name": parsed.get("name"),
            "domain": domain.lower(),
            "action_intent": parsed.get("action_intent", task.lower()),
            "code_template": parsed.get("code_template"),
            "parameter_schema": parsed.get("parameter_schema", {}),
            "dom_fingerprint": parsed.get("dom_fingerprint", {})
        }
    except Exception as e:
        logger.warning(f"Failed to compile trajectory to site adapter: {e}")
        return None


async def extract_parameters_from_task(
    task: str,
    parameter_schema: Dict[str, Any],
    llm: Any
) -> Dict[str, Any]:
    """Extract runtime values for an adapter's parameter schema from user task string."""
    if not parameter_schema:
        return {}

    prompt = f"""Given the user task: "{task}"
And the required parameter schema: {json.dumps(parameter_schema)}

Extract the values for each parameter from the user task.
Output raw JSON object only. No markdown.
Example: {{"query": "iphone 16"}}
"""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="Extract parameter dictionary from user prompt. Output JSON only."),
            HumanMessage(content=prompt)
        ])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return json.loads(extract_json_object(content))
    except Exception as e:
        logger.warning(f"Parameter extraction failed, falling back to empty: {e}")
        # Simple regex fallback if single 'query' parameter
        if len(parameter_schema) == 1 and "query" in parameter_schema:
            clean_query = re.sub(r'(search for|find|look up|on \w+)', '', task, flags=re.IGNORECASE).strip()
            return {"query": clean_query}
        return {}
