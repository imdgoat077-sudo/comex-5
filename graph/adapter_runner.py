"""
Deterministic Fast-Path Runner for Comex Site Adapters.
Executes compiled site scripts directly on active or background tabs via browser-harness,
bypassing LLM cognitive loops for sub-second execution.
"""

import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AdapterExecutionError(Exception):
    """Raised when a compiled adapter fails execution, signaling need for self-healing."""
    def __init__(self, adapter_id: str, code: str, error_detail: str, failed_selector: Optional[str] = None):
        super().__init__(f"Adapter {adapter_id} failed: {error_detail}")
        self.adapter_id = adapter_id
        self.code = code
        self.error_detail = error_detail
        self.failed_selector = failed_selector


def verify_dom_fingerprint(dom_fingerprint: Dict[str, Any], default_cdp_url: str) -> Tuple[bool, Optional[str]]:
    """Lightweight DOM pre-flight check to verify required selectors exist before executing Fast Path."""
    required = dom_fingerprint.get("required_selectors", [])
    if not required:
        return True, None

    # Construct lightweight JS test expression
    test_expr = " && ".join([f"Boolean(document.querySelector('{sel.replace('\'', '\\\\\'')}'))" for sel in required])
    try:
        import urllib.request
        import websockets
        import asyncio

        # Run quick evaluate check via CDP
        req = urllib.request.urlopen(f"{default_cdp_url}/json/list" if not default_cdp_url.startswith("ws") else "http://127.0.0.1:9222/json/list", timeout=1.0)
        targets = json.loads(req.read())
        active_target = None
        for t in targets:
            if t.get("type") == "page":
                active_target = t.get("webSocketDebuggerUrl")
                break

        if not active_target:
            return True, None  # Allow runner to try if target discovery is ambiguous

        async def _eval():
            async with websockets.connect(active_target) as ws:
                msg = {
                    "id": 100,
                    "method": "Runtime.evaluate",
                    "params": {"expression": test_expr, "returnByValue": True}
                }
                await ws.send(json.dumps(msg))
                res = json.loads(await ws.recv())
                return res.get("result", {}).get("result", {}).get("value", True)

        is_valid = asyncio.run(_eval())
        return is_valid, None if is_valid else required[0]
    except Exception as e:
        logger.warning(f"DOM fingerprint check skipped or failed: {e}")
        return True, None  # Fallback to runner execution


def execute_fast_path_adapter(
    adapter: Dict[str, Any],
    params: Dict[str, Any],
    default_cdp_url: str,
    graph_store: Optional[Any] = None,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute compiled adapter using browser-harness with sub-second latency."""
    adapter_id = adapter.get("id", "unknown")
    template = adapter.get("code_template", "")

    # 1. Parameter interpolation
    from .adapter_compiler import interpolate_template
    code = interpolate_template(template, params)

    logger.info(f"⚡ Fast-Path: Executing adapter '{adapter.get('name')}' (ID: {adapter_id})")
    started_at = time.monotonic()

    # 2. Configure environment
    env = os.environ.copy()
    if default_cdp_url.startswith("ws"):
        env["BU_CDP_WS"] = default_cdp_url
    else:
        env["BU_CDP_URL"] = default_cdp_url

    bh_path = os.path.expanduser("~/.local/bin/browser-harness")
    try:
        res = subprocess.run(
            [bh_path],
            input=code,
            text=True,
            capture_output=True,
            env=env,
            timeout=20
        )
        elapsed_ms = round((time.monotonic() - started_at) * 1000)
        output = res.stdout + "\n" + res.stderr

        if res.returncode != 0:
            failed_sel = None
            sel_match = re.search(r"querySelector\(['\"]([^'\"]+)['\"]\)", output)
            if sel_match:
                failed_sel = sel_match.group(1)

            if graph_store:
                graph_store.record_adapter_outcome(adapter_id, success=False)
                if failed_sel:
                    graph_store.record_antipattern(
                        domain=adapter.get("domain", "*"),
                        failed_selector=failed_sel,
                        trap_type="selector_not_found",
                        mitigation_hint=f"Failed during Fast-Path adapter {adapter_id}"
                    )

            raise AdapterExecutionError(
                adapter_id=adapter_id,
                code=code,
                error_detail=output[:300],
                failed_selector=failed_sel
            )

        # Record success
        if graph_store:
            graph_store.record_adapter_outcome(adapter_id, success=True)
            if agent_id:
                try:
                    tool_call_id = graph_store.record_tool_call(
                        agent_id=agent_id,
                        tool_name="fast_path_adapter",
                        args_json=json.dumps({"adapter_id": adapter_id, "params": params}),
                        result_summary=output[:200],
                        step_index=0,
                        success=True
                    )
                    # Intercept structured extractions
                    extract_pattern = r'===COMEX_EXTRACT===\s*(.*?)\s*===END_EXTRACT==='
                    for match in re.finditer(extract_pattern, output, re.DOTALL):
                        try:
                            payload = json.loads(match.group(1))
                            url = payload.get("url", "")
                            title = payload.get("title", "")
                            if url:
                                page_id = graph_store.record_navigation(tool_call_id, url, title)
                                for ent in payload.get("entities", []):
                                    graph_store.upsert_entity(
                                        tool_call_id, page_id,
                                        etype=ent.get("type", "unknown"),
                                        name=ent.get("name", ""),
                                        price=ent.get("price"),
                                        currency=ent.get("currency", "INR"),
                                        attrs=ent
                                    )
                        except Exception as pe:
                            logger.warning(f"Fast-Path: error processing extraction payload: {pe}")
                except Exception as ge:
                    logger.warning(f"Fast-Path: failed to record graph metadata: {ge}")

        clean_result = output.strip()
        return {
            "success": True,
            "fast_path": True,
            "adapter_id": adapter_id,
            "adapter_name": adapter.get("name"),
            "durationMs": elapsed_ms,
            "output": clean_result,
            "result": clean_result
        }

    except subprocess.TimeoutExpired:
        if graph_store:
            graph_store.record_adapter_outcome(adapter_id, success=False)
        raise AdapterExecutionError(
            adapter_id=adapter_id,
            code=code,
            error_detail="Timeout exceeded during Fast-Path execution."
        )
    except AdapterExecutionError:
        raise
    except Exception as e:
        if graph_store:
            graph_store.record_adapter_outcome(adapter_id, success=False)
        raise AdapterExecutionError(
            adapter_id=adapter_id,
            code=code,
            error_detail=str(e)
        )
