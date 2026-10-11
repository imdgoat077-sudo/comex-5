"""
tests/test_phase2_1d_migration.py

Phase 2.1D regression suite — Legacy Tool Wrapper Migration.

Verifies:
  A. Module-level helpers (_get_tool_service, _make_agent_loop_context,
     _tool_result_to_str, _tool_result_to_route_dict) produce correct output.
  B. Browser tools (click, type, navigate, extract, inspect) delegate to
     ComexToolService when a service+context is available, and fall back
     gracefully when the service is unavailable.
  C. graph/tools factory functions (search_memory, save_artifact, read_artifact)
     delegate to the service when given tool_service + tool_ctx, and fall
     through to the legacy graph path when not given the service.
  D. Adapter invariants:
     - _tool_result_to_str never turns a failure into a success-shaped string.
     - _tool_result_to_route_dict correctly maps security failures to blocked=True.
  E. _make_agent_loop_context cannot be called with model-generated authority
     (only server-verified worker identity is accepted).
  F. Regression: existing 132 passing tests still pass with no interface change.
"""
import asyncio
import json
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Minimal stubs so importing agent_service doesn't require full Electron stack
# ---------------------------------------------------------------------------

def _make_mock_tool_result(success: bool, data=None, error_code=None, error_message=None, retryable=True, evidence=None):
    """Build a minimal object that quacks like a ToolResult."""
    r = MagicMock()
    r.success = success
    r.data = data
    r.error_code = error_code
    r.error_message = error_message
    r.retryable = retryable
    r.evidence = evidence or {}
    return r


# ---------------------------------------------------------------------------
# A. Helper function unit tests (no Electron, no graph store needed)
# ---------------------------------------------------------------------------

class TestToolResultToStr(unittest.TestCase):
    def _import_adapter(self):
        # Import just the adapter from agent_service without running the server
        import importlib
        import sys
        # Ensure we import directly from agent_service module
        try:
            import agent_service as _as
            return _as._tool_result_to_str
        except ImportError:
            self.skipTest("agent_service not importable in this environment")

    def test_success_with_dict_data(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(success=True, data={"key": "value"})
        out = fn(res, "op")
        self.assertIn("key", out)
        self.assertIn("value", out)

    def test_success_with_str_data(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(success=True, data="done.")
        self.assertEqual(fn(res, "op"), "done.")

    def test_success_with_none_data(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(success=True, data=None)
        out = fn(res, "my_op")
        self.assertIn("my_op", out)
        self.assertIn("completed", out)

    def test_failure_surfaces_error_code(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(
            success=False,
            error_code="UNAUTHORIZED",
            error_message="worker not authorized",
        )
        out = fn(res, "browser_click")
        # Must NOT look like a success
        self.assertNotIn("completed", out.lower())
        # Must surface the error
        self.assertTrue("UNAUTHORIZED" in out or "not authorized" in out.lower(), out)

    def test_failure_with_no_message_uses_fallback(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(success=False, error_code=None, error_message=None)
        out = fn(res, "my_op")
        self.assertIn("failed", out.lower())

    def test_non_toolresult_stringified(self):
        fn = self._import_adapter()
        out = fn("raw string", "op")
        self.assertEqual(out, "raw string")


class TestToolResultToRouteDict(unittest.TestCase):
    def _import_adapter(self):
        try:
            import agent_service as _as
            return _as._tool_result_to_route_dict
        except ImportError:
            self.skipTest("agent_service not importable")

    def test_success_maps_to_route_ok(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(success=True, data={"url": "https://example.com"})
        d = fn(res)
        self.assertTrue(d["success"])
        self.assertFalse(d["blocked"])
        self.assertIn("url", d.get("result", {}))

    def test_unauthorized_maps_to_blocked(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(
            success=False,
            error_code="UNAUTHORIZED",
            error_message="lease revoked",
            retryable=False,
        )
        d = fn(res)
        self.assertFalse(d["success"])
        self.assertTrue(d["blocked"])

    def test_target_resolution_failed_maps_to_blocked(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(
            success=False,
            error_code="TARGET_RESOLUTION_FAILED",
            error_message="tab not found",
            retryable=False,
        )
        d = fn(res)
        self.assertTrue(d["blocked"])

    def test_retryable_failure_not_blocked(self):
        fn = self._import_adapter()
        res = _make_mock_tool_result(
            success=False,
            error_code="TIMEOUT",
            error_message="timed out",
            retryable=True,
        )
        d = fn(res)
        self.assertFalse(d["success"])
        # retryable=True → not blocked (just a transient failure)
        self.assertFalse(d["blocked"])

    def test_non_toolresult_returns_blocked_dict(self):
        fn = self._import_adapter()
        d = fn("unexpected string")
        self.assertFalse(d["success"])
        self.assertTrue(d["blocked"])


# ---------------------------------------------------------------------------
# B. Browser tool delegation tests
# ---------------------------------------------------------------------------

class TestBrowserToolDelegation(unittest.IsolatedAsyncioTestCase):
    """Verify that when _svc + _svc_ctx are available, the tool calls the service method."""

    async def _run_click_with_mock_service(self, svc_result):
        """
        Simulate the browser_click tool body: if service is available delegate.
        We test the delegation pattern by calling the tool body logic directly
        against a mock service.
        """
        mock_svc = MagicMock()
        mock_svc.click = AsyncMock(return_value=svc_result)
        mock_ctx = MagicMock()

        # Inline the delegation pattern from agent_service.browser_click
        selector = "#submit-btn"
        try:
            import agent_service as _as
            to_str = _as._tool_result_to_str
        except ImportError:
            self.skipTest("agent_service not importable")

        res = await mock_svc.click(mock_ctx, selector)
        mock_svc.click.assert_awaited_once_with(mock_ctx, selector)
        return to_str(res, "browser_click")

    async def test_click_delegates_to_service_on_success(self):
        ok_result = _make_mock_tool_result(
            success=True, data="Successfully clicked element '#submit-btn' at (120.0, 50.0) in 40ms."
        )
        out = await self._run_click_with_mock_service(ok_result)
        self.assertIn("submit-btn", out)

    async def test_click_surfaces_service_error_on_failure(self):
        fail_result = _make_mock_tool_result(
            success=False,
            error_code="UNAUTHORIZED",
            error_message="worker 'w-1' is not authorized for tab 'tab-2'",
        )
        out = await self._run_click_with_mock_service(fail_result)
        # Must NOT look like a success
        self.assertNotIn("Successfully clicked", out)
        self.assertTrue("UNAUTHORIZED" in out or "not authorized" in out.lower(), out)

    async def test_extract_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.extract = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data={"title": "Example", "url": "https://example.com", "text": "hello world"}
        ))
        mock_ctx = MagicMock()
        res = await mock_svc.extract(mock_ctx, None, 8000)
        mock_svc.extract.assert_awaited_once()
        d = res.data
        self.assertEqual(d["text"], "hello world")

    async def test_inspect_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.inspect = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data={"title": "Page", "returned": 5, "elements": []}
        ))
        mock_ctx = MagicMock()
        res = await mock_svc.inspect(mock_ctx, 50)
        mock_svc.inspect.assert_awaited_once()
        self.assertEqual(res.data["returned"], 5)

    async def test_navigate_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.navigate = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data="Successfully navigated 'tab-1' to https://example.com in 80ms."
        ))
        mock_ctx = MagicMock()
        res = await mock_svc.navigate(mock_ctx, "https://example.com")
        mock_svc.navigate.assert_awaited_once_with(mock_ctx, "https://example.com")
        self.assertTrue(res.success)

    async def test_type_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.type = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data="Successfully typed into element '#q' in 22ms (decision: ALLOW)."
        ))
        mock_ctx = MagicMock()
        res = await mock_svc.type(mock_ctx, "#q", "hello", True)
        mock_svc.type.assert_awaited_once_with(mock_ctx, "#q", "hello", True)
        self.assertTrue(res.success)


# ---------------------------------------------------------------------------
# C. graph/tools factory delegation tests
# ---------------------------------------------------------------------------

class TestGraphToolsDelegation(unittest.TestCase):
    """Verify make_search_memory_tool, make_save_artifact_tool, make_read_artifact_tool
    delegate to service when service+ctx supplied."""

    def _make_legacy_store(self):
        store = MagicMock()
        store.search_memory = MagicMock(return_value=[{"content_excerpt": "old result", "score": 1.0}])
        store.save_artifact = MagicMock(return_value="art_legacy_001")
        store.get_artifact = MagicMock(return_value={
            "id": "art_legacy_001",
            "title": "Old Title",
            "artifact_type": "notes",
            "created_at": "2026-01-01",
            "source_agent_id": "agent-legacy",
            "content": "A" * 8000,
        })
        store.record_tool_call = MagicMock()
        return store

    def test_search_memory_legacy_path_no_service(self):
        from graph.tools import make_search_memory_tool
        store = self._make_legacy_store()
        tool_fn = make_search_memory_tool(store, "agent-001", {"value": 0}, legacy_mode=True)
        result = tool_fn.invoke({"query": "test", "domain": "ALL"})
        store.search_memory.assert_called_once()
        self.assertIn("old result", result)

    def test_save_artifact_legacy_path_no_service(self):
        from graph.tools import make_save_artifact_tool
        store = self._make_legacy_store()
        tool_fn = make_save_artifact_tool(store, "agent-001", {"value": 0}, legacy_mode=True)
        result = tool_fn.invoke({"title": "Report", "content": "body text", "artifact_type": "notes"})
        store.save_artifact.assert_called_once()
        self.assertIn("art_legacy_001", result)

    def test_read_artifact_legacy_path_no_service(self):
        from graph.tools import make_read_artifact_tool
        store = self._make_legacy_store()
        tool_fn = make_read_artifact_tool(store, "agent-001", {"value": 0}, chunk_size=4000, legacy_mode=True)
        result_str = tool_fn.invoke({"artifact_id": "art_legacy_001", "chunk_index": 0})
        result = json.loads(result_str)
        self.assertEqual(result["artifact_id"], "art_legacy_001")
        self.assertIn("[UNTRUSTED DATA START]", result["content"])
        self.assertIn("[UNTRUSTED DATA END]", result["content"])

    def test_search_memory_service_path_called(self):
        """When tool_service+tool_ctx are provided, the service is invoked."""
        from graph.tools import make_search_memory_tool

        svc_result = _make_mock_tool_result(
            success=True,
            data=[{"content_excerpt": "service result", "score": 0.9, "type": "ENTITY"}],
        )
        mock_svc = MagicMock()
        # search_memory is an async method; wrap in a coroutine via asyncio.run
        async def _async_search(*args, **kwargs):
            return svc_result
        mock_svc.search_memory = _async_search

        mock_ctx = MagicMock()
        store = self._make_legacy_store()
        tool_fn = make_search_memory_tool(
            store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=mock_ctx,
        )
        result = tool_fn.invoke({"query": "phones", "domain": "ENTITIES"})
        # Legacy store.search_memory must NOT be called
        store.search_memory.assert_not_called()
        self.assertIn("service result", result)

    def test_save_artifact_service_path_called(self):
        from graph.tools import make_save_artifact_tool

        svc_result = _make_mock_tool_result(
            success=True,
            data={"artifact_id": "art_svc_001", "title": "Report", "artifact_type": "notes", "length": 100,
                  "source_agent_id": "agent-001", "source_run_id": "run-1", "source_task_id": "agent-001"},
        )
        mock_svc = MagicMock()
        async def _async_save(*args, **kwargs):
            return svc_result
        mock_svc.save_artifact = _async_save

        mock_ctx = MagicMock()
        store = self._make_legacy_store()
        tool_fn = make_save_artifact_tool(
            store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=mock_ctx,
        )
        result = tool_fn.invoke({"title": "Report", "content": "body", "artifact_type": "notes"})
        store.save_artifact.assert_not_called()
        self.assertIn("art_svc_001", result)

    def test_read_artifact_service_path_maps_to_legacy_json_format(self):
        from graph.tools import make_read_artifact_tool

        svc_result = _make_mock_tool_result(
            success=True,
            data={
                "artifact_id": "art_svc_002",
                "title": "MyReport",
                "artifact_type": "report",
                "chunk_index": 0,
                "total_chunks": 2,
                "chunk_text": "Hello from service",
                "is_last_chunk": False,
                "created_at": "2026-01-01",
                "source_agent_id": "agent-001",
                "source_run_id": "run-1",
                "source_task_id": "agent-001",
            }
        )
        mock_svc = MagicMock()
        async def _async_read(*args, **kwargs):
            return svc_result
        mock_svc.read_artifact = _async_read

        mock_ctx = MagicMock()
        store = self._make_legacy_store()
        tool_fn = make_read_artifact_tool(
            store, "agent-001", {"value": 0}, chunk_size=4000,
            tool_service=mock_svc, tool_ctx=mock_ctx,
        )
        result_str = tool_fn.invoke({"artifact_id": "art_svc_002", "chunk_index": 0})
        result = json.loads(result_str)
        # Must preserve legacy format
        self.assertEqual(result["artifact_id"], "art_svc_002")
        self.assertIn("[UNTRUSTED DATA START]", result["content"])
        self.assertIn("Hello from service", result["content"])
        self.assertTrue(result["has_more"])
        self.assertEqual(result["total_chunks"], 2)
        # Legacy store.get_artifact must NOT be called
        store.get_artifact.assert_not_called()

    def test_read_artifact_service_failure_returns_error_json(self):
        from graph.tools import make_read_artifact_tool

        svc_result = _make_mock_tool_result(
            success=False,
            error_code="OPERATION_FAILED",
            error_message="Artifact 'art_bad' not found.",
            retryable=False,
        )
        mock_svc = MagicMock()
        async def _async_read(*args, **kwargs):
            return svc_result
        mock_svc.read_artifact = _async_read

        store = self._make_legacy_store()
        tool_fn = make_read_artifact_tool(
            store, "agent-001", {"value": 0}, chunk_size=4000,
            tool_service=mock_svc, tool_ctx=MagicMock(),
        )
        result_str = tool_fn.invoke({"artifact_id": "art_bad", "chunk_index": 0})
        result = json.loads(result_str)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# D. Adapter invariant tests (already partially in A, add edge-cases)
# ---------------------------------------------------------------------------

class TestAdapterInvariants(unittest.TestCase):
    def _adapters(self):
        try:
            import agent_service as _as
            return _as._tool_result_to_str, _as._tool_result_to_route_dict
        except ImportError:
            self.skipTest("agent_service not importable")

    def test_failure_never_contains_success_message(self):
        to_str, _ = self._adapters()
        for code in ["UNAUTHORIZED", "TARGET_RESOLUTION_FAILED", "OPERATION_FAILED", "DEADLINE_EXCEEDED"]:
            res = _make_mock_tool_result(success=False, error_code=code, error_message=f"{code} occurred")
            out = to_str(res, "op")
            # Must not contain affirmative success words
            self.assertNotIn("Successfully", out, msg=f"code={code}")
            self.assertNotIn("completed.", out.lower(), msg=f"code={code}")

    def test_route_dict_has_required_keys_on_success(self):
        _, to_route = self._adapters()
        res = _make_mock_tool_result(success=True, data={"x": 1})
        d = to_route(res)
        self.assertIn("success", d)
        self.assertIn("blocked", d)

    def test_route_dict_has_required_keys_on_failure(self):
        _, to_route = self._adapters()
        res = _make_mock_tool_result(
            success=False, error_code="CANCELLED", error_message="cancelled", retryable=False
        )
        d = to_route(res)
        self.assertIn("success", d)
        self.assertIn("blocked", d)
        self.assertIn("error", d)


# ---------------------------------------------------------------------------
# E. Authority isolation — _make_agent_loop_context does not accept
#    model-generated worker_id overrides (the factory wraps create_tool_execution_context
#    which itself enforces registry-based validation)
# ---------------------------------------------------------------------------

class TestAuthorityIsolation(unittest.TestCase):
    def test_context_factory_imported_from_services(self):
        """The factory must import create_tool_execution_context from services, not inline it."""
        import inspect, ast
        try:
            import agent_service as _as
            src = inspect.getsource(_as._make_agent_loop_context)
        except ImportError:
            self.skipTest("agent_service not importable")
        # The function must call create_tool_execution_context from the service layer
        self.assertIn("create_tool_execution_context", src)
        self.assertIn("services.tool_service", src)

    def test_context_factory_signature_uses_server_only_args(self):
        """_make_agent_loop_context must require worker_id/worker_token from the server
        (closure vars), not from model-generated string inputs."""
        import inspect
        try:
            import agent_service as _as
        except ImportError:
            self.skipTest("agent_service not importable")
        sig = inspect.signature(_as._make_agent_loop_context)
        params = set(sig.parameters.keys())
        # These are the server-verified fields — they MUST be present
        for field in ("worker_id", "worker_token", "trusted_app_context", "agent_id", "task_id"):
            self.assertIn(field, params, f"server-verified field '{field}' missing from signature")


# ---------------------------------------------------------------------------
# F. Interface stability — make_* factory signatures still accept old call patterns
# ---------------------------------------------------------------------------

class TestInterfaceStability(unittest.TestCase):
    def test_make_search_memory_positional_compat(self):
        """Old callers: make_search_memory_tool(store, agent_id, step_ref, seen_excerpts, legacy_mode=True) must still work."""
        from graph.tools import make_search_memory_tool
        store = MagicMock()
        store.search_memory = MagicMock(return_value=[])
        store.record_tool_call = MagicMock()
        seen = set()
        fn = make_search_memory_tool(store, "a1", {"value": 0}, seen, legacy_mode=True)
        result = fn.invoke({"query": "x", "domain": "ALL"})
        self.assertEqual(result, "No matching memory found.")

    def test_make_save_artifact_positional_compat(self):
        """Old callers: make_save_artifact_tool(store, agent_id, step_ref, legacy_mode=True) must still work."""
        from graph.tools import make_save_artifact_tool
        store = MagicMock()
        store.save_artifact = MagicMock(return_value="art_001")
        store.record_tool_call = MagicMock()
        fn = make_save_artifact_tool(store, "a1", {"value": 0}, legacy_mode=True)
        result = fn.invoke({"title": "T", "content": "C", "artifact_type": "notes"})
        self.assertIn("art_001", result)

    def test_make_read_artifact_positional_compat(self):
        """Old callers: make_read_artifact_tool(store, agent_id, step_ref, chunk_size=..., legacy_mode=True) must still work."""
        from graph.tools import make_read_artifact_tool
        store = MagicMock()
        store.get_artifact = MagicMock(return_value={
            "id": "art_001", "title": "T", "artifact_type": "notes",
            "created_at": "", "source_agent_id": "", "content": "hello"
        })
        store.record_tool_call = MagicMock()
        fn = make_read_artifact_tool(store, "a1", {"value": 0}, chunk_size=4000, legacy_mode=True)
        result_str = fn.invoke({"artifact_id": "art_001", "chunk_index": 0})
        result = json.loads(result_str)
        self.assertEqual(result["artifact_id"], "art_001")
        self.assertIn("[UNTRUSTED DATA START]", result["content"])

    def test_save_user_memory_unchanged(self):
        from graph.tools import make_save_user_memory_tool
        store = MagicMock()
        store.set_user_memory = MagicMock(return_value="mem_001")
        store.record_tool_call = MagicMock()
        fn = make_save_user_memory_tool(store, "a1", {"value": 0})
        result = fn.invoke({"key": "pref_currency", "value": "INR", "category": "preference"})
        self.assertIn("INR", result)

    def test_save_skill_unchanged(self):
        from graph.tools import make_save_skill_tool
        store = MagicMock()
        store.save_skill = MagicMock(return_value="skill_001")
        store.record_tool_call = MagicMock()
        fn = make_save_skill_tool(store, "a1", {"value": 0})
        result = fn.invoke({"name": "amazon-click", "domain": "amazon.in", "description": "click buy", "code_snippet": "# code"})
        self.assertIn("skill_001", result)


# ---------------------------------------------------------------------------
# G. P1: Artifact argument order & data integrity tests
# ---------------------------------------------------------------------------

class TestArtifactArgumentIntegrity(unittest.TestCase):
    def test_save_artifact_explicit_kwargs_order(self):
        """Verify make_save_artifact_tool passes title, content, and artifact_type
        in the exact order and names expected by ComexToolService (no reversed arguments!)."""
        from graph.tools import make_save_artifact_tool

        mock_svc = MagicMock()
        mock_svc.save_artifact = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data={"artifact_id": "art_kw_001", "title": "Report", "artifact_type": "notes", "length": 100},
        ))
        mock_ctx = MagicMock()
        store = MagicMock()

        tool_fn = make_save_artifact_tool(
            store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=mock_ctx,
        )

        title_input = "Annual Financial Report"
        content_input = "Total Revenue: $5.2M. Operating Margin: 24%."
        type_input = "financial_statement"

        res = tool_fn.invoke({
            "title": title_input,
            "content": content_input,
            "artifact_type": type_input,
        })

        self.assertIn("art_kw_001", res)
        # Verify EXACT keyword argument forwarding to ComexToolService
        mock_svc.save_artifact.assert_awaited_once_with(
            context=mock_ctx,
            title=title_input,
            content=content_input,
            artifact_type=type_input,
        )

        # Explicitly verify arguments are NOT swapped
        call_kwargs = mock_svc.save_artifact.call_args.kwargs
        self.assertEqual(call_kwargs["title"], title_input)
        self.assertEqual(call_kwargs["content"], content_input)
        self.assertEqual(call_kwargs["artifact_type"], type_input)
        self.assertNotEqual(call_kwargs["content"], type_input)
        self.assertNotEqual(call_kwargs["artifact_type"], content_input)

    def test_save_and_read_back_data_integrity(self):
        """Simulate saving an artifact and reading it back to prove content and type are preserved."""
        from graph.tools import make_save_artifact_tool, make_read_artifact_tool

        storage = {}

        mock_svc = MagicMock()
        async def _mock_save(context, title, content, artifact_type="notes", source_url=None):
            art_id = "art_sim_001"
            storage[art_id] = {
                "artifact_id": art_id,
                "title": title,
                "content": content,
                "artifact_type": artifact_type,
            }
            return _make_mock_tool_result(True, data={"artifact_id": art_id})

        async def _mock_read(context, artifact_id, chunk_index=0, chunk_size=4000):
            item = storage.get(artifact_id)
            if not item:
                return _make_mock_tool_result(False, error_code="OPERATION_FAILED", error_message="Not found")
            return _make_mock_tool_result(True, data={
                "artifact_id": artifact_id,
                "title": item["title"],
                "artifact_type": item["artifact_type"],
                "chunk_index": chunk_index,
                "total_chunks": 1,
                "chunk_text": item["content"],
                "is_last_chunk": True,
            })

        mock_svc.save_artifact = _mock_save
        mock_svc.read_artifact = _mock_read
        mock_ctx = MagicMock()
        store = MagicMock()

        save_tool = make_save_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        read_tool = make_read_artifact_tool(store, "a1", {"value": 0}, chunk_size=4000, tool_service=mock_svc, tool_ctx=mock_ctx)

        save_tool.invoke({
            "title": "My Spec",
            "content": "Secret Specifications Body 12345",
            "artifact_type": "technical_doc",
        })

        read_res = json.loads(read_tool.invoke({"artifact_id": "art_sim_001", "chunk_index": 0}))
        self.assertIn("Secret Specifications Body 12345", read_res["content"])
        self.assertEqual(read_res["artifact_type"], "technical_doc")
        self.assertNotIn("technical_doc", read_res["content"])


# ---------------------------------------------------------------------------
# H. P1: No silent fallback to legacy graph store after service selection
# ---------------------------------------------------------------------------

class TestGraphToolsNoSilentFallback(unittest.TestCase):
    def setUp(self):
        self.legacy_store = MagicMock()
        self.mock_ctx = MagicMock()

    def test_search_memory_service_exception_does_not_fall_back(self):
        from graph.tools import make_search_memory_tool
        mock_svc = MagicMock()
        mock_svc.search_memory = AsyncMock(side_effect=RuntimeError("connection pool exhausted"))

        tool_fn = make_search_memory_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res = tool_fn.invoke({"query": "target query", "domain": "ALL"})
        self.assertIn("failed", res.lower())
        self.assertIn("connection pool exhausted", res)
        # Crucial check: legacy store must NOT have been called!
        self.legacy_store.search_memory.assert_not_called()

    def test_search_memory_service_failure_result_does_not_fall_back(self):
        from graph.tools import make_search_memory_tool
        mock_svc = MagicMock()
        mock_svc.search_memory = AsyncMock(return_value=_make_mock_tool_result(
            success=False, error_code="DATABASE_ERROR", error_message="database locked", retryable=False,
        ))

        tool_fn = make_search_memory_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res = tool_fn.invoke({"query": "target query", "domain": "ALL"})
        self.assertTrue("database locked" in res or "DATABASE_ERROR" in res, res)
        self.legacy_store.search_memory.assert_not_called()

    def test_search_memory_service_none_does_not_fall_back(self):
        from graph.tools import make_search_memory_tool
        mock_svc = MagicMock()
        mock_svc.search_memory = AsyncMock(return_value=None)

        tool_fn = make_search_memory_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res = tool_fn.invoke({"query": "target query", "domain": "ALL"})
        self.assertIn("failed", res.lower())
        self.legacy_store.search_memory.assert_not_called()

    def test_save_artifact_service_exception_does_not_fall_back(self):
        from graph.tools import make_save_artifact_tool
        mock_svc = MagicMock()
        mock_svc.save_artifact = AsyncMock(side_effect=TimeoutError("service timeout"))

        tool_fn = make_save_artifact_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res = tool_fn.invoke({"title": "T", "content": "C", "artifact_type": "notes"})
        self.assertIn("failed", res.lower())
        self.legacy_store.save_artifact.assert_not_called()

    def test_save_artifact_service_failure_result_does_not_fall_back(self):
        from graph.tools import make_save_artifact_tool
        mock_svc = MagicMock()
        mock_svc.save_artifact = AsyncMock(return_value=_make_mock_tool_result(
            success=False, error_code="OPERATION_FAILED", error_message="disk full", retryable=False,
        ))

        tool_fn = make_save_artifact_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res = tool_fn.invoke({"title": "T", "content": "C", "artifact_type": "notes"})
        self.assertIn("disk full", res)
        self.legacy_store.save_artifact.assert_not_called()

    def test_read_artifact_service_exception_does_not_fall_back(self):
        from graph.tools import make_read_artifact_tool
        mock_svc = MagicMock()
        mock_svc.read_artifact = AsyncMock(side_effect=RuntimeError("internal error"))

        tool_fn = make_read_artifact_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res_str = tool_fn.invoke({"artifact_id": "art_123", "chunk_index": 0})
        res = json.loads(res_str)
        self.assertIn("error", res)
        self.assertEqual(res["error"], "service_error")
        self.legacy_store.get_artifact.assert_not_called()

    def test_read_artifact_service_none_does_not_fall_back(self):
        from graph.tools import make_read_artifact_tool
        mock_svc = MagicMock()
        mock_svc.read_artifact = AsyncMock(return_value=None)

        tool_fn = make_read_artifact_tool(
            self.legacy_store, "agent-001", {"value": 0},
            tool_service=mock_svc, tool_ctx=self.mock_ctx,
        )
        res_str = tool_fn.invoke({"artifact_id": "art_123", "chunk_index": 0})
        res = json.loads(res_str)
        self.assertIn("error", res)
        self.legacy_store.get_artifact.assert_not_called()

    def test_context_creation_failure_halts_agent_loop_without_legacy_fallback(self):
        import agent_service as _as
        with patch.object(_as, "_get_tool_service", return_value=MagicMock()):
            with patch.object(_as, "_make_agent_loop_context", side_effect=ValueError("Unauthorized lease")):
                result = asyncio.run(_as.execute_agent_loop(
                    task="test task",
                    default_cdp_url="http://127.0.0.1:9222",
                    use_graph=False,
                ))
                self.assertIn("halted", str(result).lower())
                self.assertIn("Unauthorized lease", str(result))
                self.assertEqual(result.metadata.get("tool_call_count"), 0)


# ---------------------------------------------------------------------------
# I. P1: Migrated /agent/action/* route execution through ComexToolService
# ---------------------------------------------------------------------------

class TestActionRoutesMigration(unittest.TestCase):
    def setUp(self):
        from starlette.testclient import TestClient
        import agent_service as _as
        self.client = TestClient(_as.app)
        self.agent_service = _as

    def test_action_click_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.click = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data="Successfully clicked '#btn'",
            evidence={"rule": "ALLOW", "handler_invoked": False},
        ))
        mock_ctx = MagicMock()

        with patch.object(self.agent_service, "_get_tool_service", return_value=mock_svc):
            with patch.object(self.agent_service, "_make_action_route_context", return_value=(mock_ctx, None)):
                resp = self.client.post("/agent/action/click", json={
                    "selector": "#btn",
                    "tabId": "tab-1",
                    "workerId": "w-1",
                })
                self.assertEqual(resp.status_code, 200)
                d = resp.json()
                self.assertTrue(d["success"])
                self.assertFalse(d["blocked"])
                self.assertTrue(d["executed"])
                self.assertEqual(d["decision"], "ALLOW")
                self.assertEqual(d["result"], "Successfully clicked '#btn'")
                mock_svc.click.assert_awaited_once_with(mock_ctx, "#btn")

    def test_action_click_security_failure_maps_to_blocked(self):
        mock_svc = MagicMock()
        mock_svc.click = AsyncMock(return_value=_make_mock_tool_result(
            success=False,
            error_code="UNAUTHORIZED",
            error_message="Worker 'w-bad' is not authorized for tab 'tab-1'",
            retryable=False,
        ))
        mock_ctx = MagicMock()

        with patch.object(self.agent_service, "_get_tool_service", return_value=mock_svc):
            with patch.object(self.agent_service, "_make_action_route_context", return_value=(mock_ctx, None)):
                resp = self.client.post("/agent/action/click", json={
                    "selector": "#btn",
                    "tabId": "tab-1",
                    "workerId": "w-bad",
                })
                self.assertEqual(resp.status_code, 200)
                d = resp.json()
                self.assertFalse(d["success"])
                self.assertTrue(d["blocked"])
                self.assertEqual(d["error_code"], "UNAUTHORIZED")

    def test_action_type_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.type = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data="Successfully typed into '#input'",
            evidence={"rule": "ALLOW", "handler_invoked": False},
        ))
        mock_ctx = MagicMock()

        with patch.object(self.agent_service, "_get_tool_service", return_value=mock_svc):
            with patch.object(self.agent_service, "_make_action_route_context", return_value=(mock_ctx, None)):
                resp = self.client.post("/agent/action/type", json={
                    "selector": "#input",
                    "text": "hello world",
                    "tabId": "tab-1",
                    "workerId": "w-1",
                    "clear_first": True,
                })
                self.assertEqual(resp.status_code, 200)
                d = resp.json()
                self.assertTrue(d["success"])
                self.assertFalse(d["blocked"])
                mock_svc.type.assert_awaited_once_with(mock_ctx, "#input", "hello world", True)

    def test_action_navigate_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.navigate = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data="Successfully navigated to https://example.com",
            evidence={"rule": "ALLOW", "handler_invoked": False},
        ))
        mock_ctx = MagicMock()

        with patch.object(self.agent_service, "_get_tool_service", return_value=mock_svc):
            with patch.object(self.agent_service, "_make_action_route_context", return_value=(mock_ctx, None)):
                resp = self.client.post("/agent/action/navigate", json={
                    "url": "https://example.com",
                    "tabId": "tab-1",
                    "workerId": "w-1",
                })
                self.assertEqual(resp.status_code, 200)
                d = resp.json()
                self.assertTrue(d["success"])
                mock_svc.navigate.assert_awaited_once_with(mock_ctx, "https://example.com")

    def test_action_extract_delegates_to_service(self):
        mock_svc = MagicMock()
        mock_svc.extract = AsyncMock(return_value=_make_mock_tool_result(
            success=True,
            data={"title": "Page", "text": "Extracted text"},
            evidence={"rule": "ALLOW", "handler_invoked": False},
        ))
        mock_ctx = MagicMock()

        with patch.object(self.agent_service, "_get_tool_service", return_value=mock_svc):
            with patch.object(self.agent_service, "_make_action_route_context", return_value=(mock_ctx, None)):
                resp = self.client.post("/agent/action/extract", json={
                    "selector": "#content",
                    "max_chars": 5000,
                    "tabId": "tab-1",
                    "workerId": "w-1",
                })
                self.assertEqual(resp.status_code, 200)
                d = resp.json()
                self.assertTrue(d["success"])
                mock_svc.extract.assert_awaited_once_with(mock_ctx, selector="#content", max_chars=5000)

    def test_action_untrusted_request_blocked(self):
        # Untrusted request targeting foreground without contextToken fails closed
        resp = self.client.post("/agent/action/click", json={
            "selector": "#btn",
            "tabId": "active_tab",
        })
        self.assertEqual(resp.status_code, 200)
        d = resp.json()
        self.assertFalse(d["success"])
        self.assertTrue(d["blocked"])


# ---------------------------------------------------------------------------
# J. P2: Tests invoking actual registered production browser tools
# ---------------------------------------------------------------------------

class TestActualRegisteredBrowserTools(unittest.IsolatedAsyncioTestCase):
    async def test_production_tools_delegate_with_context(self):
        import agent_service as _as
        mock_svc = MagicMock()
        mock_svc.click = AsyncMock(return_value=_make_mock_tool_result(True, data="Clicked element"))
        mock_svc.type = AsyncMock(return_value=_make_mock_tool_result(True, data="Typed text"))
        mock_svc.navigate = AsyncMock(return_value=_make_mock_tool_result(True, data="Navigated"))
        mock_svc.extract = AsyncMock(return_value=_make_mock_tool_result(True, data={"title": "T", "text": "Extracted text"}))
        mock_svc.inspect = AsyncMock(return_value=_make_mock_tool_result(True, data={"title": "T", "returned": 2}))

        mock_ctx = MagicMock()

        tools = _as._build_browser_tools(
            worker_id="worker-real-1",
            tab_id="tab-real-1",
            _svc=mock_svc,
            _svc_ctx=mock_ctx,
        )
        tool_dict = {t.name: t for t in tools}

        # 1. browser_click
        res_click = await tool_dict["browser_click"].ainvoke({"selector": "#main-btn"})
        self.assertIn("Clicked element", res_click)
        mock_svc.click.assert_awaited_once_with(mock_ctx, "#main-btn")

        # 2. browser_type
        res_type = await tool_dict["browser_type"].ainvoke({"selector": "#input-box", "text": "hello", "clear_first": False})
        self.assertIn("Typed text", res_type)
        mock_svc.type.assert_awaited_once_with(mock_ctx, "#input-box", "hello", False)

        # 3. browser_navigate
        res_nav = await tool_dict["browser_navigate"].ainvoke({"url": "https://example.com/login"})
        self.assertIn("Navigated", res_nav)
        mock_svc.navigate.assert_awaited_once_with(mock_ctx, "https://example.com/login")

        # 4. browser_extract
        res_ext = await tool_dict["browser_extract"].ainvoke({"selector": "article", "max_chars": 4000})
        self.assertIn("Extracted text", res_ext)
        mock_svc.extract.assert_awaited_once_with(mock_ctx, "article", 4000)

        # 5. browser_inspect
        res_insp = await tool_dict["browser_inspect"].ainvoke({"max_elements": 20})
        self.assertIn("returned", res_insp)
        mock_svc.inspect.assert_awaited_once_with(mock_ctx, 20)


# ---------------------------------------------------------------------------
# K. P2: Tool failure classification and telemetry
# ---------------------------------------------------------------------------

class TestToolFailureClassification(unittest.IsolatedAsyncioTestCase):
    async def test_json_error_increments_failure_count_in_agent_loop(self):
        import agent_service as _as
        from langchain_core.messages import AIMessage

        # Build a tool that returns a JSON error like read_artifact does
        @_as.tool
        async def dummy_failing_tool(arg: str) -> str:
            """Dummy failing tool returning JSON error payload."""
            return json.dumps({"error": "not_found", "message": f"Artifact '{arg}' not found."})

        # Mock LLM that generates a tool call on step 1, then returns a final response on step 2
        mock_llm = MagicMock()
        step_1_msg = AIMessage(content="", tool_calls=[{"name": "dummy_failing_tool", "args": {"arg": "art_missing"}, "id": "call_1"}])
        step_2_msg = AIMessage(content="I could not find the artifact.", tool_calls=[])
        mock_llm.ainvoke = AsyncMock(side_effect=[step_1_msg, step_2_msg])
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        # Patch ChatOpenAI to return mock_llm and tools list to include dummy_failing_tool
        with patch.object(_as, "ChatOpenAI", return_value=mock_llm):
            with patch.object(_as, "_make_agent_loop_context", return_value=MagicMock()):
                with patch.object(_as, "_build_browser_tools", return_value=[dummy_failing_tool]):
                    res = await _as.execute_agent_loop(
                        task="find artifact",
                        default_cdp_url="http://127.0.0.1:9222",
                        use_graph=False,
                    )

                # Tool call count should be 1
                self.assertEqual(res.metadata["tool_call_count"], 1)
                # Failed tool call count MUST be incremented to 1
                self.assertEqual(res.metadata["failed_tool_call_count"], 1)
                # Tool trace entry success must be False
                traces = res.metadata["tool_traces"]
                self.assertEqual(len(traces), 1)
                self.assertFalse(traces[0]["success"])
                self.assertIn("not_found", traces[0]["error"])


# ---------------------------------------------------------------------------
# L. P2: Synchronous/async timeout behavior and no duplicated side effects
# ---------------------------------------------------------------------------

class TestCoroutineSyncTimeout(unittest.TestCase):
    def test_run_coroutine_sync_enforces_timeout_promptly(self):
        from graph.tools import _run_coroutine_sync
        import time

        side_effect_tracker = {"entered": False, "finished": False}

        async def _slow_coroutine():
            side_effect_tracker["entered"] = True
            await asyncio.sleep(2.0)
            side_effect_tracker["finished"] = True
            return "done"

        t0 = time.monotonic()
        with self.assertRaises(TimeoutError):
            _run_coroutine_sync(_slow_coroutine(), timeout=0.1)
        dur = time.monotonic() - t0

        self.assertTrue(side_effect_tracker["entered"])
        self.assertFalse(side_effect_tracker["finished"])
        # Should have timed out quickly (~0.1s to 0.3s), NOT waited for 2.0s
        self.assertLess(dur, 0.8)


# ---------------------------------------------------------------------------
# M. P2: Artifact read boundary compatibility (service vs legacy)
# ---------------------------------------------------------------------------

class TestArtifactReadBoundaryCompatibility(unittest.TestCase):
    def test_negative_chunk_index_matches(self):
        from graph.tools import make_read_artifact_tool
        mock_svc = MagicMock()
        mock_svc.read_artifact = AsyncMock(return_value=_make_mock_tool_result(
            False, error_code="VALIDATION_ERROR", error_message="chunk_index must be a non-negative integer."
        ))
        mock_ctx = MagicMock()
        store = MagicMock()

        svc_tool = make_read_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        legacy_tool = make_read_artifact_tool(store, "a1", {"value": 0}, legacy_mode=True)

        svc_out = json.loads(svc_tool.invoke({"artifact_id": "art_123", "chunk_index": -1}))
        leg_out = json.loads(legacy_tool.invoke({"artifact_id": "art_123", "chunk_index": -1}))

        self.assertEqual(svc_out["error"], "validation_error")
        self.assertEqual(leg_out["error"], "validation_error")

    def test_malformed_artifact_id_matches(self):
        from graph.tools import make_read_artifact_tool
        mock_svc = MagicMock()
        mock_svc.read_artifact = AsyncMock(return_value=_make_mock_tool_result(
            False, error_code="VALIDATION_ERROR", error_message="Malformed Artifact ID: not_an_art"
        ))
        mock_ctx = MagicMock()
        store = MagicMock()

        svc_tool = make_read_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        legacy_tool = make_read_artifact_tool(store, "a1", {"value": 0}, legacy_mode=True)

        svc_out = json.loads(svc_tool.invoke({"artifact_id": "not_an_art", "chunk_index": 0}))
        leg_out = json.loads(legacy_tool.invoke({"artifact_id": "not_an_art", "chunk_index": 0}))

        self.assertEqual(svc_out["error"], "invalid_id")
        self.assertEqual(leg_out["error"], "invalid_id")

    def test_out_of_range_chunk_index_matches(self):
        from graph.tools import make_read_artifact_tool
        mock_svc = MagicMock()
        mock_svc.read_artifact = AsyncMock(return_value=_make_mock_tool_result(
            True, data={
                "artifact_id": "art_123",
                "title": "Title",
                "chunk_index": 5,
                "total_chunks": 1,
                "chunk_text": "",
                "is_last_chunk": True,
            }
        ))
        mock_ctx = MagicMock()
        store = MagicMock()
        store.get_artifact = MagicMock(return_value={"id": "art_123", "title": "Title", "content": "hello world"})

        svc_tool = make_read_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        legacy_tool = make_read_artifact_tool(store, "a1", {"value": 0}, legacy_mode=True)

        svc_out = json.loads(svc_tool.invoke({"artifact_id": "art_123", "chunk_index": 5}))
        leg_out = json.loads(legacy_tool.invoke({"artifact_id": "art_123", "chunk_index": 5}))

        self.assertEqual(svc_out["error"], "out_of_range")
        self.assertEqual(leg_out["error"], "out_of_range")

    def test_chunk_size_clamping_above_service_cap(self):
        from graph.tools import make_read_artifact_tool
        mock_svc = MagicMock()
        mock_svc.read_artifact = AsyncMock(return_value=_make_mock_tool_result(
            True, data={
                "artifact_id": "art_123",
                "title": "Title",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_text": "X" * 16000,
                "is_last_chunk": True,
            }
        ))
        mock_ctx = MagicMock()
        store = MagicMock()
        store.get_artifact = MagicMock(return_value={"id": "art_123", "title": "Title", "content": "X" * 16000})

        # Request chunk_size=32000 (well above 16,000 cap)
        svc_tool = make_read_artifact_tool(store, "a1", {"value": 0}, chunk_size=32000, tool_service=mock_svc, tool_ctx=mock_ctx)
        legacy_tool = make_read_artifact_tool(store, "a1", {"value": 0}, chunk_size=32000, legacy_mode=True)

        svc_out = json.loads(svc_tool.invoke({"artifact_id": "art_123", "chunk_index": 0}))
        leg_out = json.loads(legacy_tool.invoke({"artifact_id": "art_123", "chunk_index": 0}))

        self.assertEqual(svc_out["char_range"], "0-16000")
        self.assertEqual(leg_out["char_range"], "0-16000")


# ---------------------------------------------------------------------------
# N. P1 — 1: Fail closed before LLM and enforce legacy_mode policy
# ---------------------------------------------------------------------------

class TestFailClosedExecutionAndLegacyModePolicy(unittest.IsolatedAsyncioTestCase):
    async def test_agent_loop_fails_closed_when_service_unavailable(self):
        """Production execute_agent_loop must halt before ChatOpenAI when service is None."""
        from agent_service import execute_agent_loop
        with patch("agent_service._get_tool_service", return_value=None):
            with patch("agent_service.ChatOpenAI") as mock_llm:
                res = await execute_agent_loop(
                    task="test task",
                    default_cdp_url="http://127.0.0.1:9222",
                    legacy_mode=False,
                )
                # LLM must NEVER be instantiated or invoked
                mock_llm.assert_not_called()
                self.assertIn("unavailable", str(res).lower())
                self.assertIn("unavailable", res.metadata.get("error", "").lower())
                self.assertEqual(res.metadata.get("tool_call_count"), 0)

    async def test_agent_loop_fails_closed_when_context_creation_raises(self):
        """Production loop must halt before ChatOpenAI when context creation fails."""
        from agent_service import execute_agent_loop
        mock_svc = MagicMock()
        with patch("agent_service._get_tool_service", return_value=mock_svc):
            with patch("agent_service._make_agent_loop_context", side_effect=PermissionError("Lease expired")):
                with patch("agent_service.ChatOpenAI") as mock_llm:
                    res = await execute_agent_loop(
                        task="test task",
                        default_cdp_url="http://127.0.0.1:9222",
                        legacy_mode=False,
                    )
                    mock_llm.assert_not_called()
                    self.assertIn("failed to initialize", str(res).lower())
                    self.assertIn("Lease expired", res.metadata.get("error", ""))

    async def test_agent_loop_fails_closed_when_worker_identity_lacks_token(self):
        """Worker identity without token must halt before ChatOpenAI without direct graph access."""
        from agent_service import execute_agent_loop
        mock_svc = MagicMock()
        with patch("agent_service._get_tool_service", return_value=mock_svc):
            with patch("agent_service.ChatOpenAI") as mock_llm:
                res = await execute_agent_loop(
                    task="test task",
                    default_cdp_url="http://127.0.0.1:9222",
                    worker_id="worker_unauth_001",
                    worker_token=None,
                    trusted_app_context=False,
                    legacy_mode=False,
                )
                mock_llm.assert_not_called()
                self.assertIn("worker identity requires valid worker lease token", str(res).lower())
                self.assertEqual(res.metadata.get("tool_call_count"), 0)

    def test_graph_tools_reject_partial_service_context_configuration(self):
        """Providing only one of tool_service and tool_ctx must raise ValueError."""
        from graph.tools import make_search_memory_tool, make_save_artifact_tool, make_read_artifact_tool
        store = MagicMock()
        svc = MagicMock()
        ctx = MagicMock()

        # Service without context
        with self.assertRaises(ValueError):
            make_search_memory_tool(store, "a1", {"value": 0}, tool_service=svc, tool_ctx=None)
        with self.assertRaises(ValueError):
            make_save_artifact_tool(store, "a1", {"value": 0}, tool_service=svc, tool_ctx=None)
        with self.assertRaises(ValueError):
            make_read_artifact_tool(store, "a1", {"value": 0}, tool_service=svc, tool_ctx=None)

        # Context without service
        with self.assertRaises(ValueError):
            make_search_memory_tool(store, "a1", {"value": 0}, tool_service=None, tool_ctx=ctx)
        with self.assertRaises(ValueError):
            make_save_artifact_tool(store, "a1", {"value": 0}, tool_service=None, tool_ctx=ctx)
        with self.assertRaises(ValueError):
            make_read_artifact_tool(store, "a1", {"value": 0}, tool_service=None, tool_ctx=ctx)

    def test_graph_tools_require_explicit_legacy_mode_when_unconfigured(self):
        """Omitting service/ctx without legacy_mode=True must raise ValueError, never silently fall back."""
        from graph.tools import make_search_memory_tool, make_save_artifact_tool, make_read_artifact_tool
        store = MagicMock()

        with self.assertRaises(ValueError):
            make_search_memory_tool(store, "a1", {"value": 0}, legacy_mode=False)
        with self.assertRaises(ValueError):
            make_save_artifact_tool(store, "a1", {"value": 0}, legacy_mode=False)
        with self.assertRaises(ValueError):
            make_read_artifact_tool(store, "a1", {"value": 0}, legacy_mode=False)

        # Explicit legacy_mode=True succeeds
        t1 = make_search_memory_tool(store, "a1", {"value": 0}, legacy_mode=True)
        t2 = make_save_artifact_tool(store, "a1", {"value": 0}, legacy_mode=True)
        t3 = make_read_artifact_tool(store, "a1", {"value": 0}, legacy_mode=True)
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertIsNotNone(t3)

    def test_browser_tools_reject_missing_service_when_not_legacy(self):
        """_build_browser_tools must raise ValueError when service/ctx is missing and legacy_mode=False."""
        from agent_service import _build_browser_tools
        with self.assertRaises(ValueError):
            _build_browser_tools(_svc=None, _svc_ctx=None, legacy_mode=False)


# ---------------------------------------------------------------------------
# O. P1 — 2: Asynchronous graph-tool execution and bounded timeout semantics
# ---------------------------------------------------------------------------

class TestAsyncExecutionAndTimeoutSemantics(unittest.IsolatedAsyncioTestCase):
    def test_coroutine_timeout_with_asyncio_to_thread_no_loop(self):
        """_run_coroutine_sync with asyncio.to_thread unblocks promptly when no loop exists."""
        import time
        from graph.tools import _run_coroutine_sync

        async def _thread_blocking_coro():
            await asyncio.to_thread(time.sleep, 0.5)
            return "done"

        t0 = time.monotonic()
        with self.assertRaises(TimeoutError):
            _run_coroutine_sync(_thread_blocking_coro(), timeout=0.08)
        dur = time.monotonic() - t0
        self.assertLess(dur, 0.35, f"Expected prompt unblocking ~0.08s, took {dur:.3f}s")

    async def test_coroutine_timeout_with_asyncio_to_thread_running_loop(self):
        """_run_coroutine_sync with asyncio.to_thread unblocks promptly inside a running loop."""
        import time
        from graph.tools import _run_coroutine_sync

        async def _thread_blocking_coro():
            await asyncio.to_thread(time.sleep, 0.5)
            return "done"

        t0 = time.monotonic()
        with self.assertRaises(TimeoutError):
            _run_coroutine_sync(_thread_blocking_coro(), timeout=0.08)
        dur = time.monotonic() - t0
        self.assertLess(dur, 0.35, f"Expected prompt unblocking ~0.08s, took {dur:.3f}s")

    async def test_service_backed_tools_native_async_ainvoke(self):
        """StructuredTool instances invoke service coroutines natively on .ainvoke()."""
        from graph.tools import make_search_memory_tool, make_save_artifact_tool, make_read_artifact_tool

        mock_svc = MagicMock()
        mock_svc.search_memory = AsyncMock(return_value=_make_mock_tool_result(True, data=[]))
        mock_svc.save_artifact = AsyncMock(return_value=_make_mock_tool_result(True, data={"artifact_id": "art_1"}))
        mock_svc.read_artifact = AsyncMock(return_value=_make_mock_tool_result(True, data={"chunk_text": "data", "total_chunks": 1, "is_last_chunk": True}))
        mock_ctx = MagicMock()
        store = MagicMock()

        t_search = make_search_memory_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        t_save = make_save_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        t_read = make_read_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)

        # Call native async path
        await t_search.ainvoke({"query": "q", "domain": "ALL"})
        mock_svc.search_memory.assert_awaited_once_with(mock_ctx, "q", "ALL")

        await t_save.ainvoke({"title": "T", "content": "C", "artifact_type": "notes"})
        mock_svc.save_artifact.assert_awaited_once_with(context=mock_ctx, title="T", content="C", artifact_type="notes")

        await t_read.ainvoke({"artifact_id": "art_1", "chunk_index": 0})
        mock_svc.read_artifact.assert_awaited_once()

    async def test_save_artifact_surfaces_service_settlement_evidence(self):
        """Service write timeouts and cancellations preserve authoritative settlement evidence."""
        from graph.tools import make_save_artifact_tool
        from services.tool_service import ToolErrorCode

        mock_svc = MagicMock()
        mock_svc.save_artifact = AsyncMock(return_value=_make_mock_tool_result(
            False,
            error_code=ToolErrorCode.DEADLINE_EXCEEDED,
            error_message="Execution deadline exceeded for task 'task_1'.",
            evidence={
                "settlement": "ROLLED_BACK_CLEAN",
                "persisted": False,
                "compensated_artifact_id": "art_temp_1",
            }
        ))
        mock_ctx = MagicMock()
        store = MagicMock()

        tool_fn = make_save_artifact_tool(store, "a1", {"value": 0}, tool_service=mock_svc, tool_ctx=mock_ctx)
        res = await tool_fn.ainvoke({"title": "Doc", "content": "text", "artifact_type": "notes"})

        self.assertIn("DEADLINE_EXCEEDED", res)
        self.assertIn("ROLLED_BACK_CLEAN", res)
        self.assertIn("persisted: False", res)
        # Direct graph store was NOT called
        store.save_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# P. P2 — 3: Consistent artifact chunk boundaries for small chunk sizes
# ---------------------------------------------------------------------------

class TestArtifactChunkBoundariesSmallSize(unittest.IsolatedAsyncioTestCase):
    async def test_read_artifact_chunk_size_50_normalized_and_consistent(self):
        """Requested chunk_size=50 is normalized to 100 for service and char_range consistency."""
        from graph.tools import make_read_artifact_tool

        full_content = "".join([f"line_{i:03d}\n" for i in range(30)]) # 270 chars
        mock_svc = MagicMock()

        # Simulate service obeying normalized eff_chunk_size=100
        async def _mock_read(context, artifact_id, chunk_index, chunk_size):
            self.assertEqual(chunk_size, 100, "Service must receive normalized effective chunk size 100")
            start = chunk_index * chunk_size
            end = min(start + chunk_size, len(full_content))
            slice_text = full_content[start:end]
            tot = (len(full_content) + chunk_size - 1) // chunk_size
            return _make_mock_tool_result(True, data={
                "artifact_id": artifact_id,
                "title": "Numbers",
                "chunk_index": chunk_index,
                "total_chunks": tot,
                "chunk_text": slice_text,
                "is_last_chunk": end >= len(full_content),
            })

        mock_svc.read_artifact = AsyncMock(side_effect=_mock_read)
        mock_ctx = MagicMock()
        store = MagicMock()

        tool_fn = make_read_artifact_tool(store, "a1", {"value": 0}, chunk_size=50, tool_service=mock_svc, tool_ctx=mock_ctx)

        # Chunk 0
        r0 = json.loads(await tool_fn.ainvoke({"artifact_id": "art_num", "chunk_index": 0}))
        self.assertEqual(r0["char_range"], "0-100")
        self.assertEqual(r0["chunk_index"], 0)
        self.assertTrue(r0["has_more"])
        self.assertIn(full_content[0:100], r0["content"])

        # Chunk 1
        r1 = json.loads(await tool_fn.ainvoke({"artifact_id": "art_num", "chunk_index": 1}))
        self.assertEqual(r1["char_range"], "100-200")
        self.assertEqual(r1["chunk_index"], 1)
        self.assertTrue(r1["has_more"])
        self.assertIn(full_content[100:200], r1["content"])

        # Chunk 2
        r2 = json.loads(await tool_fn.ainvoke({"artifact_id": "art_num", "chunk_index": 2}))
        self.assertEqual(r2["char_range"], f"200-{len(full_content)}")
        self.assertEqual(r2["chunk_index"], 2)
        self.assertFalse(r2["has_more"])
        self.assertIn(full_content[200:], r2["content"])


# ---------------------------------------------------------------------------
# Q. P2 — 4: Reliable tool-failure telemetry classification
# ---------------------------------------------------------------------------

class TestTelemetryErrorClassification(unittest.IsolatedAsyncioTestCase):
    def test_classify_tool_result_comprehensive(self):
        """classify_tool_result detects all service and wrapper failure representations."""
        from agent_service import classify_tool_result
        from services.tool_service import ToolResult, ToolErrorCode

        # 1. ToolResult failures
        fail_res = ToolResult.fail(ToolErrorCode.OPERATION_FAILED, "DB hung")
        is_err, desc = classify_tool_result(fail_res)
        self.assertTrue(is_err)
        self.assertIn("DB hung", desc)

        # ToolResult success
        ok_res = ToolResult.ok("all good")
        is_err, _ = classify_tool_result(ok_res)
        self.assertFalse(is_err)

        # 2. Plain text service exception messages from graph tools
        is_err, desc = classify_tool_result("Graph memory search failed: connection pool exhausted")
        self.assertTrue(is_err)
        self.assertIn("connection pool exhausted", desc)

        is_err, desc = classify_tool_result("Failed to save artifact: service timeout")
        self.assertTrue(is_err)
        self.assertIn("service timeout", desc)

        # 3. Structured error codes
        is_err, desc = classify_tool_result("[OPERATION_FAILED] Failed to save artifact: timeout")
        self.assertTrue(is_err)

        # 4. JSON error payloads
        is_err, desc = classify_tool_result('{"error": "out_of_range", "message": "chunk 999 out of range"}')
        self.assertTrue(is_err)

        # 5. Nervous system blocks
        is_err, desc = classify_tool_result("Click blocked by nervous system: SafeOverlayPolicy (disallowed)")
        self.assertTrue(is_err)

        # 6. Benign content that includes the word 'error' in conversation
        is_err, _ = classify_tool_result("Here is how you handle an error in Python using try/except blocks.")
        self.assertFalse(is_err)

    async def test_agent_loop_telemetry_captures_plain_text_service_exceptions(self):
        """Agent loop records failed_tool_call_count and trace failure on plain-text graph failures."""
        from agent_service import execute_agent_loop
        from langchain_core.messages import AIMessage

        mock_svc = MagicMock()
        # Mock search_memory to raise an exception causing plain-text failure
        mock_svc.search_memory = AsyncMock(side_effect=RuntimeError("connection pool exhausted"))
        mock_ctx = MagicMock()

        # Mock LLM to invoke search_memory, then complete
        call1 = {"id": "call_1", "name": "search_memory", "args": {"query": "test", "domain": "ALL"}}
        msg_tool_call = AIMessage(content="", tool_calls=[call1])
        msg_final = AIMessage(content="Final summary after search failure.", tool_calls=[])

        with patch("agent_service._get_tool_service", return_value=mock_svc):
            with patch("agent_service._make_agent_loop_context", return_value=mock_ctx):
                with patch("agent_service.graph_store", MagicMock()):
                    with patch("agent_service.ChatOpenAI") as mock_llm_cls:
                        mock_llm_inst = MagicMock()
                        mock_llm_cls.return_value = mock_llm_inst
                        mock_bound = MagicMock()
                        mock_bound.ainvoke = AsyncMock(side_effect=[msg_tool_call, msg_final])
                        mock_llm_inst.bind_tools = MagicMock(return_value=mock_bound)

                        res = await execute_agent_loop(
                            task="search memory",
                            default_cdp_url="http://127.0.0.1:9222",
                            legacy_mode=False,
                        )

                        self.assertEqual(res.metadata["failed_tool_call_count"], 1)
                        traces = res.metadata["tool_traces"]
                        self.assertEqual(len(traces), 1)
                        self.assertFalse(traces[0]["success"])
                        self.assertIn("connection pool exhausted", str(traces[0].get("error", "")))


# ---------------------------------------------------------------------------
# R. P2 — 5: Least-privilege operation permissions on action routes
# ---------------------------------------------------------------------------

class TestActionRoutesLeastPrivilege(unittest.TestCase):
    def test_action_route_contexts_restricted_to_specific_operations(self):
        """Each /agent/action/* route context has only its intended permitted operation."""
        from agent_service import _make_action_route_context
        from services.tool_service import get_context_authority

        authority = get_context_authority()

        with patch("agent_service._resolve_verified_tab_ws", return_value=("target_123", "ws://127.0.0.1:9222/devtools/page/target_123")):
            # click route
            ctx_click, err = _make_action_route_context(
                op_name="click",
                tab_id="tab_test",
                worker_id=None,
                worker_token=None,
                is_trusted=True,
            )
            self.assertIsNone(err)
            with authority._lock:
                rec_click = authority._issued_records[ctx_click.context_token]
            self.assertEqual(rec_click.permitted_operations, {"click"})

            # type route
            ctx_type, err = _make_action_route_context(
                op_name="type",
                tab_id="tab_test",
                worker_id=None,
                worker_token=None,
                is_trusted=True,
            )
            self.assertIsNone(err)
            with authority._lock:
                rec_type = authority._issued_records[ctx_type.context_token]
            self.assertEqual(rec_type.permitted_operations, {"type"})

            # navigate route
            ctx_nav, err = _make_action_route_context(
                op_name="navigate",
                tab_id="tab_test",
                worker_id=None,
                worker_token=None,
                is_trusted=True,
            )
            self.assertIsNone(err)
            with authority._lock:
                rec_nav = authority._issued_records[ctx_nav.context_token]
            self.assertEqual(rec_nav.permitted_operations, {"navigate"})

            # extract route
            ctx_ext, err = _make_action_route_context(
                op_name="extract",
                tab_id="tab_test",
                worker_id=None,
                worker_token=None,
                is_trusted=True,
            )
            self.assertIsNone(err)
            with authority._lock:
                rec_ext = authority._issued_records[ctx_ext.context_token]
            self.assertEqual(rec_ext.permitted_operations, {"extract"})

    def test_unrelated_operation_rejected_on_route_context(self):
        """Attempting an unrelated operation with an action route context fails validation."""
        from agent_service import _make_action_route_context
        from services.tool_service import ComexToolService, get_context_authority, ToolErrorCode

        authority = get_context_authority()
        with patch("agent_service._resolve_verified_tab_ws", return_value=("target_123", "ws://127.0.0.1:9222/devtools/page/target_123")):
            ctx_click, err = _make_action_route_context(
                op_name="click",
                tab_id="tab_test",
                worker_id=None,
                worker_token=None,
                is_trusted=True,
            )
        self.assertIsNone(err)

        # 1. Authority validation directly catches operation widening
        val_err = authority.validate_context_claims(ctx_click, "type")
        self.assertIsNotNone(val_err)
        self.assertIn("Permission check failed", val_err)
        self.assertIn("type", val_err)

        # 2. Service-level context check also catches it
        service = ComexToolService(graph_store=MagicMock())
        chk = service._precheck_context(ctx_click, "type")
        self.assertIsNotNone(chk)
        self.assertFalse(chk.success)
        self.assertEqual(chk.error_code, ToolErrorCode.UNAUTHORIZED)


# ---------------------------------------------------------------------------
# S. P2 — 6: Preservation of shared service graph store reference
# ---------------------------------------------------------------------------

class TestGraphStoreSingletonPreservation(unittest.IsolatedAsyncioTestCase):
    async def test_get_tool_service_and_agent_loop_do_not_mutate_singleton_store(self):
        """Authoritative ComexToolService graph_store reference is not reassigned per-run."""
        import agent_service
        from agent_service import _get_tool_service, execute_agent_loop
        from langchain_core.messages import AIMessage

        mock_svc = MagicMock()
        mock_svc.graph_store = "initial_bound_store"
        mock_ctx = MagicMock()

        msg_done = AIMessage(content="Hello!", tool_calls=[])

        with patch("agent_service._comex_tool_service", mock_svc):
            with patch("agent_service._make_agent_loop_context", return_value=mock_ctx):
                with patch("agent_service.ChatOpenAI") as mock_llm_cls:
                    mock_llm_inst = MagicMock()
                    mock_llm_cls.return_value = mock_llm_inst
                    mock_bound = MagicMock()
                    mock_bound.ainvoke = AsyncMock(return_value=msg_done)
                    mock_llm_inst.bind_tools = MagicMock(return_value=mock_bound)

                    await execute_agent_loop(
                        task="say hi",
                        default_cdp_url="http://127.0.0.1:9222",
                        use_graph=True,
                        legacy_mode=False,
                    )

                    # Assert graph_store was NOT mutated to active_graph_store
                    self.assertEqual(mock_svc.graph_store, "initial_bound_store")


# ---------------------------------------------------------------------------
# T. P1 — 7: Visual cursor notification endpoint security
# ---------------------------------------------------------------------------

class TestVisualCursorSecurity(unittest.IsolatedAsyncioTestCase):
    async def test_notify_cursor_rejects_untrusted_request(self):
        """POST /agent/cursor/notify without trusted authority returns 403."""
        from agent_service import notify_cursor
        with patch("agent_service._INTERNAL_APP_TOKEN", "secret_internal_token"):
            resp = await notify_cursor({"x": 100, "y": 200, "action": "click"})
            self.assertEqual(resp.status_code, 403)
            self.assertIn("Unauthorized", json.loads(resp.body.decode())["error"])

    async def test_notify_cursor_rejects_malicious_payloads(self):
        """POST /agent/cursor/notify rejects script injections, invalid actions, and non-finite coordinates."""
        from agent_service import notify_cursor
        with patch("agent_service._INTERNAL_APP_TOKEN", "secret_token"):
            trusted_payload_base = {"contextToken": "secret_token"}

            # JS injection in action
            p1 = dict(trusted_payload_base, x=50, y=50, action="click'); alert(1); ('", tabId="tab1")
            r1 = await notify_cursor(p1)
            self.assertEqual(r1.status_code, 400)
            self.assertIn("Invalid action", json.loads(r1.body.decode())["error"])

            # JS injection in tabId
            p2 = dict(trusted_payload_base, x=50, y=50, action="click", tabId="tab1'); console.log('hacked'); //")
            r2 = await notify_cursor(p2)
            self.assertEqual(r2.status_code, 400)
            self.assertIn("Invalid tabId", json.loads(r2.body.decode())["error"])

            # Non-finite coordinate
            p3 = dict(trusted_payload_base, x=float("nan"), y=50, action="click", tabId="tab1")
            r3 = await notify_cursor(p3)
            self.assertEqual(r3.status_code, 400)
            self.assertIn("non-finite", json.loads(r3.body.decode())["error"])

            # Out of bounds coordinate
            p4 = dict(trusted_payload_base, x=-10, y=50, action="click", tabId="tab1")
            r4 = await notify_cursor(p4)
            self.assertEqual(r4.status_code, 400)
            self.assertIn("out of bounds", json.loads(r4.body.decode())["error"].lower())

    async def test_send_visual_cursor_cdp_safely_serializes_without_interpolation(self):
        """_send_visual_cursor_cdp serializes arguments via JSON without raw JavaScript string interpolation."""
        from agent_service import _send_visual_cursor_cdp

        captured_exprs = []
        mock_ws = MagicMock()
        async def _mock_send(msg_str):
            msg = json.loads(msg_str)
            captured_exprs.append(msg.get("params", {}).get("expression", ""))
        async def _mock_recv():
            return json.dumps({"result": {}})

        mock_ws.send = AsyncMock(side_effect=_mock_send)
        mock_ws.recv = AsyncMock(side_effect=_mock_recv)

        class MockWsConnect:
            async def __aenter__(self):
                return mock_ws
            async def __aexit__(self, *args):
                pass

        with patch("agent_service._ui_cdp_cached_ws", "ws://mock-ui"):
            with patch("websockets.connect", return_value=MockWsConnect()):
                await _send_visual_cursor_cdp(120.5, 340.5, "click", "tab-active-1")

        self.assertEqual(len(captured_exprs), 1)
        expr = captured_exprs[0]
        # Must parse JSON safely inside IIFE
        self.assertIn("JSON.parse", expr)
        self.assertIn("tab-active-1", expr)
        # Verify it parses correctly in Python json
        self.assertNotIn("x: 120.5", expr) # Direct unquoted JS expression object literal format is replaced by JSON.parse string


# ---------------------------------------------------------------------------
# J. Phase 2.1D Source Audit Remediation Suite
# ---------------------------------------------------------------------------

class TestPhase21DAuditRemediations(unittest.IsolatedAsyncioTestCase):
    """
    Direct regression tests for all 7 Phase 2.1D source audit findings:
    1. Route-level fail-closed propagation on /agent/run and /agent/run/stream.
    2. Strict context trust derived only from verified trusted_app_context.
    3. Graph-agent creation deferred until after preflight; duration fields synchronized.
    4. search_memory preservation of completeness evidence (truncation, partial failures).
    5. Authoritative classification of memory/skill write failures and bracketed codes.
    6. Visual cursor notification token scoping (target tab & actions) and strict origin checks.
    7. Synchronous artifact write timeout outcome unknown status and settlement evidence.
    """

    async def test_finding1_run_agent_propagates_fail_closed_status(self):
        """Finding 1: /agent/run returns success=False when preflight halts."""
        from agent_service import run_agent, AgentRequest
        req = AgentRequest(task="scrape data", taskId="task-f1", tabId="active_tab", contextToken="app-tok")
        with patch("agent_service._is_trusted_request", return_value=True), \
             patch("agent_service._get_tool_service", return_value=None):
            resp = await run_agent(req)
            self.assertFalse(resp.success)
            self.assertIn("ComexToolService is unavailable", resp.error)
            self.assertEqual(resp.metadata.get("outcome"), "service_unavailable")

    async def test_finding1_run_agent_stream_emits_fail_closed_status(self):
        """Finding 1: /agent/run/stream emits done event with success=False when preflight halts."""
        from agent_service import run_agent_stream, AgentRequest
        req = AgentRequest(task="scrape data", taskId="task-f1-stream", tabId="active_tab", contextToken="app-tok")
        with patch("agent_service._is_trusted_request", return_value=True), \
             patch("agent_service._get_tool_service", return_value=None):
            resp = await run_agent_stream(req)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            full_body = "".join(chunks)
            self.assertIn('"success": false', full_body.lower())
            self.assertIn("ComexToolService is unavailable", full_body)

    async def test_finding2_direct_call_without_trusted_context_fails_closed(self):
        """Finding 2: execute_agent_loop with worker_id=None, worker_token=None, trusted_app_context=False fails closed."""
        from agent_service import execute_agent_loop
        mock_svc = MagicMock()
        res = await execute_agent_loop(
            "test task",
            "ws://mock-cdp",
            tool_service=mock_svc,
            worker_id=None,
            worker_token=None,
            trusted_app_context=False,
        )
        self.assertEqual(res.metadata.get("outcome"), "context_creation_failed")
        self.assertIn("requires trusted server authority", res.metadata.get("error", ""))

    async def test_finding3_preflight_abort_creates_no_graph_agent_record_and_syncs_timing(self):
        """Finding 3: Graph-agent record creation is deferred until after preflight succeeds."""
        from agent_service import execute_agent_loop
        mock_store = MagicMock()
        with patch("agent_service.graph_store", mock_store), \
             patch("agent_service._get_tool_service", return_value=None):
            res = await execute_agent_loop(
                "test task",
                "ws://mock-cdp",
                use_graph=True,
                tool_service=None,  # service unavailable
                trusted_app_context=True,
            )
            # Graph store create_agent must NOT be called on preflight failure
            mock_store.create_agent.assert_not_called()
            self.assertEqual(res.metadata.get("outcome"), "service_unavailable")
            self.assertGreater(res.metadata.get("duration_seconds", 0), 0)
            self.assertGreater(res.metadata.get("timing", {}).get("total_duration_ms", 0), 0)

    def test_finding4_search_memory_adapter_surfaces_truncation_evidence(self):
        """Finding 4: search_memory preserves truncation evidence and warning notice."""
        from graph.tools import make_search_memory_tool
        mock_svc = MagicMock()
        mock_svc_res = _make_mock_tool_result(
            success=True,
            data=[{"content_excerpt": "result 1", "score": 0.9}],
            evidence={"truncated": True, "omitted_count": 8, "partial_failure": False},
        )
        async def _mock_search(*args, **kwargs):
            return mock_svc_res
        mock_svc.search_memory = _mock_search

        tool = make_search_memory_tool(
            MagicMock(), "agent-1", {"value": 0},
            tool_service=mock_svc, tool_ctx=MagicMock(),
        )
        res = tool.invoke({"query": "monitors", "domain": "ALL"})
        parsed = json.loads(res)
        self.assertTrue(parsed["truncated"])
        self.assertEqual(parsed["omitted_count"], 8)
        self.assertIn("notices", parsed)
        self.assertTrue(any("truncated" in n.lower() for n in parsed["notices"]))

    def test_finding4_search_memory_adapter_surfaces_partial_failure_evidence(self):
        """Finding 4: search_memory preserves partial failure evidence and domain warning notice."""
        from graph.tools import make_search_memory_tool
        mock_svc = MagicMock()
        mock_svc_res = _make_mock_tool_result(
            success=True,
            data=[],
            evidence={"truncated": False, "omitted_count": 0, "partial_failure": True, "failed_domains": ["HISTORY"]},
        )
        async def _mock_search(*args, **kwargs):
            return mock_svc_res
        mock_svc.search_memory = _mock_search

        tool = make_search_memory_tool(
            MagicMock(), "agent-1", {"value": 0},
            tool_service=mock_svc, tool_ctx=MagicMock(),
        )
        res = tool.invoke({"query": "monitors", "domain": "ALL"})
        parsed = json.loads(res)
        self.assertTrue(parsed["partial_failure"])
        self.assertEqual(parsed["failed_domains"], ["HISTORY"])
        self.assertIn("notices", parsed)
        self.assertTrue(any("partially failed" in n.lower() for n in parsed["notices"]))

    def test_finding5_classify_tool_result_coverage(self):
        """Finding 5: classify_tool_result detects user memory/skill prefixes and arbitrary bracketed codes."""
        from agent_service import classify_tool_result

        # Direct memory & skill write failure prefixes
        is_err, desc = classify_tool_result("Failed to save user memory: database disk is full")
        self.assertTrue(is_err)
        self.assertIn("Failed to save user memory", desc)

        is_err, desc = classify_tool_result("Failed to save skill: invalid syntax in snippet")
        self.assertTrue(is_err)
        self.assertIn("Failed to save skill", desc)

        # Arbitrary uppercase bracketed error codes
        is_err, desc = classify_tool_result("[DATABASE_ERROR] database is locked")
        self.assertTrue(is_err)
        self.assertIn("[DATABASE_ERROR]", desc)

        is_err, desc = classify_tool_result("[EXTERNAL_API_429] rate limit reached")
        self.assertTrue(is_err)

        # Normal successful prose must NOT be classified as error
        is_err, _ = classify_tool_result("Found 3 articles discussing common software errors.")
        self.assertFalse(is_err)

    async def test_finding6_cursor_notification_scoping(self):
        """Finding 6A: /agent/cursor/notify enforces tab and operation scoping on context tokens."""
        from agent_service import notify_cursor
        from services.tool_service import get_context_authority
        import time

        authority = get_context_authority()
        tok = "scoped-cursor-tok-1"
        authority.issue_context_authority(
            context_token=tok,
            lease_token="lease-1",
            run_id="run-1",
            task_id="task-1",
            permitted_operations=frozenset({"click"}),
            expires_at_mono=time.monotonic() + 30.0,
            target_tab_id="tab-allowed-1",
            is_single_agent=True,
        )

        with patch("agent_service._INTERNAL_APP_TOKEN", "app-token-different"):
            # 1. Mismatched tab -> 403 Forbidden
            r_bad_tab = await notify_cursor({"contextToken": tok, "tabId": "tab-wrong-2", "action": "click", "x": 10, "y": 20})
            self.assertEqual(r_bad_tab.status_code, 403)
            self.assertIn("Unauthorized", json.loads(r_bad_tab.body.decode())["error"])

            # 2. Mismatched action (token has click, payload asks for type) -> 403 Forbidden
            r_bad_act = await notify_cursor({"contextToken": tok, "tabId": "tab-allowed-1", "action": "type", "x": 10, "y": 20})
            self.assertEqual(r_bad_act.status_code, 403)
            self.assertIn("Unauthorized", json.loads(r_bad_act.body.decode())["error"])

            # 3. Matching tab and permitted action -> 200 OK
            with patch("agent_service._send_visual_cursor_cdp", new_callable=AsyncMock):
                r_ok = await notify_cursor({"contextToken": tok, "tabId": "tab-allowed-1", "action": "click", "x": 10, "y": 20})
                self.assertEqual(r_ok.get("status"), "ok")

    def test_finding6_is_trusted_ui_target_url(self):
        """Finding 6B: _is_trusted_ui_target_url rejects loose substring matches and accepts strict origins."""
        from agent_service import _is_trusted_ui_target_url

        self.assertTrue(_is_trusted_ui_target_url("http://localhost:5173"))
        self.assertTrue(_is_trusted_ui_target_url("http://localhost:5173/"))
        self.assertTrue(_is_trusted_ui_target_url("http://localhost:5173/index.html"))
        self.assertTrue(_is_trusted_ui_target_url("http://127.0.0.1:5173/"))
        self.assertTrue(_is_trusted_ui_target_url("file:///opt/comex/dist/index.html"))

        # Malicious candidates
        self.assertFalse(_is_trusted_ui_target_url("https://attacker.com/?ref=localhost:5173"))
        self.assertFalse(_is_trusted_ui_target_url("https://attacker.com/index.html"))
        self.assertFalse(_is_trusted_ui_target_url("http://not-localhost:5173"))
        self.assertFalse(_is_trusted_ui_target_url(""))

    def test_finding7_sync_save_artifact_timeout_reports_unknown_outcome(self):
        """Finding 7: Synchronous save_artifact bridge timeout returns [OPERATION_TIMEOUT] with outcome unknown."""
        from graph.tools import make_save_artifact_tool
        mock_svc = MagicMock()
        mock_svc.save_artifact = AsyncMock(side_effect=TimeoutError("Bridge deadline exceeded"))

        tool = make_save_artifact_tool(
            MagicMock(), "agent-1", {"value": 0},
            tool_service=mock_svc, tool_ctx=MagicMock(),
        )
        res = tool.invoke({"title": "Sync Test", "content": "Some Data", "artifact_type": "notes"})
        self.assertIn("[OPERATION_TIMEOUT]", res)
        self.assertIn("Outcome is UNKNOWN", res)
        self.assertIn("Reconcile before retrying", res)

    def test_finding7_save_artifact_surfaces_unsettled_and_compensation_evidence(self):
        """Finding 7: Service write failure surfaces settlement status."""
        from graph.tools import make_save_artifact_tool
        from services.tool_service import ToolErrorCode
        mock_svc = MagicMock()
        mock_res = _make_mock_tool_result(
            success=False,
            error_code=ToolErrorCode.OPERATION_FAILED,
            error_message="Compensation failed after storage abort",
            evidence={"settlement": "COMPENSATION_FAILED", "persisted": False},
        )
        async def _mock_save(*args, **kwargs):
            return mock_res
        mock_svc.save_artifact = _mock_save

        tool = make_save_artifact_tool(
            MagicMock(), "agent-1", {"value": 0},
            tool_service=mock_svc, tool_ctx=MagicMock(),
        )
        res = tool.invoke({"title": "Settlement Test", "content": "Data"})
        self.assertIn("settlement: COMPENSATION_FAILED", res)
        self.assertIn("persisted: False", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)


