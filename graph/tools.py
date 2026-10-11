"""
LangChain tool for querying the Comex Agent Graph Memory.

Phase 2.1D Remediation:
- make_search_memory_tool, make_save_artifact_tool, and make_read_artifact_tool
  construct StructuredTool instances exposing native async coroutines for the
  production agent loop (.ainvoke()) and synchronous wrappers for legacy (.invoke()).
- Strict configuration checking: partial (service without ctx or ctx without service)
  configurations raise explicit ValueError.
- An omitted service/ctx pair without explicit legacy_mode=True raises ValueError.
  Production execution never implicitly falls back to direct graph-store access.
- Effective artifact chunk size is normalized once (min(max(100, int(chunk_size)), 16000))
  and used consistently across service requests and character range calculations.
- Synchronous bridge _run_coroutine_sync is strictly bounded and documents thread
  cancellation limitations.
"""
import inspect
import json
import logging
from langchain_core.tools import StructuredTool, tool as langchain_tool

logger = logging.getLogger(__name__)


def _run_coroutine_sync(coro_or_val, timeout: float = 30.0):
    """
    Safely execute a coroutine synchronously, or return the value directly.
    Handles running event loops (e.g. FastAPI/tests), no running event loop,
    and already-resolved values.

    Cancellation and Thread-Safety Contract:
    Standard Python executors (ThreadPoolExecutor, asyncio.to_thread) cannot forcibly
    terminate running OS threads. Work executing in background threads will continue
    until completion or cooperative exit.
    Production callers should use the native asynchronous path (.ainvoke()) so that
    ComexToolService manages deadlines, cooperative cancellation, and write settlement
    authoritatively without outer thread-pool desynchronization.
    When this helper times out, the caller is promptly unblocked with TimeoutError.
    Callers must NOT assume a cancelled thread write is rolled back unless verified
    by service settlement evidence.
    """
    if not inspect.iscoroutine(coro_or_val):
        return coro_or_val

    import asyncio
    import concurrent.futures

    async def _timed_wrapper():
        return await asyncio.wait_for(coro_or_val, timeout=timeout)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(asyncio.run, _timed_wrapper())
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout} seconds.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def make_search_memory_tool(
    graph_store,
    agent_id: str,
    step_ref: dict,
    seen_excerpts: set = None,
    *,
    tool_service=None,
    tool_ctx=None,
    legacy_mode: bool = False,
):
    if seen_excerpts is None:
        seen_excerpts = set()

    if tool_service is not None and tool_ctx is None:
        raise ValueError("Misconfigured search_memory tool: tool_service provided but tool_ctx is missing.")
    if tool_service is None and tool_ctx is not None:
        raise ValueError("Misconfigured search_memory tool: tool_ctx provided but tool_service is missing.")
    if tool_service is None and tool_ctx is None and not legacy_mode:
        raise ValueError("Production search_memory tool requires tool_service and tool_ctx. Explicit legacy_mode=True is required for direct graph-store access.")

    tool_desc = (
        "Check your persistent, long-term memory. "
        "You can search across two domains:\n"
        "- \"ENTITIES\": For structured data you extracted previously (e.g., products, prices).\n"
        "- \"HISTORY\": For your past tasks, actions, and the raw text results of your previous tool calls.\n"
        "- \"ALL\": To search everything.\n"
        "Always use search_memory before browsing if you suspect you or the user have already encountered the requested information."
    )

    if tool_service is not None and tool_ctx is not None:
        def _process_service_result(res, query: str, domain: str) -> str:
            if res is None:
                return "[OPERATION_FAILED] Graph memory search failed: service returned no result."
            from services.tool_service import ToolResult
            if isinstance(res, ToolResult) or (hasattr(res, "success") and hasattr(res, "data")):
                if res.success:
                    items = res.data if isinstance(res.data, list) else []
                    filtered = []
                    for r in items:
                        excerpt_hash = hash(r.get("content_excerpt", ""))
                        if excerpt_hash not in seen_excerpts:
                            filtered.append(r)
                            seen_excerpts.add(excerpt_hash)
                    try:
                        if graph_store:
                            graph_store.record_tool_call(
                                agent_id, "search_memory",
                                args_json=json.dumps({"query": query, "domain": domain}),
                                result_summary=f"{len(filtered)} memory items retrieved (service)",
                                step_index=step_ref.get("value", 0),
                                success=True,
                            )
                    except Exception as _re:
                        logger.warning(f"Graph: failed to record search_memory: {_re}")
                    evidence = getattr(res, "evidence", None)
                    if not isinstance(evidence, dict):
                        evidence = {}
                    is_truncated = bool(evidence.get("truncated"))
                    is_partial_fail = bool(evidence.get("partial_failure"))
                    omitted_count = evidence.get("omitted_count", 0)
                    failed_domains = evidence.get("failed_domains", [])

                    notices = []
                    if is_partial_fail:
                        notices.append(f"WARNING: Search partially failed across domain(s): {failed_domains}.")
                    if is_truncated:
                        notices.append(f"WARNING: Results truncated by output budget; {omitted_count} item(s) omitted.")

                    if is_truncated or is_partial_fail or filtered:
                        resp_payload = {
                            "results": filtered,
                            "count": len(filtered),
                            "truncated": is_truncated,
                            "omitted_count": omitted_count,
                            "partial_failure": is_partial_fail,
                        }
                        if failed_domains:
                            resp_payload["failed_domains"] = failed_domains
                        if notices:
                            resp_payload["notices"] = notices
                        return json.dumps(resp_payload, indent=2, default=str)

                    return "No matching memory found."
                # Failure path — surface error code and message without fallback
                err_parts = []
                err_code = getattr(res, "error_code", None)
                if err_code:
                    code_str = err_code.value if hasattr(err_code, "value") else str(err_code)
                    err_parts.append(f"[{code_str}]")
                err_msg = getattr(res, "error_message", None)
                if err_msg:
                    err_parts.append(f"Graph memory search failed: {err_msg}")
                else:
                    err_parts.append("Graph memory search failed.")
                return " ".join(err_parts)
            return f"[OPERATION_FAILED] Graph memory search failed: unexpected result type {type(res).__name__}."

        async def _search_memory_async(query: str, domain: str = "ALL") -> str:
            try:
                res = await tool_service.search_memory(tool_ctx, query, domain)
            except Exception as _svc_err:
                logger.warning(f"Phase 2.1D search_memory service call failed: {_svc_err}")
                return f"[OPERATION_FAILED] Graph memory search failed: {_svc_err}"
            return _process_service_result(res, query, domain)

        def _search_memory_sync(query: str, domain: str = "ALL") -> str:
            try:
                res = _run_coroutine_sync(tool_service.search_memory(tool_ctx, query, domain))
            except Exception as _svc_err:
                logger.warning(f"Phase 2.1D search_memory service call failed: {_svc_err}")
                return f"[OPERATION_FAILED] Graph memory search failed: {_svc_err}"
            return _process_service_result(res, query, domain)

        return StructuredTool.from_function(
            func=_search_memory_sync,
            coroutine=_search_memory_async,
            name="search_memory",
            description=tool_desc,
        )

    # Legacy fallback — only executed when explicitly enabled via legacy_mode=True
    def _search_memory_legacy(query: str, domain: str = "ALL") -> str:
        try:
            results = graph_store.search_memory(query, domain)
            filtered = []
            for r in results:
                excerpt_hash = hash(r.get("content_excerpt", ""))
                if excerpt_hash not in seen_excerpts:
                    filtered.append(r)
                    seen_excerpts.add(excerpt_hash)
            try:
                graph_store.record_tool_call(
                    agent_id, "search_memory",
                    args_json=json.dumps({"query": query, "domain": domain}),
                    result_summary=f"{len(filtered)} memory items retrieved",
                    step_index=step_ref.get("value", 0),
                    success=True
                )
            except Exception as re:
                logger.warning(f"Graph: failed to record search_memory tool call: {re}")
            if filtered:
                return json.dumps(filtered, indent=2, default=str)
            return "No matching memory found."
        except Exception as e:
            logger.warning(f"Graph memory search failed: {e}")
            return f"[OPERATION_FAILED] Graph memory search failed: {e}"

    return StructuredTool.from_function(
        func=_search_memory_legacy,
        name="search_memory",
        description=tool_desc,
    )


def make_save_artifact_tool(
    graph_store,
    agent_id: str,
    step_ref: dict,
    *,
    tool_service=None,
    tool_ctx=None,
    legacy_mode: bool = False,
):
    if tool_service is not None and tool_ctx is None:
        raise ValueError("Misconfigured save_artifact tool: tool_service provided but tool_ctx is missing.")
    if tool_service is None and tool_ctx is not None:
        raise ValueError("Misconfigured save_artifact tool: tool_ctx provided but tool_service is missing.")
    if tool_service is None and tool_ctx is None and not legacy_mode:
        raise ValueError("Production save_artifact tool requires tool_service and tool_ctx. Explicit legacy_mode=True is required for direct graph-store access.")

    tool_desc = (
        "Save a completed result or durable unstructured knowledge as an Artifact.\n"
        "Use this tool to persist final reports, collected lists, research summaries,\n"
        "or exact sets of questions."
    )

    if tool_service is not None and tool_ctx is not None:
        def _process_service_result(res, effective_title: str, effective_type: str, content: str) -> str:
            if res is None:
                return "[OPERATION_FAILED] Failed to save artifact: service returned no result."
            from services.tool_service import ToolResult
            if isinstance(res, ToolResult) or (hasattr(res, "success") and hasattr(res, "data")):
                if res.success:
                    art_id = (res.data or {}).get("artifact_id", "")
                    try:
                        if graph_store:
                            graph_store.record_tool_call(
                                agent_id, "save_artifact",
                                args_json=json.dumps({"title": effective_title, "artifact_type": effective_type, "content_length": len(content)}),
                                result_summary=f"Saved artifact {art_id} ({len(content)} chars) (service)",
                                step_index=step_ref.get("value", 0),
                                success=True,
                            )
                    except Exception as _re:
                        logger.warning(f"Graph: failed to record save_artifact: {_re}")
                    return f"Successfully saved artifact '{effective_title}' (ID: {art_id})."
                err_parts = []
                err_code = getattr(res, "error_code", None)
                if err_code:
                    code_str = err_code.value if hasattr(err_code, "value") else str(err_code)
                    err_parts.append(f"[{code_str}]")
                err_msg = getattr(res, "error_message", None)
                if err_msg:
                    err_parts.append(f"Failed to save artifact: {err_msg}")
                else:
                    err_parts.append("Failed to save artifact.")
                evidence = getattr(res, "evidence", None)
                if evidence and isinstance(evidence, dict):
                    settlement = evidence.get("settlement")
                    persisted = evidence.get("persisted")
                    if settlement is not None:
                        err_parts.append(f"(settlement: {settlement}, persisted: {persisted})")
                return " ".join(err_parts)
            return f"[OPERATION_FAILED] Failed to save artifact: unexpected result type {type(res).__name__}."

        async def _save_artifact_async(title: str, content: str, artifact_type: str = "notes") -> str:
            effective_title = (title or "Untitled Artifact").strip()
            effective_type = (artifact_type or "notes").strip()
            try:
                res = await tool_service.save_artifact(
                    context=tool_ctx,
                    title=effective_title,
                    content=content,
                    artifact_type=effective_type,
                )
            except Exception as _svc_err:
                logger.warning(f"Phase 2.1D save_artifact service call failed: {_svc_err}")
                return f"[OPERATION_FAILED] Failed to save artifact: {_svc_err}"
            return _process_service_result(res, effective_title, effective_type, content)

        def _save_artifact_sync(title: str, content: str, artifact_type: str = "notes") -> str:
            effective_title = (title or "Untitled Artifact").strip()
            effective_type = (artifact_type or "notes").strip()
            try:
                res = _run_coroutine_sync(tool_service.save_artifact(
                    context=tool_ctx,
                    title=effective_title,
                    content=content,
                    artifact_type=effective_type,
                ))
            except TimeoutError as _te:
                logger.warning(f"Phase 2.1D save_artifact sync bridge timed out: {_te}")
                return (
                    f"[OPERATION_TIMEOUT] Synchronous artifact write failed: timed out after bridge deadline. "
                    f"Outcome is UNKNOWN (background thread may still complete or compensate write). "
                    f"Reconcile before retrying to avoid duplicate artifacts."
                )
            except Exception as _svc_err:
                logger.warning(f"Phase 2.1D save_artifact service call failed: {_svc_err}")
                return f"[OPERATION_FAILED] Failed to save artifact: {_svc_err}"
            return _process_service_result(res, effective_title, effective_type, content)

        return StructuredTool.from_function(
            func=_save_artifact_sync,
            coroutine=_save_artifact_async,
            name="save_artifact",
            description=tool_desc,
        )

    # Legacy fallback — only executed when explicitly enabled via legacy_mode=True
    def _save_artifact_legacy(title: str, content: str, artifact_type: str = "notes") -> str:
        effective_title = (title or "Untitled Artifact").strip()
        effective_type = (artifact_type or "notes").strip()
        try:
            artifact_id = graph_store.save_artifact(agent_id, effective_title, effective_type, content)
            if artifact_id:
                try:
                    graph_store.record_tool_call(
                        agent_id, "save_artifact",
                        args_json=json.dumps({"title": effective_title, "artifact_type": effective_type, "content_length": len(content)}),
                        result_summary=f"Saved artifact {artifact_id} ({len(content)} chars)",
                        step_index=step_ref.get("value", 0),
                        success=True
                    )
                except Exception as re:
                    logger.warning(f"Graph: failed to record save_artifact tool call: {re}")
                return f"Successfully saved artifact '{effective_title}' (ID: {artifact_id})."
            return "[OPERATION_FAILED] Failed to save artifact."
        except Exception as e:
            logger.warning(f"save_artifact failed: {e}")
            return f"[OPERATION_FAILED] save_artifact failed: {e}"

    return StructuredTool.from_function(
        func=_save_artifact_legacy,
        name="save_artifact",
        description=tool_desc,
    )


def make_read_artifact_tool(
    graph_store,
    agent_id: str,
    step_ref: dict,
    chunk_size: int = 4000,
    *,
    tool_service=None,
    tool_ctx=None,
    legacy_mode: bool = False,
):
    if tool_service is not None and tool_ctx is None:
        raise ValueError("Misconfigured read_artifact tool: tool_service provided but tool_ctx is missing.")
    if tool_service is None and tool_ctx is not None:
        raise ValueError("Misconfigured read_artifact tool: tool_ctx provided but tool_service is missing.")
    if tool_service is None and tool_ctx is None and not legacy_mode:
        raise ValueError("Production read_artifact tool requires tool_service and tool_ctx. Explicit legacy_mode=True is required for direct graph-store access.")

    # Normalize effective chunk size ONCE and use identically for service and char_range
    try:
        raw_cs = int(chunk_size)
    except (ValueError, TypeError):
        raw_cs = 4000
    eff_chunk_size = min(max(100, raw_cs), 16000)

    tool_desc = (
        "Read the full content of an Artifact sequentially by chunk.\n"
        "Use this tool after finding an artifact_id (source_id) via search_memory\n"
        "to read its complete text without being bounded to a short excerpt."
    )

    if tool_service is not None and tool_ctx is not None:
        def _process_service_result(res, artifact_id: str, chunk_index: int) -> str:
            if res is None:
                return json.dumps({"error": "service_error", "message": "service returned no result"})
            from services.tool_service import ToolResult
            if isinstance(res, ToolResult) or (hasattr(res, "success") and hasattr(res, "data")):
                if res.success:
                    d = res.data or {}
                    chunk_text = d.get("chunk_text", "")
                    total_chunks = d.get("total_chunks", 1)
                    if total_chunks > 0 and chunk_index >= total_chunks:
                        return json.dumps({
                            "error": "out_of_range",
                            "message": f"chunk_index {chunk_index} is out of range. total_chunks is {total_chunks}",
                        })
                    start = chunk_index * eff_chunk_size
                    end = min(start + len(chunk_text), start + eff_chunk_size)
                    has_more = not d.get("is_last_chunk", True)
                    wrapped = f"[UNTRUSTED DATA START]\n{chunk_text}\n[UNTRUSTED DATA END]"
                    result = {
                        "artifact_id": d.get("artifact_id", artifact_id),
                        "title": d.get("title", ""),
                        "artifact_type": d.get("artifact_type", ""),
                        "created_at": d.get("created_at", ""),
                        "source_agent_id": d.get("source_agent_id", ""),
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "char_range": f"{start}-{end}",
                        "content": wrapped,
                        "has_more": has_more,
                    }
                    try:
                        if graph_store:
                            graph_store.record_tool_call(
                                agent_id, "read_artifact",
                                args_json=json.dumps({"artifact_id": artifact_id, "chunk_index": chunk_index}),
                                result_summary=f"Read artifact chunk {chunk_index}/{total_chunks} ({len(chunk_text)} chars) (service)",
                                step_index=step_ref.get("value", 0),
                                success=True,
                            )
                    except Exception as _re:
                        logger.warning(f"Graph: failed to record read_artifact: {_re}")
                    return json.dumps(result, indent=2, default=str)
                # Map failure
                err_msg = getattr(res, "error_message", None) or "read_artifact failed"
                err_code = getattr(res, "error_code", None) or "service_error"
                code_str = err_code.value if hasattr(err_code, "value") else str(err_code)
                legacy_error = "service_error"
                if "negative" in str(err_msg).lower():
                    legacy_error = "validation_error"
                elif "malformed" in str(err_msg).lower() or not (artifact_id and artifact_id.startswith("art_")):
                    legacy_error = "invalid_id"
                elif "not found" in str(err_msg).lower():
                    legacy_error = "not_found"
                elif code_str == "VALIDATION_ERROR":
                    legacy_error = "validation_error"
                else:
                    legacy_error = code_str
                return json.dumps({"error": legacy_error, "message": str(err_msg)})
            return json.dumps({"error": "service_error", "message": f"unexpected result type {type(res).__name__}"})

        async def _read_artifact_async(artifact_id: str, chunk_index: int = 0) -> str:
            try:
                res = await tool_service.read_artifact(
                    context=tool_ctx,
                    artifact_id=artifact_id,
                    chunk_index=chunk_index,
                    chunk_size=eff_chunk_size,
                )
            except Exception as _svc_err:
                logger.warning(f"Phase 2.1D read_artifact service call failed: {_svc_err}")
                return json.dumps({"error": "service_error", "message": str(_svc_err)})
            return _process_service_result(res, artifact_id, chunk_index)

        def _read_artifact_sync(artifact_id: str, chunk_index: int = 0) -> str:
            try:
                res = _run_coroutine_sync(tool_service.read_artifact(
                    context=tool_ctx,
                    artifact_id=artifact_id,
                    chunk_index=chunk_index,
                    chunk_size=eff_chunk_size,
                ))
            except Exception as _svc_err:
                logger.warning(f"Phase 2.1D read_artifact service call failed: {_svc_err}")
                return json.dumps({"error": "service_error", "message": str(_svc_err)})
            return _process_service_result(res, artifact_id, chunk_index)

        return StructuredTool.from_function(
            func=_read_artifact_sync,
            coroutine=_read_artifact_async,
            name="read_artifact",
            description=tool_desc,
        )

    # Legacy fallback — only executed when explicitly enabled via legacy_mode=True
    def _read_artifact_legacy(artifact_id: str, chunk_index: int = 0) -> str:
        try:
            if chunk_index < 0:
                return json.dumps({"error": "validation_error", "message": "chunk_index cannot be negative"})
            if not artifact_id or not artifact_id.startswith("art_"):
                return json.dumps({"error": "invalid_id", "message": f"Malformed Artifact ID: {artifact_id}"})
            artifact = graph_store.get_artifact(artifact_id)
            if not artifact:
                return json.dumps({"error": "not_found", "message": f"Artifact {artifact_id} not found."})
            content = artifact.get("content", "")
            content_len = len(content)
            if content_len == 0:
                total_chunks = 1
                start = 0
                end = 0
                chunk_content = ""
                has_more = False
            else:
                import math
                total_chunks = math.ceil(content_len / eff_chunk_size)
                if chunk_index >= total_chunks:
                    return json.dumps({"error": "out_of_range", "message": f"chunk_index {chunk_index} is out of range. total_chunks is {total_chunks}"})
                start = chunk_index * eff_chunk_size
                end = min(start + eff_chunk_size, content_len)
                chunk_content = content[start:end]
                has_more = end < content_len
            wrapped = f"[UNTRUSTED DATA START]\n{chunk_content}\n[UNTRUSTED DATA END]"
            result = {
                "artifact_id": artifact_id,
                "title": artifact.get("title", ""),
                "artifact_type": artifact.get("artifact_type", "notes"),
                "created_at": artifact.get("created_at", ""),
                "source_agent_id": artifact.get("source_agent_id", ""),
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "char_range": f"{start}-{end}",
                "content": wrapped,
                "has_more": has_more,
            }
            try:
                graph_store.record_tool_call(
                    agent_id, "read_artifact",
                    args_json=json.dumps({"artifact_id": artifact_id, "chunk_index": chunk_index}),
                    result_summary=f"Read artifact chunk {chunk_index}/{total_chunks} ({len(chunk_content)} chars)",
                    step_index=step_ref.get("value", 0),
                    success=True
                )
            except Exception as re:
                logger.warning(f"Graph: failed to record read_artifact tool call: {re}")
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            logger.warning(f"read_artifact failed: {e}")
            return json.dumps({"error": "graph_failure", "message": str(e)})

    return StructuredTool.from_function(
        func=_read_artifact_legacy,
        name="read_artifact",
        description=tool_desc,
    )



# ---------------------------------------------------------------------------
# Scope Note (Phase 2.1D Migration Scope):
# save_user_memory and save_skill remain synchronous direct graph-store tools.
# ComexToolService currently provides no service methods for user memory or
# skill creation. These tools are retained for backward compatibility, and their
# error prefixes ('Failed to save user memory:', 'Failed to save skill:') are
# classified by classify_tool_result() for authoritative telemetry.
# ---------------------------------------------------------------------------


def make_save_user_memory_tool(graph_store, agent_id: str, step_ref: dict):
    @langchain_tool
    def save_user_memory(key: str, value: str, category: str = "preference") -> str:
        """Save durable knowledge about the user, such as preferences, profile facts, or constraints.
        Categories: 'preference' (e.g. currency, preferred brands), 'profile' (e.g. location, tech stack),
        'constraint' (e.g. budget limit, non-negotiable requirements).
        """
        try:
            mem_id = graph_store.set_user_memory(key=key, value=value, category=category, agent_id=agent_id)
            try:
                graph_store.record_tool_call(
                    agent_id, "save_user_memory",
                    args_json=json.dumps({"key": key, "value": value, "category": category}),
                    result_summary=f"Saved user memory '{key}' (ID: {mem_id})",
                    step_index=step_ref.get("value", 0),
                    success=True
                )
            except Exception as re:
                logger.warning(f"Graph: failed to record save_user_memory tool call: {re}")
            return f"Successfully saved user memory: {key} = {value} (Category: {category})"
        except Exception as e:
            logger.warning(f"save_user_memory failed: {e}")
            return f"Failed to save user memory: {e}"

    return save_user_memory


def make_save_skill_tool(graph_store, agent_id: str, step_ref: dict):
    @langchain_tool
    def save_skill(name: str, domain: str, description: str, code_snippet: str) -> str:
        """Save a distilled browser interaction skill or recipe that can be reused for future tasks.
        Domain should be the target site (e.g. 'amazon.in', 'youtube.com') or '*' for generic skills.
        Code snippet must be executable Python code suitable for run_browser_harness.
        """
        try:
            skill_id = graph_store.save_skill(name=name, domain=domain, description=description, code_snippet=code_snippet, agent_id=agent_id)
            try:
                graph_store.record_tool_call(
                    agent_id, "save_skill",
                    args_json=json.dumps({"name": name, "domain": domain, "description": description}),
                    result_summary=f"Saved skill '{name}' for domain {domain} (ID: {skill_id})",
                    step_index=step_ref.get("value", 0),
                    success=True
                )
            except Exception as re:
                logger.warning(f"Graph: failed to record save_skill tool call: {re}")
            return f"Successfully saved skill '{name}' for {domain} (ID: {skill_id})."
        except Exception as e:
            logger.warning(f"save_skill failed: {e}")
            return f"Failed to save skill: {e}"

    return save_skill
