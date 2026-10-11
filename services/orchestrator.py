"""
Comex Multi-Agent Orchestrator MVP.

Provides minimal, typed, server-authorized multi-worker orchestration with:
- Bounded concurrency (maximum 2 workers).
- Server-authorized, isolated per-worker browser-tab leases.
- Asynchronous scheduling with strict DAG dependency resolution and cycle rejection.
- Failure containment: independent tasks are never corrupted by worker failures.
- Cooperative cancellation and clean lease reclamation.
- Structured worker results with source evidence aggregation.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple

from nervous.action_policy import (
    PROHIBITED_FOREGROUND_ALIASES,
    ServerTabLeaseRegistry,
    ServerTabTargetRegistry,
    TabLease,
    get_lease_registry,
    get_target_registry,
)
from services.tool_service import (
    ComexToolService,
    ToolErrorCode,
    ToolExecutionContext,
    ToolResult,
    create_tool_execution_context,
)

logger = logging.getLogger("comex.orchestrator")

# Maximum concurrent browser workers in MVP
MAX_WORKER_CONCURRENCY = 2


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerResult:
    """Structured result returned by an individual worker."""
    findings: Any
    source_urls: List[str] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    limitations: Optional[str] = None
    tool_calls_count: int = 0
    execution_time_ms: float = 0.0
    tab_id: Optional[str] = None
    worker_id: Optional[str] = None


@dataclass
class OrchestratorTask:
    """An individual unit of work scheduled by the orchestrator."""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker_id: Optional[str] = None
    assigned_tab_id: Optional[str] = None
    deadline_seconds: float = 60.0
    result: Optional[WorkerResult] = None
    error: Optional[str] = None
    requires_browser: bool = True
    sensitive_action: bool = False
    approved: bool = False
    start_time_mono: Optional[float] = None
    end_time_mono: Optional[float] = None
    task_fn: Optional[Callable[[ToolExecutionContext, ComexToolService], Coroutine[Any, Any, WorkerResult]]] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Consolidated findings compiled across all completed worker tasks."""
    summary: str
    task_results: Dict[str, WorkerResult] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    has_failures: bool = False
    execution_duration_ms: float = 0.0


@dataclass
class OrchestrationRun:
    """Complete execution record of a multi-agent orchestration workflow."""
    id: str
    user_request: str
    status: RunStatus = RunStatus.PENDING
    tasks: Dict[str, OrchestratorTask] = field(default_factory=dict)
    aggregated_result: Optional[AggregatedResult] = None
    created_at_mono: float = field(default_factory=time.monotonic)
    completed_at_mono: Optional[float] = None
    cancellation_event: asyncio.Event = field(default_factory=asyncio.Event)


def validate_dependency_graph(tasks: Dict[str, OrchestratorTask]) -> None:
    """
    Validate task dependencies:
    - Verifies all referenced dependency IDs exist in the task collection.
    - Rejects self-dependencies.
    - Rejects cyclic dependencies using 3-color DFS cycle detection.
    """
    task_ids = set(tasks.keys())

    # 1. Existence check & self-dependency check
    for tid, task in tasks.items():
        for dep in task.dependencies:
            if dep not in task_ids:
                raise ValueError(
                    f"Unknown dependency reference: task '{tid}' depends on non-existent task '{dep}'."
                )
            if dep == tid:
                raise ValueError(
                    f"Self-dependency detected: task '{tid}' cannot depend on itself."
                )

    # 2. Cycle detection via DFS (WHITE = 0, GRAY = 1, BLACK = 2)
    visited: Dict[str, int] = {tid: 0 for tid in tasks}

    def _dfs(node: str, path: List[str]):
        visited[node] = 1  # In progress
        path.append(node)
        for neighbor in tasks[node].dependencies:
            if visited[neighbor] == 1:
                cycle_path = " -> ".join(path + [neighbor])
                raise ValueError(f"Cyclic dependency detected: {cycle_path}")
            if visited[neighbor] == 0:
                _dfs(neighbor, path)
        path.pop()
        visited[node] = 2  # Completed

    for tid in tasks:
        if visited[tid] == 0:
            _dfs(tid, [])


def decompose_request_into_tasks(user_request: str) -> List[OrchestratorTask]:
    """
    Heuristic task decomposition for two-worker research workflows:
    Worker A: Technical specifications, performance, and hardware benchmarks.
    Worker B: Market pricing, availability, and user reviews.
    """
    clean_req = (user_request or "").strip()
    return [
        OrchestratorTask(
            id="task_specs_perf",
            description=f"Investigate technical specifications and performance for: {clean_req}",
            dependencies=[],
            deadline_seconds=45.0,
            requires_browser=True,
        ),
        OrchestratorTask(
            id="task_price_reviews",
            description=f"Investigate market pricing, deals, and user reviews for: {clean_req}",
            dependencies=[],
            deadline_seconds=45.0,
            requires_browser=True,
        ),
    ]


class MultiAgentOrchestrator:
    """
    Authoritative server-side coordinator for concurrent browser workers.
    """

    def __init__(
        self,
        tool_service: ComexToolService,
        lease_registry: Optional[ServerTabLeaseRegistry] = None,
        target_registry: Optional[ServerTabTargetRegistry] = None,
        max_concurrency: int = MAX_WORKER_CONCURRENCY,
    ) -> None:
        self.tool_service = tool_service
        self.lease_registry = lease_registry or get_lease_registry()
        self.target_registry = target_registry or get_target_registry()
        self.max_concurrency = max(1, min(max_concurrency, 4))
        self._runs: Dict[str, OrchestrationRun] = {}

    def create_run(
        self,
        user_request: str,
        tasks: Optional[List[OrchestratorTask]] = None,
    ) -> OrchestrationRun:
        """Create and validate a new orchestration run."""
        clean_req = (user_request or "").strip()
        if not clean_req:
            raise ValueError("user_request must be a non-empty string.")

        run_id = f"run_{secrets.token_hex(6)}"
        task_list = tasks if tasks is not None else decompose_request_into_tasks(clean_req)
        if not task_list:
            raise ValueError("At least one task is required for orchestration.")

        task_dict: Dict[str, OrchestratorTask] = {}
        for t in task_list:
            if not t.id or not isinstance(t.id, str):
                raise ValueError("Task ID must be a non-empty string.")
            if t.id in task_dict:
                raise ValueError(f"Duplicate task ID in run: '{t.id}'.")
            task_dict[t.id] = t

        # Validate DAG
        validate_dependency_graph(task_dict)

        run = OrchestrationRun(
            id=run_id,
            user_request=clean_req,
            tasks=task_dict,
        )
        self._runs[run_id] = run
        return run

    def cancel_run(self, run_id: str) -> bool:
        """Signal cooperative cancellation across all workers in a run."""
        run = self._runs.get(run_id)
        if not run:
            return False
        run.cancellation_event.set()
        run.status = RunStatus.CANCELLED
        for t in run.tasks.values():
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                t.status = TaskStatus.CANCELLED
                t.error = "Run was cancelled by caller."
        return True

    async def execute_run(self, run: OrchestrationRun) -> AggregatedResult:
        """
        Execute an orchestration run with bounded concurrency and dependency management.
        """
        run.status = RunStatus.RUNNING
        t_start = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        task_completion_events: Dict[str, asyncio.Event] = {
            tid: asyncio.Event() for tid in run.tasks
        }

        async def _worker_wrapper(task: OrchestratorTask):
            # 1. Wait for all prerequisites
            for dep_id in task.dependencies:
                dep_event = task_completion_events[dep_id]
                await dep_event.wait()
                dep_task = run.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.BLOCKED
                    task.error = f"Prerequisite task '{dep_id}' was not completed (status: {dep_task.status})."
                    task_completion_events[task.id].set()
                    return

            # Check if run cancelled before acquiring worker slot
            if run.cancellation_event.is_set():
                task.status = TaskStatus.CANCELLED
                task.error = "Task cancelled prior to worker execution."
                task_completion_events[task.id].set()
                return

            # 2. Acquire concurrency slot (bounded to max_concurrency = 2)
            async with semaphore:
                if run.cancellation_event.is_set():
                    task.status = TaskStatus.CANCELLED
                    task.error = "Task cancelled before starting execution."
                    task_completion_events[task.id].set()
                    return

                # Sensitive action authorization gate
                if task.sensitive_action and not task.approved:
                    task.status = TaskStatus.FAILED
                    task.error = (
                        f"Sensitive external action in task '{task.id}' requires explicit user approval."
                    )
                    task_completion_events[task.id].set()
                    return

                task.status = TaskStatus.RUNNING
                task.start_time_mono = time.monotonic()
                worker_id = task.assigned_worker_id or f"worker_{task.id}_{secrets.token_hex(4)}"
                tab_id = task.assigned_tab_id or f"tab_{task.id}"
                task.assigned_worker_id = worker_id
                task.assigned_tab_id = tab_id

                lease: Optional[TabLease] = None
                try:
                    # 3. Explicitly acquire server-authorized tab lease
                    if task.requires_browser:
                        lease = self.lease_registry.acquire_lease(
                            worker_id=worker_id,
                            tab_id=tab_id,
                            ttl_seconds=task.deadline_seconds,
                            trusted_issuer=True,
                        )

                    # 4. Construct authoritative ToolExecutionContext
                    context = create_tool_execution_context(
                        task_id=task.id,
                        run_id=run.id,
                        worker_id=worker_id if task.requires_browser else None,
                        target_tab=tab_id if task.requires_browser else "active_tab",
                        worker_token=lease.lease_token if lease else None,
                        timeout_seconds=task.deadline_seconds,
                        cancellation_event=run.cancellation_event,
                        lease_registry=self.lease_registry,
                        trusted_server_context=not task.requires_browser,
                    )

                    # 5. Execute task routine
                    if task.task_fn:
                        res = await task.task_fn(context, self.tool_service)
                        if not isinstance(res, WorkerResult):
                            task.result = WorkerResult(
                                findings=str(res),
                                success=True,
                                worker_id=worker_id,
                                tab_id=tab_id,
                            )
                        else:
                            res.worker_id = worker_id
                            res.tab_id = tab_id
                            task.result = res
                    else:
                        # Default browser routine: navigate & inspect
                        task.result = await self._default_browser_routine(context, task)

                    if run.cancellation_event.is_set():
                        task.status = TaskStatus.CANCELLED
                        task.error = "Task execution was cancelled."
                    elif task.result and task.result.success:
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.FAILED
                        if task.result and task.result.error:
                            task.error = task.result.error

                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    task.error = "Worker execution was cancelled."
                except PermissionError as pe:
                    task.status = TaskStatus.FAILED
                    task.error = f"Authorization failure: {pe}"
                    task.result = WorkerResult(
                        findings=None,
                        success=False,
                        error=str(pe),
                        worker_id=worker_id,
                        tab_id=tab_id,
                    )
                except Exception as exc:
                    logger.exception(f"Unhandled exception in task '{task.id}': {exc}")
                    task.status = TaskStatus.FAILED
                    task.error = f"Worker runtime exception: {exc}"
                    task.result = WorkerResult(
                        findings=None,
                        success=False,
                        error=str(exc),
                        worker_id=worker_id,
                        tab_id=tab_id,
                    )
                finally:
                    task.end_time_mono = time.monotonic()
                    # 6. Authoritative lease reclamation
                    if lease:
                        try:
                            self.lease_registry.revoke_lease(
                                worker_id=worker_id,
                                tab_id=tab_id,
                                trusted_caller=True,
                            )
                        except Exception as re:
                            logger.warning(f"Error revoking lease for worker '{worker_id}': {re}")
                    task_completion_events[task.id].set()

        # Execute all tasks concurrently (semaphore enforces max concurrency 2)
        worker_coros = [_worker_wrapper(t) for t in run.tasks.values()]
        await asyncio.gather(*worker_coros, return_exceptions=True)

        t_elapsed_ms = round((time.monotonic() - t_start) * 1000.0, 2)
        run.completed_at_mono = time.monotonic()

        # Update final run status
        any_failed = any(t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED) for t in run.tasks.values())
        all_completed = all(t.status == TaskStatus.COMPLETED for t in run.tasks.values())
        if run.cancellation_event.is_set():
            run.status = RunStatus.CANCELLED
        elif all_completed:
            run.status = RunStatus.COMPLETED
        elif any_failed:
            run.status = RunStatus.FAILED
        else:
            run.status = RunStatus.COMPLETED

        # 7. Aggregate findings
        agg_res = self._aggregate_findings(run, t_elapsed_ms)
        run.aggregated_result = agg_res
        return agg_res

    async def _default_browser_routine(
        self,
        context: ToolExecutionContext,
        task: OrchestratorTask,
    ) -> WorkerResult:
        """Default routine when no custom task_fn is provided."""
        t0 = time.monotonic()
        tool_calls = 0
        target_url = task.url or "https://example.com"
        nav_res = await self.tool_service.navigate(context, target_url)
        tool_calls += 1

        if not nav_res.success:
            return WorkerResult(
                findings=None,
                source_urls=[target_url],
                success=False,
                error=f"Navigation failed: {nav_res.error}",
                tool_calls_count=tool_calls,
                execution_time_ms=round((time.monotonic() - t0) * 1000.0, 2),
            )

        inspect_res = await self.tool_service.inspect(context, max_elements=20)
        tool_calls += 1

        findings = {
            "title": nav_res.data.get("title") if nav_res.data else "Sample Page",
            "url": target_url,
            "elements_inspected": inspect_res.data.get("returned", 0) if inspect_res.data else 0,
        }

        return WorkerResult(
            findings=findings,
            source_urls=[target_url],
            supporting_evidence=[{"stage": "inspect", "data": inspect_res.data}],
            success=True,
            tool_calls_count=tool_calls,
            execution_time_ms=round((time.monotonic() - t0) * 1000.0, 2),
        )

    def _aggregate_findings(
        self,
        run: OrchestrationRun,
        execution_duration_ms: float,
    ) -> AggregatedResult:
        """Consolidate findings across all tasks in the run."""
        task_results: Dict[str, WorkerResult] = {}
        all_sources: List[str] = []
        missing: List[str] = []
        summary_lines = [f"Orchestration Summary for: '{run.user_request}'\n"]

        has_failures = False
        for tid, task in run.tasks.items():
            if task.status == TaskStatus.COMPLETED and task.result:
                task_results[tid] = task.result
                summary_lines.append(f"### Task: {task.description}")
                summary_lines.append(f"- Status: COMPLETED (Worker: {task.assigned_worker_id}, Tab: {task.assigned_tab_id})")
                summary_lines.append(f"- Findings: {task.result.findings}")
                for src in task.result.source_urls:
                    if src not in all_sources:
                        all_sources.append(src)
            else:
                has_failures = True
                missing.append(f"{tid}: {task.error or task.status.value}")
                summary_lines.append(f"### Task: {task.description}")
                summary_lines.append(f"- Status: {task.status.value.upper()} (Error: {task.error})")

        if missing:
            summary_lines.append("\n### Limitations & Missing Information:")
            for m in missing:
                summary_lines.append(f"- {m}")

        return AggregatedResult(
            summary="\n".join(summary_lines),
            task_results=task_results,
            sources=all_sources,
            missing_information=missing,
            has_failures=has_failures,
            execution_duration_ms=execution_duration_ms,
        )
