from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.agent_memory import load_project_context
from runtime.memory_projector import build_memory_context_message, build_system_prompt
from session.models import Message
from trace.logger import TraceLogger

STATE_IDLE = "IDLE"
STATE_PLANNING = "PLANNING"
STATE_EXECUTING = "EXECUTING"
STATE_REFLECTING = "REFLECTING"
STATE_WAITING = "WAITING"

if TYPE_CHECKING:
    from ctx.compactor import Compactor
    from llm.client import LLMClient
    from runtime.executor import Executor
    from runtime.planner import Planner
    from runtime.router import Router
    from session.store import Store


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Agent:
    """Top-level orchestrator. DIRECT + SIMPLE_TOOL + PLAN_REQUIRED paths.

    Contract C: Executor owns tool-turn message persistence and returns an
    ExecutionResult describing the turn. On DIRECT this Agent appends user +
    assistant itself. In the PLAN_REQUIRED branch the Agent appends ONLY the
    final synthesized assistant answer; per-step messages are persisted by the
    executor.

    Phase 0: ReWOO + Replanner deleted (dead paths). Planner still builds the
    initial plan; failures are NOT re-planned -- the failed step's outcome is
    surfaced to synthesis so the model knows a step failed.
    """

    def __init__(
        self,
        store: Store,
        router: Router,
        executor: Executor,
        llm: LLMClient,
        trace_dir: Path,
        planner: Planner,
        workspace_root: Path | None = None,
        user_home: Path | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        self._store = store
        self._router = router
        self._executor = executor
        self._llm = llm
        self._trace_dir = trace_dir
        self._planner = planner
        self._workspace_root = workspace_root or Path.cwd()
        self._user_home = user_home
        self._compactor = compactor
        # Async per-session lock serializes the await-heavy chat turn. Store also
        # owns a synchronous with_session seam for non-async load->mutate->save
        # callers.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    async def chat(self, session_id: str, user_input: str) -> str:
        lock = await self._lock_for(session_id)
        async with lock:
            return await self._chat_impl(session_id, user_input)

    async def _chat_impl(self, session_id: str, user_input: str) -> str:
        from runtime.router import Route

        session = self._store.load(session_id)
        session.fsm_state = STATE_IDLE
        # Phase 3: compact accumulated context before the next user turn.
        # Layers are no-ops below threshold, so this is cheap when not needed.
        if self._compactor is not None:
            compacted = await self._compactor.compact(session)
            if compacted:
                self._store.save(session)  # persist compaction before proceeding
        project_context = load_project_context(self._workspace_root, self._user_home)

        trace = TraceLogger(self._trace_dir / f"{Path(session_id).name}.jsonl")
        try:
            route = await self._router.classify(user_input, session.memory)
            trace.log_route(route.value)

            if route == Route.DIRECT:
                session.fsm_state = STATE_WAITING
                sys_msg = build_system_prompt(session.memory)
                messages = [{"role": "system", "content": sys_msg}]
                memory_msg = build_memory_context_message(
                    session.memory, project_context=project_context
                )
                if memory_msg is not None:
                    messages.append(memory_msg)
                messages.extend(m.to_dict() for m in session.messages)
                answer = await self._llm.respond(messages, user_input)
                # DIRECT: agent persists user + assistant (executor does it for tool paths)
                session.messages.append(Message(role="user", content=user_input))
                session.messages.append(Message(role="assistant", content=answer))

            elif route == Route.SIMPLE_TOOL:
                # Executor owns all session.messages persistence on this path (Contract C).
                session.fsm_state = STATE_EXECUTING
                execution = await self._executor.run(
                    session, user_input, trace, project_context=project_context
                )
                session.fsm_state = STATE_WAITING
                if execution.needs_replan:
                    # No replanner now; log so the failure is not silently swallowed.
                    trace.log_truncated()
                answer = execution.text

            else:  # PLAN_REQUIRED
                session.fsm_state = STATE_PLANNING
                plan = await self._planner.make_plan(
                    user_input, session.memory, project_context=project_context
                )
                session.memory.plan = plan
                session.fsm_state = STATE_EXECUTING

                # Planner makes the initial decomposition. Failed steps are NOT
                # re-planned; their outcome (incl. ERROR) is handed to synthesis
                # so the model knows a step failed (replacement for the deleted
                # Replanner safety net).
                results: dict[int, object] = {}
                for i in range(len(plan)):
                    execution = await self._executor.run(
                        session, plan[i].prompt, trace, project_context=project_context
                    )
                    results[i] = execution
                    if execution.needs_replan:
                        trace.log_truncated()

                answer = await self._llm.synthesize(
                    [s.prompt for s in plan],
                    {str(idx): _synthesize_entry(idx, o) for idx, o in results.items()},
                    project_context,
                )
                # Contract C: agent appends ONLY the synthesized final answer here;
                # per-step messages were persisted by the executor.
                session.fsm_state = STATE_WAITING
                session.messages.append(Message(role="assistant", content=answer))

            session.fsm_state = STATE_IDLE
            session.updated_at = _now()  # Store.save does not refresh it
            self._store.save(session)
            return answer
        finally:
            trace.close()


def _synthesize_entry(idx: int, outcome: object) -> str:
    """Mark failed steps so synthesis can distinguish them from successes."""
    text = getattr(outcome, "text", str(outcome))
    needs_replan = getattr(outcome, "needs_replan", False)
    if needs_replan:
        return f"[STEP {idx} FAILED] {text}"
    return f"[STEP {idx}] {text}"
