from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.fsm import State
from session.models import Memory, Message
from trace.logger import TraceLogger

if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.executor import Executor
    from runtime.router import Router
    from session.store import Store


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_system_prompt(memory: Memory) -> str:
    """Inject memory (todos / plan / lessons) into the system prompt (spec §5.3)."""
    lines: list[str] = ["You are a helpful agent."]
    if memory.todos:
        lines.append("Todos:")
        lines.extend(f"- [#{t.id}] {t.title} [{t.status}]" for t in memory.todos)
    if memory.plan:
        lines.append("Plan: " + " | ".join(memory.plan))
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {lesson}" for lesson in memory.lessons)
    return "\n".join(lines)


class Agent:
    """Top-level orchestrator. DIRECT + SIMPLE_TOOL paths in S2.

    Contract C: on SIMPLE_TOOL the Executor owns session.messages persistence
    (user / assistant(tool_calls) / tool / final assistant). On DIRECT this
    Agent appends user + assistant itself. PLAN_REQUIRED falls back to the
    Executor until S3 wires real planning.
    """

    def __init__(
        self,
        store: Store,
        router: Router,
        executor: Executor,
        llm: LLMClient,
        trace_dir: Path,
    ) -> None:
        self._store = store
        self._router = router
        self._executor = executor
        self._llm = llm
        self._trace_dir = trace_dir

    async def chat(self, session_id: str, user_input: str) -> str:
        from runtime.router import Route

        session = self._store.load(session_id)
        session.fsm_state = State.ROUTING.value

        trace = TraceLogger(self._trace_dir / f"{session_id}.jsonl")
        try:
            route = await self._router.classify(user_input, session.memory)
            trace.log_route(route.value)

            if route == Route.DIRECT:
                session.fsm_state = State.RESPONDING.value
                sys_msg = build_system_prompt(session.memory)
                messages = [{"role": "system", "content": sys_msg}] + [
                    m.to_dict() for m in session.messages
                ]
                answer = await asyncio.to_thread(
                    self._llm.respond, messages, user_input
                )
                # DIRECT: agent persists user + assistant (executor does it for SIMPLE_TOOL)
                session.messages.append(Message(role="user", content=user_input))
                session.messages.append(Message(role="assistant", content=answer))
            else:
                # SIMPLE_TOOL, or PLAN_REQUIRED (S2 fallback -> executor).
                # Executor owns all session.messages persistence on this path.
                if route == Route.PLAN_REQUIRED:
                    # PLAN_REQUIRED not wired until S3; fall back to executor.
                    pass
                session.fsm_state = State.EXECUTING.value
                outcome = await self._executor.run(session, user_input, trace)
                session.fsm_state = State.RESPONDING.value
                answer = outcome.text

            session.fsm_state = State.IDLE.value
            session.updated_at = _now()  # Store.save does not refresh it
            self._store.save(session)
            return answer
        finally:
            trace.close()
