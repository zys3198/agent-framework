from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from session.models import Memory, Message
from trace.logger import TraceLogger

STATE_IDLE = "IDLE"
STATE_PLANNING = "PLANNING"
STATE_EXECUTING = "EXECUTING"
STATE_REFLECTING = "REFLECTING"
STATE_WAITING = "WAITING"

if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.executor import Executor
    from runtime.planner import Planner
    from runtime.router import Router
    from session.store import Store


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_system_prompt(memory: Memory) -> str:
    """Inject memory (todos / plan / lessons) into the system prompt (spec 5.3)."""
    lines: list[str] = ["You are a helpful agent."]
    if memory.todos:
        lines.append("Todos:")
        lines.extend(f"- [#{t.id}] {t.title} [{t.status}]" for t in memory.todos)
    if memory.plan:
        lines.append("Plan: " + " | ".join(s.prompt for s in memory.plan))
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {lesson}" for lesson in memory.lessons)
    return "\n".join(lines)


class Agent:
    """Top-level orchestrator. DIRECT + SIMPLE_TOOL + PLAN_REQUIRED paths.

    Contract C: on SIMPLE_TOOL and per-step in PLAN_REQUIRED, the Executor owns
    session.messages persistence (user / assistant(tool_calls) / tool / final
    assistant). On DIRECT this Agent appends user + assistant itself. In the
    PLAN_REQUIRED branch the Agent appends ONLY the final synthesized assistant
    answer -- per-step messages are persisted by the executor.

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
    ) -> None:
        self._store = store
        self._router = router
        self._executor = executor
        self._llm = llm
        self._trace_dir = trace_dir
        self._planner = planner

    async def chat(self, session_id: str, user_input: str) -> str:
        from runtime.router import Route

        session = self._store.load(session_id)
        session.fsm_state = STATE_IDLE

        trace = TraceLogger(self._trace_dir / f"{Path(session_id).name}.jsonl")
        try:
            route = await self._router.classify(user_input, session.memory)
            trace.log_route(route.value)

            if route == Route.DIRECT:
                session.fsm_state = STATE_WAITING
                sys_msg = build_system_prompt(session.memory)
                messages = [{"role": "system", "content": sys_msg}] + [
                    m.to_dict() for m in session.messages
                ]
                answer = await asyncio.to_thread(
                    self._llm.respond, messages, user_input
                )
                # DIRECT: agent persists user + assistant (executor does it for tool paths)
                session.messages.append(Message(role="user", content=user_input))
                session.messages.append(Message(role="assistant", content=answer))

            elif route == Route.SIMPLE_TOOL:
                # Executor owns all session.messages persistence on this path (Contract C).
                session.fsm_state = STATE_EXECUTING
                outcome = await self._executor.run(session, user_input, trace)
                session.fsm_state = STATE_WAITING
                if outcome.needs_replan:
                    # No replanner now; log so the failure is not silently swallowed.
                    trace.log_truncated()
                answer = outcome.text

            else:  # PLAN_REQUIRED
                session.fsm_state = STATE_PLANNING
                plan = await self._planner.make_plan(user_input, session.memory)
                session.memory.plan = plan
                session.fsm_state = STATE_EXECUTING

                # Planner makes the initial decomposition. Failed steps are NOT
                # re-planned; their outcome (incl. ERROR) is handed to synthesis
                # so the model knows a step failed (replacement for the deleted
                # Replanner safety net).
                results: dict[int, object] = {}
                for i in range(len(plan)):
                    outcome = await self._executor.run(session, plan[i].prompt, trace)
                    results[i] = outcome
                    if outcome.needs_replan:
                        trace.log_truncated()

                answer = await asyncio.to_thread(
                    self._llm.synthesize,
                    [s.prompt for s in plan],
                    {str(idx): _synthesize_entry(idx, o) for idx, o in results.items()},
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
