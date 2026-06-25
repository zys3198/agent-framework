from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.claude_memory import load_claude_context
from session.models import RECENT_LESSONS_LIMIT, Memory, MemoryEntry, Message
from trace.logger import TraceLogger

STATE_IDLE = "IDLE"
STATE_PLANNING = "PLANNING"
STATE_EXECUTING = "EXECUTING"
STATE_REFLECTING = "REFLECTING"
STATE_WAITING = "WAITING"
_MEMORY_INDEX_MAX_LINES = 200
_MEMORY_INDEX_MAX_BYTES = 25 * 1024

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
        lines.extend(f"- {lesson}" for lesson in memory.lessons[-RECENT_LESSONS_LIMIT:])
    return "\n".join(lines)


def build_memory_context_message(
    memory: Memory, claude_context: str = ""
) -> dict[str, str] | None:
    entries = _memory_index_lines(memory.entries)
    parts = [part for part in [claude_context.strip(), "\n".join(entries)] if part]
    if not parts:
        return None
    return {"role": "user", "content": "\n\n".join(parts)}


def _memory_index_lines(entries: list[MemoryEntry]) -> list[str]:
    if not entries:
        return []

    lines = ["Memory index:"]
    total_bytes = len(lines[0].encode("utf-8"))
    for entry in entries:
        line = (
            f"- id={entry.id} type={entry.type} name={entry.name} "
            f"description={entry.description} "
            f"keywords={','.join(entry.keywords)} "
            f"saved_at={entry.saved_at}"
        )
        next_bytes = total_bytes + 1 + len(line.encode("utf-8"))
        if len(lines) >= _MEMORY_INDEX_MAX_LINES or next_bytes > _MEMORY_INDEX_MAX_BYTES:
            break
        lines.append(line)
        total_bytes = next_bytes
    return lines


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
        workspace_root: Path | None = None,
        user_home: Path | None = None,
    ) -> None:
        self._store = store
        self._router = router
        self._executor = executor
        self._llm = llm
        self._trace_dir = trace_dir
        self._planner = planner
        self._workspace_root = workspace_root or Path.cwd()
        self._user_home = user_home

    async def chat(self, session_id: str, user_input: str) -> str:
        from runtime.router import Route

        session = self._store.load(session_id)
        session.fsm_state = STATE_IDLE
        claude_context = load_claude_context(self._workspace_root, self._user_home)

        trace = TraceLogger(self._trace_dir / f"{Path(session_id).name}.jsonl")
        try:
            route = await self._router.classify(user_input, session.memory)
            trace.log_route(route.value)

            if route == Route.DIRECT:
                session.fsm_state = STATE_WAITING
                sys_msg = build_system_prompt(session.memory)
                messages = [{"role": "system", "content": sys_msg}]
                memory_msg = build_memory_context_message(
                    session.memory, claude_context=claude_context
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
                outcome = await self._executor.run(
                    session, user_input, trace, claude_context=claude_context
                )
                session.fsm_state = STATE_WAITING
                if outcome.needs_replan:
                    # No replanner now; log so the failure is not silently swallowed.
                    trace.log_truncated()
                answer = outcome.text

            else:  # PLAN_REQUIRED
                session.fsm_state = STATE_PLANNING
                plan = await self._planner.make_plan(
                    user_input, session.memory, claude_context=claude_context
                )
                session.memory.plan = plan
                session.fsm_state = STATE_EXECUTING

                # Planner makes the initial decomposition. Failed steps are NOT
                # re-planned; their outcome (incl. ERROR) is handed to synthesis
                # so the model knows a step failed (replacement for the deleted
                # Replanner safety net).
                results: dict[int, object] = {}
                for i in range(len(plan)):
                    outcome = await self._executor.run(
                        session, plan[i].prompt, trace, claude_context=claude_context
                    )
                    results[i] = outcome
                    if outcome.needs_replan:
                        trace.log_truncated()

                answer = await self._llm.synthesize(
                    [s.prompt for s in plan],
                    {str(idx): _synthesize_entry(idx, o) for idx, o in results.items()},
                    claude_context,
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
