from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.recaller import Recaller
    from runtime.reflexion import Reflexion
    from session.models import Session
    from tools.base import ToolRegistry
    from trace.logger import TraceLogger

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    text: str
    needs_replan: bool = False
    message_count: int = 0
    lesson_count: int = 0


Outcome = ExecutionResult


class Executor:
    """Function-calling loop: LLM <-> tool dispatch <-> result write-back.

    On tool error, triggers Reflexion to learn a lesson, appends it to
    memory, and continues. Returns ExecutionResult(text, needs_replan, message_count, lesson_count).
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        reflexion: Reflexion,
        max_steps: int,
        recaller: Recaller | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._reflexion = reflexion
        self.max_steps = max_steps
        self._recaller = recaller

    async def run(
        self, session: Session, prompt: str, trace: TraceLogger, project_context: str = ""
    ) -> ExecutionResult:
        from datetime import UTC, datetime

        from runtime.memory_projector import (
            build_memory_context_message,
            build_system_prompt,
        )
        from session.models import Message
        from tools.base import ToolCall

        # Contract C: executor is the sole writer of session.messages on the
        # SIMPLE_TOOL path. Persist the user turn first, then build the LLM
        # message list from session.messages -- so the full sequence
        # (user -> assistant[tool_calls] -> tool -> ... -> assistant) is atomic
        # and survives reload. OpenAI/DeepSeek reject a tool message not
        # preceded by the assistant(tool_calls) that issued it.
        # System prompt is injected here too (was missing on the tool path:
        # memory/todos/lessons were invisible to the model).
        start_message_count = len(session.messages)
        start_lesson_count = len(session.memory.lessons)
        session.messages.append(Message(role="user", content=prompt))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(session.memory)}
        ]
        memory_msg = build_memory_context_message(
            session.memory, project_context=project_context
        )
        if memory_msg is not None:
            messages.append(memory_msg)
        messages.extend(m.to_dict() for m in session.messages)

        # Start parallel recall if recaller is configured and entries exist
        # Guard max_steps>0: otherwise the loop never reaches the await and
        # the task would leak ("Task was destroyed but it is pending").
        recall_task: asyncio.Task[list[str]] | None = None
        if (
            self._recaller is not None
            and session.memory.entries
            and self.max_steps > 0
        ):
            try:
                recall_task = asyncio.create_task(
                    self._recaller.recall(prompt, session.memory.entries)
                )
            except Exception:  # don't let recall setup abort the main path
                recall_task = None

        tools = self._registry.schemas()

        for step in range(self.max_steps):
            trace.log_step(step)
            resp = await self._llm.chat_with_tools(messages, tools)
            trace.log_llm_call(step, [t.name for t in resp.tool_calls])
            # First tool the LLM picks, so the tool-avoidance filter can run
            # on the recall results post-hoc (the parallel recall started
            # before the tool name was known).
            first_tool = (
                resp.tool_calls[0].name if step == 0 and resp.tool_calls else None
            )

            # After first step: inject recall results (parallel recall)
            if step == 0 and recall_task is not None:
                recall_ids = await recall_task
                recall_task = None
                # Dedup: skip ids already in memory index
                if memory_msg and recall_ids:
                    injected_ids: set[str] = set()
                    content = memory_msg.get("content", "") or ""
                    for m in re.finditer(r"id=(\S+)", content):
                        injected_ids.add(m.group(1))
                    recall_ids = [rid for rid in recall_ids if rid not in injected_ids]
                # Tool-avoidance: now that the LLM-chosen tool name is known,
                # exclude "usage" entries for that tool (keep caveat entries).
                # Pure local filter, no second LLM round-trip.
                if recall_ids and first_tool is not None:
                    recall_ids = self._recaller.filter_tool_usage(  # type: ignore[union-attr]
                        recall_ids, session.memory.entries
                    )
                # Read content and inject
                if recall_ids:
                    entry_map = {e.id: e for e in session.memory.entries}
                    now = datetime.now(UTC)
                    recall_lines: list[str] = ["", "---", "Recalled from memory:"]
                    for rid in recall_ids:
                        entry = entry_map.get(rid)
                        if entry is None:
                            continue
                        try:
                            saved = datetime.fromisoformat(entry.saved_at)
                            if saved.tzinfo is None:
                                saved = saved.replace(tzinfo=UTC)
                            delta_days = (now - saved).days
                            if delta_days == 0:
                                age = "saved today"
                            elif delta_days == 1:
                                age = "saved yesterday"
                            else:
                                age = f"saved {delta_days} days ago"
                        except (ValueError, TypeError):
                            age = ""
                        recall_lines.append(
                            f"- {entry.name}: {entry.content} "
                            f"({age}, verify before acting)"
                        )
                    messages.append(
                        {"role": "user", "content": "\n".join(recall_lines)}
                    )

            if not resp.tool_calls:
                session.messages.append(Message(role="assistant", content=resp.text))
                return ExecutionResult(
                    text=resp.text,
                    needs_replan=False,
                    message_count=len(session.messages) - start_message_count,
                    lesson_count=len(session.memory.lessons) - start_lesson_count,
                )

            tool_calls_payload = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.text or "",
                    "tool_calls": tool_calls_payload,
                }
            )
            session.messages.append(
                Message(
                    role="assistant",
                    content=resp.text or "",
                    tool_calls=tool_calls_payload,
                )
            )

            for tc in resp.tool_calls:
                trace.log_tool_call(step, tc.name, tc.args)
                exhausted = False
                try:
                    result = await self._registry.dispatch(
                        ToolCall(name=tc.name, args=tc.args), session
                    )
                except Exception as e:  # tool must not crash executor
                    result = f"ERROR: {e}"
                    log.warning("tool %s raised: %s", tc.name, e)
                    lesson = await self._reflexion.reflect(tc, result, session.memory)
                    session.memory.lessons.append(lesson.text)
                    trace.log_reflexion(step, lesson.text)
                    exhausted = lesson.reflexion_exhausted
                # Contract C: append the tool message BEFORE any early return so
                # reloaded sessions never hold an orphan tool message (API 400).
                trace.log_tool_result(step, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                session.messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )
                if exhausted:
                    # Replanner is gone (Phase 0). Flush the remaining tool
                    # messages in this batch so no orphan tool message survives
                    # a reload, then surface needs_replan so the Agent can mark
                    # the failed step in synthesis instead of silently
                    # swallowing it.
                    self._flush_pending_tools(
                        resp.tool_calls, tc.id, step, trace, session, messages
                    )
                    log.warning(
                        "step %d: reflexion exhausted on tool %s; needs_replan",
                        step,
                        tc.name,
                    )
                    return ExecutionResult(
                        text=result,
                        needs_replan=True,
                        message_count=len(session.messages) - start_message_count,
                        lesson_count=len(session.memory.lessons) - start_lesson_count,
                    )

        trace.log_truncated()
        session.messages.append(Message(role="assistant", content="(truncated)"))
        return ExecutionResult(
            text="(truncated)",
            needs_replan=True,
            message_count=len(session.messages) - start_message_count,
            lesson_count=len(session.memory.lessons) - start_lesson_count,
        )

    def _flush_pending_tools(
        self,
        all_calls: list[Any],
        after_id: str,
        step: int,
        trace: TraceLogger,
        session: Session,
        messages: list[dict[str, Any]],
    ) -> None:
        """Append tool-result messages for calls AFTER the one that triggered
        the early return, so the persisted sequence stays atomic (no orphan
        tool). Calls before/including after_id are already flushed."""
        from session.models import Message

        seen = False
        for tc in all_calls:
            if seen:
                result = f"ERROR: tool {tc.name} skipped (prior step exhausted)"
                trace.log_tool_call(step, tc.name, tc.args)
                trace.log_tool_result(step, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                session.messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )
            if tc.id == after_id:
                seen = True
