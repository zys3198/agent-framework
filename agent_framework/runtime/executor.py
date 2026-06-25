from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.reflexion import Reflexion
    from session.models import Session
    from tools.base import ToolRegistry
    from trace.logger import TraceLogger

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    text: str
    needs_replan: bool = False


class Executor:
    """Function-calling loop: LLM <-> tool dispatch <-> result write-back.

    On tool error, triggers Reflexion to learn a lesson, appends it to
    memory, and continues. Returns Outcome(text, needs_replan).
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        reflexion: Reflexion,
        max_steps: int,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._reflexion = reflexion
        self.max_steps = max_steps

    async def run(
        self, session: Session, prompt: str, trace: TraceLogger, claude_context: str = ""
    ) -> Outcome:
        from runtime.agent import build_memory_context_message, build_system_prompt
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
        session.messages.append(Message(role="user", content=prompt))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(session.memory)}
        ]
        memory_msg = build_memory_context_message(
            session.memory, claude_context=claude_context
        )
        if memory_msg is not None:
            messages.append(memory_msg)
        messages.extend(m.to_dict() for m in session.messages)
        tools = self._registry.schemas()

        for step in range(self.max_steps):
            trace.log_step(step)
            resp = await asyncio.to_thread(self._llm.chat_with_tools, messages, tools)
            trace.log_llm_call(step, [t.name for t in resp.tool_calls])

            if not resp.tool_calls:
                session.messages.append(Message(role="assistant", content=resp.text))
                return Outcome(text=resp.text, needs_replan=False)

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
                    return Outcome(text=result, needs_replan=True)

        trace.log_truncated()
        session.messages.append(Message(role="assistant", content="(truncated)"))
        return Outcome(text="(truncated)", needs_replan=True)

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
