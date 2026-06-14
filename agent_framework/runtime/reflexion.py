from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.client import LLMClient, ToolCallResult
    from session.models import Memory

# 攒到这么多条 lesson 仍失败 -> 判穷尽, 交 S3 REPLANNING 升级
_EXHAUSTION_THRESHOLD = 3


@dataclass
class Lesson:
    text: str
    reflexion_exhausted: bool = False


class Reflexion:
    """On tool failure, ask LLM for a one-line lesson; flag exhaustion."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def reflect(self, call: ToolCallResult, error: str, memory: Memory) -> Lesson:
        prompt = (
            f"A tool call failed.\n"
            f"tool: {call.name}\n"
            f"args: {call.args}\n"
            f"error: {error}\n"
            f"Write ONE short lesson to avoid this next time."
        )
        text = await asyncio.to_thread(
            self._llm.respond,
            [{"role": "system", "content": "You produce concise lessons."}],
            prompt,
        )
        # +1: this lesson is appended by the caller AFTER reflect returns,
        # so anticipate it to trigger exhaustion on the threshold-th failure
        # (not one late).
        exhausted = len(memory.lessons) + 1 >= _EXHAUSTION_THRESHOLD
        return Lesson(text=(text or "").strip(), reflexion_exhausted=exhausted)
