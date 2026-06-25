from __future__ import annotations

import inspect

from llm.client import ToolCallResult
from runtime.reflexion import Lesson, Reflexion
from session.models import Memory


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    async def respond(self, messages: list[dict], user_input: str) -> str:
        return self._responses.pop(0)


async def test_reflect_produces_lesson():
    rx = Reflexion(llm=FakeLLM(["next time pass --verbose to the tool"]))
    lesson = await rx.reflect(
        ToolCallResult(id="c1", name="search", args={"query": "x"}),
        "ERROR: no results",
        Memory(),
    )
    assert isinstance(lesson, Lesson)
    assert "verbose" in lesson.text
    assert lesson.reflexion_exhausted is False


async def test_reflect_exhaustion_flag():
    mem = Memory()
    mem.lessons = ["l1", "l2", "l3"]
    rx = Reflexion(llm=FakeLLM(["another lesson"]))
    lesson = await rx.reflect(
        ToolCallResult(id="c1", name="search", args={}),
        "ERROR: failed again",
        mem,
    )
    assert lesson.reflexion_exhausted is True


def test_reflect_is_async():
    assert inspect.iscoroutinefunction(Reflexion.reflect)
