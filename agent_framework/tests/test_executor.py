from __future__ import annotations

import inspect
from typing import Any, ClassVar

from llm.client import LLMResponse, ToolCallResult
from runtime.executor import Executor
from runtime.reflexion import Lesson, Reflexion
from session.models import MemoryEntry, Session
from tools.base import ToolRegistry
from trace.logger import TraceLogger


class FakeLLM:
    """chat_with_tools 按队列返 LLMResponse。"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)


class FakeReflexion(Reflexion):
    """绕过真 LLM: reflect 直接返固定 Lesson."""

    def __init__(self) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]

    async def reflect(self, call, error, memory) -> Lesson:  # type: ignore[override]
        return Lesson(text="fake lesson", reflexion_exhausted=False)


class ExhaustedReflexion(Reflexion):
    """Always reports reflexion exhausted -> executor must signal needs_replan."""

    def __init__(self) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]

    async def reflect(self, call, error, memory) -> Lesson:  # type: ignore[override]
        return Lesson(text="exhausted lesson", reflexion_exhausted=True)


class EchoTool:
    name = "echo"
    description = "echo back the text"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, args, session) -> str:
        return f"echo:{args.get('text')}"


def _tc(name: str, args: dict, tid: str = "c1") -> ToolCallResult:
    return ToolCallResult(id=tid, name=name, args=args)


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


def _trace(tmp_path) -> TraceLogger:
    return TraceLogger(tmp_path / "t.jsonl")


async def test_no_tool_call_returns_text(tmp_path):
    ex = Executor(
        llm=FakeLLM([LLMResponse(text="hello", tool_calls=[])]),
        registry=_registry(),
        reflexion=FakeReflexion(),
        max_steps=5,
    )
    out = await ex.run(Session(id="s"), "hi", _trace(tmp_path))
    assert out.text == "hello"
    assert out.needs_replan is False


async def test_executor_injects_memory_context_before_user_prompt(tmp_path):
    llm = FakeLLM([LLMResponse(text="hello", tool_calls=[])])
    ex = Executor(
        llm=llm,
        registry=_registry(),
        reflexion=FakeReflexion(),
        max_steps=5,
    )
    s = Session(id="s")
    s.memory.entries = [
        MemoryEntry(
            id="mem-1",
            type="project",
            name="agent-framework",
            description="Phase1 memory index",
            keywords=["memory", "index"],
            content="hidden",
            saved_at="2026-06-25T10:00:00+08:00",
        )
    ]
    out = await ex.run(s, "hi", _trace(tmp_path))
    assert out.text == "hello"
    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Memory index:" in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "hi"


async def test_one_tool_call_then_done(tmp_path):
    ex = Executor(
        llm=FakeLLM(
            [
                LLMResponse(text="", tool_calls=[_tc("echo", {"text": "A"})]),
                LLMResponse(text="got A", tool_calls=[]),
            ]
        ),
        registry=_registry(),
        reflexion=FakeReflexion(),
        max_steps=5,
    )
    s = Session(id="s")
    out = await ex.run(s, "echo A", _trace(tmp_path))
    assert out.text == "got A"
    assert any(m.role == "tool" for m in s.messages)


async def test_max_steps_truncation(tmp_path):
    loop_resp = LLMResponse(text="", tool_calls=[_tc("echo", {"text": "x"})])
    ex = Executor(
        llm=FakeLLM([loop_resp, loop_resp, loop_resp]),
        registry=_registry(),
        reflexion=FakeReflexion(),
        max_steps=2,
    )
    out = await ex.run(Session(id="s"), "loop", _trace(tmp_path))
    # spec 3.2: truncation signals needs_replan so Agent may replan
    assert out.needs_replan is True
    assert "truncated" in out.text.lower()


async def test_tool_error_triggers_reflexion(tmp_path):
    class BoomTool:
        name = "boom"
        description = "always errors"
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

        async def run(self, args, session) -> str:
            raise RuntimeError("boom!")

    reg = _registry()
    reg.register(BoomTool())
    ex = Executor(
        llm=FakeLLM(
            [
                LLMResponse(text="", tool_calls=[_tc("boom", {})]),
                LLMResponse(text="recovered", tool_calls=[]),
            ]
        ),
        registry=reg,
        reflexion=FakeReflexion(),
        max_steps=5,
    )
    s = Session(id="s")
    out = await ex.run(s, "go", _trace(tmp_path))
    assert "fake lesson" in s.memory.lessons
    assert out.text == "recovered"


def test_executor_is_async():
    assert inspect.iscoroutinefunction(Executor.run)


async def test_multiple_tool_calls_one_round(tmp_path):
    ex = Executor(
        llm=FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[
                        _tc("echo", {"text": "A"}, tid="c1"),
                        _tc("echo", {"text": "B"}, tid="c2"),
                    ],
                ),
                LLMResponse(text="both done", tool_calls=[]),
            ]
        ),
        registry=_registry(),
        reflexion=FakeReflexion(),
        max_steps=5,
    )
    s = Session(id="s")
    out = await ex.run(s, "echo A and B", _trace(tmp_path))
    assert out.text == "both done"
    tool_msgs = [m for m in s.messages if m.role == "tool"]
    assert len(tool_msgs) == 2


async def test_tool_error_exhausted_triggers_replan(tmp_path):
    class BoomTool:
        name = "boom"
        description = "always errors"
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

        async def run(self, args, session) -> str:
            raise RuntimeError("boom!")

    reg = _registry()
    reg.register(BoomTool())
    ex = Executor(
        llm=FakeLLM([LLMResponse(text="", tool_calls=[_tc("boom", {})])]),
        registry=reg,
        reflexion=ExhaustedReflexion(),
        max_steps=5,
    )
    s = Session(id="s")
    out = await ex.run(s, "go", _trace(tmp_path))
    assert out.needs_replan is True
    assert "exhausted lesson" in s.memory.lessons
    # Contract C: tool message persisted (not orphaned) even on early return
    assert any(m.role == "tool" for m in s.messages)
