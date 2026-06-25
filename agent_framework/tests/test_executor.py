from __future__ import annotations

import asyncio
import inspect
import time
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


class FakeRecaller:
    def __init__(self, return_ids: list[str]) -> None:
        self._return_ids = list(return_ids)
        self.calls: list[dict[str, object]] = []

    async def recall(self, query: str, entries: list, current_tool: str | None = None) -> list[str]:
        self.calls.append({"query": query, "entries": entries, "current_tool": current_tool})
        return list(self._return_ids)


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
    assert any(m.role == "tool" for m in s.messages)


async def test_executor_recall_calls_recaller(tmp_path):
    """recaller.recall 被调用, 结果用于 dedup."""
    s = Session(id="s")
    s.memory.entries = [
        MemoryEntry(id="mem-1", type="user", name="test", description="test entry",
                     content="important fact", saved_at="2026-06-25T00:00:00+00:00"),
    ]
    llm = FakeLLM([LLMResponse(text="final", tool_calls=[])])
    rec = FakeRecaller(["mem-1"])
    ex = Executor(llm=llm, registry=_registry(), reflexion=FakeReflexion(), max_steps=5, recaller=rec)
    out = await ex.run(s, "hi", _trace(tmp_path))
    assert out.text == "final"
    assert len(rec.calls) == 1
    assert rec.calls[0]["query"] == "hi"
    assert len(rec.calls[0]["entries"]) == 1
    # 注入结果已在 index 中 → 被 dedup 过滤, session.messages 无污染
    all_msg_text = " ".join(m.content for m in s.messages)
    assert "important fact" not in all_msg_text


async def test_executor_recall_dedup_filters_indexed(tmp_path):
    """已在 memory index 中的 id 不被注入."""
    s = Session(id="s")
    s.memory.entries = [
        MemoryEntry(id="mem-1", type="user", name="a", description="", content="c1",
                     saved_at="2026-06-25T00:00:00+00:00"),
        MemoryEntry(id="mem-2", type="user", name="b", description="", content="c2",
                     saved_at="2026-06-25T01:00:00+00:00"),
    ]
    llm = FakeLLM([LLMResponse(text="", tool_calls=[_tc("echo", {"text": "x"})]),
                   LLMResponse(text="done", tool_calls=[])])
    ex = Executor(llm=llm, registry=_registry(), reflexion=FakeReflexion(), max_steps=5,
                  recaller=FakeRecaller(["mem-1", "mem-2"]))
    await ex.run(s, "hi", _trace(tmp_path))
    # 两 id 都在 index 中 → LLM requests 中不应有额外注入内容
    for call in llm.calls:
        texts = " ".join(m.get("content", "") for m in call["messages"])
        assert "Recalled from memory" not in texts
    # session.messages 也无注入
    assert all("c1" not in m.content and "c2" not in m.content for m in s.messages)


async def test_executor_recall_not_in_session_messages(tmp_path):
    """召回注入内容不入 session.messages."""
    s = Session(id="s")
    s.memory.entries = [
        MemoryEntry(id="mem-1", type="user", name="secret", description="hidden",
                     content="should not persist", saved_at="2026-06-25T00:00:00+00:00"),
    ]
    llm = FakeLLM([LLMResponse(text="done", tool_calls=[])])
    ex = Executor(llm=llm, registry=_registry(), reflexion=FakeReflexion(), max_steps=5,
                  recaller=FakeRecaller(["mem-1"]))
    await ex.run(s, "hi", _trace(tmp_path))
    all_msg_text = " ".join(m.content for m in s.messages)
    assert "should not persist" not in all_msg_text


async def test_executor_recall_runs_parallel_with_first_step(tmp_path):
    """recall 与第一轮 chat_with_tools 并行, 总时间小于两者之和."""
    class SlowLLM:
        def __init__(self):
            self.calls = []

        async def chat_with_tools(self, messages, tools):
            self.calls.append({"messages": messages, "tools": tools})
            await asyncio.sleep(0.15)
            return LLMResponse(text="done", tool_calls=[])

    class SlowRecaller:
        async def recall(self, query, entries, current_tool=None):
            await asyncio.sleep(0.15)
            return ["mem-1"]

    s = Session(id="s")
    s.memory.entries = [
        MemoryEntry(id="mem-1", type="user", name="test", description="",
                     content="x", saved_at="2026-06-25T00:00:00+00:00"),
    ]
    ex = Executor(llm=SlowLLM(), registry=_registry(), reflexion=FakeReflexion(),
                  max_steps=5, recaller=SlowRecaller())
    start = time.monotonic()
    await ex.run(s, "hi", _trace(tmp_path))
    elapsed = time.monotonic() - start
    # 0.15 + 0.15 = 0.30 if sequential; parallel ~0.15 + overhead
    assert elapsed < 0.28, f"elapsed={elapsed:.3f}s expected <0.28"


async def test_executor_recall_no_recaller_skips(tmp_path):
    """recaller=None 时不创建 recall_task, 不报错."""
    ex = Executor(
        llm=FakeLLM([LLMResponse(text="hello", tool_calls=[])]),
        registry=_registry(),
        reflexion=FakeReflexion(),
        max_steps=5,
    )
    out = await ex.run(Session(id="s"), "hi", _trace(tmp_path))
    assert out.text == "hello"
