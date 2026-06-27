"""Phase 3 compactor tests -- TDD: threshold triggers, 3-segment chain, safety net."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ctx.compactor import Compactor
from runtime.agent import Agent
from runtime.router import Route
from session.models import Message, Session, Step, TodoItem
from session.store import Store
from tools.base import ToolRegistry

# ---- Layer 1: large result spillover ----


def test_spill_below_threshold_not_triggered(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, large_result_bytes=4096)
    session = Session(id="s1")
    session.messages = [
        Message(role="user", content="query"),
        Message(role="assistant", content="", tool_calls=[{"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
        Message(role="tool", content="x" * 3900, tool_call_id="t1"),
    ]
    c.spill_large_results(session)
    assert len(session.messages) == 3
    assert len(session.messages[2].content) == 3900
    assert not list(tmp_path.glob("*.spill"))


def test_spill_above_threshold_triggered(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, large_result_bytes=4096)
    session = Session(id="s1")
    big = "DATA:" + "y" * 5000
    session.messages = [
        Message(role="user", content="query"),
        Message(role="assistant", content="", tool_calls=[{"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
        Message(role="tool", content=big, tool_call_id="t1"),
    ]
    c.spill_large_results(session)
    tool_msg = session.messages[2]
    assert tool_msg.role == "tool"
    assert len(tool_msg.content) < 200
    assert "spill" in tool_msg.content.lower()
    spills = list(tmp_path.glob("*.spill"))
    assert len(spills) == 1
    assert spills[0].read_text(encoding="utf-8") == big


def test_spill_only_tool_results(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, large_result_bytes=4096)
    session = Session(id="s1")
    session.messages = [
        Message(role="user", content="u" * 5000),
        Message(role="assistant", content="a" * 5000),
    ]
    c.spill_large_results(session)
    assert len(session.messages[0].content) == 5000
    assert len(session.messages[1].content) == 5000
    assert not list(tmp_path.glob("*.spill"))


# ---- Layer 2: microcompact preprocessing ----


def test_microcompact_keeps_recent_n(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, microcompact_keep=2)
    session = Session(id="s1")
    msgs = [Message(role="user", content="q")]
    for i in range(5):
        msgs.append(Message(role="assistant", content="", tool_calls=[{"id": f"t{i}", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]))
        msgs.append(Message(role="tool", content=f"result-{i}", tool_call_id=f"t{i}"))
    session.messages = msgs
    c.microcompact(session)
    tool_msgs = [m for m in session.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[-1].content == "result-4"
    assert tool_msgs[-2].content == "result-3"


def test_microcompact_preserves_user_assistant(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, microcompact_keep=0)
    session = Session(id="s1")
    session.messages = [
        Message(role="user", content="keep-me"),
        Message(role="assistant", content="also-keep"),
        Message(role="assistant", content="", tool_calls=[{"id": "t0", "type": "function", "function": {"name": "x", "arguments": "{}"}}]),
        Message(role="tool", content="r0", tool_call_id="t0"),
    ]
    c.microcompact(session)
    roles = [m.role for m in session.messages]
    assert "user" in roles
    assert "assistant" in roles
    user_msg = next(m for m in session.messages if m.role == "user")
    assert user_msg.content == "keep-me"


def test_microcompact_preserves_state_info(tmp_path):
    """State info (todos/plan/workspace) lives in session.memory, not messages.
    Microcompact must not touch session.memory at all."""
    c = Compactor(AsyncMock(), tmp_path, microcompact_keep=0)
    session = Session(id="s1")
    session.memory.todos = [TodoItem(id="1", title="task")]
    session.memory.plan = [Step(prompt="do X")]
    session.memory.workspace = {"key": "val"}
    session.messages = [
        Message(role="user", content="q"),
        Message(role="assistant", content="", tool_calls=[{"id": "t0", "type": "function", "function": {"name": "x", "arguments": "{}"}}]),
        Message(role="tool", content="r0", tool_call_id="t0"),
    ]
    c.microcompact(session)
    assert session.memory.todos[0].title == "task"
    assert session.memory.plan[0].prompt == "do X"
    assert session.memory.workspace == {"key": "val"}


# ---- Layer 3: Auto-Compact full summary ----


def test_auto_compact_below_threshold(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, auto_compact_tokens=8000)
    session = Session(id="s1")
    session.messages = [Message(role="user", content="short")]
    result = asyncio.run(c.auto_compact(session))
    assert result is None


def test_auto_compact_triggers_summary(tmp_path):
    llm = AsyncMock()
    llm.respond = AsyncMock(return_value="## Primary Request\nDo the thing")
    c = Compactor(llm, tmp_path, auto_compact_tokens=100)
    session = Session(id="s1")
    session.messages = [Message(role="user", content="x" * 500), Message(role="assistant", content="y" * 500)]
    result = asyncio.run(c.auto_compact(session))
    assert result is not None
    contents = [m.content for m in result]
    boundary = [c for c in contents if "compact" in c.lower() or "interrupted" in c.lower()]
    assert len(boundary) >= 1
    summary = [c for c in contents if "Primary Request" in c]
    assert len(summary) >= 1


def test_auto_compact_preserves_attachments(tmp_path):
    llm = AsyncMock()
    llm.respond = AsyncMock(return_value="summary text")
    c = Compactor(llm, tmp_path, auto_compact_tokens=50)
    session = Session(id="s1")
    session.memory.todos = [TodoItem(id="1", title="task", status="IN_PROGRESS")]
    session.memory.plan = [Step(prompt="step")]
    session.memory.workspace = {"path": "/tmp"}
    session.messages = [Message(role="user", content="x" * 200)]
    result = asyncio.run(c.auto_compact(session))
    contents = [m.content for m in result]
    joined = "\n".join(contents)
    assert "task" in joined
    assert "step" in joined


# ---- Safety net ----


def test_circuit_breaker_trips(tmp_path):
    llm = AsyncMock()
    llm.respond = AsyncMock(side_effect=RuntimeError("LLM down"))
    c = Compactor(llm, tmp_path, auto_compact_tokens=50, circuit_breaker_limit=3)
    session = Session(id="s1")
    session.messages = [Message(role="user", content="x" * 200)]
    for _ in range(3):
        asyncio.run(c.auto_compact(session))
    assert c.is_tripped("s1") is True
    result = asyncio.run(c.auto_compact(session))
    assert result is None or result == session.messages


def test_circuit_breaker_isolated_per_session(tmp_path):
    """BLOCKER fix: session A tripping breaker must NOT affect session B."""
    llm = AsyncMock()
    llm.respond = AsyncMock(side_effect=RuntimeError("LLM down"))
    c = Compactor(llm, tmp_path, auto_compact_tokens=50, circuit_breaker_limit=3)
    sA = Session(id="A")
    sA.messages = [Message(role="user", content="x" * 200)]
    sB = Session(id="B")
    sB.messages = [Message(role="user", content="x" * 200)]
    for _ in range(3):
        asyncio.run(c.auto_compact(sA))
    assert c.is_tripped("A") is True
    assert c.is_tripped("B") is False  # B unaffected


def test_recursion_guard(tmp_path):
    c = Compactor(AsyncMock(), tmp_path, auto_compact_tokens=50)
    session = Session(id="s1")
    session.messages = [
        Message(role="user", content="[COMPACT] session continuation from interrupted context"),
        Message(role="assistant", content="summary"),
    ]
    result = asyncio.run(c.auto_compact(session))
    assert result is None


def test_summary_prompt_has_nine_sections(tmp_path):
    llm = AsyncMock()
    llm.respond = AsyncMock(return_value="summary")
    c = Compactor(llm, tmp_path, auto_compact_tokens=50)
    session = Session(id="s1")
    session.messages = [Message(role="user", content="x" * 200)]
    asyncio.run(c.auto_compact(session))
    call_args = llm.respond.call_args
    prompt = call_args[0][1]
    for section in [
        "Primary Request and Intent",
        "Key Technical Concepts",
        "Files and Code Sections",
        "Errors and fixes",
        "Problem Solving",
        "All user messages",
        "Pending Tasks",
        "Current Work",
        "Optional Next Step",
    ]:
        assert section in prompt, f"missing section: {section}"


# ---- Integration: Compactor wired into Agent ----


class _FixedRouter:
    """Router stub that always returns DIRECT -- avoids LLM call for routing."""

    async def classify(self, user_input, memory):
        return Route.DIRECT


def test_agent_compact_called_on_chat(tmp_path):
    """Agent.chat must call compactor.compact() before processing -- proves
    the integration is wired (not ghost code)."""
    from runtime.executor import Executor
    from runtime.planner import Planner
    from runtime.reflexion import Reflexion

    llm = AsyncMock()
    llm.respond = AsyncMock(return_value="answer")
    compactor = AsyncMock()
    compactor.compact = AsyncMock(return_value=False)
    agent = Agent(
        store=Store(tmp_path),
        router=_FixedRouter(),  # type: ignore[arg-type]
        executor=Executor(
            llm=llm,  # type: ignore[arg-type]
            registry=ToolRegistry(),
            reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
            max_steps=1,
        ),
        llm=llm,  # type: ignore[arg-type]
        trace_dir=tmp_path,
        planner=Planner(llm=llm),  # type: ignore[arg-type]
        compactor=compactor,  # type: ignore[arg-type]
    )
    result = asyncio.run(agent.chat("s1", "hi"))
    assert result == "answer"
    compactor.compact.assert_called_once()


def test_agent_concurrent_chat_same_session_serialized(tmp_path):
    """BLOCKER fix: concurrent requests to the same session must serialize
    via per-session asyncio.Lock so load->modify->save is atomic (no lost updates)."""
    from runtime.executor import Executor
    from runtime.planner import Planner
    from runtime.reflexion import Reflexion

    llm = AsyncMock()
    llm.respond = AsyncMock(return_value="ok")
    agent = Agent(
        store=Store(tmp_path),
        router=_FixedRouter(),  # type: ignore[arg-type]
        executor=Executor(
            llm=llm,  # type: ignore[arg-type]
            registry=ToolRegistry(),
            reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
            max_steps=1,
        ),
        llm=llm,  # type: ignore[arg-type]
        trace_dir=tmp_path,
        planner=Planner(llm=llm),  # type: ignore[arg-type]
    )

    async def _run():
        await agent.chat("s1", "init")
        await asyncio.gather(agent.chat("s1", "A"), agent.chat("s1", "B"))

    asyncio.run(_run())
    s = Store(tmp_path).load("s1")
    assert len(s.messages) == 6
    assert sorted(m.content for m in s.messages if m.role == "user") == ["A", "B", "init"]
