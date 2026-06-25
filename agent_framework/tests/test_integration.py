"""Regression tests: cross-path persistence, Contract C end-to-end.

These cover behaviors the unit tests (test_agent / test_executor) missed:
the FULL atomic 4-message sequence + id pairing, cross-path user-landing
mutual exclusion, build_system_prompt purity, and the reload invariant that
prevents orphan tool messages.

All LLM calls are mocked (FakeLLM / ScriptedExecutor).
"""

from __future__ import annotations

from typing import Any, ClassVar

from llm.client import LLMResponse, ToolCallResult
from runtime.agent import Agent, build_system_prompt
from runtime.executor import Executor
from runtime.planner import Planner
from runtime.reflexion import Reflexion
from runtime.router import Route, Router
from session.models import Memory, Step, TodoItem
from session.store import Store
from tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Shared mocks (mirrors test_agent.py patterns; kept local for self-containment)
# ---------------------------------------------------------------------------


class FakeLLM:
    """Records nothing; pops queued responds/chats for deterministic output."""

    def __init__(self, responds=None, chats=None):
        self._responds = list(responds or [])
        self._chats = list(chats or [])

    def respond(self, messages, user_input):
        return self._responds.pop(0)

    def chat_with_tools(self, messages, tools):
        return self._chats.pop(0)

    def synthesize(self, plan, results):
        return f"synth:{len(plan)}:{len(results)}"


class FixedRouter(Router):
    def __init__(self, route: Route) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]
        self._route = route

    async def classify(self, user_input, memory):  # type: ignore[override]
        return self._route


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


def _build_agent(tmp_path, llm, route: Route, executor=None) -> Agent:
    return Agent(
        store=Store(tmp_path),
        router=FixedRouter(route),
        executor=executor
        or Executor(
            llm=llm,
            registry=ToolRegistry(),
            reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
            max_steps=5,
        ),
        llm=llm,
        trace_dir=tmp_path,
        planner=Planner(llm=llm),
    )


# ---------------------------------------------------------------------------
# Test 1: Contract C 4-message persistence (SIMPLE_TOOL)
# ---------------------------------------------------------------------------


async def test_contract_c_four_message_sequence(tmp_path):
    """After a SIMPLE_TOOL turn with one real tool_call, session.messages
    holds exactly user -> assistant(tool_calls) -> tool -> assistant(final),
    and the tool message's tool_call_id matches an id in the preceding
    assistant's tool_calls.
    """
    reg = ToolRegistry()
    reg.register(EchoTool())
    llm = FakeLLM(
        chats=[
            LLMResponse(
                text="", tool_calls=[_tc("echo", {"text": "hi"}, tid="tc-001")]
            ),
            LLMResponse(text="all done", tool_calls=[]),
        ]
    )
    ex = Executor(
        llm=llm,
        registry=reg,
        reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
        max_steps=5,
    )
    agent = _build_agent(tmp_path, llm, Route.SIMPLE_TOOL, executor=ex)
    out = await agent.chat("s1", "echo hi")
    assert out == "all done"

    s = agent._store.load("s1")
    # system message is injected to the LLM but NOT persisted to session.messages
    roles = [m.role for m in s.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # user message content is the prompt
    assert s.messages[0].content == "echo hi"
    # assistant[1] carries tool_calls
    assert s.messages[1].tool_calls is not None
    assert len(s.messages[1].tool_calls) == 1
    call_id = s.messages[1].tool_calls[0]["id"]
    assert call_id == "tc-001"
    # tool message references the same id
    assert s.messages[2].role == "tool"
    assert s.messages[2].tool_call_id == "tc-001"
    assert s.messages[2].content == "echo:hi"
    # final assistant
    assert s.messages[3].content == "all done"
    assert s.messages[3].tool_calls is None


# ---------------------------------------------------------------------------
# Test 2: DIRECT vs SIMPLE_TOOL user-landing mutual exclusion
# ---------------------------------------------------------------------------


async def test_direct_vs_simple_tool_user_messages_mutually_exclusive(tmp_path):
    """In one session, a DIRECT turn then a SIMPLE_TOOL turn. Each turn's
    user message appears exactly once; DIRECT appends its own user+assistant,
    SIMPLE_TOOL's executor appends a separate user (the step prompt).
    """
    llm = FakeLLM(
        responds=["direct answer"],
        chats=[LLMResponse(text="tool answer", tool_calls=[])],
    )
    # first turn: DIRECT
    agent_direct = _build_agent(tmp_path, llm, Route.DIRECT)
    await agent_direct.chat("s2", "hello direct")

    # second turn: SIMPLE_TOOL (same session id, reloaded from store).
    ex = Executor(
        llm=llm,
        registry=ToolRegistry(),
        reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
        max_steps=5,
    )
    agent_tool = _build_agent(tmp_path, llm, Route.SIMPLE_TOOL, executor=ex)
    await agent_tool.chat("s2", "hello tool")

    s = agent_tool._store.load("s2")
    users = [m for m in s.messages if m.role == "user"]
    assistants = [m for m in s.messages if m.role == "assistant"]
    # exactly 2 user messages, 2 assistant messages, no duplicates
    assert len(users) == 2
    assert len(assistants) == 2
    assert [u.content for u in users] == ["hello direct", "hello tool"]
    assert [a.content for a in assistants] == ["direct answer", "tool answer"]


# ---------------------------------------------------------------------------
# Test 3: build_system_prompt pure function
# ---------------------------------------------------------------------------


def test_build_system_prompt_empty_memory():
    """Empty Memory -> only the base line."""
    prompt = build_system_prompt(Memory())
    assert prompt == "You are a helpful agent."


def test_build_system_prompt_all_sections():
    """Memory with todos + plan + lessons -> all three sections present."""
    mem = Memory(
        todos=[TodoItem(id="1", title="buy milk", status="PLANNED")],
        plan=[Step(prompt="step A"), Step(prompt="step B")],
        lessons=["always validate input types"],
    )
    prompt = build_system_prompt(mem)
    assert "You are a helpful agent." in prompt
    assert "Todos:" in prompt
    assert "[#1] buy milk [PLANNED]" in prompt
    assert "Plan:" in prompt
    assert "step A | step B" in prompt
    assert "Lessons learned:" in prompt
    assert "- always validate input types" in prompt


# ---------------------------------------------------------------------------
# Test 4: PLAN_REQUIRED cross-step persistence (no orphan tool on reload)
# ---------------------------------------------------------------------------


async def test_plan_required_reload_no_orphan_tool(tmp_path):
    """A 2-step PLAN_REQUIRED plan where step 1 does a tool_call. After the
    turn, reload the session from Store and assert every tool message is
    preceded by an assistant whose tool_calls references its tool_call_id.
    This is the Contract-C reload invariant that prevents the DeepSeek 400.
    """
    reg = ToolRegistry()
    reg.register(EchoTool())
    llm = FakeLLM(
        responds=['{"steps": ["echo A", "say done"]}'],
        chats=[
            # step 1: one tool_call then final text
            LLMResponse(
                text="", tool_calls=[_tc("echo", {"text": "A"}, tid="tc-step1")]
            ),
            LLMResponse(text="step1 ok", tool_calls=[]),
            # step 2: text only
            LLMResponse(text="step2 ok", tool_calls=[]),
        ],
    )
    ex = Executor(
        llm=llm,
        registry=reg,
        reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
        max_steps=5,
    )
    agent = _build_agent(tmp_path, llm, Route.PLAN_REQUIRED, executor=ex)
    out = await agent.chat("s4", "go")
    assert out == "synth:2:2"

    # reload from Store (simulates process restart)
    s = agent._store.load("s4")

    # invariant: every tool message must be preceded by an assistant message
    # whose tool_calls list contains an entry with matching id.
    for idx, msg in enumerate(s.messages):
        if msg.role == "tool":
            assert idx > 0, "tool message cannot be first"
            prev = s.messages[idx - 1]
            assert prev.role == "assistant", (
                f"tool at {idx} preceded by {prev.role}, not assistant"
            )
            assert prev.tool_calls is not None, "preceding assistant has no tool_calls"
            ids = {tc["id"] for tc in prev.tool_calls}
            assert msg.tool_call_id in ids, (
                f"tool_call_id {msg.tool_call_id} not in preceding assistant's {ids}"
            )

    # at least one tool message exists (step 1 did a real call)
    tool_msgs = [m for m in s.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    # final assistant is the synthesized answer
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "synth:2:2"
