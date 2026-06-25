from __future__ import annotations

import inspect

from llm.client import LLMResponse
from runtime.agent import Agent
from runtime.executor import Executor
from runtime.planner import Planner
from runtime.reflexion import Reflexion
from runtime.router import Route, Router
from session.models import MemoryEntry
from session.store import Store
from tools.base import ToolRegistry


class FakeLLM:
    def __init__(self, responds=None, chats=None):
        self._responds = list(responds or [])
        self._chats = list(chats or [])
        self.respond_calls: list[dict[str, object]] = []
        self.chat_calls: list[dict[str, object]] = []

    def respond(self, messages, user_input):
        self.respond_calls.append({"messages": messages, "user_input": user_input})
        return self._responds.pop(0)

    def chat_with_tools(self, messages, tools):
        self.chat_calls.append({"messages": messages, "tools": tools})
        return self._chats.pop(0)

    def synthesize(self, plan, results, claude_context: str = ""):
        return f"synth:{len(plan)}:{len(results)}"


class FixedRouter(Router):
    def __init__(self, route: Route) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]
        self._route = route

    async def classify(self, user_input, memory):  # type: ignore[override]
        return self._route


class ScriptedExecutor(Executor):
    """Returns queued Outcomes in order; records prompts seen."""

    def __init__(self, outcomes) -> None:
        super().__init__(
            llm=None,  # type: ignore[arg-type]
            registry=ToolRegistry(),
            reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
            max_steps=1,
        )
        self._outcomes = list(outcomes)
        self.prompts: list[str] = []

    async def run(self, session, prompt, trace, claude_context: str = ""):  # type: ignore[override]
        self.prompts.append(prompt)
        return self._outcomes.pop(0)


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


async def test_direct_path(tmp_path):
    llm = FakeLLM(responds=["hello world"])
    store = Store(tmp_path)
    session = store.load("s1")
    session.memory.entries = [
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
    store.save(session)
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    out = await agent.chat("s1", "hi")
    assert out == "hello world"
    messages = llm.respond_calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Memory index:" in messages[1]["content"]
    assert len(messages) == 2
    assert llm.respond_calls[0]["user_input"] == "hi"
    s = agent._store.load("s1")
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "hello world"
    assert s.messages[0].role == "user"


async def test_simple_tool_path(tmp_path):
    llm = FakeLLM(chats=[LLMResponse(text="done", tool_calls=[])])
    agent = _build_agent(tmp_path, llm, Route.SIMPLE_TOOL)
    out = await agent.chat("s2", "do it")
    assert out == "done"
    s = agent._store.load("s2")
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "done"


async def test_plan_required_runs_steps(tmp_path):
    # planner.respond -> step JSON; each step executor.chat -> text; synthesize
    llm = FakeLLM(
        responds=['{"steps": ["do A", "do B"]}'],
        chats=[
            LLMResponse(text="A done", tool_calls=[]),
            LLMResponse(text="B done", tool_calls=[]),
        ],
    )
    agent = _build_agent(tmp_path, llm, Route.PLAN_REQUIRED)
    out = await agent.chat("s3", "plan X")
    assert out == "synth:2:2"
    s = agent._store.load("s3")
    assert len(s.memory.plan) == 2
    assert [st.prompt for st in s.memory.plan] == ["do A", "do B"]
    # synthesized answer persisted (Contract C: agent appends only the final answer)
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "synth:2:2"


async def test_plan_failed_step_marked_in_synthesis(tmp_path):
    """No replanner now; a failed step's outcome is surfaced to synthesis
    (and the agent appends a truncated trace marker), not silently swallowed."""
    from runtime.executor import Outcome

    llm = FakeLLM(responds=['{"steps": ["step A"]}'])
    ex = ScriptedExecutor([Outcome(text="fail", needs_replan=True)])
    agent = _build_agent(tmp_path, llm, Route.PLAN_REQUIRED, executor=ex)
    out = await agent.chat("s4", "go")
    assert ex.prompts == ["step A"]
    # synthesize still runs; the single failed step is its only result
    assert out == "synth:1:1"


async def test_memory_persists_across_turns(tmp_path):
    llm = FakeLLM(responds=["a1", "a2"])
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    await agent.chat("s6", "first")
    await agent.chat("s6", "second")
    s = agent._store.load("s6")
    users = [m for m in s.messages if m.role == "user"]
    assistants = [m for m in s.messages if m.role == "assistant"]
    assert len(users) == 2
    assert len(assistants) == 2


def test_chat_is_async():
    assert inspect.iscoroutinefunction(Agent.chat)
