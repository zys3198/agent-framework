from __future__ import annotations

import inspect

from llm.client import LLMResponse
from runtime.agent import Agent
from runtime.executor import Executor
from runtime.reflexion import Reflexion
from runtime.router import Route, Router
from session.store import Store
from tools.base import ToolRegistry


class FakeLLM:
    def __init__(self, responds=None, chats=None):
        self._responds = list(responds or [])
        self._chats = list(chats or [])

    def respond(self, messages, user_input):
        return self._responds.pop(0)

    def chat_with_tools(self, messages, tools):
        return self._chats.pop(0)


class FixedRouter(Router):
    """Bypass LLM classification; return a fixed route."""

    def __init__(self, route: Route) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]
        self._route = route

    async def classify(self, user_input, memory):  # type: ignore[override]
        return self._route


def _build_agent(tmp_path, llm, route: Route) -> Agent:
    executor = Executor(
        llm=llm,
        registry=ToolRegistry(),
        reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
        max_steps=5,
    )
    return Agent(
        store=Store(tmp_path),
        router=FixedRouter(route),
        executor=executor,
        llm=llm,
        trace_dir=tmp_path,
    )


async def test_direct_path(tmp_path):
    llm = FakeLLM(responds=["hello world"])
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    out = await agent.chat("s1", "hi")
    assert out == "hello world"
    s = agent._store.load("s1")
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "hello world"
    assert s.messages[0].role == "user"  # user persisted


async def test_simple_tool_path(tmp_path):
    llm = FakeLLM(chats=[LLMResponse(text="done", tool_calls=[])])
    agent = _build_agent(tmp_path, llm, Route.SIMPLE_TOOL)
    out = await agent.chat("s2", "do it")
    assert out == "done"
    s = agent._store.load("s2")
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "done"


async def test_plan_required_falls_back_to_executor(tmp_path):
    llm = FakeLLM(chats=[LLMResponse(text="handled as tool", tool_calls=[])])
    agent = _build_agent(tmp_path, llm, Route.PLAN_REQUIRED)
    out = await agent.chat("s3", "plan X")
    assert out == "handled as tool"


async def test_memory_persists_across_turns(tmp_path):
    llm = FakeLLM(responds=["a1", "a2"])
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    await agent.chat("s4", "first")
    await agent.chat("s4", "second")
    s = agent._store.load("s4")
    users = [m for m in s.messages if m.role == "user"]
    assistants = [m for m in s.messages if m.role == "assistant"]
    assert len(users) == 2
    assert len(assistants) == 2


def test_chat_is_async():
    assert inspect.iscoroutinefunction(Agent.chat)
