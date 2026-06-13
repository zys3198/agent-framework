from __future__ import annotations

import inspect
from typing import ClassVar

from llm.client import LLMResponse
from runtime.agent import Agent
from runtime.executor import Executor
from runtime.planner import Planner
from runtime.reflexion import Reflexion
from runtime.replanner import Replanner
from runtime.router import Route, Router
from session.models import Step
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

    def synthesize(self, plan, results):
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

    async def run(self, session, prompt, trace):  # type: ignore[override]
        self.prompts.append(prompt)
        return self._outcomes.pop(0)


class ScriptedReplanner(Replanner):
    def __init__(self, revised_prompts) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]
        self._revised = list(revised_prompts)
        self.calls = 0

    async def revise(self, remaining, results, memory):  # type: ignore[override]
        self.calls += 1
        return [Step(prompt=p) for p in self._revised]


def _build_agent(
    tmp_path,
    llm,
    route: Route,
    executor=None,
    replanner=None,
    rewoo=None,
    max_replans: int = 2,
) -> Agent:
    from runtime.rewoo import ReWOO

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
        replanner=replanner or Replanner(llm=llm),
        rewoo=rewoo or ReWOO(llm=llm, registry=ToolRegistry()),
        max_replans=max_replans,
    )


async def test_direct_path(tmp_path):
    llm = FakeLLM(responds=["hello world"])
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    out = await agent.chat("s1", "hi")
    assert out == "hello world"
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


async def test_plan_required_runs_planning_loop(tmp_path):
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


async def test_plan_loop_replans_once_then_completes(tmp_path):
    from runtime.executor import Outcome

    llm = FakeLLM(responds=['{"steps": ["step A"]}'])  # planner only
    ex = ScriptedExecutor(
        [Outcome(text="fail", needs_replan=True), Outcome(text="ok", needs_replan=False)]
    )
    rp = ScriptedReplanner(["revised step"])
    agent = _build_agent(
        tmp_path, llm, Route.PLAN_REQUIRED, executor=ex, replanner=rp, max_replans=2
    )
    out = await agent.chat("s4", "go")
    assert rp.calls == 1
    assert ex.prompts == ["step A", "revised step"]
    # synthesize: revised 1-step plan; results keyed by plan index so the
    # re-run overwrites index 0 (fail -> ok) -> 1 result entry
    assert out == "synth:1:1"


async def test_plan_loop_caps_at_max_replans(tmp_path):
    from runtime.executor import Outcome

    llm = FakeLLM(responds=['{"steps": ["step A"]}'])
    ex = ScriptedExecutor([Outcome(text="fail", needs_replan=True)] * 10)
    rp = ScriptedReplanner(["retry step"])
    agent = _build_agent(
        tmp_path, llm, Route.PLAN_REQUIRED, executor=ex, replanner=rp, max_replans=2
    )
    out = await agent.chat("s5", "go")
    # exactly 2 replans, then cap -> continue old plan -> loop terminates
    assert rp.calls == 2
    assert out.startswith("synth:")


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


async def test_plan_loop_runs_rewoo_cluster(tmp_path):
    from runtime.rewoo import ReWOO
    from tools.base import ToolRegistry

    # planner -> rewoo_cluster step; rewoo plan_dag + solver consume responds
    llm = FakeLLM(
        responds=[
            '{"rewoo_cluster": "analyze X and Y"}',
            '{"nodes":[{"id":"E1","tool":"echo","args":{"text":"x"},"deps":[]}]}',
            '{"answer":"rewoven answer","evidence_sufficient":true}',
        ],
    )

    class EchoTool:
        name = "echo"
        description = "echo"
        parameters: ClassVar = {"type": "object", "properties": {"text": {"type": "string"}}}

        async def run(self, args, session):
            return "echo:x"

    reg = ToolRegistry()
    reg.register(EchoTool())
    ex = Executor(
        llm=llm,
        registry=reg,
        reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
        max_steps=5,
    )
    rw = ReWOO(llm=llm, registry=reg)
    agent = Agent(
        store=Store(tmp_path),
        router=FixedRouter(Route.PLAN_REQUIRED),
        executor=ex,
        llm=llm,
        trace_dir=tmp_path,
        planner=Planner(llm=llm),
        replanner=Replanner(llm=llm),
        rewoo=rw,
        max_replans=2,
    )
    out = await agent.chat("s7", "go")
    assert out == "rewoven answer"
    s = agent._store.load("s7")
    assert s.memory.plan[0].is_rewoo_cluster is True
    assert s.memory.workspace.get("E1") == "echo:x"
