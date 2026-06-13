import inspect
from typing import ClassVar

from runtime.rewoo import DagNode, ReWOO
from session.models import Memory, Session
from tools.base import ToolRegistry


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def respond(self, messages, user_input):
        self.calls += 1
        return self._replies.pop(0)


class EchoTool:
    name = "echo"
    description = "echo"
    parameters: ClassVar = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def run(self, args, session):
        return f"echo:{args.get('text')}"


def _registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


def test_substitute_vars():
    n = DagNode(id="E2", tool="echo", args={"text": "${E1} and ${E1}"}, deps=["E1"])
    out = ReWOO._substitute(n.args, {"E1": "X"})
    assert out == {"text": "X and X"}


def test_substitute_no_refs():
    out = ReWOO._substitute({"text": "plain"}, {})
    assert out == {"text": "plain"}


def test_plan_dag_parses_nodes():
    llm = ScriptedLLM([
        '{"nodes":[{"id":"E1","tool":"echo","args":{"text":"a"},"deps":[]},'
        '{"id":"E2","tool":"echo","args":{"text":"${E1}"},"deps":["E1"]}]}'
    ])
    rw = ReWOO(llm=llm, registry=_registry())
    nodes = rw._plan_dag("task")
    assert [n.id for n in nodes] == ["E1", "E2"]
    assert nodes[1].args == {"text": "${E1}"}
    assert nodes[1].deps == ["E1"]


def test_plan_dag_empty_when_unparseable():
    llm = ScriptedLLM(["no json here"])
    rw = ReWOO(llm=llm, registry=_registry())
    assert rw._plan_dag("task") == []


async def test_run_executes_and_solves(tmp_path):
    from trace.logger import TraceLogger

    llm = ScriptedLLM([
        '{"nodes":[{"id":"E1","tool":"echo","args":{"text":"hello"},"deps":[]}]}',
        '{"answer":"final synthesis","evidence_sufficient":true}',
    ])
    rw = ReWOO(llm=llm, registry=_registry())
    s = Session(id="s")
    out = await rw.run(s, Memory(), "do task", 0, TraceLogger(tmp_path / "t.jsonl"))
    assert out.text == "final synthesis"
    assert out.needs_replan is False
    assert s.memory.workspace.get("E1") == "echo:hello"


async def test_run_insufficient_evidence_triggers_replan(tmp_path):
    from trace.logger import TraceLogger

    llm = ScriptedLLM([
        '{"nodes":[{"id":"E1","tool":"echo","args":{"text":"x"},"deps":[]}]}',
        '{"answer":"not enough info","evidence_sufficient":false}',
    ])
    rw = ReWOO(llm=llm, registry=_registry())
    out = await rw.run(Session(id="s"), Memory(), "do task", 0, TraceLogger(tmp_path / "t.jsonl"))
    assert out.needs_replan is True
    assert out.text == "not enough info"


async def test_run_empty_dag_falls_back(tmp_path):
    from trace.logger import TraceLogger

    llm = ScriptedLLM(["no json", '{"answer":"empty","evidence_sufficient":true}'])
    rw = ReWOO(llm=llm, registry=_registry())
    out = await rw.run(Session(id="s"), Memory(), "task", 0, TraceLogger(tmp_path / "t.jsonl"))
    assert out.text == "empty"
    assert out.needs_replan is False


def test_run_is_async():
    assert inspect.iscoroutinefunction(ReWOO.run)
