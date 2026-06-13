# S4 ReWOO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add ReWOO (族 C') — a micro-parallel sub-mode embedded in Plan-and-Execute: when the Planner detects an independent/parallelizable step cluster, it emits a single `Step(is_rewoo_cluster=True)`; `ReWOO.run` then does plan-DAG -> worker(bind vars, tool dispatch, no LLM) -> solver(one LLM synthesis), returning an `Outcome`. If the solver judges evidence insufficient, it returns `needs_replan=True`, feeding the existing S3 REPLANNING loop (C' -> D' handoff).

**Architecture:** New `runtime/rewoo.py` owns a 3-phase micro-loop. DAG = `list[DagNode]` (id `E1`, tool, `args: dict[str,str]` with `${E1}` refs, deps). Worker executes nodes in order, substituting `${Ex}` from `memory.workspace`, dispatching each tool via `ToolRegistry` (no LLM). Solver = one LLM call returning `{"answer","evidence_sufficient"}`. `Agent` gets a `rewoo` dep + `is_rewoo_cluster` branch (spec §3.1). `Planner` emits a cluster step when LLM JSON has a `rewoo_cluster` key. Contract C: worker dispatches via the same `ToolRegistry.dispatch` path; tool errors caught as `ERROR:` strings.

**Tech Stack:** Python 3.12, asyncio, dataclasses, DeepSeek via `LLMClient`, pytest-asyncio. ruff (88, E/F/W/I/UP/B/SIM/RUF, ignore E501), mypy --strict, `from __future__ import annotations`, `dict[str, Any]`, half-width ASCII punctuation (no prime).

**Controller-made design decisions (autonomous under /goal):**
- **D1:** DAG node JSON `{"nodes":[{"id":"E1","tool":"search","args":{"query":"x"},"deps":[]}]}`. Var refs `${E1}` inside args values, substituted from `memory.workspace`.
- **D2:** Solver JSON `{"answer":"...","evidence_sufficient":bool}`. false -> `Outcome(needs_replan=True)` (C'->D').
- **D3:** Planner: if LLM JSON has top-level `rewoo_cluster` string -> `[Step(prompt=<that>, is_rewoo_cluster=True)]`; else `_parse_steps`. Backward compatible.
- **D4:** Worker executes nodes in emitted order (LLM instructed to emit deps first). No topo-sort in S4.

**Carry-over:** `log_rewoo_dag(step, nodes, edges)` + `log_rewoo_solve(step, vars, sufficient)` already in TraceLogger. `Memory.workspace` exists. `ToolRegistry.dispatch(ToolCall(name, args), session) -> str` exists. `Outcome(text, needs_replan)` exists.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `agent_framework/runtime/rewoo.py` | Create | `DagNode` + `ReWOO.run` (plan_dag/execute_dag/solve) |
| `agent_framework/runtime/planner.py` | Modify | `_extract_json` + cluster detection |
| `agent_framework/runtime/agent.py` | Modify | Inject `rewoo`; `is_rewoo_cluster` branch |
| `agent_framework/tests/test_rewoo.py` | Create | DAG parse / var subst / solve / insufficient->replan |
| `agent_framework/tests/test_planner.py` | Modify | cluster detection test |
| `agent_framework/tests/test_agent.py` | Modify | rewoo branch test |

Run from `agent_framework/`: `.venv/Scripts/python -m pytest tests/<file>.py -v`

---

## Task 1: `ReWOO` (族 C') + `DagNode`

**Files:** Create `agent_framework/runtime/rewoo.py`; Test `agent_framework/tests/test_rewoo.py`.

- [ ] **Step 1: Write failing tests** (`tests/test_rewoo.py`)

```python
import inspect

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
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

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
```

- [ ] **Step 2: Verify fail** — `.venv/Scripts/python -m pytest tests/test_rewoo.py -v` -> ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `agent_framework/runtime/rewoo.py`:

```python
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Memory, Session
    from tools.base import ToolRegistry
    from trace.logger import TraceLogger

_PLAN_DAG_PROMPT = (
    "Decompose the task into a DAG of tool calls (no observation needed yet).\n"
    'Return ONLY JSON: {"nodes":[{"id":"E1","tool":"<name>","args":{...},"deps":[]}, ...]}.\n'
    "ids are E1, E2, ... in dependency order. A later node may reference an earlier\n"
    "result inside an args value as ${E1}. deps lists the ids it depends on."
)

_SOLVE_PROMPT = (
    "Given the original task and the variable results below, synthesize the final answer.\n"
    'Return ONLY JSON: {"answer":"...","evidence_sufficient":true|false}.\n'
    "evidence_sufficient is false if the results do not justify a confident answer."
)

_VAR = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


@dataclass
class DagNode:
    id: str
    tool: str
    args: dict[str, str]
    deps: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class ReWOO:
    """Micro-parallel sub-mode (族 C'): plan DAG -> worker (tool dispatch, no LLM)
    -> solver (one synthesis). evidence insufficient -> needs_replan (C' -> D')."""

    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self._llm = llm
        self._registry = registry

    @staticmethod
    def _substitute(args: dict[str, str], workspace: dict[str, Any]) -> dict[str, str]:
        def _repl(value: str) -> str:
            return _VAR.sub(lambda m: str(workspace.get(m.group(1), m.group(0))), str(value))

        return {k: _repl(v) for k, v in args.items()}

    def _plan_dag(self, task: str) -> list[DagNode]:
        text = self._llm.respond(
            [{"role": "system", "content": _PLAN_DAG_PROMPT}], task
        )
        data = _extract_json(text or "")
        nodes_raw = data.get("nodes") if data else None
        if not isinstance(nodes_raw, list):
            return []
        out: list[DagNode] = []
        for n in nodes_raw:
            if not isinstance(n, dict) or "id" not in n or "tool" not in n:
                continue
            args = n.get("args") if isinstance(n.get("args"), dict) else {}
            deps = n.get("deps") if isinstance(n.get("deps"), list) else []
            out.append(
                DagNode(
                    id=str(n["id"]),
                    tool=str(n["tool"]),
                    args={k: str(v) for k, v in args.items()},
                    deps=[str(d) for d in deps],
                )
            )
        return out

    async def _execute_dag(
        self, nodes: list[DagNode], session: Session, trace: TraceLogger, step: int
    ) -> dict[str, str]:
        from tools.base import ToolCall

        workspace: dict[str, str] = dict(session.memory.workspace)
        for node in nodes:
            bound = self._substitute(node.args, workspace)
            trace.log_tool_call(step, node.tool, bound)
            try:
                result = await self._registry.dispatch(
                    ToolCall(name=node.tool, args=bound), session
                )
            except Exception as e:  # tool must not crash ReWOO
                result = f"ERROR: {e}"
            trace.log_tool_result(step, result)
            workspace[node.id] = result
            session.memory.workspace[node.id] = result
        return workspace

    def _solve(self, task: str, workspace: dict[str, Any]) -> tuple[str, bool]:
        lines = [f"task: {task}", "results:"]
        lines.extend(f"- {k}: {v}" for k, v in workspace.items())
        text = self._llm.respond(
            [{"role": "system", "content": _SOLVE_PROMPT}], "\n".join(lines)
        )
        data = _extract_json(text or "")
        if not data:
            return (text or "").strip(), True
        answer = str(data.get("answer", text or "")).strip()
        sufficient = bool(data.get("evidence_sufficient", True))
        return answer, sufficient

    async def run(
        self,
        session: Session,
        memory: Memory,
        task: str,
        step: int,
        trace: TraceLogger,
    ) -> Outcome:
        from runtime.executor import Outcome

        nodes = await asyncio.to_thread(self._plan_dag, task)
        trace.log_rewoo_dag(step, [n.id for n in nodes], [n.deps for n in nodes])
        if not nodes:
            answer, sufficient = await asyncio.to_thread(self._solve, task, {})
            trace.log_rewoo_solve(step, [], sufficient)
            return Outcome(text=answer, needs_replan=not sufficient)
        workspace = await self._execute_dag(nodes, session, trace, step)
        answer, sufficient = await asyncio.to_thread(self._solve, task, workspace)
        trace.log_rewoo_solve(step, list(workspace.keys()), sufficient)
        return Outcome(text=answer, needs_replan=not sufficient)
```

NOTE: `Outcome` is referenced as the return annotation but imported inline inside `run` (circular-import avoidance, same pattern as planner/replanner). To satisfy mypy on the annotation, add `if TYPE_CHECKING: from runtime.executor import Outcome` to the TYPE_CHECKING block AND keep the inline runtime import inside `run`. The `memory` param is part of the signature for symmetry; it is not unused because callers pass `session.memory` — but if ruff/mypy flag it, that's fine (it documents intent).

- [ ] **Step 4: Verify pass** — `.venv/Scripts/python -m pytest tests/test_rewoo.py -v` -> all green.

- [ ] **Step 5: Lint + type** (cwd = agent_framework/, RELATIVE paths)
- `.venv/Scripts/python -m ruff check runtime/rewoo.py tests/test_rewoo.py`
- `.venv/Scripts/python -m mypy runtime/rewoo.py`
Clean. Fix any unused-import ruff flags.

- [ ] **Step 6: Stage** — `git add agent_framework/runtime/rewoo.py agent_framework/tests/test_rewoo.py`

---

## Task 2: Planner cluster detection

**Files:** Modify `agent_framework/runtime/planner.py`; Test `agent_framework/tests/test_planner.py`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_planner.py`):

```python
async def test_make_plan_detects_rewoo_cluster():
    llm = ScriptedLLM('{"rewoo_cluster": "analyze X and Y in parallel"}')
    planner = Planner(llm)
    plan = await planner.make_plan("compare X and Y", Memory())
    assert len(plan) == 1
    assert plan[0].is_rewoo_cluster is True
    assert plan[0].prompt == "analyze X and Y in parallel"


async def test_make_plan_plain_steps_when_no_cluster_key():
    llm = ScriptedLLM('{"steps": ["a", "b"]}')
    planner = Planner(llm)
    plan = await planner.make_plan("do a then b", Memory())
    assert [s.prompt for s in plan] == ["a", "b"]
    assert all(not s.is_rewoo_cluster for s in plan)
```

- [ ] **Step 2: Verify fail** — `.venv/Scripts/python -m pytest tests/test_planner.py::test_make_plan_detects_rewoo_cluster -v` -> FAIL (no cluster detected).

- [ ] **Step 3: Implement** — in `runtime/planner.py`:
(a) Change the typing import line `from typing import TYPE_CHECKING` to `from typing import TYPE_CHECKING, Any`.
(b) Add `_extract_json` after `_parse_steps`:
```python
def _extract_json(text: str) -> dict[str, Any] | None:
    import json
    import re

    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
```
(c) Rewrite `make_plan`:
```python
    async def make_plan(self, user_input: str, memory: Memory) -> list[Step]:
        from session.models import Step

        text = await asyncio.to_thread(
            self._llm.respond,
            [{"role": "system", "content": _PLANNER_PROMPT}],
            user_input,
        )
        raw = text or ""
        data = _extract_json(raw)
        if isinstance(data, dict) and isinstance(data.get("rewoo_cluster"), str):
            return [Step(prompt=data["rewoo_cluster"], is_rewoo_cluster=True)]
        return [Step(prompt=p) for p in _parse_steps(raw)]
```

- [ ] **Step 4: Verify pass** — `.venv/Scripts/python -m pytest tests/test_planner.py -v` -> all green.

- [ ] **Step 5: Lint + type** — ruff + mypy on `runtime/planner.py` + `tests/test_planner.py`. Clean.

- [ ] **Step 6: Stage** — `git add agent_framework/runtime/planner.py agent_framework/tests/test_planner.py`

---

## Task 3: Agent `is_rewoo_cluster` branch

**Files:** Modify `agent_framework/runtime/agent.py`; Test `agent_framework/tests/test_agent.py`.

- [ ] **Step 1: Write failing test** (append to `tests/test_agent.py`):

```python
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
        parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

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
```

- [ ] **Step 2: Verify fail** — `.venv/Scripts/python -m pytest tests/test_agent.py -v` -> FAIL (Agent.__init__ rejects `rewoo` kwarg).

- [ ] **Step 3: Inject `rewoo`** — In `runtime/agent.py`:
(a) TYPE_CHECKING block: add `from runtime.rewoo import ReWOO` (after the Replanner import).
(b) `__init__` signature: add `rewoo: ReWOO,` after `replanner: Replanner,` and a line `self._rewoo = rewoo` in the body.

- [ ] **Step 4: Add the branch** — In `Agent.chat` PLAN_REQUIRED while-loop, replace:
```python
                    outcome = await self._executor.run(
                        session, plan[i].prompt, trace
                    )
```
with:
```python
                    if plan[i].is_rewoo_cluster:
                        outcome = await self._rewoo.run(
                            session, session.memory, plan[i].prompt, i, trace
                        )
                    else:
                        outcome = await self._executor.run(
                            session, plan[i].prompt, trace
                        )
```

- [ ] **Step 5: Verify pass** — `.venv/Scripts/python -m pytest tests/test_agent.py -v` -> all green. Then `.venv/Scripts/python -m pytest -q`.

- [ ] **Step 6: Lint + type** — ruff + mypy on `runtime/agent.py` + `tests/test_agent.py`. Clean.

- [ ] **Step 7: Stage** — `git add agent_framework/runtime/agent.py agent_framework/tests/test_agent.py`

---

## Task 4: Full-tree verification + slice commit

- [ ] **Step 1:** `.venv/Scripts/python -m pytest -q` -> all pass (>= 79 prior + new S4 tests).
- [ ] **Step 2:** `.venv/Scripts/python -m ruff check .` -> clean.
- [ ] **Step 3:** `.venv/Scripts/python -m mypy session runtime tools llm trace config.py` -> clean (now 22 source files).
- [ ] **Step 4:** Hand-verify: ReWOO worker uses `ToolRegistry.dispatch` (same path as executor); tool errors -> `ERROR:` string (no crash); agent still appends ONLY the synthesized answer on PLAN_REQUIRED.
- [ ] **Step 5:** Controller commits:
```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(runtime): ReWOO micro-parallel sub-mode (S4)

- rewoo.py: DagNode + plan-DAG -> worker (var substitution, tool dispatch,
  no LLM) -> solver (one synthesis); insufficient evidence -> needs_replan
- Planner detects rewoo_cluster JSON key -> Step(is_rewoo_cluster=True)
- Agent plan loop: is_rewoo_cluster branch (rewoo.run vs executor.run)
- C' -> D' handoff: solver evidence_sufficient=false feeds S3 REPLANNING
EOF
)"
```
