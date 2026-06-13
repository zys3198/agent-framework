from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.executor import Outcome
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
    """Micro-parallel sub-mode (C'): plan DAG -> worker (tool dispatch, no LLM)
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
            raw_args = n.get("args")
            args = raw_args if isinstance(raw_args, dict) else {}
            raw_deps = n.get("deps")
            deps = raw_deps if isinstance(raw_deps, list) else []
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
