# S3 REPLANNING Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire real Plan-and-Execute with macro replanning — replace the `PLAN_REQUIRED` stub in `agent.py` with a `planner → while+replan → synthesize` loop, driven by `outcome.needs_replan`.

**Architecture:** Introduce a `Step` dataclass so `Memory.plan` becomes `list[Step]` (resolving the spec §3.1-vs-§5.1 contradiction: §3.1 treats `step` as an object, §5.1 says `list[str]`). Add `Planner` (族 D) and `Replanner` (族 D′). Wire two `needs_replan=True` triggers in `Executor`: Reflexion exhaustion (primary) and max-steps truncation (spec §3.2). `Agent.chat` runs the plan loop; on `needs_replan` with `replans < MAX_REPLANS` it revises the remaining tail and re-runs the current index; at cap it continues the old plan's remaining steps (spec §6). Contract C is preserved: `Executor` stays the sole writer of `session.messages` on every step.

**Tech Stack:** Python 3.12, asyncio, dataclasses, DeepSeek (OpenAI-compatible) via `LLMClient`, pytest-asyncio. Style: `ruff` (line-length 88, E/F/W/I/UP/B/SIM/RUF, ignore E501), `mypy --strict`, `StrEnum`, `from __future__ import annotations`, `dict[str, Any]` (no bare `dict`), half-width ASCII punctuation.

**Confirmed design decisions (this session):**
- **D1:** Introduce `Step` dataclass. `Memory.plan: list[str]` → `list[Step]`. `Step.from_dict` tolerates raw `str` items from old session files (wraps as `Step(prompt=str)`).
- **D2:** At `MAX_REPLANS` cap, continue the old plan's remaining steps (spec §6), do NOT abort to RESPONDING.

**Carry-over from S1/S2 (do not regress):**
- **Contract C:** `Executor` is the sole writer of `session.messages` on the `SIMPLE_TOOL`/per-step path (`user → assistant(tool_calls) → tool → final assistant`). `Agent` must NOT append `user`/`assistant(tool_calls)`/`tool` in the plan loop — only the final synthesized answer.
- `log_replan(count, reason, revised_steps)` takes **3 params** (spec §3.1 `log_replan(replans)` 1-param is a spec bug; `TraceLogger` already implements 3-param).
- `session.updated_at = _now()` before save, and `trace.close()` in `finally` — both already in `agent.chat`; keep them when rewriting the loop.
- async tools + sync `LLMClient` wrapped via `asyncio.to_thread`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `agent_framework/session/models.py` | Modify | Add `Step` dataclass; change `Memory.plan` to `list[Step]` with tolerant `from_dict` |
| `agent_framework/runtime/planner.py` | Create | `Planner.make_plan(input, memory) -> list[Step]` (族 D); owns shared `_parse_steps` |
| `agent_framework/runtime/replanner.py` | Create | `Replanner.revise(remaining, results, memory) -> list[Step]` (族 D′) |
| `agent_framework/runtime/executor.py` | Modify | Wire `needs_replan=True` on Reflexion exhaustion + truncation; keep Contract C (no orphan tool) |
| `agent_framework/runtime/agent.py` | Modify | Inject `planner`/`replanner`/`max_replans`; real `PLAN_REQUIRED` loop; `build_system_prompt` uses `s.prompt` |
| `agent_framework/tests/test_models.py` | Modify | `Step` roundtrip + `from_dict` str tolerance + `Memory.plan` Step roundtrip |
| `agent_framework/tests/test_planner.py` | Create | JSON parse / line fallback / empty |
| `agent_framework/tests/test_replanner.py` | Create | revise returns steps / context passed / bad-JSON fallback |
| `agent_framework/tests/test_executor.py` | Modify | truncation `needs_replan=True`; exhaustion triggers replan |
| `agent_framework/tests/test_agent.py` | Modify | real planning loop; replan-once; cap-at-MAX; update `_build_agent` signature |

Test command (run from `agent_framework/`): `.venv/Scripts/python -m pytest tests/<file>.py -v`
Full check: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check . && .venv/Scripts/python -m mypy agent_framework`

---

## Task 1: `Step` dataclass + `Memory.plan` migration

**Files:**
- Modify: `agent_framework/session/models.py`
- Test: `agent_framework/tests/test_models.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_models.py`)

```python
from session.models import Step


def test_step_roundtrip():
    s = Step(prompt="do A", is_rewoo_cluster=False, done=False)
    d = s.to_dict()
    assert d == {"prompt": "do A", "is_rewoo_cluster": False, "done": False}
    s2 = Step.from_dict(d)
    assert s2.prompt == "do A"
    assert s2.is_rewoo_cluster is False
    assert s2.done is False


def test_step_defaults():
    s = Step(prompt="x")
    assert s.is_rewoo_cluster is False
    assert s.done is False


def test_step_from_dict_tolerates_str():
    # old session files stored plan items as raw strings
    s = Step.from_dict("legacy step text")
    assert s.prompt == "legacy step text"
    assert s.done is False


def test_step_from_dict_tolerates_extra_keys():
    s = Step.from_dict({"prompt": "p", "future_field": "ignore me"})
    assert s.prompt == "p"


def test_memory_plan_step_roundtrip():
    m = Memory()
    m.plan.append(Step(prompt="step one", done=True))
    m.plan.append(Step(prompt="step two"))
    d = m.to_dict()
    assert d["plan"][0] == {"prompt": "step one", "is_rewoo_cluster": False, "done": True}
    m2 = Memory.from_dict(d)
    assert len(m2.plan) == 2
    assert m2.plan[0].prompt == "step one"
    assert m2.plan[0].done is True
    assert m2.plan[1].done is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: FAIL — `Step` not importable.

- [ ] **Step 3: Add `Step` dataclass + migrate `Memory.plan`**

In `agent_framework/session/models.py`, add the `Step` dataclass after `TodoItem` (before `Memory`):

```python
@dataclass
class Step:
    prompt: str
    is_rewoo_cluster: bool = False  # S4 ReWOO marker; unused in S3
    done: bool = False  # completed steps are skipped on replan rerun

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | str) -> Step:
        # tolerate old session files that stored plan items as raw strings:
        # wrap a bare string as Step(prompt=str). Also tolerate extra keys.
        if isinstance(d, str):
            return cls(prompt=d)
        return cls(
            prompt=d["prompt"],
            is_rewoo_cluster=d.get("is_rewoo_cluster", False),
            done=d.get("done", False),
        )
```

Change `Memory` to use `list[Step]`. Replace the `plan: list[str]` field and both serialisation sites:

```python
@dataclass
class Memory:
    todos: list[TodoItem] = field(default_factory=list)
    plan: list[Step] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    workspace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "todos": [t.to_dict() for t in self.todos],
            "plan": [s.to_dict() for s in self.plan],
            "lessons": list(self.lessons),
            "workspace": dict(self.workspace),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Memory:
        return cls(
            todos=[TodoItem.from_dict(t) for t in d.get("todos", [])],
            plan=[Step.from_dict(s) for s in d.get("plan", [])],
            lessons=list(d.get("lessons", [])),
            workspace=dict(d.get("workspace", {})),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: PASS — all `test_models.py` tests green (existing `test_memory_defaults` still passes: `m.plan == []`).

- [ ] **Step 5: Lint + type check**

Run: `.venv/Scripts/python -m ruff check agent_framework/session/models.py tests/test_models.py && .venv/Scripts/python -m mypy agent_framework/session/models.py`
Expected: clean.

- [ ] **Step 6: Stage (do NOT commit — controller commits)**

```bash
git add agent_framework/session/models.py agent_framework/tests/test_models.py
```

---

## Task 2: `Planner` (族 D) + shared `_parse_steps`

**Files:**
- Create: `agent_framework/runtime/planner.py`
- Test: `agent_framework/tests/test_planner.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_planner.py`)

```python
import inspect

from runtime.planner import Planner, _parse_steps
from session.models import Memory


class ScriptedLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    def respond(self, messages, user_input):
        self.calls += 1
        return self._reply


def test_parse_steps_json():
    steps = _parse_steps('{"steps": ["do A", "do B", "do C"]}')
    assert steps == ["do A", "do B", "do C"]


def test_parse_steps_strips_numbering_and_bullets():
    text = "1. first\n2) second\n- third\n* fourth"
    assert _parse_steps(text) == ["first", "second", "third", "fourth"]


def test_parse_steps_ignores_blank_lines():
    assert _parse_steps("only one\n\n\n") == ["only one"]


def test_parse_steps_empty():
    assert _parse_steps("") == []
    assert _parse_steps("   \n  ") == []


def test_parse_steps_json_inside_prose():
    text = 'Here is the plan:\n{"steps": ["a", "b"]}\nHope it helps.'
    assert _parse_steps(text) == ["a", "b"]


def test_parse_steps_bad_json_falls_back_to_lines():
    # malformed JSON object -> fall back to line split
    text = '{"steps": [broken\nsecond line'
    out = _parse_steps(text)
    assert "second line" in out


async def test_make_plan_returns_steps():
    llm = ScriptedLLM('{"steps": ["search X", "calculate Y"]}')
    planner = Planner(llm)
    plan = await planner.make_plan("do X then Y", Memory())
    assert [s.prompt for s in plan] == ["search X", "calculate Y"]
    assert all(not s.done for s in plan)
    assert llm.calls == 1


async def test_make_plan_empty_when_no_steps():
    llm = ScriptedLLM("")
    planner = Planner(llm)
    plan = await planner.make_plan("vague", Memory())
    assert plan == []


def test_make_plan_is_async():
    assert inspect.iscoroutinefunction(Planner.make_plan)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_planner.py -v`
Expected: FAIL — module `runtime.planner` not found.

- [ ] **Step 3: Implement `Planner` + `_parse_steps`**

Create `agent_framework/runtime/planner.py`:

```python
from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Memory, Step

_PLANNER_PROMPT = (
    "You decompose a task into ordered steps for a tool-using executor.\n"
    'Return ONLY JSON: {"steps": ["...", "..."]}.\n'
    "Each step is one self-contained instruction. Empty list if no step is needed."
)

_LEAD = re.compile(r"^\s*(\d+[\.\)]|[-*])\s*")


def _parse_steps(text: str) -> list[str]:
    """Extract step prompts from LLM text.

    Prefer a JSON object {"steps": [...]} embedded anywhere in the text.
    Fall back to one prompt per non-empty line (stripping leading numbering
    and bullet markers). Empty input -> [].
    """
    if not text:
        return []
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            return [str(s).strip() for s in data["steps"] if str(s).strip()]
    out: list[str] = []
    for line in text.splitlines():
        cleaned = _LEAD.sub("", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


class Planner:
    """Produce an ordered step list for a complex task (族 D, main path)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def make_plan(self, user_input: str, memory: Memory) -> list[Step]:
        from session.models import Step

        text = await asyncio.to_thread(
            self._llm.respond,
            [{"role": "system", "content": _PLANNER_PROMPT}],
            user_input,
        )
        return [Step(prompt=p) for p in _parse_steps(text or "")]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_planner.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + type check**

Run: `.venv/Scripts/python -m ruff check agent_framework/runtime/planner.py tests/test_planner.py && .venv/Scripts/python -m mypy agent_framework/runtime/planner.py`
Expected: clean.

- [ ] **Step 6: Stage**

```bash
git add agent_framework/runtime/planner.py agent_framework/tests/test_planner.py
```

---

## Task 3: `Replanner` (族 D′)

**Files:**
- Create: `agent_framework/runtime/replanner.py`
- Test: `agent_framework/tests/test_replanner.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_replanner.py`)

```python
import inspect

from runtime.replanner import Replanner
from session.models import Memory, Step


class ScriptedLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.last_input = ""

    def respond(self, messages, user_input):
        self.last_input = user_input
        return self._reply


def _steps(*prompts: str) -> list[Step]:
    return [Step(prompt=p) for p in prompts]


async def test_revise_returns_steps():
    llm = ScriptedLLM('{"steps": ["retry with arg X", "then Y"]}')
    rp = Replanner(llm)
    out = await rp.revise(_steps("original"), {}, Memory())
    assert [s.prompt for s in out] == ["retry with arg X", "then Y"]


async def test_revise_empty_when_no_steps():
    llm = ScriptedLLM('{"steps": []}')
    rp = Replanner(llm)
    out = await rp.revise(_steps("x"), {}, Memory())
    assert out == []


async def test_revise_context_includes_remaining_results_lessons():
    llm = ScriptedLLM('{"steps": ["new"]}')
    rp = Replanner(llm)
    memory = Memory()
    memory.lessons.append("avoid empty query")

    class _O:
        text = "empty result"

    await rp.revise(_steps("step A", "step B"), {0: _O()}, memory)
    assert "step A" in llm.last_input
    assert "step B" in llm.last_input
    assert "empty result" in llm.last_input
    assert "avoid empty query" in llm.last_input


async def test_revise_bad_json_falls_back_to_lines():
    llm = ScriptedLLM("1. first fallback\n2. second fallback")
    rp = Replanner(llm)
    out = await rp.revise(_steps("x"), {}, Memory())
    assert [s.prompt for s in out] == ["first fallback", "second fallback"]


def test_revise_is_async():
    assert inspect.iscoroutinefunction(Replanner.revise)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_replanner.py -v`
Expected: FAIL — module `runtime.replanner` not found.

- [ ] **Step 3: Implement `Replanner`**

Create `agent_framework/runtime/replanner.py`:

```python
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from runtime.planner import _parse_steps

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Memory, Step

_REPLANNER_PROMPT = (
    "A step failed or hit a limit and needs a revised plan for the REMAINING work.\n"
    'Return ONLY JSON: {"steps": ["...", "..."]} listing the revised remaining\n'
    "steps. Empty list if no further step is needed."
)


def _build_context(
    remaining: list[Step], results: dict[int, Any], memory: Memory
) -> str:
    lines: list[str] = ["Remaining steps:"]
    lines.extend(f"- {s.prompt}" for s in remaining)
    if results:
        lines.append("Step results so far:")
        for idx, val in results.items():
            text = getattr(val, "text", val)
            lines.append(f"- step {idx}: {text}")
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {lesson}" for lesson in memory.lessons)
    return "\n".join(lines)


class Replanner:
    """Revise the remaining plan after a failed/limited step (族 D′, macro self-correction)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def revise(
        self,
        remaining: list[Step],
        results: dict[int, Any],
        memory: Memory,
    ) -> list[Step]:
        from session.models import Step

        context = _build_context(remaining, results, memory)
        text = await asyncio.to_thread(
            self._llm.respond,
            [{"role": "system", "content": _REPLANNER_PROMPT}],
            context,
        )
        return [Step(prompt=p) for p in _parse_steps(text or "")]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_replanner.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + type check**

Run: `.venv/Scripts/python -m ruff check agent_framework/runtime/replanner.py tests/test_replanner.py && .venv/Scripts/python -m mypy agent_framework/runtime/replanner.py`
Expected: clean.

- [ ] **Step 6: Stage**

```bash
git add agent_framework/runtime/replanner.py agent_framework/tests/test_replanner.py
```

---

## Task 4: `Executor` `needs_replan` wire

Wire two `needs_replan=True` triggers (spec §3.2): Reflexion exhaustion (primary) and max-steps truncation. Preserve Contract C — the `tool` message must be appended before returning (no orphan tool).

**Files:**
- Modify: `agent_framework/runtime/executor.py:99-115`
- Test: `agent_framework/tests/test_executor.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_executor.py`)

First add an `ExhaustedReflexion` fake near `FakeReflexion`:

```python
class ExhaustedReflexion(Reflexion):
    """Always reports reflexion exhausted -> executor must signal needs_replan."""

    def __init__(self) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]

    async def reflect(self, call, error, memory) -> Lesson:  # type: ignore[override]
        return Lesson(text="exhausted lesson", reflexion_exhausted=True)
```

Then the new test:

```python
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
```

And change the existing truncation test's expectation (find `test_max_steps_truncation`, change the `needs_replan` assertion):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_executor.py -v`
Expected: FAIL — `test_tool_error_exhausted_triggers_replan` (needs_replan still False) and `test_max_steps_truncation` (assertion flipped).

- [ ] **Step 3: Wire exhaustion trigger (no orphan tool)**

In `agent_framework/runtime/executor.py`, replace the per-tool-call block (the `for tc in resp.tool_calls:` body) so that the tool message is appended before any early return. Replace this block:

```python
            for tc in resp.tool_calls:
                trace.log_tool_call(step, tc.name, tc.args)
                try:
                    result = await self._registry.dispatch(
                        ToolCall(name=tc.name, args=tc.args), session
                    )
                except Exception as e:  # tool must not crash executor
                    result = f"ERROR: {e}"
                    log.warning("tool %s raised: %s", tc.name, e)
                    lesson = await self._reflexion.reflect(tc, result, session.memory)
                    session.memory.lessons.append(lesson.text)
                    trace.log_reflexion(step, lesson.text)
                trace.log_tool_result(step, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                session.messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )
```

with:

```python
            for tc in resp.tool_calls:
                trace.log_tool_call(step, tc.name, tc.args)
                exhausted = False
                try:
                    result = await self._registry.dispatch(
                        ToolCall(name=tc.name, args=tc.args), session
                    )
                except Exception as e:  # tool must not crash executor
                    result = f"ERROR: {e}"
                    log.warning("tool %s raised: %s", tc.name, e)
                    lesson = await self._reflexion.reflect(tc, result, session.memory)
                    session.memory.lessons.append(lesson.text)
                    trace.log_reflexion(step, lesson.text)
                    exhausted = lesson.reflexion_exhausted
                # Contract C: append the tool message BEFORE any early return so
                # reloaded sessions never hold an orphan tool message (API 400).
                trace.log_tool_result(step, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                session.messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )
                if exhausted:  # Reflexion spent -> hand off to Replanner (族 D')
                    return Outcome(text=result, needs_replan=True)
```

- [ ] **Step 4: Wire truncation trigger**

In the same file, change the truncation return (last two lines of `run`):

```python
        trace.log_truncated()
        session.messages.append(Message(role="assistant", content="(truncated)"))
        return Outcome(text="(truncated)", needs_replan=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_executor.py -v`
Expected: PASS — all executor tests green (including the two updated/new ones).

- [ ] **Step 6: Lint + type check**

Run: `.venv/Scripts/python -m ruff check agent_framework/runtime/executor.py tests/test_executor.py && .venv/Scripts/python -m mypy agent_framework/runtime/executor.py`
Expected: clean.

- [ ] **Step 7: Stage**

```bash
git add agent_framework/runtime/executor.py agent_framework/tests/test_executor.py
```

---

## Task 5: `Agent` `PLAN_REQUIRED` loop + injection

Inject `planner`/`replanner`/`max_replans` into `Agent`, replace the `PLAN_REQUIRED` stub with the real loop, and update `build_system_prompt` to read `s.prompt`. Preserve Contract C (agent appends only the synthesized final answer on this path).

**Files:**
- Modify: `agent_framework/runtime/agent.py`
- Test: `agent_framework/tests/test_agent.py`

- [ ] **Step 1: Write the failing tests** (rewrite `tests/test_agent.py`)

Full replacement of the file:

```python
from __future__ import annotations

import inspect

from llm.client import LLMResponse
from runtime.agent import Agent
from runtime.executor import Executor
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
    max_replans: int = 2,
) -> Agent:
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
        planner=None,  # type: ignore[arg-type]
        replanner=replanner,  # set per-test when needed
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
    # planner.respond -> step JSON; each step executor.chat -> text; synthesize.respond
    llm = FakeLLM(
        responds=['{"steps": ["do A", "do B"]}', "final answer"],
        chats=[
            LLMResponse(text="A done", tool_calls=[]),
            LLMResponse(text="B done", tool_calls=[]),
        ],
    )
    agent = _build_agent(tmp_path, llm, Route.PLAN_REQUIRED)
    out = await agent.chat("s3", "plan X")
    assert out == "final answer"
    s = agent._store.load("s3")
    assert len(s.memory.plan) == 2
    assert [st.prompt for st in s.memory.plan] == ["do A", "do B"]
    # synthesized answer persisted (Contract C: agent appends only the final answer)
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "final answer"


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
    # synthesize was called with the revised 1-step plan + 2 results
    assert out == "synth:1:2"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
Expected: FAIL — `Agent.__init__` rejects `planner`/`replanner`/`max_replans` kwargs.

- [ ] **Step 3: Update `Agent.__init__` + imports**

In `agent_framework/runtime/agent.py`, add the new deps to the `TYPE_CHECKING` block and `__init__`:

```python
if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.executor import Executor
    from runtime.planner import Planner
    from runtime.replanner import Replanner
    from runtime.router import Router
    from session.store import Store
```

```python
    def __init__(
        self,
        store: Store,
        router: Router,
        executor: Executor,
        llm: LLMClient,
        trace_dir: Path,
        planner: Planner,
        replanner: Replanner,
        max_replans: int,
    ) -> None:
        self._store = store
        self._router = router
        self._executor = executor
        self._llm = llm
        self._trace_dir = trace_dir
        self._planner = planner
        self._replanner = replanner
        self._max_replans = max_replans
```

- [ ] **Step 4: Update `build_system_prompt` to read `s.prompt`**

In the same file, change the plan line inside `build_system_prompt`:

```python
    if memory.plan:
        lines.append("Plan: " + " | ".join(s.prompt for s in memory.plan))
```

- [ ] **Step 5: Replace the `PLAN_REQUIRED` stub with the real loop**

In `agent.py`'s `chat`, replace the entire `else:` branch (the SIMPLE_TOOL / PLAN_REQUIRED block) with a split that gives `PLAN_REQUIRED` its own loop. The new branch structure inside the `try:` (after `trace.log_route(route.value)`):

```python
            if route == Route.DIRECT:
                session.fsm_state = State.RESPONDING.value
                sys_msg = build_system_prompt(session.memory)
                messages = [{"role": "system", "content": sys_msg}] + [
                    m.to_dict() for m in session.messages
                ]
                answer = await asyncio.to_thread(
                    self._llm.respond, messages, user_input
                )
                # DIRECT: agent persists user + assistant (executor does it for tool paths)
                session.messages.append(Message(role="user", content=user_input))
                session.messages.append(Message(role="assistant", content=answer))

            elif route == Route.SIMPLE_TOOL:
                # Executor owns all session.messages persistence on this path (Contract C).
                session.fsm_state = State.EXECUTING.value
                outcome = await self._executor.run(session, user_input, trace)
                session.fsm_state = State.RESPONDING.value
                answer = outcome.text

            else:  # PLAN_REQUIRED
                session.fsm_state = State.PLANNING.value
                plan = await self._planner.make_plan(user_input, session.memory)
                session.memory.plan = plan
                session.fsm_state = State.EXECUTING.value

                results: dict[int, object] = {}
                replans = 0
                i = 0
                while i < len(plan):
                    outcome = await self._executor.run(
                        session, plan[i].prompt, trace
                    )
                    results[i] = outcome
                    if outcome.needs_replan and replans < self._max_replans:
                        session.fsm_state = State.REPLANNING.value
                        revised = await self._replanner.revise(
                            plan[i:], results, session.memory
                        )
                        plan = plan[:i] + revised
                        replans += 1
                        session.memory.plan = plan
                        trace.log_replan(replans, "needs_replan", len(revised))
                        session.fsm_state = State.EXECUTING.value
                        continue  # re-run index i (= first revised step)
                    i += 1

                answer = await asyncio.to_thread(
                    self._llm.synthesize,
                    [s.prompt for s in plan],
                    {idx: getattr(o, "text", o) for idx, o in results.items()},
                )
                # Contract C: agent appends ONLY the synthesized final answer here;
                # per-step messages were persisted by the executor.
                session.fsm_state = State.RESPONDING.value
                session.messages.append(Message(role="assistant", content=answer))
```

Keep the existing tail unchanged (`session.fsm_state = State.IDLE.value`, `session.updated_at = _now()`, `self._store.save(session)`, `return answer`, `finally: trace.close()`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
Expected: PASS — all 8 agent tests green (including the 3 new plan-loop tests).

- [ ] **Step 7: Lint + type check**

Run: `.venv/Scripts/python -m ruff check agent_framework/runtime/agent.py tests/test_agent.py && .venv/Scripts/python -m mypy agent_framework/runtime/agent.py`
Expected: clean.

- [ ] **Step 8: Stage**

```bash
git add agent_framework/runtime/agent.py agent_framework/tests/test_agent.py
```

---

## Task 6: Full-tree verification + baseline

**Files:** none (verification + commit).

- [ ] **Step 1: Full test suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests pass (S1 + S2 + S3). Count rises above the S2 baseline of 57.

- [ ] **Step 2: Full ruff**

Run: `.venv/Scripts/python -m ruff check .`
Expected: clean.

- [ ] **Step 3: Full mypy strict**

Run: `.venv/Scripts/python -m mypy agent_framework`
Expected: clean (no errors across all source files including new planner/replanner).

- [ ] **Step 4: Smoke-check Contract C invariants by hand**

Re-read `agent.py` and confirm:
- `PLAN_REQUIRED` branch appends ONLY the synthesized `assistant` answer (no `user`, no `tool`, no `assistant(tool_calls)`).
- `SIMPLE_TOOL` branch appends nothing (executor owns it).
- `DIRECT` branch appends `user` + `assistant`.

- [ ] **Step 5: Commit the whole slice** (controller shows `git diff --cached --stat` first)

After the controller reviews and approves, commit (conventional commits, one logical change):

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(runtime): REPLANNING plan-and-execute loop (S3)

- Step dataclass; Memory.plan list[str] -> list[Step] (tolerant from_dict)
- Planner (族 D) + Replanner (族 D') with shared JSON/line _parse_steps
- Executor: needs_replan=True on Reflexion exhaustion + truncation (spec 3.2),
  tool message appended before early return (Contract C preserved)
- Agent: real PLAN_REQUIRED loop (planner -> while+replan -> synthesize),
  MAX_REPLANS cap continues old plan (spec 6)
EOF
)"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 Agent top-level orchestration (planner -> while+replan -> synthesize): Task 5.
- §3.2 `needs_replan` triggers (exhaustion + truncation): Task 4.
- §1.2 族 D Planner / 族 D′ Replanner responsibility split (planner only produces steps; replanner only revises remaining; neither executes): Tasks 2, 3.
- §5.1 `Memory.plan` typed as step-bearing structure (Step): Task 1.
- §6 Replan cap = continue old plan: Task 5 `replans < self._max_replans` guard, no abort.
- §7 trace `replan` event 3-param: Task 5 `log_replan(replans, "needs_replan", len(revised))`.
- §10 Replanner tests (exhaustion triggers replan / MAX_REPLANS cap / completed steps not re-run): Tasks 4, 5.

**2. Placeholder scan:** every code step shows full code; no TODO / "similar to" / "add error handling". `is_rewoo_cluster` is a real defaulted field used by S4, documented as such, not a placeholder.

**3. Type consistency:**
- `Step(prompt, is_rewoo_cluster=False, done=False)` — consistent across models.py (Task 1), planner.py (Task 2), replanner.py (Task 3), test files.
- `Planner.make_plan(input, memory) -> list[Step]` — Task 2 def, Task 5 call.
- `Replanner.revise(remaining, results, memory) -> list[Step]` — Task 3 def, Task 5 call passes `plan[i:]` as `remaining`.
- `Executor.run(session, prompt, trace) -> Outcome(text, needs_replan)` — Task 4 keeps signature, Task 5 calls `self._executor.run(session, plan[i].prompt, trace)`.
- `Agent.__init__(store, router, executor, llm, trace_dir, planner, replanner, max_replans)` — Task 5 def + test `_build_agent`.
- `TraceLogger.log_replan(count, reason, revised_steps)` — already 3-param in `trace/logger.py`; Task 5 calls it 3-param.
- `LLMClient.synthesize(plan: list[str], results: dict)` — Task 5 passes `[s.prompt for s in plan]` (list[str]) + dict.

No type drift across tasks.
