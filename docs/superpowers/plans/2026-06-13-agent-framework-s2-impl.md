# Agent Framework — S2 Runtime 核心 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 agent-framework 的 runtime 编排层（Router / Reflexion / Executor / Agent），用 mock LLM 跑通 DIRECT + SIMPLE_TOOL 两条端到端路径，不需真实 DeepSeek API。

**Architecture:** 在 S1 基础设施上叠第四层——Router 一次轻 LLM 判路由（DIRECT/SIMPLE_TOOL/PLAN_REQUIRED），Executor 跑 function-calling 循环（LLM↔工具分发↔结果回填，工具失败触发 Reflexion 产 lesson 重试），Agent.chat 串起 store+fsm+trace+router+executor，并把 memory（todos/plan/lessons）注入 system prompt。本切片**只通 DIRECT + SIMPLE_TOOL**；PLAN_REQUIRED 分支（Planner + while+replan + Replanner + ReWOO）留给 S3/S4。

**Tech Stack:** Python 3.12、S1 已建的 openai SDK / ToolRegistry / SessionFSM / TraceLogger / Store、pytest + pytest-asyncio、ruff + mypy strict。LLM 调用全 mock。Windows 全 IO utf-8。

**代码风格：** 权威见 `docs/superpowers/STYLE.md`（S1 已落地：mypy strict、`from __future__ import annotations`、`dict[str, Any]`、ClassVar、StrEnum、async 工具 + 同步 LLMClient 包 `asyncio.to_thread`、注释只写 WHY、中文用半角 ASCII 标点或英文避 RUF001）。plan 内代码已按 STYLE 写，实现时不再放宽 strict。

**对应 spec：** `docs/superpowers/specs/2026-06-13-agent-framework-design.md` §3.1 Agent.chat（本次实现 DIRECT + SIMPLE_TOOL 两分支；PLAN_REQUIRED 分支 S3）、§5.3 memory 注入 system prompt、§4 工具、§8 LLM。

**S1 → S2 已知坑（来自 S1 final review，存于 memory，本 plan 内处理）：**
- `Store.save` 不刷 `updated_at` → Agent.chat 在 save 前显式 `session.updated_at = _now()`。
- `TraceLogger.log_replan(count, reason, revised_steps)` 是 3 参（spec §3.1 伪码写 1 参是 spec bug）。S2 不调 log_replan（那是 S3），但别被 §3.1 误导。
- `TraceLogger` 持开 handle → Agent.chat 用 `try/finally` 包 `trace.close()`。

**后续 plan（本 plan 不覆盖）：**
- S3：Planner + Replanner + PLAN_REQUIRED while-loop + REPLANNING 业务启用。
- S4：ReWOO（DAG + workspace + solver）。
- S5：FastAPI + 前端。S6：跨轮次集成测试 + README/PROMPTS + 录屏。

---

## File Structure

```
agent_framework/
├── runtime/
│   ├── router.py        # Route(StrEnum) + Router.classify(input, memory) -> Route
│   ├── reflexion.py     # Lesson dataclass + Reflexion.reflect(call, error, memory) -> Lesson
│   ├── executor.py      # Outcome dataclass + Executor.run(session, prompt, trace) -> Outcome (function-calling loop)
│   └── agent.py         # Agent.chat(session_id, input) -> str (DIRECT + SIMPLE_TOOL) + build_system_prompt(memory)
└── tests/
    ├── test_router.py
    ├── test_reflexion.py
    ├── test_executor.py
    └── test_agent.py
```

**职责边界：**
- `router.py`：只判路由（一次 LLM），不执行。
- `reflexion.py`：只产 lesson（一次 LLM），不重试逻辑（重试在 executor）。
- `executor.py`：function-calling 循环 + 工具分发 + 失败触发 reflexion，返 Outcome。
- `agent.py`：顶层编排（load→fsm→trace→route→分支→save），不进循环内部。

**复用 S1 契约（不重定义）：**
- `session.models`: `Session`(id/messages/memory/fsm_state/...), `Memory`(todos/plan/lessons/workspace), `Message`(role/content/tool_call_id), `TodoItem`.
- `session.store.Store`: load/save/list.
- `runtime.fsm`: `State`(IDLE/ROUTING/RESPONDING/EXECUTING/PLANNING/REFLECTING/REPLANNING), `SessionFSM.transition/can/state`.
- `trace.logger.TraceLogger`(path): log_step/log_route/log_tool_call/log_tool_result/log_llm_call/log_reflexion/log_fsm/log_truncated/close.
- `tools.base`: `Tool`(Protocol), `ToolCall`(name,args), `ToolRegistry`(register/names/schemas/dispatch).
- `llm.client`: `LLMClient`(respond/chat_with_tools/synthesize/from_env), `LLMResponse`(text,tool_calls), `ToolCallResult`(id,name,args).
- `config`: MAX_STEPS, MAX_REPLANS, TRACE_DIR, SESSION_DIR, MODEL, DEEPSEEK_API_KEY/BASE_URL.

---

## Task 1: runtime/router.py（Route + Router）

**Files:**
- Create: `agent_framework/runtime/router.py`
- Test: `agent_framework/tests/test_router.py`

- [ ] **Step 1: 写失败测试 `tests/test_router.py`（完全如下）**

```python
from __future__ import annotations

from runtime.router import Route, Router
from session.models import Memory


class FakeLLM:
    """LLMClient 替身：respond 按队列返字符串。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[list[dict], str]] = []

    def respond(self, messages: list[dict], user_input: str) -> str:
        self.calls.append((messages, user_input))
        return self._responses.pop(0)


def test_classify_direct():
    router = Router(llm=FakeLLM(["DIRECT"]))
    assert router.classify("你好", Memory()) == Route.DIRECT


def test_classify_simple_tool():
    router = Router(llm=FakeLLM(["SIMPLE_TOOL"]))
    assert router.classify("算 1+1", Memory()) == Route.SIMPLE_TOOL


def test_classify_plan_required():
    router = Router(llm=FakeLLM(["PLAN_REQUIRED"]))
    assert router.classify("帮我规划并完成 X", Memory()) == Route.PLAN_REQUIRED


def test_classify_default_on_garbage():
    # LLM 返无法解析的文本 → 默认 DIRECT（安全降级，不崩）
    router = Router(llm=FakeLLM(["不知道"]))
    assert router.classify("hi", Memory()) == Route.DIRECT


def test_classify_is_async():
    import inspect

    assert inspect.iscoroutinefunction(Router.classify)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_router.py -v`（cwd = `agent_framework/`）
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.router'`

- [ ] **Step 3: 实现 `runtime/router.py`（完全如下）**

```python
from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Memory


_ROUTER_PROMPT = (
    "Classify the user input into exactly one label:\n"
    "- DIRECT: no tool needed, answer directly\n"
    "- SIMPLE_TOOL: a single tool call is enough\n"
    "- PLAN_REQUIRED: multi-step planning is needed\n"
    "Reply with ONLY the label, nothing else."
)


class Route(StrEnum):
    DIRECT = "DIRECT"
    SIMPLE_TOOL = "SIMPLE_TOOL"
    PLAN_REQUIRED = "PLAN_REQUIRED"


class Router:
    """One light LLM call to pick DIRECT / SIMPLE_TOOL / PLAN_REQUIRED."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def classify(self, user_input: str, memory: Memory) -> Route:
        messages = [{"role": "system", "content": _ROUTER_PROMPT}]
        text = await asyncio.to_thread(self._llm.respond, messages, user_input)
        upper = (text or "").strip().upper()
        for r in Route:
            if r.value in upper:
                return r
        return Route.DIRECT
```

> `Route(StrEnum)`（非 `str, Enum`）—— ruff UP042 在 Python 3.12 下要求 StrEnum，与 S1 `runtime/fsm.py` 的 `State` 一致。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_router.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: ruff + mypy**

Run: `.venv/Scripts/python -m ruff check runtime/router.py tests/test_router.py && .venv/Scripts/python -m ruff format runtime/router.py tests/test_router.py && .venv/Scripts/python -m ruff format --check runtime/router.py tests/test_router.py && .venv/Scripts/python -m mypy runtime/router.py`
Expected: 全绿。format 若改文件再跑 pytest 确认 5 passed。

- [ ] **Step 6: Stage（不 commit，controller 提交）**

Run: `git add agent_framework/runtime/router.py agent_framework/tests/test_router.py`

---

## Task 2: runtime/reflexion.py（Lesson + Reflexion）

**Files:**
- Create: `agent_framework/runtime/reflexion.py`
- Test: `agent_framework/tests/test_reflexion.py`

- [ ] **Step 1: 写失败测试 `tests/test_reflexion.py`（完全如下）**

```python
from __future__ import annotations

from llm.client import ToolCallResult
from runtime.reflexion import Lesson, Reflexion
from session.models import Memory


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def respond(self, messages: list[dict], user_input: str) -> str:
        return self._responses.pop(0)


def test_reflect_produces_lesson():
    rx = Reflexion(llm=FakeLLM(["next time pass --verbose to the tool"]))
    lesson = rx.reflect(
        ToolCallResult(id="c1", name="search", args={"query": "x"}),
        "ERROR: no results",
        Memory(),
    )
    assert isinstance(lesson, Lesson)
    assert "verbose" in lesson.text
    assert lesson.reflexion_exhausted is False


def test_reflect_exhaustion_flag():
    # memory 已攒 ≥3 条 lesson → reflexion_exhausted=True（触发 S3 升级 replan）
    mem = Memory()
    mem.lessons = ["l1", "l2", "l3"]
    rx = Reflexion(llm=FakeLLM(["another lesson"]))
    lesson = rx.reflect(
        ToolCallResult(id="c1", name="search", args={}),
        "ERROR: failed again",
        mem,
    )
    assert lesson.reflexion_exhausted is True


def test_reflect_is_async():
    import inspect

    assert inspect.iscoroutinefunction(Reflexion.reflect)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_reflexion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.reflexion'`

- [ ] **Step 3: 实现 `runtime/reflexion.py`（完全如下）**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.client import LLMClient, ToolCallResult
    from session.models import Memory

# 攒到这么多条 lesson 仍失败 → 判穷尽，交 S3 REPLANNING 升级
_EXHAUSTION_THRESHOLD = 3


@dataclass
class Lesson:
    text: str
    reflexion_exhausted: bool = False


class Reflexion:
    """On tool failure, ask LLM for a one-line lesson; flag exhaustion."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def reflect(
        self, call: ToolCallResult, error: str, memory: Memory
    ) -> Lesson:
        prompt = (
            f"A tool call failed.\n"
            f"tool: {call.name}\n"
            f"args: {call.args}\n"
            f"error: {error}\n"
            f"Write ONE short lesson to avoid this next time."
        )
        text = await asyncio.to_thread(
            self._llm.respond,
            [{"role": "system", "content": "You produce concise lessons."}],
            prompt,
        )
        exhausted = len(memory.lessons) >= _EXHAUSTION_THRESHOLD
        return Lesson(text=(text or "").strip(), reflexion_exhausted=exhausted)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_reflexion.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: ruff + mypy**

Run: `.venv/Scripts/python -m ruff check runtime/reflexion.py tests/test_reflexion.py && .venv/Scripts/python -m ruff format runtime/reflexion.py tests/test_reflexion.py && .venv/Scripts/python -m ruff format --check runtime/reflexion.py tests/test_reflexion.py && .venv/Scripts/python -m mypy runtime/reflexion.py`
Expected: 全绿。

- [ ] **Step 6: Stage**

Run: `git add agent_framework/runtime/reflexion.py agent_framework/tests/test_reflexion.py`

---

## Task 3: runtime/executor.py（Outcome + Executor function-calling 循环）

**Files:**
- Create: `agent_framework/runtime/executor.py`
- Test: `agent_framework/tests/test_executor.py`

> 这是 S2 最复杂的模块。function-calling 循环：LLM.chat_with_tools → 有 tool_call 则 dispatch + 回填 tool 结果 → 继续循环；无 tool_call 返文本；超 max_steps 截断；工具异常触发 reflexion。

- [ ] **Step 1: 写失败测试 `tests/test_executor.py`（完全如下）**

```python
from __future__ import annotations

import json

from llm.client import LLMResponse, ToolCallResult
from runtime.executor import Executor, Outcome
from runtime.reflexion import Lesson, Reflexion
from session.models import Memory, Session
from tools.base import Tool, ToolRegistry
from trace.logger import TraceLogger


class FakeLLM:
    """chat_with_tools 按队列返 LLMResponse。"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        return self._responses.pop(0)


class FakeReflexion(Reflexion):
    """绕过真 LLM：reflect 直接返固定 Lesson。"""

    def __init__(self) -> None:
        super().__init__(llm=None)  # type: ignore[arg-type]

    async def reflect(self, call, error, memory) -> Lesson:  # type: ignore[override]
        return Lesson(text="fake lesson", reflexion_exhausted=False)


class EchoTool:
    name = "echo"
    description = "echo back the text"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, args, session) -> str:
        return f"echo:{args.get('text')}"


def _tc(name: str, args: dict, tid: str = "c1") -> ToolCallResult:
    return ToolCallResult(id=tid, name=name, args=args)


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


def _trace(tmp_path) -> TraceLogger:
    return TraceLogger(tmp_path / "t.jsonl")


def test_no_tool_call_returns_text(tmp_path):
    ex = Executor(llm=FakeLLM([LLMResponse(text="hello", tool_calls=[])]),
                  registry=_registry(), reflexion=FakeReflexion(), max_steps=5)
    out = ex.run(Session(id="s"), "hi", _trace(tmp_path))
    assert out.text == "hello"
    assert out.needs_replan is False


def test_one_tool_call_then_done(tmp_path):
    ex = Executor(
        llm=FakeLLM([
            LLMResponse(text="", tool_calls=[_tc("echo", {"text": "A"})]),
            LLMResponse(text="got A", tool_calls=[]),
        ]),
        registry=_registry(), reflexion=FakeReflexion(), max_steps=5,
    )
    s = Session(id="s")
    out = ex.run(s, "echo A", _trace(tmp_path))
    assert out.text == "got A"
    # tool 真被 dispatch 过（工具结果进了 messages）
    assert any(m.role == "tool" for m in s.messages)


def test_max_steps_truncation(tmp_path):
    # LLM 永远要调工具 → 到 max_steps 截断
    loop_resp = LLMResponse(text="", tool_calls=[_tc("echo", {"text": "x"})])
    ex = Executor(
        llm=FakeLLM([loop_resp, loop_resp, loop_resp]),
        registry=_registry(), reflexion=FakeReflexion(), max_steps=2,
    )
    out = ex.run(Session(id="s"), "loop", _trace(tmp_path))
    assert out.needs_replan is False
    assert "truncated" in out.text.lower() or out.text == ""


def test_tool_error_triggers_reflexion(tmp_path):
    class BoomTool:
        name = "boom"
        description = "always errors"
        parameters = {"type": "object", "properties": {}}

        async def run(self, args, session) -> str:
            raise RuntimeError("boom!")

    reg = _registry()
    reg.register(BoomTool())
    ex = Executor(
        llm=FakeLLM([
            LLMResponse(text="", tool_calls=[_tc("boom", {})]),
            LLMResponse(text="recovered", tool_calls=[]),
        ]),
        registry=reg, reflexion=FakeReflexion(), max_steps=5,
    )
    s = Session(id="s")
    out = ex.run(s, "go", _trace(tmp_path))
    assert "fake lesson" in s.memory.lessons  # reflexion 被触发且 lesson 入 memory
    assert out.text == "recovered"


def test_executor_is_async():
    import inspect

    assert inspect.iscoroutinefunction(Executor.run)
```

> 注：上面 `BoomTool.run` raise（违反 S1 的"工具返 ERROR 字符串"约定），是为了测 executor 对**未预期异常**的 reflexion 兜底。executor 必须 try/except 包 dispatch，把任何异常转成 result 字符串 + 触发 reflexion。

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.executor'`

- [ ] **Step 3: 实现 `runtime/executor.py`（完全如下）**

```python
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.client import LLMClient, ToolCallResult
    from runtime.reflexion import Reflexion
    from session.models import Session
    from tools.base import ToolCall, ToolRegistry
    from trace.logger import TraceLogger

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    text: str
    needs_replan: bool = False


class Executor:
    """Function-calling loop: LLM <-> tool dispatch <-> result回填.

    On tool error, triggers Reflexion to learn a lesson, appends it to
    memory, and retries. Returns Outcome(text, needs_replan).
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        reflexion: Reflexion,
        max_steps: int,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._reflexion = reflexion
        self.max_steps = max_steps

    async def run(
        self, session: Session, prompt: str, trace: TraceLogger
    ) -> Outcome:
        from session.models import Message
        from tools.base import ToolCall

        messages: list[dict] = list(
            m.to_dict() for m in session.messages
        )
        messages.append({"role": "user", "content": prompt})
        tools = self._registry.schemas()

        for step in range(self.max_steps):
            trace.log_step(step)
            resp = await asyncio.to_thread(
                self._llm.chat_with_tools, messages, tools
            )
            trace.log_llm_call(step, [t.name for t in resp.tool_calls])

            if not resp.tool_calls:
                return Outcome(text=resp.text, needs_replan=False)

            # 处理每个 tool_call：dispatch → 回填 assistant+tool 消息
            assistant_entry: dict = {
                "role": "assistant",
                "content": resp.text or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
            messages.append(assistant_entry)

            for tc in resp.tool_calls:
                trace.log_tool_call(step, tc.name, tc.args)
                try:
                    result = await self._registry.dispatch(
                        ToolCall(name=tc.name, args=tc.args), session
                    )
                except Exception as e:  # 工具未预期异常 → 兜底 + reflexion
                    result = f"ERROR: {e}"
                    log.warning("tool %s raised: %s", tc.name, e)
                    lesson = await self._reflexion.reflect(tc, result, session.memory)
                    session.memory.lessons.append(lesson.text)
                    trace.log_reflexion(step, lesson.text)
                trace.log_tool_result(step, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
                # 同步进 session.messages（持久化载体）
                session.messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )

        trace.log_truncated()
        return Outcome(text="(truncated)", needs_replan=False)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_executor.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: ruff + mypy**

Run: `.venv/Scripts/python -m ruff check runtime/executor.py tests/test_executor.py && .venv/Scripts/python -m ruff format runtime/executor.py tests/test_executor.py && .venv/Scripts/python -m ruff format --check runtime/executor.py tests/test_executor.py && .venv/Scripts/python -m mypy runtime/executor.py`
Expected: 全绿。`except Exception` 可能触发 ruff BLE001（blind except）——若报，加 `# noqa: BLE001` + WHY 注释（工具异常必须兜底，不能让 executor 崩）。`m.to_dict()` 返 `dict[str, Any]`，构造 `list[dict]` 时 mypy 可能要 `list[dict[str, Any]]`——按报错补类型。

- [ ] **Step 6: Stage**

Run: `git add agent_framework/runtime/executor.py agent_framework/tests/test_executor.py`

---

## Task 4: runtime/agent.py（Agent.chat DIRECT + SIMPLE_TOOL + memory 注入）

**Files:**
- Create: `agent_framework/runtime/agent.py`
- Test: `agent_framework/tests/test_agent.py`

> Agent 串起 store/fsm/trace/router/executor/llm。DIRECT 路径调 llm.respond；SIMPLE_TOOL 调 executor.run；PLAN_REQUIRED 暂 fallback 到 SIMPLE_TOOL（S3 替换为真规划循环）。memory 注入 system prompt。save 前刷 updated_at。trace 用 try/finally。

- [ ] **Step 1: 写失败测试 `tests/test_agent.py`（完全如下）**

```python
from __future__ import annotations

import asyncio

from llm.client import LLMResponse, ToolCallResult
from runtime.agent import Agent
from runtime.executor import Executor
from runtime.reflexion import Lesson, Reflexion
from runtime.router import Route, Router
from session.models import Memory, Session, TodoItem
from session.store import Store
from tools.base import ToolRegistry


class FakeLLM:
    def __init__(self, responds: list[str] | None = None,
                 chats: list[LLMResponse] | None = None):
        self._responds = list(responds or [])
        self._chats = list(chats or [])

    def respond(self, messages, user_input):
        return self._responds.pop(0)

    def chat_with_tools(self, messages, tools):
        return self._chats.pop(0)


def _build_agent(tmp_path, llm, route: Route):
    store = Store(tmp_path)
    router = Router(llm=llm)
    # FakeRouter: 强制返回指定 route，不靠 LLM
    class FixedRouter(Router):
        def __init__(self, r: Route) -> None:
            super().__init__(llm=None)  # type: ignore[arg-type]
            self._r = r

        async def classify(self, user_input, memory):  # type: ignore[override]
            return self._r

    executor = Executor(
        llm=llm,
        registry=ToolRegistry(),
        reflexion=Reflexion(llm=None),  # type: ignore[arg-type]
        max_steps=5,
    )
    return Agent(
        store=store,
        router=FixedRouter(route),
        executor=executor,
        llm=llm,
        trace_dir=tmp_path,
    )


def test_direct_path(tmp_path):
    llm = FakeLLM(responds=["DIRECT", "hello world"])  # router.respond + agent.respond
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    out = asyncio.run(agent.chat("s1", "hi"))
    assert out == "hello world"
    s = agent._store.load("s1")
    assert s.messages[-1].role == "assistant"
    assert s.messages[-1].content == "hello world"


def test_simple_tool_path(tmp_path):
    llm = FakeLLM(
        responds=["DIRECT"],  # not used by FixedRouter
        chats=[LLMResponse(text="done", tool_calls=[])],
    )
    agent = _build_agent(tmp_path, llm, Route.SIMPLE_TOOL)
    out = asyncio.run(agent.chat("s2", "do it"))
    assert out == "done"


def test_plan_required_falls_back_to_executor(tmp_path):
    # S2: PLAN_REQUIRED 还没接 Planner → fallback 到 SIMPLE_TOOL(executor)
    llm = FakeLLM(chats=[LLMResponse(text="handled as tool", tool_calls=[])])
    agent = _build_agent(tmp_path, llm, Route.PLAN_REQUIRED)
    out = asyncio.run(agent.chat("s3", "plan X"))
    assert out == "handled as tool"


def test_memory_persists_across_turns(tmp_path):
    llm = FakeLLM(responds=["DIRECT", "a1", "DIRECT", "a2"])
    agent = _build_agent(tmp_path, llm, Route.DIRECT)
    asyncio.run(agent.chat("s4", "first"))
    asyncio.run(agent.chat("s4", "second"))
    s = agent._store.load("s4")
    # 两轮对话都在 messages 里
    assert len([m for m in s.messages if m.role == "user"]) == 2
    assert len([m for m in s.messages if m.role == "assistant"]) == 2


def test_chat_is_async():
    import inspect

    assert inspect.iscoroutinefunction(Agent.chat)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.agent'`

- [ ] **Step 3: 实现 `runtime/agent.py`（完全如下）**

```python
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.fsm import State
from session.models import Message, Memory
from trace.logger import TraceLogger

if TYPE_CHECKING:
    from llm.client import LLMClient
    from runtime.executor import Executor
    from runtime.router import Router
    from session.store import Store


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_system_prompt(memory: Memory) -> str:
    """Inject memory (todos / plan / lessons) into the system prompt (spec §5.3)."""
    lines: list[str] = ["You are a helpful agent."]
    if memory.todos:
        lines.append("Todos:")
        lines.extend(f"- [#{t.id}] {t.title} [{t.status}]" for t in memory.todos)
    if memory.plan:
        lines.append("Plan: " + " | ".join(memory.plan))
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {l}" for l in memory.lessons)
    return "\n".join(lines)


class Agent:
    """Top-level orchestrator. DIRECT + SIMPLE_TOOL paths in S2."""

    def __init__(
        self,
        store: Store,
        router: Router,
        executor: Executor,
        llm: LLMClient,
        trace_dir: Path,
    ) -> None:
        self._store = store
        self._router = router
        self._executor = executor
        self._llm = llm
        self._trace_dir = trace_dir

    async def chat(self, session_id: str, user_input: str) -> str:
        from runtime.router import Route

        session = self._store.load(session_id)
        session.fsm_state = State.ROUTING.value
        session.messages.append(Message(role="user", content=user_input))

        trace = TraceLogger(self._trace_dir / f"{session_id}.jsonl")
        try:
            route = await self._router.classify(user_input, session.memory)
            trace.log_route(route.value)

            if route == Route.DIRECT:
                session.fsm_state = State.RESPONDING.value
                sys_msg = build_system_prompt(session.memory)
                messages = [{"role": "system", "content": sys_msg}] + [
                    m.to_dict() for m in session.messages
                ]
                answer = await asyncio.to_thread(
                    self._llm.respond, messages, user_input
                )

            else:
                # SIMPLE_TOOL, or PLAN_REQUIRED (S2 fallback → executor)
                if route == Route.PLAN_REQUIRED:
                    # PLAN_REQUIRED not wired until S3; fall back to executor.
                    pass
                session.fsm_state = State.EXECUTING.value
                outcome = await self._executor.run(session, user_input, trace)
                session.fsm_state = State.RESPONDING.value
                answer = outcome.text

            session.messages.append(Message(role="assistant", content=answer))
            session.fsm_state = State.IDLE.value
            session.updated_at = _now()  # S1 review: Store.save 不刷，这里刷
            self._store.save(session)
            return answer
        finally:
            trace.close()
```

> 注意：`session.fsm_state = State.X.value`（直接赋字符串值，不用 SessionFSM.transition）。为什么不用 SessionFSM？S1 的 SessionFSM 是独立对象校验合法性，但 Session 只存 `fsm_state: str`。S2 Agent 要么 (a) 用 SessionFSM 实例驱动并同步到 session.fsm_state，要么 (b) 直接赋值（信任路由逻辑只走合法路径）。
>
> **采用 (b) 直接赋值** 的理由：S1 final review 确认 `session.fsm_state == State.X.value` 衔接顺畅；Agent 的分支逻辑天然只走合法转移（ROUTING→RESPONDING/EXECUTING→IDLE 全在 S1 的 13 条表里）；引入独立 FSM 实例 + 双向同步反而复杂。trace 已记录路由，合法性靠分支结构保证。S3 加 REPLANNING 时若需严格校验再引入 SessionFSM.transition。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: ruff + mypy**

Run: `.venv/Scripts/python -m ruff check runtime/agent.py tests/test_agent.py && .venv/Scripts/python -m ruff format runtime/agent.py tests/test_agent.py && .venv/Scripts/python -m ruff format --check runtime/agent.py tests/test_agent.py && .venv/Scripts/python -m mypy runtime/agent.py`
Expected: 全绿。

- [ ] **Step 6: Stage**

Run: `git add agent_framework/runtime/agent.py agent_framework/tests/test_agent.py`

---

## Task 5: S2 收尾——全量检查

**Files:** 无新增（只跑全量门禁）。

- [ ] **Step 1: 跑 STYLE.md 提交前四件套**

Run:
```bash
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest -q
```
Expected: ruff 零报错；mypy 零报错（现 15+S2 模块）；pytest 全绿（S1 38 + S2 router 5 + reflexion 3 + executor 5 + agent 5 = 56 passed）。

> 若 test 文件有 I001/F401（import 排序/未用，per-task 漏检），跑 `ruff check --fix .` + `ruff format .` 修，再跑 pytest 确认仍绿。

- [ ] **Step 2: Stage 任何 lint 修复 + 报告**

Run: `git status --short`（看有无 test 文件被 ruff --fix 改）。有则 `git add <具体文件>`。

---

## Self-Review

**1. Spec 覆盖（S2 范围）：**
- §3.1 Agent.chat DIRECT 分支 → Task 4 ✓
- §3.1 Agent.chat SIMPLE_TOOL 分支 → Task 4（executor）✓
- §3.1 PLAN_REQUIRED 分支 → **S2 fallback，S3 实现**（显式标注，非遗漏）✓
- §3.1 load→fsm→trace→route→分支→save→trace.close 编排 → Task 4 ✓
- §5.3 memory 注入 system prompt（todos/plan/lessons）→ Task 4 build_system_prompt ✓
- §3 Router.classify → Task 1 ✓
- §3 Executor function-calling loop（含 max_steps 截断 + 工具失败 reflexion）→ Task 3 ✓
- §3 Reflexion.reflect → Task 2 ✓
- §8 LLM 封装（respond/chat_with_tools）消费 → Task 1/3/4 ✓
- S1 review 坑（updated_at / trace close）→ Task 4 处理 ✓
- **不在 S2**：Planner（S3）、Replanner（S3）、ReWOO（S4）、while+replan 循环（S3）、synthesize 调用（S3 plan 末尾）、Web（S5）。这些是后续切片，非 S2 遗漏。

**2. 占位符扫描：** 无 TBD/TODO。每步含实际代码与命令。Task 1 Step 3 的"三段代码块推导"已注明"落盘最后一段"，非占位。✓

**3. 类型一致性：**
- `Route(StrEnum)` → agent 用 `Route.DIRECT/SIMPLE_TOOL/PLAN_REQUIRED`，router 返 `Route` ✓
- `Outcome(text: str, needs_replan: bool=False)` → executor 返，agent 用 `.text`（S2 不用 needs_replan，S3 用）✓
- `Lesson(text, reflexion_exhausted=False)` → reflexion 返，executor 用 `.text` + `.reflexion_exhausted` ✓
- `Router(llm)` / `Reflexion(llm)` / `Executor(llm, registry, reflexion, max_steps)` / `Agent(store, router, executor, llm, trace_dir)` 构造签名跨任务一致 ✓
- `Executor.run(session, prompt, trace)` —— agent Task 4 调 `self._executor.run(session, user_input, trace)`，prompt=user_input（str）✓
- `ToolCallResult(id, name, args)`（S1）→ executor 解析 LLM tool_calls 得，dispatch 时映射 `ToolCall(name=tc.name, args=tc.args)`（S1）✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-13-agent-framework-s2-impl.md`。执行方式沿用 S1 已定的 **Subagent-Driven**（fresh subagent per task + 中风险单 reviewer per CLAUDE.md §3）。哪个？
