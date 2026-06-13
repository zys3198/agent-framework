# Agent Framework — S1 基础设施 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 agent-framework 的基础设施层（config / LLM 客户端 / 工具 / session / store / trace / FSM），全部可 mock 测试，无需真实 DeepSeek API，为 S2 runtime 核心提供可组合底座。

**Architecture:** 分层 runtime 的最底层——配置集中、session JSON 持久化（原子写）、trace jsonl 日志、FSM 状态机校验、工具协议 + 注册表、DeepSeek（OpenAI 兼容）LLM 封装。本层不含 Agent 编排逻辑（S2），不含 REPLANNING/ReWOO 业务（S3/S4）。

**Tech Stack:** Python 3.12、`openai` SDK（指向 DeepSeek）、`pydantic`（可选，本层先用 dataclass）、`pytest` + `pytest-asyncio` + `ruff` + `mypy`（strict）。Windows 下全 IO 强制 utf-8。

**代码风格：** 权威见 `docs/superpowers/STYLE.md`。mypy strict 落地——所有参数/返回类型注解、禁 `Any`、`session: "Session"`、LLM client 用真实 SDK 类型。plan 内代码为示意，实现时按 STYLE 补全类型。

**对应 spec:** `docs/superpowers/specs/2026-06-13-agent-framework-design.md`（§2 模块清单、§4 工具、§5 session/memory、§6 异常、§7 trace、§8 LLM、附录 A/B）。

**后续 plan（本 plan 不覆盖，按 skill scope 拆分）:**
- S2 plan：router/planner/executor/reflexion/agent（DIRECT + SIMPLE_TOOL 路径）
- S3 plan：REPLANNING（replanner + agent while-loop）
- S4 plan：ReWOO（DAG + workspace + solver）
- S5 plan：FastAPI + 前端
- S6 plan：跨轮次集成测试 + README/PROMPTS + 录屏

---

## File Structure

```
agent_framework/
├── pyproject.toml              # 依赖与项目元信息
├── config.py                   # 环境变量集中读取
├── llm/
│   ├── __init__.py
│   └── client.py               # DeepSeek 封装：chat_with_tools/respond/synthesize
├── tools/
│   ├── __init__.py
│   ├── base.py                 # Tool Protocol + ToolRegistry
│   ├── calculator.py           # ast 白名单求值
│   ├── search.py               # mock 语料搜索
│   └── todo.py                 # CRUD 写 session.memory.todos
├── session/
│   ├── __init__.py
│   ├── models.py               # TodoItem/Memory/Message/Session dataclass
│   └── store.py                # JSON 持久化 (os.replace 原子写)
├── trace/
│   ├── __init__.py
│   └── logger.py               # jsonl 追加日志
├── runtime/
│   └── fsm.py                  # SessionFSM (含 REPLANNING 全状态，S3 启用业务)
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_models.py
    ├── test_store.py
    ├── test_trace.py
    ├── test_fsm.py
    ├── test_tools.py
    └── test_llm_client.py
```

**职责边界（每文件单一职责，独立可测）：**
- `config.py` 只读环境变量，不业务。
- `models.py` 只定义数据结构 + 序列化，不 IO。
- `store.py` 只读写 JSON，不解析语义。
- `trace/logger.py` 只追加日志，不业务。
- `fsm.py` 只校验状态合法性，不业务。
- `tools/base.py` 只定义协议 + 注册表分发，不实现具体工具。
- 各 `tools/*.py` 只实现自己的工具，不依赖其他工具。
- `llm/client.py` 只封装 SDK，不编排。

---

## Task 1: 项目骨架与依赖

**Files:**
- Create: `agent_framework/pyproject.toml`
- Create: `agent_framework/**/__init__.py`（所有包）

- [ ] **Step 1: 写 pyproject.toml**

```toml
[project]
name = "agent-framework"
version = "0.1.0"
description = "Minimal agent runtime (self-implemented, no LangChain/OpenHands)"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "openai>=1.40",
    "pydantic>=2.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "ruff>=0.6",
    "mypy>=1.11",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["config", "llm", "tools", "session", "trace", "runtime"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
strict = true
files = ["config.py", "llm", "tools", "session", "trace", "runtime"]

[[tool.mypy.overrides]]
module = "tests.*"
strict = false
disallow_untyped_defs = false
warn_return_any = false

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

> 代码风格权威见 `docs/superpowers/STYLE.md`。mypy strict：所有函数参数/返回加类型，禁 `Any`，`session: "Session"`、`LLMClient.__init__(self, client: OpenAI, model: str)`。

- [ ] **Step 2: 建包结构**

```bash
cd agent_framework
mkdir -p llm tools session trace runtime tests
touch llm/__init__.py tools/__init__.py session/__init__.py trace/__init__.py runtime/__init__.py tests/__init__.py
```

- [ ] **Step 3: 装依赖**

Run: `python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"`
Expected: 安装成功，`pip show openai` 有输出。

- [ ] **Step 4: 验证骨架**

Run: `.venv/Scripts/python -c "import llm, tools, session, trace, runtime; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml llm tools session trace runtime tests
git commit -m "chore: project scaffold + deps"
```

---

## Task 2: config.py

**Files:**
- Create: `agent_framework/config.py`
- Test: `agent_framework/tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
import os
import importlib
import config as config_mod


def test_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("MAX_STEPS", raising=False)
    monkeypatch.delenv("MAX_REPLANS", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    importlib.reload(config_mod)
    assert config_mod.MAX_STEPS == 10
    assert config_mod.MAX_REPLANS == 2
    assert config_mod.MODEL == "deepseek-chat"
    assert config_mod.REWOO_PARALLEL_ENABLED is True
    assert config_mod.HOST == "127.0.0.1"
    assert config_mod.PORT == 8000


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_STEPS", "5")
    monkeypatch.setenv("MAX_REPLANS", "7")
    monkeypatch.setenv("PORT", "9000")
    importlib.reload(config_mod)
    assert config_mod.MAX_STEPS == 5
    assert config_mod.MAX_REPLANS == 7
    assert config_mod.PORT == 9000
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: 实现 config.py**

```python
# config.py
import os
from pathlib import Path


def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL", "deepseek-chat")

MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
MAX_REPLANS = int(os.environ.get("MAX_REPLANS", "2"))
REWOO_PARALLEL_ENABLED = _bool(os.environ.get("REWOO_PARALLEL_ENABLED", "true"))

BASE_DIR = Path(__file__).resolve().parent
SESSION_DIR = BASE_DIR / os.environ.get("SESSION_DIR", "sessions")
TRACE_DIR = BASE_DIR / os.environ.get("TRACE_DIR", "trace")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(config): env-driven config with defaults"
```

---

## Task 3: session/models.py

**Files:**
- Create: `agent_framework/session/models.py`
- Test: `agent_framework/tests/test_models.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_models.py
from session.models import TodoItem, Memory, Message, Session


def test_todo_item_defaults():
    t = TodoItem(id="1", title="写大纲")
    assert t.status == "PLANNED"
    assert t.created_at  # 非空 ISO 时间戳


def test_memory_defaults():
    m = Memory()
    assert m.todos == []
    assert m.plan == []
    assert m.lessons == []
    assert m.workspace == {}


def test_session_defaults():
    s = Session(id="sid-1")
    assert s.fsm_state == "IDLE"
    assert s.step_count == 0
    assert s.memory.todos == []
    assert s.messages == []


def test_to_dict_roundtrip():
    s = Session(id="sid-1")
    s.memory.todos.append(TodoItem(id="1", title="A", status="IN_PROGRESS"))
    s.messages.append(Message(role="user", content="hi"))
    d = s.to_dict()
    s2 = Session.from_dict(d)
    assert s2.id == "sid-1"
    assert s2.memory.todos[0].title == "A"
    assert s2.memory.todos[0].status == "IN_PROGRESS"
    assert s2.messages[0].role == "user"
    assert s2.fsm_state == "IDLE"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 models.py**

```python
# session/models.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TodoItem:
    id: str
    title: str
    status: str = "PLANNED"  # PLANNED | IN_PROGRESS | DONE
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TodoItem:
        return cls(**d)


@dataclass
class Memory:
    todos: list[TodoItem] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    workspace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "todos": [t.to_dict() for t in self.todos],
            "plan": list(self.plan),
            "lessons": list(self.lessons),
            "workspace": dict(self.workspace),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Memory:
        return cls(
            todos=[TodoItem.from_dict(t) for t in d.get("todos", [])],
            plan=list(d.get("plan", [])),
            lessons=list(d.get("lessons", [])),
            workspace=dict(d.get("workspace", {})),
        )


@dataclass
class Message:
    role: str  # user | assistant | tool
    content: str
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        return cls(**d)


@dataclass
class Session:
    id: str
    messages: list[Message] = field(default_factory=list)
    memory: Memory = field(default_factory=Memory)
    fsm_state: str = "IDLE"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    step_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "memory": self.memory.to_dict(),
            "fsm_state": self.fsm_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step_count": self.step_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        return cls(
            id=d["id"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            memory=Memory.from_dict(d.get("memory", {})),
            fsm_state=d.get("fsm_state", "IDLE"),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            step_count=d.get("step_count", 0),
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add session/models.py tests/test_models.py
git commit -m "feat(session): dataclass models with dict roundtrip"
```

---

## Task 4: session/store.py（JSON 原子持久化）

**Files:**
- Create: `agent_framework/session/store.py`
- Test: `agent_framework/tests/test_store.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_store.py
import json
from pathlib import Path
from session.store import Store
from session.models import Session, TodoItem, Message


def test_load_missing_creates_new(tmp_path):
    store = Store(tmp_path)
    s = store.load("new-sid")
    assert s.id == "new-sid"
    assert s.fsm_state == "IDLE"
    assert s.memory.todos == []


def test_save_then_load_roundtrip(tmp_path):
    store = Store(tmp_path)
    s = store.load("sid-1")
    s.memory.todos.append(TodoItem(id="1", title="A", status="IN_PROGRESS"))
    s.messages.append(Message(role="user", content="hi"))
    s.fsm_state = "EXECUTING"
    store.save(s)

    s2 = store.load("sid-1")
    assert s2.memory.todos[0].title == "A"
    assert s2.messages[0].content == "hi"
    assert s2.fsm_state == "EXECUTING"


def test_corrupt_json_recovers(tmp_path):
    store = Store(tmp_path)
    f = tmp_path / "sid-x.json"
    f.write_text("{ broken json", encoding="utf-8")
    s = store.load("sid-x")
    assert s.id == "sid-x"
    assert s.memory.todos == []
    # 损坏文件应被备份
    backups = list(tmp_path.glob("*.corrupt.bak"))
    assert len(backups) == 1


def test_list_returns_summaries(tmp_path):
    store = Store(tmp_path)
    s = store.load("sid-1")
    s.memory.todos.append(TodoItem(id="1", title="A"))
    store.save(s)
    store.load("sid-2")
    store.save(store.load("sid-2"))

    items = store.list()
    ids = {it["id"] for it in items}
    assert ids == {"sid-1", "sid-2"}
    s1 = next(i for i in items if i["id"] == "sid-1")
    assert s1["todo_count"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 store.py**

```python
# session/store.py
from __future__ import annotations
import json
import os
import logging
from pathlib import Path
from session.models import Session

log = logging.getLogger(__name__)


class Store:
    """JSON 持久化。原子写 (tmp + os.replace)，损坏文件备份后重建。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        # 防路径穿越：只取文件名
        safe = Path(session_id).name
        return self.root / f"{safe}.json"

    def load(self, session_id: str) -> Session:
        p = self._path(session_id)
        if not p.exists():
            return Session(id=session_id)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            bak = p.with_suffix(p.suffix + ".corrupt.bak")
            try:
                os.replace(p, bak)
                log.warning("corrupt session file backed up: %s -> %s (%s)", p, bak, e)
            except OSError:
                pass
            return Session(id=session_id)

    def save(self, session: Session) -> None:
        p = self._path(session.id)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, p)  # 原子替换

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append({
                    "id": d["id"],
                    "todo_count": len(d.get("memory", {}).get("todos", [])),
                    "updated_at": d.get("updated_at", ""),
                    "fsm_state": d.get("fsm_state", "IDLE"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add session/store.py tests/test_store.py
git commit -m "feat(store): atomic JSON persistence with corrupt-recovery"
```

---

## Task 5: trace/logger.py（jsonl 执行日志）

**Files:**
- Create: `agent_framework/trace/logger.py`
- Test: `agent_framework/tests/test_trace.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_trace.py
import json
from pathlib import Path
from trace.logger import TraceLogger


def test_log_appends_jsonl(tmp_path):
    t = TraceLogger(tmp_path / "sid-1.jsonl")
    t.log_step(0)
    t.log_route("plan_required")
    t.log_tool_call(0, "todo.create", {"title": "A"})
    t.log_tool_result(0, "created #1")
    t.close()

    lines = (tmp_path / "sid-1.jsonl").read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines]
    assert recs[0]["type"] == "step" and recs[0]["step"] == 0
    assert recs[1]["type"] == "route" and recs[1]["value"] == "plan_required"
    assert recs[2]["name"] == "todo.create"
    assert recs[3]["result"] == "created #1"
    # 每条都有 ts
    assert all("ts" in r for r in recs)


def test_log_replan_and_rewoo(tmp_path):
    t = TraceLogger(tmp_path / "sid.jsonl")
    t.log_replan(count=1, reason="retry_exhausted", revised_steps=2)
    t.log_rewoo_dag(step=2, nodes=["E1", "E2"], edges=[["E1", "E2"]])
    t.log_rewoo_solve(step=2, vars=["E1", "E2"], sufficient=False)
    t.log_truncated()
    t.close()

    recs = [json.loads(l) for l in
            (tmp_path / "sid.jsonl").read_text(encoding="utf-8").strip().split("\n")]
    types = [r["type"] for r in recs]
    assert types == ["replan", "rewoo_dag", "rewoo_solve", "truncated"]
    assert recs[0]["count"] == 1
    assert recs[2]["evidence_sufficient"] is False
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 logger.py**

```python
# trace/logger.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceLogger:
    """每 session 一个 jsonl，每步一行。S3/S4 事件类型预留。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def _emit(self, rec: dict) -> None:
        rec.setdefault("ts", _ts())
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    # 基线事件
    def log_step(self, step: int) -> None: self._emit({"type": "step", "step": step})
    def log_llm_call(self, step: int, tools_offered: list[str]) -> None:
        self._emit({"type": "llm_call", "step": step, "tools_offered": tools_offered})
    def log_route(self, value: str) -> None: self._emit({"type": "route", "value": value})
    def log_tool_call(self, step: int, name: str, args: dict) -> None:
        self._emit({"type": "tool_call", "step": step, "name": name, "args": args})
    def log_tool_result(self, step: int, result: str) -> None:
        self._emit({"type": "tool_result", "step": step, "result": result})
    def log_reflexion(self, step: int, lesson: str) -> None:
        self._emit({"type": "reflexion", "step": step, "lesson": lesson})
    def log_fsm(self, frm: str, to: str) -> None:
        self._emit({"type": "fsm", "from": frm, "to": to})
    def log_truncated(self) -> None: self._emit({"type": "truncated"})

    # REPLANNING (S3)
    def log_replan(self, count: int, reason: str, revised_steps: int) -> None:
        self._emit({"type": "replan", "count": count,
                    "reason": reason, "revised_steps": revised_steps})

    # ReWOO (S4)
    def log_rewoo_dag(self, step: int, nodes: list[str], edges: list[list[str]]) -> None:
        self._emit({"type": "rewoo_dag", "step": step, "nodes": nodes, "edges": edges})
    def log_rewoo_solve(self, step: int, vars: list[str], sufficient: bool) -> None:
        self._emit({"type": "rewoo_solve", "step": step, "vars": vars,
                    "evidence_sufficient": sufficient})

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_trace.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add trace/logger.py tests/test_trace.py
git commit -m "feat(trace): jsonl logger with replan/rewoo event types"
```

---

## Task 6: runtime/fsm.py（SessionFSM 状态机）

**Files:**
- Create: `agent_framework/runtime/fsm.py`
- Test: `agent_framework/tests/test_fsm.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fsm.py
import pytest
from runtime.fsm import SessionFSM, InvalidTransition, State


def test_new_session_idle():
    f = SessionFSM()
    assert f.state == State.IDLE


def test_legal_path_plan():
    f = SessionFSM()
    f.transition(State.ROUTING)
    f.transition(State.PLANNING)
    f.transition(State.EXECUTING)
    f.transition(State.REPLANNING)  # replan
    f.transition(State.EXECUTING)
    f.transition(State.RESPONDING)
    f.transition(State.IDLE)
    assert f.state == State.IDLE


def test_legal_reflect_then_replan():
    f = SessionFSM()
    f.transition(State.ROUTING)
    f.transition(State.EXECUTING)
    f.transition(State.REFLECTING)
    f.transition(State.REPLANNING)  # retry_exhausted -> replan
    f.transition(State.EXECUTING)


def test_replanning_abort_to_responding():
    f = SessionFSM()
    f.transition(State.ROUTING)
    f.transition(State.EXECUTING)
    f.transition(State.REPLANNING)
    f.transition(State.RESPONDING)  # abort


def test_illegal_transition_raises():
    f = SessionFSM()
    with pytest.raises(InvalidTransition):
        f.transition(State.RESPONDING)  # IDLE -> RESPONDING 非法


def test_illegal_replanning_from_idle():
    f = SessionFSM()
    with pytest.raises(InvalidTransition):
        f.transition(State.REPLANNING)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_fsm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 fsm.py**

```python
# runtime/fsm.py
from __future__ import annotations
from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    ROUTING = "ROUTING"
    RESPONDING = "RESPONDING"
    EXECUTING = "EXECUTING"
    PLANNING = "PLANNING"
    REFLECTING = "REFLECTING"
    REPLANNING = "REPLANNING"


class InvalidTransition(Exception):
    pass


# 合法转移表 (from, to) —— 对应 spec §1.3
_TRANSITIONS: set[tuple[State, State]] = {
    (State.IDLE, State.ROUTING),
    (State.ROUTING, State.RESPONDING),   # route=direct
    (State.ROUTING, State.EXECUTING),    # route=simple
    (State.ROUTING, State.PLANNING),     # route=plan
    (State.PLANNING, State.EXECUTING),   # plan_ready
    (State.EXECUTING, State.REFLECTING), # tool_error
    (State.REFLECTING, State.EXECUTING), # retry
    (State.EXECUTING, State.REPLANNING),          # replan_needed
    (State.REFLECTING, State.REPLANNING),         # retry_exhausted
    (State.REPLANNING, State.EXECUTING),          # plan_updated
    (State.REPLANNING, State.RESPONDING),         # abort
    (State.EXECUTING, State.RESPONDING), # done
    (State.RESPONDING, State.IDLE),      # replied
}


class SessionFSM:
    """session 级状态机：只校验合法性，不持业务。"""

    def __init__(self, state: State = State.IDLE):
        self.state = state

    def transition(self, to: State) -> None:
        if (self.state, to) not in _TRANSITIONS:
            raise InvalidTransition(f"{self.state.value} -> {to.value}")
        self.state = to

    def can(self, to: State) -> bool:
        return (self.state, to) in _TRANSITIONS
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_fsm.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add runtime/fsm.py tests/test_fsm.py
git commit -m "feat(fsm): session state machine with REPLANNING transitions"
```

---

## Task 7: tools/base.py（Tool 协议 + ToolRegistry）

**Files:**
- Create: `agent_framework/tools/base.py`
- Test: `agent_framework/tests/test_tools.py`（本任务只测 Registry，具体工具后续任务补）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tools.py
import pytest
from tools.base import Tool, ToolRegistry, ToolCall


class FakeTool:
    name = "fake"
    description = "fake tool"
    parameters = {"type": "object", "properties": {}}

    async def run(self, args: dict, session) -> str:
        return f"fake:{args}"


def test_register_and_dispatch():
    reg = ToolRegistry()
    reg.register(FakeTool())
    assert "fake" in reg.names()

    import asyncio
    from session.models import Session
    res = asyncio.run(reg.dispatch(ToolCall(name="fake", args={"x": 1}), Session(id="s")))
    assert res == "fake:{'x': 1}"


def test_dispatch_unknown_raises():
    reg = ToolRegistry()
    import asyncio
    from session.models import Session
    with pytest.raises(KeyError):
        asyncio.run(reg.dispatch(ToolCall(name="nope", args={}), Session(id="s")))


def test_schemas_export():
    reg = ToolRegistry()
    reg.register(FakeTool())
    sch = reg.schemas()
    assert sch == [{"name": "fake", "description": "fake tool",
                    "parameters": {"type": "object", "properties": {}}}]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 base.py**

```python
# tools/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from session.models import Session


@dataclass
class ToolCall:
    name: str
    args: dict


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON Schema

    async def run(self, args: dict, session: "Session") -> str: ...


class ToolRegistry:
    """注册 + schema 导出 + 分发。不实现具体工具。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    async def dispatch(self, call: ToolCall, session: "Session") -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            raise KeyError(f"unknown tool: {call.name}")
        return await tool.run(call.args, session)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/base.py tests/test_tools.py
git commit -m "feat(tools): Tool protocol + registry dispatch"
```

---

## Task 8: tools/calculator.py（ast 安全求值）

**Files:**
- Create: `agent_framework/tools/calculator.py`
- Modify: `agent_framework/tests/test_tools.py`（追加 calculator 用例）

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_tools.py
from tools.calculator import Calculator


def test_calc_basic():
    import asyncio
    from session.models import Session
    c = Calculator()
    assert asyncio.run(c.run({"expr": "1 + 2 * 3"}, Session(id="s"))) == "7"
    assert asyncio.run(c.run({"expr": "(10 - 4) / 2"}, Session(id="s"))) == "3.0"


def test_calc_rejects_injection():
    import asyncio
    from session.models import Session
    c = Calculator()
    for bad in ["__import__('os')", "open('x')", "1; import os", "pow(2,3)"]:
        res = asyncio.run(c.run({"expr": bad}, Session(id="s")))
        assert res.startswith("ERROR"), f"应拒绝: {bad}"


def test_calc_schema():
    c = Calculator()
    assert c.name == "calculator"
    assert c.parameters["type"] == "object"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: tools.calculator`

- [ ] **Step 3: 实现 calculator.py**

```python
# tools/calculator.py
from __future__ import annotations
import ast
import operator as op

# 白名单二元运算符
_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


class Calculator:
    """算术表达式求值，ast 白名单，禁 eval/exec/函数调用。"""

    name = "calculator"
    description = "对算术表达式求值（支持 + - * / // % ** 和括号）。禁止函数调用与属性访问。"
    parameters = {
        "type": "object",
        "properties": {"expr": {"type": "string", "description": "算术表达式，如 (1+2)*3"}},
        "required": ["expr"],
    }

    async def run(self, args: dict, session) -> str:
        expr = args.get("expr")
        if not isinstance(expr, str):
            return "ERROR: expr 必须是字符串"
        try:
            tree = ast.parse(expr, mode="eval")
            return str(_eval(tree))
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
            return f"ERROR: {e}"
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/calculator.py tests/test_tools.py
git commit -m "feat(tools): calculator with ast whitelist (no eval)"
```

---

## Task 9: tools/search.py（mock 语料）

**Files:**
- Create: `agent_framework/tools/search.py`
- Modify: `agent_framework/tests/test_tools.py`（追加 search 用例）

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_tools.py
from tools.search import Search


def test_search_hit():
    import asyncio
    from session.models import Session
    s = Search()
    res = asyncio.run(s.run({"query": "DeepSeek"}, Session(id="s")))
    assert "DeepSeek" in res


def test_search_miss():
    import asyncio
    from session.models import Session
    s = Search()
    res = asyncio.run(s.run({"query": "zzz不存在zzz"}, Session(id="s")))
    assert "无结果" in res or res == ""
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py::test_search_hit -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 search.py**

```python
# tools/search.py
from __future__ import annotations

# 预设 mock 语料
_CORPUS: list[str] = [
    "DeepSeek 是 OpenAI 兼容的大模型 API，支持 function calling。",
    "Agent 基本循环：接收输入 → 判断直接答/调工具 → 执行 → 读结果 → 继续。",
    "Plan-and-Execute 架构先规划再分步执行，适合复杂任务。",
    "ReWOO 把规划与执行解耦，planner 一次推理产出 DAG，省 LLM round-trip。",
    "Reflexion 在工具失败后自评产出教训，带教训重试。",
]


class Search:
    """mock 搜索：关键词命中预设语料。无真实网络。"""

    name = "search"
    description = "搜索知识库（mock 预设语料），返回命中的条目。"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "搜索关键词"}},
        "required": ["query"],
    }

    async def run(self, args: dict, session) -> str:
        q = args.get("query", "")
        if not isinstance(q, str) or not q.strip():
            return "ERROR: query 必须是非空字符串"
        hits = [c for c in _CORPUS if q.lower() in c.lower()]
        return "\n".join(f"- {h}" for h in hits) if hits else "无结果"
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/search.py tests/test_tools.py
git commit -m "feat(tools): mock search corpus"
```

---

## Task 10: tools/todo.py（CRUD 写 memory）

**Files:**
- Create: `agent_framework/tools/todo.py`
- Modify: `agent_framework/tests/test_tools.py`（追加 todo 用例）

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_tools.py
from tools.todo import Todo
from session.models import Session


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_todo_create():
    t = Todo()
    s = Session(id="s")
    res = _run(t.run({"action": "create", "title": "写大纲"}, s))
    assert "created #1" in res
    assert len(s.memory.todos) == 1
    assert s.memory.todos[0].title == "写大纲"
    assert s.memory.todos[0].status == "PLANNED"

    res2 = _run(t.run({"action": "create", "title": "B"}, s))
    assert "#2" in res2
    assert len(s.memory.todos) == 2


def test_todo_list():
    t = Todo()
    s = Session(id="s")
    _run(t.run({"action": "create", "title": "A"}, s))
    _run(t.run({"action": "create", "title": "B"}, s))
    out = _run(t.run({"action": "list"}, s))
    assert "A" in out and "B" in out and "#1" in out and "#2" in out


def test_todo_update():
    t = Todo()
    s = Session(id="s")
    _run(t.run({"action": "create", "title": "A"}, s))
    res = _run(t.run({"action": "update", "id": "1", "status": "IN_PROGRESS"}, s))
    assert "updated" in res.lower()
    assert s.memory.todos[0].status == "IN_PROGRESS"


def test_todo_update_bad_status():
    t = Todo()
    s = Session(id="s")
    _run(t.run({"action": "create", "title": "A"}, s))
    res = _run(t.run({"action": "update", "id": "1", "status": "WRONG"}, s))
    assert res.startswith("ERROR")


def test_todo_update_missing_id():
    t = Todo()
    s = Session(id="s")
    res = _run(t.run({"action": "update", "id": "99", "status": "DONE"}, s))
    assert res.startswith("ERROR")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py::test_todo_create -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 todo.py**

```python
# tools/todo.py
from __future__ import annotations

_VALID_STATUS = {"PLANNED", "IN_PROGRESS", "DONE"}


class Todo:
    """任务列表 CRUD，写 session.memory.todos（跨轮次载体）。"""

    name = "todo"
    description = "管理任务列表：create/list/update。用于规划与跨轮次追踪进度。"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "update"]},
            "title": {"type": "string", "description": "create 时必填"},
            "id": {"type": "string", "description": "update 时必填，如 1"},
            "status": {"type": "string", "enum": ["PLANNED", "IN_PROGRESS", "DONE"]},
        },
        "required": ["action"],
    }

    async def run(self, args: dict, session) -> str:
        action = args.get("action")
        if action == "create":
            title = args.get("title")
            if not title:
                return "ERROR: create 需要 title"
            # session 内自增 id（取最大 +1）
            next_id = max((int(t.id) for t in session.memory.todos if t.id.isdigit()),
                          default=0) + 1
            from session.models import TodoItem
            item = TodoItem(id=str(next_id), title=title, status="PLANNED")
            session.memory.todos.append(item)
            return f"created #{next_id}: {title}"

        if action == "list":
            if not session.memory.todos:
                return "(空)"
            return "\n".join(
                f"[#{t.id}] {t.title} [{t.status}]" for t in session.memory.todos
            )

        if action == "update":
            tid = args.get("id")
            status = args.get("status")
            if status not in _VALID_STATUS:
                return f"ERROR: status 必须是 {_VALID_STATUS}"
            for t in session.memory.todos:
                if t.id == tid:
                    t.status = status
                    return f"updated #{tid} -> {status}"
            return f"ERROR: 未找到 id={tid}"

        return f"ERROR: 未知 action: {action}"
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/todo.py tests/test_tools.py
git commit -m "feat(tools): todo CRUD writing session.memory"
```

---

## Task 11: llm/client.py（DeepSeek 封装，可注入 mock）

**Files:**
- Create: `agent_framework/llm/client.py`
- Test: `agent_framework/tests/test_llm_client.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_llm_client.py
import pytest
from llm.client import LLMClient, LLMResponse, ToolCallResult


class FakeOpenAI:
    """模拟 openai SDK 的 chat.completions。"""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    class _Completions:
        def __init__(self, outer): self.outer = outer
        def create(self, **kw):
            self.outer.calls.append(kw)
            return self.outer._responses.pop(0)

    def __getattr__(self, name):
        if name == "chat":
            obj = type("Chat", (), {})()
            obj.completions = self._Completions(self)
            return obj
        raise AttributeError(name)


def _mk_choice(text="", tool_calls=None):
    msg = type("M", (), {"content": text, "tool_calls": tool_calls})()
    return type("Choice", (), {"message": msg})()


def test_respond_no_tools():
    fake = FakeOpenAI([_mk_choice(text="hello")])
    c = LLMClient(client=fake)
    out = c.respond(messages=[], user_input="hi")
    assert out == "hello"
    # 没传 tools
    assert "tools" not in fake.calls[0]


def test_chat_with_tools_returns_text():
    fake = FakeOpenAI([_mk_choice(text="42")])
    c = LLMClient(client=fake)
    resp = c.chat_with_tools(messages=[], tools=[])
    assert resp.text == "42"
    assert resp.tool_calls == []


def test_chat_with_tools_parses_tool_call():
    tc = type("TC", (), {
        "id": "call_1",
        "function": type("F", (), {
            "name": "calculator",
            "arguments": '{"expr": "1+1"}',
        })(),
    })()
    fake = FakeOpenAI([_mk_choice(text=None, tool_calls=[tc])])
    c = LLMClient(client=fake)
    resp = c.chat_with_tools(messages=[], tools=[])
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].args == {"expr": "1+1"}


def test_missing_api_key_raises():
    with pytest.raises(RuntimeError):
        LLMClient.from_env(api_key="")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 client.py**

```python
# llm/client.py
from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass
class ToolCallResult:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCallResult]


class LLMClient:
    """DeepSeek (OpenAI 兼容) 封装。client 可注入便于 mock。"""

    def __init__(self, client, model: str = "deepseek-chat"):
        self._client = client
        self.model = model

    @classmethod
    def from_env(cls, api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat") -> "LLMClient":
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY missing")
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        return cls(client, model=model)

    def respond(self, messages: list[dict], user_input: str) -> str:
        """Router=direct 路径，不传 tools。"""
        msgs = list(messages) + [{"role": "user", "content": user_input}]
        resp = self._client.chat.completions.create(model=self.model, messages=msgs)
        return resp.choices[0].message.content or ""

    def chat_with_tools(self, messages: list[dict],
                        tools: list[dict]) -> LLMResponse:
        """function-calling 路径。tools 为空时仍调用（LLM 可纯文本回）。"""
        kw = {"model": self.model, "messages": messages}
        if tools:
            kw["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self._client.chat.completions.create(**kw)
        msg = resp.choices[0].message
        text = msg.content or ""
        tcs: list[ToolCallResult] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tcs.append(ToolCallResult(id=tc.id, name=tc.function.name, args=args))
        return LLMResponse(text=text, tool_calls=tcs)

    def synthesize(self, plan: list[str], results: dict) -> str:
        """Planner 路径末尾合成最终答案。"""
        lines = [f"plan: {plan}",
                 "results: " + json.dumps(results, ensure_ascii=False, default=str)]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "\n".join(lines)
                       + "\n\n基于以上步骤结果，合成最终回答。"}],
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_llm_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/client.py tests/test_llm_client.py
git commit -m "feat(llm): DeepSeek client wrapper with injectable mock"
```

---

## Task 12: S1 收尾——全量测试 + README 段

**Files:**
- Create: `agent_framework/README.md`

- [ ] **Step 1: 跑全量检查（STYLE.md 提交前四条）**

Run:
```bash
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest -q
```
Expected: ruff 零报错；mypy 零报错；pytest 全绿（约 33 passed：config 2 + models 4 + store 4 + trace 2 + fsm 6 + tools 13 + llm 4）。

> 若 mypy strict 报 plan 示意代码的类型缺漏（如 `session` 未注解、`client` 无类型），按 STYLE.md 补全后重跑，不放宽 strict。

- [ ] **Step 2: 写 README.md（S1 段）**

````markdown
# agent-framework

从零实现的最小可用 Agent（自实现 runtime，不依赖 LangChain/OpenHands）。

## 状态
- [x] S1 基础设施（config/llm/tools/session/store/trace/fsm）
- [ ] S2 runtime 核心
- [ ] S3 REPLANNING
- [ ] S4 ReWOO
- [ ] S5 Web
- [ ] S6 集成 + 文档

## 开发
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

## 架构
详见 `docs/superpowers/specs/2026-06-13-agent-framework-design.md`。
````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with S1 status + dev setup"
```

---

## Self-Review

**1. Spec 覆盖（S1 范围）:**
- §2 模块清单 config/llm/tools/session/store/trace/fsm → Task 2/11/7-10/3/4/5/6 ✓
- §4 工具 calculator/search/todo → Task 8/9/10 ✓
- §5.1 数据结构 TodoItem/Memory/Message/Session + workspace → Task 3 ✓
- §5.2 持久化 原子写 + 损坏恢复 → Task 4 ✓
- §6 异常（损坏 JSON / 路径穿越）→ Task 4 ✓
- §7 trace（jsonl + replan/rewoo 事件）→ Task 5 ✓
- §1.3 FSM 全状态含 REPLANNING → Task 6 ✓
- 附录 A config 项 → Task 2 ✓
- 附录 B 依赖 → Task 1 ✓
- §8 LLM 封装 chat_with_tools/respond/synthesize → Task 11 ✓
- 覆盖完整，无 S1 范围遗漏。

**2. 占位符扫描:** 无 TBD/TODO/"add error handling" 裸描述。每步含实际代码与命令。✓

**3. 类型一致性:**
- `ToolCall`（base.py）vs `ToolCallResult`（client.py）——故意分开：前者注册表层，后者 LLM 返回层，名字不同是设计。Agent（S2）做映射。已记录。
- `Session.memory.todos` 类型 `list[TodoItem]`，todo.py append TodoItem ✓
- `TraceLogger.log_replan(count, reason, revised_steps)` 签名 Task 5 定义，S3 调用须匹配。
- `LLMResponse.tool_calls: list[ToolCallResult]`，S2 executor 用 `.name/.args` ✓
- `SessionFSM.transition(State.X)` 全测试用 State enum ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-13-agent-framework-s1-impl.md`. Two execution options:

**1. Subagent-Driven (recommended)** - 每个 Task 派 fresh subagent，task 间 review，快速迭代。

**2. Inline Execution** - 本会话内用 executing-plans 批量执行 + checkpoint。

哪个？
