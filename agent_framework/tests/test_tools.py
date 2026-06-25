import asyncio
from typing import ClassVar

import pytest

from session.models import Session
from tools.base import ToolCall, ToolRegistry
from tools.calculator import Calculator
from tools.memory import ReadMemoryBody, WriteMemory
from tools.search import Search
from tools.todo import Todo


class FakeTool:
    name: ClassVar[str] = "fake"
    description: ClassVar[str] = "fake tool"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args: dict, session) -> str:
        return f"fake:{args}"


def test_register_and_dispatch():
    reg = ToolRegistry()
    reg.register(FakeTool())
    assert any(s["function"]["name"] == "fake" for s in reg.schemas())

    res = asyncio.run(
        reg.dispatch(ToolCall(name="fake", args={"x": 1}), Session(id="s"))
    )
    assert res == "fake:{'x': 1}"


def test_dispatch_unknown_raises():
    reg = ToolRegistry()

    with pytest.raises(KeyError):
        asyncio.run(reg.dispatch(ToolCall(name="nope", args={}), Session(id="s")))


def test_schemas_export():
    reg = ToolRegistry()
    reg.register(FakeTool())
    sch = reg.schemas()
    assert sch == [
        {
            "type": "function",
            "function": {
                "name": "fake",
                "description": "fake tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_calc_basic():
    c = Calculator()
    assert asyncio.run(c.run({"expr": "1 + 2 * 3"}, Session(id="s"))) == "7"
    assert asyncio.run(c.run({"expr": "(10 - 4) / 2"}, Session(id="s"))) == "3.0"


def test_calc_rejects_injection():
    c = Calculator()
    for bad in ["__import__('os')", "open('x')", "1; import os", "pow(2,3)"]:
        res = asyncio.run(c.run({"expr": bad}, Session(id="s")))
        assert res.startswith("ERROR"), f"should reject: {bad}"


def test_calc_schema():
    c = Calculator()
    assert c.name == "calculator"
    assert c.parameters["type"] == "object"


def test_search_hit():
    s = Search()
    res = asyncio.run(s.run({"query": "DeepSeek"}, Session(id="s")))
    assert "DeepSeek" in res


def test_search_miss():
    s = Search()
    res = asyncio.run(s.run({"query": "zzznotexistzzz"}, Session(id="s")))
    assert "无结果" in res or res == ""


def _run(coro):
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


def test_todo_list_empty():
    t = Todo()
    s = Session(id="s")
    assert _run(t.run({"action": "list"}, s)) == "(empty)"


def test_todo_create_missing_title():
    t = Todo()
    s = Session(id="s")
    res = _run(t.run({"action": "create"}, s))
    assert res.startswith("ERROR")


def test_todo_unknown_action():
    t = Todo()
    s = Session(id="s")
    res = _run(t.run({"action": "delete", "id": "1"}, s))
    assert res.startswith("ERROR")


def test_write_memory_success_and_read_body():
    tool = WriteMemory()
    s = Session(id="s")

    out = _run(
        tool.run(
            {
                "type": "user",
                "name": "喜欢的模型",
                "description": "记录偏好",
                "content": "喜欢简洁回答",
                "keywords": ["偏好", "回答"],
            },
            s,
        )
    )
    assert out == "1"
    assert len(s.memory.entries) == 1
    assert s.memory.entries[0].name == "喜欢的模型"

    body = _run(ReadMemoryBody().run({"id": "1"}, s))
    assert body == "喜欢简洁回答"


def test_write_memory_invalid_type():
    tool = WriteMemory()
    s = Session(id="s")
    out = _run(
        tool.run(
            {
                "type": "bad",
                "name": "x",
                "description": "d",
                "content": "c",
                "keywords": [],
            },
            s,
        )
    )
    assert out.startswith("ERROR")


def test_write_memory_feedback_needs_why():
    tool = WriteMemory()
    s = Session(id="s")
    out = _run(
        tool.run(
            {
                "type": "feedback",
                "name": "code review",
                "description": "d",
                "content": "Rule: keep tests small\nHow to apply: write one assertion",
                "keywords": [],
            },
            s,
        )
    )
    assert out.startswith("ERROR")


def test_write_memory_rejects_code_and_paths_and_git():
    tool = WriteMemory()
    s = Session(id="s")

    bad_samples = [
        {
            "type": "user",
            "name": "code",
            "description": "d",
            "content": "def hello():\n    return 1",
            "keywords": [],
        },
        {
            "type": "user",
            "name": "path",
            "description": "d",
            "content": r"C:\\tmp\\file.py",
            "keywords": [],
        },
        {
            "type": "user",
            "name": "git",
            "description": "d",
            "content": "git rebase --hard HEAD~1",
            "keywords": [],
        },
        {
            "type": "user",
            "name": "assignment",
            "description": "d",
            "content": "x = 1",
            "keywords": [],
        },
        {
            "type": "user",
            "name": "loop",
            "description": "d",
            "content": "for i in range(3): pass",
            "keywords": [],
        },
    ]
    results = [_run(tool.run(sample, s)) for sample in bad_samples]
    assert all(res.startswith("ERROR") for res in results)


def test_write_memory_project_rejects_relative_time():
    tool = WriteMemory()
    s = Session(id="s")
    relative_contents = [
        "Rule: 记录项目决策\nWhy: 方便回顾\nHow to apply: 写入绝对日期, 今天完成",
        "Rule: log dates\nWhy: avoid stale context\nHow to apply: updated 2 days ago",
        "Rule: 记录项目决策\nWhy: 方便回顾\nHow to apply: 明年再处理",
        "Rule: 记录项目决策\nWhy: 方便回顾\nHow to apply: 昨晚修过",
    ]
    results = [
        _run(
            tool.run(
                {
                    "type": "project",
                    "name": f"计划-{i}",
                    "description": "d",
                    "content": content,
                    "keywords": [],
                },
                s,
            )
        )
        for i, content in enumerate(relative_contents)
    ]
    assert all(res.startswith("ERROR") for res in results)


def test_write_memory_dedup_updates_existing_entry():
    tool = WriteMemory()
    s = Session(id="s")

    first = _run(
        tool.run(
            {
                "type": "reference",
                "name": "API",
                "description": "first",
                "content": "v1",
                "keywords": ["a"],
            },
            s,
        )
    )
    second = _run(
        tool.run(
            {
                "type": "reference",
                "name": "api",
                "description": "second",
                "content": "v2",
                "keywords": ["b"],
            },
            s,
        )
    )

    assert first == "1"
    assert second == "1"
    assert len(s.memory.entries) == 1
    entry = s.memory.entries[0]
    assert entry.description == "second"
    assert entry.content == "v2"
    assert entry.keywords == ["b"]
    assert entry.name == "api"
