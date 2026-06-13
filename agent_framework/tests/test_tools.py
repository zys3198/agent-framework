import asyncio
from typing import ClassVar

import pytest

from session.models import Session
from tools.base import ToolCall, ToolRegistry
from tools.calculator import Calculator
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
    assert "fake" in reg.names()

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
            "name": "fake",
            "description": "fake tool",
            "parameters": {"type": "object", "properties": {}},
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
