import asyncio
from typing import ClassVar

import pytest

from session.models import Session
from tools.base import ToolCall, ToolRegistry
from tools.calculator import Calculator
from tools.search import Search


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
