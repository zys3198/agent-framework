import asyncio
from typing import ClassVar

import pytest

from session.models import Session
from tools.base import ToolCall, ToolRegistry


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
