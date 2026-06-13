from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from session.models import Session


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    async def run(self, args: dict[str, Any], session: Session) -> str: ...


class ToolRegistry:
    """register + schema export + dispatch. does not implement concrete tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    async def dispatch(self, call: ToolCall, session: Session) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            raise KeyError(f"unknown tool: {call.name}")
        return await tool.run(call.args, session)
