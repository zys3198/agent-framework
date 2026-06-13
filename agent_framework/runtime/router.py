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
