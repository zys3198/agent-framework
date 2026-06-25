from __future__ import annotations

import inspect

from runtime.router import Route, Router
from session.models import Memory


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[list[dict], str]] = []

    async def respond(self, messages: list[dict], user_input: str) -> str:
        self.calls.append((messages, user_input))
        return self._responses.pop(0)


async def test_classify_direct():
    router = Router(llm=FakeLLM(["DIRECT"]))
    assert await router.classify("你好", Memory()) == Route.DIRECT


async def test_classify_simple_tool():
    router = Router(llm=FakeLLM(["SIMPLE_TOOL"]))
    assert await router.classify("算 1+1", Memory()) == Route.SIMPLE_TOOL


async def test_classify_plan_required():
    router = Router(llm=FakeLLM(["PLAN_REQUIRED"]))
    assert await router.classify("帮我规划并完成 X", Memory()) == Route.PLAN_REQUIRED


async def test_classify_default_on_garbage():
    router = Router(llm=FakeLLM(["不知道"]))
    assert await router.classify("hi", Memory()) == Route.DIRECT


def test_classify_is_async():
    assert inspect.iscoroutinefunction(Router.classify)
