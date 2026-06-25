from __future__ import annotations

from runtime.recaller import Recaller
from session.models import MemoryEntry


class FakeLLM:
    """Stub LLM that returns queued respond results."""

    def __init__(self, respond_results: list[str]):
        self._respond_results = list(respond_results)
        self.respond_calls: list[dict[str, object]] = []

    async def respond(self, messages: list[dict], user_input: str) -> str:
        self.respond_calls.append(
            {"system": messages[0]["content"] if messages else "", "user_input": user_input}
        )
        return self._respond_results.pop(0)


_ENTRY_FIXTURES = [
    MemoryEntry(
        id="usage-1", type="user", name="calc用法",
        description="calculator 用法说明",
        content="怎么用计算器", saved_at="2026-06-25T00:00:00+00:00",
    ),
    MemoryEntry(
        id="caveat-1", type="user", name="calc踩坑",
        description="calculator 踩坑记录: int溢出",
        content="int 溢出注意", saved_at="2026-06-24T00:00:00+00:00",
    ),
    MemoryEntry(
        id="unrelated-1", type="user", name="python基础",
        description="Python 基础语法", content="变量定义",
        saved_at="2026-06-23T00:00:00+00:00",
    ),
]


async def test_recaller_filters_usage_when_tool_active():
    """current_tool 设置时排除用法类条目, 保留踩坑类."""
    llm = FakeLLM(['{"ids": ["usage-1", "caveat-1"]}'])
    r = Recaller(llm=llm)
    ids = await r.recall("calculator", _ENTRY_FIXTURES, current_tool="calculator")
    assert "usage-1" not in ids
    assert "caveat-1" in ids
    assert len(ids) == 1


async def test_recaller_no_tool_no_filter():
    """current_tool=None 时不过滤."""
    llm = FakeLLM(['{"ids": ["usage-1", "caveat-1", "unrelated-1"]}'])
    r = Recaller(llm=llm)
    ids = await r.recall("calculator", _ENTRY_FIXTURES)
    assert len(ids) == 3
    assert "usage-1" in ids
    assert "caveat-1" in ids
    assert "unrelated-1" in ids


async def test_recaller_empty_entries():
    llm = FakeLLM(['{"ids": []}'])
    r = Recaller(llm=llm)
    ids = await r.recall("anything", [])
    assert ids == []


async def test_recaller_parse_bad_json():
    """LLM 返回非 JSON 时安全降级."""
    llm = FakeLLM(["I don't know"])
    r = Recaller(llm=llm)
    ids = await r.recall("test", _ENTRY_FIXTURES[:1])
    assert ids == []

async def test_recaller_parse_json_wrapped_in_prose():
    r = Recaller(FakeLLM(['Sure, here: ```json\n{"ids": ["caveat-1"]}\n```']))
    ids = await r.recall("calculator", _ENTRY_FIXTURES)
    assert ids == ["caveat-1"]
