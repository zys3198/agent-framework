import inspect

from runtime.replanner import Replanner
from session.models import Memory, Step


class ScriptedLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.last_input = ""

    def respond(self, messages, user_input):
        self.last_input = user_input
        return self._reply


def _steps(*prompts: str) -> list[Step]:
    return [Step(prompt=p) for p in prompts]


async def test_revise_returns_steps():
    llm = ScriptedLLM('{"steps": ["retry with arg X", "then Y"]}')
    rp = Replanner(llm)
    out = await rp.revise(_steps("original"), {}, Memory())
    assert [s.prompt for s in out] == ["retry with arg X", "then Y"]


async def test_revise_empty_when_no_steps():
    llm = ScriptedLLM('{"steps": []}')
    rp = Replanner(llm)
    out = await rp.revise(_steps("x"), {}, Memory())
    assert out == []


async def test_revise_context_includes_remaining_results_lessons():
    llm = ScriptedLLM('{"steps": ["new"]}')
    rp = Replanner(llm)
    memory = Memory()
    memory.lessons.append("avoid empty query")

    class _O:
        text = "empty result"

    await rp.revise(_steps("step A", "step B"), {0: _O()}, memory)
    assert "step A" in llm.last_input
    assert "step B" in llm.last_input
    assert "empty result" in llm.last_input
    assert "avoid empty query" in llm.last_input


async def test_revise_bad_json_falls_back_to_lines():
    llm = ScriptedLLM("1. first fallback\n2. second fallback")
    rp = Replanner(llm)
    out = await rp.revise(_steps("x"), {}, Memory())
    assert [s.prompt for s in out] == ["first fallback", "second fallback"]


def test_revise_is_async():
    assert inspect.iscoroutinefunction(Replanner.revise)
