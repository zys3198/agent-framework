import inspect

from runtime.planner import Planner, _parse_steps
from session.models import Memory, TodoItem


class ScriptedLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    def respond(self, messages, user_input):
        self.calls += 1
        return self._reply


def test_parse_steps_json():
    steps = _parse_steps('{"steps": ["do A", "do B", "do C"]}')
    assert steps == ["do A", "do B", "do C"]


def test_parse_steps_strips_numbering_and_bullets():
    text = "1. first\n2) second\n- third\n* fourth"
    assert _parse_steps(text) == ["first", "second", "third", "fourth"]


def test_parse_steps_ignores_blank_lines():
    assert _parse_steps("only one\n\n\n") == ["only one"]


def test_parse_steps_empty():
    assert _parse_steps("") == []
    assert _parse_steps("   \n  ") == []


def test_parse_steps_json_inside_prose():
    text = 'Here is the plan:\n{"steps": ["a", "b"]}\nHope it helps.'
    assert _parse_steps(text) == ["a", "b"]


def test_parse_steps_bad_json_falls_back_to_lines():
    text = '{"steps": [broken\nsecond line'
    out = _parse_steps(text)
    assert "second line" in out


async def test_make_plan_returns_steps():
    llm = ScriptedLLM('{"steps": ["search X", "calculate Y"]}')
    planner = Planner(llm)
    plan = await planner.make_plan("do X then Y", Memory())
    assert [s.prompt for s in plan] == ["search X", "calculate Y"]
    assert llm.calls == 1


async def test_make_plan_empty_when_no_steps():
    llm = ScriptedLLM("")
    planner = Planner(llm)
    plan = await planner.make_plan("vague", Memory())
    assert plan == []


def test_make_plan_is_async():
    assert inspect.iscoroutinefunction(Planner.make_plan)


async def test_make_plan_plain_steps():
    llm = ScriptedLLM('{"steps": ["a", "b"]}')
    planner = Planner(llm)
    plan = await planner.make_plan("do a then b", Memory())
    assert [s.prompt for s in plan] == ["a", "b"]


async def test_make_plan_surfaces_memory_context():
    """memory was a dead param; now todos/lessons reach the planner prompt."""
    seen: dict[str, object] = {}

    class CapturingLLM:
        def respond(self, messages, user_input):
            seen["messages"] = messages
            return '{"steps": ["a"]}'

    mem = Memory(
        todos=[TodoItem(id="1", title="buy milk", status="PLANNED")],
        lessons=["always validate input"],
    )
    planner = Planner(CapturingLLM())  # type: ignore[arg-type]
    await planner.make_plan("go", mem)
    body = "\n".join(m["content"] for m in seen["messages"])
    assert "buy milk" in body
    assert "always validate input" in body


async def test_make_plan_limits_lessons_and_includes_claude_context():
    seen: dict[str, object] = {}

    class CapturingLLM:
        def respond(self, messages, user_input):
            seen["messages"] = messages
            return '{"steps": ["a"]}'

    mem = Memory(lessons=[f"lesson-{i}" for i in range(21)])
    planner = Planner(CapturingLLM())  # type: ignore[arg-type]
    await planner.make_plan("go", mem, claude_context="Project CLAUDE\nfollow rules")

    body = "\n".join(m["content"] for m in seen["messages"])
    assert "Project CLAUDE" in body
    assert "follow rules" in body
    assert "lesson-0" not in body
    assert "lesson-1" in body
    assert "lesson-20" in body
    assert len(mem.lessons) == 21
