import inspect

from runtime.planner import Planner, _parse_steps
from session.models import Memory


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
    # malformed JSON object -> fall back to line split
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


async def test_make_plan_detects_rewoo_cluster():
    llm = ScriptedLLM('{"rewoo_cluster": "analyze X and Y in parallel"}')
    planner = Planner(llm)
    plan = await planner.make_plan("compare X and Y", Memory())
    assert len(plan) == 1
    assert plan[0].is_rewoo_cluster is True
    assert plan[0].prompt == "analyze X and Y in parallel"


async def test_make_plan_plain_steps_when_no_cluster_key():
    llm = ScriptedLLM('{"steps": ["a", "b"]}')
    planner = Planner(llm)
    plan = await planner.make_plan("do a then b", Memory())
    assert [s.prompt for s in plan] == ["a", "b"]
    assert all(not s.is_rewoo_cluster for s in plan)
