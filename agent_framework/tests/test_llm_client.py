import pytest

from llm.client import LLMClient, LLMResponse, ToolCallResult  # noqa: F401


class FakeOpenAI:
    """Mock openai SDK chat.completions."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kw):
            self.outer.calls.append(kw)
            return self.outer._responses.pop(0)

    def __getattr__(self, name):
        if name == "chat":
            obj = type("Chat", (), {})()
            obj.completions = self._Completions(self)
            return obj
        raise AttributeError(name)


def _mk_choice(text="", tool_calls=None):
    msg = type("M", (), {"content": text, "tool_calls": tool_calls})()
    choice = type("Choice", (), {"message": msg})()
    return type("Resp", (), {"choices": [choice]})()


def test_respond_no_tools():
    fake = FakeOpenAI([_mk_choice(text="hello")])
    c = LLMClient(client=fake)
    out = c.respond(messages=[], user_input="hi")
    assert out == "hello"
    assert "tools" not in fake.calls[0]


def test_chat_with_tools_returns_text():
    fake = FakeOpenAI([_mk_choice(text="42")])
    c = LLMClient(client=fake)
    resp = c.chat_with_tools(messages=[], tools=[])
    assert resp.text == "42"
    assert resp.tool_calls == []


def test_chat_with_tools_parses_tool_call():
    tc = type(
        "TC",
        (),
        {
            "id": "call_1",
            "function": type(
                "F",
                (),
                {
                    "name": "calculator",
                    "arguments": '{"expr": "1+1"}',
                },
            )(),
        },
    )()
    fake = FakeOpenAI([_mk_choice(text=None, tool_calls=[tc])])
    c = LLMClient(client=fake)
    resp = c.chat_with_tools(messages=[], tools=[])
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].args == {"expr": "1+1"}


def test_missing_api_key_raises():
    with pytest.raises(RuntimeError):
        LLMClient.from_env(api_key="")
