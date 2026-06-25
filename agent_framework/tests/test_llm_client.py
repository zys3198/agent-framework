import logging

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

        async def create(self, **kw):
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


async def test_respond_no_tools():
    fake = FakeOpenAI([_mk_choice(text="hello")])
    c = LLMClient(client=fake)
    out = await c.respond(messages=[], user_input="hi")
    assert out == "hello"
    assert "tools" not in fake.calls[0]


async def test_chat_with_tools_returns_text():
    fake = FakeOpenAI([_mk_choice(text="42")])
    c = LLMClient(client=fake)
    resp = await c.chat_with_tools(messages=[], tools=[])
    assert resp.text == "42"
    assert resp.tool_calls == []


async def test_chat_with_tools_parses_tool_call():
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
    resp = await c.chat_with_tools(messages=[], tools=[])
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].args == {"expr": "1+1"}


def test_missing_api_key_raises():
    with pytest.raises(RuntimeError):
        LLMClient.from_env(api_key="")


async def test_synthesize_includes_claude_context_when_present():
    fake = FakeOpenAI([_mk_choice(text="final")])
    c = LLMClient(client=fake)

    out = await c.synthesize(["step"], {"0": "done"}, claude_context="User CLAUDE\nbe terse")

    assert out == "final"
    messages = fake.calls[0]["messages"]
    assert "User CLAUDE" in messages[0]["content"]
    assert "be terse" in messages[0]["content"]


async def test_synthesize_omits_claude_context_when_empty():
    fake = FakeOpenAI([_mk_choice(text="final")])
    c = LLMClient(client=fake)

    out = await c.synthesize(["step"], {"0": "done"})

    assert out == "final"
    content = fake.calls[0]["messages"][0]["content"]
    assert "CLAUDE context:" not in content


# -- Phase 3: usage logging + timeout/max_retries --

def _mk_usage(pt=10, ct=20, tt=30):
    return type("Usage", (), {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt})()


def _mk_resp_with_usage(text="", usage=None):
    msg = type("M", (), {"content": text, "tool_calls": None})()
    choice = type("Choice", (), {"message": msg})()
    attrs = {"choices": [choice]}
    if usage is not None:
        attrs["usage"] = usage
    return type("Resp", (), attrs)()


async def test_respond_logs_usage(caplog):
    caplog.set_level(logging.INFO)
    usage = _mk_usage(10, 20, 30)
    fake = FakeOpenAI([_mk_resp_with_usage("ok", usage)])
    c = LLMClient(client=fake)
    await c.respond([], "hi")
    assert "llm:respond" in caplog.text
    assert "10/20/30" in caplog.text


async def test_respond_usage_none_does_not_crash(caplog):
    caplog.set_level(logging.INFO)
    fake = FakeOpenAI([_mk_resp_with_usage("ok")])
    c = LLMClient(client=fake)
    await c.respond([], "hi")
    assert "llm:respond" in caplog.text
    assert "0/0/0" in caplog.text


async def test_chat_with_tools_logs_usage(caplog):
    caplog.set_level(logging.INFO)
    usage = _mk_usage(5, 15, 20)
    fake = FakeOpenAI([_mk_resp_with_usage("42", usage)])
    c = LLMClient(client=fake)
    await c.chat_with_tools([], [])
    assert "llm:chat_with_tools" in caplog.text


async def test_synthesize_logs_usage(caplog):
    caplog.set_level(logging.INFO)
    usage = _mk_usage(8, 12, 20)
    fake = FakeOpenAI([_mk_resp_with_usage("final", usage)])
    c = LLMClient(client=fake)
    await c.synthesize(["a"], {"0": "done"})
    assert "llm:synthesize" in caplog.text


def test_from_env_accepts_timeout_max_retries():
    import pytest
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        LLMClient.from_env(api_key="", timeout=15.0, max_retries=1)


def test_init_stores_timeout_max_retries():
    fake_client = object()
    c = LLMClient(client=fake_client, timeout=42.0, max_retries=7)
    assert c.timeout == 42.0
    assert c.max_retries == 7
