"""E2E tests against the real DeepSeek API. Skipped when no API key."""
from __future__ import annotations

import asyncio
import importlib
import os

import pytest
from dotenv import load_dotenv

# Load .env from agent_framework/ so os.environ has the key before skipif.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

KEY_AVAILABLE = bool(os.environ.get("DEEPSEEK_API_KEY"))
pytestmark = pytest.mark.skipif(not KEY_AVAILABLE, reason="DEEPSEEK_API_KEY not set")


def _build_agent():
    import config
    importlib.reload(config)
    if not config.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY empty")
    from main import build_agent
    return build_agent()


def test_e2e_direct_path():
    agent = _build_agent()
    resp = asyncio.run(agent.chat("e2e-direct", "What is 2+2? Answer with just the number."))
    assert "4" in resp


def test_e2e_tool_path():
    agent = _build_agent()
    resp = asyncio.run(
        agent.chat("e2e-tool", "Use the calculator to compute 15 * 37 and tell me the result.")
    )
    assert "555" in resp


def test_e2e_memory_write_then_recall():
    agent = _build_agent()
    asyncio.run(agent.chat("e2e-mem", "Write a memory entry: type=user, name=preferred_language, description=user prefers Python, keywords=[python], content=The user prefers Python."))
    resp = asyncio.run(agent.chat("e2e-mem", "What programming language do I prefer?"))
    assert "python" in resp.lower()


def test_e2e_compaction_triggers():
    agent = _build_agent()
    for i in range(20):
        asyncio.run(agent.chat("e2e-compact", f"Tell me fact number {i} about space."))
    import config
    from session.store import Store
    s = Store(config.SESSION_DIR).load("e2e-compact")
    assert len(s.messages) > 0
    has_marker = any("[COMPACT]" in (m.content or "") for m in s.messages)
    print(f"e2e-compact: {len(s.messages)} msgs, compacted={has_marker}")
