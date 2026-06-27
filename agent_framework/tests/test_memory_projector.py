from __future__ import annotations

from runtime.memory_projector import (
    build_memory_attachments,
    build_memory_context_message,
    build_system_prompt,
)
from session.models import Memory, MemoryEntry, Step, TodoItem


def test_build_system_prompt_empty_memory():
    assert build_system_prompt(Memory()) == "You are a helpful agent."


def test_build_system_prompt_includes_state_sections():
    memory = Memory(
        todos=[TodoItem(id="1", title="ship plan", status="IN_PROGRESS")],
        plan=[Step(prompt="write tests")],
        lessons=["check tool args"],
    )

    out = build_system_prompt(memory)

    assert "Todos:" in out
    assert "- [#1] ship plan [IN_PROGRESS]" in out
    assert "Plan: write tests" in out
    assert "Lessons learned:" in out
    assert "- check tool args" in out


def test_build_memory_context_message_includes_project_context_and_index_only():
    memory = Memory(
        entries=[
            MemoryEntry(
                id="mem-1",
                type="project",
                name="architecture",
                description="turn runner decision",
                keywords=["agent", "executor"],
                content="secret body",
                saved_at="2026-06-27T00:00:00+00:00",
            )
        ]
    )

    msg = build_memory_context_message(memory, project_context="Project rules")

    assert msg == {
        "role": "user",
        "content": "Project rules\n\nMemory index:\n- id=mem-1 type=project name=architecture description=turn runner decision keywords=agent,executor saved_at=2026-06-27T00:00:00+00:00",
    }
    assert "secret body" not in msg["content"]


def test_build_memory_context_message_returns_none_for_empty_inputs():
    assert build_memory_context_message(Memory(), project_context="") is None


def test_build_memory_attachments_preserves_state_verbatim():
    memory = Memory(
        todos=[TodoItem(id="1", title="ship", status="DONE")],
        plan=[Step(prompt="step A")],
        workspace={"E1": "result"},
    )

    out = build_memory_attachments(memory)

    assert out == "\n".join(
        [
            "--- Attachments (state info, verbatim) ---",
            "Todos:",
            "  - [#1] ship [DONE]",
            "Plan:",
            "  - step A",
            "Workspace:",
            '  {\n  "E1": "result"\n}',
        ]
    )
