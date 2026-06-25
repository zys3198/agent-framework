from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from session.models import RECENT_LESSONS_LIMIT

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Memory, Step

_PLANNER_PROMPT = (
    "You decompose a task into ordered steps for a tool-using executor.\n"
    'Return ONLY JSON: {"steps": ["...", "..."]}.\n'
    "Each step is one self-contained instruction. Empty list if no step is needed."
)

_LEAD = re.compile(r"^\s*(\d+[\.\)]|[-*])\s*")


def _parse_steps(text: str) -> list[str]:
    """Extract step prompts from LLM text.

    Prefer a JSON object {"steps": [...]} embedded anywhere in the text.
    Fall back to one prompt per non-empty line (stripping leading numbering
    and bullet markers). Empty input -> [].
    """
    if not text:
        return []
    data = _extract_json(text)
    if data and isinstance(data.get("steps"), list):
        return [str(s).strip() for s in data["steps"] if str(s).strip()]
    out: list[str] = []
    for line in text.splitlines():
        cleaned = _LEAD.sub("", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class Planner:
    """Produce an ordered step list for a complex task (族 D, main path)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def make_plan(
        self, user_input: str, memory: Memory, project_context: str = ""
    ) -> list[Step]:
        from session.models import Step

        messages = [
            {"role": "system", "content": _PLANNER_PROMPT},
            {
                "role": "user",
                "content": _build_memory_context(
                    memory, project_context=project_context
                ),
            },
        ]
        text = await self._llm.respond(messages, user_input)
        return [Step(prompt=p) for p in _parse_steps(text or "")]


def _build_memory_context(memory: Memory, project_context: str = "") -> str:
    """Surface memory into the planner prompt (was a dead param)."""
    lines: list[str] = []
    if project_context.strip():
        lines.append("AGENTS context:")
        lines.append(project_context.strip())
    if memory.todos:
        lines.append("Active todos:")
        lines.extend(f"- {t.title} [{t.status}]" for t in memory.todos)
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {lesson}" for lesson in memory.lessons[-RECENT_LESSONS_LIMIT:])
    return "\n".join(lines) if lines else "No prior context."
