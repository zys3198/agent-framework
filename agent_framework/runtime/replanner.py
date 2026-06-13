from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from runtime.planner import _parse_steps

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Memory, Step

_REPLANNER_PROMPT = (
    "A step failed or hit a limit and needs a revised plan for the REMAINING work.\n"
    'Return ONLY JSON: {"steps": ["...", "..."]} listing the revised remaining\n'
    "steps. Empty list if no further step is needed."
)


def _build_context(
    remaining: list[Step], results: dict[int, Any], memory: Memory
) -> str:
    lines: list[str] = ["Remaining steps:"]
    lines.extend(f"- {s.prompt}" for s in remaining)
    if results:
        lines.append("Step results so far:")
        for idx, val in results.items():
            text = getattr(val, "text", val)
            lines.append(f"- step {idx}: {text}")
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {lesson}" for lesson in memory.lessons)
    return "\n".join(lines)


class Replanner:
    """Revise the remaining plan after a failed/limited step (族 D', macro self-correction)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def revise(
        self,
        remaining: list[Step],
        results: dict[int, Any],
        memory: Memory,
    ) -> list[Step]:
        from session.models import Step

        context = _build_context(remaining, results, memory)
        text = await asyncio.to_thread(
            self._llm.respond,
            [{"role": "system", "content": _REPLANNER_PROMPT}],
            context,
        )
        return [Step(prompt=p) for p in _parse_steps(text or "")]
