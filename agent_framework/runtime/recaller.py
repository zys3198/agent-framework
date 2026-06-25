from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import MemoryEntry


_USAGE_KEYWORDS = ["用法", "how to use", "usage", "使用说明"]


class Recaller:
    """Memory recall filter.

    Uses LLM to select relevant entries by name+description (progressive
    disclosure). Applies tool-avoidance filter: when current_tool is set,
    excludes "usage/documentation" entries while keeping caveat ones.
    """

    def __init__(
        self, llm: LLMClient, recall_model: str | None = None
    ) -> None:
        self._llm = llm
        # TODO(2b): recall_model not wired; uses self.model
        self._recall_model = recall_model

    async def recall(
        self,
        query: str,
        entries: list[MemoryEntry],
        current_tool: str | None = None,
    ) -> list[str]:
        """Return relevant entry ids.

        1. LLM filters entries by name+description relevance.
        2. Tool-avoidance: when current_tool is set, exclude usage entries.
        """
        if not entries:
            return []

        candidates = [
            f"id={e.id} name={e.name} description={e.description}"
            for e in entries
        ]
        prompt = (
            "Memory entries:\n"
            + "\n".join(candidates)
            + f"\n\nQuery: {query}\n\n"
            'Return strict JSON {"ids": [...]} with ids of relevant entries. '
            "Be conservative \u2014 only clearly relevant."
        )
        resp = await self._llm.respond(
            [
                {
                    "role": "system",
                    "content": "You are a memory recall filter.",
                }
            ],
            prompt,
        )
        ids = self._parse_ids(resp or "")

        # Tool-avoidance filter
        if current_tool is not None and ids:
            ids = self.filter_tool_usage(ids, entries)

        return ids

    @staticmethod
    def filter_tool_usage(
        ids: list[str], entries: list[MemoryEntry]
    ) -> list[str]:
        """Exclude "usage/documentation" entries; keep caveat/bug entries.

        Pure local filter (no LLM round-trip), so the executor can apply it
        post-step-0 once the LLM-chosen tool name is known, on the ids the
        parallel recall already returned.
        """
        desc_lower = {e.id: e.description.lower() for e in entries}
        return [
            eid
            for eid in ids
            if not any(kw in desc_lower.get(eid, "") for kw in _USAGE_KEYWORDS)
        ]

    @staticmethod
    def _parse_ids(text: str) -> list[str]:
        """Extract id list from LLM response JSON."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            data: Any = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        ids = data.get("ids") if isinstance(data, dict) else None
        if not isinstance(ids, list):
            return []
        return [str(i) for i in ids if i]
