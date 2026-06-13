from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from session.models import Session

# Mock corpus (half-width punctuation to avoid RUF001; semantics preserved)
_CORPUS: list[str] = [
    "DeepSeek 是 OpenAI 兼容的大模型 API, 支持 function calling.",
    "Agent 基本循环: 接收输入 -> 判断直接答/调工具 -> 执行 -> 读结果 -> 继续.",
    "Plan-and-Execute 架构先规划再分步执行, 适合复杂任务.",
    "ReWOO 把规划与执行解耦, planner 一次推理产出 DAG, 省 LLM round-trip.",
    "Reflexion 在工具失败后自评产出教训, 带教训重试.",
]


class Search:
    """Mock search over a preset corpus. No real network."""

    name: ClassVar[str] = "search"
    description: ClassVar[str] = (
        "Search the knowledge base (mock preset corpus) and return hits."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search keyword"}},
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any], session: Session) -> str:
        q = args.get("query", "")
        if not isinstance(q, str) or not q.strip():
            return "ERROR: query must be a non-empty string"
        hits = [c for c in _CORPUS if q.lower() in c.lower()]
        return "\n".join(f"- {h}" for h in hits) if hits else "无结果"
