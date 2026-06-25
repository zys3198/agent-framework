from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass
class ToolCallResult:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCallResult]


class LLMClient:
    """DeepSeek (OpenAI-compatible) wrapper. Injectable client for tests."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "deepseek-chat",
        recall_model: str | None = None,
    ) -> None:
        self._client = client
        self.model = model
        # Phase 2b recall/gather uses a separate model; falls back to self.model.
        self.recall_model = recall_model

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ) -> LLMClient:
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY missing")
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return cls(client, model=model)

    async def respond(self, messages: list[dict[str, Any]], user_input: str) -> str:
        """Router=direct path. No tools passed."""
        msgs = [*messages, {"role": "user", "content": user_input}]
        kw: dict[str, Any] = {"model": self.model, "messages": msgs}
        resp = await self._client.chat.completions.create(**kw)
        return resp.choices[0].message.content or ""

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Function-calling path. `tools` already in standard OpenAI shape
        ({"type":"function","function":{...}}). Empty -> text-only response."""
        kw: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kw["tools"] = tools
        resp = await self._client.chat.completions.create(**kw)
        msg = resp.choices[0].message
        text = msg.content or ""
        tcs: list[ToolCallResult] = []
        for tc in msg.tool_calls or []:
            try:
                args: dict[str, Any] = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tcs.append(ToolCallResult(id=tc.id, name=tc.function.name, args=args))
        return LLMResponse(text=text, tool_calls=tcs)

    async def synthesize(
        self, plan: list[str], results: dict[str, Any], claude_context: str = ""
    ) -> str:
        """Planner path: synthesize final answer from step results."""
        lines: list[str] = []
        if claude_context.strip():
            lines.extend(["CLAUDE context:", claude_context.strip()])
        lines.extend(
            [
                f"plan: {plan}",
                "results: " + json.dumps(results, ensure_ascii=False, default=str),
            ]
        )
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "\n".join(lines)
                    + "\n\nSynthesize a final answer from the step results above.",
                }
            ],
        )
        return resp.choices[0].message.content or ""
