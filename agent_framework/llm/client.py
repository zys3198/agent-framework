from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI

log = logging.getLogger(__name__)


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
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # Phase 2b recall/gather uses a separate model; falls back to self.model.
        self.recall_model = recall_model

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> LLMClient:
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY missing")
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        return cls(client, model=model, timeout=timeout, max_retries=max_retries)

    async def respond(self, messages: list[dict[str, Any]], user_input: str) -> str:
        """Router=direct path. No tools passed."""
        msgs = [*messages, {"role": "user", "content": user_input}]
        kw: dict[str, Any] = {"model": self.model, "messages": msgs}
        start = time.perf_counter()
        resp = await self._client.chat.completions.create(**kw)
        _log_usage("respond", self.model, start, resp)
        return resp.choices[0].message.content or ""

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Function-calling path. `tools` already in standard OpenAI shape
        ({"type":"function","function":{...}}). Empty -> text-only response."""
        kw: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kw["tools"] = tools
        start = time.perf_counter()
        resp = await self._client.chat.completions.create(**kw)
        _log_usage("chat_with_tools", self.model, start, resp)
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
        self, plan: list[str], results: dict[str, Any], project_context: str = ""
    ) -> str:
        """Planner path: synthesize final answer from step results."""
        lines: list[str] = []
        if project_context.strip():
            lines.extend(["AGENTS context:", project_context.strip()])
        lines.extend(
            [
                f"plan: {plan}",
                "results: " + json.dumps(results, ensure_ascii=False, default=str),
            ]
        )
        start = time.perf_counter()
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
        _log_usage("synthesize", self.model, start, resp)
        return resp.choices[0].message.content or ""

def _log_usage(method: str, model: str, start: float, resp: Any) -> None:
    elapsed = time.perf_counter() - start
    u = getattr(resp, "usage", None)
    pt = u.prompt_tokens if u else 0
    ct = u.completion_tokens if u else 0
    tt = u.total_tokens if u else 0
    log.info("llm:%s model=%s in=%.3fs tok=%s/%s/%s", method, model, elapsed, pt, ct, tt)
