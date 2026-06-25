"""Layered compaction (Phase 3, demo-trimmed to 3 layers).

Layer 1: large result spillover -- tool results > threshold written to disk,
         message keeps a short preview. Original reclaimable via Read.
Layer 2: microcompact -- drops old tool results, keeps only recent N.
         State info (todos/plan/workspace) lives in session.memory, untouched.
Layer 3: Auto-Compact -- when estimated tokens exceed threshold, entire
         conversation sent to LLM summarizer. Output is a 3-segment chain:
         boundaryMarker + summary + attachments.

Safety net: circuit breaker (N consecutive summary failures) + recursion guard
(messages that already look like a compaction output are not re-compacted).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING

from session.models import Message

if TYPE_CHECKING:
    from llm.client import LLMClient
    from session.models import Session

log = logging.getLogger(__name__)

_BOUNDARY_PREFIX = "[COMPACT]"
_RECURSION_MARKERS = ("[COMPACT]", "session continuation", "interrupted context")
_SUMMARY_SECTIONS = [
    "1. Primary Request and Intent",
    "2. Key Technical Concepts",
    "3. Files and Code Sections",
    "4. Errors and fixes",
    "5. Problem Solving",
    "6. All user messages (enumerate, do not summarize)",
    "7. Pending Tasks",
    "8. Current Work (finest granularity: file + function name)",
    "9. Optional Next Step",
]


class Compactor:
    """3-layer compactor. Layers 1-2 are pure local (no LLM). Layer 3 uses LLM."""

    def __init__(
        self,
        llm: LLMClient,
        spill_dir: os.PathLike[str] | str,
        large_result_bytes: int = 4096,
        microcompact_keep: int = 5,
        auto_compact_tokens: int = 8000,
        circuit_breaker_limit: int = 3,
    ) -> None:
        self._llm = llm
        self._spill_dir = os.fspath(spill_dir)
        os.makedirs(self._spill_dir, exist_ok=True)
        self._large_bytes = large_result_bytes
        self._keep = microcompact_keep
        self._auto_tokens = auto_compact_tokens
        self._breaker_limit = circuit_breaker_limit
        self._failures: dict[str, int] = {}  # per-session failure count
        self._tripped: set[str] = set()  # per-session tripped sessions

    def is_tripped(self, session_id: str) -> bool:
        return session_id in self._tripped

    # ---- token estimation ----

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        # ponytail: 4 bytes ~ 1 token, no tokenizer dependency.
        # Upgrade to tiktoken if accuracy matters.
        total = 0
        for m in messages:
            total += len((m.content or "").encode("utf-8"))
            if m.tool_calls:
                total += len(json.dumps(m.tool_calls, ensure_ascii=False).encode("utf-8"))
        return total // 4

    # ---- Layer 1: large result spillover ----

    def spill_large_results(self, session: Session) -> None:
        """Tool results exceeding large_result_bytes get written to disk;
        message replaced with preview + spill path."""
        for msg in session.messages:
            if msg.role != "tool" or msg.content is None:
                continue
            raw = msg.content.encode("utf-8")
            if len(raw) <= self._large_bytes:
                continue
            digest = hashlib.sha256(raw).hexdigest()[:16]
            spill_path = os.path.join(self._spill_dir, f"{digest}.spill")
            with open(spill_path, "wb") as f:
                f.write(raw)
            preview = msg.content[:80]
            msg.content = (
                f"{preview}...\n[spill:{digest}] "
                f"({len(raw)}B) on disk, Read to reclaim."
            )
            log.info("spill_large_results: %d bytes -> %s", len(raw), spill_path)

    # ---- Layer 2: microcompact ----

    def microcompact(self, session: Session) -> None:
        """Drop old tool results, keep only recent N pairs.
        State info (todos/plan/workspace) is in session.memory -- never touched."""
        if self._keep < 0:
            return
        tool_indices = [
            i for i, m in enumerate(session.messages) if m.role == "tool"
        ]
        if len(tool_indices) <= self._keep:
            return
        # keep the last N tool messages (and their preceding assistant tool_calls)
        keep_ids = {
            session.messages[i].tool_call_id for i in tool_indices[-self._keep :]
        }
        new_msgs: list[Message] = []
        for msg in session.messages:
            if msg.role == "tool" and msg.tool_call_id not in keep_ids:
                continue  # drop old tool result
            if (
                msg.role == "assistant"
                and msg.tool_calls
                and not any(
                    tc.get("id") in keep_ids for tc in msg.tool_calls
                )
            ):
                continue  # drop assistant(tool_calls) whose results were dropped
            new_msgs.append(msg)
        session.messages = new_msgs

    # ---- Layer 3: Auto-Compact ----

    def _is_compaction_output(self, messages: list[Message]) -> bool:
        """Recursion guard: detect messages that already look like a compaction."""
        for msg in messages[:3]:
            text = (msg.content or "").lower()
            if any(marker.lower() in text for marker in _RECURSION_MARKERS):
                return True
        return False

    def _build_summary_prompt(self, session: Session) -> str:
        """9-part fixed-section summary prompt."""
        conversation = "\n".join(
            f"[{m.role}] {m.content or ''}" for m in session.messages
        )
        user_msgs = [
            m.content for m in session.messages if m.role == "user" and m.content
        ]
        sections = "\n".join(f"### {s}" for s in _SUMMARY_SECTIONS)
        return (
            "本会话是从之前一次因上下文耗尽而中断的对话延续过来的。"
            "请总结以下对话历史。\n\n"
            f"对话历史:\n{conversation}\n\n"
            f"用户消息枚举:\n" + "\n".join(f"- {u}" for u in user_msgs) + "\n\n"
            f"请按以下结构输出总结:\n{sections}"
        )

    def _build_attachments(self, session: Session) -> str:
        """Extract state info (todos/plan/workspace) as verbatim attachments."""
        parts: list[str] = ["--- Attachments (state info, verbatim) ---"]
        if session.memory.todos:
            parts.append("Todos:")
            for t in session.memory.todos:
                parts.append(f"  - [#{t.id}] {t.title} [{t.status}]")
        if session.memory.plan:
            parts.append("Plan:")
            for s in session.memory.plan:
                parts.append(f"  - {s.prompt}")
        if session.memory.workspace:
            parts.append("Workspace:")
            parts.append("  " + json.dumps(session.memory.workspace, ensure_ascii=False, indent=2))
        return "\n".join(parts)

    async def auto_compact(self, session: Session) -> list[Message] | None:
        """Full summary when estimated tokens exceed threshold.
        Returns new compacted message list, or None if not triggered."""
        if session.id in self._tripped:
            return None
        if self._is_compaction_output(session.messages):
            return None  # recursion guard
        tokens = self._estimate_tokens(session.messages)
        if tokens < self._auto_tokens:
            return None
        # build summary prompt
        prompt = self._build_summary_prompt(session)
        try:
            summary = await self._llm.respond(
                [
                    {
                        "role": "system",
                        "content": "You are a conversation summarizer.",
                    }
                ],
                prompt,
            )
        except Exception as e:
            n = self._failures.get(session.id, 0) + 1
            self._failures[session.id] = n
            log.warning("auto_compact summary failed (%d/%d): %s", n, self._breaker_limit, e)
            if n >= self._breaker_limit:
                self._tripped.add(session.id)
                log.error("circuit breaker tripped for session %s after %d failures", session.id, n)
            return None
        self._failures.pop(session.id, None)
        # build 3-segment chain: boundaryMarker + summary + attachments
        last_id = str(session.messages[-1].tool_call_id or "") if session.messages else ""
        boundary = (
            f"{_BOUNDARY_PREFIX} session continuation from interrupted context. "
            f"Pre-compaction ~{tokens} tokens. Last message ref: {last_id or 'none'}."
        )
        attachments = self._build_attachments(session)
        compacted = [
            Message(role="user", content=boundary),
            Message(role="assistant", content=summary or "(summary unavailable)"),
            Message(role="user", content=attachments),
        ]
        log.info("auto_compact: %d tokens -> %d messages", tokens, len(compacted))
        return compacted

    # ---- combined entry point ----

    async def compact(self, session: Session) -> bool:
        """Run all 3 layers in order. Returns True if compaction occurred."""
        self.spill_large_results(session)
        self.microcompact(session)
        result = await self.auto_compact(session)
        if result is not None:
            session.messages = result
            return True
        return False
