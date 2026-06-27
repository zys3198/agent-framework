from __future__ import annotations

import json

from session.models import RECENT_LESSONS_LIMIT, Memory, MemoryEntry

_MEMORY_INDEX_MAX_LINES = 200
_MEMORY_INDEX_MAX_BYTES = 25 * 1024


def build_system_prompt(memory: Memory) -> str:
    """Project stable runtime state into the system prompt."""
    lines: list[str] = ["You are a helpful agent."]
    if memory.todos:
        lines.append("Todos:")
        lines.extend(f"- [#{t.id}] {t.title} [{t.status}]" for t in memory.todos)
    if memory.plan:
        lines.append("Plan: " + " | ".join(s.prompt for s in memory.plan))
    if memory.lessons:
        lines.append("Lessons learned:")
        lines.extend(f"- {lesson}" for lesson in memory.lessons[-RECENT_LESSONS_LIMIT:])
    return "\n".join(lines)


def build_memory_context_message(
    memory: Memory, project_context: str = ""
) -> dict[str, str] | None:
    entries = _memory_index_lines(memory.entries)
    parts = [part for part in [project_context.strip(), "\n".join(entries)] if part]
    if not parts:
        return None
    return {"role": "user", "content": "\n\n".join(parts)}


def build_memory_attachments(memory: Memory) -> str:
    parts: list[str] = ["--- Attachments (state info, verbatim) ---"]
    if memory.todos:
        parts.append("Todos:")
        for t in memory.todos:
            parts.append(f"  - [#{t.id}] {t.title} [{t.status}]")
    if memory.plan:
        parts.append("Plan:")
        for s in memory.plan:
            parts.append(f"  - {s.prompt}")
    if memory.workspace:
        parts.append("Workspace:")
        parts.append("  " + json.dumps(memory.workspace, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def _memory_index_lines(entries: list[MemoryEntry]) -> list[str]:
    if not entries:
        return []

    lines = ["Memory index:"]
    total_bytes = len(lines[0].encode("utf-8"))
    for entry in entries:
        line = (
            f"- id={entry.id} type={entry.type} name={entry.name} "
            f"description={entry.description} "
            f"keywords={','.join(entry.keywords)} "
            f"saved_at={entry.saved_at}"
        )
        next_bytes = total_bytes + 1 + len(line.encode("utf-8"))
        if len(lines) >= _MEMORY_INDEX_MAX_LINES or next_bytes > _MEMORY_INDEX_MAX_BYTES:
            break
        lines.append(line)
        total_bytes = next_bytes
    return lines
