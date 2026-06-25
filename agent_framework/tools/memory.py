from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from session.models import MEMORY_ENTRY_TYPES, MemoryEntry

if TYPE_CHECKING:
    from session.models import Session


_STRUCTURE_MARKERS = ("Rule", "Why", "How to apply")
_RELATIVE_TIME_MARKERS = (
    "today",
    "yesterday",
    "tomorrow",
    "last week",
    "next week",
    "last month",
    "next month",
    "刚才",
    "刚刚",
    "今天",
    "昨天",
    "明天",
    "上周",
    "下周",
    "上个月",
    "下个月",
    "本周",
    "本月",
    "this week",
    "this month",
    "last night",
    "tonight",
    "this morning",
    "this afternoon",
    "this evening",
    "明年",
    "去年",
    "今年",
    "昨晚",
    "今晚",
    "今早",
    "今晨",
    "明早",
)
_RELATIVE_TIME_PATTERNS = (
    re.compile(r"\b\d+\s+(?:minute|hour|day|week|month|year)s?\s+ago\b", re.I),
    re.compile(r"\bin\s+\d+\s+(?:minute|hour|day|week|month|year)s?\b", re.I),
    re.compile(r"\b(?:next|last)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
)

_CODE_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"^\s*\w+\s*=\s*.+$", re.MULTILINE),
    re.compile(r"\b(?:for|while)\s+.+:\s*(?:\S.*)?$", re.MULTILINE),
    re.compile(r"\bif\s+.+:\s*(?:\S.*)?$", re.MULTILINE),
    re.compile(r"\breturn\s+.+", re.MULTILINE),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+\s*[:(]"),
    re.compile(r"\bfrom\s+\S+\s+import\s+\S+"),
    re.compile(r"\bimport\s+\S+"),
    re.compile(r"=>"),
    re.compile(r"[\w./\\-]+\.(?:py|js|ts|tsx|java|kt|go|rs|sh|bat|md|txt)\b"),
    re.compile(r"(?:[A-Za-z]:\\|/)[^\s]+"),
    re.compile(r"\bgit\s+(?:add|commit|push|pull|fetch|merge|rebase|reset|checkout|status|log|diff|restore|branch|clone)\b"),
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _error(message: str) -> str:
    return f"ERROR: {message}"


def _has_required_structure(content: str) -> bool:
    return all(marker in content for marker in _STRUCTURE_MARKERS)


def _has_relative_time(content: str) -> bool:
    lowered = content.lower()
    return any(marker in content or marker in lowered for marker in _RELATIVE_TIME_MARKERS) or any(
        pattern.search(content) for pattern in _RELATIVE_TIME_PATTERNS
    )


def _looks_like_code_or_path(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CODE_PATTERNS)


def _find_existing_entry(
    entries: list[MemoryEntry], type_: str, name: str
) -> MemoryEntry | None:
    target = name.casefold()
    for entry in entries:
        if entry.type == type_ and entry.name.casefold() == target:
            return entry
    return None


def _next_id(entries: list[MemoryEntry]) -> str:
    numeric_ids = [int(entry.id) for entry in entries if entry.id.isdigit()]
    return str(max(numeric_ids, default=0) + 1)


class WriteMemory:
    name: ClassVar[str] = "write_memory"
    description: ClassVar[str] = "Write one memory entry into session memory."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": list(MEMORY_ENTRY_TYPES),
            },
            "name": {"type": "string"},
            "description": {"type": "string"},
            "content": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["type", "name", "description", "content", "keywords"],
    }

    async def run(self, args: dict[str, Any], session: Session) -> str:
        type_ = args.get("type")
        name = args.get("name")
        description = args.get("description")
        content = args.get("content")
        keywords = args.get("keywords")

        if type_ not in MEMORY_ENTRY_TYPES:
            return _error(f"type must be one of {list(MEMORY_ENTRY_TYPES)}")
        if not isinstance(name, str) or not name.strip():
            return _error("name must be a non-empty string")
        if not isinstance(description, str):
            return _error("description must be a string")
        if not isinstance(content, str) or not content.strip():
            return _error("content must be a non-empty string")
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            return _error("keywords must be a list of strings")

        if type_ in {"feedback", "project"} and not _has_required_structure(content):
            return _error("feedback/project content must include Rule, Why, How to apply")

        if _looks_like_code_or_path(content) or _looks_like_code_or_path(description):
            return _error("content looks like code, file path, or git command")

        if type_ == "project" and _has_relative_time(content):
            return _error("project content must use absolute dates, not relative time")

        existing = _find_existing_entry(session.memory.entries, type_, name)
        if existing is None:
            entry = MemoryEntry(
                id=_next_id(session.memory.entries),
                type=type_,
                name=name,
                description=description,
                keywords=list(keywords),
                content=content,
                saved_at=_now(),
            )
            session.memory.entries.append(entry)
            return entry.id

        existing.name = name
        existing.description = description
        existing.keywords = list(keywords)
        existing.content = content
        existing.saved_at = _now()
        return existing.id


class ReadMemoryBody:
    name: ClassVar[str] = "read_memory_body"
    description: ClassVar[str] = "Read the content of a memory entry by id."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }

    async def run(self, args: dict[str, Any], session: Session) -> str:
        entry_id = args.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            return _error("id must be a non-empty string")

        for entry in session.memory.entries:
            if entry.id == entry_id:
                return entry.content
        return _error(f"memory entry not found: {entry_id}")
