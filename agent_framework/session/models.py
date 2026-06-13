from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TodoItem:
    id: str
    title: str
    status: str = "PLANNED"  # PLANNED | IN_PROGRESS | DONE
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TodoItem:
        # explicit fields, not cls(**d): tolerate extra keys; a field added
        # later gets its default instead of KeyError on old session files
        return cls(
            id=d["id"],
            title=d["title"],
            status=d.get("status", "PLANNED"),
            created_at=d.get("created_at", _now()),
        )


@dataclass
class Memory:
    todos: list[TodoItem] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    workspace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "todos": [t.to_dict() for t in self.todos],
            "plan": list(self.plan),
            "lessons": list(self.lessons),
            "workspace": dict(self.workspace),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Memory:
        return cls(
            todos=[TodoItem.from_dict(t) for t in d.get("todos", [])],
            plan=list(d.get("plan", [])),
            lessons=list(d.get("lessons", [])),
            workspace=dict(d.get("workspace", {})),
        )


@dataclass
class Message:
    role: str  # user | assistant | tool
    content: str
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        # explicit fields, not cls(**d): tolerate extra keys, future fields default
        return cls(
            role=d["role"],
            content=d["content"],
            tool_call_id=d.get("tool_call_id"),
        )


@dataclass
class Session:
    id: str
    messages: list[Message] = field(default_factory=list)
    memory: Memory = field(default_factory=Memory)
    fsm_state: str = "IDLE"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    step_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "memory": self.memory.to_dict(),
            "fsm_state": self.fsm_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step_count": self.step_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Session:
        return cls(
            id=d["id"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            memory=Memory.from_dict(d.get("memory", {})),
            fsm_state=d.get("fsm_state", "IDLE"),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            step_count=d.get("step_count", 0),
        )
