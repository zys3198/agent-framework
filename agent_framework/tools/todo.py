from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from session.models import Session

_VALID_STATUS = frozenset({"PLANNED", "IN_PROGRESS", "DONE"})


class Todo:
    """Todo list CRUD. Writes session.memory.todos (cross-turn progress)."""

    name: ClassVar[str] = "todo"
    description: ClassVar[str] = (
        "Manage a todo list (create/list/update) for planning and "
        "cross-turn progress tracking."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "update"]},
            "title": {"type": "string", "description": "required for create"},
            "id": {"type": "string", "description": "required for update, e.g. 1"},
            "status": {
                "type": "string",
                "enum": ["PLANNED", "IN_PROGRESS", "DONE"],
            },
        },
        "required": ["action"],
    }

    async def run(self, args: dict[str, Any], session: Session) -> str:
        action = args.get("action")
        if action == "create":
            title = args.get("title")
            if not title:
                return "ERROR: create requires title"
            next_id = (
                max(
                    (int(t.id) for t in session.memory.todos if t.id.isdigit()),
                    default=0,
                )
                + 1
            )
            from session.models import TodoItem

            item = TodoItem(id=str(next_id), title=title, status="PLANNED")
            session.memory.todos.append(item)
            return f"created #{next_id}: {title}"

        if action == "list":
            if not session.memory.todos:
                return "(empty)"
            return "\n".join(
                f"[#{t.id}] {t.title} [{t.status}]" for t in session.memory.todos
            )

        if action == "update":
            tid = args.get("id")
            status = args.get("status")
            if status not in _VALID_STATUS:
                return f"ERROR: status must be one of {sorted(_VALID_STATUS)}"
            for t in session.memory.todos:
                if t.id == tid:
                    t.status = status
                    return f"updated #{tid} -> {status}"
            return f"ERROR: id not found: {tid}"

        return f"ERROR: unknown action: {action}"
