from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from session.models import Session

log = logging.getLogger(__name__)

T = TypeVar("T")


def _derive_title(d: dict[str, Any]) -> str:
    """Session display title: first user message (truncated), else '新会话'."""
    for m in d.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "user":
            text = (m.get("content") or "").strip()
            if text:
                return text[:30]
    return "新会话"


class Store:
    """JSON 持久化. 原子写 (tmp + os.replace), 损坏文件备份后重建."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()


    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def _path(self, session_id: str) -> Path:
        # 防路径穿越: 只取文件名, 丢弃任何目录部分
        safe = Path(session_id).name
        return self.root / f"{safe}.json"

    def with_session(self, session_id: str, mutator: Callable[[Session], T]) -> T:
        """Run a load->mutate->save transaction under the per-session lock."""
        with self._lock_for(session_id):
            p = self._path(session_id)
            session = self._load_unlocked(session_id, p)
            result = mutator(session)
            self._save_unlocked(session, p)
            return result

    def load(self, session_id: str) -> Session:
        with self._lock_for(session_id):
            return self._load_unlocked(session_id, self._path(session_id))

    def save(self, session: Session) -> None:
        with self._lock_for(session.id):
            self._save_unlocked(session, self._path(session.id))

    def _load_unlocked(self, session_id: str, p: Path) -> Session:
        if not p.exists():
            return Session(id=session_id)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            bak = p.with_suffix(p.suffix + ".corrupt.bak")
            try:
                os.replace(p, bak)
                log.warning("corrupt session file backed up: %s -> %s (%s)", p, bak, e)
            except OSError:
                pass
            return Session(id=session_id)

    def _save_unlocked(self, session: Session, p: Path) -> None:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, p)

    def delete(self, session_id: str) -> bool:
        """Remove a session file. Returns True if a file was deleted."""
        try:
            self._path(session_id).unlink()
            return True
        except FileNotFoundError:
            return False

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append(
                    {
                        "id": d["id"],
                        "title": _derive_title(d),
                        "todo_count": len(d.get("memory", {}).get("todos", [])),
                        "updated_at": d.get("updated_at", ""),
                        "fsm_state": d.get("fsm_state", "IDLE"),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return out
