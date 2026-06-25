from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(UTC).isoformat()


class TraceLogger:
    """Per-session jsonl trace. One line per step. S3/S4 event types reserved."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Held open across log_* calls until close(); intentional for append perf.
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def _emit(self, rec: dict[str, Any]) -> None:
        # Any: trace fields are heterogeneous (str/int/list/bool), no shared schema.
        rec.setdefault("ts", _ts())
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    # Baseline events.
    def log_step(self, step: int) -> None:
        self._emit({"type": "step", "step": step})

    def log_llm_call(self, step: int, tools_offered: list[str]) -> None:
        self._emit({"type": "llm_call", "step": step, "tools_offered": tools_offered})

    def log_route(self, value: str) -> None:
        self._emit({"type": "route", "value": value})

    def log_tool_call(self, step: int, name: str, args: dict[str, Any]) -> None:
        # Any: tool args schema varies per tool, cannot enumerate statically.
        self._emit({"type": "tool_call", "step": step, "name": name, "args": args})

    def log_tool_result(self, step: int, result: str) -> None:
        self._emit({"type": "tool_result", "step": step, "result": result})

    def log_reflexion(self, step: int, lesson: str) -> None:
        self._emit({"type": "reflexion", "step": step, "lesson": lesson})

    def log_truncated(self) -> None:
        self._emit({"type": "truncated"})

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
