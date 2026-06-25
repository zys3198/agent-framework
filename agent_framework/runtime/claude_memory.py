from __future__ import annotations

from pathlib import Path


def load_claude_context(workspace_root: Path, user_home: Path | None = None) -> str:
    sections: list[str] = []

    resolved_root = workspace_root.resolve()
    for path in [*reversed(resolved_root.parents), resolved_root]:
        file = path / "CLAUDE.md"
        text = _read_text(file)
        if text:
            sections.append(_section(f"Project CLAUDE: {file}", text))

    local_file = resolved_root / "CLAUDE.local.md"
    local_text = _read_text(local_file)
    if local_text:
        sections.append(_section(f"Local CLAUDE: {local_file}", local_text))

    home = (user_home or Path.home()).resolve()
    user_file = home / ".claude" / "CLAUDE.md"
    user_text = _read_text(user_file)
    if user_text:
        sections.append(_section(f"User CLAUDE: {user_file}", user_text))

    return "\n\n".join(sections)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _section(header: str, content: str) -> str:
    return f"{header}\n{content}"
