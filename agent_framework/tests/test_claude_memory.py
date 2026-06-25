from __future__ import annotations

from runtime.claude_memory import load_claude_context


def test_load_claude_context_returns_empty_when_files_missing(tmp_path):
    assert load_claude_context(tmp_path, user_home=tmp_path / "home") == ""


def test_load_claude_context_walks_upward_root_first(tmp_path):
    top = tmp_path / "repo"
    nested = top / "apps" / "demo"
    nested.mkdir(parents=True)
    (top / "CLAUDE.md").write_text("top rules", encoding="utf-8")
    ((top / "apps") / "CLAUDE.md").write_text("apps rules", encoding="utf-8")
    (nested / "CLAUDE.md").write_text("demo rules", encoding="utf-8")

    text = load_claude_context(nested, user_home=tmp_path / "home")

    assert text.index("top rules") < text.index("apps rules") < text.index("demo rules")


def test_load_claude_context_includes_local_and_user_with_headers(tmp_path):
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    workspace.mkdir()
    (home / ".claude").mkdir(parents=True)
    (workspace / "CLAUDE.local.md").write_text("local rules", encoding="utf-8")
    ((home / ".claude") / "CLAUDE.md").write_text("user rules", encoding="utf-8")

    text = load_claude_context(workspace, user_home=home)

    assert "Project CLAUDE" not in text
    assert "Local CLAUDE" in text
    assert "User CLAUDE" in text
    assert "local rules" in text
    assert "user rules" in text
