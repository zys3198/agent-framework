from __future__ import annotations

from runtime.agent_memory import load_project_context


def test_load_project_context_no_local_files(tmp_path):
    """No AGENTS.md under workspace -> no Project section. External files
    on the parent chain (e.g. a parent dir) are out of scope. """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    text = load_project_context(workspace, user_home=tmp_path / "home")
    assert str(workspace.resolve()) not in text  # no Project section for our workspace


def test_load_project_context_walks_upward_root_first(tmp_path):
    # nested under tmp_path so traversal only finds our test files
    top = tmp_path / "repo"
    nested = top / "apps" / "demo"
    nested.mkdir(parents=True)
    (top / "AGENTS.md").write_text("top rules", encoding="utf-8")
    ((top / "apps") / "AGENTS.md").write_text("apps rules", encoding="utf-8")
    (nested / "AGENTS.md").write_text("demo rules", encoding="utf-8")

    text = load_project_context(nested, user_home=tmp_path / "home")

    assert text.index("top rules") < text.index("apps rules") < text.index("demo rules")


def test_load_project_context_includes_local_and_user_with_headers(tmp_path):
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    workspace.mkdir()
    (home / ".agents").mkdir(parents=True)
    (workspace / "AGENTS.local.md").write_text("local rules", encoding="utf-8")
    ((home / ".agents") / "AGENTS.md").write_text("user rules", encoding="utf-8")

    text = load_project_context(workspace, user_home=home)

    # External AGENTS.md on parent chain may appear; we only check
    # that our Local + User sections are present and correct.
    assert "Local AGENTS" in text
    assert "Local AGENTS" in text
    assert "User AGENTS" in text
    assert "local rules" in text
    assert "user rules" in text
