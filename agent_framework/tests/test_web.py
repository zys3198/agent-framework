"""Web layer route tests (TestClient, no real API key needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from main import create_app
from session.models import Message
from session.store import Store


class FakeAgent:
    """Records the last call; returns an echo string."""

    def __init__(self) -> None:
        self.last: tuple[str, str] | None = None

    async def chat(self, session_id: str, user_input: str) -> str:
        self.last = (session_id, user_input)
        return f"echo:{user_input}"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    store = Store(tmp_path)
    app = create_app(FakeAgent(), store, tmp_path)
    return TestClient(app)


def test_index_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "<html" in body.lower()
    assert "<title>" in body.lower()


def test_chat_returns_answer(client: TestClient) -> None:
    resp = client.post("/chat", json={"session_id": "s1", "input": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "echo:hi"
    assert data["session_id"] == "s1"


def test_chat_generates_session_id_when_omitted(client: TestClient) -> None:
    resp = client.post("/chat", json={"input": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "echo:hi"
    assert data["session_id"]
    assert len(data["session_id"]) > 0


def test_sessions_list_after_save(client: TestClient, tmp_path: Path) -> None:
    store = Store(tmp_path)
    s = store.load("list-sid")
    s.messages.append(Message(role="user", content="hello"))
    store.save(s)

    resp = client.get("/sessions")
    assert resp.status_code == 200
    rows = resp.json()
    ids = [r["id"] for r in rows]
    assert "list-sid" in ids


def test_session_detail(client: TestClient, tmp_path: Path) -> None:
    store = Store(tmp_path)
    s = store.load("detail-sid")
    s.messages.append(Message(role="assistant", content="pong"))
    store.save(s)

    resp = client.get("/sessions/detail-sid")
    assert resp.status_code == 200
    d = resp.json()
    assert d["id"] == "detail-sid"
    assert isinstance(d["messages"], list)
    assert d["messages"][0]["content"] == "pong"


def test_trace_empty_when_missing(client: TestClient) -> None:
    resp = client.get("/trace/nope")
    assert resp.status_code == 200
    assert resp.json() == []


def test_trace_returns_lines(client: TestClient, tmp_path: Path) -> None:
    sid = "trace-sid"
    path = tmp_path / f"{sid}.jsonl"
    obj = {"event": "router", "route": "CHAT"}
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    resp = client.get(f"/trace/{sid}")
    assert resp.status_code == 200
    lines = resp.json()
    assert isinstance(lines, list)
    assert len(lines) == 1
    assert lines[0]["event"] == "router"


def test_chat_503_when_no_agent(tmp_path: Path) -> None:
    store = Store(tmp_path)
    app = create_app(None, store, tmp_path)
    c = TestClient(app)
    resp = c.post("/chat", json={"session_id": "x", "input": "hi"})
    assert resp.status_code == 503


def test_trace_path_traversal_safe(client: TestClient, tmp_path: Path) -> None:
    # the guard lives in _trace_path: any "../" is reduced to its basename,
    # so a traversal id resolves inside trace_dir (empty file -> []).
    from main import _trace_path

    p = _trace_path(tmp_path, "../../etc/passwd")
    # must be inside tmp_path, not above it
    assert tmp_path in p.parents or p == tmp_path
    assert p.name == "passwd.jsonl"


def test_trace_dotdot_in_query_returns_empty(client: TestClient) -> None:
    # route still resolves; guard reduces to basename which has no file
    resp = client.get("/trace/foo")
    assert resp.status_code == 200
    assert resp.json() == []


def test_session_missing_returns_default(client: TestClient) -> None:
    # store.load on missing id returns a fresh Session; to_dict is returned
    resp = client.get("/sessions/never-saved")
    assert resp.status_code == 200
    d = resp.json()
    assert d["id"] == "never-saved"
    assert d["messages"] == []


def test_delete_removes_session_and_trace(client: TestClient, tmp_path: Path) -> None:
    sid = "del-sid"
    store = Store(tmp_path)
    store.save(store.load(sid))
    (tmp_path / f"{sid}.jsonl").write_text('{"event": "x"}\n', encoding="utf-8")

    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "session": True, "trace": True}
    assert not (tmp_path / f"{sid}.json").exists()
    assert not (tmp_path / f"{sid}.jsonl").exists()


def test_delete_404_when_both_missing(client: TestClient) -> None:
    resp = client.delete("/sessions/never-existed")
    assert resp.status_code == 404


def test_delete_only_trace_still_200(client: TestClient, tmp_path: Path) -> None:
    sid = "trace-only"
    (tmp_path / f"{sid}.jsonl").write_text('{"event": "x"}\n', encoding="utf-8")
    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "session": False, "trace": True}


def test_delete_path_traversal_safe(client: TestClient, tmp_path: Path) -> None:
    # traversal id reduced to basename inside root; nothing outside deleted.
    resp = client.delete("/sessions/..%2Fetc%2Fpasswd")
    assert resp.status_code == 404
    assert not (tmp_path / "passwd.json").exists()
    assert not (tmp_path.parent / "passwd.json").exists()


def test_delete_trace_unlink_failure_degrades(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PermissionError on the held-open trace handle (Windows) must not 500:
    # session is deleted, trace is left + reported as not-deleted.
    sid = "locked-trace"
    store = Store(tmp_path)
    store.save(store.load(sid))
    (tmp_path / f"{sid}.jsonl").write_text('{"event": "x"}\n', encoding="utf-8")

    real_unlink = Path.unlink

    def boom_on_jsonl(self: Path, *args: object, **kwargs: object) -> object:
        if self.name.endswith(".jsonl"):
            raise PermissionError(13, "file locked", str(self))
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", boom_on_jsonl)

    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "session": True, "trace": False}
    assert not (tmp_path / f"{sid}.json").exists()
    assert (tmp_path / f"{sid}.jsonl").exists()


def test_build_agent_registers_memory_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeLLM:
        pass

    monkeypatch.setattr(main.LLMClient, "from_env", lambda **_: FakeLLM())
    monkeypatch.setattr(main.config, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(main.config, "TRACE_DIR", tmp_path / "traces")

    agent = main.build_agent()
    names = {
        schema["function"]["name"] for schema in agent._executor._registry.schemas()
    }

    assert "write_memory" in names
    assert "read_memory_body" in names

# -- Phase 3: global exception handler mapping --


def _get_exception_handler(app):
    for key, value in app.exception_handlers.items():
        if key is Exception:
            return value
    return None


def test_openai_timeout_maps_to_503(tmp_path):
    from fastapi import Request
    from openai import APITimeoutError

    from main import create_app
    from session.store import Store

    store = Store(tmp_path)
    app = create_app(None, store, tmp_path)
    handler = _get_exception_handler(app)
    assert handler is not None

    scope = {"type": "http", "method": "POST", "path": "/chat", "headers": []}
    request = Request(scope)
    exc = APITimeoutError("LLM timeout")

    import inspect
    if inspect.iscoroutinefunction(handler):
        import asyncio
        response = asyncio.run(handler(request, exc))
    else:
        response = handler(request, exc)

    assert response.status_code == 503
    import json
    body = json.loads(response.body)
    assert body["detail"] == "LLM request timed out"


def test_openai_error_maps_to_502(tmp_path):
    from fastapi import Request
    from openai import OpenAIError

    from main import create_app
    from session.store import Store

    store = Store(tmp_path)
    app = create_app(None, store, tmp_path)
    handler = _get_exception_handler(app)
    assert handler is not None

    scope = {"type": "http", "method": "POST", "path": "/chat", "headers": []}
    request = Request(scope)
    exc = OpenAIError("API error")

    import inspect
    if inspect.iscoroutinefunction(handler):
        import asyncio
        response = asyncio.run(handler(request, exc))
    else:
        response = handler(request, exc)

    assert response.status_code == 502
    import json
    body = json.loads(response.body)
    assert body["detail"] == "LLM API error"


def test_generic_exception_returns_500(tmp_path):
    from fastapi import Request

    from main import create_app
    from session.store import Store

    store = Store(tmp_path)
    app = create_app(None, store, tmp_path)
    handler = _get_exception_handler(app)
    assert handler is not None

    scope = {"type": "http", "method": "POST", "path": "/chat", "headers": []}
    request = Request(scope)
    exc = ValueError("something broke")

    import inspect
    if inspect.iscoroutinefunction(handler):
        import asyncio
        response = asyncio.run(handler(request, exc))
    else:
        response = handler(request, exc)

    assert response.status_code == 500
    import json
    body = json.loads(response.body)
    assert body["detail"] == "internal error"
