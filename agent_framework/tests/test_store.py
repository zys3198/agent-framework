import json
import threading

from session.models import Message, TodoItem
from session.store import Store


def test_load_missing_creates_new(tmp_path):
    store = Store(tmp_path)
    s = store.load("new-sid")
    assert s.id == "new-sid"
    assert s.fsm_state == "IDLE"
    assert s.memory.todos == []


def test_save_then_load_roundtrip(tmp_path):
    store = Store(tmp_path)
    s = store.load("sid-1")
    s.memory.todos.append(TodoItem(id="1", title="A", status="IN_PROGRESS"))
    s.messages.append(Message(role="user", content="hi"))
    s.fsm_state = "EXECUTING"
    store.save(s)

    s2 = store.load("sid-1")
    assert s2.memory.todos[0].title == "A"
    assert s2.messages[0].content == "hi"
    assert s2.fsm_state == "EXECUTING"


def test_corrupt_json_recovers(tmp_path):
    store = Store(tmp_path)
    f = tmp_path / "sid-x.json"
    f.write_text("{ broken json", encoding="utf-8")
    s = store.load("sid-x")
    assert s.id == "sid-x"
    assert s.memory.todos == []
    # 损坏文件应被备份
    backups = list(tmp_path.glob("*.corrupt.bak"))
    assert len(backups) == 1


def test_delete_removes_session(tmp_path):
    store = Store(tmp_path)
    s = store.load("sid-del")
    s.messages.append(Message(role="user", content="hi"))
    store.save(s)
    assert (tmp_path / "sid-del.json").exists()
    assert store.delete("sid-del") is True
    assert not (tmp_path / "sid-del.json").exists()


def test_delete_missing_returns_false(tmp_path):
    store = Store(tmp_path)
    assert store.delete("nope") is False


def test_list_returns_summaries(tmp_path):
    store = Store(tmp_path)
    s = store.load("sid-1")
    s.memory.todos.append(TodoItem(id="1", title="A"))
    store.save(s)
    store.load("sid-2")
    store.save(store.load("sid-2"))
    s3 = store.load("sid-3")
    s3.messages.append(Message(role="user", content="帮我算 123 乘以 456"))
    store.save(s3)

    items = store.list()
    ids = {it["id"] for it in items}
    assert ids == {"sid-1", "sid-2", "sid-3"}
    s1 = next(i for i in items if i["id"] == "sid-1")
    assert s1["todo_count"] == 1
    assert s1["title"] == "新会话"
    s3row = next(i for i in items if i["id"] == "sid-3")
    assert s3row["title"] == "帮我算 123 乘以 456"

# -- Phase 3: concurrent save under per-session lock --


def test_concurrent_save_does_not_corrupt(tmp_path):
    """Concurrent save calls must not corrupt the JSON file on disk.

    Store's per-session lock guarantees atomic single-operation writes
    (no partial JSON). Lost updates across load->modify->save are expected
    at this layer -- the Agent.chat asyncio.Lock serializes the full
    transaction (see Agent._session_lock).
    """
    store = Store(tmp_path)
    s = store.load("conc")
    s.messages.append(Message(role="user", content="init"))
    store.save(s)

    def writer(i):
        s2 = store.load("conc")
        s2.messages.append(Message(role="user", content=f"msg{i:d}"))
        store.save(s2)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # File must always be valid, parseable JSON (no partial writes).
    raw = (tmp_path / "conc.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "messages" in data


def test_with_session_saves_mutation(tmp_path):
    store = Store(tmp_path)

    def mutate(session):
        session.messages.append(Message(role="user", content="hello"))
        return "ok"

    result = store.with_session("s1", mutate)

    assert result == "ok"
    loaded = store.load("s1")
    assert [(m.role, m.content) for m in loaded.messages] == [("user", "hello")]


def test_with_session_saves_when_mutator_returns_none(tmp_path):
    store = Store(tmp_path)

    def mutate(session):
        session.memory.lessons.append("lesson")

    result = store.with_session("s1", mutate)

    assert result is None
    assert store.load("s1").memory.lessons == ["lesson"]


def test_with_session_does_not_save_when_mutator_raises(tmp_path):
    store = Store(tmp_path)
    original = store.load("s1")
    original.messages.append(Message(role="user", content="before"))
    store.save(original)

    def mutate(session):
        session.messages.append(Message(role="user", content="after"))
        raise RuntimeError("boom")

    try:
        store.with_session("s1", mutate)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    loaded = store.load("s1")
    assert [(m.role, m.content) for m in loaded.messages] == [("user", "before")]
