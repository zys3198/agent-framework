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
