from session.models import Memory, Message, Session, TodoItem


def test_todo_item_defaults():
    t = TodoItem(id="1", title="写大纲")
    assert t.status == "PLANNED"
    assert t.created_at  # 非空 ISO 时间戳


def test_memory_defaults():
    m = Memory()
    assert m.todos == []
    assert m.plan == []
    assert m.lessons == []
    assert m.workspace == {}


def test_session_defaults():
    s = Session(id="sid-1")
    assert s.fsm_state == "IDLE"
    assert s.step_count == 0
    assert s.memory.todos == []
    assert s.messages == []


def test_to_dict_roundtrip():
    s = Session(id="sid-1")
    s.memory.todos.append(TodoItem(id="1", title="A", status="IN_PROGRESS"))
    s.messages.append(Message(role="user", content="hi"))
    d = s.to_dict()
    s2 = Session.from_dict(d)
    assert s2.id == "sid-1"
    assert s2.memory.todos[0].title == "A"
    assert s2.memory.todos[0].status == "IN_PROGRESS"
    assert s2.messages[0].role == "user"
    assert s2.fsm_state == "IDLE"
