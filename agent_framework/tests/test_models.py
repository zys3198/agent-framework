from session.models import Memory, MemoryEntry, Message, Session, Step, TodoItem


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


def test_step_roundtrip():
    s = Step(prompt="do A")
    d = s.to_dict()
    assert d == {"prompt": "do A"}
    s2 = Step.from_dict(d)
    assert s2.prompt == "do A"


def test_step_defaults():
    s = Step(prompt="x")
    assert s.prompt == "x"


def test_step_from_dict_tolerates_str():
    # old session files stored plan items as raw strings
    s = Step.from_dict("legacy step text")
    assert s.prompt == "legacy step text"


def test_step_from_dict_tolerates_extra_keys():
    s = Step.from_dict({"prompt": "p", "future_field": "ignore me"})
    assert s.prompt == "p"


def test_memory_plan_step_roundtrip():
    m = Memory()
    m.plan.append(Step(prompt="step one"))
    m.plan.append(Step(prompt="step two"))
    d = m.to_dict()
    assert d["plan"][0] == {"prompt": "step one"}
    m2 = Memory.from_dict(d)
    assert len(m2.plan) == 2
    assert m2.plan[0].prompt == "step one"


def test_memory_entry_roundtrip():
    entry = MemoryEntry(
        id="mem-1",
        type="user",
        name="Alice",
        description="remember preference",
        keywords=["pref", "user"],
        content="likes concise replies",
        saved_at="2026-06-25T10:00:00+08:00",
    )
    assert entry.to_dict() == {
        "id": "mem-1",
        "type": "user",
        "name": "Alice",
        "description": "remember preference",
        "keywords": ["pref", "user"],
        "content": "likes concise replies",
        "saved_at": "2026-06-25T10:00:00+08:00",
    }
    entry2 = MemoryEntry.from_dict(entry.to_dict())
    assert entry2 == entry


def test_memory_entries_roundtrip():
    m = Memory(entries=[MemoryEntry(id="mem-1", type="project", name="Repo", description="project note", keywords=["repo"], content="session framework", saved_at="2026-06-25T10:00:00+08:00")])
    d = m.to_dict()
    assert d["entries"][0]["id"] == "mem-1"
    m2 = Memory.from_dict(d)
    assert len(m2.entries) == 1
    assert m2.entries[0].type == "project"


def test_memory_from_dict_missing_entries_defaults_to_empty_list():
    m = Memory.from_dict({"todos": [], "plan": [], "lessons": [], "workspace": {}, "future_key": "ignore"})
    assert m.entries == []
