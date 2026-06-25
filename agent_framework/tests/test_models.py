from session.models import Memory, Message, Session, Step, TodoItem


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
