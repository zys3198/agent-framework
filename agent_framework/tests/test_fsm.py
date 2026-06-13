import pytest

from runtime.fsm import InvalidTransition, SessionFSM, State


def test_new_session_idle():
    f = SessionFSM()
    assert f.state == State.IDLE


def test_legal_path_plan():
    f = SessionFSM()
    f.transition(State.ROUTING)
    f.transition(State.PLANNING)
    f.transition(State.EXECUTING)
    f.transition(State.REPLANNING)
    f.transition(State.EXECUTING)
    f.transition(State.RESPONDING)
    f.transition(State.IDLE)
    assert f.state == State.IDLE


def test_legal_reflect_then_replan():
    f = SessionFSM()
    f.transition(State.ROUTING)
    f.transition(State.EXECUTING)
    f.transition(State.REFLECTING)
    f.transition(State.REPLANNING)
    f.transition(State.EXECUTING)


def test_replanning_abort_to_responding():
    f = SessionFSM()
    f.transition(State.ROUTING)
    f.transition(State.EXECUTING)
    f.transition(State.REPLANNING)
    f.transition(State.RESPONDING)


def test_illegal_transition_raises():
    f = SessionFSM()
    with pytest.raises(InvalidTransition):
        f.transition(State.RESPONDING)


def test_illegal_replanning_from_idle():
    f = SessionFSM()
    with pytest.raises(InvalidTransition):
        f.transition(State.REPLANNING)
