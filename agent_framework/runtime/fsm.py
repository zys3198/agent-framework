from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    IDLE = "IDLE"
    ROUTING = "ROUTING"
    RESPONDING = "RESPONDING"
    EXECUTING = "EXECUTING"
    PLANNING = "PLANNING"
    REFLECTING = "REFLECTING"
    REPLANNING = "REPLANNING"


class InvalidTransition(Exception):
    pass


# 合法转移表 (from, to)
_TRANSITIONS: set[tuple[State, State]] = {
    (State.IDLE, State.ROUTING),
    (State.ROUTING, State.RESPONDING),  # route=direct
    (State.ROUTING, State.EXECUTING),  # route=simple
    (State.ROUTING, State.PLANNING),  # route=plan
    (State.PLANNING, State.EXECUTING),  # plan_ready
    (State.EXECUTING, State.REFLECTING),  # tool_error
    (State.REFLECTING, State.EXECUTING),  # retry
    (State.EXECUTING, State.REPLANNING),  # replan_needed
    (State.REFLECTING, State.REPLANNING),  # retry_exhausted
    (State.REPLANNING, State.EXECUTING),  # plan_updated
    (State.REPLANNING, State.RESPONDING),  # abort
    (State.EXECUTING, State.RESPONDING),  # done
    (State.RESPONDING, State.IDLE),  # replied
}


class SessionFSM:
    """session 级状态机. 只校验合法性, 不持业务."""

    def __init__(self, state: State = State.IDLE) -> None:
        self.state = state

    def transition(self, to: State) -> None:
        if (self.state, to) not in _TRANSITIONS:
            raise InvalidTransition(f"{self.state.value} -> {to.value}")
        self.state = to

    def can(self, to: State) -> bool:
        return (self.state, to) in _TRANSITIONS
