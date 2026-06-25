# tests/test_tool_agent_deadline.py
"""max_execution_time is a real post-hoc deadline: ToolAgent.invoke raises
AixonError when the completed run exceeded it (audit 3.3). Before the fix it
only logged a warning and returned the result, bounding nothing."""
from __future__ import annotations

import pytest

from aixon.exceptions import AixonError
from aixon.message import Message
from tests._fakes import make_llm


def _agent(max_execution_time):
    from aixon.agents.tool_agent import ToolAgent

    return type(
        "DeadlineAgent",
        (ToolAgent,),
        {"name": "deadline", "llm": make_llm(), "max_execution_time": max_execution_time},
    )()


def test_invoke_raises_when_deadline_exceeded(monkeypatch):
    # Force the clock to jump past the deadline between start and the post-check.
    state = {"n": 0}

    def fake_monotonic():
        state["n"] += 1
        return 0.0 if state["n"] == 1 else 1e9  # deadline computed at t=0, check at t=1e9

    monkeypatch.setattr("aixon.agents.tool_agent.time.monotonic", fake_monotonic)
    agent = _agent(max_execution_time=600)
    with pytest.raises(AixonError, match="max_execution_time"):
        agent.invoke([Message(role="user", content="hi")])


def test_invoke_returns_normally_within_deadline():
    # A run that finishes within the budget returns its answer (no raise).
    agent = _agent(max_execution_time=600)
    out = agent.invoke([Message(role="user", content="hi")])
    assert out.role == "assistant"
