from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from aixon._langchain import from_langchain, to_langchain
from aixon.message import Message


# ── to_langchain ─────────────────────────────────────────────────────────────

def test_to_langchain_system():
    lc = to_langchain([Message(role="system", content="You are helpful.")])
    assert len(lc) == 1
    assert isinstance(lc[0], SystemMessage)
    assert lc[0].content == "You are helpful."


def test_to_langchain_user():
    lc = to_langchain([Message(role="user", content="Hello")])
    assert isinstance(lc[0], HumanMessage)
    assert lc[0].content == "Hello"


def test_to_langchain_assistant():
    lc = to_langchain([Message(role="assistant", content="Hi there")])
    assert isinstance(lc[0], AIMessage)
    assert lc[0].content == "Hi there"


def test_to_langchain_tool():
    lc = to_langchain(
        [Message(role="tool", content="42", tool_call_id="call_1", name="calc")]
    )
    assert isinstance(lc[0], ToolMessage)
    assert lc[0].content == "42"
    assert lc[0].tool_call_id == "call_1"


def test_to_langchain_mixed():
    lc = to_langchain(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="user msg"),
            Message(role="assistant", content="reply"),
        ]
    )
    assert [type(m).__name__ for m in lc] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
    ]


def test_to_langchain_unknown_role_raises():
    msg = Message.__new__(Message)
    object.__setattr__(msg, "role", "badrole")
    object.__setattr__(msg, "content", "x")
    object.__setattr__(msg, "name", None)
    object.__setattr__(msg, "tool_calls", [])
    object.__setattr__(msg, "tool_call_id", None)
    object.__setattr__(msg, "reasoning", None)
    with pytest.raises(ValueError, match="badrole"):
        to_langchain([msg])


# ── from_langchain ────────────────────────────────────────────────────────────

def test_from_langchain_ai_message():
    m = from_langchain(AIMessage(content="Hello back"))
    assert m.role == "assistant"
    assert m.content == "Hello back"
    assert m.tool_calls == []
    assert m.reasoning is None


def test_from_langchain_carries_tool_calls():
    lc = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "calc", "args": {"x": 1}}],
    )
    m = from_langchain(lc)
    assert len(m.tool_calls) == 1
    tc = m.tool_calls[0]
    assert tc["name"] == "calc"
    assert tc["args"] == {"x": 1}
    assert tc["id"] == "call_1"


def test_from_langchain_carries_reasoning_from_additional_kwargs():
    lc = AIMessage(
        content="answer",
        additional_kwargs={"reasoning_content": "I thought about it."},
    )
    m = from_langchain(lc)
    assert m.reasoning == "I thought about it."


def test_from_langchain_human_message():
    m = from_langchain(HumanMessage(content="Hi"))
    assert m.role == "user"
    assert m.content == "Hi"
