from typing import get_args, get_type_hints

from aixon.state import GraphState, add_messages_neutral, END
from aixon.message import Message


def test_reducer_appends_list_to_existing():
    existing = [Message(role="user", content="a")]
    out = add_messages_neutral(existing, [Message(role="assistant", content="b")])
    assert [m.content for m in out] == ["a", "b"]


def test_reducer_accepts_single_message():
    out = add_messages_neutral([], Message(role="assistant", content="solo"))
    assert [m.content for m in out] == ["solo"]


def test_reducer_treats_none_left_as_empty():
    out = add_messages_neutral(None, [Message(role="user", content="x")])
    assert [m.content for m in out] == ["x"]


def test_reducer_does_not_mutate_existing():
    existing = [Message(role="user", content="a")]
    add_messages_neutral(existing, [Message(role="assistant", content="b")])
    assert [m.content for m in existing] == ["a"]  # unchanged


def test_reducer_none_right_is_noop():
    existing = [Message(role="user", content="a")]
    out = add_messages_neutral(existing, None)
    assert [m.content for m in out] == ["a"]


def test_graphstate_messages_field_uses_reducer():
    hints = get_type_hints(GraphState, include_extras=True)
    annotated = hints["messages"]  # Annotated[list[Message], add_messages_neutral]
    args = get_args(annotated)
    assert add_messages_neutral in args


def test_graphstate_is_total_false():
    state: GraphState = {"messages": [Message(role="user", content="hi")]}
    assert state["messages"][0].content == "hi"


def test_end_is_reexported_from_langgraph():
    from langgraph.graph import END as LG_END
    assert END is LG_END
