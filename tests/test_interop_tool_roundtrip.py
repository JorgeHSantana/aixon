# tests/test_interop_tool_roundtrip.py
"""A tool Message must survive a Message -> LangChain -> Message round-trip with
its tool_call_id/name intact (audit 3.7). Without the fix, from_langchain drops
them and the re-converted ToolMessage has an empty tool_call_id."""
from __future__ import annotations

from langchain_core.messages import ToolMessage

from aixon._interop.messages import from_langchain, to_langchain
from aixon.message import Message


def test_tool_message_roundtrip_preserves_id_and_name():
    original = Message(
        role="tool",
        content="result text",
        tool_call_id="call_42",
        name="get_weather",
    )

    lc = to_langchain([original])[0]
    assert isinstance(lc, ToolMessage)
    assert lc.tool_call_id == "call_42"

    back = from_langchain(lc)
    assert back.role == "tool"
    assert back.tool_call_id == "call_42"   # was lost before the fix
    assert back.name == "get_weather"
    assert back.content == "result text"

    # And the round-tripped Message rebuilds a valid (non-empty id) ToolMessage.
    lc2 = to_langchain([back])[0]
    assert lc2.tool_call_id == "call_42"


def test_from_langchain_tool_message_directly():
    lc = ToolMessage(content="ok", tool_call_id="abc", name="lookup")
    msg = from_langchain(lc)
    assert (msg.tool_call_id, msg.name, msg.role) == ("abc", "lookup", "tool")
