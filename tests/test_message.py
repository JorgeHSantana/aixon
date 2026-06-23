from aixon import Message, Chunk


def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"
    assert m.tool_calls == []
    assert m.reasoning is None


def test_message_to_dict_omits_empty_optionals():
    m = Message(role="user", content="hi")
    assert m.to_dict() == {"role": "user", "content": "hi"}


def test_message_to_dict_includes_set_optionals():
    m = Message(role="tool", content="42", tool_call_id="call_1", name="calc")
    d = m.to_dict()
    assert d["tool_call_id"] == "call_1"
    assert d["name"] == "calc"


def test_tool_calls_are_per_instance():
    a = Message(role="assistant")
    b = Message(role="assistant")
    a.tool_calls.append({"id": "x"})
    assert b.tool_calls == []


def test_chunk_defaults():
    c = Chunk()
    assert c.content == ""
    assert c.reasoning == ""
    assert c.done is False
