from __future__ import annotations

import pytest

from aixon.server.protocol import (
    Chunk,
    Message,
    ParsedRequest,
    ProtocolAdapter,
)


def test_neutral_types_are_reexported_from_message_module():
    # protocol.py must re-export the SAME objects as aixon.message, not copies.
    import aixon.message as m

    assert Message is m.Message
    assert Chunk is m.Chunk


def test_parsed_request_fields():
    pr = ParsedRequest(
        model="echo",
        messages=[Message(role="user", content="hi")],
        params={"temperature": 0.2},
        stream=True,
    )
    assert pr.model == "echo"
    assert pr.messages[0].content == "hi"
    assert pr.params["temperature"] == 0.2
    assert pr.stream is True


def test_protocol_adapter_is_abstract():
    with pytest.raises(TypeError):
        ProtocolAdapter()  # all six methods abstract -> cannot instantiate


def test_concrete_adapter_must_implement_all_methods():
    # A subclass missing any abstract method is still abstract.
    class Partial(ProtocolAdapter):
        name = "partial"

        def parse_request(self, body, *, path):
            return ParsedRequest(model="x", messages=[], params={}, stream=False)

    with pytest.raises(TypeError):
        Partial()


def test_fully_concrete_adapter_instantiates():
    class Full(ProtocolAdapter):
        name = "full"

        def parse_request(self, body, *, path):
            return ParsedRequest(model="x", messages=[], params={}, stream=False)

        def format_response(self, *, model, message, usage):
            return {}

        def format_stream_chunk(self, *, model, chunk):
            return ""

        def format_stream_done(self, *, model):
            return "data: [DONE]\n\n"

        def format_models(self, agents):
            return {"data": []}

        def routes(self):
            return [("POST", "/x")]

    adapter = Full()
    assert adapter.name == "full"
    assert adapter.routes() == [("POST", "/x")]
