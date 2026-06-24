# tests/test_adapter_openai.py
from __future__ import annotations

import json

from aixon.message import Chunk, Message
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ParsedRequest


def _data(line: str) -> dict:
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    return json.loads(line[len("data: ") : -2])


class TestParseRequest:
    def test_parses_model_messages_stream(self):
        a = OpenAIAdapter()
        pr = a.parse_request(
            {
                "model": "echo",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "hi"},
                ],
                "stream": True,
                "temperature": 0.3,
            },
            path="/v1/chat/completions",
        )
        assert isinstance(pr, ParsedRequest)
        assert pr.model == "echo"
        assert [m.role for m in pr.messages] == ["system", "user"]
        assert all(isinstance(m, Message) for m in pr.messages)
        assert pr.stream is True
        assert pr.params == {"temperature": 0.3}

    def test_defaults_when_fields_absent(self):
        a = OpenAIAdapter()
        pr = a.parse_request({"messages": []}, path="/v1/chat/completions")
        assert pr.model == ""
        assert pr.messages == []
        assert pr.stream is False
        assert pr.params == {}


class TestFormatResponse:
    def test_chat_completion_envelope(self):
        a = OpenAIAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", content="hello"),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        assert out["object"] == "chat.completion"
        assert out["model"] == "echo"
        assert out["id"].startswith("chatcmpl-")
        choice = out["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"] == {"role": "assistant", "content": "hello"}
        assert out["usage"]["total_tokens"] == 2

    def test_reasoning_included_when_present(self):
        a = OpenAIAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", content="x", reasoning="why"),
            usage={},
        )
        assert out["choices"][0]["message"]["reasoning"] == "why"


class TestFormatStream:
    def test_content_chunk(self):
        a = OpenAIAdapter()
        d = _data(a.format_stream_chunk(model="echo", chunk=Chunk(content="hi")))
        assert d["object"] == "chat.completion.chunk"
        assert d["model"] == "echo"
        assert d["choices"][0]["delta"] == {"content": "hi"}
        assert d["choices"][0]["finish_reason"] is None

    def test_reasoning_chunk(self):
        a = OpenAIAdapter()
        d = _data(a.format_stream_chunk(model="echo", chunk=Chunk(reasoning="r")))
        assert d["choices"][0]["delta"] == {"reasoning": "r"}

    def test_done_chunk_is_finish_line(self):
        a = OpenAIAdapter()
        d = _data(a.format_stream_chunk(model="echo", chunk=Chunk(done=True)))
        assert d["choices"][0]["delta"] == {}
        assert d["choices"][0]["finish_reason"] == "stop"

    def test_empty_chunk_skipped(self):
        a = OpenAIAdapter()
        assert a.format_stream_chunk(model="echo", chunk=Chunk()) == ""

    def test_stream_done_is_done_sentinel(self):
        a = OpenAIAdapter()
        assert a.format_stream_done(model="echo") == "data: [DONE]\n\n"


class TestFormatModelsAndRoutes:
    def test_models_lists_agents_and_aliases(self):
        a = OpenAIAdapter()

        class _Fake:
            def __init__(self, name, aliases, owned_by):
                self.name = name
                self.aliases = aliases
                self.owned_by = owned_by

        out = a.format_models([_Fake("echo", ["e1"], "aixon")])
        assert out["object"] == "list"
        ids = [d["id"] for d in out["data"]]
        assert ids == ["echo", "e1"]
        assert all(d["object"] == "model" for d in out["data"])
        assert out["data"][0]["owned_by"] == "aixon"

    def test_routes(self):
        a = OpenAIAdapter()
        assert a.routes() == [
            ("POST", "/v1/chat/completions"),
            ("GET", "/v1/models"),
        ]
        assert a.name == "openai"
