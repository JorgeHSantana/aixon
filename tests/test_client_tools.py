# tests/test_client_tools.py
"""Client-declared tools: wire-level tool_calls (OpenAI adapter) and the
per-request client_tools runtime channel.

Agentic clients (editors, IDEs) send ``tools`` on the request and expect
``tool_calls`` back to execute locally. The adapter must round-trip both
directions; the Server must publish the tools where agents can read them.
"""
from __future__ import annotations

import json

from aixon.message import Chunk, Message
from aixon.runtime import client_tools, current_client_tools
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ParsedRequest

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "open_file",
        "description": "Opens a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def _deltas(raw: str) -> list[dict]:
    out = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


class TestParseRequest:
    def test_extracts_client_tools(self):
        a = OpenAIAdapter()
        pr = a.parse_request(
            {"model": "m", "messages": [], "tools": [TOOL_DEF]},
            path="/v1/chat/completions",
        )
        assert pr.tools == [TOOL_DEF]
        assert "tools" not in pr.params  # transport field, not a generation knob

    def test_tools_default_none(self):
        a = OpenAIAdapter()
        pr = a.parse_request({"model": "m", "messages": []}, path="/v1/chat/completions")
        assert pr.tools is None

    def test_inbound_assistant_tool_calls_become_neutral(self):
        a = OpenAIAdapter()
        pr = a.parse_request(
            {
                "model": "m",
                "messages": [
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "open_file",
                                     "arguments": '{"path": "/tmp/x"}'},
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                ],
            },
            path="/v1/chat/completions",
        )
        tc = pr.messages[0].tool_calls[0]
        assert tc["name"] == "open_file"
        assert tc["args"] == {"path": "/tmp/x"}
        assert tc["id"] == "call_1"
        assert pr.messages[1].tool_call_id == "call_1"

    def test_malformed_arguments_degrade_to_empty(self):
        a = OpenAIAdapter()
        pr = a.parse_request(
            {"model": "m", "messages": [
                {"role": "assistant", "tool_calls": [{
                    "id": "c", "function": {"name": "f", "arguments": "{broken"},
                }]},
            ]},
            path="/v1/chat/completions",
        )
        assert pr.messages[0].tool_calls[0]["args"] == {}


class TestFormatResponse:
    def test_tool_calls_on_the_wire(self):
        a = OpenAIAdapter()
        out = a.format_response(
            model="m",
            message=Message(role="assistant", content="",
                            tool_calls=[{"name": "open_file",
                                         "args": {"path": "/tmp/x"}, "id": "c1"}]),
            usage={},
        )
        choice = out["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None  # tool-calls-only turn
        wire = choice["message"]["tool_calls"][0]
        assert wire == {"id": "c1", "type": "function",
                        "function": {"name": "open_file",
                                     "arguments": '{"path": "/tmp/x"}'}}

    def test_wire_shaped_tool_calls_pass_through(self):
        a = OpenAIAdapter()
        wire_call = {"id": "c9", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}
        out = a.format_response(
            model="m",
            message=Message(role="assistant", tool_calls=[wire_call]),
            usage={},
        )
        assert out["choices"][0]["message"]["tool_calls"] == [wire_call]


class TestStreamSession:
    def _session(self):
        a = OpenAIAdapter()
        request = ParsedRequest(model="m", messages=[], params={}, stream=True)
        return a.open_stream(model="m", request=request)

    def test_tool_calls_stream_as_two_deltas_then_finish(self):
        s = self._session()
        raw = s.chunk(Chunk(
            tool_calls=[{"name": "open_file", "args": {"path": "/tmp/x"}, "id": "c1"}],
        ))
        raw += s.chunk(Chunk(done=True))
        deltas = [d["choices"][0]["delta"] for d in _deltas(raw)]
        finishes = [d["choices"][0]["finish_reason"] for d in _deltas(raw)]
        head, args = deltas[0]["tool_calls"][0], deltas[1]["tool_calls"][0]
        assert head["id"] == "c1" and head["function"]["name"] == "open_file"
        assert head["function"]["arguments"] == ""  # head delta carries no args
        assert json.loads(args["function"]["arguments"]) == {"path": "/tmp/x"}
        assert head["index"] == args["index"] == 0
        assert finishes[-1] == "tool_calls"

    def test_head_only_accumulator_still_gets_arguments(self):
        # Clients exist that seed args={} from the FIRST delta and only parse
        # arguments accumulated from later deltas; the split emission must
        # leave them with the full arguments.
        s = self._session()
        raw = s.chunk(Chunk(tool_calls=[{"name": "f", "args": {"a": 1}, "id": "c"}]))
        raw += s.chunk(Chunk(done=True))
        part = None
        for d in _deltas(raw):
            tcs = d["choices"][0]["delta"].get("tool_calls")
            if not tcs:
                continue
            u = tcs[0]
            if part is None:
                part = {"argsText": u.get("function", {}).get("arguments", "")}
            else:
                part["argsText"] += u.get("function", {}).get("arguments", "")
        assert json.loads(part["argsText"]) == {"a": 1}

    def test_indices_advance_across_chunks(self):
        s = self._session()
        raw = s.chunk(Chunk(tool_calls=[{"name": "f1", "args": {}, "id": "a"}]))
        raw += s.chunk(Chunk(tool_calls=[{"name": "f2", "args": {}, "id": "b"}]))
        indices = sorted({tc["index"] for d in _deltas(raw)
                          for tc in d["choices"][0]["delta"].get("tool_calls", [])})
        assert indices == [0, 1]

    def test_plain_stream_still_finishes_with_stop(self):
        s = self._session()
        raw = s.chunk(Chunk(content="hi"))
        raw += s.chunk(Chunk(done=True))
        finishes = [d["choices"][0]["finish_reason"] for d in _deltas(raw)]
        assert finishes[-1] == "stop"


class TestRuntimeChannel:
    def test_publish_and_read(self):
        assert current_client_tools() == []
        with client_tools([TOOL_DEF]):
            got = current_client_tools()
            assert got == [TOOL_DEF]
            got.clear()  # mutating the copy must not pollute the channel
            assert current_client_tools() == [TOOL_DEF]
        assert current_client_tools() == []

    def test_none_and_empty_are_empty(self):
        with client_tools(None):
            assert current_client_tools() == []
        with client_tools([]):
            assert current_client_tools() == []


class TestServerWiring:
    def test_agent_sees_client_tools_and_returns_tool_calls(self):
        import asyncio

        from fastapi.testclient import TestClient

        from aixon.agent import Agent
        from aixon.registry import get_registry
        from aixon.server.server import Server

        class _ClientToolsProbeAgent(Agent):
            """Echoes the client tools it sees as a tool call."""
            name = "ct-probe"

            def invoke(self, messages):
                return asyncio.run(self.ainvoke(messages))

            async def ainvoke(self, messages):
                tools = current_client_tools()
                name = tools[0]["function"]["name"] if tools else "none"
                return Message(role="assistant", content="",
                               tool_calls=[{"name": name, "args": {"seen": len(tools)},
                                            "id": "probe"}])

            def stream(self, messages):
                yield Chunk(done=True)

            async def astream(self, messages):
                final = await self.ainvoke(messages)
                yield Chunk(tool_calls=final.tool_calls)
                yield Chunk(done=True)

        try:
            Server._reset()
            client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
            body = {"model": "ct-probe", "tools": [TOOL_DEF],
                    "messages": [{"role": "user", "content": "go"}]}

            r = client.post("/v1/chat/completions", json=body)
            choice = r.json()["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            wire = choice["message"]["tool_calls"][0]
            assert wire["function"]["name"] == "open_file"
            assert json.loads(wire["function"]["arguments"]) == {"seen": 1}

            with client.stream("POST", "/v1/chat/completions",
                               json={**body, "stream": True}) as resp:
                raw = resp.read().decode()
            names = [tc["function"]["name"] for d in _deltas(raw) if d.get("choices")
                     for tc in d["choices"][0]["delta"].get("tool_calls", [])
                     if tc.get("function", {}).get("name")]
            assert names == ["open_file"]
        finally:
            get_registry()._agents.pop("ct-probe", None)
            Server._reset()
