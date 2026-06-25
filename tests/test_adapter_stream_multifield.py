# tests/test_adapter_stream_multifield.py
"""Stream chunks may carry content + reasoning + done together (message.py
allows it). The adapters must not drop fields (audit 3.2). These would FAIL
against the old exclusive if/return ladder."""
from __future__ import annotations

import json

from aixon.message import Chunk
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter


def _openai_delta(line: str) -> dict:
    assert line.startswith("data: ")
    return json.loads(line[len("data: "):])["choices"][0]["delta"]


# --- OpenAI ------------------------------------------------------------------

def test_openai_content_and_done_keeps_content():
    # Old bug: Chunk(content=..., done=True) emitted only {"delta": {}} — content lost.
    line = OpenAIAdapter().format_stream_chunk(
        model="m", chunk=Chunk(content="final answer", done=True)
    )
    payload = json.loads(line[len("data: "):])["choices"][0]
    assert payload["delta"].get("content") == "final answer"
    assert payload["finish_reason"] == "stop"


def test_openai_content_and_reasoning_keeps_both():
    delta = _openai_delta(
        OpenAIAdapter().format_stream_chunk(
            model="m", chunk=Chunk(content="hi", reasoning="thinking")
        )
    )
    assert delta.get("content") == "hi"
    assert delta.get("reasoning") == "thinking"


def test_openai_single_fields_unchanged():
    assert _openai_delta(OpenAIAdapter().format_stream_chunk(model="m", chunk=Chunk(content="x"))) == {"content": "x"}
    assert _openai_delta(OpenAIAdapter().format_stream_chunk(model="m", chunk=Chunk(reasoning="r"))) == {"reasoning": "r"}
    done = json.loads(OpenAIAdapter().format_stream_chunk(model="m", chunk=Chunk(done=True))[len("data: "):])
    assert done["choices"][0]["delta"] == {} and done["choices"][0]["finish_reason"] == "stop"
    assert OpenAIAdapter().format_stream_chunk(model="m", chunk=Chunk()) == ""


# --- Anthropic ---------------------------------------------------------------

def test_anthropic_content_and_done_emits_both_events():
    out = AnthropicAdapter().format_stream_chunk(
        model="m", chunk=Chunk(content="final", done=True)
    )
    assert "text_delta" in out and "final" in out      # content not dropped
    assert "message_delta" in out                       # plus the stop event


def test_anthropic_content_and_reasoning_emits_both_blocks():
    out = AnthropicAdapter().format_stream_chunk(
        model="m", chunk=Chunk(content="hi", reasoning="thinking")
    )
    assert "thinking_delta" in out and "thinking" in out
    assert "text_delta" in out and "hi" in out


def test_anthropic_single_fields_unchanged():
    assert AnthropicAdapter().format_stream_chunk(model="m", chunk=Chunk()) == ""
    only_content = AnthropicAdapter().format_stream_chunk(model="m", chunk=Chunk(content="x"))
    assert "text_delta" in only_content and "message_delta" not in only_content
