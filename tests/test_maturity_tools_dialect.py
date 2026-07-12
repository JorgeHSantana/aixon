# tests/test_maturity_tools_dialect.py
"""M2 — tool-dialect normalization.

``ParsedRequest.tools`` (and therefore ``current_client_tools()``) must ALWAYS
be OpenAI-shaped (``{"type": "function", "function": {"name", "description",
"parameters"}}``), regardless of which ``ProtocolAdapter`` parsed the wire
request. The Anthropic adapter receives Anthropic-shaped tool defs
(``{"name", "description", "input_schema"}``) and must convert them; an
OpenAI-shaped tool def passed to the Anthropic adapter (or produced by the
OpenAI adapter) must pass through untouched.
"""
from __future__ import annotations

from aixon.server.adapters.anthropic import AnthropicAdapter


def _parse(tools):
    adapter = AnthropicAdapter()
    return adapter.parse_request(
        {"model": "m", "messages": [], "tools": tools}, path="/v1/messages"
    )


def test_anthropic_tool_def_is_converted_to_openai_shape():
    pr = _parse([
        {"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}
    ])
    assert pr.tools[0]["type"] == "function"
    assert pr.tools[0]["function"]["name"] == "get_weather"
    assert pr.tools[0]["function"]["description"] == "d"
    assert pr.tools[0]["function"]["parameters"] == {"type": "object"}


def test_already_openai_shaped_tool_passes_through_intact():
    openai_tool = {
        "type": "function",
        "function": {"name": "get_weather", "description": "d", "parameters": {}},
    }
    pr = _parse([openai_tool])
    assert pr.tools == [openai_tool]
    assert pr.tools[0] is openai_tool


def test_non_dict_tool_entries_are_ignored():
    pr = _parse(["not-a-dict", 42, None])
    assert pr.tools is None


def test_no_tools_yields_none():
    pr = _parse(None)
    assert pr.tools is None


def test_mixed_anthropic_and_openai_and_junk_tools():
    openai_tool = {
        "type": "function",
        "function": {"name": "already", "description": "", "parameters": {}},
    }
    pr = _parse([
        {"name": "get_weather", "description": "d", "input_schema": {"type": "object"}},
        openai_tool,
        "junk",
    ])
    assert len(pr.tools) == 2
    assert pr.tools[0]["function"]["name"] == "get_weather"
    assert pr.tools[1] is openai_tool


def test_anthropic_server_tools_are_skipped():
    # Anthropic SERVER tools (no input_schema, e.g. web_search) cannot be
    # expressed as a client function tool — they must be skipped, not turned
    # into a bogus empty-parameters function def.
    pr = _parse([
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {"name": "real_tool", "description": "d", "input_schema": {"type": "object"}},
    ])
    assert len(pr.tools) == 1
    assert pr.tools[0]["function"]["name"] == "real_tool"
