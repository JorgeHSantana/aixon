# tests/test_agenttool_args_schema.py
"""AgentTool.args_schema: a neutral JSON-Schema dict that coerce_tools forwards
to StructuredTool, so schema-carrying tools (e.g. MCP) expose typed args to the
LLM instead of a single free-text argument."""
from __future__ import annotations

import asyncio

from aixon.agent import AgentTool
from aixon._interop.tools import coerce_tools

_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer"},
    },
    "required": ["city"],
}


def test_args_schema_defaults_to_none():
    tool = AgentTool(name="t", description="d", func=lambda text: text)
    assert tool.args_schema is None


def test_coerce_forwards_args_schema_and_calls_func_with_kwargs():
    captured: dict = {}

    def run(**kwargs) -> str:
        captured.update(kwargs)
        return "ok"

    entry = AgentTool(name="forecast", description="d", func=run,
                      args_schema=_SCHEMA)
    [tool] = coerce_tools([entry])
    out = tool.invoke({"city": "SP", "days": 2})
    assert out == "ok"
    assert captured == {"city": "SP", "days": 2}
    # The LLM-facing arg surface comes from the JSON Schema, not free text.
    assert set(tool.args) == {"city", "days"}


def test_coerce_args_schema_async_path():
    captured: dict = {}

    def run(**kwargs) -> str:
        return "sync"

    async def arun(**kwargs) -> str:
        captured.update(kwargs)
        return "async"

    entry = AgentTool(name="forecast", description="d", func=run,
                      coroutine=arun, args_schema=_SCHEMA)
    [tool] = coerce_tools([entry])
    out = asyncio.run(tool.ainvoke({"city": "POA"}))
    assert out == "async"
    assert captured == {"city": "POA"}


def test_without_args_schema_single_text_arg_unchanged():
    entry = AgentTool(name="echo", description="d", func=lambda text: text[::-1])
    [tool] = coerce_tools([entry])
    assert tool.invoke({"text": "abc"}) == "cba"
