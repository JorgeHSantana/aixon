# tests/test_tools.py
import pytest
from langchain_core.tools import BaseTool, StructuredTool, tool

from aixon.agent import AgentTool
from aixon.exceptions import AixonError
from aixon._interop.tools import coerce_tools


def test_agenttool_becomes_structuredtool():
    at = AgentTool(name="greeter", description="says hi", func=lambda text: "hi " + text)
    [coerced] = coerce_tools([at])
    assert isinstance(coerced, BaseTool)
    assert coerced.name == "greeter"
    assert coerced.description == "says hi"
    # The wrapped func runs through the LangChain tool.
    assert coerced.invoke({"text": "bob"}) == "hi bob"


def test_langchain_basetool_passes_through_unchanged():
    @tool
    def echo(text: str) -> str:
        """Echo the text."""
        return text

    result = coerce_tools([echo])
    assert result == [echo]  # same object, not re-wrapped


def test_plain_callable_is_wrapped():
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    [coerced] = coerce_tools([multiply])
    assert isinstance(coerced, BaseTool)
    assert coerced.name == "multiply"
    assert coerced.invoke({"a": 3, "b": 4}) == 12


def test_mixed_list_preserves_order_and_types():
    at = AgentTool(name="t1", description="d1", func=lambda text: text)

    @tool
    def t2(text: str) -> str:
        """second"""
        return text

    def t3(text: str) -> str:
        """third"""
        return text

    out = coerce_tools([at, t2, t3])
    assert [t.name for t in out] == ["t1", "t2", "t3"]
    assert all(isinstance(t, BaseTool) for t in out)
    assert out[1] is t2  # passthrough preserved


def test_unsupported_entry_raises_aixon_error():
    with pytest.raises(AixonError, match="cannot be used as a tool"):
        coerce_tools([42])


def test_empty_list_returns_empty():
    assert coerce_tools([]) == []
