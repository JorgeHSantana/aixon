# tests/test_as_tool_audience.py
"""#15 — moldura de subagente opcional no as_tool (audience="agent")."""
from __future__ import annotations

import asyncio

import pytest

from aixon.agent import _AGENT_AUDIENCE_SUFFIX
from aixon.exceptions import AixonError
from aixon.message import Message
from tests._fakes import make_echo_agent


def test_audience_agent_anexa_moldura():
    echo = make_echo_agent("eco15a")
    tool = echo.as_tool(audience="agent")
    result = tool.func("qual o total?")
    # O eco devolve a última mensagem: texto + moldura.
    assert result.startswith("qual o total?")
    assert _AGENT_AUDIENCE_SUFFIX in result


def test_audience_default_nao_muda_nada():
    echo = make_echo_agent("eco15b")
    tool = echo.as_tool()
    assert tool.func("oi") == "oi"


def test_audience_agent_no_caminho_async():
    echo = make_echo_agent("eco15c")
    tool = echo.as_tool(audience="agent")
    result = asyncio.run(tool.coroutine("qual o total?"))
    assert _AGENT_AUDIENCE_SUFFIX in result


def test_audience_invalida_erro_claro():
    echo = make_echo_agent("eco15d")
    with pytest.raises(AixonError, match="audience"):
        echo.as_tool(audience="robot")
