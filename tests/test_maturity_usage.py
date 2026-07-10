# tests/test_maturity_usage.py
"""M1 — real provider usage via Message.usage.

The provider's REAL token usage (LangChain ``AIMessage.usage_metadata``) must
flow through the neutral boundary: ``from_langchain`` converts it to the
OpenAI-shaped ``Message.usage``, LLMAgent passes it through, ToolAgent SUMS it
over the AI messages produced by the run, and the server's non-stream paths
prefer it over the tiktoken estimate (``build_usage`` stays as fallback).
"""
from __future__ import annotations

import asyncio
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from aixon._interop.messages import from_langchain
from aixon.agent import Agent
from aixon.agents.llm_agent import LLMAgent
from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._fakes import FakeChatModel


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _install_fake(monkeypatch, llm, script):
    """Force llm.chat_model to return our scripted fake (no provider/network)."""
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _usage_meta(inp, out):
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


# --- boundary: from_langchain -------------------------------------------------

def test_from_langchain_converts_usage_metadata_to_openai_shape():
    msg = from_langchain(
        AIMessage(content="hi", usage_metadata=_usage_meta(3, 5))
    )
    assert msg.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_from_langchain_without_usage_metadata_leaves_usage_none():
    msg = from_langchain(AIMessage(content="hi"))
    assert msg.usage is None


def test_message_usage_field_defaults_to_none():
    assert Message(role="assistant", content="x").usage is None


# --- LLMAgent: pass-through ----------------------------------------------------

def test_llm_agent_invoke_propagates_provider_usage(monkeypatch):
    class UsagellmAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")

    agent = get_registry().resolve("usagellmagent")
    _install_fake(
        monkeypatch, agent.llm,
        [AIMessage(content="hello", usage_metadata=_usage_meta(7, 2))],
    )

    result = agent.invoke([Message(role="user", content="hi")])
    assert result.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


def test_llm_agent_ainvoke_propagates_provider_usage(monkeypatch):
    class AusagellmAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")

    agent = get_registry().resolve("ausagellmagent")
    _install_fake(
        monkeypatch, agent.llm,
        [AIMessage(content="hello", usage_metadata=_usage_meta(4, 6))],
    )

    result = asyncio.run(agent.ainvoke([Message(role="user", content="hi")]))
    assert result.usage == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }


# --- ToolAgent: sum over the run's AI messages ---------------------------------

def _adder(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _tool_call_msg(usage=None):
    kwargs = {"content": "", "tool_calls": [
        {"name": "_adder", "args": {"a": 2, "b": 3}, "id": "call_1"}
    ]}
    if usage:
        kwargs["usage_metadata"] = usage
    return AIMessage(**kwargs)


def test_tool_agent_invoke_sums_usage_across_steps(monkeypatch):
    class SumusageAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [_adder]

    agent = get_registry().resolve("sumusageagent")
    _install_fake(
        monkeypatch, agent.llm,
        [
            _tool_call_msg(usage=_usage_meta(10, 4)),
            AIMessage(content="The answer is 5.", usage_metadata=_usage_meta(20, 6)),
        ],
    )

    result = agent.invoke([Message(role="user", content="add 2 and 3")])
    assert result.content == "The answer is 5."
    assert result.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    }


def test_tool_agent_ainvoke_sums_usage_across_steps(monkeypatch):
    class AsumusageAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [_adder]

    agent = get_registry().resolve("asumusageagent")
    _install_fake(
        monkeypatch, agent.llm,
        [
            _tool_call_msg(usage=_usage_meta(11, 3)),
            AIMessage(content="Sum is 5.", usage_metadata=_usage_meta(25, 7)),
        ],
    )

    result = asyncio.run(agent.ainvoke([Message(role="user", content="add")]))
    assert result.usage == {
        "prompt_tokens": 36,
        "completion_tokens": 10,
        "total_tokens": 46,
    }


def test_tool_agent_usage_none_when_provider_reports_none(monkeypatch):
    class NousageAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [_adder]

    agent = get_registry().resolve("nousageagent")
    _install_fake(
        monkeypatch, agent.llm,
        [
            _tool_call_msg(),
            AIMessage(content="Sum is 5."),
        ],
    )

    result = agent.invoke([Message(role="user", content="add")])
    assert result.usage is None


# --- server non-stream: real usage wins, estimate is fallback ------------------

REAL_USAGE = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


class _UsageEchoAgent(Agent, abstract=True):
    """Fake agent whose Message carries provider-real usage."""

    def invoke(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content="ok", usage=dict(REAL_USAGE))

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(content="ok")
        yield Chunk(done=True)


def _make_usage_echo(name: str):
    cls = type("MadeUsageEchoAgent", (_UsageEchoAgent,), {"name": name})
    return get_registry().resolve(cls.name)


def test_openai_non_stream_uses_real_usage_over_estimate():
    _make_usage_echo("realusage")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = client.post("/v1/chat/completions", json={
        "model": "realusage",
        "messages": [{"role": "user", "content": "a much longer prompt than three tokens for sure"}],
    })
    assert r.status_code == 200
    assert r.json()["usage"] == REAL_USAGE


def test_anthropic_non_stream_uses_real_usage_over_estimate():
    _make_usage_echo("realusage2")
    client = TestClient(Server(adapters=[AnthropicAdapter()]).app)
    r = client.post("/v1/messages", json={
        "model": "realusage2",
        "messages": [{"role": "user", "content": "a much longer prompt than three tokens for sure"}],
    })
    assert r.status_code == 200
    assert r.json()["usage"] == {"input_tokens": 3, "output_tokens": 5}


def test_openai_non_stream_falls_back_to_estimate_without_real_usage():
    pytest.importorskip("tiktoken")
    from tests._server_fakes import make_echo

    make_echo("plainecho")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = client.post("/v1/chat/completions", json={
        "model": "plainecho",
        "messages": [{"role": "user", "content": "hi there"}],
    })
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
