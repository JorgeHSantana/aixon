# tests/test_reasoning_agents.py
"""R3: live reasoning-channel wiring (ToolAgent) + per-request reasoning_effort
allow-listing (runtime.py).

ToolAgent: when a NEW AI message produced by the run carries reasoning
extracted by R2's ``reasoning_from_message`` (Anthropic-style thinking blocks,
or the ``reasoning_content`` convention), that text must be emitted into the
active ReasoningChannel BEFORE that same turn's tool-call label(s) — the user
sees the model's own thinking live, between tool calls, not just synthetic
step labels. ``Message.reasoning`` (invoke/ainvoke) keeps being the channel's
full drain, which now also includes this model reasoning.

LLMAgent needs no change — R2 already makes ``LLM.complete``/``LLM.stream``
carry reasoning through, and LLMAgent.invoke/stream are thin delegations to
them. Covered here with a smoke test proving the delegation preserves it.

Per-request: ``reasoning_effort`` joins ``GENERATION_PARAMS`` so a per-request
value on the wire reaches ``request_chat_model()`` -> ``provider.build()``,
mirroring the existing generation-params e2e pattern (TestClient + a fake
provider that records the kwargs it received).
"""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from aixon.agents.llm_agent import LLMAgent
from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Message
from aixon.registry import get_registry
from aixon.runtime import GENERATION_PARAMS
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._fakes import FakeChatModel, FakeProvider


def _install(monkeypatch, llm, script):
    """Force llm.chat_model to return our scripted fake (no provider/network)."""
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _thinking_then_tool_call(text: str, name: str, args: dict, id: str = "call_1"):
    """AIMessage carrying an Anthropic-style thinking block AND a tool call in
    the same turn — the shape a real Claude turn takes when it reasons before
    deciding to call a tool."""
    return AIMessage(
        content=[{"type": "thinking", "thinking": text}],
        tool_calls=[{"name": name, "args": args, "id": id}],
    )


# ── ToolAgent.stream: model reasoning surfaces before the tool-call label ────

def test_stream_emits_model_reasoning_before_tool_call_label(monkeypatch):
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return "sunny"

    class ReasoningWeatherAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [get_weather]

    agent = get_registry().resolve("reasoningweatheragent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _thinking_then_tool_call(
                "I should check the weather.", "get_weather", {"city": "Recife"}
            ),
            AIMessage(content="It's sunny."),
        ],
    )

    chunks = list(agent.stream([Message(role="user", content="weather?")]))
    reasoning_lines = [c.reasoning for c in chunks if c.reasoning]
    joined = "".join(reasoning_lines)

    assert "I should check the weather." in joined
    assert "Calling get_weather..." in joined
    # Model reasoning for this turn comes BEFORE that turn's tool-call label.
    assert joined.index("I should check the weather.") < joined.index(
        "Calling get_weather..."
    )
    assert any("It's sunny." in c.content for c in chunks if c.content)
    assert chunks[-1].done is True


def test_stream_no_reasoning_chunk_when_message_carries_none(monkeypatch):
    # Unchanged behavior: a plain tool-calling AI message (no thinking block,
    # no reasoning_content) yields only the tool-call label, never an empty
    # reasoning line for the (absent) model reasoning.
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class PlainToolAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("plaintoolagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            AIMessage(content="", tool_calls=[{"name": "adder", "args": {"a": 1, "b": 1}, "id": "c1"}]),
            AIMessage(content="Sum is 2."),
        ],
    )

    reasoning_lines = [
        c.reasoning
        for c in agent.stream([Message(role="user", content="add")])
        if c.reasoning
    ]
    assert reasoning_lines == ["Calling adder...\n"]


# ── ToolAgent.invoke: final.reasoning aggregates model thinking + labels ─────

def test_invoke_final_reasoning_includes_model_thinking_before_tool_label(monkeypatch):
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return "sunny"

    class ReasoningInvokeAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [get_weather]

    agent = get_registry().resolve("reasoninginvokeagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _thinking_then_tool_call(
                "Checking the weather.", "get_weather", {"city": "Recife"}
            ),
            AIMessage(content="It's sunny."),
        ],
    )

    result = agent.invoke([Message(role="user", content="weather?")])

    assert result.content == "It's sunny."
    assert result.reasoning is not None
    assert "Checking the weather." in result.reasoning
    assert "Calling get_weather..." in result.reasoning
    assert result.reasoning.index("Checking the weather.") < result.reasoning.index(
        "Calling get_weather..."
    )


# ── Async mirrors: ainvoke / astream ─────────────────────────────────────────

def test_ainvoke_final_reasoning_includes_model_thinking_before_tool_label(monkeypatch):
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return "sunny"

    class AReasoningInvokeAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [get_weather]

    agent = get_registry().resolve("areasoninginvokeagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _thinking_then_tool_call(
                "Checking the weather.", "get_weather", {"city": "Recife"}
            ),
            AIMessage(content="It's sunny."),
        ],
    )

    result = asyncio.run(agent.ainvoke([Message(role="user", content="weather?")]))

    assert result.content == "It's sunny."
    assert result.reasoning is not None
    assert "Checking the weather." in result.reasoning
    assert "Calling get_weather..." in result.reasoning
    assert result.reasoning.index("Checking the weather.") < result.reasoning.index(
        "Calling get_weather..."
    )


def test_astream_emits_model_reasoning_before_tool_call_label(monkeypatch):
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return "sunny"

    class AReasoningWeatherAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [get_weather]

    agent = get_registry().resolve("areasoningweatheragent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _thinking_then_tool_call(
                "I should check the weather.", "get_weather", {"city": "Recife"}
            ),
            AIMessage(content="It's sunny."),
        ],
    )

    async def collect():
        return [c async for c in agent.astream([Message(role="user", content="weather?")])]

    chunks = asyncio.run(collect())
    reasoning_lines = [c.reasoning for c in chunks if c.reasoning]
    joined = "".join(reasoning_lines)

    assert "I should check the weather." in joined
    assert "Calling get_weather..." in joined
    assert joined.index("I should check the weather.") < joined.index(
        "Calling get_weather..."
    )
    assert any("It's sunny." in c.content for c in chunks if c.content)
    assert chunks[-1].done is True


# ── LLMAgent: R2 already does the work — prove the delegation preserves it ──

def test_llm_agent_invoke_carries_reasoning_through(monkeypatch):
    class ReasoningLlmAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")

    agent = get_registry().resolve("reasoningllmagent")
    _install(
        monkeypatch,
        agent.llm,
        [AIMessage(content="answer", additional_kwargs={"reasoning_content": "because"})],
    )

    result = agent.invoke([Message(role="user", content="hi")])
    assert result.content == "answer"
    assert result.reasoning == "because"


def test_llm_agent_stream_carries_reasoning_chunks_through(monkeypatch):
    class ReasoningStreamLlmAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")

    agent = get_registry().resolve("reasoningstreamllmagent")

    class _ThinkingStreamModel(FakeChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=[{"type": "thinking", "thinking": "pondering"}]
                )
            )
            yield ChatGenerationChunk(message=AIMessageChunk(content="answer"))

    monkeypatch.setattr(
        type(agent.llm), "chat_model", property(lambda self: _ThinkingStreamModel())
    )

    chunks = list(agent.stream([Message(role="user", content="hi")]))
    assert any(c.reasoning == "pondering" for c in chunks)
    assert any(c.content == "answer" for c in chunks)
    assert chunks[-1].done is True


# ── Per-request reasoning_effort reaches provider.build() ───────────────────

def test_reasoning_effort_is_in_generation_params_allowlist():
    assert "reasoning_effort" in GENERATION_PARAMS


def test_e2e_reasoning_effort_reaches_provider_build(monkeypatch):
    captured: list[dict] = []
    original_build = FakeProvider.build

    def recording_build(self, model, **params):
        captured.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", recording_build)

    class ReasoningEffortAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = []

    Server._reset()
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    try:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "reasoningeffortagent",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r.status_code == 200
    finally:
        Server._reset()

    assert captured, "FakeProvider.build was never called"
    assert captured[-1].get("reasoning_effort") == "high"
