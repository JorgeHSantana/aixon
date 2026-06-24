# tests/test_llm_agent.py
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tests._fakes import register_fake_provider  # registers fake provider

from aixon.agents.llm_agent import LLMAgent
from aixon.exceptions import AixonError, NamingError
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry

register_fake_provider()


# ── Fixtures: Agent subclasses registered after per-test reset ────────────────

@pytest.fixture
def echo_agent():
    """Create and register EchoLLMAgent fresh for each test."""
    class EchoLLMAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")
        description = "Echoes via fake LLM"

    return get_registry().resolve("echollmagent")


@pytest.fixture
def prompted_agent():
    """Create and register PromptedLLMAgent fresh for each test."""
    class PromptedLLMAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")
        prompt = "You are a helpful assistant."

    return get_registry().resolve("promptedllmagent")


# ── Valid concrete subclass runs OFFLINE ─────────────────────────────────────

def test_llm_agent_registers_itself(echo_agent):
    assert isinstance(echo_agent, LLMAgent)
    assert echo_agent.name == "echollmagent"


def test_llm_agent_invoke_runs_offline(echo_agent):
    agent = echo_agent
    agent.llm.chat_model.script = [AIMessage(content="pong")]
    result = agent.invoke([Message(role="user", content="ping")])
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "pong"


def test_llm_agent_stream_runs_offline(echo_agent):
    agent = echo_agent
    agent.llm.chat_model.script = [AIMessage(content="streamed")]
    chunks = list(agent.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any(c.content for c in chunks)
    assert chunks[-1].done is True


# ── System prompt prepending ──────────────────────────────────────────────────

def test_prompt_prepended_as_system_message(prompted_agent):
    agent = prompted_agent
    seen: list[list[Message]] = []
    original = agent.llm.complete

    def capturing(messages):
        seen.append(list(messages))
        return original(messages)

    agent.llm.complete = capturing
    agent.invoke([Message(role="user", content="hello")])
    agent.llm.complete = original

    assert len(seen) == 1
    assert seen[0][0].role == "system"
    assert seen[0][0].content == "You are a helpful assistant."


def test_prompt_does_not_mutate_caller_list(prompted_agent):
    agent = prompted_agent
    msgs = [Message(role="user", content="hello")]
    agent.invoke(msgs)
    assert len(msgs) == 1


def test_no_prompt_does_not_prepend(echo_agent):
    agent = echo_agent
    assert agent.prompt == ""
    seen: list[list[Message]] = []
    original = agent.llm.complete

    def capturing(messages):
        seen.append(list(messages))
        return original(messages)

    agent.llm.complete = capturing
    agent.invoke([Message(role="user", content="x")])
    agent.llm.complete = original
    assert seen[0][0].role == "user"


# ── Validation ────────────────────────────────────────────────────────────────

def test_missing_llm_raises_aixon_error():
    with pytest.raises(AixonError, match="llm"):
        class NoLLMAgent(LLMAgent):
            pass


def test_llm_agent_itself_not_registered():
    names = [a.name for a in get_registry().all()]
    assert "llmagent" not in names


def test_bad_suffix_raises():
    with pytest.raises(NamingError, match="Agent"):
        class BadName(LLMAgent):
            llm = LLM("fake-1", provider="fake")
