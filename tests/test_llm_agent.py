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


# ── Valid concrete subclass runs OFFLINE ─────────────────────────────────────

class EchoLLMAgent(LLMAgent):
    llm = LLM("fake-1", provider="fake")
    description = "Echoes via fake LLM"


def test_llm_agent_registers_itself():
    assert isinstance(get_registry().resolve("echollmagent"), EchoLLMAgent)


def test_llm_agent_invoke_runs_offline():
    agent = get_registry().resolve("echollmagent")
    agent.llm.chat_model.script = [AIMessage(content="pong")]
    result = agent.invoke([Message(role="user", content="ping")])
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "pong"


def test_llm_agent_stream_runs_offline():
    agent = get_registry().resolve("echollmagent")
    agent.llm.chat_model.script = [AIMessage(content="streamed")]
    chunks = list(agent.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any(c.content for c in chunks)
    assert chunks[-1].done is True


# ── System prompt prepending ──────────────────────────────────────────────────

class PromptedLLMAgent(LLMAgent):
    llm = LLM("fake-1", provider="fake")
    prompt = "You are a helpful assistant."


def test_prompt_prepended_as_system_message():
    agent = get_registry().resolve("promptedllmagent")
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


def test_prompt_does_not_mutate_caller_list():
    agent = get_registry().resolve("promptedllmagent")
    msgs = [Message(role="user", content="hello")]
    agent.invoke(msgs)
    assert len(msgs) == 1


def test_no_prompt_does_not_prepend():
    agent = get_registry().resolve("echollmagent")
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
