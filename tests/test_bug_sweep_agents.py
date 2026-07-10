# tests/test_bug_sweep_agents.py
"""Regression tests for the agents/runtime bug sweep (A1-A6).

Follows the fake-provider/registry-reset patterns of test_llm_agent.py,
test_agent.py, and test_nested_reasoning.py. tests/conftest.py resets the
Agent registry around every test; provider registration is idempotent /
scoped to distinct provider names here so it never leaks into other test
files that rely on the shared "fake" provider.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.agent import AgentTool
from aixon.agents.llm_agent import LLMAgent
from aixon.agents.orchestrator import Orchestrator
from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Message
from aixon.providers.base import Provider, register_provider
from aixon.registry import get_registry
from aixon.runtime import client_tools, current_client_tools, generation_params
from aixon._interop.tools import coerce_tools

from tests._fakes import FakeChatModel, make_echo_agent, make_llm, register_fake_provider

register_fake_provider()


# ── A1: ToolAgent applies request generation params via request_chat_model ──

class _RecordingProvider(Provider):
    """Provider named 'fake-recording' whose build() records every kwargs
    dict it receives and returns a fresh scriptable FakeChatModel."""

    name = "fake-recording"
    env_key = "FAKE_API_KEY"

    def __init__(self):
        self.calls: list[dict] = []

    def build(self, model: str, **params):
        self.calls.append(dict(params))
        return FakeChatModel(script=[AIMessage(content="final answer")])


def test_tool_agent_applies_request_generation_params():
    provider = _RecordingProvider()
    register_provider(provider)

    def noop(text: str) -> str:
        """noop"""
        return text

    class RecordingToolAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake-recording", temperature=0.9)
        tools = [noop]

    agent = get_registry().resolve("recordingtoolagent")
    with generation_params({"temperature": 0.0}):
        agent.invoke([Message(role="user", content="hi")])

    assert provider.calls, "provider.build() was never called"
    assert provider.calls[-1]["temperature"] == 0.0


def test_request_chat_model_without_params_returns_cached():
    llm = LLM("fake-1", provider="fake")
    cm = llm.chat_model
    assert llm.request_chat_model() is cm


# ── A2: tool-call labels are not re-emitted for history AI messages ────────

def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_tool_call_labels_not_reemitted_for_history_messages(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class HistoryAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("historyagent")
    fake = FakeChatModel(script=[AIMessage(content="10 * 10 is 100.")])
    monkeypatch.setattr(type(agent.llm), "chat_model", property(lambda self: fake))

    history = [
        Message(role="user", content="add 2 and 3"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "adder", "args": {"a": 2, "b": 3}, "id": "call_1"}],
        ),
        Message(role="tool", content="5", tool_call_id="call_1", name="adder"),
        Message(role="user", content="thanks, now what is 10 * 10?"),
    ]

    result = agent.invoke(history)

    assert result.content == "10 * 10 is 100."
    # The old tool call (from the caller's history) must NOT produce a label:
    # this run never called any tool itself.
    assert not result.reasoning


# ── A3: LLMAgent client system message overrides the class prompt ──────────

@pytest.fixture
def prompted_llm_agent():
    class PromptedLLMAgent(LLMAgent):
        llm = LLM("fake-1", provider="fake")
        prompt = "class prompt"

    return get_registry().resolve("promptedllmagent")


def test_llm_agent_client_system_overrides_class_prompt(prompted_llm_agent):
    captured: list = []

    class Rec(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.extend(messages)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

    agent = prompted_llm_agent
    agent.llm._chat_model = Rec()

    agent.invoke(
        [
            Message(role="system", content="client system"),
            Message(role="user", content="hi"),
        ]
    )

    system_msgs = [m for m in captured if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
    assert system_msgs[0].content == "client system"


def test_llm_agent_prompt_still_prepended_without_client_system(prompted_llm_agent):
    agent = prompted_llm_agent
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
    assert seen[0][0].content == "class prompt"


# ── A4: Tier-1 worker named "supervisor" collides with the internal node ──

def test_orchestrator_rejects_worker_named_supervisor():
    make_echo_agent("supervisor")

    with pytest.raises(AixonError, match="supervisor"):

        class BadOrchestrator(Orchestrator):
            supervisor = make_llm()
            agents = [get_registry().resolve("supervisor")]


# ── A5: current_client_tools() deep-copies (no shared nested dicts) ────────

def test_current_client_tools_deep_copies():
    with client_tools([{"function": {"name": "f"}}]):
        first = current_client_tools()
        first[0]["function"]["name"] = "mutated"
        second = current_client_tools()
        assert second[0]["function"]["name"] == "f"


# ── A6: coerce_tools rejects duplicate tool names ──────────────────────────

def test_coerce_tools_rejects_duplicate_names():
    at1 = AgentTool(name="dup", description="d1", func=lambda text: text)
    at2 = AgentTool(name="dup", description="d2", func=lambda text: text)

    with pytest.raises(AixonError, match="dup"):
        coerce_tools([at1, at2])
