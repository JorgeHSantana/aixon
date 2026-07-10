# tests/test_tool_agent_invoke.py
import pytest

from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Message
from aixon.registry import get_registry

from langchain_core.messages import AIMessage
from tests._fakes import FakeChatModel


def _install_fake(monkeypatch, llm, script):
    """Force llm.chat_model to return our scripted fake (no provider/network)."""
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_toolagent_requires_llm():
    with pytest.raises(AixonError, match="llm"):
        # Concrete subclass missing the required `llm` attribute.
        type("NoLLMAgent", (ToolAgent,), {"tools": []})


def test_toolagent_suffix_enforced():
    from aixon.exceptions import NamingError

    with pytest.raises(NamingError, match="Agent"):
        type("BadTool", (ToolAgent,), {"llm": LLM("fake-1", provider="fake")})


def test_toolagent_invoke_runs_tool_then_returns_final_message(monkeypatch):
    calls = {"n": 0}

    def adder(a: int, b: int) -> int:
        """Add two integers."""
        calls["n"] += 1
        return a + b

    class MathAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        prompt = "You do math."
        tools = [adder]

    agent = get_registry().resolve("mathagent")
    _install_fake(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 2, "b": 3}),
            AIMessage(content="The answer is 5."),
        ],
    )

    result = agent.invoke([Message(role="user", content="add 2 and 3")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "The answer is 5."
    assert calls["n"] == 1


def test_toolagent_invoke_sets_reasoning_on_message(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class ReasonAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("reasonagent")
    _install_fake(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 1, "b": 1}),
            AIMessage(content="Sum is 2."),
        ],
    )

    result = agent.invoke([Message(role="user", content="add")])

    # A tool-call step label was collected as reasoning.
    assert result.reasoning is not None
    assert "adder" in result.reasoning


def test_toolagent_developer_role_overrides_class_prompt(monkeypatch):
    """A leading 'developer' message (OpenAI's system-role alias) must win
    over the class-level prompt exactly like a leading 'system' message
    already does — otherwise the model sees BOTH the class prompt (as
    create_agent's system_prompt) and the developer message (converted to a
    second SystemMessage by to_langchain), i.e. duplicated system
    instructions."""
    class DevAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        prompt = "class prompt must not reach the model"
        tools = []

    agent = get_registry().resolve("devagent")

    captured: list[list] = []
    fake = FakeChatModel(script=[AIMessage(content="ok")])
    original_generate = fake._generate

    def capturing_generate(messages, stop=None, run_manager=None, **kwargs):
        captured.append(list(messages))
        return original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    fake._generate = capturing_generate
    monkeypatch.setattr(type(agent.llm), "chat_model", property(lambda self: fake))

    agent.invoke([
        Message(role="developer", content="dev wins"),
        Message(role="user", content="hi"),
    ])

    assert len(captured) >= 1
    system_msgs = [m for m in captured[0] if type(m).__name__ == "SystemMessage"]
    assert len(system_msgs) == 1
    assert system_msgs[0].content == "dev wins"


def test_toolagent_is_neutral_in_and_out(monkeypatch):
    def noop(text: str) -> str:
        """noop"""
        return text

    class NeutralAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [noop]

    agent = get_registry().resolve("neutralagent")
    _install_fake(monkeypatch, agent.llm, [AIMessage(content="done immediately")])

    # Pass only neutral Messages; receive a neutral Message.
    result = agent.invoke([Message(role="user", content="hi")])
    assert type(result).__name__ == "Message"
    assert result.content == "done immediately"
