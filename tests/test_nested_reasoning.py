# tests/test_nested_reasoning.py
"""A nested agent's emit_reasoning must bubble to the parent's stream() via the
active contextvars ReasoningChannel (contract §2.3 / §2.5)."""

from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Message
from aixon.reasoning import emit_reasoning
from aixon.registry import get_registry

from langchain_core.messages import AIMessage
from tests._fakes import FakeChatModel


def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_nested_worker_reasoning_bubbles_to_parent_stream(monkeypatch):
    # The nested "agent" is a tool whose body emits reasoning exactly as a
    # nested ToolAgent would (its steps call emit_reasoning against the active
    # channel set by the parent's stream()).
    def nested_worker(text: str) -> str:
        """A nested worker that reasons before answering."""
        emit_reasoning("nested: analysing the request")
        emit_reasoning("nested: producing an answer")
        return "nested-result"

    class ParentAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [nested_worker]

    parent = get_registry().resolve("parentagent")
    fake = FakeChatModel(
        script=[
            _tool_call("nested_worker", {"text": "go"}),
            AIMessage(content="Parent done."),
        ]
    )
    monkeypatch.setattr(type(parent.llm), "chat_model", property(lambda self: fake))

    chunks = list(parent.stream([Message(role="user", content="please work")]))
    reasoning_text = "".join(c.reasoning for c in chunks if c.reasoning)

    # The parent's own tool-call label AND the nested worker's two lines all
    # surfaced through the parent stream.
    assert "Calling nested_worker..." in reasoning_text
    assert "nested: analysing the request" in reasoning_text
    assert "nested: producing an answer" in reasoning_text
    assert any("Parent done." in c.content for c in chunks if c.content)
    assert chunks[-1].done is True


def test_nested_toolagent_as_tool_propagates_reasoning(monkeypatch):
    # End-to-end with a REAL nested ToolAgent wired via as_tool() + coerce_tools.
    # Each agent gets its OWN scripted FakeChatModel via per-instance _chat_model
    # (LLM.chat_model is a cached property over self._chat_model per contract §1.3),
    # so root and child run independent scripts.
    def leaf_tool(text: str) -> str:
        """leaf"""
        return "leaf:" + text

    class ChildAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [leaf_tool]

    child = get_registry().resolve("childagent")
    child_fake = FakeChatModel(
        script=[
            _tool_call("leaf_tool", {"text": "x"}),
            AIMessage(content="child answer"),
        ]
    )
    object.__setattr__(child.llm, "_chat_model", child_fake)

    class RootAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [child.as_tool()]

    root = get_registry().resolve("rootagent")
    root_fake = FakeChatModel(
        script=[
            _tool_call("childagent", {"text": "delegate"}),
            AIMessage(content="root answer"),
        ]
    )
    object.__setattr__(root.llm, "_chat_model", root_fake)

    chunks = list(root.stream([Message(role="user", content="do it")]))
    reasoning_text = "".join(c.reasoning for c in chunks if c.reasoning)

    # Root labelled its call to the child; the child labelled its call to leaf.
    assert "Calling childagent..." in reasoning_text
    assert "Calling leaf_tool..." in reasoning_text
    assert any("root answer" in c.content for c in chunks if c.content)
    assert chunks[-1].done is True
