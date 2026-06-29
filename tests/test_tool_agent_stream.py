# tests/test_tool_agent_stream.py
from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry

from langchain_core.messages import AIMessage
from tests._fakes import FakeChatModel


def _install(monkeypatch, llm, script):
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_stream_yields_reasoning_then_content_then_done(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class StreamAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("streamagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 2, "b": 2}),
            AIMessage(content="The total is 4."),
        ],
    )

    chunks = list(agent.stream([Message(role="user", content="add 2 and 2")]))

    # All chunks are neutral Chunks.
    assert all(isinstance(c, Chunk) for c in chunks)
    # A reasoning chunk mentions the tool.
    assert any("adder" in c.reasoning for c in chunks if c.reasoning)
    # A content chunk carries the final answer.
    assert any("The total is 4." in c.content for c in chunks if c.content)
    # The stream terminates with done=True.
    assert chunks[-1].done is True


def test_stream_flattens_structured_list_content(monkeypatch):
    # Providers like Gemini 2.5 return AIMessage.content as a list of content
    # blocks. The streamed Chunk.content must be the flattened plain text (str),
    # not the raw list (which breaks SSE serialization downstream).
    def noop(text: str) -> str:
        """noop"""
        return text

    class ListContentAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [noop]

    agent = get_registry().resolve("listcontentagent")
    _install(
        monkeypatch,
        agent.llm,
        [AIMessage(content=[{"type": "text", "text": "Resposta final."}])],
    )

    chunks = list(agent.stream([Message(role="user", content="oi")]))

    content_chunks = [c.content for c in chunks if c.content]
    assert content_chunks, "expected at least one content chunk"
    for c in content_chunks:
        assert isinstance(c, str)
    assert any("Resposta final." in c for c in content_chunks)
    assert chunks[-1].done is True


def test_default_tool_call_label(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class DefaultLabelAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("defaultlabelagent")
    _install(monkeypatch, agent.llm, [_tool_call("adder", {"a": 1, "b": 1}), AIMessage(content="2")])

    reasoning = "".join(c.reasoning for c in agent.stream([Message(role="user", content="add")]) if c.reasoning)
    assert "Calling adder..." in reasoning


def test_custom_tool_call_label_is_used(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class CustomLabelAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]
        tool_call_label = "Chamando {name} 🔧"

    agent = get_registry().resolve("customlabelagent")
    _install(monkeypatch, agent.llm, [_tool_call("adder", {"a": 1, "b": 1}), AIMessage(content="2")])

    reasoning = "".join(c.reasoning for c in agent.stream([Message(role="user", content="add")]) if c.reasoning)
    assert "Chamando adder 🔧" in reasoning
    assert "Calling adder..." not in reasoning  # default phrase fully replaced


def test_stream_no_tool_still_streams_content_and_done(monkeypatch):
    def noop(text: str) -> str:
        """noop"""
        return text

    class DirectAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [noop]

    agent = get_registry().resolve("directagent")
    _install(monkeypatch, agent.llm, [AIMessage(content="immediate answer")])

    chunks = list(agent.stream([Message(role="user", content="hi")]))

    assert any("immediate answer" in c.content for c in chunks if c.content)
    assert chunks[-1].done is True
