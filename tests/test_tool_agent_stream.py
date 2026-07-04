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


def test_consecutive_duplicate_labels_are_deduped(monkeypatch):
    # A run that calls the same tool repeatedly (or any tools sharing one
    # label) must not spam the reasoning stream with identical consecutive
    # lines: 8x "Calling schema_search..." reads as noise (or a hang) in
    # chat UIs that render reasoning. Only the first of a consecutive run
    # of identical labels is emitted.
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class RepeatedCallsAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]
        tool_call_label = "Consultando o banco..."  # no {name}: same label every call

    agent = get_registry().resolve("repeatedcallsagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            # two calls in one AI message + one more in the next round: three
            # identical labels, across a drain boundary in stream().
            AIMessage(content="", tool_calls=[
                {"name": "adder", "args": {"a": 1, "b": 1}, "id": "c1"},
                {"name": "adder", "args": {"a": 2, "b": 2}, "id": "c2"},
            ]),
            _tool_call("adder", {"a": 3, "b": 3}, id="c3"),
            AIMessage(content="done"),
        ],
    )

    reasoning = [c.reasoning for c in agent.stream([Message(role="user", content="add")]) if c.reasoning]
    joined = "".join(reasoning)
    assert joined.count("Consultando o banco...") == 1


def test_distinct_labels_are_all_emitted(monkeypatch):
    # Dedupe is only for CONSECUTIVE duplicates: distinct labels (default
    # template interpolates the tool name) must all come through.
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def multiplier(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    class TwoToolsAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder, multiplier]

    agent = get_registry().resolve("twotoolsagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 1, "b": 1}, id="c1"),
            _tool_call("multiplier", {"a": 2, "b": 2}, id="c2"),
            AIMessage(content="done"),
        ],
    )

    reasoning = "".join(c.reasoning for c in agent.stream([Message(role="user", content="go")]) if c.reasoning)
    assert "Calling adder..." in reasoning
    assert "Calling multiplier..." in reasoning
