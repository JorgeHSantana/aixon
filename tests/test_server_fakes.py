from __future__ import annotations

from aixon.message import Chunk, Message
from aixon.registry import get_registry
from tests._server_fakes import EchoAgent, ReasoningAgent, make_echo


def test_make_echo_registers_and_resolves():
    inst = make_echo("alpha", aliases=["a1"], description="d")
    assert get_registry().resolve("alpha") is inst
    assert get_registry().resolve("a1") is inst
    assert inst.description == "d"


def test_echo_invoke_echoes_last_user_and_records_messages():
    inst = make_echo("alpha")
    msgs = [Message(role="system", content="s"), Message(role="user", content="hi")]
    out = inst.invoke(msgs)
    assert isinstance(out, Message)
    assert out.role == "assistant"
    assert out.content == "echo:hi"
    # It recorded exactly the neutral list it was handed.
    assert inst.seen is msgs
    assert all(isinstance(m, Message) for m in inst.seen)


def test_echo_stream_yields_content_then_done():
    inst = make_echo("alpha")
    chunks = list(inst.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[-1].done is True
    text = "".join(c.content for c in chunks if not c.done)
    assert text == "echo:hi"


def test_reasoning_agent_emits_reasoning():
    # NOTE: concrete Agent subclasses MUST end with the "Agent" suffix
    # (aixon/agent.py enforces _suffix="Agent"); a bare name like `_R` raises
    # NamingError at class-definition time. Hence `_RAgent`.
    class _RAgent(ReasoningAgent):
        name = "thinker"

    inst = get_registry().resolve("thinker")
    chunks = list(inst.stream([Message(role="user", content="hi")]))
    assert any(c.reasoning for c in chunks)
    assert chunks[-1].done is True
    assert inst.invoke([Message(role="user", content="hi")]).reasoning
