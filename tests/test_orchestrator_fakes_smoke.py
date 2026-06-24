from _fakes import make_llm, make_echo_agent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry


def test_make_llm_returns_fake_handle():
    llm = make_llm()
    assert isinstance(llm, LLM)
    assert llm.model == "fake-1"
    assert llm._provider_name == "fake"


def test_make_echo_agent_registers_and_echoes_last_user():
    make_echo_agent("alpha")
    agent = get_registry().resolve("alpha")
    out = agent.invoke([Message(role="user", content="ping")])
    assert out.role == "assistant"
    assert "ping" in out.content  # last user content is echoed back


def test_make_echo_agent_stream_yields_content_then_done():
    make_echo_agent("beta")
    agent = get_registry().resolve("beta")
    chunks = list(agent.stream([Message(role="user", content="go")]))
    assert isinstance(chunks[-1], Chunk)
    assert chunks[-1].done is True
    assert any("go" in c.content for c in chunks)


def test_make_echo_agent_distinct_names_are_distinct_agents():
    make_echo_agent("one")
    make_echo_agent("two")
    assert get_registry().resolve("one").name == "one"
    assert get_registry().resolve("two").name == "two"


def test_make_echo_agent_hidden_flag():
    make_echo_agent("seen")
    make_echo_agent("unseen", hidden=True)
    public_names = {a.name for a in get_registry().public()}
    assert "seen" in public_names
    assert "unseen" not in public_names
