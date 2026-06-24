from aixon.registry import get_registry


def test_make_cli_echo_agent_registers_and_streams():
    from tests._cli_fakes import make_cli_echo_agent
    Cls = make_cli_echo_agent("EchoAgent", description="test agent")
    agent = get_registry().resolve("echoagent")
    assert agent.description == "test agent"
    chunks = list(agent.stream([]))
    assert chunks[-1].done is True


def test_make_cli_echo_agent_reflects_user_message():
    from tests._cli_fakes import make_cli_echo_agent
    from aixon.message import Message
    Cls = make_cli_echo_agent("EchoAgent")
    agent = get_registry().resolve("echoagent")
    chunks = list(agent.stream([Message(role="user", content="hello")]))
    content_chunks = [c for c in chunks if c.content]
    assert any("hello" in c.content for c in content_chunks)


def test_make_cli_echo_agent_with_reasoning():
    from tests._cli_fakes import make_cli_echo_agent
    Cls = make_cli_echo_agent("EchoAgent", reasoning="thinking...")
    agent = get_registry().resolve("echoagent")
    chunks = list(agent.stream([]))
    assert any(c.reasoning == "thinking..." for c in chunks)


def test_hidden_agent_not_in_public():
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("PublicAgent", hidden=False)
    make_cli_echo_agent("HiddenAgent", hidden=True)
    public_names = [a.name for a in get_registry().public()]
    assert "publicagent" in public_names
    assert "hiddenagent" not in public_names
