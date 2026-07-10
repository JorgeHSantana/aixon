import pytest
from unittest.mock import patch
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _invoke_chat(runner, user_lines, *, url=None):
    """Helper: invoke 'aixon chat' with piped input lines."""
    from aixon.cli import app
    args = ["chat"]
    if url:
        args += ["--url", url]
    input_text = "\n".join(user_lines) + "\n"
    with patch("aixon.cli.autodiscover"):
        return runner.invoke(app, args, input=input_text, catch_exceptions=False)


def test_chat_menu_shows_public_agents(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("MenuAgent", description="Does menus")

    result = _invoke_chat(runner, ["0"])  # pick 0 = exit immediately
    assert "menuagent" in result.output
    assert "Does menus" in result.output


def test_chat_menu_excludes_hidden_agents(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("VisTenAgent", description="Visible")
    make_cli_echo_agent("HideTenAgent", description="Hidden", hidden=True)

    result = _invoke_chat(runner, ["0"])
    assert "vistenagent" in result.output
    assert "hidetenagent" not in result.output


def test_chat_streams_content_for_user_message(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("TalkerAgent", description="Talks")

    # Pick agent 1, send "hello", then /exit
    result = _invoke_chat(runner, ["1", "hello", "/exit"])
    assert "echo: hello" in result.output


def test_chat_exit_command_exits_cleanly(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("ExitAgent", description="Exit me")

    result = _invoke_chat(runner, ["1", "/exit"])
    assert result.exit_code == 0
    assert "Goodbye" in result.output


def test_chat_menu_command_reprints_menu(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("ReprintAgent", description="Reprint me")

    # Pick agent 1, send /menu, pick agent 1 again, then /exit
    result = _invoke_chat(runner, ["1", "/menu", "1", "/exit"])
    assert result.exit_code == 0
    # Menu must have appeared at least twice
    assert result.output.count("reprintagent") >= 2


def test_chat_no_agents_prints_message(runner):
    result = _invoke_chat(runner, [])
    assert result.exit_code == 0
    assert "No agents registered" in result.output


def test_chat_empty_input_ignored(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("BlankAgent", description="Blank test")

    # Empty line, then /exit
    result = _invoke_chat(runner, ["1", "", "/exit"])
    assert result.exit_code == 0


def test_empty_successful_turn_preserves_history_alternation(runner):
    """A turn that streams zero content chunks (but doesn't error) must still
    append an assistant message (even if empty) — otherwise the NEXT user
    message becomes a second consecutive 'user' entry in the history sent to
    the agent, breaking user/assistant alternation."""
    from aixon.agent import Agent
    from aixon.message import Chunk, Message

    calls: list[list[Message]] = []

    def _invoke(self, messages):
        return Message(role="assistant", content="")

    def _stream(self, messages):
        calls.append(list(messages))
        if len(calls) == 1:
            yield Chunk(done=True)  # zero content chunks: empty SUCCESS
        else:
            yield Chunk(content="second reply")
            yield Chunk(done=True)

    type(
        "EmptyTurnAgent",
        (Agent,),
        {"invoke": _invoke, "stream": _stream},
    )

    result = _invoke_chat(runner, ["1", "first", "second", "/exit"])
    assert result.exit_code == 0

    assert len(calls) == 2
    second_call_roles = [m.role for m in calls[1]]
    assert second_call_roles == ["user", "assistant", "user"]
    assert calls[1][0].content == "first"
    assert calls[1][1].content == ""
    assert calls[1][2].content == "second"


def test_chat_streams_reasoning_before_content(runner):
    from tests._cli_fakes import make_cli_echo_agent
    from aixon.message import Chunk
    make_cli_echo_agent(
        "ReasonAgent",
        reasoning="I am thinking",
        chunks=[
            Chunk(content="", reasoning="I am thinking"),
            Chunk(content="result"),
            Chunk(done=True),
        ],
    )

    result = _invoke_chat(runner, ["1", "prompt", "/exit"])
    # Reasoning appears in output (dim codes stripped in CliRunner non-tty)
    assert "I am thinking" in result.output
    assert "result" in result.output
    # Reasoning must appear before content
    idx_reasoning = result.output.index("I am thinking")
    idx_content = result.output.index("result")
    assert idx_reasoning < idx_content
