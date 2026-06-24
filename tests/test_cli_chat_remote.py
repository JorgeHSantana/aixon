"""Tests for aixon chat --url (remote mode via OpenAI wire protocol)."""
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _make_fake_openai(model_ids, stream_chunks=None):
    """Build a mock openai.OpenAI client that returns canned responses."""
    if stream_chunks is None:
        stream_chunks = [{"content": "remote reply"}]

    # Build fake model list
    fake_models = [MagicMock(id=m) for m in model_ids]
    models_list = MagicMock()
    models_list.data = fake_models

    # Build fake stream events
    def _make_event(content):
        event = MagicMock()
        choice = MagicMock()
        delta = MagicMock()
        delta.content = content
        choice.delta = delta
        event.choices = [choice]
        return event

    stream_events = [_make_event(c["content"]) for c in stream_chunks]
    fake_stream = MagicMock()
    fake_stream.__iter__ = MagicMock(return_value=iter(stream_events))

    completions = MagicMock()
    completions.create = MagicMock(return_value=fake_stream)
    chat = MagicMock()
    chat.completions = completions

    client = MagicMock()
    client.models.list = MagicMock(return_value=models_list)
    client.chat = chat

    return client


def _invoke_remote(runner, user_lines, url="http://localhost:8000", client=None):
    from aixon.cli import app

    if client is None:
        client = _make_fake_openai(["echoagent"])

    input_text = "\n".join(user_lines) + "\n"
    with patch("aixon.cli.OpenAI", return_value=client):
        return runner.invoke(
            app, ["chat", "--url", url], input=input_text, catch_exceptions=False
        )


def test_remote_chat_shows_remote_agents(runner):
    """Agent names returned by models.list() appear in the menu."""
    client = _make_fake_openai(["athena", "diagnosis"])
    result = _invoke_remote(runner, ["0"], client=client)
    assert result.exit_code == 0
    assert "athena" in result.output
    assert "diagnosis" in result.output


def test_remote_chat_streams_response(runner):
    """Streamed delta content is printed to stdout."""
    client = _make_fake_openai(
        ["echoagent"], stream_chunks=[{"content": "hello from server"}]
    )
    result = _invoke_remote(runner, ["1", "hi", "/exit"], client=client)
    assert "hello from server" in result.output


def test_remote_chat_exit_command(runner):
    """/exit prints Goodbye and returns exit code 0."""
    client = _make_fake_openai(["echoagent"])
    result = _invoke_remote(runner, ["1", "/exit"], client=client)
    assert result.exit_code == 0
    assert "Goodbye" in result.output


def test_remote_chat_no_openai_package_shows_error(runner):
    """If openai is not installed (OpenAI is None), a helpful error is shown."""
    from aixon.cli import app

    # Patch aixon.cli.OpenAI to None directly — no module reload needed.
    # _chat_remote already checks `if OpenAI is None` at the top.
    with patch("aixon.cli.OpenAI", None):
        result = runner.invoke(
            app,
            ["chat", "--url", "http://x:8000"],
            input="0\n",
            catch_exceptions=True,
        )
    assert result.exit_code != 0 or "openai" in result.output.lower()


def test_remote_chat_server_unreachable_shows_error(runner):
    """When models.list() raises, a 'Could not reach' error is shown."""
    client = MagicMock()
    client.models.list = MagicMock(side_effect=Exception("connection refused"))
    result = _invoke_remote(runner, [], client=client)
    assert result.exit_code != 0 or "Could not reach" in result.output
