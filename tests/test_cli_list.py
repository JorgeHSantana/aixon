import pytest
from click.testing import CliRunner
from unittest.mock import patch


@pytest.fixture
def runner():
    return CliRunner()


def test_list_with_no_agents_prints_message(runner):
    with patch("aixon.cli.autodiscover", side_effect=ImportError("no such package")):
        from aixon.cli import app
        result = runner.invoke(app, ["list", "--package", "nonexistent"])
    assert result.exit_code == 0
    assert "No agents registered" in result.output


def test_list_shows_public_agents(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("ListOneAgent", description="Lists things")

    with patch("aixon.cli.autodiscover"):
        from aixon.cli import app
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "listoneagent" in result.output
    assert "Lists things" in result.output


def test_list_excludes_hidden_agents(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("VisibleTwoAgent", description="Visible")
    make_cli_echo_agent("HiddenThreeAgent", description="Secret", hidden=True)

    with patch("aixon.cli.autodiscover"):
        from aixon.cli import app
        result = runner.invoke(app, ["list"])
    assert "visibletwoagent" in result.output
    assert "hiddenthreeagent" not in result.output


def test_list_shows_agent_type(runner):
    from tests._cli_fakes import make_cli_echo_agent
    make_cli_echo_agent("TypeFourAgent", description="Has a type")

    with patch("aixon.cli.autodiscover"):
        from aixon.cli import app
        result = runner.invoke(app, ["list"])
    # Type shown in brackets, e.g. [Agent]
    assert "[" in result.output and "]" in result.output


def test_list_default_package_is_agents(runner):
    """--package defaults to 'agents'; autodiscover is called with 'agents'."""
    calls = []

    def fake_discover(pkg):
        calls.append(pkg)

    with patch("aixon.cli.autodiscover", side_effect=fake_discover):
        from aixon.cli import app
        runner.invoke(app, ["list"])
    assert calls == ["agents"]
