"""Integration smoke tests for the CLI.

Exercises the entire CLI surface end-to-end through CliRunner,
verifying that the `app` group is importable, all four subcommands exist,
and `--help` for each returns exit 0.
"""
import pytest
from click.testing import CliRunner

from aixon.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def test_app_help(runner):
    """Test that `aixon --help` shows all four subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "new" in result.output
    assert "serve" in result.output
    assert "list" in result.output


def test_chat_help(runner):
    """Test that `aixon chat --help` shows expected options."""
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--url" in result.output
    assert "--package" in result.output


def test_new_help(runner):
    """Test that `aixon new --help` shows expected arguments."""
    result = runner.invoke(app, ["new", "--help"])
    assert result.exit_code == 0
    assert "NAME" in result.output


def test_serve_help(runner):
    """Test that `aixon serve --help` shows expected options."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--package" in result.output


def test_list_help(runner):
    """Test that `aixon list --help` shows expected options."""
    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    assert "--package" in result.output


def test_cli_entry_point_is_click_group():
    """The 'aixon.cli:app' entry point must be a click.Group, not the old stub."""
    import click

    assert isinstance(app, click.Group)


def test_pyproject_cli_extra_includes_openai():
    """cli extra must declare openai>=1.0 so remote mode is installable."""
    import os
    import tomllib

    pyproject_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
    )
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    cli_deps = data["project"]["optional-dependencies"]["cli"]
    assert any("openai" in dep for dep in cli_deps)


def test_pyproject_click_is_core_dependency():
    """click must be a CORE dependency, not in the 'cli' extra (bug-sweep I2):
    the 'aixon' console-script is installed unconditionally by [project.scripts],
    so a bare install without the 'cli' extra would traceback on `aixon --help`
    if click were optional."""
    import os
    import tomllib

    pyproject_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
    )
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    core_deps = data["project"]["dependencies"]
    assert any("click" in dep for dep in core_deps)
    cli_deps = data["project"]["optional-dependencies"]["cli"]
    assert not any("click" in dep for dep in cli_deps)
