"""Tests for the Task 4 bug-sweep findings (I1-I8): CLI, logging, packaging,
and the aixon/__init__.py server guard."""
from __future__ import annotations

import logging
import pathlib
import tomllib

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# I1 — `aixon new` scaffold must declare a hatchling wheel build target.
# ---------------------------------------------------------------------------
def test_new_scaffold_pyproject_declares_wheel_target(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app

        result = runner.invoke(app, ["new", "my-project"])
        assert result.exit_code == 0
        content = open("my-project/pyproject.toml").read()
        assert "[tool.hatch.build.targets.wheel]" in content
        parsed = tomllib.loads(content)
        assert parsed["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["agents"]


# ---------------------------------------------------------------------------
# I2/I3 — packaging: click is core, dev extra can run the suite.
# ---------------------------------------------------------------------------
def _pyproject() -> dict:
    root = pathlib.Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text())


def _dep_names(deps: list[str]) -> set[str]:
    import re

    return {re.split(r"[><=~!\[ ]", d, maxsplit=1)[0] for d in deps}


def test_click_is_core_dependency():
    data = _pyproject()
    core_names = _dep_names(data["project"]["dependencies"])
    assert "click" in core_names
    cli_extra_names = _dep_names(data["project"]["optional-dependencies"]["cli"])
    assert "click" not in cli_extra_names


def test_dev_extra_can_run_the_test_suite():
    data = _pyproject()
    dev_names = _dep_names(data["project"]["optional-dependencies"]["dev"])
    assert {"pytest", "pytest-cov", "httpx2", "fastapi", "openai"} <= dev_names


# ---------------------------------------------------------------------------
# I4 — logging: invalid LOG_LEVEL falls back safely, handler level follows
# env across instances, no propagation to root (no duplicate lines).
# ---------------------------------------------------------------------------
def test_logger_invalid_level_falls_back_to_info(monkeypatch):
    from aixon.logging import Logger

    monkeypatch.setenv("LOG_LEVEL", "basic_format")
    log = Logger("aixon.test.bugsweep.invalid")
    assert log._logger.level == logging.INFO


def test_logger_handler_level_follows_env(monkeypatch):
    from aixon.logging import Logger

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    log = Logger("aixon.test.bugsweep.followenv")
    assert log._logger.handlers[0].level == logging.DEBUG

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    log2 = Logger("aixon.test.bugsweep.followenv")
    assert log2._logger.handlers[0].level == logging.WARNING


def test_logger_does_not_propagate_to_root():
    from aixon.logging import Logger

    log = Logger("aixon.test.bugsweep.propagate")
    assert log._logger.propagate is False


# ---------------------------------------------------------------------------
# I5 — server symbols still exported after removing the dead try/except.
# ---------------------------------------------------------------------------
def test_server_symbols_exported():
    import aixon

    assert "Server" in aixon.__all__
    assert "ProtocolAdapter" in aixon.__all__
    assert "OpenAIAdapter" in aixon.__all__
    assert "AnthropicAdapter" in aixon.__all__
    assert "ParsedRequest" in aixon.__all__
    assert aixon.Server is not None


# ---------------------------------------------------------------------------
# I6 — `serve` surfaces the real ImportError from inside a user's agent
# module (via _autodiscover_quietly), not a generic warning.
# ---------------------------------------------------------------------------
def test_serve_surfaces_real_import_error_from_agent_module(runner, tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock, patch

    name = "bugsweep_serve_agents"
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            del sys.modules[mod]
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("import totally_nonexistent_lib_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    server = MagicMock()
    server.serve = MagicMock()
    server_cls = MagicMock()
    server_cls.return_value = server

    with patch.dict("sys.modules", {"aixon.server.server": MagicMock(Server=server_cls)}):
        from aixon.cli import app

        result = runner.invoke(
            app, ["serve", "--package", name], catch_exceptions=False
        )

    assert "totally_nonexistent_lib_xyz" in result.output
    assert "could not autodiscover package" not in result.output.lower()


# ---------------------------------------------------------------------------
# I7 — reasoning goes to stderr (not stdout) when stdout is not a TTY, so it
# is not interleaved with the streamed content when the caller captures stdout.
# ---------------------------------------------------------------------------
def test_chat_reasoning_goes_to_stderr_when_not_tty(runner):
    from tests._cli_fakes import make_cli_echo_agent
    from aixon.message import Chunk
    from unittest.mock import patch

    make_cli_echo_agent(
        "StderrReasonAgent",
        chunks=[
            Chunk(content="", reasoning="secret reasoning"),
            Chunk(content="visible result"),
            Chunk(done=True),
        ],
    )

    from aixon.cli import app

    with patch("aixon.cli.autodiscover"):
        result = runner.invoke(
            app, ["chat"], input="1\nprompt\n/exit\n", catch_exceptions=False
        )

    assert "secret reasoning" not in result.stdout
    assert "secret reasoning" in result.stderr
    assert "visible result" in result.stdout


# ---------------------------------------------------------------------------
# I8 — a turn exception must not leave an orphan `user` message in history
# (both chat paths).
# ---------------------------------------------------------------------------
def test_chat_error_removes_orphan_user_message(runner):
    from aixon.agent import Agent
    from aixon.message import Message, Chunk
    from unittest.mock import patch

    received_histories = []
    call_count = {"n": 0}

    def _stream(self, messages):
        received_histories.append(list(messages))
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        yield Chunk(content="ok now")
        yield Chunk(done=True)

    cls = type(
        "OrphanFixAgent",
        (Agent,),
        {
            "invoke": lambda self, m: Message(role="assistant", content="x"),
            "stream": _stream,
        },
    )

    from aixon.cli import app

    with patch("aixon.cli.autodiscover"):
        result = runner.invoke(
            app,
            ["chat"],
            input="1\nfirst\nsecond\n/exit\n",
            catch_exceptions=False,
        )

    assert len(received_histories) == 2
    second_call_messages = received_histories[1]
    roles = [m.role for m in second_call_messages]
    # No two consecutive 'user' roles: the orphaned first 'user' message
    # (whose turn raised) must have been popped before the retry.
    for i in range(len(roles) - 1):
        assert not (roles[i] == "user" and roles[i + 1] == "user"), roles


def test_chat_error_after_partial_content_appends_no_assistant_message(runner):
    """When a turn raises AFTER partial content was streamed, neither the
    popped user turn nor an assistant message built from the truncated
    output may survive in history — the next call must see only the new
    user message (plus prior successful turns, none here)."""
    from aixon.agent import Agent
    from aixon.message import Message, Chunk
    from unittest.mock import patch

    received_histories = []
    call_count = {"n": 0}

    def _stream(self, messages):
        received_histories.append(list(messages))
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield Chunk(content="partial")
            raise RuntimeError("boom mid-stream")
        yield Chunk(content="ok now")
        yield Chunk(done=True)

    type(
        "PartialErrorAgent",
        (Agent,),
        {
            "invoke": lambda self, m: Message(role="assistant", content="x"),
            "stream": _stream,
        },
    )

    from aixon.cli import app

    with patch("aixon.cli.autodiscover"):
        runner.invoke(
            app,
            ["chat"],
            input="1\nfirst\nsecond\n/exit\n",
            catch_exceptions=False,
        )

    assert len(received_histories) == 2
    second_call_messages = received_histories[1]
    # Only the second user message: the errored turn's user message was
    # popped AND no bogus assistant("partial") entry was appended.
    assert [(m.role, m.content) for m in second_call_messages] == [
        ("user", "second")
    ]
