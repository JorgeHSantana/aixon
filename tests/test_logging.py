import logging
import pytest

from aixon.logging import Logger


def test_logger_respects_log_level_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    log = Logger("aixon.test.level")
    assert log._logger.level == logging.DEBUG


def test_logger_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    log = Logger("aixon.test.default")
    assert log._logger.level == logging.INFO


def test_logger_does_not_duplicate_handlers():
    Logger("aixon.test.dup")
    Logger("aixon.test.dup")
    assert len(logging.getLogger("aixon.test.dup").handlers) == 1


def test_registering_agent_logs_info(caplog):
    from aixon.agent import Agent
    from aixon.message import Message, Chunk

    with caplog.at_level(logging.INFO, logger="aixon.registry"):
        type(
            "LoggedAgent",
            (Agent,),
            {
                "invoke": lambda self, m: Message(role="assistant"),
                "stream": lambda self, m: iter([Chunk(done=True)]),
            },
        )
    assert any("loggedagent" in r.message for r in caplog.records)


def test_autodiscover_logs(monkeypatch, tmp_path, caplog):
    import sys

    name = "logpkg_agents"
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            del sys.modules[mod]
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text(
        "from aixon.agent import Agent\n"
        "from aixon.message import Message, Chunk\n"
        "class AlphaAgent(Agent):\n"
        "    def invoke(self, messages): return Message(role='assistant')\n"
        "    def stream(self, messages): return iter([Chunk(done=True)])\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from aixon.discovery import autodiscover

    with caplog.at_level(logging.INFO, logger="aixon.discovery"):
        autodiscover(name)
    assert any(name in r.message for r in caplog.records)
