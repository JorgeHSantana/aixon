"""Testes dos achados de auditoria cli-1..cli-5 (Task 6.9)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _event(content):
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    event = MagicMock()
    event.choices = [choice]
    return event


def test_cli_1(runner):
    """Ctrl+C no meio do stream remoto descarta o 'user' do turno: o próximo
    turno não manda dois 'user' seguidos."""
    from aixon.cli import app

    sent: list[list[dict]] = []

    def _interrupted():
        yield _event("parcial")
        raise KeyboardInterrupt

    def _create(*, model, messages, stream):
        sent.append([dict(m) for m in messages])
        return _interrupted() if len(sent) == 1 else iter([_event("ok")])

    client = MagicMock()
    client.models.list.return_value = MagicMock(data=[MagicMock(id="echoagent")])
    client.chat.completions.create.side_effect = _create
    with patch("aixon.cli.OpenAI", return_value=client):
        result = runner.invoke(
            app, ["chat", "--url", "http://x"],
            input="1\nprimeiro\nsegundo\n/exit\n", catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert len(sent) == 2
    assert [m["role"] for m in sent[1]] == ["user"]
    assert sent[1][0]["content"] == "segundo"


def _make_pkg(tmp_path, monkeypatch, name, body):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, key)


def test_cli_2(tmp_path, monkeypatch, capsys):
    """ValueError levantado DENTRO do módulo do usuário vira warning; só o
    'não é pacote' (NotAPackageError) é silenciado."""
    from aixon.cli import _autodiscover_quietly
    from aixon.discovery import NotAPackageError, autodiscover

    _make_pkg(tmp_path, monkeypatch, "cli2_pkg_bad", "raise ValueError('LLM inválido')\n")
    _autodiscover_quietly("cli2_pkg_bad")
    err = capsys.readouterr().err
    assert "Warning: error importing agents from 'cli2_pkg_bad'" in err
    assert "LLM inválido" in err

    (tmp_path / "cli2_plain.py").write_text("")
    with pytest.raises(NotAPackageError):
        autodiscover("cli2_plain")
    assert issubclass(NotAPackageError, ValueError)
    _autodiscover_quietly("cli2_plain")
    assert capsys.readouterr().err == ""


def test_cli_3(monkeypatch):
    """Escolha numérica do menu: '0' sai, faixa válida, inválido repete."""
    from aixon import cli

    answers = iter(["x", "5", "2"])
    monkeypatch.setattr(cli.click, "prompt", lambda *a, **k: next(answers))
    echoed: list[str] = []
    monkeypatch.setattr(cli.click, "echo", lambda msg="", **k: echoed.append(msg))
    assert cli._prompt_choice(3) == 1
    assert echoed.count("Invalid choice, try again.") == 2

    monkeypatch.setattr(cli.click, "prompt", lambda *a, **k: " 0 ")
    assert cli._prompt_choice(3) is None


def test_cli_4(capsys):
    """Descarte do turno falho: aceita Message e dict, só tira 'user'."""
    from aixon.cli import _discard_failed_turn
    from aixon.message import Message

    msgs = [Message(role="user", content="a")]
    _discard_failed_turn(msgs, RuntimeError("boom"))
    assert msgs == []
    assert "Error: boom" in capsys.readouterr().err

    dicts = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    _discard_failed_turn(dicts, RuntimeError("x"))
    assert len(dicts) == 2


def test_cli_5():
    """_stream_inprocess só renderiza: devolve (conteúdo, erro) e não mexe na
    lista de mensagens."""
    from aixon.agent import Agent
    from aixon.cli import _stream_inprocess
    from aixon.message import Chunk, Message

    class Cli5FailAgent(Agent):
        hidden = True

        def invoke(self, messages):
            raise AssertionError

        def stream(self, messages):
            yield Chunk(content="meio")
            raise RuntimeError("caiu")

    msgs = [Message(role="user", content="oi")]
    content, err = _stream_inprocess(Cli5FailAgent(), msgs)
    assert content is None and isinstance(err, RuntimeError)
    assert len(msgs) == 1
