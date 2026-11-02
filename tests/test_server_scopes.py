# tests/test_server_scopes.py
"""A pilha de escopos por request é a mesma, na mesma ordem, nos caminhos
stream e não stream, e fecha inteira quando o agente levanta."""
from __future__ import annotations

import contextlib

import pytest
from fastapi.testclient import TestClient

import aixon.server.server as server_mod
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import make_echo, make_failing

NOMES = [
    "observe_request", "generation_params", "user_email_scope",
    "client_tools_scope", "tool_choice_scope", "tool_call_cache",
]


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


@pytest.fixture
def eventos(monkeypatch):
    """Troca os seis context managers por fakes que anotam entrada e saída."""
    log: list[tuple[str, str]] = []

    def fake(nome):
        @contextlib.contextmanager
        def cm(*args, **kwargs):
            log.append(("enter", nome))
            try:
                yield
            finally:
                log.append(("exit", nome))
        return cm

    for nome in NOMES:
        monkeypatch.setattr(server_mod, nome, fake(nome))
    return log


def _post(client, stream):
    return client.post("/v1/chat/completions", json={
        "model": "a", "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    })


ESPERADO = [("enter", n) for n in NOMES] + [("exit", n) for n in reversed(NOMES)]


def test_ordem_dos_escopos_e_a_mesma_nos_dois_caminhos(eventos):
    make_echo("a")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)

    assert _post(client, stream=True).status_code == 200
    stream = list(eventos)
    eventos.clear()
    assert _post(client, stream=False).status_code == 200
    nao_stream = list(eventos)

    assert stream == nao_stream == ESPERADO


def test_excecao_no_agente_fecha_todos_os_escopos(eventos):
    make_failing("a", RuntimeError("boom"))
    client = TestClient(
        Server(adapters=[OpenAIAdapter()]).app, raise_server_exceptions=False,
    )

    # Stream: a exceção vira evento de erro no SSE, a resposta segue 200.
    assert _post(client, stream=True).status_code == 200
    assert [e for e in eventos if e[0] == "exit"] == [
        ("exit", n) for n in reversed(NOMES)
    ]
    eventos.clear()

    assert _post(client, stream=False).status_code == 500
    assert [e for e in eventos if e[0] == "exit"] == [
        ("exit", n) for n in reversed(NOMES)
    ]
