# tests/test_server_request_hook.py
"""0.1.29 — `Server(request_hook=...)`: o hook da aplicação decide, por
request, a imposição de modelo, a identidade (e-mail e trace) e a recusa."""
from __future__ import annotations

import contextlib
import logging
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import aixon.agent as agent_mod
import aixon.server.server as server_mod
from aixon import LLMAgent
from aixon.agent import Agent
from aixon.exceptions import AixonError, RequestRejected
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from aixon.runtime import _imposed_model, current_imposed_model
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.compat import open_webui_request_hook
from aixon.server.context import RequestDecision
from aixon.server.server import Server
from tests._fakes import make_llm, make_recording_provider
from tests._server_fakes import make_echo, make_whoami, sse_content, sse_datas

URL = "/v1/chat/completions"
ANTHROPIC_URL = "/anthropic/v1/messages"


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()
    left = current_imposed_model()
    if left is not None:
        _imposed_model.set(None)
        pytest.fail(f"o teste deixou o modelo imposto {left} no contexto")


def _client(hook=None, adapters=None):
    return TestClient(Server(adapters=adapters, request_hook=hook).app)


def _post(client, *, stream=False, model="a", headers=None):
    return client.post(URL, headers=headers or {}, json={
        "model": model, "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    })


class _ImposedProbe(Agent, abstract=True):
    """Responde com ``str(current_imposed_model())`` e grava o que viu."""

    seen: list = []

    def invoke(self, messages):
        type(self).seen.append(current_imposed_model())
        return Message(role="assistant", content=str(current_imposed_model()))

    def stream(self, messages):
        type(self).seen.append(current_imposed_model())
        yield Chunk(content=str(current_imposed_model()))
        yield Chunk(content=" mais")
        yield Chunk(done=True)


def _make_probe(name):
    cls = type("MadeProbeAgent", (_ImposedProbe,), {"name": name, "seen": []})
    return get_registry().resolve(cls.name)


@pytest.mark.parametrize("stream", [False, True])
def test_hook_chamado_uma_vez_por_request(stream):
    make_echo("a")
    calls = []
    client = _client(lambda ctx: calls.append(ctx))
    r = _post(client, stream=stream)
    assert r.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_hook_recebe_contexto_normalizado(stream):
    make_echo("a", aliases=["apelido"])
    seen = []
    client = _client(lambda ctx: seen.append(ctx))
    _post(client, stream=stream, model="apelido", headers={"X-Tier": "pro"})
    (ctx,) = seen
    assert ctx.model == "apelido"
    assert ctx.agent_name == "a"
    assert ctx.headers["x-tier"] == "pro"
    assert ctx.adapter == "openai"
    assert ctx.stream is stream
    with pytest.raises(TypeError):
        ctx.params["x"] = 1  # type: ignore[index]


@pytest.mark.parametrize("stream", [False, True])
def test_user_email_do_hook_chega_ao_agente(stream):
    make_whoami("a")
    client = _client(lambda ctx: RequestDecision(user_email="ana@x.com"))
    r = _post(client, stream=stream)
    text = sse_content(r.text) if stream else r.json()["choices"][0]["message"]["content"]
    assert text == "ana@x.com"


def test_hook_proprio_exclui_o_padrao():
    make_whoami("a")
    client = _client(lambda ctx: None)
    r = _post(client, headers={"x-openwebui-user-email": "ana@x.com"})
    assert r.json()["choices"][0]["message"]["content"] == "None"


def test_sem_hook_usa_o_padrao():
    make_whoami("a")
    server = Server()
    assert server.request_hook is open_webui_request_hook
    r = _post(TestClient(server.app), headers={"x-openwebui-user-email": "ana@x.com"})
    assert r.json()["choices"][0]["message"]["content"] == "ana@x.com"


@pytest.mark.parametrize("stream", [False, True])
def test_impose_abre_escopo_nos_dois_caminhos(stream):
    probe = _make_probe("a")
    client = _client(lambda ctx: RequestDecision(impose=("pro", "gateway")))
    r = _post(client, stream=stream)
    assert r.status_code == 200
    assert type(probe).seen == [("pro", "gateway")]


def test_impose_do_servidor_vence_o_da_classe(monkeypatch, rec):
    aberturas = []
    original = server_mod.imposed_model_scope

    def contando(*args, **kwargs):
        aberturas.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(server_mod, "imposed_model_scope", contando)
    monkeypatch.setattr(agent_mod, "imposed_model_scope", contando)

    class ClasseImpoeAgent(_ImposedProbe):
        name = "a"
        seen: list = []
        llm = LLM("tier-x", provider="rec")
        impose_model = True

    client = _client(lambda ctx: RequestDecision(impose=("pro", "rec")))
    r = _post(client)
    assert r.status_code == 200
    assert ClasseImpoeAgent.seen == [("pro", "rec")]
    assert aberturas == [(("pro", "rec"), {"by": "server"})]


def _rejeita(ctx):
    raise RequestRejected("Sem permissão.")


def test_rejeicao_sem_stream():
    agent = make_echo("a")
    r = _post(_client(_rejeita))
    assert r.status_code == 403
    assert r.json() == {"error": {"message": "Sem permissão.",
                                  "type": "request_rejected",
                                  "code": "request_rejected"}}
    assert agent.seen is None


def test_rejeicao_sem_stream_status_e_code_custom():
    make_echo("a")

    def hook(ctx):
        raise RequestRejected("x", status=401, code="no_email")

    r = _post(_client(hook))
    assert r.status_code == 401
    assert r.json() == {"error": {"message": "x", "type": "request_rejected",
                                  "code": "no_email"}}


def test_rejeicao_em_stream_e_ultima_mensagem():
    agent = make_echo("a")
    r = _post(_client(_rejeita), stream=True)
    assert r.status_code == 200
    assert sse_content(r.text) == "Sem permissão."
    datas = sse_datas(r.text)
    assert datas[-1] == "[DONE]"
    assert '"finish_reason": "stop"' in datas[-2] or '"finish_reason":"stop"' in datas[-2]
    assert agent.seen is None


def test_rejeicao_adapter_anthropic():
    agent = make_echo("a")
    client = _client(_rejeita, adapters=[OpenAIAdapter(),
                                         AnthropicAdapter(mount_prefix="/anthropic")])
    body = {"model": "a", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]}
    r = client.post(ANTHROPIC_URL, json=body)
    assert r.status_code == 403
    assert r.json()["error"]["type"] == "request_rejected"
    r = client.post(ANTHROPIC_URL, json={**body, "stream": True})
    assert r.status_code == 200
    assert "Sem permissão." in r.text
    assert "message_stop" in r.text
    assert agent.seen is None


@pytest.mark.parametrize("stream", [False, True])
def test_hook_que_levanta_vira_500_sem_texto(stream, caplog):
    agent = make_echo("a")

    def hook(ctx):
        raise ValueError("segredo-do-hook")

    with caplog.at_level(logging.ERROR):
        r = _post(_client(hook), stream=stream)
    assert r.status_code == 500
    assert r.json() == {"error": {"message": "The request hook failed.",
                                  "type": "server_error"}}
    assert "segredo-do-hook" not in caplog.text
    assert "ValueError" in caplog.text
    assert agent.seen is None


def test_trace_herda_email(monkeypatch):
    make_echo("a")
    capturado = []

    @contextlib.contextmanager
    def fake_observe(name, **kwargs):
        capturado.append(kwargs)
        yield

    monkeypatch.setattr(server_mod, "observe_request", fake_observe)
    client = _client(lambda ctx: RequestDecision(user_email="ana@x.com",
                                                 trace_session_id="s1"))
    _post(client)
    assert capturado[0]["user_id"] == "ana@x.com"
    assert capturado[0]["session_id"] == "s1"


def test_singleton_hook_fixo():
    def h(ctx):
        return None

    Server(request_hook=h)
    assert Server(request_hook=None).request_hook is h
    assert Server(request_hook=h).request_hook is h
    assert Server().request_hook is h
    with pytest.raises(AixonError, match="request_hook"):
        Server(request_hook=lambda ctx: None)


def test_impose_com_provedor_inexistente_vira_500():
    make_llm()

    class UsaLLMAgent(LLMAgent):
        name = "a"
        llm = LLM("fake-1", provider="fake")

    client = _client(lambda ctx: RequestDecision(impose=("x", "provedor-que-nao-existe")))
    r = _post(client)
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "server_error"
    # o servidor segue de pé
    assert client.get("/health").status_code == 200


def test_escopo_imposto_nao_vaza_entre_requests_stream():
    probe = _make_probe("a")
    contador = []

    def hook(ctx):
        contador.append(1)
        if len(contador) == 1:
            return RequestDecision(impose=("pro", "gateway"))
        return None

    client = _client(hook)
    body = {"model": "a", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", URL, json=body) as r:
        for _ in r.iter_lines():
            break  # cliente desconecta no meio do stream
    r2 = _post(client, stream=True)
    assert r2.status_code == 200
    assert type(probe).seen == [("pro", "gateway"), None]
    assert current_imposed_model() is None


def test_cli_serve_mantem_hook_registrado_antes():
    def h(ctx):
        return None

    def fake_discover(pkg):
        # o main.py da aplicação, importado pelo autodiscover
        Server(request_hook=h)

    with (
        patch("aixon.cli.autodiscover", side_effect=fake_discover),
        patch.object(Server, "serve", lambda self, **kw: None),
    ):
        from aixon.cli import app
        result = CliRunner().invoke(app, ["serve"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert Server.get_instance().request_hook is h
