# tests/test_server_sweep_6_8.py
"""Task 6.8 (varredura): achados server-5/6/7/9/10/11 de aixon/server/server.py.

Bugs (server-9, server-10) têm teste que falha antes da correção; as extrações
(server-5/6/7/11) têm teste de igualdade de comportamento — a prova principal é
a suíte de server existente passar sem edição."""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

import aixon.server.server as server_mod
from aixon.message import Message
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import make_echo, sse_content


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _chat(model="eco", stream=False):
    return {"model": model, "stream": stream,
            "messages": [{"role": "user", "content": "oi"}]}


# --- server-9: falha em session.done() é logada, não engolida em silêncio ---
class _DoneFails(OpenAIAdapter):
    def open_stream(self, *, model, request):
        session = super().open_stream(model=model, request=request)

        def done():
            raise RuntimeError("fechamento quebrou")

        session.done = done
        return session


def test_server_9_session_done_failure_is_logged(caplog):
    make_echo("eco")
    client = TestClient(Server(adapters=[_DoneFails()]).app)
    with caplog.at_level(logging.ERROR, logger="aixon.server"):
        r = client.post("/v1/chat/completions", json=_chat(stream=True))
    assert r.status_code == 200
    assert sse_content(r.text) == "echo:oi"  # o stream segue até o fim
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "aixon.server"]
    assert any("session.done() failed: fechamento quebrou" in m for m in msgs), msgs


# --- server-10: construção do singleton e do app sob trava -------------------
def _race(fn, n=8):
    barrier = threading.Barrier(n)
    out = []

    def run():
        barrier.wait()
        out.append(fn())

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_server_10_concurrent_construction_builds_one_instance(monkeypatch):
    built = []

    class SlowAdapter(OpenAIAdapter):
        def __init__(self, *a, **k):
            built.append(1)
            time.sleep(0.05)  # alarga a janela entre o check e o set
            super().__init__(*a, **k)

    monkeypatch.setattr(server_mod, "OpenAIAdapter", SlowAdapter)
    servers = _race(Server)
    assert len({id(s) for s in servers}) == 1
    assert len(built) == 1


def test_server_10_concurrent_app_access_builds_one_app(monkeypatch):
    server = Server()
    calls = []
    real = Server._build_app

    def slow(self):
        calls.append(1)
        time.sleep(0.05)
        return real(self)

    monkeypatch.setattr(Server, "_build_app", slow)
    apps = _race(lambda: server.app)
    assert len(calls) == 1
    assert len({id(a) for a in apps}) == 1


def test_server_10_reset_still_rebuilds():
    a = Server()
    Server._reset()
    assert Server() is not a


# --- server-11: usage_for ---------------------------------------------------
def test_server_11_usage_for_prefers_provider_usage():
    from aixon.usage import usage_for

    real = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    msg = Message(role="assistant", content="x", usage=real)
    assert usage_for("gpt-4o-mini", [], msg) == real


def test_server_11_usage_for_falls_back_to_the_same_estimate():
    from aixon.server.usage import build_usage
    from aixon.usage import usage_for

    msgs = [Message(role="system", content="s"), Message(role="user", content="u")]
    msg = Message(role="assistant", content="resposta", reasoning="porque")
    assert usage_for("gpt-4o-mini", msgs, msg) == build_usage(
        "gpt-4o-mini", "s\nu", "resposta\nporque")
    sem = Message(role="assistant", content="resposta")
    assert usage_for("gpt-4o-mini", msgs, sem) == build_usage(
        "gpt-4o-mini", "s\nu", "resposta")


# --- server-7: métodos extraídos testáveis sem subir o app -------------------
class _FakeRequest:
    def __init__(self, body):
        self._body = body
        self.headers = {}

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.mark.parametrize("body", [ValueError("not json"), [1, 2], "x"])
def test_server_7_parse_body_rejects_non_object(body):
    from starlette.responses import JSONResponse

    server = Server()
    out = asyncio.run(server._parse_body(_FakeRequest(body), OpenAIAdapter(),
                                         "/v1/chat/completions"))
    assert isinstance(out, JSONResponse) and out.status_code == 400


def test_server_7_resolve_agent_unknown_model_is_404():
    from starlette.responses import JSONResponse

    server = Server()
    pr = asyncio.run(server._parse_body(_FakeRequest(_chat("nada")),
                                        OpenAIAdapter(), "/v1/chat/completions"))
    out = server._resolve_agent(pr)
    assert isinstance(out, JSONResponse) and out.status_code == 404


def test_server_7_both_paths_same_answer_through_app():
    make_echo("eco")
    client = TestClient(Server().app)
    sync = client.post("/v1/chat/completions", json=_chat()).json()
    streamed = client.post("/v1/chat/completions", json=_chat(stream=True))
    assert sync["choices"][0]["message"]["content"] == "echo:oi"
    assert sse_content(streamed.text) == "echo:oi"
