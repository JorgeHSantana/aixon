# tests/test_observability.py
"""#26 — optional Langfuse observability. Everything here is offline: the
langfuse-SDK tests point LANGFUSE_HOST at an unreachable address and disable
the per-request flush, so nothing ever leaves the process; the propagation
tests use a dummy langchain-core handler and no langfuse at all."""
from __future__ import annotations

import asyncio
import contextlib

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage

import aixon.observability as obs
from aixon.message import Message
from tests._fakes import make_llm


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Reset the lazy singletons/latches so each test decides its own env."""
    monkeypatch.setattr(obs, "_handler", None)
    monkeypatch.setattr(obs, "_disabled", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    yield


class RecordingHandler(BaseCallbackHandler):
    """Counts the LangChain callback events that reach it."""

    def __init__(self):
        self.chat_model_starts = 0
        self.chain_starts = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.chat_model_starts += 1

    def on_chain_start(self, *args, **kwargs):
        self.chain_starts += 1


# ── desligado por padrão ────────────────────────────────────────────────────

def test_disabled_without_envs_yields_false():
    with obs.observe_request("Analista") as active:
        assert active is False
        assert obs._handler_var.get() is None


def test_langfuse_enabled_requires_both_keys(monkeypatch):
    assert obs.langfuse_enabled() is False
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    assert obs.langfuse_enabled() is False
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert obs.langfuse_enabled() is True


def test_missing_dependency_disables_stickily(monkeypatch):
    """Envs presentes mas import do langfuse falhando: um warning, no-op, e o
    latch _disabled corta o custo das próximas requests."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError("langfuse não instalado")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with obs.observe_request("Analista") as active:
        assert active is False
    assert obs._disabled is True
    # segunda chamada nem tenta importar (latch)
    monkeypatch.setattr(builtins, "__import__", real_import)
    with obs.observe_request("Analista") as active:
        assert active is False


# ── identidade do trace a partir dos headers ────────────────────────────────

def test_request_identity_from_openwebui_headers():
    from starlette.datastructures import Headers

    headers = Headers({
        "X-OpenWebUI-User-Email": "jorge@zeus.com",
        "X-OpenWebUI-User-Id": "u-1",
        "X-OpenWebUI-Chat-Id": "chat-42",
    })
    ident = obs.request_identity(headers)
    assert ident == {"user_id": "jorge@zeus.com", "session_id": "chat-42"}


def test_request_identity_falls_back_to_user_id_and_none():
    from starlette.datastructures import Headers

    ident = obs.request_identity(Headers({"X-OpenWebUI-User-Id": "u-1"}))
    assert ident == {"user_id": "u-1", "session_id": None}
    assert obs.request_identity(Headers({})) == {"user_id": None, "session_id": None}


# ── propagação via configure hook (zero call sites) ─────────────────────────

def test_handler_scope_reaches_llm_calls():
    handler = RecordingHandler()
    llm = make_llm()
    llm.chat_model.script = [AIMessage(content="oi")]
    with obs._handler_scope(handler):
        out = llm.complete([Message(role="user", content="oi")])
    assert out.content == "oi"
    assert handler.chat_model_starts == 1


def test_handler_scope_reaches_tool_agent_graph():
    """O hook cobre o grafo inteiro do ToolAgent (nós/forks do langgraph):
    o handler vê os chain-starts do grafo e os DOIS turnos de modelo (tool
    call + resposta final) sem nenhum call-site do aixon anexar callbacks."""
    from aixon import ToolAgent

    handler = RecordingHandler()

    async def get_weather(city: str) -> str:
        """Clima de uma cidade."""
        return f"sol em {city}"

    class ObsProbeAgent(ToolAgent):
        name = "obs-probe"
        description = "probe"
        llm = make_llm()
        prompt = "responda"
        tools = [get_weather]

    agent = ObsProbeAgent()
    agent.llm.chat_model.script = [
        AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"city": "Recife"}, "id": "c1"}]),
        AIMessage(content="sol em Recife."),
    ]
    with obs._handler_scope(handler):
        out = asyncio.run(agent.ainvoke([Message(role="user", content="clima?")]))
    assert "sol" in out.content
    assert handler.chat_model_starts == 2
    assert handler.chain_starts >= 1


def test_handler_scope_resets_var():
    handler = RecordingHandler()
    with obs._handler_scope(handler):
        assert obs._handler_var.get() is handler
    assert obs._handler_var.get() is None


# ── com o SDK real (offline: host inalcançável, flush por request off) ──────

langfuse = pytest.importorskip("langfuse")


def _fake_langfuse_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")  # porta fechada
    monkeypatch.setenv("AIXON_LANGFUSE_FLUSH_PER_REQUEST", "0")


def test_observe_request_active_and_never_breaks(monkeypatch):
    _fake_langfuse_env(monkeypatch)
    llm = make_llm()
    llm.chat_model.script = [AIMessage(content="resposta")]
    with obs.observe_request("Analista", user_id="jorge@zeus.com",
                             session_id="chat-1",
                             metadata={"adapter": "openai", "stream": False}) as active:
        assert active is True
        out = llm.complete([Message(role="user", content="pergunta")])
    assert out.content == "resposta"
    assert obs._handler_var.get() is None  # scope fechado


def test_observe_request_flush_errors_are_swallowed(monkeypatch):
    _fake_langfuse_env(monkeypatch)
    monkeypatch.setenv("AIXON_LANGFUSE_FLUSH_PER_REQUEST", "1")

    with obs.observe_request("Analista") as active:
        assert active is True
    # flush contra porta fechada: o exporter OTel loga, nunca levanta — se
    # levantasse, o context manager acima teria propagado.


def test_observe_request_exception_propagates_but_cleans_up(monkeypatch):
    _fake_langfuse_env(monkeypatch)
    with pytest.raises(RuntimeError):
        with obs.observe_request("Analista"):
            raise RuntimeError("agente falhou")
    assert obs._handler_var.get() is None


# ── integração com o Server (observe_request recebe agente + identidade) ────

def test_server_passes_agent_and_identity(monkeypatch):
    from fastapi.testclient import TestClient

    from aixon.server.adapters.openai import OpenAIAdapter
    from aixon.server.server import Server
    import aixon.server.server as server_module
    from tests._server_fakes import make_echo

    Server._reset()
    try:
        make_echo("eco-obs", description="d")
        seen: list[dict] = []

        @contextlib.contextmanager
        def spy(agent_name, *, user_id=None, session_id=None, metadata=None):
            seen.append({"agent": agent_name, "user_id": user_id,
                         "session_id": session_id, "metadata": metadata})
            yield True

        monkeypatch.setattr(server_module, "observe_request", spy)
        client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
        headers = {"X-OpenWebUI-User-Email": "jorge@zeus.com",
                   "X-OpenWebUI-Chat-Id": "chat-7"}

        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": "eco-obs",
            "messages": [{"role": "user", "content": "oi"}],
        })
        assert r.status_code == 200
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": "eco-obs", "stream": True,
            "messages": [{"role": "user", "content": "oi"}],
        })
        assert r.status_code == 200
        assert "data:" in r.text

        assert [s["agent"] for s in seen] == ["eco-obs", "eco-obs"]
        assert all(s["user_id"] == "jorge@zeus.com" for s in seen)
        assert all(s["session_id"] == "chat-7" for s in seen)
        assert [s["metadata"]["stream"] for s in seen] == [False, True]
    finally:
        Server._reset()
