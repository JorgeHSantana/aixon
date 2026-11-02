# tests/test_audit_task_6_4.py
"""Achados observability-2..4: registro do hook num lugar, inicialização
lazy sob trava e yield do caminho de falha fora do except."""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import aixon.observability as obs


@pytest.fixture
def fake_langfuse(monkeypatch):
    """langfuse falso: CallbackHandler lento e contado; get_client falha."""
    built = []

    class CallbackHandler:
        def __init__(self):
            time.sleep(0.05)
            built.append(self)

    def get_client():
        raise RuntimeError("langfuse fora do ar")

    root = types.ModuleType("langfuse")
    root.get_client = get_client
    root.propagate_attributes = None
    sub = types.ModuleType("langfuse.langchain")
    sub.CallbackHandler = CallbackHandler
    root.langchain = sub
    monkeypatch.setitem(sys.modules, "langfuse", root)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", sub)
    monkeypatch.setattr(obs, "_handler", None)
    monkeypatch.setattr(obs, "_disabled", False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    return built


def test_observability_2(monkeypatch, fake_langfuse):
    """Igualdade: o hook é registrado uma única vez, venha o primeiro uso de
    _ensure_handler ou de _handler_scope."""
    import langchain_core.tracers.context as ctx

    calls = []
    monkeypatch.setattr(ctx, "register_configure_hook", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(obs, "_hook_registered", False)
    assert obs._ensure_handler() is fake_langfuse[0]
    with obs._handler_scope(object()):
        pass
    with obs._handler_scope(object()):
        pass
    assert calls == [(obs._handler_var,)]


def test_observability_3(fake_langfuse):
    """Primeiro uso concorrente cria um só CallbackHandler."""
    barrier = threading.Barrier(5)
    got = []

    def worker():
        barrier.wait()
        got.append(obs._ensure_handler())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(fake_langfuse) == 1
    assert all(h is fake_langfuse[0] for h in got)


def test_observability_4(fake_langfuse):
    """Com o Langfuse falhando ao abrir, a exceção do agente sai sem
    __context__ apontando para a do Langfuse."""
    with pytest.raises(ValueError) as info:
        with obs.observe_request("Analista") as active:
            assert active is False
            raise ValueError("do agente")
    assert info.value.__context__ is None
