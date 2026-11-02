# tests/test_audit_task_6_5.py
"""Achados da Task 6.5 (orchestrator e instantiate)."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Chunk, Message
from aixon.registry import get_registry


def _worker(name: str, tag: str):
    type(
        name.capitalize() + "Agent",
        (Agent,),
        {
            "name": name,
            "description": f"faz {name}",
            "invoke": lambda self, messages: Message(role="assistant", content=tag),
            "stream": lambda self, messages: iter([Chunk(content=tag), Chunk(done=True)]),
        },
    )
    return get_registry().resolve(name)


class _Sup:
    def complete(self, messages):
        if messages and messages[-1].role == "assistant":
            return Message(role="assistant", content="DONE")
        return Message(role="assistant", content="alpha")


def test_orchestrator_1(monkeypatch):
    """Erro do registry que não é 'não registrado' sobe, sem instância nova."""
    from aixon import agent as agent_mod
    from aixon.registry import Registry

    w = _worker("alpha", "a")
    cls = type(w)

    def boom(self, name):
        raise RuntimeError("registry quebrado")

    monkeypatch.setattr(Registry, "resolve", boom)
    with pytest.raises(RuntimeError, match="registry quebrado"):
        agent_mod.instantiate(cls)


def test_orchestrator_1_not_registered_falls_back():
    from aixon.agent import instantiate

    w = _worker("alpha", "a")
    assert instantiate(type(w)) is w
    assert instantiate(w) is w


def test_reflective_8():
    import aixon.agents.orchestrator as orch
    import aixon.agents.reflective as refl
    from aixon.agent import instantiate

    assert not hasattr(orch, "_instantiate")
    assert "_instantiate" not in refl.__dict__
    assert instantiate.__doc__


def _orch(name="O"):
    _worker("alpha", "[a]")

    cls = type(name + "Orchestrator", (Orchestrator,), {"supervisor": _Sup(), "agents": [get_registry().resolve("alpha")]})
    return get_registry().resolve(cls.__name__.lower())


def test_orchestrator_4(monkeypatch):
    """Os workers são resolvidos uma vez no build, não a cada roteamento."""
    o = _orch("Route")
    calls = []
    real = type(o)._worker_instances

    def counting(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(type(o), "_worker_instances", counting)
    out = o.invoke([Message(role="user", content="oi")])
    assert out.content == "[a]"
    assert len(calls) == 1


def test_orchestrator_2_invoke_ainvoke_equal():
    o = _orch("Eq")
    a = o.invoke([Message(role="user", content="oi")])
    b = asyncio.run(o.ainvoke([Message(role="user", content="oi")]))
    assert (a.content, a.usage) == (b.content, b.usage) == ("[a]", a.usage)


def test_orchestrator_3_stream_astream_equal():
    o = _orch("St")

    async def collect():
        return [c async for c in o.astream([Message(role="user", content="oi")])]

    s = list(o.stream([Message(role="user", content="oi")]))
    assert s == asyncio.run(collect())
    assert s[-1].done and s[-2].content == "[a]"


def test_orchestrator_8(monkeypatch):
    """Duas threads na primeira chamada compilam um grafo só."""
    o = _orch("Lock")
    builds = []
    real = type(o).build_graph

    def slow(self):
        builds.append(1)
        time.sleep(0.05)
        return real(self)

    monkeypatch.setattr(type(o), "build_graph", slow)
    got = []
    ts = [threading.Thread(target=lambda: got.append(o._compiled())) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(builds) == 1
    assert all(g is got[0] for g in got)
