# tests/test_tool_hooks.py
"""#17 — hooks pre/post tool call no ToolAgent."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from tests._fakes import make_llm

USER = [Message(role="user", content="soma?")]


def soma(a: int, b: int) -> str:
    """Soma dois inteiros."""
    return str(a + b)


def _script(llm):
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[
            {"name": "soma", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage(content="feito"),
    ]


def _make(name: str, **extra):
    llm = make_llm()
    _script(llm)
    cls = type(f"{name.capitalize()}Agent", (ToolAgent,), {
        "name": name, "llm": llm, "tools": [soma], **extra,
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name)


def test_on_tool_start_observa_e_reescreve_args():
    seen: list[tuple] = []

    def on_tool_start(self, name, args):
        seen.append((name, dict(args)))
        return {"a": 10, "b": 30}          # reescreve

    ends: list[tuple] = []

    def on_tool_end(self, name, args, result, error):
        ends.append((name, result, error))

    agent = _make("hk17a", on_tool_start=on_tool_start, on_tool_end=on_tool_end)
    agent.invoke(USER)
    assert seen == [("soma", {"a": 2, "b": 3})]
    assert ends == [("soma", "40", None)]   # rodou com os args reescritos


def test_hook_start_que_levanta_vira_tool_error_shieldado():
    # on_end DEVE disparar também quando on_start bloqueia a call — pareamento
    # start/end (sweep fix #5): sem isso, um hook que audita toda tentativa de
    # chamada (bloqueada ou não) perde exatamente as bloqueadas.
    ends: list[tuple] = []

    def on_tool_start(self, name, args):
        raise RuntimeError("bloqueado pela política")

    def on_tool_end(self, name, args, result, error):
        ends.append((name, args, result, error))

    agent = _make("hk17b", on_tool_start=on_tool_start, on_tool_end=on_tool_end)
    out = agent.invoke(USER)                # NÃO derruba o run (shield #9)
    assert out.content == "feito"           # modelo seguiu após o TOOL ERROR
    assert len(ends) == 1
    end_name, end_args, end_result, end_error = ends[0]
    assert end_name == "soma"
    assert end_result is None
    assert isinstance(end_error, RuntimeError)
    assert "bloqueado pela política" in str(end_error)


def test_hook_end_que_levanta_nao_corrompe_resultado():
    def on_tool_end(self, name, args, result, error):
        raise RuntimeError("telemetria quebrada")

    agent = _make("hk17c", on_tool_end=on_tool_end)
    out = agent.invoke(USER)
    assert out.content == "feito"


def test_sem_hooks_comportamento_identico():
    agent = _make("hk17d")
    out = agent.invoke(USER)
    assert out.content == "feito"


def test_hook_start_async_que_levanta_tambem_dispara_on_end_pareado():
    # Mesmo pareamento start/end (#5), mas pelo wrapper ASYNC de _guard
    # (awrapper) — a tool async precisa do mesmo comportamento do sync.
    import asyncio

    async def asoma(a: int, b: int) -> str:
        """Soma async."""
        return str(a + b)

    ends: list[tuple] = []

    def on_tool_start(self, name, args):
        raise RuntimeError("bloqueado async")

    def on_tool_end(self, name, args, result, error):
        ends.append((name, result, error))

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[
            {"name": "asoma", "args": {"a": 1, "b": 2}, "id": "c1"}]),
        AIMessage(content="feito async"),
    ]
    cls = type("Hk17eAgent", (ToolAgent,), {
        "name": "hk17e", "llm": llm, "tools": [asoma],
        "on_tool_start": on_tool_start, "on_tool_end": on_tool_end,
    })
    from aixon.registry import get_registry
    agent = get_registry().resolve("hk17e")

    out = asyncio.run(agent.ainvoke(USER))
    assert out.content == "feito async"
    assert len(ends) == 1
    end_name, end_result, end_error = ends[0]
    assert end_name == "asoma"
    assert end_result is None
    assert isinstance(end_error, RuntimeError)
