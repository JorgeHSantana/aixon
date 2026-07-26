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
    def on_tool_start(self, name, args):
        raise RuntimeError("bloqueado pela política")

    agent = _make("hk17b", on_tool_start=on_tool_start)
    out = agent.invoke(USER)                # NÃO derruba o run (shield #9)
    assert out.content == "feito"           # modelo seguiu após o TOOL ERROR


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
