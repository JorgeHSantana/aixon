# tests/test_as_tool_timeout_semantics.py
"""#22 follow-up (sweep-0123, item 1) — semantic regression: since #22 made
``as_tool()`` drive the wrapped agent through ``stream()``/``astream()``
(never ``invoke()``/``ainvoke()``), a worker that blows its OWN
``max_execution_time`` with nothing accumulated emits ``timeout_message``
as CONTENT (#19 — an agent must never die mute on the streaming path). Before
#22, ``as_tool()`` drove ``invoke()``, which raises ``AixonError`` on the
same timeout — the parent's tool-call shield (``aixon._interop.tools._guard``,
#9) converted that into a visible ``TOOL ERROR`` result. After #22, the same
timeout instead looks like the subagent's own (legitimate) answer: a normal
string return from the tool, no error, nothing for the parent's shield to
catch.

This is a regression: a failure (the subagent never really answered) must
never look like a normal response reaching the parent model as fact. The fix
(``aixon/agent.py``, ``_timeout_texts`` + the check in ``_drive_sync``/
``_drive_async``) detects a final content that byte-matches the wrapped
agent's own formatted ``timeout_message`` (or, for a ``ReflectiveAgent``
pass-through, its wrapped worker's) and raises ``AixonError`` instead of
returning it — restoring the pre-#22 semantics: the parent's shield turns it
into a ``TOOL ERROR`` the calling model can react to.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage

from aixon._interop.tools import coerce_tools
from aixon.agents.reflective import ReflectiveAgent
from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.message import Message
from aixon.registry import get_registry
from tests._fakes import make_llm


# ── (a) plain ToolAgent worker whose tool blows the deadline ────────────────

def _make_slow_child_sync(name: str, *, max_execution_time: float = 0.2):
    def lenta(x: str) -> str:
        "tool sincrona que trava além do deadline"
        time.sleep(1.0)
        return "nunca chega"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    cls = type(f"{name.capitalize()}Agent", (ToolAgent,), {
        "name": name, "llm": llm, "tools": [lenta],
        "max_execution_time": max_execution_time,
    })
    return get_registry().resolve(name)


def _make_slow_child_async(name: str, *, max_execution_time: float = 0.2):
    async def lenta(x: str) -> str:
        "tool assincrona que trava além do deadline"
        await asyncio.sleep(1.0)
        return "nunca chega"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    cls = type(f"{name.capitalize()}Agent", (ToolAgent,), {
        "name": name, "llm": llm, "tools": [lenta],
        "max_execution_time": max_execution_time,
    })
    return get_registry().resolve(name)


def test_as_tool_func_levanta_aixon_error_quando_worker_estoura_deadline():
    child = _make_slow_child_sync("timeoutastool1")
    tool = child.as_tool()

    with pytest.raises(AixonError, match="timeoutastool1"):
        tool.func("vai")


def test_as_tool_coroutine_levanta_aixon_error_quando_worker_estoura_deadline():
    child = _make_slow_child_async("timeoutastool2")
    tool = child.as_tool()

    async def run():
        await tool.coroutine("vai")

    with pytest.raises(AixonError, match="timeoutastool2"):
        asyncio.run(run())


def test_as_tool_shield_do_pai_converte_timeout_em_tool_error_sync():
    # The exact pre-#22 semantics: the parent's tool-call shield (#9) turns
    # the AixonError raised at the tool boundary into a readable TOOL ERROR
    # result — the FIX target, not just "raises somewhere".
    child = _make_slow_child_sync("timeoutastool3")
    tool = child.as_tool()
    [lc_tool] = coerce_tools([tool])  # shield_errors=True default (#9)

    result = lc_tool.invoke({"text": "vai"})

    assert result.startswith("TOOL ERROR"), result
    assert "timeoutastool3" in result


def test_as_tool_shield_do_pai_converte_timeout_em_tool_error_async():
    child = _make_slow_child_async("timeoutastool4")
    tool = child.as_tool()
    [lc_tool] = coerce_tools([tool])

    result = asyncio.run(lc_tool.ainvoke({"text": "vai"}))

    assert result.startswith("TOOL ERROR"), result
    assert "timeoutastool4" in result


# ── (b) ReflectiveAgent embrulhando um worker que trava ──────────────────────

def _make_reflective_over_slow_worker(name: str, worker, *, rounds: int = 2):
    judge = make_llm(temperature=0)
    judge.chat_model.script = [AIMessage(content="APROVADO")]  # never reached
    cls = type(f"{name.capitalize()}Agent", (ReflectiveAgent,), {
        "name": name,
        "agent": worker,
        "judge_llm": judge,
        "judge_rubric": "1. Regra qualquer.",
        "max_rounds": rounds,
    })
    return get_registry().resolve(name)


def test_as_tool_func_levanta_aixon_error_reflective_sobre_worker_travado():
    worker = _make_slow_child_sync("timeoutreflworker1")
    reflective = _make_reflective_over_slow_worker("timeoutreflparent1", worker)
    tool = reflective.as_tool()

    with pytest.raises(AixonError, match="timeoutreflparent1"):
        tool.func("vai")


def test_as_tool_coroutine_levanta_aixon_error_reflective_sobre_worker_travado():
    worker = _make_slow_child_async("timeoutreflworker2")
    reflective = _make_reflective_over_slow_worker("timeoutreflparent2", worker)
    tool = reflective.as_tool()

    async def run():
        await tool.coroutine("vai")

    with pytest.raises(AixonError, match="timeoutreflparent2"):
        asyncio.run(run())


def test_as_tool_shield_do_pai_converte_timeout_reflective_em_tool_error():
    worker = _make_slow_child_sync("timeoutreflworker3")
    reflective = _make_reflective_over_slow_worker("timeoutreflparent3", worker)
    tool = reflective.as_tool()
    [lc_tool] = coerce_tools([tool])

    result = lc_tool.invoke({"text": "vai"})

    assert result.startswith("TOOL ERROR"), result
    assert "timeoutreflparent3" in result


# ── (c) conteúdo normal continua byte-idêntico (sem regressão) ──────────────

def test_as_tool_conteudo_normal_nao_e_afetado_pelo_check_de_timeout():
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "helper", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="resposta final determinística"),
    ]

    def helper(x: str) -> str:
        "helper rapido"
        return "ok"

    cls = type("Normalastool5Agent", (ToolAgent,), {
        "name": "normalastool5", "llm": llm, "tools": [helper],
    })
    agent = get_registry().resolve("normalastool5")

    direct = agent.invoke([Message(role="user", content="oi")]).content
    object.__setattr__(llm.chat_model, "_idx", 0)

    tool = agent.as_tool()
    via_tool = tool.func("oi")

    assert via_tool == direct == "resposta final determinística"
