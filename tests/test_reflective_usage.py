# tests/test_reflective_usage.py
"""ReflectiveAgent: usage aggregated across EVERY turn of the loop — the
worker's answer AND the judge's verdict, for every round attempted (not just
the last round) — so Message.usage on the final answer is the run's SUM."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon.agent import Agent
from aixon.agents.reflective import ReflectiveAgent
from aixon.message import Message
from aixon.registry import get_registry
from tests._fakes import make_llm

USER = [Message(role="user", content="qual a capital do Ceará?")]


def _usage(p: int, c: int) -> dict:
    """Neutral (OpenAI-shaped) usage — what Message.usage carries."""
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _lc_usage(p: int, c: int) -> dict:
    """LangChain-shaped usage_metadata — what an AIMessage carries; converted
    to the neutral shape by usage_from_metadata inside LLM.complete/acomplete."""
    return {"input_tokens": p, "output_tokens": c, "total_tokens": p + c}


def make_worker(name: str, turns: "list[tuple[str, dict | None]]"):
    """Concrete Agent returning (content, usage) tuples in order, sync+async."""
    calls: list[list[Message]] = []

    def invoke(self, messages: list[Message]) -> Message:
        calls.append(list(messages))
        i = min(len(calls) - 1, len(turns) - 1)
        content, usage = turns[i]
        return Message(role="assistant", content=content, usage=usage)

    cls = type(f"{name.capitalize()}Agent", (Agent,), {
        "invoke": invoke,
        "stream": lambda self, messages: iter([]),  # unused in these tests
        "name": name,
    })
    return get_registry().resolve(name), calls


def make_judge(script_and_usage: "list[tuple[str, dict | None]]"):
    judge = make_llm(temperature=0)
    judge.chat_model.script = [
        AIMessage(content=verdict, usage_metadata=usage)
        for verdict, usage in script_and_usage
    ]
    return judge


def make_reflective(name: str, agent, judge, *, rounds: int = 3):
    cls = type(f"{name.capitalize()}Agent", (ReflectiveAgent,), {
        "name": name,
        "agent": agent,
        "judge_llm": judge,
        "judge_rubric": "1. A resposta cita a fonte.",
        "max_rounds": rounds,
    })
    return get_registry().resolve(name)


# ── sums worker + judge usage across retries ─────────────────────────────────

def test_sums_worker_and_judge_usage_across_retry_sync():
    worker, _ = make_worker("gen-usage-1", [
        ("v1", _usage(10, 5)),
        ("v2 (fonte: IBGE)", _usage(12, 6)),
    ])
    judge = make_judge([
        ("1. Falta a fonte.", _lc_usage(3, 1)),
        ("APROVADO", _lc_usage(4, 2)),
    ])
    r = make_reflective("ref-usage-1", worker, judge)
    out = r.invoke(USER)
    assert out.content == "v2 (fonte: IBGE)"
    assert out.usage == _usage(10 + 3 + 12 + 4, 5 + 1 + 6 + 2)


def test_sums_worker_and_judge_usage_across_retry_async():
    worker, _ = make_worker("gen-usage-2", [
        ("v1", _usage(10, 5)),
        ("v2 (fonte: IBGE)", _usage(12, 6)),
    ])
    judge = make_judge([
        ("1. Falta a fonte.", _lc_usage(3, 1)),
        ("APROVADO", _lc_usage(4, 2)),
    ])
    r = make_reflective("ref-usage-2", worker, judge)
    out = asyncio.run(r.ainvoke(USER))
    assert out.content == "v2 (fonte: IBGE)"
    assert out.usage == _usage(10 + 3 + 12 + 4, 5 + 1 + 6 + 2)


def test_approved_on_first_round_sums_only_that_rounds_turns():
    worker, _ = make_worker("gen-usage-3", [("v1 (fonte: IBGE)", _usage(8, 4))])
    judge = make_judge([("APROVADO", _lc_usage(2, 1))])
    r = make_reflective("ref-usage-3", worker, judge)
    out = r.invoke(USER)
    assert out.usage == _usage(8 + 2, 4 + 1)


def test_turn_without_usage_contributes_zero_not_none():
    worker, _ = make_worker("gen-usage-4", [
        ("v1", None),                       # worker reports no usage this round
        ("v2 (fonte: IBGE)", _usage(12, 6)),
    ])
    judge = make_judge([
        ("1. Falta a fonte.", _lc_usage(3, 1)),
        ("APROVADO", None),                 # judge reports no usage this round
    ])
    r = make_reflective("ref-usage-4", worker, judge)
    out = r.invoke(USER)
    # only worker-round-2 (12,6) and judge-round-1 (3,1) reported usage.
    assert out.usage == _usage(12 + 3, 6 + 1)


def test_no_turn_reports_usage_final_usage_is_none():
    worker, _ = make_worker("gen-usage-5", [("v1", None)])
    judge = make_judge([("APROVADO", None)])
    r = make_reflective("ref-usage-5", worker, judge)
    out = r.invoke(USER)
    assert out.usage is None


def test_exhausted_rounds_still_sums_all_attempted_turns():
    worker, _ = make_worker("gen-usage-6", [
        ("v1", _usage(5, 1)),
        ("v2", _usage(6, 2)),
    ])
    judge = make_judge([
        ("1. Ruim.", _lc_usage(1, 1)),
        ("1. Ainda ruim.", _lc_usage(1, 1)),
    ])
    r = make_reflective("ref-usage-6", worker, judge, rounds=2)
    out = r.invoke(USER)
    assert out.content == "v2"  # last attempt, rounds exhausted
    assert out.usage == _usage(5 + 1 + 6 + 1, 1 + 1 + 2 + 1)


def test_worker_message_is_not_mutated_in_place():
    # The worker may return a cached/shared Message (nothing in Agent's
    # contract forbids it); the reflective loop must return a COPY with the
    # summed usage, never stamp the total onto the worker's own object.
    shared = Message(role="assistant", content="v1 (fonte: IBGE)",
                     usage=_usage(8, 4))

    class SharedAgent(Agent):
        name = "gen-usage-shared"

        def invoke(self, messages: list[Message]) -> Message:
            return shared

        def stream(self, messages: list[Message]):
            return iter([])

    worker = get_registry().resolve("gen-usage-shared")
    judge = make_judge([("APROVADO", _lc_usage(2, 1))])
    r = make_reflective("ref-usage-shared", worker, judge)
    out = r.invoke(USER)
    assert out.usage == _usage(8 + 2, 4 + 1)
    assert out is not shared
    assert shared.usage == _usage(8, 4)  # worker's own Message untouched
