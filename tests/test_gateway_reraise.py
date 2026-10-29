# tests/test_gateway_reraise.py
"""0.1.28 — a verdict of the gateway (budget spent, rate limited) is not
swallowed on its way up.

Three places used to absorb errors. The tool shield (#9) turns any failure of a
tool into a `TOOL ERROR` text that the model reads and goes on from. The
ReflectiveAgent turns a judge failure (#24) into an unreviewed answer, and an
`AixonError` of the worker (#19) into content. For a verdict of the gateway
(`classify_gateway_error`) none of that is an answer to the question: the
request is over, and whoever made it has to be told. So the shield and the 8
handlers of the ReflectiveAgent re-raise the very exception object, and
everything else keeps degrading exactly as before, a 429 of another API (here,
OpenAI itself) included.

Offline: the SDK errors are built by hand (`tests/_fakes.make_gateway_status_error`)
and the models are the fakes of the suite."""
from __future__ import annotations

import asyncio

import openai
import pytest
from langchain_core.messages import AIMessage

from aixon._interop.tools import coerce_tools
from aixon.agent import Agent, AgentTool
from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import (
    AixonError,
    GatewayBudgetExceeded,
    GatewayRateLimited,
    classify_gateway_error,
)
from aixon.message import Chunk, Message
from aixon.reasoning import reasoning_channel
from aixon.registry import get_registry
from tests._fakes import BUDGET, make_gateway_status_error as status_error, make_llm
from tests.test_reflective import make_scripted_agent
from tests.test_reflective_judge_down import (
    _FailingChatModel,
    make_failing_judge,
    make_reflective_judge,
)
from tests.test_reflective_stats import reflective_log  # reuse the _ListHandler fixture

pytestmark = pytest.mark.usefixtures("gateway_base_url")

ASK = [Message(role="user", content="q")]
ENTRIES = ["invoke", "stream", "ainvoke", "astream"]
PATHS = ["invoke", "ainvoke"]  # how a coerced tool is called: the sync path or the async one
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# The verdicts of the gateway: the SDK's 422 (budget) and 429, and aixon's own
# types for them. Factories, not instances: every test raises its own, so no
# traceback and no __context__ is shared between tests.
GATEWAY = [
    pytest.param(lambda: status_error(422, BUDGET), id="422-budget"),
    pytest.param(lambda: status_error(429, "slow"), id="429"),
    pytest.param(lambda: GatewayBudgetExceeded("sem cota"), id="own-budget"),
    pytest.param(lambda: GatewayRateLimited("devagar"), id="own-rate-limited"),
]
# The two the SDK raises, for the tests that run the judge.
GATEWAY_SDK = GATEWAY[:2]


def wrapped_gateway_error():
    """An `AixonError` that carries the verdict only as its cause, as a custom
    agent does when it re-raises what its model call raised."""
    exc = AixonError("o especialista falhou")
    exc.__cause__ = status_error(422, BUDGET)
    return exc


# What a ReflectiveAgent's worker can raise that the loop must not turn into content.
WORKER_VERDICTS = [
    pytest.param(lambda: GatewayBudgetExceeded("sem cota"), id="own-budget"),
    pytest.param(wrapped_gateway_error, id="wrapped"),
]

# Failures that are NOT a verdict of the gateway.
OTHER = [
    pytest.param(
        lambda: status_error(422, "Unprocessable Entity: bad schema"), id="422-without-budget"
    ),
    pytest.param(lambda: status_error(500, BUDGET), id="500-budget"),
    pytest.param(lambda: status_error(429, "slow", url=OPENAI_URL), id="429-openai"),
    pytest.param(lambda: ValueError("boom"), id="value-error"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

async def collect(agen, out):
    """Append every chunk of `agen` to `out` as it arrives, so that `out` still
    holds what came out before an error."""
    async for chunk in agen:
        out.append(chunk)


def drive(agent, entry, out):
    """Run `agent` through `entry` to the end, appending to `out` what comes out
    as it comes out: after an error, `out` still holds what was emitted before
    it. `stream` and `astream` hand over chunks. `invoke` and `ainvoke` have none,
    so their reasoning (read off the channel, error or not) and their answer go
    to `out` as a chunk each."""
    if entry == "stream":
        for chunk in agent.stream(ASK):
            out.append(chunk)
    elif entry == "astream":
        asyncio.run(collect(agent.astream(ASK), out))
    else:
        with reasoning_channel() as channel:
            try:
                if entry == "invoke":
                    answer = agent.invoke(ASK)
                else:
                    answer = asyncio.run(agent.ainvoke(ASK))
            finally:
                out.extend(Chunk(reasoning=line + "\n") for line in channel.drain())
        out.append(Chunk(content=answer.content))


def reasoning_of(chunks):
    return "".join(c.reasoning for c in chunks)


def raising_tool(shape, path, exc, **hooks):
    """The tool that `coerce_tools` makes of an entry that raises `exc` when it
    is called: an `AgentTool` with both its `func` and its `coroutine`, or a
    plain callable, which `coerce_tools` registers on the path it fits (a
    function on `invoke`, a coroutine function on `ainvoke`)."""

    def especialista(text: str) -> str:
        """Dados internos."""
        raise exc

    async def aespecialista(text: str) -> str:
        """Dados internos."""
        raise exc

    if shape == "agent-tool":
        entry = AgentTool(
            name="especialista",
            description="Dados internos.",
            func=especialista,
            coroutine=aespecialista,
        )
    else:
        entry = aespecialista if path == "ainvoke" else especialista
    [tool] = coerce_tools([entry], **hooks)
    return tool


def call_tool(tool, path):
    """The tool called the way its agent calls it, on the sync path or the async one."""
    if path == "invoke":
        return tool.invoke({"text": "q"})
    return asyncio.run(tool.ainvoke({"text": "q"}))


def make_specialist(exc):
    """A ToolAgent whose model fails with `exc` on its first call."""
    llm = make_llm()
    llm._chat_model = _FailingChatModel(script=[exc])
    type("EspecialistaAgent", (ToolAgent,), {"name": "especialista", "llm": llm})
    return get_registry().resolve("especialista")


def make_caller(name, tool_name, tool):
    """A ToolAgent whose model calls the tool `tool_name` on its first turn."""
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {"text": "q"}, "id": "c1"}],
        )
    ]
    type(f"{name.capitalize()}Agent", (ToolAgent,), {"name": name, "llm": llm, "tools": [tool]})
    return get_registry().resolve(name)


def make_failing_worker(name, exc, answers):
    """A worker that answers `answers`, one per call, and raises `exc` on every
    call after them. `stream` and `invoke` share the count, as in
    `make_scripted_agent`, which cannot fail by itself."""
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        if calls > len(answers):
            raise exc
        return answers[calls - 1]

    def invoke(self, messages):
        return Message(role="assistant", content=step())

    def stream(self, messages):
        yield Chunk(content=step())
        yield Chunk(done=True)

    type(f"{name.capitalize()}Agent", (Agent,), {"invoke": invoke, "stream": stream, "name": name})
    return get_registry().resolve(name)


# ── the shield (#9) ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("make_exc", GATEWAY)
@pytest.mark.parametrize("shape", ["agent-tool", "callable"])
@pytest.mark.parametrize("path", PATHS)
def test_shield_reraises_same_object(make_exc, shape, path):
    exc = make_exc()
    tool = raising_tool(shape, path, exc)

    with pytest.raises(type(exc)) as info:
        call_tool(tool, path)

    assert info.value is exc  # the object the tool raised, not a copy of it


@pytest.mark.parametrize("make_exc", OTHER)
@pytest.mark.parametrize("path", PATHS)
def test_shield_still_shields_other_errors(make_exc, path):
    tool = raising_tool("agent-tool", path, make_exc())

    assert call_tool(tool, path).startswith("TOOL ERROR")


@pytest.mark.parametrize("make_exc", GATEWAY)
@pytest.mark.parametrize("path", PATHS)
def test_on_tool_end_sees_gateway_error(make_exc, path):
    exc = make_exc()
    ended = []
    tool = raising_tool(
        "agent-tool",
        path,
        exc,
        on_tool_end=lambda name, kwargs, result, error: ended.append((name, result, error)),
    )

    with pytest.raises(type(exc)):
        call_tool(tool, path)

    [(name, result, error)] = ended  # the hook ran, once, although the shield re-raised
    assert name == "especialista" and result is None
    assert error is exc


@pytest.mark.parametrize("make_exc", GATEWAY)
@pytest.mark.parametrize("path", PATHS)
def test_shield_reraises_a_verdict_raised_by_on_tool_start(make_exc, path):
    # A hook that checks the quota before the call and raises the verdict: the
    # same rule as for the tool itself, on both paths
    exc = make_exc()
    ended = []

    def on_tool_start(name, kwargs):
        raise exc

    tool = raising_tool(
        "agent-tool",
        path,
        AssertionError("the tool ran although on_tool_start raised"),
        on_tool_start=on_tool_start,
        on_tool_end=lambda name, kwargs, result, error: ended.append(error),
    )

    with pytest.raises(type(exc)) as info:
        call_tool(tool, path)

    assert info.value is exc
    assert ended == [exc]


# ── through the agents ───────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ENTRIES)
def test_tool_agent_propagates_specialist_gateway_error(entry):
    especialista = make_specialist(status_error(422, BUDGET))
    coordenador = make_caller("coordenador", "especialista", especialista.as_tool())

    with pytest.raises(openai.UnprocessableEntityError) as info:
        drive(coordenador, entry, [])

    assert classify_gateway_error(info.value) is GatewayBudgetExceeded


def test_nested_gateway_error_two_levels_deep():
    # coordenador -> revisor (reflective) -> trabalhador -> especialista: the
    # error crosses the shield of the specialist's tool, the reflective loop,
    # and the shield of the reviewer's tool.
    especialista = make_specialist(status_error(429, "slow"))
    trabalhador = make_caller("trabalhador", "especialista", especialista.as_tool())
    revisor = make_reflective_judge("revisor", trabalhador, make_failing_judge(["APROVADO"]))
    coordenador = make_caller("coordenador", "revisor", revisor.as_tool())

    chunks: list[Chunk] = []
    with pytest.raises(openai.RateLimitError) as info:
        asyncio.run(collect(coordenador.astream(ASK), chunks))

    assert classify_gateway_error(info.value) is GatewayRateLimited
    assert not any(
        "TOOL ERROR" in c.content or "agente interno falhou" in c.content for c in chunks
    )


# ── the judge (#24) ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("make_exc", GATEWAY_SDK)
def test_judge_reraises_gateway_error(entry, make_exc, reflective_log):
    exc = make_exc()
    gen, calls = make_scripted_agent("gen", ["Fortaleza."])
    r = make_reflective_judge("revisor", gen, make_failing_judge([exc]))

    out: list[Chunk] = []
    with pytest.raises(type(exc)) as info:
        drive(r, entry, out)

    assert info.value is exc
    assert r.judge_unavailable_label not in reasoning_of(out)
    assert not any(c.content for c in out)  # the unreviewed answer is not delivered
    assert reflective_log == []  # neither the outage warning nor a reflective_run line
    assert len(calls) == 1


@pytest.mark.parametrize("entry", ENTRIES)
def test_non_gateway_429_keeps_judge_degradation(entry):
    # A judge that runs straight on OpenAI, outside the gateway, behaves as it
    # always did (#24): the worker's answer is delivered, unreviewed.
    gen, calls = make_scripted_agent("gen", ["Fortaleza."])
    judge = make_failing_judge([status_error(429, "slow", url=OPENAI_URL)])
    r = make_reflective_judge("revisor", gen, judge)

    out: list[Chunk] = []
    drive(r, entry, out)

    assert "".join(c.content for c in out) == "Fortaleza."
    assert r.judge_unavailable_label in reasoning_of(out)
    assert len(calls) == 1


# ── the worker (#19) ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ["stream", "astream"])
@pytest.mark.parametrize("moment", ["round-1", "patch-round"])
@pytest.mark.parametrize("make_exc", WORKER_VERDICTS)
def test_worker_gateway_error_is_not_converted_to_content(entry, moment, make_exc):
    exc = make_exc()
    # round-1: the worker fails on its very first call. patch-round: it answers
    # round 1, the judge rejects that, and it fails on the correction, which in
    # patch mode is one more call to the worker.
    answers = [] if moment == "round-1" else ["Fortaleza."]
    worker = make_failing_worker("trabalhador", exc, answers)
    r = make_reflective_judge("revisor", worker, make_failing_judge(["1. Falta citar a fonte."]))
    type(r).revision_mode = "patch"  # as in test_reflective_stats: the loop reads it on each run

    chunks: list[Chunk] = []
    with pytest.raises(type(exc)) as info:
        drive(r, entry, chunks)

    assert info.value is exc
    assert classify_gateway_error(info.value) is GatewayBudgetExceeded
    assert not any("O agente interno falhou" in c.content for c in chunks)
    assert not any(c.content for c in chunks)  # nor the rejected candidate, as the answer
