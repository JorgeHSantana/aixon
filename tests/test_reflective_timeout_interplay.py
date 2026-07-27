# tests/test_reflective_timeout_interplay.py
"""#19 x ReflectiveAgent interplay (integration sweep, #19/#22).

#19 made a ToolAgent worker whose STREAM hits its own ``max_execution_time``
with nothing accumulated emit ``timeout_message`` as CONTENT instead of dying
mute. Wrapped in a ``ReflectiveAgent``, that content is a normal-looking
candidate answer — so it used to reach the judge like any other answer:

  (a) revision_mode="full": the judge (almost always) rejects prose that is
      actually an apology-for-timing-out, and the retry re-runs the SAME slow
      worker for ANOTHER full ``max_execution_time`` — a timeout retry storm.
  (b) revision_mode="patch": the retry calls the worker via ``invoke()``
      (patch requests SEARCH/REPLACE blocks, not a full regen) — a branch
      that was effectively unreachable from a timed-out worker before,
      because a timeout answer basically always got rejected and retried via
      the SAME code path forever. If the worker times out (or hits its
      recursion limit) AGAIN on that invoke() call, ToolAgent.invoke raises
      AixonError — which used to propagate out of ReflectiveAgent._stream,
      crashing the whole stream without ever reaching ``Chunk(done=True)``.

The fix (aixon/agents/reflective.py): (i) a candidate answer that matches the
worker's own ``timeout_message.format(seconds=...)`` byte-for-byte is passed
straight through as content, with `Chunk(done=True)`, WITHOUT ever reaching
the judge (`_is_worker_timeout_answer`); (ii) any worker call inside a
retry/patch round is wrapped in `try/except AixonError`, falling back to the
last judged answer instead of crashing the stream (`_timeout_exhausted_chunks`).

These tests exercise BOTH failure modes as RED-before-the-fix reproductions:
reverting the fix in aixon/agents/reflective.py makes both fail (1a) or raise
(1b) — see the fix's PR description for the confirmation run."""
from __future__ import annotations

import asyncio

from aixon.message import Chunk, Message
from aixon.agents.tool_agent import ToolAgent

from tests.test_reflective import USER, make_reflective
from tests.test_reflective_patch import make_patch_reflective


# ── scriptable ToolAgent-shaped worker (timeout_message/max_execution_time) ──

def make_timeout_worker(name: str, stream_answers: list[str], *,
                        invoke_raises: bool = False,
                        stream_raises: bool = False,
                        max_execution_time: int = 42):
    """Concrete Agent shaped like a real ToolAgent worker for the purpose of
    these tests: it exposes ``timeout_message``/``max_execution_time`` (the
    two attributes ReflectiveAgent's ``_is_worker_timeout_answer`` reads) and
    scripted stream()/astream() content plus invoke()/ainvoke() that can raise
    ``AixonError`` — reproducing a worker blowing its OWN deadline (or hitting
    its recursion limit) on the patch-mode retry's ``invoke``/``ainvoke``
    call. ``stream_raises`` makes stream()/astream() themselves raise on
    iteration (a round-1 failure: recursion limit / deadline before ANY
    candidate). ``stream_answers`` is consumed one entry per stream()/astream()
    call (the last entry repeats). Returns (agent, calls) where ``calls``
    counts invocations per method name."""
    from aixon.agent import Agent
    from aixon.exceptions import AixonError
    from aixon.registry import get_registry

    calls = {"stream": 0, "astream": 0, "invoke": 0, "ainvoke": 0}

    def _next(key: str) -> str:
        i = min(calls[key], len(stream_answers) - 1)
        calls[key] += 1
        return stream_answers[i]

    def stream(self, messages: list[Message]):
        # A generator: the raise fires on FIRST iteration — i.e. inside
        # ReflectiveAgent._stream's try block, like a real ToolAgent whose
        # graph blows the recursion limit mid-stream.
        if stream_raises:
            raise AixonError(
                f"agent '{name}' hit its iteration limit (max_iterations=15) "
                f"without producing a final answer."
            )
        yield Chunk(content=_next("stream"))
        yield Chunk(done=True)

    async def astream(self, messages: list[Message]):
        if stream_raises:
            raise AixonError(
                f"agent '{name}' hit its iteration limit (max_iterations=15) "
                f"without producing a final answer."
            )
        yield Chunk(content=_next("astream"))
        yield Chunk(done=True)

    def invoke(self, messages: list[Message]) -> Message:
        calls["invoke"] += 1
        if invoke_raises:
            raise AixonError(
                f"agent '{name}' exceeded max_execution_time "
                f"({max_execution_time}s)."
            )
        return Message(role="assistant", content=stream_answers[-1])

    async def ainvoke(self, messages: list[Message]) -> Message:
        calls["ainvoke"] += 1
        if invoke_raises:
            raise AixonError(
                f"agent '{name}' exceeded max_execution_time "
                f"({max_execution_time}s)."
            )
        return Message(role="assistant", content=stream_answers[-1])

    cls = type(f"{name.capitalize()}Agent", (Agent,), {
        "stream": stream, "astream": astream,
        "invoke": invoke, "ainvoke": ainvoke,
        "name": name,
        # The two attributes ReflectiveAgent pattern-matches against — the
        # SAME default text/formatting a real ToolAgent worker would use.
        "timeout_message": ToolAgent.timeout_message,
        "max_execution_time": max_execution_time,
    })
    return get_registry().resolve(name), calls


TIMEOUT_TEXT = ToolAgent.timeout_message.format(seconds=42)


# ── (1a) full mode: timeout answer passes through, judge NEVER called ───────

def test_stream_full_timeout_answer_passes_through_without_judge():
    worker, calls = make_timeout_worker("tow1", [TIMEOUT_TEXT])
    r = make_reflective("toref1", worker, ["APROVADO"], rounds=3)  # revision_mode default "full"

    chunks = list(r.stream(USER))

    content = "".join(c.content or "" for c in chunks)
    assert content == TIMEOUT_TEXT
    assert chunks[-1].done is True
    assert r.judge_llm.chat_model._idx == 0    # juiz NUNCA chamado
    assert calls["stream"] == 1                # nenhum retry — sem tempestade


def test_astream_full_timeout_answer_passes_through_without_judge():
    worker, calls = make_timeout_worker("tow1a", [TIMEOUT_TEXT])
    r = make_reflective("toref1a", worker, ["APROVADO"], rounds=3)

    async def run():
        return [c async for c in r.astream(USER)]

    chunks = asyncio.run(run())

    content = "".join(c.content or "" for c in chunks)
    assert content == TIMEOUT_TEXT
    assert chunks[-1].done is True
    assert r.judge_llm.chat_model._idx == 0
    assert calls["astream"] == 1


# ── (1b) patch mode: patch retry's worker.invoke/ainvoke raises AixonError ──
# The stream must NOT propagate the exception; it must end with the round-1
# (rejected) content, not empty, and Chunk(done=True).

def test_stream_patch_retry_worker_invoke_raises_does_not_crash():
    worker, calls = make_timeout_worker(
        "tow2", ["resposta normal da rodada 1"], invoke_raises=True)
    r = make_patch_reflective("toref2", worker, ["1. Falta citar a fonte."], rounds=3)

    chunks = list(r.stream(USER))  # must not raise AixonError

    content = "".join(c.content or "" for c in chunks)
    assert content == "resposta normal da rodada 1"
    assert chunks[-1].done is True
    assert calls["stream"] == 1
    assert calls["invoke"] == 1                # a chamada do patch que estourou
    assert r.judge_llm.chat_model._idx == 1    # juiz rodou uma vez (reprovou)


def test_astream_patch_retry_worker_ainvoke_raises_does_not_crash():
    worker, calls = make_timeout_worker(
        "tow2a", ["resposta normal da rodada 1"], invoke_raises=True)
    r = make_patch_reflective("toref2a", worker, ["1. Falta citar a fonte."], rounds=3)

    async def run():
        return [c async for c in r.astream(USER)]

    chunks = asyncio.run(run())  # must not raise AixonError

    content = "".join(c.content or "" for c in chunks)
    assert content == "resposta normal da rodada 1"
    assert chunks[-1].done is True
    assert calls["astream"] == 1
    assert calls["ainvoke"] == 1
    assert r.judge_llm.chat_model._idx == 1


# ── (1c) round 1: the FIRST worker call raises — no candidate answer yet ────
# The except-AixonError shield also covers the initial call; falling back to
# `answer` there would emit Chunk(content="") + done — an EMPTY final answer,
# the mute death #19 exists to eliminate, reopened via Reflective. The stream
# must instead close with an honest non-empty message derived from the error.

def test_stream_worker_raises_on_round_one_emits_honest_message():
    worker, _ = make_timeout_worker("tow3", ["nunca usado"], stream_raises=True)
    r = make_reflective("toref3", worker, ["APROVADO"], rounds=3)

    chunks = list(r.stream(USER))  # must not raise AixonError

    content = "".join(c.content or "" for c in chunks)
    assert "falhou antes de produzir" in content
    assert "iteration limit" in content
    assert content != ""
    assert chunks[-1].done is True
    assert r.judge_llm.chat_model._idx == 0    # nada julgado — não houve resposta


def test_astream_worker_raises_on_round_one_emits_honest_message():
    worker, _ = make_timeout_worker("tow3a", ["nunca usado"], stream_raises=True)
    r = make_reflective("toref3a", worker, ["APROVADO"], rounds=3)

    async def run():
        return [c async for c in r.astream(USER)]

    chunks = asyncio.run(run())  # must not raise AixonError

    content = "".join(c.content or "" for c in chunks)
    assert "falhou antes de produzir" in content
    assert "iteration limit" in content
    assert content != ""
    assert chunks[-1].done is True
    assert r.judge_llm.chat_model._idx == 0
