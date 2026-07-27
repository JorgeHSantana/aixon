"""#21: reasoning_channel()'s cleanup must not blow up when it runs in a
different asyncio Context than the one that opened it.

``reasoning_channel()`` is a ``@contextmanager`` wrapping a
``contextvars.ContextVar``. ``ContextVar.reset(token)`` requires the SAME
Context that produced the token via ``set()`` — calling it from a different
Context raises ``ValueError: Token was created in a different Context``.

Normally the enter/exit pair runs in the same Context (a plain ``with``
block never crosses a context boundary). But ``ToolAgent.astream()`` is an
async generator built around ``with reasoning_channel(): ...``, and async
generators that are abandoned (the caller ``break``s out of an
``async for`` without calling ``aclose()``) get finalized by the garbage
collector. asyncio's async-generator finalizer hook schedules that
``aclose()`` as a brand-new ``Task`` via ``call_soon_threadsafe`` — which
runs in whatever Context is active *then*, not the Context of the request
that originally started the stream (typically already gone by the time GC
runs, e.g. after an abrupt client disconnect). That mismatch is exactly
what reproduces the ValueError from the issue.

(a) below is the deterministic unit reproduction: no event loop, no GC
timing, just ``contextvars.copy_context()`` used directly to run the
context manager's ``__exit__`` in a different Context than its ``__enter__``.
(b) is a smoke-level integration repro against a real ``ToolAgent.astream()``
abandoned mid-stream, with ``gc.collect()`` forcing finalization — see its
docstring for why its assertion is best-effort rather than as strong as (a).
"""
from __future__ import annotations

import asyncio
import contextvars
import gc

from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Message
from aixon.reasoning import current_channel, emit_reasoning, reasoning_channel
from aixon.registry import get_registry
from tests._fakes import FakeChatModel


# ── (a) deterministic unit repro ──────────────────────────────────────────

def test_exit_in_different_context_does_not_raise_and_var_stays_usable():
    # The whole test body runs inside its OWN copied Context (`outer`,
    # entered via `outer.run(body)`) rather than the shared Context this
    # module's other tests execute in. That isolation is deliberate, not
    # incidental: because the reproduction below leaves `_current` in
    # `outer` pointing at a stale channel (see the comment further down —
    # that's the documented #21 tradeoff), running it directly in the
    # process-wide default Context would leak that stale value into every
    # later test in this process (pytest does not give each test function
    # its own Context). Confining it to `outer`, which nothing else ever
    # runs in, means that leak is inert: `outer` is simply discarded when
    # this function returns.
    def body() -> None:
        # Open the channel in `outer` (the Context `body` is running in).
        cm = reasoning_channel()
        ch = cm.__enter__()
        assert current_channel() is ch

        # Force the generator's `finally: _current.reset(token)` to run
        # inside a COPY of `outer` rather than `outer` itself — the same
        # shape as asyncio scheduling an abandoned async generator's
        # aclose() as a new Task at GC time. Must NOT raise ValueError
        # (pre-fix, this line does).
        inner = contextvars.copy_context()
        inner.run(cm.__exit__, None, None, None)

        # `outer`'s ContextVar must remain usable afterward: opening and
        # closing a brand-new channel here works cleanly end to end. (We do
        # NOT assert current_channel() is None here — because the
        # cross-context reset() above could not restore `outer`'s prior
        # value, `outer` is left holding the stale `ch` as far as
        # `_current` is concerned. That's the documented #21 tradeoff: "the
        # var dies with that context" — only the ABANDONED context's
        # bookkeeping is off, and nothing reads `_current` from it again in
        # production either, since the request/Task it belonged to is
        # already gone. The only thing that matters is that later usage of
        # the ContextVar keeps working, which this proves.)
        with reasoning_channel() as ch2:
            assert current_channel() is ch2
            emit_reasoning("after-cross-context-exit")
            assert ch2.lines == ["after-cross-context-exit"]
        # Exiting a normal (same-context) `with` block still restores correctly.
        assert current_channel() is not ch2

    outer = contextvars.copy_context()
    outer.run(body)


# ── (b) integration smoke repro (best-effort; see docstring) ─────────────

def _install(monkeypatch, llm, script):
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def test_abandoned_astream_gc_does_not_log_cross_context_token_error(monkeypatch):
    """Real-world repro of the report: a ToolAgent.astream() generator is
    abandoned (``break`` without ``aclose()``) and later finalized by the
    GC, which — per asyncio's async-generator finalizer hook — runs the
    scheduled ``aclose()`` as a fresh Task, potentially in a different
    Context than the one that started the stream.

    Capture mechanism: the exact moment the GC-scheduled finalizer Task
    actually executes is driven by ``call_soon_threadsafe`` internals, not
    something a test can await deterministically — so this pumps the loop
    a few ticks after ``gc.collect()`` to give it a chance to run. When it
    DOES run (observed reliably in manual repro via a loop exception
    handler capturing "Task exception was never retrieved"), this asserts
    no ValueError about a cross-context Token was logged. If timing means
    the finalizer never got a chance to run within the pumped ticks, this
    test still passes trivially (nothing was captured) — the strong,
    non-flaky assertion for the underlying bug is
    ``test_exit_in_different_context_does_not_raise_and_var_stays_usable``
    above, which is what actually gates the fix.
    """
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return "sunny"

    class GcReproWeatherAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [get_weather]

    agent = get_registry().resolve("gcreproweatheragent")
    _install(
        monkeypatch,
        agent.llm,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Recife"}, "id": "c1"}],
            ),
            AIMessage(content="It's sunny."),
        ],
    )

    captured: list[dict] = []

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: captured.append(context))

        agen = agent.astream([Message(role="user", content="weather?")])
        await agen.__anext__()  # start the stream, get at least one chunk
        del agen  # abandon: no aclose(), no more references

        # Give the GC-driven async-generator finalizer a chance to run its
        # scheduled aclose() Task.
        for _ in range(10):
            gc.collect()
            await asyncio.sleep(0)

    asyncio.run(scenario())

    bad = [
        c for c in captured
        if isinstance(c.get("exception"), ValueError)
        and "different Context" in str(c["exception"])
    ]
    assert bad == [], f"cross-context Token.reset() error logged: {bad}"
