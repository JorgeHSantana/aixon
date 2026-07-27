# tests/test_tool_agent_astream_task_exception.py
"""Sweep #1 (issue #20 integration sweep): ``ToolAgent.astream``'s poll loop
runs the next graph update in a background ``asyncio.Task`` (``next_task``,
#20 live-drain) and normally retrieves its outcome via ``task.result()``. But
if the CONSUMER abandons the generator (stops iterating / calls ``aclose()``)
in the exact tick where ``next_task`` just completed WITH AN EXCEPTION — right
after the reasoning-drain ``yield`` and before ``task.result()`` runs — that
exception was never fetched. asyncio then logs "Task exception was never
retrieved" when the Task is garbage-collected. The ``finally`` block must
retrieve it defensively (``next_task.exception()``) whenever the task is done
but not cancelled.

This reproduces the exact tick deterministically: a fake compiled graph whose
``astream()`` raises synchronously (no internal ``await``) on its first
``__anext__()``, so the task scheduled by ``asyncio.ensure_future`` completes
within the very first ``asyncio.wait(...)`` call — before the generator's
first ``yield`` is even reached by the test driver."""
from __future__ import annotations

import asyncio

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from aixon.reasoning import emit_reasoning
from tests._fakes import make_llm


class _FakeCompiledGraph:
    """Stand-in for the object ``langchain.agents.create_agent`` returns.
    ``astream`` yields nothing and raises on the very first advance, with NO
    ``await`` in between — so the wrapping Task finishes synchronously as
    soon as the event loop gets a chance to run it, deterministically inside
    the first ``asyncio.wait({next_task}, ...)`` in astream()'s poll loop."""

    async def astream(self, *args, **kwargs):
        # A line on the shared ReasoningChannel so the outer generator's
        # post-wait drain has something to yield BEFORE it would otherwise
        # reach `task.result()` — the exact abandon-here window under test.
        emit_reasoning("boom-tick")
        raise RuntimeError("boom")
        yield {}  # pragma: no cover — unreachable; makes this an async generator


def _agent():
    return type(
        "AstreamTaskExcAgent",
        (ToolAgent,),
        {"name": "astreamtaskexc", "llm": make_llm()},
    )()


def test_astream_retrieves_task_exception_when_abandoned_at_completion_tick(monkeypatch):
    import langchain.agents as la

    monkeypatch.setattr(la, "create_agent", lambda *a, **kw: _FakeCompiledGraph())

    created_tasks: list[asyncio.Task] = []
    real_ensure_future = asyncio.ensure_future

    def spy_ensure_future(coro_or_future, **kw):
        task = real_ensure_future(coro_or_future, **kw)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "ensure_future", spy_ensure_future)

    agent = _agent()

    async def scenario():
        it = agent.astream([Message(role="user", content="hi")])
        first = await it.__anext__()  # the drained "boom-tick" reasoning chunk
        # Abandon the generator right here: next_task is already done (with
        # RuntimeError("boom")) but astream() has not yet called
        # task.result() — the regression window.
        await it.aclose()
        return first

    first = asyncio.run(scenario())

    assert first.reasoning == "boom-tick\n"
    assert len(created_tasks) == 1
    task = created_tasks[0]
    assert task.done() and not task.cancelled()
    # asyncio.Task._log_traceback is the internal flag that drives the
    # "Task exception was never retrieved" warning from Task.__del__ on GC;
    # it flips to False the moment .exception()/.result() is called. It is a
    # private attribute, but it's the only deterministic (non-GC-timing,
    # non-log-capturing) signal that the finally block observed the
    # exception — asserting on the warning itself would depend on when the
    # garbage collector runs.
    assert task._log_traceback is False


def test_astream_still_cancels_a_pending_task_on_early_close(monkeypatch):
    # Unchanged-behavior guard: when the consumer abandons the generator
    # while next_task is still PENDING (not done), the existing
    # cancel-and-await path (not the new exception()-retrieval branch) runs.
    import langchain.agents as la

    class _SlowGraph:
        async def astream(self, *args, **kwargs):
            await asyncio.sleep(10)
            yield {}  # pragma: no cover

    monkeypatch.setattr(la, "create_agent", lambda *a, **kw: _SlowGraph())

    agent = _agent()

    async def scenario():
        it = agent.astream([Message(role="user", content="hi")])
        # Give the poll loop one tick to create next_task and start waiting,
        # then abandon before it ever completes.
        task = asyncio.ensure_future(it.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        import contextlib
        with contextlib.suppress(BaseException):
            await task
        await it.aclose()

    asyncio.run(scenario())  # must not raise / hang / warn
