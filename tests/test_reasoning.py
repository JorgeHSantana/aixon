import contextvars

from aixon.reasoning import (
    ReasoningChannel,
    current_channel,
    emit_reasoning,
    reasoning_channel,
)


def test_emit_reasoning_is_noop_without_active_channel():
    # No channel active: must not raise, must not store anywhere.
    assert current_channel() is None
    emit_reasoning("ignored")  # no-op
    assert current_channel() is None


def test_channel_collects_emitted_lines():
    with reasoning_channel() as ch:
        assert current_channel() is ch
        emit_reasoning("step one")
        emit_reasoning("step two")
        assert ch.lines == ["step one", "step two"]


def test_drain_returns_and_clears():
    with reasoning_channel() as ch:
        emit_reasoning("a")
        emit_reasoning("b")
        assert ch.drain() == ["a", "b"]
        assert ch.lines == []
        emit_reasoning("c")
        assert ch.drain() == ["c"]


def test_channel_is_reset_after_context_exits():
    with reasoning_channel():
        assert current_channel() is not None
    assert current_channel() is None


def test_nested_channels_restore_outer_on_exit():
    with reasoning_channel() as outer:
        emit_reasoning("outer-1")
        with reasoning_channel() as inner:
            assert current_channel() is inner
            emit_reasoning("inner-1")
            assert inner.lines == ["inner-1"]
        # Inner exited: the outer channel is active again, unpolluted.
        assert current_channel() is outer
        assert outer.lines == ["outer-1"]


def test_contextvar_isolation_across_independent_contexts():
    # A copied context sees its own channel; the parent context is unaffected.
    results = {}

    def run_in_child():
        with reasoning_channel() as ch:
            emit_reasoning("child")
            results["child"] = ch.lines

    ctx = contextvars.copy_context()
    ctx.run(run_in_child)
    # Parent context never had a channel.
    assert current_channel() is None
    assert results["child"] == ["child"]
