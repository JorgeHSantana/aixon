# tests/test_usage_accumulator.py
"""UsageAccumulator: thread-safety regression.

LangGraph runs Tier-2 fan-out nodes on real threads within the same superstep
(BackgroundExecutor -> ContextThreadPoolExecutor), so worker nodes can call
add_usage() concurrently against ONE accumulator. A non-atomic
read-modify-write in add() silently drops usage under that contention."""
from __future__ import annotations

import sys
import threading

from aixon.usage import UsageAccumulator


def test_add_is_thread_safe_under_contention():
    acc = UsageAccumulator()
    threads_n, adds_each = 16, 50
    barrier = threading.Barrier(threads_n)

    def hammer() -> None:
        barrier.wait()  # maximize overlap: all threads start adding at once
        for _ in range(adds_each):
            acc.add({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 1})

    threads = [threading.Thread(target=hammer) for _ in range(threads_n)]
    # Shrink the GIL switch interval so the interpreter preempts threads inside
    # add()'s read-merge-write window — without this, 800 sub-microsecond adds
    # rarely overlap and an unlocked add() would pass by luck most runs.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)

    expected = threads_n * adds_each  # 800: no add may be dropped
    assert acc.total == {
        "prompt_tokens": expected,
        "completion_tokens": expected,
        "total_tokens": expected,
    }


def test_empty_dict_counts_as_nothing_reported():
    # A worker returning usage={} must NOT materialize an all-zero total —
    # that would be truthy and silently suppress the server's tiktoken
    # estimate fallback (`if not usage:`).
    from aixon.usage import merge_usage

    assert merge_usage(None, {}) is None
    assert merge_usage({}, {}) is None
    acc = UsageAccumulator()
    acc.add({})
    assert acc.total is None
    acc.add({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    assert acc.total == {"prompt_tokens": 1, "completion_tokens": 2,
                         "total_tokens": 3}
