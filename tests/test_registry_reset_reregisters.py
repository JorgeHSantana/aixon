# tests/test_registry_reset_reregisters.py
"""reset_registry() must also clear Agent._registered so a class can re-register
into the fresh registry (audit 3.6). Before the fix, the stale class flag made
Agent.__init__ short-circuit and the registry stayed empty."""
from __future__ import annotations

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.registry import get_registry, reset_registry


def _define_worker():
    return type(
        "ResetProbeAgent",
        (Agent,),
        {
            "name": "reset-probe",
            "invoke": lambda self, messages: Message(role="assistant", content="ok"),
            "stream": lambda self, messages: iter([Chunk(done=True)]),
        },
    )


def test_reset_clears_registered_flag_and_allows_reregister():
    cls = _define_worker()                     # auto-registers on definition
    assert get_registry().resolve("reset-probe") is not None
    assert cls._registered is True

    reset_registry()

    # Registry is empty AND the class flag was cleared (not desynced).
    assert get_registry().all() == []
    assert cls._registered is False

    # Re-instantiating now re-registers into the fresh registry.
    cls()
    assert get_registry().resolve("reset-probe") is not None
