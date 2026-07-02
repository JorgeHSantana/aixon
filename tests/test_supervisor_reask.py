# tests/test_supervisor_reask.py
"""Supervisor routing hardening: whole-word matching, ambiguity detection,
and a single strict retry before the safety net."""
from __future__ import annotations

from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Chunk, Message
from aixon.registry import get_registry


class _RawScripted:
    """Returns scripted replies in order, recording every call. No implicit
    DONE logic — the script IS the whole conversation policy."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.asks = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> Message:
        self.asks += 1
        self.calls.append(list(messages))
        content = self.replies.pop(0) if self.replies else "DONE"
        return Message(role="assistant", content=content)


def _workers() -> list:
    for name, tag in (("billing", "[billing]"), ("shipping", "[shipping]")):
        type(
            name.capitalize() + "Agent",
            (Agent,),
            {
                "name": name,
                "description": f"handles {name}",
                "invoke": lambda self, messages, _t=tag: Message(
                    role="assistant", content=_t
                ),
                "stream": lambda self, messages, _t=tag: iter(
                    [Chunk(content=_t), Chunk(done=True)]
                ),
            },
        )
    return [get_registry().resolve("billing"), get_registry().resolve("shipping")]


def test_ambiguous_reply_reasks_once_and_routes_per_strict_reply():
    # Reply 1 names BOTH workers -> ambiguous -> strict retry -> "billing".
    # Pre-fix the longest-name substring fallback silently picked "shipping".
    sup = _RawScripted(["billing or shipping, hmm", "billing", "DONE"])

    class ReaskOrchestrator(Orchestrator):
        supervisor = sup
        agents = _workers()

    out = get_registry().resolve("reaskorchestrator").invoke(
        [Message(role="user", content="help")]
    )
    assert out.content == "[billing]"
    assert sup.asks == 3  # ambiguous ask + strict retry + post-answer DONE
    # The retry carried the corrective instruction as a trailing system msg.
    assert sup.calls[1][-1].role == "system"
    assert "EXACTLY one of" in sup.calls[1][-1].content


def test_unparseable_replies_reask_once_then_safety_net():
    # Neither reply parses -> "" -> _route's safety net still serves the
    # unanswered user turn with the first worker instead of stranding it.
    sup = _RawScripted(["gibberish", "still gibberish", "DONE"])

    class GibberishOrchestrator(Orchestrator):
        supervisor = sup
        agents = _workers()

    out = get_registry().resolve("gibberishorchestrator").invoke(
        [Message(role="user", content="help")]
    )
    assert out.content == "[billing]"
    assert sup.asks == 3  # first ask + strict retry (then DONE after answer)


def test_name_inside_longer_word_does_not_route():
    # "billings" must NOT fire the "billing" worker (pre-fix substring did).
    # The turn is already answered, so after the failed retry the router ENDs.
    sup = _RawScripted(["the billings look fine", "DONE"])

    class BoundaryOrchestrator(Orchestrator):
        supervisor = sup
        agents = _workers()

    out = get_registry().resolve("boundaryorchestrator").invoke(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="prior answer"),
        ]
    )
    assert out.content != "[billing]"
