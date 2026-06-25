# tests/test_orchestrator_tier1_routing.py
"""Tier 1 dynamic routing — the supervisor LLM decides who acts each turn.

These encode the three bugs of the old round-robin stub as regressions:
  1. supervisor ignored  -> always the last worker, regardless of content;
  2. multi-turn skip      -> a prior assistant message skipped workers;
  3. stale return         -> enough prior assistants returned old history verbatim.

The supervisor is a duck-typed fake (anything with .complete) so the test is
offline; workers return a self-identifying tag so we know who actually ran.
"""
from __future__ import annotations

from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Chunk, Message
from aixon.registry import get_registry


class _Supervisor:
    """Routes by keyword; says DONE once a worker has answered (last msg is
    assistant). Mirrors what a real LLM supervisor would decide."""

    def complete(self, messages: list[Message]) -> Message:
        if messages and messages[-1].role == "assistant":
            return Message(role="assistant", content="DONE")
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        t = (last_user or "").lower()
        if any(w in t for w in ("refund", "charge", "bill")):
            return Message(role="assistant", content="billing")
        if any(w in t for w in ("ship", "package", "track")):
            return Message(role="assistant", content="shipping")
        return Message(role="assistant", content="DONE")


def _make_worker(name: str, tag: str):
    type(
        name.capitalize() + "Agent",
        (Agent,),
        {
            "name": name,
            "description": f"handles {name}",
            "invoke": lambda self, messages: Message(role="assistant", content=tag),
            "stream": lambda self, messages: iter([Chunk(content=tag), Chunk(done=True)]),
        },
    )
    return get_registry().resolve(name)


def _orch():
    # Declaration order [billing, shipping] on purpose: the old stub would
    # always end on 'shipping' (the last worker), so a billing question that
    # returns the billing tag proves real content-routing.
    _make_worker("billing", "[billing handled it]")
    _make_worker("shipping", "[shipping handled it]")

    class RouteOrchestrator(Orchestrator):
        supervisor = _Supervisor()
        agents = [
            get_registry().resolve("billing"),
            get_registry().resolve("shipping"),
        ]

    return get_registry().resolve("routeorchestrator")


def test_routes_by_content_not_declaration_order():
    # Bug 1: billing question must reach billing, not the last-declared worker.
    out = _orch().invoke([Message(role="user", content="I was double charged, refund please")])
    assert out.content == "[billing handled it]"


def test_routes_shipping_question_to_shipping():
    out = _orch().invoke([Message(role="user", content="where is my package? track it")])
    assert out.content == "[shipping handled it]"


def test_prior_assistant_message_does_not_skip_workers():
    # Bug 2: a follow-up turn (history already has an assistant) must still route.
    out = _orch().invoke([
        Message(role="user", content="hi"),
        Message(role="assistant", content="an earlier answer"),
        Message(role="user", content="I need a refund"),
    ])
    assert out.content == "[billing handled it]"


def test_does_not_return_stale_history():
    # Bug 3: with several prior assistants, the old stub returned the last one
    # verbatim without running any worker. Now a worker actually handles it.
    out = _orch().invoke([
        Message(role="user", content="hi"),
        Message(role="assistant", content="r1"),
        Message(role="assistant", content="r2"),
        Message(role="user", content="track my package"),
    ])
    assert out.content == "[shipping handled it]"
    assert out.content != "r2"


def test_terminates_after_worker_answers():
    # The supervisor returns DONE once a worker has answered -> single handoff,
    # no loop to the recursion limit.
    out = _orch().invoke([Message(role="user", content="refund")])
    assert out.content == "[billing handled it]"


def test_overlapping_worker_names_route_to_most_specific():
    # A worker name that contains another ("order" vs "order-history") must not
    # mis-route: the longest matching name wins on the substring fallback.
    _make_worker("order", "[order]")
    _make_worker("order-history", "[order-history]")

    class _Sup:
        def complete(self, messages):
            if messages and messages[-1].role == "assistant":
                return Message(role="assistant", content="DONE")
            # reply is the longer name; must not match the shorter "order".
            return Message(role="assistant", content="order-history please")

    class OverlapOrchestrator(Orchestrator):
        supervisor = _Sup()
        agents = [
            get_registry().resolve("order"),
            get_registry().resolve("order-history"),
        ]

    out = get_registry().resolve("overlaporchestrator").invoke(
        [Message(role="user", content="anything")]
    )
    assert out.content == "[order-history]"
