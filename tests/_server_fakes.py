"""Hermetic fake agents for server tests. Plain Agent subclasses — no LLM, no
provider SDK, no network. They stand in for real LLM agents so the FastAPI
boundary can be exercised end-to-end with TestClient.

``EchoAgent.seen`` captures the exact ``messages`` list passed to ``invoke`` so
a test can assert the agent only ever received neutral ``Message[]`` (the
no-vendor-leak guarantee)."""

from __future__ import annotations

from typing import Iterator

from aixon.agent import Agent
from aixon.message import Chunk, Message


def _last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


class EchoAgent(Agent, abstract=True):
    """Abstract fake: echoes the last user message. Concrete via make_echo or a
    named subclass. ``seen`` records the last ``messages`` list received."""

    seen: list[Message] | None = None

    def invoke(self, messages: list[Message]) -> Message:
        self.seen = messages
        return Message(role="assistant", content="echo:" + _last_user_text(messages))

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        self.seen = messages
        text = "echo:" + _last_user_text(messages)
        # Two content deltas to prove deltas concatenate on the wire.
        yield Chunk(content=text[: len(text) // 2])
        yield Chunk(content=text[len(text) // 2 :])
        yield Chunk(done=True)


class ReasoningAgent(Agent, abstract=True):
    """Abstract fake that also emits reasoning, to prove reasoning survives a
    round trip through each dialect."""

    def invoke(self, messages: list[Message]) -> Message:
        return Message(
            role="assistant",
            content="answer",
            reasoning="because",
        )

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(reasoning="thinking...")
        yield Chunk(content="ans")
        yield Chunk(content="wer")
        yield Chunk(done=True)


def make_echo(name, *, aliases=(), hidden=False, description=""):
    """Define + register a fresh EchoAgent subclass at call time and return the
    registered instance. Defining at call time keeps each test's autouse
    reset_registry clean."""
    from aixon.registry import get_registry

    cls = type(
        "MadeEchoAgent",
        (EchoAgent,),
        {
            "name": name,
            "aliases": list(aliases),
            "hidden": hidden,
            "description": description,
        },
    )
    return get_registry().resolve(cls.name)
