"""Hermetic fake agents for server tests. Plain Agent subclasses — no LLM, no
provider SDK, no network. They stand in for real LLM agents so the FastAPI
boundary can be exercised end-to-end with TestClient.

``EchoAgent.seen`` captures the exact ``messages`` list passed to ``invoke`` so
a test can assert the agent only ever received neutral ``Message[]`` (the
no-vendor-leak guarantee).

``make_failing`` / ``make_whoami`` serve the gateway tests (an agent that raises
a given error; an agent that answers with the e-mail the Server published for the
request); ``sse_datas`` / ``sse_content`` read an SSE body back."""

from __future__ import annotations

import json
from typing import Iterator

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.runtime import current_user_email


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


class FailingAgent(Agent, abstract=True):
    """Abstract fake that fails with ``exc``. ``stream`` first emits a reasoning
    chunk (the label of a specialist being called, which a ``content``-mode
    stream turns into an OPEN ``<think>`` block) and then raises; ``invoke``
    raises at once. Concrete via make_failing."""

    exc: BaseException

    def invoke(self, messages: list[Message]) -> Message:
        raise self.exc

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(reasoning="Calling especialista...\n")
        raise self.exc


class WhoAmIAgent(Agent, abstract=True):
    """Abstract fake that answers with the user e-mail the Server published for
    the request: ``str(current_user_email())``, so ``"None"`` when there is none.
    Concrete via make_whoami."""

    def invoke(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content=str(current_user_email()))

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(content=str(current_user_email()))
        yield Chunk(done=True)


def make_failing(name: str, exc: BaseException) -> Agent:
    """Define + register an agent whose ``invoke`` raises ``exc`` and whose
    ``stream`` emits one reasoning chunk, "Calling especialista...", and then
    raises ``exc``. Returns the registered instance."""
    from aixon.registry import get_registry

    cls = type("MadeFailingAgent", (FailingAgent,), {"name": name, "exc": exc})
    return get_registry().resolve(cls.name)


def make_whoami(name: str) -> Agent:
    """Define + register an agent that answers ``str(current_user_email())`` in
    ``invoke`` and in ``stream``. Returns the registered instance."""
    from aixon.registry import get_registry

    cls = type("MadeWhoAmIAgent", (WhoAmIAgent,), {"name": name})
    return get_registry().resolve(cls.name)


def sse_datas(text: str) -> list[str]:
    """The ``data: `` lines of an SSE body, without the prefix."""
    return [l[len("data: ") :] for l in text.splitlines() if l.startswith("data: ")]


def sse_content(text: str) -> str:
    """The ``choices[0].delta.content`` of every JSON ``data:`` line of an OpenAI
    SSE body, joined. ``[DONE]`` and the chunks without ``choices`` (usage and
    error events) are skipped."""
    parts: list[str] = []
    for data in sse_datas(text):
        if data == "[DONE]":
            continue
        choices = json.loads(data).get("choices")
        if not choices:
            continue
        parts.append(choices[0]["delta"].get("content") or "")
    return "".join(parts)
