# tests/test_tool_agent_client_system.py
"""0.1.28 — `ToolAgent.client_system`: what becomes of the system message a client sends.

A chat client (Open WebUI sends the model's or the user's system prompt this
way) puts a system message first in `messages`. By default, `client_system =
"replace"`, it takes the place of the agent's own `prompt`, as it always did.
With `"append"` the agent keeps its prompt and the client's system follows it,
after a blank line, as context: the agent's instructions come first and the
client's text cannot displace them.

Everything is offline. The model is `tests/_fakes.RecordingProvider` ("rec", the
`rec` fixture), whose `calls` hold what the model really received."""
from __future__ import annotations

import asyncio

import pytest

from aixon import ToolAgent
from aixon.exceptions import AgentNotFoundError, AixonError
from aixon.llm import LLM
from aixon.message import Message
from aixon.registry import get_registry

CLIENT = [
    Message(role="system", content="CONTEXTO DO CLIENTE"),
    Message(role="user", content="oi"),
]
ENTRIES = ["invoke", "ainvoke", "stream", "astream"]


def make_agent(rec, **attrs):
    """A new ToolAgent on the model of `rec`, with a prompt of its own. `attrs`
    add to the class body, or override it."""
    body = {
        "llm": LLM("tier-x", provider=rec.name),
        "prompt": "PROMPT DO AGENTE",
        **attrs,
    }
    type("ContextoAgent", (ToolAgent,), body)
    return get_registry().resolve("contextoagent")


def systems(rec) -> list[str]:
    """The contents of the system messages of the model's first call."""
    return [m.content for m in rec.model.calls[0] if m.type == "system"]


def run(agent, entry: str, messages: list[Message]) -> None:
    """Drive `agent` through `entry` to the end. The async entries run on an
    event loop of their own."""
    if entry == "invoke":
        agent.invoke(messages)
    elif entry == "stream":
        list(agent.stream(messages))
    else:
        async def go() -> None:
            if entry == "ainvoke":
                await agent.ainvoke(messages)
            else:
                [chunk async for chunk in agent.astream(messages)]

        asyncio.run(go())


# ── "append" ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role", ["system", "developer"])
def test_append_keeps_agent_prompt_first(rec, role):
    agent = make_agent(rec, client_system="append")

    agent.invoke([
        Message(role=role, content="CONTEXTO DO CLIENTE"),
        Message(role="user", content="oi"),
    ])

    assert systems(rec) == ["PROMPT DO AGENTE\n\nCONTEXTO DO CLIENTE"]
    # the client's system is consumed, not sent twice; the conversation is intact
    assert [m.type for m in rec.model.calls[0]] == ["system", "human"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_append_holds_on_every_entry_point(rec, entry):
    run(make_agent(rec, client_system="append"), entry, CLIENT)

    assert systems(rec) == ["PROMPT DO AGENTE\n\nCONTEXTO DO CLIENTE"]


@pytest.mark.parametrize(
    "prompt, client, expected",
    [
        ("PROMPT DO AGENTE", "", "PROMPT DO AGENTE"),
        ("", "CONTEXTO DO CLIENTE", "CONTEXTO DO CLIENTE"),
    ],
    ids=["client-empty", "prompt-empty"],
)
def test_append_with_one_side_empty(rec, prompt, client, expected):
    agent = make_agent(rec, client_system="append", prompt=prompt)

    agent.invoke([
        Message(role="system", content=client),
        Message(role="user", content="oi"),
    ])

    assert systems(rec) == [expected]  # the side that is there, with no blank lines around it


def test_append_with_both_sides_empty_sends_no_system(rec):
    agent = make_agent(rec, client_system="append", prompt="")

    agent.invoke([
        Message(role="system", content=""),
        Message(role="user", content="oi"),
    ])

    assert [m.type for m in rec.model.calls[0]] == ["human"]


def test_append_without_leading_system(rec):
    make_agent(rec, client_system="append").invoke([Message(role="user", content="oi")])

    assert systems(rec) == ["PROMPT DO AGENTE"]


# ── "replace" (today's behavior) ─────────────────────────────────────────────

@pytest.mark.parametrize("attrs", [{}, {"client_system": "replace"}], ids=["default", "explicit"])
def test_replace_is_default_and_unchanged(rec, attrs):
    make_agent(rec, **attrs).invoke(CLIENT)

    assert systems(rec) == ["CONTEXTO DO CLIENTE"]


def test_replace_with_empty_client_system_keeps_agent_prompt(rec):
    make_agent(rec).invoke([
        Message(role="system", content=""),
        Message(role="user", content="oi"),
    ])

    assert systems(rec) == ["PROMPT DO AGENTE"]


# ── declaration ──────────────────────────────────────────────────────────────

def test_invalid_client_system_rejected_at_registration(rec):
    with pytest.raises(AixonError, match=r"client_system.*'replace'.*'append'"):
        make_agent(rec, client_system="merge")

    with pytest.raises(AgentNotFoundError):  # refused before it was registered
        get_registry().resolve("contextoagent")
