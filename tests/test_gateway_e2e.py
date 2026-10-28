# tests/test_gateway_e2e.py
"""0.1.28 — end to end: Server -> imposed model -> gateway.

A chat UI posts a chat completion with the user's e-mail in a header. The
coordinator, a ToolAgent that imposes its gateway tier on its whole tree, asks a
specialist (declared on OpenAI) for data and answers. Nothing between the Server
and the wire is stubbed: the real Server, the real agents, the real `ChatOpenAI`
and the real `openai` SDK run on top of `tests/_fakes.FakeGatewayWire`, so
`wire.requests` holds every HTTP request that would have left the process,
exactly as the SDK wrote it.

The specialist's own provider is booby-trapped (`explodiu`) and OPENAI_API_KEY
holds a value that must never show up on the wire, so a call that escapes the
gateway fails the test whether the tool shield swallows the error or not.

What the tests of each piece cannot show, and these do: the model and the
identity reach the call of EVERY agent of the tree, on the path the Server really
runs (the async one); and a refusal by the gateway, even one that a specialist
meets and not the agent the user talks to, ends the request with the Server's own
message instead of a generic error or a `TOOL ERROR` that the model reads and goes
on from.

Offline: nothing resolves `gateway.test`.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from aixon import ToolAgent
from aixon.llm import LLM
from aixon.runtime import user_email_scope
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._fakes import BUDGET, FakeGatewayWire
from tests._server_fakes import sse_content, sse_datas

CHAT = "/v1/chat/completions"
ANA = {"x-openwebui-user-email": "ana@example.com"}


@dataclass(frozen=True)
class E2E:
    """What the `e2e` fixture hands out."""

    wire: FakeGatewayWire
    app: Any
    client: TestClient
    explodiu: list[tuple[str, str]]


@pytest.fixture
def e2e(monkeypatch, explodiu):
    """A real Server over a coordinator that imposes the gateway tier "tier-x"
    (`client_system="append"`, and a `max_retries=0` that must stay behind) and
    calls a specialist declared on OpenAI. `wire` plays the gateway and
    `explodiu` lists every use of a provider an agent was declared with."""
    # Server is a singleton: left built with other messages by an earlier test,
    # the construction below with "SEM COTA" would raise.
    Server._reset()
    wire = FakeGatewayWire().install(monkeypatch)
    # The gateway must never read it: on the wire it would show as the Bearer key.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")

    class EspecialistaAgent(ToolAgent):
        name = "especialista-e2e"
        llm = LLM("gpt-5.4", provider="openai")

    class CoordenadorAgent(ToolAgent):
        name = "coordenador-e2e"
        # max_retries is not a portable param: it must not reach the imposed
        # model (test_e2e_rate_limited_non_stream proves it).
        llm = LLM("tier-x", provider="gateway", max_retries=0)
        prompt = "PROMPT"
        impose_model = True
        client_system = "append"
        tools = [EspecialistaAgent().as_tool(name="especialista")]

    app = Server(
        adapters=[OpenAIAdapter()],
        budget_exceeded_message="SEM COTA",
        rate_limited_message="DEVAGAR",
    ).app
    yield E2E(wire=wire, app=app, client=TestClient(app), explodiu=explodiu)
    Server._reset()


def chat_body(*, stream: bool) -> dict:
    return {
        "model": "coordenador-e2e",
        "stream": stream,
        "messages": [
            {"role": "system", "content": "CONTEXTO"},
            {"role": "user", "content": "pergunta"},
        ],
    }


def specialist_call() -> dict:
    """The gateway's answer that has the coordinator call its specialist."""
    return FakeGatewayWire.completion(
        tool_calls=[{"name": "especialista", "args": {"text": "q"}}]
    )


def queue_delegation(wire: FakeGatewayWire) -> None:
    """The three answers of a request that goes through the specialist: the
    coordinator asks for it, it answers, the coordinator closes."""
    wire.queue(body=specialist_call())
    wire.queue(body=FakeGatewayWire.completion("dado interno"))
    wire.queue(body=FakeGatewayWire.completion("resposta final"))


def assert_delegation_on_the_wire(wire: FakeGatewayWire, email: str) -> None:
    """The three HTTP requests of `queue_delegation`, as the SDK sent them: all
    to the gateway, with its key, on its tier, for `email`."""
    reqs = wire.requests
    assert len(reqs) == 3
    assert {str(q.url) for q in reqs} == {"http://gateway.test/v1/chat/completions"}
    assert {q.headers["authorization"] for q in reqs} == {"Bearer gw-test-key"}
    assert [q.headers.get("x-openwebui-user-email") for q in reqs] == [email] * 3
    # the specialist is declared on gpt-5.4 / openai: it ran on the tier anyway
    assert {json.loads(q.content)["model"] for q in reqs} == {"tier-x"}
    # the client's system message came after the agent's own prompt (append)
    assert json.loads(reqs[0].content)["messages"][0]["content"] == "PROMPT\n\nCONTEXTO"
    # and the tree really ran: the specialist was asked "q" (the 2nd call) and
    # what it answered went back to the coordinator as the tool result (the 3rd)
    assert json.loads(reqs[1].content)["messages"][-1]["content"] == "q"
    assert json.loads(reqs[2].content)["messages"][-1]["content"] == "dado interno"


# --- every call of the tree goes to the gateway, for the user ---------------
def test_e2e_stream_routes_every_call_to_gateway_with_identity(e2e):
    queue_delegation(e2e.wire)

    r = e2e.client.post(CHAT, json=chat_body(stream=True), headers=ANA)

    assert r.status_code == 200
    assert sse_content(r.text) == "resposta final"
    assert sse_datas(r.text)[-1] == "[DONE]"
    assert_delegation_on_the_wire(e2e.wire, "ana@example.com")
    assert e2e.explodiu == []


def test_e2e_non_stream_same_routing(e2e):
    queue_delegation(e2e.wire)

    r = e2e.client.post(CHAT, json=chat_body(stream=False), headers=ANA)

    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "resposta final"
    assert_delegation_on_the_wire(e2e.wire, "ana@example.com")
    assert e2e.explodiu == []


# --- the gateway says no ----------------------------------------------------
def test_e2e_budget_mid_tree_stream(e2e):
    # The budget is spent when the SPECIALIST calls the model, one level below
    # the agent the user talks to.
    e2e.wire.queue(body=specialist_call())
    e2e.wire.queue(422, {"error": {"message": BUDGET}})

    r = e2e.client.post(CHAT, json=chat_body(stream=True), headers=ANA)

    assert r.status_code == 200
    assert sse_content(r.text) == "SEM COTA"
    datas = sse_datas(r.text)
    assert json.loads(datas[-2])["choices"][0]["finish_reason"] == "stop"
    assert datas[-1] == "[DONE]"
    # a chat message of the Server's, never the generic error event nor the gateway's text
    assert "server_error" not in r.text and "ExceededBudget" not in r.text
    reqs = e2e.wire.requests
    assert len(reqs) == 2  # the SDK does not repeat a 422
    # The 2nd call is the specialist's own ("q"), not the coordinator going on
    # after a TOOL ERROR, which would carry the error text as its last message.
    assert json.loads(reqs[1].content)["messages"][-1]["content"] == "q"
    assert all(b"TOOL ERROR" not in q.content for q in reqs)
    assert e2e.explodiu == []


def test_e2e_rate_limited_non_stream(e2e):
    # "1", not "0": the SDK only honors a Retry-After above zero (with zero it
    # falls back to its exponential backoff, a second or more of sleeping).
    for _ in range(3):
        e2e.wire.queue(429, {"error": {"message": "slow"}}, headers={"retry-after-ms": "1"})

    r = e2e.client.post(CHAT, json=chat_body(stream=False), headers=ANA)

    assert r.status_code == 429
    assert r.json() == {
        "error": {"message": "DEVAGAR", "type": "rate_limit_exceeded", "code": "rate_limited"}
    }
    # 1 + DEFAULT_MAX_RETRIES (2). The class declares max_retries=0, but that is
    # not a portable param: had it reached the imposed model, this would be 1.
    assert len(e2e.wire.requests) == 3
    assert e2e.explodiu == []


# --- who the call is for ----------------------------------------------------
def test_e2e_without_email_sends_no_identity(e2e):
    queue_delegation(e2e.wire)

    r = e2e.client.post(CHAT, json=chat_body(stream=True))

    assert r.status_code == 200
    assert sse_content(r.text) == "resposta final"
    assert len(e2e.wire.requests) == 3  # the whole tree ran, and none of it carries one
    assert all("x-openwebui-user-email" not in q.headers for q in e2e.wire.requests)
    assert e2e.explodiu == []


@pytest.mark.parametrize("stream", [False, True])
def test_e2e_concurrent_requests_keep_their_own_identity(e2e, stream):
    # One agent, hence one LLM, one cached model and one HTTP client, serves
    # three people at once. The e-mail has to come from the context of the
    # request that makes the call, at the moment it is sent: baked in when the
    # model was built, or held in anything the requests share, a call would be
    # billed to somebody else.
    class EcoAgent(ToolAgent):
        name = "eco-e2e"
        llm = LLM("tier-x", provider="gateway")
        impose_model = True

    e2e.wire.delay = 0.05  # keeps the three requests in flight together
    callers = [
        ("ana", {"x-openwebui-user-email": "ana@example.com"}),
        ("bob", {"x-openwebui-user-email": "bob@example.com"}),
        ("anon", {}),
    ]

    def post(http: httpx.AsyncClient, who: str, headers: dict):
        # each caller says their own name, to tell their calls apart on the wire
        body = {
            "model": "eco-e2e",
            "stream": stream,
            "messages": [{"role": "user", "content": who}],
        }
        return http.post(CHAT, json=body, headers=headers)

    async def fire() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=e2e.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://app.test") as http:
            # Fired from a context where somebody else's e-mail is already in
            # force: the request without one must not inherit it.
            with user_email_scope("stale@example.com"):
                return await asyncio.gather(*(post(http, who, h) for who, h in callers))

    responses = asyncio.run(fire())

    assert [r.status_code for r in responses] == [200, 200, 200]
    sent = e2e.wire.requests
    assert Counter(q.headers.get("x-openwebui-user-email") for q in sent) == Counter(
        {"ana@example.com": 1, "bob@example.com": 1, None: 1}
    )
    # and each call carries the e-mail of the request it was made for
    email_of_call = {
        json.loads(q.content)["messages"][-1]["content"]: q.headers.get("x-openwebui-user-email")
        for q in sent
    }
    assert email_of_call == {"ana": "ana@example.com", "bob": "bob@example.com", "anon": None}
    assert e2e.explodiu == []
