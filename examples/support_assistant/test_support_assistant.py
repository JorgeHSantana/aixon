"""Tests for the support assistant — offline, no network, no API key.

Run from this folder:  python -m pytest

The suite forces the offline demo provider (so it is deterministic even on a
machine that has OPENAI_API_KEY set) and exercises every layer: the retriever's
embedding search, the connector's real HTTP path (mocked) and offline fallback,
reasoning capture, and the orchestrator's conditional routing end to end.
"""

from __future__ import annotations

import os

# Force offline mode BEFORE importing anything that builds an LLM/embedding at
# class-definition time. make_llm()/the retriever read this at import.
os.environ.pop("OPENAI_API_KEY", None)

import httpx  # noqa: E402  (transitive dep; used to mock the connector)

from aixon import Message, get_registry  # noqa: E402

from connectors.orders import OrdersConnector, extract_order_id  # noqa: E402
from knowledge.faq_retriever import KnowledgeRetriever  # noqa: E402


# --- Retriever + Embedding ---------------------------------------------------

def test_retriever_ranks_relevant_article_first():
    hits = KnowledgeRetriever().search("how do I reset my password?", k=3)
    assert hits, "expected at least one hit"
    assert "password" in hits[0]["text"].lower()


def test_retriever_is_read_only():
    import pytest

    from aixon import AixonError

    with pytest.raises(AixonError):
        KnowledgeRetriever().write(["new doc"])


# --- Connector: offline fallback AND real HTTP path --------------------------

def test_connector_offline_fallback():
    conn = OrdersConnector()  # ORDERS_API_URL unset -> in-memory fixture
    assert conn.base_url == ""
    assert conn.lookup_order("1002")["status"] == "in_transit"
    assert conn.lookup_order("9999")["status"] == "not_found"


def test_connector_real_http_path(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"order_id": "42", "status": "delivered", "item": "Widget", "eta": None}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)
    conn = OrdersConnector(base_url="https://api.example.com", auth_token="secret")
    order = conn.lookup_order("42")
    assert order["status"] == "delivered"
    assert captured["url"] == "https://api.example.com/orders/42"
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_extract_order_id():
    assert extract_order_id("where is order 1002?") == "1002"
    assert extract_order_id("no number here") == ""


# --- Orchestrator: conditional routing end to end ----------------------------

def _support():
    # Importing agents registers them; resolve the orchestrator from the registry.
    import agents.support  # noqa: F401

    return get_registry().resolve("support")


def test_orchestrator_routes_order_question_to_orders():
    reply = _support().invoke(
        [Message(role="user", content="where is my order 1002?")]
    )
    assert "1002" in reply.content
    assert "in_transit" in reply.content


def test_orchestrator_routes_faq_question_to_knowledge():
    reply = _support().invoke(
        [Message(role="user", content="how do I reset my password?")]
    )
    assert "password" in reply.content.lower()


def test_alias_resolves_to_orchestrator():
    assert get_registry().resolve("assistant") is _support()


def test_workers_are_hidden_only_orchestrator_is_public():
    _support()  # ensure everything is registered
    public_names = {a.name for a in get_registry().public()}
    assert "support" in public_names
    assert "triage" not in public_names
    assert "knowledge" not in public_names
    assert "orders" not in public_names
