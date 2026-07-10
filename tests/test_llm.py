# tests/test_llm.py
from __future__ import annotations

from langchain_core.messages import AIMessage

from tests._fakes import FakeChatModel, FakeProvider, make_llm  # registers fake provider
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.runtime import generation_params


# ── Construction + lazy build ─────────────────────────────────────────────────

def test_llm_construction_stores_model_and_params():
    llm = LLM("fake-1", provider="fake", temperature=0.5)
    assert llm.model == "fake-1"
    assert llm.params == {"temperature": 0.5}
    assert llm._provider_name == "fake"
    assert llm._chat_model is None  # not built yet


def test_chat_model_is_fake_chat_model():
    llm = LLM("fake-1", provider="fake")
    assert llm._chat_model is None
    cm = llm.chat_model
    assert isinstance(cm, FakeChatModel)
    assert llm._chat_model is cm  # cached


def test_chat_model_returns_same_instance_on_second_access():
    llm = LLM("fake-1", provider="fake")
    assert llm.chat_model is llm.chat_model


def test_llm_infers_provider_from_model_name():
    import aixon.providers  # registers openai/anthropic/google
    from aixon.providers.base import resolve_provider_for_model

    assert resolve_provider_for_model("gpt-4o").name == "openai"


# ── complete (offline) ────────────────────────────────────────────────────────

def test_complete_returns_neutral_message():
    llm = LLM("fake-1", provider="fake")
    llm.chat_model.script = [AIMessage(content="pong")]
    result = llm.complete([Message(role="user", content="ping")])
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "pong"


def test_complete_default_script_echoes_done():
    # Empty script -> FakeChatModel returns AIMessage("(done)")
    llm = make_llm()
    result = llm.complete([Message(role="user", content="x")])
    assert result.content == "(done)"


# ── stream (offline) ──────────────────────────────────────────────────────────

def test_stream_yields_content_then_done():
    llm = LLM("fake-1", provider="fake")
    llm.chat_model.script = [AIMessage(content="streamed")]
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any(c.content for c in chunks)
    assert chunks[-1].done is True


def test_stream_final_chunk_has_done_true():
    llm = make_llm()
    chunks = list(llm.stream([Message(role="user", content="x")]))
    assert chunks[-1].done is True
    for c in chunks[:-1]:
        assert c.done is False


# ── request_chat_model caching ────────────────────────────────────────────────

def test_request_chat_model_reuses_model_for_same_params(monkeypatch):
    build_calls: list[dict] = []
    original_build = FakeProvider.build

    def counting_build(self, model, **params):
        build_calls.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", counting_build)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"temperature": 0.3}):
        m1 = llm.request_chat_model()
        m2 = llm.request_chat_model()

    assert m1 is m2
    assert len(build_calls) == 1


def test_request_chat_model_rebuilds_for_different_params(monkeypatch):
    build_calls: list[dict] = []
    original_build = FakeProvider.build

    def counting_build(self, model, **params):
        build_calls.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", counting_build)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"temperature": 0.3}):
        m1 = llm.request_chat_model()
    with generation_params({"temperature": 0.9}):
        m2 = llm.request_chat_model()

    assert m1 is not m2
    assert len(build_calls) == 2
