# tests/test_llm.py
from __future__ import annotations

import logging

import pytest
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


# ── _bound_model: unified through request_chat_model (final-review CRITICAL) ──
#
# _bound_model() used to call `.bind(**params)` on the bare `chat_model`,
# which attaches params at INVOKE time, bypassing Provider.build() entirely —
# never reaching provider translation (reasoning_effort, temperature forcing,
# etc.). It now delegates straight to request_chat_model(), which merges
# per-request params in as constructor kwargs BEFORE calling build(), the
# same path ToolAgent already uses.


def test_bound_model_is_request_chat_model_no_params():
    """No active per-request params -> _bound_model returns the exact same
    cached chat_model request_chat_model would return (fast path preserved)."""
    llm = make_llm()
    assert llm._bound_model() is llm.chat_model


def test_bound_model_forwards_params_through_build_not_bind(monkeypatch):
    """The per-request params must reach Provider.build() as constructor
    kwargs (via request_chat_model), not get bound at invoke time. Under the
    old `.bind()` implementation, build() is only ever called with
    `self.params` (empty here) — this per-request `temperature` would never
    show up in `captured`."""
    captured: list[dict] = []
    original_build = FakeProvider.build

    def recording_build(self, model, **params):
        captured.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", recording_build)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"temperature": 0.3}):
        llm._bound_model()

    assert captured[-1]["temperature"] == 0.3


def test_bound_model_reuses_request_chat_model_cache(monkeypatch):
    """_bound_model and request_chat_model must share the same bounded model
    cache — calling one after the other with identical params must not force
    a second build."""
    build_calls: list[dict] = []
    original_build = FakeProvider.build

    def counting_build(self, model, **params):
        build_calls.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", counting_build)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"temperature": 0.3}):
        m1 = llm._bound_model()
        m2 = llm.request_chat_model()

    assert m1 is m2
    assert len(build_calls) == 1


def test_bound_model_never_calls_bind(monkeypatch):
    """No .bind() call anywhere in the _bound_model path — patch
    BaseChatModel.bind (which shadows Runnable.bind for every chat model,
    including FakeChatModel) to blow up if invoked."""
    from langchain_core.language_models.chat_models import BaseChatModel

    def _boom(self, **kwargs):
        raise AssertionError("`.bind()` must not be called by _bound_model anymore")

    monkeypatch.setattr(BaseChatModel, "bind", _boom)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"temperature": 0.3}):
        model = llm._bound_model()  # must not raise
    assert model is not None


# ── _bound_model + Anthropic: reasoning_effort/temperature reach translation ─
#
# Uses a plain (non-BaseChatModel) fake ChatAnthropic that only records
# constructor kwargs — it has no `.bind()` at all, so the OLD `.bind()`-based
# _bound_model would AttributeError on it the moment a per-request param was
# active. That is exactly the shape of the real crash: a client
# `reasoning_effort` never goes through `resolve_reasoning_spec`/provider
# translation under `.bind()`, it is handed raw to the vendor at invoke time.

class _FakeChatAnthropicCtor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_chat_anthropic_ctor(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", _FakeChatAnthropicCtor)
    import aixon.providers.anthropic  # noqa: F401  (ensure registered)

    return None


def test_bound_model_anthropic_reasoning_effort_translated_no_raw_leak(
    fake_chat_anthropic_ctor,
):
    llm = LLM("claude-sonnet-5", provider="anthropic", reasoning=True)
    with generation_params({"reasoning_effort": "high"}):
        model = llm._bound_model()

    assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "reasoning_effort" not in model.kwargs


def test_bound_model_anthropic_forces_temperature_despite_client_override(
    fake_chat_anthropic_ctor, caplog
):
    llm = LLM("claude-sonnet-5", provider="anthropic", reasoning=True)
    with caplog.at_level(logging.WARNING, logger="aixon.providers.anthropic"):
        with generation_params({"temperature": 0.2}):
            model = llm._bound_model()

    assert model.kwargs["temperature"] == 1
    assert any("temperature" in m for m in caplog.messages)


def test_bound_model_anthropic_plain_temperature_reaches_build(
    fake_chat_anthropic_ctor,
):
    """No reasoning active -> a plain client temperature still reaches
    build() as a constructor kwarg (LLMAgent with plain params still works)."""
    llm = LLM("claude-sonnet-5", provider="anthropic")
    with generation_params({"temperature": 0.2}):
        model = llm._bound_model()

    assert model.kwargs["temperature"] == 0.2
