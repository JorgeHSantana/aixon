# tests/test_reasoning_providers.py
"""R1: declarative `reasoning` knob + per-provider translation.

Covers:
- aixon.providers.base.normalize_reasoning / pop_reasoning (pure helpers).
- LLM(..., reasoning=...) plumbing: always injected into the params handed to
  Provider.build(), knob off -> byte-identical kwargs to pre-R1 behavior.
- Anthropic / OpenAI / z.AI / Google translation, using the same
  monkeypatch-the-vendor-class-and-record-kwargs pattern as
  tests/test_retriever_weaviate.py (_FakeVS): each fake vendor class just
  records **kwargs on itself so assertions inspect exactly what the provider
  would have handed the real SDK class.
"""
from __future__ import annotations

import logging

import pytest

from aixon.providers.base import normalize_reasoning, pop_reasoning


# ── normalize_reasoning (pure) ────────────────────────────────────────────────

def test_normalize_reasoning_none_is_off():
    assert normalize_reasoning(None) is None


def test_normalize_reasoning_false_is_off():
    assert normalize_reasoning(False) is None


def test_normalize_reasoning_true_is_medium():
    assert normalize_reasoning(True) == {"budget_tokens": 4096, "effort": "medium"}


@pytest.mark.parametrize(
    "effort,budget",
    [("low", 1024), ("medium", 4096), ("high", 16384)],
)
def test_normalize_reasoning_effort_fills_budget(effort, budget):
    assert normalize_reasoning({"effort": effort}) == {
        "budget_tokens": budget,
        "effort": effort,
    }


@pytest.mark.parametrize(
    "budget,effort",
    [(500, "low"), (1024, "low"), (1025, "medium"), (8192, "medium"), (8193, "high"), (20000, "high")],
)
def test_normalize_reasoning_budget_fills_effort(budget, effort):
    assert normalize_reasoning({"budget_tokens": budget}) == {
        "budget_tokens": budget,
        "effort": effort,
    }


def test_normalize_reasoning_both_given_kept_as_is():
    # Explicit dict wins outright — no derivation when both halves are given,
    # even if they disagree with the canonical table.
    assert normalize_reasoning({"budget_tokens": 100, "effort": "high"}) == {
        "budget_tokens": 100,
        "effort": "high",
    }


# ── pop_reasoning (pure) ──────────────────────────────────────────────────────

def test_pop_reasoning_extracts_key():
    params = {"reasoning": True, "temperature": 0.5}
    assert pop_reasoning(params) is True
    assert params == {"temperature": 0.5}


def test_pop_reasoning_absent_is_none():
    params = {"temperature": 0.5}
    assert pop_reasoning(params) is None
    assert params == {"temperature": 0.5}


# ── LLM plumbing (fake provider) ──────────────────────────────────────────────

from tests._fakes import FakeProvider, make_llm  # noqa: E402  (registers fake provider)


def test_llm_default_reasoning_is_none():
    llm = make_llm()
    assert llm.reasoning is None


def test_llm_stores_reasoning_spec():
    llm = make_llm(reasoning=True)
    assert llm.reasoning is True


@pytest.fixture
def recording_fake_build(monkeypatch):
    """Record the params FakeProvider.build receives, per call."""
    captured: list[dict] = []
    original_build = FakeProvider.build

    def recording_build(self, model, **params):
        captured.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", recording_build)
    return captured


def test_supporting_provider_build_receives_reasoning_key(
    monkeypatch, recording_fake_build
):
    """For a provider that declares supports_reasoning=True, LLM always
    injects params["reasoning"] before build() — even when the knob was never
    set (None) — so the provider can uniformly pop it."""
    monkeypatch.setattr(FakeProvider, "supports_reasoning", True, raising=False)

    llm = make_llm()
    llm.chat_model
    assert recording_fake_build[-1]["reasoning"] is None


def test_supporting_provider_build_passes_through_explicit_reasoning(
    monkeypatch, recording_fake_build
):
    monkeypatch.setattr(FakeProvider, "supports_reasoning", True, raising=False)

    llm = make_llm(reasoning={"effort": "high"})
    llm.chat_model
    assert recording_fake_build[-1]["reasoning"] == {"effort": "high"}


def test_unsupporting_provider_never_receives_reasoning_key(recording_fake_build):
    """Rule 5: FakeProvider does NOT declare supports_reasoning, standing in
    for any custom provider that blindly forwards **params to a strict vendor
    constructor. The stray "reasoning" key must never reach its build()."""
    assert getattr(FakeProvider, "supports_reasoning", False) is False

    llm = make_llm()
    llm.chat_model
    assert "reasoning" not in recording_fake_build[-1]


def test_unsupporting_provider_with_reasoning_on_warns_and_does_not_crash(
    recording_fake_build, caplog
):
    """Rule 5: knob on + provider without support -> build still succeeds,
    receives NO reasoning key, and a warning names the provider."""
    llm = make_llm(reasoning=True)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        model = llm.chat_model  # must not raise
    assert model is not None
    assert "reasoning" not in recording_fake_build[-1]
    assert any(
        "does not support reasoning" in m and "fake" in m for m in caplog.messages
    )


def test_unsupporting_provider_reasoning_off_does_not_warn(
    recording_fake_build, caplog
):
    llm = make_llm()
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        llm.chat_model
    assert caplog.messages == []


def test_unsupporting_provider_request_chat_model_path_also_guarded(
    recording_fake_build, caplog
):
    """request_chat_model (the per-request-params path) applies the same
    rule-5 guard as chat_model."""
    from aixon.runtime import generation_params

    llm = make_llm(reasoning=True)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with generation_params({"temperature": 0.3}):
            llm.request_chat_model()  # must not raise
    assert "reasoning" not in recording_fake_build[-1]
    assert recording_fake_build[-1]["temperature"] == 0.3
    assert any("does not support reasoning" in m for m in caplog.messages)


def test_shipped_providers_declare_reasoning_support():
    from aixon.providers.anthropic import AnthropicProvider
    from aixon.providers.google import GoogleProvider
    from aixon.providers.openai import OpenAIProvider
    from aixon.providers.zai import ZAIProvider

    for cls in (AnthropicProvider, OpenAIProvider, ZAIProvider, GoogleProvider):
        assert cls.supports_reasoning is True


# ── Anthropic translation ─────────────────────────────────────────────────────

class _FakeChatAnthropic:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_anthropic(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", _FakeChatAnthropic)
    from aixon.providers.anthropic import AnthropicProvider

    return AnthropicProvider()


def test_anthropic_reasoning_off_is_byte_identical(fake_anthropic):
    model = fake_anthropic.build("claude-sonnet-5", temperature=0.7)
    assert "thinking" not in model.kwargs
    assert model.kwargs["temperature"] == 0.7
    assert "max_tokens" not in model.kwargs


def test_anthropic_reasoning_true_sets_thinking_and_temperature(fake_anthropic):
    model = fake_anthropic.build("claude-sonnet-5", reasoning=True)
    assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert model.kwargs["temperature"] == 1
    assert model.kwargs["max_tokens"] == 4096 + 4096


def test_anthropic_reasoning_budget_spec(fake_anthropic):
    model = fake_anthropic.build("claude-sonnet-5", reasoning={"budget_tokens": 2000})
    assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert model.kwargs["max_tokens"] == 2000 + 4096


def test_anthropic_reasoning_effort_spec(fake_anthropic):
    model = fake_anthropic.build("claude-sonnet-5", reasoning={"effort": "high"})
    assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert model.kwargs["max_tokens"] == 16384 + 4096


def test_anthropic_reasoning_forces_temperature_and_warns(fake_anthropic, caplog):
    with caplog.at_level(logging.WARNING, logger="aixon.providers.anthropic"):
        model = fake_anthropic.build("claude-sonnet-5", reasoning=True, temperature=0.7)
    assert model.kwargs["temperature"] == 1
    assert any("temperature" in m for m in caplog.messages)


def test_anthropic_reasoning_no_warning_when_temperature_absent(fake_anthropic, caplog):
    with caplog.at_level(logging.WARNING, logger="aixon.providers.anthropic"):
        model = fake_anthropic.build("claude-sonnet-5", reasoning=True)
    assert model.kwargs["temperature"] == 1
    assert caplog.messages == []


def test_anthropic_max_tokens_elevated_when_absent_or_low(fake_anthropic):
    model = fake_anthropic.build("claude-sonnet-5", reasoning=True, max_tokens=100)
    # 100 <= budget(4096) -> elevated to budget + 4096
    assert model.kwargs["max_tokens"] == 4096 + 4096


def test_anthropic_max_tokens_preserved_when_already_above_budget(fake_anthropic):
    model = fake_anthropic.build("claude-sonnet-5", reasoning=True, max_tokens=20000)
    assert model.kwargs["max_tokens"] == 20000


def test_anthropic_reasoning_effort_param_overrides_knob(fake_anthropic):
    """rule 6: a per-request `reasoning_effort` in params overrides the
    class-level `reasoning` knob for this one build."""
    model = fake_anthropic.build(
        "claude-sonnet-5", reasoning={"effort": "low"}, reasoning_effort="high"
    )
    assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "reasoning_effort" not in model.kwargs  # never leaks to the vendor ctor


# ── OpenAI translation ────────────────────────────────────────────────────────

class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_openai(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    from aixon.providers.openai import OpenAIProvider

    return OpenAIProvider()


def test_openai_reasoning_off_is_byte_identical(fake_openai):
    model = fake_openai.build("gpt-5.4", temperature=0.2)
    assert "reasoning_effort" not in model.kwargs
    assert model.kwargs["temperature"] == 0.2


def test_openai_reasoning_true_sets_reasoning_effort_medium(fake_openai):
    model = fake_openai.build("gpt-5.4", reasoning=True)
    assert model.kwargs["reasoning_effort"] == "medium"


def test_openai_reasoning_budget_spec_translates_to_effort(fake_openai):
    model = fake_openai.build("gpt-5.4", reasoning={"budget_tokens": 200})
    assert model.kwargs["reasoning_effort"] == "low"


def test_openai_reasoning_effort_param_overrides_knob(fake_openai):
    model = fake_openai.build(
        "gpt-5.4", reasoning={"effort": "low"}, reasoning_effort="high"
    )
    assert model.kwargs["reasoning_effort"] == "high"


# ── z.AI / GLM translation ────────────────────────────────────────────────────

@pytest.fixture
def fake_zai(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    from aixon.providers.zai import ZAIProvider

    return ZAIProvider()


def test_zai_reasoning_off_is_byte_identical(fake_zai):
    model = fake_zai.build("glm-5.2")
    assert "extra_body" not in model.kwargs


def test_zai_reasoning_any_spec_enables_thinking(fake_zai):
    # GLM has no budget/effort dial: any non-off spec just turns thinking on.
    model = fake_zai.build("glm-5.2", reasoning={"effort": "low"})
    assert model.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_zai_reasoning_true_enables_thinking(fake_zai):
    model = fake_zai.build("glm-5.2", reasoning=True)
    assert model.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_zai_reasoning_merges_with_existing_extra_body(fake_zai):
    model = fake_zai.build(
        "glm-5.2", reasoning=True, extra_body={"some_other_flag": 1}
    )
    assert model.kwargs["extra_body"] == {
        "some_other_flag": 1,
        "thinking": {"type": "enabled"},
    }


def test_zai_reasoning_effort_param_overrides_knob(fake_zai):
    model = fake_zai.build("glm-5.2", reasoning=None, reasoning_effort="low")
    assert model.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


# ── Google translation ────────────────────────────────────────────────────────

class _FakeChatGoogleGenerativeAI:
    # Mirrors the installed langchain-google-genai (4.2.5), whose pydantic
    # model exposes these fields — the provider probes model_fields before
    # emitting the kwargs (rule 4 graceful degradation).
    model_fields = {"thinking_budget": None, "include_thoughts": None}

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOldChatGoogleGenerativeAI:
    # Stands in for an older langchain-google-genai WITHOUT thinking support:
    # no thinking_budget/include_thoughts fields, and (being pydantic-strict)
    # it rejects unknown kwargs.
    model_fields: dict = {}

    def __init__(self, **kwargs):
        for key in ("thinking_budget", "include_thoughts"):
            if key in kwargs:
                raise TypeError(f"unexpected keyword argument {key!r}")
        self.kwargs = kwargs


@pytest.fixture
def fake_google(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI", _FakeChatGoogleGenerativeAI
    )
    from aixon.providers.google import GoogleProvider

    return GoogleProvider()


@pytest.fixture
def fake_old_google(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _FakeOldChatGoogleGenerativeAI,
    )
    from aixon.providers.google import GoogleProvider

    return GoogleProvider()


def test_google_reasoning_off_is_byte_identical(fake_google):
    model = fake_google.build("gemini-2.0-flash")
    assert "thinking_budget" not in model.kwargs
    assert "include_thoughts" not in model.kwargs


def test_google_reasoning_true_sets_thinking_budget(fake_google):
    model = fake_google.build("gemini-2.0-flash", reasoning=True)
    assert model.kwargs["thinking_budget"] == 4096
    assert model.kwargs["include_thoughts"] is True


def test_google_reasoning_effort_spec(fake_google):
    model = fake_google.build("gemini-2.0-flash", reasoning={"effort": "low"})
    assert model.kwargs["thinking_budget"] == 1024
    assert model.kwargs["include_thoughts"] is True


def test_google_reasoning_effort_param_overrides_knob(fake_google):
    model = fake_google.build(
        "gemini-2.0-flash", reasoning={"effort": "low"}, reasoning_effort="high"
    )
    assert model.kwargs["thinking_budget"] == 16384


def test_google_reasoning_degrades_when_installed_package_lacks_support(
    fake_old_google, caplog
):
    """Rule 4: an installed langchain-google-genai without thinking fields ->
    build succeeds WITHOUT the kwargs (the strict fake would raise on them)
    and the mandated warning is logged."""
    with caplog.at_level(logging.WARNING, logger="aixon.providers.google"):
        model = fake_old_google.build("gemini-2.0-flash", reasoning=True)  # no raise
    assert "thinking_budget" not in model.kwargs
    assert "include_thoughts" not in model.kwargs
    assert any(
        "reasoning not supported by installed langchain-google-genai" in m
        for m in caplog.messages
    )


def test_google_reasoning_off_on_old_package_neither_warns_nor_breaks(
    fake_old_google, caplog
):
    with caplog.at_level(logging.WARNING, logger="aixon.providers.google"):
        model = fake_old_google.build("gemini-2.0-flash")
    assert "thinking_budget" not in model.kwargs
    assert caplog.messages == []
