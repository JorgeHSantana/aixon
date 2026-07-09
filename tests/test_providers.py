from __future__ import annotations

import importlib
import os

import pytest

from aixon.exceptions import AixonError
from aixon.providers.base import (
    Provider,
    get_provider,
    register_provider,
    resolve_provider_for_model,
)


# ── A throwaway provider just for registry tests ─────────────────────────────

class _DummyProvider(Provider):
    name = "dummy"
    env_key = "DUMMY_API_KEY"

    def build(self, model: str, **params):
        raise NotImplementedError  # never called in these tests


def test_register_and_get_provider():
    register_provider(_DummyProvider())
    p = get_provider("dummy")
    assert isinstance(p, _DummyProvider)


def test_get_unknown_provider_raises():
    with pytest.raises(AixonError, match="no-such-provider"):
        get_provider("no-such-provider")


# ── resolve_provider_for_model (concrete providers registered below) ─────────

@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "o3", "o1-mini", "text-davinci-003"])
def test_resolve_openai_models(model):
    importlib.import_module("aixon.providers.openai")  # self-registers
    assert resolve_provider_for_model(model).name == "openai"


@pytest.mark.parametrize("model", ["claude-3-5-sonnet-20241022", "claude-opus-4"])
def test_resolve_anthropic_models(model):
    importlib.import_module("aixon.providers.anthropic")
    assert resolve_provider_for_model(model).name == "anthropic"


@pytest.mark.parametrize("model", ["gemini-2.0-flash", "gemini-1.5-pro"])
def test_resolve_google_models(model):
    importlib.import_module("aixon.providers.google")
    assert resolve_provider_for_model(model).name == "google"


def test_resolve_unknown_model_raises():
    with pytest.raises(AixonError, match="Cannot infer"):
        resolve_provider_for_model("totally-unknown-model-xyz")


# ── Vendor build (skipped if SDK not installed) ──────────────────────────────

def test_openai_provider_build():
    pytest.importorskip("langchain_openai")
    importlib.import_module("aixon.providers.openai")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    model = get_provider("openai").build("gpt-4o-mini")
    assert hasattr(model, "invoke")


def test_anthropic_provider_build():
    pytest.importorskip("langchain_anthropic")
    importlib.import_module("aixon.providers.anthropic")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    model = get_provider("anthropic").build("claude-3-5-haiku-20241022")
    assert hasattr(model, "invoke")


def test_google_provider_build():
    pytest.importorskip("langchain_google_genai")
    importlib.import_module("aixon.providers.google")
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    model = get_provider("google").build("gemini-2.0-flash")
    assert hasattr(model, "invoke")


# ── Network-resilience defaults (timeout/max_retries) ────────────────────────

from aixon.providers.base import (  # noqa: E402
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    apply_resilience_defaults,
)


def test_apply_resilience_defaults_fills_when_absent():
    params: dict = {}
    apply_resilience_defaults(params)
    assert params["timeout"] == DEFAULT_TIMEOUT_S
    assert params["max_retries"] == DEFAULT_MAX_RETRIES


def test_apply_resilience_defaults_caller_wins():
    params = {"timeout": 7, "max_retries": 0}
    apply_resilience_defaults(params)
    assert params["timeout"] == 7  # caller value preserved
    assert params["max_retries"] == 0


def test_google_build_applies_default_timeout():
    pytest.importorskip("langchain_google_genai")
    importlib.import_module("aixon.providers.google")
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    model = get_provider("google").build("gemini-2.0-flash")
    # Without a timeout a stalled stream hangs forever; the provider now injects
    # a finite default so the request fails fast instead.
    assert model.timeout == DEFAULT_TIMEOUT_S
    assert model.max_retries == DEFAULT_MAX_RETRIES


def test_google_build_caller_timeout_overrides_default():
    pytest.importorskip("langchain_google_genai")
    importlib.import_module("aixon.providers.google")
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    model = get_provider("google").build("gemini-2.0-flash", timeout=5, max_retries=1)
    assert model.timeout == 5
    assert model.max_retries == 1


# ── z.AI (GLM) ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["glm-5.2", "glm-4.6"])
def test_resolve_zai_models(model):
    importlib.import_module("aixon.providers.zai")  # self-registers
    assert resolve_provider_for_model(model).name == "zai"


def test_zai_provider_build():
    pytest.importorskip("langchain_openai")
    importlib.import_module("aixon.providers.zai")
    os.environ.setdefault("ZAI_API_KEY", "test-key")
    model = get_provider("zai").build("glm-5.2")
    assert hasattr(model, "invoke")


def test_zai_build_returns_base_chat_model():
    pytest.importorskip("langchain_openai")
    importlib.import_module("aixon.providers.zai")
    os.environ.setdefault("ZAI_API_KEY", "test-key")
    from langchain_core.language_models.chat_models import BaseChatModel
    model = get_provider("zai").build("glm-5.2")
    assert isinstance(model, BaseChatModel)


def test_zai_build_points_to_zai_base_url_and_resilience():
    pytest.importorskip("langchain_openai")
    importlib.import_module("aixon.providers.zai")
    os.environ.setdefault("ZAI_API_KEY", "test-key")
    model = get_provider("zai").build("glm-5.2")
    assert "api.z.ai" in str(model.openai_api_base)
    assert model.request_timeout == DEFAULT_TIMEOUT_S
    assert model.max_retries == DEFAULT_MAX_RETRIES


def test_zai_base_url_env_override(monkeypatch):
    pytest.importorskip("langchain_openai")
    importlib.import_module("aixon.providers.zai")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setenv("ZAI_BASE_URL", "https://proxy.example.com/v4")
    model = get_provider("zai").build("glm-5.2")
    assert "proxy.example.com" in str(model.openai_api_base)
