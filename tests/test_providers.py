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
