# aixon/providers/anthropic.py
"""Anthropic provider — builds langchain_anthropic.ChatAnthropic.

Self-registers as 'anthropic' at import time. The langchain_anthropic import
is LAZY (inside build()) so importing this module never raises ImportError if
langchain-anthropic is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, apply_resilience_defaults, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class AnthropicProvider(Provider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_anthropic import ChatAnthropic  # lazy import
        from pydantic import SecretStr

        api_key = os.getenv(self.env_key)
        apply_resilience_defaults(params)
        # ChatAnthropic's `model` field is declared with alias "model_name"
        # (populate_by_name=True still accepts "model" at runtime, verified),
        # and its `api_key` field is a required (non-Optional) SecretStr with
        # a default_factory that re-reads ANTHROPIC_API_KEY — the same env var
        # `env_key` names. Passing `api_key=None` explicitly (the old code,
        # when the env var is unset) bypasses that default_factory and raises
        # a pydantic ValidationError; omitting the kwarg when there is no key
        # lets the SDK's own default apply instead, matching what "no key
        # configured" already looks like everywhere else in this module.
        if api_key:
            params["api_key"] = SecretStr(api_key)
        return ChatAnthropic(model_name=model, **params)


register_provider(AnthropicProvider())
