# aixon/providers/openai.py
"""OpenAI provider — builds langchain_openai.ChatOpenAI.

Self-registers as 'openai' at import time. The langchain_openai import is
LAZY (inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, apply_resilience_defaults, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class OpenAIProvider(Provider):
    name = "openai"
    env_key = "OPENAI_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import
        from pydantic import SecretStr

        api_key = os.getenv(self.env_key)
        apply_resilience_defaults(params)
        # ChatOpenAI's `api_key` field accepts SecretStr | Callable | None, not
        # a bare str (pydantic coerces at runtime, but the static field type
        # rejects it). None IS accepted and falls back to the SDK's own
        # OPENAI_API_KEY env read (same var as `env_key`), so only wrap when
        # a value is actually present.
        if api_key:
            params["api_key"] = SecretStr(api_key)
        return ChatOpenAI(model=model, **params)


register_provider(OpenAIProvider())
