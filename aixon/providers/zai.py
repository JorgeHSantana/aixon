"""z.AI provider — GLM models via the OpenAI-compatible endpoint.

The z.AI Chat Completions API follows the OpenAI wire contract, so this
provider reuses langchain_openai.ChatOpenAI pointed at the z.AI base URL.
Self-registers as 'zai' at import time. The langchain_openai import is LAZY
(inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, apply_resilience_defaults, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"


class _ChatOpenAIWrapper:
    """Wraps ChatOpenAI to expose request_timeout as 'timeout' for API consistency."""

    def __init__(self, chat_openai: Any) -> None:
        self._model = chat_openai

    def __getattr__(self, name: str) -> Any:
        if name == "timeout":
            return self._model.request_timeout
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_model":
            object.__setattr__(self, name, value)
        elif name == "timeout":
            self._model.request_timeout = value
        else:
            setattr(self._model, name, value)

    def __getstate__(self) -> dict[str, Any]:
        return {"_model": self._model}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class ZAIProvider(Provider):
    name = "zai"
    env_key = "ZAI_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import

        api_key = os.getenv(self.env_key)
        base_url = os.getenv("ZAI_BASE_URL", DEFAULT_BASE_URL)
        apply_resilience_defaults(params)
        # ChatOpenAI uses 'request_timeout', but we normalize to 'timeout'
        # for consistency with other providers (e.g. Google). Map it back.
        if "timeout" in params:
            params["request_timeout"] = params.pop("timeout")
        chat_openai = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **params)
        # Wrap to expose request_timeout as 'timeout' for API consistency
        return _ChatOpenAIWrapper(chat_openai)


register_provider(ZAIProvider())
