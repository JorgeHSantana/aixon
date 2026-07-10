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

from aixon.exceptions import AixonError
from aixon.providers.base import Provider, apply_resilience_defaults, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"


class ZAIProvider(Provider):
    name = "zai"
    env_key = "ZAI_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import

        api_key = os.getenv(self.env_key)
        if not api_key:
            raise AixonError(
                f"{self.env_key} is not set. Refusing to build the z.AI model — "
                f"the OpenAI SDK would silently fall back to OPENAI_API_KEY and "
                f"send that credential to the z.AI endpoint."
            )
        base_url = os.getenv("ZAI_BASE_URL", DEFAULT_BASE_URL)
        apply_resilience_defaults(params)
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **params)


register_provider(ZAIProvider())
