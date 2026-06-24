# aixon/providers/openai.py
"""OpenAI provider — builds langchain_openai.ChatOpenAI.

Self-registers as 'openai' at import time. The langchain_openai import is
LAZY (inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class OpenAIProvider(Provider):
    name = "openai"
    env_key = "OPENAI_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import

        api_key = os.getenv(self.env_key)
        return ChatOpenAI(model=model, api_key=api_key, **params)


register_provider(OpenAIProvider())
