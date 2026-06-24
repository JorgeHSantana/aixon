# aixon/providers/google.py
"""Google provider — builds langchain_google_genai.ChatGoogleGenerativeAI.

Self-registers as 'google' at import time. The langchain_google_genai import
is LAZY (inside build()) so importing this module never raises ImportError if
langchain-google-genai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class GoogleProvider(Provider):
    name = "google"
    env_key = "GOOGLE_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

        api_key = os.getenv(self.env_key)
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **params)


register_provider(GoogleProvider())
