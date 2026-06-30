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

        api_key = os.getenv(self.env_key)
        apply_resilience_defaults(params)
        return ChatAnthropic(model=model, api_key=api_key, **params)


register_provider(AnthropicProvider())
