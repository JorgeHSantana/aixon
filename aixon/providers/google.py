# aixon/providers/google.py
"""Google provider — builds langchain_google_genai.ChatGoogleGenerativeAI.

Self-registers as 'google' at import time. The langchain_google_genai import
is LAZY (inside build()) so importing this module never raises ImportError if
langchain-google-genai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.logging import Logger
from aixon.providers.base import (
    Provider,
    apply_resilience_defaults,
    drop_unsupported_params,
    register_provider,
    resolve_reasoning_spec,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_log = Logger("aixon.providers.google")


class GoogleProvider(Provider):
    name = "google"
    env_key = "GOOGLE_API_KEY"
    supports_reasoning = True

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

        api_key = os.getenv(self.env_key)
        apply_resilience_defaults(params)

        # `presence_penalty`/`frequency_penalty` are in the cross-provider
        # GENERATION_PARAMS allowlist but are not fields on
        # ChatGoogleGenerativeAI either — drop + warn, mirroring Anthropic.
        drop_unsupported_params(
            params, ("presence_penalty", "frequency_penalty"), self.name, _log
        )

        spec = resolve_reasoning_spec(params)
        if spec is not None:
            # `thinking_budget`/`include_thoughts` are direct constructor
            # kwargs on the installed langchain-google-genai (verified on
            # 4.2.5) — but probe the pydantic fields first so an older
            # installed version without thinking support degrades gracefully
            # (skip + warn) instead of blowing up on unknown kwargs.
            fields = getattr(ChatGoogleGenerativeAI, "model_fields", {})
            if "thinking_budget" in fields:
                params["thinking_budget"] = spec["budget_tokens"]
                params["include_thoughts"] = True
            else:
                _log.warning(
                    "reasoning not supported by installed langchain-google-genai"
                    " — ignored"
                )

        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **params)


register_provider(GoogleProvider())
