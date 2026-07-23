"""xAI provider — Grok models via the OpenAI-compatible endpoint.

The xAI Chat Completions API follows the OpenAI wire contract, so this
provider reuses langchain_openai.ChatOpenAI pointed at the xAI base URL.
Self-registers as 'xai' at import time. The langchain_openai import is LAZY
(inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.exceptions import AixonError
from aixon.providers.base import (
    Provider,
    apply_resilience_defaults,
    register_provider,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_BASE_URL = "https://api.x.ai/v1"


class XAIProvider(Provider):
    name = "xai"
    env_key = "XAI_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import
        from pydantic import SecretStr

        api_key = os.getenv(self.env_key)
        if not api_key:
            raise AixonError(
                f"{self.env_key} is not set. Refusing to build the xAI model — "
                f"the OpenAI SDK would silently fall back to OPENAI_API_KEY and "
                f"send that credential to the xAI endpoint."
            )
        base_url = os.getenv("XAI_BASE_URL", DEFAULT_BASE_URL)
        apply_resilience_defaults(params)

        # ChatOpenAI's `api_key` field is SecretStr | Callable | None, not a
        # bare str — wrap explicitly (api_key is guaranteed truthy here, the
        # `raise` above already ruled out the empty/missing case).
        return ChatOpenAI(
            model=model, api_key=SecretStr(api_key), base_url=base_url, **params
        )


register_provider(XAIProvider())
