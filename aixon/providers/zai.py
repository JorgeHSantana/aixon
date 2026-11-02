"""z.AI provider — GLM models via the OpenAI-compatible endpoint.

The z.AI Chat Completions API follows the OpenAI wire contract, so this
provider reuses langchain_openai.ChatOpenAI pointed at the z.AI base URL.
Self-registers as 'zai' at import time. The langchain_openai import is LAZY
(inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

from typing import Any

from aixon.providers.base import OpenAICompatProvider, register_provider

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"


class ZAIProvider(OpenAICompatProvider):
    name = "zai"
    env_key = "ZAI_API_KEY"
    base_url_env = "ZAI_BASE_URL"
    default_base_url = DEFAULT_BASE_URL
    require_key_message = (
        f"{env_key} is not set. Refusing to build the z.AI model — "
        f"the OpenAI SDK would silently fall back to OPENAI_API_KEY and "
        f"send that credential to the z.AI endpoint."
    )

    def apply_reasoning(self, params: dict[str, Any], spec: dict[str, Any]) -> None:
        # GLM has no budget/effort dial — any non-off spec just turns
        # thinking on. Merge with a caller-supplied extra_body if present.
        extra_body = dict(params.get("extra_body") or {})
        thinking = dict(extra_body.get("thinking") or {})
        thinking["type"] = "enabled"
        extra_body["thinking"] = thinking
        params["extra_body"] = extra_body


register_provider(ZAIProvider())
