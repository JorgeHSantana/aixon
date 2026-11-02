"""xAI provider — Grok models via the OpenAI-compatible endpoint.

The xAI Chat Completions API follows the OpenAI wire contract, so this
provider reuses langchain_openai.ChatOpenAI pointed at the xAI base URL.
Self-registers as 'xai' at import time. The langchain_openai import is LAZY
(inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

from aixon.providers.base import OpenAICompatProvider, register_provider

DEFAULT_BASE_URL = "https://api.x.ai/v1"


class XAIProvider(OpenAICompatProvider):
    name = "xai"
    env_key = "XAI_API_KEY"
    base_url_env = "XAI_BASE_URL"
    default_base_url = DEFAULT_BASE_URL
    require_key_message = (
        f"{env_key} is not set. Refusing to build the xAI model — "
        f"the OpenAI SDK would silently fall back to OPENAI_API_KEY and "
        f"send that credential to the xAI endpoint."
    )


register_provider(XAIProvider())
