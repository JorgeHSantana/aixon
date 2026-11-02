# aixon/providers/openai.py
"""OpenAI provider — builds langchain_openai.ChatOpenAI.

Self-registers as 'openai' at import time. The langchain_openai import is
LAZY (inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations


from aixon.providers.base import OpenAICompatProvider, register_provider


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    env_key = "OPENAI_API_KEY"
    # Sem chave, nada é passado: o SDK lê OPENAI_API_KEY sozinho.
    require_key = False
    # Predicted Outputs (#6): LLM._invoke_kwargs attaches the previous
    # attempt's text as `prediction` on ReflectiveAgent retries. OpenAI-only
    # feature; providers without this flag never see the kwarg.
    supports_prediction = True


register_provider(OpenAIProvider())
