"""Provider ABC, registry, and model-name inference.

Each concrete provider (OpenAI / Anthropic / Google) lives in its own
module under aixon/providers/ and self-registers at import time via
register_provider(). Provider SDK imports are LAZY (inside build()) so
importing this module — or any provider module — never fails due to a
missing vendor SDK.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from aixon.exceptions import AixonError

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# Network-resilience defaults
# ---------------------------------------------------------------------------

# Applied by every provider's build() unless the caller passes its own value.
# WITHOUT a timeout, a stalled provider stream (e.g. a half-open HTTP response
# that never delivers another byte) blocks the request FOREVER: the agent's
# ``max_execution_time`` is only checked *between* graph updates, so it cannot
# interrupt a read that is stuck mid-update. A client-side timeout turns that
# stall into a finite error; ``max_retries`` reabsorbs transient stream drops.
DEFAULT_TIMEOUT_S: float = 120.0
DEFAULT_MAX_RETRIES: int = 2


def apply_resilience_defaults(params: dict[str, Any]) -> None:
    """Set ``timeout``/``max_retries`` in *params* in place if absent.

    The caller always wins: passing ``LLM(model, timeout=...)`` overrides the
    default. All three built-in chat models (OpenAI/Anthropic/Google) accept
    both keyword arguments.
    """
    params.setdefault("timeout", DEFAULT_TIMEOUT_S)
    params.setdefault("max_retries", DEFAULT_MAX_RETRIES)


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Builds a LangChain BaseChatModel for one vendor.

    Reads the API key from the environment inside build(). Concrete
    providers live in aixon/providers/<vendor>.py.
    """

    name: str       # "openai" | "anthropic" | "google"
    env_key: str    # e.g. "OPENAI_API_KEY"

    @abstractmethod
    def build(self, model: str, **params: Any) -> "BaseChatModel":
        """Return a configured LangChain chat model.

        **params are passed through (temperature, max_tokens, top_p, etc.).
        The API key is read from os.getenv(self.env_key) inside this method.
        """


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_registry: dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Register a provider instance keyed by provider.name (overwrites)."""
    _registry[provider.name] = provider


def get_provider(name: str) -> Provider:
    """Return the registered provider for *name*.

    Raises:
        AixonError: if no provider is registered under that name.
    """
    try:
        return _registry[name]
    except KeyError:
        available = sorted(_registry)
        raise AixonError(
            f"No provider registered as '{name}'. "
            f"Available: {available}. "
            f"Install the relevant extra (e.g. pip install aixon[openai]) "
            f"or call register_provider() with a custom Provider."
        )


# ---------------------------------------------------------------------------
# Model-name → provider inference
# ---------------------------------------------------------------------------

# Rules: (compiled regex, provider name). Evaluated in order; first match wins.
_INFERENCE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(gpt-|o\d|text-)"), "openai"),
    (re.compile(r"^claude"), "anthropic"),
    (re.compile(r"^gemini"), "google"),
]


def resolve_provider_for_model(model: str) -> Provider:
    """Infer the provider from the model name and return the registered Provider.

    Rules:
    - gpt-* | o<digit>* | text-* → openai
    - claude*                    → anthropic
    - gemini*                    → google

    Raises:
        AixonError: if no rule matches or the inferred provider is not registered.
    """
    for pattern, provider_name in _INFERENCE_RULES:
        if pattern.match(model):
            return get_provider(provider_name)
    raise AixonError(
        f"Cannot infer provider for model '{model}'. "
        f"Pass provider= explicitly: LLM('{model}', provider='openai')."
    )
