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
from aixon.logging import Logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_log = Logger("aixon.providers.base")


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
    default. All built-in chat models (OpenAI/Anthropic/Google/z.AI) accept
    both keyword arguments.
    """
    params.setdefault("timeout", DEFAULT_TIMEOUT_S)
    params.setdefault("max_retries", DEFAULT_MAX_RETRIES)


def drop_unsupported_params(
    params: dict[str, Any], keys: tuple[str, ...], provider_name: str, log: Logger
) -> None:
    """Pop any of *keys* present in *params* that the vendor constructor does
    not accept, logging one warning naming them.

    ``GENERATION_PARAMS`` (aixon/runtime.py) is a fixed cross-provider
    allowlist — e.g. ``presence_penalty``/``frequency_penalty`` are valid
    ``ChatOpenAI`` kwargs but are NOT fields on ``ChatAnthropic`` or
    ``ChatGoogleGenerativeAI`` (verified against the installed SDKs' pydantic
    ``model_fields``). Both vendor models declare ``model_config =
    ConfigDict(extra="ignore")``, so passing them through unfiltered doesn't
    raise — it silently vanishes with no feedback at all, the exact kind of
    silent-drop this rule replaces with an explicit, named warning.
    """
    dropped = sorted(k for k in keys if k in params)
    for key in dropped:
        del params[key]
    if dropped:
        log.warning(
            "%s does not support %s — ignored (dropped from the request)",
            provider_name,
            ", ".join(dropped),
        )


# ---------------------------------------------------------------------------
# Reasoning knob: normalization + per-build extraction
# ---------------------------------------------------------------------------

# Canonical budget<->effort table (fixed, shared by every provider). `True` is
# equivalent to ``{"effort": "medium"}``; a bare budget is bucketed into the
# nearest effort tier for providers with only a coarse effort dial (OpenAI).
_EFFORT_TO_BUDGET: dict[str, int] = {"low": 1024, "medium": 4096, "high": 16384}


def _budget_to_effort(budget_tokens: int) -> str:
    if budget_tokens <= 1024:
        return "low"
    if budget_tokens <= 8192:
        return "medium"
    return "high"


# Budget used for an unrecognized effort string (bucket-based providers only —
# OpenAI gets the ORIGINAL string verbatim; see below). Matches "medium".
_DEFAULT_UNKNOWN_EFFORT_BUDGET = 4096


def normalize_reasoning(spec: bool | dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the declarative ``reasoning`` knob into a canonical spec.

    - ``None``/``False`` -> ``None`` (reasoning off; callers must leave every
      other kwarg untouched to preserve today's byte-for-byte behavior).
    - ``True`` -> ``{"effort": "medium"}`` normalized (see below).
    - ``dict`` -> filled in with whichever half (``budget_tokens``/``effort``)
      is missing, per the fixed table above. If both halves are already
      present they are kept exactly as given (no re-derivation).

    An ``effort`` outside the fixed table (e.g. OpenAI's own ``"minimal"``,
    or a plain typo — either reachable from a client via the per-request
    ``reasoning_effort`` override) must never crash: this is a normalization
    helper, not a validator. ``spec["effort"]`` always keeps the ORIGINAL
    string verbatim (so ``OpenAIProvider``, which forwards ``spec["effort"]``
    straight to its own ``reasoning_effort`` constructor kwarg, passes
    unknown values through untouched — OpenAI is the authority on what its
    own dial accepts, not this fixed table). Budget-based providers
    (Anthropic/Google/z.AI) have no such passthrough, so an unknown effort
    falls back to the medium budget with one warning.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {"effort": "medium"}

    budget_tokens = spec.get("budget_tokens")
    effort = spec.get("effort")
    if effort is None:
        effort = "medium" if budget_tokens is None else _budget_to_effort(budget_tokens)
    if budget_tokens is None:
        budget_tokens = _EFFORT_TO_BUDGET.get(effort)
        if budget_tokens is None:
            _log.warning(
                "unknown reasoning effort '%s' — using medium budget for "
                "budget-based providers",
                effort,
            )
            budget_tokens = _DEFAULT_UNKNOWN_EFFORT_BUDGET
    return {"budget_tokens": budget_tokens, "effort": effort}


def pop_reasoning(params: dict[str, Any]) -> bool | dict[str, Any] | None:
    """Pop and return the raw ``reasoning`` knob from *params*.

    ``LLM`` always injects ``params["reasoning"]`` (possibly ``None``) before
    calling ``Provider.build()``. Every provider must pop it here — before
    touching any other kwarg — so the raw key never reaches the vendor SDK
    constructor, regardless of whether that provider translates it.
    """
    return params.pop("reasoning", None)


def resolve_reasoning_spec(params: dict[str, Any]) -> dict[str, Any] | None:
    """Pop both reasoning-related keys from *params* and resolve the final
    canonical spec a provider's ``build()`` must apply.

    Pops the class-level ``reasoning`` knob (via ``pop_reasoning``) and, if
    present, the per-request ``reasoning_effort`` generation param (allow-
    listed onto the wire in a later round) — which OVERRIDES the knob for
    this one build with ``{"effort": reasoning_effort}``, translated the same
    way as any other effort-only spec.
    """
    spec = normalize_reasoning(pop_reasoning(params))
    reasoning_effort = params.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        spec = normalize_reasoning({"effort": reasoning_effort})
    return spec


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Builds a LangChain BaseChatModel for one vendor.

    Reads the API key from the environment inside build(). Concrete
    providers live in aixon/providers/<vendor>.py.
    """

    name: str       # "openai" | "anthropic" | "google" | "zai"
    env_key: str    # e.g. "OPENAI_API_KEY"

    # Providers that translate the declarative ``reasoning`` knob set this to
    # True; ``LLM`` only injects ``params["reasoning"]`` into build() for
    # those. Default False so a custom provider that blindly forwards
    # **params to a strict vendor constructor never receives the stray key —
    # the knob is then ignored with a warning instead of breaking the build.
    supports_reasoning: bool = False

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
    (re.compile(r"^glm"), "zai"),
    (re.compile(r"^grok"), "xai"),
]


def resolve_provider_for_model(model: str) -> Provider:
    """Infer the provider from the model name and return the registered Provider.

    Rules:
    - gpt-* | o<digit>* | text-* → openai
    - claude*                    → anthropic
    - gemini*                    → google
    - glm*                       → zai (z.AI, OpenAI-compatible)
    - grok*                      → xai (Grok, OpenAI-compatible)

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
