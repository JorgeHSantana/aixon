"""Decision providers: the engines behind a ``Decider``.

A decision provider answers CLOSED questions about a context — it never
generates text. One request carries one context and N typed questions; the
reply carries, per question, the chosen option and the chance of every option.

The neutral types here (``Question``, ``RawAnswer``, ``ProviderReply``) are the
only things that cross between ``aixon.decider`` and a provider. Vendor wire
formats and SDK objects stay inside the provider module, same rule as the LLM
``Provider``.

Registry mirrors ``aixon.providers.base``: ``register_decider_provider`` /
``get_decider_provider``. Provider modules self-register on import and import
their vendor SDK lazily, so ``import aixon`` never needs torch or an API key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from aixon.exceptions import AixonError

QuestionKind = Literal["choice", "yes_no", "scale"]

YES = "yes"
NO = "no"


@dataclass(frozen=True)
class Question:
    """One closed question.

    Exactly one of ``options`` / ``scale`` is set:

    - ``options``: option -> criterion (the text that tells the engine when
      that option applies; ``None`` = the option name speaks for itself). The
      pair ``{"yes", "no"}`` is a yes/no question.
    - ``scale``: ordered level labels, lowest first (an ordinal grade).
    """

    instruction: str
    options: Mapping[str, str | None] | None = None
    scale: tuple[str, ...] | None = None

    @property
    def kind(self) -> QuestionKind:
        if self.scale is not None:
            return "scale"
        if self.options is not None and set(self.options) == {YES, NO}:
            return "yes_no"
        return "choice"

    @property
    def labels(self) -> tuple[str, ...]:
        """Every answer this question can produce, in declaration order."""
        if self.scale is not None:
            return tuple(self.scale)
        return tuple(self.options or ())


@dataclass(frozen=True)
class RawAnswer:
    """A provider's answer to one question, before the ``Decider`` applies its
    certainty threshold.

    ``chances`` maps each label to its probability (0..1). ``certainty`` is the
    provider's own calibrated confidence when it reports one; otherwise the
    ``Decider`` uses the chance of the chosen label. ``score`` is the
    fractional position on a scale (0 = first level), scale questions only.
    """

    choice: str
    chances: dict[str, float] = field(default_factory=dict)
    certainty: float | None = None
    score: float | None = None


@dataclass(frozen=True)
class ProviderReply:
    """Answers keyed by question name, plus what the call cost."""

    answers: dict[str, RawAnswer]
    usage: dict[str, int] | None = None
    model: str | None = None


class DeciderProvider(ABC):
    """Base class for a decision engine.

    Class attributes a concrete provider sets:

    - ``name``: registry key (``provider = "<name>"`` on a ``Decider``).
    - ``hosted``: True when the context leaves the process (an HTTP API). A
      ``Decider`` on a hosted provider must declare ``redact``.
    - ``capabilities``: question kinds it answers (``"choice"``, ``"yes_no"``,
      ``"scale"``) plus ``"batch"`` when N questions fit in one request.
    - ``max_options``: most options one choice question may carry, or None.
    - ``default_model``: used when the ``Decider`` declares no ``model``.
    """

    name: str = ""
    hosted: bool = False
    capabilities: frozenset[str] = frozenset({"choice", "yes_no", "scale", "batch"})
    max_options: int | None = None
    default_model: str | None = None

    def unavailable(self) -> str | None:
        """Why this provider cannot answer right now (``"no_key"``,
        ``"not_installed"``...), or None when it can. Checked before every
        call so a missing key costs nothing and never raises."""
        return None

    @abstractmethod
    def decide(
        self,
        context: Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        model: str | None,
        timeout: float,
    ) -> ProviderReply:
        """Answer every question about ``context``. May raise: the caller
        (``aixon.decider``) is the one that never lets a failure escape."""


_registry: dict[str, DeciderProvider] = {}


def register_decider_provider(provider: DeciderProvider) -> None:
    """Register a provider instance keyed by ``provider.name`` (overwrites)."""
    if not provider.name:
        raise AixonError("DeciderProvider must define a non-empty `name`.")
    _registry[provider.name] = provider


def get_decider_provider(name: str) -> DeciderProvider:
    """Return the provider registered as *name*.

    Raises:
        AixonError: no provider is registered under that name.
    """
    try:
        return _registry[name]
    except KeyError:
        raise AixonError(
            f"No decider provider registered as '{name}'. "
            f"Available: {sorted(_registry)}. "
            f"Call register_decider_provider() with a custom DeciderProvider."
        )
