"""Decider: a closed question about a context, answered with a measured
certainty.

A ``Decider`` is declared like every other aixon resource — a class with
attributes — and asked with ``decide(context)``::

    class StoreOrProductDecider(Decider):
        provider = "jev"
        instruction = "What does the word in 'word' refer to in the user's message?"
        options = {
            "store": "a branch of the chain (a place, 'the' + name, the branch's revenue)",
            "product": "an item for sale, a category or an ingredient",
        }
        redact = None            # nothing sensitive in this context

    decision = StoreOrProductDecider.decide({"message": text, "word": "acerola"})
    if decision is None:         # engine off, no key, timeout: carry on without it
        ...
    elif decision.sure:          # decision.choice == "store", certainty 0.93
        ...
    else:                        # the engine is not sure: ask the user back
        ...

Three outcomes, always: **sure** (act), **unsure** (take the declared fallback,
typically asking the user back) and **unavailable** (``None``: behave as if the
decider did not exist). ``decide`` never raises at call time — a missing key, a
switched-off decider, a timeout or a malformed reply all yield ``None``.
Mistakes in the DECLARATION raise ``AixonError`` when the class is defined.

What a decider is for: judging LANGUAGE (tone, intent, "is this a question
back?", "which of these does the user mean?"). What it is not for: checking
numbers or arithmetic, and open questions without the options on the table
("is this message ambiguous?" flags most clear messages). See docs/deciders.md.

``Doubt`` and ``Check`` plug deciders into a ``ToolAgent`` declaratively —
before the model (read the user's message) and after it (does the answer make
sense? redo once).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence, Union

from aixon.deciders.base import (
    NO,
    YES,
    DeciderProvider,
    ProviderReply,
    Question,
    RawAnswer,
    get_decider_provider,
)
from aixon.exceptions import AixonError, NamingError
from aixon.logging import Logger
from aixon.message import Message

_log = Logger("aixon.decider")

_UNSET: Any = object()
_OFF = {"0", "off", "false", "no"}
_SUFFIX = "Decider"

Redact = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class Decision:
    """What a ``Decider`` answered.

    - ``choice``: the option (or scale level) chosen.
    - ``certainty``: 0..1 — the engine's calibrated confidence when it reports
      one, otherwise the chance of ``choice``.
    - ``chances``: chance of every option.
    - ``sure``: ``certainty >= <Decider>.certainty``. The line between "I know"
      and "I would be guessing".
    - ``score``: fractional position on a scale (0 = first level); scale
      deciders only.
    """

    choice: str
    certainty: float
    chances: dict[str, float]
    sure: bool
    decider: str
    provider: str
    model: str | None = None
    latency_ms: float = 0.0
    score: float | None = None
    usage: dict[str, int] | None = None

    def chance(self, option: str) -> float:
        """Chance of *option* (0.0 when the engine did not report it)."""
        return float(self.chances.get(option, 0.0))


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


class Decider:
    """Base class of a declarative decision. Subclass and set attributes; do
    not instantiate. The class name must end with ``Decider``.

    Attributes:
        provider: registered provider name (``"jev"``, ``"laya"``), a
            ``DeciderProvider`` instance, or None to take the name from the
            ``AIXON_DECIDER_PROVIDER`` environment variable at call time.
        model: engine model / checkpoint; None = the provider's default.
        instruction: the question. Required.
        options: ``{option: criterion}`` (criterion may be None), or just the
            option names. ``{"yes", "no"}`` makes a yes/no question. Required
            unless ``scale`` is set.
        scale: ordered level labels, lowest first, for an ordinal grade.
        certainty: threshold of ``Decision.sure`` (default 0.8).
        timeout: seconds a hosted engine may take (default 3.0).
        switch: name of an environment variable that turns THIS decider off
            when set to ``0``/``off``/``false``/``no``.
        redact: ``(context) -> context`` applied before the context leaves the
            process. REQUIRED when the provider is hosted; write
            ``redact = None`` to state that the context needs no masking.
        name: identifier in logs and traces; defaults to the class name in
            snake_case without the suffix.
    """

    provider: ClassVar[Union[str, DeciderProvider, None]] = None
    model: ClassVar[str | None] = None
    instruction: ClassVar[str] = ""
    options: ClassVar[Any] = None
    scale: ClassVar[Sequence[str] | None] = None
    certainty: ClassVar[float] = 0.8
    timeout: ClassVar[float] = 3.0
    switch: ClassVar[str | None] = None
    redact: ClassVar[Any] = _UNSET
    name: ClassVar[str] = ""

    def __init__(self) -> None:
        raise AixonError(
            f"{type(self).__name__} is declarative: call "
            f"{type(self).__name__}.decide(context), do not instantiate it."
        )

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        if not cls.__name__.endswith(_SUFFIX) or cls.__name__ == _SUFFIX:
            raise NamingError(
                f"Decider subclass '{cls.__name__}' must end with '{_SUFFIX}' "
                f"(rename to '{cls.__name__}{_SUFFIX}')."
            )
        if "name" not in cls.__dict__ or not cls.name:
            cls.name = _snake(cls.__name__[: -len(_SUFFIX)])
        cls._validate()

    # ---- declaration -----------------------------------------------------

    @classmethod
    def _fail(cls, problem: str) -> AixonError:
        return AixonError(f"Decider '{cls.__name__}': {problem}")

    @classmethod
    def _validate(cls) -> None:
        if not isinstance(cls.instruction, str) or not cls.instruction.strip():
            raise cls._fail("`instruction` is required (the question the engine answers).")
        if (cls.options is None) == (cls.scale is None):
            raise cls._fail("declare exactly one of `options` or `scale`.")
        if cls.options is not None:
            if isinstance(cls.options, str):
                raise cls._fail("`options` must be a dict {option: criterion} or a collection of names.")
            if not isinstance(cls.options, Mapping):
                cls.options = {str(option): None for option in cls.options}
            if len(cls.options) < 2:
                raise cls._fail("`options` needs at least two options.")
        else:
            cls.scale = tuple(cls.scale or ())
            if len(cls.scale) < 2:
                raise cls._fail("`scale` needs at least two levels.")
        if not 0.0 < float(cls.certainty) <= 1.0:
            raise cls._fail("`certainty` must be in (0, 1].")
        if float(cls.timeout) <= 0:
            raise cls._fail("`timeout` must be positive.")
        if cls.redact is not _UNSET and cls.redact is not None and not callable(cls.redact):
            raise cls._fail("`redact` must be a function (context) -> context, or None.")
        if cls.provider is None:
            return  # resolved from the environment at call time
        provider = cls._provider()
        assert provider is not None
        problem = cls._misfit(provider)
        if problem:
            raise cls._fail(problem)

    @classmethod
    def _misfit(cls, provider: DeciderProvider) -> str | None:
        """Why this declaration cannot run on *provider*, or None."""
        kind = cls.question().kind
        if kind not in provider.capabilities:
            return f"provider '{provider.name}' does not answer '{kind}' questions."
        if provider.max_options is not None and len(cls.question().labels) > provider.max_options:
            return (f"provider '{provider.name}' takes at most {provider.max_options} "
                    f"options; {len(cls.question().labels)} declared.")
        if provider.hosted and cls.redact is _UNSET:
            return (
                f"provider '{provider.name}' is hosted: the context leaves this process. "
                f"Declare `redact = <function>` to mask it first, or `redact = None` to "
                f"state that nothing in it needs masking."
            )
        return None

    @classmethod
    def question(cls) -> Question:
        """The neutral question this decider asks."""
        if cls.scale is not None:
            return Question(instruction=cls.instruction, scale=tuple(cls.scale))
        return Question(instruction=cls.instruction, options=dict(cls.options))

    @classmethod
    def _provider(cls) -> DeciderProvider | None:
        declared = cls.provider if cls.provider is not None else os.getenv("AIXON_DECIDER_PROVIDER")
        if not declared:
            return None
        if isinstance(declared, DeciderProvider):
            return declared
        import aixon.deciders  # noqa: F401  (self-registers the built-in providers)

        return get_decider_provider(declared)

    @classmethod
    def enabled(cls) -> bool:
        """False when the decider's ``switch`` variable turns it off."""
        if not cls.switch:
            return True
        return os.getenv(cls.switch, "").strip().lower() not in _OFF

    @classmethod
    def _redactor(cls) -> Redact | None:
        return None if cls.redact is _UNSET else cls.redact

    @classmethod
    def _why_not(cls) -> tuple[DeciderProvider | None, str | None]:
        """(provider, reason it cannot be asked right now)."""
        if not cls.enabled():
            return None, "switched_off"
        try:
            provider = cls._provider()
        except AixonError:
            return None, "unknown_provider"
        if provider is None:
            return None, "no_provider"
        if cls._misfit(provider):
            return provider, "misconfigured"
        return provider, provider.unavailable()

    # ---- asking ----------------------------------------------------------

    @classmethod
    def decide(cls, context: Mapping[str, Any]) -> Decision | None:
        """Ask the engine. ``None`` = no decision (off, no key, timeout,
        error): carry on as if this decider did not exist. Never raises."""
        return decide_many(context, [cls])[cls]

    @classmethod
    async def adecide(cls, context: Mapping[str, Any]) -> Decision | None:
        """Async ``decide`` (the engine call runs in a worker thread)."""
        return await asyncio.to_thread(cls.decide, context)

    @classmethod
    def _decision(cls, raw: RawAnswer, reply: ProviderReply, provider: DeciderProvider,
                  latency_ms: float) -> Decision:
        certainty = raw.certainty if raw.certainty is not None else raw.chances.get(raw.choice, 0.0)
        certainty = round(min(1.0, max(0.0, float(certainty))), 4)
        return Decision(
            choice=raw.choice,
            certainty=certainty,
            chances={k: round(float(v), 4) for k, v in raw.chances.items()},
            sure=certainty >= float(cls.certainty),
            decider=cls.name,
            provider=provider.name,
            model=reply.model,
            latency_ms=round(latency_ms, 1),
            score=raw.score,
            usage=reply.usage,
        )


DeciderType = type[Decider]


def _record(decider: DeciderType, context: Mapping[str, Any] | None,
            decision: Decision | None, reason: str | None,
            provider: DeciderProvider | None) -> None:
    """Log line + trace span for one decision (or one non-decision)."""
    from aixon.observability import record_decision

    if decision is not None:
        _log.info(
            f"decider '{decider.name}' -> {decision.choice} (certainty "
            f"{decision.certainty:.2f}, {'sure' if decision.sure else 'unsure'}) "
            f"via {decision.provider} in {decision.latency_ms:.0f} ms"
        )
        record_decision(
            decider.name, input=context,
            output={"choice": decision.choice, "certainty": decision.certainty,
                    "sure": decision.sure, "chances": decision.chances},
            metadata={"provider": decision.provider, "model": decision.model,
                      "latency_ms": decision.latency_ms, "usage": decision.usage,
                      "threshold": float(decider.certainty)},
        )
        return
    quiet = reason in ("switched_off", "no_key", "no_provider", "not_installed")
    (_log.debug if quiet else _log.warning)(f"decider '{decider.name}' -> no decision ({reason})")
    if not quiet:
        record_decision(decider.name, input=context, output={"no_decision": reason},
                        metadata={"provider": provider.name if provider else None}, failed=True)


def _ask(provider: DeciderProvider, context: Mapping[str, Any],
         questions: dict[str, Question], *, model: str | None, timeout: float) -> ProviderReply:
    if "batch" in provider.capabilities or len(questions) == 1:
        return provider.decide(context, questions, model=model, timeout=timeout)
    answers: dict[str, RawAnswer] = {}
    usage: dict[str, int] | None = None
    reply_model: str | None = None
    from aixon.usage import merge_usage

    for name, question in questions.items():
        one = provider.decide(context, {name: question}, model=model, timeout=timeout)
        answers.update(one.answers)
        usage = merge_usage(usage, one.usage)
        reply_model = one.model or reply_model
    return ProviderReply(answers=answers, usage=usage, model=reply_model)


def decide_many(context: Mapping[str, Any],
                deciders: Iterable[DeciderType]) -> dict[DeciderType, Decision | None]:
    """Several decisions about ONE context. Deciders that share provider,
    model and ``redact`` go in a single request (the time barely changes with
    the number of questions). Each decider maps to its ``Decision`` or to
    ``None``; one group failing does not affect another. Never raises."""
    deciders = list(dict.fromkeys(deciders))
    results: dict[DeciderType, Decision | None] = {d: None for d in deciders}
    groups: dict[tuple[int, str | None, int], list[DeciderType]] = {}
    providers: dict[int, DeciderProvider] = {}
    for decider in deciders:
        provider, reason = decider._why_not()
        if reason or provider is None:
            _record(decider, None, None, reason or "no_provider", provider)
            continue
        providers[id(provider)] = provider
        groups.setdefault((id(provider), decider.model, id(decider._redactor())), []).append(decider)
    for (provider_id, model, _), group in groups.items():
        provider = providers[provider_id]
        sent: Mapping[str, Any] | None = None
        try:
            redactor = group[0]._redactor()
            sent = redactor(dict(context)) if redactor else dict(context)
            names: dict[str, DeciderType] = {}
            for decider in group:
                key = decider.name if decider.name not in names else f"{decider.name}_{len(names)}"
                names[key] = decider
            started = time.monotonic()
            reply = _ask(provider, sent, {key: d.question() for key, d in names.items()},
                         model=model, timeout=min(float(d.timeout) for d in group))
            latency_ms = (time.monotonic() - started) * 1000
            for key, decider in names.items():
                results[decider] = decider._decision(reply.answers[key], reply, provider, latency_ms)
        except Exception as exc:  # noqa: BLE001 — a decision never breaks the caller
            for decider in group:
                results[decider] = None
                _record(decider, sent, None, type(exc).__name__, provider)
            continue
        for decider in group:
            _record(decider, sent, results[decider], None, provider)
    return results


async def adecide_many(context: Mapping[str, Any],
                       deciders: Iterable[DeciderType]) -> dict[DeciderType, Decision | None]:
    """Async ``decide_many`` (engine calls run in a worker thread)."""
    return await asyncio.to_thread(decide_many, context, list(deciders))


# ---------------------------------------------------------------------------
# Doubt: a decision about the user's message, BEFORE the model runs
# ---------------------------------------------------------------------------

Case = Mapping[str, Any]


@dataclass(frozen=True)
class Doubt:
    """Something in the user's message that could be read two ways.

    - ``decider``: who settles it.
    - ``applies``: ``(message) -> case | None``. Cheap, pure code that finds
      the concrete case in the message (the word, the candidates). ``None`` or
      ``False`` = this doubt is not in the message and the engine is NOT
      called. Omit it to ask on every message.
    - ``context``: ``(message, case) -> context`` sent to the engine. Default:
      ``{"message": message, **case}``.
    - ``on_sure``: ``(decision, case) -> directive | None``, when the engine is
      sure. The directive is appended to the user's message for the model.
    - ``on_unsure``: ``(decision, case) -> directive | None``, when it is not —
      typically "ask the user which one they mean and wait".

    A concrete doubt with the options on the table works; a generic "is this
    ambiguous?" does not. Measure before turning one on: asking back too often
    annoys the user (``aixon decide eval``).
    """

    decider: DeciderType
    applies: Callable[[str], Any] | None = None
    context: Callable[[str, Case], Mapping[str, Any]] | None = None
    on_sure: Callable[[Decision, Case], str | None] | None = None
    on_unsure: Callable[[Decision, Case], str | None] | None = None


@dataclass(frozen=True)
class DoubtOutcome:
    """One resolved doubt: ``outcome`` is ``"sure"`` or ``"unsure"``."""

    doubt: Doubt
    decision: Decision
    case: Case
    outcome: str
    directive: str | None


def _cases(message: str, doubts: Sequence[Doubt]) -> list[tuple[Doubt, Case, Mapping[str, Any]]]:
    found: list[tuple[Doubt, Case, Mapping[str, Any]]] = []
    for doubt in doubts:
        try:
            case: Any = {} if doubt.applies is None else doubt.applies(message)
            if case is None or case is False:
                continue
            if not isinstance(case, Mapping):
                case = {}
            context = doubt.context(message, case) if doubt.context else {"message": message, **case}
            found.append((doubt, case, context))
        except Exception as exc:  # noqa: BLE001 — a broken hook must not break the request
            _log.warning(f"doubt '{doubt.decider.name}' skipped: {type(exc).__name__}: {exc}")
    return found


def _outcomes(found: list[tuple[Doubt, Case, Mapping[str, Any]]],
              decisions: list[Decision | None]) -> list[DoubtOutcome]:
    outcomes: list[DoubtOutcome] = []
    for (doubt, case, _), decision in zip(found, decisions):
        if decision is None:
            continue
        hook = doubt.on_sure if decision.sure else doubt.on_unsure
        try:
            directive = hook(decision, case) if hook else None
        except Exception as exc:  # noqa: BLE001
            _log.warning(f"doubt '{doubt.decider.name}' hook failed: {type(exc).__name__}: {exc}")
            directive = None
        outcomes.append(DoubtOutcome(doubt, decision, case,
                                     "sure" if decision.sure else "unsure", directive))
    return outcomes


def _decide_cases(found: list[tuple[Doubt, Case, Mapping[str, Any]]]) -> list[Decision | None]:
    """Doubts whose contexts are identical share one request."""
    decisions: list[Decision | None] = [None] * len(found)
    by_context: dict[str, list[int]] = {}
    for i, (_, _, context) in enumerate(found):
        by_context.setdefault(json.dumps(context, sort_keys=True, default=str), []).append(i)
    for indexes in by_context.values():
        answered = decide_many(found[indexes[0]][2], [found[i][0].decider for i in indexes])
        for i in indexes:
            decisions[i] = answered[found[i][0].decider]
    return decisions


def resolve_doubts(message: str, doubts: Sequence[Doubt]) -> list[DoubtOutcome]:
    """Find which doubts are in *message*, ask the engines, run the hooks.
    Doubts not present, and doubts whose engine gave no decision, are absent
    from the result. Never raises."""
    found = _cases(message, doubts)
    return _outcomes(found, _decide_cases(found)) if found else []


async def aresolve_doubts(message: str, doubts: Sequence[Doubt]) -> list[DoubtOutcome]:
    """Async ``resolve_doubts``."""
    return await asyncio.to_thread(resolve_doubts, message, doubts)


def _with_directives(messages: list[Message], outcomes: list[DoubtOutcome]) -> list[Message]:
    directives = [o.directive for o in outcomes if o.directive]
    if not directives:
        return messages
    index = _last_user(messages)
    assert index is not None
    last = messages[index]
    patched = Message(role=last.role, content=last.content + "\n\n" + "\n".join(directives),
                      name=last.name)
    return [*messages[:index], patched, *messages[index + 1:]]


def _last_user(messages: list[Message]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return None


def _ends_with_user(messages: list[Message]) -> bool:
    return bool(messages) and messages[-1].role == "user"


def apply_doubts(messages: list[Message], doubts: Sequence[Doubt]) -> list[Message]:
    """Resolve *doubts* against the LAST message when it is the user's turn and
    return the conversation with the directives appended to it. The input list
    is not mutated. A conversation resuming after a tool result is returned
    untouched."""
    if not doubts or not _ends_with_user(messages):
        return messages
    return _with_directives(messages, resolve_doubts(messages[-1].content, doubts))


async def aapply_doubts(messages: list[Message], doubts: Sequence[Doubt]) -> list[Message]:
    """Async ``apply_doubts``."""
    if not doubts or not _ends_with_user(messages):
        return messages
    return _with_directives(messages, await aresolve_doubts(messages[-1].content, doubts))


# ---------------------------------------------------------------------------
# Check: a decision about the ANSWER, after the model ran
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """A test the answer must pass; failing it triggers ONE second attempt.

    - ``decider``: who judges. With a yes/no decider, "yes" means the answer
      is fine.
    - ``redo``: ``(decision, answer) -> instruction`` for the second attempt
      ("your answer covers September, the user asked for August: redo it").
    - ``fails``: ``(decision) -> bool``. Default (yes/no deciders only): the
      chance of "yes" is below ``fails_below``.
    - ``fails_below``: default 0.3. Deliberately far from 0.5: a check that
      fires on doubt rewrites good answers. At 0.3 the false-alarm rate
      measured on ~2,000 real answers was 0.25%.
    - ``context``: ``(message, answer) -> context`` sent to the engine.
      Default: ``{"message": message, "answer": answer}``.
    """

    decider: DeciderType
    redo: Callable[[Decision, str], str]
    fails: Callable[[Decision], bool] | None = None
    fails_below: float = 0.3
    context: Callable[[str, str], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.fails is None and self.decider.question().kind != "yes_no":
            raise AixonError(
                f"Check({self.decider.__name__}): the default failure rule needs a "
                f"yes/no decider (options {{'yes', 'no'}}); pass `fails=` for this one."
            )

    def failed(self, decision: Decision) -> bool:
        if self.fails is not None:
            return bool(self.fails(decision))
        return decision.chance(YES) < self.fails_below


@dataclass(frozen=True)
class CheckFailure:
    check: Check
    decision: Decision
    instruction: str = field(default="")


def failed_checks(message: str, answer: str, checks: Sequence[Check]) -> list[CheckFailure]:
    """Which *checks* the *answer* failed. A check whose engine gave no
    decision passes. Never raises."""
    if not checks or not answer.strip():
        return []
    failures: list[CheckFailure] = []
    by_context: dict[str, tuple[Mapping[str, Any], list[Check]]] = {}
    for check in checks:
        try:
            context = check.context(message, answer) if check.context else {
                "message": message, "answer": answer}
        except Exception as exc:  # noqa: BLE001
            _log.warning(f"check '{check.decider.name}' skipped: {type(exc).__name__}: {exc}")
            continue
        by_context.setdefault(json.dumps(context, sort_keys=True, default=str),
                              (context, []))[1].append(check)
    for context, group in by_context.values():
        answered = decide_many(context, [check.decider for check in group])
        for check in group:
            decision = answered[check.decider]
            try:
                if decision is None or not check.failed(decision):
                    continue
                failures.append(CheckFailure(check, decision, check.redo(decision, answer)))
            except Exception as exc:  # noqa: BLE001
                _log.warning(f"check '{check.decider.name}' hook failed: {type(exc).__name__}: {exc}")
    return failures


async def afailed_checks(message: str, answer: str, checks: Sequence[Check]) -> list[CheckFailure]:
    """Async ``failed_checks``."""
    return await asyncio.to_thread(failed_checks, message, answer, checks)


def redo_messages(messages: list[Message], answer: str,
                  failures: Sequence[CheckFailure]) -> list[Message]:
    """The conversation for the second attempt: the original turns, the answer
    that failed, and one user turn carrying the redo instructions."""
    instructions = "\n".join(f.instruction for f in failures if f.instruction)
    return [*messages, Message(role="assistant", content=answer),
            Message(role="user", content=instructions)]


__all__ = [
    "Check",
    "CheckFailure",
    "Decider",
    "Decision",
    "Doubt",
    "DoubtOutcome",
    "NO",
    "YES",
    "aapply_doubts",
    "adecide_many",
    "afailed_checks",
    "apply_doubts",
    "aresolve_doubts",
    "decide_many",
    "failed_checks",
    "redo_messages",
    "resolve_doubts",
]
