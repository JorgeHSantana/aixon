"""In-memory decision provider for tests and offline examples.

    from aixon.deciders import FakeDeciderProvider

    class ToneDecider(Decider):
        provider = FakeDeciderProvider({"tone": "upset"})        # always sure
        ...

    FakeDeciderProvider({"tone": ("upset", 0.55)})               # unsure
    FakeDeciderProvider(lambda context, name, question: "neutral")  # computed
    FakeDeciderProvider(error=TimeoutError())                    # engine down

A script entry is a label, a ``(label, certainty)`` pair, or a full
``RawAnswer``. Questions missing from the script get their first label with
certainty 1.0. Every call is recorded in ``.calls`` as
``(context, question_names)``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Union

from aixon.deciders.base import DeciderProvider, ProviderReply, Question, RawAnswer

Scripted = Union[str, tuple[str, float], RawAnswer]
Script = Union[Mapping[str, Scripted], Callable[[Mapping[str, Any], str, Question], Scripted]]


class FakeDeciderProvider(DeciderProvider):
    name = "fake"
    hosted = False

    def __init__(self, script: Script | None = None, *, error: Exception | None = None,
                 hosted: bool = False, batch: bool = True, unavailable: str | None = None) -> None:
        self._script: Script = script or {}
        self._error = error
        self._unavailable = unavailable
        self.hosted = hosted
        if not batch:
            self.capabilities = frozenset({"choice", "yes_no", "scale"})
        self.calls: list[tuple[dict[str, Any], tuple[str, ...]]] = []

    def unavailable(self) -> str | None:
        return self._unavailable

    def decide(
        self,
        context: Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        model: str | None,
        timeout: float,
    ) -> ProviderReply:
        self.calls.append((dict(context), tuple(questions)))
        if self._error is not None:
            raise self._error
        answers = {name: self._answer(context, name, q) for name, q in questions.items()}
        return ProviderReply(answers=answers, model=model or "fake")

    def _answer(self, context: Mapping[str, Any], name: str, question: Question) -> RawAnswer:
        if callable(self._script):
            scripted: Scripted | None = self._script(context, name, question)
        else:
            scripted = self._script.get(name)
        if isinstance(scripted, RawAnswer):
            return scripted
        labels = question.labels
        if scripted is None:
            choice, certainty = labels[0], 1.0
        elif isinstance(scripted, tuple):
            choice, certainty = scripted
        else:
            choice, certainty = scripted, 1.0
        others = [label for label in labels if label != choice]
        rest = (1.0 - certainty) / len(others) if others else 0.0
        chances = {label: (certainty if label == choice else rest) for label in labels}
        return RawAnswer(choice=choice, chances=chances, certainty=certainty)
