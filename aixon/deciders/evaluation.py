"""Calibrating a ``Decider`` against labelled examples.

A decider is only worth turning on if, at its threshold, it is rarely WRONG
WITH CERTAINTY and does not fall into doubt on cases that were clear. This
module answers that with a table — one row per threshold — so ``certainty`` is
chosen from data instead of by feel. It backs ``aixon decide eval``.

Examples file: JSON Lines, one object per line::

    {"context": {"message": "how much did the Acerola store sell?", "word": "acerola"}, "expected": "store"}
    {"context": {"message": "acerola?", "word": "acerola"}, "expected": null}

``expected`` is the right option, or ``null`` when the right behaviour is to be
UNSURE (a genuinely ambiguous case, where the caller should ask the user back).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from aixon.decider import Decider, Decision
from aixon.exceptions import AixonError

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class Example:
    context: dict[str, Any]
    expected: str | None


@dataclass(frozen=True)
class ThresholdRow:
    """What happens to the answered examples at one certainty threshold.

    - ``right``: sure and correct.
    - ``wrong_sure``: sure and WRONG (or sure where doubt was expected). The
      number that must stay near zero: nothing downstream catches it.
    - ``right_doubt``: unsure where doubt was expected.
    - ``needless_doubt``: unsure on a case that had a right answer. Costs a
      question back to the user.
    """

    threshold: float
    right: int
    wrong_sure: int
    right_doubt: int
    needless_doubt: int


@dataclass(frozen=True)
class Report:
    decider: str
    provider: str | None
    total: int
    answered: int
    accuracy: float | None
    median_latency_ms: float | None
    rows: list[ThresholdRow] = field(default_factory=list)

    @property
    def no_decision(self) -> int:
        return self.total - self.answered


def load_examples(path: str | Path) -> list[Example]:
    """Read a JSON Lines examples file. Raises ``AixonError`` naming the line
    on a malformed entry."""
    examples: list[Example] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            context, expected = data["context"], data["expected"]
            if not isinstance(context, dict) or not (expected is None or isinstance(expected, str)):
                raise ValueError("`context` must be an object and `expected` a string or null")
        except (ValueError, KeyError, TypeError) as exc:
            raise AixonError(f"{path}:{number}: invalid example ({exc}). Expected "
                             f'{{"context": {{...}}, "expected": "<option>" | null}}.')
        examples.append(Example(context, expected))
    return examples


def evaluate(decider: type[Decider], examples: Iterable[Example],
             thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> Report:
    """Ask *decider* every example and tabulate the outcome per threshold."""
    examples = list(examples)
    labels = set(decider.question().labels)
    for example in examples:
        if example.expected is not None and example.expected not in labels:
            raise AixonError(f"example expects '{example.expected}', which is not an option of "
                             f"{decider.__name__} ({sorted(labels)}).")
    answered: list[tuple[Example, Decision]] = []
    for example in examples:
        decision = decider.decide(example.context)
        if decision is not None:
            answered.append((example, decision))
    labelled = [(e, d) for e, d in answered if e.expected is not None]
    rows = []
    for threshold in thresholds:
        right = wrong_sure = right_doubt = needless_doubt = 0
        for example, decision in answered:
            sure = decision.certainty >= threshold
            if example.expected is None:
                wrong_sure += sure
                right_doubt += not sure
            elif not sure:
                needless_doubt += 1
            elif decision.choice == example.expected:
                right += 1
            else:
                wrong_sure += 1
        rows.append(ThresholdRow(threshold, right, wrong_sure, right_doubt, needless_doubt))
    return Report(
        decider=decider.name,
        provider=answered[0][1].provider if answered else None,
        total=len(examples),
        answered=len(answered),
        accuracy=(sum(d.choice == e.expected for e, d in labelled) / len(labelled)) if labelled else None,
        median_latency_ms=statistics.median(d.latency_ms for _, d in answered) if answered else None,
        rows=rows,
    )


def render(report: Report, *, current: float | None = None) -> str:
    """The report as plain text. *current* marks the decider's own threshold."""
    lines = [f"decider: {report.decider}    provider: {report.provider or '-'}",
             f"examples: {report.total}    answered: {report.answered}    "
             f"no decision: {report.no_decision}"]
    if not report.answered:
        lines.append("No example was answered: is the provider's key set and the decider switched on?")
        return "\n".join(lines)
    if report.accuracy is not None:
        lines.append(f"choice accuracy (ignoring certainty): {report.accuracy:.1%}")
    lines.append(f"median latency: {report.median_latency_ms:.0f} ms")
    lines.append("")
    lines.append(f"{'certainty':>9}  {'right':>6}  {'WRONG, sure':>11}  {'right doubt':>11}  {'needless doubt':>14}")
    for row in report.rows:
        mark = "  <- current" if current is not None and abs(row.threshold - current) < 1e-9 else ""
        lines.append(f"{row.threshold:>9.2f}  {row.right:>6}  {row.wrong_sure:>11}  "
                     f"{row.right_doubt:>11}  {row.needless_doubt:>14}{mark}")
    lines.append("")
    lines.append("Pick the lowest certainty that keeps 'WRONG, sure' at zero (or near it); "
                 "above that you only pay in needless doubts.")
    return "\n".join(lines)
