"""0.1.31 — the example ``examples/decider`` runs offline, end to end: the three
outcomes, the batch, the agent with doubts and checks, and the calibration
command printed at the end of it."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "decider"


@pytest.fixture
def example(monkeypatch):
    """Import the example's modules fresh, with the offline engine, and leave
    no trace of them (nor of the environment) for the next test."""
    monkeypatch.setenv("AIXON_DECIDER_PROVIDER", "offline")
    monkeypatch.delenv("STORE_OR_PRODUCT", raising=False)
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for name in ("main", "deciders"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("main")
    yield module
    for name in ("main", "deciders"):
        sys.modules.pop(name, None)


def test_the_example_prints_the_three_outcomes_and_the_agent_flow(example, capsys):
    example.main()
    out = capsys.readouterr().out
    assert "decision engine: offline" in out
    # 1. sure / sure / unsure / switched off
    assert "-> store  certainty 0.96  sure" in out
    assert "-> product  certainty 0.94  sure" in out
    assert "-> store  certainty 0.55  UNSURE" in out
    assert "switched off -> no decision" in out
    # 2. two decisions, one request
    assert "upset: yes  certainty 0.97  sure" in out and "urgency: right now" in out
    # 3. unsure doubt -> the agent asks back
    assert "assistant: 'Do you mean the Olive Street store or the product olive?'" in out
    # 3. sure product -> the message reaches the model untouched
    assert "model received: 'What is the margin on olive oil?'" in out
    # 3. upset + wrong period -> acknowledged, checked, redone once
    assert "[The user is upset." in out and "reasoning: 'Reviewing the answer...'" in out
    assert "assistant: 'Sorry about that. In August the Harbor store made $ 41,200.'" in out
    assert "In September" not in out.split("assistant:")[-1]
    # the hosted engine never sees the amount
    assert "'answer': 'In August the Harbor store made #'" in out


def test_the_switch_is_restored_after_the_demo(example, monkeypatch):
    import os

    example.main()
    assert "STORE_OR_PRODUCT" not in os.environ


def test_the_calibration_command_of_the_example(example, monkeypatch):
    from aixon.cli import app

    monkeypatch.chdir(EXAMPLE)
    result = CliRunner().invoke(app, ["decide", "eval", "deciders:StoreOrProductDecider", "examples.jsonl"])
    assert result.exit_code == 0, result.output
    assert "examples: 12    answered: 12" in result.output
    assert "choice accuracy (ignoring certainty): 100.0%" in result.output
    current = next(line for line in result.output.splitlines() if "<- current" in line).split()
    # at 0.8: 8 clear cases right, none wrong with certainty, 4 ambiguous ones in doubt
    assert current[:5] == ["0.80", "8", "0", "4", "0"]


def test_mask_values_keeps_dates_and_names(example):
    from deciders import mask_values

    masked = mask_values({"message": "Harbor on the 3rd?",
                          "answer": "On August 3rd the Harbor store made $ 1,234.50 in 87 sales."})
    assert masked["answer"] == "On August 3rd the Harbor store made # in # sales."
    assert masked["message"] == "Harbor on the 3rd?"
    assert mask_values({"message": "no answer here"}) == {"message": "no answer here"}
