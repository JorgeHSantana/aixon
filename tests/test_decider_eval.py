"""`aixon decide eval`: the per-threshold table that decides a decider's certainty."""
import json

import pytest
from click.testing import CliRunner

from aixon import AixonError, Decider
from aixon.cli import app
from aixon.deciders import FakeDeciderProvider
from aixon.deciders.evaluation import Example, evaluate, load_examples, render

# word -> (choice, certainty) the fake engine answers
ANSWERS = {
    "store-clear": ("store", 0.97),
    "product-clear": ("product", 0.92),
    "store-weak": ("store", 0.65),        # right, but below 0.8
    "wrong-sure": ("product", 0.85),      # expected store: wrong with certainty
    "ambiguous": ("store", 0.55),         # expected doubt
    "ambiguous-sure": ("store", 0.9),     # expected doubt, engine is sure
}


class StoreOrProductDecider(Decider):
    provider = FakeDeciderProvider(lambda context, name, question: ANSWERS[context["word"]])
    instruction = "What does the word refer to?"
    options = ["store", "product"]


class OfflineDecider(Decider):
    provider = FakeDeciderProvider(unavailable="no_key")
    instruction = "x"
    options = ["store", "product"]


EXAMPLES = [
    Example({"word": "store-clear"}, "store"),
    Example({"word": "product-clear"}, "product"),
    Example({"word": "store-weak"}, "store"),
    Example({"word": "wrong-sure"}, "store"),
    Example({"word": "ambiguous"}, None),
    Example({"word": "ambiguous-sure"}, None),
]


def test_report_per_threshold():
    report = evaluate(StoreOrProductDecider, EXAMPLES, thresholds=(0.5, 0.8, 0.95))
    assert (report.total, report.answered, report.no_decision) == (6, 6, 0)
    assert report.accuracy == 0.75  # 3 of the 4 labelled examples
    low, mid, high = report.rows
    assert (low.right, low.wrong_sure, low.right_doubt, low.needless_doubt) == (3, 3, 0, 0)
    assert (mid.right, mid.wrong_sure, mid.right_doubt, mid.needless_doubt) == (2, 2, 1, 1)
    assert (high.right, high.wrong_sure, high.right_doubt, high.needless_doubt) == (1, 0, 2, 3)


def test_render_marks_the_current_threshold():
    text = render(evaluate(StoreOrProductDecider, EXAMPLES), current=0.8)
    assert "choice accuracy (ignoring certainty): 75.0%" in text
    assert [line for line in text.splitlines() if "<- current" in line][0].split()[0] == "0.80"


def test_no_answer_is_reported_not_hidden():
    report = evaluate(OfflineDecider, EXAMPLES)
    assert report.answered == 0 and report.accuracy is None
    assert "No example was answered" in render(report)


def test_an_expected_value_that_is_not_an_option_is_an_error():
    with pytest.raises(AixonError, match="not an option"):
        evaluate(StoreOrProductDecider, [Example({"word": "store-clear"}, "shop")])


def _file(tmp_path, lines):
    path = tmp_path / "examples.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_load_examples(tmp_path):
    path = _file(tmp_path, [json.dumps({"context": {"word": "a"}, "expected": "store"}), "",
                            json.dumps({"context": {"word": "b"}, "expected": None})])
    assert load_examples(path) == [Example({"word": "a"}, "store"), Example({"word": "b"}, None)]


@pytest.mark.parametrize("line", ["not json", '{"context": {}}', '{"context": "x", "expected": "a"}',
                                  '{"context": {}, "expected": 3}'])
def test_load_examples_names_the_bad_line(tmp_path, line):
    with pytest.raises(AixonError, match="examples.jsonl:2: invalid example"):
        load_examples(_file(tmp_path, [json.dumps({"context": {}, "expected": None}), line]))


def _lines():
    return [json.dumps({"context": e.context, "expected": e.expected}) for e in EXAMPLES]


def test_cli_eval(tmp_path):
    result = CliRunner().invoke(app, ["decide", "eval", "tests.test_decider_eval:StoreOrProductDecider",
                                      _file(tmp_path, _lines()), "-t", "0.8,0.95"])
    assert result.exit_code == 0, result.output
    assert "decider: store_or_product" in result.output and "<- current" in result.output
    assert "0.95" in result.output and "0.50" not in result.output


def test_cli_eval_exits_1_when_nothing_was_answered(tmp_path):
    result = CliRunner().invoke(app, ["decide", "eval", "tests.test_decider_eval:OfflineDecider",
                                      _file(tmp_path, _lines())])
    assert result.exit_code == 1 and "No example was answered" in result.output


@pytest.mark.parametrize("target, message", [
    ("nocolon", "not in the form"),
    ("no.such.module:XDecider", "cannot import"),
    ("tests.test_decider_eval:Missing", "has no 'Missing'"),
    ("tests.test_decider_eval:Example", "is not a Decider subclass"),
])
def test_cli_eval_bad_target(tmp_path, target, message):
    result = CliRunner().invoke(app, ["decide", "eval", target, _file(tmp_path, _lines())])
    assert result.exit_code == 2 and message in result.output


def test_cli_eval_bad_thresholds_and_bad_examples(tmp_path):
    target = "tests.test_decider_eval:StoreOrProductDecider"
    bad = CliRunner().invoke(app, ["decide", "eval", target, _file(tmp_path, _lines()), "-t", "high"])
    assert bad.exit_code == 2 and "numbers separated by commas" in bad.output
    broken = CliRunner().invoke(app, ["decide", "eval", target, _file(tmp_path, ["oops"])])
    assert broken.exit_code == 1 and "invalid example" in broken.output
