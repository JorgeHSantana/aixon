import tomllib
from pathlib import Path


def _pyproject() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def test_langgraph_lives_in_the_llm_extra():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "llm" in extras, "Plan 2 must have created the 'llm' extra"
    assert any(dep.startswith("langgraph") for dep in extras["llm"]), (
        "langgraph must live in the 'llm' extra (contract §9.2) — "
        "there is NO separate 'orchestration' extra"
    )


def test_no_orchestration_extra_exists():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "orchestration" not in extras, (
        "langgraph belongs in 'llm', not a separate 'orchestration' extra "
        "(contract §9.2)"
    )


def test_all_extra_includes_langgraph():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any(dep.startswith("langgraph") for dep in extras["all"])
