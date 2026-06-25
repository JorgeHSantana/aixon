import re
import tomllib
from pathlib import Path


def _pyproject() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _dep_names(deps: list[str]) -> set[str]:
    """Extract bare package names from PEP 508 dependency strings."""
    return {re.split(r"[><=~!\[ ]", d, maxsplit=1)[0] for d in deps}


def test_langchain_stack_is_a_core_dependency():
    """langchain / langchain-core / langgraph are MANDATORY — aixon does not
    function without them (every agent subtype and the orchestrator need them),
    so they live in core `project.dependencies`, NOT an optional extra.
    `import aixon` must always pull them in."""
    names = _dep_names(_pyproject()["project"]["dependencies"])
    assert {"langchain", "langchain-core", "langgraph"} <= names, (
        "langchain/langchain-core/langgraph must be core dependencies "
        f"(got core deps: {sorted(names)})"
    )


def test_no_llm_extra_exists():
    """The old `llm` extra is gone — its contents are core dependencies now,
    so there is no optional 'llm' feature to install separately."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "llm" not in extras, (
        "langchain stack is mandatory (core deps); there is no 'llm' extra"
    )


def test_no_orchestration_extra_exists():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "orchestration" not in extras, (
        "langgraph is a core dependency, not a separate 'orchestration' extra"
    )
