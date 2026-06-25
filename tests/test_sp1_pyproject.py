from __future__ import annotations

import pathlib
import tomllib


def test_tiktoken_optional_extra_declared():
    root = pathlib.Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "tiktoken" in extras
    assert any(dep.startswith("tiktoken") for dep in extras["tiktoken"])
