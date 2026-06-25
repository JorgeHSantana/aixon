# tests/test_cli_version.py
"""`aixon --version` and `aixon.__version__` exist (audit 3.13 — the RTK/README
convention treats --version as a basic install sanity check)."""
from __future__ import annotations

import re

from click.testing import CliRunner


def test_package_exposes_dunder_version():
    import aixon

    assert isinstance(aixon.__version__, str)
    assert re.match(r"^\d+\.\d+", aixon.__version__)  # e.g. "0.0.1"


def test_cli_version_flag_prints_version():
    from aixon.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("aixon ")
    assert re.search(r"aixon \d+\.\d+", result.output)
