import sys
import types
import pytest

from aixon.discovery import autodiscover
from aixon.registry import get_registry


def _make_pkg(monkeypatch, tmp_path, name="demo_agents"):
    """Create a temp package with one agent module and one underscore module
    that must be skipped. Evicts any cached copy of the package from
    sys.modules first, so a re-import picks up THIS tmp_path (not a stale one
    from an earlier test)."""
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            del sys.modules[mod]
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text(
        "from aixon.agent import Agent\n"
        "from aixon.message import Message, Chunk\n"
        "class AlphaAgent(Agent):\n"
        "    def invoke(self, messages): return Message(role='assistant')\n"
        "    def stream(self, messages): return iter([Chunk(done=True)])\n"
    )
    (pkg / "_skip.py").write_text("raise RuntimeError('should not be imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_autodiscover_imports_modules_and_registers(monkeypatch, tmp_path):
    name = _make_pkg(monkeypatch, tmp_path)
    autodiscover(name)
    assert get_registry().resolve("alphaagent").name == "alphaagent"


def test_autodiscover_skips_underscore_modules(monkeypatch, tmp_path):
    name = _make_pkg(monkeypatch, tmp_path)
    autodiscover(name)  # must not raise from _skip.py


def test_autodiscover_rejects_non_package():
    mod = types.ModuleType("not_a_pkg")
    sys.modules["not_a_pkg"] = mod
    try:
        with pytest.raises(ValueError, match="not a package"):
            autodiscover("not_a_pkg")
    finally:
        del sys.modules["not_a_pkg"]
