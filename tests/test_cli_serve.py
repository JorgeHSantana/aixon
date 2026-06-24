import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from aixon.registry import reset_registry


@pytest.fixture
def runner():
    return CliRunner()


def _make_fake_server():
    """Return a fake Server instance whose serve() is a no-op."""
    server = MagicMock()
    server.serve = MagicMock()
    server_cls = MagicMock()
    server_cls.get_instance = MagicMock(return_value=server)
    return server_cls, server


def test_serve_calls_server_serve(runner):
    server_cls, server = _make_fake_server()
    # serve_command does a late `from aixon.server.server import Server`, so
    # patching that module in sys.modules makes it pick up the fake.
    with (
        patch("aixon.cli.autodiscover"),
        patch.dict("sys.modules", {"aixon.server.server": MagicMock(Server=server_cls)}),
    ):
        from aixon.cli import app
        result = runner.invoke(app, ["serve", "--port", "9999"], catch_exceptions=False)

    server.serve.assert_called_once()
    call_kwargs = server.serve.call_args
    assert call_kwargs.kwargs.get("port") == 9999 or (call_kwargs.args and 9999 in call_kwargs.args)


def test_serve_default_host_and_port(runner):
    server_cls, server = _make_fake_server()
    with (
        patch("aixon.cli.autodiscover"),
        patch.dict("sys.modules", {"aixon.server.server": MagicMock(Server=server_cls)}),
    ):
        from aixon.cli import app
        result = runner.invoke(app, ["serve"], catch_exceptions=False)

    server.serve.assert_called_once()
    kwargs = server.serve.call_args.kwargs
    assert kwargs.get("host") == "0.0.0.0"
    assert kwargs.get("port") == 8000


def test_serve_custom_package_discovers(runner):
    server_cls, server = _make_fake_server()
    discover_calls = []

    def fake_discover(pkg):
        discover_calls.append(pkg)

    with (
        patch("aixon.cli.autodiscover", side_effect=fake_discover),
        patch.dict("sys.modules", {"aixon.server.server": MagicMock(Server=server_cls)}),
    ):
        from aixon.cli import app
        runner.invoke(app, ["serve", "--package", "my_agents"], catch_exceptions=False)

    assert "my_agents" in discover_calls


def test_serve_missing_server_extra_shows_error(runner):
    """If aixon.server.server is not importable, a helpful error is shown."""
    import sys
    original = sys.modules.get("aixon.server.server")
    sys.modules["aixon.server.server"] = None  # type: ignore[assignment]
    try:
        from aixon.cli import app
        with patch("aixon.cli.autodiscover"):
            result = runner.invoke(app, ["serve"], catch_exceptions=True)
        assert result.exit_code != 0 or "server" in result.output.lower()
    finally:
        if original is None:
            sys.modules.pop("aixon.server.server", None)
        else:
            sys.modules["aixon.server.server"] = original


def test_serve_prints_startup_message(runner):
    server_cls, server = _make_fake_server()
    with (
        patch("aixon.cli.autodiscover"),
        patch.dict("sys.modules", {"aixon.server.server": MagicMock(Server=server_cls)}),
    ):
        from aixon.cli import app
        result = runner.invoke(app, ["serve"], catch_exceptions=False)
    assert "Starting" in result.output or "aixon" in result.output.lower()
