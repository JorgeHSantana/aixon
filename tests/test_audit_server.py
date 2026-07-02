"""Tests for audit fixes: usage token-count robustness, server error envelopes,
SSE mid-stream error handling, method-scoped auth exemptions, and CLI
hardening (serve extras probe, autodiscover diagnostics, remote /menu loop,
in-process chat error recovery)."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import make_echo


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


@pytest.fixture
def runner():
    return CliRunner()


# --- Fix 1: count_tokens degrades gracefully on special-token markers -----
def test_count_tokens_special_marker_degrades_gracefully():
    pytest.importorskip("tiktoken")
    from aixon.server.usage import count_tokens

    n = count_tokens("gpt-4o-mini", "hello <|endoftext|> world")
    assert n is not None and n > 0


def test_build_usage_special_marker_never_raises():
    from aixon.server.usage import build_usage

    usage = build_usage("gpt-4o-mini", "<|endoftext|>", "ok <|endoftext|>")
    assert isinstance(usage, dict)  # {} or populated — never an exception


# --- Fix 2: malformed / non-dict bodies -> 400 error envelope --------------
def _openai_client():
    return TestClient(
        Server(adapters=[OpenAIAdapter()]).app, raise_server_exceptions=False
    )


def test_invalid_json_body_returns_400_envelope():
    make_echo("echo")
    r = _openai_client().post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["message"]
    assert err["type"] == "invalid_request_error"


@pytest.mark.parametrize("body", ["x", [1], 42, None])
def test_non_dict_json_body_returns_400_envelope(body):
    make_echo("echo")
    r = _openai_client().post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


# --- Fix 3: agent/provider errors -> SSE error event / 500 envelope --------
def _register_boom():
    class BoomAgent(Agent):
        name = "boom"

        def invoke(self, messages):
            raise RuntimeError("provider exploded")

        def stream(self, messages):
            yield Chunk(content="partial")
            raise RuntimeError("provider exploded")


def _sse_data_lines(text: str) -> list[str]:
    return [l[len("data: "):] for l in text.splitlines() if l.startswith("data: ")]


def test_stream_error_emits_error_event_and_done():
    _register_boom()
    r = _openai_client().post("/v1/chat/completions", json={
        "model": "boom",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    datas = _sse_data_lines(r.text)
    assert datas[-1] == "[DONE]"  # stream still terminates normally
    payloads = [json.loads(d) for d in datas if d != "[DONE]"]
    errors = [p for p in payloads if "error" in p]
    assert errors, f"no error event in stream: {payloads}"
    assert "provider exploded" in errors[0]["error"]["message"]
    assert errors[0]["error"]["type"] == "server_error"
    # Chunks produced before the failure are still delivered.
    contents = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in payloads
        if p.get("choices")
    )
    assert "partial" in contents


def test_non_stream_error_returns_500_envelope():
    _register_boom()
    r = _openai_client().post("/v1/chat/completions", json={
        "model": "boom",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 500
    err = r.json()["error"]
    assert "provider exploded" in err["message"]
    assert err["type"] == "server_error"


# --- Fix 6: auth exemption is method-scoped, not path-wide -----------------
class _DualMethodAdapter(OpenAIAdapter):
    """Adapter with a GET and a POST route on the SAME path: the GET
    (model list) is public, but the POST must still require auth."""

    name = "dual"

    def routes(self):
        return [("POST", "/dual"), ("GET", "/dual")]


def test_auth_public_exemption_is_method_scoped(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "secret123")
    make_echo("echo")
    client = TestClient(Server(adapters=[_DualMethodAdapter()]).app)
    payload = {"model": "echo", "messages": [{"role": "user", "content": "hi"}]}

    assert client.get("/dual").status_code == 200  # model list stays public
    assert client.post("/dual", json=payload).status_code == 401
    assert client.post(
        "/dual", json=payload, headers={"Authorization": "Bearer secret123"}
    ).status_code == 200


# --- Fix 4: serve probes the real extra deps, not just the import ----------
def _all_output(result) -> str:
    """CliRunner stdout + stderr regardless of click's capture mode."""
    try:
        return result.output + result.stderr
    except (ValueError, AttributeError):
        return result.output


def test_serve_missing_uvicorn_shows_install_hint(runner):
    """aixon.server.server imports fine without the extra (its heavy deps are
    lazy), so serve must probe fastapi/uvicorn explicitly instead of relying
    on a dead ImportError guard."""
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "uvicorn":
            return None
        return real_find_spec(name, *args, **kwargs)

    from aixon.cli import app
    with (
        patch("aixon.cli.autodiscover"),
        # Guard: if the probe is missing/broken, don't start a real server.
        patch.dict("sys.modules", {"uvicorn": MagicMock()}),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
    ):
        result = runner.invoke(app, ["serve"], catch_exceptions=True)

    assert result.exit_code != 0
    assert "aixon[server]" in _all_output(result)


# --- Fix 5: autodiscover errors inside agent modules are surfaced ----------
def test_list_quiet_when_agents_package_absent(runner):
    err = ModuleNotFoundError("No module named 'agents'", name="agents")
    from aixon.cli import app
    with patch("aixon.cli.autodiscover", side_effect=err):
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No agents registered" in result.output
    assert "No module named" not in _all_output(result)  # quiet skip


def test_list_surfaces_import_error_from_agent_module(runner):
    """A missing third-party lib inside agents/weather.py is NOT 'no agents
    package here' — the user must see the real error."""
    err = ModuleNotFoundError("No module named 'missing_lib'", name="missing_lib")
    from aixon.cli import app
    with patch("aixon.cli.autodiscover", side_effect=err):
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "missing_lib" in _all_output(result)


def test_chat_surfaces_import_error_from_agent_module(runner):
    err = ImportError("cannot import name 'foo' from 'missing_lib'")
    from aixon.cli import app
    with patch("aixon.cli.autodiscover", side_effect=err):
        result = runner.invoke(app, ["chat"], input="0\n")
    assert result.exit_code == 0
    assert "missing_lib" in _all_output(result)


# --- Fix 7: remote /menu loops instead of recursing -------------------------
def _make_fake_openai_client(model_ids, content="remote reply"):
    """Mock openai.OpenAI client with canned models + one-chunk streams."""
    models_list = MagicMock()
    models_list.data = [MagicMock(id=m) for m in model_ids]

    def _make_stream(*args, **kwargs):
        event = MagicMock()
        delta = MagicMock()
        delta.content = content
        choice = MagicMock()
        choice.delta = delta
        event.choices = [choice]
        stream = MagicMock()
        stream.__iter__ = MagicMock(return_value=iter([event]))
        return stream

    client = MagicMock()
    client.models.list = MagicMock(return_value=models_list)
    client.chat.completions.create = MagicMock(side_effect=_make_stream)
    return client


def test_remote_menu_loops_without_recursion(runner):
    client = _make_fake_openai_client(["echoagent"])
    from aixon.cli import app
    with patch("aixon.cli.OpenAI", return_value=client):
        result = runner.invoke(
            app,
            ["chat", "--url", "http://localhost:8000"],
            input="1\n/menu\n1\nhi\n/exit\n",
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert "Goodbye" in result.output
    # /menu re-enters the menu via a loop, not by re-calling _chat_remote —
    # so the server is contacted for the model list exactly once.
    assert client.models.list.call_count == 1
    # The menu itself is shown again after /menu.
    assert result.output.count("Remote Agents") >= 2


# --- Fix 8: in-process chat survives provider errors ------------------------
def test_inprocess_chat_survives_provider_error(runner):
    class CrashingAgent(Agent):
        description = "always fails"

        def invoke(self, messages: list[Message]) -> Message:
            raise RuntimeError("provider down")

        def stream(self, messages: list[Message]):
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    from aixon.cli import app
    with patch("aixon.cli.autodiscover"):
        result = runner.invoke(
            app, ["chat"], input="1\nhello\n/exit\n", catch_exceptions=True
        )
    # The error is reported and the chat loop continues to /exit.
    assert result.exit_code == 0
    assert "provider down" in _all_output(result)
    assert "Goodbye" in result.output
