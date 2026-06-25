import json
import pytest
import httpx

from aixon.connector import Connector
from aixon.exceptions import NamingError


# --- Helpers ---

def _make_transport(routes: dict) -> httpx.MockTransport:
    """
    routes: {(method, path): (status_code, body_dict)}
    e.g. {("GET", "/health"): (200, {"ok": True})}
    """
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "not found"})
        status, body = routes[key]
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


class ApiConnector(Connector):
    """Minimal concrete connector for testing."""
    base_url_env = "API_BASE_URL"
    auth_token_env = "API_AUTH_TOKEN"


# --- Tests ---

def test_connector_suffix_enforced():
    with pytest.raises(NamingError, match="Connector"):
        class BadName(Connector):
            pass


def test_connector_abstract_subtype_exempt():
    class BaseConnector(Connector, abstract=True):
        pass
    class ConcreteConnector(BaseConnector):
        pass
    # Should not raise.


def test_connector_resolves_base_url_from_arg(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    c = ApiConnector(base_url="http://example.com")
    assert c.base_url == "http://example.com"


def test_connector_resolves_base_url_from_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://env-host.com")
    c = ApiConnector()
    assert c.base_url == "http://env-host.com"


def test_connector_resolves_auth_token_from_arg(monkeypatch):
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    c = ApiConnector(auth_token="sk-direct")
    assert c.auth_token == "sk-direct"


def test_connector_resolves_auth_token_from_env(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "sk-from-env")
    c = ApiConnector()
    assert c.auth_token == "sk-from-env"


def test_connector_headers_with_auth_token():
    c = ApiConnector(base_url="http://x.com", auth_token="mytoken")
    headers = c._headers()
    assert headers["Authorization"] == "Bearer mytoken"
    assert headers["Content-Type"] == "application/json"


def test_connector_headers_without_auth_token():
    c = ApiConnector(base_url="http://x.com", auth_token="")
    headers = c._headers()
    assert "Authorization" not in headers
    assert "Content-Type" in headers


def test_connector_get_returns_json(monkeypatch):
    transport = _make_transport({("GET", "/ping"): (200, {"pong": True})})
    c = ApiConnector(base_url="http://test.local")
    # Patch httpx.get to use the mock transport.
    original_get = httpx.get

    def mock_get(url, **kwargs):
        client = httpx.Client(transport=transport)
        return client.get(url, **kwargs)

    monkeypatch.setattr(httpx, "get", mock_get)
    result = c.get("/ping")
    assert result == {"pong": True}


def test_connector_post_returns_json(monkeypatch):
    transport = _make_transport({("POST", "/echo"): (200, {"echoed": True})})
    c = ApiConnector(base_url="http://test.local")

    def mock_post(url, **kwargs):
        client = httpx.Client(transport=transport)
        return client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "post", mock_post)
    result = c.post("/echo", json={"msg": "hi"})
    assert result == {"echoed": True}


def test_connector_get_raises_on_non_2xx(monkeypatch):
    transport = _make_transport({("GET", "/fail"): (500, {"error": "boom"})})
    c = ApiConnector(base_url="http://test.local")

    def mock_get(url, **kwargs):
        client = httpx.Client(transport=transport)
        resp = client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    monkeypatch.setattr(httpx, "get", mock_get)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("/fail")


def test_connector_default_timeout():
    c = ApiConnector(base_url="http://x.com")
    assert c.timeout == 30.0


def test_connector_custom_timeout():
    c = ApiConnector(base_url="http://x.com", timeout=60.0)
    assert c.timeout == 60.0
