import pytest

from aixon.registry import reset_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def gateway_base_url(monkeypatch) -> str:
    """Set AIXON_GATEWAY_BASE_URL to a fake gateway and return it. The `.test`
    TLD is reserved (RFC 2606), so nothing can ever resolve it. Opt in with
    `pytestmark = pytest.mark.usefixtures("gateway_base_url")`."""
    url = "http://gateway.test/v1"
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", url)
    return url


class _FakeChatOpenAI:
    """Stands in for `langchain_openai.ChatOpenAI` and records the kwargs the
    provider handed it, so a test asserts exactly what would have reached the
    SDK class."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_openai(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    from aixon.providers.openai import OpenAIProvider

    return OpenAIProvider()


@pytest.fixture
def fake_zai(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    from aixon.providers.zai import ZAIProvider

    return ZAIProvider()


@pytest.fixture
def fake_xai(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    from aixon.providers.xai import XAIProvider

    return XAIProvider()
