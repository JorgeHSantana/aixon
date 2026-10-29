import pytest

from aixon.registry import reset_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def _no_langfuse_from_the_shell(monkeypatch):
    """Langfuse turns itself on from LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY:
    with them in the developer's shell, every request of the Server tests
    would send a trace. Every test starts without them; the Langfuse tests
    set their own."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gateway_base_url(monkeypatch) -> str:
    """Set AIXON_GATEWAY_BASE_URL to a fake gateway and return it. The `.test`
    TLD is reserved (RFC 2606), so nothing can ever resolve it. Opt in with
    `pytestmark = pytest.mark.usefixtures("gateway_base_url")`."""
    url = "http://gateway.test/v1"
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", url)
    return url


@pytest.fixture
def rec():
    """A fresh RecordingProvider registered as "rec": the model that an
    `imposed_model_scope(<model>, "rec")` imposes. `rec.model` answers every
    build and `rec.builds` lists the `(model, params)` each one asked for."""
    from tests._fakes import make_recording_provider

    return make_recording_provider()


@pytest.fixture
def explodiu(monkeypatch) -> list[tuple[str, str]]:
    """Booby-trap the providers an agent tree is DECLARED with: `build()` of
    the OpenAI and Anthropic providers records `(provider, model)` in the
    returned list and raises, so a test of an imposed model asserts
    `explodiu == []`. The list is what makes the assertion mean something: a
    tool's error shield swallows the `AssertionError` (it catches `Exception`)
    and the run goes on, so without the list a specialist that really ran on
    its configured provider would still let the test pass."""
    from aixon.providers.anthropic import AnthropicProvider
    from aixon.providers.openai import OpenAIProvider

    used: list[tuple[str, str]] = []

    def build(self, model, **params):
        used.append((self.name, model))
        raise AssertionError("provedor configurado foi usado")

    monkeypatch.setattr(OpenAIProvider, "build", build)
    monkeypatch.setattr(AnthropicProvider, "build", build)
    return used


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
