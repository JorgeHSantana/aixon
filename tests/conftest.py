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
