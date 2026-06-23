import pytest

from aixon.registry import reset_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_registry()
    yield
    reset_registry()
