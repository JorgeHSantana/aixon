import pytest


@pytest.fixture(autouse=True)
def reset_registry():
    # Replaced in Task 3 with a real registry reset. No-op for now.
    yield
