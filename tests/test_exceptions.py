import pytest
from aixon.exceptions import (
    AixonError,
    NamingError,
    RegistrationError,
    AgentNotFoundError,
    CompositionCycleError,
)


@pytest.mark.parametrize(
    "exc",
    [NamingError, RegistrationError, AgentNotFoundError, CompositionCycleError],
)
def test_all_exceptions_subclass_aixon_error(exc):
    assert issubclass(exc, AixonError)


def test_exception_carries_message():
    err = NamingError("bad name")
    assert str(err) == "bad name"
    assert err.message == "bad name"
