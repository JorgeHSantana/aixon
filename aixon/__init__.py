"""aixon — declarative AI-agent framework."""

from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)

__all__ = [
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
]
