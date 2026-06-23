"""Exception hierarchy for aixon. Every error subclasses ``AixonError`` and
carries a human-readable ``message``."""


class AixonError(Exception):
    """Base exception for aixon."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NamingError(AixonError):
    """A subclass violated a required class-name suffix."""


class RegistrationError(AixonError):
    """An agent could not be registered (duplicate name or alias clash)."""


class AgentNotFoundError(AixonError):
    """No registered agent matches the requested name or alias."""


class CompositionCycleError(AixonError):
    """A cycle was detected in the agent composition graph (A uses B as a
    tool and B uses A, directly or transitively)."""
