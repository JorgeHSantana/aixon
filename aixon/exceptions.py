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


class GatewayBudgetExceeded(AixonError):
    """The gateway's budget is spent (HTTP 422 mentioning "budget").

    aixon does not raise it: the gateway's answer reaches the caller as the
    SDK's ``openai.UnprocessableEntityError``, and ``classify_gateway_error(exc)``
    returns this class for it. A tool may raise it to signal a spent quota."""


class GatewayRateLimited(AixonError):
    """The gateway is rate-limiting the caller (HTTP 429).

    aixon does not raise it: the gateway's answer reaches the caller as the
    SDK's ``openai.RateLimitError``, and ``classify_gateway_error(exc)`` returns
    this class for it. A tool may raise it to signal a rate limit."""


class ImposedModelViolation(AixonError):
    """Code asked an ``LLM`` for the model it was configured with while another
    model is imposed on the call (``aixon.runtime.imposed_model_scope``).
    Raised instead of silently running the call on the configured model."""


def classify_gateway_error(exc: BaseException) -> type[AixonError] | None:
    """Compatibilidade: a classificação mora em
    ``aixon.providers.gateway_errors.classify_gateway_error`` (contrato do
    gateway; o do LiteLLM, por exemplo); este nome continua aqui e em ``aixon`` para quem já o
    importa. Import preguiçoso: exceptions nunca importa providers no topo,
    o que criaria ciclo (gateway_errors importa este módulo)."""
    from aixon.providers.gateway_errors import classify_gateway_error as _classify

    return _classify(exc)


class RequestRejected(AixonError):
    """O ``request_hook`` recusou a request; o Server responde com ``status``
    e ``code`` no envelope de erro do protocolo."""

    def __init__(
        self, message: str, *, status: int = 403, code: str = "request_rejected"
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
