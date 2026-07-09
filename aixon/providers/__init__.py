"""Provider sub-package.

Importing this package triggers self-registration of all three concrete
providers (OpenAI, Anthropic, Google). Because their SDK imports are lazy
(inside build()), this never raises ImportError for a missing vendor SDK.
"""
from aixon.providers.base import (
    Provider,
    get_provider,
    register_provider,
    resolve_provider_for_model,
)

# Trigger self-registration. Each module's top-level register_provider() call
# fires when the module is imported. SDK imports stay inside build().
from aixon.providers import anthropic as _anthropic  # noqa: E402,F401
from aixon.providers import google as _google  # noqa: E402,F401
from aixon.providers import openai as _openai  # noqa: E402,F401
from aixon.providers import zai as _zai  # noqa: E402,F401

__all__ = [
    "Provider",
    "get_provider",
    "register_provider",
    "resolve_provider_for_model",
]
