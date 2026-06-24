"""aixon — declarative AI-agent framework."""

from aixon.agent import Agent, AgentTool
from aixon.discovery import autodiscover
from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.logging import Logger
from aixon.message import Chunk, Message, Role
from aixon.providers.base import Provider, get_provider, register_provider
from aixon.registry import Registry, get_registry, reset_registry

__all__ = [
    # Plan 1 — foundation
    "Agent",
    "AgentTool",
    "autodiscover",
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Logger",
    "Message",
    "Chunk",
    "Role",
    "Registry",
    "get_registry",
    "reset_registry",
    # Plan 2 — providers (no langchain dep at import)
    "Provider",
    "get_provider",
    "register_provider",
]

# Plan 2 — LLM + LLMAgent. These pull in langchain_core (the `llm` extra).
# Guard so `import aixon` still works on a bare install without that extra
# (contract §9.4).
try:
    from aixon.llm import LLM
    from aixon.agents.llm_agent import LLMAgent

    __all__ += ["LLM", "LLMAgent"]
except ImportError:  # pragma: no cover - bare install without [llm]
    pass
