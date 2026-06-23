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
from aixon.message import Chunk, Message, Role

__all__ = [
    "Agent",
    "AgentTool",
    "autodiscover",
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Message",
    "Chunk",
    "Role",
]
