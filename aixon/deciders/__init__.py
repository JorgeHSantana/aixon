"""Decision providers sub-package.

Importing it self-registers the built-in engines (``jev``, ``laya``). Their
vendor imports are lazy, so this never needs an API key, ``laya`` or torch.
"""

from aixon.deciders.base import (
    DeciderProvider,
    ProviderReply,
    Question,
    RawAnswer,
    get_decider_provider,
    register_decider_provider,
)
from aixon.deciders.fake import FakeDeciderProvider
from aixon.deciders.jev import JevProvider
from aixon.deciders.laya import LayaProvider
from aixon.deciders.llm import LLMDeciderProvider

__all__ = [
    "DeciderProvider",
    "FakeDeciderProvider",
    "JevProvider",
    "LLMDeciderProvider",
    "LayaProvider",
    "ProviderReply",
    "Question",
    "RawAnswer",
    "get_decider_provider",
    "register_decider_provider",
]
