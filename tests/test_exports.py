"""Smoke-test: Plan 2 symbols importable from the top-level aixon namespace."""
from tests._fakes import register_fake_provider

register_fake_provider()

import aixon


def test_llm_exported():
    from aixon import LLM
    assert LLM("fake-1", provider="fake").model == "fake-1"


def test_provider_exported():
    from aixon import Provider
    assert isinstance(aixon.Provider, type)


def test_register_get_provider_exported():
    from aixon import get_provider, register_provider
    assert callable(register_provider)
    assert callable(get_provider)


def test_llm_agent_exported():
    import inspect
    from aixon import LLMAgent
    assert inspect.isclass(LLMAgent)


def test_plan1_exports_still_present():
    from aixon import (  # noqa: F401
        Agent,
        AgentTool,
        AixonError,
        AgentNotFoundError,
        CompositionCycleError,
        NamingError,
        RegistrationError,
        Chunk,
        Message,
        Role,
        Logger,
        autodiscover,
        get_registry,
        reset_registry,
    )
    assert True
