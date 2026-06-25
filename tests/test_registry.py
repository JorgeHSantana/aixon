import pytest
from aixon.registry import Registry, get_registry, reset_registry
from aixon.exceptions import RegistrationError, AgentNotFoundError


class _FakeAgent:
    def __init__(self, name, aliases=None, hidden=False):
        self.name = name
        self.aliases = aliases or []
        self.hidden = hidden


def test_register_and_resolve_by_name():
    reg = Registry()
    a = _FakeAgent("alpha")
    reg.register(a)
    assert reg.resolve("alpha") is a


def test_resolve_by_alias():
    reg = Registry()
    a = _FakeAgent("alpha", aliases=["a1", "a2"])
    reg.register(a)
    assert reg.resolve("a2") is a


def test_duplicate_name_raises():
    reg = Registry()
    reg.register(_FakeAgent("alpha"))
    with pytest.raises(RegistrationError, match="alpha"):
        reg.register(_FakeAgent("alpha"))


def test_alias_collision_with_name_raises():
    reg = Registry()
    reg.register(_FakeAgent("alpha"))
    with pytest.raises(RegistrationError):
        reg.register(_FakeAgent("beta", aliases=["alpha"]))


def test_single_agent_resolves_only_for_empty_model():
    # An empty/missing model on a single-agent registry resolves to that agent...
    reg = Registry()
    only = _FakeAgent("alpha")
    reg.register(only)
    assert reg.resolve("") is only
    # ...but a NON-empty wrong name raises instead of masking the typo.
    with pytest.raises(AgentNotFoundError):
        reg.resolve("anything-else")


def test_unknown_name_with_multiple_agents_raises():
    reg = Registry()
    reg.register(_FakeAgent("alpha"))
    reg.register(_FakeAgent("beta"))
    with pytest.raises(AgentNotFoundError):
        reg.resolve("gamma")


def test_public_excludes_hidden():
    reg = Registry()
    visible = _FakeAgent("v")
    reg.register(visible)
    reg.register(_FakeAgent("h", hidden=True))
    assert reg.public() == [visible]


def test_global_singleton_is_stable_and_resettable():
    reset_registry()
    get_registry().register(_FakeAgent("alpha"))
    assert get_registry().resolve("alpha").name == "alpha"
    reset_registry()
    assert get_registry().all() == []
