"""Rede de igualdade: os kwargs que chegam ao ChatOpenAI não mudam com a base
OpenAICompatProvider. Os esperados foram capturados rodando contra 5c62e48."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from aixon.providers.gateway import GatewayProvider
from aixon.providers.openai import OpenAIProvider
from aixon.providers.xai import XAIProvider
from aixon.providers.zai import ZAIProvider

_ENVS = (
    "OPENAI_API_KEY", "XAI_API_KEY", "XAI_BASE_URL", "ZAI_API_KEY",
    "ZAI_BASE_URL", "AIXON_GATEWAY_API_KEY", "AIXON_GATEWAY_BASE_URL",
    "AIXON_GATEWAY_USER_HEADER",
)


class _FakeChat:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.openai_organization = "org"
        self.root_client = SimpleNamespace(organization="o", project="p")
        self.root_async_client = SimpleNamespace(organization="o", project="p")


@pytest.fixture
def fake_chat(monkeypatch):
    import langchain_openai

    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key")
    monkeypatch.setenv("AIXON_GATEWAY_API_KEY", "gw-key")
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "https://gw.example/v1")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChat)


_COM_PARAMS = {"temperature": 0.2, "reasoning": True, "timeout": 5}


def _kwargs(provider, params: dict[str, Any]) -> dict[str, Any]:
    chat = provider.build("m", **dict(params))
    out = dict(chat.kwargs)
    for key, value in out.items():
        if isinstance(value, SecretStr):
            out[key] = ("SecretStr", value.get_secret_value())
    return out


_RES = {"timeout": 120.0, "max_retries": 2}
_COM = {"temperature": 0.2, "timeout": 5, "max_retries": 2}
_GW = {
    "model": "m", "api_key": ("SecretStr", "gw-key"),
    "base_url": "https://gw.example/v1", "use_responses_api": False,
    "openai_proxy": None,
}


def test_kwargs_iguais_openai_vazio(fake_chat):
    assert _kwargs(OpenAIProvider(), {}) == {
        "model": "m", **_RES, "api_key": ("SecretStr", "sk-openai")}


def test_kwargs_iguais_openai_com_params(fake_chat):
    assert _kwargs(OpenAIProvider(), _COM_PARAMS) == {
        "model": "m", **_COM, "reasoning_effort": "medium",
        "api_key": ("SecretStr", "sk-openai")}


def test_kwargs_iguais_xai_vazio(fake_chat):
    assert _kwargs(XAIProvider(), {}) == {
        "model": "m", "api_key": ("SecretStr", "xai-key"),
        "base_url": "https://api.x.ai/v1", **_RES}


def test_kwargs_iguais_xai_com_params(fake_chat):
    assert _kwargs(XAIProvider(), _COM_PARAMS) == {
        "model": "m", "api_key": ("SecretStr", "xai-key"),
        "base_url": "https://api.x.ai/v1", **_COM, "reasoning_effort": "medium"}


def test_kwargs_iguais_zai_vazio(fake_chat):
    assert _kwargs(ZAIProvider(), {}) == {
        "model": "m", "api_key": ("SecretStr", "zai-key"),
        "base_url": "https://api.z.ai/api/paas/v4", **_RES}


def test_kwargs_iguais_zai_com_params(fake_chat):
    assert _kwargs(ZAIProvider(), _COM_PARAMS) == {
        "model": "m", "api_key": ("SecretStr", "zai-key"),
        "base_url": "https://api.z.ai/api/paas/v4", **_COM,
        "extra_body": {"thinking": {"type": "enabled"}}}


def _sem_clientes(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Os clientes httpx não têm literal: afirma o tipo e compara o resto.
    import httpx

    assert isinstance(kwargs.pop("http_client"), httpx.Client)
    assert isinstance(kwargs.pop("http_async_client"), httpx.AsyncClient)
    return kwargs


def test_kwargs_iguais_gateway_vazio(fake_chat):
    assert _sem_clientes(_kwargs(GatewayProvider(), {})) == {**_GW, **_RES}


def test_kwargs_iguais_gateway_com_params(fake_chat):
    assert _sem_clientes(_kwargs(GatewayProvider(), _COM_PARAMS)) == {
        **_GW, **_COM, "reasoning_effort": "medium"}


def test_openai_sem_chave_nao_passa_api_key(fake_chat, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "api_key" not in _kwargs(OpenAIProvider(), {})


@pytest.mark.parametrize("provider,env,trecho", [
    (XAIProvider(), "XAI_API_KEY", "xAI endpoint"),
    (ZAIProvider(), "ZAI_API_KEY", "z.AI endpoint"),
    (GatewayProvider(), "AIXON_GATEWAY_API_KEY", "never falls back"),
])
def test_chave_ausente_mantem_mensagem(fake_chat, monkeypatch, provider, env, trecho):
    from aixon.exceptions import AixonError

    monkeypatch.delenv(env)
    with pytest.raises(AixonError, match=env) as info:
        provider.build("m")
    assert trecho in str(info.value)
    assert "OPENAI_API_KEY" in str(info.value)


def test_gateway_finish_zera_organizacao_e_projeto(fake_chat):
    chat = GatewayProvider().build("m")
    assert chat.openai_organization is None
    for client in (chat.root_client, chat.root_async_client):
        assert client.organization is None and client.project is None


def test_openai_api_key_do_chamador_e_sobrescrito_pela_env(fake_chat):
    kwargs = _kwargs(OpenAIProvider(), {"api_key": "do-chamador"})
    assert kwargs["api_key"] == ("SecretStr", "sk-openai")


def _org_intacta(chat) -> None:
    assert chat.openai_organization == "org"
    for client in (chat.root_client, chat.root_async_client):
        assert (client.organization, client.project) == ("o", "p")


def test_xai_nao_toca_org(fake_chat):
    _org_intacta(XAIProvider().build("grok-4"))


def test_openai_nao_toca_org(fake_chat):
    _org_intacta(OpenAIProvider().build("gpt-4o"))
