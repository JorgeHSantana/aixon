# tests/test_audit_task_6_2.py
"""Achados exceptions-1..3: a classificação de erro do gateway mora no módulo
do provedor (aixon.providers.gateway_errors); exceptions só reexporta."""
import pytest

from aixon.exceptions import AixonError, GatewayBudgetExceeded


def test_exceptions_1(monkeypatch):
    """Um só ponto lê AIXON_GATEWAY_BASE_URL: o provedor e o classificador
    veem a mesma URL (strip), e required=True levanta o erro de sempre."""
    from aixon.providers import gateway_errors
    from aixon.providers.gateway import GatewayProvider

    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "  https://GW.example:443/v1/  ")
    assert GatewayProvider().base_url() == "https://GW.example:443/v1/"
    assert gateway_errors.gateway_base_url(required=True) == "https://GW.example:443/v1/"
    assert gateway_errors._gateway_base() == "https://gw.example/v1"

    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "   ")
    assert gateway_errors.gateway_base_url() is None
    with pytest.raises(AixonError, match="AIXON_GATEWAY_BASE_URL is missing or empty"):
        GatewayProvider().base_url()
    with pytest.raises(AixonError, match="never falls back to OPENAI_API_KEY"):
        gateway_errors.gateway_base_url(required=True)


def test_exceptions_2():
    """O contrato do LiteLLM vira constantes nomeadas no módulo do provedor."""
    from aixon.providers import gateway_errors

    assert gateway_errors.BUDGET_STATUS == 422
    assert gateway_errors.BUDGET_MARKER == "budget"
    assert gateway_errors.RATE_LIMIT_STATUS == 429


def test_exceptions_3():
    """exceptions não importa providers no topo; o classify de compatibilidade
    delega ao do módulo novo; aixon reexporta o mesmo nome."""
    import ast
    import inspect

    import aixon.exceptions as e

    tree = ast.parse(inspect.getsource(e))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [
        n.module for n in top if isinstance(n, ast.ImportFrom) and n.module
    ]
    assert not any(str(x).startswith("aixon.providers") for x in names)
    assert not hasattr(e, "_gateway_base") and not hasattr(e, "_sdk_verdict")
    assert e.classify_gateway_error(e.GatewayRateLimited("x")) is e.GatewayRateLimited

    import aixon
    from aixon.providers import gateway_errors

    assert aixon.classify_gateway_error(GatewayBudgetExceeded("x")) is GatewayBudgetExceeded
    assert gateway_errors.classify_gateway_error(GatewayBudgetExceeded("x")) is GatewayBudgetExceeded
