"""OrdersConnector — a `Connector` to an external orders/billing microservice.

`Connector` is aixon's base for HTTP clients: declare ``base_url_env`` /
``auth_token_env`` and call ``self.get`` / ``self.post`` (JSON in, JSON out,
Bearer auth, lazy ``httpx``).

So the example runs with zero infrastructure, ``lookup_order`` falls back to a
small in-memory fixture when ``ORDERS_API_URL`` is **not** set. Point that env
var at a real service and the very same method issues a real HTTP GET — nothing
else changes. The real HTTP path (``self.get``) is covered by the test suite
with a mocked transport.
"""

from __future__ import annotations

import re

from aixon import Connector

# Deterministic in-memory orders, used only when ORDERS_API_URL is unset.
_DEMO_ORDERS: dict[str, dict] = {
    "1001": {"order_id": "1001", "status": "delivered", "item": "Acme Pro (annual)", "eta": None},
    "1002": {"order_id": "1002", "status": "in_transit", "item": "USB-C hub", "eta": "2026-06-27"},
    "1003": {"order_id": "1003", "status": "processing", "item": "Mechanical keyboard", "eta": "2026-07-01"},
}


class OrdersConnector(Connector):
    """Client for the orders service. Reads base URL / token from the env."""

    base_url_env = "ORDERS_API_URL"
    auth_token_env = "ORDERS_API_TOKEN"

    def lookup_order(self, order_id: str) -> dict:
        """Return one order as a dict.

        Real path (ORDERS_API_URL set): ``GET {base_url}/orders/{order_id}``.
        Offline path: a deterministic in-memory fixture.
        """
        if self.base_url:
            return self.get(f"/orders/{order_id}")
        order = _DEMO_ORDERS.get(order_id)
        if order is None:
            return {"order_id": order_id, "status": "not_found"}
        return order

    async def alookup_order(self, order_id: str) -> dict:
        """Async variant of ``lookup_order`` — uses ``aget`` (httpx.AsyncClient)
        when ``ORDERS_API_URL`` is set, so a real deployment never blocks the
        event loop. Falls back to the in-memory fixture offline."""
        if self.base_url:
            return await self.aget(f"/orders/{order_id}")
        order = _DEMO_ORDERS.get(order_id)
        if order is None:
            return {"order_id": order_id, "status": "not_found"}
        return order


def extract_order_id(text: str) -> str:
    """Pull the first run of digits out of free text ('order 1002?' -> '1002')."""
    match = re.search(r"\d+", text or "")
    return match.group(0) if match else ""
