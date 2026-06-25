"""OrdersAgent — a `ToolAgent` that looks up orders via the `OrdersConnector`.

The tool is a plain callable (another accepted tool form) that calls the
connector and emits a reasoning line with `emit_reasoning` — which surfaces in
`stream()` as a ``Chunk(reasoning=...)`` and on `Message.reasoning` from
`invoke()`. ``hidden`` — reached through the orchestrator.
"""

from __future__ import annotations

from aixon import ToolAgent, emit_reasoning

from connectors.orders import OrdersConnector, extract_order_id
from llm_config import make_llm

_orders = OrdersConnector()


def order_status(order_query: str) -> str:
    """Look up an order's status. Accepts an order id or free text mentioning one."""
    order_id = extract_order_id(order_query)
    if not order_id:
        return "No order number found in the request. Please include your order id."
    emit_reasoning(f"Looking up order {order_id} in the orders service...")
    order = _orders.lookup_order(order_id)
    if order.get("status") == "not_found":
        return f"Order {order_id} was not found."
    eta = order.get("eta")
    eta_str = f", ETA {eta}" if eta else ""
    return (
        f"Order {order['order_id']}: {order['item']} — "
        f"status '{order['status']}'{eta_str}."
    )


class OrdersAgent(ToolAgent):
    name = "orders"
    hidden = True
    description = "Looks up order status, shipping and delivery."
    llm = make_llm()
    prompt = (
        "You are Acme's order-support assistant. Use the order_status tool to "
        "look up the customer's order, then reply clearly with the status."
    )
    tools = [order_status]
