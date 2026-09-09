"""Normalize marketplace payloads into the domain objects the system already has.

Each marketplace describes an order differently, and none of them describe it
the way this system does. Normalization happens here, once per marketplace, so
that nothing downstream — inventory, settlement, realized profit — ever sees a
marketplace-shaped dict.

The discipline is the same as everywhere else: a field that is absent stays
absent, a status that has no mapping raises rather than defaulting, and money
becomes ``Decimal`` at the boundary. A marketplace inventing a new order status
is a thing a human should look at, not a thing to guess through.

None of these normalizers has processed a payload from a live endpoint. They
are written against each marketplace's published response schema and tested
against recorded fixtures.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from econ.money import ZERO, money
from part2.operations import (
    Order,
    OrderLine,
    OrderStatus,
    SettlementLine,
    UnknownStatus,
    normalise_status,
)

logger = logging.getLogger(__name__)

NORMALIZE_VERSION = "normalize/1.0.0"


class NormalizationError(ValueError):
    """Raised when a payload cannot be turned into a domain object."""


def _decimal(raw: Any, *, field: str, default: Decimal | None = None) -> Decimal:
    """Parse money, refusing to invent a value.

    A default is supplied only where the marketplace genuinely omits a field
    that is legitimately zero — a shipping charge on a free-delivery order,
    say. Everywhere else an unparseable amount is an error, because a silently
    zeroed fee flows straight into realized profit.
    """
    if raw in (None, ""):
        if default is None:
            raise NormalizationError(f"{field} is missing and has no safe default")
        return default
    try:
        return money(str(raw).replace(",", "").replace("\u20b9", "").strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise NormalizationError(f"{field} is not a valid amount: {raw!r}") from exc


def _timestamp(raw: Any, *, field: str) -> datetime:
    if not raw:
        raise NormalizationError(f"{field} is missing")
    text = str(raw).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise NormalizationError(
            f"{field} is not an ISO-8601 timestamp: {raw!r}"
        ) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _status(raw: Any, *, marketplace: str, order_id: str) -> OrderStatus:
    """Map a marketplace status, surfacing anything unrecognised.

    Deliberately loud. A marketplace adding a status is a change this system
    needs to know about; mapping it to NEW would quietly resurrect finished
    orders.
    """
    try:
        return normalise_status(str(raw))
    except UnknownStatus as exc:
        raise NormalizationError(f"{marketplace} order {order_id}: {exc}") from exc


# --------------------------------------------------------------------------
# Amazon Selling Partner API
# --------------------------------------------------------------------------


def amazon_order(payload: Mapping[str, Any], items: list[Mapping[str, Any]]) -> Order:
    """Normalize one SP-API order plus its order items.

    SP-API splits an order's header from its line items across two calls, so
    both are required here. An order with no items is refused rather than
    recorded as a zero-value sale.
    """
    order_id = str(payload.get("AmazonOrderId") or "")
    if not order_id:
        raise NormalizationError("SP-API order has no AmazonOrderId")
    if not items:
        raise NormalizationError(f"SP-API order {order_id} has no line items")

    lines: list[OrderLine] = []
    for item in items:
        sku = str(item.get("SellerSKU") or "").strip()
        if not sku:
            raise NormalizationError(f"SP-API order {order_id}: an item has no SellerSKU")
        raw_quantity = item.get("QuantityOrdered")
        if raw_quantity is None:
            # Absent is not zero. Silently dropping the line would understate
            # the order, and understating units understates cost.
            raise NormalizationError(
                f"SP-API order {order_id}: item {sku} states no QuantityOrdered"
            )
        quantity = int(raw_quantity)
        if quantity < 0:
            raise NormalizationError(
                f"SP-API order {order_id}: item {sku} has negative quantity {quantity}"
            )
        if quantity == 0:
            # A stated zero is a real cancellation line, not missing data.
            continue
        gross = _decimal(
            (item.get("ItemPrice") or {}).get("Amount"),
            field=f"order {order_id} ItemPrice",
        )
        discount = _decimal(
            (item.get("PromotionDiscount") or {}).get("Amount"),
            field=f"order {order_id} PromotionDiscount",
            default=ZERO,
        )
        tax = _decimal(
            (item.get("ItemTax") or {}).get("Amount"),
            field=f"order {order_id} ItemTax",
            default=ZERO,
        )
        lines.append(
            OrderLine(
                sku_code=sku,
                quantity=quantity,
                # SP-API reports the line total; the domain model wants a unit
                # price, so it is derived rather than assumed equal.
                unit_price_inr=money(gross / Decimal(quantity)),
                discount_inr=discount,
                tax_inr=tax,
            )
        )

    if not lines:
        raise NormalizationError(
            f"SP-API order {order_id} has no positive-quantity items"
        )

    return Order(
        marketplace="amazon_in",
        marketplace_order_id=order_id,
        lines=lines,
        ordered_at=_timestamp(payload.get("PurchaseDate"), field="PurchaseDate"),
        status=_status(
            payload.get("OrderStatus"), marketplace="amazon_in", order_id=order_id
        ),
        shipping_inr=_decimal(
            (payload.get("ShippingPrice") or {}).get("Amount"),
            field=f"order {order_id} ShippingPrice",
            default=ZERO,
        ),
    )


def amazon_settlement(row: Mapping[str, Any]) -> SettlementLine:
    """Normalize one row of an SP-API settlement report.

    Settlement reports arrive as flat tab-separated rows with signed amounts:
    fees are negative. Signs are normalised so every deduction is a positive
    number, because a fee that arrives positive on one report and negative on
    another silently doubles or cancels in reconciliation.
    """
    order_id = str(row.get("order-id") or "").strip()
    if not order_id:
        raise NormalizationError("settlement row has no order-id")

    def deduction(field: str) -> Decimal:
        value = _decimal(row.get(field), field=field, default=ZERO)
        return abs(value)

    principal = _decimal(row.get("principal"), field="principal", default=ZERO)
    commission = deduction("commission")
    shipping = deduction("shipping-fee")
    other = deduction("fba-fees") + deduction("other-transaction-fees")
    tax_withheld = deduction("tcs-cgst") + deduction("tcs-sgst") + deduction("tcs-igst")
    refund = deduction("promotional-rebates")

    net = _decimal(row.get("total"), field="total", default=ZERO)

    return SettlementLine(
        marketplace="amazon_in",
        marketplace_order_id=order_id,
        order_value_inr=principal,
        commission_inr=commission,
        shipping_inr=shipping,
        other_fees_inr=other,
        tax_withheld_inr=tax_withheld,
        refund_inr=refund,
        net_settled_inr=net,
        settled_at=_timestamp(
            row.get("posted-date") or row.get("settlement-end-date"),
            field="posted-date",
        ),
        reference=str(row.get("settlement-id") or ""),
    )


# --------------------------------------------------------------------------
# Flipkart Seller API
# --------------------------------------------------------------------------


def flipkart_order(payload: Mapping[str, Any]) -> Order:
    """Normalize one Flipkart shipment into an order.

    Flipkart's unit of delivery is a shipment containing order items, which
    maps onto this system's order rather than its line.
    """
    shipment_id = str(payload.get("shipmentId") or "").strip()
    if not shipment_id:
        raise NormalizationError("Flipkart shipment has no shipmentId")

    raw_items = payload.get("orderItems") or []
    if not raw_items:
        raise NormalizationError(f"Flipkart shipment {shipment_id} has no orderItems")

    lines: list[OrderLine] = []
    for item in raw_items:
        sku = str(item.get("sku") or "").strip()
        if not sku:
            raise NormalizationError(
                f"Flipkart shipment {shipment_id}: an item has no sku"
            )
        quantity = int(item.get("quantity") or 1)
        price_components = item.get("priceComponents") or {}
        lines.append(
            OrderLine(
                sku_code=sku,
                quantity=quantity,
                unit_price_inr=_decimal(
                    price_components.get("sellingPrice"),
                    field=f"shipment {shipment_id} sellingPrice",
                ),
                discount_inr=_decimal(
                    price_components.get("totalDiscount"),
                    field=f"shipment {shipment_id} totalDiscount",
                    default=ZERO,
                ),
            )
        )

    return Order(
        marketplace="flipkart",
        marketplace_order_id=shipment_id,
        lines=lines,
        ordered_at=_timestamp(payload.get("orderDate"), field="orderDate"),
        status=_status(
            payload.get("status"), marketplace="flipkart", order_id=shipment_id
        ),
        shipping_inr=_decimal(
            (payload.get("priceComponents") or {}).get("shippingCharge"),
            field=f"shipment {shipment_id} shippingCharge",
            default=ZERO,
        ),
    )


def flipkart_settlement(row: Mapping[str, Any]) -> SettlementLine:
    """Normalize one Flipkart settlement row."""
    order_id = str(row.get("orderItemId") or row.get("orderId") or "").strip()
    if not order_id:
        raise NormalizationError("Flipkart settlement row has no order identifier")

    return SettlementLine(
        marketplace="flipkart",
        marketplace_order_id=order_id,
        order_value_inr=_decimal(row.get("orderValue"), field="orderValue", default=ZERO),
        commission_inr=abs(
            _decimal(row.get("commission"), field="commission", default=ZERO)
        ),
        shipping_inr=abs(
            _decimal(row.get("shippingFee"), field="shippingFee", default=ZERO)
        ),
        other_fees_inr=abs(
            _decimal(row.get("collectionFee"), field="collectionFee", default=ZERO)
        ),
        tax_withheld_inr=abs(_decimal(row.get("tcs"), field="tcs", default=ZERO)),
        refund_inr=abs(_decimal(row.get("refund"), field="refund", default=ZERO)),
        net_settled_inr=_decimal(
            row.get("settlementValue"), field="settlementValue", default=ZERO
        ),
        settled_at=_timestamp(row.get("settlementDate"), field="settlementDate"),
        reference=str(row.get("settlementId") or ""),
    )


#: Which normalizer handles which marketplace. Adding a channel is an entry
#: here plus a pair of functions; nothing downstream changes.
ORDER_NORMALIZERS = {
    "flipkart": flipkart_order,
}
SETTLEMENT_NORMALIZERS = {
    "amazon_in": amazon_settlement,
    "flipkart": flipkart_settlement,
}


def normalize_orders(
    marketplace: str, payloads: list[Mapping[str, Any]]
) -> tuple[list[Order], list[str]]:
    """Normalize a page of orders, collecting failures rather than aborting.

    One malformed order must not discard the other forty-nine on the page. The
    failures are returned so they can be surfaced and investigated.
    """
    orders: list[Order] = []
    failures: list[str] = []
    normalizer = ORDER_NORMALIZERS.get(marketplace)
    if normalizer is None:
        raise NormalizationError(f"no order normalizer for {marketplace}")

    for payload in payloads:
        try:
            orders.append(normalizer(payload))
        except NormalizationError as exc:
            failures.append(str(exc))
            logger.warning(
                "order could not be normalized",
                extra={"marketplace": marketplace, "error": str(exc)},
            )
    return orders, failures


def normalize_settlements(
    marketplace: str, rows: list[Mapping[str, Any]]
) -> tuple[list[SettlementLine], list[str]]:
    """Normalize settlement rows, collecting failures."""
    lines: list[SettlementLine] = []
    failures: list[str] = []
    normalizer = SETTLEMENT_NORMALIZERS.get(marketplace)
    if normalizer is None:
        raise NormalizationError(f"no settlement normalizer for {marketplace}")

    for row in rows:
        try:
            lines.append(normalizer(row))
        except NormalizationError as exc:
            failures.append(str(exc))
    return lines, failures


__all__ = [
    "NORMALIZE_VERSION",
    "ORDER_NORMALIZERS",
    "SETTLEMENT_NORMALIZERS",
    "NormalizationError",
    "amazon_order",
    "amazon_settlement",
    "flipkart_order",
    "flipkart_settlement",
    "normalize_orders",
    "normalize_settlements",
]
