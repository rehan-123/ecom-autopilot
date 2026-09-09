"""Orders, returns, settlement and the realized economics that follow.

This is where Part 1's predictions meet what actually happened. Three ideas
shape it:

**Order facts are immutable.** A status change appends a transition; it never
edits the order. Marketplaces revise, cancel and re-deliver, and an order table
that overwrites itself cannot answer "what did we believe on the 14th".

**Imports are idempotent.** Order feeds, webhooks and settlement files all
arrive more than once. Every import is keyed on the marketplace's own
identifier, so a replay changes nothing.

**Money is reconciled, not assumed.** A marketplace's settlement rarely equals
the order value: fees, TDS/TCS, adjustments and reversals all land later.
Unmatched money stays ``UNMATCHED`` rather than being quietly absorbed into
profit, because the discrepancy is usually where the real problem is.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from econ.money import HUNDRED, ZERO, format_inr, money, safe_div

logger = logging.getLogger(__name__)

ORDERS_VERSION = "orders/1.0.0"


class OrderStatus(StrEnum):
    """Internal statuses. Marketplace vocabularies normalise onto these."""

    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    PACKING = "PACKING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    RETURN_REQUESTED = "RETURN_REQUESTED"
    RETURNED = "RETURNED"
    RTO = "RTO"
    REFUNDED = "REFUNDED"


#: Marketplace vocabulary -> internal status. Anything unrecognised is
#: surfaced rather than guessed: silently mapping an unknown status to NEW
#: would quietly resurrect cancelled orders.
STATUS_MAP: dict[str, OrderStatus] = {
    # Amazon-style
    "pending": OrderStatus.NEW,
    "unshipped": OrderStatus.CONFIRMED,
    "partiallyshipped": OrderStatus.SHIPPED,
    "shipped": OrderStatus.SHIPPED,
    "delivered": OrderStatus.DELIVERED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    # Flipkart-style
    "approved": OrderStatus.CONFIRMED,
    "packing_in_progress": OrderStatus.PACKING,
    "ready_to_dispatch": OrderStatus.PACKING,
    "form_failed": OrderStatus.CANCELLED,
    "return_requested": OrderStatus.RETURN_REQUESTED,
    "returned": OrderStatus.RETURNED,
    "rto": OrderStatus.RTO,
    "rto_initiated": OrderStatus.RTO,
    "refunded": OrderStatus.REFUNDED,
    # Meesho-style export values
    "door_step_exchanged": OrderStatus.RETURNED,
    "rto_complete": OrderStatus.RTO,
}


class UnknownStatus(ValueError):
    """Raised when a marketplace status has no mapping."""


def normalise_status(raw: str) -> OrderStatus:
    """Map a marketplace status onto an internal one.

    Refuses to guess. An unmapped status is a signal that a marketplace
    changed its vocabulary, which is worth a human's attention rather than a
    default.
    """
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return STATUS_MAP[key]
    except KeyError as exc:
        raise UnknownStatus(
            f"unmapped marketplace status {raw!r}; add it to STATUS_MAP rather "
            "than defaulting, so the mapping stays auditable"
        ) from exc


TERMINAL_ORDER_STATES = frozenset(
    {
        OrderStatus.DELIVERED,
        OrderStatus.CANCELLED,
        OrderStatus.RETURNED,
        OrderStatus.RTO,
        OrderStatus.REFUNDED,
    }
)


@dataclass(frozen=True)
class OrderLine:
    """One SKU within an order."""

    sku_code: str
    quantity: int
    unit_price_inr: Decimal
    discount_inr: Decimal = ZERO
    tax_inr: Decimal = ZERO

    @property
    def gross_inr(self) -> Decimal:
        return money(self.unit_price_inr * self.quantity - self.discount_inr)


@dataclass
class Order:
    """A normalised order. Facts never change; status transitions append."""

    marketplace: str
    marketplace_order_id: str
    lines: list[OrderLine]
    ordered_at: datetime
    status: OrderStatus = OrderStatus.NEW
    shipping_inr: Decimal = ZERO
    customer_reference: str | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    transitions: list[tuple[OrderStatus, datetime, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.transitions:
            self.transitions.append((self.status, self.ordered_at, "imported"))

    @property
    def key(self) -> tuple[str, str]:
        """The marketplace's own identity for this order."""
        return (self.marketplace, self.marketplace_order_id)

    @property
    def gross_inr(self) -> Decimal:
        return money(
            sum((line.gross_inr for line in self.lines), ZERO) + self.shipping_inr
        )

    @property
    def units(self) -> int:
        return sum(line.quantity for line in self.lines)

    def transition(
        self, to_status: OrderStatus, *, at: datetime | None = None, note: str = ""
    ) -> None:
        """Append a status change. The order's facts are untouched."""
        # Delivered orders legitimately become returns; cancelled ones do not
        # come back to life.
        if self.status is OrderStatus.CANCELLED and to_status is not self.status:
            raise ValueError(
                f"order {self.marketplace_order_id} is CANCELLED and cannot "
                f"move to {to_status}"
            )
        self.status = to_status
        self.transitions.append((to_status, at or datetime.now(UTC), note))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "marketplace": self.marketplace,
            "marketplace_order_id": self.marketplace_order_id,
            "status": str(self.status),
            "units": self.units,
            "gross_inr": str(self.gross_inr),
            "ordered_at": self.ordered_at.isoformat(),
            "transitions": [
                {"status": str(s), "at": t.isoformat(), "note": n}
                for s, t, n in self.transitions
            ],
        }


class OrderBook:
    """Idempotent store of normalised orders."""

    def __init__(self) -> None:
        self._orders: dict[tuple[str, str], Order] = {}
        self.duplicates_ignored = 0

    def import_order(self, order: Order) -> Order:
        """Import an order, ignoring a repeat of one already held.

        Keyed on the marketplace's identifier rather than ours, because that
        is what a redelivered webhook or a re-downloaded export carries.
        """
        existing = self._orders.get(order.key)
        if existing is not None:
            self.duplicates_ignored += 1
            logger.info(
                "duplicate order ignored",
                extra={
                    "marketplace": order.marketplace,
                    "order_id": order.marketplace_order_id,
                },
            )
            return existing
        self._orders[order.key] = order
        return order

    def get(self, marketplace: str, order_id: str) -> Order | None:
        return self._orders.get((marketplace, order_id))

    def all_orders(self) -> list[Order]:
        return sorted(self._orders.values(), key=lambda o: o.ordered_at)

    def for_sku(self, sku_code: str) -> list[Order]:
        return [
            order
            for order in self.all_orders()
            if any(line.sku_code == sku_code for line in order.lines)
        ]

    def __len__(self) -> int:
        return len(self._orders)


# --------------------------------------------------------------------------
# Returns and RTO
# --------------------------------------------------------------------------


class ReturnReason(StrEnum):
    SIZE_FIT = "SIZE_FIT"
    DAMAGED_IN_TRANSIT = "DAMAGED_IN_TRANSIT"
    NOT_AS_DESCRIBED = "NOT_AS_DESCRIBED"
    QUALITY = "QUALITY"
    CHANGED_MIND = "CHANGED_MIND"
    WRONG_ITEM = "WRONG_ITEM"
    UNDELIVERED_RTO = "UNDELIVERED_RTO"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ReturnEvent:
    """One return or RTO, with its actual cost."""

    marketplace: str
    marketplace_order_id: str
    sku_code: str
    quantity: int
    reason: ReturnReason
    is_rto: bool
    return_shipping_inr: Decimal = ZERO
    refund_inr: Decimal = ZERO
    resalable: bool = True
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    @property
    def cost_inr(self) -> Decimal:
        """Direct cost of this return, excluding the lost margin."""
        return money(self.return_shipping_inr + self.refund_inr)


@dataclass(frozen=True)
class ReturnRates:
    """Measured return behaviour for one SKU."""

    sku_code: str
    shipped_units: int
    returned_units: int
    rto_units: int
    return_rate_pct: Decimal
    rto_rate_pct: Decimal
    total_return_cost_inr: Decimal
    confidence: str
    sample_size: int

    @property
    def is_measured(self) -> bool:
        return self.confidence == "MEASURED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "shipped_units": self.shipped_units,
            "return_rate_pct": str(self.return_rate_pct),
            "rto_rate_pct": str(self.rto_rate_pct),
            "total_return_cost_inr": str(self.total_return_cost_inr),
            "confidence": self.confidence,
            "sample_size": self.sample_size,
        }


class ReturnsLedger:
    """Append-only returns, folded into measured rates."""

    #: Below this many shipments a rate is too noisy to act on. Killing a
    #: product on two returns out of three orders is how good SKUs die.
    MIN_SAMPLE_FOR_MEASURED = 20

    def __init__(self) -> None:
        self._events: list[ReturnEvent] = []
        self._seen: set[tuple[str, str, str]] = set()

    def record(self, event: ReturnEvent) -> ReturnEvent:
        signature = (event.marketplace, event.marketplace_order_id, event.sku_code)
        if signature in self._seen:
            logger.info(
                "duplicate return ignored", extra={"order_id": event.marketplace_order_id}
            )
            return event
        self._events.append(event)
        self._seen.add(signature)
        return event

    def events(self, sku_code: str | None = None) -> list[ReturnEvent]:
        if sku_code is None:
            return list(self._events)
        return [e for e in self._events if e.sku_code == sku_code]

    def rates_for(self, sku_code: str, *, shipped_units: int) -> ReturnRates:
        """Measured return and RTO rates, with an honest confidence label.

        These feed back into the economics engine, replacing the category
        benchmark Part 1 had to assume. That replacement is the point of the
        whole feedback loop -- but only once the sample supports it.
        """
        events = self.events(sku_code)
        returned = sum(e.quantity for e in events if not e.is_rto)
        rto = sum(e.quantity for e in events if e.is_rto)
        cost = money(sum((e.cost_inr for e in events), ZERO))

        if shipped_units <= 0:
            return ReturnRates(sku_code, 0, returned, rto, ZERO, ZERO, cost, "NO_DATA", 0)

        return_rate = money(Decimal(returned) / Decimal(shipped_units) * HUNDRED)
        rto_rate = money(Decimal(rto) / Decimal(shipped_units) * HUNDRED)
        confidence = (
            "MEASURED" if shipped_units >= self.MIN_SAMPLE_FOR_MEASURED else "PROVISIONAL"
        )
        return ReturnRates(
            sku_code=sku_code,
            shipped_units=shipped_units,
            returned_units=returned,
            rto_units=rto,
            return_rate_pct=return_rate,
            rto_rate_pct=rto_rate,
            total_return_cost_inr=cost,
            confidence=confidence,
            sample_size=shipped_units,
        )


# --------------------------------------------------------------------------
# Settlement reconciliation
# --------------------------------------------------------------------------


class MatchState(StrEnum):
    MATCHED = "MATCHED"
    PARTIAL = "PARTIAL"
    UNMATCHED = "UNMATCHED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


@dataclass(frozen=True)
class SettlementLine:
    """One line from a marketplace settlement report."""

    marketplace: str
    marketplace_order_id: str
    order_value_inr: Decimal
    commission_inr: Decimal = ZERO
    shipping_inr: Decimal = ZERO
    other_fees_inr: Decimal = ZERO
    tax_withheld_inr: Decimal = ZERO
    refund_inr: Decimal = ZERO
    adjustment_inr: Decimal = ZERO
    net_settled_inr: Decimal = ZERO
    settled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reference: str = ""

    @property
    def total_deductions_inr(self) -> Decimal:
        return money(
            self.commission_inr
            + self.shipping_inr
            + self.other_fees_inr
            + self.tax_withheld_inr
            + self.refund_inr
        )

    @property
    def expected_net_inr(self) -> Decimal:
        return money(
            self.order_value_inr - self.total_deductions_inr + self.adjustment_inr
        )


@dataclass(frozen=True)
class Reconciliation:
    """The result of matching one settlement line to an order."""

    marketplace_order_id: str
    state: MatchState
    order_gross_inr: Decimal
    settled_net_inr: Decimal
    expected_net_inr: Decimal
    variance_inr: Decimal
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "marketplace_order_id": self.marketplace_order_id,
            "state": str(self.state),
            "order_gross_inr": str(self.order_gross_inr),
            "settled_net_inr": str(self.settled_net_inr),
            "expected_net_inr": str(self.expected_net_inr),
            "variance_inr": str(self.variance_inr),
            "notes": list(self.notes),
        }


#: Rounding differences below this are not worth a human's time.
TOLERANCE_INR = Decimal("1.00")


def reconcile(
    line: SettlementLine, order: Order | None, *, tolerance_inr: Decimal = TOLERANCE_INR
) -> Reconciliation:
    """Match one settlement line against the order it claims to settle.

    Money that cannot be tied to an order is never absorbed. An unmatched
    settlement usually means an order we never imported, a marketplace
    adjustment we do not understand, or a genuine shortfall -- and all three
    matter more than the ones that reconcile cleanly.
    """
    notes: list[str] = []
    settled = money(line.net_settled_inr)
    expected = line.expected_net_inr
    variance = money(settled - expected)

    if order is None:
        return Reconciliation(
            marketplace_order_id=line.marketplace_order_id,
            state=MatchState.UNMATCHED,
            order_gross_inr=ZERO,
            settled_net_inr=settled,
            expected_net_inr=expected,
            variance_inr=variance,
            notes=[
                "no matching order: either it was never imported or the "
                "marketplace settled an identifier we do not hold"
            ],
        )

    gross = order.gross_inr
    if abs(money(line.order_value_inr - gross)) > tolerance_inr:
        notes.append(
            f"order value mismatch: settlement says {format_inr(line.order_value_inr)}, "
            f"our order totals {format_inr(gross)}"
        )

    if abs(variance) <= tolerance_inr and not notes:
        state = MatchState.MATCHED
    elif abs(variance) <= tolerance_inr:
        state = MatchState.PARTIAL
    elif settled == ZERO:
        state = MatchState.REQUIRES_REVIEW
        notes.append("settlement reports zero net; the order may be withheld or reversed")
    else:
        state = MatchState.PARTIAL
        notes.append(
            f"net variance of {format_inr(variance)} against the itemised deductions"
        )

    return Reconciliation(
        marketplace_order_id=line.marketplace_order_id,
        state=state,
        order_gross_inr=gross,
        settled_net_inr=settled,
        expected_net_inr=expected,
        variance_inr=variance,
        notes=notes,
    )


class SettlementLedger:
    """Idempotent settlement import plus reconciliation."""

    def __init__(self) -> None:
        self._lines: dict[tuple[str, str, str], SettlementLine] = {}
        self.duplicates_ignored = 0

    def ingest(self, line: SettlementLine) -> SettlementLine:
        key = (line.marketplace, line.marketplace_order_id, line.reference)
        if key in self._lines:
            self.duplicates_ignored += 1
            return self._lines[key]
        self._lines[key] = line
        return line

    def lines(self) -> list[SettlementLine]:
        return sorted(self._lines.values(), key=lambda line: line.settled_at)

    def reconcile_all(self, book: OrderBook) -> list[Reconciliation]:
        return [
            reconcile(line, book.get(line.marketplace, line.marketplace_order_id))
            for line in self.lines()
        ]

    def net_settled_for(self, marketplace: str, order_id: str) -> Decimal:
        return money(
            sum(
                (
                    line.net_settled_inr
                    for line in self.lines()
                    if line.marketplace == marketplace
                    and line.marketplace_order_id == order_id
                ),
                ZERO,
            )
        )


# --------------------------------------------------------------------------
# Realized economics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizedEconomics:
    """What a SKU actually earned, reconciled from underlying transactions."""

    sku_code: str
    orders: int
    units_shipped: int
    gross_revenue_inr: Decimal
    net_settled_inr: Decimal
    cogs_inr: Decimal
    return_cost_inr: Decimal
    actual_contribution_inr: Decimal
    actual_margin_pct: Decimal | None
    profit_per_order_inr: Decimal | None
    unreconciled_inr: Decimal
    coverage_note: str
    sample_size: int

    @property
    def is_evidenced(self) -> bool:
        """Whether the sample is large enough to act on."""
        return self.sample_size >= 20 and self.unreconciled_inr == ZERO

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "orders": self.orders,
            "units_shipped": self.units_shipped,
            "gross_revenue_inr": str(self.gross_revenue_inr),
            "net_settled_inr": str(self.net_settled_inr),
            "cogs_inr": str(self.cogs_inr),
            "return_cost_inr": str(self.return_cost_inr),
            "actual_contribution_inr": str(self.actual_contribution_inr),
            "actual_margin_pct": (
                None if self.actual_margin_pct is None else str(self.actual_margin_pct)
            ),
            "unreconciled_inr": str(self.unreconciled_inr),
            "coverage_note": self.coverage_note,
            "sample_size": self.sample_size,
        }


def compute_realized(
    sku_code: str,
    *,
    book: OrderBook,
    settlements: SettlementLedger,
    returns: ReturnsLedger,
    unit_landed_cost_inr: Decimal,
) -> RealizedEconomics:
    """Actual economics, built up from orders, settlements and returns.

    Revenue is what the marketplace *settled*, not what the order was worth.
    The gap between the two is fees, taxes and adjustments, and it is where
    predicted margin usually goes wrong.
    """
    orders = book.for_sku(sku_code)
    shipped = [
        o for o in orders if o.status not in (OrderStatus.NEW, OrderStatus.CANCELLED)
    ]
    units = sum(
        line.quantity
        for order in shipped
        for line in order.lines
        if line.sku_code == sku_code
    )

    gross = money(
        sum(
            (
                line.gross_inr
                for order in shipped
                for line in order.lines
                if line.sku_code == sku_code
            ),
            ZERO,
        )
    )
    settled = money(
        sum(
            (
                settlements.net_settled_for(o.marketplace, o.marketplace_order_id)
                for o in shipped
            ),
            ZERO,
        )
    )
    unsettled_orders = [
        o
        for o in shipped
        if settlements.net_settled_for(o.marketplace, o.marketplace_order_id) == ZERO
    ]
    unreconciled = money(sum((o.gross_inr for o in unsettled_orders), ZERO))

    cogs = money(money(unit_landed_cost_inr) * units)
    return_cost = money(sum((e.cost_inr for e in returns.events(sku_code)), ZERO))
    contribution = money(settled - cogs - return_cost)

    margin_ratio = safe_div(contribution, gross)
    margin = money(margin_ratio * HUNDRED) if margin_ratio is not None else None
    per_order = safe_div(contribution, Decimal(len(shipped))) if shipped else None

    if unreconciled > ZERO:
        note = (
            f"{len(unsettled_orders)} order(s) worth {format_inr(unreconciled)} are not "
            "yet settled; contribution understates until they land"
        )
    elif len(shipped) < 20:
        note = f"only {len(shipped)} settled order(s): treat this as provisional"
    else:
        note = "fully reconciled against settlements"

    return RealizedEconomics(
        sku_code=sku_code,
        orders=len(shipped),
        units_shipped=units,
        gross_revenue_inr=gross,
        net_settled_inr=settled,
        cogs_inr=cogs,
        return_cost_inr=return_cost,
        actual_contribution_inr=contribution,
        actual_margin_pct=margin,
        profit_per_order_inr=per_order,
        unreconciled_inr=unreconciled,
        coverage_note=note,
        sample_size=len(shipped),
    )


@dataclass(frozen=True)
class PredictionError:
    """How far Part 1's prediction was from what happened."""

    sku_code: str
    predicted_margin_pct: Decimal
    actual_margin_pct: Decimal | None
    margin_error_points: Decimal | None
    predicted_profit_per_order_inr: Decimal
    actual_profit_per_order_inr: Decimal | None
    direction: str
    sample_size: int
    actionable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "predicted_margin_pct": str(self.predicted_margin_pct),
            "actual_margin_pct": (
                None if self.actual_margin_pct is None else str(self.actual_margin_pct)
            ),
            "margin_error_points": (
                None
                if self.margin_error_points is None
                else str(self.margin_error_points)
            ),
            "direction": self.direction,
            "sample_size": self.sample_size,
            "actionable": self.actionable,
        }


def compare_prediction(
    realized: RealizedEconomics,
    *,
    predicted_margin_pct: Decimal,
    predicted_profit_per_order_inr: Decimal,
    min_sample: int = 20,
) -> PredictionError:
    """Measure prediction error, and say whether it can be acted on.

    Prediction error is the most valuable learning signal the system produces:
    it says whether the assumptions in Part 1 were optimistic, and by how much.
    A small sample gives a number that is real but not yet actionable.
    """
    actual = realized.actual_margin_pct
    error = None if actual is None else money(actual - money(predicted_margin_pct))

    if error is None:
        direction = "NO_DATA"
    elif error < Decimal("-2"):
        direction = "OPTIMISTIC"
    elif error > Decimal("2"):
        direction = "PESSIMISTIC"
    else:
        direction = "ACCURATE"

    return PredictionError(
        sku_code=realized.sku_code,
        predicted_margin_pct=money(predicted_margin_pct),
        actual_margin_pct=actual,
        margin_error_points=error,
        predicted_profit_per_order_inr=money(predicted_profit_per_order_inr),
        actual_profit_per_order_inr=realized.profit_per_order_inr,
        direction=direction,
        sample_size=realized.sample_size,
        actionable=realized.sample_size >= min_sample
        and realized.unreconciled_inr == ZERO,
    )


__all__ = [
    "ORDERS_VERSION",
    "STATUS_MAP",
    "TOLERANCE_INR",
    "MatchState",
    "Order",
    "OrderBook",
    "OrderLine",
    "OrderStatus",
    "PredictionError",
    "RealizedEconomics",
    "Reconciliation",
    "ReturnEvent",
    "ReturnRates",
    "ReturnReason",
    "ReturnsLedger",
    "SettlementLedger",
    "SettlementLine",
    "UnknownStatus",
    "compare_prediction",
    "compute_realized",
    "normalise_status",
    "reconcile",
]
