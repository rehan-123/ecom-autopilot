"""Inventory: movements are facts, positions are derived.

Stock levels are never stored as a mutable number. Every change is an
append-only movement, and ``on_hand``, ``reserved`` and ``available`` are folded
from the movement stream. That is the same discipline Part 1 applies to prices,
for the same reason: when a count is wrong you need to see how it got there,
and a single mutable integer destroys that evidence.

Two invariants hold structurally rather than by convention:

**Stock cannot go negative.** A movement that would take on-hand below zero is
refused at the point of recording, not detected later by a reconciliation job.

**You cannot reserve what you do not have.** Reservations draw from available
stock, and available is on-hand minus reserved. Overselling is the single most
expensive marketplace mistake — it costs the order, the metric and sometimes
the account — so it is prevented rather than reported.

The reorder policy is deliberately conservative for new products. A high
expected demand is a forecast, and a forecast is not evidence; the initial
exposure is sized so that being wrong costs a known, small amount.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from econ.money import ZERO, format_inr, money

logger = logging.getLogger(__name__)

INVENTORY_VERSION = "inventory/1.0.0"


class MovementType(StrEnum):
    """Every way stock can move. Each is a fact, never an adjustment."""

    RECEIPT = "RECEIPT"
    RESERVE = "RESERVE"
    RELEASE = "RELEASE"
    SHIP = "SHIP"
    RETURN_RESALABLE = "RETURN_RESALABLE"
    RETURN_DAMAGED = "RETURN_DAMAGED"
    RTO_RECEIPT = "RTO_RECEIPT"
    WRITE_OFF = "WRITE_OFF"
    IN_TRANSIT = "IN_TRANSIT"
    TRANSIT_ARRIVED = "TRANSIT_ARRIVED"


class InventoryError(RuntimeError):
    """Raised when a movement would violate an inventory invariant."""


@dataclass(frozen=True)
class Movement:
    """One append-only stock fact."""

    sku_code: str
    movement_type: MovementType
    quantity: int
    reference: str
    unit_cost_inr: Decimal = ZERO
    marketplace: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    note: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise InventoryError(
                "movement quantity must be positive; direction is carried by "
                "movement_type, not by the sign"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "sku_code": self.sku_code,
            "type": str(self.movement_type),
            "quantity": self.quantity,
            "reference": self.reference,
            "marketplace": self.marketplace,
            "at": self.at.isoformat(),
        }


@dataclass(frozen=True)
class Position:
    """A stock position, folded from movements."""

    sku_code: str
    on_hand: int
    reserved: int
    in_transit: int
    damaged: int
    sold_units: int
    returned_units: int
    rto_units: int
    capital_locked_inr: Decimal

    @property
    def available(self) -> int:
        """What can still be sold. Never negative by construction."""
        return max(0, self.on_hand - self.reserved)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "on_hand": self.on_hand,
            "reserved": self.reserved,
            "available": self.available,
            "in_transit": self.in_transit,
            "damaged": self.damaged,
            "sold_units": self.sold_units,
            "returned_units": self.returned_units,
            "rto_units": self.rto_units,
            "capital_locked_inr": str(self.capital_locked_inr),
        }


#: How each movement affects the fold. (on_hand, reserved, in_transit, damaged)
_EFFECTS: dict[MovementType, tuple[int, int, int, int]] = {
    MovementType.RECEIPT: (+1, 0, 0, 0),
    MovementType.RESERVE: (0, +1, 0, 0),
    MovementType.RELEASE: (0, -1, 0, 0),
    MovementType.SHIP: (-1, -1, 0, 0),
    MovementType.RETURN_RESALABLE: (+1, 0, 0, 0),
    MovementType.RETURN_DAMAGED: (0, 0, 0, +1),
    MovementType.RTO_RECEIPT: (+1, 0, 0, 0),
    MovementType.WRITE_OFF: (-1, 0, 0, 0),
    MovementType.IN_TRANSIT: (0, 0, +1, 0),
    MovementType.TRANSIT_ARRIVED: (+1, 0, -1, 0),
}


class InventoryLedger:
    """Append-only movements, with positions folded on read."""

    def __init__(self) -> None:
        self._movements: list[Movement] = []
        self._seen_references: set[tuple[str, str, MovementType]] = set()

    def record(self, movement: Movement, *, idempotent: bool = True) -> Movement:
        """Append a movement after checking it cannot break an invariant.

        Duplicate references are ignored rather than applied twice: order
        webhooks and settlement files both arrive more than once, and an
        inventory system that double-counts them is worse than useless.
        """
        signature = (movement.sku_code, movement.reference, movement.movement_type)
        if idempotent and signature in self._seen_references:
            logger.info(
                "duplicate movement ignored",
                extra={"sku": movement.sku_code, "reference": movement.reference},
            )
            return movement

        self._validate(movement)
        self._movements.append(movement)
        self._seen_references.add(signature)
        return movement

    def _validate(self, movement: Movement) -> None:
        position = self.position(movement.sku_code)
        on_hand, reserved, *_ = _EFFECTS[movement.movement_type]

        if on_hand < 0 and position.on_hand < movement.quantity:
            raise InventoryError(
                f"{movement.movement_type} of {movement.quantity} would take "
                f"{movement.sku_code} below zero (on hand {position.on_hand})"
            )
        if movement.movement_type is MovementType.RESERVE and (
            position.available < movement.quantity
        ):
            raise InventoryError(
                f"cannot reserve {movement.quantity} of {movement.sku_code}: "
                f"only {position.available} available. Overselling costs the "
                "order and the account metric."
            )
        if reserved < 0 and position.reserved < movement.quantity:
            raise InventoryError(
                f"cannot release/ship {movement.quantity} of {movement.sku_code}: "
                f"only {position.reserved} reserved"
            )
        if movement.movement_type is MovementType.TRANSIT_ARRIVED and (
            position.in_transit < movement.quantity
        ):
            raise InventoryError(
                f"cannot receive {movement.quantity} in transit for "
                f"{movement.sku_code}: only {position.in_transit} outstanding"
            )

    def movements(self, sku_code: str | None = None) -> list[Movement]:
        if sku_code is None:
            return list(self._movements)
        return [m for m in self._movements if m.sku_code == sku_code]

    def position(self, sku_code: str) -> Position:
        """Fold the movement stream into a current position."""
        on_hand = reserved = in_transit = damaged = 0
        sold = returned = rto = 0
        capital = ZERO

        for movement in self._movements:
            if movement.sku_code != sku_code:
                continue
            d_hand, d_res, d_transit, d_damaged = _EFFECTS[movement.movement_type]
            on_hand += d_hand * movement.quantity
            reserved += d_res * movement.quantity
            in_transit += d_transit * movement.quantity
            damaged += d_damaged * movement.quantity

            if movement.movement_type is MovementType.SHIP:
                sold += movement.quantity
            elif movement.movement_type in (
                MovementType.RETURN_RESALABLE,
                MovementType.RETURN_DAMAGED,
            ):
                returned += movement.quantity
            elif movement.movement_type is MovementType.RTO_RECEIPT:
                rto += movement.quantity

            if movement.movement_type in (MovementType.RECEIPT, MovementType.IN_TRANSIT):
                capital += money(movement.unit_cost_inr) * movement.quantity
            elif movement.movement_type is MovementType.SHIP:
                capital -= money(movement.unit_cost_inr) * movement.quantity

        return Position(
            sku_code=sku_code,
            on_hand=max(0, on_hand),
            reserved=max(0, reserved),
            in_transit=max(0, in_transit),
            damaged=max(0, damaged),
            sold_units=sold,
            returned_units=returned,
            rto_units=rto,
            capital_locked_inr=money(max(ZERO, capital)),
        )

    def skus(self) -> list[str]:
        return sorted({m.sku_code for m in self._movements})


# --------------------------------------------------------------------------
# Reorder policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReorderPolicy:
    """Operator policy for replenishment and initial exposure."""

    lead_time_days: int = 7
    service_factor: Decimal = Decimal("1.28")  # roughly 90% service level
    max_days_of_cover: int = 60
    max_capital_per_product_inr: Decimal = Decimal("15000")
    #: Worst-case planning target for a first test of a new product.
    initial_test_orders: int = 12
    #: Evidence required before a product is treated as validated.
    min_orders_for_measured_velocity: int = 20


@dataclass(frozen=True)
class ReorderAdvice:
    """A replenishment recommendation, with its reasoning attached."""

    sku_code: str
    recommended_qty: int
    reorder_point: int
    safety_stock: int
    days_of_cover: Decimal | None
    velocity_per_day: Decimal | None
    velocity_confidence: str
    capital_required_inr: Decimal
    should_reorder: bool
    reasons: list[str] = field(default_factory=list)
    constraints_applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "recommended_qty": self.recommended_qty,
            "reorder_point": self.reorder_point,
            "safety_stock": self.safety_stock,
            "days_of_cover": None
            if self.days_of_cover is None
            else str(self.days_of_cover),
            "velocity_per_day": (
                None if self.velocity_per_day is None else str(self.velocity_per_day)
            ),
            "velocity_confidence": self.velocity_confidence,
            "capital_required_inr": str(self.capital_required_inr),
            "should_reorder": self.should_reorder,
            "reasons": list(self.reasons),
            "constraints_applied": list(self.constraints_applied),
        }


def measure_velocity(
    ledger: InventoryLedger,
    sku_code: str,
    *,
    window_days: int = 30,
    now: datetime | None = None,
) -> tuple[Decimal | None, str, int]:
    """Units shipped per day, and how much to trust it.

    Returns ``(velocity, confidence, observations)``. Confidence is
    ``MEASURED`` only once enough orders exist; below that the number is real
    but too noisy to size a purchase against, and saying so is the difference
    between a small test and a warehouse full of stock.
    """
    now = now or datetime.now(UTC)
    shipments = [
        m
        for m in ledger.movements(sku_code)
        if m.movement_type is MovementType.SHIP and (now - m.at).days <= window_days
    ]
    units = sum(m.quantity for m in shipments)
    if not shipments:
        return None, "NO_DATA", 0

    first = min(m.at for m in shipments)
    elapsed_days = max(1, (now - first).days)
    velocity = money(Decimal(units) / Decimal(elapsed_days))
    confidence = "MEASURED" if len(shipments) >= 20 else "PROVISIONAL"
    return velocity, confidence, len(shipments)


def plan_reorder(
    ledger: InventoryLedger,
    sku_code: str,
    *,
    unit_landed_cost_inr: Decimal,
    supplier_moq: int,
    policy: ReorderPolicy | None = None,
    now: datetime | None = None,
) -> ReorderAdvice:
    """Decide whether and how much to reorder.

    Deliberately conservative. A new product with no shipment history gets an
    initial test quantity sized to the policy target, not to an optimistic
    forecast, and every cap that bit is reported so the recommendation can be
    argued with.
    """
    policy = policy or ReorderPolicy()
    position = ledger.position(sku_code)
    velocity, confidence, observations = measure_velocity(ledger, sku_code, now=now)
    unit_cost = money(unit_landed_cost_inr)
    reasons: list[str] = []
    constraints: list[str] = []

    if velocity is None or velocity <= ZERO:
        # No shipment history: this is a first test, not a replenishment.
        target = max(policy.initial_test_orders, supplier_moq)
        if target > policy.initial_test_orders:
            constraints.append(
                f"supplier MOQ of {supplier_moq} exceeds the {policy.initial_test_orders}-unit "
                "test target, so the test is larger than planned"
            )
        reasons.append(
            f"no shipment history: sizing an initial test of {policy.initial_test_orders} "
            "orders rather than forecasting demand"
        )
        capital = money(unit_cost * target)
        if capital > policy.max_capital_per_product_inr:
            affordable = (
                int(policy.max_capital_per_product_inr / unit_cost) if unit_cost else 0
            )
            constraints.append(
                f"capital cap {format_inr(policy.max_capital_per_product_inr)} allows only "
                f"{affordable} units at {format_inr(unit_cost)} each"
            )
            target = max(0, min(target, affordable))
            capital = money(unit_cost * target)
            if target < supplier_moq:
                reasons.append(
                    f"MOQ {supplier_moq} cannot be met inside the capital cap; "
                    "renegotiate MOQ or raise the cap before ordering"
                )
        return ReorderAdvice(
            sku_code=sku_code,
            recommended_qty=target,
            reorder_point=0,
            safety_stock=0,
            days_of_cover=None,
            velocity_per_day=None,
            velocity_confidence=confidence,
            capital_required_inr=capital,
            should_reorder=position.available == 0 and target > 0,
            reasons=reasons,
            constraints_applied=constraints,
        )

    lead_demand = money(velocity * policy.lead_time_days)
    safety = money(
        policy.service_factor * velocity * Decimal(policy.lead_time_days).sqrt()
    )
    reorder_point = int(lead_demand + safety)
    days_of_cover = money(Decimal(position.available) / velocity) if velocity else None

    if confidence != "MEASURED":
        constraints.append(
            f"velocity is PROVISIONAL from {observations} shipments; "
            f"{policy.min_orders_for_measured_velocity} needed before scaling on it"
        )

    should = position.available <= reorder_point
    target_units = int(money(velocity * policy.max_days_of_cover))
    quantity = max(0, target_units - position.available - position.in_transit)
    quantity = max(quantity, supplier_moq) if quantity > 0 else 0

    if quantity and confidence != "MEASURED":
        capped = max(supplier_moq, int(velocity * policy.lead_time_days * 2))
        if capped < quantity:
            constraints.append(
                "velocity not yet MEASURED: order capped at twice lead-time demand"
            )
            quantity = capped

    capital = money(unit_cost * quantity)
    if capital > policy.max_capital_per_product_inr and unit_cost > ZERO:
        affordable = int(policy.max_capital_per_product_inr / unit_cost)
        constraints.append(
            f"capital cap {format_inr(policy.max_capital_per_product_inr)} limits this to "
            f"{affordable} units"
        )
        quantity = min(quantity, affordable)
        capital = money(unit_cost * quantity)

    if should:
        reasons.append(
            f"available {position.available} is at or below the reorder point "
            f"{reorder_point} ({velocity}/day over {policy.lead_time_days}-day lead time)"
        )
    else:
        reasons.append(
            f"available {position.available} is above the reorder point {reorder_point}"
        )

    return ReorderAdvice(
        sku_code=sku_code,
        recommended_qty=quantity,
        reorder_point=reorder_point,
        safety_stock=int(safety),
        days_of_cover=days_of_cover,
        velocity_per_day=velocity,
        velocity_confidence=confidence,
        capital_required_inr=capital,
        should_reorder=should and quantity > 0,
        reasons=reasons,
        constraints_applied=constraints,
    )


def low_stock_report(ledger: InventoryLedger, advice: dict[str, ReorderAdvice]) -> str:
    """Products at or below their reorder point."""
    rows = [(sku, a) for sku, a in sorted(advice.items()) if a.should_reorder]
    if not rows:
        return "No products are below their reorder point.\n"
    lines = [
        f"LOW STOCK - {len(rows)} product(s)",
        "",
        f"{'SKU':<20} {'avail':>6} {'point':>6} {'order':>6} {'capital':>12} {'velocity'}",
        f"{'-' * 20} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 12} {'-' * 18}",
    ]
    for sku, item in rows:
        position = ledger.position(sku)
        velocity = (
            f"{item.velocity_per_day}/d {item.velocity_confidence}"
            if item.velocity_per_day
            else item.velocity_confidence
        )
        lines.append(
            f"{sku[:19]:<20} {position.available:>6} {item.reorder_point:>6} "
            f"{item.recommended_qty:>6} {format_inr(item.capital_required_inr):>12} {velocity}"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "INVENTORY_VERSION",
    "InventoryError",
    "InventoryLedger",
    "Movement",
    "MovementType",
    "Position",
    "ReorderAdvice",
    "ReorderPolicy",
    "low_stock_report",
    "measure_velocity",
    "plan_reorder",
]
