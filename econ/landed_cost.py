"""Landed cost: what a unit actually costs us, delivered and ready to ship.

``supplier_cost == landed_cost`` is the single most common way an Indian
marketplace seller talks themselves into a losing SKU. This module refuses
that shortcut: every component is computed, named and recorded, and every
assumption is carried alongside the number it produced with a confidence
label.

The component that most often decides viability is ``input_tax_credit_inr``.
It is zero whenever the supplier issues no tax invoice, which means the GST on
the purchase is unrecoverable and the buyer pays tax twice -- once on the
purchase and again on the sale.

The component most often forgotten is ``moq_carry_cost_inr``. A minimum order
quantity of 50 against a realistic velocity of 8 a month is six months of
locked capital plus obsolescence exposure. That is a cost, not a supplier
attribute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from catalog.enums import ScenarioName
from catalog.schemas import SupplierOffer
from econ.money import HUNDRED, ZERO, apply_pct, input_tax_credit, money, pct, rupees

logger = logging.getLogger(__name__)

#: Bumped on any change to the formulas below. Snapshots pin it so a
#: historical calculation can be re-derived under the code that produced it.
CALC_VERSION = "landed_cost/1.0.0"


class AssumptionConfidence(StrEnum):
    """How much weight an assumption can bear."""

    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    GUESSED = "GUESSED"


@dataclass(frozen=True)
class Assumption:
    """A value the calculation needed but could not observe.

    Recorded rather than hidden. A guessed input you can see beats an omitted
    input you cannot, and the confidence label tells a reader which numbers to
    challenge first.
    """

    name: str
    value: str
    confidence: AssumptionConfidence
    rationale: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": str(self.confidence),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CarryingCostInputs:
    """Operator policy for holding stock."""

    monthly_capital_cost_pct: Decimal = Decimal("1.5")
    dead_stock_provision_pct: Decimal = Decimal("3")
    inspection_qc_per_unit_inr: Decimal = Decimal("2")
    packaging_per_unit_inr: Decimal = Decimal("8")
    inbound_defect_rate_pct: Decimal = Decimal("2")

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> CarryingCostInputs:
        payload = payload or {}
        return cls(
            monthly_capital_cost_pct=pct(payload.get("monthly_capital_cost_pct", "1.5")),
            dead_stock_provision_pct=pct(payload.get("dead_stock_provision_pct", "3")),
            inspection_qc_per_unit_inr=money(
                payload.get("inspection_qc_per_unit_inr", "2")
            ),
            packaging_per_unit_inr=money(payload.get("packaging_per_unit_inr", "8")),
            inbound_defect_rate_pct=pct(payload.get("inbound_defect_rate_pct", "2")),
        )


@dataclass(frozen=True)
class LandedCostResult:
    """A fully itemised landed cost with its assumptions."""

    scenario: ScenarioName
    calc_version: str
    supplier_offer_id: Any
    unit_cost_ex_gst_inr: Decimal
    input_tax_credit_inr: Decimal
    effective_cogs_inr: Decimal
    inbound_freight_per_unit_inr: Decimal
    moq_carry_cost_inr: Decimal
    inspection_qc_cost_inr: Decimal
    packaging_cost_inr: Decimal
    inbound_defect_provision_inr: Decimal
    total_landed_cost_inr: Decimal
    assumptions: list[Assumption] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": str(self.scenario),
            "calc_version": self.calc_version,
            "unit_cost_ex_gst_inr": str(rupees(self.unit_cost_ex_gst_inr)),
            "input_tax_credit_inr": str(rupees(self.input_tax_credit_inr)),
            "effective_cogs_inr": str(rupees(self.effective_cogs_inr)),
            "inbound_freight_per_unit_inr": str(
                rupees(self.inbound_freight_per_unit_inr)
            ),
            "moq_carry_cost_inr": str(rupees(self.moq_carry_cost_inr)),
            "inspection_qc_cost_inr": str(rupees(self.inspection_qc_cost_inr)),
            "packaging_cost_inr": str(rupees(self.packaging_cost_inr)),
            "inbound_defect_provision_inr": str(
                rupees(self.inbound_defect_provision_inr)
            ),
            "total_landed_cost_inr": str(rupees(self.total_landed_cost_inr)),
            "assumptions": [a.as_dict() for a in self.assumptions],
            "warnings": list(self.warnings),
        }


def moq_carrying_cost(
    *,
    unit_cost_inr: Decimal,
    moq_units: int,
    expected_monthly_velocity: Decimal,
    policy: CarryingCostInputs,
) -> tuple[Decimal, list[Assumption]]:
    """Cost of holding a minimum order quantity until it sells through.

    Average holding period is half the sell-through time, since stock depletes
    steadily rather than sitting until the last unit goes. Dead-stock
    provision is charged on top, because the risk that some of the batch never
    sells is not the same risk as the capital being tied up.
    """
    unit_cost = money(unit_cost_inr)
    velocity = money(expected_monthly_velocity)
    assumptions = [
        Assumption(
            name="expected_monthly_velocity",
            value=str(velocity),
            confidence=AssumptionConfidence.GUESSED,
            rationale=(
                "no sales history exists in Part 1; replace with measured "
                "velocity once the SKU has traded"
            ),
        )
    ]

    if velocity <= ZERO:
        # An unsellable SKU cannot be amortised. Charge the whole batch rather
        # than dividing by zero and reporting no carrying cost at all.
        full_batch = money(unit_cost * Decimal(moq_units))
        assumptions.append(
            Assumption(
                name="velocity_zero_fallback",
                value=str(rupees(full_batch)),
                confidence=AssumptionConfidence.GUESSED,
                rationale="velocity of zero: entire MOQ treated as dead stock",
            )
        )
        return full_batch, assumptions

    months_to_clear = money(Decimal(moq_units) / velocity)
    average_months = money(months_to_clear / 2)
    capital = money(
        unit_cost * average_months * policy.monthly_capital_cost_pct / HUNDRED
    )
    dead_stock = apply_pct(unit_cost, policy.dead_stock_provision_pct)

    assumptions.append(
        Assumption(
            name="average_holding_months",
            value=str(average_months),
            confidence=AssumptionConfidence.ESTIMATED,
            rationale=f"MOQ {moq_units} / velocity {velocity} per month, halved",
        )
    )
    return money(capital + dead_stock), assumptions


def compute_landed_cost(
    offer: SupplierOffer,
    *,
    scenario: ScenarioName = ScenarioName.EXPECTED,
    supplier_cost_delta_pct: Decimal = ZERO,
    expected_monthly_velocity: Decimal = Decimal("10"),
    policy: CarryingCostInputs | None = None,
) -> LandedCostResult:
    """Full landed cost for one supplier offer under one scenario."""
    policy = policy or CarryingCostInputs()
    warnings: list[str] = []

    adjusted_cost = money(
        offer.unit_cost_inr * (HUNDRED + pct(supplier_cost_delta_pct)) / HUNDRED
    )

    itc = input_tax_credit(
        adjusted_cost,
        gst_rate_pct=offer.gst_rate_pct,
        cost_is_gst_inclusive=offer.cost_is_gst_inclusive,
        tax_invoice_available=offer.tax_invoice_available,
    )
    if not offer.tax_invoice_available:
        warnings.append("no tax invoice: input tax credit is zero and GST is paid twice")

    unit_ex_gst = (
        money(adjusted_cost - itc) if offer.cost_is_gst_inclusive else adjusted_cost
    )
    effective_cogs = money(
        unit_ex_gst if offer.cost_is_gst_inclusive else adjusted_cost - itc
    )

    if offer.inbound_freight_inr is None:
        raise ValueError(
            "cannot compute landed cost: the catalogue did not state inbound "
            "freight. Treating it as zero understates landed cost, and a "
            "confirmed zero is a different claim from an unknown. Record it on "
            "the supplier offer."
        )
    freight = money(offer.inbound_freight_inr)
    carry, assumptions = moq_carrying_cost(
        unit_cost_inr=effective_cogs,
        moq_units=offer.moq_units,
        expected_monthly_velocity=money(expected_monthly_velocity),
        policy=policy,
    )
    if offer.moq_units > 1 and carry > effective_cogs / 5:
        warnings.append(
            f"MOQ {offer.moq_units} adds {rupees(carry)} per unit in carrying cost "
            "against the assumed velocity"
        )

    defect = apply_pct(effective_cogs, policy.inbound_defect_rate_pct)
    total = money(
        effective_cogs
        + freight
        + carry
        + policy.inspection_qc_per_unit_inr
        + policy.packaging_per_unit_inr
        + defect
    )

    assumptions.extend(
        [
            Assumption(
                name="monthly_capital_cost_pct",
                value=str(policy.monthly_capital_cost_pct),
                confidence=AssumptionConfidence.ESTIMATED,
                rationale="operator policy for the cost of tied-up working capital",
            ),
            Assumption(
                name="inbound_defect_rate_pct",
                value=str(policy.inbound_defect_rate_pct),
                confidence=AssumptionConfidence.ESTIMATED,
                rationale="provision for units unusable on arrival",
            ),
        ]
    )

    return LandedCostResult(
        scenario=scenario,
        calc_version=CALC_VERSION,
        supplier_offer_id=offer.id,
        unit_cost_ex_gst_inr=unit_ex_gst,
        input_tax_credit_inr=itc,
        effective_cogs_inr=effective_cogs,
        inbound_freight_per_unit_inr=freight,
        moq_carry_cost_inr=carry,
        inspection_qc_cost_inr=policy.inspection_qc_per_unit_inr,
        packaging_cost_inr=policy.packaging_per_unit_inr,
        inbound_defect_provision_inr=defect,
        total_landed_cost_inr=total,
        assumptions=assumptions,
        warnings=warnings,
    )


__all__ = [
    "CALC_VERSION",
    "Assumption",
    "AssumptionConfidence",
    "CarryingCostInputs",
    "LandedCostResult",
    "compute_landed_cost",
    "moq_carrying_cost",
]
