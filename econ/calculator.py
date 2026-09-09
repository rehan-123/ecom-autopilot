"""The economics calculator and the profitability gate.

Everything here is deterministic. The same inputs and the same pinned
reference versions always produce the same output, which is what makes a
decision defensible six months later when it has to be explained.

Two design choices are worth stating outright:

**Margin is per shipped order, not per delivered order.** Return logistics and
write-downs are incurred whether or not the sale sticks, so amortising them
across shipped orders is the honest denominator. Reporting per delivered order
flatters every SKU by its return rate.

**The gate considers the downside.** A SKU whose expected case clears the floor
but whose worst case is negative will lose money in the month the supplier
raises prices, and that month arrives. The asymmetry is real -- a bad SKU costs
inventory, capital, shelf-time and account metrics, while a good one only makes
money -- so the policy can require the worst case to hold up, and by default it
does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from catalog.enums import GateResult, ScenarioName
from econ.landed_cost import Assumption, LandedCostResult
from econ.money import (
    HUNDRED,
    ZERO,
    EconomicsDataUnavailable,
    FeeBreakdown,
    ReturnCost,
    apply_pct,
    compute_fees,
    compute_return_cost,
    format_inr,
    money,
    pct,
    rupees,
    safe_div,
    split_inclusive,
)
from reference.loader import PinnedReferences

logger = logging.getLogger(__name__)

CALC_VERSION = "economics/1.0.0"


@dataclass(frozen=True)
class ScenarioDeltas:
    """How one scenario differs from the expected case."""

    scenario: ScenarioName
    return_rate_multiplier: Decimal
    supplier_cost_delta_pct: Decimal
    shipping_delta_pct: Decimal
    ad_cost_multiplier: Decimal
    velocity_multiplier: Decimal

    @classmethod
    def from_payload(
        cls, scenario: ScenarioName, payload: dict[str, Any]
    ) -> ScenarioDeltas:
        return cls(
            scenario=scenario,
            return_rate_multiplier=money(payload["return_rate_multiplier"]),
            supplier_cost_delta_pct=pct(payload["supplier_cost_delta_pct"]),
            shipping_delta_pct=pct(payload["shipping_delta_pct"]),
            ad_cost_multiplier=money(payload["ad_cost_multiplier"]),
            velocity_multiplier=money(payload["velocity_multiplier"]),
        )

    def describe(self) -> dict[str, str]:
        """What differs, for display beside the numbers."""
        return {
            "return_rate": f"x{self.return_rate_multiplier}",
            "supplier_cost": f"{self.supplier_cost_delta_pct:+}%",
            "shipping": f"{self.shipping_delta_pct:+}%",
            "advertising": f"x{self.ad_cost_multiplier}",
            "velocity": f"x{self.velocity_multiplier}",
        }


@dataclass(frozen=True)
class EconomicsInputs:
    """Everything the calculator needs that is not reference data."""

    selling_price_inr: Decimal
    category: str
    hsn_code: str
    weight_g: int
    ad_allowance_pct: Decimal = Decimal("10")
    other_costs_inr: Decimal = ZERO
    registered_for_gst: bool = True


@dataclass(frozen=True)
class EconomicsResult:
    """One scenario's economics, fully itemised and reproducible."""

    scenario: ScenarioName
    calc_version: str
    pinned_versions: dict[str, str]
    selling_price_inr: Decimal
    gst_component_inr: Decimal
    net_taxable_revenue_inr: Decimal
    landed_cost_inr: Decimal
    fees: FeeBreakdown
    returns: ReturnCost
    ad_cost_inr: Decimal
    other_costs_inr: Decimal
    gross_profit_inr: Decimal
    contribution_profit_inr: Decimal
    net_profit_inr: Decimal
    net_margin_pct: Decimal
    roi_pct: Decimal
    breakeven_price_inr: Decimal
    min_profitable_price_inr: Decimal
    max_supplier_cost_inr: Decimal
    max_ad_cost_inr: Decimal
    max_return_rate_pct: Decimal
    assumptions: list[Assumption] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": str(self.scenario),
            "calc_version": self.calc_version,
            "pinned_versions": self.pinned_versions,
            "selling_price_inr": str(rupees(self.selling_price_inr)),
            "gst_component_inr": str(rupees(self.gst_component_inr)),
            "net_taxable_revenue_inr": str(rupees(self.net_taxable_revenue_inr)),
            "landed_cost_inr": str(rupees(self.landed_cost_inr)),
            "fees": self.fees.as_dict(),
            "returns": self.returns.as_dict(),
            "ad_cost_inr": str(rupees(self.ad_cost_inr)),
            "gross_profit_inr": str(rupees(self.gross_profit_inr)),
            "contribution_profit_inr": str(rupees(self.contribution_profit_inr)),
            "net_profit_inr": str(rupees(self.net_profit_inr)),
            "net_margin_pct": str(self.net_margin_pct),
            "roi_pct": str(self.roi_pct),
            "breakeven_price_inr": str(rupees(self.breakeven_price_inr)),
            "min_profitable_price_inr": str(rupees(self.min_profitable_price_inr)),
            "max_supplier_cost_inr": str(rupees(self.max_supplier_cost_inr)),
            "max_ad_cost_inr": str(rupees(self.max_ad_cost_inr)),
            "max_return_rate_pct": str(self.max_return_rate_pct),
            "warnings": list(self.warnings),
        }


def _gst_rate_for(hsn_code: str, tax_payload: dict[str, Any]) -> Decimal:
    by_hsn = tax_payload.get("gst_rate_pct_by_hsn", {}) or {}
    for length in (8, 6, 4, 2):
        prefix = hsn_code[:length]
        if prefix in by_hsn:
            return pct(by_hsn[prefix])
    if tax_payload.get("default_gst_rate_pct") is None:
        raise EconomicsDataUnavailable(
            "the tax table defines no default GST rate and none matches this "
            "HSN code. A 0% rate would understate output tax and overstate "
            "margin, so economics are unavailable rather than misleading."
        )
    return pct(tax_payload["default_gst_rate_pct"])


def compute_economics(
    inputs: EconomicsInputs,
    landed: LandedCostResult,
    references: PinnedReferences,
    deltas: ScenarioDeltas,
) -> EconomicsResult:
    """Full unit economics for one scenario."""
    price = money(inputs.selling_price_inr)
    warnings = list(landed.warnings)

    gst_rate = _gst_rate_for(inputs.hsn_code, references.tax_table.payload)
    if gst_rate == ZERO:
        warnings.append(
            f"GST rate for HSN {inputs.hsn_code} resolved to 0%; "
            "verify the tax table before relying on this"
        )
    revenue = split_inclusive(price, gst_rate)

    fees = compute_fees(
        selling_price_inr=price,
        category=inputs.category,
        weight_g=inputs.weight_g,
        rate_card=references.rate_card.payload,
        registered_for_gst=inputs.registered_for_gst,
    )
    shipping_adjusted = money(
        fees.shipping_inr * (HUNDRED + deltas.shipping_delta_pct) / HUNDRED
    )
    fees = FeeBreakdown(
        commission_inr=fees.commission_inr,
        fixed_fee_inr=fees.fixed_fee_inr,
        collection_fee_inr=fees.collection_fee_inr,
        shipping_inr=shipping_adjusted,
        gst_on_fees_inr=fees.gst_on_fees_inr,
        gst_on_fees_recoverable_inr=fees.gst_on_fees_recoverable_inr,
        notes=fees.notes,
    )
    warnings.extend(fees.notes)

    returns = compute_return_cost(
        landed_cost_inr=landed.total_landed_cost_inr,
        benchmark=references.return_benchmark.payload,
        return_rate_multiplier=deltas.return_rate_multiplier,
    )
    ad_cost = money(apply_pct(price, inputs.ad_allowance_pct) * deltas.ad_cost_multiplier)

    landed_cost = landed.total_landed_cost_inr
    other = money(inputs.other_costs_inr)

    gross = money(revenue.taxable_inr - landed_cost)
    contribution = money(gross - fees.net_cost_inr)
    net = money(contribution - returns.expected_cost_per_order_inr - ad_cost - other)

    net_margin = money((net / price) * HUNDRED) if price else ZERO
    invested = money(landed_cost + fees.net_cost_inr + ad_cost + other)
    roi_ratio = safe_div(net, invested)
    roi = money(roi_ratio * HUNDRED) if roi_ratio is not None else ZERO

    # Costs that do not scale with price, used to invert for a target price.
    fixed_costs = money(
        landed_cost + fees.net_cost_inr + returns.expected_cost_per_order_inr + other
    )
    # price/(1+g) - fixed - price*ad = 0  =>  price = fixed / (1/(1+g) - ad)
    gst_divisor = HUNDRED / (HUNDRED + gst_rate)
    ad_fraction = money(
        pct(inputs.ad_allowance_pct) * deltas.ad_cost_multiplier / HUNDRED
    )
    denominator = money(gst_divisor - ad_fraction)
    breakeven = money(fixed_costs / denominator) if denominator > ZERO else price * 10

    threshold = references.threshold_profile.payload
    min_margin = pct(threshold.get("min_net_margin_pct", ZERO))
    margin_denominator = money(denominator - (min_margin / HUNDRED))
    min_profitable = (
        money(fixed_costs / margin_denominator)
        if margin_denominator > ZERO
        else price * 10
    )

    headroom_for_cogs = money(net + landed_cost)
    max_supplier_cost = money(max(ZERO, headroom_for_cogs - apply_pct(price, min_margin)))
    max_ad = money(max(ZERO, ad_cost + net - apply_pct(price, min_margin)))
    per_return = money(returns.cost_per_return_inr + returns.unsaleable_write_down_inr)
    max_return_rate = (
        money(((net + returns.expected_cost_per_order_inr) / per_return) * HUNDRED)
        if per_return > ZERO
        else HUNDRED
    )

    return EconomicsResult(
        scenario=deltas.scenario,
        calc_version=CALC_VERSION,
        pinned_versions=references.as_versions(),
        selling_price_inr=price,
        gst_component_inr=revenue.tax_inr,
        net_taxable_revenue_inr=revenue.taxable_inr,
        landed_cost_inr=landed_cost,
        fees=fees,
        returns=returns,
        ad_cost_inr=ad_cost,
        other_costs_inr=other,
        gross_profit_inr=gross,
        contribution_profit_inr=contribution,
        net_profit_inr=net,
        net_margin_pct=net_margin,
        roi_pct=roi,
        breakeven_price_inr=breakeven,
        min_profitable_price_inr=min_profitable,
        max_supplier_cost_inr=max_supplier_cost,
        max_ad_cost_inr=max_ad,
        max_return_rate_pct=min(max_return_rate, HUNDRED),
        assumptions=list(landed.assumptions),
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Profitability gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateOutcome:
    """The gate's verdict, with what would have to change to pass.

    ``gap_to_pass`` is the point of this object. "Fails by Rs 34 of landed
    cost" is a sourcing instruction; an unexplained rejection is not.
    """

    result: GateResult
    failed_gate: str | None
    gap_to_pass: dict[str, str]
    checks: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "result": str(self.result),
            "failed_gate": self.failed_gate,
            "gap_to_pass": self.gap_to_pass,
            "checks": self.checks,
        }


def apply_profitability_gate(
    scenarios: dict[ScenarioName, EconomicsResult],
    references: PinnedReferences,
) -> GateOutcome:
    """Evaluate the profitability floors against every scenario.

    Ordering matters: the worst-case check runs first when policy requires it,
    because a SKU that only works in good conditions is the failure mode this
    whole engine exists to catch.
    """
    threshold = references.threshold_profile.payload
    expected = scenarios[ScenarioName.EXPECTED]
    worst = scenarios.get(ScenarioName.WORST, expected)
    checks: list[dict[str, str]] = []

    def record(name: str, actual: Decimal, required: Decimal, ok: bool) -> None:
        checks.append(
            {
                "gate": name,
                "actual": str(actual),
                "required": str(required),
                "result": "PASS" if ok else "FAIL",
            }
        )

    # Defaults to True deliberately: a threshold profile that omits this key
    # gets the stricter behaviour, never the laxer one. This is policy, not a
    # fabricated business fact.
    if threshold.get("require_worst_case_non_negative", True):
        floor = pct(threshold.get("worst_case_min_net_margin_pct", ZERO))
        ok = worst.net_margin_pct >= floor
        record("worst_case_min_net_margin_pct", worst.net_margin_pct, floor, ok)
        if not ok:
            shortfall = money(
                apply_pct(worst.selling_price_inr, floor) - worst.net_profit_inr
            )
            return GateOutcome(
                result=GateResult.FAIL,
                failed_gate="worst_case_min_net_margin_pct",
                gap_to_pass={
                    "measure": "worst-case net margin",
                    "actual": f"{worst.net_margin_pct}%",
                    "required": f"{floor}%",
                    "action": (
                        f"reduce landed cost by {format_inr(shortfall)} or raise price "
                        f"to {format_inr(worst.min_profitable_price_inr)}"
                    ),
                },
                checks=checks,
            )

    ordered: list[tuple[str, Decimal, Decimal, str]] = [
        (
            "min_net_margin_pct",
            expected.net_margin_pct,
            pct(threshold.get("min_net_margin_pct", ZERO)),
            "net margin",
        ),
        (
            "min_profit_per_order_inr",
            expected.net_profit_inr,
            money(threshold.get("min_profit_per_order_inr", ZERO)),
            "net profit per order",
        ),
        ("min_roi_pct", expected.roi_pct, pct(threshold.get("min_roi_pct", ZERO)), "ROI"),
    ]
    for name, actual, required, label in ordered:
        ok = actual >= required
        record(name, actual, required, ok)
        if not ok:
            deficit = money(required - actual)
            if name == "min_net_margin_pct":
                action = (
                    f"reduce landed cost by "
                    f"{format_inr(apply_pct(expected.selling_price_inr, deficit))} "
                    f"(now {format_inr(expected.landed_cost_inr)}), or raise price "
                    f"to {format_inr(expected.min_profitable_price_inr)}"
                )
            elif name == "min_profit_per_order_inr":
                action = f"improve unit economics by {format_inr(deficit)} per order"
            else:
                action = f"improve return on invested cost by {deficit} points"
            return GateOutcome(
                result=GateResult.FAIL,
                failed_gate=name,
                gap_to_pass={
                    "measure": label,
                    "actual": str(actual),
                    "required": str(required),
                    "action": action,
                },
                checks=checks,
            )

    cost_share = (
        money((expected.landed_cost_inr / expected.selling_price_inr) * HUNDRED)
        if expected.selling_price_inr
        else HUNDRED
    )
    max_cost_share = pct(threshold.get("max_supplier_cost_pct", HUNDRED))
    marginal = cost_share > max_cost_share
    record("max_supplier_cost_pct", cost_share, max_cost_share, not marginal)

    if marginal:
        return GateOutcome(
            result=GateResult.MARGINAL,
            failed_gate=None,
            gap_to_pass={
                "measure": "landed cost as a share of price",
                "actual": f"{cost_share}%",
                "required": f"<={max_cost_share}%",
                "action": "margins clear the floor but the cost ratio leaves no room",
            },
            checks=checks,
        )

    return GateOutcome(
        result=GateResult.PASS, failed_gate=None, gap_to_pass={}, checks=checks
    )


__all__ = [
    "CALC_VERSION",
    "EconomicsInputs",
    "EconomicsResult",
    "GateOutcome",
    "ScenarioDeltas",
    "apply_profitability_gate",
    "compute_economics",
]
