"""Pricing, supplier selection, portfolio ranking and optimisation.

Everything here produces *recommendations*. Nothing executes. A recommendation
carries its evidence, its expected impact and its downside, so a human
approving it is approving a reasoned case rather than a number.

Two rules run through all of it:

**Part 1's floors cannot be bypassed.** A price the optimiser likes but that
fails the worst-case gate is blocked, not warned about. The whole reason the
gate exists is that expected-case reasoning is what talks people into losing
SKUs.

**Evidence beats enthusiasm.** Rankings and kill decisions require a minimum
sample. Killing a product on three noisy orders destroys good SKUs, and
scaling one on three lucky orders destroys capital.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from catalog.schemas import SupplierOffer
from econ.landed_cost import compute_landed_cost
from econ.money import HUNDRED, ZERO, format_inr, money, pct, safe_div
from part2.operations import RealizedEconomics, ReturnRates

logger = logging.getLogger(__name__)

PRICING_VERSION = "pricing/1.0.0"
PORTFOLIO_VERSION = "portfolio/1.0.0"


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


class PriceVerdict(StrEnum):
    ALLOWED = "ALLOWED"
    BLOCKED_BELOW_FLOOR = "BLOCKED_BELOW_FLOOR"
    BLOCKED_WORST_CASE = "BLOCKED_WORST_CASE"
    BLOCKED_CHANGE_LIMIT = "BLOCKED_CHANGE_LIMIT"


@dataclass(frozen=True)
class PricePoints:
    """The prices that matter for one SKU."""

    sku_code: str
    breakeven_inr: Decimal
    minimum_viable_inr: Decimal
    target_inr: Decimal
    competitive_inr: Decimal | None
    max_discount_pct: Decimal
    current_inr: Decimal | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "breakeven_inr": str(self.breakeven_inr),
            "minimum_viable_inr": str(self.minimum_viable_inr),
            "target_inr": str(self.target_inr),
            "competitive_inr": (
                None if self.competitive_inr is None else str(self.competitive_inr)
            ),
            "max_discount_pct": str(self.max_discount_pct),
            "current_inr": None if self.current_inr is None else str(self.current_inr),
        }


@dataclass(frozen=True)
class PriceDecision:
    """A proposed price and whether policy permits it."""

    verdict: PriceVerdict
    proposed_inr: Decimal
    reason: str
    points: PricePoints
    blocked_by: str | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict is PriceVerdict.ALLOWED

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "proposed_inr": str(self.proposed_inr),
            "reason": self.reason,
            "blocked_by": self.blocked_by,
            "points": self.points.as_dict(),
        }


def compute_price_points(
    sku_code: str,
    *,
    breakeven_inr: Decimal,
    min_profitable_inr: Decimal,
    market_p25_inr: Decimal | None,
    market_median_inr: Decimal | None,
    current_inr: Decimal | None = None,
    target_margin_uplift_pct: Decimal = Decimal("15"),
) -> PricePoints:
    """Derive the price band for one SKU.

    The target sits above the minimum viable price by a configured uplift, but
    never above the market median: a price nobody pays is not a target, it is
    a listing that never converts.
    """
    minimum = money(min_profitable_inr)
    target = money(minimum * (HUNDRED + pct(target_margin_uplift_pct)) / HUNDRED)
    if market_median_inr is not None:
        target = min(target, money(market_median_inr))
        target = max(target, minimum)

    competitive = None
    if market_p25_inr is not None:
        # Match the bottom quartile only where that still clears the floor.
        competitive = max(money(market_p25_inr), minimum)

    headroom = money(target - minimum)
    max_discount = money(headroom / target * HUNDRED) if target > ZERO else ZERO
    return PricePoints(
        sku_code=sku_code,
        breakeven_inr=money(breakeven_inr),
        minimum_viable_inr=minimum,
        target_inr=target,
        competitive_inr=competitive,
        max_discount_pct=max_discount,
        current_inr=None if current_inr is None else money(current_inr),
    )


def evaluate_price_change(
    proposed_inr: Decimal,
    points: PricePoints,
    *,
    worst_case_margin_at_proposed_pct: Decimal | None = None,
    worst_case_floor_pct: Decimal = ZERO,
    max_change_pct: Decimal = Decimal("25"),
) -> PriceDecision:
    """Decide whether a proposed price may be applied.

    Blocks rather than warns. A pricing engine that emits warnings alongside
    an applied price is a pricing engine that sells below cost on a busy day.
    """
    proposed = money(proposed_inr)

    if proposed < points.minimum_viable_inr:
        return PriceDecision(
            verdict=PriceVerdict.BLOCKED_BELOW_FLOOR,
            proposed_inr=proposed,
            reason=(
                f"{format_inr(proposed)} is below the minimum viable price of "
                f"{format_inr(points.minimum_viable_inr)}"
            ),
            points=points,
            blocked_by="minimum_viable_price",
        )

    if (
        worst_case_margin_at_proposed_pct is not None
        and worst_case_margin_at_proposed_pct < pct(worst_case_floor_pct)
    ):
        return PriceDecision(
            verdict=PriceVerdict.BLOCKED_WORST_CASE,
            proposed_inr=proposed,
            reason=(
                f"worst-case margin at this price is "
                f"{worst_case_margin_at_proposed_pct}%, below the "
                f"{worst_case_floor_pct}% floor. Part 1's gate is not "
                "bypassable by a pricing recommendation."
            ),
            points=points,
            blocked_by="worst_case_min_net_margin_pct",
        )

    if points.current_inr is not None and points.current_inr > ZERO:
        change = abs(
            money((proposed - points.current_inr) / points.current_inr * HUNDRED)
        )
        if change > pct(max_change_pct):
            return PriceDecision(
                verdict=PriceVerdict.BLOCKED_CHANGE_LIMIT,
                proposed_inr=proposed,
                reason=(
                    f"a {change}% move exceeds the {max_change_pct}% single-change "
                    "limit; large jumps suppress listings and confuse buyers"
                ),
                points=points,
                blocked_by="price_change_limit",
            )

    return PriceDecision(
        verdict=PriceVerdict.ALLOWED,
        proposed_inr=proposed,
        reason=f"{format_inr(proposed)} clears every floor",
        points=points,
    )


# --------------------------------------------------------------------------
# Supplier selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SupplierScore:
    """One supplier option, scored on true cost rather than sticker price."""

    supplier_id: str
    supplier_name: str
    unit_cost_inr: Decimal
    landed_cost_inr: Decimal
    moq_units: int
    lead_time_days: int | None
    tax_invoice: bool
    capital_required_inr: Decimal
    reliability_pct: Decimal | None
    defect_rate_pct: Decimal | None
    total_score: Decimal
    disqualified: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "supplier_name": self.supplier_name,
            "unit_cost_inr": str(self.unit_cost_inr),
            "landed_cost_inr": str(self.landed_cost_inr),
            "moq_units": self.moq_units,
            "lead_time_days": self.lead_time_days,
            "tax_invoice": self.tax_invoice,
            "capital_required_inr": str(self.capital_required_inr),
            "total_score": str(self.total_score),
            "disqualified": self.disqualified,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SupplierHistory:
    """Measured performance for one supplier, where it exists."""

    on_time_pct: Decimal | None = None
    defect_rate_pct: Decimal | None = None
    orders_observed: int = 0


def rank_suppliers(
    offers: list[SupplierOffer],
    *,
    names: dict[str, str] | None = None,
    lead_times: dict[str, int] | None = None,
    history: dict[str, SupplierHistory] | None = None,
    expected_monthly_velocity: Decimal = Decimal("10"),
    capital_cap_inr: Decimal = Decimal("15000"),
) -> list[SupplierScore]:
    """Rank supplier options by true economics, not headline price.

    The cheapest unit cost is routinely the worst option: no tax invoice means
    no input tax credit, and a large MOQ means capital locked for months. Both
    are priced explicitly here so the comparison is like-for-like.
    """
    names = names or {}
    lead_times = lead_times or {}
    history = history or {}
    scored: list[SupplierScore] = []

    for offer in offers:
        supplier_id = str(offer.supplier_id)
        name = names.get(supplier_id, supplier_id[:8])
        lead_time = lead_times.get(supplier_id)
        record = history.get(supplier_id, SupplierHistory())
        notes: list[str] = []

        landed = compute_landed_cost(
            offer, expected_monthly_velocity=expected_monthly_velocity
        )
        capital = money(landed.total_landed_cost_inr * offer.moq_units)

        if not offer.tax_invoice_available:
            notes.append(
                "no tax invoice: input tax credit is zero, so the effective cost "
                "is higher than the sticker price suggests"
            )
        if offer.moq_units > 1:
            notes.append(f"MOQ {offer.moq_units} locks {format_inr(capital)} of capital")

        disqualified = False
        if capital > capital_cap_inr:
            disqualified = True
            notes.append(
                f"capital required {format_inr(capital)} exceeds the "
                f"{format_inr(capital_cap_inr)} per-product cap"
            )

        # Lower landed cost is better; everything else adjusts around it.
        cost_score = money(HUNDRED - min(landed.total_landed_cost_inr / 10, HUNDRED))
        if lead_time is None:
            # Seven days was previously assumed here, producing a 14-point
            # penalty as though a dispatch time had been measured. An unstated
            # lead time is scored neutrally and reported, so it neither
            # flatters nor punishes a supplier on invented evidence.
            lead_penalty = ZERO
            notes.append(
                "dispatch time not stated by the supplier; no lead-time "
                "penalty applied and none assumed"
            )
        else:
            lead_penalty = money(min(Decimal(lead_time) * 2, Decimal("30")))
        capital_penalty = money(
            min(capital / capital_cap_inr * Decimal("25"), Decimal("25"))
            if capital_cap_inr > ZERO
            else ZERO
        )
        # No separate penalty for a missing tax invoice: landed_cost already
        # reflects the lost input tax credit, and scoring it twice would
        # overstate the gap. The note above is what makes the reason visible.

        reliability_adj = ZERO
        if record.on_time_pct is not None and record.orders_observed >= 5:
            reliability_adj = money((record.on_time_pct - Decimal("90")) / 5)
            notes.append(
                f"measured on-time rate {record.on_time_pct}% over "
                f"{record.orders_observed} orders"
            )
        defect_adj = ZERO
        if record.defect_rate_pct is not None and record.orders_observed >= 5:
            defect_adj = money(-record.defect_rate_pct * 2)

        total = money(
            cost_score + reliability_adj + defect_adj - lead_penalty - capital_penalty
        )

        scored.append(
            SupplierScore(
                supplier_id=supplier_id,
                supplier_name=name,
                unit_cost_inr=money(offer.unit_cost_inr),
                landed_cost_inr=landed.total_landed_cost_inr,
                moq_units=offer.moq_units,
                lead_time_days=lead_time,
                tax_invoice=offer.tax_invoice_available,
                capital_required_inr=capital,
                reliability_pct=record.on_time_pct,
                defect_rate_pct=record.defect_rate_pct,
                total_score=total,
                disqualified=disqualified,
                notes=notes,
            )
        )

    return sorted(scored, key=lambda s: (s.disqualified, -s.total_score))


def select_supplier(scores: list[SupplierScore]) -> SupplierScore | None:
    """The economically strongest qualifying option, if any."""
    qualifying = [s for s in scores if not s.disqualified]
    return qualifying[0] if qualifying else None


# --------------------------------------------------------------------------
# Portfolio and lifecycle
# --------------------------------------------------------------------------


class Lifecycle(StrEnum):
    TEST = "TEST"
    VALIDATE = "VALIDATE"
    SCALE = "SCALE"
    MAINTAIN = "MAINTAIN"
    REDUCE = "REDUCE"
    KILL = "KILL"


@dataclass(frozen=True)
class PortfolioEntry:
    """One product's standing in the portfolio."""

    sku_code: str
    realized: RealizedEconomics
    returns: ReturnRates
    capital_locked_inr: Decimal
    lifecycle: Lifecycle
    rank_score: Decimal | None
    evidence_sufficient: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "lifecycle": str(self.lifecycle),
            "rank_score": None if self.rank_score is None else str(self.rank_score),
            "evidence_sufficient": self.evidence_sufficient,
            "capital_locked_inr": str(self.capital_locked_inr),
            "actual_contribution_inr": str(self.realized.actual_contribution_inr),
            "return_rate_pct": str(self.returns.return_rate_pct),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PortfolioPolicy:
    """Thresholds governing lifecycle recommendations."""

    min_sample_for_decision: int = 20
    min_sample_for_kill: int = 30
    kill_margin_pct: Decimal = Decimal("0")
    reduce_margin_pct: Decimal = Decimal("10")
    scale_margin_pct: Decimal = Decimal("25")
    max_return_rate_pct: Decimal = Decimal("35")
    max_rto_rate_pct: Decimal = Decimal("20")


def rank_portfolio(
    entries: dict[str, tuple[RealizedEconomics, ReturnRates, Decimal]],
    *,
    policy: PortfolioPolicy | None = None,
) -> list[PortfolioEntry]:
    """Rank products by risk-adjusted contribution per unit of capital.

    Not by revenue. Revenue ranks the product that turns over the most money,
    which in a capital-constrained operation is frequently the one that should
    be killed: high revenue, thin margin, heavy returns, and every rupee of
    working capital tied up in it.
    """
    policy = policy or PortfolioPolicy()
    result: list[PortfolioEntry] = []

    for sku_code, (realized, returns, capital) in entries.items():
        reasons: list[str] = []
        sufficient = realized.sample_size >= policy.min_sample_for_decision

        # Contribution per rupee of capital, discounted by return risk.
        risk_discount = (
            money(HUNDRED - min(returns.return_rate_pct + returns.rto_rate_pct, HUNDRED))
            / HUNDRED
        )
        per_capital = safe_div(realized.actual_contribution_inr, capital)
        score = (
            money(per_capital * risk_discount * HUNDRED)
            if per_capital is not None
            else None
        )

        lifecycle = _classify(realized, returns, policy, reasons, sufficient)
        result.append(
            PortfolioEntry(
                sku_code=sku_code,
                realized=realized,
                returns=returns,
                capital_locked_inr=money(capital),
                lifecycle=lifecycle,
                rank_score=score,
                evidence_sufficient=sufficient,
                reasons=reasons,
            )
        )

    return sorted(
        result,
        key=lambda e: e.rank_score if e.rank_score is not None else Decimal("-99999"),
        reverse=True,
    )


def _classify(
    realized: RealizedEconomics,
    returns: ReturnRates,
    policy: PortfolioPolicy,
    reasons: list[str],
    sufficient: bool,
) -> Lifecycle:
    """Assign a lifecycle stage, refusing to act on thin evidence."""
    margin = realized.actual_margin_pct

    if realized.sample_size == 0:
        reasons.append("no settled orders yet: this is still a test")
        return Lifecycle.TEST
    if not sufficient:
        reasons.append(
            f"{realized.sample_size} settled order(s); "
            f"{policy.min_sample_for_decision} needed before a lifecycle call"
        )
        return Lifecycle.VALIDATE
    if margin is None:
        reasons.append("margin cannot be computed; settlements are incomplete")
        return Lifecycle.VALIDATE

    if (
        realized.sample_size >= policy.min_sample_for_kill
        and margin <= policy.kill_margin_pct
    ):
        reasons.append(
            f"contribution is {format_inr(realized.actual_contribution_inr)} at "
            f"{margin}% margin over {realized.sample_size} orders: losing money "
            "consistently, not noisily"
        )
        return Lifecycle.KILL

    if returns.return_rate_pct > policy.max_return_rate_pct and returns.is_measured:
        reasons.append(
            f"measured return rate {returns.return_rate_pct}% exceeds the "
            f"{policy.max_return_rate_pct}% limit"
        )
        return Lifecycle.REDUCE
    if returns.rto_rate_pct > policy.max_rto_rate_pct and returns.is_measured:
        reasons.append(f"RTO rate {returns.rto_rate_pct}% is unsustainable")
        return Lifecycle.REDUCE

    if margin >= policy.scale_margin_pct:
        reasons.append(f"{margin}% realized margin over {realized.sample_size} orders")
        return Lifecycle.SCALE
    if margin < policy.reduce_margin_pct:
        reasons.append(
            f"{margin}% realized margin is below the {policy.reduce_margin_pct}% floor"
        )
        return Lifecycle.REDUCE

    reasons.append(f"{margin}% realized margin: steady")
    return Lifecycle.MAINTAIN


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------


class Action(StrEnum):
    CHANGE_PRICE = "CHANGE_PRICE"
    CHANGE_SUPPLIER = "CHANGE_SUPPLIER"
    REDUCE_MOQ = "REDUCE_MOQ"
    INCREASE_INVENTORY = "INCREASE_INVENTORY"
    REDUCE_INVENTORY = "REDUCE_INVENTORY"
    REDUCE_ADVERTISING = "REDUCE_ADVERTISING"
    PAUSE_PRODUCT = "PAUSE_PRODUCT"
    SCALE_PRODUCT = "SCALE_PRODUCT"
    KILL_PRODUCT = "KILL_PRODUCT"
    COLLECT_MORE_DATA = "COLLECT_MORE_DATA"


@dataclass(frozen=True)
class Recommendation:
    """A proposed change, with everything needed to judge it."""

    action: Action
    sku_code: str
    current_state: str
    proposed_change: str
    expected_impact: str
    downside: str
    confidence: str
    evidence: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": str(self.action),
            "sku_code": self.sku_code,
            "current_state": self.current_state,
            "proposed_change": self.proposed_change,
            "expected_impact": self.expected_impact,
            "downside": self.downside,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "reason": self.reason,
        }


def recommend(
    entry: PortfolioEntry, *, policy: PortfolioPolicy | None = None
) -> list[Recommendation]:
    """Deterministic recommendations for one product.

    Same inputs, same recommendations, every time. Nothing here mutates
    business state; each recommendation must be approved and executed through
    the execution gate before it can reach a marketplace.
    """
    policy = policy or PortfolioPolicy()
    realized, returns = entry.realized, entry.returns
    evidence = [
        f"{realized.sample_size} settled orders",
        f"contribution {format_inr(realized.actual_contribution_inr)}",
        f"return rate {returns.return_rate_pct}% ({returns.confidence})",
        realized.coverage_note,
    ]
    confidence = "HIGH" if entry.evidence_sufficient else "LOW"
    recommendations: list[Recommendation] = []

    if not entry.evidence_sufficient:
        recommendations.append(
            Recommendation(
                action=Action.COLLECT_MORE_DATA,
                sku_code=entry.sku_code,
                current_state=f"{realized.sample_size} settled orders, {entry.lifecycle}",
                proposed_change=(
                    f"keep exposure flat until {policy.min_sample_for_decision} "
                    "orders have settled"
                ),
                expected_impact="a lifecycle decision that is not driven by noise",
                downside="slower to scale a product that may turn out to be good",
                confidence=confidence,
                evidence=evidence,
                reason="sample is too small to distinguish signal from variance",
            )
        )
        return recommendations

    if entry.lifecycle is Lifecycle.KILL:
        recommendations.append(
            Recommendation(
                action=Action.KILL_PRODUCT,
                sku_code=entry.sku_code,
                current_state=(
                    f"{realized.actual_margin_pct}% realized margin over "
                    f"{realized.sample_size} orders"
                ),
                proposed_change="stop reordering, sell through remaining stock, delist",
                expected_impact=(
                    f"releases {format_inr(entry.capital_locked_inr)} of capital and "
                    "stops the ongoing loss"
                ),
                downside=(
                    "remaining stock may have to be cleared below cost; the loss is "
                    "realised now rather than deferred"
                ),
                confidence=confidence,
                evidence=evidence,
                reason="; ".join(entry.reasons),
            )
        )
    elif entry.lifecycle is Lifecycle.SCALE:
        recommendations.append(
            Recommendation(
                action=Action.SCALE_PRODUCT,
                sku_code=entry.sku_code,
                current_state=f"{realized.actual_margin_pct}% realized margin",
                proposed_change="raise inventory cover and reorder quantity one step",
                expected_impact="more contribution from a product that is already proving out",
                downside=(
                    "ties up more capital in one SKU; a demand or supplier shock hits harder"
                ),
                confidence=confidence,
                evidence=evidence,
                reason="; ".join(entry.reasons),
            )
        )
    elif entry.lifecycle is Lifecycle.REDUCE:
        if returns.is_measured and returns.return_rate_pct > policy.max_return_rate_pct:
            recommendations.append(
                Recommendation(
                    action=Action.PAUSE_PRODUCT,
                    sku_code=entry.sku_code,
                    current_state=f"return rate {returns.return_rate_pct}%",
                    proposed_change="pause new orders and investigate the return reasons",
                    expected_impact="stops the bleeding while the cause is diagnosed",
                    downside="loses ranking and momentum on the marketplace",
                    confidence=confidence,
                    evidence=evidence,
                    reason="; ".join(entry.reasons),
                )
            )
        recommendations.append(
            Recommendation(
                action=Action.REDUCE_INVENTORY,
                sku_code=entry.sku_code,
                current_state=f"{format_inr(entry.capital_locked_inr)} locked",
                proposed_change="cut cover to the minimum and stop replenishing",
                expected_impact="frees capital for products earning more per rupee",
                downside="stockouts if demand recovers",
                confidence=confidence,
                evidence=evidence,
                reason="; ".join(entry.reasons),
            )
        )

    if realized.unreconciled_inr > ZERO:
        recommendations.append(
            Recommendation(
                action=Action.COLLECT_MORE_DATA,
                sku_code=entry.sku_code,
                current_state=f"{format_inr(realized.unreconciled_inr)} unsettled",
                proposed_change="reconcile outstanding settlements before acting",
                expected_impact="the margin figure stops understating actual performance",
                downside="none",
                confidence="HIGH",
                evidence=evidence,
                reason="unsettled revenue makes realized margin look worse than it is",
            )
        )

    return recommendations


def optimisation_report(recommendations: list[Recommendation]) -> str:
    """Render recommendations with their reasoning, not just their verdicts."""
    if not recommendations:
        return "No optimisation actions recommended.\n"
    lines = [f"OPTIMISATION - {len(recommendations)} recommendation(s)", ""]
    for item in recommendations:
        lines += [
            f"  ACTION            {item.action}  [{item.sku_code}]",
            f"  WHY               {item.reason[:100]}",
            f"  CURRENT           {item.current_state}",
            f"  CHANGE            {item.proposed_change}",
            f"  EXPECTED BENEFIT  {item.expected_impact}",
            f"  DOWNSIDE          {item.downside}",
            f"  CONFIDENCE        {item.confidence}",
            f"  EVIDENCE          {'; '.join(item.evidence[:3])}",
            "",
        ]
    return "\n".join(lines)


def portfolio_report(entries: list[PortfolioEntry]) -> str:
    """Product portfolio, ranked by contribution per unit of capital."""
    if not entries:
        return "No products in the portfolio.\n"
    lines = [
        f"PRODUCT PORTFOLIO - {len(entries)} product(s)",
        "",
        f"{'SKU':<18} {'orders':>7} {'contribution':>14} {'margin':>8} "
        f"{'ret%':>6} {'capital':>12} {'stage':<10} {'score':>8}",
        f"{'-' * 18} {'-' * 7} {'-' * 14} {'-' * 8} {'-' * 6} {'-' * 12} {'-' * 10} {'-' * 8}",
    ]
    for entry in entries:
        realized = entry.realized
        margin = (
            "n/a"
            if realized.actual_margin_pct is None
            else f"{realized.actual_margin_pct:.1f}%"
        )
        score = "n/a" if entry.rank_score is None else f"{entry.rank_score:.1f}"
        lines.append(
            f"{entry.sku_code[:17]:<18} {realized.sample_size:>7} "
            f"{format_inr(realized.actual_contribution_inr):>14} {margin:>8} "
            f"{entry.returns.return_rate_pct:>5.1f}% "
            f"{format_inr(entry.capital_locked_inr):>12} "
            f"{entry.lifecycle!s:<10} {score:>8}"
        )
    lines += [
        "",
        "Ranked by risk-adjusted contribution per rupee of capital, not revenue.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "PORTFOLIO_VERSION",
    "PRICING_VERSION",
    "Action",
    "Lifecycle",
    "PortfolioEntry",
    "PortfolioPolicy",
    "PriceDecision",
    "PricePoints",
    "PriceVerdict",
    "Recommendation",
    "SupplierHistory",
    "SupplierScore",
    "compute_price_points",
    "evaluate_price_change",
    "optimisation_report",
    "portfolio_report",
    "rank_portfolio",
    "rank_suppliers",
    "recommend",
    "select_supplier",
]
