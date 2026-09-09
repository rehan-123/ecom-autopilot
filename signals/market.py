"""Market evidence: signals, price reference, coverage, scoring and risk.

The governing rule is that missing data stays missing. Roughly half the demand
and competition signals for Indian marketplaces have no legitimate source, and
the honest response is to record the absence rather than substitute a zero or
an average. A zero is a claim; an average is a claim; ``DATA_UNAVAILABLE`` is
the truth.

That rule has a consequence for scoring. If 40% of the weight cannot be
populated, a single 0-100 number is a lie by omission. So the score carries its
own coverage, the two are never blended, and below a configured coverage floor
no ranking value is emitted at all.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from catalog.enums import (
    HeadroomResult,
    RiskLevel,
    RiskTier,
    ScoreResult,
    SignalSource,
    SignalStatus,
)
from catalog.schemas import Product
from econ.money import HUNDRED, ZERO, format_inr, money, pct, rupees

logger = logging.getLogger(__name__)

SCORING_VERSION = "opportunity/1.0.0"
RISK_VERSION = "risk/1.0.0"

#: Signals the framework understands. Adding one is a data change here plus a
#: weight in the profile; no other module needs to know.
SIGNAL_TYPES: tuple[str, ...] = (
    "SEARCH_INTEREST",
    "TREND_DIRECTION",
    "REVIEW_VOLUME",
    "REVIEW_VELOCITY",
    "COMPETITOR_COUNT",
    "BRAND_DOMINANCE",
    "MARKET_PRICE",
    "OWN_SALES",
    "RETURN_RATE",
)

#: Default weights. Profitability and return risk carry the most because they
#: are the two components that can actually be populated reliably, and the two
#: that historically kill Indian marketplace SKUs. Weighting unobtainable
#: signals heavily only guarantees low coverage on everything.
DEFAULT_WEIGHTS: dict[str, Decimal] = {
    "PROFITABILITY": Decimal("40"),
    "RETURN_RISK": Decimal("15"),
    "SEARCH_INTEREST": Decimal("10"),
    "TREND_DIRECTION": Decimal("10"),
    "COMPETITOR_COUNT": Decimal("10"),
    "BRAND_DOMINANCE": Decimal("5"),
    "COMPLIANCE_RISK": Decimal("5"),
    "OPERATIONAL_COMPLEXITY": Decimal("5"),
}

DEFAULT_SIGNAL_MAX_AGE_DAYS = 45


@dataclass(frozen=True)
class Signal:
    """One observation, or an explicit record that none could be obtained."""

    signal_type: str
    status: SignalStatus
    source: SignalSource
    collected_by: str
    value_numeric: Decimal | None = None
    value_text: str | None = None
    source_ref: str | None = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence: str | None = None

    def __post_init__(self) -> None:
        if self.status is SignalStatus.DATA_UNAVAILABLE and (
            self.value_numeric is not None or self.value_text is not None
        ):
            raise ValueError(
                "DATA_UNAVAILABLE must not carry a value; an absent signal is "
                "not a zero and not an average"
            )
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(f"unknown signal type {self.signal_type!r}")

    def is_usable(self, *, max_age_days: int = DEFAULT_SIGNAL_MAX_AGE_DAYS) -> bool:
        if self.status is not SignalStatus.OBSERVED or self.value_numeric is None:
            return False
        return datetime.now(UTC) - self.collected_at <= timedelta(days=max_age_days)

    def effective_status(
        self, *, max_age_days: int = DEFAULT_SIGNAL_MAX_AGE_DAYS
    ) -> SignalStatus:
        if self.status is not SignalStatus.OBSERVED:
            return self.status
        if datetime.now(UTC) - self.collected_at > timedelta(days=max_age_days):
            return SignalStatus.STALE
        return SignalStatus.OBSERVED


def unavailable(signal_type: str, *, reason: str, collected_by: str) -> Signal:
    """Record that a signal has no legitimate source.

    Used rather than omitting the signal, so the gap is visible in coverage
    and in the report instead of quietly shrinking the denominator.
    """
    return Signal(
        signal_type=signal_type,
        status=SignalStatus.DATA_UNAVAILABLE,
        source=SignalSource.MANUAL,
        collected_by=collected_by,
        source_ref=reason,
    )


# --------------------------------------------------------------------------
# Market price reference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparable:
    """One observed competing listing."""

    marketplace: str
    title: str
    price_inr: Decimal
    reference: str
    observed_at: datetime

    def as_dict(self) -> dict[str, str]:
        return {
            "marketplace": self.marketplace,
            "title": self.title,
            "price_inr": str(rupees(self.price_inr)),
            "reference": self.reference,
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(frozen=True)
class MarketPriceReference:
    """What comparable items actually sell for.

    Required before economics can be trusted. A SKU can clear every internal
    gate at a price nobody in the market will pay, and this is the only input
    that tests the arithmetic against the world.
    """

    comparables: list[Comparable]
    collected_by: str
    p25_inr: Decimal
    median_inr: Decimal
    p75_inr: Decimal
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: str | None = None

    MIN_COMPARABLES = 3

    @classmethod
    def from_comparables(
        cls, comparables: list[Comparable], *, collected_by: str, notes: str | None = None
    ) -> MarketPriceReference:
        if len(comparables) < cls.MIN_COMPARABLES:
            raise ValueError(
                f"need at least {cls.MIN_COMPARABLES} comparables to form a price "
                f"reference; got {len(comparables)}"
            )
        prices = sorted(money(c.price_inr) for c in comparables)
        return cls(
            comparables=list(comparables),
            collected_by=collected_by,
            p25_inr=_percentile(prices, Decimal("0.25")),
            median_inr=money(statistics.median(prices)),
            p75_inr=_percentile(prices, Decimal("0.75")),
            notes=notes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_comparables": len(self.comparables),
            "p25_inr": str(rupees(self.p25_inr)),
            "median_inr": str(rupees(self.median_inr)),
            "p75_inr": str(rupees(self.p75_inr)),
            "collected_at": self.collected_at.isoformat(),
            "comparables": [c.as_dict() for c in self.comparables],
        }


def _percentile(sorted_prices: list[Decimal], fraction: Decimal) -> Decimal:
    """Linear-interpolation percentile over a sorted list."""
    if not sorted_prices:
        return ZERO
    if len(sorted_prices) == 1:
        return sorted_prices[0]
    position = fraction * Decimal(len(sorted_prices) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_prices) - 1)
    weight = position - Decimal(lower)
    return money(
        sorted_prices[lower] + (sorted_prices[upper] - sorted_prices[lower]) * weight
    )


@dataclass(frozen=True)
class HeadroomOutcome:
    """Whether the market supports the minimum profitable price."""

    result: HeadroomResult
    headroom_pct: Decimal
    min_profitable_price_inr: Decimal
    market_p25_inr: Decimal
    explanation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "result": str(self.result),
            "headroom_pct": str(self.headroom_pct),
            "min_profitable_price_inr": str(rupees(self.min_profitable_price_inr)),
            "market_p25_inr": str(rupees(self.market_p25_inr)),
            "explanation": self.explanation,
        }


def check_price_headroom(
    *,
    min_profitable_price_inr: Decimal,
    reference: MarketPriceReference,
    min_headroom_pct: Decimal = Decimal("10"),
    marginal_headroom_pct: Decimal = ZERO,
) -> HeadroomOutcome:
    """Compare the minimum viable price against the bottom of the market."""
    minimum = money(min_profitable_price_inr)
    p25 = money(reference.p25_inr)
    headroom = (
        HUNDRED if minimum <= ZERO else money(((p25 - minimum) / minimum) * HUNDRED)
    )

    if headroom < pct(marginal_headroom_pct):
        result = HeadroomResult.BLOCK
        explanation = (
            f"market P25 of {format_inr(p25)} is below the minimum profitable "
            f"price of {format_inr(minimum)}: this cannot be sold profitably at "
            "market rates"
        )
    elif headroom < pct(min_headroom_pct):
        result = HeadroomResult.MARGINAL
        explanation = (
            f"only {headroom}% headroom above the minimum profitable price; "
            "no room for promotions or a price war"
        )
    else:
        result = HeadroomResult.PASS
        explanation = f"{headroom}% headroom above the minimum profitable price"

    return HeadroomOutcome(result, headroom, minimum, p25, explanation)


# --------------------------------------------------------------------------
# Coverage and opportunity score
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreComponent:
    """One weighted input to the opportunity score."""

    name: str
    weight: Decimal
    normalized: Decimal | None
    status: SignalStatus
    source: str

    @property
    def populated(self) -> bool:
        return self.normalized is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": str(self.weight),
            "normalized": None if self.normalized is None else str(self.normalized),
            "status": str(self.status),
            "source": self.source,
        }


@dataclass(frozen=True)
class OpportunityScore:
    """Score value and evidence coverage, never blended.

    ``94 at 45% coverage`` and ``78 at 95% coverage`` are not comparable, so
    ``value`` never travels without ``coverage_pct`` and ``display()`` refuses
    to render a bare number.
    """

    value: Decimal | None
    coverage_pct: Decimal
    result: ScoreResult
    components: list[ScoreComponent]
    missing: list[str]
    weights_version: str = SCORING_VERSION

    def display(self) -> str:
        if self.result is ScoreResult.INSUFFICIENT_DATA:
            return f"INSUFFICIENT_DATA ({self.coverage_pct}% coverage)"
        return f"{self.value}/100 ({self.coverage_pct}% coverage)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else str(self.value),
            "coverage_pct": str(self.coverage_pct),
            "result": str(self.result),
            "weights_version": self.weights_version,
            "components": [c.as_dict() for c in self.components],
            "missing": list(self.missing),
            "display": self.display(),
        }


def compute_opportunity_score(
    components: list[ScoreComponent], *, min_coverage_pct: Decimal = Decimal("60")
) -> OpportunityScore:
    """Weighted score over populated components only.

    The denominator is the populated weight, not the total weight. Dividing by
    the total would silently treat every missing signal as a zero.
    """
    total_weight = sum((c.weight for c in components), ZERO)
    populated = [c for c in components if c.populated]
    populated_weight = sum((c.weight for c in populated), ZERO)

    coverage = (
        money((populated_weight / total_weight) * HUNDRED).quantize(Decimal("0.1"))
        if total_weight > ZERO
        else ZERO
    )
    missing = [c.name for c in components if not c.populated]

    if coverage < pct(min_coverage_pct) or populated_weight == ZERO:
        return OpportunityScore(
            value=None,
            coverage_pct=coverage,
            result=ScoreResult.INSUFFICIENT_DATA,
            components=components,
            missing=missing,
        )

    weighted = sum((c.weight * c.normalized for c in populated), ZERO)
    value = money(weighted / populated_weight).quantize(Decimal("0.1"))
    return OpportunityScore(
        value=min(max(value, ZERO), HUNDRED),
        coverage_pct=coverage,
        result=ScoreResult.SCORED,
        components=components,
        missing=missing,
    )


def build_components(
    *,
    net_margin_pct: Decimal | None,
    return_rate_pct: Decimal | None,
    signals: dict[str, Signal],
    compliance_clear: bool | None,
    operational_complexity: Decimal | None,
    weights: dict[str, Decimal] | None = None,
) -> list[ScoreComponent]:
    """Assemble score components from economics, risk and market signals."""
    weights = weights or DEFAULT_WEIGHTS
    components: list[ScoreComponent] = []

    def add(
        name: str, normalized: Decimal | None, status: SignalStatus, source: str
    ) -> None:
        components.append(
            ScoreComponent(name, weights.get(name, ZERO), normalized, status, source)
        )

    # Margin normalised so 0% scores 0 and 40%+ scores 100.
    if net_margin_pct is None:
        add("PROFITABILITY", None, SignalStatus.DATA_UNAVAILABLE, "economics")
    else:
        scaled = min(max(money(net_margin_pct * Decimal("2.5")), ZERO), HUNDRED)
        add("PROFITABILITY", scaled, SignalStatus.OBSERVED, "economics")

    if return_rate_pct is None:
        add("RETURN_RISK", None, SignalStatus.DATA_UNAVAILABLE, "benchmark")
    else:
        scaled = min(max(money(HUNDRED - return_rate_pct * Decimal("2")), ZERO), HUNDRED)
        add("RETURN_RISK", scaled, SignalStatus.OBSERVED, "benchmark")

    for name in (
        "SEARCH_INTEREST",
        "TREND_DIRECTION",
        "COMPETITOR_COUNT",
        "BRAND_DOMINANCE",
    ):
        signal = signals.get(name)
        if signal is None:
            add(name, None, SignalStatus.DATA_UNAVAILABLE, "none")
        elif signal.is_usable():
            add(
                name,
                _normalise(name, signal.value_numeric),
                SignalStatus.OBSERVED,
                str(signal.source),
            )
        else:
            add(name, None, signal.effective_status(), str(signal.source))

    if compliance_clear is None:
        add("COMPLIANCE_RISK", None, SignalStatus.REQUIRES_REVIEW, "compliance")
    else:
        add(
            "COMPLIANCE_RISK",
            HUNDRED if compliance_clear else ZERO,
            SignalStatus.OBSERVED,
            "compliance",
        )

    if operational_complexity is None:
        add("OPERATIONAL_COMPLEXITY", None, SignalStatus.DATA_UNAVAILABLE, "risk")
    else:
        add(
            "OPERATIONAL_COMPLEXITY",
            min(max(money(HUNDRED - operational_complexity), ZERO), HUNDRED),
            SignalStatus.OBSERVED,
            "risk",
        )

    return components


def _normalise(signal_type: str, raw: Decimal | None) -> Decimal:
    """Map a raw signal onto 0-100, higher meaning more attractive."""
    if raw is None:
        return ZERO
    value = money(raw)
    if signal_type in ("COMPETITOR_COUNT", "BRAND_DOMINANCE"):
        # More competitors or stronger brand dominance is worse.
        return min(max(money(HUNDRED - value), ZERO), HUNDRED)
    return min(max(value, ZERO), HUNDRED)


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAssessment:
    """Deterministic risk profile driving which thresholds apply."""

    tier: RiskTier
    return_risk: RiskLevel
    fragility_risk: RiskLevel
    sizing_risk: RiskLevel
    warranty_risk: RiskLevel
    supplier_concentration_risk: RiskLevel
    seasonality_risk: RiskLevel
    operational_complexity: Decimal
    rationale: list[str]
    ruleset_version: str = RISK_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": str(self.tier),
            "return_risk": str(self.return_risk),
            "fragility_risk": str(self.fragility_risk),
            "sizing_risk": str(self.sizing_risk),
            "warranty_risk": str(self.warranty_risk),
            "supplier_concentration_risk": str(self.supplier_concentration_risk),
            "seasonality_risk": str(self.seasonality_risk),
            "operational_complexity": str(self.operational_complexity),
            "rationale": list(self.rationale),
            "ruleset_version": self.ruleset_version,
        }


def assess_risk(
    product: Product,
    *,
    supplier_count: int,
    benchmark_return_rate_pct: Decimal | None,
    seasonal: bool = False,
) -> RiskAssessment:
    """Classify a product's risk from its own attributes.

    Deterministic and explainable: every elevated level appends its reason, so
    a HIGH_RETURN tier can always be traced to the attribute that caused it.
    """
    rationale: list[str] = []
    complexity = ZERO

    if benchmark_return_rate_pct is None:
        return_risk = RiskLevel.UNKNOWN
        rationale.append("no return benchmark available for this category")
    elif benchmark_return_rate_pct >= Decimal("25"):
        return_risk = RiskLevel.HIGH
        rationale.append(f"category return rate {benchmark_return_rate_pct}% is high")
    elif benchmark_return_rate_pct >= Decimal("12"):
        return_risk = RiskLevel.MEDIUM
    else:
        return_risk = RiskLevel.LOW

    fragility = RiskLevel.HIGH if product.fragile else RiskLevel.LOW
    if product.fragile:
        rationale.append("fragile goods carry damage and dispute exposure")
        complexity += Decimal("25")

    sizing = RiskLevel.HIGH if product.sized else RiskLevel.LOW
    if product.sized:
        rationale.append("sized goods return far more often than unsized ones")
        complexity += Decimal("25")

    if product.warranty_months and product.warranty_months > 0:
        warranty = RiskLevel.MEDIUM if product.warranty_months <= 12 else RiskLevel.HIGH
        rationale.append(f"{product.warranty_months}-month warranty creates a claim path")
        complexity += Decimal("15")
    else:
        warranty = RiskLevel.LOW

    if supplier_count <= 1:
        concentration = RiskLevel.HIGH
        rationale.append("single supplier: no fallback if they fail or raise prices")
        complexity += Decimal("20")
    elif supplier_count == 2:
        concentration = RiskLevel.MEDIUM
    else:
        concentration = RiskLevel.LOW

    seasonality = RiskLevel.HIGH if seasonal else RiskLevel.LOW
    if seasonal:
        rationale.append("seasonal demand concentrates dead-stock risk")
        complexity += Decimal("15")

    if product.weight_g and product.weight_g > 2000:
        rationale.append("heavy item: shipping dominates the economics")
        complexity += Decimal("10")

    high_levels = [fragility, sizing, warranty, return_risk].count(RiskLevel.HIGH)
    if product.sized or return_risk is RiskLevel.HIGH or high_levels >= 2:
        tier = RiskTier.HIGH_RETURN
    elif high_levels == 1 or complexity >= Decimal("30"):
        tier = RiskTier.MEDIUM
    else:
        tier = RiskTier.LOW

    if not rationale:
        rationale.append("no elevated risk attributes detected")

    return RiskAssessment(
        tier=tier,
        return_risk=return_risk,
        fragility_risk=fragility,
        sizing_risk=sizing,
        warranty_risk=warranty,
        supplier_concentration_risk=concentration,
        seasonality_risk=seasonality,
        operational_complexity=min(complexity, HUNDRED),
        rationale=rationale,
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "RISK_VERSION",
    "SCORING_VERSION",
    "SIGNAL_TYPES",
    "Comparable",
    "HeadroomOutcome",
    "MarketPriceReference",
    "OpportunityScore",
    "RiskAssessment",
    "ScoreComponent",
    "Signal",
    "assess_risk",
    "build_components",
    "check_price_headroom",
    "compute_opportunity_score",
    "unavailable",
]
