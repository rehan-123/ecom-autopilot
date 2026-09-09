"""Opportunity ranking: from candidates to RECOMMENDED, BACKUP or REJECTED.

This layer orchestrates; it does not calculate. Landed cost, scenarios, the
profitability gate, price headroom, compliance and scoring all come from the
existing engines, so there is exactly one definition of what a product costs
and one of whether it clears policy. What is new is the classification and the
ordering.

The three verdicts mean different things:

``RECOMMENDED`` cleared every automated gate and has enough evidence behind it
to be worth a human's time.

``BACKUP`` would clear the gates but rests on thin evidence, or is held up by
something recoverable — missing comparables, an unverified rate card. These are
the ones to fix, not to discard.

``REJECTED`` failed a gate that matters, and the report says which and by how
much. That gap is the useful output: "reduce landed cost by ₹34" is a sourcing
instruction.

Nothing here executes. A ranked opportunity is an argument for a human to
approve, and the approval gate in ``part2.execution`` is the only route onward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from catalog.enums import (
    ComplianceResult,
    GateResult,
    HeadroomResult,
    IdentityConfidence,
    ScenarioName,
)
from catalog.schemas import Product
from decisions.pipeline import Pipeline, PipelineConfig, PipelineResult
from decisions.rules import Disposition
from econ.money import HUNDRED, ZERO, EconomicsDataUnavailable, format_inr, money
from signals.market import Comparable, MarketPriceReference
from sourcing.matching import (
    MatchConfidence,
    ProductGroup,
    SupplierCandidate,
    SupplierMatch,
    match_suppliers,
    rank_matched_suppliers,
)

logger = logging.getLogger(__name__)

OPPORTUNITY_VERSION = "opportunities/1.0.0"


class Verdict(StrEnum):
    RECOMMENDED = "RECOMMENDED"
    BACKUP = "BACKUP"
    REJECTED = "REJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class OpportunityPolicy:
    """Thresholds governing classification."""

    min_coverage_pct: Decimal = Decimal("45")
    min_supplier_confidence: MatchConfidence = MatchConfidence.HIGH
    min_comparables: int = 3
    target_marketplace: str = "test_marketplace"
    expected_monthly_velocity: Decimal = Decimal("10")
    capital_cap_inr: Decimal = Decimal("15000")


@dataclass
class Opportunity:
    """One ranked candidate, with everything behind the verdict."""

    group_key: str
    title: str
    verdict: Verdict
    slug: str = ""
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    recoverable: list[str] = field(default_factory=list)

    discovery_coverage_pct: Decimal = ZERO
    sources: list[str] = field(default_factory=list)
    supplier_matches: list[SupplierMatch] = field(default_factory=list)
    selected_supplier: str | None = None

    pipeline: PipelineResult | None = None
    expected_margin_pct: Decimal | None = None
    worst_margin_pct: Decimal | None = None
    expected_profit_inr: Decimal | None = None
    worst_profit_inr: Decimal | None = None
    landed_cost_inr: Decimal | None = None
    target_price_inr: Decimal | None = None

    confidence: str = "LOW"
    rank_score: Decimal | None = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def requires_approval(self) -> bool:
        """Every opportunity does. Stated explicitly so it is testable."""
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "slug": self.slug or self.group_key,
            "title": self.title,
            "verdict": str(self.verdict),
            "confidence": self.confidence,
            "rank_score": None if self.rank_score is None else str(self.rank_score),
            "discovery_coverage_pct": str(self.discovery_coverage_pct),
            "sources": list(self.sources),
            "selected_supplier": self.selected_supplier,
            "expected_margin_pct": (
                None
                if self.expected_margin_pct is None
                else str(self.expected_margin_pct)
            ),
            "worst_margin_pct": (
                None if self.worst_margin_pct is None else str(self.worst_margin_pct)
            ),
            "landed_cost_inr": (
                None if self.landed_cost_inr is None else str(self.landed_cost_inr)
            ),
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "recoverable": list(self.recoverable),
            "requires_approval": self.requires_approval,
        }


def _price_reference(
    group: ProductGroup, policy: OpportunityPolicy
) -> MarketPriceReference | None:
    """Build a price reference from observed prices across the group.

    Returns None below the minimum comparables rather than extrapolating. Two
    observations do not describe a market, and pretending otherwise puts a
    confident number in front of a purchase decision.
    """
    observations = [
        (member, money(member.observed_price_inr.value))
        for member in group.members
        if member.observed_price_inr.present
    ]
    if len(observations) < policy.min_comparables:
        return None

    return MarketPriceReference.from_comparables(
        [
            Comparable(
                marketplace=member.source,
                title=member.raw_title[:80],
                price_inr=price,
                reference=member.observed_price_inr.reference or member.external_ref,
                observed_at=member.observed_at,
            )
            for member, price in observations
        ],
        collected_by="discovery",
        notes="assembled from observed marketplace listings",
    )


def evaluate_group(
    group: ProductGroup,
    supplier_candidates: list[SupplierCandidate],
    *,
    pipeline: Pipeline,
    policy: OpportunityPolicy | None = None,
) -> Opportunity:
    """Run one candidate product through the full existing decision stack."""
    policy = policy or OpportunityPolicy()
    canonical = group.canonical
    opportunity = Opportunity(
        group_key=group.key,
        slug=group.slug,
        title=canonical.raw_title,
        verdict=Verdict.INSUFFICIENT_DATA,
        discovery_coverage_pct=canonical.coverage_pct,
        sources=sorted({m.source for m in group.members}),
    )

    if group.requires_review:
        opportunity.recoverable.append(
            "identity is uncertain: at least one observation matched only on "
            "title similarity and needs a human to confirm"
        )

    # -- supplier ---------------------------------------------------------
    matches = match_suppliers(group, supplier_candidates)
    opportunity.supplier_matches = matches
    if not matches:
        opportunity.blockers.append(
            "no supplier match: nothing in the ingested catalogues supplies this"
        )
        opportunity.verdict = Verdict.REJECTED
        opportunity.reasons.append("cannot be sourced, so cannot be sold")
        return opportunity

    _scores, selected = rank_matched_suppliers(
        matches,
        expected_monthly_velocity=policy.expected_monthly_velocity,
        capital_cap_inr=policy.capital_cap_inr,
    )
    if selected is None:
        weak = matches[0].confidence
        if not weak.is_actionable:
            opportunity.recoverable.append(
                f"supplier match is only {weak}: "
                f"{matches[0].supplier_name} matched on title similarity "
                f"({matches[0].similarity}), which is a hypothesis rather than "
                "proof they stock this item. Confirm it before ordering."
            )
        else:
            opportunity.recoverable.append(
                "every matched supplier exceeds the per-product capital cap"
            )
        opportunity.verdict = Verdict.BACKUP
        opportunity.reasons.append("sourcing is unresolved but not ruled out")
        return opportunity

    opportunity.selected_supplier = selected.supplier_name
    opportunity.landed_cost_inr = selected.landed_cost_inr
    chosen = next(m for m in matches if str(m.offer.supplier_id) == selected.supplier_id)

    # -- market price -----------------------------------------------------
    reference = _price_reference(group, policy)
    if reference is None:
        opportunity.recoverable.append(
            f"only {len(group.observed_prices)} observed price(s); "
            f"{policy.min_comparables} comparables are needed before economics "
            "can be tested against the market"
        )

    target_price = (
        money(reference.median_inr)
        if reference is not None
        else (money(group.observed_prices[0]) if group.observed_prices else None)
    )
    if target_price is None:
        opportunity.blockers.append(
            "no observed price anywhere: nothing to price against"
        )
        opportunity.verdict = Verdict.INSUFFICIENT_DATA
        return opportunity
    opportunity.target_price_inr = target_price

    # -- the existing Part 1 pipeline does all the economics ---------------
    product = Product(
        canonical_name=canonical.raw_title[:120],
        category=canonical.category or "general",
        hsn_code=canonical.hsn_code,
        brand=canonical.brand,
        weight_g=canonical.weight_g or 500,
        identity_confidence=(
            IdentityConfidence.HIGH
            if group.confidence.is_actionable and not group.requires_review
            else IdentityConfidence.MEDIUM
        ),
    )
    try:
        result = pipeline.run(
            product=product,
            offers=[chosen.offer],
            sku_code=f"SKU-{group.key[:24]}",
            config=PipelineConfig(
                marketplace=policy.target_marketplace,
                selling_price_inr=target_price,
                expected_monthly_velocity=policy.expected_monthly_velocity,
            ),
            signals=canonical.to_signals(),
            price_reference=reference,
        )
    except EconomicsDataUnavailable as exc:
        # A required cost input is absent from the reference data. Producing a
        # number without it would overstate profit, and crashing would take
        # the whole ranking down over one product, so this degrades to
        # INSUFFICIENT_DATA with the missing input named.
        logger.info(
            "economics unavailable", extra={"group": group.key, "error": str(exc)}
        )
        opportunity.verdict = Verdict.INSUFFICIENT_DATA
        opportunity.blockers.append(f"economics unavailable: {exc}")
        opportunity.reasons.append(
            "required cost data is missing, so no profitability can be stated"
        )
        return _score(opportunity)

    opportunity.pipeline = result
    _absorb_pipeline(opportunity, result)

    return classify(opportunity, group, policy)


def _absorb_pipeline(opportunity: Opportunity, result: PipelineResult) -> None:
    """Copy the numbers the report needs out of the pipeline result."""
    expected = result.economics.get(ScenarioName.EXPECTED)
    worst = result.economics.get(ScenarioName.WORST)
    if expected:
        opportunity.expected_margin_pct = expected.net_margin_pct
        opportunity.expected_profit_inr = expected.net_profit_inr
        opportunity.landed_cost_inr = expected.landed_cost_inr
    if worst:
        opportunity.worst_margin_pct = worst.net_margin_pct
        opportunity.worst_profit_inr = worst.net_profit_inr


def classify(
    opportunity: Opportunity, group: ProductGroup, policy: OpportunityPolicy
) -> Opportunity:
    """Turn a pipeline outcome into a verdict, with reasons.

    Hard failures reject. Recoverable gaps demote to BACKUP rather than
    discarding, because a product blocked only by missing comparables is a
    research task, not a dead end.
    """
    result = opportunity.pipeline
    blocking = result.blocked_by if result else None

    # --- hard gates, checked independently of the decision findings --------
    # These duplicate conditions the rules layer also detects, deliberately.
    # Relying on `blocked_by` alone made a single upstream omission enough to
    # let a non-compliant or uncosted product reach RECOMMENDED, and that is
    # too much weight for one code path to carry.
    compliance = getattr(result, "compliance", None) if result else None
    compliance_state = getattr(compliance, "overall", None)

    if compliance_state is ComplianceResult.FAIL:
        opportunity.verdict = Verdict.REJECTED
        # Name the requirements that failed. A generic "compliance failed" tells
        # an operator nothing about what to obtain.
        failures = list(getattr(compliance, "blocking_failures", []) or [])
        detail = ", ".join(failures) if failures else "see the compliance report"
        opportunity.blockers.append(
            f"compliance.fail: blocking requirements unmet: {detail}"
        )
        return _score(opportunity)

    if compliance_state in (
        ComplianceResult.UNKNOWN,
        ComplianceResult.REQUIRES_HUMAN_REVIEW,
    ):
        # Not a rejection -- the product may well be fine -- but unresolved
        # compliance can never be a recommendation.
        conflicts = list(getattr(compliance, "conflicts", []) or [])
        reason = f"; conflicting evidence: {conflicts[0]}" if conflicts else ""
        opportunity.recoverable.append(
            f"compliance is {compliance_state}: it must be resolved by a human "
            f"before this can be recommended{reason}"
        )

    if result and result.gate and result.gate.result is GateResult.FAIL:
        opportunity.verdict = Verdict.REJECTED
        gap = result.gate.gap_to_pass.get("action", "")
        opportunity.blockers.append(
            f"failed {result.gate.failed_gate}" + (f": {gap}" if gap else "")
        )
        opportunity.reasons.append(gap or "economics do not clear the floor")
        return _score(opportunity)

    if result and result.headroom and result.headroom.result is HeadroomResult.BLOCK:
        opportunity.verdict = Verdict.REJECTED
        opportunity.blockers.append(result.headroom.explanation)
        opportunity.reasons.append("cannot be sold profitably at market prices")
        return _score(opportunity)

    if blocking and blocking.disposition in (Disposition.BLOCKED, Disposition.REJECTED):
        opportunity.verdict = Verdict.REJECTED
        opportunity.blockers.append(f"{blocking.rule_id}: {blocking.reason[:120]}")
        return _score(opportunity)

    if blocking and blocking.disposition is Disposition.REQUIRES_HUMAN_REVIEW:
        opportunity.recoverable.append(f"{blocking.rule_id}: {blocking.reason[:120]}")

    # No economics means no profitability claim. A verdict of RECOMMENDED
    # asserts the numbers clear the gates; without numbers there is nothing to
    # clear, so this is INSUFFICIENT_DATA rather than a pass by omission.
    if opportunity.worst_margin_pct is None or opportunity.expected_margin_pct is None:
        opportunity.verdict = Verdict.INSUFFICIENT_DATA
        opportunity.blockers.append(
            "no economics were computed, so no profitability can be asserted"
        )
        return _score(opportunity)

    if opportunity.selected_supplier is None:
        # Reached when a supplier matched but none was usable. A probable
        # match is not a confirmed source of supply.
        opportunity.recoverable.append(
            "no usable supplier was confirmed, so there is no verified cost "
            "basis for a recommendation"
        )

    if opportunity.discovery_coverage_pct < policy.min_coverage_pct:
        opportunity.recoverable.append(
            f"discovery coverage {opportunity.discovery_coverage_pct}% is below the "
            f"{policy.min_coverage_pct}% floor; missing "
            f"{', '.join(group.canonical.missing_signals[:4])}"
        )

    if opportunity.recoverable:
        opportunity.verdict = Verdict.BACKUP
        opportunity.reasons.append(
            "clears the economics but rests on evidence that can be strengthened"
        )
        opportunity.confidence = "MEDIUM"
        return _score(opportunity)

    opportunity.verdict = Verdict.RECOMMENDED
    opportunity.confidence = "HIGH"
    opportunity.reasons.append(
        f"clears every gate; expected margin {opportunity.expected_margin_pct}% "
        f"with worst case {opportunity.worst_margin_pct}%"
    )
    return _score(opportunity)


def _score(opportunity: Opportunity) -> Opportunity:
    """Rank score: worst-case profit, discounted by evidence coverage.

    Deliberately built on the *worst* case. Ranking on expected profit puts
    the most optimistic assumptions at the top of the list, which is precisely
    the wrong order for deciding where to put capital.
    """
    if opportunity.worst_profit_inr is None:
        opportunity.rank_score = None
        return opportunity

    coverage_factor = money(opportunity.discovery_coverage_pct / HUNDRED)
    penalty = {
        Verdict.RECOMMENDED: Decimal("1.0"),
        Verdict.BACKUP: Decimal("0.6"),
        Verdict.REJECTED: Decimal("0"),
        Verdict.INSUFFICIENT_DATA: Decimal("0"),
    }[opportunity.verdict]
    opportunity.rank_score = money(
        opportunity.worst_profit_inr * coverage_factor * penalty
    ).quantize(Decimal("0.01"))
    return opportunity


def rank_opportunities(opportunities: list[Opportunity]) -> list[Opportunity]:
    """Order by verdict, then by risk-adjusted worst-case score."""
    order = {
        Verdict.RECOMMENDED: 0,
        Verdict.BACKUP: 1,
        Verdict.INSUFFICIENT_DATA: 2,
        Verdict.REJECTED: 3,
    }
    return sorted(
        opportunities,
        key=lambda o: (
            order[o.verdict],
            -(o.rank_score if o.rank_score is not None else Decimal("-99999")),
        ),
    )


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def opportunity_report(opportunities: list[Opportunity]) -> str:
    """The ranked list, with coverage and worst case always visible."""
    if not opportunities:
        return "No opportunities evaluated.\n"

    tally: dict[str, int] = {}
    for item in opportunities:
        tally[str(item.verdict)] = tally.get(str(item.verdict), 0) + 1

    lines = [
        "OPPORTUNITIES  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())),
        "",
        f"{'Product':<34} {'verdict':<18} {'expect':>8} {'worst':>8} "
        f"{'landed':>11} {'cov':>6} {'supplier'}",
        f"{'-' * 34} {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 11} {'-' * 6} {'-' * 16}",
    ]
    for item in opportunities:
        expected = (
            "n/a"
            if item.expected_margin_pct is None
            else f"{item.expected_margin_pct:.1f}%"
        )
        worst = (
            "n/a" if item.worst_margin_pct is None else f"{item.worst_margin_pct:.1f}%"
        )
        landed = (
            "n/a" if item.landed_cost_inr is None else format_inr(item.landed_cost_inr)
        )
        lines.append(
            f"{item.title[:33]:<34} {item.verdict!s:<18} {expected:>8} {worst:>8} "
            f"{landed:>11} {item.discovery_coverage_pct:>5.0f}% "
            f"{(item.selected_supplier or '-')[:16]}"
        )
    lines += [
        "",
        "Ranked on worst-case profit discounted by evidence coverage, not on",
        "expected profit. Every entry requires human approval before execution.",
        "",
    ]
    return "\n".join(lines)


def recommendation_report(opportunities: list[Opportunity]) -> str:
    """The detail a human needs in order to approve or decline."""
    shortlist = [
        o for o in opportunities if o.verdict in (Verdict.RECOMMENDED, Verdict.BACKUP)
    ]
    if not shortlist:
        return "Nothing reached RECOMMENDED or BACKUP.\n"

    lines = [f"FOR APPROVAL - {len(shortlist)} product(s)", ""]
    for item in shortlist:
        lines += [
            f"  {item.title[:70]}",
            f"    VERDICT        {item.verdict}   (confidence {item.confidence})",
            f"    SOURCE         observed on {', '.join(item.sources)}",
            f"    SUPPLIER       {item.selected_supplier or 'unresolved'}",
            f"    LANDED COST    {format_inr(item.landed_cost_inr) if item.landed_cost_inr else 'n/a'}",
            f"    TARGET PRICE   {format_inr(item.target_price_inr) if item.target_price_inr else 'n/a'}",
            f"    EXPECTED       {item.expected_margin_pct}% "
            f"({format_inr(item.expected_profit_inr) if item.expected_profit_inr else 'n/a'}/order)",
            f"    WORST CASE     {item.worst_margin_pct}% "
            f"({format_inr(item.worst_profit_inr) if item.worst_profit_inr else 'n/a'}/order)",
            f"    COVERAGE       {item.discovery_coverage_pct}%",
        ]
        for note in item.recoverable:
            lines.append(f"    TO STRENGTHEN  {note[:80]}")
        for note in item.blockers:
            lines.append(f"    BLOCKER        {note[:80]}")
        lines.append("")
    lines += [
        "  Nothing here is executed. Approve an action explicitly to proceed;",
        "  the system cannot purchase, publish or pay on its own.",
        "",
    ]
    return "\n".join(lines)


def rejection_report(opportunities: list[Opportunity]) -> str:
    """Why each rejected product failed, and what would change it."""
    rejected = [o for o in opportunities if o.verdict is Verdict.REJECTED]
    if not rejected:
        return "No products were rejected.\n"
    lines = [
        f"REJECTED - {len(rejected)} product(s)",
        "",
        f"{'Product':<34} {'Blocker'}",
        f"{'-' * 34} {'-' * 60}",
    ]
    for item in rejected:
        blocker = (
            item.blockers[0]
            if item.blockers
            else (item.reasons[0] if item.reasons else "")
        )
        lines.append(f"{item.title[:33]:<34} {blocker[:70]}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "OPPORTUNITY_VERSION",
    "Opportunity",
    "OpportunityPolicy",
    "Verdict",
    "classify",
    "evaluate_group",
    "opportunity_report",
    "rank_opportunities",
    "recommendation_report",
    "rejection_report",
]
