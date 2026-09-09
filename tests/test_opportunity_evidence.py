"""Phase 2E: missing evidence must never reach RECOMMENDED.

The hole this closes: `classify` decided verdicts from the decision-rules
finding alone. Compliance state and computed economics were never checked
independently, so a single upstream omission was enough for a non-compliant or
entirely uncosted product to be recommended. These gates now duplicate
conditions the rules layer also detects -- deliberately, because one code path
was carrying too much weight.
"""

from __future__ import annotations

import collections
from decimal import Decimal

import pytest

from catalog.enums import ComplianceResult
from sourcing.matching import MatchConfidence, ProductGroup
from sourcing.model import DiscoveredProduct, FieldSource, Observed
from sourcing.opportunities import (
    Opportunity,
    OpportunityPolicy,
    Verdict,
    _score,
    classify,
    rank_opportunities,
)


class _Compliance:
    def __init__(self, overall, failures=(), conflicts=()):
        self.overall = overall
        self.blocking_failures = list(failures)
        self.conflicts = tuple(conflicts)


class _Result:
    """A pipeline result with no decision findings at all.

    Deliberately empty: it isolates the hard gates from the rules layer, which
    is exactly the single-point-of-failure being tested.
    """

    def __init__(self, compliance):
        self.gate = None
        self.headroom = None
        self.blocked_by = None
        self.compliance = compliance
        self.economics = {}
        self.decisions = []
        self.score = None


def _group() -> ProductGroup:
    product = DiscoveredProduct(source="s", external_ref="r", raw_title="Widget")
    product.observed_price_inr = Observed(Decimal("500"), FieldSource.MANUAL, "ref")
    for name in (
        "search_interest",
        "trend",
        "review_count",
        "review_velocity",
        "competitor_count",
        "brand_dominance",
        "rank",
        "availability",
    ):
        setattr(product, name, Observed(Decimal("50"), FieldSource.MANUAL, "ref"))
    return ProductGroup(key="k", members=[product], confidence=MatchConfidence.EXACT)


def verdict_for(
    compliance=ComplianceResult.PASS,
    *,
    economics=True,
    supplier="Acme Traders",
    failures=(),
    conflicts=(),
) -> Opportunity:
    opportunity = Opportunity(
        group_key="k",
        title="Widget",
        verdict=Verdict.INSUFFICIENT_DATA,
        discovery_coverage_pct=Decimal("100"),
    )
    opportunity.pipeline = _Result(_Compliance(compliance, failures, conflicts))
    opportunity.selected_supplier = supplier
    if economics:
        opportunity.expected_margin_pct = Decimal("30")
        opportunity.worst_margin_pct = Decimal("20")
        opportunity.worst_profit_inr = Decimal("100")
    return classify(opportunity, _group(), OpportunityPolicy())


# ==========================================================================
# Compliance can never be bypassed
# ==========================================================================


def test_compliance_fail_cannot_be_recommended():
    outcome = verdict_for(ComplianceResult.FAIL, failures=["BIS_ISI_SMALL_APPLIANCE"])
    assert outcome.verdict is Verdict.REJECTED
    assert "BIS_ISI_SMALL_APPLIANCE" in outcome.blockers[0]


def test_compliance_unknown_cannot_be_recommended():
    outcome = verdict_for(ComplianceResult.UNKNOWN)
    assert outcome.verdict is not Verdict.RECOMMENDED
    assert any("compliance is" in r for r in outcome.recoverable)


def test_compliance_requires_review_cannot_be_recommended():
    outcome = verdict_for(ComplianceResult.REQUIRES_HUMAN_REVIEW)
    assert outcome.verdict is not Verdict.RECOMMENDED


def test_a_compliance_conflict_is_surfaced_in_the_reason():
    outcome = verdict_for(
        ComplianceResult.REQUIRES_HUMAN_REVIEW,
        conflicts=("bis_certificate_no: sources disagree",),
    )
    assert any("conflicting evidence" in r for r in outcome.recoverable)


def test_compliance_pass_with_everything_present_is_recommended():
    """The gates must not block products that genuinely clear them."""
    assert verdict_for(ComplianceResult.PASS).verdict is Verdict.RECOMMENDED


def test_a_blocked_verdict_names_the_blocking_condition():
    outcome = verdict_for(ComplianceResult.FAIL, failures=["BIS_ISI", "LABEL_RULES"])
    assert "BIS_ISI" in outcome.blockers[0]
    assert "LABEL_RULES" in outcome.blockers[0]


# ==========================================================================
# Economics
# ==========================================================================


def test_missing_economics_cannot_be_recommended():
    outcome = verdict_for(economics=False)
    assert outcome.verdict is Verdict.INSUFFICIENT_DATA
    assert any("no economics" in b for b in outcome.blockers)


def test_missing_economics_produces_no_numeric_profitability():
    outcome = verdict_for(economics=False)
    assert outcome.expected_margin_pct is None
    assert outcome.worst_margin_pct is None


def test_missing_economics_scores_nothing():
    """No fabricated rank from absent numbers."""
    outcome = verdict_for(economics=False)
    assert outcome.rank_score in (None, 0)


# ==========================================================================
# Supplier
# ==========================================================================


def test_no_confirmed_supplier_cannot_be_recommended():
    outcome = verdict_for(supplier=None)
    assert outcome.verdict is not Verdict.RECOMMENDED
    assert any("no usable supplier" in r for r in outcome.recoverable)


def test_a_probable_match_is_not_a_confirmed_supplier():
    """Regression on the known LED case, at unit level."""
    from sourcing.feeds import build_registry, collect
    from sourcing.matching import group_products, match_suppliers, rank_matched_suppliers
    from sourcing.supplier_ingest import ingest_suppliers

    products, _ = collect(build_registry())
    candidates = ingest_suppliers("data/fixtures/discovery/suppliers.csv").accepted
    led = next(
        g for g in group_products(products) if "led strip light 5m" in g.key.lower()
    )
    matches = match_suppliers(led, candidates)
    scores, selected = rank_matched_suppliers(matches)

    assert matches and str(matches[0].confidence) == "PROBABLE"
    assert not matches[0].usable
    assert scores == [] and selected is None


# ==========================================================================
# Ranking
# ==========================================================================


def test_a_missing_score_gains_no_ranking_advantage():
    """An unscoreable product must sort last, not first."""

    def make(key, worst, coverage="100"):
        item = Opportunity(
            group_key=key,
            title=key,
            verdict=Verdict.RECOMMENDED,
            discovery_coverage_pct=Decimal(coverage),
        )
        item.worst_profit_inr = None if worst is None else Decimal(worst)
        return _score(item)

    ranked = rank_opportunities(
        [make("none", None), make("low", "10"), make("high", "300")]
    )
    assert [r.group_key for r in ranked] == ["high", "low", "none"]
    assert ranked[-1].rank_score is None


def test_ranking_is_deterministic():
    def build():
        return [
            verdict_for(ComplianceResult.PASS),
            verdict_for(ComplianceResult.FAIL, failures=["X"]),
            verdict_for(economics=False),
        ]

    first = [
        (o.group_key, str(o.verdict), o.rank_score) for o in rank_opportunities(build())
    ]
    second = [
        (o.group_key, str(o.verdict), o.rank_score) for o in rank_opportunities(build())
    ]
    assert first == second


def test_ranking_never_places_a_rejected_product_above_a_recommended_one():
    """Ranking cannot override a hard gate."""
    recommended = verdict_for(ComplianceResult.PASS)
    rejected = verdict_for(ComplianceResult.FAIL, failures=["X"])
    rejected.worst_profit_inr = Decimal("99999")
    ranked = rank_opportunities([_score(rejected), recommended])
    assert ranked[0].verdict is Verdict.RECOMMENDED


def test_a_rejected_product_scores_zero_however_profitable():
    item = Opportunity(
        group_key="x",
        title="x",
        verdict=Verdict.REJECTED,
        discovery_coverage_pct=Decimal("100"),
    )
    item.worst_profit_inr = Decimal("50000")
    assert _score(item).rank_score == 0


# ==========================================================================
# Pipeline robustness and unchanged outcomes
# ==========================================================================


def _run_pipeline():
    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import evaluate_group

    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())
    return rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )


def test_one_unevaluable_product_does_not_crash_the_ranking():
    """Verified by the economics-unavailable path in 2C, re-asserted here."""
    opportunities = _run_pipeline()
    assert len(opportunities) == 7


def test_existing_outcomes_are_unchanged():
    opportunities = _run_pipeline()
    tally = collections.Counter(str(o.verdict) for o in opportunities)
    assert tally == {"BACKUP": 1, "REJECTED": 6}

    by_title = {o.title: o for o in opportunities}
    kemei = next(o for t, o in by_title.items() if "KX880" in t)
    assert kemei.verdict is Verdict.REJECTED
    assert "BIS" in kemei.blockers[0]

    lunchbox = next(o for t, o in by_title.items() if "Lunchbox" in t)
    assert lunchbox.verdict is Verdict.REJECTED
    assert "worst_case" in lunchbox.blockers[0]

    led = next(o for t, o in by_title.items() if "LED Strip Light 5m" in t)
    assert led.verdict is Verdict.BACKUP
    assert led.selected_supplier is None
    assert "PROBABLE" in led.recoverable[0]

    generics = [
        o
        for t, o in by_title.items()
        if o.selected_supplier is None and "no supplier match" in " ".join(o.blockers)
    ]
    assert generics
    assert all(o.verdict is not Verdict.RECOMMENDED for o in generics)


def test_every_opportunity_still_requires_human_approval():
    for opportunity in _run_pipeline():
        assert opportunity.requires_approval


@pytest.mark.parametrize(
    "term", ["purchase", "payment", "publish", "transfer", "fulfil", "checkout"]
)
def test_opportunity_intelligence_has_no_execution_capability(term: str):
    from pathlib import Path

    source = Path("sourcing/opportunities.py").read_text(encoding="utf-8").lower()
    assert f"def {term}" not in source
