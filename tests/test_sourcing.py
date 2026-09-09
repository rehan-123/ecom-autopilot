"""Tests for the discovery, matching and opportunity-ranking layer.

The load-bearing assertions are about *refusal*: that a product with no
supplier cannot be recommended, that similarity alone never merges two
records, that an unavailable feed produces no candidates rather than plausible
ones, and that nothing here can reach a marketplace without an approval.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catalog.enums import AIService, SignalStatus
from catalog.schemas import SupplierOffer, new_id
from decisions.pipeline import Pipeline
from llm.services import LLMService, RecordedLLMClient
from reference.loader import ReferenceStore
from sourcing.feeds import (
    CsvObservationFeed,
    FeedError,
    SellerApiFeed,
    TrendsExportFeed,
    UnavailableFeed,
    build_registry,
    collect,
    coverage_report,
    load_supplier_candidates,
)
from sourcing.matching import (
    PROBABLE_THRESHOLD,
    MatchConfidence,
    ProductGroup,
    SupplierCandidate,
    compare,
    consult_identity_judge,
    group_products,
    jaccard,
    match_suppliers,
    rank_matched_suppliers,
)
from sourcing.model import (
    COVERAGE_WEIGHTS,
    SIGNAL_FIELDS,
    DiscoveredProduct,
    FieldSource,
    Observed,
    enrich,
    extract_gtin,
    extract_model_code,
    guess_brand,
    normalize_model_code,
    normalize_title,
    tokenize,
)
from sourcing.opportunities import (
    Opportunity,
    OpportunityPolicy,
    Verdict,
    evaluate_group,
    opportunity_report,
    rank_opportunities,
    recommendation_report,
    rejection_report,
)

FIXTURES = Path("data/fixtures/discovery")
REFERENCE = Path("tests/fixtures/reference")


def a_product(title: str, price: str | None = "1299", **kwargs) -> DiscoveredProduct:
    base = {
        "source": "manual_research",
        "external_ref": f"ref://{abs(hash(title)) % 10000}",
        "raw_title": title,
        "category": "personal_care_electrical",
        "hsn_code": "8510",
        "weight_g": 310,
    }
    base.update(kwargs)
    product = DiscoveredProduct(**base)
    if price is not None:
        product.observed_price_inr = Observed(
            Decimal(price), FieldSource.MANUAL, "ref://x"
        )
    return enrich(product)


def a_candidate(
    name: str,
    title: str,
    cost: str,
    *,
    moq: int = 20,
    invoice: bool = True,
    gtin: str | None = None,
) -> SupplierCandidate:
    return SupplierCandidate(
        supplier_name=name,
        title=title,
        gtin=gtin,
        model_code=extract_model_code(title),
        offer=SupplierOffer(
            supplier_id=new_id(),
            product_id=new_id(),
            unit_cost_inr=Decimal(cost),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=invoice,
            moq_units=moq,
            inbound_freight_inr=Decimal("10"),
        ),
    )


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline(ReferenceStore(REFERENCE).load())


# ==========================================================================
# Normalization
# ==========================================================================


def test_normalization_strips_marketing_noise():
    tokens = set(tokenize("New Original Premium Rechargeable Hair Trimmer"))
    assert "trimmer" in tokens
    assert "premium" not in tokens
    assert "new" not in tokens


def test_model_code_survives_spacing_and_hyphens():
    """KX-880, KX 880 and KX880 are one product, not three."""
    assert extract_model_code("Kemei KX-880 Trimmer") == "KX880"
    assert extract_model_code("Kemei KX 880 Trimmer") == "KX880"
    assert extract_model_code("Rechargeable KX880 Clipper") == "KX880"


def test_supplied_model_code_is_normalized_like_an_extracted_one():
    """A catalogue writing ML-750 must match a title saying ML750.

    When only extraction normalised, these produced two half-evidenced
    products where there was one well-evidenced product.
    """
    assert normalize_model_code("ML-750") == "ML750"
    assert normalize_model_code("ml 750") == "ML750"
    supplied = a_product("Steel Lunchbox 750ml", model_code="ML-750")
    derived = a_product("Milton ML750 Steel Lunch Box")
    assert supplied.model_code == derived.model_code


def test_a_unit_reading_is_not_a_model_code():
    assert extract_model_code("Steel Lunchbox 750ml Insulated") != "750ML"


def test_gtin_is_extracted_when_present():
    assert extract_gtin("Trimmer EAN 8901234567890") == "8901234567890"
    assert extract_gtin("Trimmer with 5 combs") is None


def test_brand_is_never_guessed_from_an_unknown_word():
    """Guessing the first token is the brand produces confident nonsense."""
    assert guess_brand("Rechargeable Cordless Trimmer") is None
    assert guess_brand("Kemei KX-880 Trimmer", frozenset({"Kemei"})) == "Kemei"


def test_normalization_is_accent_and_case_insensitive():
    assert normalize_title("Café Máquina") == "cafe maquina"


# ==========================================================================
# The discovery model
# ==========================================================================


def test_unavailable_field_cannot_carry_a_value():
    with pytest.raises(ValueError, match="cannot carry a value"):
        Observed(Decimal("5"), FieldSource.UNAVAILABLE)


def test_coverage_reflects_what_was_actually_obtained():
    bare = a_product("Trimmer", price=None)
    assert bare.coverage_pct == 0

    priced = a_product("Trimmer", price="1299")
    assert priced.coverage_pct > bare.coverage_pct
    assert "search_interest" in priced.missing_signals


def test_coverage_weights_cover_every_signal_field():
    assert set(COVERAGE_WEIGHTS) == set(SIGNAL_FIELDS)


def test_missing_signals_project_as_data_unavailable():
    """An absent signal must reach the scorer as absent, never as zero."""
    signals = a_product("Trimmer").to_signals()
    assert signals["SEARCH_INTEREST"].status is SignalStatus.DATA_UNAVAILABLE
    assert signals["SEARCH_INTEREST"].value_numeric is None


def test_present_signals_project_as_observed():
    product = a_product("Trimmer")
    product.search_interest = Observed(
        Decimal("58"), FieldSource.PUBLIC_DATASET, "trends"
    )
    signals = product.to_signals()
    assert signals["SEARCH_INTEREST"].status is SignalStatus.OBSERVED
    assert signals["SEARCH_INTEREST"].value_numeric == Decimal("58")


# ==========================================================================
# Matching
# ==========================================================================


def test_shared_gtin_is_conclusive():
    left = a_product("Trimmer A", gtin="8901234567890")
    right = a_product("Completely Different Words", gtin="8901234567890")
    assert compare(left, right).confidence is MatchConfidence.EXACT


def test_different_gtins_prove_they_are_not_the_same():
    """Positive evidence against, not merely absence of evidence for."""
    left = a_product("Kemei KX-880 Trimmer", gtin="8901234567890")
    right = a_product("Kemei KX-880 Trimmer", gtin="8909999999999")
    assert compare(left, right).confidence is MatchConfidence.NONE


def test_shared_model_code_is_a_high_confidence_match():
    left = a_product("Kemei KX-880 Professional Trimmer")
    right = a_product("Rechargeable KX880 Clipper Cordless")
    match = compare(left, right)
    assert match.confidence is MatchConfidence.HIGH
    assert match.confidence.is_actionable


def test_same_model_different_brand_needs_a_human():
    left = a_product("KX-880 Trimmer", brand="Kemei")
    right = a_product("KX-880 Trimmer", brand="Nova")
    match = compare(left, right)
    assert match.requires_review
    assert not match.confidence.is_actionable


def test_similarity_alone_never_merges():
    """A hypothesis about identity must not become a purchase order."""
    left = a_product("Blue Cotton Round Neck T Shirt Large")
    right = a_product("Blue Cotton Round Neck T Shirt Medium")
    match = compare(left, right)
    # Two sizes of one garment overlap heavily but are different SKUs.
    assert match.similarity >= PROBABLE_THRESHOLD
    assert not match.confidence.is_actionable
    assert match.requires_review


def test_unrelated_products_do_not_match():
    assert compare(a_product("Hair Trimmer"), a_product("Steel Lunchbox")).confidence is (
        MatchConfidence.NONE
    )


def test_grouping_merges_only_actionable_matches():
    products = [
        a_product("Kemei KX-880 Professional Trimmer"),
        a_product("Rechargeable KX880 Clipper"),
        a_product("Steel Lunchbox 750ml"),
    ]
    groups = group_products(products)
    assert len(groups) == 2
    assert max(len(g.members) for g in groups) == 2


def test_group_canonical_is_the_best_evidenced_member():
    thin = a_product("KX-880 Trimmer", price=None)
    rich = a_product("Kemei KX-880 Trimmer", price="1299")
    rich.search_interest = Observed(Decimal("60"), FieldSource.PUBLIC_DATASET, "t")
    group = group_products([thin, rich])[0]
    assert group.canonical is rich


def test_jaccard_is_bounded():
    assert jaccard(frozenset(), frozenset({"a"})) == 0
    assert jaccard(frozenset({"a"}), frozenset({"a"})) == 1


# ==========================================================================
# LLM assistance is advisory only
# ==========================================================================


def _judge(payload: dict) -> LLMService:
    return LLMService(
        AIService.JUDGE_IDENTITY,
        RecordedLLMClient({"JUDGE_IDENTITY": json.dumps(payload)}),
    )


def test_identity_judge_findings_do_not_change_confidence():
    """Claude advises; the confidence band is set by deterministic evidence."""
    left = a_product("Blue Cotton Round Neck T Shirt Large")
    right = a_product("Blue Cotton Round Neck T Shirt Medium")
    before = compare(left, right)
    assert before.requires_review
    judge = _judge(
        {
            "findings": [
                {"flag": "SAME_PRODUCT", "confidence": "high", "evidence_refs": []}
            ],
            "unresolved_questions": [],
        }
    )
    after = consult_identity_judge(left, right, before, judge)
    assert after.confidence is before.confidence
    assert after.requires_review
    assert after.findings


def test_unusable_judge_response_records_insufficient_evidence():
    left = a_product("Blue Cotton Round Neck T Shirt Large")
    right = a_product("Blue Cotton Round Neck T Shirt Medium")
    judge = LLMService(
        AIService.JUDGE_IDENTITY, RecordedLLMClient({"JUDGE_IDENTITY": "garbage"})
    )
    after = consult_identity_judge(left, right, compare(left, right), judge)
    assert after.findings[0]["flag"] == "INSUFFICIENT_EVIDENCE"


def test_judge_cannot_cause_a_merge():
    """Even an enthusiastic judge cannot merge two weakly-related records."""
    products = [
        a_product("Blue Cotton Round Neck T Shirt Large"),
        a_product("Blue Cotton Round Neck T Shirt Medium"),
    ]
    judge = _judge(
        {
            "findings": [
                {"flag": "SAME_PRODUCT", "confidence": "high", "evidence_refs": []}
            ],
            "unresolved_questions": [],
        }
    )
    assert len(group_products(products, judge=judge)) == 2


# ==========================================================================
# Supplier matching
# ==========================================================================


def _group(*products: DiscoveredProduct) -> ProductGroup:
    return group_products(list(products))[0]


def test_supplier_matches_on_model_code():
    group = _group(a_product("Kemei KX-880 Trimmer"))
    matches = match_suppliers(
        group, [a_candidate("Shenzhen Direct", "Kemei KX-880 Trimmer Cordless", "268")]
    )
    assert matches[0].confidence is MatchConfidence.HIGH
    assert matches[0].usable


def test_weak_supplier_match_is_not_usable():
    group = _group(a_product("Kemei KX-880 Trimmer"))
    matches = match_suppliers(group, [a_candidate("Random", "Kemei Trimmer", "200")])
    assert matches
    assert all(not m.usable for m in matches)


def test_unrelated_supplier_is_not_matched_at_all():
    group = _group(a_product("Kemei KX-880 Trimmer"))
    assert (
        match_suppliers(group, [a_candidate("Kolkata", "LED Strip 5m RGB", "240")]) == []
    )


def test_supplier_ranking_reuses_the_existing_engine():
    """No second definition of what a supplier costs.

    The input tax credit is worth real money but a finite amount: a large
    enough discount outweighs it. At a modest gap the invoice wins; the point
    is that landed cost decides, not the sticker price.
    """
    group = _group(a_product("Kemei KX-880 Trimmer"))
    matches = match_suppliers(
        group,
        [
            a_candidate("Cheap No Invoice", "Kemei KX-880 Trimmer", "250", invoice=False),
            a_candidate(
                "Dearer With Invoice", "Kemei KX-880 Trimmer", "268", invoice=True
            ),
        ],
    )
    scores, selected = rank_matched_suppliers(matches, capital_cap_inr=Decimal("100000"))
    assert selected is not None
    cheap = next(s for s in scores if not s.tax_invoice)
    assert cheap.unit_cost_inr < selected.unit_cost_inr
    assert cheap.landed_cost_inr > selected.landed_cost_inr


def test_a_large_enough_discount_outweighs_the_tax_invoice():
    """Stated because it is counterintuitive and the engine is right.

    At Rs 220 against Rs 268 the discount exceeds the recoverable credit, so
    the no-invoice supplier genuinely lands cheaper. The system must not
    pretend otherwise.
    """
    group = _group(a_product("Kemei KX-880 Trimmer"))
    matches = match_suppliers(
        group,
        [
            a_candidate("Deep Discount", "Kemei KX-880 Trimmer", "220", invoice=False),
            a_candidate("With Invoice", "Kemei KX-880 Trimmer", "268", invoice=True),
        ],
    )
    scores, _selected = rank_matched_suppliers(matches, capital_cap_inr=Decimal("100000"))
    cheap = next(s for s in scores if not s.tax_invoice)
    dearer = next(s for s in scores if s.tax_invoice)
    assert cheap.landed_cost_inr < dearer.landed_cost_inr


def test_no_usable_matches_selects_nothing():
    group = _group(a_product("Kemei KX-880 Trimmer"))
    matches = match_suppliers(group, [a_candidate("X", "Kemei Trimmer", "200")])
    scores, selected = rank_matched_suppliers(matches)
    assert scores == []
    assert selected is None


# ==========================================================================
# Feeds
# ==========================================================================


def test_unavailable_feed_yields_no_candidates_and_says_why():
    result = UnavailableFeed("meesho", "no public API").fetch()
    assert result.products == []
    assert not result.available
    assert "no public API" in result.unavailable_reason


def test_seller_api_feed_reports_missing_credentials():
    result = SellerApiFeed("amazon_in", environ={}).fetch()
    assert not result.available
    assert "credentials are not configured" in result.unavailable_reason


def test_seller_api_feed_with_credentials_is_honest_about_transport():
    result = SellerApiFeed(
        "amazon_in",
        environ={
            "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_ID": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER",
        },
    ).fetch()
    assert not result.available
    assert "no transport is bundled" in result.unavailable_reason


def test_csv_feed_records_missing_columns_as_unavailable(tmp_path: Path):
    """A column that is absent is absent, not zero."""
    path = tmp_path / "obs.csv"
    path.write_text("title,price\nHair Trimmer,1299\n", encoding="utf-8")
    product = CsvObservationFeed(path).fetch().products[0]
    assert product.observed_price_inr.present
    assert not product.search_interest.present
    assert product.search_interest.source is FieldSource.UNAVAILABLE


def test_csv_feed_handles_rupee_and_thousands_separator(tmp_path: Path):
    path = tmp_path / "obs.csv"
    path.write_text('title,price\nTrimmer,"\u20b9 1,299.00"\n', encoding="utf-8")
    product = CsvObservationFeed(path).fetch().products[0]
    assert product.observed_price_inr.value == Decimal("1299.00")


def test_csv_feed_skips_rows_without_a_title(tmp_path: Path):
    path = tmp_path / "obs.csv"
    path.write_text("title,price\n,1299\nTrimmer,1299\n", encoding="utf-8")
    result = CsvObservationFeed(path).fetch()
    assert len(result.products) == 1
    assert any("no title" in w for w in result.warnings)


def test_missing_feed_file_raises_a_clear_error(tmp_path: Path):
    with pytest.raises(FeedError, match="not found"):
        CsvObservationFeed(tmp_path / "absent.csv").fetch()


def test_trends_feed_never_carries_a_price():
    """Trends supplies interest. Treating it as price would be fabrication."""
    result = TrendsExportFeed(FIXTURES / "trends.csv").fetch()
    assert result.products
    for product in result.products:
        assert not product.observed_price_inr.present
        assert "never price" in product.observed_price_inr.reason


def test_registry_registers_unavailable_sources_visibly():
    feeds = build_registry(fixture_dir=FIXTURES)
    _products, results = collect(feeds)
    unavailable_sources = [r.source for r in results if not r.available]
    assert "meesho" in unavailable_sources
    report = coverage_report(results)
    assert "DATA_UNAVAILABLE" in report
    assert "meesho" in report


def test_fixtures_make_the_project_work_offline():
    """No network, no credentials, still a complete run."""
    products, results = collect(build_registry(fixture_dir=FIXTURES))
    assert products, "fixtures must yield candidates with no credentials"
    assert any(r.available for r in results)


def test_supplier_candidates_reuse_the_part_one_ingester():
    candidates = load_supplier_candidates(FIXTURES / "suppliers.csv")
    assert len(candidates) >= 4
    assert all(c.offer.unit_cost_inr > 0 for c in candidates)


def test_unreadable_supplier_rows_never_become_candidates(tmp_path: Path):
    """A price the ingester flagged must not reach the economics engine.

    The catalogue now states the facts that used to be defaulted — GST, MOQ,
    tax invoice, currency — so the readable row is legitimately accepted while
    the unreadable one is refused. Previously a row carrying only a supplier,
    a title and a price was accepted by inventing the rest.
    """
    from sourcing.supplier_ingest import RejectionReason, ingest_suppliers

    path = tmp_path / "s.csv"
    path.write_text(
        "Supplier,Product,Rate (INR),GST,MOQ,Tax Invoice,GST Inclusive,Freight\n"
        "Acme,Widget,call for quote,18,10,yes,yes,0\n"
        "Acme,Gadget,150,18,10,yes,yes,0\n",
        encoding="utf-8",
    )
    candidates = load_supplier_candidates(path)
    assert len(candidates) == 1
    assert candidates[0].offer.unit_cost_inr == Decimal("150")

    report = ingest_suppliers(path)
    assert [r.reason for r in report.rejected] == [RejectionReason.UNREADABLE_COST]


def test_a_row_missing_required_supplier_facts_is_refused_not_defaulted(tmp_path: Path):
    """The fabrication this layer exists to remove.

    A catalogue giving only supplier, title and price used to yield a usable
    offer with GST invented at 18%, MOQ at 1 and — most consequentially — a
    tax invoice assumed available, which silently granted input tax credit
    worth roughly 15% of cost.
    """
    from sourcing.supplier_ingest import RejectionReason, ingest_suppliers

    path = tmp_path / "bare.csv"
    path.write_text("Supplier,Product,Rate (INR)\nAcme,Widget,150\n", encoding="utf-8")
    report = ingest_suppliers(path)
    assert report.accepted == []
    assert report.rejected[0].reason is RejectionReason.MISSING_GST


# ==========================================================================
# Opportunity evaluation
# ==========================================================================


def _evaluate(products, candidates, pipeline, **policy_kwargs) -> Opportunity:
    group = group_products(products)[0]
    return evaluate_group(
        group, candidates, pipeline=pipeline, policy=OpportunityPolicy(**policy_kwargs)
    )


def test_no_supplier_means_rejected():
    """Cannot be sourced, so cannot be sold."""
    store = ReferenceStore(REFERENCE).load()
    opportunity = _evaluate([a_product("Kemei KX-880 Trimmer")], [], Pipeline(store))
    assert opportunity.verdict is Verdict.REJECTED
    assert any("no supplier match" in b for b in opportunity.blockers)


def test_weak_supplier_match_demotes_to_backup(pipeline):
    opportunity = _evaluate(
        [a_product("Kemei KX-880 Trimmer")],
        [a_candidate("X", "Kemei Trimmer", "200")],
        pipeline,
    )
    assert opportunity.verdict is Verdict.BACKUP
    assert any("supplier match is only" in r for r in opportunity.recoverable)


def test_no_observed_price_is_insufficient_data(pipeline):
    opportunity = _evaluate(
        [a_product("Kemei KX-880 Trimmer", price=None)],
        [a_candidate("Shenzhen", "Kemei KX-880 Trimmer", "268")],
        pipeline,
    )
    assert opportunity.verdict is Verdict.INSUFFICIENT_DATA


def test_too_few_comparables_is_recoverable_not_fatal(pipeline):
    """A research task, not a dead end."""
    opportunity = _evaluate(
        [a_product("Kemei KX-880 Trimmer", price="1299")],
        [a_candidate("Shenzhen", "Kemei KX-880 Trimmer", "268")],
        pipeline,
    )
    assert opportunity.verdict is not Verdict.RECOMMENDED
    assert any("comparables" in r for r in opportunity.recoverable)


def test_failing_worst_case_is_rejected_with_a_gap(pipeline):
    products = [
        a_product("Kemei KX-880 Trimmer", price="620"),
        a_product("Kemei KX-880 Trimmer Cordless", price="640"),
        a_product("KX880 Trimmer Rechargeable", price="660"),
    ]
    opportunity = _evaluate(
        products, [a_candidate("Shenzhen", "Kemei KX-880 Trimmer", "420")], pipeline
    )
    assert opportunity.verdict is Verdict.REJECTED
    assert opportunity.blockers


def test_every_opportunity_requires_approval(pipeline):
    opportunity = _evaluate(
        [a_product("Kemei KX-880 Trimmer")],
        [a_candidate("Shenzhen", "Kemei KX-880 Trimmer", "268")],
        pipeline,
    )
    assert opportunity.requires_approval


def test_ranking_uses_worst_case_not_expected():
    """Ranking on expected profit puts the most optimistic assumptions first."""
    optimistic = Opportunity(
        group_key="A",
        title="A",
        verdict=Verdict.RECOMMENDED,
        discovery_coverage_pct=Decimal("100"),
    )
    optimistic.expected_profit_inr = Decimal("900")
    optimistic.worst_profit_inr = Decimal("10")
    steady = Opportunity(
        group_key="B",
        title="B",
        verdict=Verdict.RECOMMENDED,
        discovery_coverage_pct=Decimal("100"),
    )
    steady.expected_profit_inr = Decimal("400")
    steady.worst_profit_inr = Decimal("300")

    from sourcing.opportunities import _score

    ranked = rank_opportunities([_score(optimistic), _score(steady)])
    assert ranked[0].group_key == "B"


def test_recommended_outranks_backup():
    recommended = Opportunity(group_key="R", title="R", verdict=Verdict.RECOMMENDED)
    backup = Opportunity(group_key="B", title="B", verdict=Verdict.BACKUP)
    assert rank_opportunities([backup, recommended])[0].verdict is Verdict.RECOMMENDED


def test_rejected_opportunities_score_zero():
    from sourcing.opportunities import _score

    rejected = Opportunity(
        group_key="X",
        title="X",
        verdict=Verdict.REJECTED,
        discovery_coverage_pct=Decimal("100"),
    )
    rejected.worst_profit_inr = Decimal("500")
    assert _score(rejected).rank_score == 0


def test_reports_render_and_state_the_approval_requirement(pipeline):
    opportunity = _evaluate(
        [a_product("Kemei KX-880 Trimmer")],
        [a_candidate("Shenzhen", "Kemei KX-880 Trimmer", "268")],
        pipeline,
    )
    ranked = rank_opportunities([opportunity])
    assert "OPPORTUNITIES" in opportunity_report(ranked)
    assert "approval" in opportunity_report(ranked).lower()
    assert isinstance(rejection_report(ranked), str)
    assert isinstance(recommendation_report(ranked), str)


# ==========================================================================
# End to end, entirely offline
# ==========================================================================


def test_full_discovery_to_ranking_runs_offline(pipeline):
    """Fixtures only: no network, no credentials, complete result."""
    products, results = collect(build_registry(fixture_dir=FIXTURES))
    groups = group_products(products)
    candidates = load_supplier_candidates(FIXTURES / "suppliers.csv")

    opportunities = rank_opportunities(
        [evaluate_group(g, candidates, pipeline=pipeline) for g in groups]
    )
    assert opportunities
    assert any(not r.available for r in results), "unavailable sources must stay visible"

    verdicts = {o.verdict for o in opportunities}
    assert verdicts <= set(Verdict)
    for opportunity in opportunities:
        assert opportunity.requires_approval
        if opportunity.verdict is Verdict.REJECTED:
            assert opportunity.blockers or opportunity.reasons


def test_compliance_can_reject_an_otherwise_profitable_product(pipeline):
    """A high margin cannot carry a product past a certification failure."""
    products = [
        a_product("Kemei KX-880 Trimmer", price="1299"),
        a_product("Kemei KX-880 Trimmer Cordless", price="1349"),
        a_product("KX880 Trimmer Rechargeable", price="1420"),
    ]
    opportunity = _evaluate(
        products, [a_candidate("Shenzhen", "Kemei KX-880 Trimmer", "268")], pipeline
    )
    assert opportunity.expected_margin_pct is not None
    assert opportunity.expected_margin_pct > 0
    assert opportunity.verdict is not Verdict.RECOMMENDED


# ==========================================================================
# Properties
# ==========================================================================


@given(
    titles=st.lists(
        st.sampled_from(
            ["Kemei KX-880 Trimmer", "Steel Lunchbox 750ml", "LED Strip 5m RGB"]
        ),
        min_size=1,
        max_size=9,
    )
)
@settings(max_examples=30)
def test_grouping_never_loses_or_duplicates_an_observation(titles: list[str]):
    products = [a_product(t) for t in titles]
    groups = group_products(products)
    assert sum(len(g.members) for g in groups) == len(products)


@given(price=st.decimals(min_value=Decimal("1"), max_value=Decimal("99999"), places=2))
@settings(max_examples=30)
def test_coverage_is_always_a_percentage(price: Decimal):
    assert 0 <= a_product("Trimmer", price=str(price)).coverage_pct <= 100


@given(
    a=st.text(min_size=1, max_size=40),
    b=st.text(min_size=1, max_size=40),
)
@settings(max_examples=40)
def test_comparison_is_symmetric_in_confidence(a: str, b: str):
    """Order of observation must not change whether two records match."""
    left, right = a_product(a), a_product(b)
    assert compare(left, right).confidence is compare(right, left).confidence
