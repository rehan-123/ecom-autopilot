"""Unit and property tests for the Part 1 engines.

The property tests matter more than the unit tests here. Money arithmetic fails
in ways that individual examples miss: a sign error shows up only for certain
inputs, and a rounding bug only at certain magnitudes. The invariants below
hold for every input, so they catch classes of defect rather than instances.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from catalog.enums import (
    ComplianceResult,
    GateResult,
    HeadroomResult,
    IdentityConfidence,
    RiskTier,
    ScenarioName,
    ScoreResult,
    SignalSource,
    SignalStatus,
)
from catalog.schemas import Product, SupplierOffer, new_id
from compliance.engine import HsnProvenance
from compliance.engine import evaluate as evaluate_compliance
from econ.calculator import (
    EconomicsInputs,
    ScenarioDeltas,
    apply_profitability_gate,
    compute_economics,
)
from econ.landed_cost import CarryingCostInputs, compute_landed_cost, moq_carrying_cost
from econ.money import (
    ZERO,
    compute_fees,
    compute_return_cost,
    input_tax_credit,
    money,
    rupees,
    split_inclusive,
)
from reference.loader import ReferenceStore, VerificationState, pin
from signals.market import (
    Comparable,
    MarketPriceReference,
    ScoreComponent,
    Signal,
    assess_risk,
    build_components,
    check_price_headroom,
    compute_opportunity_score,
    unavailable,
)

FIXTURES = "tests/fixtures/reference"

# Bound generated values to plausible retail magnitudes; the invariants are
# about relationships, not about surviving absurd inputs.
prices = st.decimals(min_value=Decimal("50"), max_value=Decimal("50000"), places=2)
costs = st.decimals(min_value=Decimal("10"), max_value=Decimal("20000"), places=2)
rates = st.decimals(min_value=Decimal("0"), max_value=Decimal("28"), places=0)


@pytest.fixture(scope="module")
def store() -> ReferenceStore:
    return ReferenceStore(FIXTURES).load()


@pytest.fixture
def references(store: ReferenceStore):
    return pin(
        store,
        marketplace="test_marketplace",
        category="personal_care_electrical",
        risk_tier="MEDIUM",
        scenario="EXPECTED",
    )


def make_offer(
    cost: str = "280", *, invoice: bool = True, moq: int = 50
) -> SupplierOffer:
    return SupplierOffer(
        supplier_id=new_id(),
        product_id=new_id(),
        unit_cost_inr=Decimal(cost),
        cost_is_gst_inclusive=True,
        gst_rate_pct=Decimal("18"),
        tax_invoice_available=invoice,
        moq_units=moq,
        inbound_freight_inr=Decimal("12.50"),
    )


def economics_for(
    references, offer: SupplierOffer, price: str = "1299", scenario=ScenarioName.EXPECTED
):
    deltas = ScenarioDeltas.from_payload(
        scenario,
        ReferenceStore(FIXTURES).load().latest(f"scenario_profile:{scenario}").payload,
    )
    landed = compute_landed_cost(offer, scenario=scenario)
    return compute_economics(
        EconomicsInputs(
            selling_price_inr=Decimal(price),
            category="personal_care_electrical",
            hsn_code="8510",
            weight_g=310,
        ),
        landed,
        references,
        deltas,
    )


# ==========================================================================
# Money primitives
# ==========================================================================


def test_money_refuses_floats():
    with pytest.raises(TypeError, match="float"):
        money(280.33)


def test_gst_split_reconstructs_the_gross():
    split = split_inclusive(Decimal("799"), Decimal("18"))
    assert rupees(split.taxable_inr + split.tax_inr) == rupees(Decimal("799"))


@given(gross=prices, rate=rates)
def test_gst_split_never_exceeds_the_gross(gross: Decimal, rate: Decimal):
    split = split_inclusive(gross, rate)
    assert split.tax_inr >= ZERO
    assert split.taxable_inr <= gross


def test_itc_is_zero_without_a_tax_invoice():
    """No tax invoice means the GST paid is unrecoverable."""
    assert (
        input_tax_credit(
            Decimal("399"),
            gst_rate_pct=Decimal("18"),
            cost_is_gst_inclusive=True,
            tax_invoice_available=False,
        )
        == ZERO
    )


@given(cost=costs, rate=rates)
def test_itc_never_exceeds_the_cost(cost: Decimal, rate: Decimal):
    itc = input_tax_credit(
        cost,
        gst_rate_pct=rate,
        cost_is_gst_inclusive=True,
        tax_invoice_available=True,
    )
    assert ZERO <= itc <= cost


def test_returns_are_charged_per_shipped_order():
    """Amortising across shipped orders is the honest denominator."""
    cost = compute_return_cost(
        landed_cost_inr=Decimal("280"),
        benchmark={
            "return_rate_pct": 10,
            "rto_rate_pct": 5,
            "cost_per_return_inr": 150,
            "unsaleable_share_pct": 40,
        },
    )
    assert cost.expected_cost_per_order_inr > ZERO
    assert cost.delivered_fraction == money("0.85")


def test_unmapped_category_falls_back_and_says_so():
    fees = compute_fees(
        selling_price_inr=Decimal("799"),
        category="nonexistent_category",
        weight_g=310,
        rate_card=ReferenceStore(FIXTURES)
        .load()
        .latest("rate_card:test_marketplace")
        .payload,
    )
    assert any("no commission slab" in n for n in fees.notes)


def test_unregistered_seller_cannot_recover_gst_on_fees():
    card = ReferenceStore(FIXTURES).load().latest("rate_card:test_marketplace").payload
    registered = compute_fees(
        selling_price_inr=Decimal("799"), category="default", weight_g=310, rate_card=card
    )
    unregistered = compute_fees(
        selling_price_inr=Decimal("799"),
        category="default",
        weight_g=310,
        rate_card=card,
        registered_for_gst=False,
    )
    assert unregistered.net_cost_inr > registered.net_cost_inr


# ==========================================================================
# Landed cost
# ==========================================================================


def test_landed_cost_exceeds_supplier_cost():
    """`supplier_cost == landed_cost` is the shortcut this engine refuses."""
    landed = compute_landed_cost(make_offer("280"))
    assert landed.total_landed_cost_inr > Decimal("280") - landed.input_tax_credit_inr


def test_missing_tax_invoice_raises_landed_cost():
    with_invoice = compute_landed_cost(make_offer("280", invoice=True))
    without = compute_landed_cost(make_offer("280", invoice=False))
    assert without.total_landed_cost_inr > with_invoice.total_landed_cost_inr
    assert any("tax invoice" in w for w in without.warnings)


def test_moq_carrying_cost_rises_with_moq():
    policy = CarryingCostInputs()
    small, _ = moq_carrying_cost(
        unit_cost_inr=Decimal("280"),
        moq_units=10,
        expected_monthly_velocity=Decimal("10"),
        policy=policy,
    )
    large, _ = moq_carrying_cost(
        unit_cost_inr=Decimal("280"),
        moq_units=200,
        expected_monthly_velocity=Decimal("10"),
        policy=policy,
    )
    assert large > small


def test_zero_velocity_charges_the_whole_batch():
    """An unsellable SKU cannot be amortised; it must not divide by zero."""
    cost, assumptions = moq_carrying_cost(
        unit_cost_inr=Decimal("100"),
        moq_units=50,
        expected_monthly_velocity=ZERO,
        policy=CarryingCostInputs(),
    )
    assert cost == Decimal("5000")
    assert any(a.name == "velocity_zero_fallback" for a in assumptions)


def test_every_landed_cost_records_its_assumptions():
    landed = compute_landed_cost(make_offer())
    names = {a.name for a in landed.assumptions}
    assert "expected_monthly_velocity" in names
    assert all(a.confidence for a in landed.assumptions)


@given(cost=costs)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_landed_cost_is_monotonic_in_supplier_cost(cost: Decimal):
    cheaper = compute_landed_cost(make_offer(str(cost)))
    dearer = compute_landed_cost(make_offer(str(cost + Decimal("50"))))
    assert dearer.total_landed_cost_inr > cheaper.total_landed_cost_inr


# ==========================================================================
# Economics properties
# ==========================================================================


def test_higher_landed_cost_cannot_improve_margin(references):
    cheap = economics_for(references, make_offer("200"))
    dear = economics_for(references, make_offer("400"))
    assert dear.net_margin_pct < cheap.net_margin_pct


def test_higher_price_improves_margin_above_breakeven(references):
    low = economics_for(references, make_offer(), price="1200")
    high = economics_for(references, make_offer(), price="1600")
    assert high.net_margin_pct > low.net_margin_pct


def test_missing_itc_cannot_improve_economics(references):
    with_invoice = economics_for(references, make_offer("280", invoice=True))
    without = economics_for(references, make_offer("280", invoice=False))
    assert without.net_profit_inr < with_invoice.net_profit_inr


def test_worst_is_not_better_than_expected_is_not_better_than_best(store, references):
    results = {
        scenario: economics_for(references, make_offer(), scenario=scenario)
        for scenario in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
    }
    assert (
        results[ScenarioName.WORST].net_margin_pct
        <= results[ScenarioName.EXPECTED].net_margin_pct
        <= results[ScenarioName.BEST].net_margin_pct
    )


def test_min_profitable_price_exceeds_breakeven(references):
    economics = economics_for(references, make_offer())
    assert economics.min_profitable_price_inr > economics.breakeven_price_inr


def test_selling_at_min_profitable_price_meets_the_floor(references):
    """The inverse must agree with the forward calculation."""
    first = economics_for(references, make_offer())
    at_min = economics_for(
        references, make_offer(), price=str(first.min_profitable_price_inr)
    )
    floor = Decimal(str(references.threshold_profile.payload["min_net_margin_pct"]))
    assert at_min.net_margin_pct >= floor - Decimal("1")


@given(price=prices)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_all_money_outputs_stay_decimal(references, price: Decimal):
    economics = economics_for(references, make_offer(), price=str(price))
    for value in (
        economics.net_profit_inr,
        economics.landed_cost_inr,
        economics.breakeven_price_inr,
    ):
        assert isinstance(value, Decimal)


# ==========================================================================
# Profitability gate
# ==========================================================================


def test_gate_passes_a_healthy_sku(references):
    scenarios = {
        s: economics_for(references, make_offer("200"), price="1499", scenario=s)
        for s in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
    }
    assert apply_profitability_gate(scenarios, references).result is GateResult.PASS


def test_gate_rejects_on_negative_worst_case(references):
    """The core policy: expected-case profit does not excuse a losing downside."""
    scenarios = {
        s: economics_for(references, make_offer("620"), price="899", scenario=s)
        for s in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
    }
    outcome = apply_profitability_gate(scenarios, references)
    assert outcome.result is GateResult.FAIL
    assert outcome.gap_to_pass


def test_failed_gate_always_names_a_gate_and_a_gap(references):
    scenarios = {
        s: economics_for(references, make_offer("700"), price="799", scenario=s)
        for s in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
    }
    outcome = apply_profitability_gate(scenarios, references)
    assert outcome.result is GateResult.FAIL
    assert outcome.failed_gate
    assert "action" in outcome.gap_to_pass


# ==========================================================================
# Price headroom
# ==========================================================================


def _reference(*prices_: str) -> MarketPriceReference:
    return MarketPriceReference.from_comparables(
        [
            Comparable("m", f"item {i}", Decimal(p), f"http://x/{i}", datetime.now(UTC))
            for i, p in enumerate(prices_)
        ],
        collected_by="tester",
    )


def test_price_reference_requires_three_comparables():
    with pytest.raises(ValueError, match="at least 3"):
        _reference("100", "200")


def test_percentiles_are_ordered():
    ref = _reference("400", "500", "600", "900")
    assert ref.p25_inr <= ref.median_inr <= ref.p75_inr


def test_headroom_blocks_when_market_is_below_viable_price():
    outcome = check_price_headroom(
        min_profitable_price_inr=Decimal("512"), reference=_reference("440", "449", "470")
    )
    assert outcome.result is HeadroomResult.BLOCK
    assert outcome.headroom_pct < ZERO


def test_headroom_marginal_between_zero_and_floor():
    outcome = check_price_headroom(
        min_profitable_price_inr=Decimal("500"), reference=_reference("510", "520", "530")
    )
    assert outcome.result is HeadroomResult.MARGINAL


def test_headroom_passes_with_room():
    outcome = check_price_headroom(
        min_profitable_price_inr=Decimal("500"),
        reference=_reference("900", "950", "1000"),
    )
    assert outcome.result is HeadroomResult.PASS


# ==========================================================================
# Signals and scoring
# ==========================================================================


def test_data_unavailable_cannot_carry_a_value():
    with pytest.raises(ValueError, match="not a zero"):
        Signal(
            "SEARCH_INTEREST",
            SignalStatus.DATA_UNAVAILABLE,
            SignalSource.MANUAL,
            "tester",
            value_numeric=Decimal("50"),
        )


def test_unknown_signal_type_is_rejected():
    with pytest.raises(ValueError, match="unknown signal type"):
        Signal("MADE_UP", SignalStatus.OBSERVED, SignalSource.MANUAL, "tester")


def test_stale_signal_is_not_usable():
    old = Signal(
        "SEARCH_INTEREST",
        SignalStatus.OBSERVED,
        SignalSource.MANUAL,
        "tester",
        value_numeric=Decimal("50"),
        collected_at=datetime.now(UTC) - timedelta(days=400),
    )
    assert not old.is_usable()
    assert old.effective_status() is SignalStatus.STALE


def test_insufficient_coverage_emits_no_score():
    components = [
        ScoreComponent(
            "PROFITABILITY", Decimal("40"), Decimal("80"), SignalStatus.OBSERVED, "e"
        ),
        ScoreComponent(
            "SEARCH_INTEREST", Decimal("60"), None, SignalStatus.DATA_UNAVAILABLE, "n"
        ),
    ]
    score = compute_opportunity_score(components, min_coverage_pct=Decimal("60"))
    assert score.result is ScoreResult.INSUFFICIENT_DATA
    assert score.value is None


def test_score_display_always_carries_coverage():
    """A bare number invites comparison between differently-evidenced products."""
    components = [
        ScoreComponent(
            "PROFITABILITY", Decimal("40"), Decimal("90"), SignalStatus.OBSERVED, "e"
        ),
        ScoreComponent(
            "RETURN_RISK", Decimal("60"), Decimal("70"), SignalStatus.OBSERVED, "b"
        ),
    ]
    score = compute_opportunity_score(components)
    assert score.result is ScoreResult.SCORED
    assert "coverage" in score.display()


def test_missing_signal_is_excluded_not_zeroed():
    """Dividing by total weight would treat every gap as a zero."""
    present_only = compute_opportunity_score(
        [
            ScoreComponent(
                "PROFITABILITY", Decimal("40"), Decimal("80"), SignalStatus.OBSERVED, "e"
            )
        ],
        min_coverage_pct=Decimal("0"),
    )
    with_gap = compute_opportunity_score(
        [
            ScoreComponent(
                "PROFITABILITY", Decimal("40"), Decimal("80"), SignalStatus.OBSERVED, "e"
            ),
            ScoreComponent(
                "SEARCH_INTEREST", Decimal("10"), None, SignalStatus.DATA_UNAVAILABLE, "n"
            ),
        ],
        min_coverage_pct=Decimal("0"),
    )
    assert present_only.value == with_gap.value
    assert with_gap.coverage_pct < present_only.coverage_pct


def test_build_components_marks_absent_signals():
    components = build_components(
        net_margin_pct=Decimal("25"),
        return_rate_pct=Decimal("10"),
        signals={},
        compliance_clear=True,
        operational_complexity=Decimal("20"),
    )
    missing = [c.name for c in components if not c.populated]
    assert "SEARCH_INTEREST" in missing


def test_unavailable_helper_records_the_reason():
    signal = unavailable(
        "REVIEW_VELOCITY", reason="no legitimate API", collected_by="sys"
    )
    assert signal.status is SignalStatus.DATA_UNAVAILABLE
    assert signal.source_ref == "no legitimate API"


# ==========================================================================
# Risk
# ==========================================================================


def _product(**kwargs) -> Product:
    base = {
        "canonical_name": "Test Item",
        "category": "home",
        "identity_confidence": IdentityConfidence.HIGH,
    }
    return Product(**{**base, **kwargs})


def test_sized_goods_are_high_return():
    risk = assess_risk(
        _product(sized=True), supplier_count=3, benchmark_return_rate_pct=Decimal("5")
    )
    assert risk.tier is RiskTier.HIGH_RETURN


def test_single_supplier_raises_concentration_risk():
    risk = assess_risk(
        _product(), supplier_count=1, benchmark_return_rate_pct=Decimal("5")
    )
    assert any("single supplier" in r for r in risk.rationale)


def test_missing_benchmark_is_unknown_not_low():
    risk = assess_risk(_product(), supplier_count=3, benchmark_return_rate_pct=None)
    assert str(risk.return_risk) == "UNKNOWN"


def test_plain_product_is_low_risk():
    risk = assess_risk(
        _product(), supplier_count=5, benchmark_return_rate_pct=Decimal("4")
    )
    assert risk.tier is RiskTier.LOW


def test_risk_always_explains_itself():
    risk = assess_risk(
        _product(fragile=True), supplier_count=2, benchmark_return_rate_pct=Decimal("8")
    )
    assert risk.rationale


# ==========================================================================
# Compliance
# ==========================================================================


@pytest.fixture
def ruleset(store: ReferenceStore):
    return store.get("compliance_ruleset:india", "baseline-0")


def test_blocking_requirement_without_evidence_fails(ruleset):
    outcome = evaluate_compliance(
        ruleset=ruleset,
        hsn_code="8510",
        product_text="Trimmer",
        evidence={},
        hsn_provenance=HsnProvenance.SOURCED,
        now=ruleset.effective_from,
    )
    assert outcome.overall is ComplianceResult.FAIL


def test_unknown_never_becomes_pass(ruleset):
    """A keyword-only match is a question for a human, not a clearance."""
    outcome = evaluate_compliance(
        ruleset=ruleset,
        hsn_code="9999",
        product_text="deluxe toy set",
        evidence={},
        now=ruleset.effective_from,
    )
    assert outcome.overall is not ComplianceResult.PASS


def test_stale_ruleset_cannot_clear_a_product(ruleset):
    """An out-of-date rulebook may omit requirements that apply.

    Satisfying the rules a stale book contains proves nothing about the rules
    it does not, so nothing clears against it -- even with full evidence.
    """
    full_evidence = {
        "bis_licence_number": "CM/L-1",
        "manufacturer_name": "KX",
        "licence_valid_until": "2027-01-01",
    }
    fresh = evaluate_compliance(
        ruleset=ruleset,
        hsn_code="8510",
        product_text="Trimmer",
        evidence=full_evidence,
        hsn_provenance=HsnProvenance.SOURCED,
        now=ruleset.last_verified_at,
    )
    assert fresh.overall is ComplianceResult.PASS

    long_after = ruleset.last_verified_at + timedelta(days=400)
    stale = evaluate_compliance(
        ruleset=ruleset,
        hsn_code="8510",
        product_text="Trimmer",
        evidence=full_evidence,
        hsn_provenance=HsnProvenance.SOURCED,
        now=long_after,
    )
    assert ruleset.verification_state(now=long_after) is VerificationState.STALE
    assert stale.overall is ComplianceResult.REQUIRES_HUMAN_REVIEW


def test_shipped_ruleset_is_unverified_and_clears_nothing():
    """The bundled regulatory list has never been checked against a source."""
    production = ReferenceStore("data/reference").load()
    shipped = production.get("compliance_ruleset:india", "baseline-0")
    assert shipped.verification_state() is VerificationState.UNVERIFIED
    outcome = evaluate_compliance(
        ruleset=shipped, hsn_code="3924", product_text="Kitchen Rack", evidence={}
    )
    assert outcome.overall is ComplianceResult.REQUIRES_HUMAN_REVIEW


def test_compliance_records_the_ruleset_version(ruleset):
    outcome = evaluate_compliance(
        ruleset=ruleset, hsn_code="3924", product_text="Rack", evidence={}
    )
    assert outcome.ruleset_version == "baseline-0"


# ==========================================================================
# Reference data
# ==========================================================================


def test_shipped_reference_templates_are_unverified():
    """Templates carry shapes, not authoritative figures."""
    production = ReferenceStore("data/reference").load()
    audit = production.audit()
    assert audit["rate_card:flipkart:template-0"] is VerificationState.UNVERIFIED
    assert audit["tax_table:template-0"] is VerificationState.UNVERIFIED


def test_own_policy_ships_verified():
    audit = ReferenceStore("data/reference").load().audit()
    assert audit["threshold_profile:LOW:policy-2026.08"] is VerificationState.VERIFIED


def test_pinned_references_report_their_weakest_link(references):
    assert references.weakest_state() is VerificationState.VERIFIED
    assert references.unverified_docs() == []


def test_stale_document_is_detected():
    store_ = ReferenceStore(FIXTURES).load()
    doc = store_.latest("threshold_profile:LOW")
    future = doc.last_verified_at + timedelta(days=500)
    assert doc.verification_state(now=future) is VerificationState.STALE


def test_yaml_numbers_become_decimal_not_float():
    payload = ReferenceStore(FIXTURES).load().latest("rate_card:test_marketplace").payload
    assert isinstance(payload["collection_fee_pct"], Decimal)


# ==========================================================================
# Gaps surfaced by mutation testing
# ==========================================================================


def test_landed_cost_includes_freight_and_carry_not_just_cogs(references):
    """The sum of components must actually reach the total.

    A mutation that zeroed freight and carry survived until this test existed:
    nothing asserted the total was more than the effective COGS by exactly the
    other components.
    """
    landed = compute_landed_cost(make_offer("280", moq=100))
    component_sum = (
        landed.effective_cogs_inr
        + landed.inbound_freight_per_unit_inr
        + landed.moq_carry_cost_inr
        + landed.inspection_qc_cost_inr
        + landed.packaging_cost_inr
        + landed.inbound_defect_provision_inr
    )
    assert landed.total_landed_cost_inr == component_sum
    assert landed.inbound_freight_per_unit_inr > ZERO
    assert landed.moq_carry_cost_inr > ZERO
    assert landed.total_landed_cost_inr > landed.effective_cogs_inr


def test_worst_case_gate_actually_runs_when_policy_requires_it(store, references):
    """The worst-case gate fires first, before the expected-case floors.

    A mutation skipping the worst-case block must not be able to pass by
    having a later gate fail instead, so the expected-case floors are set
    trivially low: the ONLY thing that can fail here is the worst case.
    """
    lenient = dict(references.threshold_profile.payload)
    lenient.update(
        require_worst_case_non_negative=True,
        worst_case_min_net_margin_pct=0,
        min_net_margin_pct=0,
        min_profit_per_order_inr=0,
        min_roi_pct=0,
        max_supplier_cost_pct=100,
    )
    object.__setattr__(references.threshold_profile, "payload", lenient)

    # Wide-spread offer: expected clears zero, worst does not.
    scenarios = {
        s: economics_for(references, make_offer("300"), price="680", scenario=s)
        for s in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
    }
    assert scenarios[ScenarioName.EXPECTED].net_margin_pct > ZERO
    assert scenarios[ScenarioName.WORST].net_margin_pct < ZERO
    outcome = apply_profitability_gate(scenarios, references)
    assert outcome.result is GateResult.FAIL
    assert outcome.failed_gate == "worst_case_min_net_margin_pct"


def test_worst_case_check_can_be_disabled_by_policy(store):
    """The guard is policy-driven, not hardcoded: proving the branch is live."""
    references = pin(
        store,
        marketplace="test_marketplace",
        category="personal_care_electrical",
        risk_tier="MEDIUM",
        scenario="EXPECTED",
    )
    # Policy with the worst-case requirement switched off.
    relaxed = dict(references.threshold_profile.payload)
    relaxed["require_worst_case_non_negative"] = False
    relaxed["min_net_margin_pct"] = 0
    relaxed["min_profit_per_order_inr"] = 0
    relaxed["min_roi_pct"] = 0
    object.__setattr__(references.threshold_profile, "payload", relaxed)

    scenarios = {
        s: economics_for(references, make_offer("560"), price="999", scenario=s)
        for s in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
    }
    outcome = apply_profitability_gate(scenarios, references)
    # With the worst-case gate off, a negative downside no longer fails here.
    assert outcome.failed_gate != "worst_case_min_net_margin_pct"


def test_failed_gate_action_is_never_empty(references):
    """Every rejection must carry an actionable instruction, not a blank.

    A mutation that blanked the 'action' string survived because no test
    asserted the string was non-empty -- only that the key was present.
    """
    for cost in ("560", "700", "900"):
        scenarios = {
            s: economics_for(references, make_offer(cost), price="899", scenario=s)
            for s in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST)
        }
        outcome = apply_profitability_gate(scenarios, references)
        if outcome.result is GateResult.FAIL:
            assert outcome.gap_to_pass["action"].strip(), (
                f"failed gate for cost {cost} produced an empty action"
            )
            assert any(ch.isdigit() for ch in outcome.gap_to_pass["action"])
