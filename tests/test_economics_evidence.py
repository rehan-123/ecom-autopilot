"""Phase 2C: a missing economics input must never become a confirmed zero.

Every value guarded here is a cost. Substituting zero understates cost and so
overstates profit — the one direction of error that turns a loss-making SKU
into an apparent opportunity. An explicitly configured zero is a real claim
and is honoured; absence is not.
"""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from econ.calculator import _gst_rate_for
from econ.money import EconomicsDataUnavailable, compute_fees, compute_return_cost

RATE_CARD = {
    "commission_pct_by_category": {"default": Decimal("8"), "home_kitchen": Decimal("6")},
    "fixed_fee_bands": [{"max_price": Decimal("500"), "fee": Decimal("10")}],
    "collection_fee_pct": Decimal("2"),
    "shipping_by_weight_g": [{"max_weight_g": Decimal("500"), "fee": Decimal("60")}],
    "gst_on_fees_pct": Decimal("18"),
}
BENCHMARK = {
    "return_rate_pct": Decimal("6"),
    "rto_rate_pct": Decimal("4"),
    "cost_per_return_inr": Decimal("70"),
    "unsaleable_share_pct": Decimal("30"),
}


def fees(**overrides):
    card = copy.deepcopy(RATE_CARD)
    card.update(overrides)
    return compute_fees(
        selling_price_inr=Decimal("999"),
        category="home_kitchen",
        weight_g=300,
        rate_card=card,
    )


def without(mapping: dict, key: str) -> dict:
    stripped = copy.deepcopy(mapping)
    stripped.pop(key)
    return stripped


# ==========================================================================
# Marketplace fees
# ==========================================================================


@pytest.mark.parametrize(
    "key",
    [
        "commission_pct_by_category",
        "collection_fee_pct",
        "gst_on_fees_pct",
        "fixed_fee_bands",
        "shipping_by_weight_g",
    ],
)
def test_a_missing_fee_input_is_refused(key: str):
    card = without(RATE_CARD, key)
    with pytest.raises(EconomicsDataUnavailable, match=key.split("_", maxsplit=1)[0]):
        compute_fees(
            selling_price_inr=Decimal("999"),
            category="home_kitchen",
            weight_g=300,
            rate_card=card,
        )


def test_missing_collection_fee_is_not_a_confirmed_zero():
    """The distinction the whole change exists to make."""
    stated_zero = fees(collection_fee_pct=Decimal("0"))
    assert stated_zero.collection_fee_inr == 0
    with pytest.raises(EconomicsDataUnavailable):
        compute_fees(
            selling_price_inr=Decimal("999"),
            category="home_kitchen",
            weight_g=300,
            rate_card=without(RATE_CARD, "collection_fee_pct"),
        )


def test_missing_commission_slab_with_no_default_is_refused():
    card = copy.deepcopy(RATE_CARD)
    card["commission_pct_by_category"] = {"electronics": Decimal("5")}
    with pytest.raises(EconomicsDataUnavailable, match="no commission slab"):
        compute_fees(
            selling_price_inr=Decimal("999"),
            category="home_kitchen",
            weight_g=300,
            rate_card=card,
        )


def test_a_category_falls_back_to_an_explicit_default_slab():
    """A stated default is legitimate; only absence is refused."""
    card = copy.deepcopy(RATE_CARD)
    result = compute_fees(
        selling_price_inr=Decimal("999"),
        category="not_listed",
        weight_g=300,
        rate_card=card,
    )
    assert result.commission_inr > 0


def test_an_empty_band_table_is_not_a_zero_fee():
    card = copy.deepcopy(RATE_CARD)
    card["fixed_fee_bands"] = []
    with pytest.raises(EconomicsDataUnavailable, match="no max_price bands"):
        compute_fees(
            selling_price_inr=Decimal("999"),
            category="home_kitchen",
            weight_g=300,
            rate_card=card,
        )


def test_explicit_zero_fees_still_calculate():
    """A rate card stating zero is making a real claim."""
    result = fees(
        collection_fee_pct=Decimal("0"),
        gst_on_fees_pct=Decimal("0"),
        fixed_fee_bands=[{"max_price": Decimal("100000"), "fee": Decimal("0")}],
    )
    assert result.collection_fee_inr == 0
    assert result.gst_on_fees_inr == 0
    assert result.commission_inr > 0, "commission and shipping still apply"
    assert result.shipping_inr > 0


def test_a_none_value_is_treated_as_absent():
    with pytest.raises(EconomicsDataUnavailable):
        fees(collection_fee_pct=None)


# ==========================================================================
# Returns / RTO
# ==========================================================================


@pytest.mark.parametrize(
    "key",
    ["return_rate_pct", "rto_rate_pct", "cost_per_return_inr", "unsaleable_share_pct"],
)
def test_a_missing_return_benchmark_field_is_refused(key: str):
    with pytest.raises(EconomicsDataUnavailable, match=key):
        compute_return_cost(
            landed_cost_inr=Decimal("200"), benchmark=without(BENCHMARK, key)
        )


def test_explicit_zero_return_rates_still_calculate():
    zeroed = {**BENCHMARK, "return_rate_pct": Decimal("0"), "rto_rate_pct": Decimal("0")}
    result = compute_return_cost(landed_cost_inr=Decimal("200"), benchmark=zeroed)
    assert result.expected_cost_per_order_inr == 0


def test_a_missing_return_rate_is_not_the_same_as_zero_returns():
    """Zero returns is a claim; not knowing is not."""
    stated = compute_return_cost(
        landed_cost_inr=Decimal("200"),
        benchmark={
            **BENCHMARK,
            "return_rate_pct": Decimal("0"),
            "rto_rate_pct": Decimal("0"),
        },
    )
    assert stated.expected_cost_per_order_inr == 0
    with pytest.raises(EconomicsDataUnavailable):
        compute_return_cost(
            landed_cost_inr=Decimal("200"),
            benchmark=without(BENCHMARK, "return_rate_pct"),
        )


# ==========================================================================
# GST / tax table
# ==========================================================================


def test_a_missing_default_gst_rate_is_refused():
    with pytest.raises(EconomicsDataUnavailable, match="no default GST rate"):
        _gst_rate_for("9999", {"gst_rate_pct_by_hsn": {}})


def test_an_hsn_specific_rate_is_used_when_present():
    rate = _gst_rate_for(
        "8510",
        {
            "gst_rate_pct_by_hsn": {"8510": Decimal("18")},
            "default_gst_rate_pct": Decimal("12"),
        },
    )
    assert rate == Decimal("18")


def test_an_explicit_zero_gst_rate_is_honoured():
    """Some HSN codes genuinely attract 0%."""
    rate = _gst_rate_for(
        "0401",
        {
            "gst_rate_pct_by_hsn": {"0401": Decimal("0")},
            "default_gst_rate_pct": Decimal("18"),
        },
    )
    assert rate == 0


# ==========================================================================
# Pipeline degrades rather than crashing
# ==========================================================================


def _stripped_pipeline(key: str):
    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore

    store = ReferenceStore("tests/fixtures/reference").load()
    store.latest("rate_card:test_marketplace").payload.pop(key, None)
    return Pipeline(store), store


def test_missing_fee_data_degrades_to_insufficient_data_not_a_crash():
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import Verdict, evaluate_group

    pipeline, _store = _stripped_pipeline("collection_fee_pct")
    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    group = next(g for g in group_products(products) if "KX880" in g.key)

    opportunity = evaluate_group(group, candidates, pipeline=pipeline)
    assert opportunity.verdict is Verdict.INSUFFICIENT_DATA
    assert "economics unavailable" in opportunity.blockers[0]


def test_degraded_economics_produce_no_numeric_profitability():
    """No fake numbers, and no false RECOMMENDED."""
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import Verdict, evaluate_group

    pipeline, _store = _stripped_pipeline("gst_on_fees_pct")
    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    group = next(g for g in group_products(products) if "KX880" in g.key)

    opportunity = evaluate_group(group, candidates, pipeline=pipeline)
    assert opportunity.expected_margin_pct is None
    assert opportunity.worst_margin_pct is None
    assert opportunity.verdict is not Verdict.RECOMMENDED
    assert opportunity.rank_score in (None, 0)


# ==========================================================================
# Existing outcomes unchanged
# ==========================================================================


def test_existing_verdicts_are_unchanged_with_complete_reference_data():
    """Kemei / lunchbox / LED / generic behaviour must be untouched."""
    import collections

    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import evaluate_group, rank_opportunities

    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())
    opportunities = rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )
    tally = collections.Counter(str(o.verdict) for o in opportunities)
    assert tally == {"BACKUP": 1, "REJECTED": 6}

    by_title = {o.title: o for o in opportunities}
    kemei = next(o for t, o in by_title.items() if "KX880" in t)
    assert "BIS" in kemei.blockers[0]
    lunchbox = next(o for t, o in by_title.items() if "Lunchbox" in t)
    assert "worst_case" in lunchbox.blockers[0]
    led = next(o for t, o in by_title.items() if "LED Strip Light 5m" in t)
    assert str(led.verdict) == "BACKUP"
    assert "PROBABLE" in led.recoverable[0]
