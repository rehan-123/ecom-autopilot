"""Supplier ingestion tests.

The load-bearing assertions are about refusal. A supplier catalogue that does
not state GST, MOQ or whether a tax invoice is issued must not produce a usable
offer, because the previous layer defaulted all three — and the tax-invoice
default silently granted input tax credit worth roughly 15% of unit cost to
every supplier in the system.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from econ.money import input_tax_credit
from sourcing.supplier_ingest import (
    RejectionReason,
    SupplierIngestReport,
    dedupe_key,
    detect_currency,
    ingest_suppliers,
    parse_bool,
    supplier_id_for,
)

FIXTURE = Path("data/fixtures/discovery/suppliers.csv")

FULL_HEADER = (
    "Supplier Name,Item Description,Rate (INR),MOQ,HSN,Weight,GST,Dispatch,"
    "Tax Invoice,Currency,GST Inclusive,Inbound Freight\n"
)
FULL_ROW = "Acme Traders,Steel Lunchbox 750ml,182.00,24,7323,320,18,5,yes,INR,yes,0\n"


def write(tmp_path: Path, body: str, name: str = "s.csv") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ==========================================================================
# Valid ingestion
# ==========================================================================


def test_valid_csv_ingests(tmp_path: Path):
    report = ingest_suppliers(write(tmp_path, FULL_HEADER + FULL_ROW))
    assert report.accepted_count == 1
    assert report.rejected == []
    offer = report.accepted[0].offer
    assert offer.unit_cost_inr == Decimal("182.00")
    assert offer.moq_units == 24
    assert offer.gst_rate_pct == Decimal("18")
    assert offer.tax_invoice_available is True


def test_the_shipped_fixture_still_ingests_completely():
    """Existing fixture suppliers must keep working."""
    report = ingest_suppliers(FIXTURE)
    assert report.accepted_count == 4
    assert report.rejected == []


def test_every_supplier_fact_comes_from_the_file(tmp_path: Path):
    """Change a stated fact and the offer changes with it."""
    row = "Acme,Widget,100,50,7323,200,12,3,no,INR,yes,0\n"
    offer = ingest_suppliers(write(tmp_path, FULL_HEADER + row)).accepted[0].offer
    assert offer.gst_rate_pct == Decimal("12")
    assert offer.moq_units == 50
    assert offer.tax_invoice_available is False


# ==========================================================================
# Missing facts are refused, never defaulted
# ==========================================================================


@pytest.mark.parametrize(
    ("header", "row", "reason"),
    [
        (
            "Supplier,Product,Rate (INR),MOQ,Tax Invoice,GST Inclusive,Freight\n",
            "Acme,Widget,100,10,yes,yes,0\n",
            RejectionReason.MISSING_GST,
        ),
        (
            "Supplier,Product,Rate (INR),GST,MOQ,Freight\n",
            "Acme,Widget,100,18,10,0\n",
            RejectionReason.MISSING_TAX_INVOICE,
        ),
        (
            "Supplier,Product,Rate (INR),GST,Tax Invoice,GST Inclusive,Freight\n",
            "Acme,Widget,100,18,yes,yes,0\n",
            RejectionReason.MISSING_MOQ,
        ),
        (
            "Supplier,Product,GST,MOQ,Tax Invoice,Currency,GST Inclusive,Freight\n",
            "Acme,Widget,18,10,yes,INR,yes,0\n",
            RejectionReason.MISSING_COST,
        ),
    ],
)
def test_a_missing_required_fact_is_refused(
    tmp_path: Path, header: str, row: str, reason: RejectionReason
):
    report = ingest_suppliers(write(tmp_path, header + row))
    assert report.accepted == []
    assert [r.reason for r in report.rejected] == [reason]


def test_the_tax_invoice_default_that_was_removed_is_worth_real_money():
    """Why the tax-invoice field is mandatory rather than defaulted.

    On a Rs 240 cost at 18% GST the assumption was worth about Rs 37 per unit
    of input tax credit — frequently the difference between a viable SKU and a
    loss-making one.
    """
    cost = Decimal("240")
    granted = input_tax_credit(
        cost,
        gst_rate_pct=Decimal("18"),
        cost_is_gst_inclusive=True,
        tax_invoice_available=True,
    )
    actual = input_tax_credit(
        cost,
        gst_rate_pct=Decimal("18"),
        cost_is_gst_inclusive=True,
        tax_invoice_available=False,
    )
    assert granted > actual
    assert granted - actual > Decimal("30")


def test_missing_supplier_or_product_is_refused(tmp_path: Path):
    report = ingest_suppliers(
        write(
            tmp_path, FULL_HEADER + ",Steel Lunchbox,182,24,7323,320,18,5,yes,INR,yes,0\n"
        )
    )
    assert [r.reason for r in report.rejected] == [RejectionReason.MISSING_SUPPLIER]


# ==========================================================================
# Malformed rows are reported, never silently dropped
# ==========================================================================


def test_unreadable_cost_is_distinguished_from_a_missing_one(tmp_path: Path):
    """The operator fixes the two differently."""
    body = FULL_HEADER + "Acme,Widget,call for quote,10,7323,200,18,5,yes,INR,yes,0\n"
    report = ingest_suppliers(write(tmp_path, body))
    assert [r.reason for r in report.rejected] == [RejectionReason.UNREADABLE_COST]
    assert "call for quote" in report.rejected[0].detail


def test_rejected_rows_are_reported_with_a_row_number(tmp_path: Path):
    """A silent skip makes a truncated import look like a small catalogue."""
    body = FULL_HEADER + FULL_ROW + "Bad Co,Thing,,10,7323,200,18,5,yes,INR,yes,0\n"
    report = ingest_suppliers(write(tmp_path, body))
    assert report.accepted_count == 1
    assert len(report.rejected) == 1
    assert report.rejected[0].row_number == 3
    assert report.rejected[0].supplier_name == "Bad Co"
    assert any("refused" in w for w in report.warnings)


def test_non_positive_values_are_refused(tmp_path: Path):
    body = FULL_HEADER + "Acme,Widget,0,10,7323,200,18,5,yes,INR,yes,0\n"
    report = ingest_suppliers(write(tmp_path, body))
    assert report.rejected[0].reason is RejectionReason.IMPLAUSIBLE_VALUE


# ==========================================================================
# Currency
# ==========================================================================


def test_a_foreign_currency_is_refused_not_converted(tmp_path: Path):
    """No exchange rate is configured; converting would invent a number."""
    header = "Supplier,Product,Rate,GST,MOQ,Tax Invoice,Currency,GST Inclusive,Freight\n"
    report = ingest_suppliers(
        write(tmp_path, header + "Acme,W,100,18,10,yes,USD,yes,0\n")
    )
    assert report.rejected[0].reason is RejectionReason.UNSUPPORTED_CURRENCY
    assert "USD" in report.rejected[0].detail


def test_an_unlabelled_price_is_refused(tmp_path: Path):
    header = "Supplier,Product,Rate,GST,MOQ,Tax Invoice,GST Inclusive,Freight\n"
    report = ingest_suppliers(write(tmp_path, header + "Acme,W,100,18,10,yes,yes,0\n"))
    assert report.rejected[0].reason is RejectionReason.UNSUPPORTED_CURRENCY


def test_currency_is_read_from_the_rate_header(tmp_path: Path):
    assert detect_currency(["Supplier", "Rate (INR)"], {}) == "INR"
    assert detect_currency(["Supplier", "Cost (USD)"], {}) == "USD"
    assert detect_currency(["Supplier", "Rate"], {}) is None


def test_an_explicit_currency_column_wins():
    assert detect_currency(["Rate (INR)"], {"currency": "usd"}) == "USD"


# ==========================================================================
# Identity and deduplication
# ==========================================================================


def test_supplier_identity_is_stable_across_runs_and_files():
    """The old loader minted a fresh UUID per row, so one supplier on ten
    lines became ten suppliers, defeating concentration risk scoring."""
    assert supplier_id_for("Acme Traders") == supplier_id_for("Acme Traders")
    assert supplier_id_for("acme  traders") == supplier_id_for("Acme Traders")
    assert supplier_id_for("Acme Traders") != supplier_id_for("Other Co")


def test_the_same_supplier_on_two_rows_is_one_supplier(tmp_path: Path):
    body = (
        FULL_HEADER
        + FULL_ROW
        + "Acme Traders,Different Item,90,10,7323,200,18,5,yes,INR,yes,0\n"
    )
    report = ingest_suppliers(write(tmp_path, body))
    assert report.accepted_count == 2
    assert len({c.offer.supplier_id for c in report.accepted}) == 1


def test_duplicate_rows_are_deduplicated_deterministically(tmp_path: Path):
    """Re-importing the same catalogue must change nothing."""
    body = FULL_HEADER + FULL_ROW + FULL_ROW
    report = ingest_suppliers(write(tmp_path, body))
    assert report.accepted_count == 1
    assert report.duplicates_ignored == 1


def test_dedupe_prefers_identity_over_title():
    from catalog.schemas import SupplierOffer, new_id
    from sourcing.matching import SupplierCandidate

    def candidate(title: str, gtin: str | None) -> SupplierCandidate:
        return SupplierCandidate(
            supplier_name="Acme",
            title=title,
            gtin=gtin,
            offer=SupplierOffer(
                supplier_id=new_id(),
                product_id=new_id(),
                unit_cost_inr=Decimal("10"),
                cost_is_gst_inclusive=True,
                gst_rate_pct=Decimal("18"),
                tax_invoice_available=True,
                moq_units=1,
            ),
        )

    assert dedupe_key(candidate("Widget A", "890123")) == dedupe_key(
        candidate("Widget A Deluxe Pack", "890123")
    )


def test_ingesting_twice_produces_identical_results(tmp_path: Path):
    path = write(tmp_path, FULL_HEADER + FULL_ROW)
    first, second = ingest_suppliers(path), ingest_suppliers(path)
    assert [c.offer.supplier_id for c in first.accepted] == [
        c.offer.supplier_id for c in second.accepted
    ]


# ==========================================================================
# Provenance
# ==========================================================================


def test_every_accepted_record_carries_provenance(tmp_path: Path):
    report = ingest_suppliers(write(tmp_path, FULL_HEADER + FULL_ROW))
    candidate = report.accepted[0]
    provenance = report.provenance[str(candidate.offer.id)]
    assert provenance.row_number == 2
    assert str(tmp_path) in provenance.source_reference
    assert candidate.offer.source_document_ref.endswith("#row=2")


def test_provenance_survives_serialisation(tmp_path: Path):
    body = report_body = FULL_HEADER + FULL_ROW
    report = ingest_suppliers(write(tmp_path, body))
    assert report_body
    entry = report.as_dict()["suppliers"][0]
    assert entry["provenance"]["row_number"] == 2
    assert entry["tax_invoice_available"] is True


# ==========================================================================
# The LLM cannot create supplier data
# ==========================================================================


def test_no_llm_is_reachable_from_supplier_ingestion():
    """Supplier facts have exactly one origin: the ingested file.

    Inspects the import graph rather than the text, so a docstring mentioning
    the rule neither satisfies nor violates it.
    """
    import ast

    tree = ast.parse(Path("sourcing/supplier_ingest.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {"llm", "research", "openai", "anthropic"}
    offenders = {name for name in imported if name.split(".")[0] in forbidden}
    assert not offenders, f"supplier ingestion must not import {offenders}"


def test_research_candidates_never_carry_supplier_facts():
    """Cross-checks the research layer's half of the same rule."""
    from research.assistant import FACTUAL_FIELDS

    for field in ("supplier_name", "unit_cost_inr", "moq_units", "gst_rate_pct"):
        assert field in FACTUAL_FIELDS


def test_a_report_with_no_accepted_rows_is_still_a_report(tmp_path: Path):
    report = ingest_suppliers(write(tmp_path, FULL_HEADER))
    assert isinstance(report, SupplierIngestReport)
    assert report.accepted == []
    assert report.as_dict()["accepted"] == 0


# ==========================================================================
# Boolean parsing
# ==========================================================================


@pytest.mark.parametrize("raw", ["yes", "Y", "true", "1", "Available"])
def test_affirmative_values_parse_true(raw: str):
    assert parse_bool(raw) is True


@pytest.mark.parametrize("raw", ["no", "N", "false", "0", "none"])
def test_negative_values_parse_false(raw: str):
    assert parse_bool(raw) is False


@pytest.mark.parametrize("raw", [None, "", "   ", "maybe", "TBC"])
def test_unclear_values_parse_as_unknown(raw):
    """Silence about a tax invoice must never read as yes."""
    assert parse_bool(raw) is None


# ==========================================================================
# Existing matching must be unaffected
# ==========================================================================


def test_existing_supplier_matching_still_works():
    """Regression: the matcher consumes what this layer produces."""
    from sourcing.feeds import build_registry, collect
    from sourcing.matching import group_products, match_suppliers

    candidates = ingest_suppliers(FIXTURE).accepted
    products, _ = collect(build_registry())
    groups = group_products(products)

    kemei = next(g for g in groups if "KX880" in g.key)
    matches = match_suppliers(kemei, candidates)
    assert matches
    assert any(m.usable for m in matches)


def test_the_led_probable_match_behaviour_is_unchanged():
    """The known LED case must stay PROBABLE and unusable."""
    from sourcing.feeds import build_registry, collect
    from sourcing.matching import group_products, match_suppliers

    candidates = ingest_suppliers(FIXTURE).accepted
    products, _ = collect(build_registry())
    led = next(
        g for g in group_products(products) if "led strip light 5m" in g.key.lower()
    )
    matches = match_suppliers(led, candidates)
    assert len(matches) == 1
    assert str(matches[0].confidence) == "PROBABLE"
    assert not matches[0].usable


# ==========================================================================
# The two remaining fabrication paths, and the two the audit did not name
# ==========================================================================


def test_missing_gst_inclusivity_is_never_true(tmp_path: Path):
    """A wholesale convention is not evidence.

    On an inclusive quote the tax is a fraction of the stated price; on an
    exclusive one it is added on top. Guessing misstates recoverable credit.
    """
    header = "Supplier,Product,Rate (INR),GST,MOQ,Tax Invoice,Freight\n"
    report = ingest_suppliers(write(tmp_path, header + "Acme,W,100,18,10,yes,0\n"))
    assert report.accepted == []
    assert report.rejected[0].reason is RejectionReason.MISSING_GST_INCLUSIVITY


def test_inclusivity_is_moot_without_a_tax_invoice(tmp_path: Path):
    """Not required when it cannot change any number.

    Without an invoice the credit is zero either way, so demanding the field
    would refuse a row over something that has no effect.
    """
    header = "Supplier,Product,Rate (INR),GST,MOQ,Tax Invoice,Currency,Freight\n"
    report = ingest_suppliers(write(tmp_path, header + "Acme,W,100,18,10,no,INR,0\n"))
    assert report.accepted_count == 1
    assert report.accepted[0].offer.cost_is_gst_inclusive is None


def test_unknown_inclusivity_cannot_reach_economics_as_true():
    """Economics must refuse rather than assume."""
    with pytest.raises(ValueError, match="did not state whether"):
        input_tax_credit(
            Decimal("240"),
            gst_rate_pct=Decimal("18"),
            cost_is_gst_inclusive=None,
            tax_invoice_available=True,
        )


def test_unknown_inclusivity_is_harmless_without_an_invoice():
    assert (
        input_tax_credit(
            Decimal("240"),
            gst_rate_pct=Decimal("18"),
            cost_is_gst_inclusive=None,
            tax_invoice_available=False,
        )
        == 0
    )


def test_missing_dispatch_is_not_seven_days(tmp_path: Path):
    header = (
        "Supplier,Product,Rate (INR),GST,MOQ,Tax Invoice,Currency,GST Inclusive,Freight\n"
    )
    report = ingest_suppliers(
        write(tmp_path, header + "Acme,W,100,18,10,yes,INR,yes,0\n")
    )
    candidate = report.accepted[0]
    assert candidate.lead_time_days is None
    assert candidate.offer.dispatch_days is None
    assert any("no lead time is assumed" in w for w in report.warnings)


def test_a_stated_dispatch_time_is_preserved(tmp_path: Path):
    report = ingest_suppliers(write(tmp_path, FULL_HEADER + FULL_ROW))
    assert report.accepted[0].lead_time_days == 5


def test_supplier_candidate_does_not_default_a_lead_time():
    """The dataclass default was a third fabrication path."""
    from catalog.schemas import SupplierOffer, new_id
    from sourcing.matching import SupplierCandidate

    candidate = SupplierCandidate(
        supplier_name="Acme",
        title="Widget",
        offer=SupplierOffer(
            supplier_id=new_id(),
            product_id=new_id(),
            unit_cost_inr=Decimal("10"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
            moq_units=1,
        ),
    )
    assert candidate.lead_time_days is None


def test_ranking_does_not_invent_a_lead_time_penalty():
    """The fourth path: rank_suppliers defaulted to 7 and penalised for it."""
    from part2.intelligence import rank_suppliers

    offers = [c.offer for c in ingest_suppliers(FIXTURE).accepted[:1]]
    scores = rank_suppliers(offers, lead_times={}, capital_cap_inr=Decimal("999999"))
    assert scores[0].lead_time_days is None
    assert any("not stated" in n for n in scores[0].notes)


def test_a_stated_lead_time_still_penalises():
    """Removing the default must not remove the signal."""
    from part2.intelligence import rank_suppliers

    offers = [c.offer for c in ingest_suppliers(FIXTURE).accepted[:1]]
    supplier_id = str(offers[0].supplier_id)
    fast = rank_suppliers(
        offers, lead_times={supplier_id: 2}, capital_cap_inr=Decimal("999999")
    )[0]
    slow = rank_suppliers(
        offers, lead_times={supplier_id: 14}, capital_cap_inr=Decimal("999999")
    )[0]
    assert fast.total_score > slow.total_score


# ==========================================================================
# Field classification contract
# ==========================================================================


def test_every_audited_field_is_explicitly_classified():
    from sourcing.supplier_ingest import FIELD_CLASSIFICATION

    for field in (
        "gst_rate_pct",
        "cost_is_gst_inclusive",
        "moq_units",
        "tax_invoice",
        "dispatch_days",
        "currency",
        "hsn_code",
        "weight_g",
        "available_qty",
    ):
        assert field in FIELD_CLASSIFICATION, f"{field} is unclassified"


def test_derived_values_are_never_read_from_the_catalogue():
    """A DERIVED value must not be presentable as a supplier-stated fact."""
    import ast

    from sourcing.supplier_ingest import FIELD_CLASSIFICATION, FieldClass

    derived = {f for f, c in FIELD_CLASSIFICATION.items() if c is FieldClass.DERIVED}
    tree = ast.parse(Path("sourcing/supplier_ingest.py").read_text(encoding="utf-8"))
    read_from_row = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert not (derived & read_from_row), (
        f"derived fields must not be read from the file: {derived & read_from_row}"
    )


# ==========================================================================
# Inbound freight: UNKNOWN is not zero
# ==========================================================================


def test_missing_inbound_freight_is_refused_not_zeroed(tmp_path: Path):
    """An unstated freight charge silently understates every margin."""
    header = "Supplier,Product,Rate (INR),GST,MOQ,Tax Invoice,GST Inclusive\n"
    report = ingest_suppliers(write(tmp_path, header + "Acme,W,100,18,10,yes,yes\n"))
    assert report.accepted == []
    assert report.rejected[0].reason is RejectionReason.MISSING_FREIGHT


def test_an_explicitly_stated_zero_freight_is_kept(tmp_path: Path):
    """A confirmed zero is a real claim and must survive."""
    report = ingest_suppliers(write(tmp_path, FULL_HEADER + FULL_ROW))
    assert report.accepted[0].offer.inbound_freight_inr == Decimal("0")


def test_a_stated_freight_is_preserved(tmp_path: Path):
    row = "Acme,Widget,100,10,7323,200,18,5,yes,INR,yes,12.50\n"
    report = ingest_suppliers(write(tmp_path, FULL_HEADER + row))
    assert report.accepted[0].offer.inbound_freight_inr == Decimal("12.50")


def test_unknown_and_confirmed_zero_freight_are_distinguishable():
    """The whole point: None and Decimal('0') are different states."""
    from catalog.schemas import SupplierOffer, new_id

    def offer(freight):
        return SupplierOffer(
            supplier_id=new_id(),
            product_id=new_id(),
            unit_cost_inr=Decimal("240"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
            moq_units=10,
            inbound_freight_inr=freight,
        )

    assert offer(None).inbound_freight_inr is None
    assert offer(Decimal("0")).inbound_freight_inr == Decimal("0")
    assert offer(None).inbound_freight_inr != offer(Decimal("0")).inbound_freight_inr


def test_unknown_freight_cannot_produce_falsely_precise_economics():
    """Landed cost must refuse rather than quietly treat unknown as zero."""
    from catalog.schemas import SupplierOffer, new_id
    from econ.landed_cost import compute_landed_cost

    unknown = SupplierOffer(
        supplier_id=new_id(),
        product_id=new_id(),
        unit_cost_inr=Decimal("240"),
        cost_is_gst_inclusive=True,
        gst_rate_pct=Decimal("18"),
        tax_invoice_available=True,
        moq_units=10,
        inbound_freight_inr=None,
    )
    with pytest.raises(ValueError, match="did not state inbound freight"):
        compute_landed_cost(unknown, expected_monthly_velocity=Decimal("10"))


def test_confirmed_zero_freight_computes_normally():
    """Removing the default must not break the legitimate zero case."""
    from catalog.schemas import SupplierOffer, new_id
    from econ.landed_cost import compute_landed_cost

    stated = SupplierOffer(
        supplier_id=new_id(),
        product_id=new_id(),
        unit_cost_inr=Decimal("240"),
        cost_is_gst_inclusive=True,
        gst_rate_pct=Decimal("18"),
        tax_invoice_available=True,
        moq_units=10,
        inbound_freight_inr=Decimal("0"),
    )
    landed = compute_landed_cost(stated, expected_monthly_velocity=Decimal("10"))
    assert landed.total_landed_cost_inr > 0
    assert landed.inbound_freight_per_unit_inr == Decimal("0")


# ==========================================================================
# MOQ is mandatory on the schema, not just in ingestion
# ==========================================================================


def test_supplier_offer_cannot_be_built_without_moq():
    """The schema default of 1 understated locked capital on every offer."""
    from pydantic import ValidationError

    from catalog.schemas import SupplierOffer, new_id

    with pytest.raises(ValidationError) as excinfo:
        SupplierOffer(
            supplier_id=new_id(),
            product_id=new_id(),
            unit_cost_inr=Decimal("240"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
            inbound_freight_inr=Decimal("0"),
        )
    assert "moq_units" in str(excinfo.value)


def test_supplier_offer_has_no_factual_defaults():
    """Every optional field must be an honest UNKNOWN, never a value."""
    from catalog.schemas import SupplierOffer

    factual = {
        "moq_units",
        "unit_cost_inr",
        "gst_rate_pct",
        "tax_invoice_available",
        "cost_is_gst_inclusive",
        "inbound_freight_inr",
        "dispatch_days",
        "available_qty",
    }
    for name in factual:
        field = SupplierOffer.model_fields[name]
        if field.is_required():
            continue
        assert field.get_default() is None, (
            f"{name} defaults to {field.get_default()!r}; a factual supplier "
            "field must be required or UNKNOWN, never a substituted value"
        )
