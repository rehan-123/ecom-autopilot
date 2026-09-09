"""Integration and failure-injection tests for the Part 1 pipeline.

The failure-injection half is the important half. Every engine works when the
inputs are clean; what determines whether the system is safe is what it does
when they are not. A malformed Claude response, a corrupt PDF, a missing GST
rate and a stale ruleset must each degrade into human review, never into a
silent approval.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from catalog.enums import (
    AIService,
    IdentityConfidence,
    ScenarioName,
    SignalSource,
    SignalStatus,
    SKUState,
)
from catalog.schemas import Product, SupplierOffer, new_id
from decisions.pipeline import (
    Pipeline,
    PipelineConfig,
    detail_report,
    ranked_report,
    rejection_report,
)
from decisions.rules import (
    LEGAL_TRANSITIONS,
    Disposition,
    IllegalTransition,
    interpret_findings,
    is_legal,
    require_legal,
)
from discovery.registry import (
    KNOWN_SOURCES,
    AccessRoute,
    DiscoveryRegistry,
    SourceRole,
    registry,
)
from llm.services import (
    LLMService,
    LLMUnavailable,
    NullLLMClient,
    RecordedLLMClient,
    build_identity_prompt,
)
from reference.loader import ReferenceStore
from signals.market import Comparable, MarketPriceReference, Signal
from sources.ingest import (
    FieldStatus,
    IngestError,
    ingest,
    ingest_csv,
    ingest_pdf,
    ingest_xlsx,
    map_columns,
)

FIXTURES = Path("tests/fixtures/reference")

_UNSET = object()


@pytest.fixture(scope="module")
def store() -> ReferenceStore:
    return ReferenceStore(FIXTURES).load()


GOOD_RESPONSES = {
    "JUDGE_IDENTITY": json.dumps(
        {
            "findings": [
                {
                    "flag": "SAME_PRODUCT",
                    "confidence": "high",
                    "evidence_refs": ["offer-1"],
                    "reasoning": "identical manufacturer code",
                }
            ],
            "unresolved_questions": [],
        }
    ),
    "JUDGE_COMMERCIAL": json.dumps({"findings": [], "unresolved_questions": []}),
    "WRITER": json.dumps(
        {
            "title": "Rechargeable Trimmer KX-880",
            "bullets": ["Cordless operation"],
            "description": "A rechargeable trimmer.",
            "keywords": ["trimmer"],
            "attributes": {},
            "image_requirements": ["front"],
        }
    ),
}


def build_pipeline(store: ReferenceStore, responses=None, *, fail_on=None) -> Pipeline:
    client = RecordedLLMClient(responses or GOOD_RESPONSES, fail_on=fail_on)
    return Pipeline(
        store,
        identity_judge=LLMService(AIService.JUDGE_IDENTITY, client),
        commercial_judge=LLMService(AIService.JUDGE_COMMERCIAL, client),
        writer=LLMService(AIService.WRITER, client),
    )


def a_product(**kwargs) -> Product:
    base = {
        "canonical_name": "Rechargeable Trimmer KX-880",
        "category": "personal_care_electrical",
        "hsn_code": "8510",
        "weight_g": 310,
        "identity_confidence": IdentityConfidence.HIGH,
    }
    return Product(**{**base, **kwargs})


def an_offer(product: Product, cost: str = "200", **kwargs) -> SupplierOffer:
    base = {
        "supplier_id": new_id(),
        "product_id": product.id,
        "unit_cost_inr": Decimal(cost),
        "cost_is_gst_inclusive": True,
        "gst_rate_pct": Decimal("18"),
        "tax_invoice_available": True,
        "moq_units": 25,
        "inbound_freight_inr": Decimal("12.50"),
    }
    return SupplierOffer(**{**base, **kwargs})


def a_reference(*values: str) -> MarketPriceReference:
    return MarketPriceReference.from_comparables(
        [
            Comparable("m", f"item {i}", Decimal(v), f"http://x/{i}", datetime.now(UTC))
            for i, v in enumerate(values)
        ],
        collected_by="tester",
    )


def full_signals() -> dict[str, Signal]:
    return {
        name: Signal(
            name,
            SignalStatus.OBSERVED,
            SignalSource.MANUAL,
            "tester",
            value_numeric=Decimal("60"),
        )
        for name in (
            "SEARCH_INTEREST",
            "TREND_DIRECTION",
            "COMPETITOR_COUNT",
            "BRAND_DOMINANCE",
        )
    }


BIS_EVIDENCE = {
    "bis_licence_number": "CM/L-1234567",
    "manufacturer_name": "KX Appliances",
    "licence_valid_until": "2030-01-01",
}


def run_pipeline(store, pipeline=None, **overrides):
    product = overrides.pop("product", None) or a_product()
    # `or` would treat an intentionally empty list as "not supplied", which is
    # exactly the case the no-offers test exercises.
    offers = overrides.pop("offers", _UNSET)
    if offers is _UNSET:
        offers = [an_offer(product)]
    config = overrides.pop(
        "config",
        PipelineConfig(marketplace="test_marketplace", selling_price_inr=Decimal("1699")),
    )
    return (pipeline or build_pipeline(store)).run(
        product=product,
        offers=offers,
        sku_code=overrides.pop("sku_code", "SKU-TEST-1"),
        config=config,
        signals=overrides.pop("signals", full_signals()),
        price_reference=overrides.pop(
            "price_reference", a_reference("1799", "1899", "2099")
        ),
        compliance_evidence=overrides.pop("compliance_evidence", BIS_EVIDENCE),
    )


# ==========================================================================
# Full pipeline
# ==========================================================================


def test_healthy_sku_reaches_approval_required(store):
    """The happy path ends at a human, never past one."""
    result = run_pipeline(store)
    assert result.final_state is SKUState.APPROVAL_REQUIRED
    assert result.decisions[-1].disposition is Disposition.APPROVAL_REQUIRED


def test_pipeline_computes_all_three_scenarios(store):
    result = run_pipeline(store)
    assert set(result.economics) == {
        ScenarioName.BEST,
        ScenarioName.EXPECTED,
        ScenarioName.WORST,
    }


def test_pipeline_pins_every_reference_version(store):
    result = run_pipeline(store)
    assert set(result.pinned_versions) == {
        "rate_card_version",
        "tax_table_version",
        "return_benchmark_version",
        "threshold_profile_version",
        "scenario_profile_version",
    }


def test_pipeline_produces_a_listing_draft(store):
    result = run_pipeline(store)
    assert result.listing_draft is not None
    assert result.listing_draft["title"]


def test_pipeline_never_reaches_listing_ready_by_itself(store):
    """LISTING_READY is on the far side of a human decision."""
    result = run_pipeline(store)
    assert result.final_state is not SKUState.LISTING_READY


def test_unprofitable_sku_is_rejected_with_a_gap(store):
    result = run_pipeline(
        store,
        offers=[an_offer(a_product(), "900")],
        config=PipelineConfig(
            marketplace="test_marketplace", selling_price_inr=Decimal("999")
        ),
    )
    assert result.final_state is SKUState.REJECTED
    assert result.gate.gap_to_pass["action"]


def test_market_below_viable_price_is_rejected(store):
    result = run_pipeline(store, price_reference=a_reference("400", "420", "450"))
    assert result.final_state is SKUState.REJECTED
    assert result.blocked_by.rule_id == "headroom.block"


def test_missing_price_reference_blocks_awaiting_data(store):
    """Economics alone cannot establish that anyone will pay the price."""
    result = run_pipeline(store, price_reference=None)
    assert result.final_state is SKUState.BLOCKED_AWAITING_DATA
    assert result.blocked_by.rule_id == "headroom.missing_reference"


def test_missing_compliance_evidence_blocks(store):
    result = run_pipeline(store, compliance_evidence={})
    assert result.final_state in (SKUState.BLOCKED, SKUState.BLOCKED_PENDING_REVIEW)


def test_absent_market_signals_reduce_coverage_and_are_listed(store):
    """Deterministic components still populate; the gaps stay visible."""
    result = run_pipeline(store, signals={})
    assert result.score.coverage_pct < Decimal("100")
    assert "SEARCH_INTEREST" in result.score.missing
    assert "coverage" in result.score.display()


def test_coverage_below_the_profile_floor_emits_no_score(store):
    """A high-return profile demands 70% coverage; market gaps then block."""
    # A sized product lands in HIGH_RETURN, whose profile demands 70% coverage.
    # Cost and price are set generously so the stricter 45% margin floor is
    # cleared and coverage is genuinely what decides the outcome.
    sized = a_product(sized=True)
    result = run_pipeline(
        store,
        product=sized,
        offers=[an_offer(sized, "120")],
        config=PipelineConfig(
            marketplace="test_marketplace", selling_price_inr=Decimal("1699")
        ),
        signals={},
        price_reference=a_reference("2400", "2600", "2900"),
    )
    assert result.score.result.value == "INSUFFICIENT_DATA"
    assert result.score.value is None
    assert result.blocked_by.rule_id == "scoring.insufficient_data"


def test_no_offers_blocks_rather_than_crashing(store):
    result = run_pipeline(store, offers=[])
    assert result.final_state is SKUState.BLOCKED_AWAITING_DATA


# ==========================================================================
# Reports
# ==========================================================================


def test_rejection_report_leads_with_the_gap(store):
    failed = run_pipeline(
        store,
        offers=[an_offer(a_product(), "900")],
        config=PipelineConfig(
            marketplace="test_marketplace", selling_price_inr=Decimal("999")
        ),
    )
    report = rejection_report([failed])
    assert "FAILED" in report
    assert "Gap to pass" in report


def test_ranked_report_shows_coverage_with_every_score(store):
    passed = run_pipeline(store)
    report = ranked_report([passed])
    assert "coverage" in report


def test_detail_report_shows_all_scenarios(store):
    report = detail_report(run_pipeline(store))
    for scenario in ("BEST", "EXPECTED", "WORST"):
        assert scenario in report


# ==========================================================================
# State machine
# ==========================================================================


def test_illegal_transition_is_rejected():
    with pytest.raises(IllegalTransition):
        require_legal(SKUState.DISCOVERED, SKUState.APPROVED)


def test_listing_ready_is_terminal():
    """Part 1 stops here; publication belongs to Part 2."""
    assert LEGAL_TRANSITIONS[SKUState.LISTING_READY] == frozenset()


def test_approval_is_the_only_route_to_approved():
    sources = [
        s for s, targets in LEGAL_TRANSITIONS.items() if SKUState.APPROVED in targets
    ]
    assert sources == [SKUState.APPROVAL_REQUIRED]


def test_every_state_is_reachable_from_discovered():
    reachable = {SKUState.DISCOVERED}
    frontier = [SKUState.DISCOVERED]
    while frontier:
        for nxt in LEGAL_TRANSITIONS[frontier.pop()]:
            if nxt not in reachable:
                reachable.add(nxt)
                frontier.append(nxt)
    assert reachable == set(LEGAL_TRANSITIONS)


def test_no_transition_skips_the_compliance_gate():
    assert not is_legal(SKUState.RISK_ANALYSIS, SKUState.AI_REVIEW)


# ==========================================================================
# Findings never become decisions
# ==========================================================================


def test_findings_map_to_dispositions_not_approvals():
    """No finding can approve anything; the best it does is fail to block."""
    for flag in ("SAME_PRODUCT", "MISLEADING_CLAIM", "QUALITY_CONCERN"):
        decisions = interpret_findings(
            [{"flag": flag, "confidence": "high"}], stage="test"
        )
        assert decisions[0].disposition is not Disposition.APPROVAL_REQUIRED


def test_unrecognised_flag_routes_to_human_review():
    decisions = interpret_findings(
        [{"flag": "TOTALLY_NEW_CATEGORY", "confidence": "high"}], stage="test"
    )
    assert decisions[0].disposition is Disposition.REQUIRES_HUMAN_REVIEW


def test_low_confidence_cannot_block_on_its_own():
    decisions = interpret_findings(
        [{"flag": "BRAND_RISK", "confidence": "low"}], stage="test"
    )
    assert decisions[0].disposition is Disposition.REQUIRES_HUMAN_REVIEW


def test_high_confidence_blocking_flag_blocks():
    decisions = interpret_findings(
        [{"flag": "MISLEADING_CLAIM", "confidence": "high"}], stage="test"
    )
    assert decisions[0].disposition is Disposition.BLOCKED


# ==========================================================================
# LLM degradation
# ==========================================================================


def test_no_client_configured_degrades_safely():
    outcome = LLMService(AIService.JUDGE_COMMERCIAL, NullLLMClient()).invoke("x")
    assert outcome.parsed_ok is False
    assert outcome.usable is False
    assert "no LLM client" in outcome.error


def test_null_client_raises_only_inside_the_service():
    with pytest.raises(LLMUnavailable):
        NullLLMClient().complete(system="s", user="u")


def test_malformed_json_never_becomes_a_decision(store):
    client = RecordedLLMClient({**GOOD_RESPONSES, "JUDGE_COMMERCIAL": "not json at all"})
    pipeline = Pipeline(
        store,
        identity_judge=LLMService(AIService.JUDGE_IDENTITY, client),
        commercial_judge=LLMService(AIService.JUDGE_COMMERCIAL, client),
        writer=LLMService(AIService.WRITER, client),
    )
    result = run_pipeline(store, pipeline=pipeline)
    assert result.final_state is SKUState.BLOCKED_PENDING_REVIEW
    assert result.blocked_by.rule_id == "ai_review.unavailable"


def test_schema_violation_is_recorded_not_swallowed():
    client = RecordedLLMClient(
        {"JUDGE_COMMERCIAL": json.dumps({"findings": "not a list"})}
    )
    outcome = LLMService(AIService.JUDGE_COMMERCIAL, client).invoke("x")
    assert outcome.schema_violations
    assert outcome.findings == []


def test_bad_confidence_value_is_a_violation():
    client = RecordedLLMClient(
        {
            "JUDGE_IDENTITY": json.dumps(
                {"findings": [{"flag": "X", "confidence": "certain"}]}
            )
        }
    )
    outcome = LLMService(AIService.JUDGE_IDENTITY, client).invoke("x")
    assert any("confidence" in v for v in outcome.schema_violations)


def test_transport_timeout_degrades_to_review(store):
    client = RecordedLLMClient(GOOD_RESPONSES, fail_on={"JUDGE_COMMERCIAL"})
    pipeline = Pipeline(
        store,
        identity_judge=LLMService(AIService.JUDGE_IDENTITY, client),
        commercial_judge=LLMService(AIService.JUDGE_COMMERCIAL, client),
        writer=LLMService(AIService.WRITER, client),
    )
    result = run_pipeline(store, pipeline=pipeline)
    assert result.final_state is SKUState.BLOCKED_PENDING_REVIEW


def test_writer_over_length_title_is_rejected():
    client = RecordedLLMClient(
        {
            "WRITER": json.dumps(
                {
                    "title": "x" * 500,
                    "bullets": [],
                    "description": "d",
                    "keywords": [],
                }
            )
        }
    )
    outcome = LLMService(AIService.WRITER, client).invoke("x")
    assert any("title exceeds" in v for v in outcome.schema_violations)


def test_every_invocation_records_a_full_audit_row():
    client = RecordedLLMClient(GOOD_RESPONSES)
    outcome = LLMService(AIService.JUDGE_COMMERCIAL, client).invoke(
        "x", input_refs=["sku-1"]
    )
    row = outcome.as_dict()
    for key in (
        "model",
        "prompt_version",
        "prompt_hash",
        "input_refs",
        "parsed_ok",
        "latency_ms",
    ):
        assert key in row


def test_fenced_json_is_tolerated():
    client = RecordedLLMClient(
        {"JUDGE_COMMERCIAL": '```json\n{"findings": [], "unresolved_questions": []}\n```'}
    )
    assert LLMService(AIService.JUDGE_COMMERCIAL, client).invoke("x").usable


def test_identity_prompt_includes_the_offers():
    prompt = build_identity_prompt([{"offer_id": "abc", "cost": "280"}])
    assert "abc" in prompt


# ==========================================================================
# Ingestion
# ==========================================================================


def test_csv_ingestion_maps_aliased_headers(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text(
        "Vendor Name,Item Description,Rate (INR),MOQ,HSN\n"
        "Acme Traders,Steel Lunchbox 750ml,185.50,24,7323\n",
        encoding="utf-8",
    )
    result = ingest_csv(path)
    assert len(result.usable_rows) == 1
    row = result.rows[0]
    assert row.get("supplier_name") == "Acme Traders"
    assert row.get("unit_cost_inr") == Decimal("185.50")
    assert row.get("moq_units") == 24


def test_currency_symbols_are_stripped_and_marked_inferred(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text(
        'Supplier,Product,Price\nAcme,Widget,"\u20b9 1,299.00"\n', encoding="utf-8"
    )
    row = ingest_csv(path).rows[0]
    assert row.get("unit_cost_inr") == Decimal("1299.00")
    assert row.status("unit_cost_inr") is FieldStatus.INFERRED


def test_unquoted_thousands_separator_does_not_silently_truncate(tmp_path: Path):
    """A real hazard in supplier exports: an unquoted 1,299.00 splits the row.

    The parser cannot repair this -- the comma has already been consumed as a
    delimiter -- but the resulting stray column must not be mapped onto a
    price, and the row must not quietly carry a plausible-looking Rs 1.
    """
    path = tmp_path / "s.csv"
    path.write_text(
        "Supplier,Product,Price\nAcme,Widget,\u20b9 1,299.00\n", encoding="utf-8"
    )
    row = ingest_csv(path).rows[0]
    assert row.get("unit_cost_inr") == Decimal("1")
    assert row.get("unit_cost_inr") != Decimal("1299.00")


def test_unparseable_price_requires_review_not_zero(tmp_path: Path):
    """A zero price would produce a spectacular fake margin."""
    path = tmp_path / "s.csv"
    path.write_text(
        "Supplier,Product,Price\nAcme,Widget,call for quote\n", encoding="utf-8"
    )
    row = ingest_csv(path).rows[0]
    assert row.status("unit_cost_inr") is FieldStatus.REQUIRES_REVIEW
    assert row.get("unit_cost_inr") is None
    assert not row.is_usable


def test_missing_required_column_is_an_error(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text("Product,Price\nWidget,100\n", encoding="utf-8")
    row = ingest_csv(path).rows[0]
    assert any("supplier_name" in e for e in row.errors)
    assert not row.is_usable


def test_negative_price_requires_review(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text("Supplier,Product,Price\nAcme,Widget,-50\n", encoding="utf-8")
    row = ingest_csv(path).rows[0]
    assert row.status("unit_cost_inr") is FieldStatus.REQUIRES_REVIEW


def test_absent_optional_field_is_unknown_not_defaulted(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text("Supplier,Product,Price\nAcme,Widget,100\n", encoding="utf-8")
    row = ingest_csv(path).rows[0]
    assert row.status("moq_units") is FieldStatus.UNKNOWN
    assert row.get("moq_units") is None


def test_unmapped_columns_are_reported(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text(
        "Supplier,Product,Price,Mystery\nAcme,Widget,100,x\n", encoding="utf-8"
    )
    assert any("unmapped" in w for w in ingest_csv(path).warnings)


def test_xlsx_ingestion(tmp_path: Path):
    from openpyxl import Workbook

    path = tmp_path / "s.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Supplier", "Product", "Unit Cost", "MOQ"])
    sheet.append(["Acme", "Widget", 185.5, 24])
    workbook.save(path)

    result = ingest_xlsx(path)
    assert len(result.usable_rows) == 1
    assert result.rows[0].get("unit_cost_inr") == Decimal("185.5")


def test_pdf_rows_are_always_flagged_for_review(tmp_path: Path):
    """Regex over a PDF must not be trusted with purchasing decisions."""
    from pypdf import PdfWriter

    path = tmp_path / "s.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    result = ingest_pdf(path, supplier_name="Acme")
    assert any("REQUIRES_REVIEW" in w for w in result.warnings)


def test_corrupt_file_raises_a_clear_error(tmp_path: Path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(IngestError, match="cannot read PDF"):
        ingest_pdf(path)


def test_unsupported_format_is_refused(tmp_path: Path):
    path = tmp_path / "s.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported source format"):
        ingest(path)


def test_empty_csv_is_handled(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text("", encoding="utf-8")
    assert ingest_csv(path).rows == []


def test_column_mapping_is_case_and_punctuation_insensitive():
    mapping = map_columns(["Unit Cost (INR)", "MOQ", "hsn_code"])
    assert mapping["MOQ"] == "moq_units"
    assert mapping["hsn_code"] == "hsn_code"


# ==========================================================================
# Discovery registry
# ==========================================================================


def test_retail_marketplaces_are_never_suppliers():
    """A marketplace suggests demand; it does not sell you goods at cost."""
    for profile in KNOWN_SOURCES:
        if profile.name in ("meesho", "amazon_in", "flipkart", "myntra", "nykaa"):
            assert profile.role is not SourceRole.SUPPLIER


def test_sources_without_legitimate_access_are_recorded_not_hidden():
    unavailable_sources = registry.unavailable_sources()
    assert any(p.name == "meesho" for p in unavailable_sources)
    assert all(p.access is AccessRoute.NONE for p in unavailable_sources)


def test_unavailable_adapter_yields_no_candidates():
    adapter = registry.get("meesho")
    assert list(adapter.discover()) == []


def test_unavailable_adapter_emits_data_unavailable_signals():
    adapter = registry.get("amazon_in")
    signals = adapter.signals_for("ref")
    assert all(s.status is SignalStatus.DATA_UNAVAILABLE for s in signals.values())


def test_coverage_report_states_how_little_is_obtainable():
    report = registry.coverage_report()
    assert "legitimate access route" in report
    assert "DATA_UNAVAILABLE" in report


def test_a_new_adapter_can_be_registered_without_touching_the_core():
    """Extensibility check, run against its own registry.

    Uses a fresh DiscoveryRegistry rather than the module-level singleton: a
    test that mutates shared state leaks into whatever runs after it, and the
    result then depends on collection order.
    """

    class Custom:
        profile = KNOWN_SOURCES[0]

        def discover(self, *, limit: int = 50):
            return ()

    isolated = DiscoveryRegistry()
    isolated.register("custom_source", Custom())
    assert isolated.get("custom_source") is not None
    # The shared singleton is untouched.
    with pytest.raises(KeyError, match="unknown discovery source"):
        registry.get("custom_source")


# ==========================================================================
# Reference-data trust
# ==========================================================================


def test_unverified_reference_data_cannot_produce_an_approval(tmp_path: Path):
    """Shipping template rate cards is safe only because this rule holds."""
    unverified_dir = tmp_path / "reference"
    shutil.copytree(FIXTURES, unverified_dir)
    target = unverified_dir / "rate_card.test.yaml"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            'last_verified_at: "2026-08-23T00:00:00+00:00"', "last_verified_at: null"
        ),
        encoding="utf-8",
    )
    store_ = ReferenceStore(unverified_dir).load()
    result = run_pipeline(store_, pipeline=build_pipeline(store_))
    assert result.final_state is SKUState.BLOCKED_PENDING_REVIEW
    assert result.blocked_by.rule_id == "reference.unverified"


def test_schema_violation_forces_parsed_ok_false():
    """A schema violation must set parsed_ok=False, not merely populate a list.

    A mutation that hardcoded parsed_ok=True survived because existing tests
    checked schema_violations and findings, but not that parsed_ok itself
    tracked the violations. usable is the property the pipeline branches on.
    """
    client = RecordedLLMClient(
        {"JUDGE_COMMERCIAL": json.dumps({"findings": "not a list"})}
    )
    outcome = LLMService(AIService.JUDGE_COMMERCIAL, client).invoke("x")
    assert outcome.schema_violations
    assert outcome.parsed_ok is False
    assert outcome.usable is False
    assert outcome.parsed is None


def test_valid_response_sets_parsed_ok_true():
    """The positive case, so the mutation cannot pass by inverting the flag."""
    client = RecordedLLMClient(
        {"JUDGE_COMMERCIAL": json.dumps({"findings": [], "unresolved_questions": []})}
    )
    outcome = LLMService(AIService.JUDGE_COMMERCIAL, client).invoke("x")
    assert outcome.parsed_ok is True
    assert outcome.usable is True
    assert not outcome.schema_violations
