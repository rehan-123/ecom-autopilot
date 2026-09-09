"""Pydantic schema tests: enums, required fields, constraints, timezone rules."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from catalog.enums import (
    AIService,
    ApprovalDecision,
    ComplianceResult,
    EventType,
    GateResult,
    GSTStatus,
    ScenarioName,
    ScoreResult,
    SignalSource,
    SignalStatus,
    SKUState,
    SupplierType,
    TransitionTrigger,
)
from catalog.schemas import (
    SKU,
    AIInvocation,
    Approval,
    AuditEvent,
    EconomicsSnapshot,
    MarketPriceReference,
    MarketSignal,
    OpportunityScore,
    Product,
    SKUStateTransition,
    Supplier,
    SupplierOffer,
)

UTC_NOW = datetime.now(UTC)


# -- valid construction ----------------------------------------------------


def test_valid_supplier_is_accepted():
    supplier = Supplier(
        name="Wholesale B",
        type=SupplierType.WHOLESALER,
        gst_status=GSTStatus.REGISTERED,
        gstin="07AABCU9603R1ZX",
        issues_tax_invoice=True,
    )
    assert supplier.type is SupplierType.WHOLESALER
    assert isinstance(supplier.id, uuid.UUID)


def test_valid_product_is_accepted():
    product = Product(canonical_name="Trimmer", category="personal_care", hsn_code="8510")
    assert product.hsn_code == "8510"


def test_ids_are_uuids_and_unique():
    a = Product(canonical_name="A", category="x")
    b = Product(canonical_name="B", category="x")
    assert isinstance(a.id, uuid.UUID) and isinstance(b.id, uuid.UUID)
    assert a.id != b.id


# -- enum validation -------------------------------------------------------


def test_invalid_supplier_type_is_rejected():
    with pytest.raises(ValidationError):
        Supplier(
            name="X",
            type="RANDOM_TYPE",
            gst_status=GSTStatus.UNKNOWN,
            issues_tax_invoice=False,
        )


def test_invalid_sku_state_is_rejected():
    with pytest.raises(ValidationError):
        SKUStateTransition(
            sku_id=uuid.uuid4(),
            to_state="PUBLISHED",
            trigger=TransitionTrigger.RULE,
            actor="SYSTEM",
        )


def test_part_two_states_do_not_exist():
    """No lifecycle state may represent publication, purchase or fulfilment."""
    forbidden = {"PUBLISHED", "ORDERED", "PURCHASED", "SHIPPED", "DELIVERED"}
    assert forbidden.isdisjoint({member.value for member in SKUState})


def test_part_two_event_types_do_not_exist():
    forbidden = {"ORDER_PLACED", "PAYMENT_MADE", "LISTING_PUBLISHED"}
    assert forbidden.isdisjoint({member.value for member in EventType})


# -- required fields -------------------------------------------------------


def test_missing_required_field_is_rejected():
    with pytest.raises(ValidationError):
        Product(category="x")  # canonical_name missing


def test_extra_field_is_rejected():
    with pytest.raises(ValidationError):
        Product(canonical_name="A", category="x", surprise="value")


def test_models_are_frozen():
    product = Product(canonical_name="A", category="x")
    with pytest.raises(ValidationError):
        product.canonical_name = "B"


# -- numeric constraints ---------------------------------------------------


def test_negative_cost_is_rejected():
    with pytest.raises(ValidationError):
        SupplierOffer(
            supplier_id=uuid.uuid4(),
            product_id=uuid.uuid4(),
            unit_cost_inr=Decimal("-1"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
        )


def test_zero_moq_is_rejected():
    with pytest.raises(ValidationError):
        SupplierOffer(
            supplier_id=uuid.uuid4(),
            product_id=uuid.uuid4(),
            unit_cost_inr=Decimal("100"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
            moq_units=0,
        )


def test_percentage_above_100_is_rejected():
    with pytest.raises(ValidationError):
        SupplierOffer(
            supplier_id=uuid.uuid4(),
            product_id=uuid.uuid4(),
            unit_cost_inr=Decimal("100"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("101"),
            tax_invoice_available=True,
        )


def test_hsn_must_be_numeric():
    with pytest.raises(ValidationError):
        Product(canonical_name="A", category="x", hsn_code="85AB")


# -- datetimes -------------------------------------------------------------


def test_naive_datetime_is_rejected():
    with pytest.raises(ValidationError, match="naive datetime"):
        Product(canonical_name="A", category="x", created_at=datetime(2026, 8, 23))


def test_aware_datetime_is_normalised_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    product = Product(
        canonical_name="A",
        category="x",
        created_at=datetime(2026, 8, 23, 17, 30, tzinfo=ist),
    )
    assert product.created_at.tzinfo == UTC
    assert product.created_at.hour == 12


def test_default_timestamps_are_aware():
    assert Product(canonical_name="A", category="x").created_at.tzinfo is not None


# -- cross-field business rules -------------------------------------------


def test_registered_supplier_requires_gstin():
    with pytest.raises(ValidationError, match="must have a gstin"):
        Supplier(
            name="X",
            type=SupplierType.WHOLESALER,
            gst_status=GSTStatus.REGISTERED,
            issues_tax_invoice=False,
        )


def test_unregistered_supplier_cannot_issue_tax_invoice():
    """No tax invoice means no input tax credit. The schema refuses to record
    a combination that would let a later calculation assume otherwise."""
    with pytest.raises(ValidationError, match="tax invoice"):
        Supplier(
            name="Retail Source",
            type=SupplierType.MARKETPLACE_RETAIL,
            gst_status=GSTStatus.UNREGISTERED,
            issues_tax_invoice=True,
        )


def test_marketplace_retail_supplier_is_representable():
    """Retail sources can be recorded — and flagged — but never as wholesale."""
    supplier = Supplier(
        name="Retail Marketplace Source",
        type=SupplierType.MARKETPLACE_RETAIL,
        gst_status=GSTStatus.UNREGISTERED,
        issues_tax_invoice=False,
    )
    assert supplier.type is SupplierType.MARKETPLACE_RETAIL
    assert supplier.issues_tax_invoice is False


def test_data_unavailable_signal_cannot_carry_a_value():
    with pytest.raises(ValidationError, match="must not carry a value"):
        MarketSignal(
            sku_id=uuid.uuid4(),
            signal_type="SEARCH_INTEREST",
            value_numeric=Decimal("50"),
            status=SignalStatus.DATA_UNAVAILABLE,
            source=SignalSource.MANUAL,
            collected_by="rehan",
        )


def test_observed_signal_may_carry_a_value():
    signal = MarketSignal(
        sku_id=uuid.uuid4(),
        signal_type="COMPETITOR_COUNT",
        value_numeric=Decimal("42"),
        status=SignalStatus.OBSERVED,
        source=SignalSource.MANUAL,
        collected_by="rehan",
    )
    assert signal.value_numeric == Decimal("42")


def test_insufficient_data_score_cannot_carry_a_value():
    with pytest.raises(ValidationError, match="must not carry a score value"):
        OpportunityScore(
            sku_id=uuid.uuid4(),
            weights_version="v1",
            value=Decimal("94"),
            coverage_pct=Decimal("45"),
            result=ScoreResult.INSUFFICIENT_DATA,
        )


def test_scored_result_requires_a_value():
    with pytest.raises(ValidationError, match="SCORED requires a value"):
        OpportunityScore(
            sku_id=uuid.uuid4(),
            weights_version="v1",
            coverage_pct=Decimal("95"),
            result=ScoreResult.SCORED,
        )


def test_score_carries_coverage_alongside_value():
    score = OpportunityScore(
        sku_id=uuid.uuid4(),
        weights_version="v1",
        value=Decimal("78"),
        coverage_pct=Decimal("95"),
        result=ScoreResult.SCORED,
    )
    assert score.coverage_pct == Decimal("95")


def test_failed_economics_gate_must_name_the_gate():
    with pytest.raises(ValidationError, match="must name the gate"):
        EconomicsSnapshot(
            sku_id=uuid.uuid4(),
            landed_cost_id=uuid.uuid4(),
            scenario=ScenarioName.EXPECTED,
            calc_version="1.0.0",
            rate_card_version="rc1",
            tax_table_version="tt1",
            return_benchmark_version="rb1",
            threshold_profile_version="tp1",
            scenario_profile_version="sp1",
            selling_price_inr=Decimal("799"),
            net_profit_inr=Decimal("-20"),
            net_margin_pct=Decimal("-2.5"),
            roi_pct=Decimal("-7"),
            breakeven_price_inr=Decimal("820"),
            min_profitable_price_inr=Decimal("900"),
            gate_result=GateResult.FAIL,
        )


def test_percentiles_must_be_ordered():
    with pytest.raises(ValidationError, match="p25 <= median <= p75"):
        MarketPriceReference(
            sku_id=uuid.uuid4(),
            collected_by="rehan",
            n_comparables=3,
            price_p25_inr=Decimal("700"),
            price_median_inr=Decimal("500"),
            price_p75_inr=Decimal("900"),
        )


def test_transition_must_change_state():
    with pytest.raises(ValidationError, match="must change state"):
        SKUStateTransition(
            sku_id=uuid.uuid4(),
            from_state=SKUState.APPROVED,
            to_state=SKUState.APPROVED,
            trigger=TransitionTrigger.RULE,
            actor="SYSTEM",
        )


# -- audit event -----------------------------------------------------------


@pytest.mark.parametrize("actor", ["SYSTEM", "HUMAN:rehan", "AI:JUDGE_IDENTITY"])
def test_well_formed_actors_are_accepted(actor: str):
    event = AuditEvent(event_type=EventType.SUPPLIER_CREATED, actor=actor)
    assert event.actor == actor


@pytest.mark.parametrize("actor", ["human", "HUMAN:", "AI", "robot:x", ""])
def test_malformed_actors_are_rejected(actor: str):
    with pytest.raises(ValidationError):
        AuditEvent(event_type=EventType.SUPPLIER_CREATED, actor=actor)


def test_audit_event_defaults_are_empty_not_none():
    event = AuditEvent(event_type=EventType.ERROR, actor="SYSTEM")
    assert event.inputs_json == {}
    assert event.reference_versions_json == {}
    assert event.result_json == {}


# -- reproducibility fields exist -----------------------------------------


def test_economics_snapshot_pins_every_reference_version():
    """Without these pins a replayed calculation silently uses today's rates."""
    required = {
        "rate_card_version",
        "tax_table_version",
        "return_benchmark_version",
        "threshold_profile_version",
        "scenario_profile_version",
        "calc_version",
    }
    assert required <= set(EconomicsSnapshot.model_fields)


def test_sku_has_no_state_column():
    """State is projected from the transition stream, not stored on the SKU."""
    assert "state" not in SKU.model_fields
    assert "state_updated_at" not in SKU.model_fields


def test_supplier_offer_has_no_superseded_backreference():
    """Setting one would require mutating an already-written immutable row."""
    assert "superseded_by_offer_id" not in SupplierOffer.model_fields


def test_supplier_model_holds_no_credentials():
    leaked = {"password", "api_key", "token", "secret", "bank_account"}
    assert leaked.isdisjoint(set(Supplier.model_fields))


# -- remaining entities construct -----------------------------------------


def test_ai_invocation_records_model_and_prompt_versions():
    invocation = AIInvocation(
        service=AIService.EXTRACTOR,
        model="claude-sonnet-4-6",
        model_version="2026-05-01",
        prompt_version="extractor-v1",
        prompt_hash="a" * 64,
        parsed_ok=True,
    )
    assert invocation.service is AIService.EXTRACTOR
    assert invocation.cost_inr == Decimal("0")


def test_approval_allows_null_assessment_refs_this_milestone():
    approval = Approval(
        sku_id=uuid.uuid4(), decision=ApprovalDecision.DEFERRED, approver="rehan"
    )
    assert approval.economics_snapshot_id is None


def test_compliance_result_vocabulary_is_complete():
    assert {member.value for member in ComplianceResult} == {
        "PASS",
        "FAIL",
        "UNKNOWN",
        "REQUIRES_HUMAN_REVIEW",
    }
