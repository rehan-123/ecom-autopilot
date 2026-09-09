"""SQLAlchemy Core table definitions.

Split from ``migrations.py`` deliberately: this module is *declarative* (what
the schema is), migrations is *procedural* (creating it, installing triggers,
recording versions). Mixing them makes both harder to read.

Tables are grouped into three tiers, and the tier determines the guarantees:

Tier 1 - immutable facts
    supplier, product, sku, supplier_offer, product_identity_evidence, all
    reference tables, all assessment snapshots, ai_invocation,
    sku_state_transition, approval.
    Never UPDATEd for business data. New information means a new row.

Tier 2 - the ledger
    audit_event. Append-only, enforced by database triggers and a hash chain.

Tier 3 - projections
    sku_current_state, supplier_offer_current. Derived, droppable, and
    rebuildable in full from Tier 2. Never a source of truth.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

from catalog.enums import (
    AIService,
    ApprovalDecision,
    ComplianceResult,
    EventType,
    EvidenceType,
    GateResult,
    GSTStatus,
    HeadroomResult,
    IdentityConfidence,
    RiskLevel,
    RiskTier,
    ScenarioName,
    ScoreResult,
    SignalSource,
    SignalStatus,
    SKUState,
    SupplierType,
    TransitionTrigger,
)
from database.types import DecimalText, JSONText, UTCDateTime, UUIDText

metadata = MetaData()

#: Tables that must never be mutated. Enforced by triggers in migrations.
IMMUTABLE_TABLES: tuple[str, ...] = (
    "audit_event",
    "supplier_offer",
    "sku_state_transition",
    "landed_cost",
    "economics_snapshot",
    "price_headroom_check",
    "risk_assessment",
    "compliance_assessment",
    "opportunity_score",
    "ai_invocation",
    "market_signal",
    "market_price_reference",
)

#: Tables rebuildable from the ledger. Safe to drop and recreate.
PROJECTION_TABLES: tuple[str, ...] = (
    "sku_current_state",
    "supplier_offer_current",
)


def _enum_check(column: str, enum_cls: type) -> CheckConstraint:
    """Constrain a text column to an enum's members.

    Belt and braces alongside Pydantic: raw SQL inserts bypass the schemas,
    and a controlled vocabulary that only holds at the application layer will
    eventually not hold.
    """
    members = ", ".join(f"'{member.value}'" for member in enum_cls)
    return CheckConstraint(f"{column} IN ({members})", name=f"ck_{column}_valid")


# --------------------------------------------------------------------------
# Tier 1: catalog
# --------------------------------------------------------------------------

supplier = Table(
    "supplier",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("name", String(200), nullable=False),
    Column("type", String(30), nullable=False),
    Column("gst_status", String(20), nullable=False),
    Column("gstin", String(15)),
    Column("issues_tax_invoice", Boolean, nullable=False),
    Column("state_code", String(2)),
    Column("contact", JSONText, nullable=False, default=dict),
    Column("notes", Text),
    Column("created_at", UTCDateTime, nullable=False),
    _enum_check("type", SupplierType),
    _enum_check("gst_status", GSTStatus),
    UniqueConstraint("name", name="uq_supplier_name"),
)

product = Table(
    "product",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("canonical_name", String(300), nullable=False),
    Column("category", String(120), nullable=False),
    Column("hsn_code", String(8)),
    Column("brand", String(200)),
    Column("manufacturer", String(200)),
    Column("country_of_origin", String(60)),
    Column("weight_g", Integer),
    Column("length_mm", Integer),
    Column("width_mm", Integer),
    Column("height_mm", Integer),
    Column("fragile", Boolean, nullable=False, default=False),
    Column("sized", Boolean, nullable=False, default=False),
    Column("warranty_months", Integer),
    Column("identity_confidence", String(20), nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    _enum_check("identity_confidence", IdentityConfidence),
)

sku = Table(
    "sku",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("product_id", UUIDText, ForeignKey("product.id"), nullable=False),
    Column("sku_code", String(64), nullable=False),
    Column("variant", JSONText, nullable=False, default=dict),
    Column("pack_qty", Integer, nullable=False, default=1),
    Column("created_at", UTCDateTime, nullable=False),
    CheckConstraint("pack_qty > 0", name="ck_pack_qty_positive"),
    UniqueConstraint("sku_code", name="uq_sku_code"),
    # No `state` column by design. Current state is projected from
    # sku_state_transition into sku_current_state.
)

product_identity_evidence = Table(
    "product_identity_evidence",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("product_id", UUIDText, ForeignKey("product.id"), nullable=False),
    Column("evidence_type", String(30), nullable=False),
    Column("value", Text, nullable=False),
    Column("source_offer_id", UUIDText, ForeignKey("supplier_offer.id")),
    Column("collected_at", UTCDateTime, nullable=False),
    _enum_check("evidence_type", EvidenceType),
)

supplier_offer = Table(
    "supplier_offer",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("supplier_id", UUIDText, ForeignKey("supplier.id"), nullable=False),
    Column("product_id", UUIDText, ForeignKey("product.id"), nullable=False),
    Column("unit_cost_inr", DecimalText, nullable=False),
    Column("cost_is_gst_inclusive", Boolean, nullable=False),
    Column("gst_rate_pct", DecimalText, nullable=False),
    Column("tax_invoice_available", Boolean, nullable=False),
    Column("moq_units", Integer, nullable=False, default=1),
    Column("available_qty", Integer),
    Column("inbound_freight_inr", DecimalText, nullable=False),
    Column("dispatch_days", Integer),
    Column("return_terms", Text),
    Column("warranty_terms", Text),
    Column("compliance_docs", JSONText, nullable=False, default=dict),
    Column("observed_at", UTCDateTime, nullable=False),
    Column("source_document_ref", String(400)),
    CheckConstraint("moq_units > 0", name="ck_moq_positive"),
    # No `superseded_by_offer_id`: setting it would require UPDATEing an
    # immutable row. Supersession is derived from observed_at ordering and
    # materialised in the supplier_offer_current projection.
)

# --------------------------------------------------------------------------
# Tier 1: versioned reference data
# --------------------------------------------------------------------------

rate_card = Table(
    "rate_card",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("marketplace", String(60), nullable=False),
    Column("version", String(40), nullable=False),
    Column("effective_from", UTCDateTime, nullable=False),
    Column("effective_to", UTCDateTime),
    Column("rules_json", JSONText, nullable=False, default=dict),
    Column("source_url", Text),
    Column("last_verified_at", UTCDateTime),
    Column("created_at", UTCDateTime, nullable=False),
    UniqueConstraint("marketplace", "version", name="uq_rate_card_version"),
)

tax_table = Table(
    "tax_table",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("version", String(40), nullable=False),
    Column("effective_from", UTCDateTime, nullable=False),
    Column("effective_to", UTCDateTime),
    Column("hsn_code", String(8), nullable=False),
    Column("gst_rate_pct", DecimalText, nullable=False),
    Column("cess_pct", DecimalText, nullable=False),
    Column("tcs_rate_pct", DecimalText, nullable=False),
    Column("source_url", Text),
    Column("last_verified_at", UTCDateTime),
    UniqueConstraint("version", "hsn_code", name="uq_tax_table_version_hsn"),
)

return_benchmark = Table(
    "return_benchmark",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("version", String(40), nullable=False),
    Column("category", String(120), nullable=False),
    Column("channel", String(60), nullable=False),
    Column("return_rate_pct", DecimalText, nullable=False),
    Column("rto_rate_pct", DecimalText, nullable=False),
    Column("cost_per_return_inr", DecimalText, nullable=False),
    Column("unsaleable_share_pct", DecimalText, nullable=False),
    Column("source", String(120), nullable=False),
    Column("confidence", String(40), nullable=False),
    Column("effective_from", UTCDateTime, nullable=False),
    Column("source_url", Text),
    Column("last_verified_at", UTCDateTime),
    UniqueConstraint(
        "version", "category", "channel", name="uq_return_benchmark_version"
    ),
)

threshold_profile = Table(
    "threshold_profile",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("version", String(40), nullable=False),
    Column("name", String(60), nullable=False),
    Column("min_net_margin_pct", DecimalText, nullable=False),
    Column("min_profit_per_order_inr", DecimalText, nullable=False),
    Column("max_return_rate_pct", DecimalText, nullable=False),
    Column("max_supplier_cost_pct", DecimalText, nullable=False),
    Column("max_ad_spend_pct", DecimalText, nullable=False),
    Column("min_roi_pct", DecimalText, nullable=False),
    Column("min_price_headroom_pct", DecimalText, nullable=False),
    Column("min_coverage_pct", DecimalText, nullable=False),
    Column("effective_from", UTCDateTime, nullable=False),
    Column("source_url", Text),
    Column("last_verified_at", UTCDateTime),
    UniqueConstraint("version", "name", name="uq_threshold_profile_version"),
)

scenario_profile = Table(
    "scenario_profile",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("version", String(40), nullable=False),
    Column("scenario", String(20), nullable=False),
    Column("return_rate_multiplier", DecimalText, nullable=False),
    Column("supplier_cost_delta_pct", DecimalText, nullable=False),
    Column("shipping_delta_pct", DecimalText, nullable=False),
    Column("ad_cost_multiplier", DecimalText, nullable=False),
    Column("velocity_multiplier", DecimalText, nullable=False),
    Column("effective_from", UTCDateTime, nullable=False),
    Column("source_url", Text),
    Column("last_verified_at", UTCDateTime),
    _enum_check("scenario", ScenarioName),
    UniqueConstraint("version", "scenario", name="uq_scenario_profile_version"),
)

# --------------------------------------------------------------------------
# Tier 1: assessment snapshots (structures only this milestone)
# --------------------------------------------------------------------------

landed_cost = Table(
    "landed_cost",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column(
        "supplier_offer_id", UUIDText, ForeignKey("supplier_offer.id"), nullable=False
    ),
    Column("scenario", String(20), nullable=False),
    Column("calc_version", String(40), nullable=False),
    Column("unit_cost_ex_gst_inr", DecimalText, nullable=False),
    Column("input_tax_credit_inr", DecimalText, nullable=False),
    Column("effective_cogs_inr", DecimalText, nullable=False),
    Column("inbound_freight_per_unit_inr", DecimalText, nullable=False),
    Column("moq_carry_cost_inr", DecimalText, nullable=False),
    Column("inspection_qc_cost_inr", DecimalText, nullable=False),
    Column("packaging_cost_inr", DecimalText, nullable=False),
    Column("inbound_defect_provision_inr", DecimalText, nullable=False),
    Column("total_landed_cost_inr", DecimalText, nullable=False),
    Column("assumptions_json", JSONText, nullable=False, default=dict),
    Column("computed_at", UTCDateTime, nullable=False),
    _enum_check("scenario", ScenarioName),
)

market_signal = Table(
    "market_signal",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("signal_type", String(60), nullable=False),
    Column("value_numeric", DecimalText),
    Column("value_text", Text),
    Column("status", String(30), nullable=False),
    Column("source", String(30), nullable=False),
    Column("source_ref", Text),
    Column("confidence", String(40)),
    Column("collected_at", UTCDateTime, nullable=False),
    Column("collected_by", String(120), nullable=False),
    _enum_check("status", SignalStatus),
    _enum_check("source", SignalSource),
    CheckConstraint(
        "status <> 'DATA_UNAVAILABLE' OR (value_numeric IS NULL AND value_text IS NULL)",
        name="ck_unavailable_carries_no_value",
    ),
)

market_price_reference = Table(
    "market_price_reference",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("collected_at", UTCDateTime, nullable=False),
    Column("collected_by", String(120), nullable=False),
    Column("comparable_json", JSONText, nullable=False, default=list),
    Column("n_comparables", Integer, nullable=False),
    Column("price_p25_inr", DecimalText, nullable=False),
    Column("price_median_inr", DecimalText, nullable=False),
    Column("price_p75_inr", DecimalText, nullable=False),
    Column("notes", Text),
    CheckConstraint("n_comparables >= 0", name="ck_n_comparables_non_negative"),
)

economics_snapshot = Table(
    "economics_snapshot",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("landed_cost_id", UUIDText, ForeignKey("landed_cost.id"), nullable=False),
    Column("scenario", String(20), nullable=False),
    Column("calc_version", String(40), nullable=False),
    # Pinned reference versions: without these the calculation is not replayable.
    Column("rate_card_version", String(40), nullable=False),
    Column("tax_table_version", String(40), nullable=False),
    Column("return_benchmark_version", String(40), nullable=False),
    Column("threshold_profile_version", String(40), nullable=False),
    Column("scenario_profile_version", String(40), nullable=False),
    Column("selling_price_inr", DecimalText, nullable=False),
    Column("net_profit_inr", DecimalText, nullable=False),
    Column("net_margin_pct", DecimalText, nullable=False),
    Column("roi_pct", DecimalText, nullable=False),
    Column("breakeven_price_inr", DecimalText, nullable=False),
    Column("min_profitable_price_inr", DecimalText, nullable=False),
    Column("gate_result", String(20), nullable=False),
    Column("failed_gate", String(120)),
    Column("gap_to_pass_json", JSONText, nullable=False, default=dict),
    Column("computed_at", UTCDateTime, nullable=False),
    _enum_check("scenario", ScenarioName),
    _enum_check("gate_result", GateResult),
    CheckConstraint(
        "gate_result <> 'FAIL' OR failed_gate IS NOT NULL",
        name="ck_fail_names_gate",
    ),
)

price_headroom_check = Table(
    "price_headroom_check",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column(
        "economics_snapshot_id",
        UUIDText,
        ForeignKey("economics_snapshot.id"),
        nullable=False,
    ),
    Column(
        "market_price_reference_id",
        UUIDText,
        ForeignKey("market_price_reference.id"),
        nullable=False,
    ),
    Column("min_profitable_price_inr", DecimalText, nullable=False),
    Column("market_p25_inr", DecimalText, nullable=False),
    Column("headroom_pct", DecimalText, nullable=False),
    Column("result", String(20), nullable=False),
    Column("computed_at", UTCDateTime, nullable=False),
    _enum_check("result", HeadroomResult),
)

risk_assessment = Table(
    "risk_assessment",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("risk_tier", String(20), nullable=False),
    Column("return_risk", String(20), nullable=False),
    Column("fragility_risk", String(20), nullable=False),
    Column("sizing_risk", String(20), nullable=False),
    Column("warranty_risk", String(20), nullable=False),
    Column("supplier_concentration_risk", String(20), nullable=False),
    Column("seasonality_risk", String(20), nullable=False),
    Column("rationale_json", JSONText, nullable=False, default=dict),
    Column("ruleset_version", String(40), nullable=False),
    Column("computed_at", UTCDateTime, nullable=False),
    _enum_check("risk_tier", RiskTier),
    _enum_check("return_risk", RiskLevel),
)

compliance_assessment = Table(
    "compliance_assessment",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("ruleset_version", String(40), nullable=False),
    Column("ruleset_last_verified_at", UTCDateTime),
    Column("overall_result", String(30), nullable=False),
    Column("checks_json", JSONText, nullable=False, default=list),
    Column("blocking_failures_json", JSONText, nullable=False, default=list),
    Column("reviewed_by", String(120)),
    Column("reviewed_at", UTCDateTime),
    Column("review_note", Text),
    Column("computed_at", UTCDateTime, nullable=False),
    _enum_check("overall_result", ComplianceResult),
)

opportunity_score = Table(
    "opportunity_score",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("weights_version", String(40), nullable=False),
    Column("value", DecimalText),
    Column("coverage_pct", DecimalText, nullable=False),
    Column("components_json", JSONText, nullable=False, default=list),
    Column("missing_signals_json", JSONText, nullable=False, default=list),
    Column("result", String(30), nullable=False),
    Column("computed_at", UTCDateTime, nullable=False),
    _enum_check("result", ScoreResult),
    CheckConstraint(
        "(result = 'INSUFFICIENT_DATA' AND value IS NULL) "
        "OR (result = 'SCORED' AND value IS NOT NULL)",
        name="ck_score_value_matches_result",
    ),
)

# --------------------------------------------------------------------------
# Tier 1: AI, transitions, approvals
# --------------------------------------------------------------------------

ai_invocation = Table(
    "ai_invocation",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("service", String(30), nullable=False),
    Column("sku_id", UUIDText, ForeignKey("sku.id")),
    Column("product_id", UUIDText, ForeignKey("product.id")),
    Column("model", String(120), nullable=False),
    Column("model_version", String(120)),
    Column("prompt_version", String(40), nullable=False),
    Column("prompt_hash", String(64), nullable=False),
    Column("input_refs", JSONText, nullable=False, default=list),
    Column("input_token_count", Integer),
    Column("output_token_count", Integer),
    Column("latency_ms", Integer),
    Column("cost_inr", DecimalText, nullable=False),
    Column("raw_output", Text),
    Column("parsed_ok", Boolean, nullable=False),
    Column("schema_violations", JSONText, nullable=False, default=list),
    Column("findings", JSONText, nullable=False, default=list),
    Column("created_at", UTCDateTime, nullable=False),
    _enum_check("service", AIService),
)

sku_state_transition = Table(
    "sku_state_transition",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("from_state", String(30)),
    Column("to_state", String(30), nullable=False),
    Column("trigger", String(20), nullable=False),
    Column("rule_id", String(120)),
    Column("actor", String(120), nullable=False),
    Column("rationale", Text),
    Column("created_at", UTCDateTime, nullable=False),
    _enum_check("to_state", SKUState),
    _enum_check("trigger", TransitionTrigger),
    CheckConstraint(
        "from_state IS NULL OR from_state <> to_state",
        name="ck_transition_changes_state",
    ),
)

approval = Table(
    "approval",
    metadata,
    Column("id", UUIDText, primary_key=True),
    Column("sku_id", UUIDText, ForeignKey("sku.id"), nullable=False),
    Column("decision", String(20), nullable=False),
    Column("approver", String(120), nullable=False),
    Column("decided_at", UTCDateTime, nullable=False),
    Column("note", Text),
    Column("economics_snapshot_id", UUIDText, ForeignKey("economics_snapshot.id")),
    Column("compliance_assessment_id", UUIDText, ForeignKey("compliance_assessment.id")),
    Column("opportunity_score_id", UUIDText, ForeignKey("opportunity_score.id")),
    _enum_check("decision", ApprovalDecision),
)

# --------------------------------------------------------------------------
# Tier 2: the ledger
# --------------------------------------------------------------------------

audit_event = Table(
    "audit_event",
    metadata,
    # `seq` is the ordering key. Timestamps alone cannot totally order events
    # written within the same microsecond, and replay determinism requires a
    # total order. `id` remains the externally meaningful identifier.
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("id", UUIDText, nullable=False, unique=True),
    Column("timestamp", UTCDateTime, nullable=False),
    Column("sku_id", UUIDText),
    Column("product_id", UUIDText),
    Column("supplier_id", UUIDText),
    Column("event_type", String(50), nullable=False),
    Column("actor", String(160), nullable=False),
    Column("inputs_json", JSONText, nullable=False, default=dict),
    Column("reference_versions_json", JSONText, nullable=False, default=dict),
    Column("calc_version", String(40)),
    Column("rule_id", String(120)),
    Column("result_json", JSONText, nullable=False, default=dict),
    Column("error_json", JSONText),
    Column("prev_hash", String(64)),
    Column("event_hash", String(64), nullable=False),
    _enum_check("event_type", EventType),
    # A linear chain can never have two events sharing a predecessor, so a
    # fork is a schema violation rather than something to detect afterwards.
    # This holds even against writers that bypass EventRepository.
    UniqueConstraint("prev_hash", name="uq_audit_event_prev_hash"),
    UniqueConstraint("event_hash", name="uq_audit_event_hash"),
    # Deliberately no foreign keys. The ledger records what happened, including
    # events about entities that failed validation and were never persisted.
    # An FK here would make the ledger refuse to record exactly the failures
    # it most needs to capture.
)

schema_version = Table(
    "schema_version",
    metadata,
    Column("version", Integer, primary_key=True),
    Column("applied_at", UTCDateTime, nullable=False),
    Column("description", Text, nullable=False),
)

# --------------------------------------------------------------------------
# Tier 3: projections (droppable, rebuilt from the ledger)
# --------------------------------------------------------------------------

sku_current_state = Table(
    "sku_current_state",
    metadata,
    Column("sku_id", UUIDText, primary_key=True),
    Column("state", String(30), nullable=False),
    Column("state_updated_at", UTCDateTime, nullable=False),
    Column("transition_count", Integer, nullable=False),
    Column("rebuilt_from_seq", Integer, nullable=False),
    _enum_check("state", SKUState),
)

supplier_offer_current = Table(
    "supplier_offer_current",
    metadata,
    Column("supplier_id", UUIDText, primary_key=True),
    Column("product_id", UUIDText, primary_key=True),
    Column("offer_id", UUIDText, nullable=False),
    Column("unit_cost_inr", DecimalText, nullable=False),
    Column("observed_at", UTCDateTime, nullable=False),
    Column("superseded_count", Integer, nullable=False),
    Column("rebuilt_from_seq", Integer, nullable=False),
)

# --------------------------------------------------------------------------
# Indexes
# --------------------------------------------------------------------------
# Chosen for the access patterns this system actually has: history lookups by
# entity, ordered ledger scans, and reference lookups by version. Not every
# column is indexed; unused indexes cost write throughput and buy nothing.

INDEXES: tuple[Index, ...] = (
    Index("ix_supplier_offer_supplier", supplier_offer.c.supplier_id),
    Index("ix_supplier_offer_product", supplier_offer.c.product_id),
    Index(
        "ix_supplier_offer_lookup",
        supplier_offer.c.supplier_id,
        supplier_offer.c.product_id,
        supplier_offer.c.observed_at,
    ),
    Index("ix_sku_product", sku.c.product_id),
    Index("ix_identity_evidence_product", product_identity_evidence.c.product_id),
    Index("ix_audit_event_timestamp", audit_event.c.timestamp),
    Index("ix_audit_event_type", audit_event.c.event_type),
    Index("ix_audit_event_sku", audit_event.c.sku_id),
    Index("ix_audit_event_product", audit_event.c.product_id),
    Index("ix_audit_event_supplier", audit_event.c.supplier_id),
    Index(
        "ix_transition_sku_created",
        sku_state_transition.c.sku_id,
        sku_state_transition.c.created_at,
    ),
    Index("ix_landed_cost_sku", landed_cost.c.sku_id),
    Index("ix_economics_sku", economics_snapshot.c.sku_id),
    Index("ix_market_signal_sku", market_signal.c.sku_id),
    Index("ix_compliance_sku", compliance_assessment.c.sku_id),
    Index("ix_opportunity_sku", opportunity_score.c.sku_id),
    Index("ix_ai_invocation_sku", ai_invocation.c.sku_id),
    Index("ix_approval_sku", approval.c.sku_id),
    Index("ix_rate_card_marketplace", rate_card.c.marketplace, rate_card.c.version),
    Index("ix_tax_table_hsn", tax_table.c.hsn_code, tax_table.c.version),
    Index("ix_sku_current_state_state", sku_current_state.c.state),
)
