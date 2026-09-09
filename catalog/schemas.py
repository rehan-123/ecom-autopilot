"""Pydantic v2 entity schemas.

These models are the single source of truth for entity *shape*. The SQLAlchemy
tables in ``database.tables`` mirror them; later milestones will derive JSON
schemas for Claude structured output from the same models, so that the
database, the LLM contract and the report shapes cannot drift apart.

Conventions enforced here:

* Every business identifier is a UUID. Sequential integers are never exposed
  as meaningful identifiers.
* Every datetime is timezone-aware and normalised to UTC on construction.
  Naive datetimes are rejected at the boundary, not coerced.
* Monetary amounts are ``Decimal``, never ``float``. Binary floating point
  silently loses rupees over a long enough ledger.
* Models are frozen. These describe immutable records; mutating one in place
  would defeat the point of the architecture.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from catalog.enums import (
    ActorKind,
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

# --------------------------------------------------------------------------
# Shared field types
# --------------------------------------------------------------------------

Money = Annotated[Decimal, Field(ge=0, decimal_places=4)]
"""A non-negative rupee amount. Four decimal places so that per-unit
apportionment (freight over an MOQ, say) does not round prematurely."""

Percent = Annotated[Decimal, Field(ge=0, le=100, decimal_places=4)]

JsonDict = dict[str, Any]


def new_id() -> uuid.UUID:
    """Generate a business identifier."""
    return uuid.uuid4()


def utcnow() -> datetime:
    """Current time, timezone-aware, UTC."""
    return datetime.now(UTC)


class _Base(BaseModel):
    """Common configuration for every entity model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def _require_aware_utc(cls, value: Any) -> Any:
        """Reject naive datetimes and normalise aware ones to UTC.

        Mixing naive and aware datetimes is the most common way an event
        ledger quietly becomes unorderable, so this is enforced on every
        datetime field of every model rather than case by case.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                raise ValueError("naive datetime rejected; supply a timezone-aware value")
            return value.astimezone(UTC)
        return value


# --------------------------------------------------------------------------
# Catalog: the immutable factual core
# --------------------------------------------------------------------------


class Supplier(_Base):
    """A source we can buy from.

    Deliberately carries no credentials. Portal logins, API keys and bank
    details are out of scope for this system and must not be persisted here.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    name: str = Field(min_length=1, max_length=200)
    type: SupplierType
    gst_status: GSTStatus
    gstin: str | None = Field(default=None, min_length=15, max_length=15)
    issues_tax_invoice: bool
    state_code: str | None = Field(default=None, min_length=2, max_length=2)
    contact: JsonDict = Field(default_factory=dict)
    notes: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _gstin_consistent_with_status(self) -> Supplier:
        if self.gst_status is GSTStatus.REGISTERED and not self.gstin:
            raise ValueError("a REGISTERED supplier must have a gstin")
        if self.gst_status is GSTStatus.UNREGISTERED and self.gstin:
            raise ValueError("an UNREGISTERED supplier must not have a gstin")
        if self.issues_tax_invoice and self.gst_status is not GSTStatus.REGISTERED:
            raise ValueError(
                "only a REGISTERED supplier can issue a tax invoice; "
                "without one there is no input tax credit"
            )
        return self


class Product(_Base):
    """The canonical real-world product.

    Not a listing and not a sellable unit. Several supplier offers resolve to
    one Product; that resolution is what makes supplier comparison possible.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    canonical_name: str = Field(min_length=1, max_length=300)
    category: str = Field(min_length=1, max_length=120)
    hsn_code: str | None = Field(default=None, min_length=4, max_length=8)
    brand: str | None = None
    manufacturer: str | None = None
    country_of_origin: str | None = Field(default=None, max_length=60)
    weight_g: int | None = Field(default=None, gt=0)
    length_mm: int | None = Field(default=None, gt=0)
    width_mm: int | None = Field(default=None, gt=0)
    height_mm: int | None = Field(default=None, gt=0)
    fragile: bool = False
    sized: bool = False
    warranty_months: int | None = Field(default=None, ge=0)
    identity_confidence: IdentityConfidence = IdentityConfidence.UNRESOLVED
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("hsn_code")
    @classmethod
    def _hsn_is_numeric(cls, value: str | None) -> str | None:
        if value is not None and not value.isdigit():
            raise ValueError("hsn_code must be numeric")
        return value


class SKU(_Base):
    """Our sellable unit: a product, a variant, a pack size.

    Note the absence of a ``state`` field. Current state is a *projection*
    derived from the SKUStateTransition stream, not a mutable column. See
    ``ledger.projections``. Use ``SKUView`` when a state-bearing read model
    is needed.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    product_id: uuid.UUID
    sku_code: str = Field(min_length=1, max_length=64)
    variant: JsonDict = Field(default_factory=dict)
    pack_qty: int = Field(default=1, gt=0)
    created_at: datetime = Field(default_factory=utcnow)


class SKUView(_Base):
    """Read model: a SKU together with its projected current state."""

    sku: SKU
    state: SKUState
    state_updated_at: datetime


class ProductIdentityEvidence(_Base):
    """Why two supplier offers were judged to be the same product.

    Retained so an identity decision made by the Identity Judge can be
    re-examined, and so that replaying history reproduces the same evidence
    the judgement was made on.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    product_id: uuid.UUID
    evidence_type: EvidenceType
    value: str = Field(min_length=1, max_length=2000)
    source_offer_id: uuid.UUID | None = None
    collected_at: datetime = Field(default_factory=utcnow)


class SupplierOffer(_Base):
    """An immutable observation of what a supplier quoted, when.

    A price change creates a *new* offer. The previous row is never updated,
    so the price history for any (supplier, product) pair is the ordered set
    of its offers. Supersession is derived from ``observed_at`` ordering
    rather than stored as a back-reference, because writing a back-reference
    onto an earlier row would require mutating an immutable record.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    supplier_id: uuid.UUID
    product_id: uuid.UUID
    unit_cost_inr: Money
    #: None means the catalogue did not say. It is not False and it is not
    #: True: without it the GST component of the quoted cost is unknown, so
    #: recoverable input tax credit cannot be computed.
    cost_is_gst_inclusive: bool | None = None
    gst_rate_pct: Percent
    tax_invoice_available: bool
    #: Required. MOQ is supplier-stated fact and drives carrying cost, so a
    #: default of 1 would understate locked capital on every offer built
    #: without one.
    moq_units: int = Field(..., gt=0)
    available_qty: int | None = Field(default=None, ge=0)
    #: None means the catalogue did not state inbound freight. It is not zero:
    #: an unstated freight charge silently understates landed cost, and a
    #: confirmed zero is a different claim from an unknown.
    inbound_freight_inr: Money | None = None
    dispatch_days: int | None = Field(default=None, ge=0)
    return_terms: str | None = None
    warranty_terms: str | None = None
    compliance_docs: JsonDict = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utcnow)
    source_document_ref: str | None = None


# --------------------------------------------------------------------------
# Reference data: versioned, dated, staleness-aware
# --------------------------------------------------------------------------


class _VersionedReference(_Base):
    """Shared shape for reference data.

    Every rate, rule and benchmark used by a calculation must be reachable
    through one of these, so that an assessment can pin the exact version it
    used and be replayed years later against the same inputs.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    version: str = Field(min_length=1, max_length=40)
    effective_from: datetime
    source_url: str | None = None
    last_verified_at: datetime | None = None


class RateCard(_VersionedReference):
    """Marketplace fee schedule. Rules kept as JSON because the shape differs
    per marketplace (commission slabs, fixed-fee bands, weight x zone shipping
    matrices) and is not usefully queryable relationally."""

    marketplace: str = Field(min_length=1, max_length=60)
    effective_to: datetime | None = None
    rules_json: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class TaxTable(_VersionedReference):
    effective_to: datetime | None = None
    hsn_code: str = Field(min_length=4, max_length=8)
    gst_rate_pct: Percent
    cess_pct: Percent = Decimal("0")
    tcs_rate_pct: Percent = Decimal("0")


class ReturnBenchmark(_VersionedReference):
    category: str = Field(min_length=1, max_length=120)
    channel: str = Field(min_length=1, max_length=60)
    return_rate_pct: Percent
    rto_rate_pct: Percent
    cost_per_return_inr: Money
    unsaleable_share_pct: Percent
    source: str = Field(min_length=1, max_length=120)
    confidence: str = Field(min_length=1, max_length=40)


class ThresholdProfile(_VersionedReference):
    """Configurable business floors. Versioned as data so that changing a
    threshold does not silently reinterpret historical decisions."""

    name: str = Field(min_length=1, max_length=60)
    min_net_margin_pct: Percent
    min_profit_per_order_inr: Money
    max_return_rate_pct: Percent
    max_supplier_cost_pct: Percent
    max_ad_spend_pct: Percent
    min_roi_pct: Percent
    min_price_headroom_pct: Percent
    min_coverage_pct: Percent


class ScenarioProfile(_VersionedReference):
    """Best/expected/worst expressed as parameter deltas rather than
    hand-picked numbers, so scenarios are reproducible and comparable."""

    scenario: ScenarioName
    return_rate_multiplier: Decimal = Field(gt=0)
    supplier_cost_delta_pct: Decimal
    shipping_delta_pct: Decimal
    ad_cost_multiplier: Decimal = Field(gt=0)
    velocity_multiplier: Decimal = Field(gt=0)


# --------------------------------------------------------------------------
# Assessments: structures only. No calculation logic in this milestone.
# --------------------------------------------------------------------------


class LandedCost(_Base):
    """What a unit actually costs us, delivered and ready to ship.

    ``input_tax_credit_inr`` is zero whenever the supplier issues no tax
    invoice. It is kept as its own line rather than folded into the cost
    because it is frequently the difference between a viable SKU and a dead
    one, and must stay visible in every report.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    supplier_offer_id: uuid.UUID
    scenario: ScenarioName
    calc_version: str = Field(min_length=1, max_length=40)
    unit_cost_ex_gst_inr: Money
    input_tax_credit_inr: Money
    effective_cogs_inr: Money
    inbound_freight_per_unit_inr: Money
    moq_carry_cost_inr: Money
    inspection_qc_cost_inr: Money
    packaging_cost_inr: Money
    inbound_defect_provision_inr: Money
    total_landed_cost_inr: Money
    assumptions_json: JsonDict = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=utcnow)


class MarketSignal(_Base):
    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    signal_type: str = Field(min_length=1, max_length=60)
    value_numeric: Decimal | None = None
    value_text: str | None = None
    status: SignalStatus
    source: SignalSource
    source_ref: str | None = None
    confidence: str | None = Field(default=None, max_length=40)
    collected_at: datetime = Field(default_factory=utcnow)
    collected_by: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def _unavailable_carries_no_value(self) -> MarketSignal:
        if self.status is SignalStatus.DATA_UNAVAILABLE and (
            self.value_numeric is not None or self.value_text is not None
        ):
            raise ValueError(
                "DATA_UNAVAILABLE must not carry a value; an absent signal is "
                "not a zero and not an average"
            )
        return self


class MarketPriceReference(_Base):
    """Manually collected evidence of what comparable items actually sell for.

    Required before economics can be trusted: a SKU can clear every internal
    gate at a price nobody in the market will pay.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    collected_at: datetime = Field(default_factory=utcnow)
    collected_by: str = Field(min_length=1, max_length=120)
    comparable_json: list[JsonDict] = Field(default_factory=list)
    n_comparables: int = Field(ge=0)
    price_p25_inr: Money
    price_median_inr: Money
    price_p75_inr: Money
    notes: str | None = None

    @model_validator(mode="after")
    def _percentiles_ordered(self) -> MarketPriceReference:
        if not (self.price_p25_inr <= self.price_median_inr <= self.price_p75_inr):
            raise ValueError("percentiles must satisfy p25 <= median <= p75")
        return self


class EconomicsSnapshot(_Base):
    """An immutable unit-economics result.

    Every reference version used is pinned. Without those pins a February
    calculation replayed with August's rate card silently produces a
    different answer, and the ledger becomes fiction.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    landed_cost_id: uuid.UUID
    scenario: ScenarioName
    calc_version: str = Field(min_length=1, max_length=40)
    rate_card_version: str = Field(min_length=1, max_length=40)
    tax_table_version: str = Field(min_length=1, max_length=40)
    return_benchmark_version: str = Field(min_length=1, max_length=40)
    threshold_profile_version: str = Field(min_length=1, max_length=40)
    scenario_profile_version: str = Field(min_length=1, max_length=40)
    selling_price_inr: Money
    net_profit_inr: Decimal
    net_margin_pct: Decimal
    roi_pct: Decimal
    breakeven_price_inr: Money
    min_profitable_price_inr: Money
    gate_result: GateResult
    failed_gate: str | None = None
    gap_to_pass_json: JsonDict = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _failure_explains_itself(self) -> EconomicsSnapshot:
        if self.gate_result is GateResult.FAIL and not self.failed_gate:
            raise ValueError(
                "a FAIL must name the gate it failed; 'fails by X' is the "
                "only actionable output of a rejection"
            )
        return self


class PriceHeadroomCheck(_Base):
    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    economics_snapshot_id: uuid.UUID
    market_price_reference_id: uuid.UUID
    min_profitable_price_inr: Money
    market_p25_inr: Money
    headroom_pct: Decimal
    result: HeadroomResult
    computed_at: datetime = Field(default_factory=utcnow)


class RiskAssessment(_Base):
    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    risk_tier: RiskTier
    return_risk: RiskLevel
    fragility_risk: RiskLevel
    sizing_risk: RiskLevel
    warranty_risk: RiskLevel
    supplier_concentration_risk: RiskLevel
    seasonality_risk: RiskLevel
    rationale_json: JsonDict = Field(default_factory=dict)
    ruleset_version: str = Field(min_length=1, max_length=40)
    computed_at: datetime = Field(default_factory=utcnow)


class ComplianceAssessment(_Base):
    """Compliance outcome, with the ruleset version and its verification date.

    ``ruleset_last_verified_at`` is carried explicitly because a stale
    ruleset makes "no matching requirement" meaningless: absence of a rule is
    only evidence when the rulebook is current.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    ruleset_version: str = Field(min_length=1, max_length=40)
    ruleset_last_verified_at: datetime | None = None
    overall_result: ComplianceResult
    checks_json: list[JsonDict] = Field(default_factory=list)
    blocking_failures_json: list[JsonDict] = Field(default_factory=list)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    computed_at: datetime = Field(default_factory=utcnow)


class OpportunityScore(_Base):
    """Score value and evidence coverage, never blended.

    A score of 94 at 45% coverage and a score of 78 at 95% coverage are not
    comparable. ``coverage_pct`` travels with ``value`` everywhere so that a
    bare number can never be rendered.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    weights_version: str = Field(min_length=1, max_length=40)
    value: Decimal | None = Field(default=None, ge=0, le=100)
    coverage_pct: Percent
    components_json: list[JsonDict] = Field(default_factory=list)
    missing_signals_json: list[str] = Field(default_factory=list)
    result: ScoreResult
    computed_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _insufficient_data_has_no_value(self) -> OpportunityScore:
        if self.result is ScoreResult.INSUFFICIENT_DATA and self.value is not None:
            raise ValueError(
                "INSUFFICIENT_DATA must not carry a score value; emitting one "
                "would let a low-coverage product be ranked against a "
                "well-evidenced one"
            )
        if self.result is ScoreResult.SCORED and self.value is None:
            raise ValueError("SCORED requires a value")
        return self


# --------------------------------------------------------------------------
# AI, transitions, approvals
# --------------------------------------------------------------------------


class AIInvocation(_Base):
    """Persistence model for a Claude call. No API client exists yet.

    ``model_version`` and ``prompt_version`` are recorded because when a model
    updates and extraction quality shifts, this table is the only way to find
    out which outputs came from which behaviour.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        protected_namespaces=(),  # permit the model_* field names below
    )

    id: uuid.UUID = Field(default_factory=new_id)
    service: AIService
    sku_id: uuid.UUID | None = None
    product_id: uuid.UUID | None = None
    model: str = Field(min_length=1, max_length=120)
    model_version: str | None = Field(default=None, max_length=120)
    prompt_version: str = Field(min_length=1, max_length=40)
    prompt_hash: str = Field(min_length=8, max_length=64)
    input_refs: list[str] = Field(default_factory=list)
    input_token_count: int | None = Field(default=None, ge=0)
    output_token_count: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    cost_inr: Money = Decimal("0")
    raw_output: str | None = None
    parsed_ok: bool
    schema_violations: list[str] = Field(default_factory=list)
    findings: list[JsonDict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class SKUStateTransition(_Base):
    """The authoritative record of a lifecycle change.

    State is never changed anywhere else. This stream is what
    ``sku_current_state`` is projected from.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    from_state: SKUState | None = None
    to_state: SKUState
    trigger: TransitionTrigger
    rule_id: str | None = Field(default=None, max_length=120)
    actor: str = Field(min_length=1, max_length=120)
    rationale: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _no_self_transition(self) -> SKUStateTransition:
        if self.from_state is not None and self.from_state == self.to_state:
            raise ValueError("a transition must change state")
        return self


class Approval(_Base):
    """A human decision. The boundary before LISTING_READY.

    The three assessment references are optional at this milestone because
    those engines do not exist yet; once they do, an approval without them
    should be rejected by the approval workflow rather than by this schema.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    sku_id: uuid.UUID
    decision: ApprovalDecision
    approver: str = Field(min_length=1, max_length=120)
    decided_at: datetime = Field(default_factory=utcnow)
    note: str | None = None
    economics_snapshot_id: uuid.UUID | None = None
    compliance_assessment_id: uuid.UUID | None = None
    opportunity_score_id: uuid.UUID | None = None


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


class AuditEvent(_Base):
    """One immutable entry in the append-only ledger.

    ``seq`` is assigned by the database on append and is the ordering key.
    Timestamps alone are insufficient: two events in the same microsecond
    would be unorderable, and replay determinism depends on a total order.

    ``prev_hash``/``event_hash`` form a chain. SQLite triggers prevent normal
    UPDATE and DELETE; the chain makes any out-of-band tampering *detectable*,
    which triggers alone cannot do.
    """

    id: uuid.UUID = Field(default_factory=new_id)
    seq: int | None = Field(default=None, ge=1)
    timestamp: datetime = Field(default_factory=utcnow)
    sku_id: uuid.UUID | None = None
    product_id: uuid.UUID | None = None
    supplier_id: uuid.UUID | None = None
    event_type: EventType
    actor: str = Field(min_length=1, max_length=160)
    inputs_json: JsonDict = Field(default_factory=dict)
    reference_versions_json: JsonDict = Field(default_factory=dict)
    calc_version: str | None = Field(default=None, max_length=40)
    rule_id: str | None = Field(default=None, max_length=120)
    result_json: JsonDict = Field(default_factory=dict)
    error_json: JsonDict | None = None
    prev_hash: str | None = Field(default=None, max_length=64)
    event_hash: str | None = Field(default=None, max_length=64)

    @field_validator("actor")
    @classmethod
    def _actor_is_well_formed(cls, value: str) -> str:
        """Actors are ``SYSTEM``, ``HUMAN:<id>`` or ``AI:<service>``."""
        if value == ActorKind.SYSTEM:
            return value
        for kind in (ActorKind.HUMAN, ActorKind.AI):
            prefix = f"{kind}:"
            if value.startswith(prefix) and len(value) > len(prefix):
                return value
        raise ValueError("actor must be 'SYSTEM', 'HUMAN:<identifier>' or 'AI:<service>'")


__all__ = [
    "SKU",
    "AIInvocation",
    "Approval",
    "AuditEvent",
    "ComplianceAssessment",
    "EconomicsSnapshot",
    "JsonDict",
    "LandedCost",
    "MarketPriceReference",
    "MarketSignal",
    "Money",
    "OpportunityScore",
    "Percent",
    "PriceHeadroomCheck",
    "Product",
    "ProductIdentityEvidence",
    "RateCard",
    "ReturnBenchmark",
    "RiskAssessment",
    "SKUStateTransition",
    "SKUView",
    "ScenarioProfile",
    "Supplier",
    "SupplierOffer",
    "TaxTable",
    "ThresholdProfile",
    "new_id",
    "utcnow",
]
