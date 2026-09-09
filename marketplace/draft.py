"""Canonical, marketplace-agnostic listing draft.

Phase 3 is readiness infrastructure. Nothing here publishes, prices, orders or
syncs anything: it decides whether a product *could* be listed and says
plainly what is missing when it could not.

The draft is the join point between Phase 2's evidence and a marketplace's
requirements, so the discipline that made Phase 2 trustworthy has to survive
the crossing. Every factual field carries the provenance it arrived with, and
a field that was UNAVAILABLE upstream stays UNAVAILABLE here. A draft is never
made ready by filling a gap — it is made ready by the gap being closed
upstream with real evidence.

Three states are kept deliberately distinct, because collapsing them is how a
dashboard starts lying:

``UNSUPPORTED``     the marketplace cannot do this, ever
``NOT_CONFIGURED``  it could, but no credentials exist
``NOT_READY``       everything works, but this product is not fit to list
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

DRAFT_VERSION = "listing-draft/1.0.0"


class FieldProvenance(StrEnum):
    """Where a listing field's value came from.

    Carried across from Phase 2 unchanged. The one transition this module must
    never perform is AI_INFERENCE -> SOURCED: a model's suggested copy is
    perfectly good draft text and is never evidence of a fact.
    """

    SOURCED = "SOURCED"
    DERIVED = "DERIVED"
    AI_DRAFT = "AI_DRAFT"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def is_evidence(self) -> bool:
        return self is FieldProvenance.SOURCED


class CapabilityStatus(StrEnum):
    """Whether a marketplace can perform an operation.

    ``NOT_CONFIGURED`` is not a softer ``UNSUPPORTED``. One is fixed by
    supplying credentials; the other cannot be fixed at all, and a UI that
    shows them the same way sends an operator hunting for a key that would
    change nothing.
    """

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_CONFIGURED = "NOT_CONFIGURED"

    @property
    def is_usable(self) -> bool:
        return self is CapabilityStatus.SUPPORTED


class Readiness(StrEnum):
    """Whether a draft is fit to be listed."""

    READY = "READY"
    NOT_READY = "NOT_READY"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"

    @property
    def is_ready(self) -> bool:
        return self is Readiness.READY


class OperationResult(StrEnum):
    """Outcome of an adapter operation in Phase 3."""

    DRY_RUN = "DRY_RUN"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    #: Structurally possible, deliberately switched off for this phase.
    NOT_ENABLED_FOR_PHASE_3 = "NOT_ENABLED_FOR_PHASE_3"


class InventoryState(StrEnum):
    """Stock position, with unknown kept separate from zero."""

    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    REORDER_REQUIRED = "REORDER_REQUIRED"


class Severity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationFinding:
    """One structured reason a draft is not listable."""

    code: str
    field: str
    message: str
    severity: Severity = Severity.ERROR

    @property
    def blocks(self) -> bool:
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "severity": str(self.severity),
            "blocks": self.blocks,
        }


@dataclass(frozen=True)
class DraftField:
    """One listing value and the provenance it arrived with."""

    value: Any
    provenance: FieldProvenance
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.provenance is FieldProvenance.UNAVAILABLE and self.value is not None:
            raise ValueError("an UNAVAILABLE field cannot carry a value")
        if self.provenance is FieldProvenance.SOURCED and not self.reference:
            raise ValueError(
                "a SOURCED field must name its source, or it is indistinguishable "
                "from a guess"
            )

    @property
    def present(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else str(self.value),
            "provenance": str(self.provenance),
            "reference": self.reference,
        }


def unavailable() -> DraftField:
    """A field no source could supply.

    Carries no reference by construction: an UNAVAILABLE field has no source
    to name, and the DraftField invariants reject one.
    """
    return DraftField(None, FieldProvenance.UNAVAILABLE)


@dataclass
class ListingDraft:
    """A marketplace-agnostic listing, assembled from Phase 2 evidence."""

    sku_code: str
    title: str
    #: Identity and commercial facts, each carrying its provenance.
    brand: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    model_code: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    gtin: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    hsn_code: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    weight_g: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    price_inr: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    gst_rate_pct: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    category: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )

    #: Draft copy. May legitimately come from a model; never evidence.
    description: DraftField = field(
        default_factory=lambda: DraftField(None, FieldProvenance.UNAVAILABLE)
    )
    bullets: tuple[str, ...] = ()

    currency: str = "INR"
    images: tuple[str, ...] = ()
    dimensions_cm: tuple[Decimal, Decimal, Decimal] | None = None
    marketplace_attributes: dict[str, str] = field(default_factory=dict)

    #: Carried through from Phase 2 rather than recomputed.
    compliance_state: str | None = None
    compliance_blockers: tuple[str, ...] = ()
    economics_available: bool = False
    expected_margin_pct: Decimal | None = None
    worst_margin_pct: Decimal | None = None
    supplier_name: str | None = None
    supplier_confirmed: bool = False
    inventory_state: InventoryState = InventoryState.UNKNOWN
    inventory_qty: int | None = None

    opportunity_key: str | None = None
    opportunity_verdict: str | None = None
    evidence_coverage_pct: Decimal | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def factual_fields(self) -> dict[str, DraftField]:
        """Every field that asserts something about the world."""
        return {
            name: getattr(self, name)
            for name in (
                "brand",
                "model_code",
                "gtin",
                "hsn_code",
                "weight_g",
                "price_inr",
                "gst_rate_pct",
                "category",
            )
        }

    @property
    def missing_factual_fields(self) -> list[str]:
        return [n for n, f in self.factual_fields().items() if not f.present]

    def as_dict(self) -> dict[str, Any]:
        """Serialise. Contains no credential, token or secret by construction."""
        return {
            "sku_code": self.sku_code,
            "title": self.title,
            "currency": self.currency,
            "fields": {n: f.as_dict() for n, f in self.factual_fields().items()},
            "description": self.description.as_dict(),
            "bullets": list(self.bullets),
            "images": list(self.images),
            "dimensions_cm": (
                None
                if self.dimensions_cm is None
                else [str(d) for d in self.dimensions_cm]
            ),
            "marketplace_attributes": dict(self.marketplace_attributes),
            "compliance_state": self.compliance_state,
            "compliance_blockers": list(self.compliance_blockers),
            "economics_available": self.economics_available,
            "expected_margin_pct": (
                None
                if self.expected_margin_pct is None
                else str(self.expected_margin_pct)
            ),
            "worst_margin_pct": (
                None if self.worst_margin_pct is None else str(self.worst_margin_pct)
            ),
            "supplier_name": self.supplier_name,
            "supplier_confirmed": self.supplier_confirmed,
            "inventory_state": str(self.inventory_state),
            "inventory_qty": self.inventory_qty,
            "opportunity_key": self.opportunity_key,
            "opportunity_verdict": self.opportunity_verdict,
            "evidence_coverage_pct": (
                None
                if self.evidence_coverage_pct is None
                else str(self.evidence_coverage_pct)
            ),
            "missing_factual_fields": self.missing_factual_fields,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ListingAssessment:
    """One marketplace's verdict on one draft."""

    marketplace: str
    readiness: Readiness
    findings: list[ValidationFinding] = field(default_factory=list)
    capability: CapabilityStatus = CapabilityStatus.SUPPORTED
    notes: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.blocks]

    def as_dict(self) -> dict[str, Any]:
        return {
            "marketplace": self.marketplace,
            "readiness": str(self.readiness),
            "capability": str(self.capability),
            "findings": [f.as_dict() for f in self.findings],
            "blocking_count": len(self.blocking),
            "notes": list(self.notes),
            # Stated on every assessment so no client can infer otherwise.
            "published": False,
            "external_side_effects": False,
        }


__all__ = [
    "DRAFT_VERSION",
    "CapabilityStatus",
    "DraftField",
    "FieldProvenance",
    "InventoryState",
    "ListingAssessment",
    "ListingDraft",
    "OperationResult",
    "Readiness",
    "Severity",
    "ValidationFinding",
    "unavailable",
]
