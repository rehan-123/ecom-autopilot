"""Marketplace registry, validation, and the opportunity → draft transformation.

Three responsibilities, none of which recompute Phase 2 work:

**Registry.** Reports what each marketplace can do, distinguishing "cannot"
from "not configured". Capability declarations and credential detection come
from the existing ``part2.marketplace`` adapters rather than a second list
that would drift out of step with them.

**Validation.** Checks a draft against generic marketplace requirements and
returns structured findings. It never repairs a factual value: a missing HSN
stays missing and blocks, because silently filling it is the failure mode the
whole system is built to avoid.

**Transformation.** Turns a Phase 2 ``Opportunity`` into a ``ListingDraft``,
carrying each field's provenance across unchanged. Compliance state,
economics and supplier confirmation are read from the opportunity, never
re-derived — Phase 2 already decided them, and deciding twice invites the two
answers to disagree.

Phase 3 performs no live marketplace operation. Write capabilities return
``NOT_ENABLED_FOR_PHASE_3`` even where an adapter and credentials exist.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from marketplace.draft import (
    CapabilityStatus,
    DraftField,
    FieldProvenance,
    InventoryState,
    ListingAssessment,
    ListingDraft,
    OperationResult,
    Readiness,
    Severity,
    ValidationFinding,
)
from part2.marketplace import ADAPTERS, Capability, get_adapter

logger = logging.getLogger(__name__)

REGISTRY_VERSION = "marketplace-registry/1.0.0"

#: Operations that would change external state. Blocked for this phase
#: regardless of adapter or credentials.
WRITE_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.CREATE_LISTING,
        Capability.UPDATE_LISTING,
        Capability.UPDATE_PRICE,
        Capability.UPDATE_INVENTORY,
    }
)

#: Minimum title length any marketplace will accept. Deliberately generic:
#: inventing Amazon's or Flipkart's real limits without their documentation
#: would be fabricating an API contract.
MIN_TITLE_LEN = 10
MAX_TITLE_LEN = 200
MIN_IMAGES = 1


@dataclass(frozen=True)
class CapabilityReport:
    """One marketplace capability and why it is or is not available."""

    capability: str
    status: CapabilityStatus
    reason: str = ""
    #: True when the operation is switched off for this phase rather than
    #: absent. Kept separate so the UI never implies it could be run.
    write_blocked_for_phase_3: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "status": str(self.status),
            "reason": self.reason,
            "write_blocked_for_phase_3": self.write_blocked_for_phase_3,
        }


@dataclass
class MarketplaceStatus:
    """Everything the UI needs about one marketplace."""

    name: str
    access_model: str
    credentials_configured: bool
    live_transport: bool
    capabilities: list[CapabilityReport] = field(default_factory=list)
    notes: str = ""

    def capability(self, name: str) -> CapabilityReport | None:
        return next((c for c in self.capabilities if c.capability == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "access_model": self.access_model,
            "credentials_configured": self.credentials_configured,
            "live_transport": self.live_transport,
            "capabilities": [c.as_dict() for c in self.capabilities],
            "notes": self.notes,
            "phase_3_write_operations_enabled": False,
        }


def capability_status(
    marketplace: str, capability: Capability, *, environ: Mapping[str, str] | None = None
) -> CapabilityReport:
    """Classify one capability for one marketplace.

    Order matters. A capability the adapter does not declare is UNSUPPORTED
    whatever the credentials say — supplying a key cannot make a marketplace
    grow an API it does not have.
    """
    adapter = get_adapter(
        marketplace, environ=dict(environ) if environ is not None else None
    )
    profile = adapter.profile

    if capability not in profile.capabilities:
        return CapabilityReport(
            capability=str(capability),
            status=CapabilityStatus.UNSUPPORTED,
            reason=f"{marketplace} does not expose {capability}",
        )

    env = dict(environ) if environ is not None else dict(os.environ)
    if not profile.credentials_present(env):
        required = ", ".join(profile.credential_env_vars) or "credentials"
        return CapabilityReport(
            capability=str(capability),
            status=CapabilityStatus.NOT_CONFIGURED,
            reason=f"set {required} in the environment to configure this",
        )

    if capability in WRITE_CAPABILITIES:
        return CapabilityReport(
            capability=str(capability),
            status=CapabilityStatus.SUPPORTED,
            reason="structurally available but disabled for Phase 3",
            write_blocked_for_phase_3=True,
        )

    return CapabilityReport(
        capability=str(capability), status=CapabilityStatus.SUPPORTED, reason=""
    )


def marketplace_status(
    marketplace: str, *, environ: Mapping[str, str] | None = None
) -> MarketplaceStatus:
    adapter = get_adapter(
        marketplace, environ=dict(environ) if environ is not None else None
    )
    env = dict(environ) if environ is not None else dict(os.environ)
    return MarketplaceStatus(
        name=marketplace,
        access_model=str(adapter.profile.access),
        credentials_configured=adapter.profile.credentials_present(env),
        live_transport=adapter.live,
        capabilities=[
            capability_status(marketplace, capability, environ=env)
            for capability in Capability
        ],
        notes=adapter.profile.notes,
    )


def registry(*, environ: Mapping[str, str] | None = None) -> list[MarketplaceStatus]:
    """Every configured marketplace, in a stable order."""
    return [marketplace_status(name, environ=environ) for name in sorted(ADAPTERS)]


# --------------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------------


def _sourced(value: Any, reference: str) -> DraftField:
    if value in (None, ""):
        return DraftField(None, FieldProvenance.UNAVAILABLE)
    return DraftField(value, FieldProvenance.SOURCED, reference=reference)


def build_draft(opportunity: Any, *, inventory_qty: int | None = None) -> ListingDraft:
    """Turn a Phase 2 opportunity into a marketplace-agnostic draft.

    Deterministic: the same opportunity always produces the same draft. Every
    factual field is copied with the provenance it already had, and nothing is
    filled in to make the draft look complete.
    """
    pipeline = getattr(opportunity, "pipeline", None)
    compliance = getattr(pipeline, "compliance", None) if pipeline else None
    reference = f"opportunity://{opportunity.group_key}"

    draft = ListingDraft(
        sku_code=f"SKU-{opportunity.group_key[:24]}",
        title=opportunity.title,
        opportunity_key=opportunity.group_key,
        opportunity_verdict=str(opportunity.verdict),
        evidence_coverage_pct=opportunity.discovery_coverage_pct,
    )

    canonical = None
    matches = getattr(opportunity, "supplier_matches", []) or []
    if matches:
        canonical = matches[0]

    draft.price_inr = _sourced(getattr(opportunity, "target_price_inr", None), reference)
    draft.hsn_code = _sourced(
        getattr(getattr(pipeline, "product", None), "hsn_code", None), reference
    )

    # Compliance and economics are read, never recomputed.
    draft.compliance_state = str(compliance.overall) if compliance else None
    draft.compliance_blockers = tuple(getattr(compliance, "blocking_failures", []) or [])
    draft.expected_margin_pct = opportunity.expected_margin_pct
    draft.worst_margin_pct = opportunity.worst_margin_pct
    draft.economics_available = (
        opportunity.expected_margin_pct is not None
        and opportunity.worst_margin_pct is not None
    )

    # A probable supplier match is not a confirmed source of supply.
    draft.supplier_name = opportunity.selected_supplier
    draft.supplier_confirmed = opportunity.selected_supplier is not None
    if canonical is not None and not draft.supplier_confirmed:
        draft.marketplace_attributes["supplier_match_confidence"] = str(
            canonical.confidence
        )

    if inventory_qty is None:
        draft.inventory_state = InventoryState.UNKNOWN
        draft.inventory_qty = None
    elif inventory_qty < 0:
        raise ValueError(f"inventory cannot be negative: {inventory_qty}")
    elif inventory_qty == 0:
        draft.inventory_state = InventoryState.OUT_OF_STOCK
        draft.inventory_qty = 0
    else:
        draft.inventory_state = InventoryState.KNOWN
        draft.inventory_qty = inventory_qty

    return draft


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


BLOCKING_COMPLIANCE_STATES = frozenset({"FAIL", "UNKNOWN", "REQUIRES_HUMAN_REVIEW"})


def validate_draft(
    draft: ListingDraft, *, marketplace: str = "generic"
) -> ListingAssessment:
    """Assess a draft against generic marketplace requirements.

    Findings are structured and nothing is repaired. Rules are deliberately
    generic: this project has no marketplace API documentation, and inventing
    Amazon's real title limits or category IDs would be fabricating a contract.
    """
    findings: list[ValidationFinding] = []

    def error(code: str, field_name: str, message: str) -> None:
        findings.append(ValidationFinding(code, field_name, message, Severity.ERROR))

    def warn(code: str, field_name: str, message: str) -> None:
        findings.append(ValidationFinding(code, field_name, message, Severity.WARNING))

    # -- identity -------------------------------------------------------
    if not draft.sku_code or len(draft.sku_code) < 3:
        error("INVALID_SKU", "sku_code", "SKU is missing or too short")
    if not draft.title or len(draft.title) < MIN_TITLE_LEN:
        error(
            "INVALID_TITLE", "title", f"title must be at least {MIN_TITLE_LEN} characters"
        )
    elif len(draft.title) > MAX_TITLE_LEN:
        error("INVALID_TITLE", "title", f"title exceeds {MAX_TITLE_LEN} characters")

    # -- factual fields, never filled in --------------------------------
    for name, value in draft.factual_fields().items():
        if not value.present and name in ("hsn_code", "price_inr", "category"):
            error(
                "MISSING_REQUIRED_FIELD",
                name,
                f"{name} has no sourced value and is required to list",
            )
        elif not value.present:
            warn("MISSING_FIELD", name, f"{name} is unavailable")
        elif not value.provenance.is_evidence:
            error(
                "UNSOURCED_FACTUAL_FIELD",
                name,
                f"{name} has provenance {value.provenance}; a listing fact must "
                "be sourced, not inferred",
            )

    # -- price ----------------------------------------------------------
    if draft.price_inr.present and Decimal(str(draft.price_inr.value)) <= 0:
        error("INVALID_PRICE", "price_inr", "price must be positive")
    if draft.currency != "INR":
        error("INVALID_CURRENCY", "currency", f"unsupported currency {draft.currency}")

    # -- images ---------------------------------------------------------
    if len(draft.images) < MIN_IMAGES:
        error("MISSING_IMAGES", "images", f"at least {MIN_IMAGES} image is required")

    # -- compliance -----------------------------------------------------
    if draft.compliance_state is None:
        error("COMPLIANCE_UNKNOWN", "compliance_state", "no compliance evaluation exists")
    elif draft.compliance_state in BLOCKING_COMPLIANCE_STATES:
        detail = ", ".join(draft.compliance_blockers) or draft.compliance_state
        error(
            "COMPLIANCE_NOT_CLEARED",
            "compliance_state",
            f"compliance is {draft.compliance_state}: {detail}",
        )

    # -- economics ------------------------------------------------------
    if not draft.economics_available:
        error(
            "ECONOMICS_UNAVAILABLE",
            "economics",
            "no economics were computed, so this cannot be presented as listable",
        )

    # -- supplier -------------------------------------------------------
    if not draft.supplier_confirmed:
        confidence = draft.marketplace_attributes.get("supplier_match_confidence")
        detail = f" (match is only {confidence})" if confidence else ""
        error(
            "SUPPLIER_NOT_CONFIRMED",
            "supplier_name",
            f"no confirmed supplier{detail}; a probable match is not a source of supply",
        )

    # -- inventory ------------------------------------------------------
    if draft.inventory_state is InventoryState.UNKNOWN:
        warn("INVENTORY_UNKNOWN", "inventory_qty", "stock position is not known")
    elif draft.inventory_state is InventoryState.OUT_OF_STOCK:
        error("OUT_OF_STOCK", "inventory_qty", "no stock on hand")
    if draft.inventory_qty is not None and draft.inventory_qty < 0:
        error("INVALID_QUANTITY", "inventory_qty", "inventory cannot be negative")

    blocking = [f for f in findings if f.blocks]
    readiness = Readiness.READY if not blocking else Readiness.NOT_READY

    return ListingAssessment(
        marketplace=marketplace,
        readiness=readiness,
        findings=findings,
        capability=CapabilityStatus.SUPPORTED,
        notes=[
            "Phase 3 assesses readiness only. No listing was created, updated "
            "or published, and no external request was made."
        ],
    )


def assess_for_marketplace(
    draft: ListingDraft, marketplace: str, *, environ: Mapping[str, str] | None = None
) -> ListingAssessment:
    """Assess a draft for one marketplace, including its capability state.

    A marketplace that cannot create listings is reported as such rather than
    being validated against rules it will never apply.
    """
    capability = capability_status(
        marketplace, Capability.CREATE_LISTING, environ=environ
    )
    if capability.status is not CapabilityStatus.SUPPORTED:
        return ListingAssessment(
            marketplace=marketplace,
            readiness=Readiness.NOT_READY,
            capability=capability.status,
            notes=[capability.reason],
        )

    assessment = validate_draft(draft, marketplace=marketplace)
    assessment.capability = capability.status
    if capability.write_blocked_for_phase_3:
        assessment.notes.append(
            "listing creation is disabled for Phase 3 even though this "
            "marketplace and its credentials would support it"
        )
    return assessment


def assess_all(
    draft: ListingDraft, *, environ: Mapping[str, str] | None = None
) -> list[ListingAssessment]:
    """Assess against every marketplace, isolating failures.

    One adapter raising must not make another marketplace unavailable, so each
    is evaluated independently and a failure becomes a finding rather than an
    exception.
    """
    assessments: list[ListingAssessment] = []
    for name in sorted(ADAPTERS):
        try:
            assessments.append(assess_for_marketplace(draft, name, environ=environ))
        except Exception as exc:  # noqa: BLE001 - an adapter fault is a finding
            logger.warning(
                "adapter assessment failed",
                extra={"marketplace": name, "error": str(exc)},
            )
            assessments.append(
                ListingAssessment(
                    marketplace=name,
                    readiness=Readiness.NOT_READY,
                    capability=CapabilityStatus.UNSUPPORTED,
                    findings=[
                        ValidationFinding(
                            "ADAPTER_ERROR", "adapter", f"adapter failed: {exc}"
                        )
                    ],
                )
            )
    return assessments


# --------------------------------------------------------------------------
# Dry-run operations
# --------------------------------------------------------------------------


def preview_listing(draft: ListingDraft, marketplace: str) -> dict[str, Any]:
    """Render what would be sent. Makes no request."""
    return {
        "marketplace": marketplace,
        "result": str(OperationResult.DRY_RUN),
        "draft": draft.as_dict(),
        "detail": (
            "This is a preview. No marketplace request was made and no listing exists."
        ),
    }


def publish_listing(draft: ListingDraft, marketplace: str) -> dict[str, Any]:
    """Refuse to publish. Present so callers get a structured refusal.

    Never returns success. Phase 3 has no publication capability at all, and a
    function that quietly did nothing while reporting OK would be worse than
    its absence.
    """
    return {
        "marketplace": marketplace,
        "result": str(OperationResult.NOT_ENABLED_FOR_PHASE_3),
        "sku_code": draft.sku_code,
        "detail": (
            "Publishing is not enabled in Phase 3. No listing was created and "
            "no external request was made."
        ),
        "published": False,
    }


__all__ = [
    "BLOCKING_COMPLIANCE_STATES",
    "REGISTRY_VERSION",
    "WRITE_CAPABILITIES",
    "CapabilityReport",
    "MarketplaceStatus",
    "assess_all",
    "assess_for_marketplace",
    "build_draft",
    "capability_status",
    "marketplace_status",
    "preview_listing",
    "publish_listing",
    "registry",
    "validate_draft",
]
