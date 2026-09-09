"""Phase 3: marketplace adapter and readiness layer.

Readiness infrastructure only. Nothing here publishes a listing, places an
order, syncs inventory, moves money or contacts a customer. Write capabilities
return NOT_ENABLED_FOR_PHASE_3 even where an adapter and credentials exist.

Consumes Phase 2 outputs unchanged: compliance state, economics and supplier
confirmation are read from the opportunity rather than recomputed.
"""

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
from marketplace.registry import (
    CapabilityReport,
    MarketplaceStatus,
    assess_all,
    assess_for_marketplace,
    build_draft,
    capability_status,
    marketplace_status,
    preview_listing,
    publish_listing,
    registry,
    validate_draft,
)

MARKETPLACE_LAYER_VERSION = "marketplace/1.0.0"

__all__ = [
    "MARKETPLACE_LAYER_VERSION",
    "CapabilityReport",
    "CapabilityStatus",
    "DraftField",
    "FieldProvenance",
    "InventoryState",
    "ListingAssessment",
    "ListingDraft",
    "MarketplaceStatus",
    "OperationResult",
    "Readiness",
    "Severity",
    "ValidationFinding",
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
