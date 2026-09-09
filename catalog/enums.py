"""Controlled vocabularies shared by Pydantic schemas and SQLAlchemy tables.

Kept in a dedicated module (rather than inside ``schemas.py``) so that the
persistence layer can validate against the same vocabulary without importing
the application-level schemas. That keeps the dependency direction one-way:

    enums  <-  schemas
    enums  <-  database.tables

Adding a member is backwards compatible. Removing or renaming one is not:
historical audit events carry these strings verbatim, so a rename breaks
replay of existing history. Deprecate instead of renaming.
"""

from __future__ import annotations

from enum import StrEnum


class SupplierType(StrEnum):
    """How a supplier sources the goods it offers.

    ``MARKETPLACE_RETAIL`` exists so retail sources can be *recorded* and
    explicitly flagged. A retail marketplace is not a wholesale source: its
    prices sit at or near landed cost and it issues B2C receipts rather than
    tax invoices, so no input tax credit is available. Later milestones treat
    this type as a penalty signal, never as a preferred supplier.
    """

    WHOLESALER = "WHOLESALER"
    MANUFACTURER = "MANUFACTURER"
    DISTRIBUTOR = "DISTRIBUTOR"
    MARKETPLACE_RETAIL = "MARKETPLACE_RETAIL"
    OTHER = "OTHER"


class GSTStatus(StrEnum):
    REGISTERED = "REGISTERED"
    UNREGISTERED = "UNREGISTERED"
    UNKNOWN = "UNKNOWN"


class SKUState(StrEnum):
    """Lifecycle states for a sellable unit.

    Part 1 terminates at ``LISTING_READY``. There is deliberately no state
    representing publication, purchase, or fulfilment: those belong to Part 2
    and must not be representable by this codebase.
    """

    DISCOVERED = "DISCOVERED"
    DATA_COLLECTION = "DATA_COLLECTION"
    ECONOMIC_ANALYSIS = "ECONOMIC_ANALYSIS"
    SIGNAL_COLLECTION = "SIGNAL_COLLECTION"
    BLOCKED_AWAITING_DATA = "BLOCKED_AWAITING_DATA"
    PRICE_HEADROOM_CHECK = "PRICE_HEADROOM_CHECK"
    SUPPLIER_ANALYSIS = "SUPPLIER_ANALYSIS"
    RISK_ANALYSIS = "RISK_ANALYSIS"
    COMPLIANCE_GATE = "COMPLIANCE_GATE"
    BLOCKED = "BLOCKED"
    BLOCKED_PENDING_REVIEW = "BLOCKED_PENDING_REVIEW"
    AI_REVIEW = "AI_REVIEW"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVED = "APPROVED"
    LISTING_READY = "LISTING_READY"
    REJECTED = "REJECTED"


#: States from which no further transition is expected within Part 1.
TERMINAL_STATES: frozenset[SKUState] = frozenset(
    {SKUState.LISTING_READY, SKUState.REJECTED}
)


class EvidenceType(StrEnum):
    TITLE = "TITLE"
    SPECIFICATION = "SPECIFICATION"
    IMAGE_HASH = "IMAGE_HASH"
    MANUFACTURER_CODE = "MANUFACTURER_CODE"
    EAN = "EAN"
    BARCODE = "BARCODE"
    MODEL_NUMBER = "MODEL_NUMBER"
    OTHER = "OTHER"


class IdentityConfidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNRESOLVED = "UNRESOLVED"


class ScenarioName(StrEnum):
    BEST = "BEST"
    EXPECTED = "EXPECTED"
    WORST = "WORST"


class SignalStatus(StrEnum):
    """Status of a market signal.

    ``DATA_UNAVAILABLE`` is a first-class outcome, not an error. Roughly half
    the demand and competition signals for Indian marketplaces have no
    legitimate source. Recording the absence explicitly is what stops a
    missing value from being silently treated as an average one.
    """

    OBSERVED = "OBSERVED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    STALE = "STALE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class SignalSource(StrEnum):
    MANUAL = "MANUAL"
    GOOGLE_TRENDS = "GOOGLE_TRENDS"
    SELLER_REPORT = "SELLER_REPORT"
    OWN_HISTORY = "OWN_HISTORY"
    SUPPLIER = "SUPPLIER"
    OTHER = "OTHER"


class GateResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    MARGINAL = "MARGINAL"


class HeadroomResult(StrEnum):
    PASS = "PASS"
    MARGINAL = "MARGINAL"
    BLOCK = "BLOCK"


class RiskTier(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH_RETURN = "HIGH_RETURN"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class ComplianceResult(StrEnum):
    """Outcome of a compliance check.

    ``UNKNOWN`` must never be promoted to ``PASS`` by any automated path.
    It routes to ``REQUIRES_HUMAN_REVIEW``. Categories under mandatory BIS
    certification carry statutory penalties starting at Rs 2 lakh, so an
    unresolved check is a blocker, not a warning.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class ScoreResult(StrEnum):
    SCORED = "SCORED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class TransitionTrigger(StrEnum):
    RULE = "RULE"
    HUMAN = "HUMAN"
    INGEST = "INGEST"
    SCHEDULE = "SCHEDULE"


class ApprovalDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


class AIService(StrEnum):
    """Claude service call sites.

    Two distinct Judges: identity resolution runs early over raw supplier
    offers, commercial judgement runs late over the fully assembled record.
    Different evidence, different schemas, different failure modes.
    """

    EXTRACTOR = "EXTRACTOR"
    JUDGE_IDENTITY = "JUDGE_IDENTITY"
    JUDGE_COMMERCIAL = "JUDGE_COMMERCIAL"
    WRITER = "WRITER"
    #: Suggests *what to investigate*, never what is true. Its output is a
    #: hypothesis list; every factual field it touches stays UNAVAILABLE until
    #: a source supplies it.
    RESEARCHER = "RESEARCHER"


class EventType(StrEnum):
    """Audit ledger event types.

    Every mutation of business state emits exactly one of these. The string
    values are persisted verbatim and are load-bearing for replay: treat them
    as a public interface.
    """

    SUPPLIER_CREATED = "SUPPLIER_CREATED"
    PRODUCT_CREATED = "PRODUCT_CREATED"
    PRODUCT_IDENTITY_EVIDENCE_ADDED = "PRODUCT_IDENTITY_EVIDENCE_ADDED"
    SKU_CREATED = "SKU_CREATED"
    SKU_STATE_CHANGED = "SKU_STATE_CHANGED"
    SUPPLIER_OFFER_CREATED = "SUPPLIER_OFFER_CREATED"
    REFERENCE_DATA_LOADED = "REFERENCE_DATA_LOADED"
    LANDED_COST_COMPUTED = "LANDED_COST_COMPUTED"
    MARKET_SIGNAL_RECORDED = "MARKET_SIGNAL_RECORDED"
    MARKET_PRICE_REFERENCE_RECORDED = "MARKET_PRICE_REFERENCE_RECORDED"
    ECONOMICS_CALCULATED = "ECONOMICS_CALCULATED"
    PRICE_HEADROOM_CHECKED = "PRICE_HEADROOM_CHECKED"
    RISK_ASSESSED = "RISK_ASSESSED"
    COMPLIANCE_ASSESSED = "COMPLIANCE_ASSESSED"
    OPPORTUNITY_SCORED = "OPPORTUNITY_SCORED"
    AI_INVOKED = "AI_INVOKED"
    APPROVAL_RECORDED = "APPROVAL_RECORDED"
    ERROR = "ERROR"


class ActorKind(StrEnum):
    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"
    AI = "AI"
