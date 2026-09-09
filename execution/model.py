"""Phase 4 execution domain model.

Builds on Part 2's execution primitives rather than replacing them: the
idempotency store, the ``Outcome`` taxonomy and the content-derived
idempotency key all come from ``part2.execution``. What this adds is the
apparatus that makes an approval mean something specific.

The central idea is that an approval authorises **one exact operation**, not a
product or an intention. It is bound to the operation type, the SKU, the
marketplace and a fingerprint of the payload, and any of those changing
invalidates it. Without that, "approved" degrades into a general permission
that outlives the thing it was granted for — which is how an operator ends up
having authorised a price change and received a publication.

Nothing here performs an external write. Live mode exists structurally and is
off unless explicitly enabled; dry run is the default and cannot be changed by
a client request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

EXECUTION_MODEL_VERSION = "execution/1.0.0"

#: Live execution is enabled only by this variable being explicitly true.
#: Absent, empty or anything else means dry run. There is deliberately no
#: "default to live in production" path.
LIVE_ENABLED_ENV_VAR = "ECOM_AUTOPILOT_LIVE_EXECUTION"


class ExecutionMode(StrEnum):
    DRY_RUN = "DRY_RUN"
    LIVE = "LIVE"


class ExecutionStatus(StrEnum):
    """Lifecycle of one execution request."""

    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"

    @property
    def is_terminal(self) -> bool:
        return self in (
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.REJECTED,
            ExecutionStatus.CANCELLED,
        )


#: Legal lifecycle transitions. EXECUTING is reachable only from APPROVED, so
#: there is no path from PENDING_APPROVAL to any external effect.
LEGAL_TRANSITIONS: dict[ExecutionStatus, frozenset[ExecutionStatus]] = {
    ExecutionStatus.PENDING_APPROVAL: frozenset(
        {ExecutionStatus.APPROVED, ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED}
    ),
    ExecutionStatus.APPROVED: frozenset(
        {
            ExecutionStatus.EXECUTING,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.REQUIRES_REVIEW,
        }
    ),
    ExecutionStatus.EXECUTING: frozenset(
        {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.REQUIRES_REVIEW,
        }
    ),
    ExecutionStatus.REQUIRES_REVIEW: frozenset(
        {ExecutionStatus.APPROVED, ExecutionStatus.CANCELLED, ExecutionStatus.REJECTED}
    ),
    ExecutionStatus.SUCCEEDED: frozenset(),
    ExecutionStatus.FAILED: frozenset({ExecutionStatus.REQUIRES_REVIEW}),
    ExecutionStatus.REJECTED: frozenset(),
    ExecutionStatus.CANCELLED: frozenset(),
}


class OperationType(StrEnum):
    """Operations the existing adapters actually declare.

    Deliberately mapped one-to-one onto ``part2.marketplace.Capability``. An
    operation with no corresponding adapter capability cannot be requested at
    all, so there is no way to queue something that could never run.
    """

    LISTING_PUBLISH = "LISTING_PUBLISH"
    LISTING_UPDATE = "LISTING_UPDATE"
    PRICE_UPDATE = "PRICE_UPDATE"
    INVENTORY_UPDATE = "INVENTORY_UPDATE"


#: Operation -> the adapter capability it requires.
OPERATION_CAPABILITY: dict[OperationType, str] = {
    OperationType.LISTING_PUBLISH: "CREATE_LISTING",
    OperationType.LISTING_UPDATE: "UPDATE_LISTING",
    OperationType.PRICE_UPDATE: "UPDATE_PRICE",
    OperationType.INVENTORY_UPDATE: "UPDATE_INVENTORY",
}


class RetryClass(StrEnum):
    """How an outcome may be retried.

    ``UNKNOWN_OUTCOME`` is the one that matters. A request that was sent but
    whose response was lost may already have taken effect; retrying it blindly
    is how one price change becomes two. It requires reconciliation, never an
    automatic retry.
    """

    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"

    @property
    def may_retry_automatically(self) -> bool:
        return self is RetryClass.RETRYABLE


class GateResult(StrEnum):
    PASS = "PASS"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class GateOutcome:
    """One safety check and its verdict."""

    gate: str
    result: GateResult
    detail: str = ""

    @property
    def blocks(self) -> bool:
        return self.result is GateResult.BLOCK

    def as_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "result": str(self.result), "detail": self.detail}


def fingerprint(payload: Mapping[str, Any]) -> str:
    """Stable content hash of an executable payload.

    Derived from the content alone -- sorted keys, no timestamps, no random
    salt -- so the same payload always fingerprints identically and any change
    is detectable. A timestamp-based key would make every resubmission look
    novel, which would defeat both idempotency and change detection.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_idempotency_key(
    operation: OperationType, marketplace: str, sku_code: str, payload_fingerprint: str
) -> str:
    """Deterministic key for one intended external effect.

    Keyed on the intent, not the attempt: the same operation on the same SKU
    with the same payload always produces the same key, so a resubmission is
    recognised as the same effect rather than a new one.
    """
    return hashlib.sha256(
        "|".join([str(operation), marketplace, sku_code, payload_fingerprint]).encode()
    ).hexdigest()[:32]


@dataclass(frozen=True)
class Approval:
    """A human authorising one exact operation.

    Bound to the operation, SKU, marketplace and payload fingerprint. It
    authorises that combination and nothing else.
    """

    approved_by: str
    approved_at: datetime
    operation: OperationType
    marketplace: str
    sku_code: str
    payload_fingerprint: str

    def authorises(
        self,
        *,
        operation: OperationType,
        marketplace: str,
        sku_code: str,
        payload_fingerprint: str,
    ) -> tuple[bool, str]:
        """Whether this approval covers the operation now being attempted."""
        if self.operation is not operation:
            return False, f"approved {self.operation}, attempted {operation}"
        if self.marketplace != marketplace:
            return False, f"approved for {self.marketplace}, attempted {marketplace}"
        if self.sku_code != sku_code:
            return False, f"approved for {self.sku_code}, attempted {sku_code}"
        if self.payload_fingerprint != payload_fingerprint:
            return False, (
                "the payload changed after approval; the approved version no "
                "longer matches what would be sent"
            )
        return True, ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat(),
            "operation": str(self.operation),
            "marketplace": self.marketplace,
            "sku_code": self.sku_code,
            "payload_fingerprint": self.payload_fingerprint,
        }


@dataclass(frozen=True)
class AuditEntry:
    """One append-only lifecycle record."""

    execution_id: str
    at: datetime
    actor: str
    from_status: str | None
    to_status: str
    detail: str = ""
    mode: str = str(ExecutionMode.DRY_RUN)

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "at": self.at.isoformat(),
            "actor": self.actor,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "detail": self.detail,
            "mode": self.mode,
        }


@dataclass
class ExecutionResult:
    """The structured outcome of one execution attempt.

    ``side_effect_occurred`` is stated explicitly on every result rather than
    inferred from the status, because "did this change anything out there" is
    the question an operator actually needs answered after a failure.
    """

    execution_id: str
    idempotency_key: str
    operation: OperationType
    marketplace: str
    sku_code: str
    status: ExecutionStatus
    mode: ExecutionMode
    payload_fingerprint: str
    started_at: datetime
    completed_at: datetime | None = None
    external_reference: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_class: RetryClass | None = None
    side_effect_occurred: bool = False
    gates: list[GateOutcome] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "idempotency_key": self.idempotency_key,
            "operation": str(self.operation),
            "marketplace": self.marketplace,
            "sku_code": self.sku_code,
            "status": str(self.status),
            "mode": str(self.mode),
            "payload_fingerprint": self.payload_fingerprint,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "external_reference": self.external_reference,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_class": str(self.retry_class) if self.retry_class else None,
            "side_effect_occurred": self.side_effect_occurred,
            "gates": [g.as_dict() for g in self.gates],
        }


class IllegalTransition(RuntimeError):
    """Raised when a lifecycle transition is not permitted."""


@dataclass
class ExecutionRequest:
    """One requested operation and its whole lifecycle."""

    operation: OperationType
    marketplace: str
    sku_code: str
    payload: dict[str, Any]
    requested_by: str
    reason: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING_APPROVAL
    approval: Approval | None = None
    result: ExecutionResult | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    audit: list[AuditEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._record(actor=self.requested_by, to_status=self.status, detail=self.reason)

    @property
    def payload_fingerprint(self) -> str:
        """Recomputed from the live payload every time it is read.

        Deliberately not cached: the point is to detect a payload that changed
        after approval, and a cached value would agree with itself forever.
        """
        return fingerprint(self.payload)

    @property
    def idempotency_key(self) -> str:
        return build_idempotency_key(
            self.operation, self.marketplace, self.sku_code, self.payload_fingerprint
        )

    def _record(
        self,
        *,
        actor: str,
        to_status: ExecutionStatus,
        detail: str = "",
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
        from_status: ExecutionStatus | None = None,
    ) -> None:
        self.audit.append(
            AuditEntry(
                execution_id=str(self.id),
                at=datetime.now(UTC),
                actor=actor,
                from_status=str(from_status) if from_status else None,
                to_status=str(to_status),
                detail=detail,
                mode=str(mode),
            )
        )

    def transition(
        self,
        to_status: ExecutionStatus,
        *,
        actor: str,
        detail: str = "",
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
    ) -> None:
        """Move to a new status, refusing illegal moves.

        Holds a lock so two concurrent callers cannot both observe APPROVED
        and both proceed to EXECUTING.
        """
        with self._lock:
            allowed = LEGAL_TRANSITIONS.get(self.status, frozenset())
            if to_status not in allowed:
                raise IllegalTransition(
                    f"cannot move from {self.status} to {to_status}; "
                    f"permitted: {sorted(str(s) for s in allowed) or 'none'}"
                )
            previous = self.status
            self.status = to_status
            self._record(
                actor=actor,
                to_status=to_status,
                detail=detail,
                mode=mode,
                from_status=previous,
            )

    def approve(
        self, approver: str, *, operation: OperationType | None = None
    ) -> Approval:
        """Record a human approval bound to the current payload."""
        approval = Approval(
            approved_by=approver,
            approved_at=datetime.now(UTC),
            operation=operation or self.operation,
            marketplace=self.marketplace,
            sku_code=self.sku_code,
            payload_fingerprint=self.payload_fingerprint,
        )
        self.transition(
            ExecutionStatus.APPROVED,
            actor=approver,
            detail=f"approved fingerprint {approval.payload_fingerprint[:12]}",
        )
        self.approval = approval
        return approval

    def reject(self, actor: str, reason: str = "") -> None:
        self.transition(ExecutionStatus.REJECTED, actor=actor, detail=reason)

    def cancel(self, actor: str, reason: str = "") -> None:
        self.transition(ExecutionStatus.CANCELLED, actor=actor, detail=reason)

    def as_dict(self) -> dict[str, Any]:
        """Serialise. Contains no credential by construction."""
        return {
            "id": str(self.id),
            "operation": str(self.operation),
            "marketplace": self.marketplace,
            "sku_code": self.sku_code,
            "payload": dict(self.payload),
            "payload_fingerprint": self.payload_fingerprint,
            "idempotency_key": self.idempotency_key,
            "status": str(self.status),
            "requested_by": self.requested_by,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "approval": self.approval.as_dict() if self.approval else None,
            "result": self.result.as_dict() if self.result else None,
            "audit": [a.as_dict() for a in self.audit],
        }


def live_execution_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether live execution has been explicitly turned on.

    Anything other than an exact affirmative is dry run. There is no
    environment in which the absence of configuration means live.
    """
    env = dict(environ) if environ is not None else dict(os.environ)
    return env.get(LIVE_ENABLED_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


__all__ = [
    "EXECUTION_MODEL_VERSION",
    "LEGAL_TRANSITIONS",
    "LIVE_ENABLED_ENV_VAR",
    "OPERATION_CAPABILITY",
    "Approval",
    "AuditEntry",
    "ExecutionMode",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "GateOutcome",
    "GateResult",
    "IllegalTransition",
    "OperationType",
    "RetryClass",
    "build_idempotency_key",
    "fingerprint",
    "live_execution_enabled",
]
