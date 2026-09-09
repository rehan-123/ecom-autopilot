"""Execution safety: gates, idempotency, failure classification, retries.

Part 1 decides what *should* happen. Part 2 makes things happen outside this
process, which introduces three failure modes Part 1 never had: an action can
be attempted twice, an action can fail halfway, and an action's outcome can be
genuinely unknown. Everything in this module exists to keep those three from
turning into duplicate orders, phantom listings, or silent corruption.

Three rules are absolute:

**UNKNOWN is never success.** A timeout after the request was sent means the
marketplace may or may not have acted. Recording that as failure invites a
retry that duplicates; recording it as success invites drift. It is recorded
as ``UNKNOWN`` and routed to a human.

**Every external write is idempotent.** Each carries a caller-supplied key
derived from the intent, not from the attempt. A replay of the same intent
returns the first result rather than acting again.

**Recommendation is not execution.** They are separate states with an approval
between them, so nothing reaches a marketplace because a ranking function
liked it.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

EXECUTION_VERSION = "execution/1.0.0"


class Outcome(StrEnum):
    """How an external interaction ended.

    ``UNKNOWN`` is a first-class result, not an error case. Most integration
    bugs come from systems that only model success and failure and therefore
    have to guess which one a timeout was.
    """

    OK = "OK"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"
    UNKNOWN = "UNKNOWN"
    DRY_RUN = "DRY_RUN"

    @property
    def is_success(self) -> bool:
        """Only OK counts. DRY_RUN did not touch anything external."""
        return self is Outcome.OK

    @property
    def needs_human(self) -> bool:
        return self in (Outcome.UNKNOWN, Outcome.PERMANENT_FAILURE, Outcome.NOT_SUPPORTED)


class ExecutionState(StrEnum):
    """Lifecycle of an action from proposal to completion."""

    RECOMMENDED = "RECOMMENDED"
    APPROVED = "APPROVED"
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


#: Legal execution transitions. A recommendation cannot reach a marketplace
#: without passing through APPROVED, which is the whole point of the split.
EXECUTION_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.RECOMMENDED: frozenset(
        {ExecutionState.APPROVED, ExecutionState.REQUIRES_REVIEW, ExecutionState.FAILED}
    ),
    ExecutionState.APPROVED: frozenset({ExecutionState.QUEUED, ExecutionState.FAILED}),
    ExecutionState.QUEUED: frozenset({ExecutionState.EXECUTING, ExecutionState.FAILED}),
    ExecutionState.EXECUTING: frozenset(
        {
            ExecutionState.EXECUTED,
            ExecutionState.FAILED,
            # A genuinely unknown outcome parks here rather than guessing.
            ExecutionState.REQUIRES_REVIEW,
        }
    ),
    ExecutionState.REQUIRES_REVIEW: frozenset(
        {ExecutionState.APPROVED, ExecutionState.FAILED, ExecutionState.EXECUTED}
    ),
    ExecutionState.EXECUTED: frozenset(),
    ExecutionState.FAILED: frozenset({ExecutionState.RECOMMENDED}),
}


class IllegalExecutionTransition(ValueError):
    """Raised when an execution state change is not permitted."""


def require_execution_transition(
    from_state: ExecutionState, to_state: ExecutionState
) -> None:
    allowed = EXECUTION_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise IllegalExecutionTransition(
            f"{from_state} -> {to_state} is not legal; "
            f"allowed: {sorted(str(s) for s in allowed) or 'none (terminal)'}"
        )


class NotApproved(PermissionError):
    """Raised when execution is attempted without an approval."""


@dataclass(frozen=True)
class ActionResult:
    """The outcome of one attempted external action."""

    outcome: Outcome
    action: str
    idempotency_key: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    duration_ms: int = 0
    correlation_id: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def changed_external_state(self) -> bool:
        """Whether anything outside this process actually moved."""
        return self.outcome is Outcome.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "action": self.action,
            "idempotency_key": self.idempotency_key,
            "detail": self.detail,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "correlation_id": self.correlation_id,
            "at": self.at.isoformat(),
            # Payload is deliberately excluded from the audit summary: it can
            # carry marketplace responses, and this dict is logged.
        }


def idempotency_key(*parts: Any) -> str:
    """Derive a stable key from an action's *intent*.

    Built from what the caller means to do -- SKU, marketplace, target price --
    never from a timestamp or attempt counter. Two attempts at the same intent
    must produce the same key, or retrying would duplicate.
    """
    material = "|".join(str(p) for p in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class IdempotencyStore:
    """Remembers the result of every external write, keyed by intent.

    Held in memory here; the ledger is the durable record. The interface is
    what matters: ``begin`` either reserves the key or hands back the earlier
    result, so a duplicate can never reach the transport.
    """

    def __init__(self) -> None:
        self._results: dict[str, ActionResult] = {}
        self._in_flight: set[str] = set()

    def lookup(self, key: str) -> ActionResult | None:
        return self._results.get(key)

    def begin(self, key: str) -> ActionResult | None:
        """Reserve a key. Returns a prior result if this intent already ran.

        An in-flight duplicate returns UNKNOWN rather than waiting: two workers
        racing on the same intent is itself a condition worth surfacing, and
        blocking would just move the race.
        """
        existing = self._results.get(key)
        if existing is not None:
            logger.info(
                "idempotent replay", extra={"key": key, "action": existing.action}
            )
            return existing
        if key in self._in_flight:
            return ActionResult(
                outcome=Outcome.UNKNOWN,
                action="concurrent-duplicate",
                idempotency_key=key,
                detail="another worker is executing this exact intent",
            )
        self._in_flight.add(key)
        return None

    def complete(self, key: str, result: ActionResult) -> ActionResult:
        """Record a terminal result.

        Retryable failures are not stored: the intent has not been settled, so
        a later attempt must be allowed to proceed.
        """
        self._in_flight.discard(key)
        if result.outcome is not Outcome.RETRYABLE_FAILURE:
            self._results[key] = result
        return result

    def abandon(self, key: str) -> None:
        self._in_flight.discard(key)

    def __len__(self) -> int:
        return len(self._results)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries with exponential backoff and jitter.

    ``retry_non_idempotent`` defaults to False and should stay that way for
    anything that creates an order or moves money: retrying a write whose
    outcome is unknown is how duplicates happen.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.05
    max_delay_s: float = 2.0
    jitter: bool = True
    retry_non_idempotent: bool = False

    def delay_for(self, attempt: int) -> float:
        delay = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        if self.jitter:
            # Jitter stops a fleet of workers retrying in lockstep.
            delay *= 0.5 + random.random() / 2
        return delay


def execute_with_retry(
    operation: Callable[[], ActionResult],
    *,
    key: str,
    store: IdempotencyStore,
    policy: RetryPolicy | None = None,
    idempotent: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> ActionResult:
    """Run an external write once, safely, with bounded retries.

    Consults the idempotency store first, so a repeated intent returns the
    original result without touching the transport. A non-idempotent operation
    is never retried after an unknown outcome, because the first attempt may
    have succeeded.
    """
    policy = policy or RetryPolicy()

    prior = store.begin(key)
    if prior is not None:
        return prior

    started = time.perf_counter()
    last: ActionResult | None = None
    try:
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = operation()
            except Exception as exc:  # noqa: BLE001 - any transport error is data
                result = ActionResult(
                    outcome=Outcome.UNKNOWN,
                    action="unhandled",
                    idempotency_key=key,
                    detail=f"{type(exc).__name__}: {exc}",
                )

            elapsed = int((time.perf_counter() - started) * 1000)
            result = ActionResult(
                outcome=result.outcome,
                action=result.action,
                idempotency_key=key,
                detail=result.detail,
                payload=result.payload,
                attempts=attempt,
                duration_ms=elapsed,
                correlation_id=result.correlation_id,
            )
            last = result

            if result.outcome is not Outcome.RETRYABLE_FAILURE:
                return store.complete(key, result)
            if not idempotent and not policy.retry_non_idempotent:
                # The first attempt may have taken effect. Stop and ask.
                return store.complete(
                    key,
                    ActionResult(
                        outcome=Outcome.UNKNOWN,
                        action=result.action,
                        idempotency_key=key,
                        detail=(
                            "non-idempotent operation failed retryably; a retry "
                            "could duplicate it, so the outcome is unknown"
                        ),
                        attempts=attempt,
                        duration_ms=elapsed,
                    ),
                )
            if attempt < policy.max_attempts:
                sleep(policy.delay_for(attempt))

        assert last is not None
        return store.complete(key, last)
    finally:
        store.abandon(key)


@dataclass
class ExecutionRecord:
    """One auditable action, from recommendation through to outcome."""

    action: str
    sku_code: str
    marketplace: str
    proposed: dict[str, Any]
    reason: str
    state: ExecutionState = ExecutionState.RECOMMENDED
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    approved_by: str | None = None
    result: ActionResult | None = None
    history: list[tuple[ExecutionState, datetime, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append((self.state, datetime.now(UTC), "created"))

    def transition(self, to_state: ExecutionState, *, note: str = "") -> None:
        require_execution_transition(self.state, to_state)
        self.state = to_state
        self.history.append((to_state, datetime.now(UTC), note))

    def approve(self, approver: str) -> None:
        """A human authorises this specific action."""
        self.transition(ExecutionState.APPROVED, note=f"approved by {approver}")
        self.approved_by = approver

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "correlation_id": self.correlation_id,
            "action": self.action,
            "sku_code": self.sku_code,
            "marketplace": self.marketplace,
            "state": str(self.state),
            "approved_by": self.approved_by,
            "reason": self.reason,
            "proposed": {k: str(v) for k, v in self.proposed.items()},
            "result": self.result.as_dict() if self.result else None,
            "history": [
                {"state": str(s), "at": t.isoformat(), "note": n}
                for s, t, n in self.history
            ],
        }


def run_approved(
    record: ExecutionRecord,
    operation: Callable[[], ActionResult],
    *,
    store: IdempotencyStore,
    policy: RetryPolicy | None = None,
    idempotent: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> ActionResult:
    """Execute an action that has been approved, and only then.

    The approval check is here rather than in each adapter so there is exactly
    one place an unapproved action could slip through, and it is tested.
    """
    if record.state is not ExecutionState.APPROVED:
        raise NotApproved(
            f"{record.action} for {record.sku_code} is {record.state}; "
            "only an APPROVED action may execute"
        )

    record.transition(ExecutionState.QUEUED, note="queued for execution")
    record.transition(ExecutionState.EXECUTING, note="calling adapter")

    key = idempotency_key(
        record.action,
        record.sku_code,
        record.marketplace,
        *sorted(f"{k}={v}" for k, v in record.proposed.items()),
    )
    result = execute_with_retry(
        operation, key=key, store=store, policy=policy, idempotent=idempotent, sleep=sleep
    )
    record.result = result

    if result.outcome in (Outcome.OK, Outcome.DRY_RUN):
        record.transition(ExecutionState.EXECUTED, note=str(result.outcome))
    elif result.outcome.needs_human:
        record.transition(ExecutionState.REQUIRES_REVIEW, note=result.detail[:120])
    else:
        record.transition(ExecutionState.FAILED, note=result.detail[:120])

    logger.info(
        "action executed",
        extra={
            "correlation_id": record.correlation_id,
            "operation": record.action,
            "sku": record.sku_code,
            "marketplace": record.marketplace,
            "result": str(result.outcome),
            "duration": result.duration_ms,
        },
    )
    return result


__all__ = [
    "EXECUTION_TRANSITIONS",
    "EXECUTION_VERSION",
    "ActionResult",
    "ExecutionRecord",
    "ExecutionState",
    "IdempotencyStore",
    "IllegalExecutionTransition",
    "NotApproved",
    "Outcome",
    "RetryPolicy",
    "execute_with_retry",
    "idempotency_key",
    "require_execution_transition",
    "run_approved",
]
