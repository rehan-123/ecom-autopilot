"""Durable execution ledger: the database arbitrates idempotency, not memory.

Phase 4 kept idempotency in a per-process dictionary. That is correct for one
worker and worthless for two: each process holds its own store, both see a free
key, and both execute. This module moves the decision into SQLite, where a
UNIQUE index on ``idempotency_key`` makes the claim atomic across processes.

The claim is the whole mechanism. A worker inserts its claim row inside a
``BEGIN IMMEDIATE`` transaction; exactly one insert can succeed, and the loser
reads back the winner's row rather than executing. No Python lock is involved,
so it holds across processes, containers and restarts.

Crash recovery follows from the same table. A row left in ``EXECUTING`` means a
process died mid-flight, and whether the external effect landed is genuinely
unknown. It is never re-executed and never assumed successful: it becomes
``REQUIRES_REVIEW`` awaiting reconciliation.

No credential ever enters this ledger. It stores what happened, not what
authorised it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from database.connection import read_only, transaction

logger = logging.getLogger(__name__)

LEDGER_VERSION = "execution-ledger/1.0.0"

#: A row still EXECUTING after this long is presumed to belong to a dead
#: process. It is not retried -- it is surfaced for reconciliation.
STALE_EXECUTION_AFTER = timedelta(minutes=15)

execution_metadata = MetaData()

#: The durable execution ledger. UNIQUE(idempotency_key) is the arbiter: it is
#: what makes two concurrent workers resolve to one execution.
execution_ledger = Table(
    "execution_ledger",
    execution_metadata,
    Column("idempotency_key", String(64), primary_key=True),
    Column("execution_id", String(36), nullable=False),
    Column("operation", String(32), nullable=False),
    Column("sku_code", String(64), nullable=False),
    Column("marketplace", String(32), nullable=False),
    Column("payload_fingerprint", String(64), nullable=False),
    Column("approval_id", String(64)),
    Column("execution_state", String(32), nullable=False),
    Column("execution_mode", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("external_reference", String(128)),
    Column("side_effect_occurred", Boolean, nullable=False, default=False),
    Column("outcome", String(32)),
    Column("error_code", String(64)),
    Column("error_message", Text),
    Column("retry_count", Integer, nullable=False, default=0),
    Column("reconciliation_state", String(32), nullable=False),
    UniqueConstraint("idempotency_key", name="uq_execution_idempotency_key"),
)


class ReconciliationState(StrEnum):
    """Where an uncertain execution stands.

    ``UNKNOWN_OUTCOME`` is not a failure and not a success. Only
    ``CONFIRMED_FAILURE`` makes an operation safely retryable, because only
    then is it known that nothing landed.
    """

    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    RECONCILING = "RECONCILING"
    CONFIRMED_SUCCESS = "CONFIRMED_SUCCESS"
    CONFIRMED_FAILURE = "CONFIRMED_FAILURE"

    @property
    def safe_to_retry(self) -> bool:
        return self is ReconciliationState.CONFIRMED_FAILURE


@dataclass(frozen=True)
class ClaimResult:
    """The outcome of attempting to claim an idempotency key."""

    won: bool
    idempotency_key: str
    execution_id: str
    existing_state: str | None = None
    existing_outcome: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "won": self.won,
            "idempotency_key": self.idempotency_key,
            "execution_id": self.execution_id,
            "existing_state": self.existing_state,
            "existing_outcome": self.existing_outcome,
            "detail": self.detail,
        }


def install_execution_ledger(engine: Engine) -> None:
    """Create the ledger table if absent.

    Additive only. Existing tables and rows are untouched, so this is safe to
    run against a database that already holds history.
    """
    execution_metadata.create_all(engine, checkfirst=True)


class PersistentIdempotencyStore:
    """Database-backed idempotency, authoritative across processes.

    Deliberately not a drop-in replacement for the in-memory store: claiming
    is a distinct operation from looking up, because the whole point is that
    the check and the claim happen in one atomic step. A ``lookup`` followed by
    an ``insert`` would reintroduce exactly the race this exists to close.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        install_execution_ledger(engine)

    def claim(
        self,
        *,
        idempotency_key: str,
        execution_id: str,
        operation: str,
        sku_code: str,
        marketplace: str,
        payload_fingerprint: str,
        execution_mode: str,
        approval_id: str | None = None,
    ) -> ClaimResult:
        """Atomically claim the right to execute this operation.

        Exactly one caller wins. The insert happens inside BEGIN IMMEDIATE, so
        the UNIQUE constraint resolves the race in the database rather than in
        any one process's memory.
        """
        now = datetime.now(UTC)
        try:
            with transaction(self.engine) as conn:
                conn.execute(
                    execution_ledger.insert().values(
                        idempotency_key=idempotency_key,
                        execution_id=execution_id,
                        operation=operation,
                        sku_code=sku_code,
                        marketplace=marketplace,
                        payload_fingerprint=payload_fingerprint,
                        approval_id=approval_id,
                        execution_state="EXECUTING",
                        execution_mode=execution_mode,
                        created_at=now,
                        updated_at=now,
                        started_at=now,
                        side_effect_occurred=False,
                        retry_count=0,
                        reconciliation_state=str(ReconciliationState.NOT_REQUIRED),
                    )
                )
        except IntegrityError:
            existing = self.get(idempotency_key)
            logger.info(
                "idempotency claim lost to an existing execution",
                extra={"operation": "claim", "result": idempotency_key[:12]},
            )
            return ClaimResult(
                won=False,
                idempotency_key=idempotency_key,
                execution_id=(existing or {}).get("execution_id", ""),
                existing_state=(existing or {}).get("execution_state"),
                existing_outcome=(existing or {}).get("outcome"),
                detail=(
                    "this exact operation has already been claimed; returning the "
                    "existing execution rather than running it again"
                ),
            )

        return ClaimResult(
            won=True, idempotency_key=idempotency_key, execution_id=execution_id
        )

    def complete(
        self,
        idempotency_key: str,
        *,
        execution_state: str,
        outcome: str | None = None,
        side_effect_occurred: bool = False,
        external_reference: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        reconciliation_state: ReconciliationState = ReconciliationState.NOT_REQUIRED,
    ) -> None:
        """Record the result against a claimed key."""
        with transaction(self.engine) as conn:
            conn.execute(
                execution_ledger.update()
                .where(execution_ledger.c.idempotency_key == idempotency_key)
                .values(
                    execution_state=execution_state,
                    outcome=outcome,
                    side_effect_occurred=side_effect_occurred,
                    external_reference=external_reference,
                    error_code=error_code,
                    error_message=error_message,
                    reconciliation_state=str(reconciliation_state),
                    completed_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )

    def release(self, idempotency_key: str) -> None:
        """Release a claim that never ran.

        Used only when a gate blocked before anything was attempted, so a
        blocked attempt does not permanently consume the key.
        """
        with transaction(self.engine) as conn:
            conn.execute(
                execution_ledger.delete().where(
                    execution_ledger.c.idempotency_key == idempotency_key
                )
            )

    def get(self, idempotency_key: str) -> dict[str, Any] | None:
        with read_only(self.engine) as conn:
            row = (
                conn.execute(
                    select(execution_ledger).where(
                        execution_ledger.c.idempotency_key == idempotency_key
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def history(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Paginated history. Bounded so an export cannot become unbounded."""
        limit = max(1, min(limit, 500))
        with read_only(self.engine) as conn:
            rows = (
                conn.execute(
                    select(execution_ledger)
                    .order_by(execution_ledger.c.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]

    # -- crash recovery ---------------------------------------------------

    def stale_executions(
        self,
        *,
        now: datetime | None = None,
        older_than: timedelta = STALE_EXECUTION_AFTER,
    ) -> list[dict[str, Any]]:
        """Rows left EXECUTING by a process that died."""
        cutoff = (now or datetime.now(UTC)) - older_than
        with read_only(self.engine) as conn:
            rows = (
                conn.execute(
                    select(execution_ledger).where(
                        execution_ledger.c.execution_state == "EXECUTING",
                        execution_ledger.c.started_at < cutoff,
                    )
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]

    def recover_stale(
        self,
        *,
        now: datetime | None = None,
        older_than: timedelta = STALE_EXECUTION_AFTER,
    ) -> list[str]:
        """Mark abandoned executions for review. Never re-runs them.

        A process that died mid-execution may or may not have caused an
        external effect. Re-running risks a duplicate; assuming success risks a
        phantom. Both are worse than saying "a human must check this".
        """
        recovered: list[str] = []
        for row in self.stale_executions(now=now, older_than=older_than):
            with transaction(self.engine) as conn:
                conn.execute(
                    execution_ledger.update()
                    .where(execution_ledger.c.idempotency_key == row["idempotency_key"])
                    .values(
                        execution_state="REQUIRES_REVIEW",
                        outcome="UNKNOWN_OUTCOME",
                        error_code="STALE_EXECUTION",
                        error_message=(
                            "the executing process did not finish. Whether the "
                            "external effect occurred is unknown; reconcile "
                            "against the marketplace before any retry."
                        ),
                        reconciliation_state=str(ReconciliationState.REQUIRES_REVIEW),
                        updated_at=datetime.now(UTC),
                    )
                )
            recovered.append(row["idempotency_key"])
            logger.warning(
                "stale execution marked for review",
                extra={"operation": "recover", "result": row["idempotency_key"][:12]},
            )
        return recovered

    # -- reconciliation ---------------------------------------------------

    def set_reconciliation(
        self, idempotency_key: str, state: ReconciliationState, *, detail: str = ""
    ) -> None:
        with transaction(self.engine) as conn:
            conn.execute(
                execution_ledger.update()
                .where(execution_ledger.c.idempotency_key == idempotency_key)
                .values(
                    reconciliation_state=str(state),
                    error_message=detail or None,
                    updated_at=datetime.now(UTC),
                )
            )

    def may_retry(self, idempotency_key: str) -> tuple[bool, str]:
        """Whether an operation may safely be retried.

        Only a confirmed failure qualifies. An unknown outcome never does,
        however long ago it happened.
        """
        row = self.get(idempotency_key)
        if row is None:
            return True, "no prior execution"
        state = ReconciliationState(row["reconciliation_state"])
        if state.safe_to_retry:
            return True, "reconciliation confirmed the operation did not take effect"
        if row["execution_state"] == "FAILED" and row["outcome"] != "UNKNOWN_OUTCOME":
            return True, "the prior attempt failed without external effect"
        return False, (
            f"prior execution is {row['execution_state']} with reconciliation "
            f"{state}; retrying could duplicate an effect that already occurred"
        )


__all__ = [
    "LEDGER_VERSION",
    "STALE_EXECUTION_AFTER",
    "ClaimResult",
    "PersistentIdempotencyStore",
    "ReconciliationState",
    "execution_ledger",
    "install_execution_ledger",
]
