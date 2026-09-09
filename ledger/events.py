"""The append-only audit ledger.

Deliberately offers no ``update_event`` and no ``delete_event``. A method that
should never be called should not exist: its presence invites use, and its
absence is a guarantee that survives refactoring.

Each event carries ``prev_hash`` and ``event_hash``, chaining every entry to
its predecessor. The database triggers in ``database.migrations`` block normal
mutation; the chain makes out-of-band mutation *detectable*. Triggers prevent,
the chain proves.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, Engine, func, select

from catalog.enums import EventType
from catalog.schemas import AuditEvent
from database.connection import read_only
from database.tables import audit_event
from database.types import canonical_json

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64
"""Predecessor hash for the first event in an empty ledger."""


class LedgerIntegrityError(RuntimeError):
    """Raised when the hash chain does not verify."""


def _compute_hash(event: AuditEvent, prev_hash: str) -> str:
    """Hash an event's content together with its predecessor's hash.

    ``seq`` and ``event_hash`` are excluded: ``seq`` is assigned by the
    database after hashing, and an event cannot contain its own hash. Every
    other field is covered, so altering any of them breaks verification.
    """
    body = {
        "id": str(event.id),
        "timestamp": event.timestamp.isoformat(timespec="microseconds"),
        "sku_id": str(event.sku_id) if event.sku_id else None,
        "product_id": str(event.product_id) if event.product_id else None,
        "supplier_id": str(event.supplier_id) if event.supplier_id else None,
        "event_type": str(event.event_type),
        "actor": event.actor,
        "inputs_json": event.inputs_json,
        "reference_versions_json": event.reference_versions_json,
        "calc_version": event.calc_version,
        "rule_id": event.rule_id,
        "result_json": event.result_json,
        "error_json": event.error_json,
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class EventRepository:
    """Read and append operations on the ledger.

    Instantiate with an ``Engine`` for standalone use, or pass a ``Connection``
    to ``append_event`` to enlist in a caller's transaction so that an entity
    write and its event commit together.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # -- write ------------------------------------------------------------

    def append_event(
        self, event: AuditEvent, *, conn: Connection | None = None
    ) -> AuditEvent:
        """Append one event and return it with ``seq`` and hashes populated.

        The read of the previous hash and the insert happen in the same
        transaction, so concurrent appends cannot fork the chain.
        """
        if conn is not None:
            return self._append(conn, event)
        with self._engine.begin() as owned_conn:
            return self._append(owned_conn, event)

    def _append(self, conn: Connection, event: AuditEvent) -> AuditEvent:
        prev_hash = (
            conn.execute(
                select(audit_event.c.event_hash)
                .order_by(audit_event.c.seq.desc())
                .limit(1)
            ).scalar()
            or GENESIS_HASH
        )
        event_hash = _compute_hash(event, prev_hash)

        result = conn.execute(
            audit_event.insert().values(
                id=event.id,
                timestamp=event.timestamp,
                sku_id=event.sku_id,
                product_id=event.product_id,
                supplier_id=event.supplier_id,
                event_type=str(event.event_type),
                actor=event.actor,
                inputs_json=event.inputs_json,
                reference_versions_json=event.reference_versions_json,
                calc_version=event.calc_version,
                rule_id=event.rule_id,
                result_json=event.result_json,
                error_json=event.error_json,
                prev_hash=prev_hash,
                event_hash=event_hash,
            )
        )
        seq = result.inserted_primary_key[0]

        stored = event.model_copy(
            update={"seq": seq, "prev_hash": prev_hash, "event_hash": event_hash}
        )
        logger.info(
            "event appended",
            extra={"seq": seq, "event_type": str(event.event_type)},
        )
        return stored

    # -- read -------------------------------------------------------------

    def get_event(
        self, event_id: uuid.UUID, *, conn: Connection | None = None
    ) -> AuditEvent | None:
        """Fetch one event by its business identifier."""
        stmt = select(audit_event).where(audit_event.c.id == event_id)
        row = self._execute(stmt, conn).first()
        return None if row is None else _row_to_event(row)

    def get_event_by_seq(
        self, seq: int, *, conn: Connection | None = None
    ) -> AuditEvent | None:
        stmt = select(audit_event).where(audit_event.c.seq == seq)
        row = self._execute(stmt, conn).first()
        return None if row is None else _row_to_event(row)

    def get_events(
        self,
        *,
        sku_id: uuid.UUID | None = None,
        product_id: uuid.UUID | None = None,
        supplier_id: uuid.UUID | None = None,
        event_type: EventType | None = None,
        since_seq: int | None = None,
        until_seq: int | None = None,
        until_timestamp: datetime | None = None,
        limit: int | None = None,
        conn: Connection | None = None,
    ) -> list[AuditEvent]:
        """Return events in ledger order, optionally filtered.

        Ordering is always by ``seq``. Timestamp ordering would be
        non-deterministic for events written in the same microsecond.
        """
        stmt = select(audit_event).order_by(audit_event.c.seq.asc())
        if sku_id is not None:
            stmt = stmt.where(audit_event.c.sku_id == sku_id)
        if product_id is not None:
            stmt = stmt.where(audit_event.c.product_id == product_id)
        if supplier_id is not None:
            stmt = stmt.where(audit_event.c.supplier_id == supplier_id)
        if event_type is not None:
            stmt = stmt.where(audit_event.c.event_type == str(event_type))
        if since_seq is not None:
            stmt = stmt.where(audit_event.c.seq > since_seq)
        if until_seq is not None:
            stmt = stmt.where(audit_event.c.seq <= until_seq)
        if until_timestamp is not None:
            stmt = stmt.where(audit_event.c.timestamp <= until_timestamp)
        if limit is not None:
            stmt = stmt.limit(limit)
        return [_row_to_event(row) for row in self._execute(stmt, conn).fetchall()]

    def count(self, *, conn: Connection | None = None) -> int:
        stmt = select(func.count()).select_from(audit_event)
        return int(self._execute(stmt, conn).scalar() or 0)

    def head_seq(self, *, conn: Connection | None = None) -> int:
        """Highest sequence number in the ledger, or 0 when empty."""
        stmt = select(func.max(audit_event.c.seq))
        return int(self._execute(stmt, conn).scalar() or 0)

    # -- integrity --------------------------------------------------------

    def verify_chain(self, *, conn: Connection | None = None) -> dict[str, Any]:
        """Recompute every hash and confirm the chain is unbroken.

        Detects out-of-band tampering that bypassed the triggers. Returns a
        summary; raises nothing so it can be used as a health check.
        """
        events = self.get_events(conn=conn)
        prev_hash = GENESIS_HASH
        for event in events:
            if event.prev_hash != prev_hash:
                return {
                    "valid": False,
                    "checked": len(events),
                    "broken_at_seq": event.seq,
                    "reason": "prev_hash does not match preceding event_hash",
                }
            expected = _compute_hash(event, prev_hash)
            if expected != event.event_hash:
                return {
                    "valid": False,
                    "checked": len(events),
                    "broken_at_seq": event.seq,
                    "reason": "event content does not match its recorded hash",
                }
            prev_hash = event.event_hash
        return {"valid": True, "checked": len(events), "head_hash": prev_hash}

    def require_valid_chain(self, *, conn: Connection | None = None) -> None:
        """Raise ``LedgerIntegrityError`` if the chain does not verify."""
        report = self.verify_chain(conn=conn)
        if not report["valid"]:
            raise LedgerIntegrityError(
                f"ledger integrity failure at seq {report['broken_at_seq']}: "
                f"{report['reason']}"
            )

    # -- internal ---------------------------------------------------------

    def _execute(self, stmt: Any, conn: Connection | None) -> _Rows:
        """Run a read and materialise it.

        Always returns the same type regardless of whether the caller supplied
        a connection. The earlier version returned a raw SQLAlchemy ``Result``
        when given one and a wrapper otherwise, so behaviour differed between
        the two call paths -- the kind of inconsistency that holds until
        someone calls a ``Result``-only method on the wrapper.
        """
        if conn is not None:
            return _Rows(conn.execute(stmt).fetchall())
        with read_only(self._engine) as owned_conn:
            return _Rows(owned_conn.execute(stmt).fetchall())


class _Rows:
    """Rows detached from the connection that produced them."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def fetchall(self) -> list[Any]:
        return self._rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalar(self) -> Any:
        return self._rows[0][0] if self._rows else None


def _row_to_event(row: Any) -> AuditEvent:
    """Rehydrate an ``AuditEvent`` from a database row."""
    mapping = row._mapping
    return AuditEvent(
        id=mapping["id"],
        seq=mapping["seq"],
        timestamp=mapping["timestamp"],
        sku_id=mapping["sku_id"],
        product_id=mapping["product_id"],
        supplier_id=mapping["supplier_id"],
        event_type=EventType(mapping["event_type"]),
        actor=mapping["actor"],
        inputs_json=mapping["inputs_json"] or {},
        reference_versions_json=mapping["reference_versions_json"] or {},
        calc_version=mapping["calc_version"],
        rule_id=mapping["rule_id"],
        result_json=mapping["result_json"] or {},
        error_json=mapping["error_json"],
        prev_hash=mapping["prev_hash"],
        event_hash=mapping["event_hash"],
    )
