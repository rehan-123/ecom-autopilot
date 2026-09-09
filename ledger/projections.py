"""Projections: derived, droppable, rebuilt from the ledger.

A projection is never a source of truth. Every projection table here can be
dropped entirely and reconstructed from ``audit_event`` alone. If that is not
true of a table, it does not belong in this module.

That constraint is what makes the ledger meaningful. A ledger that cannot
reproduce current state is decoration.

Two projections exist at this milestone:

``sku_current_state``
    A SKU's current lifecycle state. This is why ``sku`` has no ``state``
    column: state changes would otherwise be UPDATEs against an immutable
    entity, and history would live in two places.

``supplier_offer_current``
    The latest offer per (supplier, product), with a count of superseded
    predecessors. Replaces the ``superseded_by_offer_id`` back-reference,
    which could only be maintained by mutating an already-written offer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, Engine, Table

from catalog.enums import EventType, SKUState
from database.tables import (
    PROJECTION_TABLES,
    metadata,
    sku_current_state,
    supplier_offer_current,
)
from ledger.events import EventRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Projection:
    """A named derived view and the function that rebuilds it."""

    name: str
    table: Table
    builder: Callable[[Connection, EventRepository, int], int]
    description: str


def _build_sku_current_state(
    conn: Connection, repo: EventRepository, head_seq: int
) -> int:
    """Fold SKU_CREATED and SKU_STATE_CHANGED events into current state."""
    created = repo.get_events(
        event_type=EventType.SKU_CREATED, until_seq=head_seq, conn=conn
    )
    changed = repo.get_events(
        event_type=EventType.SKU_STATE_CHANGED, until_seq=head_seq, conn=conn
    )

    state: dict[Any, dict[str, Any]] = {}

    for event in created:
        if event.sku_id is None:
            continue
        initial = event.result_json.get("state", SKUState.DISCOVERED)
        state[event.sku_id] = {
            "state": str(initial),
            "state_updated_at": event.timestamp,
            "transition_count": 0,
        }

    # Events arrive in seq order, so the last write per SKU wins deterministically.
    for event in changed:
        if event.sku_id is None:
            continue
        entry = state.setdefault(
            event.sku_id,
            {
                "state": str(SKUState.DISCOVERED),
                "state_updated_at": event.timestamp,
                "transition_count": 0,
            },
        )
        entry["state"] = str(event.result_json["to_state"])
        entry["state_updated_at"] = event.timestamp
        entry["transition_count"] += 1

    rows = [
        {
            "sku_id": sku_id,
            "state": entry["state"],
            "state_updated_at": entry["state_updated_at"],
            "transition_count": entry["transition_count"],
            "rebuilt_from_seq": head_seq,
        }
        for sku_id, entry in state.items()
    ]
    if rows:
        conn.execute(sku_current_state.insert(), rows)
    return len(rows)


def _observed_at(event: Any) -> datetime:
    """Observation date from the payload, falling back to the event time.

    The payload is authoritative: a quote observed on 23 August that is
    backfilled on 27 August must sort by the former.
    """
    raw = (event.result_json or {}).get("observed_at")
    if raw is None:
        return event.timestamp
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _build_supplier_offer_current(
    conn: Connection, repo: EventRepository, head_seq: int
) -> int:
    """Reduce the offer stream to the latest offer per (supplier, product)."""
    offers = repo.get_events(
        event_type=EventType.SUPPLIER_OFFER_CREATED, until_seq=head_seq, conn=conn
    )

    latest: dict[tuple[Any, Any], dict[str, Any]] = {}
    for event in offers:
        payload = event.result_json
        key = (event.supplier_id, event.product_id)
        entry = latest.get(key)
        observed_at = _observed_at(event)

        if entry is None:
            latest[key] = {
                "offer_id": payload["offer_id"],
                "unit_cost_inr": str(payload["unit_cost_inr"]),
                "observed_at": observed_at,
                "superseded_count": 0,
            }
            continue

        # Later observation supersedes the earlier one. The earlier offer row
        # itself is untouched -- it remains queryable in supplier_offer.
        if observed_at >= entry["observed_at"]:
            latest[key] = {
                "offer_id": payload["offer_id"],
                "unit_cost_inr": str(payload["unit_cost_inr"]),
                "observed_at": observed_at,
                "superseded_count": entry["superseded_count"] + 1,
            }
        else:
            entry["superseded_count"] += 1

    rows = [
        {
            "supplier_id": supplier_id,
            "product_id": product_id,
            "offer_id": entry["offer_id"],
            "unit_cost_inr": entry["unit_cost_inr"],
            "observed_at": entry["observed_at"],
            "superseded_count": entry["superseded_count"],
            "rebuilt_from_seq": head_seq,
        }
        for (supplier_id, product_id), entry in latest.items()
    ]
    if rows:
        conn.execute(supplier_offer_current.insert(), rows)
    return len(rows)


PROJECTIONS: tuple[Projection, ...] = (
    Projection(
        name="sku_current_state",
        table=sku_current_state,
        builder=_build_sku_current_state,
        description="Current lifecycle state per SKU, folded from the transition stream",
    ),
    Projection(
        name="supplier_offer_current",
        table=supplier_offer_current,
        builder=_build_supplier_offer_current,
        description="Latest offer per (supplier, product), with supersession count",
    ),
)


def drop_projections(engine: Engine) -> list[str]:
    """Drop every projection table. Safe: nothing here is a source of truth."""
    dropped: list[str] = []
    with engine.begin() as conn:
        for projection in PROJECTIONS:
            projection.table.drop(bind=conn, checkfirst=True)
            dropped.append(projection.name)
    logger.info("projections dropped", extra={"tables": dropped})
    return dropped


def rebuild_projections(engine: Engine) -> dict[str, int]:
    """Drop and rebuild every projection from the ledger.

    Deterministic: the same ledger always yields the same projections. The
    rebuild runs in one transaction, so a failure part-way leaves the previous
    projections in place rather than an empty database.
    """
    repo = EventRepository(engine)
    counts: dict[str, int] = {}

    with engine.begin() as conn:
        head_seq = repo.head_seq(conn=conn)
        for projection in PROJECTIONS:
            projection.table.drop(bind=conn, checkfirst=True)
            projection.table.create(bind=conn, checkfirst=True)
            counts[projection.name] = projection.builder(conn, repo, head_seq)

    logger.info("projections rebuilt", extra={"head_seq": head_seq, "counts": counts})
    return counts


def ensure_projections(engine: Engine) -> None:
    """Create projection tables if absent, without rebuilding contents."""
    with engine.begin() as conn:
        for projection in PROJECTIONS:
            projection.table.create(bind=conn, checkfirst=True)


def projection_table_names() -> tuple[str, ...]:
    """Names of every registered projection table."""
    return PROJECTION_TABLES


def assert_projections_are_derived() -> None:
    """Guard: every registered projection must exist in the metadata.

    Cheap protection against someone adding a source-of-truth table to the
    projection registry, which ``rebuild_projections`` would then delete.
    """
    for projection in PROJECTIONS:
        if projection.table.name not in metadata.tables:
            raise RuntimeError(f"projection {projection.name!r} is not a declared table")
        if projection.table.name not in PROJECTION_TABLES:
            raise RuntimeError(
                f"{projection.name!r} is registered as a projection but is not "
                "listed in PROJECTION_TABLES; rebuilding would destroy data"
            )
