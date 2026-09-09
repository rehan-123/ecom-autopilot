"""Replay: reconstruct state from the event stream.

Two capabilities, both of which exist because a ledger you cannot replay is a
log, not a source of truth:

``replay_events``
    Fold the whole stream into a state snapshot. Running it twice over the
    same ledger must produce identical output; the test suite asserts this.

``reconstruct_at``
    Fold the stream up to a point in time or a sequence number. Answers
    "what did we believe on 15 August, and on what evidence?" -- which is the
    question that matters when a past decision has to be defended or debugged.

Later milestones add ``replay_snapshot``, which re-derives an
``EconomicsSnapshot`` from its pinned reference versions and asserts the
result is unchanged. That is regression testing for money arithmetic, and it
is the reason every snapshot pins its inputs.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Engine

from catalog.enums import EventType, SKUState
from catalog.schemas import AuditEvent
from ledger.events import EventRepository

logger = logging.getLogger(__name__)


@dataclass
class SKUHistory:
    """Reconstructed lifecycle for one SKU."""

    sku_id: uuid.UUID
    sku_code: str | None = None
    state: SKUState = SKUState.DISCOVERED
    state_updated_at: datetime | None = None
    transitions: list[tuple[str | None, str, datetime]] = field(default_factory=list)


@dataclass
class OfferHistory:
    """Reconstructed price history for one (supplier, product) pair."""

    supplier_id: uuid.UUID
    product_id: uuid.UUID
    observations: list[tuple[uuid.UUID, Decimal, datetime]] = field(default_factory=list)

    @property
    def in_observation_order(self) -> list[tuple[uuid.UUID, Decimal, datetime]]:
        """Observations sorted by when they were observed, not recorded."""
        return sorted(self.observations, key=lambda o: o[2])

    @property
    def current_cost(self) -> Decimal | None:
        """Cost from the latest *observation*.

        Ordered by ``observed_at`` rather than ledger position: a quote from
        the 23rd backfilled on the 27th must not displace the 27th's quote.
        The projection already ordered this way, so replay disagreeing with
        it meant two code paths gave two different answers for the same
        question.
        """
        ordered = self.in_observation_order
        return ordered[-1][1] if ordered else None

    @property
    def superseded_count(self) -> int:
        return max(0, len(self.observations) - 1)


@dataclass
class ReplayState:
    """The result of folding the event stream."""

    events_applied: int = 0
    head_seq: int = 0
    suppliers: dict[uuid.UUID, str] = field(default_factory=dict)
    products: dict[uuid.UUID, str] = field(default_factory=dict)
    skus: dict[uuid.UUID, SKUHistory] = field(default_factory=dict)
    offers: dict[tuple[uuid.UUID, uuid.UUID], OfferHistory] = field(default_factory=dict)
    unhandled_event_types: dict[str, int] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable summary of the reconstructed state.

        Used to assert that two replays of the same ledger agree. Comparing
        fingerprints rather than object graphs keeps the determinism test
        readable and independent of dataclass identity.
        """
        parts: list[str] = [
            f"events={self.events_applied}",
            f"head={self.head_seq}",
            f"suppliers={sorted(str(k) for k in self.suppliers)}",
            f"products={sorted(str(k) for k in self.products)}",
        ]
        for sku_id in sorted(self.skus, key=str):
            history = self.skus[sku_id]
            parts.append(f"sku:{sku_id}:{history.state}:{len(history.transitions)}")
        for key in sorted(self.offers, key=lambda k: (str(k[0]), str(k[1]))):
            history = self.offers[key]
            parts.append(
                f"offer:{key[0]}:{key[1]}:{history.current_cost}:"
                f"{history.superseded_count}"
            )
        return "|".join(parts)


def _fold_supplier_created(state: ReplayState, event: AuditEvent) -> None:
    if event.supplier_id is not None:
        state.suppliers[event.supplier_id] = (event.result_json or {}).get("name", "")


def _fold_product_created(state: ReplayState, event: AuditEvent) -> None:
    if event.product_id is not None:
        state.products[event.product_id] = (event.result_json or {}).get(
            "canonical_name", ""
        )


def _fold_sku_created(state: ReplayState, event: AuditEvent) -> None:
    if event.sku_id is None:
        return
    payload = event.result_json or {}
    state.skus[event.sku_id] = SKUHistory(
        sku_id=event.sku_id,
        sku_code=payload.get("sku_code"),
        state=SKUState(payload.get("state", SKUState.DISCOVERED)),
        state_updated_at=event.timestamp,
    )


def _fold_sku_state_changed(state: ReplayState, event: AuditEvent) -> None:
    if event.sku_id is None:
        return
    payload = event.result_json or {}
    history = state.skus.setdefault(event.sku_id, SKUHistory(sku_id=event.sku_id))
    to_state = SKUState(payload["to_state"])
    history.transitions.append(
        (payload.get("from_state"), str(to_state), event.timestamp)
    )
    history.state = to_state
    history.state_updated_at = event.timestamp


def _fold_supplier_offer_created(state: ReplayState, event: AuditEvent) -> None:
    if event.supplier_id is None or event.product_id is None:
        return
    payload = event.result_json or {}
    key = (event.supplier_id, event.product_id)
    history = state.offers.setdefault(
        key, OfferHistory(supplier_id=key[0], product_id=key[1])
    )
    history.observations.append(
        (
            uuid.UUID(str(payload["offer_id"])),
            Decimal(str(payload["unit_cost_inr"])),
            _observed_at(payload, event.timestamp),
        )
    )


def _observed_at(payload: dict[str, Any], fallback: datetime) -> datetime:
    """Observation date from the payload, falling back to the record time."""
    raw = payload.get("observed_at")
    if raw is None:
        return fallback
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


#: Event type -> fold function. A dispatch table rather than a branch ladder:
#: adding an event type means adding one entry, and the fold for each type
#: stays independently readable and testable.
_FOLDERS: dict[EventType, Callable[[ReplayState, AuditEvent], None]] = {
    EventType.SUPPLIER_CREATED: _fold_supplier_created,
    EventType.PRODUCT_CREATED: _fold_product_created,
    EventType.SKU_CREATED: _fold_sku_created,
    EventType.SKU_STATE_CHANGED: _fold_sku_state_changed,
    EventType.SUPPLIER_OFFER_CREATED: _fold_supplier_offer_created,
}


def _apply(state: ReplayState, event: AuditEvent) -> None:
    """Apply one event to the accumulating state.

    Event types with no fold are counted rather than ignored, so replay can
    report what it did not account for instead of silently dropping it.
    """
    folder = _FOLDERS.get(event.event_type)
    if folder is None:
        key = str(event.event_type)
        state.unhandled_event_types[key] = state.unhandled_event_types.get(key, 0) + 1
    else:
        folder(state, event)

    state.events_applied += 1
    if event.seq is not None:
        state.head_seq = max(state.head_seq, event.seq)


def replay_events(
    engine: Engine,
    *,
    until_seq: int | None = None,
    until_timestamp: datetime | None = None,
    verify_integrity: bool = True,
) -> ReplayState:
    """Fold the ledger into a reconstructed state snapshot.

    ``verify_integrity`` checks the hash chain first. Replaying a tampered
    ledger produces a confident wrong answer, which is worse than an error,
    so verification is on by default.
    """
    repo = EventRepository(engine)
    if verify_integrity:
        repo.require_valid_chain()

    events = repo.get_events(until_seq=until_seq, until_timestamp=until_timestamp)
    state = ReplayState()
    for event in events:
        _apply(state, event)

    logger.info(
        "replay complete",
        extra={"events": state.events_applied, "head_seq": state.head_seq},
    )
    return state


def reconstruct_at(
    engine: Engine,
    *,
    at: datetime | None = None,
    seq: int | None = None,
) -> ReplayState:
    """Reconstruct state as it stood at a timestamp or sequence number."""
    if at is None and seq is None:
        raise ValueError("supply either `at` or `seq`")
    return replay_events(engine, until_seq=seq, until_timestamp=at)


def sku_state_at(engine: Engine, sku_id: uuid.UUID, *, at: datetime) -> SKUState | None:
    """The lifecycle state of one SKU at a past moment."""
    history = reconstruct_at(engine, at=at).skus.get(sku_id)
    return None if history is None else history.state


def offer_history(
    engine: Engine, supplier_id: uuid.UUID, product_id: uuid.UUID
) -> OfferHistory | None:
    """Full price history for a (supplier, product) pair.

    Every observation is preserved. A price rise from Rs 280 to Rs 310 leaves
    both records intact; the earlier one is superseded, never overwritten.
    """
    return replay_events(engine).offers.get((supplier_id, product_id))
