"""Append-only enforcement.

These tests exist because Python-layer immutability protects only code that
goes through the repository. A stray UPDATE from a SQL console, a future
module, or an agent refactoring the codebase would otherwise rewrite history
silently. The database has to refuse.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from catalog.enums import EventType, SKUState, TransitionTrigger
from catalog.repository import InvalidTransitionError
from catalog.schemas import AuditEvent
from database.tables import IMMUTABLE_TABLES, rate_card
from ledger.projections import rebuild_projections
from tests.conftest import make_offer, utc


def _append(events) -> AuditEvent:
    return events.append_event(
        AuditEvent(
            event_type=EventType.SUPPLIER_CREATED,
            actor="SYSTEM",
            result_json={"name": "original"},
        )
    )


# -- the core guarantee ----------------------------------------------------


def test_insert_event_succeeds(events):
    assert _append(events).seq == 1


def test_update_audit_event_fails(engine, events):
    _append(events)
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE audit_event SET actor = 'HUMAN:x' WHERE seq = 1"))


def test_delete_audit_event_fails(engine, events):
    _append(events)
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM audit_event WHERE seq = 1"))


def test_bulk_delete_fails(engine, events):
    for _ in range(5):
        _append(events)
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM audit_event"))
    assert events.count() == 5


def test_event_content_survives_a_rejected_update(engine, events):
    stored = _append(events)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE audit_event SET result_json = :forged").bindparams(
                    forged='{"name":"forged"}'
                )
            )
    assert events.get_event(stored.id).result_json == {"name": "original"}


# -- every immutable table is protected -----------------------------------


@pytest.mark.parametrize("table", IMMUTABLE_TABLES)
def test_every_immutable_table_is_populated_by_the_fixture(engine, populated, table):
    """Guard: triggers fire per-row, so an empty table would pass vacuously."""
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    assert count > 0, f"{table} has no rows; the immutability test below is vacuous"


@pytest.mark.parametrize("table", IMMUTABLE_TABLES)
def test_every_immutable_table_rejects_update(engine, populated, table: str):
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text(f"UPDATE {table} SET id = id"))


@pytest.mark.parametrize("table", IMMUTABLE_TABLES)
def test_every_immutable_table_rejects_delete(engine, populated, table: str):
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {table}"))


# -- supplier offer history ------------------------------------------------


def test_supplier_price_change_creates_a_new_offer(catalog, supplier, product):
    """Rs 280 -> Rs 310 is a new observation, never an edit."""
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23)))
    catalog.record_offer(make_offer(supplier.id, product.id, "310.00", utc(2026, 8, 27)))

    offers = catalog.offers_for(supplier.id, product.id)
    assert len(offers) == 2
    assert [o.unit_cost_inr for o in offers] == [Decimal("280.00"), Decimal("310.00")]
    assert offers[0].id != offers[1].id


def test_historical_offer_remains_intact(catalog, supplier, product):
    original = catalog.record_offer(
        make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23))
    )
    catalog.record_offer(make_offer(supplier.id, product.id, "310.00", utc(2026, 8, 27)))

    offers = catalog.offers_for(supplier.id, product.id)
    preserved = next(o for o in offers if o.id == original.id)
    assert preserved.unit_cost_inr == Decimal("280.00")
    assert preserved.observed_at == utc(2026, 8, 23)


def test_current_offer_is_the_latest_observation(catalog, supplier, product):
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23)))
    catalog.record_offer(make_offer(supplier.id, product.id, "310.00", utc(2026, 8, 27)))
    assert catalog.current_offer(supplier.id, product.id).unit_cost_inr == Decimal(
        "310.00"
    )


def test_offer_price_cannot_be_edited(engine, catalog, supplier, product):
    offer = catalog.record_offer(
        make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23))
    )
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE supplier_offer SET unit_cost_inr = '1.00' WHERE id = :i"),
                {"i": str(offer.id)},
            )
    assert catalog.current_offer(supplier.id, product.id).unit_cost_inr == Decimal(
        "280.00"
    )


# -- state transitions -----------------------------------------------------


def test_state_transition_is_recorded_not_overwritten(catalog, sku):
    catalog.transition_sku(
        sku.id, SKUState.DATA_COLLECTION, trigger=TransitionTrigger.INGEST
    )
    catalog.transition_sku(
        sku.id, SKUState.ECONOMIC_ANALYSIS, trigger=TransitionTrigger.RULE
    )

    transitions = catalog.transitions_for(sku.id)
    assert [t.to_state for t in transitions] == [
        SKUState.DISCOVERED,
        SKUState.DATA_COLLECTION,
        SKUState.ECONOMIC_ANALYSIS,
    ]
    assert transitions[0].from_state is None


def test_transition_history_cannot_be_rewritten(engine, catalog, sku):
    catalog.transition_sku(sku.id, SKUState.REJECTED, trigger=TransitionTrigger.RULE)
    with pytest.raises(IntegrityError, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE sku_state_transition SET to_state = 'APPROVED'"))


def test_transition_to_same_state_is_rejected(catalog, sku):
    with pytest.raises(InvalidTransitionError):
        catalog.transition_sku(
            sku.id, SKUState.DISCOVERED, trigger=TransitionTrigger.RULE
        )


def test_every_transition_emits_a_ledger_event(catalog, events, sku):
    before = len(events.get_events(event_type=EventType.SKU_STATE_CHANGED))
    catalog.transition_sku(
        sku.id, SKUState.DATA_COLLECTION, trigger=TransitionTrigger.RULE
    )
    after = events.get_events(event_type=EventType.SKU_STATE_CHANGED)
    assert len(after) == before + 1
    assert after[-1].result_json["to_state"] == "DATA_COLLECTION"


# -- projections remain mutable -------------------------------------------


def test_projection_tables_are_not_trigger_protected(engine, populated):
    """Projections must stay droppable; protecting them would break rebuild."""
    rebuild_projections(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM sku_current_state")).scalar() > 0

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM sku_current_state"))  # must not raise

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM sku_current_state")).scalar() == 0


def test_reference_tables_accept_new_versions(engine):
    """Reference data is versioned by insert, so inserts must be allowed."""

    with engine.begin() as conn:
        conn.execute(
            rate_card.insert().values(
                id=uuid.uuid4(),
                marketplace="test_marketplace",
                version="2026.08",
                effective_from=datetime.now(UTC),
                rules_json={},
                created_at=datetime.now(UTC),
            )
        )
