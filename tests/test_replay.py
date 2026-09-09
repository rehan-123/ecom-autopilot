"""Replay and projection tests.

The claim under test is that the ledger is the source of truth: drop every
projection, rebuild from events alone, and arrive at the same state. A ledger
that cannot do this is a log, not a source of truth.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import inspect, select, text

from catalog.enums import SKUState, TransitionTrigger
from catalog.schemas import SKU
from database.tables import sku_current_state, supplier_offer_current
from ledger.events import EventRepository, LedgerIntegrityError
from ledger.projections import (
    PROJECTIONS,
    assert_projections_are_derived,
    drop_projections,
    rebuild_projections,
)
from ledger.replay import (
    offer_history,
    reconstruct_at,
    replay_events,
    sku_state_at,
)
from tests.conftest import make_offer, utc


@pytest.fixture
def seeded(catalog, supplier, product, sku):
    """A small history: two offers and three lifecycle transitions."""
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23)))
    catalog.record_offer(make_offer(supplier.id, product.id, "310.00", utc(2026, 8, 27)))
    for state in (
        SKUState.DATA_COLLECTION,
        SKUState.ECONOMIC_ANALYSIS,
        SKUState.BLOCKED_AWAITING_DATA,
    ):
        catalog.transition_sku(sku.id, state, trigger=TransitionTrigger.RULE)
    return {"supplier": supplier, "product": product, "sku": sku}


# -- replay ----------------------------------------------------------------


def test_replay_reconstructs_entities(engine, seeded):
    state = replay_events(engine)
    assert len(state.suppliers) == 1
    assert len(state.products) == 1
    assert len(state.skus) == 1
    assert state.events_applied == state.head_seq


def test_replay_reconstructs_final_sku_state(engine, seeded):
    history = replay_events(engine).skus[seeded["sku"].id]
    assert history.state is SKUState.BLOCKED_AWAITING_DATA
    assert len(history.transitions) == 3


def test_replay_preserves_full_offer_history(engine, seeded):
    history = offer_history(engine, seeded["supplier"].id, seeded["product"].id)
    assert [cost for _, cost, _ in history.observations] == [
        Decimal("280.00"),
        Decimal("310.00"),
    ]
    assert history.current_cost == Decimal("310.00")
    assert history.superseded_count == 1


def test_replay_is_deterministic(engine, seeded):
    """Same ledger, same result. Every time."""
    fingerprints = {replay_events(engine).fingerprint() for _ in range(5)}
    assert len(fingerprints) == 1


def test_replay_of_empty_ledger_is_safe(engine):
    state = replay_events(engine)
    assert state.events_applied == 0
    assert state.skus == {}


def test_replay_verifies_integrity_by_default(engine, events, seeded):
    with engine.begin() as conn:
        conn.execute(text("DROP TRIGGER trg_audit_event_no_update"))
        conn.execute(text("UPDATE audit_event SET actor = 'HUMAN:mallory' WHERE seq = 1"))
    with pytest.raises(LedgerIntegrityError):
        replay_events(engine)


def test_replay_can_skip_verification_when_asked(engine, seeded):
    state = replay_events(engine, verify_integrity=False)
    assert state.events_applied > 0


# -- point-in-time reconstruction -----------------------------------------


def test_reconstruct_at_earlier_time_shows_earlier_price(engine, seeded):
    """The question this answers: what did we believe on 23 August?"""
    early = reconstruct_at(engine, at=utc(2026, 8, 23) + timedelta(hours=1))
    key = (seeded["supplier"].id, seeded["product"].id)
    assert early.offers[key].current_cost == Decimal("280.00")

    now = replay_events(engine)
    assert now.offers[key].current_cost == Decimal("310.00")


def test_reconstruct_at_sequence_number(engine, seeded):
    repo = EventRepository(engine)
    head = repo.head_seq()
    partial = reconstruct_at(engine, seq=head - 1)
    assert partial.events_applied == head - 1


def test_sku_state_at_past_moment(engine, catalog, seeded):
    sku_id = seeded["sku"].id
    transitions = catalog.transitions_for(sku_id)
    second = transitions[1].created_at

    assert sku_state_at(engine, sku_id, at=second) is SKUState.DATA_COLLECTION
    assert catalog.current_state(sku_id) is SKUState.BLOCKED_AWAITING_DATA


def test_reconstruct_requires_a_bound(engine):
    with pytest.raises(ValueError, match="either `at` or `seq`"):
        reconstruct_at(engine)


# -- projections -----------------------------------------------------------


def test_rebuild_projections_populates_state(engine, seeded):
    counts = rebuild_projections(engine)
    assert counts["sku_current_state"] == 1
    assert counts["supplier_offer_current"] == 1


def test_projections_survive_a_full_drop_and_rebuild(engine, catalog, seeded):
    """The whole point: projections are derived, never authoritative."""
    before = catalog.current_state(seeded["sku"].id)

    drop_projections(engine)
    with engine.connect() as conn:
        assert "sku_current_state" not in inspect(conn).get_table_names()

    rebuild_projections(engine)
    assert catalog.current_state(seeded["sku"].id) == before


def test_rebuild_is_idempotent(engine, seeded):
    first = rebuild_projections(engine)
    second = rebuild_projections(engine)
    third = rebuild_projections(engine)
    assert first == second == third


def test_rebuild_produces_identical_rows_every_time(engine, seeded):
    def snapshot() -> list[tuple]:
        rebuild_projections(engine)
        with engine.connect() as conn:
            states = conn.execute(
                select(
                    sku_current_state.c.sku_id,
                    sku_current_state.c.state,
                    sku_current_state.c.transition_count,
                ).order_by(sku_current_state.c.sku_id)
            ).fetchall()
            offers = conn.execute(
                select(
                    supplier_offer_current.c.offer_id,
                    supplier_offer_current.c.unit_cost_inr,
                    supplier_offer_current.c.superseded_count,
                ).order_by(supplier_offer_current.c.offer_id)
            ).fetchall()
        return [tuple(r) for r in states] + [tuple(r) for r in offers]

    assert snapshot() == snapshot() == snapshot()


def test_projection_records_the_sequence_it_was_built_from(engine, seeded):
    rebuild_projections(engine)
    head = EventRepository(engine).head_seq()
    with engine.connect() as conn:
        built_from = conn.execute(select(sku_current_state.c.rebuilt_from_seq)).scalar()
    assert built_from == head


def test_projection_tracks_transition_count(engine, seeded):
    rebuild_projections(engine)
    with engine.connect() as conn:
        count = conn.execute(select(sku_current_state.c.transition_count)).scalar()
    assert count == 3


def test_current_offer_projection_points_at_latest(engine, catalog, seeded):
    rebuild_projections(engine)
    latest = catalog.current_offer(seeded["supplier"].id, seeded["product"].id)
    with engine.connect() as conn:
        row = conn.execute(select(supplier_offer_current)).first()
    assert row._mapping["offer_id"] == latest.id
    assert Decimal(row._mapping["unit_cost_inr"]) == Decimal("310.00")
    assert row._mapping["superseded_count"] == 1


def test_current_state_falls_back_to_transitions_without_projection(
    engine, catalog, seeded
):
    """A missing projection degrades performance, not correctness."""
    drop_projections(engine)
    assert catalog.current_state(seeded["sku"].id) is SKUState.BLOCKED_AWAITING_DATA


def test_rebuild_reflects_events_added_after_last_rebuild(engine, catalog, seeded):
    rebuild_projections(engine)
    catalog.transition_sku(
        seeded["sku"].id,
        SKUState.REJECTED,
        trigger=TransitionTrigger.HUMAN,
        actor="HUMAN:rehan",
        rationale="market price headroom negative",
    )
    rebuild_projections(engine)
    with engine.connect() as conn:
        state = conn.execute(select(sku_current_state.c.state)).scalar()
    assert state == "REJECTED"


def test_multiple_skus_project_independently(engine, catalog, product, seeded):
    other = catalog.create_sku(SKU(product_id=product.id, sku_code="SKU-SECOND"))
    catalog.transition_sku(other.id, SKUState.REJECTED, trigger=TransitionTrigger.RULE)

    counts = rebuild_projections(engine)
    assert counts["sku_current_state"] == 2
    assert catalog.current_state(other.id) is SKUState.REJECTED
    assert catalog.current_state(seeded["sku"].id) is SKUState.BLOCKED_AWAITING_DATA


def test_every_projection_is_registered_as_droppable():
    assert_projections_are_derived()
    assert len(PROJECTIONS) == 2
