"""Ledger tests: append, retrieve, order, filter, and hash-chain integrity."""

from __future__ import annotations

import itertools
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from catalog.enums import EventType, GSTStatus, SupplierType
from catalog.schemas import SKU, AuditEvent, Supplier, utcnow
from ledger.events import (
    GENESIS_HASH,
    EventRepository,
    LedgerIntegrityError,
    _compute_hash,
)


def _event(event_type: EventType = EventType.ERROR, **kwargs) -> AuditEvent:
    return AuditEvent(event_type=event_type, actor="SYSTEM", **kwargs)


# -- append and retrieve ---------------------------------------------------


def test_append_event_succeeds(events: EventRepository):
    stored = events.append_event(_event())
    assert stored.seq == 1
    assert stored.event_hash is not None
    assert stored.prev_hash == GENESIS_HASH


def test_retrieve_event_matches_exactly(events: EventRepository):
    original = _event(
        EventType.ECONOMICS_CALCULATED,
        inputs_json={"supplier_cost": "280", "selling_price": "799"},
        reference_versions_json={"rate_card_version": "2026.08"},
        calc_version="1.0.0",
        rule_id="minimum_net_margin",
        result_json={"net_margin_pct": "24.7", "gate": "PASS"},
    )
    stored = events.append_event(original)
    fetched = events.get_event(stored.id)

    assert fetched is not None
    assert fetched.id == original.id
    assert fetched.event_type is EventType.ECONOMICS_CALCULATED
    assert fetched.inputs_json == original.inputs_json
    assert fetched.reference_versions_json == original.reference_versions_json
    assert fetched.calc_version == "1.0.0"
    assert fetched.rule_id == "minimum_net_margin"
    assert fetched.result_json == original.result_json
    assert fetched.timestamp == original.timestamp


def test_get_missing_event_returns_none(events: EventRepository):
    assert events.get_event(uuid.uuid4()) is None


def test_history_is_ordered_by_sequence(events: EventRepository):
    for _ in range(10):
        events.append_event(_event())
    seqs = [e.seq for e in events.get_events()]
    assert seqs == sorted(seqs) == list(range(1, 11))


def test_sequence_is_monotonic_even_for_identical_timestamps(events: EventRepository):
    """Timestamps alone cannot order events written in the same microsecond."""
    moment = utcnow()
    for _ in range(5):
        events.append_event(_event(timestamp=moment))

    stored = events.get_events()
    assert len({e.timestamp for e in stored}) == 1
    assert [e.seq for e in stored] == [1, 2, 3, 4, 5]


def test_events_filter_by_sku(events: EventRepository):
    sku_a, sku_b = uuid.uuid4(), uuid.uuid4()
    events.append_event(_event(EventType.SKU_CREATED, sku_id=sku_a))
    events.append_event(_event(EventType.SKU_CREATED, sku_id=sku_b))
    events.append_event(_event(EventType.SKU_STATE_CHANGED, sku_id=sku_a))

    assert len(events.get_events(sku_id=sku_a)) == 2
    assert len(events.get_events(sku_id=sku_b)) == 1


def test_events_filter_by_type(events: EventRepository):
    events.append_event(_event(EventType.SUPPLIER_CREATED))
    events.append_event(_event(EventType.PRODUCT_CREATED))
    events.append_event(_event(EventType.SUPPLIER_CREATED))

    found = events.get_events(event_type=EventType.SUPPLIER_CREATED)
    assert len(found) == 2
    assert all(e.event_type is EventType.SUPPLIER_CREATED for e in found)


def test_events_filter_by_sequence_window(events: EventRepository):
    for _ in range(6):
        events.append_event(_event())
    window = events.get_events(since_seq=2, until_seq=4)
    assert [e.seq for e in window] == [3, 4]


def test_count_and_head_seq(events: EventRepository):
    assert events.count() == 0
    assert events.head_seq() == 0
    for _ in range(3):
        events.append_event(_event())
    assert events.count() == 3
    assert events.head_seq() == 3


# -- interface guarantees --------------------------------------------------


def test_repository_exposes_no_mutation_methods():
    """A method that should never be called should not exist."""
    surface = dir(EventRepository)
    for forbidden in ("update_event", "delete_event", "edit_event", "purge"):
        assert forbidden not in surface


# -- hash chain ------------------------------------------------------------


def test_first_event_chains_to_genesis(events: EventRepository):
    assert events.append_event(_event()).prev_hash == GENESIS_HASH


def test_each_event_chains_to_its_predecessor(events: EventRepository):
    stored = [events.append_event(_event()) for _ in range(5)]
    for previous, current in itertools.pairwise(stored):
        assert current.prev_hash == previous.event_hash


def test_chain_verifies_on_untouched_ledger(events: EventRepository):
    for _ in range(8):
        events.append_event(_event())
    report = events.verify_chain()
    assert report["valid"] is True
    assert report["checked"] == 8


def test_chain_detects_out_of_band_tampering(engine, events: EventRepository):
    """Triggers prevent tampering. The chain proves it did not happen.

    Dropping the trigger and editing a row is exactly the attack triggers
    cannot stop, so the chain has to catch it.
    """
    for _ in range(4):
        events.append_event(_event())
    assert events.verify_chain()["valid"] is True

    with engine.begin() as conn:
        conn.execute(text("DROP TRIGGER trg_audit_event_no_update"))
        conn.execute(text("UPDATE audit_event SET actor = 'HUMAN:mallory' WHERE seq = 2"))

    report = events.verify_chain()
    assert report["valid"] is False
    assert report["broken_at_seq"] == 2
    assert "does not match its recorded hash" in report["reason"]


def test_require_valid_chain_raises_on_tampering(engine, events: EventRepository):
    events.append_event(_event())
    events.append_event(_event())

    with engine.begin() as conn:
        conn.execute(text("DROP TRIGGER trg_audit_event_no_update"))
        conn.execute(
            text("UPDATE audit_event SET result_json = :forged WHERE seq = 1").bindparams(
                forged='{"forged":1}'
            )
        )

    with pytest.raises(LedgerIntegrityError, match="integrity failure at seq 1"):
        events.require_valid_chain()


def test_empty_ledger_verifies(events: EventRepository):
    report = events.verify_chain()
    assert report["valid"] is True
    assert report["checked"] == 0


def test_tampering_propagates_forward_through_the_chain(engine, events):
    """The property that makes this a chain rather than a set of hashes.

    A sophisticated attacker does not just edit a row; they edit it and
    recompute its hash, then repair the successor's ``prev_hash`` pointer.
    That must still fail, because every downstream hash is computed over its
    predecessor's hash and so must also be recomputed. If altering one event
    could be concealed by fixing up only its immediate neighbour, the hashes
    would be independent and the ledger would not be tamper-evident.
    """
    stored = [events.append_event(_event()) for _ in range(4)]
    assert events.verify_chain()["valid"] is True

    target = stored[1]
    forged_content = {"forged": True}
    forged = target.model_copy(update={"result_json": forged_content})
    forged_hash = _compute_hash(forged, target.prev_hash)

    with engine.begin() as conn:
        conn.execute(text("DROP TRIGGER trg_audit_event_no_update"))
        # Rewrite the target and recompute its own hash correctly.
        conn.execute(
            text(
                "UPDATE audit_event SET result_json = :body, event_hash = :h "
                "WHERE seq = :s"
            ).bindparams(body='{"forged":true}', h=forged_hash, s=target.seq)
        )
        # Repair the successor's pointer so the linkage check passes.
        conn.execute(
            text("UPDATE audit_event SET prev_hash = :h WHERE seq = :s").bindparams(
                h=forged_hash, s=stored[2].seq
            )
        )

    report = events.verify_chain()
    assert report["valid"] is False, (
        "a repaired-pointer tamper went undetected: event hashes are not "
        "chained to their predecessors"
    )
    # The break surfaces downstream of the edit, not at it.
    assert report["broken_at_seq"] == stored[2].seq


# -- transactional coupling ------------------------------------------------


def test_entity_and_event_commit_together(engine, catalog, events: EventRepository):
    """A fact without its event is a broken audit trail, and vice versa."""
    before = events.count()
    catalog.create_supplier(
        Supplier(
            name="Atomic Test Supplier",
            type=SupplierType.DISTRIBUTOR,
            gst_status=GSTStatus.UNKNOWN,
            issues_tax_invoice=False,
        )
    )
    assert events.count() == before + 1


def test_failed_entity_write_leaves_no_orphan_event(engine, catalog, events, product):
    """If the entity insert fails, its event must roll back with it."""
    catalog.create_sku(SKU(product_id=product.id, sku_code="ATOMIC-1"))
    before = events.count()

    with pytest.raises(IntegrityError):
        catalog.create_sku(SKU(product_id=product.id, sku_code="ATOMIC-1"))

    assert events.count() == before, "rolled-back write must not leave an event"
