"""Tests for defects found in the Milestone 1 audit.

Each test here corresponds to a defect that the original suite did not catch.
They are kept together so the regression they guard is obvious.
"""

from __future__ import annotations

import threading
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, OperationalError

from catalog.enums import (
    AIService,
    ApprovalDecision,
    EventType,
    SKUState,
    TransitionTrigger,
)
from catalog.repository import CatalogRepository, InvalidTransitionError
from catalog.schemas import SKU, AIInvocation, Approval, AuditEvent
from database.tables import supplier_offer_current
from ledger.events import EventRepository
from ledger.projections import drop_projections, rebuild_projections
from ledger.replay import offer_history, replay_events
from tests.conftest import make_offer, utc

# ==========================================================================
# D1 - concurrent appends forked the hash chain
# ==========================================================================


def _append_concurrently(engine, n: int) -> tuple[list[int], list[str]]:
    repo = EventRepository(engine)
    barrier = threading.Barrier(n)
    ok: list[int] = []
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=30)
            repo.append_event(
                AuditEvent(
                    event_type=EventType.ERROR, actor="SYSTEM", result_json={"w": i}
                )
            )
            ok.append(i)
        # An exception raised in a worker thread is discarded by the runtime,
        # so the worker must capture whatever occurs and the test asserts the
        # collected list is empty. Narrowing this would let an unanticipated
        # failure mode pass silently, which is the opposite of the intent.
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return ok, errors


@pytest.mark.parametrize("writers", [8, 24])
def test_concurrent_appends_do_not_fork_the_chain(engine, writers: int):
    """Two writers must never read the same predecessor hash.

    pysqlite's DEFERRED transactions took a read snapshot first and the write
    lock only at INSERT, so concurrent appends could both read the same
    predecessor and both commit. The chain forked and, because the table is
    append-only, could never be repaired. BEGIN IMMEDIATE plus a UNIQUE
    constraint on prev_hash closes it from both ends.
    """
    ok, errors = _append_concurrently(engine, writers)
    assert errors == []
    assert len(ok) == writers

    repo = EventRepository(engine)
    events = repo.get_events()
    prev_hashes = [e.prev_hash for e in events]

    assert len(events) == writers
    assert len(set(prev_hashes)) == writers, "two events share a predecessor"
    assert repo.verify_chain()["valid"] is True


def test_duplicate_prev_hash_is_rejected_by_the_schema(engine, events):
    """The constraint holds even against writers that bypass the repository."""
    events.append_event(AuditEvent(event_type=EventType.ERROR, actor="SYSTEM"))
    first = events.get_events()[0]

    with pytest.raises(IntegrityError, match="prev_hash"), engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO audit_event "
                "(id, timestamp, event_type, actor, inputs_json, "
                " reference_versions_json, result_json, prev_hash, event_hash) "
                "VALUES (:i, :t, 'ERROR', 'SYSTEM', '{}', '{}', '{}', :p, :h)"
            ).bindparams(
                i=str(uuid.uuid4()),
                t=first.timestamp.isoformat(),
                p=first.prev_hash,  # same predecessor -> a fork
                h="f" * 64,
            )
        )


def test_duplicate_event_hash_is_rejected(engine, events):
    events.append_event(AuditEvent(event_type=EventType.ERROR, actor="SYSTEM"))
    first = events.get_events()[0]

    with pytest.raises(IntegrityError, match="event_hash"), engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO audit_event "
                "(id, timestamp, event_type, actor, inputs_json, "
                " reference_versions_json, result_json, prev_hash, event_hash) "
                "VALUES (:i, :t, 'ERROR', 'SYSTEM', '{}', '{}', '{}', :p, :h)"
            ).bindparams(
                i=str(uuid.uuid4()),
                t=first.timestamp.isoformat(),
                p="e" * 64,
                h=first.event_hash,
            )
        )


def test_concurrent_catalog_writes_keep_the_ledger_consistent(engine, product):
    """End-to-end: parallel SKU creation must not corrupt the ledger."""
    repo = CatalogRepository(engine)
    barrier = threading.Barrier(10)
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=30)
            repo.create_sku(SKU(product_id=product.id, sku_code=f"SKU-{i:03d}"))
        except Exception as exc:  # noqa: BLE001 - see note above
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    assert EventRepository(engine).verify_chain()["valid"] is True
    assert rebuild_projections(engine)["sku_current_state"] == 10


# ==========================================================================
# D4 - TERMINAL_STATES was declared but never enforced
# ==========================================================================


def test_leaving_a_terminal_state_requires_explicit_reopen(catalog, sku):
    catalog.transition_sku(sku.id, SKUState.REJECTED, trigger=TransitionTrigger.RULE)

    with pytest.raises(InvalidTransitionError, match="terminal state"):
        catalog.transition_sku(sku.id, SKUState.APPROVED, trigger=TransitionTrigger.RULE)


def test_deliberate_reopen_is_permitted_and_flagged(catalog, events, sku):
    """Reopening is legitimate — a supplier price can fall — but not silent."""
    catalog.transition_sku(sku.id, SKUState.REJECTED, trigger=TransitionTrigger.RULE)
    catalog.transition_sku(
        sku.id,
        SKUState.ECONOMIC_ANALYSIS,
        trigger=TransitionTrigger.HUMAN,
        actor="HUMAN:rehan",
        rationale="supplier requoted at a lower landed cost",
        allow_reopen=True,
    )
    assert catalog.current_state(sku.id) is SKUState.ECONOMIC_ANALYSIS

    latest = events.get_events(event_type=EventType.SKU_STATE_CHANGED)[-1]
    assert latest.result_json["reopened"] is True


def test_listing_ready_is_also_terminal(catalog, sku):
    for state in (
        SKUState.DATA_COLLECTION,
        SKUState.APPROVED,
        SKUState.LISTING_READY,
    ):
        catalog.transition_sku(sku.id, state, trigger=TransitionTrigger.RULE)

    with pytest.raises(InvalidTransitionError, match="terminal state"):
        catalog.transition_sku(sku.id, SKUState.REJECTED, trigger=TransitionTrigger.RULE)


def test_transitioning_an_unknown_sku_is_rejected(catalog):
    with pytest.raises(InvalidTransitionError, match="does not exist"):
        catalog.transition_sku(
            uuid.uuid4(), SKUState.APPROVED, trigger=TransitionTrigger.RULE
        )


# ==========================================================================
# D2/D3 - Approval, AIInvocation and errors had no persistence path
# ==========================================================================


def test_approval_is_persisted_and_ledgered(catalog, events, sku):
    approval = catalog.record_approval(
        Approval(
            sku_id=sku.id,
            decision=ApprovalDecision.DEFERRED,
            approver="rehan",
            note="awaiting market price reference",
        )
    )
    stored = catalog.approvals_for(sku.id)
    assert [a.id for a in stored] == [approval.id]
    assert stored[0].decision is ApprovalDecision.DEFERRED

    recorded = events.get_events(event_type=EventType.APPROVAL_RECORDED)
    assert len(recorded) == 1
    assert recorded[0].actor == "HUMAN:rehan"
    assert recorded[0].result_json["decision"] == "DEFERRED"


def test_approval_does_not_move_sku_state(catalog, sku):
    """Recording a decision is persistence; acting on it is a later milestone."""
    before = catalog.current_state(sku.id)
    catalog.record_approval(
        Approval(sku_id=sku.id, decision=ApprovalDecision.APPROVED, approver="rehan")
    )
    assert catalog.current_state(sku.id) == before


def test_ai_invocation_is_persisted_and_ledgered(catalog, events, sku, product):
    catalog.record_ai_invocation(
        AIInvocation(
            service=AIService.JUDGE_IDENTITY,
            sku_id=sku.id,
            product_id=product.id,
            model="none-milestone-1",
            model_version="n/a",
            prompt_version="identity-v0",
            prompt_hash="a" * 64,
            parsed_ok=False,
            schema_violations=["no client configured"],
        )
    )
    recorded = events.get_events(event_type=EventType.AI_INVOKED)
    assert len(recorded) == 1
    assert recorded[0].actor == "AI:JUDGE_IDENTITY"
    assert recorded[0].reference_versions_json["prompt_version"] == "identity-v0"
    assert recorded[0].result_json["parsed_ok"] is False


def test_error_is_recorded_for_an_entity_that_was_never_persisted(catalog, events):
    """The ledger has no foreign keys precisely so this is possible."""
    phantom = uuid.uuid4()
    catalog.record_error(
        "supplier catalogue row failed validation",
        sku_id=phantom,
        context={"row": 42, "field": "unit_cost_inr"},
    )
    recorded = events.get_events(event_type=EventType.ERROR)
    assert len(recorded) == 1
    assert recorded[0].sku_id == phantom
    assert recorded[0].error_json["message"].startswith("supplier catalogue row")
    assert recorded[0].inputs_json["row"] == 42


def test_error_events_do_not_break_the_chain(catalog, events, sku):
    catalog.record_error("first", sku_id=sku.id)
    catalog.record_error("second")
    assert events.verify_chain()["valid"] is True


# ==========================================================================
# D12 - observed_at was inferred from the event timestamp
# ==========================================================================


def test_backfilled_offer_sorts_by_observation_not_record_time(
    engine, catalog, supplier, product
):
    """A quote observed on the 23rd but recorded on the 27th is still the 23rd.

    Projections previously read the event timestamp, so a backfill would have
    been treated as the newest quote and the wrong price would have become
    current.
    """
    catalog.record_offer(make_offer(supplier.id, product.id, "310.00", utc(2026, 8, 27)))
    # Recorded second, observed first.
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23)))

    history = offer_history(engine, supplier.id, product.id)
    assert history.current_cost == Decimal("310.00"), (
        "the 27 Aug quote is the latest observation despite being recorded second"
    )

    rebuild_projections(engine)
    with engine.connect() as conn:
        row = conn.execute(select(supplier_offer_current)).first()
    assert Decimal(row._mapping["unit_cost_inr"]) == Decimal("310.00")
    assert row._mapping["observed_at"] == utc(2026, 8, 27)


def test_offer_payload_carries_observed_at(catalog, events, supplier, product):
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23)))
    event = events.get_events(event_type=EventType.SUPPLIER_OFFER_CREATED)[0]
    assert event.result_json["observed_at"].startswith("2026-08-23")


# ==========================================================================
# D5 - _execute returned different types on its two call paths
# ==========================================================================


def test_reads_behave_identically_with_and_without_a_connection(engine, events):
    for _ in range(3):
        events.append_event(AuditEvent(event_type=EventType.ERROR, actor="SYSTEM"))

    without = events.get_events()
    with engine.connect() as conn:
        within = events.get_events(conn=conn)

    assert [e.id for e in without] == [e.id for e in within]
    assert events.count() == len(without)
    with engine.connect() as conn:
        assert events.count(conn=conn) == len(within)
        assert events.head_seq(conn=conn) == events.head_seq()


# ==========================================================================
# D6 - current_state introspected the schema on every call
# ==========================================================================


def test_state_read_recovers_after_projection_is_dropped_and_rebuilt(
    engine, catalog, sku
):
    """A dropped projection must cost a fallback query, never a wrong answer."""
    catalog.transition_sku(
        sku.id, SKUState.DATA_COLLECTION, trigger=TransitionTrigger.RULE
    )
    rebuild_projections(engine)
    assert catalog.current_state(sku.id) is SKUState.DATA_COLLECTION

    drop_projections(engine)
    assert catalog.current_state(sku.id) is SKUState.DATA_COLLECTION

    rebuild_projections(engine)
    assert catalog.current_state(sku.id) is SKUState.DATA_COLLECTION


def test_state_read_does_not_swallow_unrelated_database_errors(engine, catalog, sku):
    """The fallback triggers on a missing table only, not on any failure."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE sku_current_state"))
        conn.execute(text("CREATE TABLE sku_current_state (wrong_column TEXT)"))

    with pytest.raises(OperationalError):
        catalog.current_state(sku.id)


# ==========================================================================
# Cross-cutting: the ledger stays replayable through all of the above
# ==========================================================================


def test_full_history_replays_after_every_new_write_path(
    engine, catalog, events, supplier, product, sku
):
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23)))
    catalog.transition_sku(
        sku.id, SKUState.DATA_COLLECTION, trigger=TransitionTrigger.RULE
    )
    catalog.record_ai_invocation(
        AIInvocation(
            service=AIService.EXTRACTOR,
            sku_id=sku.id,
            model="none-milestone-1",
            prompt_version="v0",
            prompt_hash="b" * 64,
            parsed_ok=True,
        )
    )
    catalog.record_error("transient supplier feed timeout", sku_id=sku.id)
    catalog.record_approval(
        Approval(sku_id=sku.id, decision=ApprovalDecision.DEFERRED, approver="rehan")
    )

    assert events.verify_chain()["valid"] is True
    first, second = replay_events(engine), replay_events(engine)
    assert first.fingerprint() == second.fingerprint()
    assert first.skus[sku.id].state is SKUState.DATA_COLLECTION


def test_unhandled_event_types_are_counted_not_silently_dropped(
    engine, catalog, events, sku
):
    """Replay must surface event types it does not fold, not ignore them."""
    catalog.record_error("noted", sku_id=sku.id)
    state = replay_events(engine)
    assert state.unhandled_event_types.get("ERROR") == 1
