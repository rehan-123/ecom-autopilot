"""Database tests: initialisation, idempotency, foreign keys, indexes, types."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, StatementError

from catalog.schemas import SKU
from database.connection import create_db_engine, foreign_keys_enabled
from database.migrations import (
    SCHEMA_VERSION,
    get_schema_version,
    init_db,
    list_indexes,
    list_tables,
    list_triggers,
)
from database.tables import (
    IMMUTABLE_TABLES,
    PROJECTION_TABLES,
    audit_event,
    metadata,
    supplier_offer,
)
from database.tables import (
    sku as sku_table,
)
from database.tables import (
    supplier_offer as offer_table,
)
from ledger.projections import assert_projections_are_derived
from tests.conftest import make_offer, utc


def test_database_initialises(engine):
    assert get_schema_version(engine) == SCHEMA_VERSION
    assert len(list_tables(engine)) == len(metadata.tables)


def test_init_db_is_idempotent(tmp_path):
    eng = create_db_engine(tmp_path / "idem.db")
    first = init_db(eng)
    second = init_db(eng)
    third = init_db(eng)
    assert first == second == third

    with eng.connect() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM schema_version")).scalar()
    assert rows == 1, "schema version must be recorded once, not per run"
    eng.dispose()


def test_all_expected_tables_exist(engine):
    tables = set(list_tables(engine))
    expected = {
        "supplier",
        "product",
        "sku",
        "product_identity_evidence",
        "supplier_offer",
        "rate_card",
        "tax_table",
        "return_benchmark",
        "threshold_profile",
        "scenario_profile",
        "landed_cost",
        "market_signal",
        "market_price_reference",
        "economics_snapshot",
        "price_headroom_check",
        "risk_assessment",
        "compliance_assessment",
        "opportunity_score",
        "ai_invocation",
        "sku_state_transition",
        "approval",
        "audit_event",
        "schema_version",
        "sku_current_state",
        "supplier_offer_current",
    }
    assert expected <= tables


def test_no_part_two_tables_exist(engine):
    """Part 2 concerns must not be representable in this schema."""
    forbidden = {"order", "orders", "purchase", "payment", "listing", "shipment"}
    assert forbidden.isdisjoint(set(list_tables(engine)))


def test_indexes_exist(engine):
    assert "ix_audit_event_timestamp" in list_indexes(engine, "audit_event")
    assert "ix_audit_event_type" in list_indexes(engine, "audit_event")
    assert "ix_supplier_offer_lookup" in list_indexes(engine, "supplier_offer")
    assert "ix_sku_product" in list_indexes(engine, "sku")


def test_triggers_exist_for_every_immutable_table(engine):
    triggers = set(list_triggers(engine))
    for table in IMMUTABLE_TABLES:
        assert f"trg_{table}_no_update" in triggers
        assert f"trg_{table}_no_delete" in triggers


def test_foreign_keys_are_enabled(engine):
    with engine.connect() as conn:
        assert foreign_keys_enabled(conn) is True


def test_foreign_key_violation_is_rejected(engine):
    """A SKU cannot reference a product that does not exist."""
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                sku_table.insert().values(
                    id=uuid.uuid4(),
                    product_id=uuid.uuid4(),  # no such product
                    sku_code="ORPHAN",
                    variant={},
                    pack_qty=1,
                    created_at=datetime.now(UTC),
                )
            )


def test_check_constraint_rejects_invalid_enum_via_raw_sql(engine, product):
    """Vocabulary is enforced in the database, not only in Pydantic."""
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO sku_state_transition "
                    "(id, sku_id, to_state, trigger, actor, created_at) "
                    "VALUES (:i, :s, 'PUBLISHED', 'RULE', 'SYSTEM', :t)"
                ),
                {
                    "i": str(uuid.uuid4()),
                    "s": str(uuid.uuid4()),
                    "t": datetime.now(UTC).isoformat(),
                },
            )


def test_unique_constraint_on_sku_code(engine, catalog, product):
    catalog.create_sku(SKU(product_id=product.id, sku_code="DUPE"))
    with pytest.raises(IntegrityError):
        catalog.create_sku(SKU(product_id=product.id, sku_code="DUPE"))


# -- custom types round-trip ----------------------------------------------


def test_datetime_round_trips_as_aware_utc(engine, supplier, product, catalog):
    stored_at = utc(2026, 8, 23, 9)
    catalog.record_offer(make_offer(supplier.id, product.id, "280.00", stored_at))
    offer = catalog.current_offer(supplier.id, product.id)
    assert offer.observed_at.tzinfo is not None
    assert offer.observed_at == stored_at


def test_decimal_round_trips_without_float_drift(engine, supplier, product, catalog):
    exact = "280.3300"
    catalog.record_offer(make_offer(supplier.id, product.id, exact, utc(2026, 8, 23)))
    offer = catalog.current_offer(supplier.id, product.id)
    assert offer.unit_cost_inr == Decimal(exact)
    assert isinstance(offer.unit_cost_inr, Decimal)


def test_float_money_is_refused_at_the_database_boundary(engine, supplier, product):
    """Money must never be stored as a float.

    SQLite's NUMERIC affinity converts to float, which loses precision
    silently and accumulates over a long ledger. The type decorator refuses
    the float rather than rounding it, because a rounding bug in money
    arithmetic surfaces only once the numbers are already wrong.
    """

    with (
        pytest.raises((TypeError, StatementError), match="float"),
        engine.begin() as conn,
    ):
        conn.execute(
            offer_table.insert().values(
                id=uuid.uuid4(),
                supplier_id=supplier.id,
                product_id=product.id,
                unit_cost_inr=280.33,  # float, not Decimal
                cost_is_gst_inclusive=True,
                gst_rate_pct=Decimal("18"),
                tax_invoice_available=True,
                moq_units=1,
                inbound_freight_inr=Decimal("0"),
                compliance_docs={},
                observed_at=datetime.now(UTC),
            )
        )


def test_decimal_scale_survives_a_round_trip(engine, supplier, product, catalog):
    """Trailing precision is preserved exactly, not normalised away."""
    catalog.record_offer(
        make_offer(supplier.id, product.id, "1234.5678", utc(2026, 8, 23))
    )
    stored = catalog.current_offer(supplier.id, product.id).unit_cost_inr
    assert stored == Decimal("1234.5678")
    assert str(stored) == "1234.5678"
    assert stored != float(stored) or isinstance(stored, Decimal)


def test_naive_datetime_is_refused_at_the_database_boundary(engine):
    """Defence in depth: the type decorator refuses naive values too."""
    with pytest.raises((ValueError, StatementError), match="naive datetime"):
        with engine.begin() as conn:
            conn.execute(
                audit_event.insert().values(
                    id=uuid.uuid4(),
                    timestamp=datetime(2026, 8, 23),  # naive
                    event_type="ERROR",
                    actor="SYSTEM",
                    inputs_json={},
                    reference_versions_json={},
                    result_json={},
                    event_hash="a" * 64,
                )
            )


def test_projection_tables_are_registered_as_droppable():
    assert_projections_are_derived()
    assert set(PROJECTION_TABLES) == {"sku_current_state", "supplier_offer_current"}


def test_audit_event_has_no_foreign_keys():
    """The ledger must be able to record events about entities that were
    never persisted, including failures."""
    assert len(audit_event.foreign_keys) == 0


def test_supplier_offer_has_no_superseded_column():
    assert "superseded_by_offer_id" not in supplier_offer.c
