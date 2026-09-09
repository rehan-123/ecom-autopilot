"""Shared fixtures.

Every test gets a fresh file-backed database in a temp directory rather than
``:memory:``. In-memory SQLite drops the schema when the pooled connection is
recycled, which would make the trigger tests pass for the wrong reason.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import Engine

from catalog.enums import GSTStatus, IdentityConfidence, SupplierType
from catalog.repository import CatalogRepository
from catalog.schemas import SKU, Product, Supplier, SupplierOffer
from database.connection import create_db_engine
from database.migrations import init_db
from ledger.events import EventRepository


@pytest.fixture
def engine(tmp_path) -> Engine:
    """An initialised database, isolated per test."""
    eng = create_db_engine(tmp_path / "test.db")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def catalog(engine: Engine) -> CatalogRepository:
    return CatalogRepository(engine)


@pytest.fixture
def events(engine: Engine) -> EventRepository:
    return EventRepository(engine)


@pytest.fixture
def supplier(catalog: CatalogRepository) -> Supplier:
    return catalog.create_supplier(
        Supplier(
            name="Wholesale Supplier B",
            type=SupplierType.WHOLESALER,
            gst_status=GSTStatus.REGISTERED,
            gstin="07AABCU9603R1ZX",
            issues_tax_invoice=True,
            state_code="07",
        )
    )


@pytest.fixture
def product(catalog: CatalogRepository) -> Product:
    return catalog.create_product(
        Product(
            canonical_name="Rechargeable Trimmer KX-880",
            category="personal_care_electrical",
            hsn_code="8510",
            weight_g=310,
            identity_confidence=IdentityConfidence.HIGH,
        )
    )


@pytest.fixture
def sku(catalog: CatalogRepository, product: Product) -> SKU:
    return catalog.create_sku(
        SKU(product_id=product.id, sku_code="SKU-TRIM-880", pack_qty=1)
    )


def make_offer(
    supplier_id, product_id, cost: str, observed_at: datetime
) -> SupplierOffer:
    """Build an offer with sensible defaults for tests."""
    return SupplierOffer(
        supplier_id=supplier_id,
        product_id=product_id,
        unit_cost_inr=Decimal(cost),
        cost_is_gst_inclusive=True,
        gst_rate_pct=Decimal("18"),
        tax_invoice_available=True,
        moq_units=50,
        available_qty=200,
        inbound_freight_inr=Decimal("12.50"),
        observed_at=observed_at,
    )


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


@pytest.fixture
def populated(engine: Engine, catalog, events, supplier, product, sku) -> dict:
    """One valid row in every immutable table.

    Needed because ``BEFORE UPDATE`` and ``BEFORE DELETE`` triggers fire
    per-row: on an empty table they never run, and an immutability test
    against an empty table passes for the wrong reason.

    Doubles as an end-to-end check that every assessment structure accepts a
    valid row and that the foreign key graph is satisfiable.
    """
    # Imports are local to this fixture deliberately: it exists to seed one
    # row in every immutable table, and keeping the (long) table and enum
    # imports beside that code keeps the fixture readable as one unit.
    import uuid as _uuid

    from sqlalchemy import insert

    from catalog.enums import (
        AIService,
        ComplianceResult,
        GateResult,
        HeadroomResult,
        RiskLevel,
        RiskTier,
        ScenarioName,
        ScoreResult,
        SignalSource,
        SignalStatus,
    )
    from database.tables import (
        ai_invocation,
        compliance_assessment,
        economics_snapshot,
        landed_cost,
        market_price_reference,
        market_signal,
        opportunity_score,
        price_headroom_check,
        risk_assessment,
    )

    now = datetime.now(UTC)
    offer = catalog.record_offer(
        make_offer(supplier.id, product.id, "280.00", utc(2026, 8, 23))
    )

    ids = {
        "landed_cost": _uuid.uuid4(),
        "economics_snapshot": _uuid.uuid4(),
        "market_price_reference": _uuid.uuid4(),
    }

    with engine.begin() as conn:
        conn.execute(
            insert(market_price_reference).values(
                id=ids["market_price_reference"],
                sku_id=sku.id,
                collected_at=now,
                collected_by="rehan",
                comparable_json=[],
                n_comparables=3,
                price_p25_inr=Decimal("499"),
                price_median_inr=Decimal("549"),
                price_p75_inr=Decimal("649"),
            )
        )
        conn.execute(
            insert(market_signal).values(
                id=_uuid.uuid4(),
                sku_id=sku.id,
                signal_type="SEARCH_INTEREST",
                status=str(SignalStatus.DATA_UNAVAILABLE),
                source=str(SignalSource.MANUAL),
                collected_at=now,
                collected_by="rehan",
            )
        )
        conn.execute(
            insert(landed_cost).values(
                id=ids["landed_cost"],
                sku_id=sku.id,
                supplier_offer_id=offer.id,
                scenario=str(ScenarioName.EXPECTED),
                calc_version="0.0.0-placeholder",
                unit_cost_ex_gst_inr=Decimal("237.29"),
                input_tax_credit_inr=Decimal("42.71"),
                effective_cogs_inr=Decimal("237.29"),
                inbound_freight_per_unit_inr=Decimal("12.50"),
                moq_carry_cost_inr=Decimal("0"),
                inspection_qc_cost_inr=Decimal("0"),
                packaging_cost_inr=Decimal("0"),
                inbound_defect_provision_inr=Decimal("0"),
                total_landed_cost_inr=Decimal("249.79"),
                assumptions_json={"note": "structure only; no engine this milestone"},
                computed_at=now,
            )
        )
        conn.execute(
            insert(economics_snapshot).values(
                id=ids["economics_snapshot"],
                sku_id=sku.id,
                landed_cost_id=ids["landed_cost"],
                scenario=str(ScenarioName.EXPECTED),
                calc_version="0.0.0-placeholder",
                rate_card_version="unset",
                tax_table_version="unset",
                return_benchmark_version="unset",
                threshold_profile_version="unset",
                scenario_profile_version="unset",
                selling_price_inr=Decimal("799"),
                net_profit_inr=Decimal("0"),
                net_margin_pct=Decimal("0"),
                roi_pct=Decimal("0"),
                breakeven_price_inr=Decimal("0"),
                min_profitable_price_inr=Decimal("0"),
                gate_result=str(GateResult.MARGINAL),
                gap_to_pass_json={},
                computed_at=now,
            )
        )
        conn.execute(
            insert(price_headroom_check).values(
                id=_uuid.uuid4(),
                sku_id=sku.id,
                economics_snapshot_id=ids["economics_snapshot"],
                market_price_reference_id=ids["market_price_reference"],
                min_profitable_price_inr=Decimal("0"),
                market_p25_inr=Decimal("499"),
                headroom_pct=Decimal("0"),
                result=str(HeadroomResult.MARGINAL),
                computed_at=now,
            )
        )
        conn.execute(
            insert(risk_assessment).values(
                id=_uuid.uuid4(),
                sku_id=sku.id,
                risk_tier=str(RiskTier.MEDIUM),
                return_risk=str(RiskLevel.UNKNOWN),
                fragility_risk=str(RiskLevel.UNKNOWN),
                sizing_risk=str(RiskLevel.UNKNOWN),
                warranty_risk=str(RiskLevel.UNKNOWN),
                supplier_concentration_risk=str(RiskLevel.UNKNOWN),
                seasonality_risk=str(RiskLevel.UNKNOWN),
                rationale_json={},
                ruleset_version="unset",
                computed_at=now,
            )
        )
        conn.execute(
            insert(compliance_assessment).values(
                id=_uuid.uuid4(),
                sku_id=sku.id,
                ruleset_version="unset",
                overall_result=str(ComplianceResult.UNKNOWN),
                checks_json=[],
                blocking_failures_json=[],
                computed_at=now,
            )
        )
        conn.execute(
            insert(opportunity_score).values(
                id=_uuid.uuid4(),
                sku_id=sku.id,
                weights_version="unset",
                value=None,
                coverage_pct=Decimal("30"),
                components_json=[],
                missing_signals_json=["DEMAND", "COMPETITION"],
                result=str(ScoreResult.INSUFFICIENT_DATA),
                computed_at=now,
            )
        )
        conn.execute(
            insert(ai_invocation).values(
                id=_uuid.uuid4(),
                service=str(AIService.EXTRACTOR),
                sku_id=sku.id,
                product_id=product.id,
                model="none",
                prompt_version="unset",
                prompt_hash="0" * 64,
                input_refs=[],
                cost_inr=Decimal("0"),
                parsed_ok=False,
                schema_violations=[],
                findings=[],
                created_at=now,
            )
        )

    return {"supplier": supplier, "product": product, "sku": sku, "offer": offer}
