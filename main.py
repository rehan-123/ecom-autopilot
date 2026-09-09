"""CLI entrypoint for the Intelligence & Product Engine foundation.

Commands
--------
``init-db``               create schema, triggers, indexes; safe to re-run
``demo``                  seed a worked example and print the ledger
``rebuild-projections``   drop and rebuild every projection from the ledger
``verify-ledger``         recompute the hash chain and report integrity
``status``                schema version, table and event counts

Part 2 capabilities are absent by construction. There is no purchase command,
no publish command, and no marketplace client anywhere in this codebase — not
a disabled one, an absent one. A configuration flag can be flipped; a
capability that does not exist cannot be.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from catalog.enums import (
    AIService,
    ApprovalDecision,
    EvidenceType,
    GSTStatus,
    IdentityConfidence,
    SKUState,
    SupplierType,
    TransitionTrigger,
)
from catalog.repository import CatalogRepository, InvalidTransitionError
from catalog.schemas import (
    SKU,
    AIInvocation,
    Approval,
    Product,
    ProductIdentityEvidence,
    Supplier,
    SupplierOffer,
    new_id,
)
from database.connection import DEFAULT_DB_PATH, create_db_engine
from database.migrations import (
    get_schema_version,
    init_db,
    list_tables,
    list_triggers,
)
from ledger.events import EventRepository
from ledger.projections import rebuild_projections
from ledger.replay import offer_history, replay_events

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def _rule(title: str = "") -> None:
    print(f"\n{'-' * 74}")
    if title:
        print(title)
        print("-" * 74)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_init_db(db_path: Path) -> int:
    engine = create_db_engine(db_path)
    summary = init_db(engine)
    _rule("DATABASE INITIALISED")
    for key, value in summary.items():
        print(f"  {key:<20} {value}")
    print(f"  {'path':<20} {db_path}")
    return 0


def cmd_status(db_path: Path) -> int:
    engine = create_db_engine(db_path)
    version = get_schema_version(engine)
    if version is None:
        print("Database is not initialised. Run: python main.py init-db")
        return 1
    repo = EventRepository(engine)
    _rule("STATUS")
    print(f"  schema version   {version}")
    print(f"  tables           {len(list_tables(engine))}")
    print(f"  triggers         {len(list_triggers(engine))}")
    print(f"  ledger events    {repo.count()}")
    print(f"  ledger head seq  {repo.head_seq()}")
    return 0


def cmd_verify_ledger(db_path: Path) -> int:
    engine = create_db_engine(db_path)
    report = EventRepository(engine).verify_chain()
    _rule("LEDGER INTEGRITY")
    for key, value in report.items():
        print(f"  {key:<16} {value}")
    return 0 if report["valid"] else 2


def cmd_rebuild_projections(db_path: Path) -> int:
    engine = create_db_engine(db_path)
    counts = rebuild_projections(engine)
    _rule("PROJECTIONS REBUILT FROM LEDGER")
    for name, count in counts.items():
        print(f"  {name:<28} {count} row(s)")
    return 0


def _demo_catalog(catalog: CatalogRepository) -> tuple[Supplier, Product, SKU]:
    """Demo section: catalog."""
    _rule("1. SUPPLIER, PRODUCT, SKU")

    supplier = catalog.create_supplier(
        Supplier(
            name="Wholesale Supplier B",
            type=SupplierType.WHOLESALER,
            gst_status=GSTStatus.REGISTERED,
            gstin="07AABCU9603R1ZX",
            issues_tax_invoice=True,
            state_code="07",
            notes="Issues tax invoice, so input tax credit is available.",
        )
    )
    print(f"  supplier  {supplier.name}  ({supplier.id})")
    print(f"            tax invoice: {supplier.issues_tax_invoice}  -> ITC available")

    product = catalog.create_product(
        Product(
            canonical_name="Rechargeable Trimmer KX-880",
            category="personal_care_electrical",
            hsn_code="8510",
            manufacturer="KX Appliances",
            country_of_origin="India",
            weight_g=310,
            warranty_months=6,
            identity_confidence=IdentityConfidence.HIGH,
        )
    )
    print(f"  product   {product.canonical_name}  ({product.id})")

    catalog.add_identity_evidence(
        ProductIdentityEvidence(
            product_id=product.id,
            evidence_type=EvidenceType.MANUFACTURER_CODE,
            value="KX-880",
        )
    )
    print("  evidence  MANUFACTURER_CODE=KX-880 recorded for identity resolution")

    sku = catalog.create_sku(
        SKU(product_id=product.id, sku_code="SKU-TRIM-880", pack_qty=1)
    )
    print(
        f"  sku       {sku.sku_code}  ({sku.id})  state={catalog.current_state(sku.id)}"
    )

    return supplier, product, sku


def _demo_offers(
    catalog: CatalogRepository, supplier: Supplier, product: Product
) -> datetime:
    """Demo section: offers."""
    _rule("2. TWO HISTORICAL SUPPLIER OFFERS (a price change, not an edit)")

    day_one = datetime(2026, 8, 23, 9, 0, tzinfo=UTC)
    offer_a = catalog.record_offer(
        SupplierOffer(
            supplier_id=supplier.id,
            product_id=product.id,
            unit_cost_inr=Decimal("280.00"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
            moq_units=50,
            available_qty=400,
            inbound_freight_inr=Decimal("12.50"),
            dispatch_days=2,
            observed_at=day_one,
            source_document_ref="quote-2026-08-23.pdf",
        )
    )
    offer_b = catalog.record_offer(
        SupplierOffer(
            supplier_id=supplier.id,
            product_id=product.id,
            unit_cost_inr=Decimal("310.00"),
            cost_is_gst_inclusive=True,
            gst_rate_pct=Decimal("18"),
            tax_invoice_available=True,
            moq_units=50,
            available_qty=250,
            inbound_freight_inr=Decimal("12.50"),
            dispatch_days=2,
            observed_at=day_one + timedelta(days=4),
            source_document_ref="quote-2026-08-27.pdf",
        )
    )
    print(
        f"  offer #1  Rs {offer_a.unit_cost_inr}  observed {offer_a.observed_at.date()}"
    )
    print(
        f"  offer #2  Rs {offer_b.unit_cost_inr}  observed {offer_b.observed_at.date()}"
    )

    preserved = catalog.offers_for(supplier.id, product.id)
    print(f"\n  both rows preserved: {len(preserved)} offers on record")
    print(f"  costs in order:      {[str(o.unit_cost_inr) for o in preserved]}")
    current = catalog.current_offer(supplier.id, product.id)
    print(f"  current (derived):   Rs {current.unit_cost_inr}")

    return day_one


def _demo_transitions(catalog: CatalogRepository, sku: SKU) -> None:
    """Demo section: transitions."""
    _rule("3. LIFECYCLE TRANSITIONS (state is never silently changed)")

    for to_state, rationale in (
        (SKUState.DATA_COLLECTION, "supplier offer ingested"),
        (SKUState.ECONOMIC_ANALYSIS, "landed cost inputs complete"),
        (SKUState.BLOCKED_AWAITING_DATA, "MarketPriceReference not yet collected"),
    ):
        catalog.transition_sku(
            sku.id,
            to_state,
            trigger=TransitionTrigger.RULE,
            rule_id="milestone1.demo",
            rationale=rationale,
        )
        print(f"  -> {to_state:<24} {rationale}")

    view = catalog.get_sku_view(sku.id)
    print(f"\n  projected state: {view.state} @ {view.state_updated_at.isoformat()}")


def _demo_write_paths(catalog: CatalogRepository, sku: SKU, product: Product) -> bool:
    """Demo section: write paths."""
    _rule("4. TERMINAL STATE GUARD, APPROVAL, AI INVOCATION, ERROR")

    catalog.record_error(
        "supplier catalogue row 42 failed validation",
        sku_id=sku.id,
        context={"row": 42, "field": "unit_cost_inr"},
    )
    print("  error     recorded against a row that was never persisted")

    catalog.record_ai_invocation(
        AIInvocation(
            service=AIService.JUDGE_IDENTITY,
            sku_id=sku.id,
            product_id=product.id,
            model="none-milestone-1",
            prompt_version="identity-v0",
            prompt_hash="0" * 64,
            parsed_ok=False,
            schema_violations=["no client configured in Milestone 1"],
        )
    )
    print("  ai        invocation logged (persistence only; no client exists)")

    catalog.record_approval(
        Approval(
            sku_id=sku.id,
            decision=ApprovalDecision.DEFERRED,
            approver="rehan",
            note="awaiting MarketPriceReference",
        )
    )
    print(
        f"  approval  DEFERRED recorded; SKU state unchanged "
        f"({catalog.current_state(sku.id)})"
    )

    catalog.transition_sku(
        sku.id,
        SKUState.REJECTED,
        trigger=TransitionTrigger.RULE,
        rationale="price headroom negative against market p25",
    )
    try:
        catalog.transition_sku(sku.id, SKUState.APPROVED, trigger=TransitionTrigger.RULE)
        print("  reopen    SUCCEEDED without intent  <-- guarantee broken")
        return 2
    except InvalidTransitionError:
        print("  reopen    REJECTED  (terminal state needs allow_reopen=True)")

    catalog.transition_sku(
        sku.id,
        SKUState.ECONOMIC_ANALYSIS,
        trigger=TransitionTrigger.HUMAN,
        actor="HUMAN:rehan",
        rationale="supplier requoted lower",
        allow_reopen=True,
    )
    print(
        f"  reopen    PERMITTED with explicit intent, flagged in the ledger "
        f"-> {catalog.current_state(sku.id)}"
    )

    return True


def _demo_ledger(events: EventRepository) -> None:
    """Demo section: ledger."""
    _rule("5. IMMUTABLE LEDGER")

    print(f"  events appended: {events.count()}")
    for event in events.get_events():
        print(f"    seq {event.seq:>2}  {event.event_type:<34} {event.actor}")


def _demo_tamper(engine, events: EventRepository, offer_a: SupplierOffer) -> bool:
    """Demo section: tamper."""
    _rule("6. TAMPER RESISTANCE")

    for statement, label in (
        ("UPDATE audit_event SET actor = 'SYSTEM' WHERE seq = 1", "UPDATE audit_event"),
        ("DELETE FROM audit_event WHERE seq = 1", "DELETE audit_event"),
        (
            f"UPDATE supplier_offer SET unit_cost_inr = '1.00' WHERE id = '{offer_a.id}'",
            "UPDATE supplier_offer",
        ),
    ):
        try:
            with engine.begin() as conn:
                conn.execute(text(statement))
            print(f"  {label:<24} SUCCEEDED  <-- guarantee broken")
            return 2
        except IntegrityError as exc:
            # Narrower than `Exception` on purpose: the trigger raises
            # IntegrityError specifically, so anything else reaching here is a
            # different bug and should surface rather than read as a pass.
            reason = str(exc).split("\n")[0]
            print(f"  {label:<24} REJECTED   {reason[:70]}")

    integrity = events.verify_chain()
    print(f"\n  hash chain valid: {integrity['valid']}  ({integrity['checked']} events)")

    return True


def _demo_projections(engine, catalog: CatalogRepository, sku: SKU) -> None:
    """Demo section: projections."""
    _rule("7. PROJECTION REBUILD FROM LEDGER")

    counts = rebuild_projections(engine)
    for name, count in counts.items():
        print(f"  {name:<28} {count} row(s)")
    print(f"  state after rebuild:         {catalog.current_state(sku.id)}")


def _demo_replay(engine, supplier: Supplier, product: Product, day_one: datetime) -> None:
    """Demo section: replay."""
    _rule("8. REPLAY")

    state = replay_events(engine)
    print(f"  events applied:  {state.events_applied}")
    print(f"  suppliers:       {len(state.suppliers)}")
    print(f"  products:        {len(state.products)}")
    print(f"  SKUs:            {len(state.skus)}")

    history = offer_history(engine, supplier.id, product.id)
    print(f"  offer history:   {[str(c) for _, c, _ in history.observations]}")
    print(f"  superseded:      {history.superseded_count}")

    at_day_one = replay_events(engine, until_timestamp=day_one + timedelta(hours=1))
    early = at_day_one.offers.get((supplier.id, product.id))
    print(f"\n  reconstructed at {day_one.date()}: cost was Rs {early.current_cost}")
    print(f"  today:                      cost is  Rs {history.current_cost}")

    first, second = replay_events(engine), replay_events(engine)
    print(f"\n  replay deterministic: {first.fingerprint() == second.fingerprint()}")


def _has_execution_history(db_path: Path) -> bool:
    """Whether this database holds recorded executions.

    Read-only and failure-tolerant: a database without the table simply has no
    history, which is the common case for a fresh checkout.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        rows = conn.execute(
            "SELECT count(*) FROM sqlite_master "
            "WHERE type='table' AND name='execution_ledger'"
        ).fetchone()[0]
        if not rows:
            return False
        return conn.execute("SELECT count(*) FROM execution_ledger").fetchone()[0] > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def cmd_demo(db_path: Path) -> int:
    """Seed a worked example demonstrating every Milestone 1 guarantee.

    Resets the database first. A demonstration seeds fixed suppliers and
    products by name, so a second run previously died on a UNIQUE constraint
    -- a demo that cannot be run twice is broken. Resetting is safe because
    the ledger's guarantees are about append-only behaviour *within* a
    database, not about this file surviving a demo re-run; `init-db` and
    `status` operate on whatever database you point them at.
    """
    # Refuse to reset a database that holds real execution history. The demo
    # is destructive by design, and the default path is the same one a real
    # install uses -- running it against production would delete the very
    # ledger the execution guarantees rest on.
    if db_path.exists() and _has_execution_history(db_path):
        print(
            f"refusing to reset {db_path}: it contains execution ledger history.\n"
            "The demo is destructive. Point it at a scratch database instead:\n"
            f"  python main.py demo --db /tmp/demo.db",
            file=sys.stderr,
        )
        return 1

    for suffix in ("", "-wal", "-shm"):
        stale = db_path.with_name(db_path.name + suffix)
        if stale.exists():
            stale.unlink()
            print(f"  reset      removed previous demonstration database {stale.name}")

    engine = create_db_engine(db_path)
    init_db(engine)
    catalog = CatalogRepository(engine)
    events = EventRepository(engine)

    supplier, product, sku = _demo_catalog(catalog)
    day_one = _demo_offers(catalog, supplier, product)
    offer_a = catalog.offers_for(supplier.id, product.id)[0]
    _demo_transitions(catalog, sku)
    if not _demo_write_paths(catalog, sku, product):
        return 2
    _demo_ledger(events)
    if not _demo_tamper(engine, events, offer_a):
        return 2
    _demo_projections(engine, catalog, sku)
    _demo_replay(engine, supplier, product, day_one)

    _rule("MILESTONE 1 DEMONSTRATION COMPLETE")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecom-autopilot",
        description="Intelligence & Product Engine - Part 1, Milestone 1",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="database path")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "command",
        choices=[
            "init-db",
            "demo",
            "status",
            "verify-ledger",
            "rebuild-projections",
            "analyze",
            "sources",
            "reference",
            "ingest",
            "marketplaces",
            "inventory",
            "portfolio",
            "optimize",
            "reconcile",
            "publish",
            "discover",
            "trends",
            "match",
            "suppliers",
            "opportunities",
            "rank",
            "recommend",
            "credentials",
            "backup",
        ],
    )
    parser.add_argument("--source", help="supplier file to ingest")
    parser.add_argument("--to", help="backup destination path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="describe external actions without performing them (default)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="attempt real external calls (requires credentials and a transport)",
    )
    return parser


def _force_utf8_stdio() -> None:
    """Make stdout and stderr UTF-8 regardless of the console code page.

    Reports quote prices in rupees, and U+20B9 has no cp1252 code point. On a
    Windows console the first report line raised UnicodeEncodeError before any
    of it reached the screen. Reconfiguring the streams fixes the output path
    without touching the domain: the currency stays the currency.

    ``reconfigure`` mutates the existing TextIOWrapper in place, so logging
    handlers already bound to these streams pick up the change too.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    handlers = {
        "init-db": cmd_init_db,
        "demo": cmd_demo,
        "rebuild-projections": cmd_rebuild_projections,
        "verify-ledger": cmd_verify_ledger,
        "status": cmd_status,
        "analyze": cmd_analyze,
        "sources": cmd_sources,
        "reference": cmd_reference,
        "marketplaces": cmd_marketplaces,
        "inventory": cmd_inventory,
        "portfolio": cmd_portfolio,
        "optimize": cmd_optimize,
        "reconcile": cmd_reconcile,
        "trends": cmd_trends,
        "match": cmd_match,
        "suppliers": cmd_suppliers,
        "opportunities": cmd_opportunities,
        "rank": cmd_rank,
        "recommend": cmd_recommend,
        "credentials": cmd_credentials,
    }
    if args.command == "discover":
        return cmd_discover(args.db, args.source)
    if args.command == "backup":
        return cmd_backup(args.db, args.to)
    if args.command == "publish":
        return cmd_publish(args.db, dry_run=args.dry_run)
    if args.command == "ingest":
        return cmd_ingest(args.db, args.source)
    return handlers[args.command](args.db)


# --------------------------------------------------------------------------
# Part 1 analysis commands
# --------------------------------------------------------------------------


def _demo_catalogue() -> list[dict[str, object]]:
    """A small catalogue exercising every outcome the pipeline can produce."""
    from decimal import Decimal as D

    return [
        {
            "name": "Rechargeable Trimmer KX-880",
            "category": "personal_care_electrical",
            "hsn": "8510",
            "weight_g": 310,
            "cost": D("200"),
            "invoice": True,
            "moq": 25,
            "price": D("1699"),
            "market": ["1799", "1899", "2099"],
            "evidence": {
                "bis_licence_number": "CM/L-1234567",
                "manufacturer_name": "KX Appliances",
                "licence_valid_until": "2030-01-01",
            },
        },
        {
            "name": "Kitchen Organiser Rack",
            "category": "personal_care_electrical",
            "hsn": "3924",
            "weight_g": 900,
            "cost": D("350"),
            "invoice": True,
            "moq": 100,
            "price": D("599"),
            "market": ["449", "479", "520"],
            "evidence": {},
        },
        {
            "name": "Retail-sourced Trimmer",
            "category": "personal_care_electrical",
            "hsn": "8510",
            "weight_g": 310,
            "cost": D("399"),
            "invoice": False,
            "moq": 1,
            "price": D("799"),
            "market": ["749", "799", "849"],
            "evidence": {},
        },
    ]


def cmd_analyze(_db_path: Path) -> int:
    """Run the full Part 1 pipeline over a demonstration catalogue."""
    import json
    from datetime import datetime as _dt
    from decimal import Decimal as D

    from catalog.enums import AIService, SignalSource, SignalStatus
    from decisions.pipeline import (
        Pipeline,
        PipelineConfig,
        detail_report,
        ranked_report,
        rejection_report,
    )
    from llm.services import LLMService, RecordedLLMClient
    from reference.loader import ReferenceStore
    from signals.market import Comparable, MarketPriceReference, Signal

    reference_dir = Path("tests/fixtures/reference")
    if not reference_dir.is_dir():
        print("demonstration reference fixtures not found", file=sys.stderr)
        return 1

    store = ReferenceStore(reference_dir).load()
    _rule("REFERENCE DATA VERIFICATION")
    for name, state in store.audit().items():
        print(f"  {state:<12} {name}")

    responses = {
        "JUDGE_IDENTITY": json.dumps({"findings": [], "unresolved_questions": []}),
        "JUDGE_COMMERCIAL": json.dumps({"findings": [], "unresolved_questions": []}),
        "WRITER": json.dumps(
            {
                "title": "Product listing draft",
                "bullets": ["Drafted from verified facts only"],
                "description": "Draft description.",
                "keywords": ["draft"],
                "attributes": {},
                "image_requirements": ["front", "back"],
            }
        ),
    }
    client = RecordedLLMClient(responses)
    pipeline = Pipeline(
        store,
        identity_judge=LLMService(AIService.JUDGE_IDENTITY, client),
        commercial_judge=LLMService(AIService.JUDGE_COMMERCIAL, client),
        writer=LLMService(AIService.WRITER, client),
    )

    signals = {
        name: Signal(
            name,
            SignalStatus.OBSERVED,
            SignalSource.MANUAL,
            "demo",
            value_numeric=D("60"),
        )
        for name in (
            "SEARCH_INTEREST",
            "TREND_DIRECTION",
            "COMPETITOR_COUNT",
            "BRAND_DOMINANCE",
        )
    }

    results = []
    for index, spec in enumerate(_demo_catalogue()):
        product = Product(
            canonical_name=str(spec["name"]),
            category=str(spec["category"]),
            hsn_code=str(spec["hsn"]),
            weight_g=int(spec["weight_g"]),
            identity_confidence=IdentityConfidence.HIGH,
        )
        offer = SupplierOffer(
            supplier_id=new_id(),
            product_id=product.id,
            unit_cost_inr=spec["cost"],
            cost_is_gst_inclusive=True,
            gst_rate_pct=D("18"),
            tax_invoice_available=bool(spec["invoice"]),
            moq_units=int(spec["moq"]),
            inbound_freight_inr=D("12.50"),
        )
        reference = MarketPriceReference.from_comparables(
            [
                Comparable("demo", f"comparable {i}", D(p), f"ref://{i}", _dt.now(UTC))
                for i, p in enumerate(spec["market"])
            ],
            collected_by="demo",
        )
        results.append(
            pipeline.run(
                product=product,
                offers=[offer],
                sku_code=f"SKU-DEMO-{index + 1}",
                config=PipelineConfig(
                    marketplace="test_marketplace",
                    selling_price_inr=spec["price"],
                ),
                signals=signals,
                price_reference=reference,
                compliance_evidence=spec["evidence"],
            )
        )

    _rule("REJECTION REPORT (the primary output)")
    print(rejection_report(results))
    _rule("AWAITING APPROVAL")
    print(ranked_report(results))
    _rule("DETAIL")
    for result in results:
        print(detail_report(result))
    return 0


def cmd_sources(_db_path: Path) -> int:
    """Report which discovery sources have a legitimate access route."""
    from discovery.registry import registry

    print(registry.coverage_report())
    return 0


def cmd_reference(_db_path: Path) -> int:
    """Report the verification state of all bundled reference data."""
    from reference.loader import ReferenceStore

    _rule("REFERENCE DATA")
    for name, state in ReferenceStore("data/reference").load().audit().items():
        print(f"  {state:<12} {name}")
    print(
        "\n  UNVERIFIED entries are shipped as shapes, not figures. The decision\n"
        "  engine refuses to emit an approval for any assessment pinned to one.\n"
    )
    return 0


def cmd_ingest(_db_path: Path, source: str | None = None) -> int:
    """Ingest a supplier CSV/XLSX/PDF and report what was and was not read."""
    from sources.ingest import IngestError, ingest

    if not source:
        print("usage: python main.py ingest --source <file>", file=sys.stderr)
        return 1
    try:
        result = ingest(source)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    _rule(f"INGESTED {result.format.upper()}")
    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key:<16} {value}")
    if result.review_rows:
        print("\n  Rows needing review:")
        for row in result.review_rows[:10]:
            reason = row.errors[0] if row.errors else "field flagged for review"
            print(f"    row {row.row_number}: {reason}")
    return 0


# --------------------------------------------------------------------------
# Part 2 operational commands
# --------------------------------------------------------------------------


def _demo_operations():
    """Build an in-memory operating picture for the demonstration commands."""
    from decimal import Decimal as D

    from part2.inventory import InventoryLedger, Movement, MovementType
    from part2.operations import (
        Order,
        OrderBook,
        OrderLine,
        OrderStatus,
        ReturnEvent,
        ReturnReason,
        ReturnsLedger,
        SettlementLedger,
        SettlementLine,
    )

    inventory = InventoryLedger()
    book, settlements, returns = OrderBook(), SettlementLedger(), ReturnsLedger()
    now = datetime.now(UTC)

    specs = [
        ("SKU-TRIM-880", 40, "300", "1299", "1135", 25, 1),
        ("SKU-ORG-100", 30, "420", "599", "455", 22, 9),
    ]
    for sku, received, cost, price, settled, orders, return_count in specs:
        inventory.record(
            Movement(
                sku, MovementType.RECEIPT, received, f"PO-{sku}", unit_cost_inr=D(cost)
            )
        )
        for i in range(orders):
            order_id = f"{sku}-{i}"
            book.import_order(
                Order(
                    marketplace="amazon_in",
                    marketplace_order_id=order_id,
                    lines=[OrderLine(sku_code=sku, quantity=1, unit_price_inr=D(price))],
                    ordered_at=now - timedelta(days=orders - i),
                    status=OrderStatus.DELIVERED,
                )
            )
            inventory.record(Movement(sku, MovementType.RESERVE, 1, f"R-{order_id}"))
            inventory.record(
                Movement(
                    sku,
                    MovementType.SHIP,
                    1,
                    f"S-{order_id}",
                    unit_cost_inr=D(cost),
                    at=now - timedelta(days=orders - i),
                )
            )
            settlements.ingest(
                SettlementLine(
                    "amazon_in",
                    order_id,
                    D(price),
                    commission_inr=D("104"),
                    shipping_inr=D("60"),
                    net_settled_inr=D(settled),
                    reference=f"ST-{order_id}",
                )
            )
        for i in range(return_count):
            returns.record(
                ReturnEvent(
                    "amazon_in",
                    f"{sku}-{i}",
                    sku,
                    1,
                    ReturnReason.SIZE_FIT,
                    False,
                    return_shipping_inr=D("150"),
                    refund_inr=D(price),
                )
            )
    return inventory, book, settlements, returns, specs


def cmd_marketplaces(_db_path: Path) -> int:
    """Report what each marketplace adapter can legitimately do."""
    from part2.marketplace import capability_matrix

    print(capability_matrix())
    return 0


def cmd_inventory(_db_path: Path) -> int:
    """Show stock positions and replenishment advice."""
    from decimal import Decimal as D

    from part2.inventory import low_stock_report, plan_reorder

    inventory, *_rest, specs = _demo_operations()
    _rule("INVENTORY POSITIONS")
    print(f"  {'SKU':<18} {'on hand':>8} {'reserved':>9} {'available':>10} {'sold':>6}")
    advice = {}
    for sku, _r, cost, *_ in specs:
        position = inventory.position(sku)
        print(
            f"  {sku:<18} {position.on_hand:>8} {position.reserved:>9} "
            f"{position.available:>10} {position.sold_units:>6}"
        )
        advice[sku] = plan_reorder(
            inventory, sku, unit_landed_cost_inr=D(cost), supplier_moq=20
        )
    print()
    print(low_stock_report(inventory, advice))
    return 0


def cmd_portfolio(_db_path: Path) -> int:
    """Rank products by risk-adjusted contribution per unit of capital."""
    from decimal import Decimal as D

    from part2.intelligence import portfolio_report, rank_portfolio
    from part2.operations import compute_realized

    inventory, book, settlements, returns, specs = _demo_operations()
    entries = {}
    for sku, _r, cost, _p, _s, orders, _rc in specs:
        realized = compute_realized(
            sku,
            book=book,
            settlements=settlements,
            returns=returns,
            unit_landed_cost_inr=D(cost),
        )
        entries[sku] = (
            realized,
            returns.rates_for(sku, shipped_units=orders),
            inventory.position(sku).capital_locked_inr,
        )
    print(portfolio_report(rank_portfolio(entries)))
    return 0


def cmd_optimize(_db_path: Path) -> int:
    """Recommend changes, with evidence. Recommends only; never executes."""
    from decimal import Decimal as D

    from part2.intelligence import optimisation_report, rank_portfolio, recommend
    from part2.operations import compute_realized

    inventory, book, settlements, returns, specs = _demo_operations()
    entries = {}
    for sku, _r, cost, _p, _s, orders, _rc in specs:
        entries[sku] = (
            compute_realized(
                sku,
                book=book,
                settlements=settlements,
                returns=returns,
                unit_landed_cost_inr=D(cost),
            ),
            returns.rates_for(sku, shipped_units=orders),
            inventory.position(sku).capital_locked_inr,
        )
    recommendations = []
    for entry in rank_portfolio(entries):
        recommendations.extend(recommend(entry))
    print(optimisation_report(recommendations))
    print("  Recommendations require explicit approval before any execution.\n")
    return 0


def cmd_reconcile(_db_path: Path) -> int:
    """Reconcile settlements against orders."""
    _inventory, book, settlements, _returns, _specs = _demo_operations()
    _rule("SETTLEMENT RECONCILIATION")
    tally: dict[str, int] = {}
    for result in settlements.reconcile_all(book):
        tally[str(result.state)] = tally.get(str(result.state), 0) + 1
    for state, count in sorted(tally.items()):
        print(f"  {state:<20} {count:>4}")
    print("\n  Unmatched money is never absorbed into profit.\n")
    return 0


def cmd_publish(_db_path: Path, dry_run: bool = True) -> int:
    """Demonstrate the publication gate. Dry run by default."""
    from decimal import Decimal as D

    from catalog.enums import ApprovalDecision, ComplianceResult, SKUState
    from part2.marketplace import ListingPayload, get_adapter
    from part2.publisher import ListingPublisher, check_prerequisites

    payload = ListingPayload(
        sku_code="SKU-TRIM-880",
        title="Rechargeable Trimmer KX-880",
        description="Cordless trimmer with adjustable comb.",
        bullets=("Cordless operation", "USB charging"),
        price_inr=D("1299"),
        quantity=15,
        attributes={"brand": "Kemei", "country_of_origin": "India"},
        images=("front.jpg", "back.jpg"),
    )
    _rule("PUBLICATION GATE" + ("  (DRY RUN)" if dry_run else ""))
    for label, kwargs in [
        ("fully approved", {}),
        ("no human approval", {"approval_decision": ApprovalDecision.DEFERRED}),
        ("compliance unresolved", {"compliance": ComplianceResult.REQUIRES_HUMAN_REVIEW}),
        ("price no longer viable", {"current_min_profitable_inr": D("1500")}),
    ]:
        check = check_prerequisites(
            sku_state=kwargs.pop("sku_state", SKUState.LISTING_READY),
            approval_decision=kwargs.pop("approval_decision", ApprovalDecision.APPROVED),
            compliance=kwargs.pop("compliance", ComplianceResult.PASS),
            payload=payload,
            current_min_profitable_inr=kwargs.pop(
                "current_min_profitable_inr", D("1000")
            ),
        )
        publisher = ListingPublisher(get_adapter("amazon_in", environ={}))
        _record, result = publisher.publish(
            payload, check, approver="rehan", dry_run=dry_run
        )
        print(f"  {label:<26} {result.outcome:<20} {result.detail[:60]}")
    print()
    return 0


# --------------------------------------------------------------------------
# Discovery, matching and opportunity ranking
# --------------------------------------------------------------------------

DISCOVERY_FIXTURES = Path("data/fixtures/discovery")
SUPPLIER_FIXTURE = DISCOVERY_FIXTURES / "suppliers.csv"


def _discovery_inputs(source: str | None = None):
    """Collect observations, group them, and load supplier candidates."""
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products

    feeds = build_registry(fixture_dir=DISCOVERY_FIXTURES)
    products, results = collect(feeds, only=[source] if source else None)
    groups = group_products(products)
    candidates = (
        load_supplier_candidates(SUPPLIER_FIXTURE) if SUPPLIER_FIXTURE.is_file() else []
    )
    return products, results, groups, candidates


def _evaluate_all(source: str | None = None):
    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.opportunities import (
        OpportunityPolicy,
        evaluate_group,
        rank_opportunities,
    )

    reference_dir = Path("tests/fixtures/reference")
    if not reference_dir.is_dir():
        print("reference fixtures not found", file=sys.stderr)
        return None, None
    store = ReferenceStore(reference_dir).load()
    pipeline = Pipeline(store)
    _products, _results, groups, candidates = _discovery_inputs(source)
    opportunities = [
        evaluate_group(group, candidates, pipeline=pipeline, policy=OpportunityPolicy())
        for group in groups
    ]
    return rank_opportunities(opportunities), groups


def cmd_discover(_db_path: Path, source: str | None = None) -> int:
    """Collect candidate products from every legitimate feed."""
    from sourcing.feeds import coverage_report

    products, results, groups, _candidates = _discovery_inputs(source)
    print(coverage_report(results))
    _rule(f"CANDIDATES  {len(products)} observations -> {len(groups)} distinct products")
    print(f"  {'Product':<44} {'sources':>8} {'coverage':>9}  missing signals")
    for group in groups:
        canonical = group.canonical
        missing = ", ".join(canonical.missing_signals[:3]) or "none"
        print(
            f"  {canonical.raw_title[:43]:<44} {len(group.members):>8} "
            f"{canonical.coverage_pct:>8.0f}%  {missing[:38]}"
        )
    print()
    return 0


def cmd_trends(_db_path: Path) -> int:
    """Show search-interest evidence, and be clear about its limits."""
    _products, results, _groups, _candidates = _discovery_inputs("google_trends")
    trends = next((r for r in results if r.source == "google_trends"), None)
    _rule("SEARCH INTEREST")
    if trends is None or not trends.available:
        reason = trends.unavailable_reason if trends else "feed not registered"
        print(f"  DATA_UNAVAILABLE  {reason}")
        print()
        return 0
    print(f"  {'Term':<36} {'interest':>9} {'direction':>10}")
    for product in trends.products:
        interest = product.search_interest
        trend = product.trend
        print(
            f"  {product.raw_title[:35]:<36} "
            f"{(interest.value if interest.present else 'n/a')!s:>9} "
            f"{(trend.value if trend.present else 'n/a')!s:>10}"
        )
    print(
        "\n  Interest is category-level. It corroborates demand; it never\n"
        "  establishes it for a specific SKU, and it is never treated as sales.\n"
    )
    return 0


def cmd_match(_db_path: Path) -> int:
    """Show how observations were grouped, and what needs a human."""
    _products, _results, groups, candidates = _discovery_inputs()
    from sourcing.matching import match_suppliers

    _rule("PRODUCT MATCHING")
    for group in groups:
        flag = "REVIEW" if group.requires_review else ""
        print(
            f"  {group.canonical.raw_title[:42]:<44} {len(group.members)} obs  "
            f"{group.confidence!s:<8} {flag}"
        )
        for member in group.members[1:]:
            print(f"      + {member.raw_title[:60]}  [{member.source}]")
        supplier_matches = match_suppliers(group, candidates)
        for supplier_match in supplier_matches[:2]:
            print(
                f"      supplier: {supplier_match.supplier_name[:28]:<30} "
                f"{supplier_match.confidence!s:<9} "
                f"{supplier_match.evidence[0].detail[:34] if supplier_match.evidence else ''}"
            )
        if not supplier_matches:
            print("      supplier: none matched")
    print()
    return 0


def cmd_suppliers(_db_path: Path) -> int:
    """Rank supplier options for each matched product, on true landed cost."""
    from econ.money import format_inr
    from sourcing.matching import match_suppliers, rank_matched_suppliers

    _products, _results, groups, candidates = _discovery_inputs()
    _rule("SUPPLIER OPTIONS")
    for group in groups:
        matches = match_suppliers(group, candidates)
        scores, selected = rank_matched_suppliers(matches)
        if not scores:
            continue
        print(f"  {group.canonical.raw_title[:60]}")
        print(
            f"    {'supplier':<30} {'unit':>10} {'landed':>11} {'MOQ':>5} "
            f"{'invoice':>8} {'score':>7}"
        )
        for score in scores:
            marker = (
                "*" if selected and score.supplier_id == selected.supplier_id else " "
            )
            print(
                f"  {marker} {score.supplier_name[:29]:<30} "
                f"{format_inr(score.unit_cost_inr):>10} "
                f"{format_inr(score.landed_cost_inr):>11} {score.moq_units:>5} "
                f"{'yes' if score.tax_invoice else 'NO':>8} {score.total_score:>7.1f}"
            )
        for note in (selected.notes if selected else [])[:2]:
            print(f"      note: {note[:70]}")
        print()
    print("  * selected on landed cost, not sticker price.\n")
    return 0


def cmd_opportunities(_db_path: Path) -> int:
    """Rank every candidate through the existing economics engine."""
    from sourcing.opportunities import opportunity_report, rejection_report

    opportunities, _groups = _evaluate_all()
    if opportunities is None:
        return 1
    print(opportunity_report(opportunities))
    print(rejection_report(opportunities))
    return 0


def cmd_rank(_db_path: Path) -> int:
    """Ranked opportunities only, without the rejection detail."""
    from sourcing.opportunities import opportunity_report

    opportunities, _groups = _evaluate_all()
    if opportunities is None:
        return 1
    print(opportunity_report(opportunities))
    return 0


def cmd_recommend(_db_path: Path) -> int:
    """The shortlist a human is asked to approve."""
    from sourcing.opportunities import recommendation_report

    opportunities, _groups = _evaluate_all()
    if opportunities is None:
        return 1
    print(recommendation_report(opportunities))
    return 0


def cmd_credentials(_db_path: Path) -> int:
    """Report which marketplace credentials are configured.

    Presence only. No value is ever read into the output, so this is safe to
    paste into a ticket.
    """
    from part2.auth import credential_report
    from part2.marketplace import ADAPTERS, get_adapter

    print(credential_report())
    _rule("LIVE TRANSPORT")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        base = adapter.base_url() or "n/a (export-based adapter)"
        print(f"  {name:<12} {'offline':<9} {base}")
    print(
        "\n  Adapters are offline unless a transport is supplied explicitly.\n"
        "  Credentials alone never enable a network call.\n"
    )
    return 0


def cmd_backup(db_path: Path, destination: str | None = None) -> int:
    """Copy the database to a timestamped file. Never overwrites anything.

    Uses SQLite's own backup API rather than a file copy, so a backup taken
    while the application is running is internally consistent instead of a
    torn snapshot. The live database is only ever read.

    Restore is deliberately manual: stop the application, move the backup over
    the database path yourself, and restart. An automatic restore is one
    mistyped argument away from destroying the ledger it exists to protect.
    """
    import sqlite3
    from datetime import UTC, datetime

    if not db_path.exists():
        print(f"no database at {db_path}; nothing to back up", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = (
        Path(destination)
        if destination
        else db_path.parent / f"{db_path.stem}-{stamp}.db"
    )
    if target.exists():
        print(f"refusing to overwrite an existing file: {target}", file=sys.stderr)
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        backup = sqlite3.connect(target)
        try:
            source.backup(backup)
        finally:
            backup.close()
    finally:
        source.close()

    size_kb = target.stat().st_size / 1024
    _rule("DATABASE BACKUP")
    print(f"  source       {db_path}")
    print(f"  backup       {target}")
    print(f"  size         {size_kb:.1f} KiB")
    print(f"  integrity    {'ok' if _verify_backup(target) else 'FAILED'}")
    print()
    print("  To restore: stop the application, move this file over")
    print(f"  {db_path}, then restart. Restore is manual by design.")
    print()
    return 0


def _verify_backup(path: Path) -> bool:
    """Confirm the copy is a readable SQLite database."""
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            conn.close()
    except sqlite3.Error:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
