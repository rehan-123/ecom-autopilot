"""Service layer: assembles domain state for the API without duplicating logic.

Every function here delegates to an existing engine. Nothing computes a price,
a margin, a verdict or a state transition — those live in ``econ``,
``sourcing``, ``decisions`` and ``part2`` and stay there. This module exists
only to build the objects those engines need and to hold the result.

State is cached in memory for the process lifetime because rebuilding the
discovery and operations picture on every HTTP request would make the
dashboard unusable, and the underlying fixtures do not change under it. The
ledger is read live from SQLite, because that genuinely does change.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from database.connection import DEFAULT_DB_PATH, create_db_engine
from decisions.pipeline import Pipeline
from econ.money import ZERO, money
from ledger.events import EventRepository
from part2.auth import CREDENTIAL_SPECS, credential_report
from part2.execution import ExecutionRecord, ExecutionState, IdempotencyStore
from part2.intelligence import (
    PortfolioEntry,
    rank_portfolio,
    recommend,
)
from part2.inventory import InventoryLedger, ReorderAdvice, plan_reorder
from part2.marketplace import ADAPTERS, get_adapter
from part2.operations import compute_realized
from reference.loader import ReferenceStore
from sourcing.feeds import (
    FeedResult,
    build_registry,
    collect,
    load_supplier_candidates,
)
from sourcing.matching import (
    ProductGroup,
    group_products,
    match_suppliers,
    rank_matched_suppliers,
)
from sourcing.opportunities import (
    Opportunity,
    OpportunityPolicy,
    evaluate_group,
    rank_opportunities,
)

logger = logging.getLogger(__name__)

SERVICE_VERSION = "api-service/1.0.0"

DISCOVERY_FIXTURES = Path("data/fixtures/discovery")
SUPPLIER_FIXTURE = DISCOVERY_FIXTURES / "suppliers.csv"
REFERENCE_FIXTURES = Path("tests/fixtures/reference")


@dataclass
class DomainState:
    """One assembled view of the whole system."""

    groups: list[ProductGroup]
    feeds: list[FeedResult]
    opportunities: list[Opportunity]
    supplier_candidates: list[Any]
    inventory: InventoryLedger
    portfolio: list[PortfolioEntry]
    reorder: dict[str, ReorderAdvice]
    built_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def opportunity_by_key(self, key: str) -> Opportunity | None:
        """Resolve by slug or raw key; the UI routes on the slug."""
        return next((o for o in self.opportunities if key in (o.slug, o.group_key)), None)

    def group_by_key(self, key: str) -> ProductGroup | None:
        return next((g for g in self.groups if key in (g.slug, g.key)), None)


class ServiceError(RuntimeError):
    """Raised when the domain cannot be assembled at all."""


class DomainService:
    """Builds and caches domain state; the API's only door into the engines."""

    def __init__(
        self,
        *,
        fixture_dir: Path = DISCOVERY_FIXTURES,
        reference_dir: Path = REFERENCE_FIXTURES,
        supplier_file: Path = SUPPLIER_FIXTURE,
        db_path: Path = DEFAULT_DB_PATH,
    ) -> None:
        self.fixture_dir = fixture_dir
        self.reference_dir = reference_dir
        self.supplier_file = supplier_file
        self.db_path = db_path
        self._state: DomainState | None = None
        self._lock = threading.Lock()
        #: Execution records live for the process lifetime. They are proposals
        #: awaiting approval, not durable business facts; anything that
        #: actually happens is written to the ledger by the engines.
        self.executions: dict[str, ExecutionRecord] = {}
        self.idempotency = IdempotencyStore()

    # -- assembly ---------------------------------------------------------

    def state(self, *, refresh: bool = False) -> DomainState:
        with self._lock:
            if self._state is None or refresh:
                self._state = self._build()
            return self._state

    def _build(self) -> DomainState:
        if not self.reference_dir.is_dir():
            raise ServiceError(
                f"reference data not found at {self.reference_dir}; the API "
                "cannot price anything without it"
            )
        store = ReferenceStore(self.reference_dir).load()
        pipeline = Pipeline(store)

        products, feeds = collect(build_registry(fixture_dir=self.fixture_dir))
        groups = group_products(products)
        candidates = (
            load_supplier_candidates(self.supplier_file)
            if self.supplier_file.is_file()
            else []
        )

        opportunities = rank_opportunities(
            [
                evaluate_group(
                    group, candidates, pipeline=pipeline, policy=OpportunityPolicy()
                )
                for group in groups
            ]
        )

        inventory, portfolio, reorder = self._operations()

        logger.info(
            "domain state assembled",
            extra={
                "operation": "build_state",
                "result": f"{len(groups)} groups, {len(opportunities)} opportunities",
            },
        )
        return DomainState(
            groups=groups,
            feeds=feeds,
            opportunities=opportunities,
            supplier_candidates=candidates,
            inventory=inventory,
            portfolio=portfolio,
            reorder=reorder,
        )

    def _operations(
        self,
    ) -> tuple[InventoryLedger, list[PortfolioEntry], dict[str, ReorderAdvice]]:
        """Build the operating picture using the CLI's own demonstration data.

        Reuses ``main._demo_operations`` rather than reimplementing it, so the
        API and the CLI show the same numbers. When a real order feed is
        connected this is the single function that changes.
        """
        from main import _demo_operations  # noqa: PLC0415 - avoids a cycle at import

        inventory, book, settlements, returns, specs = _demo_operations()

        entries: dict[str, tuple[Any, Any, Decimal]] = {}
        reorder: dict[str, ReorderAdvice] = {}
        for sku, _received, cost, _price, _settled, orders, _returns in specs:
            realized = compute_realized(
                sku,
                book=book,
                settlements=settlements,
                returns=returns,
                unit_landed_cost_inr=money(cost),
            )
            position = inventory.position(sku)
            entries[sku] = (
                realized,
                returns.rates_for(sku, shipped_units=orders),
                position.capital_locked_inr,
            )
            reorder[sku] = plan_reorder(
                inventory, sku, unit_landed_cost_inr=money(cost), supplier_moq=20
            )

        self._book = book
        self._settlements = settlements
        self._returns = returns
        return inventory, rank_portfolio(entries), reorder

    # -- accessors --------------------------------------------------------

    @property
    def order_book(self) -> Any:
        self.state()
        return self._book

    @property
    def settlements(self) -> Any:
        self.state()
        return self._settlements

    @property
    def returns(self) -> Any:
        self.state()
        return self._returns

    def supplier_options(self, group_key: str) -> tuple[list[Any], Any]:
        """Ranked supplier options for one product, via the Part 2 engine."""
        state = self.state()
        group = state.group_by_key(group_key)
        if group is None:
            return [], None
        matches = match_suppliers(group, state.supplier_candidates)
        return rank_matched_suppliers(matches)

    def ledger_events(
        self, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[Any], int, dict]:
        """Read the append-only ledger, plus its integrity status.

        Read live rather than cached: the ledger is the one thing in this
        system that genuinely changes underneath the API.
        """
        if not self.db_path.exists():
            return (
                [],
                0,
                {"valid": None, "reason": "no database; run `python main.py init-db`"},
            )
        engine = create_db_engine(self.db_path)
        repository = EventRepository(engine)
        total = repository.count()
        events = repository.get_events()
        window = list(reversed(events))[offset : offset + limit]
        return window, total, repository.verify_chain()

    def settings_snapshot(self) -> dict[str, Any]:
        """Configuration and capability, with nothing secret in it."""
        store = ReferenceStore(self.reference_dir).load()
        adapters = []
        for name in sorted(ADAPTERS):
            adapter = get_adapter(name)
            adapters.append(
                {
                    "name": name,
                    "access": str(adapter.profile.access),
                    "capabilities": sorted(str(c) for c in adapter.profile.capabilities),
                    "credentials_configured": adapter.profile.credentials_present(),
                    "required_env_vars": list(adapter.profile.credential_env_vars),
                    "live_transport": adapter.live,
                    "notes": adapter.profile.notes,
                }
            )
        return {
            "mode": "PAPER",
            "dry_run_default": True,
            "execution_requires_approval": True,
            "capabilities_absent": [
                "purchasing",
                "payment processing",
                "money transfer",
                "marketplace publishing",
                "automated fulfilment",
            ],
            "reference_data": {k: str(v) for k, v in store.audit().items()},
            "marketplaces": adapters,
            "credentials": {
                name: {
                    "required": list(spec.required),
                    "missing": spec.missing(),
                }
                for name, spec in sorted(CREDENTIAL_SPECS.items())
            },
            "credential_report": credential_report(),
        }

    # -- execution --------------------------------------------------------

    def propose(
        self,
        *,
        action: str,
        sku_code: str,
        marketplace: str,
        proposed: dict[str, Any],
        reason: str,
    ) -> ExecutionRecord:
        """Record a proposed action in RECOMMENDED state.

        Proposing is not doing. The record cannot reach a marketplace until a
        human approves it, and the state machine in ``part2.execution``
        enforces that rather than this layer.
        """
        record = ExecutionRecord(
            action=action,
            sku_code=sku_code,
            marketplace=marketplace,
            proposed=proposed,
            reason=reason,
        )
        self.executions[str(record.id)] = record
        return record

    def approve(self, record_id: str, approver: str) -> ExecutionRecord:
        record = self.executions.get(record_id)
        if record is None:
            raise KeyError(record_id)
        record.approve(approver)
        return record

    def pending_approvals(self) -> list[ExecutionRecord]:
        return [
            record
            for record in self.executions.values()
            if record.state
            in (ExecutionState.RECOMMENDED, ExecutionState.REQUIRES_REVIEW)
        ]

    def seed_proposals(self) -> None:
        """Turn current recommendations into proposals awaiting approval.

        Uses the optimisation engine's own recommendations rather than
        inventing actions, so the approval queue shows exactly what the
        analytics screen justifies.
        """
        if self.executions:
            return
        for entry in self.state().portfolio:
            for suggestion in recommend(entry):
                self.propose(
                    action=str(suggestion.action),
                    sku_code=suggestion.sku_code,
                    marketplace="amazon_in",
                    proposed={"change": suggestion.proposed_change},
                    reason=suggestion.reason,
                )

    # -- dashboard --------------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        """Headline numbers, every one of them derived from an engine."""
        state = self.state()
        verdicts: dict[str, int] = {}
        for opportunity in state.opportunities:
            key = str(opportunity.verdict)
            verdicts[key] = verdicts.get(key, 0) + 1

        realized_total = ZERO
        capital_total = ZERO
        orders_total = 0
        for entry in state.portfolio:
            realized_total += entry.realized.actual_contribution_inr
            capital_total += entry.capital_locked_inr
            orders_total += entry.realized.sample_size

        _events, event_count, integrity = self.ledger_events(limit=1)
        available_feeds = [f for f in state.feeds if f.available]

        return {
            "mode": "PAPER",
            "built_at": state.built_at.isoformat(),
            "discovery": {
                "observations": sum(len(g.members) for g in state.groups),
                "distinct_products": len(state.groups),
                "feeds_available": len(available_feeds),
                "feeds_total": len(state.feeds),
            },
            "opportunities": {"by_verdict": verdicts, "total": len(state.opportunities)},
            "operations": {
                "orders": orders_total,
                "realized_contribution_inr": str(realized_total),
                "capital_locked_inr": str(capital_total),
                "products": len(state.portfolio),
                "low_stock": sum(1 for a in state.reorder.values() if a.should_reorder),
            },
            "approvals_pending": len(self.pending_approvals()),
            "ledger": {"events": event_count, "chain_valid": integrity.get("valid")},
        }


#: Process-wide service. The API constructs one and shares it.
_service: DomainService | None = None
_service_lock = threading.Lock()


def get_service() -> DomainService:
    global _service  # noqa: PLW0603 - one shared instance is the point
    with _service_lock:
        if _service is None:
            _service = DomainService()
        return _service


def reset_service(service: DomainService | None = None) -> None:
    """Replace the shared service. Used by tests to inject fixtures."""
    global _service  # noqa: PLW0603
    with _service_lock:
        _service = service


__all__ = [
    "SERVICE_VERSION",
    "DomainService",
    "DomainState",
    "ServiceError",
    "get_service",
    "reset_service",
]
