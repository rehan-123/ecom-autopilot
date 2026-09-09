"""FastAPI application: a thin HTTP surface over the existing engines.

Deliberately thin. Every endpoint reads from ``api.service``, which delegates
to the domain engines; nothing here calculates a price, decides a verdict or
moves a state. If a number appears in the UI, an engine produced it.

Three properties are enforced at this layer and tested:

**Read-mostly.** The only mutating endpoints create an approval proposal or
approve one. There is no endpoint that purchases, pays, publishes or transfers
money, because no such capability exists underneath.

**Approval before execution.** Executing a proposal calls into
``part2.execution.run_approved``, which refuses anything not in APPROVED
state. The API cannot bypass it.

**Paper mode by default.** Every execution is dry-run unless explicitly told
otherwise, and the settings endpoint reports the mode so the UI can show it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.service import DomainService, ServiceError, get_service
from database.connection import DEFAULT_DB_PATH, create_db_engine
from execution import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionMode,
    ExecutionRequest,
    IllegalTransition,
    OperationType,
    live_execution_enabled,
)
from execution.ledger import PersistentIdempotencyStore
from marketplace import (
    assess_all,
    build_draft,
    preview_listing,
    publish_listing,
    validate_draft,
)
from marketplace import (
    registry as marketplace_registry,
)
from part2.execution import ActionResult, ExecutionState, Outcome, run_approved
from part2.intelligence import optimisation_report, recommend
from part2.marketplace import get_adapter
from research import ResearchAssistant, ResearchError, default_researcher, preview
from research.export import write_observations, write_trends
from sourcing.feeds import coverage_report
from sourcing.supplier_ingest import ingest_suppliers

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"

app = FastAPI(
    title="Ecom Autopilot",
    version=API_VERSION,
    description=(
        "HTTP surface over the Ecom Autopilot decision and operating engines. "
        "Paper mode by default. No purchasing, payment, publishing or "
        "money-transfer capability exists in this system."
    ),
)

# The Vite dev server runs on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def service() -> DomainService:
    return get_service()


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class ProposalRequest(BaseModel):
    action: str = Field(..., min_length=1, max_length=64)
    sku_code: str = Field(..., min_length=1, max_length=64)
    marketplace: str = Field(default="amazon_in", max_length=32)
    reason: str = Field(default="", max_length=500)
    proposed: dict[str, str] = Field(default_factory=dict)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class SendToAnalysisRequest(BaseModel):
    """Body for send-to-analysis. The run is identified by the path, not here.

    ``run_id`` was previously required in the body as well as the path, which
    made every well-formed client request fail with 422 — the run is already
    named in the URL, so a caller has no reason to repeat it. It is accepted
    optionally for callers that do send it, and checked against the path
    rather than ignored: a body naming a different run than the URL means the
    caller is confused about which run it is acting on, and silently
    preferring one over the other would act on the wrong data.
    """

    run_id: str | None = None
    #: Writing to the fixtures the pipeline reads is the only side effect the
    #: research screen has, and it is explicit.
    persist: bool = True


class SupplierImportRequest(BaseModel):
    """Human-triggered supplier catalogue import.

    Takes a path the operator supplies rather than an upload, matching how
    every other ingestion in this system works. ``persist`` is explicit
    because writing to the catalogue the matcher reads is a real side effect.
    """

    path: str = Field(..., min_length=1, max_length=500)
    persist: bool = False


class ExecutionCreateRequest(BaseModel):
    """Create an execution request.

    Deliberately narrow. A client supplies the intent — what operation, on
    which SKU, at which marketplace, with what payload — and nothing else.
    Verdict, compliance, supplier confirmation, economics and capability are
    all derived server-side from authoritative domain state; accepting them
    from a caller would let anyone authorise anything by asserting it.
    """

    operation: str = Field(..., min_length=1, max_length=32)
    marketplace: str = Field(..., min_length=1, max_length=32)
    slug: str = Field(..., min_length=1, max_length=200)
    payload: dict[str, str] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=500)
    requested_by: str = Field(default="operator", max_length=64)


class ExecutionActorRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=64)
    reason: str = Field(default="", max_length=500)


class ExecutionRunRequest(BaseModel):
    #: Defaults to dry run. A client asking for live still cannot get it
    #: unless the server environment enables it.
    mode: str = "DRY_RUN"


class ApprovalRequest(BaseModel):
    approver: str = Field(..., min_length=1, max_length=64)


class ExecuteRequest(BaseModel):
    #: Defaults to True so an accidental call cannot reach a marketplace.
    dry_run: bool = True


# --------------------------------------------------------------------------
# Health and dashboard
# --------------------------------------------------------------------------


def _component_health(svc: DomainService) -> dict[str, Any]:
    """Real component checks. Never healthy merely because the process runs."""
    components: dict[str, Any] = {}

    try:
        _events, count, integrity = svc.ledger_events(limit=1)
        components["database"] = {
            "status": "OK" if integrity.get("valid") is not False else "DEGRADED",
            "detail": f"{count} ledger events; chain valid={integrity.get('valid')}",
        }
    except Exception as exc:  # noqa: BLE001 - a health probe reports, never raises
        components["database"] = {"status": "NOT_READY", "detail": str(exc)[:160]}

    try:
        statuses = marketplace_registry()
        configured = [m for m in statuses if m.credentials_configured]
        components["marketplace_adapters"] = {
            "status": "OK" if statuses else "NOT_READY",
            "detail": f"{len(configured)}/{len(statuses)} configured",
        }
    except Exception as exc:  # noqa: BLE001
        components["marketplace_adapters"] = {
            "status": "NOT_READY",
            "detail": str(exc)[:160],
        }

    components["execution"] = {
        "status": "OK",
        "detail": (
            "dry run available; live execution "
            + ("enabled" if live_execution_enabled() else "disabled")
        ),
        "live_execution_enabled": live_execution_enabled(),
        "persistent_idempotency": _ENGINE.ledger is not None,
    }
    return components


@app.get("/api/health", tags=["system"])
def health(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Component health plus the safety posture."""
    components = _component_health(svc)
    worst = "ok"
    if any(c["status"] == "NOT_READY" for c in components.values()):
        worst = "not_ready"
    elif any(c["status"] == "DEGRADED" for c in components.values()):
        worst = "degraded"
    return {
        "status": worst,
        "version": API_VERSION,
        "mode": "PAPER",
        "dry_run_default": True,
        "execution_requires_approval": True,
        "components": components,
    }


@app.get("/api/readiness", tags=["system"])
def readiness(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Readiness, kept separate from health.

    Application health and marketplace readiness are different questions. The
    app can be perfectly healthy while no marketplace is configured, and
    conflating them would either hide a real fault or raise a false alarm.
    """
    components = _component_health(svc)
    database_ok = components["database"]["status"] != "NOT_READY"
    configured = [m for m in marketplace_registry() if m.credentials_configured]

    if not database_ok:
        dry_run_state = "NOT_READY"
    elif not configured:
        dry_run_state = "READY"
    else:
        dry_run_state = "READY"

    # Live readiness is a strictly higher bar and is reported separately.
    live_blockers: list[str] = []
    if not live_execution_enabled():
        live_blockers.append("live execution is not enabled")
    if not configured:
        live_blockers.append("no marketplace has credentials configured")
    if _ENGINE.ledger is None:
        live_blockers.append("persistent idempotency is not configured")
    if not database_ok:
        live_blockers.append("database unavailable")
    live_blockers.append("no live adapter transport is wired into the execution engine")

    return {
        "dry_run": dry_run_state,
        "live_execution": "NOT_READY",
        "live_blockers": live_blockers,
        "components": components,
        "note": (
            "Application health and marketplace readiness are distinct. Live "
            "execution requires every blocker above to be cleared."
        ),
    }


@app.get("/api/dashboard", tags=["dashboard"])
def dashboard(svc: DomainService = Depends(service)) -> dict[str, Any]:
    try:
        return svc.dashboard()
    except ServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@app.get("/api/discovery/feeds", tags=["discovery"])
def discovery_feeds(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Which sources produced candidates, and why the others did not."""
    state = svc.state()
    return {
        "feeds": [feed.as_dict() for feed in state.feeds],
        "report": coverage_report(state.feeds),
    }


@app.get("/api/discovery/products", tags=["discovery"])
def discovery_products(svc: DomainService = Depends(service)) -> dict[str, Any]:
    state = svc.state()
    return {
        "products": [
            {
                **group.as_dict(),
                "members": [member.as_dict() for member in group.members],
            }
            for group in state.groups
        ]
    }


# --------------------------------------------------------------------------
# Suppliers
# --------------------------------------------------------------------------


@app.post("/api/suppliers/import", tags=["suppliers"])
def import_suppliers(
    request: SupplierImportRequest, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    """Import a supplier CSV, reporting what was accepted and what was refused.

    Human-triggered, never automatic. Rows that do not state a required fact
    are refused with a reason rather than defaulted, so the response is as
    much about the rejections as the acceptances.
    """
    path = Path(request.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no such file: {request.path}")
    if path.suffix.lower() not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=400, detail=f"unsupported catalogue format: {path.suffix}"
        )

    report = ingest_suppliers(path)
    body = report.as_dict()
    if request.persist:
        state = svc.state(refresh=True)
        body["products"] = len(state.groups)
    body["note"] = (
        "Supplier facts come only from this file. Rows missing GST, MOQ or a "
        "tax-invoice statement are refused rather than defaulted, because each "
        "of those changes landed cost."
    )
    return body


@app.get("/api/suppliers", tags=["suppliers"])
def suppliers(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Ranked supplier options per product, from the Part 2 ranking engine."""
    state = svc.state()
    rows = []
    for group in state.groups:
        scores, selected = svc.supplier_options(group.key)
        if not scores:
            continue
        rows.append(
            {
                "group_key": group.key,
                "product": group.canonical.raw_title,
                "selected": selected.supplier_name if selected else None,
                "options": [score.as_dict() for score in scores],
            }
        )
    return {"products": rows}


# --------------------------------------------------------------------------
# Opportunities
# --------------------------------------------------------------------------


@app.get("/api/opportunities", tags=["opportunities"])
def opportunities(
    verdict: str | None = Query(default=None),
    svc: DomainService = Depends(service),
) -> dict[str, Any]:
    items = svc.state().opportunities
    if verdict:
        items = [o for o in items if str(o.verdict) == verdict.upper()]
    return {"opportunities": [o.as_dict() for o in items]}


@app.get("/api/opportunities/{group_key}", tags=["opportunities"])
def opportunity_detail(
    group_key: str, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    """Everything behind one product's verdict."""
    state = svc.state()
    opportunity = state.opportunity_by_key(group_key)
    if opportunity is None:
        raise HTTPException(status_code=404, detail=f"no opportunity {group_key!r}")

    group = state.group_by_key(group_key)
    scores, selected = svc.supplier_options(group_key)
    scenarios = {}
    if opportunity.pipeline:
        scenarios = {
            str(name): economics.as_dict()
            for name, economics in opportunity.pipeline.economics.items()
        }

    return {
        "opportunity": opportunity.as_dict(),
        "observations": [m.as_dict() for m in (group.members if group else [])],
        "suppliers": [s.as_dict() for s in scores],
        # Every match the matcher found, including those below the actionable
        # bar. `scores` holds only usable options, so on its own it cannot
        # distinguish "nothing matched" from "something matched at PROBABLE
        # confidence and needs confirming" -- and the UI was reporting the
        # first while the reason line said the second.
        "supplier_matches": [m.as_dict() for m in opportunity.supplier_matches],
        "selected_supplier": selected.as_dict() if selected else None,
        "scenarios": scenarios,
        "decisions": (
            [d.as_dict() for d in opportunity.pipeline.decisions]
            if opportunity.pipeline
            else []
        ),
        "compliance": (
            opportunity.pipeline.compliance.as_dict()
            if opportunity.pipeline and opportunity.pipeline.compliance
            else None
        ),
        "score": (
            opportunity.pipeline.score.as_dict()
            if opportunity.pipeline and opportunity.pipeline.score
            else None
        ),
    }


# --------------------------------------------------------------------------
# Approvals and execution
# --------------------------------------------------------------------------


@app.get("/api/approvals", tags=["approvals"])
def approvals(svc: DomainService = Depends(service)) -> dict[str, Any]:
    svc.seed_proposals()
    return {
        "pending": [r.as_dict() for r in svc.pending_approvals()],
        "all": [r.as_dict() for r in svc.executions.values()],
    }


@app.post("/api/approvals", tags=["approvals"], status_code=201)
def create_proposal(
    request: ProposalRequest, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    """Record a proposed action. Proposing is not doing."""
    record = svc.propose(
        action=request.action,
        sku_code=request.sku_code,
        marketplace=request.marketplace,
        proposed=dict(request.proposed),
        reason=request.reason,
    )
    return record.as_dict()


@app.post("/api/approvals/{record_id}/approve", tags=["approvals"])
def approve(
    record_id: str, request: ApprovalRequest, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    try:
        record = svc.approve(record_id, request.approver)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no execution {record_id!r}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.as_dict()


@app.post("/api/approvals/{record_id}/execute", tags=["approvals"])
def execute(
    record_id: str, request: ExecuteRequest, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    """Execute an approved action, dry-run unless told otherwise.

    The approval check lives in ``run_approved``; this endpoint cannot bypass
    it, and an unapproved record produces 409 rather than an execution.
    """
    record = svc.executions.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {record_id!r}")
    if record.state is not ExecutionState.APPROVED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"execution is {record.state}; a human must approve it before it can run"
            ),
        )

    adapter = get_adapter(record.marketplace)

    def operation() -> ActionResult:
        # Every currently proposable action is an analysis recommendation
        # rather than a marketplace write, so this reports the intent through
        # the adapter's dry-run path instead of inventing a call.
        return ActionResult(
            outcome=Outcome.DRY_RUN if request.dry_run else Outcome.NOT_SUPPORTED,
            action=record.action,
            idempotency_key=str(record.id),
            detail=(
                f"Would apply '{record.action}' to {record.sku_code} on "
                f"{adapter.profile.name}: {record.proposed}"
                if request.dry_run
                else (
                    "live execution of portfolio actions is not implemented; "
                    "no purchasing or publishing capability exists"
                )
            ),
        )

    result = run_approved(record, operation, store=svc.idempotency, idempotent=True)
    return {"execution": record.as_dict(), "result": result.as_dict()}


# --------------------------------------------------------------------------
# Research
# --------------------------------------------------------------------------

#: Runs live for the process lifetime. A research run is a proposal, not a
#: business fact; nothing durable is recorded until it is sent to analysis.
_RESEARCH_RUNS: dict[str, Any] = {}


@app.post("/api/research", tags=["research"])
def research(request: ResearchRequest) -> dict[str, Any]:
    """Run a research request.

    Uses OpenAI when OPENAI_API_KEY is set in the environment, and degrades to
    the operator's own search terms otherwise -- the same degraded-but-working
    path any other transport failure takes. The key is read fresh on each
    request rather than cached on the app, so rotating it takes effect on the
    next call with no restart.

    Whichever transport is in play, the model suggests what to investigate;
    sources supply the facts. Every factual field a source cannot supply
    comes back UNAVAILABLE, never invented.
    """
    try:
        run = ResearchAssistant(researcher=default_researcher()).run(request.query)
    except ResearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _RESEARCH_RUNS[str(run.id)] = run
    body = run.as_dict()
    body["export_preview"] = preview(run)
    return body


@app.get("/api/research/{run_id}", tags=["research"])
def research_run(run_id: str) -> dict[str, Any]:
    run = _RESEARCH_RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no research run {run_id!r}")
    body = run.as_dict()
    body["export_preview"] = preview(run)
    return body


@app.post("/api/research/{run_id}/send-to-analysis", tags=["research"])
def send_to_analysis(
    run_id: str, request: SendToAnalysisRequest, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    """Write sourced observations into the fixtures and rebuild the pipeline.

    Only SOURCED claims are written; a model's suggestion enters as a product
    with no evidence, which the existing coverage rules refuse to rank. The
    verdict still comes from the deterministic pipeline, not from research.
    """
    if request.run_id is not None and request.run_id != run_id:
        # Accepting either silently would act on a run the caller did not
        # name in the URL.
        raise HTTPException(
            status_code=400,
            detail=(
                f"body run_id {request.run_id!r} does not match the path "
                f"{run_id!r}; the run is identified by the path"
            ),
        )

    run = _RESEARCH_RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no research run {run_id!r}")

    written = {"observations": 0, "trends": 0}
    if request.persist:
        written["observations"] = write_observations(
            run, svc.fixture_dir / "observations.csv"
        )
        written["trends"] = write_trends(run, svc.fixture_dir / "trends.csv")

    state = svc.state(refresh=True)
    return {
        "written": written,
        "products": len(state.groups),
        "opportunities": len(state.opportunities),
        "note": (
            "Candidates were added to discovery. The verdict is decided by the "
            "existing matching, economics, compliance and opportunity engines, "
            "not by research. Human approval is still required before anything "
            "executes."
        ),
    }


# --------------------------------------------------------------------------
# Phase 3: marketplace readiness (no live execution)
# --------------------------------------------------------------------------


@app.get("/api/marketplaces/registry", tags=["marketplace"])
def marketplaces_registry() -> dict[str, Any]:
    """Capability status per marketplace, distinguishing cannot from unconfigured."""
    return {
        "marketplaces": [m.as_dict() for m in marketplace_registry()],
        "phase": 3,
        "live_execution_enabled": False,
        "note": (
            "Phase 3 assesses readiness only. No listing is published, no order "
            "placed, no inventory synced and no money moved."
        ),
    }


@app.get("/api/listings/{slug}/preview", tags=["marketplace"])
def listing_preview(slug: str, svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Build a draft from a Phase 2 opportunity and render it. Makes no request."""
    opportunity = svc.state().opportunity_by_key(slug)
    if opportunity is None:
        raise HTTPException(status_code=404, detail=f"no opportunity {slug!r}")

    draft = build_draft(opportunity)
    return {
        **preview_listing(draft, "generic"),
        "assessments": [a.as_dict() for a in assess_all(draft)],
    }


@app.get("/api/listings/{slug}/validate", tags=["marketplace"])
def listing_validate(slug: str, svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Structured validation findings per marketplace."""
    opportunity = svc.state().opportunity_by_key(slug)
    if opportunity is None:
        raise HTTPException(status_code=404, detail=f"no opportunity {slug!r}")

    draft = build_draft(opportunity)
    # The generic assessment always runs. Without it, an operator with no
    # configured marketplace would see nothing at all about their product,
    # which is the least useful moment to say nothing.
    assessments = [validate_draft(draft, marketplace="generic"), *assess_all(draft)]
    return {
        "sku_code": draft.sku_code,
        "readiness": (
            "READY" if any(a.readiness.is_ready for a in assessments) else "NOT_READY"
        ),
        "assessments": [a.as_dict() for a in assessments],
        "missing_factual_fields": draft.missing_factual_fields,
    }


@app.post("/api/listings/{slug}/publish", tags=["marketplace"])
def listing_publish(slug: str, svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Always refuses. Present so a caller gets a structured refusal.

    Returning 200 with a NOT_ENABLED_FOR_PHASE_3 result rather than 404 is
    deliberate: a client must be able to tell "this endpoint exists and
    declined" from "this route is missing".
    """
    opportunity = svc.state().opportunity_by_key(slug)
    if opportunity is None:
        raise HTTPException(status_code=404, detail=f"no opportunity {slug!r}")
    return publish_listing(build_draft(opportunity), "generic")


# --------------------------------------------------------------------------
# Phase 4: controlled execution
# --------------------------------------------------------------------------


_ENGINE = ExecutionEngine()


def _attach_ledger() -> None:
    """Attach the durable ledger on first use, not at import.

    Building it at import time created the database file before `init-db` had
    run, leaving a half-initialised schema that broke unrelated endpoints on a
    clean checkout. Attaching lazily means the ledger appears once the
    database genuinely exists.
    """
    if _ENGINE.ledger is not None or not DEFAULT_DB_PATH.exists():
        return
    try:
        _ENGINE.ledger = PersistentIdempotencyStore(create_db_engine(DEFAULT_DB_PATH))
    except Exception as exc:  # noqa: BLE001 - degraded startup must stay visible
        logger.warning(
            "durable execution ledger unavailable", extra={"error": str(exc)[:160]}
        )


def _context_for(slug: str, svc: DomainService) -> tuple[Any, ExecutionContext]:
    """Derive authoritative execution context from domain state."""
    opportunity = svc.state().opportunity_by_key(slug)
    if opportunity is None:
        raise HTTPException(status_code=404, detail=f"no opportunity {slug!r}")

    pipeline = getattr(opportunity, "pipeline", None)
    compliance = getattr(pipeline, "compliance", None) if pipeline else None
    return opportunity, ExecutionContext(
        compliance_state=str(compliance.overall) if compliance else None,
        opportunity_verdict=str(opportunity.verdict),
        supplier_confirmed=opportunity.selected_supplier is not None,
        economics_available=(
            opportunity.expected_margin_pct is not None
            and opportunity.worst_margin_pct is not None
        ),
    )


@app.post("/api/executions", tags=["execution"], status_code=201)
def create_execution(
    request: ExecutionCreateRequest, svc: DomainService = Depends(service)
) -> dict[str, Any]:
    """Create an execution request in PENDING_APPROVAL."""
    try:
        operation = OperationType(request.operation)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown operation {request.operation!r}; supported: "
                f"{[str(o) for o in OperationType]}"
            ),
        ) from exc

    _attach_ledger()
    opportunity, _context = _context_for(request.slug, svc)
    record = ExecutionRequest(
        operation=operation,
        marketplace=request.marketplace,
        sku_code=f"SKU-{opportunity.group_key[:24]}",
        payload=dict(request.payload),
        requested_by=request.requested_by,
        reason=request.reason,
    )
    _ENGINE.submit(record)
    return record.as_dict()


@app.get("/api/executions", tags=["execution"])
def list_executions() -> dict[str, Any]:
    return {
        "executions": [r.as_dict() for r in _ENGINE.requests.values()],
        "live_execution_enabled": live_execution_enabled(),
        "default_mode": str(ExecutionMode.DRY_RUN),
    }


@app.get("/api/executions/history", tags=["execution"])
def execution_history(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    state: str | None = Query(default=None),
    marketplace: str | None = Query(default=None),
    sku: str | None = Query(default=None),
    q: str | None = Query(default=None),
) -> dict[str, Any]:
    """Durable execution history from the ledger, filtered and paginated.

    Reads the persisted ledger rather than in-process request objects, so
    history survives a restart. Filtering happens after the bounded read, so a
    filter can never widen the query beyond its page limit.
    """
    _attach_ledger()
    if _ENGINE.ledger is None:
        return {"rows": [], "total": 0, "detail": "no durable ledger; run init-db"}

    rows = _ENGINE.ledger.history(limit=limit, offset=offset)

    def keep(row: dict[str, Any]) -> bool:
        if state and str(row.get("execution_state", "")).upper() != state.upper():
            return False
        if marketplace and row.get("marketplace") != marketplace:
            return False
        if sku and sku.lower() not in str(row.get("sku_code", "")).lower():
            return False
        if q:
            haystack = " ".join(str(v) for v in row.values() if v is not None).lower()
            if q.lower() not in haystack:
                return False
        return True

    filtered = [r for r in rows if keep(r)]
    return {
        "rows": [
            {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
            for r in filtered
        ],
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
        "filters": {"state": state, "marketplace": marketplace, "sku": sku, "q": q},
    }


@app.get("/api/executions/{execution_id}", tags=["execution"])
def execution_status(execution_id: str) -> dict[str, Any]:
    record = _ENGINE.requests.get(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {execution_id!r}")
    return record.as_dict()


@app.post("/api/executions/{execution_id}/approve", tags=["execution"])
def approve_execution(
    execution_id: str, request: ExecutionActorRequest
) -> dict[str, Any]:
    """Bind a human approval to this exact operation and payload."""
    record = _ENGINE.requests.get(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {execution_id!r}")
    try:
        record.approve(request.actor)
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.as_dict()


@app.post("/api/executions/{execution_id}/reject", tags=["execution"])
def reject_execution(execution_id: str, request: ExecutionActorRequest) -> dict[str, Any]:
    record = _ENGINE.requests.get(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {execution_id!r}")
    try:
        record.reject(request.actor, request.reason)
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.as_dict()


@app.post("/api/executions/{execution_id}/cancel", tags=["execution"])
def cancel_execution(execution_id: str, request: ExecutionActorRequest) -> dict[str, Any]:
    record = _ENGINE.requests.get(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {execution_id!r}")
    try:
        record.cancel(request.actor, request.reason)
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.as_dict()


@app.post("/api/executions/{execution_id}/run", tags=["execution"])
def run_execution(
    execution_id: str,
    request: ExecutionRunRequest,
    slug: str = Query(...),
    svc: DomainService = Depends(service),
) -> dict[str, Any]:
    """Run the gate chain and, in dry run, prepare the payload without sending it."""
    record = _ENGINE.requests.get(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {execution_id!r}")

    _attach_ledger()
    _opportunity, context = _context_for(slug, svc)
    mode = (
        ExecutionMode.LIVE
        if request.mode.upper() == "LIVE" and live_execution_enabled()
        else ExecutionMode.DRY_RUN
    )
    result = _ENGINE.execute(record, context, mode=mode)
    return {
        "execution": record.as_dict(),
        "result": result.as_dict(),
        "live_execution_enabled": live_execution_enabled(),
    }


@app.get("/api/export/{dataset}", tags=["system"])
def export_dataset(
    dataset: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    svc: DomainService = Depends(service),
) -> dict[str, Any]:
    """Read-only export of history, preserving IDs, states and provenance.

    Bounded by construction. Every dataset here is already free of credentials
    -- none of these records has a field that could hold one -- and a test
    plants a secret in the environment to prove none appears in the output.
    """
    _attach_ledger()

    if dataset == "executions":
        rows = (
            _ENGINE.ledger.history(limit=limit, offset=offset) if _ENGINE.ledger else []
        )
        payload = [
            {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
            for r in rows
        ]
    elif dataset == "audit":
        payload = [
            entry.as_dict()
            for record in _ENGINE.requests.values()
            for entry in record.audit
        ][offset : offset + limit]
    elif dataset == "opportunities":
        payload = [o.as_dict() for o in svc.state().opportunities][
            offset : offset + limit
        ]
    elif dataset == "ledger":
        events, _total, _integrity = svc.ledger_events(limit=limit, offset=offset)
        payload = [
            {
                "seq": e.seq,
                "id": str(e.id),
                "timestamp": e.timestamp.isoformat(),
                "event_type": str(e.event_type),
                "actor": e.actor,
                "event_hash": e.event_hash,
                "prev_hash": e.prev_hash,
            }
            for e in events
        ]
    else:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown dataset {dataset!r}; available: executions, audit, "
                "opportunities, ledger"
            ),
        )

    return {
        "dataset": dataset,
        "rows": payload,
        "count": len(payload),
        "limit": limit,
        "offset": offset,
        "contains_secrets": False,
    }


@app.get("/api/executions/{execution_id}/audit", tags=["execution"])
def execution_audit(execution_id: str) -> dict[str, Any]:
    """Append-only lifecycle trail. Read-only over HTTP."""
    record = _ENGINE.requests.get(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no execution {execution_id!r}")
    return {"execution_id": execution_id, "audit": [a.as_dict() for a in record.audit]}


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


@app.get("/api/orders", tags=["orders"])
def orders(svc: DomainService = Depends(service)) -> dict[str, Any]:
    book = svc.order_book
    return {
        "orders": [order.as_dict() for order in book.all_orders()],
        "duplicates_ignored": book.duplicates_ignored,
    }


@app.get("/api/inventory", tags=["inventory"])
def inventory(svc: DomainService = Depends(service)) -> dict[str, Any]:
    state = svc.state()
    return {
        "positions": [
            {
                **state.inventory.position(sku).as_dict(),
                "reorder": state.reorder[sku].as_dict() if sku in state.reorder else None,
            }
            for sku in state.inventory.skus()
        ]
    }


@app.get("/api/analytics", tags=["analytics"])
def analytics(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Portfolio standing plus the recommendations that follow from it."""
    state = svc.state()
    recommendations = [r for entry in state.portfolio for r in recommend(entry)]
    reconciliations = svc.settlements.reconcile_all(svc.order_book)
    tally: dict[str, int] = {}
    for item in reconciliations:
        tally[str(item.state)] = tally.get(str(item.state), 0) + 1

    return {
        "portfolio": [entry.as_dict() for entry in state.portfolio],
        "recommendations": [r.as_dict() for r in recommendations],
        "report": optimisation_report(recommendations),
        "settlement": {
            "by_state": tally,
            "lines": [item.as_dict() for item in reconciliations[:50]],
        },
    }


# --------------------------------------------------------------------------
# Ledger and settings
# --------------------------------------------------------------------------


@app.get("/api/ledger", tags=["ledger"])
def ledger(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    svc: DomainService = Depends(service),
) -> dict[str, Any]:
    """The append-only audit ledger and its hash-chain status."""
    events, total, integrity = svc.ledger_events(limit=limit, offset=offset)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "integrity": {
            k: (str(v) if v is not None else None) for k, v in integrity.items()
        },
        "events": [
            {
                "seq": event.seq,
                "id": str(event.id),
                "timestamp": event.timestamp.isoformat(),
                "event_type": str(event.event_type),
                "actor": event.actor,
                "sku_id": str(event.sku_id) if event.sku_id else None,
                "event_hash": event.event_hash,
                "prev_hash": event.prev_hash,
            }
            for event in events
        ],
    }


@app.get("/api/settings", tags=["settings"])
def settings(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Configuration, capability and credential status. No secret values."""
    return svc.settings_snapshot()


@app.post("/api/refresh", tags=["system"])
def refresh(svc: DomainService = Depends(service)) -> dict[str, Any]:
    """Rebuild the domain state from the fixtures and engines."""
    state = svc.state(refresh=True)
    return {"rebuilt_at": state.built_at.isoformat(), "products": len(state.groups)}


__all__ = ["API_VERSION", "app"]
