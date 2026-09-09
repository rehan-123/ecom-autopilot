"""Persistence for catalog entities.

Structural deviation from the specified layout, and the reason matters: an
entity write and its ledger event must land in the *same* transaction. A fact
stored without its event, or an event without its fact, is a broken audit
trail — and it breaks silently, which is the worst way for an audit trail to
break. Leaving that coordination to callers guarantees someone eventually
forgets. So every write goes through here, and here is the only place that
knows how to pair the two.

No ``update_*`` methods. Entities in this layer are immutable: new
information means a new row (a new offer, a new transition, a new snapshot),
never an edit to an existing one.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import Connection, Engine, select
from sqlalchemy.exc import OperationalError

from catalog.enums import (
    TERMINAL_STATES,
    EventType,
    SKUState,
    TransitionTrigger,
)
from catalog.schemas import (
    SKU,
    AIInvocation,
    Approval,
    AuditEvent,
    Product,
    ProductIdentityEvidence,
    SKUStateTransition,
    SKUView,
    Supplier,
    SupplierOffer,
)
from database.connection import read_only
from database.tables import (
    ai_invocation as ai_invocation_table,
)
from database.tables import (
    approval as approval_table,
)
from database.tables import (
    product as product_table,
)
from database.tables import (
    product_identity_evidence as evidence_table,
)
from database.tables import (
    sku as sku_table,
)
from database.tables import (
    sku_current_state,
)
from database.tables import (
    sku_state_transition as transition_table,
)
from database.tables import (
    supplier as supplier_table,
)
from database.tables import (
    supplier_offer as offer_table,
)
from ledger.events import EventRepository

logger = logging.getLogger(__name__)

SYSTEM_ACTOR = "SYSTEM"


class InvalidTransitionError(ValueError):
    """Raised when a state change is not permitted."""


class CatalogRepository:
    """Transactional writes and reads for catalog entities."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._events = EventRepository(engine)

    # -- suppliers --------------------------------------------------------

    def create_supplier(
        self, supplier: Supplier, *, actor: str = SYSTEM_ACTOR
    ) -> Supplier:
        with self._engine.begin() as conn:
            conn.execute(
                supplier_table.insert().values(
                    id=supplier.id,
                    name=supplier.name,
                    type=str(supplier.type),
                    gst_status=str(supplier.gst_status),
                    gstin=supplier.gstin,
                    issues_tax_invoice=supplier.issues_tax_invoice,
                    state_code=supplier.state_code,
                    contact=supplier.contact,
                    notes=supplier.notes,
                    created_at=supplier.created_at,
                )
            )
            self._events.append_event(
                AuditEvent(
                    supplier_id=supplier.id,
                    event_type=EventType.SUPPLIER_CREATED,
                    actor=actor,
                    result_json={
                        "name": supplier.name,
                        "type": str(supplier.type),
                        "gst_status": str(supplier.gst_status),
                        "issues_tax_invoice": supplier.issues_tax_invoice,
                    },
                ),
                conn=conn,
            )
        return supplier

    def get_supplier(self, supplier_id: uuid.UUID) -> Supplier | None:
        with read_only(self._engine) as conn:
            row = conn.execute(
                select(supplier_table).where(supplier_table.c.id == supplier_id)
            ).first()
        return None if row is None else Supplier(**dict(row._mapping))

    # -- products ---------------------------------------------------------

    def create_product(self, product: Product, *, actor: str = SYSTEM_ACTOR) -> Product:
        with self._engine.begin() as conn:
            conn.execute(product_table.insert().values(**_product_values(product)))
            self._events.append_event(
                AuditEvent(
                    product_id=product.id,
                    event_type=EventType.PRODUCT_CREATED,
                    actor=actor,
                    result_json={
                        "canonical_name": product.canonical_name,
                        "category": product.category,
                        "hsn_code": product.hsn_code,
                    },
                ),
                conn=conn,
            )
        return product

    def get_product(self, product_id: uuid.UUID) -> Product | None:
        with read_only(self._engine) as conn:
            row = conn.execute(
                select(product_table).where(product_table.c.id == product_id)
            ).first()
        return None if row is None else Product(**dict(row._mapping))

    def add_identity_evidence(
        self, evidence: ProductIdentityEvidence, *, actor: str = SYSTEM_ACTOR
    ) -> ProductIdentityEvidence:
        with self._engine.begin() as conn:
            conn.execute(
                evidence_table.insert().values(
                    id=evidence.id,
                    product_id=evidence.product_id,
                    evidence_type=str(evidence.evidence_type),
                    value=evidence.value,
                    source_offer_id=evidence.source_offer_id,
                    collected_at=evidence.collected_at,
                )
            )
            self._events.append_event(
                AuditEvent(
                    product_id=evidence.product_id,
                    event_type=EventType.PRODUCT_IDENTITY_EVIDENCE_ADDED,
                    actor=actor,
                    result_json={
                        "evidence_type": str(evidence.evidence_type),
                        "value": evidence.value,
                    },
                ),
                conn=conn,
            )
        return evidence

    # -- SKUs -------------------------------------------------------------

    def create_sku(
        self,
        sku: SKU,
        *,
        initial_state: SKUState = SKUState.DISCOVERED,
        actor: str = SYSTEM_ACTOR,
    ) -> SKU:
        """Create a SKU and record its initial state.

        The initial state is written to the ledger, not to the ``sku`` row —
        ``sku`` has no state column. ``sku_current_state`` is projected from
        these events.
        """
        with self._engine.begin() as conn:
            conn.execute(
                sku_table.insert().values(
                    id=sku.id,
                    product_id=sku.product_id,
                    sku_code=sku.sku_code,
                    variant=sku.variant,
                    pack_qty=sku.pack_qty,
                    created_at=sku.created_at,
                )
            )
            transition = SKUStateTransition(
                sku_id=sku.id,
                from_state=None,
                to_state=initial_state,
                trigger=TransitionTrigger.INGEST,
                actor=actor,
                rationale="SKU created",
                created_at=sku.created_at,
            )
            _insert_transition(conn, transition)
            self._events.append_event(
                AuditEvent(
                    sku_id=sku.id,
                    product_id=sku.product_id,
                    event_type=EventType.SKU_CREATED,
                    actor=actor,
                    timestamp=sku.created_at,
                    result_json={
                        "sku_code": sku.sku_code,
                        "state": str(initial_state),
                        "pack_qty": sku.pack_qty,
                    },
                ),
                conn=conn,
            )
        return sku

    def transition_sku(
        self,
        sku_id: uuid.UUID,
        to_state: SKUState,
        *,
        trigger: TransitionTrigger,
        actor: str = SYSTEM_ACTOR,
        rule_id: str | None = None,
        rationale: str | None = None,
        allow_reopen: bool = False,
    ) -> SKUStateTransition:
        """Record a lifecycle change. The only way state ever changes.

        Leaving a terminal state (``LISTING_READY`` or ``REJECTED``) requires
        ``allow_reopen=True``. This is a data-integrity guard, not a transition
        graph: reopening a decided SKU is legitimate — a supplier price can
        fall — but it must be deliberate rather than the result of a
        misrouted rule. The full legality graph is a business rule and belongs
        to a later milestone.
        """
        current = self.current_state(sku_id)
        if current is None:
            raise InvalidTransitionError(f"SKU {sku_id} does not exist")
        if current == to_state:
            raise InvalidTransitionError(f"SKU {sku_id} is already in state {to_state}")
        if current in TERMINAL_STATES and not allow_reopen:
            raise InvalidTransitionError(
                f"SKU {sku_id} is in terminal state {current}; pass "
                "allow_reopen=True to reopen it deliberately"
            )

        transition = SKUStateTransition(
            sku_id=sku_id,
            from_state=current,
            to_state=to_state,
            trigger=trigger,
            rule_id=rule_id,
            actor=actor,
            rationale=rationale,
        )
        with self._engine.begin() as conn:
            _insert_transition(conn, transition)
            self._events.append_event(
                AuditEvent(
                    sku_id=sku_id,
                    event_type=EventType.SKU_STATE_CHANGED,
                    actor=actor,
                    rule_id=rule_id,
                    timestamp=transition.created_at,
                    inputs_json={"from_state": str(current) if current else None},
                    result_json={
                        "from_state": str(current) if current else None,
                        "to_state": str(to_state),
                        "trigger": str(trigger),
                        "rationale": rationale,
                        "reopened": bool(current in TERMINAL_STATES),
                    },
                ),
                conn=conn,
            )
        return transition

    def current_state(self, sku_id: uuid.UUID) -> SKUState | None:
        """Read current state, preferring the projection.

        Reads the projection directly and falls back to folding the transition
        stream when it is absent. Earlier revisions ran a schema introspection
        on every call (expensive) and then cached the answer (stale once a
        rebuild dropped the table). Attempting the read and handling its
        absence is both cheaper and self-healing: a dropped projection costs
        one extra query, never a wrong answer.
        """
        with read_only(self._engine) as conn:
            try:
                value = conn.execute(
                    select(sku_current_state.c.state).where(
                        sku_current_state.c.sku_id == sku_id
                    )
                ).scalar()
            except OperationalError as exc:
                if "no such table" not in str(exc).lower():
                    raise
                logger.debug("state projection absent; folding transitions")
                value = None
            if value is not None:
                return SKUState(value)

            value = conn.execute(
                select(transition_table.c.to_state)
                .where(transition_table.c.sku_id == sku_id)
                .order_by(transition_table.c.created_at.desc())
                .limit(1)
            ).scalar()
        return None if value is None else SKUState(value)

    def get_sku_view(self, sku_id: uuid.UUID) -> SKUView | None:
        """A SKU together with its current projected state."""
        with read_only(self._engine) as conn:
            row = conn.execute(select(sku_table).where(sku_table.c.id == sku_id)).first()
            if row is None:
                return None
            updated_at = conn.execute(
                select(transition_table.c.created_at)
                .where(transition_table.c.sku_id == sku_id)
                .order_by(transition_table.c.created_at.desc())
                .limit(1)
            ).scalar()
        state = self.current_state(sku_id)
        if state is None or updated_at is None:
            return None
        return SKUView(
            sku=SKU(**dict(row._mapping)), state=state, state_updated_at=updated_at
        )

    def transitions_for(self, sku_id: uuid.UUID) -> list[SKUStateTransition]:
        with read_only(self._engine) as conn:
            rows = conn.execute(
                select(transition_table)
                .where(transition_table.c.sku_id == sku_id)
                .order_by(transition_table.c.created_at.asc())
            ).fetchall()
        return [SKUStateTransition(**dict(row._mapping)) for row in rows]

    # -- supplier offers --------------------------------------------------

    def record_offer(
        self, offer: SupplierOffer, *, actor: str = SYSTEM_ACTOR
    ) -> SupplierOffer:
        """Record a supplier observation.

        A price change is a *new* offer. The previous row is never touched;
        database triggers would refuse the UPDATE in any case.
        """
        with self._engine.begin() as conn:
            conn.execute(offer_table.insert().values(**_offer_values(offer)))
            self._events.append_event(
                AuditEvent(
                    supplier_id=offer.supplier_id,
                    product_id=offer.product_id,
                    event_type=EventType.SUPPLIER_OFFER_CREATED,
                    actor=actor,
                    timestamp=offer.observed_at,
                    result_json={
                        "offer_id": str(offer.id),
                        "unit_cost_inr": str(offer.unit_cost_inr),
                        "moq_units": offer.moq_units,
                        "tax_invoice_available": offer.tax_invoice_available,
                        # Carried explicitly rather than inferred from the
                        # event timestamp: an observation date is a business
                        # fact about the quote, not about when we recorded it,
                        # and the two diverge on any backfill.
                        "observed_at": offer.observed_at.isoformat(),
                    },
                ),
                conn=conn,
            )
        return offer

    def offers_for(
        self, supplier_id: uuid.UUID, product_id: uuid.UUID
    ) -> list[SupplierOffer]:
        """Every observation for a pair, oldest first. Nothing is dropped."""
        with read_only(self._engine) as conn:
            rows = conn.execute(
                select(offer_table)
                .where(offer_table.c.supplier_id == supplier_id)
                .where(offer_table.c.product_id == product_id)
                .order_by(offer_table.c.observed_at.asc())
            ).fetchall()
        return [SupplierOffer(**dict(row._mapping)) for row in rows]

    def current_offer(
        self, supplier_id: uuid.UUID, product_id: uuid.UUID
    ) -> SupplierOffer | None:
        """The latest observation, derived from ``observed_at`` ordering."""
        offers = self.offers_for(supplier_id, product_id)
        return offers[-1] if offers else None

    # -- approvals --------------------------------------------------------

    def record_approval(self, approval: Approval) -> Approval:
        """Persist a human decision.

        Persistence only. This records that a decision was made; it does not
        evaluate whether the SKU deserved it and does not move the SKU's
        state. Approval *workflow* — requiring the three assessment
        references, gating the move to LISTING_READY — is a later milestone.
        """
        with self._engine.begin() as conn:
            conn.execute(
                approval_table.insert().values(
                    id=approval.id,
                    sku_id=approval.sku_id,
                    decision=str(approval.decision),
                    approver=approval.approver,
                    decided_at=approval.decided_at,
                    note=approval.note,
                    economics_snapshot_id=approval.economics_snapshot_id,
                    compliance_assessment_id=approval.compliance_assessment_id,
                    opportunity_score_id=approval.opportunity_score_id,
                )
            )
            self._events.append_event(
                AuditEvent(
                    sku_id=approval.sku_id,
                    event_type=EventType.APPROVAL_RECORDED,
                    actor=f"HUMAN:{approval.approver}",
                    timestamp=approval.decided_at,
                    result_json={
                        "decision": str(approval.decision),
                        "note": approval.note,
                        "economics_snapshot_id": _opt_str(approval.economics_snapshot_id),
                        "compliance_assessment_id": _opt_str(
                            approval.compliance_assessment_id
                        ),
                        "opportunity_score_id": _opt_str(approval.opportunity_score_id),
                    },
                ),
                conn=conn,
            )
        return approval

    def approvals_for(self, sku_id: uuid.UUID) -> list[Approval]:
        with read_only(self._engine) as conn:
            rows = conn.execute(
                select(approval_table)
                .where(approval_table.c.sku_id == sku_id)
                .order_by(approval_table.c.decided_at.asc())
            ).fetchall()
        return [Approval(**dict(row._mapping)) for row in rows]

    # -- AI invocations ---------------------------------------------------

    def record_ai_invocation(self, invocation: AIInvocation) -> AIInvocation:
        """Persist a record of an AI call.

        No Claude client exists in this milestone. This is the persistence
        side, written now so that when a service is added the audit trail is
        already in place rather than retrofitted.
        """
        with self._engine.begin() as conn:
            conn.execute(
                ai_invocation_table.insert().values(
                    id=invocation.id,
                    service=str(invocation.service),
                    sku_id=invocation.sku_id,
                    product_id=invocation.product_id,
                    model=invocation.model,
                    model_version=invocation.model_version,
                    prompt_version=invocation.prompt_version,
                    prompt_hash=invocation.prompt_hash,
                    input_refs=invocation.input_refs,
                    input_token_count=invocation.input_token_count,
                    output_token_count=invocation.output_token_count,
                    latency_ms=invocation.latency_ms,
                    cost_inr=invocation.cost_inr,
                    raw_output=invocation.raw_output,
                    parsed_ok=invocation.parsed_ok,
                    schema_violations=invocation.schema_violations,
                    findings=invocation.findings,
                    created_at=invocation.created_at,
                )
            )
            self._events.append_event(
                AuditEvent(
                    sku_id=invocation.sku_id,
                    product_id=invocation.product_id,
                    event_type=EventType.AI_INVOKED,
                    actor=f"AI:{invocation.service}",
                    timestamp=invocation.created_at,
                    reference_versions_json={
                        "model": invocation.model,
                        "model_version": invocation.model_version,
                        "prompt_version": invocation.prompt_version,
                    },
                    result_json={
                        "parsed_ok": invocation.parsed_ok,
                        "schema_violations": invocation.schema_violations,
                        "finding_count": len(invocation.findings),
                        "cost_inr": str(invocation.cost_inr),
                    },
                ),
                conn=conn,
            )
        return invocation

    # -- errors -----------------------------------------------------------

    def record_error(
        self,
        message: str,
        *,
        actor: str = SYSTEM_ACTOR,
        sku_id: uuid.UUID | None = None,
        product_id: uuid.UUID | None = None,
        supplier_id: uuid.UUID | None = None,
        context: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record a failure in the ledger.

        The ledger deliberately has no foreign keys precisely so it can record
        events about entities that failed validation and were never persisted
        — which is exactly the case this method serves.
        """
        # NB: "message" is a reserved LogRecord attribute; passing it via
        # extra= raises KeyError, which would make the error-handling path
        # itself throw.
        logger.error("recording error event", extra={"detail": message})
        return self._events.append_event(
            AuditEvent(
                sku_id=sku_id,
                product_id=product_id,
                supplier_id=supplier_id,
                event_type=EventType.ERROR,
                actor=actor,
                inputs_json=context or {},
                error_json={"message": message},
            )
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _opt_str(value: uuid.UUID | None) -> str | None:
    return None if value is None else str(value)


def _insert_transition(conn: Connection, transition: SKUStateTransition) -> None:
    conn.execute(
        transition_table.insert().values(
            id=transition.id,
            sku_id=transition.sku_id,
            from_state=str(transition.from_state) if transition.from_state else None,
            to_state=str(transition.to_state),
            trigger=str(transition.trigger),
            rule_id=transition.rule_id,
            actor=transition.actor,
            rationale=transition.rationale,
            created_at=transition.created_at,
        )
    )


def _product_values(product: Product) -> dict[str, Any]:
    return {
        "id": product.id,
        "canonical_name": product.canonical_name,
        "category": product.category,
        "hsn_code": product.hsn_code,
        "brand": product.brand,
        "manufacturer": product.manufacturer,
        "country_of_origin": product.country_of_origin,
        "weight_g": product.weight_g,
        "length_mm": product.length_mm,
        "width_mm": product.width_mm,
        "height_mm": product.height_mm,
        "fragile": product.fragile,
        "sized": product.sized,
        "warranty_months": product.warranty_months,
        "identity_confidence": str(product.identity_confidence),
        "created_at": product.created_at,
    }


def _offer_values(offer: SupplierOffer) -> dict[str, Any]:
    return {
        "id": offer.id,
        "supplier_id": offer.supplier_id,
        "product_id": offer.product_id,
        "unit_cost_inr": offer.unit_cost_inr,
        "cost_is_gst_inclusive": offer.cost_is_gst_inclusive,
        "gst_rate_pct": offer.gst_rate_pct,
        "tax_invoice_available": offer.tax_invoice_available,
        "moq_units": offer.moq_units,
        "available_qty": offer.available_qty,
        "inbound_freight_inr": offer.inbound_freight_inr,
        "dispatch_days": offer.dispatch_days,
        "return_terms": offer.return_terms,
        "warranty_terms": offer.warranty_terms,
        "compliance_docs": offer.compliance_docs,
        "observed_at": offer.observed_at,
        "source_document_ref": offer.source_document_ref,
    }
