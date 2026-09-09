"""The execution engine: twelve gates, then a dry run.

Every gate is checked immediately before the attempt, not once at approval
time. An approval granted yesterday does not carry a product past a compliance
result that changed this morning — that is the difference between authorising
an action and authorising an outcome.

The gates are evaluated in full rather than short-circuiting, so an operator
sees every reason an operation is blocked instead of fixing them one at a
time. Only after all twelve pass does anything reach an adapter, and in dry
run — the default, and the only mode enabled unless the environment
explicitly says otherwise — the adapter is never called at all.

Nothing here fabricates success. A dry run reports DRY_RUN. An unsupported
capability reports why. A lost response reports UNKNOWN_OUTCOME and demands
reconciliation rather than retrying something that may already have happened.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from execution.ledger import PersistentIdempotencyStore
from execution.model import (
    OPERATION_CAPABILITY,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    GateOutcome,
    GateResult,
    OperationType,
    RetryClass,
    live_execution_enabled,
)
from marketplace.draft import CapabilityStatus
from marketplace.registry import capability_status
from part2.execution import ActionResult, IdempotencyStore, Outcome
from part2.marketplace import Capability

logger = logging.getLogger(__name__)

ENGINE_VERSION = "execution-engine/1.0.0"

#: Compliance states that can never authorise an external write.
BLOCKING_COMPLIANCE = frozenset({"FAIL", "UNKNOWN", "REQUIRES_HUMAN_REVIEW"})

#: Opportunity verdicts that can never execute, whatever a human approves.
#: A human may approve a mistake; they may not approve one the evidence has
#: already ruled out.
NON_EXECUTABLE_VERDICTS = frozenset({"REJECTED", "INSUFFICIENT_DATA"})


class ExecutionContext:
    """Authoritative domain state for one execution attempt.

    Supplied by the server from its own domain state, never by the client. A
    request that could carry its own compliance result or supplier
    confirmation would let a caller authorise anything by asserting it.
    """

    def __init__(
        self,
        *,
        compliance_state: str | None = None,
        opportunity_verdict: str | None = None,
        supplier_confirmed: bool = False,
        economics_available: bool = False,
        inventory_qty: int | None = None,
        cancelled: bool = False,
    ) -> None:
        self.compliance_state = compliance_state
        self.opportunity_verdict = opportunity_verdict
        self.supplier_confirmed = supplier_confirmed
        self.economics_available = economics_available
        self.inventory_qty = inventory_qty
        self.cancelled = cancelled


def evaluate_gates(
    request: ExecutionRequest,
    context: ExecutionContext,
    *,
    mode: ExecutionMode,
    store: IdempotencyStore,
    environ: Mapping[str, str] | None = None,
    ledger: Any | None = None,
) -> list[GateOutcome]:
    """Run every safety gate. Returns all outcomes, blocking or not."""
    gates: list[GateOutcome] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        gates.append(
            GateOutcome(name, GateResult.PASS if ok else GateResult.BLOCK, detail)
        )

    # 1-3. Approval exists, is still valid, and matches the current payload.
    approval = request.approval
    check(
        "approval_exists", approval is not None, "" if approval else "no human approval"
    )
    if approval is not None:
        authorised, why = approval.authorises(
            operation=request.operation,
            marketplace=request.marketplace,
            sku_code=request.sku_code,
            payload_fingerprint=request.payload_fingerprint,
        )
        check("approval_matches_operation", authorised, why)
        check(
            "payload_unchanged",
            approval.payload_fingerprint == request.payload_fingerprint,
            "payload changed after approval" if not authorised else "",
        )
    else:
        check("approval_matches_operation", False, "no approval to match")
        check("payload_unchanged", False, "no approved fingerprint to compare")

    check(
        "status_is_approved",
        request.status is ExecutionStatus.APPROVED,
        f"status is {request.status}",
    )

    # 4-5. Marketplace capability, derived from the adapter, never from creds.
    capability_name = OPERATION_CAPABILITY.get(request.operation)
    capability = Capability[capability_name] if capability_name else None
    if capability is None:
        check("capability_supported", False, f"no capability maps to {request.operation}")
        check("adapter_configured", False, "no capability to configure")
    else:
        report = capability_status(request.marketplace, capability, environ=environ)
        check(
            "capability_supported",
            report.status is not CapabilityStatus.UNSUPPORTED,
            report.reason,
        )
        check(
            "adapter_configured",
            report.status is not CapabilityStatus.NOT_CONFIGURED,
            report.reason,
        )

    # 6. Live execution is enabled only when explicitly configured.
    if mode is ExecutionMode.LIVE:
        check(
            "live_execution_enabled",
            live_execution_enabled(environ),
            "live execution is not enabled; set the environment flag deliberately",
        )
    else:
        check("live_execution_enabled", True, "dry run needs no live authorisation")

    # 7. Compliance, read from Phase 2 rather than recomputed.
    check(
        "compliance_cleared",
        context.compliance_state is not None
        and context.compliance_state not in BLOCKING_COMPLIANCE,
        f"compliance is {context.compliance_state}",
    )

    # 8. Opportunity verdict and economics readiness.
    check(
        "verdict_executable",
        context.opportunity_verdict not in NON_EXECUTABLE_VERDICTS,
        f"verdict is {context.opportunity_verdict}",
    )
    check(
        "economics_available",
        context.economics_available,
        "no economics were computed",
    )

    # 9. Supplier confirmation.
    check(
        "supplier_confirmed",
        context.supplier_confirmed,
        "no confirmed supplier; a probable match is not a source of supply",
    )

    # 10. Inventory sanity, where the operation touches it.
    if request.operation is OperationType.INVENTORY_UPDATE:
        quantity = request.payload.get("quantity")
        check(
            "inventory_valid",
            isinstance(quantity, int) and quantity >= 0,
            f"invalid quantity {quantity!r}",
        )
    else:
        check("inventory_valid", True, "operation does not change inventory")

    # 11. Idempotency: this exact effect must not already have been attempted.
    # The database is authoritative when present; the in-memory store is the
    # single-process fallback.
    if ledger is not None:
        allowed, why = ledger.may_retry(request.idempotency_key)
        check("idempotency_available", allowed, why)
    else:
        check(
            "idempotency_available",
            store.lookup(request.idempotency_key) is None,
            "this exact operation has already been executed",
        )

    # 12. No cancellation.
    check("not_cancelled", not context.cancelled, "execution was cancelled")

    return gates


class ExecutionEngine:
    """Runs execution requests through the gates. Dry run by default."""

    def __init__(
        self,
        *,
        store: IdempotencyStore | None = None,
        ledger: PersistentIdempotencyStore | None = None,
    ) -> None:
        self.store = store or IdempotencyStore()
        #: When supplied, the database arbitrates idempotency instead of the
        #: in-memory store. That is what makes the guarantee hold across
        #: processes rather than only within one.
        self.ledger = ledger
        self.requests: dict[str, ExecutionRequest] = {}
        self._lock = threading.Lock()

    def submit(self, request: ExecutionRequest) -> ExecutionRequest:
        with self._lock:
            self.requests[str(request.id)] = request
        return request

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
        *,
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
        environ: Mapping[str, str] | None = None,
        adapter_call: Any | None = None,
    ) -> ExecutionResult:
        """Attempt an execution.

        Gates first, always. In dry run the adapter is never invoked at all,
        so a dry run cannot have an external effect even if an adapter were
        misconfigured to perform one.
        """
        started = datetime.now(UTC)
        # Requested live but not enabled: degrade to dry run rather than
        # refusing outright, so the operator still sees the full gate report.
        if mode is ExecutionMode.LIVE and not live_execution_enabled(environ):
            logger.info("live execution requested but not enabled; running dry")

        gates = evaluate_gates(
            request,
            context,
            mode=mode,
            store=self.store,
            environ=environ,
            ledger=self.ledger,
        )
        blocking = [g for g in gates if g.blocks]

        result = ExecutionResult(
            execution_id=str(request.id),
            idempotency_key=request.idempotency_key,
            operation=request.operation,
            marketplace=request.marketplace,
            sku_code=request.sku_code,
            status=ExecutionStatus.REQUIRES_REVIEW,
            mode=mode,
            payload_fingerprint=request.payload_fingerprint,
            started_at=started,
            gates=gates,
        )

        if blocking:
            result.status = ExecutionStatus.REQUIRES_REVIEW
            result.error_code = "GATES_BLOCKED"
            result.error_message = (
                "; ".join(f"{g.gate}: {g.detail}" for g in blocking if g.detail)
                or "one or more safety gates blocked this operation"
            )
            result.retry_class = RetryClass.NON_RETRYABLE
            result.side_effect_occurred = False
            result.completed_at = datetime.now(UTC)
            request.result = result
            request._record(
                actor="engine",
                to_status=request.status,
                detail=f"blocked by {len(blocking)} gate(s)",
                mode=mode,
            )
            return result

        # Claim the key in the database before anything else. Two workers
        # reaching this line concurrently resolve here, in SQLite, not in the
        # memory of either process.
        if self.ledger is not None:
            claim = self.ledger.claim(
                idempotency_key=request.idempotency_key,
                execution_id=str(request.id),
                operation=str(request.operation),
                sku_code=request.sku_code,
                marketplace=request.marketplace,
                payload_fingerprint=request.payload_fingerprint,
                execution_mode=str(mode),
                approval_id=request.approval.approved_by if request.approval else None,
            )
            if not claim.won:
                result.status = ExecutionStatus.REQUIRES_REVIEW
                result.error_code = "DUPLICATE_EXECUTION"
                result.error_message = claim.detail
                result.retry_class = RetryClass.NON_RETRYABLE
                result.side_effect_occurred = False
                result.completed_at = datetime.now(UTC)
                request.result = result
                return result

        request.transition(
            ExecutionStatus.EXECUTING,
            actor="engine",
            detail="all gates passed",
            mode=mode,
        )

        if mode is ExecutionMode.DRY_RUN or not live_execution_enabled(environ):
            # The adapter is not called. A dry run that invoked a live client
            # would be a live run with a reassuring label.
            result.status = ExecutionStatus.SUCCEEDED
            result.side_effect_occurred = False
            result.external_reference = None
            result.error_code = None
            result.error_message = (
                "DRY_RUN: all gates passed and the payload was prepared. No "
                "adapter was called and nothing was changed externally."
            )
            result.completed_at = datetime.now(UTC)
            request.transition(
                ExecutionStatus.SUCCEEDED,
                actor="engine",
                detail="dry run completed with no external effect",
                mode=mode,
            )
            self._remember(request, result)
            if self.ledger is not None:
                self.ledger.complete(
                    request.idempotency_key,
                    execution_state=str(ExecutionStatus.SUCCEEDED),
                    outcome="DRY_RUN",
                    side_effect_occurred=False,
                )
            request.result = result
            return result

        return self._execute_live(request, result, adapter_call, mode)

    def _remember(self, request: ExecutionRequest, result: ExecutionResult) -> None:
        """Record the attempt against its idempotency key.

        Uses the Part 2 store rather than a second one, so a Phase 4 execution
        and a Part 2 action cannot both fire for the same intent.
        """
        self.store.complete(
            request.idempotency_key,
            ActionResult(
                outcome=Outcome.DRY_RUN
                if result.mode is ExecutionMode.DRY_RUN
                else (
                    Outcome.OK
                    if result.status is ExecutionStatus.SUCCEEDED
                    else Outcome.UNKNOWN
                ),
                action=str(result.operation),
                idempotency_key=result.idempotency_key,
                detail=result.error_message or str(result.status),
            ),
        )

    def _execute_live(
        self,
        request: ExecutionRequest,
        result: ExecutionResult,
        adapter_call: Any | None,
        mode: ExecutionMode,
    ) -> ExecutionResult:
        """Perform a live call. Only reachable with live explicitly enabled.

        Success is claimed only when the adapter confirms it. A lost response
        becomes UNKNOWN_OUTCOME and REQUIRES_REVIEW, never an automatic retry:
        the request may already have taken effect.
        """
        if adapter_call is None:
            result.status = ExecutionStatus.FAILED
            result.error_code = "NOT_CONFIGURED"
            result.error_message = (
                "live execution is enabled but no adapter transport was supplied; "
                "nothing was sent"
            )
            result.retry_class = RetryClass.NON_RETRYABLE
            result.side_effect_occurred = False
            result.completed_at = datetime.now(UTC)
            request.transition(
                ExecutionStatus.FAILED,
                actor="engine",
                detail=result.error_message,
                mode=mode,
            )
            request.result = result
            return result

        try:
            outcome = adapter_call(request.payload)
        except Exception as exc:  # noqa: BLE001 - an adapter fault is an outcome
            # The request may or may not have landed. That is genuinely
            # unknown, and guessing either way is worse than saying so.
            result.status = ExecutionStatus.REQUIRES_REVIEW
            result.error_code = "UNKNOWN_OUTCOME"
            result.error_message = (
                f"the adapter raised after the request may have been sent: {exc}. "
                "Reconcile against the marketplace before retrying."
            )
            result.retry_class = RetryClass.UNKNOWN_OUTCOME
            result.side_effect_occurred = False
            result.completed_at = datetime.now(UTC)
            request.transition(
                ExecutionStatus.REQUIRES_REVIEW,
                actor="engine",
                detail=result.error_message[:120],
                mode=mode,
            )
            request.result = result
            return result

        confirmed = bool(getattr(outcome, "confirmed", False))
        if confirmed:
            result.status = ExecutionStatus.SUCCEEDED
            result.side_effect_occurred = True
            result.external_reference = getattr(outcome, "external_reference", None)
            request.transition(
                ExecutionStatus.SUCCEEDED,
                actor="engine",
                detail=f"confirmed by adapter: {result.external_reference}",
                mode=mode,
            )
        elif getattr(outcome, "uncertain", False):
            result.status = ExecutionStatus.REQUIRES_REVIEW
            result.error_code = "UNKNOWN_OUTCOME"
            result.error_message = getattr(outcome, "detail", "outcome not confirmed")
            result.retry_class = RetryClass.UNKNOWN_OUTCOME
            request.transition(
                ExecutionStatus.REQUIRES_REVIEW,
                actor="engine",
                detail="outcome unconfirmed; reconciliation required",
                mode=mode,
            )
        else:
            result.status = ExecutionStatus.FAILED
            result.error_code = getattr(outcome, "error_code", "ADAPTER_FAILED")
            result.error_message = getattr(
                outcome, "detail", "the adapter reported failure"
            )
            result.retry_class = RetryClass.RETRYABLE
            result.side_effect_occurred = False
            request.transition(
                ExecutionStatus.FAILED,
                actor="engine",
                detail=result.error_message[:120],
                mode=mode,
            )

        result.completed_at = datetime.now(UTC)
        self._remember(request, result)
        request.result = result
        return result


__all__ = [
    "BLOCKING_COMPLIANCE",
    "ENGINE_VERSION",
    "NON_EXECUTABLE_VERDICTS",
    "ExecutionContext",
    "ExecutionEngine",
    "evaluate_gates",
]
