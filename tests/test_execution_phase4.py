"""Phase 4: nothing executes externally without approval and every gate.

An approval authorises one exact operation — its type, SKU, marketplace and
payload fingerprint. Any of those changing invalidates it. Every gate is
rechecked immediately before the attempt, so an approval granted before a
compliance result changed cannot carry a product past it.
"""

from __future__ import annotations

import collections
import json
import threading
from datetime import UTC, datetime

import pytest

from execution import (
    Approval,
    ExecutionContext,
    ExecutionEngine,
    ExecutionMode,
    ExecutionRequest,
    ExecutionStatus,
    IllegalTransition,
    OperationType,
    RetryClass,
    build_idempotency_key,
    evaluate_gates,
    fingerprint,
    live_execution_enabled,
)
from part2.execution import IdempotencyStore

CREDS = {
    "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER",
    "AMAZON_SP_CLIENT_ID": "PLACEHOLDER",
    "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER",
}


def a_request(**overrides) -> ExecutionRequest:
    defaults = {
        "operation": OperationType.PRICE_UPDATE,
        "marketplace": "amazon_in",
        "sku_code": "SKU-TRIM-880",
        "payload": {"price_inr": "1299"},
        "requested_by": "operator",
    }
    defaults.update(overrides)
    return ExecutionRequest(**defaults)


def a_context(**overrides) -> ExecutionContext:
    defaults = {
        "compliance_state": "PASS",
        "opportunity_verdict": "RECOMMENDED",
        "supplier_confirmed": True,
        "economics_available": True,
    }
    defaults.update(overrides)
    return ExecutionContext(**defaults)


def blocked_gates(result) -> set[str]:
    return {g.gate for g in result.gates if g.blocks}


# ==========================================================================
# Approval
# ==========================================================================


def test_without_approval_nothing_executes():
    engine = ExecutionEngine()
    result = engine.execute(a_request(), a_context(), environ=CREDS)
    assert result.status is ExecutionStatus.REQUIRES_REVIEW
    assert "approval_exists" in blocked_gates(result)
    assert result.side_effect_occurred is False


def test_a_rejected_request_cannot_execute():
    request = a_request()
    request.reject("operator", "not now")
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert "status_is_approved" in blocked_gates(result)


def test_a_cancelled_request_cannot_execute():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(cancelled=True), environ=CREDS)
    assert "not_cancelled" in blocked_gates(result)


def test_changing_the_payload_after_approval_invalidates_it():
    """The core promise: approval covers one exact payload."""
    request = a_request()
    request.approve("operator")
    request.payload["price_inr"] = "49"

    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert result.status is ExecutionStatus.REQUIRES_REVIEW
    assert "payload_unchanged" in blocked_gates(result)


def test_an_approval_for_another_sku_does_not_authorise_this_one():
    approval = Approval(
        approved_by="operator",
        approved_at=datetime.now(UTC),
        operation=OperationType.PRICE_UPDATE,
        marketplace="amazon_in",
        sku_code="SKU-OTHER",
        payload_fingerprint=fingerprint({"price_inr": "1299"}),
    )
    ok, why = approval.authorises(
        operation=OperationType.PRICE_UPDATE,
        marketplace="amazon_in",
        sku_code="SKU-TRIM-880",
        payload_fingerprint=fingerprint({"price_inr": "1299"}),
    )
    assert not ok
    assert "SKU-OTHER" in why


def test_an_approval_for_another_marketplace_does_not_authorise_this_one():
    approval = Approval(
        approved_by="operator",
        approved_at=datetime.now(UTC),
        operation=OperationType.PRICE_UPDATE,
        marketplace="flipkart",
        sku_code="SKU-1",
        payload_fingerprint="abc",
    )
    ok, why = approval.authorises(
        operation=OperationType.PRICE_UPDATE,
        marketplace="amazon_in",
        sku_code="SKU-1",
        payload_fingerprint="abc",
    )
    assert not ok
    assert "flipkart" in why


def test_an_approval_for_another_operation_does_not_authorise_this_one():
    approval = Approval(
        approved_by="operator",
        approved_at=datetime.now(UTC),
        operation=OperationType.PRICE_UPDATE,
        marketplace="amazon_in",
        sku_code="SKU-1",
        payload_fingerprint="abc",
    )
    ok, _ = approval.authorises(
        operation=OperationType.LISTING_PUBLISH,
        marketplace="amazon_in",
        sku_code="SKU-1",
        payload_fingerprint="abc",
    )
    assert not ok


def test_executing_is_unreachable_without_passing_through_approved():
    request = a_request()
    with pytest.raises(IllegalTransition):
        request.transition(ExecutionStatus.EXECUTING, actor="attacker")


def test_a_succeeded_request_is_terminal():
    request = a_request()
    request.approve("operator")
    request.transition(ExecutionStatus.EXECUTING, actor="engine")
    request.transition(ExecutionStatus.SUCCEEDED, actor="engine")
    with pytest.raises(IllegalTransition):
        request.transition(ExecutionStatus.EXECUTING, actor="engine")


# ==========================================================================
# Capabilities
# ==========================================================================


def test_an_unsupported_capability_blocks_execution():
    request = a_request(marketplace="meesho", operation=OperationType.LISTING_PUBLISH)
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert "capability_supported" in blocked_gates(result)


def test_an_unconfigured_adapter_blocks_execution():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(), environ={})
    assert "adapter_configured" in blocked_gates(result)


def test_credentials_cannot_make_an_unsupported_capability_supported():
    request = a_request(marketplace="meesho", operation=OperationType.LISTING_PUBLISH)
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(), environ={**CREDS, "MEESHO_TOKEN": "x"}
    )
    assert "capability_supported" in blocked_gates(result)


# ==========================================================================
# Compliance, verdict, supplier, economics
# ==========================================================================


@pytest.mark.parametrize("state", ["FAIL", "UNKNOWN", "REQUIRES_HUMAN_REVIEW", None])
def test_uncleared_compliance_blocks_execution(state):
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(compliance_state=state), environ=CREDS
    )
    assert "compliance_cleared" in blocked_gates(result)


@pytest.mark.parametrize("verdict", ["REJECTED", "INSUFFICIENT_DATA"])
def test_a_non_executable_verdict_blocks_execution(verdict):
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(opportunity_verdict=verdict), environ=CREDS
    )
    assert "verdict_executable" in blocked_gates(result)


@pytest.mark.parametrize("verdict", ["RECOMMENDED", "BACKUP"])
def test_a_favourable_verdict_still_requires_approval(verdict):
    """A verdict is evidence, not authorisation."""
    result = ExecutionEngine().execute(
        a_request(), a_context(opportunity_verdict=verdict), environ=CREDS
    )
    assert "approval_exists" in blocked_gates(result)


def test_missing_economics_blocks_execution():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(economics_available=False), environ=CREDS
    )
    assert "economics_available" in blocked_gates(result)


def test_an_unconfirmed_supplier_blocks_execution():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(supplier_confirmed=False), environ=CREDS
    )
    assert "supplier_confirmed" in blocked_gates(result)


# ==========================================================================
# Idempotency
# ==========================================================================


def test_the_idempotency_key_is_derived_from_content_not_time():
    first = build_idempotency_key(OperationType.PRICE_UPDATE, "amazon_in", "SKU-1", "fp")
    second = build_idempotency_key(OperationType.PRICE_UPDATE, "amazon_in", "SKU-1", "fp")
    assert first == second


def test_a_different_payload_produces_a_different_key():
    request = a_request()
    before = request.idempotency_key
    request.payload["price_inr"] = "1499"
    assert request.idempotency_key != before


def test_the_same_operation_twice_executes_once():
    engine = ExecutionEngine()
    request = a_request()
    request.approve("operator")

    first = engine.execute(request, a_context(), environ=CREDS)
    assert first.status is ExecutionStatus.SUCCEEDED

    duplicate = a_request()
    duplicate.approve("operator")
    second = engine.execute(duplicate, a_context(), environ=CREDS)
    assert "idempotency_available" in blocked_gates(second)


def test_concurrent_submissions_produce_exactly_one_execution():
    engine = ExecutionEngine()
    outcomes: list[ExecutionStatus] = []
    barrier = threading.Barrier(8)

    def attempt() -> None:
        request = a_request()
        request.approve("operator")
        barrier.wait(timeout=20)
        outcomes.append(engine.execute(request, a_context(), environ=CREDS).status)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    succeeded = [s for s in outcomes if s is ExecutionStatus.SUCCEEDED]
    assert len(succeeded) == 1, (
        f"expected one winner, got {collections.Counter(outcomes)}"
    )


# ==========================================================================
# Dry run and live mode
# ==========================================================================


def test_dry_run_is_the_default_and_has_no_side_effect():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert result.mode is ExecutionMode.DRY_RUN
    assert result.side_effect_occurred is False
    assert result.external_reference is None


def test_dry_run_never_calls_an_adapter():
    called: list[str] = []

    def adapter(_payload):
        called.append("called")
        raise AssertionError("a dry run must never reach an adapter")

    request = a_request()
    request.approve("operator")
    ExecutionEngine().execute(
        request,
        a_context(),
        environ=CREDS,
        adapter_call=adapter,
        mode=ExecutionMode.DRY_RUN,
    )
    assert called == []


def test_live_mode_is_disabled_unless_explicitly_enabled():
    assert not live_execution_enabled({})
    assert not live_execution_enabled({"ECOM_AUTOPILOT_LIVE_EXECUTION": ""})
    assert not live_execution_enabled({"ECOM_AUTOPILOT_LIVE_EXECUTION": "maybe"})
    assert live_execution_enabled({"ECOM_AUTOPILOT_LIVE_EXECUTION": "true"})


def test_requesting_live_without_the_flag_runs_dry():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(), environ=CREDS, mode=ExecutionMode.LIVE
    )
    assert result.side_effect_occurred is False


def test_dry_run_still_runs_every_gate():
    request = a_request()
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert len(result.gates) >= 12
    assert result.status is ExecutionStatus.REQUIRES_REVIEW


# ==========================================================================
# Live results are never fabricated
# ==========================================================================


class _Outcome:
    def __init__(self, **kwargs):
        self.confirmed = kwargs.get("confirmed", False)
        self.uncertain = kwargs.get("uncertain", False)
        self.external_reference = kwargs.get("external_reference")
        self.detail = kwargs.get("detail", "")
        self.error_code = kwargs.get("error_code")


LIVE = {**CREDS, "ECOM_AUTOPILOT_LIVE_EXECUTION": "true"}


def _live_result(outcome_or_exc):
    request = a_request()
    request.approve("operator")

    def adapter(_payload):
        if isinstance(outcome_or_exc, Exception):
            raise outcome_or_exc
        return outcome_or_exc

    return ExecutionEngine().execute(
        request, a_context(), environ=LIVE, mode=ExecutionMode.LIVE, adapter_call=adapter
    )


def test_success_is_claimed_only_when_the_adapter_confirms_it():
    result = _live_result(_Outcome(confirmed=True, external_reference="AMZ-1"))
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.side_effect_occurred is True
    assert result.external_reference == "AMZ-1"


def test_an_adapter_failure_is_not_success():
    result = _live_result(_Outcome(confirmed=False, detail="rejected by marketplace"))
    assert result.status is ExecutionStatus.FAILED
    assert result.side_effect_occurred is False


def test_an_uncertain_outcome_requires_review_and_is_never_auto_retried():
    """A request that may have landed must not be blindly repeated."""
    result = _live_result(_Outcome(uncertain=True, detail="response lost"))
    assert result.status is ExecutionStatus.REQUIRES_REVIEW
    assert result.retry_class is RetryClass.UNKNOWN_OUTCOME
    assert not result.retry_class.may_retry_automatically


def test_an_adapter_exception_becomes_unknown_outcome_not_failure():
    result = _live_result(RuntimeError("connection reset"))
    assert result.status is ExecutionStatus.REQUIRES_REVIEW
    assert result.retry_class is RetryClass.UNKNOWN_OUTCOME
    assert "Reconcile" in result.error_message


def test_live_without_a_transport_fails_rather_than_pretending():
    request = a_request()
    request.approve("operator")
    result = ExecutionEngine().execute(
        request, a_context(), environ=LIVE, mode=ExecutionMode.LIVE, adapter_call=None
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error_code == "NOT_CONFIGURED"
    assert result.side_effect_occurred is False


# ==========================================================================
# Inventory
# ==========================================================================


def test_a_negative_inventory_quantity_is_rejected():
    request = a_request(
        operation=OperationType.INVENTORY_UPDATE, payload={"quantity": -5}
    )
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert "inventory_valid" in blocked_gates(result)


def test_a_zero_inventory_quantity_is_allowed():
    request = a_request(operation=OperationType.INVENTORY_UPDATE, payload={"quantity": 0})
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert "inventory_valid" not in blocked_gates(result)


def test_a_blocked_operation_records_no_side_effect():
    request = a_request(
        operation=OperationType.INVENTORY_UPDATE, payload={"quantity": -1}
    )
    request.approve("operator")
    result = ExecutionEngine().execute(request, a_context(), environ=CREDS)
    assert result.side_effect_occurred is False


# ==========================================================================
# Audit trail
# ==========================================================================


def test_every_transition_is_audited():
    request = a_request()
    request.approve("operator")
    ExecutionEngine().execute(request, a_context(), environ=CREDS)
    statuses = [entry.to_status for entry in request.audit]
    assert "PENDING_APPROVAL" in statuses
    assert "APPROVED" in statuses
    assert "SUCCEEDED" in statuses


def test_the_audit_trail_is_append_only_through_the_lifecycle():
    request = a_request()
    before = len(request.audit)
    request.approve("operator")
    assert len(request.audit) > before
    assert all(isinstance(e.at, datetime) for e in request.audit)


def test_the_audit_records_the_approved_fingerprint():
    request = a_request()
    request.approve("operator")
    entry = next(e for e in request.audit if e.to_status == "APPROVED")
    assert request.payload_fingerprint[:12] in entry.detail


# ==========================================================================
# Security
# ==========================================================================


def test_no_credential_appears_in_a_serialised_execution():
    request = a_request(payload={"price_inr": "1299"})
    request.approve("operator")
    ExecutionEngine().execute(
        request,
        a_context(),
        environ={**CREDS, "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER-super-secret-value"},
    )
    payload = json.dumps(request.as_dict())
    assert "PLACEHOLDER-super-secret-value" not in payload


def test_no_credential_is_stored_in_the_payload_or_approval():
    request = a_request()
    request.approve("operator")
    body = request.as_dict()
    for blob in (json.dumps(body["payload"]), json.dumps(body["approval"])):
        for term in ("secret", "token", "password", "api_key"):
            assert term not in blob.lower()


# ==========================================================================
# Regression: Phase 2 and 3 unchanged
# ==========================================================================


def test_phase_2_and_3_behaviour_is_unchanged():
    from decisions.pipeline import Pipeline
    from marketplace import build_draft, validate_draft
    from reference.loader import ReferenceStore
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import evaluate_group, rank_opportunities

    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())
    opportunities = rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )
    assert collections.Counter(str(o.verdict) for o in opportunities) == {
        "BACKUP": 1,
        "REJECTED": 6,
    }
    for opportunity in opportunities:
        assert not validate_draft(build_draft(opportunity)).readiness.is_ready


def test_gates_are_evaluated_in_full_not_short_circuited():
    """An operator should see every reason, not the first one."""
    request = a_request()
    gates = evaluate_gates(
        request,
        a_context(compliance_state="FAIL", supplier_confirmed=False),
        mode=ExecutionMode.DRY_RUN,
        store=IdempotencyStore(),
        environ={},
    )
    blocked = {g.gate for g in gates if g.blocks}
    assert {"approval_exists", "compliance_cleared", "supplier_confirmed"} <= blocked
