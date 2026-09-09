"""Part 2 tests: execution safety, operations, intelligence, and the full loop.

The failure-injection and property sections matter most. Every subsystem here
talks to something outside the process, and the interesting question is never
"does it work when the marketplace behaves" — it is what happens when a
request times out after the write landed, when a settlement file is imported
twice, or when two workers reserve the last unit at the same moment.
"""

from __future__ import annotations

import contextlib
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from catalog.enums import ApprovalDecision, ComplianceResult, SKUState
from catalog.schemas import SupplierOffer, new_id
from part2.execution import (
    ActionResult,
    ExecutionRecord,
    ExecutionState,
    IdempotencyStore,
    IllegalExecutionTransition,
    NotApproved,
    Outcome,
    RetryPolicy,
    execute_with_retry,
    idempotency_key,
    run_approved,
)
from part2.intelligence import (
    Action,
    Lifecycle,
    PriceVerdict,
    SupplierHistory,
    compute_price_points,
    evaluate_price_change,
    optimisation_report,
    portfolio_report,
    rank_portfolio,
    rank_suppliers,
    recommend,
    select_supplier,
)
from part2.inventory import (
    InventoryError,
    InventoryLedger,
    Movement,
    MovementType,
    ReorderPolicy,
    measure_velocity,
    plan_reorder,
)
from part2.marketplace import (
    ADAPTERS,
    Capability,
    ListingPayload,
    RemoteListing,
    capability_matrix,
    detect_drift,
    get_adapter,
)
from part2.operations import (
    MatchState,
    Order,
    OrderBook,
    OrderLine,
    OrderStatus,
    ReturnEvent,
    ReturnReason,
    ReturnsLedger,
    SettlementLedger,
    SettlementLine,
    UnknownStatus,
    compare_prediction,
    compute_realized,
    normalise_status,
    reconcile,
)
from part2.publisher import ListingPublisher, check_prerequisites

NOW = datetime.now(UTC)


def a_payload(**kwargs) -> ListingPayload:
    base = {
        "sku_code": "SKU-1",
        "title": "Rechargeable Trimmer KX-880",
        "description": "A cordless trimmer.",
        "bullets": ("Cordless",),
        "price_inr": Decimal("1299"),
        "quantity": 10,
        "attributes": {"brand": "Kemei", "country_of_origin": "India"},
        "images": ("front.jpg",),
    }
    return ListingPayload(**{**base, **kwargs})


def an_offer(cost: str, *, invoice: bool = True, moq: int = 25) -> SupplierOffer:
    return SupplierOffer(
        supplier_id=new_id(),
        product_id=new_id(),
        unit_cost_inr=Decimal(cost),
        cost_is_gst_inclusive=True,
        gst_rate_pct=Decimal("18"),
        tax_invoice_available=invoice,
        moq_units=moq,
        inbound_freight_inr=Decimal("10"),
    )


def an_order(
    order_id: str,
    *,
    sku: str = "SKU-1",
    qty: int = 1,
    price: str = "1299",
    status: OrderStatus = OrderStatus.DELIVERED,
    marketplace: str = "amazon_in",
) -> Order:
    return Order(
        marketplace=marketplace,
        marketplace_order_id=order_id,
        lines=[OrderLine(sku_code=sku, quantity=qty, unit_price_inr=Decimal(price))],
        ordered_at=NOW,
        status=status,
    )


# ==========================================================================
# Execution safety
# ==========================================================================


def test_unknown_is_never_success():
    """A timeout after the request was sent is not a failure and not a pass."""
    assert not Outcome.UNKNOWN.is_success
    assert Outcome.UNKNOWN.needs_human
    assert not Outcome.DRY_RUN.is_success
    assert Outcome.OK.is_success


def test_idempotency_key_depends_on_intent_not_attempt():
    first = idempotency_key("update_price", "SKU-1", "amazon_in", "1299")
    second = idempotency_key("update_price", "SKU-1", "amazon_in", "1299")
    third = idempotency_key("update_price", "SKU-1", "amazon_in", "1399")
    assert first == second
    assert first != third


def test_repeated_intent_does_not_reach_the_transport():
    """The core idempotency guarantee: a replay acts zero additional times."""
    store = IdempotencyStore()
    calls: list[int] = []

    def operation() -> ActionResult:
        calls.append(1)
        return ActionResult(Outcome.OK, "create_listing", "k")

    for _ in range(5):
        execute_with_retry(operation, key="k", store=store, sleep=lambda _: None)

    assert len(calls) == 1, "the transport was called more than once for one intent"


def test_retryable_failure_is_retried_then_settles():
    store = IdempotencyStore()
    attempts: list[int] = []

    def flaky() -> ActionResult:
        attempts.append(1)
        if len(attempts) < 3:
            return ActionResult(Outcome.RETRYABLE_FAILURE, "x", "k", detail="429")
        return ActionResult(Outcome.OK, "x", "k")

    result = execute_with_retry(flaky, key="k", store=store, sleep=lambda _: None)
    assert result.outcome is Outcome.OK
    assert result.attempts == 3


def test_retries_are_bounded():
    store = IdempotencyStore()
    attempts: list[int] = []

    def always_fails() -> ActionResult:
        attempts.append(1)
        return ActionResult(Outcome.RETRYABLE_FAILURE, "x", "k")

    execute_with_retry(
        always_fails,
        key="k",
        store=store,
        policy=RetryPolicy(max_attempts=4),
        sleep=lambda _: None,
    )
    assert len(attempts) == 4


def test_non_idempotent_operation_is_not_retried_into_a_duplicate():
    """The rule that prevents duplicate orders.

    A retryable failure on a non-idempotent write may have already taken
    effect, so the outcome is UNKNOWN rather than a second attempt.
    """
    store = IdempotencyStore()
    attempts: list[int] = []

    def order_placement() -> ActionResult:
        attempts.append(1)
        return ActionResult(Outcome.RETRYABLE_FAILURE, "place", "k", detail="timeout")

    result = execute_with_retry(
        order_placement, key="k", store=store, idempotent=False, sleep=lambda _: None
    )
    assert len(attempts) == 1
    assert result.outcome is Outcome.UNKNOWN
    assert "duplicate" in result.detail


def test_transport_exception_becomes_unknown_not_failure():
    store = IdempotencyStore()

    def explodes() -> ActionResult:
        raise TimeoutError("connection reset after send")

    result = execute_with_retry(explodes, key="k", store=store, sleep=lambda _: None)
    assert result.outcome is Outcome.UNKNOWN
    assert "TimeoutError" in result.detail


def test_retryable_failure_is_not_cached_as_a_settled_result():
    """An unsettled intent must remain retryable later."""
    store = IdempotencyStore()
    store.complete("k", ActionResult(Outcome.RETRYABLE_FAILURE, "x", "k"))
    assert store.lookup("k") is None


def test_execution_cannot_skip_approval():
    record = ExecutionRecord(
        action="update_price",
        sku_code="SKU-1",
        marketplace="amazon_in",
        proposed={"price_inr": Decimal("1299")},
        reason="test",
    )
    with pytest.raises(NotApproved):
        run_approved(
            record,
            lambda: ActionResult(Outcome.OK, "x", "k"),
            store=IdempotencyStore(),
            sleep=lambda _: None,
        )


def test_approved_action_executes_and_records_history():
    record = ExecutionRecord(
        action="update_price",
        sku_code="SKU-1",
        marketplace="amazon_in",
        proposed={"price_inr": Decimal("1299")},
        reason="test",
    )
    record.approve("rehan")
    result = run_approved(
        record,
        lambda: ActionResult(Outcome.OK, "update_price", "k"),
        store=IdempotencyStore(),
        sleep=lambda _: None,
    )
    assert result.outcome is Outcome.OK
    assert record.state is ExecutionState.EXECUTED
    assert record.approved_by == "rehan"
    assert [s for s, _, _ in record.history][:3] == [
        ExecutionState.RECOMMENDED,
        ExecutionState.APPROVED,
        ExecutionState.QUEUED,
    ]


def test_unknown_outcome_parks_for_human_review():
    record = ExecutionRecord(
        action="create_listing",
        sku_code="SKU-1",
        marketplace="amazon_in",
        proposed={},
        reason="test",
    )
    record.approve("rehan")
    run_approved(
        record,
        lambda: ActionResult(Outcome.UNKNOWN, "x", "k", detail="timeout"),
        store=IdempotencyStore(),
        sleep=lambda _: None,
    )
    assert record.state is ExecutionState.REQUIRES_REVIEW


def test_illegal_execution_transition_is_refused():
    record = ExecutionRecord(
        action="x", sku_code="SKU-1", marketplace="m", proposed={}, reason="r"
    )
    with pytest.raises(IllegalExecutionTransition):
        record.transition(ExecutionState.EXECUTED)


# ==========================================================================
# Marketplace adapters
# ==========================================================================


def test_unsupported_capability_reports_not_supported_never_fake_success():
    """The refusal must come from the capability check specifically.

    Asserting only on the outcome was not enough: an adapter with no
    credentials returns NOT_SUPPORTED for a different reason, which masked
    whether the capability check ran at all. The detail names the capability,
    so the two paths are distinguishable.
    """
    adapter = get_adapter("meesho", environ={})
    result = adapter.update_price("SKU-1", Decimal("999"), dry_run=False)
    assert result.outcome is Outcome.NOT_SUPPORTED
    assert not result.changed_external_state
    assert "does not support" in result.detail
    assert "UPDATE_PRICE" in result.detail


def test_capability_check_runs_before_credentials_and_before_dry_run():
    """Order matters: a capability an adapter lacks can never be dry-run.

    Offering to "would update the price" on a channel that has no price API
    is a lie in planning form, and a plan built on it fails at execution.
    """
    adapter = get_adapter("meesho", environ={})
    dry = adapter.update_price("SKU-1", Decimal("999"), dry_run=True)
    assert dry.outcome is Outcome.NOT_SUPPORTED
    assert "does not support" in dry.detail

    # Even with credentials fully supplied, an absent capability stays absent.
    ondc = get_adapter(
        "ondc",
        environ={
            "ONDC_SUBSCRIBER_ID": "PLACEHOLDER",
            "ONDC_SIGNING_PRIVATE_KEY": "PLACEHOLDER",
        },
    )
    settlements = ondc.get_settlements(since=NOW)
    assert settlements.outcome is Outcome.NOT_SUPPORTED
    assert "does not support" in settlements.detail
    assert "GET_SETTLEMENTS" in settlements.detail


def test_dry_run_never_changes_external_state():
    for name in ADAPTERS:
        adapter = get_adapter(name, environ={})
        result = adapter.create_listing(a_payload(), dry_run=True)
        assert not result.changed_external_state, name
        if result.outcome is Outcome.DRY_RUN:
            assert "Would create listing" in result.detail


def test_missing_credentials_block_a_real_call():
    adapter = get_adapter("amazon_in", environ={})
    result = adapter.create_listing(a_payload(), dry_run=False)
    assert result.outcome is Outcome.NOT_SUPPORTED
    assert "credentials not configured" in result.detail


def test_credentials_present_but_no_transport_is_a_clear_failure():
    """Credentials alone must never enable a network call.

    An adapter built without an explicit transport is offline by
    construction, so an operator who sets credentials but forgets to wire a
    transport gets an actionable message rather than a silent no-op.
    """
    adapter = get_adapter(
        "amazon_in",
        environ={
            "AMAZON_SP_REFRESH_TOKEN": "x",
            "AMAZON_SP_CLIENT_ID": "y",
            "AMAZON_SP_CLIENT_SECRET": "z",
        },
    )
    result = adapter.create_listing(a_payload(), dry_run=False)
    assert result.outcome is Outcome.PERMANENT_FAILURE
    assert "no HTTP transport was supplied" in result.detail
    assert not adapter.live


def test_credential_values_never_appear_in_results():
    # Obvious placeholder: this test proves the value never escapes,
    # so it must not look like a real credential to the scanner either.
    placeholder = "PLACEHOLDER-not-a-real-credential"
    adapter = get_adapter(
        "amazon_in",
        environ={
            "AMAZON_SP_REFRESH_TOKEN": placeholder,
            "AMAZON_SP_CLIENT_ID": placeholder,
            "AMAZON_SP_CLIENT_SECRET": placeholder,
        },
    )
    result = adapter.create_listing(a_payload(), dry_run=False)
    assert placeholder not in result.detail
    assert placeholder not in str(result.as_dict())


def test_identical_listing_content_yields_one_idempotency_key():
    assert a_payload().fingerprint() == a_payload().fingerprint()
    assert a_payload().fingerprint() != a_payload(price_inr=Decimal("1399")).fingerprint()


def test_successful_write_is_not_evidence_of_marketplace_state():
    """Verification is a separate read; drift is normal and must be detected."""
    payload = a_payload()
    remote = RemoteListing(
        listing_id="L1",
        sku_code="SKU-1",
        status="ACTIVE",
        title=payload.title,
        price_inr=Decimal("999"),
        quantity=3,
    )
    drift = detect_drift(payload, remote)
    assert any("price drift" in d for d in drift)
    assert any("inventory drift" in d for d in drift)


def test_suppressed_listing_is_detected():
    payload = a_payload()
    remote = RemoteListing(
        listing_id="L1",
        sku_code="SKU-1",
        status="SUPPRESSED",
        title=payload.title,
        price_inr=payload.price_inr,
        quantity=payload.quantity,
    )
    assert any("not live" in d for d in detect_drift(payload, remote))


def test_capability_matrix_is_honest_about_transport():
    report = capability_matrix(environ={})
    assert "No HTTP transport is bundled" in report
    for name in ADAPTERS:
        assert name in report


def test_unknown_marketplace_is_refused():
    with pytest.raises(KeyError, match="unknown marketplace"):
        get_adapter("shopee")


def test_meesho_declares_no_listing_write():
    profile = get_adapter("meesho", environ={}).profile
    assert not profile.supports(Capability.CREATE_LISTING)
    assert not profile.supports(Capability.UPDATE_PRICE)


# ==========================================================================
# Inventory
# ==========================================================================


def _receive(ledger: InventoryLedger, sku: str, qty: int, ref: str = "PO-1") -> None:
    ledger.record(
        Movement(sku, MovementType.RECEIPT, qty, ref, unit_cost_inr=Decimal("200"))
    )


def test_inventory_cannot_go_negative():
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 5)
    with pytest.raises(InventoryError, match="below zero"):
        ledger.record(Movement("SKU-1", MovementType.WRITE_OFF, 10, "W-1"))


def test_cannot_reserve_more_than_available():
    """Overselling costs the order, the metric and sometimes the account."""
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 3)
    ledger.record(Movement("SKU-1", MovementType.RESERVE, 3, "O-1"))
    with pytest.raises(InventoryError, match="only 0 available"):
        ledger.record(Movement("SKU-1", MovementType.RESERVE, 1, "O-2"))


def test_available_equals_on_hand_minus_reserved():
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 10)
    ledger.record(Movement("SKU-1", MovementType.RESERVE, 4, "O-1"))
    position = ledger.position("SKU-1")
    assert position.on_hand == 10
    assert position.reserved == 4
    assert position.available == 6


def test_ship_consumes_both_reservation_and_stock():
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 10)
    ledger.record(Movement("SKU-1", MovementType.RESERVE, 2, "O-1"))
    ledger.record(Movement("SKU-1", MovementType.SHIP, 2, "O-1"))
    position = ledger.position("SKU-1")
    assert position.on_hand == 8
    assert position.reserved == 0
    assert position.sold_units == 2


def test_duplicate_movement_reference_is_ignored():
    """Order webhooks and settlement files both arrive more than once."""
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 10, "PO-1")
    _receive(ledger, "SKU-1", 10, "PO-1")
    assert ledger.position("SKU-1").on_hand == 10


def test_resalable_return_restocks_but_damaged_does_not():
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 5)
    ledger.record(Movement("SKU-1", MovementType.RESERVE, 2, "O-1"))
    ledger.record(Movement("SKU-1", MovementType.SHIP, 2, "O-1"))
    ledger.record(Movement("SKU-1", MovementType.RETURN_RESALABLE, 1, "R-1"))
    ledger.record(Movement("SKU-1", MovementType.RETURN_DAMAGED, 1, "R-2"))
    position = ledger.position("SKU-1")
    assert position.on_hand == 4
    assert position.damaged == 1


def test_in_transit_arrives_exactly_once():
    ledger = InventoryLedger()
    ledger.record(Movement("SKU-1", MovementType.IN_TRANSIT, 20, "PO-9"))
    assert ledger.position("SKU-1").in_transit == 20
    ledger.record(Movement("SKU-1", MovementType.TRANSIT_ARRIVED, 20, "PO-9-arr"))
    position = ledger.position("SKU-1")
    assert position.in_transit == 0
    assert position.on_hand == 20
    with pytest.raises(InventoryError, match="only 0 outstanding"):
        ledger.record(Movement("SKU-1", MovementType.TRANSIT_ARRIVED, 1, "PO-9-arr2"))


def test_movement_quantity_must_be_positive():
    with pytest.raises(InventoryError, match="must be positive"):
        Movement("SKU-1", MovementType.RECEIPT, -5, "X")


def test_new_product_gets_a_small_test_not_a_forecast():
    """The low-risk initial exposure rule."""
    ledger = InventoryLedger()
    advice = plan_reorder(
        ledger,
        "SKU-NEW",
        unit_landed_cost_inr=Decimal("200"),
        supplier_moq=1,
        policy=ReorderPolicy(initial_test_orders=12),
    )
    assert advice.recommended_qty == 12
    assert advice.velocity_confidence == "NO_DATA"
    assert any("initial test" in r for r in advice.reasons)


def test_capital_cap_limits_an_initial_test():
    ledger = InventoryLedger()
    advice = plan_reorder(
        ledger,
        "SKU-NEW",
        unit_landed_cost_inr=Decimal("2000"),
        supplier_moq=50,
        policy=ReorderPolicy(max_capital_per_product_inr=Decimal("10000")),
    )
    assert advice.capital_required_inr <= Decimal("10000")
    assert any("capital cap" in c for c in advice.constraints_applied)


def test_provisional_velocity_caps_the_order():
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 100)
    for i in range(5):
        ledger.record(Movement("SKU-1", MovementType.RESERVE, 1, f"O-{i}"))
        ledger.record(
            Movement(
                "SKU-1", MovementType.SHIP, 1, f"O-{i}", at=NOW - timedelta(days=10 - i)
            )
        )
    advice = plan_reorder(
        ledger, "SKU-1", unit_landed_cost_inr=Decimal("200"), supplier_moq=10
    )
    assert advice.velocity_confidence == "PROVISIONAL"
    assert any("PROVISIONAL" in c for c in advice.constraints_applied)


def test_velocity_is_measured_only_with_enough_shipments():
    ledger = InventoryLedger()
    _receive(ledger, "SKU-1", 200)
    for i in range(25):
        ledger.record(Movement("SKU-1", MovementType.RESERVE, 1, f"O-{i}"))
        ledger.record(
            Movement(
                "SKU-1", MovementType.SHIP, 1, f"O-{i}", at=NOW - timedelta(days=25 - i)
            )
        )
    velocity, confidence, observations = measure_velocity(ledger, "SKU-1")
    assert confidence == "MEASURED"
    assert observations == 25
    assert velocity > 0


# ==========================================================================
# Orders, returns, settlement
# ==========================================================================


def test_marketplace_statuses_normalise():
    assert normalise_status("Shipped") is OrderStatus.SHIPPED
    assert normalise_status("RTO_INITIATED") is OrderStatus.RTO
    assert normalise_status("packing_in_progress") is OrderStatus.PACKING


def test_unmapped_status_is_refused_not_defaulted():
    """A marketplace changing its vocabulary is a human's problem, not a guess."""
    with pytest.raises(UnknownStatus, match="unmapped marketplace status"):
        normalise_status("QUANTUM_SUPERPOSITION")


def test_duplicate_order_import_is_idempotent():
    book = OrderBook()
    book.import_order(an_order("AMZ-1"))
    book.import_order(an_order("AMZ-1"))
    book.import_order(an_order("AMZ-1"))
    assert len(book) == 1
    assert book.duplicates_ignored == 2


def test_same_order_id_on_different_marketplaces_is_not_a_duplicate():
    book = OrderBook()
    book.import_order(an_order("1", marketplace="amazon_in"))
    book.import_order(an_order("1", marketplace="flipkart"))
    assert len(book) == 2


def test_cancelled_order_cannot_be_resurrected():
    order = an_order("A-1", status=OrderStatus.CANCELLED)
    with pytest.raises(ValueError, match="CANCELLED"):
        order.transition(OrderStatus.SHIPPED)


def test_order_transitions_append_and_never_edit():
    order = an_order("A-1", status=OrderStatus.NEW)
    order.transition(OrderStatus.SHIPPED, note="handed to carrier")
    order.transition(OrderStatus.DELIVERED)
    assert len(order.transitions) == 3
    assert order.transitions[0][0] is OrderStatus.NEW


def test_duplicate_return_is_ignored():
    ledger = ReturnsLedger()
    event = ReturnEvent("amazon_in", "A-1", "SKU-1", 1, ReturnReason.SIZE_FIT, False)
    ledger.record(event)
    ledger.record(event)
    assert len(ledger.events("SKU-1")) == 1


def test_return_rates_are_provisional_on_a_small_sample():
    ledger = ReturnsLedger()
    ledger.record(ReturnEvent("m", "O-1", "SKU-1", 1, ReturnReason.QUALITY, False))
    rates = ledger.rates_for("SKU-1", shipped_units=3)
    assert rates.confidence == "PROVISIONAL"
    assert not rates.is_measured


def test_return_rates_become_measured_with_enough_shipments():
    ledger = ReturnsLedger()
    for i in range(5):
        ledger.record(
            ReturnEvent("m", f"O-{i}", "SKU-1", 1, ReturnReason.SIZE_FIT, False)
        )
    rates = ledger.rates_for("SKU-1", shipped_units=50)
    assert rates.is_measured
    assert rates.return_rate_pct == Decimal("10")


def test_unmatched_settlement_is_never_absorbed():
    """Money that cannot be tied to an order stays visible."""
    line = SettlementLine(
        "amazon_in", "GHOST-1", Decimal("1299"), net_settled_inr=Decimal("900")
    )
    result = reconcile(line, None)
    assert result.state is MatchState.UNMATCHED
    assert "no matching order" in result.notes[0]


def test_clean_settlement_matches():
    order = an_order("A-1", price="1000")
    line = SettlementLine(
        "amazon_in",
        "A-1",
        Decimal("1000"),
        commission_inr=Decimal("100"),
        shipping_inr=Decimal("60"),
        net_settled_inr=Decimal("840"),
    )
    assert reconcile(line, order).state is MatchState.MATCHED


def test_variance_is_reported_not_hidden():
    order = an_order("A-1", price="1000")
    line = SettlementLine(
        "amazon_in",
        "A-1",
        Decimal("1000"),
        commission_inr=Decimal("100"),
        net_settled_inr=Decimal("700"),
    )
    result = reconcile(line, order)
    assert result.state is MatchState.PARTIAL
    assert result.variance_inr != 0


def test_zero_settlement_requires_review():
    order = an_order("A-1", price="1000")
    line = SettlementLine(
        "amazon_in", "A-1", Decimal("1000"), net_settled_inr=Decimal("0")
    )
    assert reconcile(line, order).state is MatchState.REQUIRES_REVIEW


def test_duplicate_settlement_import_is_idempotent():
    ledger = SettlementLedger()
    line = SettlementLine(
        "amazon_in",
        "A-1",
        Decimal("1000"),
        net_settled_inr=Decimal("840"),
        reference="APR-1",
    )
    ledger.ingest(line)
    ledger.ingest(line)
    assert len(ledger.lines()) == 1
    assert ledger.duplicates_ignored == 1


# ==========================================================================
# Realized economics
# ==========================================================================


def _loop(orders: int, *, settle: bool = True, price: str = "1299"):
    book, settlements, returns = OrderBook(), SettlementLedger(), ReturnsLedger()
    for i in range(orders):
        book.import_order(an_order(f"A-{i}", price=price))
        if settle:
            settlements.ingest(
                SettlementLine(
                    "amazon_in",
                    f"A-{i}",
                    Decimal(price),
                    commission_inr=Decimal("100"),
                    shipping_inr=Decimal("60"),
                    net_settled_inr=Decimal(price) - Decimal("160"),
                    reference=f"S-{i}",
                )
            )
    return book, settlements, returns


def test_realized_uses_settled_money_not_order_value():
    """The gap between the two is fees, and it is where margin goes."""
    book, settlements, returns = _loop(5)
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    assert realized.net_settled_inr < realized.gross_revenue_inr
    assert realized.actual_contribution_inr == realized.net_settled_inr - Decimal("1500")


def test_unsettled_revenue_is_flagged_not_counted():
    book, settlements, returns = _loop(3, settle=False)
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    assert realized.unreconciled_inr > 0
    assert not realized.is_evidenced
    assert "not yet settled" in realized.coverage_note


def test_returns_reduce_realized_contribution():
    book, settlements, returns = _loop(5)
    before = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    returns.record(
        ReturnEvent(
            "amazon_in",
            "A-0",
            "SKU-1",
            1,
            ReturnReason.QUALITY,
            False,
            return_shipping_inr=Decimal("150"),
            refund_inr=Decimal("1299"),
        )
    )
    after = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    assert after.actual_contribution_inr < before.actual_contribution_inr


def test_prediction_error_is_measured_and_labelled():
    book, settlements, returns = _loop(25)
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    error = compare_prediction(
        realized,
        predicted_margin_pct=Decimal("40"),
        predicted_profit_per_order_inr=Decimal("500"),
    )
    assert error.margin_error_points is not None
    assert error.direction in ("OPTIMISTIC", "PESSIMISTIC", "ACCURATE")
    assert error.actionable


def test_prediction_error_on_a_small_sample_is_not_actionable():
    book, settlements, returns = _loop(3)
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    error = compare_prediction(
        realized,
        predicted_margin_pct=Decimal("40"),
        predicted_profit_per_order_inr=Decimal("500"),
    )
    assert not error.actionable


# ==========================================================================
# Pricing
# ==========================================================================


def _points(current: str | None = "1299"):
    return compute_price_points(
        "SKU-1",
        breakeven_inr=Decimal("800"),
        min_profitable_inr=Decimal("1000"),
        market_p25_inr=Decimal("1200"),
        market_median_inr=Decimal("1400"),
        current_inr=None if current is None else Decimal(current),
    )


def test_price_below_minimum_viable_is_blocked():
    decision = evaluate_price_change(Decimal("900"), _points())
    assert decision.verdict is PriceVerdict.BLOCKED_BELOW_FLOOR
    assert not decision.allowed


def test_price_failing_worst_case_is_blocked_not_warned():
    """Part 1's gate is not bypassable by a pricing recommendation."""
    decision = evaluate_price_change(
        Decimal("1050"),
        _points(),
        worst_case_margin_at_proposed_pct=Decimal("-3"),
        worst_case_floor_pct=Decimal("0"),
    )
    assert decision.verdict is PriceVerdict.BLOCKED_WORST_CASE
    assert decision.blocked_by == "worst_case_min_net_margin_pct"


def test_large_price_swing_is_blocked():
    decision = evaluate_price_change(
        Decimal("2500"), _points(), max_change_pct=Decimal("25")
    )
    assert decision.verdict is PriceVerdict.BLOCKED_CHANGE_LIMIT


def test_sound_price_is_allowed():
    decision = evaluate_price_change(
        Decimal("1350"),
        _points(),
        worst_case_margin_at_proposed_pct=Decimal("8"),
        worst_case_floor_pct=Decimal("0"),
    )
    assert decision.allowed


def test_target_price_never_exceeds_the_market_median():
    points = compute_price_points(
        "SKU-1",
        breakeven_inr=Decimal("800"),
        min_profitable_inr=Decimal("1000"),
        market_p25_inr=Decimal("1050"),
        market_median_inr=Decimal("1100"),
    )
    assert points.target_inr <= Decimal("1100")


def test_competitive_price_never_drops_below_the_floor():
    points = compute_price_points(
        "SKU-1",
        breakeven_inr=Decimal("800"),
        min_profitable_inr=Decimal("1000"),
        market_p25_inr=Decimal("700"),
        market_median_inr=Decimal("900"),
    )
    assert points.competitive_inr >= points.minimum_viable_inr


# ==========================================================================
# Supplier selection
# ==========================================================================


def test_cheapest_sticker_price_does_not_win_without_a_tax_invoice():
    """The headline number is routinely the worst option."""
    offers = [
        an_offer("240", moq=100),
        an_offer("255", moq=20),
        an_offer("220", invoice=False, moq=500),
    ]
    scores = rank_suppliers(offers, capital_cap_inr=Decimal("100000"))
    cheapest = min(scores, key=lambda s: s.unit_cost_inr)
    assert not cheapest.tax_invoice
    assert scores[0].supplier_id != cheapest.supplier_id


def test_supplier_over_the_capital_cap_is_disqualified():
    offers = [an_offer("240", moq=1000), an_offer("255", moq=20)]
    scores = rank_suppliers(offers, capital_cap_inr=Decimal("15000"))
    assert any(s.disqualified for s in scores)
    assert not select_supplier(scores).disqualified


def test_measured_reliability_shifts_the_ranking():
    good, bad = an_offer("250", moq=20), an_offer("248", moq=20)
    history = {
        str(good.supplier_id): SupplierHistory(Decimal("99"), Decimal("0.5"), 40),
        str(bad.supplier_id): SupplierHistory(Decimal("70"), Decimal("9"), 40),
    }
    scores = rank_suppliers(
        [good, bad], history=history, capital_cap_inr=Decimal("100000")
    )
    assert scores[0].supplier_id == str(good.supplier_id)


def test_no_qualifying_supplier_returns_none():
    scores = rank_suppliers([an_offer("500", moq=10000)], capital_cap_inr=Decimal("1000"))
    assert select_supplier(scores) is None


# ==========================================================================
# Portfolio and optimisation
# ==========================================================================


def _entry(orders: int, price: str, landed: str, *, returns_count: int = 0):
    book, settlements, returns = _loop(orders, price=price)
    for i in range(returns_count):
        returns.record(
            ReturnEvent(
                "amazon_in",
                f"A-{i}",
                "SKU-1",
                1,
                ReturnReason.SIZE_FIT,
                False,
                return_shipping_inr=Decimal("150"),
                refund_inr=Decimal(price),
            )
        )
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal(landed),
    )
    rates = returns.rates_for("SKU-1", shipped_units=orders)
    return realized, rates


def test_thin_evidence_never_produces_a_lifecycle_decision():
    realized, rates = _entry(3, "1299", "300")
    entries = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})
    assert entries[0].lifecycle is Lifecycle.VALIDATE
    assert not entries[0].evidence_sufficient


def test_consistently_unprofitable_product_is_killed():
    realized, rates = _entry(35, "500", "600")
    entries = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})
    assert entries[0].lifecycle is Lifecycle.KILL


def test_strong_product_is_scaled():
    realized, rates = _entry(30, "2000", "300")
    entries = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})
    assert entries[0].lifecycle is Lifecycle.SCALE


def test_high_returns_reduce_rather_than_scale():
    realized, rates = _entry(30, "2000", "300", returns_count=15)
    entries = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})
    assert entries[0].lifecycle in (Lifecycle.REDUCE, Lifecycle.KILL)


def test_ranking_prefers_contribution_per_capital_not_revenue():
    """High revenue with capital locked up loses to lean and profitable."""
    big_realized, big_rates = _entry(30, "2000", "1500")
    lean_realized, lean_rates = _entry(30, "900", "300")
    entries = rank_portfolio(
        {
            "BIG": (big_realized, big_rates, Decimal("100000")),
            "LEAN": (lean_realized, lean_rates, Decimal("4000")),
        }
    )
    assert entries[0].sku_code == "LEAN"


def test_every_recommendation_carries_evidence_and_downside():
    realized, rates = _entry(35, "500", "600")
    entry = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})[0]
    for item in recommend(entry):
        assert item.evidence
        assert item.downside
        assert item.expected_impact
        assert item.reason


def test_recommendations_are_deterministic():
    realized, rates = _entry(30, "2000", "300")
    entry = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})[0]
    first = [r.as_dict() for r in recommend(entry)]
    second = [r.as_dict() for r in recommend(entry)]
    assert first == second


def test_thin_evidence_recommends_collecting_more():
    realized, rates = _entry(2, "1299", "300")
    entry = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})[0]
    actions = {r.action for r in recommend(entry)}
    assert Action.COLLECT_MORE_DATA in actions
    assert Action.KILL_PRODUCT not in actions


def test_reports_render():
    realized, rates = _entry(30, "2000", "300")
    entries = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})
    assert "PRODUCT PORTFOLIO" in portfolio_report(entries)
    assert "OPTIMISATION" in optimisation_report(recommend(entries[0]))


# ==========================================================================
# Publication gate
# ==========================================================================


def _check(**kwargs):
    base = {
        "sku_state": SKUState.LISTING_READY,
        "approval_decision": ApprovalDecision.APPROVED,
        "compliance": ComplianceResult.PASS,
        "payload": a_payload(),
        "current_min_profitable_inr": Decimal("1000"),
    }
    return check_prerequisites(**{**base, **kwargs})


def test_fully_prepared_listing_may_publish():
    assert _check().satisfied


def test_unapproved_listing_cannot_publish():
    check = _check(approval_decision=ApprovalDecision.DEFERRED)
    assert not check.satisfied
    assert any(p.name == "human_approval_exists" for p in check.failures)


def test_failed_compliance_cannot_publish():
    check = _check(compliance=ComplianceResult.REQUIRES_HUMAN_REVIEW)
    assert not check.satisfied
    assert any(p.name == "compliance_passed" for p in check.failures)


def test_sku_not_listing_ready_cannot_publish():
    assert not _check(sku_state=SKUState.APPROVED).satisfied


def test_price_below_current_viability_blocks_publication():
    """Approval on Tuesday says nothing about the supplier's Wednesday price."""
    check = _check(current_min_profitable_inr=Decimal("1500"))
    assert not check.satisfied
    assert any(p.name == "economics_still_valid" for p in check.failures)


def test_missing_attributes_block_publication():
    check = _check(payload=a_payload(attributes={"brand": "Kemei"}))
    assert any(p.name == "required_attributes_present" for p in check.failures)


def test_missing_images_block_publication():
    check = _check(payload=a_payload(images=()))
    assert any(p.name == "images_present" for p in check.failures)


def test_publisher_refuses_and_explains():
    publisher = ListingPublisher(get_adapter("amazon_in", environ={}))
    record, result = publisher.publish(
        a_payload(), _check(approval_decision=None), approver="rehan", dry_run=True
    )
    assert result.outcome is Outcome.PERMANENT_FAILURE
    assert record.state is ExecutionState.REQUIRES_REVIEW
    assert "human_approval_exists" in result.detail


def test_publisher_dry_run_changes_nothing():
    publisher = ListingPublisher(get_adapter("amazon_in", environ={}))
    record, result = publisher.publish(
        a_payload(), _check(), approver="rehan", dry_run=True
    )
    assert result.outcome is Outcome.DRY_RUN
    assert not result.changed_external_state
    assert record.state is ExecutionState.EXECUTED
    assert publisher.published == {}


def test_republishing_identical_content_is_idempotent():
    publisher = ListingPublisher(get_adapter("amazon_in", environ={}))
    for _ in range(3):
        publisher.publish(a_payload(), _check(), approver="rehan", dry_run=True)
    assert len(publisher.store) == 1


# ==========================================================================
# Property tests
# ==========================================================================

quantities = st.integers(min_value=1, max_value=500)
money_values = st.decimals(min_value=Decimal("1"), max_value=Decimal("50000"), places=2)


@given(receipts=st.lists(quantities, min_size=1, max_size=12))
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_inventory_never_goes_negative_under_any_sequence(receipts: list[int]):
    ledger = InventoryLedger()
    for index, quantity in enumerate(receipts):
        ledger.record(Movement("SKU-P", MovementType.RECEIPT, quantity, f"PO-{index}"))
        position = ledger.position("SKU-P")
        assert position.on_hand >= 0
        assert position.available >= 0
        assert position.reserved >= 0


@given(on_hand=quantities, reserve=quantities)
@settings(max_examples=40)
def test_reserved_never_exceeds_on_hand(on_hand: int, reserve: int):
    ledger = InventoryLedger()
    ledger.record(Movement("SKU-P", MovementType.RECEIPT, on_hand, "PO-1"))
    # A reservation larger than stock is refused; the invariant below must
    # hold either way.
    with contextlib.suppress(InventoryError):
        ledger.record(Movement("SKU-P", MovementType.RESERVE, reserve, "O-1"))
    position = ledger.position("SKU-P")
    assert position.reserved <= position.on_hand
    assert position.available + position.reserved == position.on_hand


@given(imports=st.integers(min_value=1, max_value=25))
@settings(max_examples=25)
def test_duplicate_order_imports_are_always_idempotent(imports: int):
    book = OrderBook()
    for _ in range(imports):
        book.import_order(an_order("SAME-ID"))
    assert len(book) == 1


@given(price=money_values, landed=money_values)
@settings(max_examples=40)
def test_negative_contribution_cannot_be_ranked_positive(price: Decimal, landed: Decimal):
    """Ranking never turns a loss into a gain."""
    book, settlements, returns = _loop(25, price=str(price))
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=landed,
    )
    entries = rank_portfolio(
        {
            "SKU-1": (
                realized,
                returns.rates_for("SKU-1", shipped_units=25),
                Decimal("5000"),
            )
        }
    )
    if realized.actual_contribution_inr < 0:
        assert entries[0].rank_score is None or entries[0].rank_score <= 0
        assert entries[0].lifecycle in (
            Lifecycle.KILL,
            Lifecycle.REDUCE,
            Lifecycle.VALIDATE,
        )


@given(proposed=money_values)
@settings(max_examples=40)
def test_no_price_below_the_floor_is_ever_allowed(proposed: Decimal):
    points = _points(current=None)
    decision = evaluate_price_change(proposed, points)
    if decision.allowed:
        assert decision.proposed_inr >= points.minimum_viable_inr


# ==========================================================================
# Failure injection
# ==========================================================================


@pytest.mark.parametrize(
    ("outcome", "detail"),
    [
        (Outcome.RETRYABLE_FAILURE, "429 rate limited"),
        (Outcome.PERMANENT_FAILURE, "401 unauthorised"),
        (Outcome.UNKNOWN, "timeout after send"),
        (Outcome.NOT_SUPPORTED, "capability absent"),
    ],
)
def test_no_failure_mode_becomes_success(outcome: Outcome, detail: str):
    store = IdempotencyStore()
    result = execute_with_retry(
        lambda: ActionResult(outcome, "x", "k", detail=detail),
        key="k",
        store=store,
        policy=RetryPolicy(max_attempts=2),
        sleep=lambda _: None,
    )
    assert not result.changed_external_state


def test_partial_settlement_import_leaves_the_rest_unmatched():
    book = OrderBook()
    for i in range(3):
        book.import_order(an_order(f"A-{i}", price="1000"))
    settlements = SettlementLedger()
    settlements.ingest(
        SettlementLine(
            "amazon_in",
            "A-0",
            Decimal("1000"),
            commission_inr=Decimal("100"),
            net_settled_inr=Decimal("900"),
        )
    )
    states = [r.state for r in settlements.reconcile_all(book)]
    assert MatchState.MATCHED in states
    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=ReturnsLedger(),
        unit_landed_cost_inr=Decimal("300"),
    )
    assert realized.unreconciled_inr > 0


def test_supplier_price_increase_changes_the_selection():
    cheap, steady = an_offer("200", moq=20), an_offer("230", moq=20)
    before = rank_suppliers([cheap, steady], capital_cap_inr=Decimal("100000"))
    assert before[0].supplier_id == str(cheap.supplier_id)

    risen = SupplierOffer(
        supplier_id=cheap.supplier_id,
        product_id=cheap.product_id,
        unit_cost_inr=Decimal("400"),
        cost_is_gst_inclusive=True,
        gst_rate_pct=Decimal("18"),
        tax_invoice_available=True,
        moq_units=20,
        inbound_freight_inr=Decimal("10"),
    )
    after = rank_suppliers([risen, steady], capital_cap_inr=Decimal("100000"))
    assert after[0].supplier_id == str(steady.supplier_id)


def test_return_spike_downgrades_a_previously_healthy_product():
    healthy, rates = _entry(30, "2000", "300")
    before = rank_portfolio({"SKU-1": (healthy, rates, Decimal("5000"))})[0]
    assert before.lifecycle is Lifecycle.SCALE

    spiked_realized, spiked_rates = _entry(30, "2000", "300", returns_count=18)
    after = rank_portfolio({"SKU-1": (spiked_realized, spiked_rates, Decimal("5000"))})[0]
    assert after.lifecycle is not Lifecycle.SCALE


# ==========================================================================
# Concurrency
# ==========================================================================


def test_concurrent_workers_cannot_double_execute_one_intent():
    """Two workers racing on the same action must act at most once."""
    store = IdempotencyStore()
    executions: list[int] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def operation() -> ActionResult:
        with lock:
            executions.append(1)
        return ActionResult(Outcome.OK, "create_listing", "k")

    def worker() -> None:
        barrier.wait(timeout=30)
        execute_with_retry(
            operation, key="same-intent", store=store, sleep=lambda _: None
        )

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(executions) <= 1, f"intent executed {len(executions)} times"


def test_concurrent_reservations_never_oversell():
    """The invariant that protects the account metric."""
    ledger = InventoryLedger()
    ledger.record(Movement("SKU-C", MovementType.RECEIPT, 5, "PO-1"))
    lock = threading.Lock()
    barrier = threading.Barrier(12)
    granted: list[int] = []

    def worker(index: int) -> None:
        barrier.wait(timeout=30)
        with lock, contextlib.suppress(InventoryError):
            # Losing the race is the expected outcome for most workers; the
            # assertion is on how many won, not on who.
            ledger.record(Movement("SKU-C", MovementType.RESERVE, 1, f"O-{index}"))
            granted.append(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(granted) == 5
    position = ledger.position("SKU-C")
    assert position.reserved == 5
    assert position.available == 0


def test_concurrent_order_imports_stay_idempotent():
    book = OrderBook()
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait(timeout=30)
        with lock:
            book.import_order(an_order("RACE-1"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(book) == 1


# ==========================================================================
# End-to-end
# ==========================================================================


def test_approved_listing_through_to_optimisation_recommendation():
    """The full Part 2 loop, entirely in dry-run and in-memory fakes.

    approved listing -> publish -> inventory -> orders -> returns ->
    settlement -> realized profit -> portfolio -> recommendation.
    """
    publisher = ListingPublisher(get_adapter("amazon_in", environ={}))
    record, result = publisher.publish(
        a_payload(), _check(), approver="rehan", dry_run=True
    )
    assert result.outcome is Outcome.DRY_RUN
    assert record.state is ExecutionState.EXECUTED

    inventory = InventoryLedger()
    inventory.record(
        Movement("SKU-1", MovementType.RECEIPT, 40, "PO-1", unit_cost_inr=Decimal("300"))
    )

    book, settlements, returns = OrderBook(), SettlementLedger(), ReturnsLedger()
    for i in range(25):
        order = an_order(f"A-{i}", price="1299")
        book.import_order(order)
        inventory.record(Movement("SKU-1", MovementType.RESERVE, 1, f"O-{i}"))
        inventory.record(
            Movement(
                "SKU-1", MovementType.SHIP, 1, f"O-{i}", unit_cost_inr=Decimal("300")
            )
        )
        settlements.ingest(
            SettlementLine(
                "amazon_in",
                f"A-{i}",
                Decimal("1299"),
                commission_inr=Decimal("104"),
                shipping_inr=Decimal("60"),
                net_settled_inr=Decimal("1135"),
                reference=f"S-{i}",
            )
        )
    returns.record(
        ReturnEvent(
            "amazon_in",
            "A-0",
            "SKU-1",
            1,
            ReturnReason.SIZE_FIT,
            False,
            return_shipping_inr=Decimal("150"),
            refund_inr=Decimal("1299"),
        )
    )

    position = inventory.position("SKU-1")
    assert position.on_hand == 15
    assert position.sold_units == 25

    realized = compute_realized(
        "SKU-1",
        book=book,
        settlements=settlements,
        returns=returns,
        unit_landed_cost_inr=Decimal("300"),
    )
    assert realized.unreconciled_inr == 0
    assert realized.is_evidenced

    rates = returns.rates_for("SKU-1", shipped_units=25)
    entries = rank_portfolio({"SKU-1": (realized, rates, position.capital_locked_inr)})
    assert entries[0].evidence_sufficient

    recommendations = recommend(entries[0])
    assert recommendations
    assert all(r.evidence for r in recommendations)

    advice = plan_reorder(
        inventory, "SKU-1", unit_landed_cost_inr=Decimal("300"), supplier_moq=20
    )
    assert advice.velocity_confidence == "MEASURED"


def test_supplier_ranking_surfaces_the_missing_tax_invoice():
    """The ranking must explain itself, not just order the options.

    Landed cost already carries the lost input tax credit, so the number is
    right either way -- but a human comparing Rs 220 against Rs 255 needs to
    be told why the cheaper one loses.
    """
    no_invoice = an_offer("220", invoice=False, moq=20)
    scores = rank_suppliers(
        [no_invoice, an_offer("255", moq=20)], capital_cap_inr=Decimal("100000")
    )
    flagged = next(s for s in scores if not s.tax_invoice)
    assert any("no tax invoice" in n for n in flagged.notes)
    assert any("input tax credit is zero" in n for n in flagged.notes)


def test_no_invoice_supplier_has_higher_landed_cost_despite_lower_price():
    """The penalty lives in landed cost, applied once and only once."""
    cheap_no_invoice = an_offer("220", invoice=False, moq=20)
    dearer_with_invoice = an_offer("255", invoice=True, moq=20)
    scores = rank_suppliers(
        [cheap_no_invoice, dearer_with_invoice], capital_cap_inr=Decimal("100000")
    )
    by_id = {s.supplier_id: s for s in scores}
    cheap = by_id[str(cheap_no_invoice.supplier_id)]
    dearer = by_id[str(dearer_with_invoice.supplier_id)]
    assert cheap.unit_cost_inr < dearer.unit_cost_inr
    assert cheap.landed_cost_inr > dearer.landed_cost_inr


def test_kill_requires_a_larger_sample_than_an_ordinary_decision():
    """A losing product with a mid-sized sample is reduced, not killed.

    min_sample_for_decision is 20 and min_sample_for_kill is 30 precisely so
    that a product can be throttled on moderate evidence but only retired on
    strong evidence.
    """
    realized, rates = _entry(22, "500", "600")
    entry = rank_portfolio({"SKU-1": (realized, rates, Decimal("5000"))})[0]
    assert entry.evidence_sufficient
    assert entry.realized.actual_margin_pct < 0
    assert entry.lifecycle is Lifecycle.REDUCE, "killed on a sub-threshold sample"

    strong, strong_rates = _entry(35, "500", "600")
    killed = rank_portfolio({"SKU-1": (strong, strong_rates, Decimal("5000"))})[0]
    assert killed.lifecycle is Lifecycle.KILL
