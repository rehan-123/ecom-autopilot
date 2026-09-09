"""Integration tests for the live HTTP transports, against a fake network.

Every path here is exercised without touching a real endpoint. That is a
limitation worth stating rather than hiding: these tests prove the request is
built as each marketplace's specification describes, that failures classify
correctly, and that secrets never escape — they do not prove any marketplace
accepts the request. The first live call remains an experiment.

The fake transport records what would have been sent, which is what makes the
request-construction assertions possible at all.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from part2.auth import (
    CREDENTIAL_SPECS,
    AccessToken,
    CredentialsMissing,
    CredentialSpec,
    OAuth2TokenProvider,
    StaticTokenProvider,
    build_authorization_header,
    build_signing_string,
    credential_report,
    flipkart_provider,
    lwa_provider,
)
from part2.execution import Outcome
from part2.marketplace import ListingPayload, get_adapter
from part2.normalize import (
    NormalizationError,
    amazon_order,
    amazon_settlement,
    flipkart_order,
    flipkart_settlement,
    normalize_orders,
    normalize_settlements,
)
from part2.operations import OrderStatus
from part2.transport import (
    REDACTED_HEADERS,
    HttpResponse,
    NullTransport,
    RateLimiter,
    TokenBucket,
    TransportError,
    UrllibTransport,
    classify_status,
    paginate,
    parse_rate_limit_header,
    redact,
    retry_after_seconds,
)

NOW = datetime.now(UTC)


class FakeTransport:
    """Records requests and replays scripted responses.

    Deliberately not a mock library: the assertions are about the exact URL,
    headers and body the adapters construct, and a recorded list is the
    clearest way to check that.
    """

    def __init__(self, responses: list[HttpResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[dict] = []

    def request(
        self,
        method,
        url,
        *,
        headers=None,
        params=None,
        json_body=None,
        form_body=None,
        operation="default",
        idempotent=True,
    ):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "params": dict(params or {}),
                "json_body": json_body,
                "form_body": form_body,
                "operation": operation,
                "idempotent": idempotent,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return ok_response({})


def ok_response(
    payload: dict, *, status: int = 200, headers: dict | None = None
) -> HttpResponse:
    return HttpResponse(
        status=status,
        body=json.dumps(payload).encode("utf-8"),
        headers=headers or {},
        outcome=Outcome.OK,
        url="https://example.test/x",
    )


def error_response(
    status: int, *, idempotent: bool = True, headers: dict | None = None
) -> HttpResponse:
    outcome, detail = classify_status(status, idempotent=idempotent)
    return HttpResponse(
        status, b"{}", headers or {}, outcome, detail, url="https://example.test/x"
    )


# ==========================================================================
# Status classification
# ==========================================================================


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, Outcome.OK),
        (204, Outcome.OK),
        (400, Outcome.PERMANENT_FAILURE),
        (401, Outcome.PERMANENT_FAILURE),
        (403, Outcome.PERMANENT_FAILURE),
        (404, Outcome.PERMANENT_FAILURE),
        (409, Outcome.PERMANENT_FAILURE),
        (429, Outcome.RETRYABLE_FAILURE),
    ],
)
def test_status_classification(status: int, expected: Outcome):
    assert classify_status(status, idempotent=True)[0] is expected


def test_gateway_error_splits_on_idempotency():
    """A 502 on a write may have been applied before the gateway gave up."""
    assert classify_status(502, idempotent=True)[0] is Outcome.RETRYABLE_FAILURE
    assert classify_status(502, idempotent=False)[0] is Outcome.UNKNOWN


def test_no_failure_status_is_ever_success():
    for status in (400, 401, 403, 404, 409, 429, 500, 502, 503, 504):
        assert not classify_status(status, idempotent=True)[0].is_success


# ==========================================================================
# Secrets never escape
# ==========================================================================


def test_authorization_headers_are_redacted():
    headers = {"Authorization": "Bearer super-secret", "Accept": "application/json"}
    assert redact(headers)["Authorization"] == "<redacted>"
    assert redact(headers)["Accept"] == "application/json"


def test_every_credential_carrying_header_is_redacted():
    for name in ("authorization", "x-amz-access-token", "cookie", "x-api-key"):
        assert name in REDACTED_HEADERS
        assert redact({name: "secret"})[name] == "<redacted>"


def test_response_dict_never_exposes_a_token():
    response = HttpResponse(
        200, b"{}", {"Authorization": "Bearer leaked-value"}, Outcome.OK
    )
    assert "leaked-value" not in json.dumps(response.as_dict())


def test_credential_report_names_variables_not_values():
    report = credential_report({"AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER-secret"})
    assert "AMAZON_SP_REFRESH_TOKEN" in report
    assert "PLACEHOLDER-secret" not in report


# ==========================================================================
# Rate limiting
# ==========================================================================


def test_token_bucket_allows_a_burst_then_throttles():
    bucket = TokenBucket(rate_per_second=1.0, burst=3.0)
    slept: list[float] = []
    assert all(bucket.acquire(sleep=slept.append) for _ in range(3))
    bucket.acquire(sleep=slept.append)
    assert slept, "the fourth request should have waited"


def test_token_bucket_refuses_when_the_wait_exceeds_the_budget():
    bucket = TokenBucket(rate_per_second=0.01, burst=1.0)
    assert bucket.acquire(sleep=lambda _: None)
    assert not bucket.acquire(timeout_s=1.0, sleep=lambda _: None)


def test_rate_limit_is_only_ever_tightened():
    """A header claiming a higher limit is not a reason to speed up.

    The configured value may reflect a quota shared with another process.
    """
    bucket = TokenBucket(rate_per_second=1.0, burst=5.0)
    bucket.observe_limit(0.5)
    assert bucket.rate_per_second == 0.5
    bucket.observe_limit(10.0)
    assert bucket.rate_per_second == 0.5


def test_rate_limit_header_is_parsed():
    assert parse_rate_limit_header({"x-amzn-RateLimit-Limit": "0.0166"}) == pytest.approx(
        0.0166
    )
    assert parse_rate_limit_header({"X-RateLimit-Limit": "5"}) == 5.0
    assert parse_rate_limit_header({"unrelated": "x"}) is None


def test_remaining_quota_headers_are_deliberately_ignored():
    """A remaining count says nothing about the refill rate."""
    assert parse_rate_limit_header({"x-ratelimit-remaining": "3"}) is None


def test_retry_after_is_honoured_when_present():
    assert retry_after_seconds({"Retry-After": "2.5"}) == 2.5
    assert retry_after_seconds({"Retry-After": "not a number"}) is None
    assert retry_after_seconds({}) is None


def test_rate_limiter_configures_per_operation():
    limiter = RateLimiter()
    limiter.configure("get_orders", 0.5, 2.0)
    assert limiter.bucket("get_orders").rate_per_second == 0.5
    assert limiter.bucket("other") is not limiter.bucket("get_orders")


def test_buckets_are_shared_across_threads():
    limiter = RateLimiter(default_rate=100.0, default_burst=100.0)
    seen: list[int] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait(timeout=20)
        seen.append(id(limiter.bucket("shared")))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert len(set(seen)) == 1, "one bucket per operation, shared"


# ==========================================================================
# Retries
# ==========================================================================


def test_retryable_failure_is_retried_then_succeeds():
    transport = UrllibTransport(sleep=lambda _: None)
    transport._opener = _FakeOpener(
        [_raise(429), _raise(429), _reply(200, b'{"ok":true}')]
    )
    response = transport.request("GET", "https://example.test/x")
    assert response.ok
    assert len(transport._opener.calls) == 3


def test_retries_are_bounded():
    transport = UrllibTransport(max_attempts=2, sleep=lambda _: None)
    transport._opener = _FakeOpener([_raise(503), _raise(503), _raise(503)])
    response = transport.request("GET", "https://example.test/x")
    assert not response.ok
    assert len(transport._opener.calls) == 2


def test_permanent_failure_is_not_retried():
    transport = UrllibTransport(sleep=lambda _: None)
    transport._opener = _FakeOpener([_raise(401)])
    response = transport.request("GET", "https://example.test/x")
    assert response.outcome is Outcome.PERMANENT_FAILURE
    assert len(transport._opener.calls) == 1


def test_timeout_on_a_write_is_unknown_not_failure():
    """The request was sent; whether it was applied is genuinely unknown."""
    transport = UrllibTransport(sleep=lambda _: None)
    transport._opener = _FakeOpener([TimeoutError("timed out")])
    response = transport.request("POST", "https://example.test/x", idempotent=False)
    assert response.outcome is Outcome.UNKNOWN


def test_timeout_on_a_read_is_retryable():
    transport = UrllibTransport(max_attempts=1, sleep=lambda _: None)
    transport._opener = _FakeOpener([TimeoutError("timed out")])
    response = transport.request("GET", "https://example.test/x", idempotent=True)
    assert response.outcome is Outcome.RETRYABLE_FAILURE


def test_form_and_json_bodies_are_mutually_exclusive():
    transport = UrllibTransport()
    with pytest.raises(TransportError, match="not both"):
        transport.request("POST", "https://x.test", json_body={}, form_body="a=b")


# ==========================================================================
# Offline stays the default
# ==========================================================================


def test_null_transport_refuses_and_explains():
    response = NullTransport().request("GET", "https://example.test")
    assert response.outcome is Outcome.PERMANENT_FAILURE
    assert "no HTTP transport configured" in response.detail


def test_adapters_are_offline_unless_a_transport_is_supplied():
    """Credentials alone must not enable a network call."""
    adapter = get_adapter(
        "amazon_in",
        environ={
            "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_ID": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER",
        },
    )
    assert not adapter.live
    result = adapter.get_orders(since=NOW)
    assert not result.changed_external_state


def test_dry_run_still_makes_no_request_even_with_a_live_transport():
    fake = FakeTransport()
    adapter = get_adapter("amazon_in", environ={}, transport=fake)
    payload = ListingPayload(
        sku_code="SKU-1",
        title="t",
        description="d",
        bullets=(),
        price_inr=Decimal("100"),
        quantity=1,
    )
    result = adapter.create_listing(payload, dry_run=True)
    assert result.outcome is Outcome.DRY_RUN
    assert fake.requests == [], "dry run must not reach the transport"


def test_unsupported_capability_never_reaches_the_transport():
    fake = FakeTransport()
    adapter = get_adapter("meesho", environ={}, transport=fake)
    result = adapter.update_price("SKU-1", Decimal("999"), dry_run=False)
    assert result.outcome is Outcome.NOT_SUPPORTED
    assert fake.requests == []


# ==========================================================================
# Authentication
# ==========================================================================


def test_token_is_cached_between_calls():
    fake = FakeTransport([ok_response({"access_token": "abc", "expires_in": 3600})])
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={"grant_type": "x"},
        transport=fake,
        marketplace="test",
    )
    assert provider.token() == "abc"
    assert provider.token() == "abc"
    assert provider.refreshes == 1, "a cached token must not be refetched"


def test_expiring_token_is_refreshed_early():
    """A token that expires mid-flight produces a 401 that looks like bad creds."""
    fake = FakeTransport(
        [
            ok_response({"access_token": "first", "expires_in": 60}),
            ok_response({"access_token": "second", "expires_in": 3600}),
        ]
    )
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={},
        transport=fake,
        marketplace="test",
    )
    assert provider.token() == "first"
    assert provider.token() == "second"


def test_invalidate_forces_a_refresh():
    fake = FakeTransport(
        [
            ok_response({"access_token": "one", "expires_in": 3600}),
            ok_response({"access_token": "two", "expires_in": 3600}),
        ]
    )
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={},
        transport=fake,
        marketplace="test",
    )
    assert provider.token() == "one"
    provider.invalidate()
    assert provider.token() == "two"


def test_concurrent_callers_trigger_one_refresh():
    """Without the lock, a burst produces a cascade of competing refreshes."""
    fake = FakeTransport([ok_response({"access_token": "x", "expires_in": 3600})] * 20)
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={},
        transport=fake,
        marketplace="test",
    )
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait(timeout=20)
        provider.token()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert provider.refreshes == 1


def test_token_endpoint_receives_a_form_body_not_json():
    """OAuth token endpoints reject JSON; this is a real interop detail."""
    fake = FakeTransport([ok_response({"access_token": "x", "expires_in": 60})])
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={"grant_type": "refresh_token", "refresh_token": "r"},
        transport=fake,
        marketplace="test",
    )
    provider.token()
    sent = fake.requests[0]
    assert sent["json_body"] is None
    assert "grant_type=refresh_token" in sent["form_body"]


def test_failed_refresh_raises_rather_than_returning_an_empty_token():
    fake = FakeTransport([error_response(401)])
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={},
        transport=fake,
        marketplace="test",
    )
    with pytest.raises(CredentialsMissing, match="token refresh failed"):
        provider.token()


def test_token_response_without_a_token_is_refused():
    fake = FakeTransport([ok_response({"expires_in": 3600})])
    provider = OAuth2TokenProvider(
        token_url="https://auth.test/token",
        form_fields={},
        transport=fake,
        marketplace="test",
    )
    with pytest.raises(CredentialsMissing, match="no access_token"):
        provider.token()


def test_missing_credentials_are_named():
    spec = CredentialSpec("test", ("A_KEY", "B_KEY"))
    assert spec.missing({"A_KEY": "x"}) == ["B_KEY"]
    with pytest.raises(CredentialsMissing, match="B_KEY"):
        spec.read({"A_KEY": "x"})


def test_every_marketplace_declares_its_credentials():
    for name in ("amazon_in", "flipkart", "ondc"):
        assert CREDENTIAL_SPECS[name].required


def test_lwa_provider_builds_a_refresh_token_grant():
    fake = FakeTransport([ok_response({"access_token": "x", "expires_in": 60})])
    provider = lwa_provider(
        fake,
        environ={
            "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER-r",
            "AMAZON_SP_CLIENT_ID": "PLACEHOLDER-c",
            "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER-s",
        },
    )
    provider.token()
    assert "grant_type=refresh_token" in fake.requests[0]["form_body"]
    assert fake.requests[0]["url"] == "https://api.amazon.com/auth/o2/token"


def test_flipkart_provider_builds_a_client_credentials_grant():
    fake = FakeTransport([ok_response({"access_token": "x", "expires_in": 60})])
    provider = flipkart_provider(
        fake,
        environ={
            "FLIPKART_CLIENT_ID": "PLACEHOLDER-c",
            "FLIPKART_CLIENT_SECRET": "PLACEHOLDER-s",
        },
    )
    provider.token()
    assert "grant_type=client_credentials" in fake.requests[0]["form_body"]


def test_static_token_provider_is_usable_and_inert():
    provider = StaticTokenProvider("supplied-out-of-band")
    assert provider.token() == "supplied-out-of-band"
    provider.invalidate()
    assert provider.token() == "supplied-out-of-band"


def test_access_token_freshness_accounts_for_the_margin():
    import time as time_module

    assert not AccessToken("x", time_module.time() + 30).is_fresh()
    assert AccessToken("x", time_module.time() + 3600).is_fresh()


def test_ondc_signing_string_matches_the_specification():
    """Field order and line endings are specified and significant."""
    signing = build_signing_string(
        digest_base64="ZGln", created=1700000000, expires=1700000030
    )
    assert signing == (
        "(created): 1700000000\n(expires): 1700000030\ndigest: BLAKE-512=ZGln"
    )


def test_ondc_authorization_header_shape():
    header = build_authorization_header(
        subscriber_id="sub.example",
        unique_key_id="key1",
        signature="sig",
        created=1,
        expires=2,
    )
    assert 'keyId="sub.example|key1|ed25519"' in header
    assert 'algorithm="ed25519"' in header
    assert 'headers="(created) (expires) digest"' in header


# ==========================================================================
# Adapter wiring
# ==========================================================================


def _live_amazon(responses=None):
    fake = FakeTransport(responses)
    adapter = get_adapter(
        "amazon_in",
        environ={
            "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_ID": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER",
            "AMAZON_SP_MARKETPLACE_ID": "A21TJRUUN4KGV",
        },
        transport=fake,
        tokens=StaticTokenProvider("token-value"),
    )
    return adapter, fake


def test_amazon_uses_its_own_auth_header_not_bearer():
    """SP-API expects x-amz-access-token, not Authorization: Bearer."""
    adapter, fake = _live_amazon([ok_response({"payload": {"Orders": []}})])
    adapter.get_orders(since=NOW)
    headers = fake.requests[0]["headers"]
    assert headers["x-amz-access-token"] == "token-value"
    assert "Authorization" not in headers


def test_amazon_get_orders_builds_the_documented_request():
    adapter, fake = _live_amazon([ok_response({"payload": {"Orders": []}})])
    result = adapter.get_orders(since=NOW)
    sent = fake.requests[0]
    assert sent["method"] == "GET"
    assert sent["url"].endswith("/orders/v0/orders")
    assert sent["params"]["MarketplaceIds"] == "A21TJRUUN4KGV"
    assert "CreatedAfter" in sent["params"]
    assert result.outcome is Outcome.OK


def test_a_401_invalidates_the_token_and_retries_once():
    """Clock skew and server-side revocation both produce a 401 on a token
    the client believed valid."""

    class CountingProvider:
        def __init__(self) -> None:
            self.invalidations = 0

        def token(self) -> str:
            return "t"

        def invalidate(self) -> None:
            self.invalidations += 1

    provider = CountingProvider()
    fake = FakeTransport([error_response(401), ok_response({"payload": {}})])
    adapter = get_adapter(
        "amazon_in",
        environ={
            "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_ID": "PLACEHOLDER",
            "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER",
        },
        transport=fake,
        tokens=provider,
    )
    result = adapter.get_orders(since=NOW)
    assert provider.invalidations == 1
    assert len(fake.requests) == 2
    assert result.outcome is Outcome.OK


def test_flipkart_posts_a_shipment_filter():
    fake = FakeTransport([ok_response({"shipments": []})])
    adapter = get_adapter(
        "flipkart",
        environ={
            "FLIPKART_CLIENT_ID": "PLACEHOLDER",
            "FLIPKART_CLIENT_SECRET": "PLACEHOLDER",
        },
        transport=fake,
        tokens=StaticTokenProvider("tok"),
    )
    adapter.get_orders(since=NOW)
    sent = fake.requests[0]
    assert sent["method"] == "POST"
    assert sent["url"].endswith("/v3/shipments/filter")
    assert sent["headers"]["Authorization"] == "Bearer tok"


def test_unparseable_body_is_a_permanent_failure_not_a_crash():
    fake = FakeTransport(
        [HttpResponse(200, b"<html>maintenance</html>", {}, Outcome.OK, url="x")]
    )
    adapter = get_adapter(
        "flipkart",
        environ={
            "FLIPKART_CLIENT_ID": "PLACEHOLDER",
            "FLIPKART_CLIENT_SECRET": "PLACEHOLDER",
        },
        transport=fake,
        tokens=StaticTokenProvider("tok"),
    )
    result = adapter.get_orders(since=NOW)
    assert result.outcome is Outcome.PERMANENT_FAILURE
    assert "unparseable" in result.detail


def test_published_rate_limits_seed_the_limiter():
    limiter = RateLimiter()
    get_adapter("amazon_in", environ={}).configure_limits(limiter)
    assert limiter.bucket("get_orders").rate_per_second == pytest.approx(0.0166)


# ==========================================================================
# Pagination
# ==========================================================================


def test_pagination_walks_every_page():
    pages = [
        ok_response({"items": [1, 2], "next": "c1"}),
        ok_response({"items": [3], "next": None}),
    ]
    collected = []
    for items, _response in paginate(
        lambda _cursor: pages.pop(0),
        extract_items=lambda p: p["items"],
        extract_cursor=lambda p: p.get("next"),
    ):
        collected.extend(items)
    assert collected == [1, 2, 3]


def test_pagination_stops_on_a_repeated_cursor():
    """A marketplace returning the same cursor forever is a real failure mode."""
    collected = []
    for items, _response in paginate(
        lambda _cursor: ok_response({"items": [1], "next": "same"}),
        extract_items=lambda p: p["items"],
        extract_cursor=lambda p: p["next"],
        max_pages=50,
    ):
        collected.extend(items)
    assert len(collected) == 2, "one page, then the repeat is detected"


def test_pagination_is_bounded():
    counter = {"n": 0}

    def fetch(_cursor):
        counter["n"] += 1
        return ok_response({"items": [1], "next": f"c{counter['n']}"})

    list(
        paginate(
            fetch,
            extract_items=lambda p: p["items"],
            extract_cursor=lambda p: p["next"],
            max_pages=5,
        )
    )
    assert counter["n"] == 5


def test_pagination_stops_on_a_failed_page():
    responses = [ok_response({"items": [1], "next": "c1"}), error_response(500)]
    pages = list(
        paginate(
            lambda _c: responses.pop(0),
            extract_items=lambda p: p["items"],
            extract_cursor=lambda p: p.get("next"),
        )
    )
    assert not pages[-1][1].ok


# ==========================================================================
# Normalization
# ==========================================================================


AMAZON_ORDER = {
    "AmazonOrderId": "403-1234567-1234567",
    "PurchaseDate": "2026-08-01T10:00:00Z",
    "OrderStatus": "Shipped",
    "ShippingPrice": {"Amount": "60.00"},
}
AMAZON_ITEMS = [
    {
        "SellerSKU": "SKU-TRIM-880",
        "QuantityOrdered": 2,
        "ItemPrice": {"Amount": "2598.00"},
        "ItemTax": {"Amount": "396.00"},
    }
]


def test_amazon_order_normalizes_to_the_domain_model():
    order = amazon_order(AMAZON_ORDER, AMAZON_ITEMS)
    assert order.marketplace == "amazon_in"
    assert order.status is OrderStatus.SHIPPED
    assert order.units == 2
    # SP-API reports a line total; the domain wants a unit price.
    assert order.lines[0].unit_price_inr == Decimal("1299.00")


def test_amazon_order_without_items_is_refused():
    """An order with no items must not be recorded as a zero-value sale."""
    with pytest.raises(NormalizationError, match="no line items"):
        amazon_order(AMAZON_ORDER, [])


def test_amazon_order_with_an_unmapped_status_is_refused():
    payload = {**AMAZON_ORDER, "OrderStatus": "QuantumSuperposition"}
    with pytest.raises(NormalizationError, match="unmapped marketplace status"):
        amazon_order(payload, AMAZON_ITEMS)


def test_item_without_a_sku_is_refused():
    with pytest.raises(NormalizationError, match="no SellerSKU"):
        amazon_order(
            AMAZON_ORDER, [{"QuantityOrdered": 1, "ItemPrice": {"Amount": "10"}}]
        )


def test_missing_price_is_an_error_not_a_zero():
    """A silently zeroed amount flows straight into realized profit."""
    with pytest.raises(NormalizationError, match="no safe default"):
        amazon_order(AMAZON_ORDER, [{"SellerSKU": "S", "QuantityOrdered": 1}])


def test_absent_optional_amount_defaults_to_zero():
    order = amazon_order(
        {**AMAZON_ORDER, "ShippingPrice": {}},
        [{"SellerSKU": "S", "QuantityOrdered": 1, "ItemPrice": {"Amount": "100"}}],
    )
    assert order.shipping_inr == Decimal("0")


def test_amazon_settlement_normalizes_signs():
    """Fees arrive negative; every deduction must end up positive."""
    line = amazon_settlement(
        {
            "order-id": "403-1",
            "settlement-id": "S-1",
            "posted-date": "2026-08-05T00:00:00Z",
            "principal": "1299.00",
            "commission": "-104.00",
            "shipping-fee": "-60.00",
            "tcs-cgst": "-6.50",
            "tcs-sgst": "-6.50",
            "total": "1122.00",
        }
    )
    assert line.commission_inr == Decimal("104.00")
    assert line.tax_withheld_inr == Decimal("13.00")
    assert line.net_settled_inr == Decimal("1122.00")


def test_flipkart_order_normalizes():
    order = flipkart_order(
        {
            "shipmentId": "SHP-1",
            "orderDate": "2026-08-01T10:00:00+05:30",
            "status": "shipped",
            "orderItems": [
                {
                    "sku": "SKU-1",
                    "quantity": 1,
                    "priceComponents": {"sellingPrice": "599", "totalDiscount": "50"},
                }
            ],
            "priceComponents": {"shippingCharge": "40"},
        }
    )
    assert order.marketplace == "flipkart"
    assert order.lines[0].discount_inr == Decimal("50")
    assert order.shipping_inr == Decimal("40")


def test_flipkart_settlement_normalizes():
    line = flipkart_settlement(
        {
            "orderItemId": "OI-1",
            "settlementId": "FS-1",
            "settlementDate": "2026-08-06T00:00:00Z",
            "orderValue": "599",
            "commission": "-48",
            "shippingFee": "-40",
            "settlementValue": "511",
        }
    )
    assert line.commission_inr == Decimal("48")
    assert line.net_settled_inr == Decimal("511")


def test_one_malformed_order_does_not_discard_the_page():
    """Forty-nine good orders must survive one bad one."""
    orders, failures = normalize_orders(
        "flipkart",
        [
            {
                "shipmentId": "GOOD",
                "orderDate": "2026-08-01T10:00:00Z",
                "status": "shipped",
                "orderItems": [
                    {
                        "sku": "S",
                        "quantity": 1,
                        "priceComponents": {"sellingPrice": "100"},
                    }
                ],
            },
            {
                "shipmentId": "BAD",
                "orderDate": "2026-08-01T10:00:00Z",
                "status": "shipped",
                "orderItems": [],
            },
        ],
    )
    assert len(orders) == 1
    assert len(failures) == 1
    assert "BAD" in failures[0]


def test_settlement_normalization_collects_failures():
    lines, failures = normalize_settlements(
        "flipkart",
        [
            {
                "orderItemId": "OK",
                "settlementDate": "2026-08-06T00:00:00Z",
                "settlementValue": "100",
            },
            {"settlementValue": "100"},
        ],
    )
    assert len(lines) == 1
    assert len(failures) == 1


def test_unknown_marketplace_has_no_normalizer():
    with pytest.raises(NormalizationError, match="no order normalizer"):
        normalize_orders("shopee", [])


def test_normalized_money_is_always_decimal():
    order = amazon_order(AMAZON_ORDER, AMAZON_ITEMS)
    assert isinstance(order.lines[0].unit_price_inr, Decimal)
    assert isinstance(order.shipping_inr, Decimal)


# ==========================================================================
# Fake urllib opener
# ==========================================================================


class _FakeResponse:
    def __init__(self, status: int, body: bytes, headers: dict | None = None) -> None:
        self.status = status
        self._body = body
        self.headers = _Headers(headers or {})

    def read(self, _n: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Headers(dict):
    def items(self):
        return super().items()


def _reply(status: int, body: bytes):
    return _FakeResponse(status, body)


def _raise(status: int):
    import urllib.error

    return urllib.error.HTTPError(
        "https://example.test/x", status, f"HTTP {status}", _Headers({}), None
    )


class _FakeOpener:
    """Replays scripted urllib outcomes."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list = []

    def open(self, request, timeout=None):
        self.calls.append(request)
        item = self.script.pop(0) if self.script else _reply(200, b"{}")
        if isinstance(item, Exception):
            raise item
        return item
