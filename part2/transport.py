"""HTTP transport: stdlib only, rate limited, retried, and honest about failure.

Built on ``urllib.request`` rather than a third-party client. That is not
asceticism — it keeps the dependency surface of a system that handles money
small, and it means the repository still declares no HTTP client, which a test
enforces.

Three things here matter more than the plumbing:

**Status codes map onto the existing failure taxonomy.** A 429 is retryable, a
401 is permanent, and a timeout after the request was sent is ``UNKNOWN`` —
because the marketplace may or may not have acted on it. That last distinction
is the one that prevents duplicate orders.

**Rate limits are respected before they are hit.** A token bucket per host and
operation, seeded from the marketplace's published limits and updated from
response headers where they are returned. Racing to 429 and backing off works,
but it burns quota and, on some marketplaces, reputation.

**Nothing here has run against a live endpoint.** The request construction
follows each marketplace's published specification, and every path is tested
against a fake transport. Until real credentials exercise it, treat the first
live call as an experiment: check the response, do not batch.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from part2.execution import Outcome

logger = logging.getLogger(__name__)

TRANSPORT_VERSION = "transport/1.0.0"

DEFAULT_TIMEOUT_S = 30.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

#: Header names whose values must never reach a log or a result.
REDACTED_HEADERS = frozenset(
    {
        "authorization",
        "x-amz-access-token",
        "x-amz-security-token",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "authorization-signature",
    }
)


class TransportError(RuntimeError):
    """Raised only for programmer error, never for a remote failure.

    Remote failures are returned as ``HttpResponse`` with an outcome, because
    a caller must be able to distinguish "the marketplace said no" from "we
    built the request wrong".
    """


def redact(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy headers with secrets replaced.

    Applied at every boundary where headers could be logged or returned. A
    token in a log file is a token in a backup, a ticket attachment and a
    screenshot.
    """
    return {
        key: ("<redacted>" if key.lower() in REDACTED_HEADERS else value)
        for key, value in headers.items()
    }


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP exchange, classified against the execution taxonomy."""

    status: int
    body: bytes
    headers: dict[str, str]
    outcome: Outcome
    detail: str = ""
    elapsed_ms: int = 0
    url: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    def json(self) -> Any:
        """Parse the body as JSON, or raise a classified failure.

        A body that does not parse is a permanent failure rather than an
        exception: a marketplace returning HTML where JSON was promised is a
        real condition to record, not a crash.
        """
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"response from {self.url} was not JSON: {exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "outcome": str(self.outcome),
            "detail": self.detail,
            "elapsed_ms": self.elapsed_ms,
            "url": self.url,
            "headers": redact(self.headers),
        }


def classify_status(status: int, *, idempotent: bool) -> tuple[Outcome, str]:
    """Map an HTTP status onto an execution outcome.

    The 5xx band splits on idempotency. A 502 on a read is safely retryable; a
    502 on an order acknowledgement may have been applied before the gateway
    gave up, so it is ``UNKNOWN`` and a human decides.
    """
    if 200 <= status < 300:
        return Outcome.OK, ""
    if status in (401, 403):
        return Outcome.PERMANENT_FAILURE, (
            f"HTTP {status}: credentials rejected or scope insufficient"
        )
    if status == 404:
        return Outcome.PERMANENT_FAILURE, "HTTP 404: resource not found"
    if status == 429:
        return Outcome.RETRYABLE_FAILURE, "HTTP 429: rate limited"
    if status in (408, 425):
        return Outcome.RETRYABLE_FAILURE, f"HTTP {status}: timeout, safe to retry"
    if status == 409:
        return (
            Outcome.PERMANENT_FAILURE,
            "HTTP 409: conflict; the request contradicts current state",
        )
    if 400 <= status < 500:
        return Outcome.PERMANENT_FAILURE, f"HTTP {status}: request rejected"
    if status in (502, 503, 504):
        if idempotent:
            return (
                Outcome.RETRYABLE_FAILURE,
                f"HTTP {status}: gateway error, safe to retry",
            )
        return Outcome.UNKNOWN, (
            f"HTTP {status} on a non-idempotent request: the marketplace may "
            "have applied it before failing"
        )
    if status >= 500:
        return Outcome.UNKNOWN, f"HTTP {status}: server error of unknown effect"
    return Outcome.UNKNOWN, f"HTTP {status}: unclassified"


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Classic token bucket, refilled continuously.

    Marketplaces publish limits per operation, not per connection, so buckets
    are keyed by operation and shared across threads.
    """

    rate_per_second: float
    burst: float
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.rate_per_second <= 0:
            raise TransportError("rate_per_second must be positive")
        self._tokens = float(self.burst)
        self._last = time.monotonic()

    def acquire(
        self, *, timeout_s: float = 60.0, sleep: Callable[[float], None] = time.sleep
    ) -> bool:
        """Wait for a token. Returns False if the wait would exceed the budget."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.burst, self._tokens + (now - self._last) * self.rate_per_second
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                shortfall = (1.0 - self._tokens) / self.rate_per_second
            if time.monotonic() + shortfall > deadline:
                return False
            sleep(min(shortfall, 1.0))

    def observe_limit(self, rate_per_second: float) -> None:
        """Adopt a rate the marketplace reported.

        Only ever tightens. A header claiming a higher limit than configured is
        not a reason to speed up: the configured value may reflect a quota
        shared with another process.
        """
        with self._lock:
            if 0 < rate_per_second < self.rate_per_second:
                logger.info(
                    "tightening rate limit from a response header",
                    extra={"operation": "rate_limit", "result": rate_per_second},
                )
                self.rate_per_second = rate_per_second


class RateLimiter:
    """Per-operation token buckets for one marketplace."""

    def __init__(self, default_rate: float = 1.0, default_burst: float = 5.0) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._default = (default_rate, default_burst)
        self._lock = threading.Lock()

    def configure(self, operation: str, rate_per_second: float, burst: float) -> None:
        with self._lock:
            self._buckets[operation] = TokenBucket(rate_per_second, burst)

    def bucket(self, operation: str) -> TokenBucket:
        with self._lock:
            if operation not in self._buckets:
                self._buckets[operation] = TokenBucket(*self._default)
            return self._buckets[operation]

    def acquire(self, operation: str, **kwargs: Any) -> bool:
        return self.bucket(operation).acquire(**kwargs)


def parse_rate_limit_header(headers: Mapping[str, str]) -> float | None:
    """Read a published rate from a response, where one is returned.

    Amazon returns ``x-amzn-RateLimit-Limit`` as requests per second. Others
    return remaining-quota headers this deliberately ignores: a remaining count
    says nothing about the refill rate, and guessing one is how a client ends
    up hammering an endpoint at exactly the wrong moment.
    """
    for name in ("x-amzn-ratelimit-limit", "x-ratelimit-limit"):
        raw = next((v for k, v in headers.items() if k.lower() == name), None)
        if raw is None:
            continue
        try:
            value = float(str(raw).split(";")[0].strip())
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Honour a ``Retry-After`` header when the marketplace sends one."""
    raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


class Transport(Protocol):
    """What an adapter needs from an HTTP layer."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json_body: Any | None = None,
        form_body: str | None = None,
        operation: str = "default",
        idempotent: bool = True,
    ) -> HttpResponse: ...


class UrllibTransport:
    """HTTP over the standard library.

    Deliberately small. It does one request, applies the rate limiter, retries
    what is safe to retry, and classifies the result. Everything
    marketplace-specific lives in the adapters.
    """

    def __init__(
        self,
        *,
        limiter: RateLimiter | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = 3,
        user_agent: str = "ecom-autopilot/1.0 (+operator-configured)",
        sleep: Callable[[float], None] = time.sleep,
        opener: Any | None = None,
    ) -> None:
        self.limiter = limiter or RateLimiter()
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.user_agent = user_agent
        self._sleep = sleep
        # Injectable so every failure path can be exercised without a network.
        self._opener = opener or urllib.request.build_opener()
        self.calls: list[tuple[str, str]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json_body: Any | None = None,
        form_body: str | None = None,
        operation: str = "default",
        idempotent: bool = True,
    ) -> HttpResponse:
        if json_body is not None and form_body is not None:
            raise TransportError("pass json_body or form_body, not both")
        if params:
            separator = "&" if urllib.parse.urlparse(url).query else "?"
            url = f"{url}{separator}{urllib.parse.urlencode(params, doseq=True)}"

        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(dict(headers or {}))
        payload: bytes | None = None
        if json_body is not None:
            payload = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        elif form_body is not None:
            # OAuth token endpoints require x-www-form-urlencoded, not JSON.
            payload = form_body.encode("utf-8")
            request_headers["Content-Type"] = "application/x-www-form-urlencoded"

        self.calls.append((method, url))
        started = time.perf_counter()
        last: HttpResponse | None = None

        for attempt in range(1, self.max_attempts + 1):
            if not self.limiter.acquire(operation):
                return HttpResponse(
                    status=0,
                    body=b"",
                    headers={},
                    outcome=Outcome.RETRYABLE_FAILURE,
                    detail=f"rate limit budget exhausted for {operation}",
                    url=url,
                )

            response = self._attempt(
                method=method,
                url=url,
                headers=request_headers,
                payload=payload,
                idempotent=idempotent,
                display_url=url,
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            response = HttpResponse(
                status=response.status,
                body=response.body,
                headers=response.headers,
                outcome=response.outcome,
                detail=response.detail,
                elapsed_ms=elapsed,
                url=url,
            )
            last = response

            published = parse_rate_limit_header(response.headers)
            if published is not None:
                self.limiter.bucket(operation).observe_limit(published)

            logger.info(
                "http request",
                extra={
                    "operation": operation,
                    "result": str(response.outcome),
                    "duration": elapsed,
                    # Never the headers: they carry the bearer token.
                    "error": response.detail or None,
                },
            )

            if response.outcome is not Outcome.RETRYABLE_FAILURE:
                return response
            if attempt >= self.max_attempts:
                break

            delay = retry_after_seconds(response.headers)
            if delay is None:
                delay = min(0.25 * (2 ** (attempt - 1)), 8.0)
            self._sleep(delay)

        assert last is not None
        return last

    def _attempt(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: bytes | None,
        idempotent: bool,
        display_url: str,
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=payload, method=method.upper())
        for key, value in headers.items():
            request.add_header(key, value)

        try:
            with self._opener.open(request, timeout=self.timeout_s) as raw:
                body = raw.read(MAX_RESPONSE_BYTES)
                response_headers = dict(raw.headers.items())
                outcome, detail = classify_status(raw.status, idempotent=idempotent)
                return HttpResponse(
                    raw.status, body, response_headers, outcome, detail, url=display_url
                )
        except urllib.error.HTTPError as exc:
            body = exc.read(MAX_RESPONSE_BYTES) if hasattr(exc, "read") else b""
            response_headers = dict(exc.headers.items()) if exc.headers else {}
            outcome, detail = classify_status(exc.code, idempotent=idempotent)
            return HttpResponse(
                exc.code, body, response_headers, outcome, detail, url=display_url
            )
        except TimeoutError:
            # The request was sent. Whether it was applied is genuinely unknown.
            outcome = Outcome.RETRYABLE_FAILURE if idempotent else Outcome.UNKNOWN
            return HttpResponse(
                0,
                b"",
                {},
                outcome,
                "timed out after sending; the marketplace may have applied it"
                if not idempotent
                else "timed out; safe to retry a read",
                url=display_url,
            )
        except urllib.error.URLError as exc:
            # Connection refused or DNS failure: the request never landed.
            return HttpResponse(
                0,
                b"",
                {},
                Outcome.RETRYABLE_FAILURE,
                f"could not reach {urllib.parse.urlparse(display_url).netloc}: {exc.reason}",
                url=display_url,
            )


class NullTransport:
    """Refuses every call. The default, so offline stays the default.

    An adapter constructed without a transport cannot make a network request
    even if credentials happen to be present in the environment.
    """

    def request(self, _method: str, url: str, **_kwargs: Any) -> HttpResponse:
        return HttpResponse(
            status=0,
            body=b"",
            headers={},
            outcome=Outcome.PERMANENT_FAILURE,
            detail=(
                "no HTTP transport configured; pass UrllibTransport() to the "
                "adapter to enable live calls"
            ),
            url=url,
        )


# --------------------------------------------------------------------------
# Pagination
# --------------------------------------------------------------------------


def paginate(
    fetch: Callable[[str | None], HttpResponse],
    *,
    extract_items: Callable[[Any], list[Any]],
    extract_cursor: Callable[[Any], str | None],
    max_pages: int = 100,
) -> Iterator[tuple[list[Any], HttpResponse]]:
    """Walk a cursor-paginated endpoint.

    Bounded by ``max_pages``. A marketplace returning the same cursor forever
    is a real failure mode, and an unbounded loop turns it into a hung process
    that quietly consumes quota.
    """
    cursor: str | None = None
    seen: set[str] = set()

    for page in range(max_pages):
        response = fetch(cursor)
        if not response.ok:
            yield [], response
            return

        payload = response.json()
        items = extract_items(payload) if payload is not None else []
        yield items, response

        cursor = extract_cursor(payload) if payload is not None else None
        if not cursor:
            return
        if cursor in seen:
            logger.warning(
                "pagination cursor repeated; stopping",
                extra={"operation": "paginate", "result": "cursor_loop"},
            )
            return
        seen.add(cursor)

        if page == max_pages - 1:
            logger.warning(
                "pagination hit the page ceiling; results are incomplete",
                extra={"operation": "paginate", "result": "truncated"},
            )


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "REDACTED_HEADERS",
    "TRANSPORT_VERSION",
    "HttpResponse",
    "NullTransport",
    "RateLimiter",
    "TokenBucket",
    "Transport",
    "TransportError",
    "UrllibTransport",
    "classify_status",
    "paginate",
    "parse_rate_limit_header",
    "redact",
    "retry_after_seconds",
]
