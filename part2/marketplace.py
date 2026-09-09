"""Marketplace adapters: one contract, explicit capabilities, no pretending.

An adapter declares what it can legitimately do. Anything it cannot do returns
``NOT_SUPPORTED`` — never a fabricated success, and never a silent no-op. That
distinction matters more than it looks: a fake success propagates into
inventory, into settlement reconciliation, and eventually into a profit figure
that is simply wrong.

None of the adapters here carry a transport. Amazon, Flipkart, Meesho and ONDC
all require seller registration, approved developer credentials and signed
requests; none of that can be bundled. So each adapter declares its real
capability surface and reports ``DATA_UNAVAILABLE`` or ``NOT_SUPPORTED`` until
credentials and a transport are supplied by the operator.

That is not a stub in the dismissive sense. The capability model, the dry-run
path, the idempotency contract, the failure classification and the response
verification are all real and tested; what is absent is a network client that
would be illegal to guess at.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from econ.money import format_inr, money
from part2.auth import TokenProvider
from part2.execution import ActionResult, Outcome, idempotency_key
from part2.transport import NullTransport, Transport

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "marketplace/1.0.0"


class Capability(StrEnum):
    """One operation an adapter may or may not be able to perform."""

    GET_CATALOG = "GET_CATALOG"
    GET_LISTING = "GET_LISTING"
    CREATE_LISTING = "CREATE_LISTING"
    UPDATE_LISTING = "UPDATE_LISTING"
    UPDATE_PRICE = "UPDATE_PRICE"
    UPDATE_INVENTORY = "UPDATE_INVENTORY"
    GET_ORDERS = "GET_ORDERS"
    GET_ORDER = "GET_ORDER"
    GET_INVENTORY = "GET_INVENTORY"
    GET_RETURNS = "GET_RETURNS"
    GET_SETTLEMENTS = "GET_SETTLEMENTS"


class AccessModel(StrEnum):
    """How access to a marketplace is legitimately obtained."""

    SELLER_API = "SELLER_API"
    AUTHORISED_FEED = "AUTHORISED_FEED"
    NETWORK_PARTICIPANT = "NETWORK_PARTICIPANT"
    MANUAL_EXPORT = "MANUAL_EXPORT"
    NONE = "NONE"


class CredentialsMissing(RuntimeError):
    """Raised when an operation needs credentials that are not configured."""


@dataclass(frozen=True)
class MarketplaceProfile:
    """What one marketplace genuinely offers."""

    name: str
    access: AccessModel
    capabilities: frozenset[Capability]
    credential_env_vars: tuple[str, ...]
    notes: str

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def credentials_present(self, environ: dict[str, str] | None = None) -> bool:
        """Whether every required variable is set.

        Only presence is checked, never the value, and values are never logged
        or returned. Credentials live in the environment; nothing in this
        repository stores or transmits one.
        """
        env = environ if environ is not None else dict(os.environ)
        return bool(self.credential_env_vars) and all(
            env.get(name) for name in self.credential_env_vars
        )


@dataclass(frozen=True)
class ListingPayload:
    """Everything a marketplace needs to create or update a listing."""

    sku_code: str
    title: str
    description: str
    bullets: tuple[str, ...]
    price_inr: Decimal
    quantity: int
    attributes: dict[str, Any] = field(default_factory=dict)
    images: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        """Stable identity for this exact listing content.

        Used as the idempotency key so republishing identical content is a
        no-op while a genuine edit is a new intent.
        """
        return idempotency_key(
            self.sku_code,
            self.title,
            self.description,
            "|".join(self.bullets),
            str(money(self.price_inr)),
            self.quantity,
            "|".join(f"{k}={v}" for k, v in sorted(self.attributes.items())),
        )


@dataclass(frozen=True)
class RemoteListing:
    """A listing as the marketplace reports it."""

    listing_id: str
    sku_code: str
    status: str
    title: str
    price_inr: Decimal
    quantity: int
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MarketplaceAdapter(Protocol):
    """The contract every channel implements."""

    profile: MarketplaceProfile

    def create_listing(
        self, payload: ListingPayload, *, dry_run: bool
    ) -> ActionResult: ...
    def update_price(
        self, sku_code: str, price_inr: Decimal, *, dry_run: bool
    ) -> ActionResult: ...
    def update_inventory(
        self, sku_code: str, quantity: int, *, dry_run: bool
    ) -> ActionResult: ...
    def get_listing(self, sku_code: str) -> ActionResult: ...
    def get_orders(self, *, since: datetime) -> ActionResult: ...
    def get_settlements(self, *, since: datetime) -> ActionResult: ...


class BaseAdapter:
    """Shared behaviour: capability checks, dry-run, credential gating.

    Every operation passes through ``_guard``, so an unsupported call, a
    missing credential and a dry run are handled identically for every
    marketplace instead of being reimplemented per adapter and drifting.
    """

    profile: MarketplaceProfile

    #: Published per-operation rate limits, requests per second. Conservative
    #: where a marketplace does not publish one: exceeding a limit costs quota
    #: and, on some channels, standing.
    RATE_LIMITS: ClassVar[dict[str, tuple[float, float]]] = {
        "get_orders": (0.0166, 20.0),
        "get_order": (0.5, 30.0),
        "get_settlements": (0.0222, 10.0),
        "create_listing": (5.0, 10.0),
        "update_price": (5.0, 10.0),
        "update_inventory": (5.0, 30.0),
        "get_listing": (2.0, 20.0),
    }

    #: Base URL for live calls. Overridable per marketplace via the
    #: environment, because sandbox and regional endpoints differ.
    BASE_URL: str = ""
    ENDPOINT_ENV_VAR: str = ""

    def __init__(
        self,
        *,
        environ: dict[str, str] | None = None,
        transport: Transport | None = None,
        tokens: TokenProvider | None = None,
    ) -> None:
        self._environ = environ
        # Offline stays the default: without an explicitly supplied transport
        # an adapter cannot make a network call even if credentials happen to
        # be present in the environment.
        self._transport = transport or NullTransport()
        self._tokens = tokens

    @property
    def live(self) -> bool:
        """Whether this adapter could actually reach the marketplace."""
        return not isinstance(self._transport, NullTransport)

    def base_url(self) -> str:
        env = self._environ if self._environ is not None else dict(os.environ)
        return env.get(self.ENDPOINT_ENV_VAR or "", "") or self.BASE_URL

    def auth_headers(self) -> dict[str, str]:
        """Authentication headers for a live request.

        Overridden per marketplace. The token itself is fetched here and never
        stored on the adapter, so it cannot end up in a repr or a pickle.
        """
        if self._tokens is None:
            return {}
        return {"Authorization": f"Bearer {self._tokens.token()}"}

    def configure_limits(self, limiter: Any) -> None:
        """Seed a rate limiter with this marketplace's published limits."""
        for operation, (rate, burst) in self.RATE_LIMITS.items():
            limiter.configure(operation, rate, burst)

    # -- guards -----------------------------------------------------------

    def _guard(
        self,
        capability: Capability,
        action: str,
        key: str,
        *,
        dry_run: bool,
        plan: str,
    ) -> ActionResult | None:
        """Return a terminal result, or None if the caller may proceed."""
        if not self.profile.supports(capability):
            return ActionResult(
                outcome=Outcome.NOT_SUPPORTED,
                action=action,
                idempotency_key=key,
                detail=(
                    f"{self.profile.name} does not support {capability}: "
                    f"{self.profile.notes}"
                ),
            )
        if dry_run:
            logger.info(
                "dry run", extra={"operation": action, "marketplace": self.profile.name}
            )
            return ActionResult(
                outcome=Outcome.DRY_RUN,
                action=action,
                idempotency_key=key,
                detail=plan,
            )
        if not self.profile.credentials_present(self._environ):
            missing = ", ".join(self.profile.credential_env_vars) or "none declared"
            return ActionResult(
                outcome=Outcome.NOT_SUPPORTED,
                action=action,
                idempotency_key=key,
                detail=(
                    f"credentials not configured for {self.profile.name}; "
                    f"set {missing} in the environment"
                ),
            )
        return None

    def _no_transport(self, action: str, key: str) -> ActionResult:
        """Credentials exist but no transport was supplied.

        Reporting UNKNOWN would be wrong: nothing was attempted. This is a
        permanent failure with an actionable message.
        """
        return ActionResult(
            outcome=Outcome.PERMANENT_FAILURE,
            action=action,
            idempotency_key=key,
            detail=(
                f"{self.profile.name} credentials are present but no HTTP transport "
                "was supplied; pass transport=UrllibTransport() to enable live calls"
            ),
        )

    def _call(
        self,
        *,
        method: str,
        path: str,
        action: str,
        key: str,
        operation: str,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        idempotent: bool = True,
    ) -> ActionResult:
        """Make one live request and translate it into an ActionResult.

        The single place a marketplace call becomes a domain result, so the
        failure taxonomy is applied identically for every operation. A 401 is
        retried exactly once after invalidating the cached token, because
        clock skew and server-side revocation both produce a 401 on a token
        the client believed valid.
        """
        base = self.base_url()
        if not base:
            return self._no_transport(action, key)

        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            headers = self.auth_headers()
        except Exception as exc:  # noqa: BLE001 - credential problems are data
            return ActionResult(
                outcome=Outcome.PERMANENT_FAILURE,
                action=action,
                idempotency_key=key,
                detail=f"authentication failed: {exc}",
            )

        response = self._transport.request(
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            operation=operation,
            idempotent=idempotent,
        )

        if response.status in (401, 403) and self._tokens is not None:
            self._tokens.invalidate()
            response = self._transport.request(
                method,
                url,
                headers=self.auth_headers(),
                params=params,
                json_body=json_body,
                operation=operation,
                idempotent=idempotent,
            )

        payload: dict[str, Any] = {}
        if response.ok:
            try:
                parsed = response.json()
                payload = parsed if isinstance(parsed, dict) else {"items": parsed}
            except Exception as exc:  # noqa: BLE001 - a bad body is a real outcome
                return ActionResult(
                    outcome=Outcome.PERMANENT_FAILURE,
                    action=action,
                    idempotency_key=key,
                    detail=f"marketplace returned an unparseable body: {exc}",
                )

        return ActionResult(
            outcome=response.outcome,
            action=action,
            idempotency_key=key,
            detail=response.detail or f"HTTP {response.status}",
            payload=payload,
            duration_ms=response.elapsed_ms,
        )

    # -- operations -------------------------------------------------------

    def create_listing(
        self, payload: ListingPayload, *, dry_run: bool = True
    ) -> ActionResult:
        key = payload.fingerprint()
        plan = (
            f"Would create listing on {self.profile.name} for {payload.sku_code} "
            f"at {format_inr(payload.price_inr)}, quantity {payload.quantity}"
        )
        guarded = self._guard(
            Capability.CREATE_LISTING, "create_listing", key, dry_run=dry_run, plan=plan
        )
        return guarded if guarded else self._no_transport("create_listing", key)

    def update_price(
        self, sku_code: str, price_inr: Decimal, *, dry_run: bool = True
    ) -> ActionResult:
        key = idempotency_key(
            "update_price", self.profile.name, sku_code, money(price_inr)
        )
        plan = (
            f"Would set price to {format_inr(price_inr)} for {sku_code} "
            f"on {self.profile.name}"
        )
        guarded = self._guard(
            Capability.UPDATE_PRICE, "update_price", key, dry_run=dry_run, plan=plan
        )
        return guarded if guarded else self._no_transport("update_price", key)

    def update_inventory(
        self, sku_code: str, quantity: int, *, dry_run: bool = True
    ) -> ActionResult:
        key = idempotency_key("update_inventory", self.profile.name, sku_code, quantity)
        plan = (
            f"Would update inventory to {quantity} for {sku_code} on {self.profile.name}"
        )
        guarded = self._guard(
            Capability.UPDATE_INVENTORY,
            "update_inventory",
            key,
            dry_run=dry_run,
            plan=plan,
        )
        return guarded if guarded else self._no_transport("update_inventory", key)

    def get_listing(self, sku_code: str) -> ActionResult:
        key = idempotency_key("get_listing", self.profile.name, sku_code)
        guarded = self._guard(
            Capability.GET_LISTING, "get_listing", key, dry_run=False, plan=""
        )
        return guarded if guarded else self._no_transport("get_listing", key)

    def get_orders(self, *, since: datetime) -> ActionResult:
        key = idempotency_key("get_orders", self.profile.name, since.isoformat())
        guarded = self._guard(
            Capability.GET_ORDERS, "get_orders", key, dry_run=False, plan=""
        )
        return guarded if guarded else self._no_transport("get_orders", key)

    def get_settlements(self, *, since: datetime) -> ActionResult:
        key = idempotency_key("get_settlements", self.profile.name, since.isoformat())
        guarded = self._guard(
            Capability.GET_SETTLEMENTS, "get_settlements", key, dry_run=False, plan=""
        )
        return guarded if guarded else self._no_transport("get_settlements", key)


# --------------------------------------------------------------------------
# Concrete channels
# --------------------------------------------------------------------------

_SELLER_CAPS = frozenset(
    {
        Capability.GET_CATALOG,
        Capability.GET_LISTING,
        Capability.CREATE_LISTING,
        Capability.UPDATE_LISTING,
        Capability.UPDATE_PRICE,
        Capability.UPDATE_INVENTORY,
        Capability.GET_ORDERS,
        Capability.GET_ORDER,
        Capability.GET_INVENTORY,
        Capability.GET_RETURNS,
        Capability.GET_SETTLEMENTS,
    }
)


class AmazonIndiaAdapter(BaseAdapter):
    """Selling Partner API.

    Since Amazon dropped the AWS SigV4 requirement for most operations, the
    Login-with-Amazon access token in ``x-amz-access-token`` is the whole of
    authentication for the calls made here.
    """

    profile = MarketplaceProfile(
        name="amazon_in",
        access=AccessModel.SELLER_API,
        capabilities=_SELLER_CAPS,
        credential_env_vars=(
            "AMAZON_SP_REFRESH_TOKEN",
            "AMAZON_SP_CLIENT_ID",
            "AMAZON_SP_CLIENT_SECRET",
        ),
        notes=(
            "Selling Partner API covers your own listings, orders, inventory "
            "and settlement reports. It does not expose competitor data."
        ),
    )
    BASE_URL = "https://sellingpartnerapi-eu.amazon.com"
    ENDPOINT_ENV_VAR = "AMAZON_SP_ENDPOINT"

    def auth_headers(self) -> dict[str, str]:
        if self._tokens is None:
            return {}
        # SP-API uses its own header, not Authorization.
        return {"x-amz-access-token": self._tokens.token()}

    def get_orders(self, *, since: datetime) -> ActionResult:
        key = idempotency_key("get_orders", self.profile.name, since.isoformat())
        guarded = self._guard(
            Capability.GET_ORDERS, "get_orders", key, dry_run=False, plan=""
        )
        if guarded:
            return guarded
        env = self._environ if self._environ is not None else dict(os.environ)
        return self._call(
            method="GET",
            path="/orders/v0/orders",
            action="get_orders",
            key=key,
            operation="get_orders",
            params={
                "CreatedAfter": since.isoformat(),
                "MarketplaceIds": env.get("AMAZON_SP_MARKETPLACE_ID", "A21TJRUUN4KGV"),
            },
        )

    def update_price(
        self, sku_code: str, price_inr: Decimal, *, dry_run: bool = True
    ) -> ActionResult:
        key = idempotency_key(
            "update_price", self.profile.name, sku_code, money(price_inr)
        )
        plan = (
            f"Would set price to {format_inr(price_inr)} for {sku_code} on "
            f"{self.profile.name}"
        )
        guarded = self._guard(
            Capability.UPDATE_PRICE, "update_price", key, dry_run=dry_run, plan=plan
        )
        if guarded:
            return guarded
        return self._call(
            method="PATCH",
            path=f"/listings/2021-08-01/items/{sku_code}",
            action="update_price",
            key=key,
            operation="update_price",
            json_body={
                "productType": "PRODUCT",
                "patches": [
                    {
                        "op": "replace",
                        "path": "/attributes/purchasable_offer",
                        "value": [
                            {
                                "our_price": [
                                    {
                                        "schedule": [
                                            {"value_with_tax": str(money(price_inr))}
                                        ]
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
        )


class FlipkartAdapter(BaseAdapter):
    """Seller API, OAuth2 client-credentials with a Bearer token."""

    BASE_URL = "https://api.flipkart.net/sellers"
    ENDPOINT_ENV_VAR = "FLIPKART_ENDPOINT"

    def get_orders(self, *, since: datetime) -> ActionResult:
        key = idempotency_key("get_orders", self.profile.name, since.isoformat())
        guarded = self._guard(
            Capability.GET_ORDERS, "get_orders", key, dry_run=False, plan=""
        )
        if guarded:
            return guarded
        return self._call(
            method="POST",
            path="/v3/shipments/filter",
            action="get_orders",
            key=key,
            operation="get_orders",
            json_body={
                "filter": {
                    "type": "preDispatch",
                    "orderDate": {"from": since.isoformat()},
                },
                "pagination": {"pageSize": 20},
            },
        )

    def update_inventory(
        self, sku_code: str, quantity: int, *, dry_run: bool = True
    ) -> ActionResult:
        key = idempotency_key("update_inventory", self.profile.name, sku_code, quantity)
        plan = (
            f"Would update inventory to {quantity} for {sku_code} on {self.profile.name}"
        )
        guarded = self._guard(
            Capability.UPDATE_INVENTORY,
            "update_inventory",
            key,
            dry_run=dry_run,
            plan=plan,
        )
        if guarded:
            return guarded
        return self._call(
            method="POST",
            path="/skus/v2/inventory",
            action="update_inventory",
            key=key,
            operation="update_inventory",
            json_body={sku_code: {"inventory": [{"quantity": quantity}]}},
        )

    profile = MarketplaceProfile(
        name="flipkart",
        access=AccessModel.SELLER_API,
        capabilities=_SELLER_CAPS,
        credential_env_vars=("FLIPKART_CLIENT_ID", "FLIPKART_CLIENT_SECRET"),
        notes=(
            "Seller API covers your own listings, orders and settlements. "
            "Market intelligence is not available through it."
        ),
    )


class MeeshoAdapter(BaseAdapter):
    profile = MarketplaceProfile(
        name="meesho",
        access=AccessModel.MANUAL_EXPORT,
        capabilities=frozenset({Capability.GET_ORDERS, Capability.GET_SETTLEMENTS}),
        credential_env_vars=(),
        notes=(
            "No general public seller API. Orders and settlements arrive as "
            "panel exports, so they are ingested as files rather than fetched. "
            "Listing and price writes are not available programmatically."
        ),
    )

    def get_orders(self, *, since: datetime) -> ActionResult:
        """Orders come from a downloaded export, not an API call."""
        key = idempotency_key("get_orders", self.profile.name, since.isoformat())
        return ActionResult(
            outcome=Outcome.NOT_SUPPORTED,
            action="get_orders",
            idempotency_key=key,
            detail=(
                "Meesho orders are ingested from a seller-panel export; "
                "use `ingest-orders --source <file>` rather than a live fetch"
            ),
        )

    get_settlements = get_orders  # same route, same explanation


class ONDCAdapter(BaseAdapter):
    profile = MarketplaceProfile(
        name="ondc",
        access=AccessModel.NETWORK_PARTICIPANT,
        capabilities=frozenset(
            {
                Capability.GET_CATALOG,
                Capability.CREATE_LISTING,
                Capability.UPDATE_LISTING,
                Capability.UPDATE_PRICE,
                Capability.UPDATE_INVENTORY,
                Capability.GET_ORDERS,
                Capability.GET_ORDER,
            }
        ),
        credential_env_vars=("ONDC_SUBSCRIBER_ID", "ONDC_SIGNING_PRIVATE_KEY"),
        notes=(
            "Open network with published protocol specifications. Requires "
            "onboarding as a registered network participant and signed "
            "requests. Settlement is handled by the network, not this adapter."
        ),
    )


#: Every channel the system knows about. Adding one is an entry here plus a
#: class above; nothing in economics, inventory, orders or pricing changes.
ADAPTERS: dict[str, type[BaseAdapter]] = {
    "amazon_in": AmazonIndiaAdapter,
    "flipkart": FlipkartAdapter,
    "meesho": MeeshoAdapter,
    "ondc": ONDCAdapter,
}


def get_adapter(
    name: str,
    *,
    environ: dict[str, str] | None = None,
    transport: Transport | None = None,
    tokens: TokenProvider | None = None,
) -> BaseAdapter:
    """Build an adapter. Without a transport it stays offline by construction."""
    try:
        return ADAPTERS[name](environ=environ, transport=transport, tokens=tokens)
    except KeyError as exc:
        raise KeyError(
            f"unknown marketplace {name!r}; known: {sorted(ADAPTERS)}"
        ) from exc


def capability_matrix(environ: dict[str, str] | None = None) -> str:
    """Honest report of what each channel can do right now."""
    lines = [
        "MARKETPLACE CAPABILITIES",
        "",
        f"{'Channel':<12} {'Access':<22} {'Creds':<8} {'Capabilities'}",
        f"{'-' * 12} {'-' * 22} {'-' * 8} {'-' * 34}",
    ]
    for name, cls in sorted(ADAPTERS.items()):
        profile = cls(environ=environ).profile
        creds = "set" if profile.credentials_present(environ) else "missing"
        caps = f"{len(profile.capabilities)} declared"
        lines.append(f"{name:<12} {profile.access!s:<22} {creds:<8} {caps}")
    lines += [
        "",
        "No HTTP transport is bundled for any channel. Operations report",
        "NOT_SUPPORTED until credentials and a transport are supplied, and",
        "dry-run mode describes the intended call without making one.",
        "",
    ]
    return "\n".join(lines)


def detect_drift(intended: ListingPayload, remote: RemoteListing) -> list[str]:
    """Compare what we asked for against what the marketplace reports.

    A successful API response is not evidence the marketplace state changed:
    listings get suppressed, prices get overridden by promotions, quantities
    get clamped. Verification is a separate read.
    """
    drift: list[str] = []
    if money(remote.price_inr) != money(intended.price_inr):
        drift.append(
            f"price drift: intended {format_inr(intended.price_inr)}, "
            f"marketplace reports {format_inr(remote.price_inr)}"
        )
    if remote.quantity != intended.quantity:
        drift.append(
            f"inventory drift: intended {intended.quantity}, "
            f"marketplace reports {remote.quantity}"
        )
    if remote.title.strip() != intended.title.strip():
        drift.append("title drift: marketplace title differs from the submitted one")
    if remote.status.upper() not in ("ACTIVE", "LIVE"):
        drift.append(f"listing is not live: marketplace status {remote.status!r}")
    return drift


def dry_run_plan(results: Iterable[ActionResult]) -> str:
    """Render what a dry run would have done."""
    lines = ["DRY RUN - no external state was changed", ""]
    for result in results:
        marker = "would" if result.outcome is Outcome.DRY_RUN else str(result.outcome)
        lines.append(f"  [{marker}] {result.detail or result.action}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ADAPTERS",
    "ADAPTER_VERSION",
    "AccessModel",
    "AmazonIndiaAdapter",
    "BaseAdapter",
    "Capability",
    "CredentialsMissing",
    "FlipkartAdapter",
    "ListingPayload",
    "MarketplaceAdapter",
    "MarketplaceProfile",
    "MeeshoAdapter",
    "ONDCAdapter",
    "RemoteListing",
    "capability_matrix",
    "detect_drift",
    "dry_run_plan",
    "get_adapter",
]
