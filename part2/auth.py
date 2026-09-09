"""Authentication: credentials from the environment, tokens cached and refreshed.

Credentials are read from the environment at the moment they are needed and
never stored, logged, returned in a result, or written to the ledger. The only
thing that travels is the short-lived access token, and that goes into a header
the transport redacts.

Both marketplaces here use OAuth2 with a refresh step. The mechanics differ —
Amazon calls it Login with Amazon and exchanges a long-lived refresh token,
Flipkart uses client credentials — but the shape is the same: a cached token,
refreshed slightly before it expires, under a lock so a hundred concurrent
requests trigger one refresh rather than a hundred.

Nothing in this module has been exercised against a live authorisation server.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

from part2.transport import HttpResponse, Transport

logger = logging.getLogger(__name__)

AUTH_VERSION = "auth/1.0.0"

#: Refresh this many seconds before nominal expiry. A token that expires
#: mid-flight produces a 401 that looks like a credential problem.
REFRESH_MARGIN_S = 120.0


class CredentialsMissing(RuntimeError):
    """Raised when a required credential is absent from the environment."""


@dataclass(frozen=True)
class CredentialSpec:
    """Which environment variables one marketplace needs."""

    marketplace: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    def missing(self, environ: Mapping[str, str] | None = None) -> list[str]:
        env = environ if environ is not None else os.environ
        return [name for name in self.required if not env.get(name)]

    def read(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Read the credentials, raising if any required one is absent.

        Returns a plain dict the caller must not log. Nothing in this project
        persists the result.
        """
        env = environ if environ is not None else os.environ
        absent = self.missing(env)
        if absent:
            raise CredentialsMissing(
                f"{self.marketplace}: set {', '.join(absent)} in the environment"
            )
        values = {name: env[name] for name in self.required}
        values.update({name: env[name] for name in self.optional if env.get(name)})
        return values


#: Every credential the project can consume. Documented in one place so an
#: operator can see exactly what is required before anything is configured.
CREDENTIAL_SPECS: dict[str, CredentialSpec] = {
    "amazon_in": CredentialSpec(
        marketplace="amazon_in",
        required=(
            "AMAZON_SP_REFRESH_TOKEN",
            "AMAZON_SP_CLIENT_ID",
            "AMAZON_SP_CLIENT_SECRET",
        ),
        optional=("AMAZON_SP_ENDPOINT", "AMAZON_SP_MARKETPLACE_ID"),
    ),
    "flipkart": CredentialSpec(
        marketplace="flipkart",
        required=("FLIPKART_CLIENT_ID", "FLIPKART_CLIENT_SECRET"),
        optional=("FLIPKART_ENDPOINT",),
    ),
    "ondc": CredentialSpec(
        marketplace="ondc",
        required=("ONDC_SUBSCRIBER_ID", "ONDC_SIGNING_PRIVATE_KEY"),
        optional=("ONDC_ENDPOINT", "ONDC_UNIQUE_KEY_ID"),
    ),
}


@dataclass
class AccessToken:
    """A short-lived token with the moment it stops being usable."""

    value: str
    expires_at: float

    def is_fresh(self, *, margin_s: float = REFRESH_MARGIN_S) -> bool:
        return time.time() < (self.expires_at - margin_s)


class TokenProvider(Protocol):
    """Supplies a currently-valid access token."""

    def token(self) -> str: ...
    def invalidate(self) -> None: ...


class StaticTokenProvider:
    """A token supplied directly. Used in tests and by operators who mint
    their own out of band."""

    def __init__(self, value: str) -> None:
        self._value = value

    def token(self) -> str:
        return self._value

    def invalidate(self) -> None:
        """Nothing to do: a static token cannot be refreshed."""


class OAuth2TokenProvider:
    """Caches and refreshes an OAuth2 access token.

    The lock matters. Without it, a burst of concurrent requests each sees an
    expired token and each starts a refresh, which on some authorisation
    servers invalidates the previous one and produces a cascade of 401s that
    looks exactly like bad credentials.
    """

    def __init__(
        self,
        *,
        token_url: str,
        form_fields: Mapping[str, str],
        transport: Transport,
        marketplace: str,
        default_ttl_s: float = 3600.0,
    ) -> None:
        self._token_url = token_url
        self._form_fields = dict(form_fields)
        self._transport = transport
        self._marketplace = marketplace
        self._default_ttl_s = default_ttl_s
        self._cached: AccessToken | None = None
        self._lock = threading.Lock()
        self.refreshes = 0

    def token(self) -> str:
        with self._lock:
            if self._cached is not None and self._cached.is_fresh():
                return self._cached.value
            self._cached = self._refresh()
            self.refreshes += 1
            return self._cached.value

    def invalidate(self) -> None:
        """Drop the cached token so the next call refreshes.

        Called when a request comes back 401 despite a token we believed fresh
        — clock skew and server-side revocation both cause that.
        """
        with self._lock:
            self._cached = None

    def _refresh(self) -> AccessToken:
        response: HttpResponse = self._transport.request(
            "POST",
            self._token_url,
            form_body=self.form_body,
            operation="auth:token",
            idempotent=True,
        )
        if not response.ok:
            raise CredentialsMissing(
                f"{self._marketplace}: token refresh failed "
                f"({response.detail or response.status})"
            )

        payload = response.json() or {}
        value = payload.get("access_token")
        if not value:
            raise CredentialsMissing(
                f"{self._marketplace}: token response contained no access_token"
            )
        ttl = float(payload.get("expires_in") or self._default_ttl_s)
        logger.info(
            "access token refreshed",
            extra={"marketplace": self._marketplace, "operation": "auth:token"},
        )
        return AccessToken(value=str(value), expires_at=time.time() + ttl)

    @property
    def form_body(self) -> str:
        """The encoded form the token endpoint expects.

        Exposed so a transport can send it as the request body. Kept as a
        property rather than stored on the request so the secret never sits in
        a structure that might be logged.
        """
        return urlencode(self._form_fields)


def lwa_provider(
    transport: Transport, *, environ: Mapping[str, str] | None = None
) -> OAuth2TokenProvider:
    """Login with Amazon: exchange a long-lived refresh token for an access token.

    Since Amazon dropped the AWS SigV4 requirement for most Selling Partner
    API operations, this token in ``x-amz-access-token`` is the whole of
    authentication for the calls this project makes.
    """
    values = CREDENTIAL_SPECS["amazon_in"].read(environ)
    return OAuth2TokenProvider(
        token_url="https://api.amazon.com/auth/o2/token",
        form_fields={
            "grant_type": "refresh_token",
            "refresh_token": values["AMAZON_SP_REFRESH_TOKEN"],
            "client_id": values["AMAZON_SP_CLIENT_ID"],
            "client_secret": values["AMAZON_SP_CLIENT_SECRET"],
        },
        transport=transport,
        marketplace="amazon_in",
    )


def flipkart_provider(
    transport: Transport, *, environ: Mapping[str, str] | None = None
) -> OAuth2TokenProvider:
    """Flipkart Seller API: client-credentials grant, Bearer token."""
    values = CREDENTIAL_SPECS["flipkart"].read(environ)
    endpoint = values.get("FLIPKART_ENDPOINT", "https://api.flipkart.net")
    return OAuth2TokenProvider(
        token_url=f"{endpoint}/oauth-service/oauth/token",
        form_fields={
            "grant_type": "client_credentials",
            "scope": "Seller_Api",
            "client_id": values["FLIPKART_CLIENT_ID"],
            "client_secret": values["FLIPKART_CLIENT_SECRET"],
        },
        transport=transport,
        marketplace="flipkart",
    )


class Signer(Protocol):
    """Signs a request body for networks that require it.

    ONDC requires an Ed25519 signature over a BLAKE-512 digest of the payload.
    The standard library has neither primitive, and this project declines to
    add a cryptography dependency it cannot exercise against a live network —
    so the signer is injected. ``build_signing_string`` below produces exactly
    the string the specification says to sign, which is the part worth getting
    right and testing.
    """

    def sign(self, signing_string: str) -> str: ...


def build_signing_string(*, digest_base64: str, created: int, expires: int) -> str:
    """Assemble the ONDC signing string.

    Field order and line endings are specified and significant; a signature
    over a differently-assembled string verifies nowhere.
    """
    return (
        f"(created): {created}\n(expires): {expires}\ndigest: BLAKE-512={digest_base64}"
    )


def build_authorization_header(
    *, subscriber_id: str, unique_key_id: str, signature: str, created: int, expires: int
) -> str:
    """The ONDC ``Authorization`` header around a computed signature."""
    return (
        f'Signature keyId="{subscriber_id}|{unique_key_id}|ed25519",'
        f'algorithm="ed25519",created="{created}",expires="{expires}",'
        f'headers="(created) (expires) digest",signature="{signature}"'
    )


def credential_report(environ: Mapping[str, str] | None = None) -> str:
    """What is configured and what is missing, without revealing any value."""
    lines = [
        "MARKETPLACE CREDENTIALS",
        "",
        f"{'Marketplace':<14} {'Status':<12} Required environment variables",
        f"{'-' * 14} {'-' * 12} {'-' * 48}",
    ]
    for name, spec in sorted(CREDENTIAL_SPECS.items()):
        absent = spec.missing(environ)
        status = "configured" if not absent else f"{len(absent)} missing"
        lines.append(f"{name:<14} {status:<12} {', '.join(spec.required)}")
    lines += [
        "",
        "Values are read from the environment when needed and never stored,",
        "logged, or written to the ledger. Presence is reported; contents are not.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "AUTH_VERSION",
    "CREDENTIAL_SPECS",
    "REFRESH_MARGIN_S",
    "AccessToken",
    "CredentialSpec",
    "CredentialsMissing",
    "OAuth2TokenProvider",
    "Signer",
    "StaticTokenProvider",
    "TokenProvider",
    "build_authorization_header",
    "build_signing_string",
    "credential_report",
    "flipkart_provider",
    "lwa_provider",
]
