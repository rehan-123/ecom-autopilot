"""Discovery source adapters: where product candidates come from.

A discovery source tells us a product *appears* to be selling. It does not
tell us where to buy it. That distinction is the point of this package: a
retail marketplace is evidence, never a supplier, and the registry records
which role each source can play.

**No source here scrapes anything.** An adapter is legitimate only if it uses
an official API, an authorised feed, a public dataset, a permitted export, or
data the operator collected by hand. Sources with no legitimate access route
are still registered — with ``access=NONE`` — so the gap is visible instead of
being quietly filled by something that violates terms of service.

Where a source cannot supply a signal, the adapter returns
``DATA_UNAVAILABLE``. Demand, ranking, review velocity, competition and trend
figures are never fabricated, because a made-up number that reaches the
opportunity score is indistinguishable from a real one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from signals.market import Signal, unavailable

logger = logging.getLogger(__name__)

DISCOVERY_VERSION = "discovery/1.0.0"


class AccessRoute(StrEnum):
    """How, legitimately, data can be obtained from a source."""

    OFFICIAL_API = "OFFICIAL_API"
    AUTHORISED_FEED = "AUTHORISED_FEED"
    PUBLIC_DATASET = "PUBLIC_DATASET"
    SELLER_EXPORT = "SELLER_EXPORT"
    MANUAL = "MANUAL"
    NONE = "NONE"


class SourceRole(StrEnum):
    """What a source may be used for.

    ``DISCOVERY_ONLY`` is the important one. A retail marketplace can suggest a
    product is moving; it cannot become the supplier, because its prices sit at
    or near landed cost and it issues no tax invoice.
    """

    DISCOVERY_ONLY = "DISCOVERY_ONLY"
    SIGNAL_ONLY = "SIGNAL_ONLY"
    SUPPLIER = "SUPPLIER"


@dataclass(frozen=True)
class SourceProfile:
    """What is actually possible with one source, honestly stated."""

    name: str
    role: SourceRole
    access: AccessRoute
    notes: str
    supplies_signals: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.access is not AccessRoute.NONE


#: Sources named in the project scope, each recorded with the access route it
#: genuinely offers. Entries with ``AccessRoute.NONE`` are deliberate: the
#: source is real, the legitimate access is not, and pretending otherwise
#: would be the start of a scraper.
KNOWN_SOURCES: tuple[SourceProfile, ...] = (
    SourceProfile(
        "amazon_in",
        SourceRole.DISCOVERY_ONLY,
        AccessRoute.SELLER_EXPORT,
        "Own seller reports only. Competitor rank and review velocity have no "
        "legitimate API; record those as DATA_UNAVAILABLE.",
        ("OWN_SALES", "RETURN_RATE"),
    ),
    SourceProfile(
        "flipkart",
        SourceRole.DISCOVERY_ONLY,
        AccessRoute.SELLER_EXPORT,
        "Seller APIs cover your own listings and orders, not market intelligence.",
        ("OWN_SALES", "RETURN_RATE"),
    ),
    SourceProfile(
        "meesho",
        SourceRole.DISCOVERY_ONLY,
        AccessRoute.NONE,
        "No public API. A retail marketplace is not a wholesale source: prices "
        "sit near landed cost and purchases yield no input tax credit.",
    ),
    SourceProfile(
        "myntra", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "ajio", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "nykaa", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "tata_cliq", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "snapdeal", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "jiomart", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "firstcry", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "pepperfry", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "ondc",
        SourceRole.DISCOVERY_ONLY,
        AccessRoute.OFFICIAL_API,
        "Open network with published specifications; participation requires "
        "onboarding as a network participant.",
    ),
    SourceProfile(
        "blinkit", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "zepto", SourceRole.DISCOVERY_ONLY, AccessRoute.NONE, "No public data API."
    ),
    SourceProfile(
        "swiggy_instamart",
        SourceRole.DISCOVERY_ONLY,
        AccessRoute.NONE,
        "No public data API.",
    ),
    SourceProfile(
        "google_trends",
        SourceRole.SIGNAL_ONLY,
        AccessRoute.PUBLIC_DATASET,
        "Relative interest over time. Category-level at best; too coarse for "
        "SKU-level decisions, so treat as directional only.",
        ("SEARCH_INTEREST", "TREND_DIRECTION"),
    ),
    SourceProfile(
        "manual_research",
        SourceRole.SIGNAL_ONLY,
        AccessRoute.MANUAL,
        "Human browsing, recorded with URLs and dates. The most reliable "
        "market-price source available, and the slowest.",
        ("MARKET_PRICE", "COMPETITOR_COUNT", "BRAND_DOMINANCE"),
    ),
    SourceProfile(
        "supplier_direct",
        SourceRole.SUPPLIER,
        AccessRoute.MANUAL,
        "Manufacturers, authorised wholesalers, distributors, B2B suppliers "
        "and importers. The only sources that may become preferred suppliers.",
    ),
)

#: Preference order for actual sourcing. A marketplace never appears here.
SUPPLIER_PREFERENCE: tuple[str, ...] = (
    "MANUFACTURER",
    "AUTHORISED_WHOLESALER",
    "DISTRIBUTOR",
    "B2B_SUPPLIER",
    "IMPORTER",
)


@dataclass(frozen=True)
class DiscoveryCandidate:
    """A product a discovery source suggests is worth investigating."""

    source: str
    external_ref: str
    title: str
    observed_price_inr: Decimal | None
    signals: dict[str, Signal] = field(default_factory=dict)
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "external_ref": self.external_ref,
            "title": self.title,
            "observed_price_inr": (
                None if self.observed_price_inr is None else str(self.observed_price_inr)
            ),
            "signals": {
                k: {"status": str(v.status), "type": v.signal_type}
                for k, v in self.signals.items()
            },
            "observed_at": self.observed_at.isoformat(),
        }


class DiscoveryAdapter(Protocol):
    """Contract every discovery source implements."""

    profile: SourceProfile

    def discover(self, *, limit: int = 50) -> Iterable[DiscoveryCandidate]:
        """Yield candidates, or nothing if no legitimate route exists."""
        ...


class UnavailableAdapter:
    """Adapter for a source with no legitimate access route.

    Returns no candidates and explains why, rather than raising. A source
    being inaccessible is a normal state of the world, not an error, and the
    registry is more useful listing it than pretending it does not exist.
    """

    def __init__(self, profile: SourceProfile) -> None:
        self.profile = profile

    def discover(self, *, limit: int = 50) -> Iterable[DiscoveryCandidate]:
        logger.info(
            "discovery source unavailable",
            extra={"source": self.profile.name, "detail": self.profile.notes},
        )
        return ()

    def signals_for(self, external_ref: str) -> dict[str, Signal]:
        return {
            name: unavailable(
                name,
                reason=f"{self.profile.name}: {self.profile.notes}",
                collected_by="system",
            )
            for name in self.profile.supplies_signals
        }


class ManualAdapter:
    """Candidates and signals entered by a human.

    The slowest source and the only one that reliably works for Indian
    marketplaces, which is why the architecture treats manual research as a
    first-class input rather than a fallback.
    """

    def __init__(self, profile: SourceProfile) -> None:
        self.profile = profile
        self._candidates: list[DiscoveryCandidate] = []

    def record(self, candidate: DiscoveryCandidate) -> None:
        self._candidates.append(candidate)

    def discover(self, *, limit: int = 50) -> Iterable[DiscoveryCandidate]:
        return tuple(self._candidates[:limit])


class DiscoveryRegistry:
    """Holds the adapters and reports what each can actually do.

    Adding a source later is registering an adapter here; no other module
    changes. That is the extensibility the architecture promises, and it costs
    nothing because every adapter already has to declare its access route.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, object] = {}
        for profile in KNOWN_SOURCES:
            self._adapters[profile.name] = (
                ManualAdapter(profile)
                if profile.access is AccessRoute.MANUAL
                else UnavailableAdapter(profile)
            )

    def register(self, name: str, adapter: object) -> None:
        """Install a real adapter, replacing the placeholder."""
        self._adapters[name] = adapter
        logger.info("discovery adapter registered", extra={"source": name})

    def get(self, name: str) -> object:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown discovery source {name!r}; known: {sorted(self._adapters)}"
            ) from exc

    def usable_sources(self) -> list[SourceProfile]:
        return [p for p in KNOWN_SOURCES if p.usable]

    def unavailable_sources(self) -> list[SourceProfile]:
        return [p for p in KNOWN_SOURCES if not p.usable]

    def coverage_report(self) -> str:
        """Which sources can supply data, and which cannot.

        Worth reading before trusting any opportunity score: it shows exactly
        how much of the demand picture is unobtainable.
        """
        lines = [
            "DISCOVERY SOURCE COVERAGE",
            "",
            f"{'Source':<20} {'Role':<16} {'Access':<18} Notes",
            f"{'-' * 20} {'-' * 16} {'-' * 18} {'-' * 30}",
        ]
        for profile in KNOWN_SOURCES:
            lines.append(
                f"{profile.name:<20} {profile.role!s:<16} "
                f"{profile.access!s:<18} {profile.notes[:44]}"
            )
        usable = len(self.usable_sources())
        lines += [
            "",
            f"{usable} of {len(KNOWN_SOURCES)} sources have a legitimate access route.",
            "Sources marked NONE are recorded so the gap is visible; signals from",
            "them are DATA_UNAVAILABLE and never estimated.",
            "",
        ]
        return "\n".join(lines)


registry = DiscoveryRegistry()


__all__ = [
    "DISCOVERY_VERSION",
    "KNOWN_SOURCES",
    "SUPPLIER_PREFERENCE",
    "AccessRoute",
    "DiscoveryAdapter",
    "DiscoveryCandidate",
    "DiscoveryRegistry",
    "ManualAdapter",
    "SourceProfile",
    "SourceRole",
    "UnavailableAdapter",
    "registry",
]
