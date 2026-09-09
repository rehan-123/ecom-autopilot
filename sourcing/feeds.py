"""Discovery feeds: where candidate products legitimately come from.

Every feed here reads something the operator is entitled to have — their own
seller export, a public dataset download, a manually recorded observation, or
an authorised API where credentials exist. Nothing scrapes, nothing bypasses a
rate limit, and nothing fabricates a number that could not be obtained.

The uncomfortable consequence is that most Indian marketplaces yield nothing.
`discovery/registry.py` already records which ones have a legitimate route;
this module turns that judgement into feeds that either produce observations or
say plainly that they cannot. A feed returning zero candidates with a reason is
a correct outcome, not a failure.
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from discovery.registry import KNOWN_SOURCES, AccessRoute
from econ.money import money
from sourcing.matching import SupplierCandidate
from sourcing.model import (
    DiscoveredProduct,
    FieldSource,
    Observed,
    enrich,
    unavailable,
)

logger = logging.getLogger(__name__)

FEEDS_VERSION = "feeds/1.0.0"

DEFAULT_FIXTURE_DIR = Path("data/fixtures/discovery")


class FeedError(RuntimeError):
    """Raised when a feed's input cannot be read at all."""


@dataclass
class FeedResult:
    """What one feed produced, including what it could not produce."""

    source: str
    access: AccessRoute
    products: list[DiscoveredProduct] = field(default_factory=list)
    unavailable_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "access": str(self.access),
            "available": self.available,
            "products": len(self.products),
            "unavailable_reason": self.unavailable_reason,
            "warnings": list(self.warnings),
        }


class DiscoveryFeed(Protocol):
    """Contract every feed implements."""

    source: str
    access: AccessRoute

    def fetch(self) -> FeedResult: ...


def _decimal(raw: Any) -> Decimal | None:
    if raw in (None, ""):
        return None
    try:
        return money(str(raw).replace(",", "").replace("\u20b9", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def _observed(
    raw: Any, source: FieldSource, reference: str | None, missing: str
) -> Observed:
    value = _decimal(raw)
    if value is None:
        return unavailable(missing)
    return Observed(value, source, reference=reference)


class UnavailableFeed:
    """A source with no legitimate programmatic route.

    Returns no products and says why. Registered rather than omitted so the
    gap is visible in every report — an operator deciding whether to trust an
    opportunity score needs to know how much of the market was invisible.
    """

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.access = AccessRoute.NONE
        self._reason = reason

    def fetch(self) -> FeedResult:
        logger.info("discovery source unavailable", extra={"source": self.source})
        return FeedResult(
            source=self.source, access=self.access, unavailable_reason=self._reason
        )


class CsvObservationFeed:
    """Observations recorded by a human or exported from a seller panel.

    The slowest source and the only one that works for most Indian
    marketplaces. Treated as first-class rather than a fallback, because a
    manually collected price with a URL and a date is better evidence than an
    automated number of unknown provenance.
    """

    COLUMNS = (
        "title",
        "price",
        "category",
        "brand",
        "model",
        "gtin",
        "hsn",
        "weight_g",
        "rank",
        "trend",
        "reviews",
        "review_velocity",
        "search_interest",
        "competitors",
        "brand_dominance",
        "availability",
        "reference",
    )

    def __init__(
        self,
        path: Path | str,
        *,
        source: str = "manual_research",
        access: AccessRoute = AccessRoute.MANUAL,
    ) -> None:
        self.path = Path(path)
        self.source = source
        self.access = access

    def fetch(self) -> FeedResult:
        result = FeedResult(source=self.source, access=self.access)
        if not self.path.is_file():
            raise FeedError(f"observation file not found: {self.path}")

        try:
            text = self.path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise FeedError(f"cannot read {self.path}: {exc}") from exc

        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames:
            result.warnings.append("file has no header row")
            return result

        field_source = (
            FieldSource.MANUAL
            if self.access is AccessRoute.MANUAL
            else FieldSource.SELLER_EXPORT
        )
        for number, row in enumerate(reader, start=2):
            title = (row.get("title") or "").strip()
            if not title:
                result.warnings.append(f"row {number}: no title, skipped")
                continue

            reference = (row.get("reference") or "").strip() or None
            product = DiscoveredProduct(
                source=self.source,
                external_ref=reference or f"{self.source}:{self.path.name}:{number}",
                raw_title=title,
                category=(row.get("category") or "").strip() or None,
                brand=(row.get("brand") or "").strip() or None,
                model_code=(row.get("model") or "").strip() or None,
                gtin=(row.get("gtin") or "").strip() or None,
                hsn_code=(row.get("hsn") or "").strip() or None,
                weight_g=int(_decimal(row.get("weight_g")) or 0) or None,
                observed_price_inr=_observed(
                    row.get("price"), field_source, reference, "price not recorded"
                ),
                rank=_observed(
                    row.get("rank"),
                    field_source,
                    reference,
                    "rank has no legitimate source",
                ),
                trend=_observed(
                    row.get("trend"), field_source, reference, "trend not supplied"
                ),
                review_count=_observed(
                    row.get("reviews"),
                    field_source,
                    reference,
                    "review counts have no legitimate API",
                ),
                review_velocity=_observed(
                    row.get("review_velocity"),
                    field_source,
                    reference,
                    "review velocity has no legitimate API",
                ),
                search_interest=_observed(
                    row.get("search_interest"),
                    field_source,
                    reference,
                    "search interest not supplied",
                ),
                competitor_count=_observed(
                    row.get("competitors"),
                    field_source,
                    reference,
                    "competitor count not recorded",
                ),
                brand_dominance=_observed(
                    row.get("brand_dominance"),
                    field_source,
                    reference,
                    "brand dominance not recorded",
                ),
                availability=_observed(
                    row.get("availability"),
                    field_source,
                    reference,
                    "availability not recorded",
                ),
            )
            result.products.append(enrich(product))

        unmapped = [c for c in (reader.fieldnames or []) if c not in self.COLUMNS]
        if unmapped:
            result.warnings.append(f"unrecognised columns ignored: {unmapped}")
        return result


class TrendsExportFeed:
    """Relative search interest from a Google Trends CSV export.

    A public dataset the operator downloads themselves. Category-level at
    best, so it corroborates interest rather than establishing demand for a
    specific SKU — recorded as `search_interest` and `trend`, never as sales.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.source = "google_trends"
        self.access = AccessRoute.PUBLIC_DATASET

    def fetch(self) -> FeedResult:
        result = FeedResult(source=self.source, access=self.access)
        if not self.path.is_file():
            raise FeedError(f"trends export not found: {self.path}")

        rows = [
            line
            for line in self.path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.startswith("Category:")
        ]
        reader = csv.DictReader(rows)
        for row in reader:
            term = (row.get("term") or row.get("Term") or "").strip()
            if not term:
                continue
            interest = _decimal(row.get("interest") or row.get("Interest"))
            direction = _decimal(row.get("direction") or row.get("Direction"))
            product = DiscoveredProduct(
                source=self.source,
                external_ref=f"trends:{term}",
                raw_title=term,
                category=(row.get("category") or "").strip() or None,
                observed_price_inr=unavailable(
                    "Google Trends carries interest, never price"
                ),
                search_interest=(
                    Observed(
                        interest, FieldSource.PUBLIC_DATASET, reference=str(self.path)
                    )
                    if interest is not None
                    else unavailable("no interest value in export")
                ),
                trend=(
                    Observed(
                        direction, FieldSource.PUBLIC_DATASET, reference=str(self.path)
                    )
                    if direction is not None
                    else unavailable("no direction value in export")
                ),
            )
            product.notes.append(
                "search interest is category-level; it corroborates demand but "
                "does not establish it for a specific SKU"
            )
            result.products.append(enrich(product))
        if not result.products:
            result.warnings.append("no usable rows in the trends export")
        return result


class SellerApiFeed:
    """Your own catalogue and performance, via an authorised seller API.

    Credentials come from the environment. Without them the feed reports
    itself unavailable rather than failing, because operating with no seller
    API is the normal case, not an error.
    """

    def __init__(
        self, marketplace: str, *, environ: dict[str, str] | None = None
    ) -> None:
        self.source = marketplace
        self.access = AccessRoute.SELLER_EXPORT
        self._environ = environ

    def fetch(self) -> FeedResult:
        from part2.marketplace import get_adapter  # noqa: PLC0415

        adapter = get_adapter(self.source, environ=self._environ)
        environ = self._environ if self._environ is not None else dict(os.environ)
        if not adapter.profile.credentials_present(environ):
            return FeedResult(
                source=self.source,
                access=self.access,
                unavailable_reason=(
                    f"{self.source} seller API credentials are not configured; "
                    f"set {', '.join(adapter.profile.credential_env_vars) or 'the required vars'}"
                ),
            )
        return FeedResult(
            source=self.source,
            access=self.access,
            unavailable_reason=(
                f"{self.source} credentials are present but no transport is bundled; "
                "supply a client implementing the adapter's calls"
            ),
        )


def build_registry(
    *,
    fixture_dir: Path | str | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, DiscoveryFeed]:
    """Assemble every feed, available or not.

    Sources with no legitimate route get an ``UnavailableFeed`` so they appear
    in coverage reporting. Adding a real integration later means replacing one
    entry, with nothing else in the system changing.
    """
    directory = Path(fixture_dir) if fixture_dir else DEFAULT_FIXTURE_DIR
    feeds: dict[str, DiscoveryFeed] = {}

    for profile in KNOWN_SOURCES:
        if profile.access is AccessRoute.NONE:
            feeds[profile.name] = UnavailableFeed(profile.name, profile.notes)
        elif profile.access is AccessRoute.SELLER_EXPORT:
            feeds[profile.name] = SellerApiFeed(profile.name, environ=environ)

    observations = directory / "observations.csv"
    if observations.is_file():
        feeds["manual_research"] = CsvObservationFeed(observations)
    else:
        feeds["manual_research"] = UnavailableFeed(
            "manual_research",
            f"no observation file at {observations}; record one to supply candidates",
        )

    trends = directory / "trends.csv"
    feeds["google_trends"] = (
        TrendsExportFeed(trends)
        if trends.is_file()
        else UnavailableFeed(
            "google_trends",
            f"no trends export at {trends}; download one from Google Trends",
        )
    )
    return feeds


def collect(
    feeds: dict[str, DiscoveryFeed], *, only: list[str] | None = None
) -> tuple[list[DiscoveredProduct], list[FeedResult]]:
    """Run every feed, returning candidates and the full availability picture."""
    products: list[DiscoveredProduct] = []
    results: list[FeedResult] = []

    for name, feed in sorted(feeds.items()):
        if only and name not in only:
            continue
        try:
            result = feed.fetch()
        except FeedError as exc:
            result = FeedResult(
                source=name,
                access=getattr(feed, "access", AccessRoute.NONE),
                unavailable_reason=str(exc),
            )
        results.append(result)
        products.extend(result.products)
    return products, results


def coverage_report(results: list[FeedResult]) -> str:
    """How much of the market was actually visible."""
    available = [r for r in results if r.available]
    lines = [
        "DISCOVERY FEED COVERAGE",
        "",
        f"{'Source':<20} {'Access':<18} {'Products':>9}  Note",
        f"{'-' * 20} {'-' * 18} {'-' * 9}  {'-' * 34}",
    ]
    for result in results:
        note = "" if result.available else (result.unavailable_reason or "")[:44]
        lines.append(
            f"{result.source:<20} {result.access!s:<18} {len(result.products):>9}  {note}"
        )
    lines += [
        "",
        f"{len(available)} of {len(results)} feeds produced candidates.",
        "Sources with no legitimate route yield DATA_UNAVAILABLE and are never",
        "estimated. Read any opportunity score with that in mind.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Supplier catalogue loading
# --------------------------------------------------------------------------


def load_supplier_candidates(path: Path | str) -> list[SupplierCandidate]:
    """Accepted supplier candidates from a catalogue.

    Thin wrapper over ``sourcing.supplier_ingest.ingest_suppliers`` so there is
    one supplier path rather than two with different rules. Callers wanting the
    rejected rows and provenance should use ``ingest_suppliers`` directly; this
    returns only what was accepted, which is what the matcher needs.
    """
    from sourcing.supplier_ingest import ingest_suppliers  # noqa: PLC0415

    return ingest_suppliers(path).accepted


def iter_fixture_files(directory: Path | str = DEFAULT_FIXTURE_DIR) -> Iterator[Path]:
    directory = Path(directory)
    if directory.is_dir():
        yield from sorted(p for p in directory.iterdir() if p.suffix in (".csv", ".json"))


__all__ = [
    "DEFAULT_FIXTURE_DIR",
    "FEEDS_VERSION",
    "CsvObservationFeed",
    "DiscoveryFeed",
    "FeedError",
    "FeedResult",
    "SellerApiFeed",
    "TrendsExportFeed",
    "UnavailableFeed",
    "build_registry",
    "collect",
    "coverage_report",
    "iter_fixture_files",
    "load_supplier_candidates",
]
