"""The normalized discovery model: a candidate product, and how well evidenced.

A discovered product is an *observation*, not a fact about the market. It says
"this appeared on this source at this price on this date", and every field
carries where it came from. That distinction is the whole reason this layer
exists separately from the catalogue: a marketplace listing tells you
something was on sale, not that it sells, and certainly not that you can
source it profitably.

Coverage is computed, never assumed. Roughly half the interesting signals for
Indian marketplaces have no legitimate source, so a candidate with a price and
nothing else is normal — and must be visibly distinguishable from one backed by
trend data and competitor counts.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from econ.money import HUNDRED, ZERO, money
from signals.market import Signal

logger = logging.getLogger(__name__)

DISCOVERY_MODEL_VERSION = "sourcing/1.0.0"


class FieldSource(StrEnum):
    """Where one observed field came from."""

    SELLER_EXPORT = "SELLER_EXPORT"
    PUBLIC_DATASET = "PUBLIC_DATASET"
    OFFICIAL_API = "OFFICIAL_API"
    MANUAL = "MANUAL"
    DERIVED = "DERIVED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class Observed:
    """One observed value with its provenance.

    ``value is None`` means the field was not obtainable, and the reason says
    why. That is a different statement from zero, and the difference decides
    whether a product is ranked or held back.
    """

    value: Decimal | int | str | None
    source: FieldSource
    reference: str | None = None
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and self.source is FieldSource.UNAVAILABLE:
            raise ValueError(
                "an UNAVAILABLE field cannot carry a value; that is the whole "
                "point of recording it as unavailable"
            )

    @property
    def present(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else str(self.value),
            "source": str(self.source),
            "reference": self.reference,
            "reason": self.reason,
        }


def unavailable(reason: str) -> Observed:
    """Record that a field has no legitimate source."""
    return Observed(None, FieldSource.UNAVAILABLE, reason=reason)


#: Signal fields the discovery model can carry. Each maps to a weight in the
#: coverage calculation; adding one is an entry here plus a weight.
SIGNAL_FIELDS: tuple[str, ...] = (
    "observed_price_inr",
    "rank",
    "trend",
    "review_count",
    "review_velocity",
    "search_interest",
    "competitor_count",
    "brand_dominance",
    "availability",
)

#: Relative importance for coverage. Price dominates because without it there
#: is nothing to test economics against; the rest are corroboration.
COVERAGE_WEIGHTS: dict[str, Decimal] = {
    "observed_price_inr": Decimal("30"),
    "search_interest": Decimal("15"),
    "trend": Decimal("15"),
    "competitor_count": Decimal("12"),
    "review_count": Decimal("8"),
    "review_velocity": Decimal("8"),
    "brand_dominance": Decimal("6"),
    "rank": Decimal("3"),
    "availability": Decimal("3"),
}


@dataclass
class DiscoveredProduct:
    """A candidate product as observed on one source."""

    source: str
    external_ref: str
    raw_title: str
    category: str | None = None
    brand: str | None = None
    model_code: str | None = None
    gtin: str | None = None
    hsn_code: str | None = None
    weight_g: int | None = None

    observed_price_inr: Observed = field(
        default_factory=lambda: unavailable("not collected")
    )
    rank: Observed = field(default_factory=lambda: unavailable("not collected"))
    trend: Observed = field(default_factory=lambda: unavailable("not collected"))
    review_count: Observed = field(default_factory=lambda: unavailable("not collected"))
    review_velocity: Observed = field(
        default_factory=lambda: unavailable("not collected")
    )
    search_interest: Observed = field(
        default_factory=lambda: unavailable("not collected")
    )
    competitor_count: Observed = field(
        default_factory=lambda: unavailable("not collected")
    )
    brand_dominance: Observed = field(
        default_factory=lambda: unavailable("not collected")
    )
    availability: Observed = field(default_factory=lambda: unavailable("not collected"))

    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    notes: list[str] = field(default_factory=list)

    # -- derived -----------------------------------------------------------

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.raw_title)

    @property
    def match_tokens(self) -> frozenset[str]:
        return frozenset(tokenize(self.raw_title))

    def signal_field(self, name: str) -> Observed:
        return getattr(self, name)

    @property
    def coverage_pct(self) -> Decimal:
        """Share of weighted signal fields that were actually obtained."""
        total = sum(COVERAGE_WEIGHTS.values(), ZERO)
        got = sum(
            (
                COVERAGE_WEIGHTS[name]
                for name in SIGNAL_FIELDS
                if self.signal_field(name).present
            ),
            ZERO,
        )
        return money(got / total * HUNDRED) if total else ZERO

    @property
    def missing_signals(self) -> list[str]:
        return [name for name in SIGNAL_FIELDS if not self.signal_field(name).present]

    @property
    def has_price(self) -> bool:
        return self.observed_price_inr.present

    def to_signals(self, *, collected_by: str = "discovery") -> dict[str, Signal]:
        """Project onto the Part 1 signal vocabulary.

        The scoring engine already knows how to weigh signals and how to refuse
        to score when coverage is thin. Translating rather than reimplementing
        keeps one definition of what a signal means.
        """
        from catalog.enums import SignalSource, SignalStatus  # noqa: PLC0415

        mapping = {
            "search_interest": "SEARCH_INTEREST",
            "trend": "TREND_DIRECTION",
            "review_count": "REVIEW_VOLUME",
            "review_velocity": "REVIEW_VELOCITY",
            "competitor_count": "COMPETITOR_COUNT",
            "brand_dominance": "BRAND_DOMINANCE",
        }
        source_map = {
            FieldSource.SELLER_EXPORT: SignalSource.SELLER_REPORT,
            FieldSource.OFFICIAL_API: SignalSource.OTHER,
            FieldSource.PUBLIC_DATASET: SignalSource.GOOGLE_TRENDS,
            FieldSource.MANUAL: SignalSource.MANUAL,
            FieldSource.DERIVED: SignalSource.OTHER,
            FieldSource.UNAVAILABLE: SignalSource.MANUAL,
        }

        signals: dict[str, Signal] = {}
        for attribute, signal_type in mapping.items():
            observed = self.signal_field(attribute)
            source = source_map.get(observed.source, SignalSource.MANUAL)
            if not observed.present:
                signals[signal_type] = Signal(
                    signal_type=signal_type,
                    status=SignalStatus.DATA_UNAVAILABLE,
                    source=source,
                    collected_by=collected_by,
                    source_ref=observed.reason,
                )
                continue
            signals[signal_type] = Signal(
                signal_type=signal_type,
                status=SignalStatus.OBSERVED,
                source=source,
                collected_by=collected_by,
                value_numeric=money(observed.value),
                source_ref=observed.reference,
                collected_at=observed.observed_at,
            )
        return signals

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "source": self.source,
            "external_ref": self.external_ref,
            "raw_title": self.raw_title,
            "normalized_title": self.normalized_title,
            "brand": self.brand,
            "model_code": self.model_code,
            "gtin": self.gtin,
            "category": self.category,
            "coverage_pct": str(self.coverage_pct),
            "missing_signals": self.missing_signals,
            "signals": {
                name: self.signal_field(name).as_dict() for name in SIGNAL_FIELDS
            },
            "observed_at": self.observed_at.isoformat(),
        }


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

#: Marketing noise that carries no identity. Removing it is what lets three
#: suppliers' wildly different titles collapse onto one product.
_NOISE = frozenset(
    {
        "new",
        "original",
        "genuine",
        "premium",
        "best",
        "quality",
        "hot",
        "sale",
        "offer",
        "combo",
        "pack",
        "set",
        "free",
        "shipping",
        "with",
        "for",
        "and",
        "the",
        "a",
        "an",
        "of",
        "in",
        "professional",
        "pro",
        "rechargeable",
        "portable",
        "high",
        "super",
        "ultra",
        "latest",
    }
)

_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:ml|l|g|kg|mm|cm|m|w|v|mah|inch|in|pcs?|pieces?)\b",
    re.IGNORECASE,
)
_MODEL_PATTERN = re.compile(r"\b([A-Z]{1,4}[-\s]?\d{2,5}[A-Z]?)\b")
_GTIN_PATTERN = re.compile(r"\b(\d{8}|\d{12,14})\b")


def normalize_title(raw: str) -> str:
    """Reduce a title to comparable form.

    Case, accents, punctuation and marketing noise are stripped; units and
    model codes are kept because they carry identity. "Kemei KX-880
    Professional Trimmer" and "Rechargeable KX880 Clipper" should end up
    sharing their distinguishing token.
    """
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(raw: str) -> list[str]:
    """Identity-bearing tokens from a title, noise removed."""
    normalized = normalize_title(raw)
    # Collapse a spaced or hyphenated model code so KX-880 and KX880 agree.
    normalized = re.sub(r"\b([a-z]{1,4})[-\s](\d{2,5}[a-z]?)\b", r"\1\2", normalized)
    tokens = [t for t in re.split(r"[\s.-]+", normalized) if t]
    return [t for t in tokens if t not in _NOISE and len(t) > 1]


def normalize_model_code(code: str | None) -> str | None:
    """Reduce a model code to one comparable form.

    Applied to codes supplied in a column as well as to extracted ones. When
    only extraction normalised, a catalogue that wrote "ML-750" and a title
    that said "ML750" produced two products where there was one -- and two
    half-evidenced products rank below the one well-evidenced product they
    should have been.
    """
    if code is None:
        return None
    cleaned = re.sub(r"[\s\-_.]+", "", code).upper()
    return cleaned or None


def extract_model_code(raw: str) -> str | None:
    """Pull a manufacturer model code out of a title, if one is present.

    A model code is the strongest identity signal available without a GTIN:
    two listings sharing one are almost certainly the same product, whatever
    else their titles say.
    """
    candidate = re.sub(r"\b([A-Za-z]{1,4})[-\s](\d{2,5}[A-Za-z]?)\b", r"\1\2", raw)
    match = _MODEL_PATTERN.search(candidate.upper())
    if not match:
        return None
    code = normalize_model_code(match.group(1))
    # A bare unit reading like "750ML" is not a model code.
    if code is None or _UNIT_PATTERN.fullmatch(code):
        return None
    return code


def extract_gtin(raw: str) -> str | None:
    match = _GTIN_PATTERN.search(raw)
    return match.group(1) if match else None


def guess_brand(raw: str, known_brands: frozenset[str] | None = None) -> str | None:
    """Identify a brand only when it is recognised.

    Guessing that the first word of a title is the brand produces confident
    nonsense on half of Indian marketplace listings, so an unrecognised title
    returns None and the field stays empty.
    """
    if not known_brands:
        return None
    tokens = set(tokenize(raw))
    for brand in sorted(known_brands, key=len, reverse=True):
        if normalize_title(brand) in tokens:
            return brand
    return None


def enrich(
    product: DiscoveredProduct, *, known_brands: frozenset[str] | None = None
) -> DiscoveredProduct:
    """Fill derivable identity fields, never overwriting observed ones.

    A supplied model code is normalised rather than trusted verbatim, so it
    compares equal to one extracted from a title.
    """
    product.model_code = normalize_model_code(product.model_code) or extract_model_code(
        product.raw_title
    )
    if product.gtin is None:
        product.gtin = extract_gtin(product.raw_title)
    if product.brand is None:
        product.brand = guess_brand(product.raw_title, known_brands)
    return product


__all__ = [
    "COVERAGE_WEIGHTS",
    "DISCOVERY_MODEL_VERSION",
    "SIGNAL_FIELDS",
    "DiscoveredProduct",
    "FieldSource",
    "Observed",
    "enrich",
    "extract_gtin",
    "extract_model_code",
    "guess_brand",
    "normalize_model_code",
    "normalize_title",
    "tokenize",
    "unavailable",
]
