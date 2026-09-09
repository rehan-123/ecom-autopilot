"""Versioned reference data: rate cards, tax tables, benchmarks, rulesets.

Every rate, rule and benchmark a calculation depends on is reachable only
through this module, so an assessment can pin the exact version it used and be
replayed years later against the same inputs.

Two properties are load-bearing:

**Nothing here is invented.** The bundled files under ``data/reference/`` ship
with ``last_verified_at: null`` and ``status: UNVERIFIED``. They carry the
*shape* of the data, not authoritative figures. Real rates must be entered and
verified against the published source before any decision built on them can
pass.

**Unverified and stale data cannot yield a PASS.** ``verification_state``
reports whether a version is usable, and the decision engine downgrades any
result resting on unverified or stale reference data to
``REQUIRES_HUMAN_REVIEW``. "No matching rule" is only evidence when the
rulebook is current.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_DIR = Path("data/reference")

#: A reference version older than this is stale. Rate cards and statutory
#: lists change by notification, so an unchecked figure decays quickly.
DEFAULT_MAX_AGE_DAYS = 90


class VerificationState(StrEnum):
    """Whether a reference version may be relied upon."""

    VERIFIED = "VERIFIED"
    STALE = "STALE"
    UNVERIFIED = "UNVERIFIED"
    MISSING = "MISSING"


class ReferenceError(RuntimeError):
    """Raised when required reference data is absent or malformed."""


@dataclass(frozen=True)
class ReferenceDoc:
    """One versioned reference document."""

    kind: str
    version: str
    effective_from: datetime
    effective_to: datetime | None
    source: str | None
    last_verified_at: datetime | None
    payload: dict[str, Any]
    path: Path

    def verification_state(
        self, *, now: datetime | None = None, max_age_days: int = DEFAULT_MAX_AGE_DAYS
    ) -> VerificationState:
        if self.last_verified_at is None:
            return VerificationState.UNVERIFIED
        now = now or datetime.now(UTC)
        if now - self.last_verified_at > timedelta(days=max_age_days):
            return VerificationState.STALE
        return VerificationState.VERIFIED

    def is_usable_for_pass(
        self, *, now: datetime | None = None, max_age_days: int = DEFAULT_MAX_AGE_DAYS
    ) -> bool:
        return (
            self.verification_state(now=now, max_age_days=max_age_days)
            is VerificationState.VERIFIED
        )


def _parse_dt(value: Any, field: str, path: Path) -> datetime | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ReferenceError(f"{path}: {field} is not ISO-8601: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _decimalise(value: Any) -> Any:
    """Convert numeric leaves to Decimal.

    YAML parses ``18.5`` as a float. Letting a float reach the economics engine
    would defeat the Decimal discipline everywhere else, so conversion happens
    at the boundary where the data enters the system.
    """
    if isinstance(value, dict):
        return {k: _decimalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimalise(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return value


class ReferenceStore:
    """Loads and indexes versioned reference documents from disk."""

    REQUIRED_KEYS = ("kind", "version", "effective_from")

    def __init__(self, root: Path | str = DEFAULT_REFERENCE_DIR) -> None:
        self.root = Path(root)
        self._docs: dict[tuple[str, str], ReferenceDoc] = {}
        self._loaded = False

    def load(self) -> ReferenceStore:
        """Read every YAML document under the reference root."""
        self._docs.clear()
        if not self.root.is_dir():
            raise ReferenceError(f"reference directory not found: {self.root}")

        for path in sorted(self.root.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            missing = [k for k in self.REQUIRED_KEYS if k not in raw]
            if missing:
                raise ReferenceError(f"{path}: missing key(s) {missing}")

            doc = ReferenceDoc(
                kind=str(raw["kind"]),
                version=str(raw["version"]),
                effective_from=_parse_dt(raw["effective_from"], "effective_from", path),
                effective_to=_parse_dt(raw.get("effective_to"), "effective_to", path),
                source=raw.get("source"),
                last_verified_at=_parse_dt(
                    raw.get("last_verified_at"), "last_verified_at", path
                ),
                payload=_decimalise(raw.get("payload", {})),
                path=path,
            )
            key = (doc.kind, doc.version)
            if key in self._docs:
                raise ReferenceError(f"duplicate reference {key} in {path}")
            self._docs[key] = doc

        self._loaded = True
        logger.info("reference data loaded", extra={"documents": len(self._docs)})
        return self

    def _require_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, kind: str, version: str) -> ReferenceDoc:
        self._require_loaded()
        try:
            return self._docs[(kind, version)]
        except KeyError as exc:
            available = sorted(v for k, v in self._docs if k == kind)
            raise ReferenceError(
                f"no {kind} version {version!r}; available: {available or 'none'}"
            ) from exc

    def versions(self, kind: str) -> list[str]:
        self._require_loaded()
        return sorted(v for k, v in self._docs if k == kind)

    def latest(self, kind: str, *, as_of: datetime | None = None) -> ReferenceDoc:
        """The version in force at ``as_of``, newest ``effective_from`` first."""
        self._require_loaded()
        as_of = as_of or datetime.now(UTC)
        candidates = [
            doc
            for (k, _), doc in self._docs.items()
            if k == kind
            and doc.effective_from <= as_of
            and (doc.effective_to is None or doc.effective_to > as_of)
        ]
        if not candidates:
            raise ReferenceError(f"no {kind} in force at {as_of.isoformat()}")
        return max(candidates, key=lambda d: d.effective_from)

    def all_docs(self) -> list[ReferenceDoc]:
        self._require_loaded()
        return sorted(self._docs.values(), key=lambda d: (d.kind, d.version))

    def audit(
        self, *, now: datetime | None = None, max_age_days: int = DEFAULT_MAX_AGE_DAYS
    ) -> dict[str, VerificationState]:
        """Verification state for every loaded document, for reporting."""
        return {
            f"{doc.kind}:{doc.version}": doc.verification_state(
                now=now, max_age_days=max_age_days
            )
            for doc in self.all_docs()
        }


@dataclass(frozen=True)
class PinnedReferences:
    """The exact reference versions one assessment used.

    Carried on every economics snapshot. Without it, a February calculation
    replayed today silently uses today's rate card and produces a different
    answer -- and the ledger becomes fiction, which is worse than no ledger
    because it is trusted.
    """

    rate_card: ReferenceDoc
    tax_table: ReferenceDoc
    return_benchmark: ReferenceDoc
    threshold_profile: ReferenceDoc
    scenario_profile: ReferenceDoc

    def as_versions(self) -> dict[str, str]:
        return {
            "rate_card_version": self.rate_card.version,
            "tax_table_version": self.tax_table.version,
            "return_benchmark_version": self.return_benchmark.version,
            "threshold_profile_version": self.threshold_profile.version,
            "scenario_profile_version": self.scenario_profile.version,
        }

    def docs(self) -> tuple[ReferenceDoc, ...]:
        return (
            self.rate_card,
            self.tax_table,
            self.return_benchmark,
            self.threshold_profile,
            self.scenario_profile,
        )

    def weakest_state(
        self, *, now: datetime | None = None, max_age_days: int = DEFAULT_MAX_AGE_DAYS
    ) -> VerificationState:
        """The worst verification state among the pinned documents.

        A chain is only as trustworthy as its least-verified link, so a single
        unverified rate card taints the whole assessment.
        """
        order = [
            VerificationState.VERIFIED,
            VerificationState.STALE,
            VerificationState.UNVERIFIED,
            VerificationState.MISSING,
        ]
        states = [
            d.verification_state(now=now, max_age_days=max_age_days) for d in self.docs()
        ]
        return max(states, key=order.index)

    def unverified_docs(
        self, *, now: datetime | None = None, max_age_days: int = DEFAULT_MAX_AGE_DAYS
    ) -> list[str]:
        return [
            f"{d.kind}:{d.version}"
            for d in self.docs()
            if not d.is_usable_for_pass(now=now, max_age_days=max_age_days)
        ]


def _benchmark_for(
    store: ReferenceStore, category: str, *, as_of: datetime | None = None
) -> ReferenceDoc:
    """Return benchmark for a category, falling back to the default.

    A catalogue contains far more categories than anyone has settlement
    history for, and refusing to price a product because its category has no
    bespoke benchmark would make the engine unusable on real data. The default
    is a weaker input, not a missing one -- and it carries its own verification
    state, so an assessment resting on it is still gated accordingly.
    """
    try:
        return store.latest(f"return_benchmark:{category}", as_of=as_of)
    except ReferenceError:
        logger.info(
            "no category return benchmark; using the default",
            extra={"category": category},
        )
        return store.latest("return_benchmark:default", as_of=as_of)


def pin(
    store: ReferenceStore,
    *,
    marketplace: str,
    category: str,
    risk_tier: str,
    scenario: str,
    as_of: datetime | None = None,
) -> PinnedReferences:
    """Resolve the reference versions in force for one assessment.

    No ``hsn_code`` parameter: a single tax_table document covers every HSN,
    and the per-HSN rate is resolved inside the economics calculator. Taking
    an argument that never affects the result would imply a versioning
    dimension that does not exist.
    """
    return PinnedReferences(
        rate_card=store.latest(f"rate_card:{marketplace}", as_of=as_of),
        tax_table=store.latest("tax_table", as_of=as_of),
        return_benchmark=_benchmark_for(store, category, as_of=as_of),
        threshold_profile=store.latest(f"threshold_profile:{risk_tier}", as_of=as_of),
        scenario_profile=store.latest(f"scenario_profile:{scenario}", as_of=as_of),
    )


__all__ = [
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_REFERENCE_DIR",
    "PinnedReferences",
    "ReferenceDoc",
    "ReferenceError",
    "ReferenceStore",
    "VerificationState",
    "pin",
]
