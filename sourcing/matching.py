"""Matching: which observations are the same product, and who can supply it.

Two matching problems, one discipline. Deterministic evidence decides;
similarity only proposes. A GTIN or a manufacturer model code shared between
two records is near-conclusive. A token overlap is a hypothesis, and a
hypothesis about product identity that turns into a purchase order is how you
end up with 200 units of the wrong thing.

So every match carries a confidence and the evidence that produced it, and
anything below the confident band is routed to review rather than resolved.
Claude may be asked for an opinion; its answer is a *finding* appended to the
evidence, never the verdict.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from catalog.schemas import SupplierOffer
from econ.money import ZERO, money
from part2.intelligence import (
    SupplierHistory,
    SupplierScore,
    rank_suppliers,
    select_supplier,
)
from sourcing.model import DiscoveredProduct, normalize_title, tokenize

logger = logging.getLogger(__name__)

MATCHING_VERSION = "matching/1.0.0"


class MatchConfidence(StrEnum):
    """How much weight a match can bear."""

    EXACT = "EXACT"
    HIGH = "HIGH"
    PROBABLE = "PROBABLE"
    WEAK = "WEAK"
    NONE = "NONE"

    @property
    def is_actionable(self) -> bool:
        """Whether a match may be acted on without a human confirming it."""
        return self in (MatchConfidence.EXACT, MatchConfidence.HIGH)


#: Token overlap needed before similarity is even called PROBABLE. Set high
#: deliberately: Indian marketplace titles share a lot of generic vocabulary,
#: so a low threshold matches everything to everything.
PROBABLE_THRESHOLD = Decimal("0.62")
HIGH_THRESHOLD = Decimal("0.82")


@dataclass(frozen=True)
class MatchEvidence:
    """One reason two records might describe the same product."""

    kind: str
    detail: str
    weight: MatchConfidence

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail, "weight": str(self.weight)}


@dataclass(frozen=True)
class Match:
    """A proposed identity relationship, with its evidence."""

    left_ref: str
    right_ref: str
    confidence: MatchConfidence
    similarity: Decimal
    evidence: list[MatchEvidence] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    requires_review: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "confidence": str(self.confidence),
            "similarity": str(self.similarity),
            "requires_review": self.requires_review,
            "evidence": [e.as_dict() for e in self.evidence],
            "findings": list(self.findings),
        }


def jaccard(left: frozenset[str], right: frozenset[str]) -> Decimal:
    """Token overlap between two titles, 0 to 1."""
    if not left or not right:
        return ZERO
    intersection = len(left & right)
    union = len(left | right)
    return money(Decimal(intersection) / Decimal(union))


def compare(left: DiscoveredProduct, right: DiscoveredProduct) -> Match:
    """Assess whether two observations describe the same product.

    Ordered by strength of evidence. A GTIN match ends the question; a model
    code very nearly does; similarity alone never does.
    """
    evidence: list[MatchEvidence] = []
    similarity = jaccard(left.match_tokens, right.match_tokens)

    if left.gtin and right.gtin:
        if left.gtin == right.gtin:
            evidence.append(
                MatchEvidence(
                    "gtin", f"both carry GTIN {left.gtin}", MatchConfidence.EXACT
                )
            )
            return Match(
                left.external_ref,
                right.external_ref,
                MatchConfidence.EXACT,
                similarity,
                evidence,
            )
        # Two different GTINs is positive evidence they are NOT the same.
        evidence.append(
            MatchEvidence(
                "gtin_conflict",
                f"different GTINs ({left.gtin} vs {right.gtin})",
                MatchConfidence.NONE,
            )
        )
        return Match(
            left.external_ref,
            right.external_ref,
            MatchConfidence.NONE,
            similarity,
            evidence,
        )

    if left.model_code and right.model_code and left.model_code == right.model_code:
        evidence.append(
            MatchEvidence(
                "model_code", f"both carry model {left.model_code}", MatchConfidence.HIGH
            )
        )
        if (
            left.brand
            and right.brand
            and normalize_title(left.brand) != normalize_title(right.brand)
        ):
            evidence.append(
                MatchEvidence(
                    "brand_conflict",
                    f"model matches but brands differ ({left.brand} vs {right.brand})",
                    MatchConfidence.PROBABLE,
                )
            )
            return Match(
                left.external_ref,
                right.external_ref,
                MatchConfidence.PROBABLE,
                similarity,
                evidence,
                requires_review=True,
            )
        return Match(
            left.external_ref,
            right.external_ref,
            MatchConfidence.HIGH,
            similarity,
            evidence,
        )

    if similarity >= HIGH_THRESHOLD:
        evidence.append(
            MatchEvidence(
                "similarity", f"token overlap {similarity}", MatchConfidence.PROBABLE
            )
        )
        return Match(
            left.external_ref,
            right.external_ref,
            MatchConfidence.PROBABLE,
            similarity,
            evidence,
            requires_review=True,
        )

    if similarity >= PROBABLE_THRESHOLD:
        evidence.append(
            MatchEvidence(
                "similarity", f"token overlap {similarity}", MatchConfidence.WEAK
            )
        )
        return Match(
            left.external_ref,
            right.external_ref,
            MatchConfidence.WEAK,
            similarity,
            evidence,
            requires_review=True,
        )

    return Match(
        left.external_ref, right.external_ref, MatchConfidence.NONE, similarity, evidence
    )


def consult_identity_judge(
    left: DiscoveredProduct, right: DiscoveredProduct, match: Match, judge: Any
) -> Match:
    """Ask Claude about an uncertain match and append its findings.

    Advisory only. The findings become part of the evidence a human reads; the
    confidence band is unchanged, because a model's opinion about product
    identity is not the kind of thing that should silently authorise a
    purchase.
    """
    from llm.services import build_identity_prompt  # noqa: PLC0415

    outcome = judge.invoke(
        build_identity_prompt(
            [
                {"ref": left.external_ref, "title": left.raw_title, "brand": left.brand},
                {
                    "ref": right.external_ref,
                    "title": right.raw_title,
                    "brand": right.brand,
                },
            ]
        ),
        input_refs=[left.external_ref, right.external_ref],
    )
    findings = list(outcome.findings) if outcome.usable else []
    if not outcome.usable:
        findings = [
            {
                "flag": "INSUFFICIENT_EVIDENCE",
                "confidence": "low",
                "reasoning": f"identity judge unavailable: {outcome.error or outcome.schema_violations}",
            }
        ]
    return Match(
        left_ref=match.left_ref,
        right_ref=match.right_ref,
        confidence=match.confidence,
        similarity=match.similarity,
        evidence=match.evidence,
        findings=findings,
        requires_review=match.requires_review,
    )


@dataclass
class ProductGroup:
    """A set of observations believed to describe one product."""

    key: str
    members: list[DiscoveredProduct]
    confidence: MatchConfidence
    matches: list[Match] = field(default_factory=list)

    @property
    def canonical(self) -> DiscoveredProduct:
        """The best-evidenced member, used as the group's representative."""
        return max(self.members, key=lambda p: (p.coverage_pct, p.observed_at))

    @property
    def requires_review(self) -> bool:
        return any(m.requires_review for m in self.matches)

    @property
    def slug(self) -> str:
        """URL-safe identifier for this group.

        The key is a human-readable identity string -- a GTIN, a model code, or
        a normalized title -- and titles contain spaces and occasionally
        slashes. A slash survives URL encoding into a path segment and breaks
        routing, so routing uses this instead and the key stays for display.
        """
        cleaned = re.sub(r"[^a-z0-9]+", "-", self.key.lower()).strip("-")
        if not cleaned:
            cleaned = "group"
        digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned[:48]}-{digest}"

    @property
    def observed_prices(self) -> list[Decimal]:
        return [
            money(p.observed_price_inr.value)
            for p in self.members
            if p.observed_price_inr.present
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "slug": self.slug,
            "size": len(self.members),
            "confidence": str(self.confidence),
            "requires_review": self.requires_review,
            "canonical_title": self.canonical.raw_title,
            "sources": sorted({p.source for p in self.members}),
            "coverage_pct": str(self.canonical.coverage_pct),
        }


def group_products(
    products: list[DiscoveredProduct], *, judge: Any | None = None
) -> list[ProductGroup]:
    """Collapse observations onto distinct products.

    Only actionable matches merge. A PROBABLE or WEAK relationship keeps the
    records apart and records the suspicion, because merging two different
    products silently averages their prices and produces economics for
    something that does not exist.
    """
    groups: list[ProductGroup] = []

    for product in products:
        placed = False
        for group in groups:
            match = compare(group.canonical, product)
            if judge is not None and match.requires_review:
                match = consult_identity_judge(group.canonical, product, match, judge)

            if match.confidence.is_actionable:
                group.members.append(product)
                group.matches.append(match)
                if match.confidence is MatchConfidence.HIGH:
                    group.confidence = MatchConfidence.HIGH
                placed = True
                break
            if match.requires_review:
                # Kept apart, but the suspicion is recorded on both sides.
                group.matches.append(match)

        if not placed:
            key = product.gtin or product.model_code or product.normalized_title[:48]
            groups.append(
                ProductGroup(key=key, members=[product], confidence=MatchConfidence.EXACT)
            )

    return groups


# --------------------------------------------------------------------------
# Supplier matching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SupplierMatch:
    """A supplier offer proposed as a source for a discovered product."""

    group_key: str
    offer: SupplierOffer
    supplier_name: str
    confidence: MatchConfidence
    similarity: Decimal
    evidence: list[MatchEvidence] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.confidence.is_actionable

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "supplier_name": self.supplier_name,
            "unit_cost_inr": str(self.offer.unit_cost_inr),
            "moq_units": self.offer.moq_units,
            "tax_invoice": self.offer.tax_invoice_available,
            "confidence": str(self.confidence),
            "similarity": str(self.similarity),
            "evidence": [e.as_dict() for e in self.evidence],
        }


@dataclass(frozen=True)
class SupplierCandidate:
    """A supplier catalogue row, before it is matched to anything."""

    supplier_name: str
    title: str
    offer: SupplierOffer
    gtin: str | None = None
    model_code: str | None = None
    #: None means the catalogue did not state a dispatch time. Seven days was
    #: previously invented here, which fed a lead-time penalty in supplier
    #: ranking as though it had been measured.
    lead_time_days: int | None = None

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(tokenize(self.title))


def match_suppliers(
    group: ProductGroup, candidates: list[SupplierCandidate]
) -> list[SupplierMatch]:
    """Find supplier offers that plausibly supply this product.

    A discovery source is never a candidate here. The candidates come from
    ingested supplier catalogues, which is the only place an actual
    acquisition cost can come from.
    """
    canonical = group.canonical
    matches: list[SupplierMatch] = []

    for candidate in candidates:
        evidence: list[MatchEvidence] = []
        similarity = jaccard(canonical.match_tokens, candidate.tokens)

        if canonical.gtin and candidate.gtin and canonical.gtin == candidate.gtin:
            confidence = MatchConfidence.EXACT
            evidence.append(
                MatchEvidence("gtin", f"GTIN {canonical.gtin}", MatchConfidence.EXACT)
            )
        elif (
            canonical.model_code
            and candidate.model_code
            and canonical.model_code == candidate.model_code
        ):
            confidence = MatchConfidence.HIGH
            evidence.append(
                MatchEvidence(
                    "model_code", f"model {canonical.model_code}", MatchConfidence.HIGH
                )
            )
        elif similarity >= HIGH_THRESHOLD:
            confidence = MatchConfidence.PROBABLE
            evidence.append(
                MatchEvidence(
                    "similarity", f"token overlap {similarity}", MatchConfidence.PROBABLE
                )
            )
        elif similarity >= PROBABLE_THRESHOLD:
            confidence = MatchConfidence.WEAK
            evidence.append(
                MatchEvidence(
                    "similarity", f"token overlap {similarity}", MatchConfidence.WEAK
                )
            )
        else:
            continue

        matches.append(
            SupplierMatch(
                group_key=group.key,
                offer=candidate.offer,
                supplier_name=candidate.supplier_name,
                confidence=confidence,
                similarity=similarity,
                evidence=evidence,
            )
        )

    order = [
        MatchConfidence.EXACT,
        MatchConfidence.HIGH,
        MatchConfidence.PROBABLE,
        MatchConfidence.WEAK,
    ]
    return sorted(matches, key=lambda m: (order.index(m.confidence), -m.similarity))


def rank_matched_suppliers(
    matches: list[SupplierMatch],
    *,
    history: dict[str, SupplierHistory] | None = None,
    expected_monthly_velocity: Decimal = Decimal("10"),
    capital_cap_inr: Decimal = Decimal("15000"),
) -> tuple[list[SupplierScore], SupplierScore | None]:
    """Rank usable supplier matches using the existing Part 2 engine.

    No economics are recomputed here. ``rank_suppliers`` already prices landed
    cost, MOQ capital and the missing input tax credit; duplicating any of it
    would create a second definition of what a supplier costs.
    """
    usable = [m for m in matches if m.usable]
    if not usable:
        return [], None

    names = {str(m.offer.supplier_id): m.supplier_name for m in usable}
    scores = rank_suppliers(
        [m.offer for m in usable],
        names=names,
        history=history,
        expected_monthly_velocity=expected_monthly_velocity,
        capital_cap_inr=capital_cap_inr,
    )
    return scores, select_supplier(scores)


__all__ = [
    "HIGH_THRESHOLD",
    "MATCHING_VERSION",
    "PROBABLE_THRESHOLD",
    "Match",
    "MatchConfidence",
    "MatchEvidence",
    "ProductGroup",
    "SupplierCandidate",
    "SupplierMatch",
    "compare",
    "consult_identity_judge",
    "group_products",
    "jaccard",
    "match_suppliers",
    "rank_matched_suppliers",
]
