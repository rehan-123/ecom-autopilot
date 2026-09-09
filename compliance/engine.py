"""Compliance evaluation over versioned rules-as-data.

The regulatory knowledge lives in YAML under ``data/reference/compliance/``,
not in this file. BIS and QCO lists change by notification, and a hardcoded
list has no version and no verification date — so "we found no rule" would be
meaningless.

Two rules are absolute:

**UNKNOWN is never PASS.** An unresolved check routes to human review. The
categories under mandatory BIS certification carry statutory penalties, so an
ambiguous match is a blocker, not a warning.

**A stale rulebook cannot clear a product.** If the ruleset has not been
verified recently, "no matching requirement" stops being evidence of anything
and the result degrades to ``REQUIRES_HUMAN_REVIEW``.

Claude may read a supplier's certificate and report what it appears to say.
Claude never sets the result: category entry is a legal judgement with criminal
exposure attached.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from catalog.enums import ComplianceResult
from reference.loader import ReferenceDoc, VerificationState

logger = logging.getLogger(__name__)

COMPLIANCE_VERSION = "compliance/1.0.0"


class Severity(StrEnum):
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"


@dataclass(frozen=True)
class Requirement:
    """One rule from the versioned ruleset."""

    id: str
    requirement: str
    severity: Severity
    hsn_prefixes: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    evidence_required: tuple[str, ...] = ()
    note: str | None = None

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> Requirement:
        return cls(
            id=str(raw["id"]),
            requirement=str(raw.get("requirement", "EVIDENCE_REQUIRED")),
            severity=Severity(raw.get("severity", "BLOCKING")),
            hsn_prefixes=tuple(str(p) for p in raw.get("hsn_prefix", [])),
            keywords=tuple(str(k).lower() for k in raw.get("keywords", [])),
            evidence_required=tuple(str(e) for e in raw.get("evidence_required", [])),
            note=raw.get("note"),
        )

    def matches(self, *, hsn_code: str | None, text: str) -> tuple[bool, bool]:
        """Return ``(matched, exact)``.

        An HSN prefix hit is exact. A keyword hit alone is a fuzzy match, which
        resolves to UNKNOWN rather than FAIL: a product whose title merely
        mentions a regulated word may or may not be in scope, and that is a
        question for a human.
        """
        if hsn_code and any(hsn_code.startswith(p) for p in self.hsn_prefixes):
            return True, True
        haystack = text.lower()
        if any(re.search(rf"\b{re.escape(k)}\b", haystack) for k in self.keywords):
            return True, False
        return False, False


@dataclass(frozen=True)
class CheckOutcome:
    """The result of evaluating one requirement."""

    requirement_id: str
    result: ComplianceResult
    reason: str
    evidence_present: tuple[str, ...] = ()
    evidence_missing: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "result": str(self.result),
            "reason": self.reason,
            "evidence_present": list(self.evidence_present),
            "evidence_missing": list(self.evidence_missing),
        }


@dataclass(frozen=True)
class ComplianceOutcome:
    """The overall compliance verdict for a product."""

    overall: ComplianceResult
    checks: list[CheckOutcome]
    blocking_failures: list[str]
    ruleset_version: str
    ruleset_last_verified_at: datetime | None
    engine_version: str = COMPLIANCE_VERSION
    notes: list[str] = field(default_factory=list)
    #: Fields where two sources disagree. Never resolved silently.
    conflicts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": str(self.overall),
            "ruleset_version": self.ruleset_version,
            "ruleset_last_verified_at": (
                self.ruleset_last_verified_at.isoformat()
                if self.ruleset_last_verified_at
                else None
            ),
            "engine_version": self.engine_version,
            "checks": [c.as_dict() for c in self.checks],
            "blocking_failures": list(self.blocking_failures),
            "conflicts": list(self.conflicts),
            "notes": list(self.notes),
        }


def detect_conflicts(evidence: dict[str, Any]) -> list[str]:
    """Find compliance fields where two sources disagree.

    Convention: a caller supplying the same fact from more than one source
    passes a list. Two different values in that list is a contradiction, and
    picking one silently would make a compliance decision on evidence the
    system knows to be unreliable. The field is surfaced instead so a human
    resolves it.
    """
    conflicts: list[str] = []
    for key, value in sorted(evidence.items()):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        distinct = {str(v).strip().lower() for v in value if v not in (None, "")}
        if len(distinct) > 1:
            conflicts.append(f"{key}: sources disagree ({sorted(distinct)})")
    return conflicts


class HsnProvenance(StrEnum):
    """Where an HSN code came from.

    Only a sourced code is evidence. An HSN inferred from a product name, a
    category or a model's description is a hypothesis: HSN determines which
    regulatory rules apply at all, so a wrong one silently routes a regulated
    product around every requirement that governs it.
    """

    SOURCED = "SOURCED"
    AI_INFERENCE = "AI_INFERENCE"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def is_evidence(self) -> bool:
        return self is HsnProvenance.SOURCED


def _evidence_state(
    requirement: Requirement, evidence: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present = tuple(
        key
        for key in requirement.evidence_required
        if evidence.get(key) not in (None, "", [], {})
    )
    missing = tuple(k for k in requirement.evidence_required if k not in present)
    return present, missing


def evaluate(
    *,
    ruleset: ReferenceDoc,
    hsn_code: str | None,
    product_text: str,
    evidence: dict[str, Any] | None = None,
    hsn_provenance: HsnProvenance = HsnProvenance.UNAVAILABLE,
    now: datetime | None = None,
    max_age_days: int = 90,
) -> ComplianceOutcome:
    """Evaluate a product against a versioned compliance ruleset."""
    evidence = evidence or {}
    now = now or datetime.now(UTC)
    notes: list[str] = []

    conflicts = detect_conflicts(evidence)
    for conflict in conflicts:
        notes.append(f"conflicting evidence: {conflict}")

    requirements = [
        Requirement.from_payload(r) for r in ruleset.payload.get("requirements", [])
    ]
    state = ruleset.verification_state(now=now, max_age_days=max_age_days)
    ruleset_fresh = state is VerificationState.VERIFIED
    if not ruleset_fresh:
        notes.append(
            f"ruleset {ruleset.version} is {state}; absence of a matching rule is "
            "not evidence of compliance"
        )

    # An HSN code decides which requirements apply. Without one, "no rule
    # matched" means "no rule could match", which is not the same as "no rule
    # applies" -- and the difference is the whole question.
    hsn_is_evidence = bool(hsn_code) and hsn_provenance.is_evidence
    if hsn_code and not hsn_provenance.is_evidence:
        notes.append(
            f"HSN {hsn_code} has provenance {hsn_provenance}; an inferred code "
            "is not evidence of which requirements apply and cannot clear them"
        )
    elif not hsn_code:
        notes.append(
            "no HSN code: the requirements that apply to this product cannot "
            "be determined, so absence of a matching rule proves nothing"
        )

    checks: list[CheckOutcome] = []
    blocking: list[str] = []
    matched_any = False

    for requirement in requirements:
        matched, exact = requirement.matches(
            hsn_code=hsn_code if hsn_is_evidence else None, text=product_text
        )
        if not matched:
            continue
        matched_any = True
        present, missing = _evidence_state(requirement, evidence)

        if not exact:
            outcome = CheckOutcome(
                requirement.id,
                ComplianceResult.UNKNOWN,
                "keyword match only; a human must confirm whether this applies",
                present,
                missing,
            )
        elif not missing:
            outcome = CheckOutcome(
                requirement.id,
                ComplianceResult.PASS,
                "all required evidence supplied",
                present,
                missing,
            )
        elif requirement.severity is Severity.BLOCKING:
            outcome = CheckOutcome(
                requirement.id,
                ComplianceResult.FAIL,
                f"required evidence missing: {', '.join(missing)}",
                present,
                missing,
            )
            blocking.append(requirement.id)
        else:
            outcome = CheckOutcome(
                requirement.id,
                ComplianceResult.REQUIRES_HUMAN_REVIEW,
                f"advisory requirement lacks evidence: {', '.join(missing)}",
                present,
                missing,
            )
        checks.append(outcome)

    results = {c.result for c in checks}
    if ComplianceResult.FAIL in results:
        overall = ComplianceResult.FAIL
    elif (
        ComplianceResult.UNKNOWN in results
        or ComplianceResult.REQUIRES_HUMAN_REVIEW in results
    ):
        # UNKNOWN never resolves itself upward.
        overall = ComplianceResult.REQUIRES_HUMAN_REVIEW
    elif not ruleset_fresh:
        # A rulebook that has not been verified may be missing requirements
        # entirely, so satisfying the rules it does contain proves nothing
        # about the rules it does not. Nothing clears on a stale ruleset.
        overall = ComplianceResult.REQUIRES_HUMAN_REVIEW
        notes.append(
            "the rulebook is not current, so it may omit requirements that "
            "apply; no product can be cleared against it"
        )
    elif conflicts:
        # Two sources disagreeing is not a tie to be broken silently.
        overall = ComplianceResult.REQUIRES_HUMAN_REVIEW
    elif not hsn_is_evidence:
        # Reached when nothing matched and no sourced HSN exists. Clearing here
        # would mean a product with no classification evidence passes simply
        # because no rule could be tested against it.
        overall = ComplianceResult.UNKNOWN
        notes.append(
            "compliance cannot be determined without a sourced HSN code; this "
            "is UNKNOWN, not clearance"
        )
    elif not matched_any:
        overall = ComplianceResult.PASS
        notes.append("no requirement in the current ruleset applies to this product")
    else:
        overall = ComplianceResult.PASS

    logger.info(
        "compliance evaluated",
        extra={"overall": str(overall), "matched": len(checks)},
    )
    return ComplianceOutcome(
        overall=overall,
        checks=checks,
        blocking_failures=blocking,
        conflicts=tuple(conflicts),
        ruleset_version=ruleset.version,
        ruleset_last_verified_at=ruleset.last_verified_at,
        notes=notes,
    )


__all__ = [
    "COMPLIANCE_VERSION",
    "CheckOutcome",
    "ComplianceOutcome",
    "HsnProvenance",
    "Requirement",
    "Severity",
    "detect_conflicts",
    "evaluate",
]
