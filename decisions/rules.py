"""Deterministic decision rules and the Part 1 state machine.

Claude does not own state. The Judges emit findings; the table below decides
what a finding *means*, and the state machine decides where the SKU may go
next. Both are data, both are versioned, and both produce the same answer for
the same input every time.

That reproducibility is the whole point. When a SKU is rejected and the
supplier asks why, the answer has to be a rule name and its inputs — not a
paraphrase of what a model said on a Tuesday.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from catalog.enums import (
    ComplianceResult,
    GateResult,
    HeadroomResult,
    ScoreResult,
    SKUState,
)

logger = logging.getLogger(__name__)

RULES_VERSION = "rules/1.0.0"


class Disposition(StrEnum):
    """What the rule engine concluded."""

    CONTINUE = "CONTINUE"
    BLOCKED = "BLOCKED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    REJECTED = "REJECTED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"


@dataclass(frozen=True)
class Decision:
    """One rule firing, with the inputs that caused it."""

    rule_id: str
    disposition: Disposition
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)
    next_state: SKUState | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "disposition": str(self.disposition),
            "reason": self.reason,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "next_state": str(self.next_state) if self.next_state else None,
            "rules_version": RULES_VERSION,
        }


#: What each Judge finding flag means. Findings never map to approval — the
#: best a finding can do is fail to block.
FINDING_DISPOSITIONS: dict[str, Disposition] = {
    "SAME_PRODUCT": Disposition.CONTINUE,
    "DIFFERENT_PRODUCT": Disposition.CONTINUE,
    "SUSPECTED_DUPLICATE": Disposition.REQUIRES_HUMAN_REVIEW,
    "INSUFFICIENT_EVIDENCE": Disposition.REQUIRES_HUMAN_REVIEW,
    "QUALITY_CONCERN": Disposition.REQUIRES_HUMAN_REVIEW,
    "MISLEADING_CLAIM": Disposition.BLOCKED,
    "RETURN_RISK": Disposition.REQUIRES_HUMAN_REVIEW,
    "BRAND_RISK": Disposition.BLOCKED,
    "EVIDENCE_CONTRADICTION": Disposition.REQUIRES_HUMAN_REVIEW,
}

#: A low-confidence finding cannot block on its own; it routes to review so a
#: human decides whether the model was onto something.
_CONFIDENCE_SOFTENS = {"low"}


def interpret_findings(findings: list[dict[str, Any]], *, stage: str) -> list[Decision]:
    """Map Judge findings onto dispositions.

    An unrecognised flag is not ignored: a model inventing a new category is
    exactly the situation a human should look at.
    """
    decisions: list[Decision] = []
    for finding in findings:
        flag = str(finding.get("flag", "")).upper()
        confidence = str(finding.get("confidence", "low")).lower()
        disposition = FINDING_DISPOSITIONS.get(flag)

        if disposition is None:
            decisions.append(
                Decision(
                    rule_id=f"{stage}.unknown_flag",
                    disposition=Disposition.REQUIRES_HUMAN_REVIEW,
                    reason=f"unrecognised finding flag {flag!r}",
                    inputs={"flag": flag, "confidence": confidence},
                )
            )
            continue

        if disposition is Disposition.BLOCKED and confidence in _CONFIDENCE_SOFTENS:
            decisions.append(
                Decision(
                    rule_id=f"{stage}.{flag.lower()}_low_confidence",
                    disposition=Disposition.REQUIRES_HUMAN_REVIEW,
                    reason=f"{flag} reported at low confidence; a human decides",
                    inputs={"flag": flag, "confidence": confidence},
                )
            )
            continue

        decisions.append(
            Decision(
                rule_id=f"{stage}.{flag.lower()}",
                disposition=disposition,
                reason=str(finding.get("reasoning", flag))[:300],
                inputs={"flag": flag, "confidence": confidence},
            )
        )
    return decisions


def decide_profitability(gate: Any) -> Decision:
    if gate.result is GateResult.FAIL:
        return Decision(
            rule_id=f"profitability.{gate.failed_gate}",
            disposition=Disposition.REJECTED,
            reason=gate.gap_to_pass.get("action", "profitability floor not met"),
            inputs={"failed_gate": gate.failed_gate},
            next_state=SKUState.REJECTED,
        )
    if gate.result is GateResult.MARGINAL:
        return Decision(
            rule_id="profitability.marginal",
            disposition=Disposition.REQUIRES_HUMAN_REVIEW,
            reason=gate.gap_to_pass.get("action", "economics are marginal"),
            inputs={"result": gate.result},
            next_state=SKUState.BLOCKED_PENDING_REVIEW,
        )
    return Decision(
        rule_id="profitability.pass",
        disposition=Disposition.CONTINUE,
        reason="all profitability floors met, including the worst case",
        inputs={"result": gate.result},
    )


def decide_headroom(outcome: Any) -> Decision:
    if outcome.result is HeadroomResult.BLOCK:
        return Decision(
            rule_id="headroom.block",
            disposition=Disposition.REJECTED,
            reason=outcome.explanation,
            inputs={"headroom_pct": outcome.headroom_pct},
            next_state=SKUState.REJECTED,
        )
    if outcome.result is HeadroomResult.MARGINAL:
        return Decision(
            rule_id="headroom.marginal",
            disposition=Disposition.REQUIRES_HUMAN_REVIEW,
            reason=outcome.explanation,
            inputs={"headroom_pct": outcome.headroom_pct},
            next_state=SKUState.BLOCKED_PENDING_REVIEW,
        )
    return Decision(
        rule_id="headroom.pass",
        disposition=Disposition.CONTINUE,
        reason=outcome.explanation,
        inputs={"headroom_pct": outcome.headroom_pct},
    )


def decide_compliance(outcome: Any) -> Decision:
    """Compliance overrides everything.

    A high opportunity score cannot carry a product past a blocking
    certification failure, and UNKNOWN never resolves upward on its own.
    """
    if outcome.overall is ComplianceResult.FAIL:
        return Decision(
            rule_id="compliance.fail",
            disposition=Disposition.BLOCKED,
            reason=f"blocking requirements unmet: {', '.join(outcome.blocking_failures)}",
            inputs={"ruleset_version": outcome.ruleset_version},
            next_state=SKUState.BLOCKED,
        )
    if outcome.overall in (
        ComplianceResult.UNKNOWN,
        ComplianceResult.REQUIRES_HUMAN_REVIEW,
    ):
        return Decision(
            rule_id="compliance.requires_review",
            disposition=Disposition.REQUIRES_HUMAN_REVIEW,
            reason="compliance could not be established without human judgement",
            inputs={"overall": outcome.overall},
            next_state=SKUState.BLOCKED_PENDING_REVIEW,
        )
    return Decision(
        rule_id="compliance.pass",
        disposition=Disposition.CONTINUE,
        reason="all applicable requirements satisfied against a current ruleset",
        inputs={"ruleset_version": outcome.ruleset_version},
    )


def decide_coverage(score: Any) -> Decision:
    if score.result is ScoreResult.INSUFFICIENT_DATA:
        return Decision(
            rule_id="scoring.insufficient_data",
            disposition=Disposition.REQUIRES_HUMAN_REVIEW,
            reason=(
                f"evidence coverage {score.coverage_pct}% is below the floor; "
                f"missing: {', '.join(score.missing) or 'none'}"
            ),
            inputs={"coverage_pct": score.coverage_pct},
            next_state=SKUState.BLOCKED_AWAITING_DATA,
        )
    return Decision(
        rule_id="scoring.scored",
        disposition=Disposition.CONTINUE,
        reason=f"scored {score.display()}",
        inputs={"value": score.value, "coverage_pct": score.coverage_pct},
    )


def decide_reference_trust(weakest_state: Any, unverified: list[str]) -> Decision:
    """Refuse to approve on unverified or stale reference data.

    An assessment is only as trustworthy as the least-verified figure it rests
    on. Shipping template rate cards is safe precisely because this rule stops
    them producing a decision.
    """
    if str(weakest_state) == "VERIFIED":
        return Decision(
            rule_id="reference.verified",
            disposition=Disposition.CONTINUE,
            reason="all pinned reference data is verified and current",
        )
    return Decision(
        rule_id="reference.unverified",
        disposition=Disposition.REQUIRES_HUMAN_REVIEW,
        reason=(
            f"reference data is {weakest_state}: {', '.join(unverified)}. "
            "Verify against the published source before acting on this result."
        ),
        inputs={"weakest_state": weakest_state},
        next_state=SKUState.BLOCKED_PENDING_REVIEW,
    )


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

#: Legal transitions. Anything not listed is rejected, so a misrouted rule
#: cannot quietly advance a SKU past a gate it never cleared.
LEGAL_TRANSITIONS: dict[SKUState, frozenset[SKUState]] = {
    SKUState.DISCOVERED: frozenset({SKUState.DATA_COLLECTION, SKUState.REJECTED}),
    SKUState.DATA_COLLECTION: frozenset(
        {SKUState.ECONOMIC_ANALYSIS, SKUState.BLOCKED_AWAITING_DATA, SKUState.REJECTED}
    ),
    SKUState.ECONOMIC_ANALYSIS: frozenset(
        {
            SKUState.SIGNAL_COLLECTION,
            SKUState.REJECTED,
            SKUState.BLOCKED_PENDING_REVIEW,
        }
    ),
    SKUState.SIGNAL_COLLECTION: frozenset(
        {SKUState.PRICE_HEADROOM_CHECK, SKUState.BLOCKED_AWAITING_DATA}
    ),
    SKUState.BLOCKED_AWAITING_DATA: frozenset(
        {SKUState.SIGNAL_COLLECTION, SKUState.ECONOMIC_ANALYSIS, SKUState.REJECTED}
    ),
    SKUState.PRICE_HEADROOM_CHECK: frozenset(
        {
            SKUState.SUPPLIER_ANALYSIS,
            SKUState.REJECTED,
            SKUState.BLOCKED_PENDING_REVIEW,
        }
    ),
    SKUState.SUPPLIER_ANALYSIS: frozenset(
        {
            SKUState.RISK_ANALYSIS,
            SKUState.BLOCKED_PENDING_REVIEW,
            SKUState.BLOCKED_AWAITING_DATA,
        }
    ),
    SKUState.RISK_ANALYSIS: frozenset(
        {
            SKUState.COMPLIANCE_GATE,
            SKUState.BLOCKED_PENDING_REVIEW,
            SKUState.BLOCKED_AWAITING_DATA,
        }
    ),
    SKUState.COMPLIANCE_GATE: frozenset(
        {
            SKUState.AI_REVIEW,
            SKUState.BLOCKED,
            SKUState.BLOCKED_PENDING_REVIEW,
            # Evidence coverage is assessed here, and a gap means we are
            # waiting on data rather than on a human judgement.
            SKUState.BLOCKED_AWAITING_DATA,
        }
    ),
    SKUState.BLOCKED_PENDING_REVIEW: frozenset(
        {
            SKUState.ECONOMIC_ANALYSIS,
            SKUState.COMPLIANCE_GATE,
            SKUState.AI_REVIEW,
            SKUState.REJECTED,
            SKUState.BLOCKED,
        }
    ),
    SKUState.BLOCKED: frozenset({SKUState.COMPLIANCE_GATE, SKUState.REJECTED}),
    SKUState.AI_REVIEW: frozenset(
        {
            SKUState.APPROVAL_REQUIRED,
            SKUState.BLOCKED_PENDING_REVIEW,
            SKUState.BLOCKED_AWAITING_DATA,
            SKUState.REJECTED,
        }
    ),
    SKUState.APPROVAL_REQUIRED: frozenset({SKUState.APPROVED, SKUState.REJECTED}),
    SKUState.APPROVED: frozenset({SKUState.LISTING_READY, SKUState.REJECTED}),
    # Part 1 ends at LISTING_READY. There is deliberately no onward transition:
    # publication belongs to Part 2 and cannot be represented here.
    SKUState.LISTING_READY: frozenset(),
    SKUState.REJECTED: frozenset(),
}


class IllegalTransition(ValueError):
    """Raised when a transition is not permitted by the state machine."""


def is_legal(from_state: SKUState, to_state: SKUState) -> bool:
    return to_state in LEGAL_TRANSITIONS.get(from_state, frozenset())


def require_legal(from_state: SKUState, to_state: SKUState) -> None:
    if not is_legal(from_state, to_state):
        allowed = sorted(str(s) for s in LEGAL_TRANSITIONS.get(from_state, frozenset()))
        raise IllegalTransition(
            f"{from_state} -> {to_state} is not a legal transition; "
            f"allowed: {allowed or 'none (terminal)'}"
        )


def path_to(target: SKUState, from_state: SKUState) -> list[SKUState] | None:
    """Shortest legal path between two states, or None if unreachable.

    Used by the pipeline to advance a SKU through intermediate states it has
    not visited yet, without any stage being able to skip a gate.
    """
    queue: deque[tuple[SKUState, list[SKUState]]] = deque([(from_state, [])])
    seen = {from_state}
    while queue:
        current, path = queue.popleft()
        for nxt in sorted(LEGAL_TRANSITIONS.get(current, frozenset()), key=str):
            if nxt in seen:
                continue
            if nxt is target:
                return [*path, nxt]
            seen.add(nxt)
            queue.append((nxt, [*path, nxt]))
    return None


__all__ = [
    "FINDING_DISPOSITIONS",
    "LEGAL_TRANSITIONS",
    "RULES_VERSION",
    "Decision",
    "Disposition",
    "IllegalTransition",
    "decide_compliance",
    "decide_coverage",
    "decide_headroom",
    "decide_profitability",
    "decide_reference_trust",
    "interpret_findings",
    "is_legal",
    "path_to",
    "require_legal",
]
