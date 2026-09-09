"""Listing publication: prerequisites first, then a gated external write.

Part 1 ends at ``LISTING_READY`` plus an ``Approval``. This is the only path
from there to a marketplace, and it refuses to take it unless every
precondition still holds.

The checks run *at publish time*, not at approval time. An approval granted on
Tuesday says nothing about whether the supplier raised prices on Wednesday, so
economics are re-validated against current inputs before anything is sent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from catalog.enums import ApprovalDecision, ComplianceResult, SKUState
from econ.money import format_inr, money
from part2.execution import (
    ActionResult,
    ExecutionRecord,
    ExecutionState,
    IdempotencyStore,
    Outcome,
    run_approved,
)
from part2.marketplace import BaseAdapter, ListingPayload, RemoteListing, detect_drift

logger = logging.getLogger(__name__)

PUBLISHER_VERSION = "publisher/1.0.0"


@dataclass(frozen=True)
class Prerequisite:
    """One publication precondition and whether it holds."""

    name: str
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class PublishCheck:
    """The full precondition assessment for one publication."""

    prerequisites: list[Prerequisite]

    @property
    def satisfied(self) -> bool:
        return all(p.satisfied for p in self.prerequisites)

    @property
    def failures(self) -> list[Prerequisite]:
        return [p for p in self.prerequisites if not p.satisfied]

    def as_dict(self) -> dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "prerequisites": [
                {"name": p.name, "satisfied": p.satisfied, "detail": p.detail}
                for p in self.prerequisites
            ],
        }


def check_prerequisites(
    *,
    sku_state: SKUState,
    approval_decision: ApprovalDecision | None,
    compliance: ComplianceResult | None,
    payload: ListingPayload,
    current_min_profitable_inr: Decimal | None,
    required_attributes: tuple[str, ...] = ("brand", "country_of_origin"),
    require_images: bool = True,
) -> PublishCheck:
    """Assess whether this listing may be published right now.

    Every check is evaluated -- none short-circuits -- so a caller sees all
    the reasons at once rather than fixing them one round-trip at a time.
    """
    checks: list[Prerequisite] = [
        Prerequisite(
            "sku_is_listing_ready",
            sku_state is SKUState.LISTING_READY,
            f"SKU state is {sku_state}; Part 1 must have reached LISTING_READY",
        ),
        Prerequisite(
            "human_approval_exists",
            approval_decision is ApprovalDecision.APPROVED,
            f"approval decision is {approval_decision}; publication requires APPROVED",
        ),
        Prerequisite(
            "compliance_passed",
            compliance is ComplianceResult.PASS,
            f"compliance is {compliance}; UNKNOWN and REQUIRES_HUMAN_REVIEW do not permit listing",
        ),
        Prerequisite(
            "title_present",
            bool(payload.title.strip()),
            "listing has no title",
        ),
        Prerequisite(
            "description_present",
            bool(payload.description.strip()),
            "listing has no description",
        ),
        Prerequisite(
            "price_positive",
            money(payload.price_inr) > 0,
            f"price is {format_inr(payload.price_inr)}",
        ),
    ]

    missing_attrs = [a for a in required_attributes if not payload.attributes.get(a)]
    checks.append(
        Prerequisite(
            "required_attributes_present",
            not missing_attrs,
            f"missing required attributes: {missing_attrs}"
            if missing_attrs
            else "all present",
        )
    )

    checks.append(
        Prerequisite(
            "images_present",
            bool(payload.images) or not require_images,
            "no images supplied; marketplaces suppress listings without them",
        )
    )

    if current_min_profitable_inr is None:
        checks.append(
            Prerequisite(
                "economics_still_valid",
                False,
                "current economics unavailable: cannot confirm the price is still viable",
            )
        )
    else:
        viable = money(payload.price_inr) >= money(current_min_profitable_inr)
        checks.append(
            Prerequisite(
                "economics_still_valid",
                viable,
                (
                    f"price {format_inr(payload.price_inr)} is below the current minimum "
                    f"viable price {format_inr(current_min_profitable_inr)}"
                    if not viable
                    else "price still clears the current minimum viable price"
                ),
            )
        )

    return PublishCheck(prerequisites=checks)


class ListingPublisher:
    """Publishes approved listings, and refuses everything else."""

    def __init__(
        self, adapter: BaseAdapter, *, store: IdempotencyStore | None = None
    ) -> None:
        self.adapter = adapter
        self.store = store or IdempotencyStore()
        self.published: dict[str, str] = {}

    def publish(
        self,
        payload: ListingPayload,
        check: PublishCheck,
        *,
        approver: str,
        dry_run: bool = True,
    ) -> tuple[ExecutionRecord, ActionResult]:
        """Publish one listing through the execution gate.

        Returns the execution record alongside the result so the caller has
        the full audit trail, including the reason when nothing was sent.
        """
        record = ExecutionRecord(
            action="create_listing",
            sku_code=payload.sku_code,
            marketplace=self.adapter.profile.name,
            proposed={
                "price_inr": money(payload.price_inr),
                "quantity": payload.quantity,
            },
            reason="approved listing publication",
        )

        if not check.satisfied:
            reasons = "; ".join(f"{p.name}: {p.detail}" for p in check.failures)
            record.transition(
                ExecutionState.REQUIRES_REVIEW, note="prerequisites not met"
            )
            result = ActionResult(
                outcome=Outcome.PERMANENT_FAILURE,
                action="create_listing",
                idempotency_key=payload.fingerprint(),
                detail=f"not published: {reasons}",
            )
            record.result = result
            logger.warning(
                "publication refused",
                extra={
                    "sku": payload.sku_code,
                    "marketplace": self.adapter.profile.name,
                    "result": "PREREQUISITES_FAILED",
                },
            )
            return record, result

        record.approve(approver)
        result = run_approved(
            record,
            lambda: self.adapter.create_listing(payload, dry_run=dry_run),
            store=self.store,
            idempotent=True,
        )
        if result.outcome is Outcome.OK:
            self.published[payload.sku_code] = payload.fingerprint()
        return record, result

    def verify(self, payload: ListingPayload, remote: RemoteListing) -> list[str]:
        """Confirm the marketplace actually reflects what we sent.

        A 200 response is not evidence of state change. Listings get
        suppressed, prices get overridden by marketplace promotions, and
        quantities get clamped -- all with a successful write.
        """
        drift = detect_drift(payload, remote)
        if drift:
            logger.warning(
                "listing drift detected",
                extra={"sku": payload.sku_code, "marketplace": self.adapter.profile.name},
            )
        return drift


__all__ = [
    "PUBLISHER_VERSION",
    "ListingPublisher",
    "Prerequisite",
    "PublishCheck",
    "check_prerequisites",
]
