"""Supplier ingestion: every supplier fact comes from the file or is refused.

The layer this replaces defaulted five fields when a catalogue did not supply
them — GST rate to 18%, GST-inclusivity to true, MOQ to 1, dispatch to 7 days,
and, most consequentially, ``tax_invoice_available`` to ``True``. That last one
is not a cosmetic default: it decides whether input tax credit is recoverable,
which is roughly ₹37 per unit on a ₹240 item and is the single largest driver
of whether a SKU is viable at all. Every supplier in the system was assumed to
issue a tax invoice because no column ever said so.

So this module refuses to guess. A row that does not state a required fact is
rejected with a reason naming the field, and the operator adds it. Rejected
rows are reported rather than silently skipped, because a catalogue that
quietly loses half its lines looks identical to a small catalogue.

Deduplication is deterministic and supplier identity is stable across runs, so
re-importing the same file twice changes nothing and the same supplier is the
same supplier.

No LLM touches anything here. Supplier facts have exactly one origin: the file
the operator ingested.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from catalog.schemas import SupplierOffer
from econ.money import money
from sourcing.matching import SupplierCandidate
from sourcing.model import extract_model_code, normalize_model_code, normalize_title

logger = logging.getLogger(__name__)

SUPPLIER_INGEST_VERSION = "supplier-ingest/1.0.0"

#: Namespace for deriving stable supplier identifiers. A supplier named the
#: same way in two files is the same supplier, in this run and the next.
_SUPPLIER_NAMESPACE = uuid.UUID("6f1b6d84-3f8a-4c2e-9c1f-2b7a5d0e4a11")

#: The only currency the economics engine models. Anything else is refused
#: rather than converted: this project has no FX rates, and inventing one
#: would put a fabricated number underneath every downstream calculation.
SUPPORTED_CURRENCY = "INR"

_TRUE = {"y", "yes", "true", "1", "t", "available", "issued"}
_FALSE = {"n", "no", "false", "0", "f", "unavailable", "none", "not issued"}


class FieldClass(StrEnum):
    """How each supplier field is treated when the catalogue is silent."""

    #: The row is refused. These change landed cost, so a default would be a
    #: fabricated business fact.
    SOURCE_REQUIRED = "SOURCE_REQUIRED"
    #: Recorded as unknown and carried through as unknown. Never substituted.
    OPTIONAL_UNAVAILABLE = "OPTIONAL_UNAVAILABLE"
    #: Computed from other sourced fields. Never presented or persisted as a
    #: supplier-stated fact.
    DERIVED = "DERIVED"


#: The explicit contract for every supplier field. Anything not listed here is
#: not read from a catalogue at all.
FIELD_CLASSIFICATION: dict[str, FieldClass] = {
    "supplier_name": FieldClass.SOURCE_REQUIRED,
    "product_name": FieldClass.SOURCE_REQUIRED,
    "unit_cost_inr": FieldClass.SOURCE_REQUIRED,
    "currency": FieldClass.SOURCE_REQUIRED,
    "gst_rate_pct": FieldClass.SOURCE_REQUIRED,
    "tax_invoice": FieldClass.SOURCE_REQUIRED,
    "moq_units": FieldClass.SOURCE_REQUIRED,
    # Required only when a tax invoice is available: without an invoice the
    # input tax credit is zero regardless, so inclusivity cannot change any
    # number. With one, it decides how much tax is recoverable.
    "cost_is_gst_inclusive": FieldClass.SOURCE_REQUIRED,
    "dispatch_days": FieldClass.OPTIONAL_UNAVAILABLE,
    "hsn_code": FieldClass.OPTIONAL_UNAVAILABLE,
    "weight_g": FieldClass.OPTIONAL_UNAVAILABLE,
    "available_qty": FieldClass.OPTIONAL_UNAVAILABLE,
    # Required because landed cost cannot be computed without it. A stated
    # zero is fine and common (ex-works, freight absorbed); an unstated one is
    # not the same claim.
    "inbound_freight_inr": FieldClass.SOURCE_REQUIRED,
    "ean": FieldClass.OPTIONAL_UNAVAILABLE,
    "supplier_product_id": FieldClass.OPTIONAL_UNAVAILABLE,
    "observed_at": FieldClass.OPTIONAL_UNAVAILABLE,
    # Derived from the supplier name, never read from the file.
    "supplier_id": FieldClass.DERIVED,
    "product_id": FieldClass.DERIVED,
    "model_code": FieldClass.DERIVED,
}


class RejectionReason(StrEnum):
    """Why a row could not become a usable supplier offer."""

    MISSING_SUPPLIER = "MISSING_SUPPLIER"
    MISSING_PRODUCT = "MISSING_PRODUCT"
    MISSING_COST = "MISSING_COST"
    UNREADABLE_COST = "UNREADABLE_COST"
    MISSING_GST = "MISSING_GST"
    MISSING_TAX_INVOICE = "MISSING_TAX_INVOICE"
    MISSING_MOQ = "MISSING_MOQ"
    MISSING_GST_INCLUSIVITY = "MISSING_GST_INCLUSIVITY"
    MISSING_FREIGHT = "MISSING_FREIGHT"
    UNSUPPORTED_CURRENCY = "UNSUPPORTED_CURRENCY"
    IMPLAUSIBLE_VALUE = "IMPLAUSIBLE_VALUE"


@dataclass(frozen=True)
class RejectedRow:
    """A row that was refused, and exactly why.

    Surfaced rather than dropped. A silent skip makes a truncated import look
    like a small catalogue, and the operator never learns which column to fix.
    """

    row_number: int
    reason: RejectionReason
    detail: str
    supplier_name: str | None = None
    product_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "reason": str(self.reason),
            "detail": self.detail,
            "supplier_name": self.supplier_name,
            "product_name": self.product_name,
        }


@dataclass(frozen=True)
class SupplierProvenance:
    """Where one accepted supplier record came from."""

    source_reference: str
    row_number: int
    observed_at: datetime
    ingested_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_reference": self.source_reference,
            "row_number": self.row_number,
            "observed_at": self.observed_at.isoformat(),
            "ingested_at": self.ingested_at.isoformat(),
        }


@dataclass
class SupplierIngestReport:
    """Everything one import produced, including what it refused."""

    source: str
    accepted: list[SupplierCandidate] = field(default_factory=list)
    provenance: dict[str, SupplierProvenance] = field(default_factory=dict)
    rejected: list[RejectedRow] = field(default_factory=list)
    duplicates_ignored: int = 0
    warnings: list[str] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "accepted": self.accepted_count,
            "rejected": len(self.rejected),
            "duplicates_ignored": self.duplicates_ignored,
            "ingested_at": self.ingested_at.isoformat(),
            "suppliers": [
                {
                    "supplier_name": c.supplier_name,
                    "title": c.title,
                    "unit_cost_inr": str(c.offer.unit_cost_inr),
                    "moq_units": c.offer.moq_units,
                    "gst_rate_pct": str(c.offer.gst_rate_pct),
                    "tax_invoice_available": c.offer.tax_invoice_available,
                    "lead_time_days": c.lead_time_days,
                    "provenance": self.provenance[str(c.offer.id)].as_dict()
                    if str(c.offer.id) in self.provenance
                    else None,
                }
                for c in self.accepted
            ],
            "rejected_rows": [r.as_dict() for r in self.rejected],
            "warnings": list(self.warnings),
        }


def supplier_id_for(name: str) -> uuid.UUID:
    """Stable identifier for a supplier name.

    Derived rather than random so the same supplier is the same entity across
    rows, files and runs. The previous loader minted a fresh UUID per row, so
    one supplier appearing on ten lines became ten suppliers — which quietly
    defeated supplier-concentration risk scoring.
    """
    return uuid.uuid5(_SUPPLIER_NAMESPACE, normalize_title(name))


def parse_bool(raw: Any) -> bool | None:
    """Parse a yes/no field, returning None when it does not say.

    None is the important case: it means the catalogue is silent, and silence
    about a tax invoice must not be read as 'yes'.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def detect_currency(headers: list[str], row: dict[str, Any]) -> str | None:
    """Determine the currency of the cost column.

    Read from an explicit ``currency`` column when present, otherwise from a
    currency named in the rate column's header (``Rate (INR)``). If neither
    states it, the answer is None and the row is refused — an unlabelled
    number is not a price.
    """
    explicit = next(
        (v for k, v in row.items() if k and "currency" in k.strip().lower()), None
    )
    if explicit and str(explicit).strip():
        return str(explicit).strip().upper()

    for header in headers:
        match = re.search(r"\b(INR|USD|EUR|GBP|CNY|AED)\b", header or "", re.IGNORECASE)
        if match and re.search(r"rate|cost|price", header, re.IGNORECASE):
            return match.group(1).upper()
    return None


def _positive(value: Decimal | int | None) -> bool:
    return value is not None and value > 0


def build_candidate(
    row: Any,
    *,
    row_number: int,
    source: str,
    currency: str | None,
    ingested_at: datetime,
) -> tuple[SupplierCandidate | None, RejectedRow | None, SupplierProvenance | None]:
    """Turn one ingested row into a candidate, or explain why it cannot be.

    Every required fact must be present in the row. Nothing is defaulted:
    the fields this refuses to guess are precisely the ones that change landed
    cost, and a plausible default is indistinguishable from a measurement once
    it reaches the economics engine.
    """
    supplier = row.get("supplier_name")
    product = row.get("product_name")
    cost = row.get("unit_cost_inr")

    def reject(reason: RejectionReason, detail: str) -> tuple[None, RejectedRow, None]:
        return (
            None,
            RejectedRow(
                row_number=row_number,
                reason=reason,
                detail=detail,
                supplier_name=str(supplier) if supplier else None,
                product_name=str(product) if product else None,
            ),
            None,
        )

    if not supplier:
        return reject(RejectionReason.MISSING_SUPPLIER, "no supplier name")
    if not product:
        return reject(RejectionReason.MISSING_PRODUCT, "no product title")

    if cost is None:
        # Distinguish "absent" from "present but unreadable": the ingester
        # already flags the latter, and the operator fixes them differently.
        status = row.status("unit_cost_inr")
        if str(status) == "REQUIRES_REVIEW":
            return reject(
                RejectionReason.UNREADABLE_COST,
                f"unit cost could not be parsed: {row.fields['unit_cost_inr'].raw!r}",
            )
        return reject(RejectionReason.MISSING_COST, "no unit cost")

    if currency is None:
        return reject(
            RejectionReason.UNSUPPORTED_CURRENCY,
            "the catalogue does not state a currency; add a currency column or "
            "name it in the rate header, e.g. 'Rate (INR)'",
        )
    if currency != SUPPORTED_CURRENCY:
        return reject(
            RejectionReason.UNSUPPORTED_CURRENCY,
            f"cost is in {currency}; only {SUPPORTED_CURRENCY} is supported and "
            "no exchange rate is configured",
        )

    gst = row.get("gst_rate_pct")
    if gst is None:
        return reject(
            RejectionReason.MISSING_GST,
            "no GST rate. It is not defaulted: the rate determines recoverable "
            "input tax credit and therefore landed cost",
        )

    tax_invoice = parse_bool(row.get("tax_invoice"))
    if tax_invoice is None:
        return reject(
            RejectionReason.MISSING_TAX_INVOICE,
            "the catalogue does not say whether a tax invoice is issued. "
            "Without one there is no input tax credit, which is frequently the "
            "difference between a viable SKU and a dead one, so it is never "
            "assumed",
        )

    moq = row.get("moq_units")
    if moq is None:
        return reject(
            RejectionReason.MISSING_MOQ,
            "no minimum order quantity. MOQ drives carrying cost, so a default "
            "of 1 would understate locked capital",
        )

    if not _positive(cost) or not _positive(moq):
        return reject(
            RejectionReason.IMPLAUSIBLE_VALUE,
            f"cost {cost} and MOQ {moq} must both be positive",
        )

    if row.get("inbound_freight_inr") is None:
        return reject(
            RejectionReason.MISSING_FREIGHT,
            "inbound freight is not stated. Landed cost cannot be computed "
            "without it, and defaulting to zero understates every downstream "
            "margin. State 0 explicitly if freight is genuinely not charged",
        )

    inclusive = parse_bool(row.get("cost_is_gst_inclusive"))
    if inclusive is None and tax_invoice:
        # Previously defaulted to True on the grounds that Indian wholesale
        # quotes are ordinarily GST-inclusive. A convention is not evidence:
        # on an inclusive quote the tax is a fraction of the stated price, on
        # an exclusive one it is added on top, and picking wrong misstates the
        # recoverable credit. Left unknown when there is no tax invoice,
        # because the credit is zero either way and the field cannot matter.
        return reject(
            RejectionReason.MISSING_GST_INCLUSIVITY,
            "the catalogue does not state whether the quoted cost is "
            "GST-inclusive. With a tax invoice this decides how much input tax "
            "credit is recoverable, so it is required rather than assumed",
        )

    observed_raw = row.get("observed_at")
    observed_at = ingested_at
    if observed_raw:
        try:
            parsed = datetime.fromisoformat(str(observed_raw))
            observed_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            observed_at = ingested_at

    offer = SupplierOffer(
        supplier_id=supplier_id_for(str(supplier)),
        product_id=uuid.uuid5(_SUPPLIER_NAMESPACE, normalize_title(str(product))),
        unit_cost_inr=money(cost),
        cost_is_gst_inclusive=inclusive,
        gst_rate_pct=money(gst),
        tax_invoice_available=tax_invoice,
        moq_units=int(moq),
        inbound_freight_inr=money(row.get("inbound_freight_inr"))
        if row.get("inbound_freight_inr") is not None
        else None,
        dispatch_days=int(row.get("dispatch_days")) if row.get("dispatch_days") else None,
        available_qty=int(row.get("available_qty")) if row.get("available_qty") else None,
        source_document_ref=f"{source}#row={row_number}",
        observed_at=observed_at,
    )

    candidate = SupplierCandidate(
        supplier_name=str(supplier),
        title=str(product),
        offer=offer,
        gtin=row.get("ean"),
        model_code=normalize_model_code(row.get("model_number"))
        or extract_model_code(str(product)),
        lead_time_days=int(row.get("dispatch_days"))
        if row.get("dispatch_days") is not None
        else None,
    )
    provenance = SupplierProvenance(
        source_reference=f"{source}#row={row_number}",
        row_number=row_number,
        observed_at=observed_at,
        ingested_at=ingested_at,
    )
    return candidate, None, provenance


def dedupe_key(candidate: SupplierCandidate) -> tuple[str, str]:
    """Deterministic identity for deduplication.

    Supplier plus product identity, preferring a GTIN or model code over the
    title. Re-importing the same catalogue must change nothing.
    """
    identity = candidate.gtin or candidate.model_code or normalize_title(candidate.title)
    return (normalize_title(candidate.supplier_name), identity)


def ingest_suppliers(path: Path | str) -> SupplierIngestReport:
    """Import a supplier catalogue, refusing rows that do not state their facts.

    Reuses the existing ``sources.ingest`` reader so column aliasing, currency
    symbol handling and the refusal to parse an unreadable price all behave
    exactly as they do elsewhere. What this adds is the supplier-specific
    contract: which facts are mandatory, and what happens when they are absent.
    """
    from sources.ingest import ingest  # noqa: PLC0415

    path = Path(path)
    result = ingest(path)
    ingested_at = datetime.now(UTC)
    report = SupplierIngestReport(source=str(path), ingested_at=ingested_at)
    report.warnings.extend(result.warnings)

    headers: list[str] = []
    if path.suffix.lower() == ".csv" and path.is_file():
        first = path.read_text(encoding="utf-8-sig").splitlines()
        headers = first[0].split(",") if first else []

    seen: set[tuple[str, str]] = set()
    unstated_dispatch = 0

    for row in result.rows:
        currency = detect_currency(headers, {f: row.get(f) for f in ("currency",)})
        candidate, rejection, provenance = build_candidate(
            row,
            row_number=row.row_number,
            source=str(path),
            currency=currency,
            ingested_at=ingested_at,
        )
        if rejection is not None:
            report.rejected.append(rejection)
            continue

        assert candidate is not None and provenance is not None
        key = dedupe_key(candidate)
        if key in seen:
            report.duplicates_ignored += 1
            continue
        seen.add(key)

        if candidate.lead_time_days is None:
            unstated_dispatch += 1

        report.accepted.append(candidate)
        report.provenance[str(candidate.offer.id)] = provenance

    if unstated_dispatch:
        report.warnings.append(
            f"{unstated_dispatch} row(s) did not state a dispatch time; it is "
            "carried as unknown and no lead time is assumed."
        )
    if report.rejected:
        report.warnings.append(
            f"{len(report.rejected)} row(s) were refused because they did not "
            "state a required fact; see rejected_rows"
        )

    logger.info(
        "supplier catalogue ingested",
        extra={
            "operation": "ingest_suppliers",
            "result": f"{report.accepted_count} accepted, {len(report.rejected)} rejected",
        },
    )
    return report


__all__ = [
    "FIELD_CLASSIFICATION",
    "SUPPLIER_INGEST_VERSION",
    "SUPPORTED_CURRENCY",
    "FieldClass",
    "RejectedRow",
    "RejectionReason",
    "SupplierIngestReport",
    "SupplierProvenance",
    "build_candidate",
    "dedupe_key",
    "detect_currency",
    "ingest_suppliers",
    "parse_bool",
    "supplier_id_for",
]
