"""Supplier ingestion: messy files in, structured records with provenance out.

Three formats, one contract. Every field that reaches the rest of the system
carries a ``FieldStatus`` saying how it got there:

``KNOWN``            read directly from the source
``INFERRED``         derived deterministically from other columns
``UNKNOWN``          absent, and recorded as absent
``REQUIRES_REVIEW``  present but unparseable or implausible

Nothing is silently invented. A missing MOQ becomes ``UNKNOWN``, not 1; an
unparseable price becomes ``REQUIRES_REVIEW``, not 0. The alternative — filling
gaps with plausible defaults — produces a catalogue that looks complete and
costs money to discover otherwise.

PDF is deliberately the weakest path. Tabular PDFs vary too much for reliable
deterministic parsing, so the extractor produces candidate rows flagged
``REQUIRES_REVIEW`` and defers to the LLM extractor or a human.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INGEST_VERSION = "ingest/1.0.0"


class FieldStatus(StrEnum):
    KNOWN = "KNOWN"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class IngestError(RuntimeError):
    """Raised when a source file cannot be read at all."""


@dataclass(frozen=True)
class Field_:
    """A single extracted value and how it was obtained."""

    value: Any
    status: FieldStatus
    raw: str | None = None
    note: str | None = None

    @property
    def usable(self) -> bool:
        return self.status in (FieldStatus.KNOWN, FieldStatus.INFERRED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else str(self.value),
            "status": str(self.status),
            "raw": self.raw,
            "note": self.note,
        }


@dataclass
class ExtractedRow:
    """One candidate supplier offer, with per-field provenance."""

    source_ref: str
    row_number: int
    fields: dict[str, Field_] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get(self, name: str) -> Any:
        f = self.fields.get(name)
        return f.value if f and f.usable else None

    def status(self, name: str) -> FieldStatus:
        f = self.fields.get(name)
        return f.status if f else FieldStatus.UNKNOWN

    @property
    def needs_review(self) -> bool:
        return bool(self.errors) or any(
            f.status is FieldStatus.REQUIRES_REVIEW for f in self.fields.values()
        )

    @property
    def is_usable(self) -> bool:
        """Whether the row has the minimum needed to become an offer."""
        required = ("product_name", "unit_cost_inr", "supplier_name")
        return all(self.get(name) is not None for name in required) and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "row_number": self.row_number,
            "usable": self.is_usable,
            "needs_review": self.needs_review,
            "fields": {k: v.as_dict() for k, v in self.fields.items()},
            "errors": list(self.errors),
        }


@dataclass
class IngestResult:
    """Everything one source file yielded, including what it failed to yield."""

    source_path: str
    format: str
    rows: list[ExtractedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ingest_version: str = INGEST_VERSION

    @property
    def usable_rows(self) -> list[ExtractedRow]:
        return [r for r in self.rows if r.is_usable]

    @property
    def review_rows(self) -> list[ExtractedRow]:
        return [r for r in self.rows if r.needs_review or not r.is_usable]

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "format": self.format,
            "rows": len(self.rows),
            "usable": len(self.usable_rows),
            "needs_review": len(self.review_rows),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Column mapping and parsing
# --------------------------------------------------------------------------

#: Header aliases seen in real supplier files. Matching is case- and
#: punctuation-insensitive so "Unit Cost (INR)" and "unit_cost_inr" agree.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "supplier_name": ("supplier", "suppliername", "vendor", "vendorname", "seller"),
    "product_name": (
        "product",
        "productname",
        "item",
        "itemname",
        "description",
        "title",
    ),
    "sku_code": ("sku", "skucode", "itemcode", "articlecode", "stylecode"),
    "brand": ("brand", "make"),
    "manufacturer": ("manufacturer", "mfr", "mfg"),
    "model_number": ("model", "modelno", "modelnumber", "partno", "partnumber"),
    "hsn_code": ("hsn", "hsncode", "hsnsac"),
    "unit_cost_inr": (
        "price",
        "rate",
        "cost",
        "unitcost",
        "unitprice",
        "costinr",
        "priceinr",
    ),
    "gst_rate_pct": ("gst", "gstrate", "gstpct", "taxrate", "tax"),
    "moq_units": ("moq", "minorder", "minorderqty", "minimumorderquantity"),
    "available_qty": ("qty", "quantity", "stock", "available", "availableqty"),
    "weight_g": ("weight", "weightg", "weightgrams", "grossweight"),
    "inbound_freight_inr": ("freight", "shipping", "inboundfreight", "transport"),
    "dispatch_days": ("dispatch", "leadtime", "dispatchdays", "leaddays"),
    "ean": ("ean", "barcode", "upc", "gtin"),
    # Supplier-catalogue facts that must be stated, never assumed.
    "tax_invoice": ("taxinvoice", "gstinvoice", "invoice", "taxinvoiceavailable"),
    "cost_is_gst_inclusive": ("gstinclusive", "inclusiveofgst", "costgstinclusive"),
    "currency": ("currency", "ccy"),
    "observed_at": ("observedat", "quotedate", "pricedate", "asof"),
    "supplier_product_id": ("supplierproductid", "supplierpid", "vendorsku"),
}


def _canonical(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.lower())


def map_columns(headers: list[str]) -> dict[str, str]:
    """Map source headers onto canonical field names.

    Two passes. Exact matches win outright; only then are headers matched by
    containment, longest alias first. Real supplier files write "Item
    Description" and "Rate (INR)" rather than "product_name" and
    "unit_cost_inr", and refusing to recognise those would push every file to
    the review queue for no benefit. Exact-first ordering stops a short alias
    such as "tax" claiming a header a longer alias describes better.
    """
    mapping: dict[str, str] = {}
    claimed: set[str] = set()

    for header in headers:
        key = _canonical(header)
        for canonical_name, aliases in COLUMN_ALIASES.items():
            if key == _canonical(canonical_name) or key in aliases:
                mapping[header] = canonical_name
                claimed.add(canonical_name)
                break

    candidates = sorted(
        (
            (len(alias), alias, canonical_name)
            for canonical_name, aliases in COLUMN_ALIASES.items()
            for alias in (*aliases, _canonical(canonical_name))
        ),
        reverse=True,
    )
    for header in headers:
        if header in mapping:
            continue
        key = _canonical(header)
        for _, alias, canonical_name in candidates:
            if canonical_name in claimed:
                continue
            if alias in key:
                mapping[header] = canonical_name
                claimed.add(canonical_name)
                break
    return mapping


_NUMERIC = re.compile(r"-?\d+(?:[.,]\d+)?")


def parse_decimal(raw: Any, *, field_name: str) -> Field_:
    """Parse a money or rate value, refusing to guess.

    Currency symbols, thousands separators and stray words are stripped, but an
    unparseable value becomes ``REQUIRES_REVIEW`` rather than zero. A zero
    price would flow straight into the economics engine and produce a
    spectacular fake margin.
    """
    if raw is None or str(raw).strip() == "":
        return Field_(None, FieldStatus.UNKNOWN, None, f"{field_name} absent")
    text = str(raw).strip()
    cleaned = (
        text.replace(",", "").replace("\u20b9", "").replace("Rs.", "").replace("Rs", "")
    )
    match = _NUMERIC.search(cleaned)
    if not match:
        return Field_(None, FieldStatus.REQUIRES_REVIEW, text, "no numeric value found")
    try:
        value = Decimal(match.group().replace(",", ""))
    except InvalidOperation:
        return Field_(None, FieldStatus.REQUIRES_REVIEW, text, "not a valid decimal")
    if value < 0:
        return Field_(None, FieldStatus.REQUIRES_REVIEW, text, "negative value")
    # Compare against the ORIGINAL text, not the cleaned one: if a currency
    # symbol or separator had to be removed, the value was derived rather than
    # read, and the provenance must say so.
    status = FieldStatus.KNOWN if match.group() == text.strip() else FieldStatus.INFERRED
    note = None if status is FieldStatus.KNOWN else f"extracted {value} from {text!r}"
    return Field_(value, status, text, note)


def parse_int(raw: Any, *, field_name: str) -> Field_:
    parsed = parse_decimal(raw, field_name=field_name)
    if not parsed.usable:
        return parsed
    return Field_(int(parsed.value), parsed.status, parsed.raw, parsed.note)


def parse_text(raw: Any, *, field_name: str) -> Field_:
    if raw is None or str(raw).strip() == "":
        return Field_(None, FieldStatus.UNKNOWN, None, f"{field_name} absent")
    return Field_(str(raw).strip(), FieldStatus.KNOWN, str(raw))


_PARSERS = {
    "unit_cost_inr": parse_decimal,
    "gst_rate_pct": parse_decimal,
    "inbound_freight_inr": parse_decimal,
    "moq_units": parse_int,
    "available_qty": parse_int,
    "weight_g": parse_int,
    "dispatch_days": parse_int,
}


def build_row(
    raw_row: dict[str, Any], mapping: dict[str, str], *, source_ref: str, row_number: int
) -> ExtractedRow:
    """Turn one raw record into a provenance-carrying ExtractedRow."""
    row = ExtractedRow(source_ref=source_ref, row_number=row_number)
    for header, value in raw_row.items():
        canonical_name = mapping.get(header)
        if canonical_name is None:
            continue
        parser = _PARSERS.get(canonical_name, parse_text)
        row.fields[canonical_name] = parser(value, field_name=canonical_name)

    for required in ("supplier_name", "product_name", "unit_cost_inr"):
        if required not in row.fields:
            row.fields[required] = Field_(
                None, FieldStatus.UNKNOWN, None, "column not present in source"
            )
            row.errors.append(f"required column {required!r} missing")
        elif row.fields[required].status is FieldStatus.REQUIRES_REVIEW:
            row.errors.append(
                f"{required} could not be parsed: {row.fields[required].raw!r}"
            )
    return row


# --------------------------------------------------------------------------
# Format adapters
# --------------------------------------------------------------------------


def ingest_csv(path: Path | str) -> IngestResult:
    path = Path(path)
    result = IngestResult(source_path=str(path), format="csv")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestError(f"cannot read {path}: {exc}") from exc

    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        result.warnings.append("file has no header row")
        return result

    mapping = map_columns(list(reader.fieldnames))
    unmapped = [h for h in reader.fieldnames if h not in mapping]
    if unmapped:
        result.warnings.append(f"unmapped columns ignored: {unmapped}")

    for number, raw in enumerate(reader, start=2):
        result.rows.append(
            build_row(raw, mapping, source_ref=str(path), row_number=number)
        )
    return result


def ingest_xlsx(path: Path | str, *, sheet: str | None = None) -> IngestResult:
    path = Path(path)
    result = IngestResult(source_path=str(path), format="xlsx")
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - dependency declared
        raise IngestError("openpyxl is required to read .xlsx files") from exc

    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise IngestError(f"cannot open workbook {path}: {exc}") from exc

    worksheet = workbook[sheet] if sheet else workbook.active
    rows = worksheet.iter_rows(values_only=True)
    try:
        headers = [str(h) if h is not None else "" for h in next(rows)]
    except StopIteration:
        result.warnings.append("worksheet is empty")
        return result

    mapping = map_columns(headers)
    for number, values in enumerate(rows, start=2):
        if all(v is None for v in values):
            continue
        raw = dict(zip(headers, values, strict=False))
        result.rows.append(
            build_row(raw, mapping, source_ref=str(path), row_number=number)
        )
    workbook.close()
    return result


_PDF_ROW = re.compile(
    r"^(?P<product>.{4,80}?)\s{2,}(?P<cost>[\u20b9Rs.\s]*[\d,]+(?:\.\d+)?)\s*$"
)


def ingest_pdf(path: Path | str, *, supplier_name: str | None = None) -> IngestResult:
    """Extract candidate rows from a PDF price list.

    Every row is flagged ``REQUIRES_REVIEW``. Tabular PDFs vary far too much
    for a regex to be trusted with purchasing decisions, so this produces
    candidates for the LLM extractor or a human to confirm — never finished
    records.
    """
    path = Path(path)
    result = IngestResult(source_path=str(path), format="pdf")
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency declared
        raise IngestError("pypdf is required to read .pdf files") from exc

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise IngestError(f"cannot read PDF {path}: {exc}") from exc

    result.warnings.append(
        "PDF rows are candidates only and are flagged REQUIRES_REVIEW; confirm "
        "them with the LLM extractor or by hand before use"
    )

    number = 1
    for page_index, text in enumerate(pages, start=1):
        for line in text.splitlines():
            match = _PDF_ROW.match(line.strip())
            if not match:
                continue
            number += 1
            row = ExtractedRow(source_ref=f"{path}#page={page_index}", row_number=number)
            cost = parse_decimal(match.group("cost"), field_name="unit_cost_inr")
            row.fields["product_name"] = Field_(
                match.group("product").strip(),
                FieldStatus.REQUIRES_REVIEW,
                line.strip(),
                "extracted from unstructured PDF text",
            )
            row.fields["unit_cost_inr"] = Field_(
                cost.value,
                FieldStatus.REQUIRES_REVIEW if cost.usable else cost.status,
                cost.raw,
                "extracted from unstructured PDF text",
            )
            row.fields["supplier_name"] = (
                Field_(supplier_name, FieldStatus.INFERRED, None, "supplied by caller")
                if supplier_name
                else Field_(None, FieldStatus.UNKNOWN, None, "not present in PDF")
            )
            result.rows.append(row)

    if not result.rows:
        result.warnings.append("no candidate rows matched the expected layout")
    return result


def ingest(path: Path | str, **kwargs: Any) -> IngestResult:
    """Dispatch on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ingest_csv(path)
    if suffix in (".xlsx", ".xlsm"):
        return ingest_xlsx(path, **kwargs)
    if suffix == ".pdf":
        return ingest_pdf(path, **kwargs)
    raise IngestError(f"unsupported source format: {suffix or path.name!r}")


__all__ = [
    "COLUMN_ALIASES",
    "INGEST_VERSION",
    "ExtractedRow",
    "FieldStatus",
    "Field_",
    "IngestError",
    "IngestResult",
    "build_row",
    "ingest",
    "ingest_csv",
    "ingest_pdf",
    "ingest_xlsx",
    "map_columns",
    "parse_decimal",
    "parse_int",
    "parse_text",
]
