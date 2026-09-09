"""Export research results into the existing fixture schemas.

The enforcement point. Two rules, at different levels.

**Per field:** a claim that is not ``SOURCED`` never becomes a value in a
factual column — it is written blank, which the existing ingester already
reads as UNKNOWN. That keeps a model's suggestion from acquiring the authority
of a measurement once it is sitting in a CSV next to real observations.

**Per candidate:** a candidate with no sourced claim at all is not written
as an observation. A row carrying only a title is still a row, and the
ingester would read it back as a product that someone observed. It remains
visible in the Research screen — it is a research result, just not evidence.

Supplier rows are never written from research at all. An acquisition cost can
only come from a supplier catalogue the operator ingested.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

from research.assistant import Provenance, ResearchCandidate, ResearchRun

logger = logging.getLogger(__name__)

EXPORT_VERSION = "research-export/1.0.0"

#: Column order of the existing observations fixture. Matched exactly so the
#: file the assistant writes is the file the ingester already reads.
OBSERVATION_COLUMNS = (
    "title",
    "price",
    "category",
    "brand",
    "model",
    "gtin",
    "hsn",
    "weight_g",
    "trend",
    "search_interest",
    "competitors",
    "brand_dominance",
    "reference",
)
TRENDS_COLUMNS = ("term", "interest", "direction", "category")

#: Which claim maps to which observations column.
_COLUMN_CLAIM = {
    "price": "observed_price_inr",
    "gtin": "gtin",
    "hsn": "hsn_code",
    "weight_g": "weight_g",
    "trend": "trend",
    "search_interest": "search_interest",
    "competitors": "competitor_count",
    "brand_dominance": "brand_dominance",
}


def _sourced(candidate: ResearchCandidate, field: str) -> str:
    """The value only if a source supplied it; blank otherwise.

    Blank is meaningful: the existing ingester records an absent column as
    UNKNOWN rather than defaulting it, so an unavailable fact stays unavailable
    all the way through.
    """
    claim = candidate.claim(field)
    if claim is None or claim.provenance is not Provenance.SOURCED:
        return ""
    return "" if claim.value is None else str(claim.value)


def observation_rows(run: ResearchRun) -> list[dict[str, str]]:
    """Rows for observations.csv, carrying only sourced facts.

    A candidate with no ``SOURCED`` claim at all produces no row. It is a
    hypothesis the model proposed and nothing corroborated, and writing it
    would file a bare title into the observations fixture as though someone
    had observed it. The ingester would then read that row back as a real
    product with every field UNKNOWN, and it would go on to dilute discovery
    coverage and compete for a place in the opportunity ranking on the
    strength of a suggestion.

    Such candidates stay fully visible in the Research screen, with their
    rationale and their UNAVAILABLE claims intact -- they are a research
    result, just not an observation. The distinction is the point: the model
    may put a product on the list, but only a source can put it in the
    evidence base.
    """
    rows: list[dict[str, str]] = []
    for candidate in run.candidates:
        if not candidate.has_any_evidence:
            logger.info(
                "candidate not exported: no sourced claim",
                extra={"operation": "export", "result": candidate.name[:60]},
            )
            continue
        reference = next(
            (
                c.reference
                for c in candidate.claims
                if c.provenance is Provenance.SOURCED and c.reference
            ),
            f"research://{candidate.id}",
        )
        row = dict.fromkeys(OBSERVATION_COLUMNS, "")
        row["title"] = candidate.name
        row["category"] = candidate.category or ""
        row["reference"] = reference
        for column, field in _COLUMN_CLAIM.items():
            row[column] = _sourced(candidate, field)
        rows.append(row)
    return rows


def trends_rows(run: ResearchRun) -> list[dict[str, str]]:
    """Rows for trends.csv. Written only where a source supplied interest."""
    rows: list[dict[str, str]] = []
    for candidate in run.candidates:
        interest = _sourced(candidate, "search_interest")
        direction = _sourced(candidate, "trend")
        if not interest and not direction:
            continue
        rows.append(
            {
                "term": candidate.name,
                "interest": interest,
                "direction": direction,
                "category": candidate.category or "",
            }
        )
    return rows


def to_csv(rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_observations(run: ResearchRun, path: Path) -> int:
    """Append research observations to the fixture the pipeline reads.

    Appends rather than replaces: manually recorded observations are usually
    better evidence than anything research produces, and overwriting them
    would discard the good data to make room for hypotheses.
    """
    rows = observation_rows(run)
    if not rows:
        return 0
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(OBSERVATION_COLUMNS), lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "research observations written",
        extra={"operation": "export", "result": len(rows)},
    )
    return len(rows)


def write_trends(run: ResearchRun, path: Path) -> int:
    rows = trends_rows(run)
    if not rows:
        return 0
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(TRENDS_COLUMNS), lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def preview(run: ResearchRun) -> dict[str, str]:
    """What would be written, for review before anything touches disk."""
    return {
        "observations.csv": to_csv(observation_rows(run), OBSERVATION_COLUMNS),
        "trends.csv": to_csv(trends_rows(run), TRENDS_COLUMNS),
        "suppliers.csv": (
            "# not written: supplier facts come from an ingested catalogue,\n"
            "# never from research. An acquisition cost has one legitimate source.\n"
        ),
    }


__all__ = [
    "EXPORT_VERSION",
    "OBSERVATION_COLUMNS",
    "TRENDS_COLUMNS",
    "observation_rows",
    "preview",
    "to_csv",
    "trends_rows",
    "write_observations",
    "write_trends",
]
