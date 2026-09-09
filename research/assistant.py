"""AI research assistant: hypotheses from a model, facts from sources.

The organising idea is a hard separation. A language model is good at
suggesting *what to investigate* — categories worth a look, search terms to
try. It is incapable of knowing what a trimmer costs in Delhi this week, and
any number it produces is a fabrication that becomes indistinguishable from a
measurement the moment it lands in a CSV.

So every field carries one of three provenances:

``SOURCED``       a legitimate source supplied it, with a reference and a date
``AI_INFERENCE``  a model suggested it; never a number, never evidence
``UNAVAILABLE``   nothing could supply it, and that is recorded rather than filled

The export path enforces this: a value with ``AI_INFERENCE`` provenance is
never written into a factual column of ``observations.csv``. The AI can put a
product on the list; it cannot put a price on it.

Degradation is normal. With no LLM client configured the assistant still runs
using the operator's own search terms, and with no live sources every factual
field comes back ``UNAVAILABLE`` — which is the honest answer for most Indian
marketplaces today, not a failure.
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

from catalog.enums import AIService
from llm.services import LLMOutcome, LLMService, LLMUnavailable, NullLLMClient
from sourcing.feeds import DEFAULT_FIXTURE_DIR, FeedResult, build_registry, collect
from sourcing.model import (
    DiscoveredProduct,
    FieldSource,
    Observed,
    enrich,
    tokenize,
    unavailable,
)

logger = logging.getLogger(__name__)

RESEARCH_VERSION = "research/1.0.0"

#: Fields a model may never populate. Each is something an engine downstream
#: treats as evidence, so a fabricated value would change a business decision.
FACTUAL_FIELDS: tuple[str, ...] = (
    "observed_price_inr",
    "rank",
    "trend",
    "review_count",
    "review_velocity",
    "search_interest",
    "competitor_count",
    "brand_dominance",
    "availability",
    "gtin",
    "hsn_code",
    "weight_g",
    "moq_units",
    "gst_rate_pct",
    "supplier_name",
    "unit_cost_inr",
)


class Provenance(StrEnum):
    """Where a value came from. The whole design turns on this distinction."""

    SOURCED = "SOURCED"
    AI_INFERENCE = "AI_INFERENCE"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def is_fact(self) -> bool:
        """Whether a value may be treated as evidence by the engines."""
        return self is Provenance.SOURCED


class ResearchError(RuntimeError):
    """Raised when a research request cannot be run at all."""


@dataclass(frozen=True)
class Claim:
    """One value, and the honest account of where it came from."""

    field: str
    value: Any
    provenance: Provenance
    reference: str | None = None
    observed_at: datetime | None = None
    confidence: str = "low"
    note: str | None = None

    def __post_init__(self) -> None:
        if self.provenance is Provenance.UNAVAILABLE and self.value is not None:
            raise ValueError(f"{self.field}: an UNAVAILABLE claim cannot carry a value")
        if self.provenance is Provenance.SOURCED and not self.reference:
            raise ValueError(
                f"{self.field}: a SOURCED claim must name its source, or it is "
                "indistinguishable from a guess"
            )
        if self.provenance is Provenance.AI_INFERENCE and self.field in FACTUAL_FIELDS:
            raise ValueError(
                f"{self.field}: a model may not supply a factual field. "
                "Suggesting what to investigate is not the same as knowing it."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": None if self.value is None else str(self.value),
            "provenance": str(self.provenance),
            "reference": self.reference,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "confidence": self.confidence,
            "note": self.note,
        }


@dataclass
class ResearchCandidate:
    """One product the assistant proposes investigating."""

    name: str
    category: str | None = None
    rationale: str = ""
    search_terms: list[str] = field(default_factory=list)
    ai_confidence: str = "low"
    claims: list[Claim] = field(default_factory=list)
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def claim(self, name: str) -> Claim | None:
        return next((c for c in self.claims if c.field == name), None)

    @property
    def sourced_fields(self) -> list[str]:
        return [c.field for c in self.claims if c.provenance.is_fact]

    @property
    def unavailable_fields(self) -> list[str]:
        return [c.field for c in self.claims if c.provenance is Provenance.UNAVAILABLE]

    @property
    def evidence_ratio(self) -> Decimal:
        """Share of factual fields an actual source supplied."""
        if not self.claims:
            return Decimal("0")
        return Decimal(len(self.sourced_fields)) / Decimal(len(self.claims)) * 100

    @property
    def has_any_evidence(self) -> bool:
        return bool(self.sourced_fields)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "category": self.category,
            "rationale": self.rationale,
            "search_terms": list(self.search_terms),
            "ai_confidence": self.ai_confidence,
            "provenance_summary": {
                "sourced": self.sourced_fields,
                "unavailable": self.unavailable_fields,
                "evidence_pct": str(self.evidence_ratio.quantize(Decimal("0.1"))),
            },
            "claims": [c.as_dict() for c in self.claims],
            # Stated on every candidate so a reader never has to infer it.
            "is_ai_suggestion": True,
            "facts_are_ai_generated": False,
        }


@dataclass
class ResearchRun:
    """One research request and everything it produced."""

    request: str
    candidates: list[ResearchCandidate] = field(default_factory=list)
    llm_outcome: LLMOutcome | None = None
    feed_results: list[FeedResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "COMPLETE"
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def ai_available(self) -> bool:
        return self.llm_outcome is not None and self.llm_outcome.usable

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "request": self.request,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "ai_available": self.ai_available,
            "ai_error": (
                None
                if self.llm_outcome is None or self.llm_outcome.usable
                else (self.llm_outcome.error or str(self.llm_outcome.schema_violations))
            ),
            "candidates": [c.as_dict() for c in self.candidates],
            "sources": [f.as_dict() for f in self.feed_results],
            "warnings": list(self.warnings),
            "disclaimer": (
                "Candidates are AI suggestions about what to investigate. Every "
                "factual field is either SOURCED with a reference or recorded as "
                "UNAVAILABLE. No price, cost, MOQ, GST, demand or supplier figure "
                "here was generated by a model."
            ),
        }


# --------------------------------------------------------------------------
# Request parsing
# --------------------------------------------------------------------------

_LIMIT = re.compile(r"\b(\d{1,3})\s*(?:products?|items?|candidates?|skus?)\b", re.I)
_BUDGET = re.compile(
    r"(?:under|below|<|max(?:imum)?)\s*(?:\u20b9|rs\.?|inr)?\s*([\d,]+)", re.I
)


@dataclass(frozen=True)
class ParsedRequest:
    """Structured constraints read out of a free-text request.

    Parsed deterministically rather than by the model: a misread budget would
    silently change which candidates are considered, and regex failure is
    visible where a model's misinterpretation is not.
    """

    text: str
    limit: int = 20
    max_supplier_cost_inr: Decimal | None = None

    @classmethod
    def parse(cls, text: str) -> ParsedRequest:
        limit_match = _LIMIT.search(text)
        budget_match = _BUDGET.search(text)
        limit = int(limit_match.group(1)) if limit_match else 20
        budget = Decimal(budget_match.group(1).replace(",", "")) if budget_match else None
        return cls(
            text=text.strip(), limit=max(1, min(limit, 50)), max_supplier_cost_inr=budget
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "limit": self.limit,
            "max_supplier_cost_inr": (
                None
                if self.max_supplier_cost_inr is None
                else str(self.max_supplier_cost_inr)
            ),
        }


# --------------------------------------------------------------------------
# The assistant
# --------------------------------------------------------------------------


#: A candidate must share at least this fraction of its own identifying tokens
#: with an observation before that observation's facts may be attributed to it.
ATTACH_CONTAINMENT = Decimal("0.8")

#: And it must share at least this many. A single shared token is what caused
#: every "LED <anything>" suggestion to inherit one LED strip light's price:
#: "led" alone is a category, not an identity.
ATTACH_MIN_SHARED_TOKENS = 2


def _identity_tokens(text: str) -> frozenset[str]:
    """Identity-bearing tokens, using the project's existing tokenizer.

    Reuses ``sourcing.model.tokenize`` so research matches on exactly the
    vocabulary the discovery matcher uses -- marketing noise stripped, model
    codes preserved -- rather than a second, looser notion of a word.
    """
    return frozenset(tokenize(text))


def _attribution_score(
    candidate_tokens: frozenset[str], product_tokens: frozenset[str]
) -> tuple[Decimal, int]:
    """How well an observation supports a candidate.

    Containment of the *candidate's* tokens, not Jaccard. A generic suggestion
    ("Hair Trimmer") is legitimately supported by a specific observation
    ("Kemei KX-880 Professional Hair Trimmer") whose extra tokens should not
    count against it -- Jaccard would punish exactly the case that is correct.
    The reverse never holds: an observation is only evidence for a candidate
    whose terms it actually covers.
    """
    if not candidate_tokens:
        return Decimal("0"), 0
    shared = candidate_tokens & product_tokens
    containment = Decimal(len(shared)) / Decimal(len(candidate_tokens))
    return containment, len(shared)


def _assign_observations(
    candidates: list[ResearchCandidate], observed: list[DiscoveredProduct]
) -> dict[uuid.UUID, DiscoveredProduct]:
    """Attribute each observation to at most one candidate.

    Exclusivity is the second half of the fix. An observation describes one
    real product someone actually saw; if two candidates both clear the
    similarity bar for it, that is ambiguity, not corroboration of both. The
    best-scoring candidate takes it and the others get nothing, which is why a
    list of near-identical suggestions can no longer all inherit the same
    price and reference.

    Deterministic: pairs are ranked by score, then by shared-token count, then
    by stable identifiers, so the same inputs always produce the same
    attribution.
    """
    pairs: list[tuple[Decimal, int, str, str, ResearchCandidate, DiscoveredProduct]] = []
    for candidate in candidates:
        candidate_tokens = _identity_tokens(candidate.name)
        for product in observed:
            containment, shared = _attribution_score(
                candidate_tokens, product.match_tokens
            )
            if containment < ATTACH_CONTAINMENT or shared < ATTACH_MIN_SHARED_TOKENS:
                continue
            pairs.append(
                (
                    containment,
                    shared,
                    candidate.name,
                    product.external_ref,
                    candidate,
                    product,
                )
            )

    pairs.sort(key=lambda p: (-p[0], -p[1], p[2], p[3]))

    assigned: dict[uuid.UUID, DiscoveredProduct] = {}
    taken: set[str] = set()
    for _containment, _shared, _name, ref, candidate, product in pairs:
        if candidate.id in assigned or ref in taken:
            continue
        assigned[candidate.id] = product
        taken.add(ref)
    return assigned


class ResearchAssistant:
    """Runs a research request: model suggests, sources supply, nothing invents."""

    def __init__(
        self,
        *,
        researcher: LLMService | None = None,
        fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    ) -> None:
        # Defaults to the refusing client, so nothing calls out unless a
        # transport is deliberately supplied.
        self.researcher = researcher or LLMService(AIService.RESEARCHER, NullLLMClient())
        self.fixture_dir = fixture_dir

    def run(self, request: str) -> ResearchRun:
        if not request.strip():
            raise ResearchError("a research request cannot be empty")

        parsed = ParsedRequest.parse(request)
        run = ResearchRun(request=parsed.text)

        candidates, outcome = self._suggest(parsed)
        run.llm_outcome = outcome
        if outcome is not None and not outcome.usable:
            run.status = "DEGRADED"
            run.warnings.append(
                "no research model is configured, so candidates were derived "
                "from the words in your request rather than suggested"
            )
            if not candidates:
                run.warnings.append(
                    "your request names no specific product or category, and "
                    "without a research model there is nothing to derive terms "
                    "from. Either configure a research model or name what to "
                    "look for, e.g. 'hair trimmers under 500'. An empty result "
                    "here means the assistant could not search, not that no "
                    "promising products exist."
                )

        observed, feeds = collect(build_registry(fixture_dir=self.fixture_dir))
        run.feed_results = feeds
        unavailable_sources = [f for f in feeds if not f.available]
        if unavailable_sources:
            run.warnings.append(
                f"{len(unavailable_sources)} of {len(feeds)} sources have no "
                "legitimate route; fields they would supply stay UNAVAILABLE"
            )

        selected = candidates[: parsed.limit]
        # Computed once across the whole run: exclusivity only means anything
        # if every candidate is considered together, since the rule is that
        # one observation supports at most one of them.
        assignment = _assign_observations(selected, observed)

        for candidate in selected:
            self._attach_evidence(candidate, observed, assignment)
            run.candidates.append(candidate)

        if not any(c.has_any_evidence for c in run.candidates):
            run.warnings.append(
                "no candidate has any sourced fact. These are hypotheses only "
                "and cannot be analysed until observations are recorded."
            )

        logger.info(
            "research complete",
            extra={
                "operation": "research",
                "result": f"{len(run.candidates)} candidates, status {run.status}",
            },
        )
        return run

    # -- model ------------------------------------------------------------

    def _suggest(
        self, parsed: ParsedRequest
    ) -> tuple[list[ResearchCandidate], LLMOutcome | None]:
        """Ask the model for hypotheses; fall back to the request itself."""
        prompt = (
            "Suggest product categories worth investigating for an Indian "
            "marketplace seller.\n\n"
            f"Request: {parsed.text}\n"
            f"Suggest at most {parsed.limit} candidates.\n"
            "Remember: name, category, rationale, search_terms and confidence "
            "only. No prices, costs, MOQ, GST, demand or supplier information — "
            "those are gathered from sources."
        )
        outcome = self.researcher.invoke(prompt, input_refs=[parsed.text[:80]])

        if not outcome.usable or not outcome.parsed:
            return self._fallback(parsed), outcome

        candidates: list[ResearchCandidate] = []
        for raw in outcome.parsed.get("candidates", []):
            candidates.append(
                ResearchCandidate(
                    name=str(raw.get("name", "")).strip(),
                    category=(raw.get("category") or None),
                    rationale=str(raw.get("rationale", ""))[:400],
                    search_terms=[str(t) for t in raw.get("search_terms", [])][:8],
                    ai_confidence=str(raw.get("confidence", "low")).lower(),
                )
            )
        return [c for c in candidates if c.name], outcome

    def _fallback(self, parsed: ParsedRequest) -> list[ResearchCandidate]:
        """Derive search terms from the request when no model is available.

        Not a simulation of the model. It produces one candidate built from the
        operator's own words, clearly marked low confidence, so the rest of the
        pipeline can still be exercised without an LLM.
        """
        words = [
            w
            for w in re.findall(r"[a-zA-Z]{3,}", parsed.text.lower())
            if w
            not in {
                "find",
                "promising",
                "products",
                "product",
                "under",
                "india",
                "indian",
                "commerce",
                "ecommerce",
                "supplier",
                "cost",
                "with",
            }
        ]
        if not words:
            # A generic request with no product noun leaves nothing to search
            # for. Returning an empty list silently would look like "no
            # promising products", which is a different claim entirely.
            return []
        return [
            ResearchCandidate(
                name=" ".join(words[:4]).title(),
                category=None,
                rationale=(
                    "derived from the request text because no research model is "
                    "configured; this is not a model suggestion"
                ),
                search_terms=words[:6],
                ai_confidence="low",
            )
        ]

    # -- evidence ---------------------------------------------------------

    def _attach_evidence(
        self,
        candidate: ResearchCandidate,
        observed: list[DiscoveredProduct],
        assignment: dict[uuid.UUID, DiscoveredProduct] | None = None,
    ) -> None:
        """Attach facts from the observation assigned to this candidate.

        The assignment is computed once across the whole run by
        ``_assign_observations``, which enforces two rules an earlier version
        did not: an observation must cover most of a candidate's identifying
        tokens (not merely share one), and it may support only one candidate.

        Without those, a single shared word was enough to copy one product's
        price, HSN, weight and demand figures onto every superficially similar
        suggestion -- eighteen different LED products all reporting one LED
        strip light's facts under its reference. Facts are attributed to the
        product they describe or to nothing.
        """
        if assignment is None:
            assignment = _assign_observations([candidate], observed)
        match = assignment.get(candidate.id)

        for name in (
            "observed_price_inr",
            "search_interest",
            "trend",
            "review_count",
            "review_velocity",
            "competitor_count",
            "brand_dominance",
            "rank",
            "availability",
        ):
            if match is None:
                candidate.claims.append(
                    Claim(
                        field=name,
                        value=None,
                        provenance=Provenance.UNAVAILABLE,
                        note="no source observation matched this candidate",
                    )
                )
                continue

            observation: Observed = match.signal_field(name)
            if observation.present:
                candidate.claims.append(
                    Claim(
                        field=name,
                        value=observation.value,
                        provenance=Provenance.SOURCED,
                        reference=observation.reference or match.external_ref,
                        observed_at=observation.observed_at,
                        confidence="high"
                        if observation.source is FieldSource.OFFICIAL_API
                        else "medium",
                    )
                )
            else:
                candidate.claims.append(
                    Claim(
                        field=name,
                        value=None,
                        provenance=Provenance.UNAVAILABLE,
                        note=observation.reason or "not supplied by any source",
                    )
                )

        # Identity fields, only where a source carried them.
        if match is not None:
            for name, value in (
                ("gtin", match.gtin),
                ("hsn_code", match.hsn_code),
                ("weight_g", match.weight_g),
            ):
                candidate.claims.append(
                    Claim(
                        field=name,
                        value=value,
                        provenance=Provenance.SOURCED
                        if value
                        else Provenance.UNAVAILABLE,
                        reference=match.external_ref if value else None,
                        note=None if value else "not present in the source record",
                    )
                )
            candidate.category = candidate.category or match.category

        # Supplier facts never come from research. They come from an ingested
        # catalogue, which is the only place an acquisition cost can exist.
        for name in ("supplier_name", "unit_cost_inr", "moq_units", "gst_rate_pct"):
            candidate.claims.append(
                Claim(
                    field=name,
                    value=None,
                    provenance=Provenance.UNAVAILABLE,
                    note=(
                        "supplier facts come from an ingested supplier catalogue, "
                        "never from research"
                    ),
                )
            )

    # -- handoff ----------------------------------------------------------

    def to_discovered(self, candidate: ResearchCandidate) -> DiscoveredProduct:
        """Convert to the existing discovery model for the analysis pipeline.

        Only SOURCED claims become values. An AI suggestion enters the pipeline
        as a product with no evidence, which the existing coverage rules will
        correctly refuse to rank.
        """
        product = DiscoveredProduct(
            source="ai_research",
            external_ref=f"research://{candidate.id}",
            raw_title=candidate.name,
            category=candidate.category,
        )
        for claim in candidate.claims:
            if not hasattr(product, claim.field):
                continue
            if claim.provenance.is_fact and claim.value is not None:
                setattr(
                    product,
                    claim.field,
                    Observed(
                        claim.value,
                        FieldSource.MANUAL,
                        reference=claim.reference,
                        observed_at=claim.observed_at or datetime.now(UTC),
                    ),
                )
            elif claim.field in {f.name for f in _observed_fields()}:
                setattr(
                    product,
                    claim.field,
                    unavailable(claim.note or "not supplied by any source"),
                )
        product.notes.append(
            "candidate suggested by AI research; all factual fields are sourced "
            "or UNAVAILABLE"
        )
        return enrich(product)


def _observed_fields() -> list[Any]:
    """The DiscoveredProduct fields that hold an Observed value."""
    import dataclasses  # noqa: PLC0415 - local to keep the import surface small

    return [
        f
        for f in dataclasses.fields(DiscoveredProduct)
        if f.type == "Observed" or "Observed" in str(f.type)
    ]


def build_researcher(client: Any | None = None) -> LLMService:
    """Construct the research service using the project's LLM abstraction.

    Credentials never appear here: the caller supplies a transport, and the
    default refuses to call anything.
    """
    return LLMService(AIService.RESEARCHER, client or NullLLMClient())


def default_researcher(environ: dict[str, str] | None = None) -> LLMService:
    """The researcher the API uses when nothing is explicitly injected.

    Reads OPENAI_API_KEY from the environment at call time and never stores,
    logs, or returns it. With no key set this degrades to NullLLMClient, which
    is exactly the same degraded-but-working path the assistant already takes
    on any other transport failure -- there is no special case for "no key"
    versus "key rejected" versus "network unreachable" beyond where the
    failure is detected.
    """
    from llm.openai_client import OpenAIClient  # noqa: PLC0415 - optional transport

    try:
        client: Any = OpenAIClient(environ=environ)
    except LLMUnavailable as exc:
        logger.info("research model unavailable", extra={"detail": str(exc)})
        client = NullLLMClient()
    return LLMService(AIService.RESEARCHER, client)


__all__ = [
    "FACTUAL_FIELDS",
    "RESEARCH_VERSION",
    "Claim",
    "ParsedRequest",
    "Provenance",
    "ResearchAssistant",
    "ResearchCandidate",
    "ResearchError",
    "ResearchRun",
    "build_researcher",
    "default_researcher",
]
