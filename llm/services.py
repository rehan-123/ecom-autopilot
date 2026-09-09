"""The four Claude call sites, with schema validation and safe degradation.

Extractor, Judge Identity, Judge Commercial, Writer. Each builds a prompt,
calls a transport, validates the response against a schema, and records a full
invocation record whether it succeeded or not.

Two properties matter more than the prompts:

**A malformed response never becomes a business decision.** Every response is
schema-validated. A violation produces an ``LLMOutcome`` with ``parsed_ok =
False`` and no findings, which the decision engine treats as missing evidence
and routes to human review. There is no path from bad JSON to an approval.

**The Judges emit findings, not decisions.** A finding is an observation with
evidence and a confidence; what a finding *means* is decided by deterministic
rules in ``decisions.rules``. If the Judge returned a verdict, the verdict
would be unauditable and non-reproducible, which is exactly what the rest of
this architecture exists to avoid.

Transport is injected. The default ``NullLLMClient`` raises
``LLMUnavailable``, so nothing in this repository makes a network call unless
a client is deliberately supplied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from catalog.enums import AIService

logger = logging.getLogger(__name__)

PROMPT_VERSIONS: dict[AIService, str] = {
    AIService.EXTRACTOR: "extractor/1.0.0",
    AIService.JUDGE_IDENTITY: "judge-identity/1.0.0",
    AIService.JUDGE_COMMERCIAL: "judge-commercial/1.0.0",
    AIService.WRITER: "writer/1.0.0",
    AIService.RESEARCHER: "researcher/1.0.0",
}


class LLMUnavailable(RuntimeError):
    """No transport is configured, or the transport failed."""


class LLMClient(Protocol):
    """Transport contract.

    Deliberately minimal: prompt in, text out. Keeping the interface this
    narrow means a real client, a recorded fixture and a failing stub are
    interchangeable, so every degradation path is testable without a network.
    """

    def complete(self, *, system: str, user: str, max_tokens: int = 2000) -> str:
        """Return the model's raw text response."""
        ...


class NullLLMClient:
    """Default transport: refuses to call anything.

    Part 1 is fully functional without an LLM — every gate before the Judges
    is deterministic — so the absence of a client degrades the pipeline rather
    than breaking it.
    """

    def complete(self, *, system: str, user: str, max_tokens: int = 2000) -> str:
        raise LLMUnavailable(
            "no LLM client configured; supply one to enable Claude services"
        )


class RecordedLLMClient:
    """Replays canned responses, keyed by service.

    Used by tests and the demo so that the full pipeline — including the Claude
    stages — can be exercised deterministically and offline.
    """

    def __init__(self, responses: dict[str, str], *, fail_on: set[str] | None = None):
        self._responses = responses
        self._fail_on = fail_on or set()
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 2000) -> str:
        key = system.split("\n", 1)[0].strip()
        self.calls.append((key, user))
        if key in self._fail_on:
            raise LLMUnavailable(f"simulated transport failure for {key}")
        try:
            return self._responses[key]
        except KeyError as exc:
            raise LLMUnavailable(f"no recorded response for {key!r}") from exc


@dataclass
class LLMOutcome:
    """One invocation, successful or not, with everything needed to audit it."""

    service: AIService
    model: str
    model_version: str | None
    prompt_version: str
    prompt_hash: str
    input_refs: list[str]
    parsed_ok: bool
    parsed: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    schema_violations: list[str] = field(default_factory=list)
    raw_output: str | None = None
    latency_ms: int = 0
    input_token_count: int | None = None
    output_token_count: int | None = None
    cost_inr: Decimal = Decimal("0")
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def usable(self) -> bool:
        return self.parsed_ok and not self.schema_violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": str(self.service),
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "input_refs": list(self.input_refs),
            "parsed_ok": self.parsed_ok,
            "findings": list(self.findings),
            "schema_violations": list(self.schema_violations),
            "latency_ms": self.latency_ms,
            "cost_inr": str(self.cost_inr),
            "error": self.error,
        }


def _hash_prompt(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    Models sometimes wrap JSON in prose or fences. Stripping those is
    tolerated; anything still unparseable is a schema violation, not something
    to guess at.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------

ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def _validate_findings(payload: dict[str, Any]) -> list[str]:
    """Validate a Judge response.

    Enforces the shape that keeps decisions deterministic: findings carry a
    flag, a confidence and evidence references, and nothing else is treated as
    meaningful.
    """
    violations: list[str] = []
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return ["'findings' must be a list"]

    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            violations.append(f"findings[{index}] is not an object")
            continue
        if not isinstance(finding.get("flag"), str) or not finding.get("flag"):
            violations.append(f"findings[{index}].flag missing or not a string")
        confidence = str(finding.get("confidence", "")).lower()
        if confidence not in ALLOWED_CONFIDENCE:
            violations.append(
                f"findings[{index}].confidence must be one of {sorted(ALLOWED_CONFIDENCE)}"
            )
        if not isinstance(finding.get("evidence_refs", []), list):
            violations.append(f"findings[{index}].evidence_refs must be a list")

    if not isinstance(payload.get("unresolved_questions", []), list):
        violations.append("'unresolved_questions' must be a list")
    return violations


def _validate_extraction(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ["'rows' must be a list"]
    allowed = {"KNOWN", "INFERRED", "UNKNOWN", "REQUIRES_REVIEW"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(f"rows[{index}] is not an object")
            continue
        for name, value in row.items():
            if not isinstance(value, dict) or "status" not in value:
                violations.append(f"rows[{index}].{name} must carry a status")
            elif value["status"] not in allowed:
                violations.append(
                    f"rows[{index}].{name}.status must be one of {sorted(allowed)}"
                )
    return violations


WRITER_LIMITS = {"title": 200, "description": 5000}


def _validate_research(payload: dict[str, Any]) -> list[str]:
    """Validate a researcher response.

    The strictest validator in the system, because this is the one call site
    where a model is asked to be creative. A candidate may carry a name, a
    category and a rationale -- nothing else. Any numeric field is rejected
    outright: a price or review count from a language model is a fabrication
    wearing the costume of a fact, and the rest of the pipeline cannot tell
    the difference once it is in a CSV.
    """
    violations: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ["'candidates' must be a list"]

    #: Fields a model must never populate. Each is something the economics,
    #: scoring or compliance engines treat as evidence.
    FORBIDDEN = (
        "price",
        "cost",
        "rate",
        "mrp",
        "moq",
        "gst",
        "hsn",
        "weight",
        "reviews",
        "review_count",
        "review_velocity",
        "rank",
        "sales",
        "units",
        "revenue",
        "margin",
        "demand",
        "search_interest",
        "competitors",
        "competitor_count",
        "supplier",
        "supplier_name",
        "trend",
        "interest",
        "stock",
        "availability",
        "gtin",
        "ean",
    )
    allowed = {"name", "category", "rationale", "search_terms", "confidence"}

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            violations.append(f"candidates[{index}] is not an object")
            continue
        if not str(candidate.get("name", "")).strip():
            violations.append(f"candidates[{index}].name is missing")
        confidence = str(candidate.get("confidence", "")).lower()
        if confidence not in ALLOWED_CONFIDENCE:
            violations.append(
                f"candidates[{index}].confidence must be one of {sorted(ALLOWED_CONFIDENCE)}"
            )
        for key in candidate:
            lowered = key.lower()
            if key not in allowed:
                violations.append(f"candidates[{index}] has unexpected field {key!r}")
            if any(term in lowered for term in FORBIDDEN):
                violations.append(
                    f"candidates[{index}].{key} looks like a factual claim; the "
                    "researcher may suggest what to investigate, never what is true"
                )
        terms = candidate.get("search_terms", [])
        if not isinstance(terms, list):
            violations.append(f"candidates[{index}].search_terms must be a list")
    return violations


def _validate_listing(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for required in ("title", "bullets", "description", "keywords"):
        if required not in payload:
            violations.append(f"missing required field {required!r}")
    title = payload.get("title")
    if isinstance(title, str) and len(title) > WRITER_LIMITS["title"]:
        violations.append(f"title exceeds {WRITER_LIMITS['title']} characters")
    for list_field in ("bullets", "keywords"):
        if list_field in payload and not isinstance(payload[list_field], list):
            violations.append(f"{list_field!r} must be a list")
    description = payload.get("description")
    if isinstance(description, str) and len(description) > WRITER_LIMITS["description"]:
        violations.append("description exceeds the character limit")
    # A listing that invents specifications is worse than no listing.
    if payload.get("invented_specifications"):
        violations.append("response admits to inventing specifications")
    return violations


_VALIDATORS = {
    AIService.EXTRACTOR: _validate_extraction,
    AIService.JUDGE_IDENTITY: _validate_findings,
    AIService.JUDGE_COMMERCIAL: _validate_findings,
    AIService.WRITER: _validate_listing,
    AIService.RESEARCHER: _validate_research,
}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[AIService, str] = {
    AIService.EXTRACTOR: (
        "EXTRACTOR\n"
        "You convert messy supplier catalogue text into structured rows.\n"
        'Return ONLY a JSON object: {"rows": [{"field": {"value": ..., '
        '"status": "KNOWN|INFERRED|UNKNOWN|REQUIRES_REVIEW"}}]}.\n'
        "Never invent a value. If a field is absent, status is UNKNOWN and "
        "value is null. If a value is present but ambiguous, status is "
        "REQUIRES_REVIEW."
    ),
    AIService.JUDGE_IDENTITY: (
        "JUDGE_IDENTITY\n"
        "You assess whether supplier offers describe the same real-world "
        "product.\n"
        'Return ONLY a JSON object: {"findings": [{"flag": ..., '
        '"confidence": "high|medium|low", "evidence_refs": [...], '
        '"reasoning": ...}], "unresolved_questions": [...]}.\n'
        "Report observations, not decisions. Use flags SAME_PRODUCT, "
        "SUSPECTED_DUPLICATE, DIFFERENT_PRODUCT, INSUFFICIENT_EVIDENCE."
    ),
    AIService.JUDGE_COMMERCIAL: (
        "JUDGE_COMMERCIAL\n"
        "You review assembled evidence about a product and report concerns.\n"
        "Return ONLY a JSON object with 'findings' and 'unresolved_questions'.\n"
        "Report observations, not decisions. Never state whether to list the "
        "product. Use flags such as QUALITY_CONCERN, MISLEADING_CLAIM, "
        "RETURN_RISK, BRAND_RISK, EVIDENCE_CONTRADICTION."
    ),
    AIService.RESEARCHER: (
        "RESEARCHER\n"
        "You suggest product categories worth investigating for an Indian "
        "marketplace seller.\n"
        'Return ONLY a JSON object: {"candidates": [{"name": ..., '
        '"category": ..., "rationale": ..., "search_terms": [...], '
        '"confidence": "high|medium|low"}]}.\n'
        "You are generating HYPOTHESES, not facts. You must NEVER state or "
        "estimate a price, cost, MOQ, GST rate, HSN code, weight, review "
        "count, sales figure, ranking, search volume, competitor count, or "
        "supplier. Those are gathered from sources, and a value you invent "
        "would be indistinguishable from a measured one once recorded. "
        "Include only name, category, rationale, search_terms and confidence."
    ),
    AIService.WRITER: (
        "WRITER\n"
        "You draft marketplace listing content from verified product facts.\n"
        "Return ONLY a JSON object with title, bullets, description, keywords, "
        "attributes, image_requirements.\n"
        "Use only facts supplied to you. Never invent specifications, "
        "certifications, or performance claims. Never copy competitor text."
    ),
}


class LLMService:
    """Wraps one call site: prompt, transport, validation, audit record."""

    def __init__(
        self,
        service: AIService,
        client: LLMClient | None = None,
        *,
        model: str = "claude-sonnet-4-6",
        model_version: str | None = None,
    ) -> None:
        self.service = service
        self.client = client or NullLLMClient()
        self.model = model
        self.model_version = model_version

    def invoke(
        self, user_prompt: str, *, input_refs: list[str] | None = None
    ) -> LLMOutcome:
        """Call the transport and validate the response.

        Never raises for a transport or schema problem: the failure is
        recorded in the outcome so the caller sees missing evidence rather
        than an exception that might be swallowed somewhere upstream.
        """
        system = SYSTEM_PROMPTS[self.service]
        refs = input_refs or []
        outcome_base = {
            "service": self.service,
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": PROMPT_VERSIONS[self.service],
            "prompt_hash": _hash_prompt(system, user_prompt),
            "input_refs": refs,
        }

        started = time.perf_counter()
        try:
            raw = self.client.complete(system=system, user=user_prompt)
        except Exception as exc:  # noqa: BLE001 - any transport failure degrades
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.warning("LLM transport failed", extra={"detail": str(exc)})
            return LLMOutcome(
                **outcome_base, parsed_ok=False, latency_ms=elapsed, error=str(exc)
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        try:
            payload = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return LLMOutcome(
                **outcome_base,
                parsed_ok=False,
                raw_output=raw,
                latency_ms=elapsed,
                schema_violations=[f"unparseable response: {exc}"],
            )

        violations = _VALIDATORS[self.service](payload)
        findings = payload.get("findings", []) if not violations else []
        return LLMOutcome(
            **outcome_base,
            parsed_ok=not violations,
            parsed=payload if not violations else None,
            findings=findings if isinstance(findings, list) else [],
            schema_violations=violations,
            raw_output=raw,
            latency_ms=elapsed,
        )


def build_identity_prompt(offers: list[dict[str, Any]]) -> str:
    return (
        "Assess whether these supplier offers describe the same product.\n\n"
        + json.dumps(offers, indent=2, default=str)
    )


def build_commercial_prompt(evidence: dict[str, Any]) -> str:
    return (
        "Review this assembled evidence and report concerns as findings.\n\n"
        + json.dumps(evidence, indent=2, default=str)
    )


def build_writer_prompt(facts: dict[str, Any]) -> str:
    return "Draft listing content using ONLY these verified facts.\n\n" + json.dumps(
        facts, indent=2, default=str
    )


def build_extractor_prompt(text: str) -> str:
    return "Extract supplier catalogue rows from this text.\n\n" + text


__all__ = [
    "PROMPT_VERSIONS",
    "SYSTEM_PROMPTS",
    "LLMClient",
    "LLMOutcome",
    "LLMService",
    "LLMUnavailable",
    "NullLLMClient",
    "RecordedLLMClient",
    "build_commercial_prompt",
    "build_extractor_prompt",
    "build_identity_prompt",
    "build_writer_prompt",
]
