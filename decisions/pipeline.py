"""The Part 1 pipeline, and the reports it produces.

One SKU, one pass, from supplier offer to either a rejection with a gap or a
listing draft awaiting human approval. Every stage records a decision; the
first blocking disposition stops the run and the SKU keeps the state that
stage assigned.

The primary output is the **rejection report**, not the ranked list. A first
run is typically thirty to fifty candidates with most failing on economics, and
"fails by Rs 34 of landed cost" is a sourcing instruction you can act on. A
ranked list of the six survivors throws away the actionable part.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from catalog.enums import AIService, ScenarioName, SKUState
from catalog.schemas import Product, SupplierOffer
from compliance.engine import ComplianceOutcome, HsnProvenance
from compliance.engine import evaluate as evaluate_compliance
from decisions.rules import (
    Decision,
    Disposition,
    decide_compliance,
    decide_coverage,
    decide_headroom,
    decide_profitability,
    decide_reference_trust,
    interpret_findings,
    require_legal,
)
from econ.calculator import (
    EconomicsInputs,
    EconomicsResult,
    GateOutcome,
    ScenarioDeltas,
    apply_profitability_gate,
    compute_economics,
)
from econ.landed_cost import LandedCostResult, compute_landed_cost
from econ.money import ZERO, format_inr, money
from llm.services import (
    LLMOutcome,
    LLMService,
    build_commercial_prompt,
    build_identity_prompt,
    build_writer_prompt,
)
from reference.loader import ReferenceDoc, ReferenceStore, pin
from signals.market import (
    HeadroomOutcome,
    MarketPriceReference,
    OpportunityScore,
    RiskAssessment,
    Signal,
    assess_risk,
    build_components,
    check_price_headroom,
    compute_opportunity_score,
)

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "pipeline/1.0.0"


@dataclass
class PipelineResult:
    """Everything one SKU's pass produced."""

    sku_code: str
    product_name: str
    final_state: SKUState
    decisions: list[Decision] = field(default_factory=list)
    landed_costs: dict[ScenarioName, LandedCostResult] = field(default_factory=dict)
    economics: dict[ScenarioName, EconomicsResult] = field(default_factory=dict)
    gate: GateOutcome | None = None
    headroom: HeadroomOutcome | None = None
    risk: RiskAssessment | None = None
    compliance: ComplianceOutcome | None = None
    score: OpportunityScore | None = None
    llm_outcomes: list[LLMOutcome] = field(default_factory=list)
    listing_draft: dict[str, Any] | None = None
    pinned_versions: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def blocked_by(self) -> Decision | None:
        """The first decision that stopped the run, if any."""
        return next(
            (
                d
                for d in self.decisions
                if d.disposition
                in (
                    Disposition.REJECTED,
                    Disposition.BLOCKED,
                    Disposition.REQUIRES_HUMAN_REVIEW,
                )
            ),
            None,
        )

    @property
    def expected(self) -> EconomicsResult | None:
        return self.economics.get(ScenarioName.EXPECTED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku_code": self.sku_code,
            "product_name": self.product_name,
            "final_state": str(self.final_state),
            "pipeline_version": PIPELINE_VERSION,
            "pinned_versions": self.pinned_versions,
            "decisions": [d.as_dict() for d in self.decisions],
            "economics": {str(k): v.as_dict() for k, v in self.economics.items()},
            "gate": self.gate.as_dict() if self.gate else None,
            "headroom": self.headroom.as_dict() if self.headroom else None,
            "risk": self.risk.as_dict() if self.risk else None,
            "compliance": self.compliance.as_dict() if self.compliance else None,
            "score": self.score.as_dict() if self.score else None,
            "llm": [o.as_dict() for o in self.llm_outcomes],
            "listing_draft": self.listing_draft,
            "warnings": list(self.warnings),
        }


@dataclass
class PipelineConfig:
    """Inputs the pipeline needs that are not reference data."""

    marketplace: str
    selling_price_inr: Decimal
    ad_allowance_pct: Decimal = Decimal("10")
    expected_monthly_velocity: Decimal = Decimal("10")
    compliance_ruleset_kind: str = "compliance_ruleset:india"
    compliance_ruleset_version: str = "baseline-0"
    scenarios: tuple[ScenarioName, ...] = (
        ScenarioName.BEST,
        ScenarioName.EXPECTED,
        ScenarioName.WORST,
    )


class Pipeline:
    """Runs one SKU through every Part 1 stage."""

    def __init__(
        self,
        store: ReferenceStore,
        *,
        identity_judge: LLMService | None = None,
        commercial_judge: LLMService | None = None,
        writer: LLMService | None = None,
    ) -> None:
        self.store = store
        self.identity_judge = identity_judge or LLMService(AIService.JUDGE_IDENTITY)
        self.commercial_judge = commercial_judge or LLMService(AIService.JUDGE_COMMERCIAL)
        self.writer = writer or LLMService(AIService.WRITER)

    def run(
        self,
        *,
        product: Product,
        offers: list[SupplierOffer],
        sku_code: str,
        config: PipelineConfig,
        signals: dict[str, Signal] | None = None,
        price_reference: MarketPriceReference | None = None,
        compliance_evidence: dict[str, Any] | None = None,
        hsn_provenance: HsnProvenance = HsnProvenance.SOURCED,
    ) -> PipelineResult:
        """Execute the pipeline, stopping at the first blocking decision."""
        result = PipelineResult(
            sku_code=sku_code,
            product_name=product.canonical_name,
            final_state=SKUState.DISCOVERED,
        )
        if not offers:
            result.warnings.append("no supplier offers: nothing to cost")
            self._advance(result, SKUState.DATA_COLLECTION)
            self._advance(result, SKUState.BLOCKED_AWAITING_DATA)
            return result

        self._advance(result, SKUState.DATA_COLLECTION)

        # --- risk first: it selects which threshold profile applies ---------
        benchmark_doc = self._benchmark(product.category)
        benchmark_rate = (
            money(benchmark_doc.payload.get("return_rate_pct", ZERO))
            if benchmark_doc
            else None
        )
        risk = assess_risk(
            product,
            supplier_count=len({o.supplier_id for o in offers}),
            benchmark_return_rate_pct=benchmark_rate,
        )
        result.risk = risk

        references = pin(
            self.store,
            marketplace=config.marketplace,
            category=product.category,
            risk_tier=str(risk.tier),
            scenario=str(ScenarioName.EXPECTED),
        )
        result.pinned_versions = references.as_versions()

        # --- economics across all three scenarios ---------------------------
        self._advance(result, SKUState.ECONOMIC_ANALYSIS)
        best_offer = min(offers, key=lambda o: o.unit_cost_inr)

        for scenario in config.scenarios:
            deltas = ScenarioDeltas.from_payload(
                scenario, self.store.latest(f"scenario_profile:{scenario}").payload
            )
            landed = compute_landed_cost(
                best_offer,
                scenario=scenario,
                supplier_cost_delta_pct=deltas.supplier_cost_delta_pct,
                expected_monthly_velocity=money(
                    config.expected_monthly_velocity * deltas.velocity_multiplier
                ),
            )
            scoped = pin(
                self.store,
                marketplace=config.marketplace,
                category=product.category,
                risk_tier=str(risk.tier),
                scenario=str(scenario),
            )
            economics = compute_economics(
                EconomicsInputs(
                    selling_price_inr=config.selling_price_inr,
                    category=product.category,
                    hsn_code=product.hsn_code or "",
                    weight_g=product.weight_g or 500,
                    ad_allowance_pct=config.ad_allowance_pct,
                ),
                landed,
                scoped,
                deltas,
            )
            result.landed_costs[scenario] = landed
            result.economics[scenario] = economics
            result.warnings.extend(
                w for w in economics.warnings if w not in result.warnings
            )

        gate = apply_profitability_gate(result.economics, references)
        result.gate = gate
        if self._record(result, decide_profitability(gate)):
            return result

        # --- signals and the market reality check ---------------------------
        self._advance(result, SKUState.SIGNAL_COLLECTION)
        expected = result.economics[ScenarioName.EXPECTED]

        if price_reference is None:
            result.warnings.append(
                "no MarketPriceReference: economics cannot be tested against the market"
            )
            self._record(
                result,
                Decision(
                    rule_id="headroom.missing_reference",
                    disposition=Disposition.REQUIRES_HUMAN_REVIEW,
                    reason=(
                        "a MarketPriceReference is required before a SKU can be "
                        "considered viable; collect 3-5 comparables"
                    ),
                    next_state=SKUState.BLOCKED_AWAITING_DATA,
                ),
            )
            return result

        self._advance(result, SKUState.PRICE_HEADROOM_CHECK)
        threshold = references.threshold_profile.payload
        headroom = check_price_headroom(
            min_profitable_price_inr=expected.min_profitable_price_inr,
            reference=price_reference,
            min_headroom_pct=money(threshold.get("min_price_headroom_pct", 10)),
            marginal_headroom_pct=money(threshold.get("marginal_price_headroom_pct", 0)),
        )
        result.headroom = headroom
        if self._record(result, decide_headroom(headroom)):
            return result

        self._advance(result, SKUState.SUPPLIER_ANALYSIS)
        self._advance(result, SKUState.RISK_ANALYSIS)

        # --- compliance overrides everything --------------------------------
        self._advance(result, SKUState.COMPLIANCE_GATE)
        ruleset = self.store.get(
            config.compliance_ruleset_kind, config.compliance_ruleset_version
        )
        # A Product's HSN reaches this point only from an ingested supplier
        # catalogue or a recorded observation, both of which are sourced
        # evidence. Callers holding an inferred code must say so explicitly.
        compliance = evaluate_compliance(
            ruleset=ruleset,
            hsn_code=product.hsn_code,
            product_text=f"{product.canonical_name} {product.category}",
            evidence=compliance_evidence or {},
            hsn_provenance=hsn_provenance,
        )
        result.compliance = compliance

        # --- score, with coverage carried alongside -------------------------
        components = build_components(
            net_margin_pct=expected.net_margin_pct,
            return_rate_pct=benchmark_rate,
            signals=signals or {},
            compliance_clear=compliance.overall.value == "PASS",
            operational_complexity=risk.operational_complexity,
        )
        score = compute_opportunity_score(
            components, min_coverage_pct=money(threshold.get("min_coverage_pct", 60))
        )
        result.score = score

        if self._record(result, decide_compliance(compliance)):
            return result
        if self._record(result, decide_coverage(score)):
            return result

        weakest = references.weakest_state()
        if self._record(
            result, decide_reference_trust(weakest, references.unverified_docs())
        ):
            return result

        # --- Claude review ---------------------------------------------------
        self._advance(result, SKUState.AI_REVIEW)
        if len(offers) > 1:
            identity = self.identity_judge.invoke(
                build_identity_prompt(
                    [
                        {"offer_id": str(o.id), "cost": str(o.unit_cost_inr)}
                        for o in offers
                    ]
                ),
                input_refs=[str(o.id) for o in offers],
            )
            result.llm_outcomes.append(identity)
            if identity.usable:
                for decision in interpret_findings(identity.findings, stage="identity"):
                    if self._record(result, decision):
                        return result

        commercial = self.commercial_judge.invoke(
            build_commercial_prompt(
                {
                    "product": product.canonical_name,
                    "net_margin_pct": str(expected.net_margin_pct),
                    "risk_tier": str(risk.tier),
                    "compliance": str(compliance.overall),
                    "headroom_pct": str(headroom.headroom_pct),
                }
            ),
            input_refs=[sku_code],
        )
        result.llm_outcomes.append(commercial)
        if commercial.usable:
            for decision in interpret_findings(commercial.findings, stage="commercial"):
                if self._record(result, decision):
                    return result
        # A failed Judge is missing evidence, not a clean bill of health.
        elif self._record(
            result,
            Decision(
                rule_id="ai_review.unavailable",
                disposition=Disposition.REQUIRES_HUMAN_REVIEW,
                reason=(
                    "commercial review produced no usable findings "
                    f"({commercial.error or commercial.schema_violations})"
                ),
                next_state=SKUState.BLOCKED_PENDING_REVIEW,
            ),
        ):
            return result

        # --- listing draft, then hand to a human ----------------------------
        draft = self.writer.invoke(
            build_writer_prompt(
                {
                    "product": product.canonical_name,
                    "brand": product.brand,
                    "category": product.category,
                    "weight_g": product.weight_g,
                    "warranty_months": product.warranty_months,
                }
            ),
            input_refs=[sku_code],
        )
        result.llm_outcomes.append(draft)
        if draft.usable:
            result.listing_draft = draft.parsed
        else:
            result.warnings.append(
                "listing draft unavailable; a human must write the content"
            )

        self._advance(result, SKUState.APPROVAL_REQUIRED)
        result.decisions.append(
            Decision(
                rule_id="approval.required",
                disposition=Disposition.APPROVAL_REQUIRED,
                reason="every automated gate passed; a human decides whether to list",
                next_state=SKUState.APPROVAL_REQUIRED,
            )
        )
        return result

    # -- internals --------------------------------------------------------

    def _benchmark(self, category: str) -> ReferenceDoc | None:
        for kind in (f"return_benchmark:{category}", "return_benchmark:default"):
            try:
                return self.store.latest(kind)
            except Exception:  # noqa: BLE001 - absence is an expected outcome
                continue
        return None

    def _advance(self, result: PipelineResult, to_state: SKUState) -> None:
        require_legal(result.final_state, to_state)
        result.final_state = to_state

    def _record(self, result: PipelineResult, decision: Decision) -> bool:
        """Record a decision; return True if it stops the run."""
        result.decisions.append(decision)
        if decision.disposition is Disposition.CONTINUE:
            return False
        if decision.next_state:
            require_legal(result.final_state, decision.next_state)
            result.final_state = decision.next_state
        logger.info(
            "pipeline halted",
            extra={"rule": decision.rule_id, "state": str(result.final_state)},
        )
        return True


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def rejection_report(results: list[PipelineResult]) -> str:
    """The primary report: what failed, and what would have to change.

    Sorted by how close each product came, so the ones worth renegotiating
    appear first.
    """
    failures = [r for r in results if r.blocked_by]
    if not failures:
        return "No products failed.\n"

    lines = [
        f"FAILED / BLOCKED - {len(failures)} of {len(results)} products",
        "",
        f"{'Product':<30} {'Failed gate':<34} {'Gap to pass'}",
        f"{'-' * 30} {'-' * 34} {'-' * 40}",
    ]
    for result in sorted(failures, key=_closeness):
        decision = result.blocked_by
        gap = ""
        if result.gate and result.gate.gap_to_pass:
            gap = result.gate.gap_to_pass.get("action", "")
        if not gap:
            gap = decision.reason
        lines.append(
            f"{result.product_name[:29]:<30} {decision.rule_id[:33]:<34} {gap[:60]}"
        )
    lines.append("")
    return "\n".join(lines)


def _closeness(result: PipelineResult) -> Decimal:
    """How near a failure came to passing; smaller is nearer."""
    expected = result.expected
    if expected is None:
        return Decimal("9999")
    return -expected.net_margin_pct


def ranked_report(results: list[PipelineResult]) -> str:
    """Survivors, ranked. Secondary to the rejection report by design."""
    survivors = [r for r in results if r.blocked_by is None]
    if not survivors:
        return "No products cleared every gate.\n"

    lines = [
        f"AWAITING APPROVAL - {len(survivors)} of {len(results)} products",
        "",
        f"{'Product':<28} {'Score':<26} {'Margin':>8} {'Risk':<12} {'Compliance'}",
        f"{'-' * 28} {'-' * 26} {'-' * 8} {'-' * 12} {'-' * 22}",
    ]
    for result in sorted(
        survivors,
        key=lambda r: r.score.value if r.score and r.score.value else ZERO,
        reverse=True,
    ):
        expected = result.expected
        # Score is always rendered with its coverage; a bare number would
        # invite comparison between differently-evidenced products.
        score = result.score.display() if result.score else "no score"
        lines.append(
            f"{result.product_name[:27]:<28} {score:<26} "
            f"{expected.net_margin_pct:>7.1f}% "
            f"{str(result.risk.tier)[:11]:<12} "
            f"{str(result.compliance.overall) if result.compliance else '-'}"
        )
    lines.append("")
    return "\n".join(lines)


def detail_report(result: PipelineResult) -> str:
    """Full itemisation for one product, including every scenario."""
    lines = [
        f"{result.product_name}  [{result.sku_code}]",
        "=" * 74,
        f"Final state : {result.final_state}",
        f"Risk tier   : {result.risk.tier if result.risk else '-'}",
        f"Score       : {result.score.display() if result.score else '-'}",
        f"Compliance  : {result.compliance.overall if result.compliance else '-'}",
        "",
        "SCENARIOS",
        f"  {'':<10} {'landed':>12} {'net':>12} {'margin':>9} {'min price':>13}",
    ]
    for scenario in (ScenarioName.BEST, ScenarioName.EXPECTED, ScenarioName.WORST):
        economics = result.economics.get(scenario)
        if economics is None:
            continue
        lines.append(
            f"  {scenario!s:<10} {format_inr(economics.landed_cost_inr):>13} "
            f"{format_inr(economics.net_profit_inr):>13} "
            f"{economics.net_margin_pct:>8.1f}% "
            f"{format_inr(economics.min_profitable_price_inr):>14}"
        )

    if result.headroom:
        lines += ["", "MARKET", f"  {result.headroom.explanation}"]

    lines += ["", "DECISIONS"]
    for decision in result.decisions:
        lines.append(f"  {decision.disposition:<24} {decision.rule_id}")
        lines.append(f"  {'':<24} {decision.reason[:70]}")

    if result.warnings:
        lines += ["", "WARNINGS"]
        lines.extend(f"  - {w}" for w in dict.fromkeys(result.warnings))
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PIPELINE_VERSION",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "detail_report",
    "ranked_report",
    "rejection_report",
]
