"""Research assistant tests: the model must not be able to invent a fact.

Almost every assertion here is about refusal. A language model asked to find
promising products will happily supply prices, review counts and supplier
names, and every one of them would be indistinguishable from a measurement
once written to a CSV. These tests are the thing standing between a plausible
sentence and a purchase decision.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from catalog.enums import AIService
from llm.services import LLMService, NullLLMClient, RecordedLLMClient
from research.assistant import (
    FACTUAL_FIELDS,
    Claim,
    ParsedRequest,
    Provenance,
    ResearchAssistant,
    ResearchError,
)
from research.export import (
    OBSERVATION_COLUMNS,
    observation_rows,
    preview,
    trends_rows,
    write_observations,
)
from sourcing.model import DiscoveredProduct

FIXTURES = Path("data/fixtures/discovery")


def responder(payload: dict) -> LLMService:
    return LLMService(
        AIService.RESEARCHER, RecordedLLMClient({"RESEARCHER": json.dumps(payload)})
    )


def good_payload(**overrides):
    candidate = {
        "name": "Rechargeable Hair Trimmer",
        "category": "personal_care_electrical",
        "rationale": "grooming is a steady repeat category",
        "search_terms": ["trimmer", "clipper"],
        "confidence": "medium",
    }
    candidate.update(overrides)
    return {"candidates": [candidate]}


# ==========================================================================
# Hallucination prevention
# ==========================================================================


@pytest.mark.parametrize(
    "field",
    [
        "price",
        "unit_cost_inr",
        "moq",
        "gst_rate",
        "review_count",
        "rank",
        "search_interest",
        "supplier_name",
        "competitor_count",
        "gtin",
    ],
)
def test_model_cannot_supply_a_factual_field(field: str):
    """Every one of these would become evidence downstream if accepted."""
    outcome = responder(good_payload(**{field: 499})).invoke("x")
    assert not outcome.usable
    assert outcome.schema_violations
    assert outcome.findings == []


def test_a_rejected_response_yields_no_candidates_from_the_model():
    """A schema violation is missing evidence, never a partial acceptance."""
    assistant = ResearchAssistant(researcher=responder(good_payload(price=499)))
    run = assistant.run("hair trimmers under 500")
    assert run.status == "DEGRADED"
    assert not run.ai_available
    # Anything present came from the fallback, not from the rejected response.
    for candidate in run.candidates:
        assert candidate.name != "Rechargeable Hair Trimmer"


def test_claim_refuses_ai_provenance_on_a_factual_field():
    """The type system enforces it, not just the prompt."""
    for field in ("observed_price_inr", "unit_cost_inr", "moq_units", "supplier_name"):
        with pytest.raises(ValueError, match="may not supply a factual field"):
            Claim(field=field, value=Decimal("499"), provenance=Provenance.AI_INFERENCE)


def test_every_factual_field_is_protected():
    """The guard list must cover everything an engine treats as evidence."""
    for field in FACTUAL_FIELDS:
        with pytest.raises(ValueError):
            Claim(field=field, value="x", provenance=Provenance.AI_INFERENCE)


def test_sourced_claim_must_name_its_source():
    """A fact without a reference is indistinguishable from a guess."""
    with pytest.raises(ValueError, match="must name its source"):
        Claim(
            field="observed_price_inr",
            value=Decimal("499"),
            provenance=Provenance.SOURCED,
        )


def test_unavailable_claim_cannot_carry_a_value():
    with pytest.raises(ValueError, match="cannot carry a value"):
        Claim(
            field="observed_price_inr",
            value=Decimal("1"),
            provenance=Provenance.UNAVAILABLE,
        )


def test_unexpected_fields_are_rejected_even_if_harmless():
    """An allowlist, not a blocklist: new fabrications are refused by default."""
    outcome = responder(good_payload(estimated_demand="high")).invoke("x")
    assert not outcome.usable


# ==========================================================================
# Provenance
# ==========================================================================


def test_every_claim_carries_a_provenance():
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    assert run.candidates
    for candidate in run.candidates:
        assert candidate.claims
        for claim in candidate.claims:
            assert claim.provenance in set(Provenance)


def test_sourced_claims_carry_a_reference_and_a_date():
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    sourced = [c for c in run.candidates[0].claims if c.provenance is Provenance.SOURCED]
    assert sourced, "the fixture should supply at least one fact"
    for claim in sourced:
        assert claim.reference
        assert claim.value is not None


def test_candidates_are_labelled_as_ai_suggestions():
    """A reader must never have to infer where a candidate came from."""
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    body = run.candidates[0].as_dict()
    assert body["is_ai_suggestion"] is True
    assert body["facts_are_ai_generated"] is False


def test_run_states_the_separation_explicitly():
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    assert "UNAVAILABLE" in run.as_dict()["disclaimer"]
    assert "generated by a model" in run.as_dict()["disclaimer"]


# ==========================================================================
# Unavailable data
# ==========================================================================


def test_unmatched_candidate_has_every_fact_unavailable():
    """The correct outcome for most Indian marketplaces, not a failure."""
    run = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    ).run("quantum widgets")
    candidate = run.candidates[0]
    assert candidate.sourced_fields == []
    assert candidate.unavailable_fields
    assert candidate.evidence_ratio == 0


def test_supplier_facts_never_come_from_research():
    """An acquisition cost has exactly one legitimate origin."""
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    for field in ("supplier_name", "unit_cost_inr", "moq_units", "gst_rate_pct"):
        claim = run.candidates[0].claim(field)
        assert claim.provenance is Provenance.UNAVAILABLE
        assert "catalogue" in claim.note


def test_unavailable_sources_are_reported_not_hidden():
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    assert any(not f.available for f in run.feed_results)
    assert any("no legitimate route" in w for w in run.warnings)


# ==========================================================================
# Export: the enforcement point
# ==========================================================================


def test_export_writes_only_sourced_facts():
    """The rule that stops a suggestion becoming a measurement.

    Uses a candidate that *does* match a source, so it is exported, and checks
    the per-field rule: only fields the source actually supplied carry a
    value. The separate candidate-level rule — that a candidate with no
    sourced claim at all is not exported — is covered below.
    """
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    row = observation_rows(run)[0]
    assert row["title"] == "Rechargeable Hair Trimmer"
    # The fixture supplies no brand, model or GTIN for this product, so those
    # columns must stay blank rather than being filled from the suggestion.
    for column in ("brand", "model", "gtin"):
        assert row[column] == "", f"{column} must be blank when no source supplied it"


def test_ai_only_candidate_is_not_written_as_an_observation():
    """A candidate nothing corroborated is a hypothesis, not an observation.

    Writing a bare title would file it into the observations fixture as though
    someone had seen it, and the ingester would read it back as a real product
    with every field UNKNOWN.
    """
    run = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    ).run("quantum widgets")

    assert run.candidates, "the candidate must still exist"
    assert not run.candidates[0].has_any_evidence
    assert observation_rows(run) == []


def test_ai_only_candidate_remains_visible_in_the_research_result():
    """Not exported is not the same as discarded.

    The operator still sees the suggestion, its rationale and every
    UNAVAILABLE claim — that is what tells them what to go and look for.
    """
    run = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    ).run("quantum widgets")

    body = run.as_dict()
    assert len(body["candidates"]) == 1
    candidate = body["candidates"][0]
    assert candidate["name"] == "Quantum Widget XZ9"
    assert candidate["is_ai_suggestion"] is True
    assert candidate["provenance_summary"]["unavailable"], "its gaps stay visible"
    assert candidate["provenance_summary"]["evidence_pct"] == "0.0"


def test_ai_only_candidate_is_not_persisted_to_disk(tmp_path: Path):
    """write_observations must not create a row, or a file, for it."""
    run = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    ).run("quantum widgets")

    path = tmp_path / "observations.csv"
    assert write_observations(run, path) == 0
    assert not path.exists(), "nothing to write means no file is created"


def test_ai_only_candidate_does_not_reach_the_ingester(tmp_path: Path):
    """The end-to-end consequence: it never becomes a discovered product."""
    from sourcing.feeds import CsvObservationFeed

    run = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    ).run("quantum widgets")

    path = tmp_path / "observations.csv"
    path.write_text(",".join(OBSERVATION_COLUMNS) + "\n", encoding="utf-8")
    write_observations(run, path)

    titles = [p.raw_title for p in CsvObservationFeed(path).fetch().products]
    assert "Quantum Widget XZ9" not in titles


def test_a_mixed_run_exports_only_the_evidenced_candidates():
    """One sourced candidate and one hypothesis: exactly one row."""
    payload = {
        "candidates": [
            {
                "name": "Rechargeable Hair Trimmer",
                "category": "personal_care_electrical",
                "rationale": "matches a recorded observation",
                "search_terms": ["trimmer"],
                "confidence": "medium",
            },
            {
                "name": "Quantum Widget XZ9",
                "category": "unknown",
                "rationale": "nothing corroborates this",
                "search_terms": ["quantumwidget"],
                "confidence": "low",
            },
        ]
    }
    run = ResearchAssistant(researcher=responder(payload)).run("mixed request")
    assert len(run.candidates) == 2

    rows = observation_rows(run)
    assert len(rows) == 1
    assert rows[0]["title"] == "Rechargeable Hair Trimmer"


def test_a_candidate_with_a_single_sourced_claim_is_exported():
    """The bar is one sourced claim, not a full record.

    Partial evidence is still evidence; the empty columns already say what is
    unknown.
    """
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    candidate = run.candidates[0]
    assert len(candidate.sourced_fields) >= 1
    assert len(observation_rows(run)) == 1


def test_export_carries_a_real_price_when_a_source_has_one():
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    row = observation_rows(run)[0]
    assert row["price"], "a sourced price should be exported"
    assert row["reference"].startswith(("manual://", "research://", "http"))


def test_export_matches_the_existing_fixture_schema():
    """The file written must be the file the ingester already reads."""
    header = (FIXTURES / "observations.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(OBSERVATION_COLUMNS)


def test_exported_rows_survive_the_existing_ingester(tmp_path: Path):
    from sourcing.feeds import CsvObservationFeed

    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    path = tmp_path / "observations.csv"
    assert write_observations(run, path) == 1
    products = CsvObservationFeed(path).fetch().products
    assert products
    assert products[0].raw_title == "Rechargeable Hair Trimmer"


def test_suppliers_csv_is_never_written_from_research():
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    body = preview(run)["suppliers.csv"]
    assert "not written" in body
    assert "ingested catalogue" in body


def test_trends_export_only_where_interest_was_sourced():
    run = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    ).run("quantum widgets")
    assert trends_rows(run) == []


# ==========================================================================
# AI failure and degradation
# ==========================================================================


def test_no_model_configured_degrades_rather_than_failing():
    """Part 1 works without an LLM; research must too."""
    run = ResearchAssistant(
        researcher=LLMService(AIService.RESEARCHER, NullLLMClient())
    ).run("hair trimmers and lunchboxes")
    assert run.status == "DEGRADED"
    assert not run.ai_available
    assert any("no research model is configured" in w for w in run.warnings)


def test_malformed_model_output_degrades_safely():
    service = LLMService(
        AIService.RESEARCHER, RecordedLLMClient({"RESEARCHER": "not json"})
    )
    run = ResearchAssistant(researcher=service).run("hair trimmers")
    assert run.status == "DEGRADED"
    assert run.candidates or run.warnings


def test_transport_failure_degrades_safely():
    service = LLMService(
        AIService.RESEARCHER, RecordedLLMClient({}, fail_on={"RESEARCHER"})
    )
    run = ResearchAssistant(researcher=service).run("hair trimmers")
    assert run.status == "DEGRADED"
    assert run.as_dict()["ai_error"]


def test_empty_result_is_explained_not_silent():
    """An empty list must not read as 'no promising products exist'."""
    run = ResearchAssistant(
        researcher=LLMService(AIService.RESEARCHER, NullLLMClient())
    ).run("find promising products under 500")
    assert run.candidates == []
    assert any("could not search" in w for w in run.warnings)


def test_empty_request_is_refused():
    with pytest.raises(ResearchError, match="cannot be empty"):
        ResearchAssistant().run("   ")


# ==========================================================================
# Request parsing is deterministic
# ==========================================================================


def test_constraints_are_parsed_deterministically():
    """A misread budget would silently change what is considered."""
    parsed = ParsedRequest.parse("Find 20 promising products under ₹500 supplier cost")
    assert parsed.limit == 20
    assert parsed.max_supplier_cost_inr == Decimal("500")


def test_limit_is_bounded():
    assert ParsedRequest.parse("find 900 products").limit == 50
    assert ParsedRequest.parse("no numbers here").limit == 20


# ==========================================================================
# Handoff to the existing pipeline
# ==========================================================================


def test_candidate_converts_to_the_existing_discovery_model():
    assistant = ResearchAssistant(researcher=responder(good_payload()))
    run = assistant.run("hair trimmers")
    product = assistant.to_discovered(run.candidates[0])
    assert isinstance(product, DiscoveredProduct)
    assert product.source == "ai_research"
    assert product.raw_title == "Rechargeable Hair Trimmer"


def test_unsourced_candidate_enters_the_pipeline_with_no_evidence():
    """An AI suggestion must not gain coverage it did not earn."""
    assistant = ResearchAssistant(
        researcher=responder(
            good_payload(name="Quantum Widget XZ9", search_terms=["quantumwidget"])
        )
    )
    run = assistant.run("quantum widgets")
    product = assistant.to_discovered(run.candidates[0])
    assert product.coverage_pct == 0
    assert not product.has_price


def test_research_output_flows_through_the_whole_pipeline(tmp_path: Path):
    """Research → discovery → matching → economics → compliance → opportunity.

    The verdict comes from the deterministic engines. Research decides nothing.
    """
    import shutil

    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.feeds import (
        build_registry,
        collect,
        load_supplier_candidates,
    )
    from sourcing.matching import group_products
    from sourcing.opportunities import Verdict, evaluate_group

    workdir = tmp_path / "discovery"
    shutil.copytree(FIXTURES, workdir)

    run = ResearchAssistant(
        researcher=responder(good_payload()), fixture_dir=workdir
    ).run("hair trimmers")
    assert write_observations(run, workdir / "observations.csv") == 1

    products, _feeds = collect(build_registry(fixture_dir=workdir))
    assert any(p.source == "manual_research" for p in products)

    groups = group_products(products)
    candidates = load_supplier_candidates(workdir / "suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())

    verdicts = {
        evaluate_group(group, candidates, pipeline=pipeline).verdict for group in groups
    }
    assert verdicts <= set(Verdict)
    # The engines decided; research only supplied candidates.
    assert verdicts & {Verdict.REJECTED, Verdict.BACKUP, Verdict.INSUFFICIENT_DATA}


# ==========================================================================
# Evidence attribution: facts belong to one product, never to a category
# ==========================================================================


def _led_run(*names: str):
    """A run of several LED candidates against the LED strip light fixture."""
    payload = {
        "candidates": [
            {
                "name": name,
                "category": "home_lighting",
                "rationale": "lighting is a steady category",
                "search_terms": [],
                "confidence": "low",
            }
            for name in names
        ]
    }
    return ResearchAssistant(researcher=responder(payload)).run("led products")


def test_one_sourced_led_product_does_not_populate_other_led_candidates():
    """The reported bug, pinned.

    `data/fixtures/discovery/observations.csv` carries exactly one LED
    observation -- an LED Strip Light at manual://obs/7. Sharing the token
    "led" previously copied its price, HSN, weight, trend, search interest,
    competitor count and brand dominance onto every other LED suggestion.
    "led" is a category, not an identity.
    """
    run = _led_run(
        "LED Fairy Lights",
        "LED Ceiling Panel",
        "LED Bulb 9W",
        "LED Desk Lamp",
        "LED Mirror Light",
        "LED Strip Light",
    )

    by_name = {c.name: c in run.candidates and c for c in run.candidates}
    strip = by_name["LED Strip Light"]
    others = [c for c in run.candidates if c.name != "LED Strip Light"]

    # The candidate that genuinely is the observed product keeps its facts.
    assert strip.has_any_evidence
    assert strip.claim("observed_price_inr").value is not None

    # Nothing else inherits any of them.
    for candidate in others:
        assert candidate.sourced_fields == [], (
            f"{candidate.name} inherited facts it has no source for"
        )
        for field in (
            "observed_price_inr",
            "hsn_code",
            "weight_g",
            "trend",
            "search_interest",
            "competitor_count",
            "brand_dominance",
        ):
            claim = candidate.claim(field)
            if claim is not None:
                assert claim.value is None, f"{candidate.name}.{field} was inferred"


def test_a_source_reference_is_never_reused_across_candidates():
    """Two candidates citing one reference is the signature of the bug."""
    run = _led_run("LED Fairy Lights", "LED Ceiling Panel", "LED Strip Light")

    references = [
        claim.reference
        for candidate in run.candidates
        for claim in candidate.claims
        if claim.provenance is Provenance.SOURCED and claim.reference
    ]
    assert len(references) == len(set(references)) or len(set(references)) == 1, (
        "a reference may back one candidate's claims, never several candidates'"
    )
    citing = {
        candidate.name
        for candidate in run.candidates
        for claim in candidate.claims
        if claim.provenance is Provenance.SOURCED and claim.reference == "manual://obs/7"
    }
    assert citing == {"LED Strip Light"}


def test_only_the_matching_candidate_is_exported():
    """The export consequence: one observation, one row."""
    run = _led_run("LED Fairy Lights", "LED Bulb 9W", "LED Strip Light")
    rows = observation_rows(run)
    assert [row["title"] for row in rows] == ["LED Strip Light"]
    assert {row["reference"] for row in rows} == {"manual://obs/7"}


def test_a_single_shared_token_is_never_enough():
    """ "led" alone must not attach anything, however high the containment."""
    run = _led_run("LED")
    candidate = run.candidates[0]
    # Containment is 1.0 -- every token the candidate has is in the
    # observation -- but one token is a category, not a product.
    assert candidate.sourced_fields == []
    assert observation_rows(run) == []


def test_no_single_source_backs_more_than_one_candidate():
    """Exclusivity, stated precisely.

    Two candidates may both hold evidence when there are genuinely two
    distinct sources — the fixtures carry an LED observation and a separate
    Trends row. What must never happen is one source backing several
    candidates, which is the signature of the fan-out bug.
    """
    run = _led_run("LED Strip Light", "LED Strip Light 5m", "LED Fairy Lights")

    backers: dict[str, set[str]] = {}
    for candidate in run.candidates:
        for claim in candidate.claims:
            if claim.provenance is Provenance.SOURCED and claim.reference:
                backers.setdefault(claim.reference, set()).add(candidate.name)

    for reference, names in backers.items():
        assert len(names) == 1, (
            f"{reference} was attributed to {sorted(names)}; a source describes "
            "one product and may back only one candidate"
        )


def test_a_generic_candidate_still_matches_its_specific_observation():
    """The fix must not break legitimate attribution.

    "Rechargeable Hair Trimmer" is genuinely supported by the observed
    "Kemei KX-880 Professional Hair Trimmer": every identifying token the
    candidate has appears in the observation. Containment rather than Jaccard
    is what keeps this working -- the observation's extra tokens are detail,
    not disagreement.
    """
    run = ResearchAssistant(researcher=responder(good_payload())).run("hair trimmers")
    candidate = run.candidates[0]
    assert candidate.has_any_evidence
    assert candidate.claim("observed_price_inr").value is not None
    assert candidate.claim("observed_price_inr").reference == "manual://obs/1"


def test_attribution_is_deterministic():
    """The same inputs must always produce the same attribution."""
    first = _led_run("LED Fairy Lights", "LED Strip Light", "LED Bulb 9W")
    second = _led_run("LED Fairy Lights", "LED Strip Light", "LED Bulb 9W")
    assert [c.sourced_fields for c in first.candidates] == [
        c.sourced_fields for c in second.candidates
    ]


def test_unrelated_candidate_gets_nothing_from_any_observation():
    run = _led_run("Quantum Widget XZ9")
    assert run.candidates[0].sourced_fields == []
    assert observation_rows(run) == []


def test_supplier_claims_stay_unavailable_regardless_of_attribution():
    """Supplier data is untouched by this change and must remain so."""
    run = _led_run("LED Strip Light")
    candidate = run.candidates[0]
    assert candidate.has_any_evidence
    for field in ("supplier_name", "unit_cost_inr", "moq_units", "gst_rate_pct"):
        claim = candidate.claim(field)
        assert claim.provenance is Provenance.UNAVAILABLE
        assert claim.value is None
