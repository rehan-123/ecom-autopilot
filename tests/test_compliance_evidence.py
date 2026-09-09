"""Phase 2D: missing compliance evidence must never become a positive result.

The hole this closes: a product with no HSN code received PASS, because no
HSN-keyed rule could match and "nothing matched" was read as "nothing
applies". HSN determines which requirements govern a product at all, so
without one, absence of a matching rule proves nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from catalog.enums import ComplianceResult
from compliance.engine import HsnProvenance, detect_conflicts
from compliance.engine import evaluate as evaluate_compliance
from reference.loader import ReferenceDoc


def ruleset(*, verified: bool = True) -> ReferenceDoc:
    """A minimal verified ruleset with one blocking BIS requirement."""
    return ReferenceDoc(
        kind="compliance_ruleset:india",
        version="test-1",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        effective_to=None,
        source="test",
        last_verified_at=datetime.now(UTC) if verified else None,
        payload={
            "requirements": [
                {
                    "id": "BIS_ISI_SMALL_APPLIANCE",
                    "requirement": "BIS registration",
                    "severity": "BLOCKING",
                    "hsn_prefix": ["8510"],
                    "keywords": ["trimmer"],
                    "evidence_required": ["bis_certificate_no"],
                }
            ]
        },
        path=Path("test"),
    )


def evaluate(**kwargs):
    defaults = {
        "ruleset": ruleset(),
        "product_text": "Stainless Steel Water Bottle",
        "evidence": {},
    }
    defaults.update(kwargs)
    return evaluate_compliance(**defaults)


# ==========================================================================
# HSN is what decides which rules apply
# ==========================================================================


def test_a_product_without_an_hsn_cannot_pass():
    """The reported hole. No classification evidence is not clearance."""
    outcome = evaluate(hsn_code=None, hsn_provenance=HsnProvenance.UNAVAILABLE)
    assert outcome.overall is ComplianceResult.UNKNOWN
    assert outcome.overall is not ComplianceResult.PASS
    assert any("without a sourced HSN" in n for n in outcome.notes)


def test_an_empty_hsn_string_is_treated_as_absent():
    outcome = evaluate(hsn_code="", hsn_provenance=HsnProvenance.UNAVAILABLE)
    assert outcome.overall is ComplianceResult.UNKNOWN


def test_an_ai_inferred_hsn_cannot_establish_compliance():
    """A model guessing an HSN would route a regulated product around its rules."""
    outcome = evaluate(hsn_code="7323", hsn_provenance=HsnProvenance.AI_INFERENCE)
    assert outcome.overall is ComplianceResult.UNKNOWN
    assert any("not evidence" in n for n in outcome.notes)


def test_an_ai_inferred_hsn_cannot_dodge_a_blocking_rule():
    """The dangerous direction: inferring an unregulated code for a regulated item."""
    outcome = evaluate(
        hsn_code="7323",
        product_text="Kemei KX-880 Hair Trimmer",
        hsn_provenance=HsnProvenance.AI_INFERENCE,
    )
    assert outcome.overall is not ComplianceResult.PASS


def test_a_sourced_hsn_outside_every_rule_still_passes():
    """The fix must not block products that are genuinely unregulated."""
    outcome = evaluate(hsn_code="7323", hsn_provenance=HsnProvenance.SOURCED)
    assert outcome.overall is ComplianceResult.PASS


def test_provenance_is_evidence_only_when_sourced():
    assert HsnProvenance.SOURCED.is_evidence
    assert not HsnProvenance.AI_INFERENCE.is_evidence
    assert not HsnProvenance.UNAVAILABLE.is_evidence


# ==========================================================================
# Regulatory evidence
# ==========================================================================


def test_missing_blocking_evidence_fails_not_passes():
    """Existing Kemei/BIS behaviour, pinned."""
    outcome = evaluate(
        hsn_code="8510",
        product_text="Kemei KX-880 Hair Trimmer",
        hsn_provenance=HsnProvenance.SOURCED,
    )
    assert outcome.overall is ComplianceResult.FAIL
    assert "BIS_ISI_SMALL_APPLIANCE" in outcome.blocking_failures


def test_supplied_evidence_clears_the_requirement():
    outcome = evaluate(
        hsn_code="8510",
        product_text="Kemei KX-880 Hair Trimmer",
        evidence={"bis_certificate_no": "R-41001234"},
        hsn_provenance=HsnProvenance.SOURCED,
    )
    assert outcome.overall is ComplianceResult.PASS


@pytest.mark.parametrize("empty", [None, "", [], {}])
def test_an_empty_evidence_value_is_not_evidence(empty):
    """A present-but-empty field must not satisfy a requirement."""
    outcome = evaluate(
        hsn_code="8510",
        product_text="Kemei KX-880 Hair Trimmer",
        evidence={"bis_certificate_no": empty},
        hsn_provenance=HsnProvenance.SOURCED,
    )
    assert outcome.overall is ComplianceResult.FAIL


def test_a_keyword_only_match_is_a_question_not_a_clearance():
    outcome = evaluate(
        hsn_code="7323",
        product_text="cordless trimmer accessory",
        hsn_provenance=HsnProvenance.SOURCED,
    )
    assert outcome.overall is not ComplianceResult.PASS


def test_a_stale_ruleset_cannot_clear_anything():
    """A rulebook that may omit requirements proves nothing about them."""
    outcome = evaluate(
        ruleset=ruleset(verified=False),
        hsn_code="8510",
        product_text="Kemei KX-880 Hair Trimmer",
        evidence={"bis_certificate_no": "R-41001234"},
        hsn_provenance=HsnProvenance.SOURCED,
    )
    assert outcome.overall is not ComplianceResult.PASS


# ==========================================================================
# Conflicting evidence
# ==========================================================================


def test_conflicting_evidence_is_not_resolved_silently():
    outcome = evaluate(
        hsn_code="8510",
        product_text="Kemei KX-880 Hair Trimmer",
        evidence={"bis_certificate_no": ["R-41001234", "R-99999999"]},
        hsn_provenance=HsnProvenance.SOURCED,
    )
    assert outcome.overall is not ComplianceResult.PASS
    assert outcome.conflicts
    assert "bis_certificate_no" in outcome.conflicts[0]


def test_agreeing_sources_are_not_a_conflict():
    assert detect_conflicts({"bis_certificate_no": ["R-1", "R-1"]}) == []
    assert detect_conflicts({"bis_certificate_no": "R-1"}) == []
    assert detect_conflicts({}) == []


def test_a_conflict_is_surfaced_with_both_values():
    conflicts = detect_conflicts({"bis_certificate_no": ["R-1", "R-2"]})
    assert len(conflicts) == 1
    assert "r-1" in conflicts[0] and "r-2" in conflicts[0]


# ==========================================================================
# Pipeline safety and unchanged outcomes
# ==========================================================================


def test_missing_compliance_evidence_does_not_crash_the_pipeline():
    import collections

    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import evaluate_group, rank_opportunities

    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())
    opportunities = rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )
    assert opportunities
    tally = collections.Counter(str(o.verdict) for o in opportunities)
    assert tally == {"BACKUP": 1, "REJECTED": 6}


def test_existing_kemei_bis_rejection_is_unchanged():
    import collections

    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import evaluate_group, rank_opportunities

    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())
    opportunities = rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )
    by_title = {o.title: o for o in opportunities}

    kemei = next(o for t, o in by_title.items() if "KX880" in t)
    assert "BIS" in kemei.blockers[0]
    lunchbox = next(o for t, o in by_title.items() if "Lunchbox" in t)
    assert "worst_case" in lunchbox.blockers[0]
    led = next(o for t, o in by_title.items() if "LED Strip Light 5m" in t)
    assert str(led.verdict) == "BACKUP"
    assert collections.Counter(str(o.verdict) for o in opportunities)["REJECTED"] == 6


def test_no_compliance_code_defaults_to_a_positive_status():
    """Guards the whole class: no `.get(..., True/PASS/COMPLIANT)` anywhere."""
    source = Path("compliance/engine.py").read_text(encoding="utf-8")
    for pattern in ('.get("', ", True)", ', "PASS")', ', "COMPLIANT")'):
        if pattern in (", True)", ', "PASS")', ', "COMPLIANT")'):
            assert pattern not in source, (
                f"compliance code contains {pattern!r}: a missing value must "
                "never default to a positive status"
            )
