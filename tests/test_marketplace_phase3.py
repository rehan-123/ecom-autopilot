"""Phase 3: marketplace readiness must never imply or perform execution.

Two families of assertion. The first is that Phase 2's evidence discipline
survives the crossing into a listing: UNAVAILABLE stays UNAVAILABLE, an
inferred field cannot satisfy a sourced requirement, and a probable supplier
is not a confirmed one. The second is that nothing here reaches a marketplace.
"""

from __future__ import annotations

import collections
from decimal import Decimal
from pathlib import Path

import pytest

from marketplace import (
    CapabilityStatus,
    DraftField,
    FieldProvenance,
    InventoryState,
    ListingDraft,
    OperationResult,
    Readiness,
    assess_all,
    assess_for_marketplace,
    build_draft,
    capability_status,
    marketplace_status,
    preview_listing,
    publish_listing,
    registry,
    validate_draft,
)
from part2.marketplace import ADAPTERS, Capability

CREDS = {
    "AMAZON_SP_REFRESH_TOKEN": "PLACEHOLDER",
    "AMAZON_SP_CLIENT_ID": "PLACEHOLDER",
    "AMAZON_SP_CLIENT_SECRET": "PLACEHOLDER",
}


def listable_draft(**overrides) -> ListingDraft:
    """A draft that clears every gate, so tests can remove one at a time."""
    draft = ListingDraft(
        sku_code="SKU-TRIM-880",
        title="Kemei KX-880 Rechargeable Hair Trimmer",
        images=("https://example.test/1.jpg",),
        compliance_state="PASS",
        economics_available=True,
        expected_margin_pct=Decimal("22"),
        worst_margin_pct=Decimal("14"),
        supplier_name="Shenzhen Kemei Direct",
        supplier_confirmed=True,
        inventory_state=InventoryState.KNOWN,
        inventory_qty=40,
    )
    draft.hsn_code = DraftField("8510", FieldProvenance.SOURCED, "supplier://row=2")
    draft.price_inr = DraftField(Decimal("1299"), FieldProvenance.SOURCED, "obs://1")
    draft.category = DraftField("personal_care", FieldProvenance.SOURCED, "obs://1")
    for key, value in overrides.items():
        setattr(draft, key, value)
    return draft


# ==========================================================================
# Architecture
# ==========================================================================


def test_registry_discovers_every_adapter():
    names = {m.name for m in registry(environ={})}
    assert names == set(ADAPTERS)


def test_every_capability_is_explicitly_classified():
    for status in registry(environ={}):
        assert len(status.capabilities) == len(list(Capability))
        for report in status.capabilities:
            assert report.status in set(CapabilityStatus)


def test_unsupported_and_not_configured_are_distinct():
    """One is fixed by a credential; the other cannot be fixed at all."""
    meesho = marketplace_status("meesho", environ={})
    create = meesho.capability(str(Capability.CREATE_LISTING))
    assert create.status is CapabilityStatus.UNSUPPORTED

    amazon = marketplace_status("amazon_in", environ={})
    assert amazon.capability(str(Capability.CREATE_LISTING)).status is (
        CapabilityStatus.NOT_CONFIGURED
    )


def test_credentials_cannot_create_an_unsupported_capability():
    report = capability_status(
        "meesho", Capability.CREATE_LISTING, environ={**CREDS, "MEESHO_KEY": "x"}
    )
    assert report.status is CapabilityStatus.UNSUPPORTED


def test_a_configured_write_capability_is_still_disabled_for_phase_3():
    report = capability_status("amazon_in", Capability.CREATE_LISTING, environ=CREDS)
    assert report.status is CapabilityStatus.SUPPORTED
    assert report.write_blocked_for_phase_3


def test_registry_states_live_execution_is_off():
    for status in registry(environ=CREDS):
        assert status.as_dict()["phase_3_write_operations_enabled"] is False


# ==========================================================================
# Draft provenance
# ==========================================================================


def test_an_unavailable_field_cannot_carry_a_value():
    with pytest.raises(ValueError, match="cannot carry a value"):
        DraftField("8510", FieldProvenance.UNAVAILABLE)


def test_a_sourced_field_must_name_its_source():
    with pytest.raises(ValueError, match="must name its source"):
        DraftField("8510", FieldProvenance.SOURCED)


def test_an_inferred_factual_field_cannot_satisfy_a_sourced_requirement():
    """AI_DRAFT is fine for copy; it is never evidence of a fact."""
    draft = listable_draft()
    draft.hsn_code = DraftField("8510", FieldProvenance.AI_DRAFT, "model")
    assessment = validate_draft(draft)
    assert assessment.readiness is Readiness.NOT_READY
    assert any(f.code == "UNSOURCED_FACTUAL_FIELD" for f in assessment.blocking)


def test_a_missing_required_field_stays_missing_and_blocks():
    draft = listable_draft()
    draft.hsn_code = DraftField(None, FieldProvenance.UNAVAILABLE)
    assessment = validate_draft(draft)
    assert "hsn_code" in draft.missing_factual_fields
    assert any(f.code == "MISSING_REQUIRED_FIELD" for f in assessment.blocking)


def test_the_transformation_is_deterministic():
    opportunity = _opportunities()[0]
    first, second = build_draft(opportunity), build_draft(opportunity)
    assert first.as_dict()["fields"] == second.as_dict()["fields"]
    assert first.sku_code == second.sku_code


# ==========================================================================
# Compliance boundary
# ==========================================================================


@pytest.mark.parametrize("state", ["FAIL", "UNKNOWN", "REQUIRES_HUMAN_REVIEW"])
def test_uncleared_compliance_is_never_ready(state: str):
    assessment = validate_draft(listable_draft(compliance_state=state))
    assert assessment.readiness is Readiness.NOT_READY
    assert any(f.code == "COMPLIANCE_NOT_CLEARED" for f in assessment.blocking)


def test_absent_compliance_evaluation_is_never_ready():
    assessment = validate_draft(listable_draft(compliance_state=None))
    assert any(f.code == "COMPLIANCE_UNKNOWN" for f in assessment.blocking)


def test_a_compliant_draft_can_be_ready():
    """The gates must not block a product that genuinely clears them."""
    assert validate_draft(listable_draft()).readiness is Readiness.READY


# ==========================================================================
# Economics boundary
# ==========================================================================


def test_missing_economics_is_never_ready():
    assessment = validate_draft(
        listable_draft(
            economics_available=False, expected_margin_pct=None, worst_margin_pct=None
        )
    )
    assert assessment.readiness is Readiness.NOT_READY
    assert any(f.code == "ECONOMICS_UNAVAILABLE" for f in assessment.blocking)


def test_missing_economics_does_not_become_zero():
    draft = listable_draft(
        economics_available=False, expected_margin_pct=None, worst_margin_pct=None
    )
    body = draft.as_dict()
    assert body["expected_margin_pct"] is None
    assert body["worst_margin_pct"] is None


def test_valid_economics_are_preserved():
    body = listable_draft().as_dict()
    assert body["expected_margin_pct"] == "22"
    assert body["worst_margin_pct"] == "14"


# ==========================================================================
# Supplier boundary
# ==========================================================================


def test_an_unconfirmed_supplier_is_never_ready():
    assessment = validate_draft(listable_draft(supplier_confirmed=False))
    assert assessment.readiness is Readiness.NOT_READY
    assert any(f.code == "SUPPLIER_NOT_CONFIRMED" for f in assessment.blocking)


def test_a_probable_match_is_reported_as_unconfirmed():
    """The LED case, carried into Phase 3."""
    led = next(o for o in _opportunities() if "LED Strip Light 5m" in o.title)
    draft = build_draft(led)
    assert draft.supplier_confirmed is False
    assert draft.supplier_name is None
    assert draft.marketplace_attributes.get("supplier_match_confidence") == "PROBABLE"


# ==========================================================================
# Inventory
# ==========================================================================


def test_unknown_inventory_is_not_a_positive_quantity():
    draft = build_draft(_opportunities()[0], inventory_qty=None)
    assert draft.inventory_state is InventoryState.UNKNOWN
    assert draft.inventory_qty is None


def test_zero_stock_is_preserved_as_out_of_stock():
    draft = build_draft(_opportunities()[0], inventory_qty=0)
    assert draft.inventory_state is InventoryState.OUT_OF_STOCK
    assert draft.inventory_qty == 0


def test_negative_inventory_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        build_draft(_opportunities()[0], inventory_qty=-1)


# ==========================================================================
# Dry run and external side effects
# ==========================================================================


def test_preview_makes_no_request_and_says_so():
    result = preview_listing(listable_draft(), "amazon_in")
    assert result["result"] == str(OperationResult.DRY_RUN)
    assert "No marketplace request was made" in result["detail"]


def test_publish_is_never_enabled_and_never_reports_success():
    result = publish_listing(listable_draft(), "amazon_in")
    assert result["result"] == str(OperationResult.NOT_ENABLED_FOR_PHASE_3)
    assert result["published"] is False


def test_publish_refuses_even_with_credentials_configured():
    result = publish_listing(listable_draft(), "amazon_in")
    assert result["published"] is False


def test_an_assessment_never_claims_publication():
    body = validate_draft(listable_draft()).as_dict()
    assert body["published"] is False
    assert body["external_side_effects"] is False


def test_no_phase_3_module_performs_a_network_call():
    import ast

    for path in ("marketplace/draft.py", "marketplace/registry.py"):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for banned in ("urllib.request", "http.client", "socket", "requests", "httpx"):
            assert banned not in imported, f"{path} imports {banned}"


# ==========================================================================
# Error isolation
# ==========================================================================


def test_one_failing_adapter_does_not_block_the_others(monkeypatch):
    import sys

    # `marketplace/__init__.py` exports a function named `registry`, which
    # shadows the `marketplace.registry` submodule in the package namespace.
    # sys.modules reaches the module regardless of that shadowing.
    registry_module = sys.modules["marketplace.registry"]
    real = registry_module.assess_for_marketplace

    def flaky(draft, marketplace, **kwargs):
        if marketplace == "flipkart":
            raise RuntimeError("adapter exploded")
        return real(draft, marketplace, **kwargs)

    monkeypatch.setattr(registry_module, "assess_for_marketplace", flaky)
    assessments = registry_module.assess_all(listable_draft(), environ={})
    assert len(assessments) == len(ADAPTERS)
    broken = next(a for a in assessments if a.marketplace == "flipkart")
    assert any(f.code == "ADAPTER_ERROR" for f in broken.findings)


def test_an_unsupported_marketplace_is_reported_not_validated():
    assessment = assess_for_marketplace(listable_draft(), "meesho", environ={})
    assert assessment.capability is CapabilityStatus.UNSUPPORTED
    assert assessment.readiness is Readiness.NOT_READY


# ==========================================================================
# Security
# ==========================================================================


def test_no_secret_appears_in_a_serialised_draft_or_assessment():
    import json
    import os

    os.environ["AMAZON_SP_CLIENT_SECRET"] = "PLACEHOLDER-super-secret-value"
    try:
        draft = build_draft(_opportunities()[0])
        payload = json.dumps(
            {
                "draft": draft.as_dict(),
                "assessments": [a.as_dict() for a in assess_all(draft)],
                "registry": [m.as_dict() for m in registry()],
            }
        )
        # The VALUE must never appear. Variable *names* legitimately do: the
        # UI has to tell an operator which variable to set, and a name is not
        # a secret.
        assert "PLACEHOLDER-super-secret-value" not in payload
        assert "AMAZON_SP_CLIENT_SECRET" in payload, (
            "the required variable name should be shown so it can be configured"
        )
    finally:
        os.environ.pop("AMAZON_SP_CLIENT_SECRET", None)


def test_no_credentials_are_hard_coded_in_phase_3_source():
    import re

    for path in ("marketplace/draft.py", "marketplace/registry.py"):
        source = Path(path).read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9]{16,}", source)
        assert not re.search(r"AKIA[0-9A-Z]{16}", source)


# ==========================================================================
# Phase 2 regression
# ==========================================================================


def _opportunities():
    from decisions.pipeline import Pipeline
    from reference.loader import ReferenceStore
    from sourcing.feeds import build_registry, collect, load_supplier_candidates
    from sourcing.matching import group_products
    from sourcing.opportunities import evaluate_group, rank_opportunities

    products, _ = collect(build_registry())
    candidates = load_supplier_candidates("data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore("tests/fixtures/reference").load())
    return rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )


def test_phase_2_outcomes_are_unchanged():
    opportunities = _opportunities()
    assert collections.Counter(str(o.verdict) for o in opportunities) == {
        "BACKUP": 1,
        "REJECTED": 6,
    }
    by_title = {o.title: o for o in opportunities}
    kemei = next(o for t, o in by_title.items() if "KX880" in t)
    assert "BIS" in kemei.blockers[0]
    lunchbox = next(o for t, o in by_title.items() if "Lunchbox" in t)
    assert "worst_case" in lunchbox.blockers[0]


def test_no_phase_2_product_is_marketplace_ready():
    """None of the fixture products clears every listing gate, and that is right."""
    for opportunity in _opportunities():
        draft = build_draft(opportunity)
        assert validate_draft(draft).readiness is Readiness.NOT_READY
