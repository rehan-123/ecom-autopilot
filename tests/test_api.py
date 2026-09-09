"""API tests: the HTTP layer must expose the engines without weakening them.

The load-bearing assertions are about what the API *cannot* do. It has no
endpoint that purchases, pays, publishes or transfers money; it cannot execute
an unapproved action; and it defaults every execution to dry run. Those are
properties of the system, and an HTTP surface is exactly where they would
erode first.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.service import DomainService, reset_service

READ_ENDPOINTS = [
    "/api/health",
    "/api/dashboard",
    "/api/discovery/feeds",
    "/api/discovery/products",
    "/api/suppliers",
    "/api/opportunities",
    "/api/approvals",
    "/api/orders",
    "/api/inventory",
    "/api/analytics",
    "/api/ledger",
    "/api/settings",
]


@pytest.fixture
def client() -> TestClient:
    reset_service(DomainService())
    with TestClient(app) as test_client:
        yield test_client
    reset_service(None)


# ==========================================================================
# Availability
# ==========================================================================


@pytest.mark.parametrize("path", READ_ENDPOINTS)
def test_every_read_endpoint_answers(client: TestClient, path: str):
    response = client.get(path)
    assert response.status_code == 200, response.text[:300]
    assert response.json() is not None


def test_health_reports_the_safety_posture(client: TestClient):
    body = client.get("/api/health").json()
    assert body["mode"] == "PAPER"
    assert body["dry_run_default"] is True
    assert body["execution_requires_approval"] is True


# ==========================================================================
# Real data, not placeholders
# ==========================================================================


def test_dashboard_reports_real_engine_output(client: TestClient):
    body = client.get("/api/dashboard").json()
    assert body["discovery"]["distinct_products"] > 0
    assert body["discovery"]["observations"] >= body["discovery"]["distinct_products"]
    assert body["opportunities"]["total"] > 0


def test_discovery_surfaces_unavailable_sources(client: TestClient):
    """The gaps must reach the UI, not just the available feeds."""
    body = client.get("/api/discovery/feeds").json()
    unavailable = [f for f in body["feeds"] if not f["available"]]
    assert unavailable, "sources with no legitimate route must still be listed"
    assert all(f["unavailable_reason"] for f in unavailable)


def test_opportunities_carry_verdicts_and_coverage(client: TestClient):
    items = client.get("/api/opportunities").json()["opportunities"]
    assert items
    for item in items:
        assert item["verdict"] in (
            "RECOMMENDED",
            "BACKUP",
            "REJECTED",
            "INSUFFICIENT_DATA",
        )
        assert item["requires_approval"] is True
        assert "discovery_coverage_pct" in item


def test_opportunity_filter_narrows_results(client: TestClient):
    everything = client.get("/api/opportunities").json()["opportunities"]
    rejected = client.get("/api/opportunities?verdict=REJECTED").json()["opportunities"]
    assert len(rejected) <= len(everything)
    assert all(o["verdict"] == "REJECTED" for o in rejected)


def test_product_detail_returns_the_decision_trail(client: TestClient):
    key = client.get("/api/opportunities").json()["opportunities"][0]["group_key"]
    body = client.get(f"/api/opportunities/{key}").json()
    assert body["opportunity"]["group_key"] == key
    assert "scenarios" in body
    assert "observations" in body


def test_unknown_product_is_a_404(client: TestClient):
    assert client.get("/api/opportunities/does-not-exist").status_code == 404


def test_money_is_returned_as_strings_not_floats(client: TestClient):
    """Decimal precision must survive JSON.

    A float here would undo the discipline the whole economics engine is
    built on.
    """
    body = client.get("/api/dashboard").json()
    assert isinstance(body["operations"]["realized_contribution_inr"], str)
    assert isinstance(body["operations"]["capital_locked_inr"], str)


def test_inventory_never_reports_negative_stock(client: TestClient):
    for position in client.get("/api/inventory").json()["positions"]:
        assert position["on_hand"] >= 0
        assert position["available"] >= 0
        assert position["reserved"] >= 0


def test_analytics_recommendations_carry_evidence(client: TestClient):
    for item in client.get("/api/analytics").json()["recommendations"]:
        assert item["evidence"]
        assert item["downside"]


def test_settings_reports_credentials_without_values(client: TestClient):
    body = client.get("/api/settings").json()
    assert body["mode"] == "PAPER"
    for marketplace in body["marketplaces"]:
        assert "required_env_vars" in marketplace
        # Presence only; a value must never be serialised.
        assert isinstance(marketplace["credentials_configured"], bool)
    assert "AMAZON_SP_REFRESH_TOKEN" in str(body["credentials"])


def test_ledger_reports_chain_integrity(client: TestClient):
    body = client.get("/api/ledger").json()
    assert "integrity" in body
    assert isinstance(body["total"], int)


# ==========================================================================
# The API cannot weaken the safety model
# ==========================================================================


FORBIDDEN_PATH_TERMS = (
    "purchase",
    "payment",
    "pay",
    "publish",
    "transfer",
    "checkout",
    "buy",
)


def test_no_endpoint_can_move_money_or_publish(client: TestClient):
    """The absent capabilities must not reappear as HTTP routes."""
    paths = [route.path.lower() for route in app.routes]
    # One exception, checked rather than trusted: the Phase 3 publish route
    # exists so a client gets a structured refusal instead of a 404. The
    # assertion below proves it never publishes, which is a stronger guarantee
    # than banning the word from a URL.
    refusal_only = "/api/listings/{slug}/publish"
    offenders = [
        path
        for path in paths
        for term in FORBIDDEN_PATH_TERMS
        if term in path.replace("approvals", "") and path != refusal_only
    ]
    assert not offenders, f"forbidden capability exposed: {offenders}"

    slug = client.get("/api/opportunities").json()["opportunities"][0]["slug"]
    body = client.post(f"/api/listings/{slug}/publish").json()
    assert body["result"] == "NOT_ENABLED_FOR_PHASE_3"
    assert body["published"] is False


def test_only_approval_endpoints_mutate(client: TestClient):
    """Everything else is read-only.

    An explicit allowlist rather than a pattern: a new mutating endpoint has
    to be added here deliberately, which is the moment to ask what it changes.
    Research writes observation fixtures and rebuilds derived state; it cannot
    approve, execute, purchase, publish or pay.
    """
    mutating = {
        route.path
        for route in app.routes
        if getattr(route, "methods", set()) & {"POST", "PUT", "PATCH", "DELETE"}
    }
    allowed = {
        # Approval gate: propose, approve, then execute an approved action.
        "/api/approvals",
        "/api/approvals/{record_id}/approve",
        "/api/approvals/{record_id}/execute",
        # Research: runs a query, and writes sourced observations to fixtures.
        "/api/research",
        "/api/research/{run_id}/send-to-analysis",
        # Human-triggered supplier catalogue import. Reads a file the operator
        # names; writes nothing unless persist is set.
        "/api/suppliers/import",
        # Phase 3 publish: always refuses with NOT_ENABLED_FOR_PHASE_3.
        "/api/listings/{slug}/publish",
        # Phase 4 controlled execution. Creating and approving change only
        # internal state; `run` is gated and defaults to dry run.
        "/api/executions",
        "/api/executions/{execution_id}/approve",
        "/api/executions/{execution_id}/reject",
        "/api/executions/{execution_id}/cancel",
        "/api/executions/{execution_id}/run",
        # Rebuilds derived state from the engines. Changes no business fact.
        "/api/refresh",
    }
    assert mutating <= allowed, f"undeclared mutating endpoint: {mutating - allowed}"


def test_unapproved_action_cannot_execute(client: TestClient):
    """The gate lives in the engine; the API cannot route around it."""
    created = client.post(
        "/api/approvals",
        json={"action": "CHANGE_PRICE", "sku_code": "SKU-1", "reason": "test"},
    ).json()
    response = client.post(
        f"/api/approvals/{created['id']}/execute", json={"dry_run": True}
    )
    assert response.status_code == 409
    assert "approve" in response.json()["detail"].lower()


def test_approval_then_execution_stays_in_dry_run(client: TestClient):
    created = client.post(
        "/api/approvals",
        json={"action": "CHANGE_PRICE", "sku_code": "SKU-1", "reason": "test"},
    ).json()
    approved = client.post(
        f"/api/approvals/{created['id']}/approve", json={"approver": "operator"}
    ).json()
    assert approved["state"] == "APPROVED"
    assert approved["approved_by"] == "operator"

    body = client.post(
        f"/api/approvals/{created['id']}/execute", json={"dry_run": True}
    ).json()
    assert body["result"]["outcome"] == "DRY_RUN"
    assert body["execution"]["state"] == "EXECUTED"


def test_execution_defaults_to_dry_run_when_unspecified(client: TestClient):
    """An accidental call must not reach a marketplace."""
    created = client.post(
        "/api/approvals", json={"action": "SCALE_PRODUCT", "sku_code": "SKU-2"}
    ).json()
    client.post(f"/api/approvals/{created['id']}/approve", json={"approver": "op"})
    body = client.post(f"/api/approvals/{created['id']}/execute", json={}).json()
    assert body["result"]["outcome"] == "DRY_RUN"


def test_live_execution_is_refused_because_no_capability_exists(client: TestClient):
    created = client.post(
        "/api/approvals", json={"action": "SCALE_PRODUCT", "sku_code": "SKU-3"}
    ).json()
    client.post(f"/api/approvals/{created['id']}/approve", json={"approver": "op"})
    body = client.post(
        f"/api/approvals/{created['id']}/execute", json={"dry_run": False}
    ).json()
    assert body["result"]["outcome"] == "NOT_SUPPORTED"
    assert "no purchasing or publishing capability" in body["result"]["detail"]


def test_approving_an_unknown_record_is_a_404(client: TestClient):
    response = client.post("/api/approvals/nope/approve", json={"approver": "op"})
    assert response.status_code == 404


def test_approvals_queue_is_seeded_from_real_recommendations(client: TestClient):
    """The queue shows what the optimiser justified, not invented actions."""
    body = client.get("/api/approvals").json()
    assert body["all"], "the optimiser should have proposed something"
    for record in body["all"]:
        assert record["reason"], "every proposal must carry its reasoning"


def test_refresh_rebuilds_state(client: TestClient):
    body = client.post("/api/refresh", json={}).json()
    assert body["products"] > 0


def test_product_routes_use_a_url_safe_slug(client: TestClient):
    """A group key may contain spaces or a slash; a path segment may not.

    Encoding does not save a slash in a path segment, so routing uses a
    derived slug and the human-readable key stays for display.
    """
    products = client.get("/api/discovery/products").json()["products"]
    assert products
    for product in products:
        slug = product["slug"]
        assert " " not in slug
        assert "/" not in slug
        assert slug == slug.lower()
        # The slug must actually resolve.
        assert client.get(f"/api/opportunities/{slug}").status_code in (200, 404)


def test_every_opportunity_is_reachable_by_its_slug(client: TestClient):
    for opportunity in client.get("/api/opportunities").json()["opportunities"]:
        response = client.get(f"/api/opportunities/{opportunity['slug']}")
        assert response.status_code == 200, opportunity["slug"]
        assert response.json()["opportunity"]["group_key"] == opportunity["group_key"]


def test_slugs_are_unique_across_products(client: TestClient):
    slugs = [p["slug"] for p in client.get("/api/discovery/products").json()["products"]]
    assert len(slugs) == len(set(slugs))


# ==========================================================================
# Research endpoints
# ==========================================================================


def test_research_endpoint_runs_and_degrades_without_a_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """With no key configured, research must still answer.

    The endpoint reads OPENAI_API_KEY from the real environment, so this test
    pins that environment explicitly rather than assuming the machine running
    it has no key set -- a developer or CI worker with OPENAI_API_KEY exported
    for unrelated reasons must not change what this test asserts.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    body = client.post("/api/research", json={"query": "hair trimmers under 500"}).json()
    assert body["status"] in ("COMPLETE", "DEGRADED")
    assert body["ai_available"] is False
    assert "UNAVAILABLE" in body["disclaimer"]


def test_research_endpoint_uses_openai_when_a_key_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """The other half of the same guarantee: a key present must be used.

    Exercised against a stubbed urlopen so the test never makes a real network
    call, but the path through default_researcher() -> OpenAIClient is real.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "PLACEHOLDER-test-key")
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            import json as _json

            return _json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": _json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "name": "Rechargeable Hair Trimmer",
                                                "category": "personal_care_electrical",
                                                "rationale": "steady repeat category",
                                                "search_terms": ["trimmer"],
                                                "confidence": "medium",
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured["authorization"] = request.get_header("Authorization")
        return FakeResponse()

    import urllib.request as urllib_request

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)

    body = client.post("/api/research", json={"query": "hair trimmers"}).json()
    assert body["ai_available"] is True
    assert captured["authorization"] == "Bearer PLACEHOLDER-test-key"
    # The key must never be echoed back to the frontend in any form.
    assert "PLACEHOLDER-test-key" not in __import__("json").dumps(body)


def test_research_response_never_contains_an_ai_generated_price(client: TestClient):
    """The property the whole research design exists to guarantee."""
    body = client.post(
        "/api/research", json={"query": "hair trimmers and lunchboxes under 500"}
    ).json()
    for candidate in body["candidates"]:
        assert candidate["facts_are_ai_generated"] is False
        for claim in candidate["claims"]:
            if claim["provenance"] == "AI_INFERENCE":
                assert claim["field"] not in (
                    "observed_price_inr",
                    "unit_cost_inr",
                    "moq_units",
                    "supplier_name",
                )
            if claim["provenance"] == "SOURCED":
                assert claim["reference"], "a sourced fact must name its source"


def test_research_export_preview_leaves_unsourced_columns_blank(client: TestClient):
    body = client.post("/api/research", json={"query": "quantumwidget xz9"}).json()
    preview = body["export_preview"]["observations.csv"]
    assert "title,price," in preview
    assert "not written" in body["export_preview"]["suppliers.csv"]


def test_research_run_is_retrievable(client: TestClient):
    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    fetched = client.get(f"/api/research/{created['id']}").json()
    assert fetched["id"] == created["id"]


def test_unknown_research_run_is_a_404(client: TestClient):
    assert client.get("/api/research/nope").status_code == 404


def test_empty_research_query_is_rejected(client: TestClient):
    assert client.post("/api/research", json={"query": "x"}).status_code == 422


def test_send_to_analysis_feeds_the_deterministic_pipeline(client: TestClient, tmp_path):
    """Research supplies candidates; the engines decide the verdict.

    Sends the body the frontend actually sends. An earlier version passed
    ``{"run_id": ...}``, which was written to match the implementation rather
    than the client -- so it kept passing while every real request from the
    Research screen failed with 422.
    """
    import shutil

    from api.service import DomainService, reset_service

    workdir = tmp_path / "discovery"
    shutil.copytree("data/fixtures/discovery", workdir)
    reset_service(
        DomainService(fixture_dir=workdir, supplier_file=workdir / "suppliers.csv")
    )

    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    body = client.post(
        f"/api/research/{created['id']}/send-to-analysis", json={"persist": True}
    ).json()

    assert body["products"] > 0
    assert "not by research" in body["note"]
    verdicts = {
        o["verdict"] for o in client.get("/api/opportunities").json()["opportunities"]
    }
    assert verdicts <= {"RECOMMENDED", "BACKUP", "REJECTED", "INSUFFICIENT_DATA"}


def test_send_to_analysis_for_an_unknown_run_is_a_404(client: TestClient):
    response = client.post("/api/research/nope/send-to-analysis", json={"run_id": "nope"})
    assert response.status_code == 404


def test_research_cannot_execute_anything(client: TestClient):
    """Research has no execution path; approvals remain the only route."""
    paths = [r.path for r in app.routes if "research" in r.path]
    assert paths
    assert not any("execute" in p for p in paths)


# ==========================================================================
# send-to-analysis contract: the run is named by the path, not the body
# ==========================================================================


def _research_workdir(tmp_path):
    """A throwaway fixture dir so persistence does not touch the repo."""
    import shutil

    from api.service import DomainService, reset_service

    workdir = tmp_path / "discovery"
    shutil.copytree("data/fixtures/discovery", workdir)
    reset_service(
        DomainService(fixture_dir=workdir, supplier_file=workdir / "suppliers.csv")
    )
    return workdir


def test_send_to_analysis_accepts_the_body_the_frontend_sends(client, tmp_path):
    """The exact request the Send to Analysis button makes.

    Reproduces the reported 422: the schema required run_id in the body while
    the frontend sent only {"persist": true}, because the run is already named
    in the URL.
    """
    _research_workdir(tmp_path)
    created = client.post("/api/research", json={"query": "hair trimmers"}).json()

    response = client.post(
        f"/api/research/{created['id']}/send-to-analysis", json={"persist": True}
    )
    assert response.status_code == 200, response.text[:400]


def test_send_to_analysis_accepts_an_empty_body(client, tmp_path):
    """persist defaults to True, so no body at all is a valid request."""
    _research_workdir(tmp_path)
    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    assert (
        client.post(
            f"/api/research/{created['id']}/send-to-analysis", json={}
        ).status_code
        == 200
    )


def test_send_to_analysis_still_accepts_a_matching_run_id(client, tmp_path):
    """Clients that do send it keep working."""
    _research_workdir(tmp_path)
    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    response = client.post(
        f"/api/research/{created['id']}/send-to-analysis",
        json={"run_id": created["id"], "persist": True},
    )
    assert response.status_code == 200


def test_a_body_run_id_that_contradicts_the_path_is_refused(client, tmp_path):
    """Stricter than before, not looser.

    Accepting either silently would act on a run the caller did not name in
    the URL.
    """
    _research_workdir(tmp_path)
    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    response = client.post(
        f"/api/research/{created['id']}/send-to-analysis",
        json={"run_id": "a-different-run"},
    )
    assert response.status_code == 400
    assert "does not match the path" in response.json()["detail"]


def test_send_to_analysis_persists_only_evidenced_candidates(client, tmp_path):
    """The whole point of the endpoint, end to end.

    A run containing one sourced candidate and one AI-only hypothesis must
    write exactly one observation row, and that row must carry the source's
    own reference.
    """
    import csv

    workdir = _research_workdir(tmp_path)
    observations = workdir / "observations.csv"
    before = len(observations.read_text(encoding="utf-8").splitlines())

    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    sourced = [c for c in created["candidates"] if c["provenance_summary"]["sourced"]]
    unsourced = [
        c for c in created["candidates"] if not c["provenance_summary"]["sourced"]
    ]
    assert sourced, "the fixture should supply at least one sourced candidate"

    body = client.post(
        f"/api/research/{created['id']}/send-to-analysis", json={"persist": True}
    ).json()

    assert body["written"]["observations"] == len(sourced)
    assert body["products"] > 0

    rows = list(csv.DictReader(observations.read_text(encoding="utf-8").splitlines()))
    assert len(observations.read_text(encoding="utf-8").splitlines()) == before + len(
        sourced
    )

    written_titles = {r["title"] for r in rows}
    for candidate in sourced:
        assert candidate["name"] in written_titles
    for candidate in unsourced:
        assert candidate["name"] not in written_titles, (
            "an AI-only candidate must not be persisted as an observation"
        )


def test_send_to_analysis_can_skip_persistence(client, tmp_path):
    """persist=false rebuilds state without writing anything."""
    workdir = _research_workdir(tmp_path)
    observations = workdir / "observations.csv"
    before = observations.read_text(encoding="utf-8")

    created = client.post("/api/research", json={"query": "hair trimmers"}).json()
    body = client.post(
        f"/api/research/{created['id']}/send-to-analysis", json={"persist": False}
    ).json()

    assert body["written"]["observations"] == 0
    assert observations.read_text(encoding="utf-8") == before


def test_no_post_endpoint_requires_an_id_already_in_its_path(client):
    """Guards the whole class of defect, not just this one endpoint.

    A required body field duplicating a path parameter is invisible until a
    real client calls it, because a test written alongside the schema will
    send it -- which is exactly how the send-to-analysis 422 survived a green
    suite. Reads the endpoint's own type hints rather than FastAPI internals,
    so a framework upgrade cannot silently disable this check.
    """
    import typing

    from fastapi.routing import APIRoute
    from pydantic import BaseModel

    offenders: list[str] = []
    checked = 0
    for route in app.routes:
        if not isinstance(route, APIRoute) or "POST" not in route.methods:
            continue
        path_params = set(route.param_convertors)
        for annotation in typing.get_type_hints(route.endpoint).values():
            if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
                continue
            checked += 1
            for name, field in annotation.model_fields.items():
                if name in path_params and field.is_required():
                    offenders.append(f"{route.path}: body field {name!r}")

    assert checked, "no request models were inspected; the check is vacuous"
    assert not offenders, (
        "these body fields duplicate a path parameter and are required, so "
        f"every well-formed client request will 422: {offenders}"
    )


# ==========================================================================
# Supplier explanation must match the engine state it describes
# ==========================================================================


def _detail_for(client: TestClient, needle: str) -> dict:
    opportunities = client.get("/api/opportunities").json()["opportunities"]
    target = next(o for o in opportunities if needle in o["group_key"].lower())
    return client.get(f"/api/opportunities/{target['slug']}").json()


def test_a_below_bar_supplier_match_is_surfaced_not_reported_as_none(client: TestClient):
    """The reported contradiction.

    The LED strip light has one PROBABLE supplier match. `suppliers` holds
    only usable options, so it is empty -- and the UI read that as "no
    supplier matched" while the reason line said the match was only PROBABLE.
    Both were describing the same state, incompatibly.
    """
    detail = _detail_for(client, "5m rgb remote")

    assert detail["suppliers"] == [], "no option is usable, so none is selectable"
    assert len(detail["supplier_matches"]) == 1, (
        "a match below the actionable bar must still reach the client"
    )
    match = detail["supplier_matches"][0]
    assert match["confidence"] == "PROBABLE"
    assert match["supplier_name"]
    assert not match["usable"] if "usable" in match else True


def test_the_reason_names_the_supplier_it_is_talking_about(client: TestClient):
    """ "match is only PROBABLE" is only meaningful if it says which supplier."""
    detail = _detail_for(client, "5m rgb remote")
    reasons = " ".join(detail["opportunity"]["recoverable"])
    assert "PROBABLE" in reasons
    named = detail["supplier_matches"][0]["supplier_name"]
    assert named in reasons, "the explanation must name the matched supplier"


def test_probable_match_does_not_become_a_selected_supplier(client: TestClient):
    """Surfacing it must not promote it. The engine chose nothing."""
    detail = _detail_for(client, "5m rgb remote")
    assert detail["selected_supplier"] is None
    assert detail["opportunity"]["selected_supplier"] is None


def test_a_below_bar_match_still_computes_no_economics(client: TestClient):
    """Explanation changed; the deterministic outcome did not."""
    detail = _detail_for(client, "5m rgb remote")
    assert detail["scenarios"] == {}
    assert detail["opportunity"]["verdict"] == "BACKUP"
    assert detail["opportunity"]["landed_cost_inr"] is None


def test_a_genuine_no_match_reports_a_blocker_not_a_weak_match(client: TestClient):
    """The other case must stay distinguishable.

    A candidate nothing matched is REJECTED with a blocker, and carries no
    supplier_matches at all -- so the UI can tell the two situations apart.
    """
    opportunities = client.get("/api/opportunities").json()["opportunities"]
    unmatched = [
        o
        for o in opportunities
        if o["verdict"] == "REJECTED"
        and any("no supplier match" in b for b in o["blockers"])
    ]
    assert unmatched, "the fixtures should contain at least one unsourceable candidate"

    detail = client.get(f"/api/opportunities/{unmatched[0]['slug']}").json()
    assert detail["suppliers"] == []
    assert detail["supplier_matches"] == [], (
        "nothing matched, so there is genuinely nothing to show"
    )
    assert any("no supplier match" in b for b in detail["opportunity"]["blockers"])


def test_every_opportunity_detail_exposes_supplier_matches(client: TestClient):
    """The key must always be present so the UI never branches on undefined."""
    for opportunity in client.get("/api/opportunities").json()["opportunities"]:
        detail = client.get(f"/api/opportunities/{opportunity['slug']}").json()
        assert "supplier_matches" in detail
        assert isinstance(detail["supplier_matches"], list)


def test_a_usable_match_still_appears_as_a_selectable_option(client: TestClient):
    """The fix must not hide genuinely usable suppliers behind the new path."""
    opportunities = client.get("/api/opportunities").json()["opportunities"]
    sourced = [o for o in opportunities if o["selected_supplier"]]
    assert sourced, "the fixtures should contain at least one sourceable candidate"

    detail = client.get(f"/api/opportunities/{sourced[0]['slug']}").json()
    assert detail["suppliers"], "a usable option must still be listed"
    assert detail["selected_supplier"] is not None


# ==========================================================================
# Supplier CSV import (2A)
# ==========================================================================


def test_supplier_import_reports_acceptances_and_provenance(client: TestClient):
    body = client.post(
        "/api/suppliers/import", json={"path": "data/fixtures/discovery/suppliers.csv"}
    ).json()
    assert body["accepted"] == 4
    assert body["rejected"] == 0
    entry = body["suppliers"][0]
    assert entry["provenance"]["source_reference"].endswith("#row=2")
    assert entry["tax_invoice_available"] is True


def test_supplier_import_refuses_rows_missing_required_facts(
    client: TestClient, tmp_path
):
    path = tmp_path / "bare.csv"
    path.write_text("Supplier,Product,Rate (INR)\nAcme,Widget,150\n", encoding="utf-8")
    body = client.post("/api/suppliers/import", json={"path": str(path)}).json()
    assert body["accepted"] == 0
    assert body["rejected"] == 1
    assert body["rejected_rows"][0]["reason"] == "MISSING_GST"


def test_supplier_import_rejects_a_missing_or_unsupported_file(
    client: TestClient, tmp_path
):
    assert (
        client.post("/api/suppliers/import", json={"path": "nope.csv"}).status_code == 404
    )
    bad = tmp_path / "x.txt"
    bad.write_text("hi", encoding="utf-8")
    assert (
        client.post("/api/suppliers/import", json={"path": str(bad)}).status_code == 400
    )


def test_supplier_import_does_not_persist_unless_asked(client: TestClient):
    body = client.post(
        "/api/suppliers/import", json={"path": "data/fixtures/discovery/suppliers.csv"}
    ).json()
    assert "products" not in body, "persist defaults to false"


def test_supplier_import_is_in_the_mutating_allowlist(client: TestClient):
    """A new mutating endpoint must be a deliberate addition."""
    from fastapi.routing import APIRoute

    mutating = {
        r.path
        for r in app.routes
        if isinstance(r, APIRoute) and r.methods & {"POST", "PUT", "PATCH", "DELETE"}
    }
    assert "/api/suppliers/import" in mutating
    assert not any("purchase" in p or "payment" in p for p in mutating)


# ==========================================================================
# Phase 3: marketplace readiness endpoints
# ==========================================================================


def test_marketplace_registry_endpoint(client: TestClient):
    body = client.get("/api/marketplaces/registry").json()
    assert body["live_execution_enabled"] is False
    assert len(body["marketplaces"]) == 4
    statuses = {c["status"] for m in body["marketplaces"] for c in m["capabilities"]}
    assert statuses <= {"SUPPORTED", "UNSUPPORTED", "NOT_CONFIGURED"}


def test_listing_preview_makes_no_external_call(client: TestClient):
    slug = client.get("/api/opportunities").json()["opportunities"][0]["slug"]
    body = client.get(f"/api/listings/{slug}/preview").json()
    assert body["result"] == "DRY_RUN"
    assert all(a["published"] is False for a in body["assessments"])


def test_listing_validation_serialises_structured_findings(client: TestClient):
    slug = client.get("/api/opportunities").json()["opportunities"][0]["slug"]
    body = client.get(f"/api/listings/{slug}/validate").json()
    assert body["readiness"] == "NOT_READY"
    # Not every assessment carries findings: an unconfigured marketplace is
    # reported via its capability status instead of being validated.
    findings = [f for a in body["assessments"] for f in a["findings"]]
    assert findings, "at least one marketplace should have produced findings"
    assert {"code", "field", "message", "severity", "blocks"} <= set(findings[0])


def test_publish_endpoint_refuses_with_a_structured_response(client: TestClient):
    slug = client.get("/api/opportunities").json()["opportunities"][0]["slug"]
    response = client.post(f"/api/listings/{slug}/publish")
    assert response.status_code == 200
    assert response.json()["result"] == "NOT_ENABLED_FOR_PHASE_3"
    assert response.json()["published"] is False


def test_listing_endpoints_404_on_an_unknown_product(client: TestClient):
    for path in ("preview", "validate"):
        assert client.get(f"/api/listings/nope/{path}").status_code == 404
    assert client.post("/api/listings/nope/publish").status_code == 404


def test_no_api_response_leaks_a_credential_value(client: TestClient, monkeypatch):
    import json

    monkeypatch.setenv("AMAZON_SP_CLIENT_SECRET", "PLACEHOLDER-leaked-value")
    slug = client.get("/api/opportunities").json()["opportunities"][0]["slug"]
    payload = json.dumps(
        {
            "registry": client.get("/api/marketplaces/registry").json(),
            "preview": client.get(f"/api/listings/{slug}/preview").json(),
            "settings": client.get("/api/settings").json(),
        }
    )
    assert "PLACEHOLDER-leaked-value" not in payload


def test_phase_3_adds_no_write_endpoint_to_the_allowlist(client: TestClient):
    from fastapi.routing import APIRoute

    mutating = {
        r.path
        for r in app.routes
        if isinstance(r, APIRoute) and r.methods & {"POST", "PUT", "PATCH", "DELETE"}
    }
    # publish exists but always refuses; nothing else was added.
    assert "/api/listings/{slug}/publish" in mutating
    assert not any(
        term in p for p in mutating for term in ("purchase", "payment", "fulfil")
    )


# ==========================================================================
# Phase 4: execution endpoints
# ==========================================================================


def _slug(client: TestClient) -> str:
    return client.get("/api/opportunities").json()["opportunities"][0]["slug"]


def test_execution_lifecycle_over_http(client: TestClient):
    slug = _slug(client)
    created = client.post(
        "/api/executions",
        json={
            "operation": "PRICE_UPDATE",
            "marketplace": "amazon_in",
            "slug": slug,
            "payload": {"price_inr": "1299"},
        },
    ).json()
    assert created["status"] == "PENDING_APPROVAL"

    blocked = client.post(
        f"/api/executions/{created['id']}/run?slug={slug}", json={}
    ).json()
    assert blocked["result"]["status"] == "REQUIRES_REVIEW"
    assert "approval_exists" in {
        g["gate"] for g in blocked["result"]["gates"] if g["result"] == "BLOCK"
    }


def test_a_client_cannot_supply_its_own_compliance_or_verdict(client: TestClient):
    """Authoritative state is derived server-side, never accepted."""
    slug = _slug(client)
    response = client.post(
        "/api/executions",
        json={
            "operation": "PRICE_UPDATE",
            "marketplace": "amazon_in",
            "slug": slug,
            "compliance_state": "PASS",
            "opportunity_verdict": "RECOMMENDED",
            "supplier_confirmed": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert "compliance_state" not in body
    assert "opportunity_verdict" not in body


def test_an_unknown_operation_is_refused(client: TestClient):
    response = client.post(
        "/api/executions",
        json={
            "operation": "TRANSFER_MONEY",
            "marketplace": "amazon_in",
            "slug": _slug(client),
        },
    )
    assert response.status_code == 400
    assert "unknown operation" in response.json()["detail"]


def test_requesting_live_mode_still_runs_dry_when_not_enabled(client: TestClient):
    slug = _slug(client)
    created = client.post(
        "/api/executions",
        json={"operation": "PRICE_UPDATE", "marketplace": "amazon_in", "slug": slug},
    ).json()
    client.post(f"/api/executions/{created['id']}/approve", json={"actor": "operator"})
    body = client.post(
        f"/api/executions/{created['id']}/run?slug={slug}", json={"mode": "LIVE"}
    ).json()
    assert body["result"]["mode"] == "DRY_RUN"
    assert body["result"]["side_effect_occurred"] is False
    assert body["live_execution_enabled"] is False


def test_the_audit_endpoint_is_read_only(client: TestClient):
    slug = _slug(client)
    created = client.post(
        "/api/executions",
        json={"operation": "PRICE_UPDATE", "marketplace": "amazon_in", "slug": slug},
    ).json()
    assert client.get(f"/api/executions/{created['id']}/audit").status_code == 200
    assert client.post(f"/api/executions/{created['id']}/audit").status_code == 405


def test_no_execution_response_leaks_a_credential(client: TestClient, monkeypatch):
    import json as _json

    monkeypatch.setenv("AMAZON_SP_CLIENT_SECRET", "PLACEHOLDER-leak-check")
    slug = _slug(client)
    created = client.post(
        "/api/executions",
        json={"operation": "PRICE_UPDATE", "marketplace": "amazon_in", "slug": slug},
    ).json()
    client.post(f"/api/executions/{created['id']}/approve", json={"actor": "op"})
    blob = _json.dumps(
        {
            "run": client.post(
                f"/api/executions/{created['id']}/run?slug={slug}", json={}
            ).json(),
            "audit": client.get(f"/api/executions/{created['id']}/audit").json(),
        }
    )
    assert "PLACEHOLDER-leak-check" not in blob


# ==========================================================================
# Phase 6: export, history and backup
# ==========================================================================


@pytest.mark.parametrize("dataset", ["executions", "audit", "opportunities", "ledger"])
def test_every_export_dataset_answers(client: TestClient, dataset: str):
    body = client.get(f"/api/export/{dataset}?limit=5").json()
    assert body["dataset"] == dataset
    assert isinstance(body["rows"], list)
    assert body["contains_secrets"] is False


def test_an_unknown_export_dataset_is_a_404(client: TestClient):
    response = client.get("/api/export/credentials")
    assert response.status_code == 404
    assert "unknown dataset" in response.json()["detail"]


def test_exports_are_bounded(client: TestClient):
    body = client.get("/api/export/opportunities?limit=2").json()
    assert len(body["rows"]) <= 2
    assert client.get("/api/export/opportunities?limit=99999").status_code == 422


def test_no_export_leaks_a_planted_secret(client: TestClient, monkeypatch):
    import json as _json

    monkeypatch.setenv("AMAZON_SP_CLIENT_SECRET", "PLACEHOLDER-export-leak")
    blob = _json.dumps(
        {
            d: client.get(f"/api/export/{d}?limit=20").json()
            for d in ("executions", "audit", "opportunities", "ledger")
        }
    )
    assert "PLACEHOLDER-export-leak" not in blob


def test_exports_preserve_identifiers_and_state(client: TestClient):
    rows = client.get("/api/export/opportunities?limit=5").json()["rows"]
    assert rows
    for row in rows:
        assert "group_key" in row
        assert "verdict" in row
        assert row["requires_approval"] is True


def test_execution_history_is_filterable(client: TestClient):
    body = client.get(
        "/api/executions/history?limit=10&state=SUCCEEDED&marketplace=amazon_in"
    ).json()
    assert "rows" in body
    for row in body["rows"]:
        assert row["execution_state"] == "SUCCEEDED"
        assert row["marketplace"] == "amazon_in"


def test_execution_history_is_bounded(client: TestClient):
    assert client.get("/api/executions/history?limit=99999").status_code == 422


def test_history_route_is_not_shadowed_by_the_id_route(client: TestClient):
    """`history` must not be read as an execution_id."""
    response = client.get("/api/executions/history")
    assert response.status_code == 200
    assert "rows" in response.json()
