#!/usr/bin/env python3
"""Mutation harness: break each guarantee, confirm a test catches it.

A green suite proves the tests pass. It does not prove they would fail if the
code were wrong. This harness deliberately breaks each invariant the milestone
claims and asserts that at least one test notices.

Any mutation reported as SURVIVED is a test gap, not a passing result.

Usage:  python tools/mutation_check.py
Exit:   0 if every mutation is caught, 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
IGNORE = shutil.ignore_patterns(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    ".venv*",
    ".mutation_scratch*",
    "*.db",
    "*.db-wal",
    "*.db-shm",
)


@dataclass(frozen=True)
class Mutation:
    """A single deliberate defect."""

    name: str
    target: str
    old: str
    new: str
    guards: str
    """What this mutation proves is actually tested."""


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="audit ledger UPDATE trigger disabled",
        target="database/migrations.py",
        old="BEFORE UPDATE ON {table}",
        new="BEFORE UPDATE ON {table} WHEN 0",
        guards="append-only enforcement at the database layer",
    ),
    Mutation(
        name="audit ledger DELETE trigger disabled",
        target="database/migrations.py",
        old="BEFORE DELETE ON {table}",
        new="BEFORE DELETE ON {table} WHEN 0",
        guards="deletion is refused, not merely discouraged",
    ),
    Mutation(
        name="foreign key enforcement off",
        target="database/connection.py",
        old="PRAGMA foreign_keys=ON",
        new="PRAGMA foreign_keys=OFF",
        guards="referential integrity is actually enforced",
    ),
    Mutation(
        name="hash chain unlinked from predecessor",
        target="ledger/events.py",
        old='"prev_hash": prev_hash,\n    }',
        new="}",
        guards="tamper evidence propagates forward",
    ),
    Mutation(
        name="chain verification never reports a problem",
        target="ledger/events.py",
        old="        events = self.get_events(conn=conn)\n        prev_hash = GENESIS_HASH",
        new='        return {"valid": True, "checked": 0, "head_hash": GENESIS_HASH}\n        events = self.get_events(conn=conn)\n        prev_hash = GENESIS_HASH',
        guards="verification actually inspects the chain",
    ),
    Mutation(
        name="landed cost equals supplier cost",
        target="econ/landed_cost.py",
        old="    total = money(\n        effective_cogs\n        + freight\n        + carry",
        new="    total = money(\n        effective_cogs\n        + ZERO * freight\n        + ZERO * carry",
        guards="landed cost includes freight, carry, QC and defect provision",
    ),
    Mutation(
        name="MOQ carrying cost dropped",
        target="econ/landed_cost.py",
        old="    return money(capital + dead_stock), assumptions",
        new="    return ZERO, assumptions",
        guards="MOQ is priced as locked capital, not treated as an attribute",
    ),
    Mutation(
        name="ITC granted without a tax invoice",
        target="econ/money.py",
        old="    if not tax_invoice_available:\n        # No invoice, no credit",
        new="    if False:\n        return ZERO",
        guards="a retail receipt yields no input tax credit",
    ),
    Mutation(
        name="worst-case gate skipped",
        target="econ/calculator.py",
        old='    if threshold.get("require_worst_case_non_negative", True):',
        new="    if False:",
        guards="approval requires the downside to hold, not just the expected case",
    ),
    Mutation(
        name="failed gate stops reporting its gap",
        target="econ/calculator.py",
        old='                    "action": action,',
        new='                    "action": "",',
        guards="every rejection says what would have to change",
    ),
    Mutation(
        name="score divides by total instead of populated weight",
        target="signals/market.py",
        old='    value = money(weighted / populated_weight).quantize(Decimal("0.1"))',
        new='    value = money(weighted / total_weight).quantize(Decimal("0.1"))',
        guards="missing signals are excluded, not silently treated as zero",
    ),
    Mutation(
        name="coverage floor ignored",
        target="signals/market.py",
        old="    if coverage < pct(min_coverage_pct) or populated_weight == ZERO:",
        new="    if populated_weight == ZERO:",
        guards="a low-coverage product emits no ranking value",
    ),
    Mutation(
        name="price headroom never blocks",
        target="signals/market.py",
        old="    if headroom < pct(marginal_headroom_pct):",
        new="    if False:",
        guards="a SKU unsellable at market price is rejected",
    ),
    Mutation(
        name="stale ruleset can clear a product",
        target="compliance/engine.py",
        old="    elif not ruleset_fresh:",
        new="    elif False:",
        guards="an unverified rulebook clears nothing",
    ),
    Mutation(
        name="UNKNOWN compliance resolves to PASS",
        target="compliance/engine.py",
        old=(
            "    elif (\n"
            "        ComplianceResult.UNKNOWN in results\n"
            "        or ComplianceResult.REQUIRES_HUMAN_REVIEW in results\n"
            "    ):"
        ),
        new="    elif False:",
        guards="UNKNOWN never becomes PASS",
    ),
    Mutation(
        name="unverified reference data can approve",
        target="decisions/rules.py",
        old='    if str(weakest_state) == "VERIFIED":',
        new="    if True:",
        guards="template rate cards cannot produce a business decision",
    ),
    Mutation(
        name="state machine accepts any transition",
        target="decisions/rules.py",
        old="    if not is_legal(from_state, to_state):",
        new="    if False:",
        guards="no stage can skip a gate",
    ),
    Mutation(
        name="unknown Judge flag ignored",
        target="decisions/rules.py",
        old="        if disposition is None:",
        new="        if False:",
        guards="an unrecognised finding routes to a human",
    ),
    Mutation(
        name="malformed LLM output treated as valid",
        target="llm/services.py",
        old="            parsed_ok=not violations,",
        new="            parsed_ok=True,",
        guards="a schema violation never becomes a business decision",
    ),
    Mutation(
        name="unparseable price becomes zero",
        target="sources/ingest.py",
        old='        return Field_(None, FieldStatus.REQUIRES_REVIEW, text, "no numeric value found")',
        new='        return Field_(ZERO_PRICE, FieldStatus.KNOWN, text, "defaulted")',
        guards="an unreadable price is flagged, never defaulted",
    ),
    Mutation(
        name="DATA_UNAVAILABLE may carry a signal value",
        target="signals/market.py",
        old="        if self.status is SignalStatus.DATA_UNAVAILABLE and (",
        new="        if False and (",
        guards="an absent market signal is not a zero",
    ),
    Mutation(
        name="idempotency store ignores prior results",
        target="part2/execution.py",
        old="        existing = self._results.get(key)\n        if existing is not None:",
        new="        existing = None\n        if existing is not None:",
        guards="a repeated intent never reaches the transport twice",
    ),
    Mutation(
        name="non-idempotent write is retried",
        target="part2/execution.py",
        old="            if not idempotent and not policy.retry_non_idempotent:",
        new="            if False:",
        guards="an unknown-outcome order is not retried into a duplicate",
    ),
    Mutation(
        name="UNKNOWN counts as success",
        target="part2/execution.py",
        old="        return self is Outcome.OK",
        new="        return self is not Outcome.PERMANENT_FAILURE",
        guards="only OK is success; UNKNOWN and DRY_RUN are not",
    ),
    Mutation(
        name="approval gate removed",
        target="part2/execution.py",
        old="    if record.state is not ExecutionState.APPROVED:",
        new="    if False:",
        guards="no external write executes without an approval",
    ),
    Mutation(
        name="inventory may go negative",
        target="part2/inventory.py",
        old="        if on_hand < 0 and position.on_hand < movement.quantity:",
        new="        if False:",
        guards="stock cannot be driven below zero",
    ),
    Mutation(
        name="reservation ignores availability",
        target="part2/inventory.py",
        old="        if movement.movement_type is MovementType.RESERVE and (\n            position.available < movement.quantity\n        ):",
        new="        if False:",
        guards="overselling is prevented, not reported afterwards",
    ),
    Mutation(
        name="duplicate inventory movement applied twice",
        target="part2/inventory.py",
        old="        if idempotent and signature in self._seen_references:",
        new="        if False:",
        guards="a replayed webhook does not double-count stock",
    ),
    Mutation(
        name="new product sized by forecast not by test",
        target="part2/inventory.py",
        old="        target = max(policy.initial_test_orders, supplier_moq)",
        new="        target = supplier_moq * 10",
        guards="a product with no history gets a small controlled test",
    ),
    Mutation(
        name="duplicate order import creates a second order",
        target="part2/operations.py",
        old="        existing = self._orders.get(order.key)\n        if existing is not None:",
        new="        existing = None\n        if existing is not None:",
        guards="order imports are idempotent",
    ),
    Mutation(
        name="unmapped marketplace status defaults silently",
        target="part2/operations.py",
        old="        raise UnknownStatus(",
        new="        return OrderStatus.NEW\n        raise UnknownStatus(",
        guards="an unrecognised status is surfaced, not guessed",
    ),
    Mutation(
        name="unmatched settlement is treated as matched",
        target="part2/operations.py",
        old="            state=MatchState.UNMATCHED,",
        new="            state=MatchState.MATCHED,",
        guards="money that ties to no order is never absorbed",
    ),
    Mutation(
        name="duplicate settlement import counted twice",
        target="part2/operations.py",
        old="        if key in self._lines:",
        new="        if False:",
        guards="settlement imports are idempotent",
    ),
    Mutation(
        name="realized profit ignores return costs",
        target="part2/operations.py",
        old="    contribution = money(settled - cogs - return_cost)",
        new="    contribution = money(settled - cogs)",
        guards="returns reduce realized contribution",
    ),
    Mutation(
        name="realized profit uses order value not settled money",
        target="part2/operations.py",
        old="    contribution = money(settled - cogs - return_cost)",
        new="    contribution = money(gross - cogs - return_cost)",
        guards="realized economics reconcile from settlements",
    ),
    Mutation(
        name="price floor ignored",
        target="part2/intelligence.py",
        old="    if proposed < points.minimum_viable_inr:",
        new="    if False:",
        guards="no price below the minimum viable price is allowed",
    ),
    Mutation(
        name="worst-case gate bypassed by pricing",
        target="part2/intelligence.py",
        old="        and worst_case_margin_at_proposed_pct < pct(worst_case_floor_pct)",
        new="        and False",
        guards="optimisation cannot bypass Part 1's worst-case gate",
    ),
    Mutation(
        name="kill decision ignores sample size",
        target="part2/intelligence.py",
        old=(
            "        realized.sample_size >= policy.min_sample_for_kill\n"
            "        and margin <= policy.kill_margin_pct"
        ),
        new="        margin <= policy.kill_margin_pct",
        guards="a product is not killed on one noisy observation",
    ),
    Mutation(
        name="ranking ignores capital and return risk",
        target="part2/intelligence.py",
        old="        per_capital = safe_div(realized.actual_contribution_inr, capital)",
        new="        per_capital = realized.gross_revenue_inr",
        guards="ranking is by risk-adjusted contribution, not revenue",
    ),
    Mutation(
        name="missing tax invoice is not surfaced in supplier ranking",
        target="part2/intelligence.py",
        old="        if not offer.tax_invoice_available:",
        new="        if False:",
        guards="the ranking explains why a cheap supplier is not the best one",
    ),
    Mutation(
        name="publication skips prerequisite checks",
        target="part2/publisher.py",
        old="        if not check.satisfied:",
        new="        if False:",
        guards="an unapproved or non-compliant listing cannot publish",
    ),
    Mutation(
        name="unsupported capability faked as success",
        target="part2/marketplace.py",
        old="        if not self.profile.supports(capability):",
        new="        if False:",
        guards="a missing capability reports NOT_SUPPORTED",
    ),
    Mutation(
        name="dry run performs the real call",
        target="part2/marketplace.py",
        old="        if dry_run:",
        new="        if False:",
        guards="dry run never touches external state",
    ),
    Mutation(
        name="listing drift never detected",
        target="part2/marketplace.py",
        old="    if money(remote.price_inr) != money(intended.price_inr):",
        new="    if False:",
        guards="a successful write is verified against marketplace state",
    ),
    Mutation(
        name="similarity alone merges products",
        target="sourcing/matching.py",
        old="        return self in (MatchConfidence.EXACT, MatchConfidence.HIGH)",
        new="        return True",
        guards="a title-similarity hypothesis never becomes an identity decision",
    ),
    Mutation(
        name="different GTINs no longer disprove a match",
        target="sourcing/matching.py",
        old="        # Two different GTINs is positive evidence they are NOT the same.",
        new="        return Match(left.external_ref, right.external_ref, MatchConfidence.HIGH,\n                     similarity, evidence)\n        # disabled",
        guards="conflicting GTINs are evidence against, not absence of evidence",
    ),
    Mutation(
        name="supplied model codes stop being normalized",
        target="sourcing/model.py",
        old='    cleaned = re.sub(r"[\\s\\-_.]+", "", code).upper()',
        new="    cleaned = code",
        guards="ML-750 and ML750 resolve to one product",
    ),
    Mutation(
        name="unavailable signal projected as zero",
        target="sourcing/model.py",
        old="            if not observed.present:",
        new="            if False:",
        guards="a missing signal reaches the scorer as absent, never as zero",
    ),
    Mutation(
        name="coverage always reported as complete",
        target="sourcing/model.py",
        old="        return money(got / total * HUNDRED) if total else ZERO",
        new="        return HUNDRED",
        guards="evidence coverage reflects what was actually obtained",
    ),
    Mutation(
        name="unavailable field may carry a value",
        target="sourcing/model.py",
        old="        if self.value is not None and self.source is FieldSource.UNAVAILABLE:",
        new="        if False:",
        guards="DATA_UNAVAILABLE cannot carry a number",
    ),
    Mutation(
        name="products without a supplier still rank",
        target="sourcing/opportunities.py",
        old="    if not matches:",
        new="    if False:",
        guards="a product that cannot be sourced cannot be recommended",
    ),
    Mutation(
        name="ranking uses expected instead of worst case",
        target="sourcing/opportunities.py",
        old="        opportunity.worst_profit_inr * coverage_factor * penalty",
        new="        (opportunity.expected_profit_inr or ZERO) * coverage_factor * penalty",
        guards="ranking is on the downside, not the optimistic case",
    ),
    Mutation(
        name="thin comparables no longer demote to BACKUP",
        target="sourcing/opportunities.py",
        old="    if opportunity.recoverable:",
        new="    if False:",
        guards="weak evidence cannot reach RECOMMENDED",
    ),
    Mutation(
        name="price reference built from too few comparables",
        target="sourcing/opportunities.py",
        old="    if len(observations) < policy.min_comparables:",
        new="    if False:",
        guards="two observations do not describe a market",
    ),
    Mutation(
        name="unavailable feed invents candidates",
        target="sourcing/feeds.py",
        old="        return FeedResult(\n            source=self.source, access=self.access, unavailable_reason=self._reason\n        )",
        new="        return FeedResult(source=self.source, access=self.access)",
        guards="a source with no legitimate route yields nothing, visibly",
    ),
    Mutation(
        name="seller feed ignores missing credentials",
        target="sourcing/feeds.py",
        old="        if not adapter.profile.credentials_present(environ):",
        new="        if False:",
        guards="an unconfigured seller API reports itself unavailable",
    ),
    Mutation(
        name="category benchmark fallback silently disabled",
        target="reference/loader.py",
        old='        return store.latest("return_benchmark:default", as_of=as_of)',
        new="        raise",
        guards="an uncatalogued category still prices against the default benchmark",
    ),
    Mutation(
        name="timeout on a write reported as retryable",
        target="part2/transport.py",
        old='        return Outcome.UNKNOWN, (\n            f"HTTP {status} on a non-idempotent request: the marketplace may "',
        new='        return Outcome.RETRYABLE_FAILURE, (\n            f"HTTP {status} on a non-idempotent request: the marketplace may "',
        guards="a write of unknown effect is never retried into a duplicate",
    ),
    Mutation(
        name="429 treated as permanent",
        target="part2/transport.py",
        old='        return Outcome.RETRYABLE_FAILURE, "HTTP 429: rate limited"',
        new='        return Outcome.PERMANENT_FAILURE, "HTTP 429: rate limited"',
        guards="rate limiting is retried, not abandoned",
    ),
    Mutation(
        name="auth headers no longer redacted",
        target="part2/transport.py",
        old='        key: ("<redacted>" if key.lower() in REDACTED_HEADERS else value)',
        new="        key: value",
        guards="a bearer token never reaches a log or a result",
    ),
    Mutation(
        name="rate limiter accepts a looser published limit",
        target="part2/transport.py",
        old="            if 0 < rate_per_second < self.rate_per_second:",
        new="            if rate_per_second > 0:",
        guards="an observed rate limit only ever tightens",
    ),
    Mutation(
        name="pagination cursor loop no longer detected",
        target="part2/transport.py",
        old="        if cursor in seen:",
        new="        if False:",
        guards="a repeated cursor stops the walk instead of hanging",
    ),
    Mutation(
        name="null transport starts making calls",
        target="part2/transport.py",
        old='            outcome=Outcome.PERMANENT_FAILURE,\n            detail=(\n                "no HTTP transport configured;',
        new='            outcome=Outcome.OK,\n            detail=(\n                "no HTTP transport configured;',
        guards="offline is the default and cannot silently succeed",
    ),
    Mutation(
        name="token cache never expires",
        target="part2/auth.py",
        old="        return time.time() < (self.expires_at - margin_s)",
        new="        return True",
        guards="an expiring token is refreshed before it fails in flight",
    ),
    Mutation(
        name="token refresh failure returns an empty token",
        target="part2/auth.py",
        old="        if not value:",
        new="        if False:",
        guards="a token response without a token is refused",
    ),
    Mutation(
        name="adapter becomes live without a transport",
        target="part2/marketplace.py",
        old="        return not isinstance(self._transport, NullTransport)",
        new="        return True",
        guards="credentials alone never enable a network call",
    ),
    Mutation(
        name="unmapped order status defaults during normalization",
        target="part2/normalize.py",
        old='        raise NormalizationError(f"{marketplace} order {order_id}: {exc}") from exc',
        new="        return OrderStatus.NEW",
        guards="a marketplace's new status is surfaced, not guessed",
    ),
    Mutation(
        name="missing amount silently becomes zero",
        target="part2/normalize.py",
        old='            raise NormalizationError(f"{field} is missing and has no safe default")',
        new="            return ZERO",
        guards="a zeroed fee never flows into realized profit",
    ),
    Mutation(
        name="settlement deduction signs no longer normalized",
        target="part2/normalize.py",
        old="        return abs(value)",
        new="        return value",
        guards="deductions are positive regardless of report sign convention",
    ),
    Mutation(
        name="one bad order discards the whole page",
        target="part2/normalize.py",
        old="        except NormalizationError as exc:\n            failures.append(str(exc))\n            logger.warning(",
        new="        except NormalizationError:\n            raise\n            logger.warning(",
        guards="a malformed order does not lose the rest of the page",
    ),
    Mutation(
        name="(control) comment-only edit",
        target="ledger/events.py",
        old='GENESIS_HASH = "0" * 64',
        new='GENESIS_HASH = "0" * 64  # inert control mutation',
        guards="(control) inert — must survive, proving the harness can tell",
    ),
    Mutation(
        name="naive datetimes silently coerced",
        target="database/types.py",
        old='raise ValueError(\n                "refusing to store a naive datetime; supply tzinfo explicitly"\n            )',
        new="value = value.replace(tzinfo=timezone.utc)",
        guards="naive datetimes are rejected, not assumed to be UTC",
    ),
    Mutation(
        name="write transactions become DEFERRED",
        target="database/connection.py",
        old='conn.exec_driver_sql("BEGIN IMMEDIATE")',
        new='conn.exec_driver_sql("BEGIN")',
        guards="concurrent appends cannot fork the hash chain",
    ),
    Mutation(
        name="prev_hash uniqueness dropped",
        target="database/tables.py",
        old='UniqueConstraint("prev_hash", name="uq_audit_event_prev_hash"),',
        new="",
        guards="a chain fork is a schema violation",
    ),
    Mutation(
        name="terminal state guard removed",
        target="catalog/repository.py",
        old="if current in TERMINAL_STATES and not allow_reopen:",
        new="if False:",
        guards="a decided SKU cannot be silently reopened",
    ),
    Mutation(
        name="state transition emits no ledger event",
        target="catalog/repository.py",
        old="event_type=EventType.SKU_STATE_CHANGED,",
        new="event_type=EventType.ERROR,",
        guards="every state change is recorded as such",
    ),
    Mutation(
        name="offer observation ordering ignored",
        target="ledger/replay.py",
        old="ordered = self.in_observation_order",
        new="ordered = self.observations",
        guards="replay and the projection agree on backfilled offers",
    ),
    Mutation(
        name="projection reads become authoritative",
        target="ledger/projections.py",
        old="projection.table.drop(bind=conn, checkfirst=True)\n            projection.table.create(bind=conn, checkfirst=True)",
        new="projection.table.create(bind=conn, checkfirst=True)",
        guards="rebuild fully replaces derived state rather than appending",
    ),
    Mutation(
        name="DATA_UNAVAILABLE may carry a value",
        target="catalog/schemas.py",
        old="if self.status is SignalStatus.DATA_UNAVAILABLE and (",
        new="if False and (",
        guards="an absent signal is not a zero",
    ),
    Mutation(
        name="INSUFFICIENT_DATA may carry a score",
        target="catalog/schemas.py",
        old="if self.result is ScoreResult.INSUFFICIENT_DATA and self.value is not None:",
        new="if False:",
        guards="a low-coverage product cannot be ranked as if evidenced",
    ),
    Mutation(
        name="unregistered supplier may issue a tax invoice",
        target="catalog/schemas.py",
        old="if self.issues_tax_invoice and self.gst_status is not GSTStatus.REGISTERED:",
        new="if False:",
        guards="input tax credit cannot be assumed without a tax invoice",
    ),
    Mutation(
        name="Decimal money silently becomes float",
        target="database/types.py",
        old='raise TypeError("refusing to store a float as a monetary value; use Decimal")',
        new="value = Decimal(str(value))",
        guards="monetary precision is preserved",
    ),
)


#: Tests about the repository rather than the code under mutation. A mutation
#: makes them fail by construction (or, for the README checks, they fail
#: whenever documentation is mid-edit), so leaving them in would report CAUGHT
#: for every mutation -- including the inert control, which is how this fault
#: reveals itself.
_META_TESTS: tuple[str, ...] = (
    "tests/test_repository_integrity.py::test_every_mutation_pattern_still_matches_its_source",
    "tests/test_repository_integrity.py::test_every_mutation_target_exists",
    "tests/test_repository_integrity.py::test_readme_per_module_test_counts_are_accurate",
    "tests/test_repository_integrity.py::test_readme_total_matches_the_sum_of_its_table",
    "tests/test_repository_integrity.py::test_readme_mutation_count_matches_the_harness",
)

_DESELECT_META_TESTS: tuple[str, ...] = tuple(
    arg for name in _META_TESTS for arg in ("--deselect", name)
)


def utf8_env() -> dict[str, str]:
    """Environment forcing UTF-8 for a child Python process.

    ``PYTHONUTF8`` enables UTF-8 mode; ``PYTHONIOENCODING`` covers interpreters
    where UTF-8 mode is unavailable. Without these, a subprocess on Windows
    dies with UnicodeEncodeError the moment a report prints a rupee sign.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _scratch_parent() -> Path | None:
    """A temp location guaranteed not to sit inside the project tree.

    ``copytree`` recurses forever if the destination is a descendant of the
    source, which happens whenever the project itself lives under the system
    temp directory. Returning None makes tempfile pick its default.
    """
    default = Path(tempfile.gettempdir()).resolve()
    if default == ROOT or ROOT.is_relative_to(default):
        sibling = ROOT.parent / f".mutation_scratch_{ROOT.name}"
        sibling.mkdir(exist_ok=True)
        return sibling
    return None


def run_one(mutation: Mutation) -> tuple[str, str]:
    """Apply one mutation to a scratch copy and run the suite."""
    with tempfile.TemporaryDirectory(dir=_scratch_parent()) as tmp:
        project = Path(tmp) / "project"
        if Path(tmp).resolve().is_relative_to(ROOT):
            return "ERROR", "scratch directory would nest inside the project"
        shutil.copytree(ROOT, project, ignore=IGNORE)
        (project / "data").mkdir(exist_ok=True)

        target = project / mutation.target
        source = target.read_text(encoding="utf-8")
        if mutation.old not in source:
            return "ERROR", "mutation pattern not found (source drifted)"
        target.write_text(source.replace(mutation.old, mutation.new, 1), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                "--no-header",
                # These two assert that mutation patterns still match their
                # source, so a mutation makes them fail by construction. Left
                # in, they would be the first failure for every mutation and
                # would mask whether the invariant's real guard still works --
                # a mutation would report CAUGHT while its actual protection
                # had silently disappeared.
                *_DESELECT_META_TESTS,
            ],
            cwd=project,
            capture_output=True,
            text=True,
            # Windows defaults to the locale codec (cp1252), which cannot
            # encode the rupee sign that appears throughout the reports. Both
            # ends of the pipe are pinned to UTF-8 so captured output survives.
            encoding="utf-8",
            errors="replace",
            env=utf8_env(),
            timeout=600,
            check=False,  # a non-zero exit is the signal, not an error
        )
        if result.returncode != 0:
            failed = [
                line.split("::")[-1].split(" ")[0]
                for line in result.stdout.splitlines()
                if line.startswith("FAILED")
            ]
            return "CAUGHT", failed[0] if failed else "suite failed"
        return "SURVIVED", "no test failed"


def verify_targets() -> list[str]:
    """Confirm every mutation target exists before running anything.

    A harness that silently references a module the repository does not
    contain reports FileNotFoundError instead of a test result. Checking up
    front turns a confusing stack trace into a clear message naming the
    missing files.
    """
    return sorted({m.target for m in MUTATIONS if not (ROOT / m.target).is_file()})


def main(argv: list[str] | None = None) -> int:
    """Run the mutation set, optionally a slice of it.

    ``--slice i/n`` runs every n-th mutation starting at i. The full set runs
    the whole suite once per mutation, which is slow enough that CI benefits
    from splitting it; the slices together cover exactly the same set.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = MUTATIONS
    if "--slice" in argv:
        spec = argv[argv.index("--slice") + 1]
        index, total = (int(part) for part in spec.split("/"))
        selected = tuple(m for i, m in enumerate(MUTATIONS) if i % total == index - 1)

    missing = verify_targets()
    if missing:
        print("Mutation targets missing from the repository:")
        for path in missing:
            print(f"  - {path}")
        print(
            "\nThe harness tests the real implementation, so these files must "
            "be present.\nIf you downloaded individual files rather than the "
            "whole project, re-download the\ncomplete repository."
        )
        return 1

    print(f"Running {len(selected)} of {len(MUTATIONS)} mutations\n")
    survived: list[Mutation] = []
    errored: list[Mutation] = []

    for mutation in selected:
        status, detail = run_one(mutation)
        control = mutation.guards.startswith("(control)")
        marker = {"CAUGHT": "ok  ", "SURVIVED": "GAP ", "ERROR": "ERR "}[status]
        if control and status == "SURVIVED":
            marker = "ok  "  # an inert mutation is expected to survive
        print(f"  {marker} {mutation.name:<44} {status:<9} {detail}")
        if status == "SURVIVED" and not control:
            survived.append(mutation)
        elif status == "ERROR":
            errored.append(mutation)

    print()
    if survived:
        print(f"{len(survived)} test gap(s) — these invariants are unguarded:")
        for mutation in survived:
            print(f"  - {mutation.guards}")
    if errored:
        print(f"{len(errored)} mutation(s) could not be applied:")
        for mutation in errored:
            print(f"  - {mutation.name} ({mutation.target})")

    if not survived and not errored:
        print("All guarded invariants have a test that fails when broken.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
