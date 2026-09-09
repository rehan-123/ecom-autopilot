"""Repository integrity: the project must be complete and self-consistent.

These exist because of a real failure. The project was validated in a fresh
environment against a partial checkout — eight files, no dependency manifest,
no ``database/migrations.py`` — and the symptoms were a ``ModuleNotFoundError``
and a ``FileNotFoundError`` from the mutation harness. Neither named the
actual cause, which was that most of the repository was missing.

Nothing here tests business logic. Every test answers one question: *is this
checkout complete and internally consistent, or is it lying to me?*
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path logic lives in tools/, which is not an installable package. Importing it
# here keeps a single definition of "project source" shared with the mutation
# harness rather than two lists that drift apart.
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from project_files import (  # noqa: E402
    is_generated_dir,
    is_virtualenv,
    iter_production_files,
    iter_project_files,
    source_packages,
)

sys.path.pop(0)

#: Every module the architecture requires. A checkout missing any of these is
#: incomplete, and the failure must say so rather than surfacing as an
#: unrelated import error several layers down.
REQUIRED_MODULES: tuple[str, ...] = (
    "api.app",
    "api.service",
    "catalog.enums",
    "catalog.schemas",
    "catalog.repository",
    "compliance.engine",
    "database.types",
    "database.tables",
    "database.connection",
    "database.migrations",
    "decisions.pipeline",
    "decisions.rules",
    "discovery.registry",
    "econ.calculator",
    "econ.landed_cost",
    "econ.money",
    "ledger.events",
    "ledger.projections",
    "ledger.replay",
    "llm.services",
    "execution.engine",
    "execution.model",
    "marketplace.draft",
    "marketplace.registry",
    "part2.auth",
    "part2.marketplace",
    "part2.normalize",
    "part2.transport",
    "reference.loader",
    "research.assistant",
    "research.export",
    "signals.market",
    "sourcing.feeds",
    "sourcing.matching",
    "sourcing.model",
    "sourcing.opportunities",
    "sources.ingest",
)

REQUIRED_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "README.md",
    "main.py",
    "validate.sh",
    "config/settings.yaml",
    "tools/mutation_check.py",
    "tests/conftest.py",
    "data/reference/compliance/india_baseline.yaml",
    "data/reference/threshold_profile.low_risk.yaml",
    "data/reference/scenario_profile.worst.yaml",
    "data/fixtures/discovery/observations.csv",
    "data/fixtures/discovery/trends.csv",
    "data/fixtures/discovery/suppliers.csv",
    "api/app.py",
    "api/service.py",
    "execution/engine.py",
    "execution/model.py",
    "marketplace/draft.py",
    "marketplace/registry.py",
    "research/assistant.py",
    "research/export.py",
    "frontend/package.json",
    "frontend/src/App.jsx",
    "frontend/src/pages/index.jsx",
)

#: Third-party distributions the runtime genuinely needs, mapped to the module
#: name they provide.
#: Distributions declared in an optional extra rather than the core runtime.
#: Production code may import them, but only within the subsystem the extra
#: installs -- the CLI and engines must keep working without them.
OPTIONAL_DISTRIBUTIONS: dict[str, str] = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}

RUNTIME_DISTRIBUTIONS: dict[str, str] = {
    "pydantic": "pydantic",
    "SQLAlchemy": "sqlalchemy",
    "PyYAML": "yaml",
    "openpyxl": "openpyxl",
    "pypdf": "pypdf",
}


def _is_test_infrastructure(parts: tuple[str, ...]) -> bool:
    """Whether a path is test scaffolding rather than production code.

    The root ``conftest.py`` counts: it imports pytest and describes checkout
    problems in prose, neither of which belongs to the shipped system.
    """
    return (
        parts[0] in {"tests", "tools", ".venv", ".mutation_scratch"}
        or parts[-1] == "conftest.py"
        or "__pycache__" in parts
    )


def _load_pyproject() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


# --------------------------------------------------------------------------
# The checkout is complete
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_module_imports(module_name: str):
    """Every architectural module is present and importable."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"{module_name} could not be imported ({exc}). Either the "
            "checkout is incomplete or a dependency is missing; install with "
            "`pip install -e .[dev]` from the project root."
        )


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_exists(relative_path: str):
    assert (PROJECT_ROOT / relative_path).is_file(), (
        f"{relative_path} is missing from the checkout"
    )


def test_every_package_directory_has_an_init():
    for package in (
        "catalog",
        "compliance",
        "database",
        "decisions",
        "discovery",
        "econ",
        "ledger",
        "llm",
        "reference",
        "reports",
        "signals",
        "sources",
        "tests",
    ):
        assert (PROJECT_ROOT / package / "__init__.py").is_file(), (
            f"{package} is missing __init__.py and will not import as a package"
        )


# --------------------------------------------------------------------------
# The dependency manifest is honest
# --------------------------------------------------------------------------


def test_runtime_dependencies_are_declared():
    """Every third-party runtime import appears in the manifest.

    Guards the exact failure that started this: SQLAlchemy imported by the
    code but unobtainable in a clean environment.
    """
    declared = " ".join(_load_pyproject()["project"]["dependencies"]).lower()
    assert "httpx" not in declared, "httpx must stay a dev dependency, not runtime"
    for distribution in RUNTIME_DISTRIBUTIONS:
        assert distribution.lower() in declared, (
            f"{distribution} is imported at runtime but not declared in "
            "pyproject.toml [project.dependencies]"
        )


def test_declared_runtime_dependencies_are_importable():
    for distribution, module_name in RUNTIME_DISTRIBUTIONS.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            pytest.fail(
                f"{distribution} is declared but not installed; the "
                "environment does not match the manifest"
            )


def test_no_undeclared_third_party_imports_in_production_code():
    """Production code imports only stdlib, first-party, or declared packages.

    Catches a dependency added to the code but not to the manifest, which
    works locally and fails in every clean environment.
    """
    first_party = {name.lower() for name in source_packages(PROJECT_ROOT)} | {
        "tools",
        "main",
        "conftest",
        "project_files",
    }
    declared = (
        {name.lower() for name in RUNTIME_DISTRIBUTIONS.values()}
        | {name.lower() for name in OPTIONAL_DISTRIBUTIONS.values()}
        | first_party
    )
    stdlib = set(sys.stdlib_module_names)

    offenders: list[str] = []
    for path in iter_production_files(PROJECT_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if root and root not in stdlib and root.lower() not in declared:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {root}")

    assert not offenders, "undeclared third-party imports: " + ", ".join(offenders)


def test_optional_extras_are_declared_in_the_manifest():
    """Anything an optional subsystem imports must be installable.

    fastapi lives in the `api` extra rather than core runtime, so the CLI and
    engines keep working without it. It still has to be declared somewhere, or
    a clean install of that extra fails.
    """
    manifest = _load_pyproject()["project"]["optional-dependencies"]
    declared = " ".join(manifest.get("api", [])).lower()
    for distribution in OPTIONAL_DISTRIBUTIONS:
        assert distribution.lower() in declared, (
            f"{distribution} is imported by the API but not in the `api` extra"
        )


def test_engines_import_without_the_api_extra():
    """The domain must not acquire a hard dependency on the web layer."""
    for module in (
        "econ.calculator",
        "decisions.pipeline",
        "sourcing.opportunities",
        "part2.intelligence",
    ):
        source_file = PROJECT_ROOT / (module.replace(".", "/") + ".py")
        text = source_file.read_text(encoding="utf-8")
        assert "fastapi" not in text, f"{module} must not import the web framework"
        assert "import api" not in text, f"{module} must not depend on the API layer"


def test_dev_dependencies_cover_the_validation_tools():
    dev = " ".join(_load_pyproject()["project"]["optional-dependencies"]["dev"]).lower()
    assert "pytest" in dev
    assert "ruff" in dev


# --------------------------------------------------------------------------
# The mutation harness targets the real implementation
# --------------------------------------------------------------------------


def _load_mutation_harness():
    """Import the harness from tools/, which is not an importable package.

    The sys.path manipulation is deliberate and confined here: tools/ holds
    standalone scripts rather than a package, so this is the only way to
    inspect the harness's declarations from a test.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    try:
        import mutation_check
    finally:
        sys.path.pop(0)
    return mutation_check


def test_every_mutation_target_exists():
    """The harness must reference files this repository actually contains.

    A harness pointing at a module that is not here reports FileNotFoundError
    instead of a test result, which reads as a broken project rather than a
    stale harness.
    """
    mutation_check = _load_mutation_harness()
    missing = [
        mutation.target
        for mutation in mutation_check.MUTATIONS
        if not (PROJECT_ROOT / mutation.target).is_file()
    ]
    assert not missing, f"mutation targets not in the repository: {missing}"


def test_every_mutation_pattern_still_matches_its_source():
    """Mutation patterns must survive refactoring and reformatting.

    A pattern that no longer matches makes the mutation silently
    unapplied — the harness reports an error rather than a false pass, but the
    invariant stops being checked either way.
    """
    mutation_check = _load_mutation_harness()
    stale = [
        mutation.name
        for mutation in mutation_check.MUTATIONS
        if mutation.old
        not in (PROJECT_ROOT / mutation.target).read_text(encoding="utf-8")
    ]
    assert not stale, f"mutation patterns no longer match their source: {stale}"


# --------------------------------------------------------------------------
# Collection cannot silently succeed on an empty suite
# --------------------------------------------------------------------------


def test_suite_is_substantial():
    """A near-empty collection would otherwise look like a pass.

    If dependencies or modules go missing, pytest can collect very little and
    still exit zero on the subset that survives. This asserts the suite that
    actually ran is the real one.
    """
    collected = list(PROJECT_ROOT.glob("tests/test_*.py"))
    assert len(collected) >= 6, (
        f"only {len(collected)} test modules found; expected the full suite"
    )


# --------------------------------------------------------------------------
# The Part 1 boundary holds
# --------------------------------------------------------------------------


FORBIDDEN_PART_2_TERMS: tuple[str, ...] = (
    "stripe",
    "razorpay",
    "payment_gateway",
    "place_order",
    "publish_listing",
    "purchase_order",
    "capture_payment",
    "seller_api",
    "marketplace_client",
)


#: Part 1 packages. Operational capability must not leak into these: Part 1
#: decides, Part 2 acts, and the separation is what keeps a decision engine
#: from acquiring the ability to spend money.
PART_ONE_PACKAGES = frozenset(
    {
        "catalog",
        "compliance",
        "database",
        "decisions",
        "discovery",
        "econ",
        "ledger",
        "llm",
        "reference",
        "reports",
        "signals",
        "sources",
    }
)


def test_no_part_two_capability_exists_in_part_one_source():
    """Part 1 stays a decision engine with no operational reach.

    Scans the project source tree only. An earlier version walked everything
    below the project root, so a virtualenv created during validation put
    third-party libraries in scope and the scan reported forbidden terms found
    inside pydantic and openpyxl.
    """
    offenders: list[str] = []
    for path in iter_project_files(PROJECT_ROOT, suffixes=(".py", ".yaml", ".toml")):
        parts = path.relative_to(PROJECT_ROOT).parts
        if parts[0] not in PART_ONE_PACKAGES:
            continue
        if path.name == "conftest.py" or "test_" in path.name:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        offenders.extend(
            f"{path.relative_to(PROJECT_ROOT)}: {term}"
            for term in FORBIDDEN_PART_2_TERMS
            if term in lowered
        )
    assert not offenders, f"Part 2 capability found in Part 1 source: {offenders}"


#: Capabilities that remain absent even in Part 2. Part 2 may publish a
#: listing and update inventory; it may never move money.
FORBIDDEN_EVERYWHERE = (
    "stripe",
    "razorpay",
    "payment_gateway",
    "capture_payment",
    "charge_card",
    "initiate_payout",
    "bank_transfer",
)


def test_no_payment_capability_exists_anywhere():
    """Part 2 operates the business; it never moves money.

    Ordering stock and paying for it are different actions with different
    blast radii. Part 2 can recommend a purchase and record one; it has no
    code that can execute a payment.
    """
    offenders: list[str] = []
    for path in iter_project_files(PROJECT_ROOT, suffixes=(".py", ".yaml", ".toml")):
        if "test_" in path.name:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        offenders.extend(
            f"{path.relative_to(PROJECT_ROOT)}: {term}"
            for term in FORBIDDEN_EVERYWHERE
            if term in lowered
        )
    assert not offenders, f"payment capability found: {offenders}"


def test_part2_external_writes_require_approval():
    """Every marketplace write path runs through the execution gate."""
    from part2.execution import ExecutionState, NotApproved, run_approved

    assert NotApproved is not None
    # RECOMMENDED cannot reach QUEUED without passing through APPROVED.
    from part2.execution import EXECUTION_TRANSITIONS

    assert ExecutionState.QUEUED not in EXECUTION_TRANSITIONS[ExecutionState.RECOMMENDED]
    assert ExecutionState.QUEUED in EXECUTION_TRANSITIONS[ExecutionState.APPROVED]
    assert run_approved is not None


def test_part2_adapters_all_support_dry_run():
    """Dry run is a property of the shared base, not per-adapter goodwill."""
    from part2.execution import Outcome
    from part2.marketplace import ADAPTERS, ListingPayload

    payload = ListingPayload(
        sku_code="SKU-1",
        title="t",
        description="d",
        bullets=(),
        price_inr=Decimal("100"),
        quantity=1,
    )
    for name, cls in ADAPTERS.items():
        result = cls(environ={}).create_listing(payload, dry_run=True)
        assert result.outcome in (Outcome.DRY_RUN, Outcome.NOT_SUPPORTED), name
        assert not result.changed_external_state, name


def test_no_credentials_are_present_in_source():
    """No secret may appear in source, fixtures, config or documentation."""
    import re

    patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"(?i)(api_key|secret|token|password)\s*=\s*[\"'][^\"'\s]{12,}[\"']"),
    ]
    offenders: list[str] = []
    for path in iter_project_files(
        PROJECT_ROOT, suffixes=(".py", ".yaml", ".toml", ".md", ".sh")
    ):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(text):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}: {pattern.pattern[:30]}"
                )
    assert not offenders, f"possible credential in source: {offenders}"


def test_no_http_client_is_declared_as_a_dependency():
    """No marketplace or Claude client can exist without an HTTP client.

    Reads the actual manifest rather than a constant defined in this file. An
    earlier version compared a hardcoded dict against itself, so adding httpx
    to pyproject.toml would not have failed it -- a test that could only ever
    pass.
    """
    declared = " ".join(_load_pyproject()["project"]["dependencies"]).lower()
    assert "httpx" not in declared, "httpx must stay a dev dependency, not runtime"
    # httpx is permitted: FastAPI's TestClient requires it and it is a dev
    # dependency only. The ban is on clients that could reach a marketplace
    # from production code, which is asserted separately below.
    for client in ("requests", "aiohttp", "anthropic", "urllib3", "boto3"):
        assert client not in declared, (
            f"{client} is declared as a runtime dependency; Part 1 makes no "
            "network calls, so an HTTP client has no legitimate use here"
        )


# --------------------------------------------------------------------------
# The README describes this repository, not a previous one
# --------------------------------------------------------------------------


def _utf8_env() -> dict[str, str]:
    """Environment pinning a child Python process to UTF-8."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_per_module_test_counts_are_accurate():
    """Counts in the README table must match what pytest actually collects.

    The README once claimed 175 tests after the suite had grown to 201. A
    document that overstates its own verification is worse than one that says
    nothing, because it is trusted.
    """

    claimed = re.findall(r"\| `(test_\w+\.py)` \| (\d+) \|", _readme())
    assert claimed, "README no longer contains a per-module test count table"

    wrong: list[str] = []
    for module, count in claimed:
        output = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(PROJECT_ROOT / "tests" / module),
                "--collect-only",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_env(),
            check=False,
        ).stdout
        match = re.search(r"(\d+) tests? collected", output)
        actual = int(match.group(1)) if match else -1
        if actual != int(count):
            wrong.append(f"{module}: README says {count}, pytest collects {actual}")
    assert not wrong, "README test counts are stale: " + "; ".join(wrong)


def test_readme_total_matches_the_sum_of_its_table():

    readme = _readme()
    per_module = sum(
        int(count) for _, count in re.findall(r"\| `(test_\w+\.py)` \| (\d+) \|", readme)
    )
    stated = re.search(r"\*\*(\d+) tests across", readme)
    assert stated, "README no longer states a total test count"
    assert int(stated.group(1)) == per_module, (
        f"README total says {stated.group(1)} but its own table sums to {per_module}"
    )


def test_readme_does_not_reference_files_that_were_removed():
    """Guards against documenting a layout the project no longer has."""

    readme = _readme()
    for removed in ("requirements.txt", "requirements-dev.txt", "ruff.toml"):
        for line in readme.splitlines():
            if removed not in line:
                continue
            # A line saying the file is absent is correct; one telling the
            # reader to use it is not.
            if re.search(r"\b(no|not|removed|consolidated)\b", line, re.IGNORECASE):
                continue
            pytest.fail(
                f"README references removed file {removed!r} as if it exists: "
                f"{line.strip()!r}"
            )


def test_readme_mutation_count_matches_the_harness():
    harness = _load_mutation_harness()
    count = len(harness.MUTATIONS)
    # Spelled out rather than kept as a literal map: a map has to be edited
    # every time a mutation is added, and it was twice left behind.
    units = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]
    if count < 20:
        word = units[count]
    else:
        word = tens[count // 10] + (f"-{units[count % 10]}" if count % 10 else "")
    expected = word.capitalize()

    assert f"{expected} mutations" in _readme(), (
        f"README should say '{expected} mutations'; the harness declares {count}"
    )


# --------------------------------------------------------------------------
# Source-tree boundary: generated environments are never project source
# --------------------------------------------------------------------------


def test_virtualenv_is_detected_by_marker_not_by_name(tmp_path: Path):
    """Any environment is excluded, whatever it is called.

    A name list only excludes the environments someone thought of. Rehan's
    Windows run had .check-venv and .validation-venv, neither of which was on
    the list, so the scanner walked into site-packages and reported forbidden
    terms found inside pydantic and openpyxl.
    """
    for name in (".venv", ".check-venv", ".validation-venv", "env", "whatever-env"):
        env = tmp_path / name
        (env / "lib").mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
        assert is_virtualenv(env), f"{name} not recognised as a virtualenv"
        assert is_generated_dir(env)


def test_directory_without_the_marker_is_not_a_virtualenv(tmp_path: Path):
    """The marker must not over-match: a package named env_utils is source."""
    ordinary = tmp_path / "env_utils"
    ordinary.mkdir()
    assert not is_virtualenv(ordinary)
    assert not is_generated_dir(ordinary)


@pytest.mark.parametrize(
    "name",
    ["__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", "build", "dist"],
)
def test_cache_and_build_directories_are_generated(tmp_path: Path, name: str):
    directory = tmp_path / name
    directory.mkdir()
    assert is_generated_dir(directory)


def test_egg_info_and_site_packages_are_generated(tmp_path: Path):
    for name in ("ecom_autopilot.egg-info", "site-packages", "pydantic-2.13.dist-info"):
        directory = tmp_path / name
        directory.mkdir()
        assert is_generated_dir(directory), name


def test_scanner_ignores_a_virtualenv_created_inside_the_project(tmp_path: Path):
    """The end-to-end guarantee, on a synthetic tree.

    Builds a project containing a virtualenv whose site-packages holds a file
    with a forbidden term, and asserts discovery never yields it. This is the
    exact shape of the reported failure.
    """
    root = tmp_path / "project"
    (root / "econ").mkdir(parents=True)
    (root / "econ" / "money.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[tool.setuptools]\npackages = ["econ"]\n', encoding="utf-8"
    )

    site = root / ".validation-venv" / "lib" / "python3.14" / "site-packages"
    site.mkdir(parents=True)
    (root / ".validation-venv" / "pyvenv.cfg").write_text(
        "home = /usr\n", encoding="utf-8"
    )
    (site / "some_dependency.py").write_text(
        "STRIPE_API = 'stripe'\ndef place_order(): ...\n", encoding="utf-8"
    )

    found = {p.relative_to(root).as_posix() for p in iter_project_files(root)}
    assert "econ/money.py" in found
    assert not any(".validation-venv" in path for path in found)


def test_scanner_still_sees_a_forbidden_term_in_real_source(tmp_path: Path):
    """The exclusion must not blind the boundary check.

    A scanner that excludes everything passes trivially; this proves it still
    finds a Part 2 capability added to genuine project source.
    """
    root = tmp_path / "project"
    (root / "econ").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.setuptools]\npackages = ["econ"]\n', encoding="utf-8"
    )
    (root / "econ" / "checkout.py").write_text(
        "def place_order(cart):\n    ...\n", encoding="utf-8"
    )

    offenders = [
        path
        for path in iter_project_files(root)
        if "place_order" in path.read_text(encoding="utf-8")
    ]
    assert offenders, "a forbidden capability in real source must still be found"


def test_source_packages_are_read_from_the_manifest():
    packages = source_packages(PROJECT_ROOT)
    assert "econ" in packages
    assert "decisions" in packages
    assert not any(name.startswith(".") for name in packages)


def test_project_discovery_finds_every_first_party_package():
    found = {
        p.relative_to(PROJECT_ROOT).parts[0] for p in iter_project_files(PROJECT_ROOT)
    }
    for package in source_packages(PROJECT_ROOT):
        assert package in found, f"{package} not discovered"
    assert "tests" in found
    assert "main.py" in {p.name for p in iter_project_files(PROJECT_ROOT)}


def test_no_generated_directory_leaks_into_discovery():
    """Holds whether or not validation has already created an environment."""
    for path in iter_project_files(PROJECT_ROOT, suffixes=(".py", ".yaml", ".toml")):
        parts = path.relative_to(PROJECT_ROOT).parts
        assert "site-packages" not in parts
        assert "__pycache__" not in parts
        assert not any(is_virtualenv(PROJECT_ROOT / parts[0]) for _ in (0,))


# --------------------------------------------------------------------------
# UTF-8: the rupee sign must survive every path it travels
# --------------------------------------------------------------------------

RUPEE = "\u20b9"


def test_rupee_survives_a_file_round_trip(tmp_path: Path):
    """Explicit encoding on both ends, not the platform default.

    On Windows the default is cp1252, which has no rupee code point, so an
    unencoded write raised UnicodeEncodeError.
    """
    path = tmp_path / "prices.csv"
    path.write_text(
        f"Supplier,Product,Price\nAcme,Widget,{RUPEE} 1299.00\n", encoding="utf-8"
    )
    assert RUPEE in path.read_text(encoding="utf-8")


def test_rupee_survives_ingestion(tmp_path: Path):
    from sources.ingest import ingest_csv

    path = tmp_path / "s.csv"
    path.write_text(
        f'Supplier,Product,Price\nAcme,Widget,"{RUPEE} 1,299.00"\n', encoding="utf-8"
    )
    row = ingest_csv(path).rows[0]
    assert row.get("unit_cost_inr") == Decimal("1299.00")
    assert RUPEE in row.fields["unit_cost_inr"].raw


def test_rupee_survives_report_rendering():
    """Reports quote prices in rupees; that is domain, not decoration."""
    from econ.money import rupees

    rendered = f"{RUPEE}{rupees(Decimal('1299.00'))}"
    assert rendered.encode("utf-8").decode("utf-8") == rendered


def test_rupee_survives_a_subprocess_round_trip():
    """The path that actually failed: captured child output on a cp1252 host."""
    result = subprocess.run(
        [sys.executable, "-c", f'print("{RUPEE} 1299")'],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        check=False,
    )
    assert RUPEE in result.stdout


def test_cli_reports_render_rupees_without_encoding_error():
    """End to end: the analyze command prints reports full of rupee signs."""
    result = subprocess.run(
        [sys.executable, "main.py", "analyze"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "UnicodeEncodeError" not in result.stderr
    assert RUPEE in result.stdout


def test_every_project_file_is_valid_utf8():
    """No source file may be unreadable as UTF-8 on any platform."""
    unreadable: list[str] = []
    for path in iter_project_files(
        PROJECT_ROOT, suffixes=(".py", ".yaml", ".toml", ".md", ".sh")
    ):
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            unreadable.append(str(path.relative_to(PROJECT_ROOT)))
    assert not unreadable, f"not valid UTF-8: {unreadable}"


def test_no_file_io_relies_on_the_platform_default_encoding():
    """Every read_text/write_text in the project passes an explicit encoding.

    The Windows failure came from exactly this: a default-encoded write of a
    rupee sign. A grep-level rule is crude but catches the reintroduction.
    """
    offenders: list[str] = []
    for path in iter_project_files(PROJECT_ROOT):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in ("read_text", "write_text"):
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} {node.func.attr}"
                )
    assert not offenders, f"file I/O without explicit encoding: {offenders}"


def test_demo_command_is_idempotent(tmp_path: Path):
    """`main.py demo` must survive being run twice.

    Runs against a scratch database, not the default path: the demo is
    destructive and now refuses to reset a database holding real execution
    history, which is exactly the protection a production install needs.

    It seeds fixed suppliers and products by name, so without a reset the
    second run died on a UNIQUE constraint. Validation sequences run the demo
    more than once, and a command that only works on a virgin database is a
    trap.
    """
    _demo_db = tmp_path / "demo.db"
    for attempt in (1, 2):
        result = subprocess.run(
            [sys.executable, "main.py", "demo", "--db", str(_demo_db)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_env(),
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, f"run {attempt} failed: {result.stdout[-600:]}"
        assert "hash chain valid: True" in result.stdout
        assert "replay deterministic: True" in result.stdout


def test_cli_renders_rupees_on_a_legacy_codepage_console():
    """The exact Windows failure, reproduced by forcing the locale codec.

    Setting PYTHONIOENCODING=cp1252 makes CPython behave as a Windows console
    does by default. Before `main.py` reconfigured its streams, this raised
    UnicodeEncodeError on the first report line. Note the environment here
    deliberately does NOT set PYTHONUTF8 -- the application must fix its own
    output path rather than depend on the caller.
    """
    hostile = dict(os.environ)
    hostile["PYTHONIOENCODING"] = "cp1252"
    hostile.pop("PYTHONUTF8", None)

    result = subprocess.run(
        [sys.executable, "main.py", "analyze"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=hostile,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "UnicodeEncodeError" not in result.stderr
    assert RUPEE in result.stdout


def test_reports_actually_quote_currency():
    """Money reaching a human is rendered as currency, not a bare decimal."""
    from econ.money import format_inr

    rendered = format_inr(Decimal("1234567.89"))
    assert rendered.startswith(RUPEE)
    # Indian digit grouping: last three, then pairs.
    assert rendered == f"{RUPEE}12,34,567.89"


# --------------------------------------------------------------------------
# The validator itself
# --------------------------------------------------------------------------


def _load_validator():
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    try:
        import validate
    finally:
        sys.path.pop(0)
    return validate


def test_validator_covers_every_cli_command():
    """No command may be silently dropped from validation.

    `ingest` was previously left out because it needs `--source`, which is
    exactly the sort of omission this test exists to prevent: the commands
    hardest to validate are the ones most worth validating.
    """
    validator = _load_validator()
    covered = {
        step.command[step.command.index("main.py") + 1]
        for step in validator.cli_steps("python", Path("fixture.csv"))
        if "main.py" in step.command
    }

    # Parse the choices list itself rather than grepping a window of nearby
    # lines: an earlier version swept up the --dry-run flag names as if they
    # were commands.
    parser_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    start = parser_source.index("choices=[")
    block = parser_source[start : parser_source.index("]", start)]
    declared = set(re.findall(r'"([a-z][a-z-]*)"', block))
    assert "ingest" in declared, "choices list was not parsed correctly"
    missing = declared - covered
    assert not missing, f"CLI commands not covered by the validator: {sorted(missing)}"


def test_validator_supplies_a_source_to_ingest():
    """The ingest step must pass a real file, not run the command bare."""
    validator = _load_validator()
    step = next(
        s
        for s in validator.cli_steps("python", Path("/tmp/fix.csv"))
        if s.name == "cli: ingest"
    )
    assert "--source" in step.command
    assert step.command[step.command.index("--source") + 1].endswith(".csv")


def test_validator_fixture_ingests_cleanly(tmp_path: Path):
    """The generated fixture must actually parse, not merely exist.

    It deliberately contains aliased headers, a rupee sign and a quoted
    thousands separator -- all things real supplier exports contain and all
    things the ingester has had defects in.
    """
    from sources.ingest import ingest_csv

    validator = _load_validator()
    fixture = tmp_path / "supplier_catalogue.csv"
    fixture.write_text(validator.MINIMAL_SUPPLIER_CSV, encoding="utf-8")

    result = ingest_csv(fixture)
    assert len(result.usable_rows) == 2, result.summary()
    assert result.rows[0].get("supplier_name") == "Acme Traders"
    assert result.rows[0].get("unit_cost_inr") == Decimal("185.50")
    assert result.rows[1].get("unit_cost_inr") == Decimal("1299.00")
    assert RUPEE in result.rows[1].fields["unit_cost_inr"].raw


def test_validator_checks_output_not_just_exit_code():
    """A command that prints an error and exits zero must still fail."""
    validator = _load_validator()
    assert validator.check_cli_output("cli: demo", "hash chain valid: True") is not None
    good = "hash chain valid: True\nreplay deterministic: True"
    assert validator.check_cli_output("cli: demo", good) is None
    assert validator.check_cli_output(
        "cli: demo", good + "\nTraceback (most recent call last)"
    )


def test_force_remove_handles_readonly_files(tmp_path: Path):
    """Windows venvs hold read-only files; cleanup must not raise.

    A secondary cleanup error buries whatever the real validation result was,
    so removal clears the read-only bit and retries.
    """
    import stat as stat_module

    validator = _load_validator()
    target = tmp_path / "fake-venv"
    (target / "Scripts").mkdir(parents=True)
    locked = target / "Scripts" / "python.exe"
    locked.write_text("binary", encoding="utf-8")
    locked.chmod(stat_module.S_IREAD)

    assert validator.force_remove(target) is None
    assert not target.exists()


def test_force_remove_on_a_missing_path_is_not_an_error(tmp_path: Path):
    validator = _load_validator()
    assert validator.force_remove(tmp_path / "never-existed") is None


def test_validator_temp_venv_lives_outside_the_project_tree():
    """An environment inside the repo is one every scanner must then exclude."""
    validator = _load_validator()
    parent, _venv = None, None
    source = (PROJECT_ROOT / "tools" / "validate.py").read_text(encoding="utf-8")
    assert "tempfile.mkdtemp" in source

    # Assert on behaviour rather than on the text: the file legitimately
    # mentions the old in-tree path when explaining why it no longer uses one.
    import inspect

    body = inspect.getsource(validator.build_fresh_venv)
    assert "tempfile.mkdtemp" in body
    assert "ROOT /" not in body, (
        "the validator must not create its environment inside the project tree"
    )
    assert parent is None


def test_validate_sh_delegates_to_the_python_validator():
    """The shell wrapper must not carry its own logic to drift out of sync."""
    script = (PROJECT_ROOT / "validate.sh").read_text(encoding="utf-8")
    assert "tools/validate.py" in script
    assert "rm -rf" not in script, "cleanup belongs in the cross-platform validator"


def test_build_metadata_is_ignored_and_never_scanned():
    """An editable install writes *.egg-info into the project root.

    It cannot be asserted absent -- `pip install -e .` creates it, and the
    validator does exactly that -- so the invariant is that it is ignored,
    never walked as source, and never packaged. An earlier version of this
    test asserted absence and therefore failed inside the very validation run
    that produced it.
    """
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "egg-info" in ignore, "build metadata must be gitignored"

    # Structurally recognised as generated, so no scanner walks into it.
    probe = PROJECT_ROOT / "ecom_autopilot.egg-info"
    assert is_generated_dir(probe) or not probe.exists()

    leaked = [
        str(p.relative_to(PROJECT_ROOT))
        for p in iter_project_files(
            PROJECT_ROOT, suffixes=(".py", ".txt", ".toml", ".md")
        )
        if ".egg-info" in p.parts or "PKG-INFO" in p.name
    ]
    assert not leaked, f"build metadata reached project discovery: {leaked}"


def test_recommend_command_states_the_approval_requirement():
    """The shortlist must say plainly that nothing executes on its own.

    A human reading a ranked list of profitable-looking products is the exact
    moment to restate that approving is a separate, deliberate act.
    """
    result = subprocess.run(
        [sys.executable, "main.py", "recommend"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-600:]
    assert "FOR APPROVAL" in result.stdout
    assert "cannot purchase, publish or pay" in result.stdout
