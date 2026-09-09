#!/usr/bin/env python3
"""A-Z release gate: validate the whole repository.

Complements ``tools/validate.py`` rather than replacing it. That one proves the
code is correct and safe: that the safety properties every phase established still hold, that no
safe: that the safety properties every phase established still hold, that no
secret can escape, and that the documentation still matches reality.

The distinction matters because a green test suite and a safe release are not
the same claim. Tests can pass while the README promises live marketplace
publishing that does not exist, or while a dependency carries a known
vulnerability.

Run it from the project root:

    python tools/release_gate.py
    python tools/release_gate.py --skip-frontend   # no Node toolchain
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Patterns that would indicate a real credential in the repository. Written to
#: match credential *values*, never variable names -- the UI legitimately shows
#: an operator which variable to set.
SECRET_PATTERNS = (
    r"sk-[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"ghp_[A-Za-z0-9]{30,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
)

#: Claims the documentation must never make, because they are not true.
FORBIDDEN_DOC_CLAIMS = (
    "live marketplace publishing is supported",
    "orders are placed automatically",
    "payments are processed",
)


@dataclass
class Check:
    name: str
    run: Callable[[], tuple[bool, str]]
    blocking: bool = True


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    warned: list[tuple[str, str]] = field(default_factory=list)


def sh(command: list[str], *, cwd: Path = ROOT, timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
        check=False,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


# --------------------------------------------------------------------------
# Structure and code quality
# --------------------------------------------------------------------------


def check_structure() -> tuple[bool, str]:
    """Every phase's package is present."""
    required = [
        "catalog",
        "database",
        "ledger",
        "econ",
        "compliance",
        "sourcing",
        "decisions",
        "research",
        "marketplace",
        "execution",
        "api",
        "part2",
    ]
    missing = [p for p in required if not (ROOT / p / "__init__.py").exists()]
    return (
        not missing,
        f"missing packages: {missing}" if missing else f"{len(required)} packages",
    )


def check_pytest() -> tuple[bool, str]:
    code, out = sh(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    )
    match = re.search(r"(\d+) passed", out)
    return code == 0, f"{match.group(1)} passed" if match else out[-200:]


def check_ruff() -> tuple[bool, str]:
    code, out = sh([sys.executable, "-m", "ruff", "check", "."])
    return code == 0, "clean" if code == 0 else out[-300:]


def check_format() -> tuple[bool, str]:
    code, out = sh([sys.executable, "-m", "ruff", "format", "--check", "."])
    return code == 0, out.strip().splitlines()[-1] if out.strip() else "clean"


def check_compile() -> tuple[bool, str]:
    code, out = sh([sys.executable, "-m", "compileall", "-q", "."])
    return code == 0, "clean" if code == 0 else out[-300:]


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------


def check_no_secrets() -> tuple[bool, str]:
    """No credential value anywhere in tracked source."""
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {
            ".py",
            ".jsx",
            ".js",
            ".json",
            ".md",
            ".yaml",
            ".yml",
            ".toml",
            ".csv",
        }:
            continue
        if any(
            part in {"node_modules", ".git", "dist", "__pycache__"} for part in path.parts
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT)} ({pattern})")
    return (
        not offenders,
        f"found: {offenders[:3]}" if offenders else "no credential values",
    )


def check_frontend_bundle_clean() -> tuple[bool, str]:
    """A built bundle must contain no credential."""
    dist = ROOT / "frontend" / "dist"
    if not dist.exists():
        return True, "no build present (skipped)"
    for asset in dist.rglob("*.js"):
        text = asset.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                return False, f"{asset.name} contains a credential-shaped value"
    return True, "bundle clean"


def check_npm_audit() -> tuple[bool, str]:
    """Known fixable vulnerabilities are a blocker, not a warning."""
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        return True, "dependencies not installed (skipped)"
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    code, out = sh([npm_cmd, "audit", "--json"], cwd=frontend, timeout=300)
    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return code == 0, "npm audit produced unparseable output"
    total = report.get("metadata", {}).get("vulnerabilities", {}).get("total", 0)
    return total == 0, f"{total} vulnerabilities"


# --------------------------------------------------------------------------
# Safety properties
# --------------------------------------------------------------------------


def check_execution_safety() -> tuple[bool, str]:
    """The execution guarantees, asserted directly rather than via tests."""
    sys.path.insert(0, str(ROOT))
    from execution import (  # noqa: PLC0415
        ExecutionContext,
        ExecutionEngine,
        ExecutionRequest,
        ExecutionStatus,
        IllegalTransition,
        OperationType,
        live_execution_enabled,
    )

    findings: list[str] = []
    if live_execution_enabled({}):
        findings.append("live execution enabled without the flag")

    request = ExecutionRequest(
        operation=OperationType.PRICE_UPDATE,
        marketplace="amazon_in",
        sku_code="SKU-1",
        payload={"price_inr": "1"},
        requested_by="gate",
    )
    try:
        request.transition(ExecutionStatus.EXECUTING, actor="gate")
        findings.append("EXECUTING reachable without approval")
    except IllegalTransition:
        pass

    context = ExecutionContext(
        compliance_state="PASS",
        opportunity_verdict="RECOMMENDED",
        supplier_confirmed=True,
        economics_available=True,
    )
    result = ExecutionEngine().execute(request, context, environ={})
    if result.side_effect_occurred:
        findings.append("an unapproved request reported a side effect")

    return not findings, "; ".join(findings) if findings else "gates intact"


def check_marketplace_safety() -> tuple[bool, str]:
    """Credentials must never convert UNSUPPORTED into SUPPORTED."""
    sys.path.insert(0, str(ROOT))
    from marketplace import CapabilityStatus, capability_status  # noqa: PLC0415
    from part2.marketplace import Capability  # noqa: PLC0415

    report = capability_status(
        "meesho",
        Capability.CREATE_LISTING,
        environ={"MEESHO_TOKEN": "x", "AMAZON_SP_CLIENT_ID": "x"},
    )
    if report.status is not CapabilityStatus.UNSUPPORTED:
        return False, "credentials converted an unsupported capability"
    return True, "unsupported stays unsupported"


def check_regression() -> tuple[bool, str]:
    """The known Phase 2 outcomes."""
    sys.path.insert(0, str(ROOT))
    import collections  # noqa: PLC0415

    from decisions.pipeline import Pipeline  # noqa: PLC0415
    from reference.loader import ReferenceStore  # noqa: PLC0415
    from sourcing.feeds import (  # noqa: PLC0415
        build_registry,
        collect,
        load_supplier_candidates,
    )
    from sourcing.matching import group_products  # noqa: PLC0415
    from sourcing.opportunities import evaluate_group, rank_opportunities  # noqa: PLC0415

    products, _ = collect(build_registry(fixture_dir=ROOT / "data/fixtures/discovery"))
    candidates = load_supplier_candidates(ROOT / "data/fixtures/discovery/suppliers.csv")
    pipeline = Pipeline(ReferenceStore(ROOT / "tests/fixtures/reference").load())
    opportunities = rank_opportunities(
        [
            evaluate_group(g, candidates, pipeline=pipeline)
            for g in group_products(products)
        ]
    )
    tally = dict(collections.Counter(str(o.verdict) for o in opportunities))
    expected = {"BACKUP": 1, "REJECTED": 6}
    if tally != expected:
        return False, f"tally is {tally}, expected {expected}"

    titles = {o.title: o for o in opportunities}
    kemei = next((o for t, o in titles.items() if "KX880" in t), None)
    if kemei is None or "BIS" not in " ".join(kemei.blockers):
        return False, "Kemei BIS block missing"
    return True, f"{tally}, Kemei BIS blocked"


def check_docs_honest() -> tuple[bool, str]:
    """The README must not claim capabilities that do not exist."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    claims = [c for c in FORBIDDEN_DOC_CLAIMS if c in readme]
    if claims:
        return False, f"README claims: {claims}"
    required = ["dry run", "human approval", "advisory"]
    missing = [r for r in required if r not in readme]
    return not missing, f"missing sections: {missing}" if missing else "accurate"


def check_database_lifecycle() -> tuple[bool, str]:
    """Clean init, warm restart, and a non-destructive backup."""
    import tempfile  # noqa: PLC0415

    workdir = Path(tempfile.mkdtemp())
    database = workdir / "gate.db"

    code, out = sh([sys.executable, "main.py", "init-db", "--db", str(database)])
    if code != 0:
        return False, f"init-db failed: {out[-200:]}"
    if not database.exists():
        return False, "init-db created no database"

    before = database.stat().st_size
    code, _out = sh([sys.executable, "main.py", "init-db", "--db", str(database)])
    if code != 0 or database.stat().st_size < before:
        return False, "re-running init-db was destructive"

    code, out = sh(
        [
            sys.executable,
            "main.py",
            "backup",
            "--db",
            str(database),
            "--to",
            str(workdir / "b.db"),
        ]
    )
    if code != 0 or "integrity    ok" not in out:
        return False, f"backup failed: {out[-200:]}"
    return True, "clean init, warm restart, verified backup"


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------


def check_frontend_tests() -> tuple[bool, str]:
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        return True, "dependencies not installed (skipped)"
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    code, out = sh([npm_cmd, "test"], cwd=frontend, timeout=600)
    match = re.search(r"Tests\s+(\d+) passed", out)
    return code == 0, f"{match.group(1)} passed" if match else out[-200:]


def check_frontend_build() -> tuple[bool, str]:
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        return True, "dependencies not installed (skipped)"
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    code, out = sh([npm_cmd, "run", "build"], cwd=frontend, timeout=900)
    return code == 0, "built" if code == 0 else out[-300:]


def build_checks(*, skip_frontend: bool) -> list[Check]:
    checks = [
        Check("structure", check_structure),
        Check("pytest", check_pytest),
        Check("ruff check", check_ruff),
        Check("ruff format", check_format),
        Check("compileall", check_compile),
        Check("secret scan", check_no_secrets),
        Check("execution safety", check_execution_safety),
        Check("marketplace safety", check_marketplace_safety),
        Check("database lifecycle", check_database_lifecycle),
        Check("phase 2 regression", check_regression),
        Check("documentation honesty", check_docs_honest),
    ]
    if not skip_frontend:
        checks += [
            Check("npm audit", check_npm_audit),
            Check("frontend tests", check_frontend_tests),
            Check("frontend build", check_frontend_build),
            Check("frontend bundle", check_frontend_bundle_clean),
        ]
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A-Z release gate.")
    parser.add_argument("--skip-frontend", action="store_true")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    report = Report()
    checks = build_checks(skip_frontend=args.skip_frontend)
    print(f"ECOM AUTOPILOT - RELEASE GATE ({len(checks)} checks)")

    for index, check in enumerate(checks, start=1):
        try:
            ok, detail = check.run()
        except Exception as exc:  # noqa: BLE001 - a gate reports, never crashes
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        status = "PASS" if ok else ("FAIL" if check.blocking else "WARN")
        print(f"  {index:>2}/{len(checks)}  {check.name:<24} {status:<5} {detail[:70]}")
        if ok:
            report.passed.append(check.name)
        elif check.blocking:
            report.failed.append((check.name, detail))
        else:
            report.warned.append((check.name, detail))

    print()
    if report.failed:
        print(f"RELEASE GATE: BLOCKED - {len(report.failed)} check(s) failed")
        for name, detail in report.failed:
            print(f"  - {name}: {detail[:200]}")
        return 1

    print(f"RELEASE GATE: PASS - {len(checks)}/{len(checks)} checks passed")
    print(
        "\nThis validates dry-run readiness. Live marketplace execution remains\n"
        "unavailable: no real adapter transport is implemented and no operation\n"
        "has been confirmed by a marketplace."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
