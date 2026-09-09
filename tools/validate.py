#!/usr/bin/env python3
"""Complete project validation, cross-platform.

Replaces a bash-only validator that could not run on Windows at all. Three
things it gets right that the previous one did not:

**It smoke-tests the CLI.** Running the commands is the only way to catch a
command that imports fine but dies on first use. ``ingest`` legitimately
requires ``--source``, so the validator generates a minimal supplier file in a
temporary directory, ingests it, checks the result, and removes it. Skipping
the command would have hidden exactly the class of defect this step exists to
find.

**Its temporary environment lives outside the project tree.** A virtualenv
created inside the repository is something every integrity scanner then has to
be taught to ignore. Putting it in the system temp directory removes the
problem instead of working around it.

**Cleanup cannot produce a secondary error.** On Windows, files under a venv's
``Scripts`` directory are frequently read-only and still held open briefly
after the last subprocess exits, so a plain ``rmtree`` raises ``PermissionError``
and buries whatever the real validation result was. Removal clears the
read-only bit, retries, and if it still cannot finish it reports a warning
rather than failing the run: a leftover temp directory is untidy, not a
validation failure.

Usage:
    python tools/validate.py              validate in the current environment
    python tools/validate.py --fresh      build a throwaway venv from the manifest
    python tools/validate.py --quick      skip mutation testing
    python tools/validate.py --slice 1/2  run half the mutation set
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


#: Packages byte-compiled by the first gate. Derived from the manifest so a
#: new package is covered without editing this file.
def _packages() -> list[str]:
    # tools/ holds standalone scripts rather than a package, so the path
    # manipulation is deliberate and confined here.
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from project_files import source_packages  # noqa: PLC0415 - see comment

        return source_packages(ROOT)
    finally:
        sys.path.pop(0)


def utf8_env() -> dict[str, str]:
    """Environment that keeps a child Python on UTF-8.

    Reports quote prices in rupees, and U+20B9 has no cp1252 code point. Both
    variables are set because UTF-8 mode is unavailable on some builds.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def force_remove(path: Path, *, attempts: int = 4) -> str | None:
    """Delete a directory tree, tolerating Windows' read-only venv files.

    Returns None on success, or a warning string if the tree could not be
    fully removed. A leftover temporary directory is untidy; failing the whole
    validation over it would report the wrong problem.
    """
    if not path.exists():
        return None

    def clear_readonly(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)  # noqa: PTH101 - needs the raw call
            func(target)
        except OSError:
            pass

    for attempt in range(1, attempts + 1):
        try:
            # onexc superseded onerror in 3.12, which is this project's floor.
            shutil.rmtree(path, onexc=clear_readonly)
            return None
        except (OSError, PermissionError):
            if attempt == attempts:
                return f"could not fully remove {path}; delete it manually"
            # A just-exited subprocess can still hold a handle briefly.
            time.sleep(0.3 * attempt)
    return None


@dataclass
class Step:
    """One validation gate."""

    name: str
    command: Sequence[str]
    cwd: Path = ROOT


def run(
    command: Sequence[str], *, cwd: Path = ROOT, timeout: int = 1800
) -> tuple[int, str]:
    """Run a subprocess with UTF-8 pinned on both ends."""
    result = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env(),
        timeout=timeout,
        check=False,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


# --------------------------------------------------------------------------
# CLI smoke tests
# --------------------------------------------------------------------------

MINIMAL_SUPPLIER_CSV = (
    "Supplier Name,Item Description,Rate (INR),MOQ,HSN,Weight\n"
    "Acme Traders,Steel Lunchbox 750ml,185.50,24,7323,320\n"
    'Acme Traders,Rechargeable Trimmer KX-880,"\u20b9 1,299.00",12,8510,310\n'
)
"""A supplier file that exercises the real parsing paths.

Deliberately not trivial: aliased headers, a rupee sign, and a quoted
thousands separator are all things real supplier exports contain and all
things the ingester has had defects in.
"""


def cli_steps(python: str, fixture: Path) -> list[Step]:
    """Every CLI command, each invoked the way it is meant to be used."""
    return [
        Step("cli: init-db", [python, "main.py", "init-db"]),
        Step("cli: demo", [python, "main.py", "demo"]),
        Step("cli: status", [python, "main.py", "status"]),
        Step("cli: verify-ledger", [python, "main.py", "verify-ledger"]),
        Step("cli: rebuild-projections", [python, "main.py", "rebuild-projections"]),
        Step("cli: verify-ledger (post-rebuild)", [python, "main.py", "verify-ledger"]),
        Step("cli: analyze", [python, "main.py", "analyze"]),
        Step("cli: sources", [python, "main.py", "sources"]),
        Step("cli: reference", [python, "main.py", "reference"]),
        # ingest requires a source; the validator supplies one rather than
        # skipping the command.
        Step("cli: ingest", [python, "main.py", "ingest", "--source", str(fixture)]),
        Step("cli: discover", [python, "main.py", "discover"]),
        Step("cli: trends", [python, "main.py", "trends"]),
        Step("cli: match", [python, "main.py", "match"]),
        Step("cli: suppliers", [python, "main.py", "suppliers"]),
        Step("cli: opportunities", [python, "main.py", "opportunities"]),
        Step("cli: rank", [python, "main.py", "rank"]),
        Step("cli: recommend", [python, "main.py", "recommend"]),
        Step("cli: credentials", [python, "main.py", "credentials"]),
        # backup writes a timestamped copy; it never touches the live database.
        Step("cli: backup", [python, "main.py", "backup"]),
        Step("cli: marketplaces", [python, "main.py", "marketplaces"]),
        Step("cli: publish (dry run)", [python, "main.py", "publish", "--dry-run"]),
        Step("cli: inventory", [python, "main.py", "inventory"]),
        Step("cli: reconcile", [python, "main.py", "reconcile"]),
        Step("cli: portfolio", [python, "main.py", "portfolio"]),
        Step("cli: optimize", [python, "main.py", "optimize"]),
    ]


def check_cli_output(name: str, output: str) -> str | None:
    """Assert each command actually did its job, not merely exited zero.

    An exit code alone is weak evidence: a command that prints an error and
    returns 0 would pass. These check for the specific evidence each command
    is supposed to produce.
    """
    expectations = {
        "cli: demo": ["hash chain valid: True", "replay deterministic: True"],
        "cli: verify-ledger": ["valid"],
        "cli: rebuild-projections": ["REBUILT"],
        "cli: ingest": ["INGESTED CSV", "usable"],
        "cli: analyze": ["\u20b9"],
        "cli: marketplaces": ["No HTTP transport is bundled"],
        "cli: publish (dry run)": ["PUBLICATION GATE"],
        "cli: portfolio": ["PRODUCT PORTFOLIO"],
        "cli: optimize": ["OPTIMISATION"],
        "cli: discover": ["DISCOVERY FEED COVERAGE", "DATA_UNAVAILABLE"],
        "cli: trends": ["SEARCH INTEREST"],
        "cli: match": ["PRODUCT MATCHING"],
        "cli: suppliers": ["SUPPLIER OPTIONS"],
        "cli: opportunities": ["OPPORTUNITIES"],
        "cli: rank": ["OPPORTUNITIES"],
        # The shipped fixtures deterministically yield at least one BACKUP, so
        # the heading is a strict expectation rather than an either/or.
        "cli: recommend": ["FOR APPROVAL", "cannot purchase, publish or pay"],
        "cli: credentials": ["MARKETPLACE CREDENTIALS", "never enable a network call"],
        "cli: backup": ["DATABASE BACKUP", "integrity    ok"],
    }
    for needle in expectations.get(name, []):
        if needle not in output:
            return f"expected {needle!r} in output"
    if "Traceback" in output or "UnicodeEncodeError" in output:
        return "command produced a traceback"
    return None


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def venv_python(venv: Path) -> str:
    """Interpreter path inside a venv, on either platform layout."""
    candidate = (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    return str(candidate)


def build_fresh_venv() -> tuple[Path, str]:
    """Create a throwaway venv OUTSIDE the project tree.

    Outside is the point. An environment inside the repository is something
    every integrity scanner then has to exclude, and the previous validator
    created one at `.venv-validate` for exactly that reason.
    """
    parent = Path(tempfile.mkdtemp(prefix="ecom-validate-"))
    venv = parent / "venv"
    print(f"==> creating throwaway venv at {venv}")
    code, output = run([sys.executable, "-m", "venv", str(venv)], timeout=300)
    if code != 0:
        raise RuntimeError(f"could not create venv:\n{output}")

    python = venv_python(venv)
    print("==> installing from pyproject.toml only")
    for args in (
        [python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        [python, "-m", "pip", "install", "--quiet", "-e", ".[dev]"],
    ):
        code, output = run(args, timeout=900)
        if code != 0:
            raise RuntimeError(f"dependency install failed:\n{output[-2000:]}")
    return parent, python


def build_steps(
    python: str, fixture: Path, *, quick: bool, mutation_slice: str | None
) -> list[Step]:
    """Assemble every gate, in dependency order."""
    steps: list[Step] = [
        Step(
            "compileall",
            [
                python,
                "-m",
                "compileall",
                "-q",
                *_packages(),
                "tools",
                "tests",
                "main.py",
                "conftest.py",
            ],
        ),
        Step("pytest", [python, "-m", "pytest"]),
        Step("ruff check", [python, "-m", "ruff", "check", "."]),
        Step("ruff format --check", [python, "-m", "ruff", "format", "--check", "."]),
        *cli_steps(python, fixture),
    ]
    if not quick:
        mutation = [python, "tools/mutation_check.py"]
        if mutation_slice:
            mutation += ["--slice", mutation_slice]
        steps.append(Step("mutation check", mutation))
    return steps


def execute(steps: list[Step]) -> list[tuple[str, str]]:
    """Run each gate, returning the failures.

    A step passes only if it exits zero AND produced the evidence it is
    supposed to produce; an exit code alone would let a command that prints an
    error and returns 0 slip through.
    """
    failures: list[tuple[str, str]] = []
    for index, step in enumerate(steps, start=1):
        print(f"\n==> {index}/{len(steps)}  {step.name}")
        try:
            code, output = run(step.command, cwd=step.cwd)
        except subprocess.TimeoutExpired:
            failures.append((step.name, "timed out"))
            print("    TIMED OUT")
            continue

        problem = (
            f"exit code {code}" if code != 0 else check_cli_output(step.name, output)
        )
        if problem:
            failures.append((step.name, f"{problem}\n{output[-1500:]}"))
            print(f"    FAILED - {problem.splitlines()[0]}")
        else:
            tail = [line for line in output.strip().splitlines() if line.strip()]
            print(f"    ok  {tail[-1][:96] if tail else ''}")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the project end to end.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="build a throwaway venv outside the project tree first",
    )
    parser.add_argument("--quick", action="store_true", help="skip mutation testing")
    parser.add_argument(
        "--slice",
        dest="mutation_slice",
        default=None,
        help="run a slice of the mutation set, e.g. 1/2",
    )
    return parser


def force_utf8_stdio() -> None:
    """Pin this process's streams to UTF-8 before anything prints rupees."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def report(failures: list[tuple[str, str]]) -> int:
    """Print the outcome and return the process exit code."""
    print()
    if failures:
        print(f"{len(failures)} step(s) failed:")
        for name, detail in failures:
            print(f"\n--- {name} ---\n{detail}")
        return 1
    print("All validation steps passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force_utf8_stdio()

    temp_root: Path | None = None
    python = sys.executable
    warnings: list[str] = []

    try:
        if args.fresh:
            temp_root, python = build_fresh_venv()

        fixture_dir = Path(tempfile.mkdtemp(prefix="ecom-fixture-"))
        fixture = fixture_dir / "supplier_catalogue.csv"
        fixture.write_text(MINIMAL_SUPPLIER_CSV, encoding="utf-8")

        steps: list[Step] = [
            Step(
                "compileall",
                [
                    python,
                    "-m",
                    "compileall",
                    "-q",
                    *_packages(),
                    "tools",
                    "tests",
                    "main.py",
                    "conftest.py",
                ],
            ),
            Step("pytest", [python, "-m", "pytest"]),
            Step("ruff check", [python, "-m", "ruff", "check", "."]),
            Step("ruff format --check", [python, "-m", "ruff", "format", "--check", "."]),
            *cli_steps(python, fixture),
        ]
        if not args.quick:
            mutation = [python, "tools/mutation_check.py"]
            if args.mutation_slice:
                mutation += ["--slice", args.mutation_slice]
            steps.append(Step("mutation check", mutation))

        failures: list[tuple[str, str]] = []
        for index, step in enumerate(steps, start=1):
            print(f"\n==> {index}/{len(steps)}  {step.name}")
            try:
                code, output = run(step.command, cwd=step.cwd)
            except subprocess.TimeoutExpired:
                failures.append((step.name, "timed out"))
                print("    TIMED OUT")
                continue

            problem = None if code != 0 else check_cli_output(step.name, output)
            if code != 0:
                problem = f"exit code {code}"

            if problem:
                failures.append((step.name, f"{problem}\n{output[-1500:]}"))
                print(f"    FAILED - {problem.splitlines()[0]}")
            else:
                tail = [line for line in output.strip().splitlines() if line.strip()]
                print(f"    ok  {tail[-1][:96] if tail else ''}")

        warning = force_remove(fixture_dir)
        if warning:
            warnings.append(warning)

        return report(failures)
    finally:
        if temp_root is not None:
            warning = force_remove(temp_root)
            if warning:
                warnings.append(warning)
        for warning in warnings:
            # Reported, never fatal: a leftover temp directory is untidy, not
            # a validation failure, and conflating the two hides real results.
            print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
