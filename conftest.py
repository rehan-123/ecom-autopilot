"""Root conftest: preflight validation before anything else is collected.

This exists because of a specific failure. A validation run against an
incomplete checkout produced:

    ImportError while loading conftest 'tests/conftest.py'
    ModuleNotFoundError: No module named 'sqlalchemy'

That message names a symptom, not the cause. The cause was that the checkout
was missing most of its files, including the dependency manifest. Worse, the
failure happens while *loading* ``tests/conftest.py``, so every test —
including the repository-integrity tests written to diagnose exactly this —
never runs.

pytest loads conftest files from the root downward, and conftests along
``testpaths`` are imported during argument parsing -- *before*
``pytest_configure`` fires. A hook is therefore too late: ``tests/conftest.py``
has already failed by then. The check runs at module import time instead,
which is the first project code pytest executes.

It validates nothing about business logic. It answers one question: *is this
checkout complete enough for the suite to mean anything?*
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent

#: Modules the architecture requires. Absence means an incomplete checkout.
REQUIRED_MODULE_FILES: tuple[str, ...] = (
    "econ/calculator.py",
    "econ/landed_cost.py",
    "econ/money.py",
    "reference/loader.py",
    "signals/market.py",
    "sources/ingest.py",
    "compliance/engine.py",
    "decisions/rules.py",
    "decisions/pipeline.py",
    "discovery/registry.py",
    "llm/services.py",
    "part2/transport.py",
    "part2/auth.py",
    "part2/normalize.py",
    "api/app.py",
    "api/service.py",
    "execution/engine.py",
    "execution/model.py",
    "marketplace/draft.py",
    "marketplace/registry.py",
    "research/assistant.py",
    "research/export.py",
    "sourcing/model.py",
    "sourcing/matching.py",
    "sourcing/feeds.py",
    "sourcing/opportunities.py",
    "catalog/__init__.py",
    "catalog/enums.py",
    "catalog/schemas.py",
    "catalog/repository.py",
    "database/__init__.py",
    "database/types.py",
    "database/tables.py",
    "database/connection.py",
    "database/migrations.py",
    "ledger/__init__.py",
    "ledger/events.py",
    "ledger/projections.py",
    "ledger/replay.py",
    "main.py",
    "pyproject.toml",
)

#: Third-party module name -> the distribution that provides it.
REQUIRED_THIRD_PARTY: dict[str, str] = {
    "pydantic": "pydantic",
    "sqlalchemy": "SQLAlchemy",
    "yaml": "PyYAML",
    "openpyxl": "openpyxl",
    "pypdf": "pypdf",
}


def _missing_files() -> list[str]:
    return [p for p in REQUIRED_MODULE_FILES if not (PROJECT_ROOT / p).is_file()]


def _missing_packages() -> list[str]:
    return [
        distribution
        for module_name, distribution in REQUIRED_THIRD_PARTY.items()
        if importlib.util.find_spec(module_name) is None
    ]


def _force_utf8_stdio() -> None:
    """Pin the test process's streams to UTF-8.

    pytest captures output, but a failure report that quotes a rupee sign is
    still written to the real stdout at the end of the run. On a cp1252
    console that raises UnicodeEncodeError and buries the actual failure.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _preflight() -> None:
    """Fail with a diagnosis rather than an import traceback."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    missing_files = _missing_files()
    missing_packages = _missing_packages()
    if not missing_files and not missing_packages:
        return

    lines = [
        "",
        "=" * 72,
        "PREFLIGHT FAILED - this checkout cannot be validated",
        "=" * 72,
    ]

    if missing_files:
        lines += [
            "",
            f"{len(missing_files)} required file(s) are missing from the project:",
            *(f"    {path}" for path in missing_files),
            "",
            "This usually means individual files were downloaded rather than the",
            "whole project. Every module above is required by the architecture;",
            "none of them is optional, and none should be recreated by hand.",
        ]

    if missing_packages:
        lines += [
            "",
            f"{len(missing_packages)} runtime dependency(ies) are not installed:",
            *(f"    {name}" for name in missing_packages),
            "",
            "Install them from the project's own manifest, not globally:",
            "",
            "    python -m venv .venv",
            "    source .venv/bin/activate      # Windows: .venv\\\\Scripts\\\\activate",
            "    pip install -e '.[dev]'",
        ]

    lines += ["", "=" * 72, ""]
    message = "\n".join(lines)
    print(message, file=sys.stderr)
    # pytest.exit stops the run without a traceback: the diagnosis above is
    # the useful output, and a stack trace from conftest would bury it.
    pytest.exit(message, returncode=2)


# Runs at import time, before tests/conftest.py is loaded.
_force_utf8_stdio()
_preflight()
