"""Project source discovery: what belongs to this repository, and what does not.

Anything that walks the repository -- integrity scanners, the safety-boundary
check, the mutation harness -- must agree on where the project ends. Getting
that boundary wrong is not a cosmetic problem: a scanner that wanders into
``site-packages`` reports forbidden terms found inside third-party libraries
and fails the build for code the project never wrote.

The boundary is defined two ways, deliberately overlapping:

**Positively**, from the manifest. ``pyproject.toml`` already lists the
first-party packages, so the source roots are derived from it rather than from
a literal that drifts when a package is added.

**Negatively**, by detecting generated directories structurally rather than by
name. Every virtualenv contains ``pyvenv.cfg``, whatever it is called --
``.venv``, ``.check-venv``, ``.validation-venv``, ``env``. Matching on that
marker catches environments this file has never heard of, which a hardcoded
name list cannot. Caches and build outputs are matched by name because they
have no equivalent marker.

Everything here is pure path logic. No project module imports it, so it stays
usable from the mutation harness, which runs against scratch copies.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from pathlib import Path

#: Directory names that are always generated, never source.
GENERATED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        ".git",
        ".idea",
        ".vscode",
        "build",
        "dist",
        "site-packages",
        "node_modules",
        # Emitted by the mutation harness when the project sits under /tmp.
        ".mutation_scratch",
    }
)

#: Suffixes that mark a directory as generated regardless of its exact name.
GENERATED_DIR_SUFFIXES: tuple[str, ...] = (".egg-info", ".dist-info")

#: Top-level files that are project source even though they sit outside a
#: package directory.
ROOT_SOURCE_FILES: tuple[str, ...] = (
    "main.py",
    "conftest.py",
    "pyproject.toml",
    "README.md",
    "validate.sh",
)

#: Directories that hold project source but are not installable packages.
NON_PACKAGE_SOURCE_DIRS: tuple[str, ...] = ("tests", "tools", "config", "data")


def is_virtualenv(path: Path) -> bool:
    """Whether a directory is a Python virtual environment.

    Detected by ``pyvenv.cfg``, which every environment created by ``venv`` or
    ``virtualenv`` contains. Structural detection rather than a name list, so
    an environment called ``.check-venv`` or ``my-env`` is recognised without
    this file having to know about it in advance.
    """
    return path.is_dir() and (path / "pyvenv.cfg").is_file()


def is_generated_dir(path: Path) -> bool:
    """Whether a directory is generated output rather than project source."""
    name = path.name
    if name in GENERATED_DIR_NAMES:
        return True
    if name.endswith(GENERATED_DIR_SUFFIXES):
        return True
    return is_virtualenv(path)


def is_inside_generated(path: Path, root: Path) -> bool:
    """Whether any directory between ``root`` and ``path`` is generated.

    Checked for every ancestor, not just the first component: a virtualenv can
    sit at any depth, and ``site-packages`` is always several levels down.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    current = root
    for part in relative.parts[:-1] if path.is_file() else relative.parts:
        current = current / part
        if is_generated_dir(current):
            return True
    return False


def source_packages(root: Path) -> list[str]:
    """First-party package names, read from the manifest.

    Derived rather than hardcoded so that adding a package cannot leave a
    scanner asserting against a stale list.
    """
    manifest = root / "pyproject.toml"
    if not manifest.is_file():
        return []
    with manifest.open("rb") as handle:
        data = tomllib.load(handle)
    return list(data.get("tool", {}).get("setuptools", {}).get("packages", []))


def source_roots(root: Path) -> list[Path]:
    """Every directory that holds project source."""
    names = [*source_packages(root), *NON_PACKAGE_SOURCE_DIRS]
    return [root / name for name in names if (root / name).is_dir()]


def iter_project_files(
    root: Path,
    *,
    suffixes: tuple[str, ...] = (".py",),
    include_tests: bool = True,
    include_tools: bool = True,
) -> Iterator[Path]:
    """Yield project source files, never entering a generated directory.

    Walks only the roots derived from the manifest plus the known non-package
    source directories, and skips generated directories at every level. A
    virtualenv created inside the project -- which normal validation does --
    is therefore invisible to every caller.
    """
    skip = set()
    if not include_tests:
        skip.add("tests")
    if not include_tools:
        skip.add("tools")

    for directory in source_roots(root):
        if directory.name in skip:
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if is_inside_generated(path, root):
                continue
            yield path

    for name in ROOT_SOURCE_FILES:
        candidate = root / name
        if candidate.is_file() and candidate.suffix in suffixes:
            yield candidate


def iter_production_files(
    root: Path, *, suffixes: tuple[str, ...] = (".py",)
) -> Iterator[Path]:
    """Project files excluding test scaffolding.

    ``conftest.py`` counts as scaffolding: it imports pytest and describes
    checkout problems in prose, neither of which belongs to the shipped
    system.
    """
    for path in iter_project_files(
        root, suffixes=suffixes, include_tests=False, include_tools=False
    ):
        if path.name == "conftest.py":
            continue
        yield path


def copytree_ignore(directory: str, entries: list[str]) -> set[str]:
    """``shutil.copytree`` ignore callable that skips generated directories.

    Used by the mutation harness so scratch copies contain the project and
    nothing else -- copying a virtualenv would be slow, and copying a stale
    one would silently change what the tests import.
    """
    base = Path(directory)
    ignored = {
        entry
        for entry in entries
        if (base / entry).is_dir() and is_generated_dir(base / entry)
    }
    ignored |= {
        entry
        for entry in entries
        if entry.endswith((".pyc", ".pyo", ".db", ".db-wal", ".db-shm"))
    }
    return ignored


__all__ = [
    "GENERATED_DIR_NAMES",
    "GENERATED_DIR_SUFFIXES",
    "NON_PACKAGE_SOURCE_DIRS",
    "ROOT_SOURCE_FILES",
    "copytree_ignore",
    "is_generated_dir",
    "is_inside_generated",
    "is_virtualenv",
    "iter_production_files",
    "iter_project_files",
    "source_packages",
    "source_roots",
]
