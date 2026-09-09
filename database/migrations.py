"""Database initialisation: tables, immutability triggers, indexes, versioning.

Procedural counterpart to the declarative ``tables.py``.

Immutability is enforced at the database layer, not just in Python. Python
guarantees hold only for code that goes through the repository; a stray
``UPDATE`` from a SQL console, a future module, or a coding agent would
otherwise silently rewrite history. Triggers refuse both.

Honest limitation: triggers are a strong guardrail, not a cryptographic one.
Anyone with file access can drop a trigger or edit the file directly. That is
what the ledger hash chain is for -- it cannot prevent tampering, but it makes
tampering detectable. See ``ledger.events.verify_chain``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import Connection, Engine, inspect, select, text

from database.tables import (
    IMMUTABLE_TABLES,
    INDEXES,
    PROJECTION_TABLES,
    metadata,
    schema_version,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SCHEMA_DESCRIPTION = (
    "Milestone 1: catalog entities, versioned reference data, assessment "
    "structures, append-only audit ledger, projections"
)


def _trigger_names(table: str) -> tuple[str, str]:
    return f"trg_{table}_no_update", f"trg_{table}_no_delete"


def install_immutability_triggers(conn: Connection) -> list[str]:
    """Install BEFORE UPDATE / BEFORE DELETE guards on immutable tables.

    Returns the names of the triggers now present. Idempotent.
    """
    installed: list[str] = []
    for table in IMMUTABLE_TABLES:
        update_trg, delete_trg = _trigger_names(table)
        conn.execute(
            text(
                f"""
                CREATE TRIGGER IF NOT EXISTS {update_trg}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT,
                        '{table} is append-only: UPDATE is forbidden');
                END;
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TRIGGER IF NOT EXISTS {delete_trg}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT,
                        '{table} is append-only: DELETE is forbidden');
                END;
                """
            )
        )
        installed.extend((update_trg, delete_trg))
    logger.info("immutability triggers installed", extra={"count": len(installed)})
    return installed


def create_indexes(conn: Connection) -> list[str]:
    """Create declared indexes. Idempotent."""
    created: list[str] = []
    for index in INDEXES:
        index.create(bind=conn, checkfirst=True)
        created.append(index.name)
    return created


def _record_schema_version(conn: Connection) -> None:
    """Record the applied schema version, once."""
    existing = conn.execute(
        select(schema_version.c.version).where(schema_version.c.version == SCHEMA_VERSION)
    ).first()
    if existing is not None:
        return
    conn.execute(
        schema_version.insert().values(
            version=SCHEMA_VERSION,
            applied_at=datetime.now(UTC),
            description=SCHEMA_DESCRIPTION,
        )
    )


def init_db(engine: Engine) -> dict[str, object]:
    """Create the database. Safe to run repeatedly.

    Creates tables, installs immutability triggers, creates indexes and
    records the schema version. Returns a summary for the CLI to print.
    """
    with engine.begin() as conn:
        metadata.create_all(bind=conn, checkfirst=True)
        triggers = install_immutability_triggers(conn)
        indexes = create_indexes(conn)
        _record_schema_version(conn)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tables": len(metadata.tables),
        "immutable_tables": len(IMMUTABLE_TABLES),
        "triggers": len(triggers),
        "indexes": len(indexes),
        "projections": len(PROJECTION_TABLES),
    }
    logger.info("database initialised", extra=summary)
    return summary


def get_schema_version(engine: Engine) -> int | None:
    """Return the highest applied schema version, or None if uninitialised."""
    if not inspect(engine).has_table(schema_version.name):
        return None
    with engine.connect() as conn:
        return conn.execute(
            select(schema_version.c.version).order_by(schema_version.c.version.desc())
        ).scalar()


def list_tables(engine: Engine) -> list[str]:
    """Return every table currently present."""
    return sorted(inspect(engine).get_table_names())


def list_indexes(engine: Engine, table: str) -> list[str]:
    """Return index names on ``table``."""
    return sorted(idx["name"] for idx in inspect(engine).get_indexes(table))


def list_triggers(engine: Engine) -> list[str]:
    """Return every trigger currently installed."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'trigger' ORDER BY name")
        ).fetchall()
    return [row[0] for row in rows]
