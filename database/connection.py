"""Engine and connection management.

SQLite defaults are wrong for this workload in two ways that matter, and both
are fixed per-connection here rather than hoped for:

* ``PRAGMA foreign_keys`` is OFF by default. Every foreign key in
  ``tables.py`` is decorative until it is switched on.
* The default journal mode blocks readers during writes. WAL lets the replay
  and projection reads proceed while events are being appended.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event, text

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/ecom_autopilot.db")

READ_ONLY_FLAG = "ecom_autopilot_read_only"
"""Connection-info key marking a connection as read-only."""


def _install_pragmas(engine: Engine) -> None:
    """Apply SQLite pragmas on every new connection."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def _begin(conn: Connection) -> None:
        # Writes take the lock up front so read-then-write is atomic.
        # Reads stay DEFERRED: making every transaction exclusive would
        # serialise readers against each other and deadlock a read issued
        # while another connection is open.
        if conn.info.get(READ_ONLY_FLAG):
            conn.exec_driver_sql("BEGIN")
        else:
            conn.exec_driver_sql("BEGIN IMMEDIATE")


def create_db_engine(
    db_path: Path | str = DEFAULT_DB_PATH, *, echo: bool = False
) -> Engine:
    """Create an engine for a SQLite database at ``db_path``.

    Pass ``":memory:"`` for an ephemeral database (used by the test suite).
    """
    if str(db_path) == ":memory:":
        url = "sqlite:///:memory:"
    else:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"

    engine = create_engine(url, echo=echo, future=True)
    _install_pragmas(engine)
    logger.debug("engine created", extra={"url": url})
    return engine


@contextmanager
def transaction(engine: Engine) -> Iterator[Connection]:
    """Run a unit of work in a transaction, committing on success.

    Used rather than autocommit so that an entity write and its ledger event
    either both land or neither does. A fact recorded without its event, or
    an event without its fact, is a broken audit trail.
    """
    with engine.begin() as conn:
        yield conn


@contextmanager
def read_only(engine: Engine) -> Iterator[Connection]:
    """Open a connection for reads, using a shared rather than write lock."""
    with engine.connect() as conn:
        conn.info[READ_ONLY_FLAG] = True
        yield conn


def foreign_keys_enabled(conn: Connection) -> bool:
    """Report whether FK enforcement is active on this connection."""
    return bool(conn.execute(text("PRAGMA foreign_keys")).scalar())
