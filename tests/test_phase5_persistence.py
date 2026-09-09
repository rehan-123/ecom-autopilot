"""Phase 5: the database arbitrates idempotency, not any one process.

Phase 4's guarantee held within a worker and was worthless across two: each
process had its own store, both saw a free key, both would execute. These
tests use real subprocesses so the claim is proved where it actually matters.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from database.connection import create_db_engine
from execution.ledger import (
    PersistentIdempotencyStore,
    ReconciliationState,
    execution_ledger,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def store(tmp_path: Path) -> PersistentIdempotencyStore:
    return PersistentIdempotencyStore(create_db_engine(tmp_path / "exec.db"))


def a_claim(
    store: PersistentIdempotencyStore, key: str = "key-1", execution_id: str = "e1"
):
    return store.claim(
        idempotency_key=key,
        execution_id=execution_id,
        operation="PRICE_UPDATE",
        sku_code="SKU-1",
        marketplace="amazon_in",
        payload_fingerprint="fp",
        execution_mode="DRY_RUN",
    )


# ==========================================================================
# Atomic claim
# ==========================================================================


def test_a_claim_is_recorded_durably(store):
    assert a_claim(store).won
    row = store.get("key-1")
    assert row["execution_state"] == "EXECUTING"
    assert row["reconciliation_state"] == "NOT_REQUIRED"


def test_a_second_claim_on_the_same_key_loses(store):
    assert a_claim(store, execution_id="first").won
    second = a_claim(store, execution_id="second")
    assert not second.won
    assert second.existing_state == "EXECUTING"
    assert "already been claimed" in second.detail


def test_the_loser_receives_the_winners_execution(store):
    a_claim(store, execution_id="winner")
    loser = a_claim(store, execution_id="loser")
    assert loser.execution_id == "winner"


def test_different_keys_do_not_collide(store):
    assert a_claim(store, key="a").won
    assert a_claim(store, key="b").won


# ==========================================================================
# Multi-process concurrency â€” the claim Phase 4 could not make
# ==========================================================================


CHILD = """
import sys

# The child is a fresh interpreter with no inherited sys.path, so the project
# root is passed in explicitly rather than assumed.
sys.path.insert(0, sys.argv[3])

from database.connection import create_db_engine
from execution.ledger import PersistentIdempotencyStore

store = PersistentIdempotencyStore(create_db_engine(sys.argv[1]))
result = store.claim(
    idempotency_key="shared-key", execution_id=sys.argv[2],
    operation="PRICE_UPDATE", sku_code="SKU-1", marketplace="amazon_in",
    payload_fingerprint="fp", execution_mode="DRY_RUN",
)
print("WON" if result.won else "LOST")
"""


def test_separate_processes_produce_exactly_one_winner(tmp_path: Path):
    """Real subprocesses, not threads.

    Threading proves nothing about a multi-worker deployment: two threads
    share a store, two processes do not. This is the test that justifies the
    guarantee.
    """
    database = tmp_path / "exec.db"
    PersistentIdempotencyStore(create_db_engine(database))  # create the table

    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(database),
                f"proc-{i}",
                str(PROJECT_ROOT),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT,
        )
        for i in range(6)
    ]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=90)
        assert process.returncode == 0, stderr[-400:]
        outputs.append(stdout.strip())

    assert outputs.count("WON") == 1, f"expected exactly one winner, got {outputs}"
    assert outputs.count("LOST") == 5


def test_the_winner_is_the_only_row_in_the_ledger(tmp_path: Path):
    database = tmp_path / "exec.db"
    store = PersistentIdempotencyStore(create_db_engine(database))
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")

    for i in range(4):
        subprocess.run(
            [
                sys.executable,
                str(script),
                str(database),
                f"proc-{i}",
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
            cwd=PROJECT_ROOT,
        )
    assert len(store.history()) == 1


# ==========================================================================
# Crash recovery
# ==========================================================================


def test_a_stale_execution_is_not_re_executed(store):
    """A dead process's row must never be assumed successful or retried."""
    a_claim(store)
    with store.engine.begin() as conn:
        conn.execute(
            execution_ledger.update()
            .where(execution_ledger.c.idempotency_key == "key-1")
            .values(started_at=datetime.now(UTC) - timedelta(hours=2))
        )

    recovered = store.recover_stale()
    assert recovered == ["key-1"]

    row = store.get("key-1")
    assert row["execution_state"] == "REQUIRES_REVIEW"
    assert row["outcome"] == "UNKNOWN_OUTCOME"
    assert row["side_effect_occurred"] is False
    assert "unknown" in row["error_message"].lower()


def test_a_recent_execution_is_not_treated_as_stale(store):
    a_claim(store)
    assert store.recover_stale() == []
    assert store.get("key-1")["execution_state"] == "EXECUTING"


def test_a_recovered_execution_cannot_simply_be_retried(store):
    a_claim(store)
    with store.engine.begin() as conn:
        conn.execute(
            execution_ledger.update()
            .where(execution_ledger.c.idempotency_key == "key-1")
            .values(started_at=datetime.now(UTC) - timedelta(hours=2))
        )
    store.recover_stale()
    allowed, why = store.may_retry("key-1")
    assert not allowed
    assert "duplicate" in why.lower()


# ==========================================================================
# Reconciliation
# ==========================================================================


def test_only_a_confirmed_failure_permits_a_retry(store):
    a_claim(store)
    store.set_reconciliation("key-1", ReconciliationState.REQUIRES_REVIEW)
    assert not store.may_retry("key-1")[0]

    store.set_reconciliation("key-1", ReconciliationState.RECONCILING)
    assert not store.may_retry("key-1")[0]

    store.set_reconciliation("key-1", ReconciliationState.CONFIRMED_SUCCESS)
    assert not store.may_retry("key-1")[0]

    store.set_reconciliation("key-1", ReconciliationState.CONFIRMED_FAILURE)
    allowed, why = store.may_retry("key-1")
    assert allowed
    assert "did not take effect" in why


def test_an_unseen_key_may_execute(store):
    allowed, _ = store.may_retry("never-seen")
    assert allowed


def test_reconciliation_states_are_explicit():
    assert ReconciliationState.CONFIRMED_FAILURE.safe_to_retry
    for state in (
        ReconciliationState.REQUIRES_REVIEW,
        ReconciliationState.RECONCILING,
        ReconciliationState.CONFIRMED_SUCCESS,
        ReconciliationState.NOT_REQUIRED,
    ):
        assert not state.safe_to_retry


# ==========================================================================
# Ledger hygiene
# ==========================================================================


def test_the_ledger_holds_no_credential_field(store):
    a_claim(store)
    row = store.get("key-1")
    for column in row:
        assert not any(
            term in column.lower()
            for term in ("secret", "token", "password", "credential", "api_key")
        )


def test_history_is_bounded(store):
    for i in range(20):
        a_claim(store, key=f"k{i}", execution_id=f"e{i}")
    assert len(store.history(limit=5)) == 5
    assert len(store.history(limit=10_000)) <= 500


def test_installing_the_ledger_twice_is_safe(tmp_path: Path):
    """Startup must not destroy existing data."""
    database = tmp_path / "exec.db"
    first = PersistentIdempotencyStore(create_db_engine(database))
    a_claim(first)
    second = PersistentIdempotencyStore(create_db_engine(database))
    assert second.get("key-1") is not None


# ==========================================================================
# Phase 6: backup is non-destructive
# ==========================================================================


def test_backup_creates_a_readable_copy_without_touching_the_source(tmp_path: Path):
    import main

    database = tmp_path / "src.db"
    store = PersistentIdempotencyStore(create_db_engine(database))
    a_claim(store)
    before = database.read_bytes()

    target = tmp_path / "backup.db"
    assert main.cmd_backup(database, str(target)) == 0
    assert target.exists()
    assert database.read_bytes() == before, "the live database must not be modified"

    restored = PersistentIdempotencyStore(create_db_engine(target))
    assert restored.get("key-1") is not None


def test_backup_refuses_to_overwrite_an_existing_file(tmp_path: Path):
    import main

    database = tmp_path / "src.db"
    PersistentIdempotencyStore(create_db_engine(database))
    target = tmp_path / "existing.db"
    target.write_text("do not clobber me", encoding="utf-8")

    assert main.cmd_backup(database, str(target)) == 1
    assert target.read_text(encoding="utf-8") == "do not clobber me"


def test_backup_of_a_missing_database_fails_cleanly(tmp_path: Path):
    import main

    assert main.cmd_backup(tmp_path / "absent.db", str(tmp_path / "out.db")) == 1


# ==========================================================================
# Phase 7: destructive commands must not reach a live database
# ==========================================================================


def test_demo_refuses_to_reset_a_database_with_execution_history(tmp_path: Path):
    """The demo is destructive and shares the default path with a real install."""
    import main

    database = tmp_path / "live.db"
    store = PersistentIdempotencyStore(create_db_engine(database))
    a_claim(store)

    assert main.cmd_demo(database) == 1
    assert database.exists(), "the database must not have been deleted"
    assert store.get("key-1") is not None, "execution history must survive"


def test_demo_still_resets_a_database_without_execution_history(tmp_path: Path):
    """The guard must not break the legitimate demo case."""
    import main

    database = tmp_path / "scratch.db"
    engine = create_db_engine(database)
    PersistentIdempotencyStore(engine)
    engine.dispose()
    assert main.cmd_demo(database) == 0


def test_history_detection_tolerates_a_missing_table(tmp_path: Path):
    import main

    database = tmp_path / "bare.db"
    database.write_bytes(b"")
    assert main._has_execution_history(database) is False
