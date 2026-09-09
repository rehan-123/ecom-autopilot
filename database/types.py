"""Custom SQLAlchemy column types.

SQLite has no native datetime, UUID, decimal or JSON type. Left to defaults,
SQLAlchemy will happily round-trip a timezone-aware datetime as a naive one
and a Decimal as a float. Both failures are silent and both corrupt an audit
ledger, so every affected column goes through an explicit TypeDecorator.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import String, Text, TypeDecorator


class UTCDateTime(TypeDecorator):
    """Store timezone-aware datetimes as ISO-8601 UTC strings.

    Naive datetimes are rejected on write rather than assumed to be UTC.
    ISO-8601 in UTC sorts lexicographically in the same order it sorts
    chronologically, so ``ORDER BY`` on the raw column is correct.
    """

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"expected datetime, got {type(value).__name__}")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(
                "refusing to store a naive datetime; supply tzinfo explicitly"
            )
        return value.astimezone(UTC).isoformat(timespec="microseconds")

    def process_result_value(self, value: Any, dialect: Any) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)


class UUIDText(TypeDecorator):
    """Store UUIDs as canonical lowercase hyphenated text."""

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def process_result_value(self, value: Any, dialect: Any) -> uuid.UUID | None:
        return None if value is None else uuid.UUID(value)


class DecimalText(TypeDecorator):
    """Store Decimals as text to preserve exact value and scale.

    SQLite's NUMERIC affinity converts to float, which loses precision. Money
    accumulated over a long ledger must not drift, so the string form is the
    stored form.
    """

    impl = String(40)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, float):
            raise TypeError("refusing to store a float as a monetary value; use Decimal")
        return str(Decimal(str(value)))

    def process_result_value(self, value: Any, dialect: Any) -> Decimal | None:
        return None if value is None else Decimal(value)


class JSONText(TypeDecorator):
    """Store structured payloads as canonical JSON.

    ``sort_keys=True`` makes the serialisation deterministic, which matters
    because the ledger hash chain is computed over these payloads.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        return None if value is None else json.loads(value)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON encoding, used for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
