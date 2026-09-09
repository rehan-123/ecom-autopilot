"""Compliance evaluation over versioned rules-as-data."""

from compliance.engine import (
    COMPLIANCE_VERSION,
    CheckOutcome,
    ComplianceOutcome,
    Requirement,
    Severity,
    evaluate,
)

__all__ = [
    "COMPLIANCE_VERSION",
    "CheckOutcome",
    "ComplianceOutcome",
    "Requirement",
    "Severity",
    "evaluate",
]
