"""Phase 4: controlled execution.

Human approval is mandatory and is bound to one exact operation — its type,
SKU, marketplace and payload fingerprint. Any material change invalidates it.
Every gate is rechecked immediately before an attempt, so a stale approval
cannot carry a product past a condition that changed after it was granted.

Dry run is the default and the only mode enabled unless the environment
explicitly says otherwise. In dry run no adapter is called at all.
"""

from execution.engine import (
    BLOCKING_COMPLIANCE,
    NON_EXECUTABLE_VERDICTS,
    ExecutionContext,
    ExecutionEngine,
    evaluate_gates,
)
from execution.model import (
    LEGAL_TRANSITIONS,
    LIVE_ENABLED_ENV_VAR,
    OPERATION_CAPABILITY,
    Approval,
    AuditEntry,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    GateOutcome,
    GateResult,
    IllegalTransition,
    OperationType,
    RetryClass,
    build_idempotency_key,
    fingerprint,
    live_execution_enabled,
)

EXECUTION_LAYER_VERSION = "execution/1.0.0"

__all__ = [
    "BLOCKING_COMPLIANCE",
    "EXECUTION_LAYER_VERSION",
    "LEGAL_TRANSITIONS",
    "LIVE_ENABLED_ENV_VAR",
    "NON_EXECUTABLE_VERDICTS",
    "OPERATION_CAPABILITY",
    "Approval",
    "AuditEntry",
    "ExecutionContext",
    "ExecutionEngine",
    "ExecutionMode",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "GateOutcome",
    "GateResult",
    "IllegalTransition",
    "OperationType",
    "RetryClass",
    "build_idempotency_key",
    "evaluate_gates",
    "fingerprint",
    "live_execution_enabled",
]
