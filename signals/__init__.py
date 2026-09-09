"""Market evidence: signals, price reference, coverage, scoring and risk."""

from signals.market import (
    DEFAULT_WEIGHTS,
    SIGNAL_TYPES,
    Comparable,
    HeadroomOutcome,
    MarketPriceReference,
    OpportunityScore,
    RiskAssessment,
    ScoreComponent,
    Signal,
    assess_risk,
    build_components,
    check_price_headroom,
    compute_opportunity_score,
    unavailable,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "SIGNAL_TYPES",
    "Comparable",
    "HeadroomOutcome",
    "MarketPriceReference",
    "OpportunityScore",
    "RiskAssessment",
    "ScoreComponent",
    "Signal",
    "assess_risk",
    "build_components",
    "check_price_headroom",
    "compute_opportunity_score",
    "unavailable",
]
