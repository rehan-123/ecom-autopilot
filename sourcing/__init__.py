"""Live product discovery, supplier matching and opportunity ranking.

Sits between Part 1's decision engine and Part 2's operating layer: it finds
candidate products from legitimate sources, works out who can actually supply
them, runs the existing economics, and ranks what survives.

No money logic lives here. Landed cost, scenarios, gates, compliance and
scoring all come from the Part 1 engines, so there is one definition of what a
product costs and one of whether it clears policy.
"""

from sourcing.feeds import FeedResult, build_registry, collect, coverage_report
from sourcing.matching import (
    Match,
    MatchConfidence,
    ProductGroup,
    SupplierCandidate,
    group_products,
    match_suppliers,
)
from sourcing.model import DiscoveredProduct, FieldSource, Observed, unavailable
from sourcing.opportunities import (
    Opportunity,
    OpportunityPolicy,
    Verdict,
    evaluate_group,
    opportunity_report,
    rank_opportunities,
    recommendation_report,
)

SOURCING_VERSION = "sourcing/1.0.0"

__all__ = [
    "SOURCING_VERSION",
    "DiscoveredProduct",
    "FeedResult",
    "FieldSource",
    "Match",
    "MatchConfidence",
    "Observed",
    "Opportunity",
    "OpportunityPolicy",
    "ProductGroup",
    "SupplierCandidate",
    "Verdict",
    "build_registry",
    "collect",
    "coverage_report",
    "evaluate_group",
    "group_products",
    "match_suppliers",
    "opportunity_report",
    "rank_opportunities",
    "recommendation_report",
    "unavailable",
]
