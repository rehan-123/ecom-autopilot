"""AI research assistant: hypotheses from a model, facts from sources.

A language model suggests what to investigate. It never supplies a price, a
cost, an MOQ, a demand figure or a supplier — those are gathered from sources
or recorded as UNAVAILABLE. The export path enforces that separation, so a
suggestion cannot acquire the authority of a measurement.
"""

from research.assistant import (
    Claim,
    ParsedRequest,
    Provenance,
    ResearchAssistant,
    ResearchCandidate,
    ResearchError,
    ResearchRun,
    build_researcher,
    default_researcher,
)
from research.export import observation_rows, preview, trends_rows, write_observations

RESEARCH_PACKAGE_VERSION = "research/1.0.0"

__all__ = [
    "RESEARCH_PACKAGE_VERSION",
    "Claim",
    "ParsedRequest",
    "Provenance",
    "ResearchAssistant",
    "ResearchCandidate",
    "ResearchError",
    "ResearchRun",
    "build_researcher",
    "default_researcher",
    "observation_rows",
    "preview",
    "trends_rows",
    "write_observations",
]
