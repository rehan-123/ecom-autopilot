"""Discovery source adapters. See discovery.registry."""

from discovery.registry import (
    KNOWN_SOURCES,
    DiscoveryAdapter,
    DiscoveryCandidate,
    DiscoveryRegistry,
    registry,
)

__all__ = [
    "KNOWN_SOURCES",
    "DiscoveryAdapter",
    "DiscoveryCandidate",
    "DiscoveryRegistry",
    "registry",
]
