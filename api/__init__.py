"""HTTP application layer over the existing Ecom Autopilot engines.

Thin by design: ``api.service`` assembles domain state from the engines and
``api.app`` exposes it. No business logic lives here — prices, verdicts,
compliance and state transitions all come from the domain packages.
"""

API_LAYER_VERSION = "api/1.0.0"

__all__ = ["API_LAYER_VERSION"]
