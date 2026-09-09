"""Reports are produced by decisions.pipeline; this package re-exports them."""

from decisions.pipeline import detail_report, ranked_report, rejection_report

__all__ = ["detail_report", "ranked_report", "rejection_report"]
