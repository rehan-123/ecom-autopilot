"""The four Claude call sites, with validation and safe degradation."""

from llm.services import (
    LLMClient,
    LLMOutcome,
    LLMService,
    LLMUnavailable,
    NullLLMClient,
    RecordedLLMClient,
    build_commercial_prompt,
    build_extractor_prompt,
    build_identity_prompt,
    build_writer_prompt,
)

__all__ = [
    "LLMClient",
    "LLMOutcome",
    "LLMService",
    "LLMUnavailable",
    "NullLLMClient",
    "RecordedLLMClient",
    "build_commercial_prompt",
    "build_extractor_prompt",
    "build_identity_prompt",
    "build_writer_prompt",
]
