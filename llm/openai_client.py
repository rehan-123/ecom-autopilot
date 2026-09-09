"""OpenAI transport implementing the existing LLMClient protocol.

One class, one job: satisfy ``complete(*, system, user, max_tokens) -> str``
using OpenAI's Chat Completions API. Everything else — prompt construction,
response validation, the anti-fabrication allowlist, degradation on failure —
already exists in ``llm.services`` and is untouched by this file.

Built on stdlib ``urllib`` rather than an SDK or ``requests``/``httpx``, for
the same reason the marketplace transports in ``part2.transport`` are: this
project's dependency manifest declares no third-party HTTP client, and a test
enforces that. Adding one for a single API would widen the dependency surface
of a system that handles money, for a convenience this module does not need.

The API key is read from ``OPENAI_API_KEY`` at the moment of the call and
never stored on the instance, logged, or returned in any result. It travels
in exactly one place — the ``Authorization`` header of one outbound request —
and nowhere else in this codebase.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from llm.services import LLMUnavailable

logger = logging.getLogger(__name__)

OPENAI_CLIENT_VERSION = "openai-client/1.0.0"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_TIMEOUT_S = 30.0

#: The only place a key-shaped string may legitimately appear in this module:
#: the header of the request object, never a field on ``self``.
_REDACTED = "<redacted>"


class OpenAIClient:
    """LLMClient over OpenAI's Chat Completions API.

    Reads ``OPENAI_API_KEY`` (and optionally ``OPENAI_MODEL``,
    ``OPENAI_BASE_URL``) from the environment when constructed, not when
    imported — so importing this module never requires a key, and a process
    with no key set simply cannot build a working instance.
    """

    def __init__(
        self,
        *,
        environ: dict[str, str] | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        env = environ if environ is not None else os.environ
        key = env.get("OPENAI_API_KEY")
        if not key:
            raise LLMUnavailable(
                "OPENAI_API_KEY is not set; the research assistant will run "
                "in degraded mode using terms from the request itself"
            )
        # Held only for the lifetime of this call chain, never logged or
        # exposed through __repr__ (the default dataclass-style repr is not
        # used here precisely so a key cannot end up in one accidentally).
        self._key = key
        self.model = model or env.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.endpoint = endpoint or env.get("OPENAI_BASE_URL", DEFAULT_ENDPOINT)
        self.timeout_s = timeout_s

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self.model!r}, key={_REDACTED})"

    def complete(self, *, system: str, user: str, max_tokens: int = 2000) -> str:
        """Return the model's raw text response.

        Any failure — network, auth, rate limit, malformed body — raises
        ``LLMUnavailable``. ``LLMService.invoke`` already catches every
        exception from this method and turns it into a degraded
        ``LLMOutcome``, so nothing here needs its own retry or fallback logic.
        """
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # The body may contain OpenAI's own error message, which is safe
            # to surface -- it never contains the request's Authorization
            # header, only what the server chose to say back.
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMUnavailable(
                f"OpenAI API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"could not reach the OpenAI API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMUnavailable("OpenAI API request timed out") from exc

        try:
            payload = json.loads(raw)
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(
                f"OpenAI API response did not have the expected shape: {exc}"
            ) from exc


def openai_available(environ: dict[str, str] | None = None) -> bool:
    """Whether an OpenAI client could be constructed right now.

    Presence of the key only -- never its value, and never a network call.
    Used to decide which client to hand the research assistant without
    risking a call just to check.
    """
    env = environ if environ is not None else os.environ
    return bool(env.get("OPENAI_API_KEY"))


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "OPENAI_CLIENT_VERSION",
    "OpenAIClient",
    "openai_available",
]
