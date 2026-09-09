"""Tests for the OpenAI transport.

Two kinds of assertion matter here. The first is about the key: it must be
readable only from the environment, must never appear in a log, a repr, an
exception message beyond what OpenAI itself returned, or a result the API
sends to the frontend. The second is that plugging in a real-shaped transport
does not weaken anything upstream -- the anti-fabrication validator in
``llm.services`` must reject a fabricated field from this client exactly as it
rejects one from the recorded test client, because the validator does not
know or care which transport produced the response.

No test here makes a real network call. ``urllib.request.urlopen`` is patched
in every case; this proves request construction and response handling are
correct, not that OpenAI's API behaves as documented.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from catalog.enums import AIService
from llm.openai_client import (
    DEFAULT_MODEL,
    OpenAIClient,
    openai_available,
)
from llm.services import LLMService, LLMUnavailable
from research.assistant import default_researcher

# ==========================================================================
# Construction and key handling
# ==========================================================================


def test_client_refuses_to_build_without_a_key():
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY is not set"):
        OpenAIClient(environ={})


def test_client_builds_with_a_key_present():
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-key"})
    assert client.model == DEFAULT_MODEL


def test_key_never_appears_in_repr():
    """A key in a repr ends up in a log the moment someone prints the object."""
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-super-secret"})
    assert "PLACEHOLDER-super-secret" not in repr(client)
    assert "<redacted>" in repr(client)


def test_key_is_not_a_public_attribute_by_the_obvious_name():
    """A stray `client.key` or `client.api_key` access would be too easy."""
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"})
    assert not hasattr(client, "key")
    assert not hasattr(client, "api_key")


def test_model_and_endpoint_are_overridable_from_the_environment():
    client = OpenAIClient(
        environ={
            "OPENAI_API_KEY": "PLACEHOLDER-x",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_BASE_URL": "https://example.test/v1/chat/completions",
        }
    )
    assert client.model == "gpt-4o"
    assert client.endpoint == "https://example.test/v1/chat/completions"


def test_openai_available_checks_presence_only_never_calls_out(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("openai_available must not make a network call")

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)
    assert not openai_available({})
    assert openai_available({"OPENAI_API_KEY": "PLACEHOLDER-x"})


# ==========================================================================
# A successful call: request shape and response parsing
# ==========================================================================


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def _ok(content: dict) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps(content)}}]}
    ).encode("utf-8")


def test_successful_call_returns_the_message_content(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(_ok({"candidates": []}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"})
    result = client.complete(system="SYS", user="USER")

    assert json.loads(result) == {"candidates": []}
    request = captured["request"]
    assert request.get_header("Authorization") == "Bearer PLACEHOLDER-x"
    assert request.get_header("Content-type") == "application/json"
    body = json.loads(request.data)
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0


def test_max_tokens_is_forwarded(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data)
        return _FakeResponse(_ok({"candidates": []}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"}).complete(
        system="s", user="u", max_tokens=500
    )
    assert captured["body"]["max_tokens"] == 500


def test_authorization_header_never_leaks_into_the_returned_text(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_ok({"candidates": []}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-super-secret-abc"})
    result = client.complete(system="s", user="u")
    assert "PLACEHOLDER-super-secret-abc" not in result


# ==========================================================================
# Failure modes -- every one must degrade, never raise past LLMService
# ==========================================================================


def test_http_error_becomes_llm_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            __import__("io").BytesIO(b'{"error": {"message": "Invalid API key"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-bad"})
    with pytest.raises(LLMUnavailable, match="HTTP 401"):
        client.complete(system="s", user="u")


def test_http_error_body_does_not_contain_the_key(monkeypatch):
    """The error message surfaces what OpenAI said, never what we sent."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            __import__("io").BytesIO(b'{"error": {"message": "Invalid API key"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-super-secret"})
    with pytest.raises(LLMUnavailable) as excinfo:
        client.complete(system="s", user="u")
    assert "PLACEHOLDER-super-secret" not in str(excinfo.value)


def test_rate_limit_becomes_llm_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            __import__("io").BytesIO(b"{}"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMUnavailable, match="429"):
        OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"}).complete(
            system="s", user="u"
        )


def test_network_unreachable_becomes_llm_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMUnavailable, match="could not reach"):
        OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"}).complete(
            system="s", user="u"
        )


def test_timeout_becomes_llm_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMUnavailable, match="timed out"):
        OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"}).complete(
            system="s", user="u"
        )


def test_malformed_response_shape_becomes_llm_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(json.dumps({"unexpected": "shape"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMUnavailable, match="expected shape"):
        OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"}).complete(
            system="s", user="u"
        )


def test_non_json_response_body_becomes_llm_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"not json at all")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMUnavailable):
        OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"}).complete(
            system="s", user="u"
        )


# ==========================================================================
# End to end through LLMService: degradation is invisible from the outside
# ==========================================================================


def test_llmservice_degrades_on_a_real_shaped_401(monkeypatch):
    """LLMService.invoke already catches every exception; confirm this one too."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            __import__("io").BytesIO(b"{}"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-bad"})
    outcome = LLMService(AIService.RESEARCHER, client).invoke("x")
    assert not outcome.usable
    assert outcome.error is not None
    assert "401" in outcome.error


def test_the_anti_fabrication_validator_applies_to_openai_responses_too(monkeypatch):
    """The validator does not know or care which transport produced this.

    A real model asked to research products will happily supply a price if
    not stopped; the allowlist in llm.services rejects it regardless of
    whether the text came from the recorded test client or a live API.
    """

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(
            _ok(
                {
                    "candidates": [
                        {
                            "name": "Rechargeable Hair Trimmer",
                            "category": "personal_care_electrical",
                            "rationale": "steady demand",
                            "search_terms": ["trimmer"],
                            "confidence": "high",
                            "price": 499,  # a real model would do exactly this
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"})
    outcome = LLMService(AIService.RESEARCHER, client).invoke("x")
    assert not outcome.usable
    assert any("price" in v for v in outcome.schema_violations)
    assert outcome.findings == []


# ==========================================================================
# default_researcher(): the wiring the API endpoint uses
# ==========================================================================


def test_default_researcher_falls_back_to_null_client_with_no_key():
    service = default_researcher(environ={})
    outcome = service.invoke("x")
    assert not outcome.usable
    assert "no LLM client configured" in (outcome.error or "")


def test_default_researcher_uses_openai_when_a_key_is_present(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_ok({"candidates": []}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    service = default_researcher(environ={"OPENAI_API_KEY": "PLACEHOLDER-x"})
    outcome = service.invoke("x")
    assert outcome.usable


def test_default_researcher_never_raises_even_when_the_key_is_invalid(monkeypatch):
    """The API endpoint calls this with no try/except of its own; it must not need one."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            __import__("io").BytesIO(b"{}"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    service = default_researcher(environ={"OPENAI_API_KEY": "PLACEHOLDER-wrong"})
    outcome = service.invoke("x")
    assert not outcome.usable
