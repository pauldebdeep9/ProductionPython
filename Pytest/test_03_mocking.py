"""
tests/unit/test_03_mocking.py — where to draw the mock boundary.

Run:  pytest tests/unit/test_03_mocking.py -v

THE DECISION THAT MATTERS
-------------------------
Not "should I mock?" but "WHERE?". Mock too deep and your tests assert that
your code calls its own functions in a particular order — which breaks on
every refactor and proves nothing. Mock too shallow and your tests need a
network.

THE RULE: mock at the boundary you do not own.

    ┌──────────────────────────────────────────────────┐
    │  your code                                        │
    │    RagService.answer                              │
    │    RagService.parse          <- NEVER mock these  │
    │    the repair loop                                │
    ├──────────────────────────────────────────────────┤
    │  YOUR PROTOCOLS  (Retriever, ModelClient)         │
    │    <- substitute a FAKE here                      │
    ├──────────────────────────────────────────────────┤
    │  the HTTP client                                  │
    │    <- or intercept here, with respx/vcr           │
    ├──────────────────────────────────────────────────┤
    │  Azure OpenAI, Azure AI Search  (not yours)       │
    └──────────────────────────────────────────────────┘

FAKE vs MOCK, because the words get used interchangeably and should not be:

    FAKE   a working implementation with real behaviour, simplified.
           `InMemoryRetriever` genuinely filters by permission.
           You assert on OUTCOMES.
    MOCK   an object that records calls and returns canned values.
           You assert on INTERACTIONS ("was it called with X?").
    STUB   returns canned values, records nothing.

PREFER FAKES. A test suite built on mocks asserts that your code makes
particular calls; a suite built on fakes asserts that it produces particular
results. Only the second survives a refactor.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from rag.service import (
    Chunk,
    InMemoryRetriever,
    ModelClient,
    PermissionViolation,
    RagService,
    Retriever,
    ScriptedModel,
    good_response,
)


# ===========================================================================
# 1. THE SAME TEST, THREE WAYS
# ===========================================================================

async def test_with_a_fake(make_service) -> None:
    """A FAKE. Asserts on the OUTCOME.

    Survives: renaming internal methods, reordering the repair loop, changing
    how the prompt is built, extracting a helper.
    Fails when: the behaviour actually changes.
    """
    svc = make_service(["not json", good_response()])
    answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))
    assert answer.repairs == 1
    assert answer.proposal is not None
    assert answer.proposal.disposition == "adjust_quantity"


async def test_with_a_mock_asserting_interactions() -> None:
    """A MOCK. Asserts on the CALLS.

    This is legitimate here — "the retriever was called with the caller's
    groups" is a SECURITY property about an interaction, not an implementation
    detail. That is the narrow case where interaction assertions earn their
    place.
    """
    retriever = create_autospec(Retriever, instance=True)
    retriever.search = AsyncMock(return_value=[
        Chunk("c1", "po-1001", "PO 1001 quantity 400", frozenset({"isc-all"}))
    ])
    model = ScriptedModel([good_response()])
    svc = RagService(retriever, model)

    groups = frozenset({"isc-all"})
    await svc.answer("invoice 88 units", groups)

    retriever.search.assert_awaited_once()
    _, kwargs = retriever.search.call_args
    args, _ = retriever.search.call_args
    assert groups in args, (
        "the caller's group set must reach the retriever — without it, "
        "permission trimming cannot happen"
    )


async def test_with_a_mock_asserting_the_wrong_thing() -> None:
    """THE ANTI-PATTERN, shown so it is recognisable.

    This asserts that `parse` was called exactly three times — which is an
    implementation detail of the repair loop. Rewrite the loop to validate
    incrementally, or to parse once and repair the object, and this test
    fails while the behaviour is unchanged and correct.

    A test that fails on a correct refactor is a test that will be deleted,
    and the deletion takes the useful tests around it.
    """
    svc = RagService(InMemoryRetriever(), ScriptedModel(["bad", "bad", good_response()]))
    calls: list[str] = []
    real_parse = RagService.parse

    def counting_parse(text: str):
        calls.append(text[:12])
        return real_parse(text)

    # Patching a STATIC METHOD ON YOUR OWN CLASS is the smell. It couples the
    # test to the internal call graph.
    RagService.parse = staticmethod(counting_parse)  # type: ignore[method-assign]
    try:
        await svc.answer("invoice 88 units", frozenset({"isc-all"}))
        assert len(calls) == 3, "brittle: asserts on the internal call count"
    finally:
        RagService.parse = staticmethod(real_parse)  # type: ignore[method-assign]


# ===========================================================================
# 2. MagicMock SATISFIES EVERY INTERFACE AND NONE
# ===========================================================================

async def test_magicmock_accepts_anything() -> None:
    """The problem with a bare MagicMock: it never says no.

    It accepts a call with the wrong argument names, returns a Mock for any
    attribute, and drifts silently when the real interface changes. A test
    using one can pass against code that could not possibly work.
    """
    m = MagicMock()
    m.search(completely_wrong_argument=1, another=2)
    m.method_that_does_not_exist()
    assert m.anything.at.all.nested is not None
    # Every one of those succeeded. None of them would against the real thing.


def test_autospec_rejects_the_wrong_signature() -> None:
    """`create_autospec` builds a mock CONSTRAINED to the real signature.

    If you must use a mock, use an autospec'd one. It catches the case where
    the interface changed and the test did not.
    """
    retriever = create_autospec(Retriever, instance=True)
    with pytest.raises(TypeError):
        retriever.search(completely_wrong_argument=1)


async def test_a_handwritten_fake_is_checked_by_the_type_checker() -> None:
    """Better still: a hand-written fake, verified by mypy against the
    Protocol.

    `InMemoryRetriever` and `ScriptedModel` ship WITH the production code
    (rag/service.py), so a Protocol change breaks them in CI — in the same
    package, immediately. A mock in a conftest is checked by nothing.
    """
    fake: Retriever = InMemoryRetriever()
    chunks = await fake.search("invoice 88 units", frozenset({"isc-all"}), k=4)
    assert chunks
    assert all(c.visible_to(frozenset({"isc-all"})) for c in chunks)


# ===========================================================================
# 3. monkeypatch — for things you do not own and cannot inject
# ===========================================================================

def test_monkeypatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars, module attributes, and dict entries. Auto-reversed."""
    import os
    monkeypatch.setenv("RAG_DEPLOYMENT", "gpt-4o-mini-test")
    assert os.environ["RAG_DEPLOYMENT"] == "gpt-4o-mini-test"


def test_monkeypatch_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """`setattr` with a STRING target patches where the name is LOOKED UP,
    not where it is defined. That distinction causes most patching confusion:

        # module_under_test.py
        from time import time          # bound at import
        ...
        monkeypatch.setattr("time.time", fake)             # NO EFFECT
        monkeypatch.setattr("module_under_test.time", fake) # works

    The rule: patch the name in the module that USES it.
    """
    import rag.service as svc_module
    monkeypatch.setattr(svc_module, "CORPUS", [
        Chunk("z1", "doc-z", "a replaced corpus", frozenset({"isc-all"}))
    ])
    assert svc_module.CORPUS[0].chunk_id == "z1"


def test_monkeypatch_was_reversed() -> None:
    import rag.service as svc_module
    assert svc_module.CORPUS[0].chunk_id == "c1"


def test_monkeypatch_raising_guards_against_typos(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """`raising=True` (the default) errors if the attribute does not exist.

    Keep it. With `raising=False`, a typo in the attribute name silently
    patches nothing and the test passes against unpatched code — which is a
    green test that verifies nothing.
    """
    import rag.service as svc_module
    with pytest.raises(AttributeError):
        monkeypatch.setattr(svc_module, "CORPSU", [])


# ===========================================================================
# 4. INTERCEPTING AT THE HTTP EDGE
# ===========================================================================

class _FakeHttpTransport:
    """Stands in for `respx` / `vcr.py` / `aioresponses`.

    WHEN TO INTERCEPT AT THE HTTP LAYER rather than substituting a Protocol:

      * you are testing YOUR ADAPTER — the code that builds the request and
        parses the response. Substituting the Protocol skips exactly the code
        under test.
      * you need to exercise transport behaviour: a 429 with Retry-After,
        a truncated body, a connection reset mid-stream.
      * you want to verify the WIRE FORMAT: headers, auth, the JSON shape you
        actually send.

    WHEN NOT TO: for everything above the adapter. Testing `RagService`
    through an HTTP mock means every test carries JSON serialisation it does
    not care about, and a change to the wire format breaks a hundred tests
    instead of five.

    THE SHAPE: HTTP-level tests for the adapter (a handful), Protocol-level
    fakes for everything else (most of the suite).
    """

    def __init__(self, responses: list[tuple[int, dict[str, Any]]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self._i = 0

    async def post(self, url: str, *, json: dict[str, Any],
                   headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        self.requests.append({"url": url, "json": json, "headers": headers})
        status, body = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return status, body


class AzureOpenAIAdapter:
    """The code under test in this section: request building and response
    parsing. This is the ONLY layer where HTTP-level mocking belongs."""

    def __init__(self, transport: _FakeHttpTransport, deployment: str,
                 api_version: str = "2024-10-21") -> None:
        self.transport = transport
        self.deployment = deployment
        self.api_version = api_version

    async def complete(self, prompt: str, *,
                       max_tokens: int) -> tuple[str, int, int]:
        url = (f"https://x.openai.azure.com/openai/deployments/"
               f"{self.deployment}/chat/completions"
               f"?api-version={self.api_version}")
        status, body = await self.transport.post(
            url,
            json={"messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens},
            headers={"Authorization": "Bearer <token>"},
        )
        if status == 429:
            raise RuntimeError(f"429; retry after {body.get('retry_after')}")
        if status >= 500:
            raise RuntimeError(f"{status} upstream")
        choice = body["choices"][0]
        usage = body.get("usage", {})
        return (choice["message"]["content"],
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0))


def _ok_body(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40}}


async def test_adapter_builds_the_right_request() -> None:
    """HTTP-level: assert on the WIRE FORMAT.

    A Protocol-level fake cannot catch a wrong api-version, a missing header,
    or a malformed body — because it never sees them.
    """
    transport = _FakeHttpTransport([(200, _ok_body(good_response()))])
    adapter = AzureOpenAIAdapter(transport, "gpt-4o-mini-prod")

    await adapter.complete("hello", max_tokens=256)

    req = transport.requests[0]
    assert "deployments/gpt-4o-mini-prod/" in req["url"], (
        "the DEPLOYMENT name must be in the path — not the model name"
    )
    assert "api-version=2024-10-21" in req["url"]
    assert req["headers"]["Authorization"].startswith("Bearer ")
    assert req["json"]["max_tokens"] == 256
    assert req["json"]["messages"][0]["role"] == "user"


@pytest.mark.parametrize("status,expected", [
    pytest.param(429, "429", id="rate_limited"),
    pytest.param(500, "500", id="server_error"),
    pytest.param(503, "503", id="unavailable"),
])
async def test_adapter_translates_transport_errors(status: int,
                                                   expected: str) -> None:
    """Transport-level failures, which only an HTTP-level fake can produce."""
    transport = _FakeHttpTransport([(status, {"retry_after": 2.0})])
    adapter = AzureOpenAIAdapter(transport, "gpt-4o-mini-prod")
    with pytest.raises(RuntimeError, match=expected):
        await adapter.complete("hello", max_tokens=256)


async def test_adapter_extracts_usage() -> None:
    transport = _FakeHttpTransport([(200, _ok_body(good_response()))])
    adapter = AzureOpenAIAdapter(transport, "d")
    text, tin, tout = await adapter.complete("hello", max_tokens=256)
    assert json.loads(text)["disposition"] == "adjust_quantity"
    assert (tin, tout) == (100, 40)


# ===========================================================================
# 5. CONTRACT TESTS — keeping the fake honest
# ===========================================================================

async def test_fake_and_adapter_satisfy_the_same_protocol() -> None:
    """THE RISK WITH FAKES: the fake drifts from the real implementation, your
    tests stay green, and production breaks.

    THE MITIGATION is a CONTRACT TEST — one body of assertions run against
    BOTH implementations. See tests/contract/ for the full pattern; this is
    the idea in miniature.

    It catches: the real client returning a different tuple shape, raising a
    different exception type, or handling an empty response differently from
    the fake.
    """
    implementations: list[ModelClient] = [
        ScriptedModel([good_response()]),
        AzureOpenAIAdapter(
            _FakeHttpTransport([(200, _ok_body(good_response()))]), "d"),
    ]
    for impl in implementations:
        text, tin, tout = await impl.complete("hello", max_tokens=256)
        assert isinstance(text, str)
        assert isinstance(tin, int) and isinstance(tout, int)
        assert RagService.parse(text).disposition == "adjust_quantity"


# ===========================================================================
# 6. WHAT NEVER TO MOCK
# ===========================================================================

async def test_never_mock_the_security_check(make_service) -> None:
    """THE PERMISSION FILTER MUST NEVER BE MOCKED OR STUBBED.

    A test that patches out `visible_to` or substitutes a retriever that
    ignores groups is testing a system that does not exist. Worse, it makes
    the suite GREEN for exactly the failure mode you most need to catch.

    So the fake retriever implements permission trimming FOR REAL, and this
    test verifies the invariant end to end. `trim=False` exists only to prove
    the service fails closed when a retriever misbehaves — which is a
    different assertion from "trimming works".
    """
    leaky = make_service([good_response()], trim=False)
    with pytest.raises(PermissionViolation):
        await leaky.answer("settlement terms confidential",
                           frozenset({"isc-all"}))


async def test_the_real_filter_is_exercised(make_service) -> None:
    """The positive case: with a correct retriever, invisible chunks never
    appear, and nothing raises."""
    svc = make_service([good_response()])
    answer = await svc.answer("settlement terms confidential compensation",
                              frozenset({"isc-all"}))
    assert "c5" not in answer.citations
    assert "c6" not in answer.citations
