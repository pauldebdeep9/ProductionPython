"""
tests/contract/test_06_contracts.py — keeping fakes honest.

Run:  pytest tests/contract -v

THE RISK THIS ADDRESSES
-----------------------
Fakes are the right default (see test_03), but they carry one specific danger:
the fake DRIFTS from the real implementation, your unit suite stays green, and
production breaks in a way no test could have caught.

Concretely, a fake retriever that returns `list[Chunk]` while the real Azure
AI Search adapter returns them sorted differently, or raises a different
exception on an empty index, or handles `k=0` differently.

THE FIX is a CONTRACT TEST: one body of assertions, executed against EVERY
implementation of the interface — the fake, the real adapter, and any future
one. The fake earns its place by passing the same tests as the real thing.

THE MECHANISM in pytest is a parametrised fixture with `params=`. Each test in
the file runs once per implementation, automatically.

WHAT A CONTRACT TEST ASSERTS: only what the INTERFACE promises. Not
implementation details, not performance, not ordering unless the interface
guarantees ordering. If you cannot state the promise in one sentence, it does
not belong in a contract test.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from rag.service import (
    CORPUS,
    Chunk,
    GenerationFailed,
    InMemoryRetriever,
    ModelClient,
    RagService,
    Retriever,
    ScriptedModel,
    good_response,
)


# ===========================================================================
# A second implementation, to have something to compare against
# ===========================================================================

class SortedListRetriever:
    """A DIFFERENT implementation of the same Protocol.

    Stands in for "the real Azure AI Search adapter". Deliberately implemented
    differently — it pre-sorts and uses a different scoring approach — so the
    contract test is actually comparing two things rather than one thing
    twice.
    """

    def __init__(self, corpus: list[Chunk] | None = None) -> None:
        self.corpus = sorted(corpus if corpus is not None else CORPUS,
                             key=lambda c: c.chunk_id)

    async def search(self, query: str, groups: frozenset[str], *,
                     k: int) -> list[Chunk]:
        await asyncio.sleep(0)
        terms = set(query.lower().split())
        out: list[Chunk] = []
        for c in self.corpus:
            if not c.visible_to(groups):
                continue
            overlap = len(terms & set(c.text.lower().split()))
            if overlap:
                out.append(Chunk(c.chunk_id, c.doc_id, c.text,
                                 c.allowed_groups, float(overlap)))
        out.sort(key=lambda c: (-c.score, c.chunk_id))
        return out[:k]


class EchoModel:
    """A second ModelClient, deriving its answer from the prompt."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: str, *,
                       max_tokens: int) -> tuple[str, int, int]:
        self.calls += 1
        cited = [c for c in ("c1", "c2", "c3", "c4") if f"[{c}]" in prompt]
        return good_response(evidence=cited or ["c1"]), len(prompt) // 4, 40


# ===========================================================================
# THE PARAMETRISED FIXTURE — this is the whole mechanism
# ===========================================================================

@pytest.fixture(params=["in_memory", "sorted_list"])
def retriever_impl(request: pytest.FixtureRequest) -> Retriever:
    """`params=` makes every test using this fixture run once per value.

    The ids in the report become:
        test_returns_at_most_k[in_memory]
        test_returns_at_most_k[sorted_list]

    ADDING AN IMPLEMENTATION is one line here, and it immediately inherits
    every contract test in the file. That is the property that makes this
    pattern worth the setup.
    """
    return {"in_memory": InMemoryRetriever,
            "sorted_list": SortedListRetriever}[request.param]()


@pytest.fixture(params=["scripted", "echo"])
def model_impl(request: pytest.FixtureRequest) -> ModelClient:
    return ({"scripted": lambda: ScriptedModel([good_response()]),
             "echo": EchoModel}[request.param])()


# ===========================================================================
# THE RETRIEVER CONTRACT
# ===========================================================================

async def test_returns_at_most_k(retriever_impl: Retriever) -> None:
    """PROMISE: never more than k results."""
    for k in (1, 2, 4, 10):
        chunks = await retriever_impl.search("units billed PO invoice receipt",
                                             frozenset({"isc-all"}), k=k)
        assert len(chunks) <= k


async def test_never_returns_invisible_chunks(retriever_impl: Retriever) -> None:
    """PROMISE: permission trimming, applied by every implementation.

    THE MOST IMPORTANT CONTRACT TEST HERE. A new retriever that forgets to
    trim fails this immediately, in the contract suite, rather than in
    production — and it fails with a message naming the implementation.
    """
    for groups in (frozenset({"isc-all"}), frozenset({"hr-only"}),
                   frozenset({"legal-only"}), frozenset()):
        chunks = await retriever_impl.search(
            "units billed compensation settlement confidential",
            groups, k=10)
        for c in chunks:
            assert c.visible_to(groups), (
                f"{type(retriever_impl).__name__} returned {c.chunk_id}, "
                f"which is not visible to {sorted(groups)}"
            )


async def test_empty_groups_returns_nothing(retriever_impl: Retriever) -> None:
    """PROMISE: no groups means no results. Fails CLOSED.

    The alternative — treating an empty group set as "no filter" — is the
    fail-open bug, and it is exactly the kind of edge case where two
    implementations diverge without anyone noticing.
    """
    chunks = await retriever_impl.search("units billed", frozenset(), k=10)
    assert chunks == []


async def test_no_match_returns_empty_not_none(retriever_impl: Retriever) -> None:
    """PROMISE: an empty LIST, never None.

    A trivial-looking contract that matters: the caller does
    `for c in chunks`, and one implementation returning None turns a
    no-results case into a TypeError at 3am.
    """
    chunks = await retriever_impl.search("zzzz nonexistent qqqq",
                                         frozenset({"isc-all"}), k=5)
    assert chunks == []
    assert isinstance(chunks, list)


async def test_results_are_score_ordered(retriever_impl: Retriever) -> None:
    """PROMISE: descending by score.

    NOTE what this does NOT assert: the ordering of TIES. `InMemoryRetriever`
    and `SortedListRetriever` break ties differently, and that is allowed
    because the interface does not promise it.

    Writing the contract to require a specific tie order would be asserting an
    implementation detail — and it would have forced one implementation to
    change for no user-visible benefit. Contract tests should assert the
    promise and nothing more.
    """
    chunks = await retriever_impl.search("units billed PO invoice receipt",
                                         frozenset({"isc-all"}), k=10)
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)


async def test_k_of_zero_is_handled(retriever_impl: Retriever) -> None:
    """An edge case both implementations must agree on."""
    chunks = await retriever_impl.search("units billed",
                                         frozenset({"isc-all"}), k=0)
    assert chunks == []


# ===========================================================================
# THE MODEL CONTRACT
# ===========================================================================

async def test_returns_the_expected_tuple_shape(model_impl: ModelClient) -> None:
    """PROMISE: (text, tokens_in, tokens_out), with those types."""
    text, tin, tout = await model_impl.complete("hello", max_tokens=256)
    assert isinstance(text, str)
    assert isinstance(tin, int) and tin >= 0
    assert isinstance(tout, int) and tout >= 0


async def test_output_is_parseable_by_the_service(model_impl: ModelClient) -> None:
    """PROMISE: a healthy model produces output the parser accepts.

    This is the contract that keeps a fake honest in the way that matters
    most: if the real client's output shape changes and the fake's does not,
    every unit test still passes and every production request fails.
    """
    text, _, _ = await model_impl.complete(
        "Context:\n[c1] PO 1001 quantity 400\nQ: units?", max_tokens=256)
    proposal = RagService.parse(text)
    assert proposal.disposition in {
        "approve", "request_credit", "adjust_quantity",
        "hold_pending_receipt", "escalate_to_buyer"}


# ===========================================================================
# THE FULL PIPELINE, ACROSS THE PRODUCT OF IMPLEMENTATIONS
# ===========================================================================

async def test_pipeline_works_with_any_combination(
    retriever_impl: Retriever, model_impl: ModelClient
) -> None:
    """2 retrievers x 2 models = 4 tests, generated automatically.

    Requesting both parametrised fixtures produces the Cartesian product. For
    an interface with several implementations this is the cheapest integration
    coverage available — and it catches the combination nobody tried.
    """
    svc = RagService(retriever_impl, model_impl)
    answer = await svc.answer("invoice 88 units billed", frozenset({"isc-all"}))
    assert answer.proposal is not None
    assert answer.outcome == "ok"
    assert "c5" not in answer.citations and "c6" not in answer.citations


# ===========================================================================
# WHAT A CONTRACT TEST MUST NOT ASSERT
# ===========================================================================

async def test_contract_does_not_pin_performance(
    retriever_impl: Retriever
) -> None:
    """A NOTE RATHER THAN A REAL TEST, because the point is what is absent.

    Things that must NOT be in a contract test:

      TIMING        "search completes in under 10ms". The fake is in-memory
                    and the real one crosses a network. A contract that the
                    real implementation cannot meet is a contract that gets
                    deleted.

      CALL COUNTS   "search issues exactly one HTTP request". That is an
                    implementation detail of one implementation.

      ORDERING OF TIES, EXACT SCORES, INTERNAL STATE — same reason.

    THE TEST FOR A CONTRACT ASSERTION: could a reasonable alternative
    implementation fail this while still being correct? If yes, it does not
    belong.
    """
    chunks = await retriever_impl.search("units billed",
                                         frozenset({"isc-all"}), k=4)
    assert isinstance(chunks, list)
