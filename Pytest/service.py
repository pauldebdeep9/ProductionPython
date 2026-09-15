"""
rag/service.py — the system under test.

WHY THIS EXISTS
---------------
Testing lessons taught on `def add(a, b)` do not transfer. The questions that
actually matter — where to put a fixture, where to mock, how to test something
whose output is nondeterministic — only have answers in the context of a real
system with real seams.

So everything in this tutorial tests ONE system: a permission-aware RAG
service with a repair loop. It has the seams that matter:

    * an HTTP boundary to a model provider      (where mocks belong)
    * an HTTP boundary to a search index        (where mocks belong)
    * a permission filter                       (a security invariant)
    * a repair loop over nondeterministic output (the LLM-specific problem)
    * config, retries, and a cost budget        (cross-cutting concerns)

NOTE ON DESIGN FOR TESTABILITY: every dependency arrives through the
constructor. There is no module-level client, no `os.environ` read at import
time, and no global state. That is not an accident — it is what makes the
tests in this tutorial short. Code that is hard to test is usually code with
hidden dependencies, and the fix is almost always dependency injection rather
than a cleverer mock.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    allowed_groups: frozenset[str]
    score: float = 0.0

    def visible_to(self, groups: frozenset[str]) -> bool:
        return bool(self.allowed_groups & groups)


Disposition = Literal["approve", "request_credit", "adjust_quantity",
                      "hold_pending_receipt", "escalate_to_buyer"]

VALID_DISPOSITIONS: frozenset[str] = frozenset({
    "approve", "request_credit", "adjust_quantity",
    "hold_pending_receipt", "escalate_to_buyer",
})


@dataclass(frozen=True, slots=True)
class Proposal:
    disposition: Disposition
    confidence: float
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass
class Answer:
    proposal: Proposal | None
    citations: list[str] = field(default_factory=list)
    repairs: int = 0
    degraded: bool = False
    outcome: str = "ok"
    cost_usd: Decimal = Decimal("0")
    trimmed_count: int = 0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RagError(Exception):
    """Base for everything this service raises."""


class PermissionViolation(RagError):
    """A chunk survived retrieval that the caller may not see.

    Should be impossible. If it fires, the request must fail CLOSED.
    """


class BudgetExceeded(RagError):
    pass


class GenerationFailed(RagError):
    pass


# ---------------------------------------------------------------------------
# Protocols — the seams
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    """`groups` is required, so a new implementation cannot silently omit
    permission trimming from its signature."""

    async def search(self, query: str, groups: frozenset[str], *,
                     k: int) -> list[Chunk]: ...


class ModelClient(Protocol):
    async def complete(self, prompt: str, *,
                       max_tokens: int) -> tuple[str, int, int]: ...


class Clock(Protocol):
    """Injected so tests never sleep. See test file 01."""

    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    top_k: int = 4
    max_tokens: int = 512
    max_repairs: int = 2
    min_confidence: float = 0.6
    daily_budget_usd: Decimal = Decimal("10.00")
    price_in_per_1k: Decimal = Decimal("0.00015")
    price_out_per_1k: Decimal = Decimal("0.00060")


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

class RagService:
    def __init__(self, retriever: Retriever, model: ModelClient,
                 config: Config | None = None,
                 clock: Clock | None = None) -> None:
        self.retriever = retriever
        self.model = model
        self.config = config or Config()
        self.clock = clock
        self.spend = Decimal("0")

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def parse(text: str) -> Proposal:
        """Extract and validate a proposal from model output.

        Deliberately does the FREE repairs first (fence stripping, brace
        extraction) before raising — a model call to fix a markdown fence is
        pure waste. See test file 07.
        """
        candidate = text.strip()
        if candidate.startswith("```"):
            inner = candidate.split("```")
            if len(inner) >= 2:
                candidate = inner[1]
                if candidate.startswith("json"):
                    candidate = candidate[4:]
        if not candidate.lstrip().startswith("{"):
            start, end = candidate.find("{"), candidate.rfind("}")
            if 0 <= start < end:
                candidate = candidate[start:end + 1]

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise GenerationFailed(f"not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise GenerationFailed("expected a JSON object")

        disp = data.get("disposition")
        if disp not in VALID_DISPOSITIONS:
            raise GenerationFailed(
                f"disposition {disp!r} is not one of {sorted(VALID_DISPOSITIONS)}")

        conf = data.get("confidence")
        if isinstance(conf, str):
            try:
                conf = float(conf.rstrip("%")) / (100 if "%" in conf else 1)
            except ValueError:
                raise GenerationFailed(
                    f"confidence {conf!r} is not numeric") from None
        if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
            raise GenerationFailed(f"confidence {conf!r} out of range [0,1]")

        reason = data.get("reason")
        if not isinstance(reason, str) or len(reason) < 10:
            raise GenerationFailed("reason must be at least 10 characters")

        evidence = data.get("evidence_ids")
        if not isinstance(evidence, list) or not evidence:
            raise GenerationFailed("evidence_ids must be a non-empty list")

        return Proposal(disposition=disp, confidence=float(conf),
                        reason=reason,
                        evidence_ids=tuple(str(x) for x in evidence))

    # -- cost -------------------------------------------------------------

    def _charge(self, tokens_in: int, tokens_out: int) -> Decimal:
        cost = (Decimal(tokens_in) / 1000 * self.config.price_in_per_1k
                + Decimal(tokens_out) / 1000 * self.config.price_out_per_1k)
        if self.spend + cost > self.config.daily_budget_usd:
            raise BudgetExceeded(
                f"spend {self.spend + cost} exceeds budget "
                f"{self.config.daily_budget_usd}")
        self.spend += cost
        return cost

    # -- the request path --------------------------------------------------

    async def answer(self, question: str, groups: frozenset[str]) -> Answer:
        chunks = await self.retriever.search(question, groups,
                                             k=self.config.top_k)

        # THE HARD INVARIANT. Fails closed, always.
        for c in chunks:
            if not c.visible_to(groups):
                raise PermissionViolation(
                    f"chunk {c.chunk_id} from {c.doc_id} is not visible")

        prompt = ("Context:\n"
                  + "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
                  + f"\nQ: {question}\nRespond with JSON.")

        repairs = 0
        cost = Decimal("0")
        last_error: str = ""
        for attempt in range(self.config.max_repairs + 1):
            text, tin, tout = await self.model.complete(
                prompt, max_tokens=self.config.max_tokens)
            cost += self._charge(tin, tout)
            try:
                proposal = self.parse(text)
            except GenerationFailed as e:
                repairs += 1
                last_error = str(e)
                # A REPAIR, not a retry: the prompt changes.
                prompt = (f"{prompt}\n\nYour previous reply was invalid: {e}\n"
                          f"Return ONLY a corrected JSON object.")
                continue

            if proposal.confidence < self.config.min_confidence:
                return Answer(proposal=proposal,
                              citations=[c.chunk_id for c in chunks],
                              repairs=repairs, degraded=True,
                              outcome="low_confidence", cost_usd=cost,
                              trimmed_count=0)
            return Answer(proposal=proposal,
                          citations=[c.chunk_id for c in chunks],
                          repairs=repairs, degraded=repairs > 0,
                          outcome="ok", cost_usd=cost, trimmed_count=0)

        return Answer(proposal=None, citations=[c.chunk_id for c in chunks],
                      repairs=repairs, degraded=True,
                      outcome=f"repair_exhausted: {last_error}"[:60],
                      cost_usd=cost)


# ---------------------------------------------------------------------------
# Test doubles that ship WITH the code, not with the tests
# ---------------------------------------------------------------------------
# Shipping the fakes alongside the production code (rather than defining them
# in a conftest) means:
#   * they are type-checked against the Protocol in CI
#   * downstream consumers of this package can use them in THEIR tests
#   * a Protocol change breaks the fake immediately, in the same package
# This is a deliberate design choice and it is worth arguing for in review.

CORPUS: list[Chunk] = [
    Chunk("c1", "po-1001", "PO 1001 quantity 400 units", frozenset({"isc-all"})),
    Chunk("c2", "po-1001", "PO 1001 unit price 12.50 USD", frozenset({"isc-all"})),
    Chunk("c3", "inv-88", "Invoice 88 billed 402 units", frozenset({"isc-all"})),
    Chunk("c4", "gr-55", "Goods receipt 55 recorded 400 units", frozenset({"isc-all"})),
    Chunk("c5", "hr-comp", "Compensation band for plant leads", frozenset({"hr-only"})),
    Chunk("c6", "legal-1", "Settlement terms, confidential", frozenset({"legal-only"})),
]


class InMemoryRetriever:
    """A real implementation over a fixed corpus. Not a mock — it has real
    behaviour, including the permission filter."""

    def __init__(self, corpus: list[Chunk] | None = None, *,
                 trim: bool = True) -> None:
        self.corpus = list(corpus if corpus is not None else CORPUS)
        self.trim = trim
        self.calls: list[tuple[str, frozenset[str], int]] = []

    async def search(self, query: str, groups: frozenset[str], *,
                     k: int) -> list[Chunk]:
        self.calls.append((query, groups, k))
        pool = ([c for c in self.corpus if c.visible_to(groups)]
                if self.trim else list(self.corpus))
        terms = set(query.lower().split())
        scored = [
            Chunk(c.chunk_id, c.doc_id, c.text, c.allowed_groups,
                  float(len(terms & set(c.text.lower().split()))))
            for c in pool
        ]
        return sorted((c for c in scored if c.score > 0),
                      key=lambda c: -c.score)[:k]


class ScriptedModel:
    """Returns a fixed sequence of responses.

    THE KEY PROPERTY for testing an LLM system: `[bad, bad, good]` is an
    assertion about repair behaviour, not a coin flip. Random or live models
    make tests flaky; a script makes them deterministic.
    """

    def __init__(self, responses: list[str] | None = None, *,
                 tokens_in: int = 100, tokens_out: int = 40) -> None:
        self.responses = list(responses or [])
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.prompts: list[str] = []
        self._i = 0

    async def complete(self, prompt: str, *,
                       max_tokens: int) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("ScriptedModel has no responses configured")
        text = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return text, self.tokens_in, self.tokens_out


class FakeClock:
    """A controllable clock. Tests must never sleep for real."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        # Records the request and advances instantly. A test asserting
        # `clock.slept == [0.1, 0.2, 0.4]` verifies backoff without spending
        # 0.7 seconds — see test file 01.
        self.slept.append(seconds)
        self.t += seconds


# ---------------------------------------------------------------------------
# Builders for valid responses — used everywhere in the tests
# ---------------------------------------------------------------------------

def good_response(disposition: str = "adjust_quantity", confidence: float = 0.84,
                  reason: str = "Invoice billed 402 against a 400 unit receipt.",
                  evidence: list[str] | None = None) -> str:
    return json.dumps({
        "disposition": disposition, "confidence": confidence,
        "reason": reason, "evidence_ids": evidence or ["c3", "c4"],
    })
