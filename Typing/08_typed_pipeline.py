"""
08 — Capstone: a fully typed RAG + agent pipeline.

WHAT THIS COMPOSES
------------------
  01  boundaries          -> Pydantic in, dataclass inside, TypedDict out
  02  Protocols           -> swappable retriever and model provider
  03  generics + ParamSpec-> typed pipeline stages and a signature-preserving
                             decorator
  04  discriminated unions-> exhaustive dispatch on model responses
  05  container choice    -> the right shape at each layer
  06  Pydantic            -> schema generation + validation from one model
  07  mypy --strict       -> the whole file passes

THE CLAIM BEING DEMONSTRATED
----------------------------
A pipeline can be fully statically checked end-to-end WITHOUT any runtime type
checking on internal calls, provided the boundaries convert untrusted data into
domain types once. Everything after the boundary is proven consistent by mypy
at zero runtime cost.

Run:  python 08_typed_pipeline.py
      mypy --strict 08_typed_pipeline.py
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, ParamSpec, Protocol, TypedDict, TypeVar, assert_never

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from typing_lab import (
    Chunk,
    FinishReason,
    ChunkId,
    CORPUS,
    DeploymentName,
    Disposition,
    PrincipalId,
    Stage,
    TraceId,
    banner,
    section,
    show,
)

P = ParamSpec("P")
R = TypeVar("R")


# ===========================================================================
# LAYER 1 — Protocols: what the pipeline needs from its dependencies
# ===========================================================================

class Retriever(Protocol):
    """`groups` is required by the signature, so permission trimming cannot be
    silently omitted by a new implementation."""

    async def search(
        self, query: str, groups: frozenset[str], *, k: int
    ) -> Sequence[Chunk]: ...


class ModelClient(Protocol):
    async def complete(
        self, prompt: str, *, max_tokens: int, deployment: DeploymentName
    ) -> RawCompletion: ...


@dataclass(frozen=True, slots=True)
class RawCompletion:
    """What a provider returns BEFORE interpretation. Deliberately dumb — it
    carries the wire facts and makes no claims about meaning."""

    text: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


# ===========================================================================
# LAYER 2 — the boundary models (Pydantic), used ONLY at the edge
# ===========================================================================

class ProposalSchema(BaseModel):
    """Validates model output AND generates the JSON Schema sent to the API."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    disposition: Disposition = Field(description="Action for this exception.")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=10, max_length=400)
    evidence_ids: list[str] = Field(min_length=1)

    def to_domain(self, trace_id: TraceId) -> Proposal:
        """Convert to the internal type. After this line, no Pydantic."""
        return Proposal(
            disposition=self.disposition,
            confidence=self.confidence,
            reason=self.reason,
            evidence_ids=tuple(ChunkId(x) for x in self.evidence_ids),
            trace_id=trace_id,
        )


# ===========================================================================
# LAYER 3 — domain types (frozen dataclasses), used everywhere internally
# ===========================================================================

@dataclass(frozen=True, slots=True)
class Proposal:
    disposition: Disposition
    confidence: float
    reason: str
    evidence_ids: tuple[ChunkId, ...]
    trace_id: TraceId


@dataclass(frozen=True, slots=True)
class RetrievalOutput:
    chunks: tuple[Chunk, ...]
    candidates_seen: int
    trimmed_by_acl: int


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything a request carries. Frozen, so no stage can mutate another
    stage's view of the world."""

    trace_id: TraceId
    principal: PrincipalId
    groups: frozenset[str]
    deployment: DeploymentName
    max_tokens: int = 256


# --- the discriminated union of model outcomes (script 04) -----------------

@dataclass(frozen=True, slots=True)
class Parsed:
    kind: Literal["parsed"] = "parsed"
    proposal: Proposal | None = None


@dataclass(frozen=True, slots=True)
class NeedsRepair:
    kind: Literal["needs_repair"] = "needs_repair"
    hint: str = ""
    raw: str = ""


@dataclass(frozen=True, slots=True)
class TruncatedOut:
    kind: Literal["truncated"] = "truncated"
    tokens_used: int = 0


@dataclass(frozen=True, slots=True)
class FilteredOut:
    kind: Literal["filtered"] = "filtered"


ModelOutcome = Parsed | NeedsRepair | TruncatedOut | FilteredOut


# ===========================================================================
# LAYER 4 — the wire type on the way OUT (TypedDict)
# ===========================================================================

class ProposalResponse(TypedDict):
    """The JSON we serialise back to the caller. A TypedDict because it IS a
    dict — converting to an object just to call json.dumps is ceremony."""

    trace_id: str
    disposition: str
    confidence: float
    reason: str
    citations: list[str]
    degraded: bool


# ===========================================================================
# LAYER 5 — generic pipeline stages
# ===========================================================================

class PipelineStage[TIn, TOut](Protocol):
    """A generic Protocol. Any stage transforms TIn -> TOut given a context.

    Two type parameters connecting two positions each — the test from script
    03 part 6 for whether a generic is earning its keep.
    """

    async def __call__(self, value: TIn, ctx: RequestContext) -> TOut: ...


def instrumented[**Q, S](
    stage: Stage,
) -> Callable[[Callable[Q, Awaitable[S]]], Callable[Q, Awaitable[S]]]:
    """A decorator that preserves the wrapped signature exactly.

    Note the PEP 695 form: `[**Q, S]` declares a ParamSpec and a TypeVar
    inline, which is the 3.12 equivalent of module-level `ParamSpec("Q")`.
    """

    def decorator(fn: Callable[Q, Awaitable[S]]) -> Callable[Q, Awaitable[S]]:
        @functools.wraps(fn)
        async def wrapper(*args: Q.args, **kwargs: Q.kwargs) -> S:
            TIMINGS.setdefault(stage, 0)
            TIMINGS[stage] += 1
            return await fn(*args, **kwargs)

        return wrapper

    return decorator


TIMINGS: dict[Stage, int] = {}


# ===========================================================================
# LAYER 6 — implementations
# ===========================================================================

class InMemoryRetriever:
    """Satisfies `Retriever` structurally. Imports nothing from this module's
    Protocol section — in a real codebase it would be a separate package."""

    def __init__(self, corpus: Sequence[Chunk]) -> None:
        self.corpus = tuple(corpus)

    async def search(
        self, query: str, groups: frozenset[str], *, k: int
    ) -> Sequence[Chunk]:
        await asyncio.sleep(0)
        terms = set(query.lower().split())
        # Trim BEFORE ranking. Always.
        visible = [c for c in self.corpus if c.visible_to(groups)]
        scored = [
            Chunk(c.chunk_id, c.doc_id, c.text, c.allowed_groups, c.sensitivity,
                  float(len(terms & set(c.text.lower().split()))))
            for c in visible
        ]
        return tuple(sorted((c for c in scored if c.score > 0),
                            key=lambda c: -c.score)[:k])


class ScriptedModel:
    """Satisfies `ModelClient`. A hand-written fake, verified by mypy against
    the Protocol — see `_conformance` at the bottom."""

    # NOTE THE PARAMETER TYPE. It was `Sequence[tuple[str, str]]` first, and
    # mypy rejected the RawCompletion construction below:
    #
    #   error: Argument "finish_reason" to "RawCompletion" has incompatible
    #          type "str"; expected "Literal['stop', 'length', ...]"
    #
    # The tempting fix is `cast(FinishReason, finish)` or an `assert finish in
    # (...)`. NEITHER WORKS as narrowing — `assert x in tuple` does not narrow
    # `str` to a Literal (script 04 part 1: helper-shaped checks need a
    # TypeGuard).
    #
    # The RIGHT fix is to type the data at its SOURCE. Declaring the parameter
    # as carrying FinishReason means the literal strings in the call sites are
    # checked there, where a typo is visible, rather than being laundered
    # through a cast here.
    def __init__(self, responses: Sequence[tuple[str, FinishReason]]) -> None:
        self.responses = tuple(responses)
        self._i = 0
        self.prompts: list[str] = []

    async def complete(
        self, prompt: str, *, max_tokens: int, deployment: DeploymentName
    ) -> RawCompletion:
        await asyncio.sleep(0)
        self.prompts.append(prompt)
        text, finish = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return RawCompletion(
            text=text,
            finish_reason=finish,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
        )


# ===========================================================================
# LAYER 7 — the boundary: RawCompletion -> ModelOutcome
# ===========================================================================

def interpret(raw: RawCompletion, trace_id: TraceId) -> ModelOutcome:
    """THE BOUNDARY FUNCTION. Untrusted text in, closed union out.

    Order matters: finish_reason is checked BEFORE parsing, so truncation is
    diagnosed as truncation rather than as a confusing JSON error.
    """
    match raw.finish_reason:
        case "length":
            return TruncatedOut(tokens_used=raw.completion_tokens)
        case "content_filter":
            return FilteredOut()
        case "tool_calls":
            return NeedsRepair(hint="Return a JSON proposal, not a tool call.",
                               raw=raw.text)
        case "stop":
            pass
        case _:
            assert_never(raw.finish_reason)

    try:
        model = ProposalSchema.model_validate_json(raw.text)
    except ValidationError as e:
        return NeedsRepair(hint=_repair_hint(e), raw=raw.text)
    return Parsed(proposal=model.to_domain(trace_id))


def _repair_hint(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(object)"
        parts.append(f"- '{loc}': {err['msg']}")
    return ("Your JSON did not satisfy the schema:\n" + "\n".join(parts)
            + "\nReturn ONLY a corrected JSON object.")


# ===========================================================================
# LAYER 8 — the pipeline
# ===========================================================================

@dataclass
class Pipeline:
    retriever: Retriever
    model: ModelClient
    max_repairs: int = 2
    acl_violations: int = 0

    @instrumented(Stage.RETRIEVE)
    async def retrieve(self, query: str, ctx: RequestContext) -> RetrievalOutput:
        chunks = await self.retriever.search(query, ctx.groups, k=4)

        # THE HARD INVARIANT, asserted after retrieval. Cheap (a set
        # intersection per chunk) and catches a retriever that forgot to trim.
        for c in chunks:
            if not c.visible_to(ctx.groups):
                self.acl_violations += 1
                raise PermissionError(
                    f"chunk {c.chunk_id} not visible to {ctx.principal}"
                )

        return RetrievalOutput(
            chunks=tuple(chunks),
            candidates_seen=len(CORPUS),
            trimmed_by_acl=len(CORPUS) - len(chunks),
        )

    @instrumented(Stage.GENERATE)
    async def generate(
        self, query: str, retrieval: RetrievalOutput, ctx: RequestContext
    ) -> tuple[ModelOutcome, list[str]]:
        """Repair loop, dispatching exhaustively on the outcome union."""
        trail: list[str] = []
        context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in retrieval.chunks)
        prompt = f"Q: {query}\nContext:\n{context}\nRespond with JSON."
        max_tokens = ctx.max_tokens

        for _ in range(self.max_repairs + 1):
            raw = await self.model.complete(
                prompt, max_tokens=max_tokens, deployment=ctx.deployment)
            outcome = interpret(raw, ctx.trace_id)

            match outcome:
                case Parsed():
                    trail.append("parsed")
                    return outcome, trail
                case TruncatedOut(tokens_used=n):
                    trail.append(f"truncated@{n}")
                    max_tokens *= 2          # REPAIR, not retry
                case NeedsRepair(hint=hint):
                    trail.append("repair")
                    prompt = f"{prompt}\n\n{hint}"
                case FilteredOut():
                    trail.append("filtered")
                    return outcome, trail     # FAIL FAST
                case _:
                    assert_never(outcome)

        return NeedsRepair(hint="repair budget exhausted"), trail

    async def run(self, query: str, ctx: RequestContext) -> ProposalResponse:
        retrieval = await self.retrieve(query, ctx)
        outcome, trail = await self.generate(query, retrieval, ctx)

        # Exhaustive dispatch again, now producing the wire type.
        match outcome:
            case Parsed(proposal=p) if p is not None:
                return ProposalResponse(
                    trace_id=str(ctx.trace_id),
                    disposition=p.disposition,
                    confidence=p.confidence,
                    reason=p.reason,
                    citations=[str(c) for c in p.evidence_ids],
                    degraded=len(trail) > 1,
                )
            case Parsed():
                return _degraded(ctx, "empty proposal")
            case NeedsRepair(hint=hint):
                return _degraded(ctx, hint.splitlines()[0])
            case TruncatedOut():
                return _degraded(ctx, "output truncated")
            case FilteredOut():
                return _degraded(ctx, "content filtered")
            case _:
                assert_never(outcome)


def _degraded(ctx: RequestContext, reason: str) -> ProposalResponse:
    """A degraded response, LABELLED as such — the rule from the
    failure-handling tutorial, enforced here by the type having a `degraded`
    field that must be filled in."""
    return ProposalResponse(
        trace_id=str(ctx.trace_id), disposition="escalate_to_buyer",
        confidence=0.0, reason=f"automated analysis unavailable: {reason}",
        citations=[], degraded=True,
    )


# ===========================================================================
# Static conformance assertions — cost nothing, fail in the right file
# ===========================================================================

def _conformance() -> None:
    _r: Retriever = InMemoryRetriever(CORPUS)
    _m: ModelClient = ScriptedModel([])
    del _r, _m


# ===========================================================================
# Demonstrations
# ===========================================================================

GOOD = json.dumps({
    "disposition": "adjust_quantity", "confidence": 0.84,
    "reason": "Invoice billed 402 units against a 400 unit goods receipt.",
    "evidence_ids": ["c3", "c4"],
})
BAD_ENUM = json.dumps({
    "disposition": "partial_approve", "confidence": 0.6,
    "reason": "Partially acceptable given the observed variance in units.",
    "evidence_ids": ["c3"],
})


async def demo1_happy() -> None:
    banner("DEMO 1 — clean run, fully typed end to end")

    p = Pipeline(InMemoryRetriever(CORPUS), ScriptedModel([(GOOD, "stop")]))
    ctx = RequestContext(TraceId("tr-0001"), PrincipalId("debdeep@contoso.com"),
                         frozenset({"isc-all"}), DeploymentName("gpt-4o-mini-prod"))
    resp = await p.run("units billed receipt", ctx)
    for k, v in resp.items():
        show(k, v)


async def demo2_repair() -> None:
    banner("DEMO 2 — hallucinated enum repaired, then parsed")

    model = ScriptedModel([(BAD_ENUM, "stop"), (GOOD, "stop")])
    p = Pipeline(InMemoryRetriever(CORPUS), model)
    ctx = RequestContext(TraceId("tr-0002"), PrincipalId("a@contoso.com"),
                         frozenset({"isc-all"}), DeploymentName("d"))
    resp = await p.run("units billed receipt", ctx)
    show("disposition", resp["disposition"])
    show("degraded (a repair was needed)", resp["degraded"])
    section("the repair hint that was appended to the prompt")
    hint_line = [ln for ln in model.prompts[1].splitlines()
                 if ln.startswith("- ")]
    for ln in hint_line:
        print(f"      {ln[:88]}")


async def demo3_truncation_and_filter() -> None:
    banner("DEMO 3 — truncation repairs; content filter fails fast")

    section("truncated, then succeeds at double max_tokens")
    model = ScriptedModel([("{\"disp", "length"), (GOOD, "stop")])
    p = Pipeline(InMemoryRetriever(CORPUS), model)
    ctx = RequestContext(TraceId("tr-0003"), PrincipalId("a@contoso.com"),
                         frozenset({"isc-all"}), DeploymentName("d"),
                         max_tokens=32)
    resp = await p.run("units billed", ctx)
    show("disposition", resp["disposition"])
    show("degraded", resp["degraded"])

    section("content filtered — no repair attempted")
    model2 = ScriptedModel([("", "content_filter"), (GOOD, "stop")])
    p2 = Pipeline(InMemoryRetriever(CORPUS), model2)
    resp2 = await p2.run("units billed", RequestContext(
        TraceId("tr-0004"), PrincipalId("a@contoso.com"),
        frozenset({"isc-all"}), DeploymentName("d")))
    show("disposition", resp2["disposition"])
    show("reason", resp2["reason"])
    show("model calls made", len(model2.prompts))
    print("      One call. A content filter rejects the same prompt every")
    print("      time, so repairing it is pure waste.")


async def demo4_permission_trimming() -> None:
    banner("DEMO 4 — permission trimming, and the assertion that guards it")

    p = Pipeline(InMemoryRetriever(CORPUS), ScriptedModel([(GOOD, "stop")]))
    for principal, groups in [
        ("debdeep@contoso.com", frozenset({"isc-all"})),
        ("legal@contoso.com", frozenset({"legal-only"})),
    ]:
        ctx = RequestContext(TraceId("tr-x"), PrincipalId(principal), groups,
                             DeploymentName("d"))
        out = await p.retrieve("settlement units billed confidential", ctx)
        show(principal, [c.chunk_id for c in out.chunks])

    section("a retriever that forgets to trim")

    class LeakyRetriever:
        async def search(self, query: str, groups: frozenset[str], *,
                         k: int) -> Sequence[Chunk]:
            return tuple(CORPUS[:6])      # no filtering at all

    leaky = Pipeline(LeakyRetriever(), ScriptedModel([(GOOD, "stop")]))
    try:
        await leaky.retrieve("settlement", RequestContext(
            TraceId("tr-y"), PrincipalId("a@contoso.com"),
            frozenset({"isc-all"}), DeploymentName("d")))
    except PermissionError as e:
        show("post-retrieval assertion", f"raised: {e}")
    print("""
      NOTE WHAT TYPING DID AND DID NOT DO HERE. The Protocol forced
      LeakyRetriever to ACCEPT `groups` — it could not silently omit the
      parameter. But nothing in the type system can force it to USE the
      parameter. That is a runtime assertion's job, and it is why the
      post-retrieval check exists.

      Static typing constrains SHAPE, never BEHAVIOUR. Any security invariant
      needs a runtime check as well.""")


async def main() -> None:
    _conformance()
    await demo1_happy()
    await demo2_repair()
    await demo3_truncation_and_filter()
    await demo4_permission_trimming()

    banner("WHAT THE TYPES BOUGHT")
    print("""
  1. ONE Pydantic model generates the JSON Schema sent to the API AND
     validates what comes back. They cannot drift.
  2. Validation happens exactly ONCE, at `interpret()`. Every function after
     that takes frozen dataclasses and pays no runtime type cost.
  3. `ModelOutcome` is a closed union with `assert_never` in both dispatch
     sites. Adding a fifth outcome produces two checker errors naming the
     exact lines to update.
  4. Truncation, filtering, and parse failure are DIFFERENT types, so they
     cannot accidentally share a code path — filtered fails fast, truncated
     doubles max_tokens, parse failure appends a hint.
  5. The `Retriever` Protocol makes `groups` a required parameter, so a new
     retriever cannot silently drop permission trimming from its signature.
  6. `ProposalResponse` has a required `degraded` field, so a degraded answer
     cannot be returned unlabelled — the compiler enforces the honesty rule.
  7. `instrumented` uses ParamSpec, so decorating a method does not erase its
     signature for callers.
  8. `mypy --strict` passes on this entire file with no `type: ignore`.

  AND THE LIMIT, from demo 4: types constrain shape, never behaviour. The
  Protocol forced LeakyRetriever to accept `groups`; only the runtime
  assertion caught it ignoring them. Every security invariant needs both.
""")


if __name__ == "__main__":
    asyncio.run(main())
