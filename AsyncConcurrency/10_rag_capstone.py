"""
10 — Capstone: a permission-aware RAG service using every pattern in this set.

WHAT THIS COMPOSES
------------------
  01  Task/loop mechanics            -> concurrent retrieval fan-out
  02  TaskGroup + except*            -> structured multi-source retrieval
  03  to_thread                      -> CPU-bound reranking off the loop
  04  Semaphore + TokenBucket        -> bounded concurrency and RPM control
  05  timeout_at + shielded cleanup  -> one end-to-end deadline, audit survives
  06  classified retries + breaker   -> resilient model calls
  07  aclosing + SSE                 -> token streaming with clean shutdown
  08  bounded queue                  -> (referenced; see 08 for ingestion)
  09  contextvars, client lifetime   -> trace propagation, shared clients

THE INVARIANT THIS SERVICE PROTECTS
-----------------------------------
Permission-trimmed retrieval. The security filter is applied PRE-RANKING, and
a post-retrieval assertion re-checks every returned chunk against the caller's
identity. That assertion is not defence-in-depth theatre — it is there because
permission bugs in RAG fail silently and UPWARD: the user gets a fluent,
well-cited answer built from a document they were never allowed to see, and
nothing in the response indicates anything went wrong.

Concurrency makes this harder, which is why it belongs in an async tutorial:
  * a shared cache keyed only on the query text leaks across identities,
  * a cached ACL that goes stale mid-request leaks,
  * a fan-out where one source applies the filter and another does not leaks,
  * a partial-failure path that falls back to an unfiltered index leaks.

Every one of those is a concurrency-shaped bug, not a cryptography bug.

Run:  python 10_rag_capstone.py
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars

# Reuse the primitives we built earlier rather than redefining them.
import importlib.util
import json
import pathlib
import sys
import time
import uuid
from dataclasses import dataclass, field

from fake_llm import (
    FakeLLMClient,
    LLMError,
    Timer,
    TransientServerError,
    banner,
)


def _load(mod_name: str, filename: str):
    """Load a numbered tutorial script as a module so we can reuse its
    primitives without copy-pasting them.

    GOTCHA worth knowing independently of this tutorial: `@dataclass` resolves
    type annotations via `sys.modules[cls.__module__]`. A module executed with
    `exec_module` but never registered in `sys.modules` therefore blows up with
    a confusing `'NoneType' object has no attribute '__dict__'`. Register the
    module BEFORE executing it. The same trap hits plugin loaders and any
    dynamic-import machinery.
    """
    spec = importlib.util.spec_from_file_location(
        mod_name, pathlib.Path(__file__).parent / filename
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m          # <-- must happen before exec_module
    spec.loader.exec_module(m)
    return m


_bc = _load("bc", "04_bounded_concurrency.py")
_rb = _load("rb", "06_retries_backoff.py")
TokenBucket = _bc.TokenBucket
RetryPolicy = _rb.RetryPolicy
with_retries = _rb.with_retries
CircuitBreaker = _rb.CircuitBreaker


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
principal_var: contextvars.ContextVar[str] = contextvars.ContextVar("principal", default="-")


VERBOSE = True


def log(event: str, **fields) -> None:
    """Structured log line carrying request context automatically.

    Note what is NOT here: no prompt text, no chunk text, no retrieved
    content. Logging retrieved content to Application Insights moves data
    across a trust boundary that your permission model does not cover — the
    people who can read the telemetry workspace are not the people who could
    read the source document. Log IDs and counts; never bodies.
    """
    if not VERBOSE:
        return
    record = {
        "event": event,
        "trace_id": trace_id_var.get(),
        "principal": principal_var.get(),
        **fields,
    }
    print(f"    {json.dumps(record)}")


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    # The ACL travels WITH the chunk, in the index. It is not looked up
    # separately at answer time — that lookup is a race waiting to happen.
    allowed_groups: frozenset[str]
    score: float = 0.0


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    trace_id: str = ""
    usage_tokens: int = 0
    degraded: bool = False
    # Every candidate that survived permission trimming, before top-k.
    # Exposed so the demo can show trimming directly rather than by inference.
    retrieved: list[str] = field(default_factory=list)


class PermissionError_(Exception):
    """Raised when the post-retrieval assertion fails. This should be
    impossible; if it fires, it is a P1 and the request must fail closed."""


# ---------------------------------------------------------------------------
# Fake indices
# ---------------------------------------------------------------------------

CORPUS = [
    Chunk("c1", "po-1001", "PO 1001 unit price 12.50 USD", frozenset({"isc-all"})),
    Chunk("c2", "po-1001", "PO 1001 quantity 400 units", frozenset({"isc-all"})),
    Chunk("c3", "inv-88", "Invoice 88 billed 402 units", frozenset({"isc-all"})),
    Chunk("c4", "gr-55", "Goods receipt 55 recorded 400 units", frozenset({"isc-all"})),
    Chunk("c5", "hr-comp", "Compensation band for plant leads", frozenset({"hr-only"})),
    Chunk("c6", "legal-1", "Settlement terms, confidential", frozenset({"legal-only"})),
]


class SearchIndex:
    """A vector/keyword index that applies a security filter PRE-RANKING."""

    def __init__(self, name: str, latency: float) -> None:
        self.name = name
        self.latency = latency

    async def search(self, query: str, groups: frozenset[str], k: int = 6) -> list[Chunk]:
        await asyncio.sleep(self.latency)

        # THE CRITICAL LINE. The filter is applied to the candidate set BEFORE
        # top-k selection. Filtering after ranking is the classic bug: you ask
        # for the top 3, get 3 documents the user cannot see, filter them out,
        # and return nothing — or worse, you return them.
        #
        # In Azure AI Search this is the `filter` parameter with a
        # `search.in(group_ids, ...)` expression, evaluated during retrieval.
        visible = [c for c in CORPUS if c.allowed_groups & groups]

        scored = []
        for c in visible:
            overlap = len(set(query.lower().split()) & set(c.text.lower().split()))
            if overlap:
                scored.append(Chunk(c.chunk_id, c.doc_id, c.text, c.allowed_groups,
                                    score=float(overlap)))
        scored.sort(key=lambda c: -c.score)
        return scored[:k]


class BrokenIndex(SearchIndex):
    """Deliberately omits the security filter — used to prove the assertion
    catches it. In real life this is a code path someone added for a
    'quick internal tool' that later got wired into the main service."""

    async def search(self, query: str, groups: frozenset[str], k: int = 3) -> list[Chunk]:
        await asyncio.sleep(self.latency)
        # Note the signature still ACCEPTS `groups` — it just never uses it.
        # That is what makes this bug survive review: the call site looks
        # identical to the correct one, and the type checker is satisfied.
        scored = []
        for c in CORPUS:
            overlap = len(set(query.lower().split()) & set(c.text.lower().split()))
            if overlap:
                scored.append(Chunk(c.chunk_id, c.doc_id, c.text,
                                    c.allowed_groups, float(overlap)))
        scored.sort(key=lambda c: -c.score)
        return scored[:k]


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

class RagService:
    """Clients and limiters are constructed ONCE and live for the process.

    See PITFALL 5: building these per request costs a TLS handshake per call
    and defeats the credential's internal token cache.
    """

    def __init__(self) -> None:
        self.llm = FakeLLMClient(seed=101, base_latency_s=0.04, jitter_s=0.02,
                                 server_error_rate=0.25)
        self.embedder = FakeLLMClient(seed=102, base_latency_s=0.02)

        self.indices = [
            SearchIndex("sharepoint", 0.05),
            SearchIndex("blob-docs", 0.08),
        ]

        # Bounded concurrency, bounded rate. Shared across all requests,
        # because the QUOTA is shared across all requests.
        self.sem = asyncio.Semaphore(8)
        self.bucket = TokenBucket(rate=50.0, capacity=10.0)
        self.breaker = CircuitBreaker(threshold=6, recovery_time=0.5)

        self.audit: list[dict] = []

    # -- audit -------------------------------------------------------------

    async def _write_audit(self, record: dict) -> None:
        await asyncio.sleep(0.02)   # a real append to an audit store
        self.audit.append(record)

    # -- model call with all the protections --------------------------------

    async def _generate(self, prompt: str) -> str:
        """Rate limit -> concurrency slot -> circuit breaker -> retries.

        ORDER MATTERS. Rate token first (cheap to hold), then the concurrency
        slot (scarce), so we never occupy a slot merely waiting on rate.
        Retries live INSIDE the semaphore so that a retrying request does not
        secretly double the real in-flight count.
        """
        await self.bucket.acquire(1.0)
        async with self.sem:
            async def call():
                return await self.breaker.call(lambda: self.llm.complete(prompt))

            completion = await with_retries(
                call,
                policy=RetryPolicy(max_attempts=4, base_delay=0.02,
                                   overall_deadline=1.0),
                op_name="generate",
            )
            return completion.text

    # -- retrieval fan-out --------------------------------------------------

    async def _retrieve(self, query: str, groups: frozenset[str]) -> list[Chunk]:
        """Query every index concurrently, then fuse.

        Uses gather(return_exceptions=True) rather than TaskGroup deliberately:
        losing ONE index should degrade the answer, not fail the request. That
        is a product decision, and it is the kind of thing worth stating
        explicitly in an ADR rather than leaving implicit in a primitive choice.
        """
        results = await asyncio.gather(
            *(idx.search(query, groups) for idx in self.indices),
            return_exceptions=True,
        )

        merged: dict[str, Chunk] = {}
        failures = 0
        for idx, res in zip(self.indices, results, strict=True):
            if isinstance(res, BaseException):
                failures += 1
                log("retrieval.source_failed", source=idx.name,
                    error=type(res).__name__)
                continue
            # Reciprocal Rank Fusion, simplified: 1/(k+rank), summed.
            for rank, chunk in enumerate(res):
                prev = merged.get(chunk.chunk_id)
                bump = 1.0 / (60 + rank)
                if prev is None:
                    merged[chunk.chunk_id] = Chunk(
                        chunk.chunk_id, chunk.doc_id, chunk.text,
                        chunk.allowed_groups, bump,
                    )
                else:
                    merged[chunk.chunk_id] = Chunk(
                        prev.chunk_id, prev.doc_id, prev.text,
                        prev.allowed_groups, prev.score + bump,
                    )

        if failures == len(self.indices):
            # Fail closed. Never fall back to an unfiltered source.
            raise TransientServerError("all retrieval sources unavailable")

        return sorted(merged.values(), key=lambda c: -c.score)

    # -- the hard invariant -------------------------------------------------

    @staticmethod
    def _assert_permitted(chunks: list[Chunk], groups: frozenset[str]) -> None:
        """Post-retrieval assertion. Should never fire.

        This is cheap (a set intersection per chunk) and catches: a new index
        added without the filter, a filter expression broken by a schema
        change, a cache returning another principal's entry, and a stale ACL
        baked into the index.

        Treat a firing here as fail-closed, page-a-human, not log-and-continue.
        """
        for c in chunks:
            if not (c.allowed_groups & groups):
                raise PermissionError_(
                    f"chunk {c.chunk_id} from {c.doc_id} is not visible to caller"
                )

    # -- CPU-bound rerank ---------------------------------------------------

    @staticmethod
    def _rerank_sync(chunks: list[Chunk], query: str) -> list[Chunk]:
        """Stands in for a cross-encoder. Pure CPU, hundreds of ms in reality.

        MUST go through to_thread, or it freezes every concurrent request —
        see 03. This is the most commonly missed blocking call in RAG code,
        because it does not look like I/O and does not look slow in a unit test
        with three documents.
        """
        # A deliberate CPU burner standing in for a cross-encoder forward pass.
        acc = 0.0
        for i in range(50_000):
            acc += (i % 7) * 0.5
        return sorted(chunks, key=lambda c: (-c.score, c.chunk_id))

    # -- the request handler -------------------------------------------------

    async def answer(
        self,
        query: str,
        principal: str,
        groups: frozenset[str],
        *,
        budget_s: float = 2.0,
    ) -> Answer:
        trace_id = f"tr-{uuid.uuid4().hex[:8]}"
        trace_id_var.set(trace_id)
        principal_var.set(principal)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget_s   # ONE end-to-end budget
        started = time.perf_counter()
        degraded = False
        outcome = "ok"

        try:
            async with asyncio.timeout_at(deadline):
                log("request.start", query_len=len(query))

                chunks = await self._retrieve(query, groups)
                log("retrieval.done", candidates=len(chunks))

                # The invariant, checked before anything reaches the model.
                self._assert_permitted(chunks, groups)

                chunks = await asyncio.to_thread(self._rerank_sync, chunks, query)
                top = chunks[:3]

                # Re-check after rerank. Cheap, and reranking is exactly the
                # kind of stage where a future refactor could reintroduce
                # unfiltered candidates from a cache.
                self._assert_permitted(top, groups)

                context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in top)
                try:
                    text = await self._generate(f"Q: {query}\nContext:\n{context}")
                except (LLMError, TimeoutError) as e:
                    # Degrade rather than fail: return the citations we found
                    # with an honest note. Better than a 500 for many use cases
                    # — but it MUST be labelled, or users cannot tell.
                    degraded = True
                    outcome = f"degraded:{type(e).__name__}"
                    text = "Generation unavailable; showing retrieved sources only."

                return Answer(
                    text=text,
                    retrieved=[c.chunk_id for c in chunks],
                    citations=[c.chunk_id for c in top],
                    trace_id=trace_id,
                    usage_tokens=self.llm.total_usage.total_tokens,
                    degraded=degraded,
                )

        except PermissionError_ as e:
            outcome = "permission_violation"
            log("SECURITY.assertion_failed", error=str(e))
            raise
        except TimeoutError:
            outcome = "deadline_exceeded"
            raise
        except asyncio.CancelledError:
            outcome = "client_disconnect"   # NOT an error; count separately
            raise
        finally:
            # Shielded, bounded audit write. Survives a shutdown drain — see 05.
            record = {
                "trace_id": trace_id,
                "principal": principal,
                "outcome": outcome,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(0.5):
                    await asyncio.shield(self._write_audit(record))


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

async def demo1_permission_trimming() -> None:
    banner("DEMO 1 — permission-trimmed retrieval, three identities, one query")

    svc = RagService()
    # One query touching ISC content, HR content, and legal content at once.
    query = "settlement units billed compensation band confidential"

    for principal, groups in [
        ("debdeep@contoso.com ", frozenset({"isc-all"})),
        ("hr.lead@contoso.com ", frozenset({"isc-all", "hr-only"})),
        ("legal@contoso.com   ", frozenset({"legal-only"})),
    ]:
        ans = await svc.answer(query, principal.strip(), groups)
        print(f"      {principal} groups={sorted(groups)}")
        print(f"        retrieved (post-trim): {ans.retrieved}")
        print(f"        cited (top-3):         {ans.citations}")

    print("""
      Identical query string, three different candidate sets. The ISC user
      never sees c5 (compensation) or c6 (settlement); the legal user sees
      ONLY c6 and none of the supply-chain chunks.

      Two consequences for concurrent code:
        * A cache keyed on the query alone would serve the legal chunk to the
          ISC user. The key must include identity or the resolved group set.
        * If you key on the group set, you have accepted an ACL staleness
          window. State it as a number — "up to 5 minutes stale" is a decision
          a reviewer can accept or reject; "we cache results" is not.""")


async def demo2_assertion_catches_a_broken_index() -> None:
    banner("DEMO 2 — the post-retrieval assertion catches an unfiltered source")

    svc = RagService()
    svc.indices = [BrokenIndex("legacy-index", 0.02)]  # someone wired this in

    try:
        await svc.answer("settlement terms confidential",
                         "debdeep@contoso.com", frozenset({"isc-all"}))
        print("      *** LEAK: the request succeeded — assertion did not fire ***")
    except PermissionError_ as e:
        print(f"      assertion fired, request failed CLOSED: {e}")
        print("      Without this check the caller would have received a fluent,")
        print("      well-cited answer built from a confidential legal document.")


async def demo3_concurrency_under_load() -> None:
    banner("DEMO 3 — 30 concurrent requests, mixed identities")

    global VERBOSE
    VERBOSE = False          # 30 requests x N events is unreadable; count instead
    svc = RagService()
    people = [
        ("a@contoso.com", frozenset({"isc-all"})),
        ("b@contoso.com", frozenset({"isc-all", "hr-only"})),
        ("c@contoso.com", frozenset({"legal-only"})),
    ]

    async def one(i: int) -> Answer:
        principal, groups = people[i % len(people)]
        return await svc.answer(f"units billed {i}", principal, groups)

    with Timer("30 concurrent requests"):
        results = await asyncio.gather(*(one(i) for i in range(30)),
                                       return_exceptions=True)

    ok = [r for r in results if isinstance(r, Answer) and not r.degraded]
    deg = [r for r in results if isinstance(r, Answer) and r.degraded]
    err = [r for r in results if isinstance(r, BaseException)]
    print(f"\n      ok={len(ok)} degraded={len(deg)} errored={len(err)}")
    print(f"      audit records written: {len(svc.audit)}  (must equal 30)")
    print(f"      llm client: {svc.llm.stats()}")

    # Report as k/n, not as a percentage — n=30 with an injected 25% error
    # rate cannot support a capability claim about the real system.
    print(f"\n      RESULT: {len(ok)}/30 clean, {len(deg)}/30 degraded, "
          f"{len(err)}/30 failed")
    print("      (k/n of named runs. A percentage here would imply a rate")
    print("      estimate this sample size cannot support.)")
    VERBOSE = True


async def demo4_deadline_and_disconnect() -> None:
    banner("DEMO 4 — deadline exceeded and client disconnect")

    svc = RagService()
    svc.indices = [SearchIndex("slow-index", 2.0)]

    print("\n  A) budget exceeded")
    try:
        await svc.answer("q", "a@contoso.com", frozenset({"isc-all"}), budget_s=0.2)
    except TimeoutError:
        print(f"      TimeoutError; audit written anyway: {svc.audit[-1]}")

    print("\n  B) client disconnects mid-request (task cancelled)")
    svc2 = RagService()
    svc2.indices = [SearchIndex("slow-index", 2.0)]
    t = asyncio.create_task(
        svc2.answer("q", "b@contoso.com", frozenset({"isc-all"}), budget_s=5.0)
    )
    await asyncio.sleep(0.1)
    t.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t
    await asyncio.sleep(0.1)
    print(f"      audit written on cancel: {svc2.audit[-1] if svc2.audit else 'NONE'}")
    print("      outcome is 'client_disconnect', not an error — count it")
    print("      separately or your error-rate dashboard will lie to you.")


async def demo5_streaming() -> None:
    banner("DEMO 5 — the streaming variant")

    svc = RagService()
    svc.llm = FakeLLMClient(seed=110, base_latency_s=0.03, jitter_s=0.0)

    async def stream_answer(query: str, principal: str, groups: frozenset[str]):
        trace_id = f"tr-{uuid.uuid4().hex[:8]}"
        trace_id_var.set(trace_id)
        principal_var.set(principal)

        chunks = await svc._retrieve(query, groups)
        svc._assert_permitted(chunks, groups)
        top = chunks[:3]

        # Send citations FIRST. The user sees provenance before prose, which
        # is the right ordering for a system whose answers must be auditable —
        # and it means a mid-stream failure still leaves them something usable.
        yield f"event: citations\ndata: {json.dumps([c.chunk_id for c in top])}\n\n"

        ctx = " ".join(c.text for c in top)
        async with contextlib.aclosing(svc.llm.stream(f"{query} {ctx}", n_tokens=6)) as s:
            async for tok in s:
                yield f"data: {json.dumps({'delta': tok})}\n\n"
        yield "event: done\ndata: {}\n\n"

    async with contextlib.aclosing(
        stream_answer("units billed", "a@contoso.com", frozenset({"isc-all"}))
    ) as s:
        async for frame in s:
            print(f"      {frame!r}")


# ---------------------------------------------------------------------------

async def main() -> None:
    await demo1_permission_trimming()
    await demo2_assertion_catches_a_broken_index()
    await demo3_concurrency_under_load()
    await demo4_deadline_and_disconnect()
    await demo5_streaming()

    banner("WHAT TO TAKE INTO A DESIGN REVIEW")
    print("""
  The concurrency decisions in this service that are worth defending out loud:

  1. Security filter is PRE-ranking, and there is a post-retrieval assertion
     that fails the request closed. Permission bugs fail silently and upward;
     the assertion is the only thing that makes them loud.
  2. Cache keys include the resolved group set, never the query alone, and
     carry a TTL that bounds ACL staleness. State the staleness window as a
     number — "we may serve up to 5 minutes of stale ACL" is a decision
     someone can accept or reject; "we cache" is not.
  3. Retrieval degrades on partial source failure; it never falls back to an
     unfiltered index. Fail closed, always.
  4. One end-to-end deadline per request, above the transport timeouts.
  5. Retries sit INSIDE the semaphore, so retrying does not silently double
     real concurrency. The rate bucket is shared per deployment, because the
     quota is per deployment.
  6. Audit writes are shielded and bounded, so they survive a shutdown drain.
  7. Client disconnect is counted separately from errors.
  8. Logs carry trace_id and principal and NEVER carry prompt or chunk text —
     telemetry readers are not the same audience as document readers.
  9. Reranking goes through to_thread; it is CPU-bound and would otherwise
     freeze every concurrent request.
 10. Results reported as k/n named runs, not percentages, at this sample size.
""")


if __name__ == "__main__":
    asyncio.run(main())
