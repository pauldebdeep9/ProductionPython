"""
06 — Fallbacks and graceful degradation: what to serve when you cannot serve.

THE DESIGN QUESTION
-------------------
Retries assume the thing will work if you wait. Fallbacks assume it will not,
and ask a different question: what is the best answer we can still produce?

For a GenAI system the ladder is unusually rich, because there are many
partial answers that are genuinely useful:

    full RAG answer with citations           ideal
    answer from a smaller/cheaper model      slightly worse, much cheaper
    answer from a different region           same quality, higher latency
    cached answer from 10 minutes ago        stale but often fine
    retrieved passages with NO generation    no synthesis, but sourced
    "unavailable, here is who to contact"    honest failure

Most systems implement rung 1 and rung 6 and nothing in between. The middle
rungs are where the availability actually comes from.

THE RULE THAT GOVERNS ALL OF IT
-------------------------------
A degraded answer must be LABELLED as degraded. An unlabelled fallback is
worse than an error, because the user cannot tell they are getting reduced
service and will make decisions on it as though it were the real thing.

And the security corollary: NEVER fall back in a direction that widens access.
Falling back from a permission-trimmed index to an untrimmed one is a data
leak wearing a resilience costume.

Run:  python 06_fallbacks.py
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field

from failure_lab import (
    AppError,
    Fail,
    FaultInjector,
    Metrics,
    Ok,
    ServiceUnavailable,
    banner,
    section,
)

# ---------------------------------------------------------------------------
# The answer object carries its own provenance
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    text: str
    source: str                      # which rung produced this
    degraded: bool = False
    quality: str = "full"            # full | reduced | sourced_only | none
    citations: list[str] = field(default_factory=list)
    age_seconds: float = 0.0
    cost: float = 0.0
    attempts: list[str] = field(default_factory=list)

    def user_notice(self) -> str:
        """What the USER is told. This is not optional decoration — it is the
        contract that makes degradation honest."""
        if not self.degraded:
            return ""
        return {
            "reduced": "Answered using a smaller model; quality may be lower.",
            "sourced_only": "Answer generation is unavailable. Showing the "
                            "source passages that matched your question.",
            "none": "This service is temporarily unavailable.",
        }.get(self.quality, "Response is degraded.")


# ---------------------------------------------------------------------------
# PART 1 — the fallback chain
# ---------------------------------------------------------------------------

class FallbackChain:
    """Try rungs in order; return the first that succeeds; record the path.

    THE THREE MISTAKES:
      1. No deadline. A 4-rung chain with a 30s timeout each is a 2-minute
         hang. The deadline must span the WHOLE chain, not each rung.
      2. Falling back on non-transient errors. If the prompt was rejected by
         a content filter, every rung rejects it. You have just spent 4x the
         money to fail 4x slower.
      3. Not labelling. See above.
    """

    def __init__(self, deadline_s: float = 3.0, metrics: Metrics | None = None) -> None:
        self.deadline_s = deadline_s
        self.metrics = metrics
        self.rungs: list[tuple[str, object, str]] = []

    def add(self, name: str, fn, quality: str = "full") -> FallbackChain:
        self.rungs.append((name, fn, quality))
        return self

    async def run(self, query: str) -> Answer:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.deadline_s
        path: list[str] = []

        for i, (name, fn, quality) in enumerate(self.rungs):
            remaining = deadline - loop.time()
            if remaining <= 0:
                path.append(f"{name}:deadline")
                break
            try:
                async with asyncio.timeout(remaining):
                    ans: Answer = await fn(query)
                ans.attempts = [*path, f"{name}:ok"]
                ans.degraded = i > 0
                ans.quality = quality if i > 0 else "full"
                if self.metrics:
                    self.metrics.incr(f"served_by.{name}")
                return ans
            except AppError as e:
                # ONLY fall back on errors a different provider could survive.
                # A content filter or a validation error will reject the same
                # prompt at every rung.
                from failure_lab import Disposition, classify
                if classify(e) is Disposition.FAIL_FAST:
                    path.append(f"{name}:fail_fast")
                    if self.metrics:
                        self.metrics.incr("chain.aborted_fail_fast")
                    break
                path.append(f"{name}:{type(e).__name__}")
                if self.metrics:
                    self.metrics.incr(f"fallback_from.{name}")
            except TimeoutError:
                path.append(f"{name}:timeout")

        if self.metrics:
            self.metrics.incr("chain.exhausted")
        return Answer(text="", source="none", degraded=True, quality="none",
                      attempts=path)


async def part1_chain() -> None:
    banner("PART 1 — a fallback ladder for a RAG service")

    m = Metrics()

    def make_rung(name: str, injector: FaultInjector, cost: float,
                  text: str, cites: list[str]):
        async def rung(query: str) -> Answer:
            await injector.maybe_fail()
            return Answer(text=text, source=name, citations=cites, cost=cost)
        return rung

    section("all healthy — the primary serves")
    chain = FallbackChain(deadline_s=2.0, metrics=m)
    chain.add("gpt-4o-primary",
              make_rung("gpt-4o-primary", FaultInjector(script=[Ok(0.02)]),
                        0.0100, "Full synthesised answer.", ["c1", "c2"]))
    chain.add("gpt-4o-mini",
              make_rung("gpt-4o-mini", FaultInjector(script=[Ok(0.02)]),
                        0.0006, "Shorter answer.", ["c1"]), quality="reduced")
    a = await chain.run("why is invoice 88 over-billed?")
    print(f"      served by: {a.source}  degraded={a.degraded}  path={a.attempts}")

    section("primary down — degrade to the smaller model")
    chain2 = FallbackChain(deadline_s=2.0, metrics=m)
    chain2.add("gpt-4o-primary",
               make_rung("gpt-4o-primary",
                         FaultInjector(script=[Fail(ServiceUnavailable, 0.01)]),
                         0.0100, "x", []))
    chain2.add("gpt-4o-eastus2",
               make_rung("gpt-4o-eastus2",
                         FaultInjector(script=[Fail(ServiceUnavailable, 0.01)]),
                         0.0100, "x", []))
    chain2.add("gpt-4o-mini",
               make_rung("gpt-4o-mini", FaultInjector(script=[Ok(0.02)]),
                         0.0006, "Shorter answer, same citations.", ["c1", "c2"]),
               quality="reduced")
    a2 = await chain2.run("why is invoice 88 over-billed?")
    print(f"      served by: {a2.source}  degraded={a2.degraded}")
    print(f"      path: {a2.attempts}")
    print(f"      user sees: {a2.user_notice()!r}")

    section("everything down — retrieval-only, still useful")
    chain3 = FallbackChain(deadline_s=2.0, metrics=m)
    chain3.add("gpt-4o-primary",
               make_rung("p", FaultInjector(script=[Fail(ServiceUnavailable, 0.01)]),
                         0.01, "x", []))
    chain3.add("gpt-4o-mini",
               make_rung("m", FaultInjector(script=[Fail(ServiceUnavailable, 0.01)]),
                         0.0006, "x", []))

    async def retrieval_only(query: str) -> Answer:
        return Answer(
            text="(no generated answer)\n"
                 "  [c1] PO 1001 quantity 400 units\n"
                 "  [c3] Invoice 88 billed 402 units",
            source="retrieval-only", citations=["c1", "c3"], cost=0.0)

    chain3.add("retrieval-only", retrieval_only, quality="sourced_only")
    a3 = await chain3.run("why is invoice 88 over-billed?")
    print(f"      served by: {a3.source}")
    print(f"      user sees: {a3.user_notice()!r}")
    print(f"      content:\n        {a3.text.splitlines()[1].strip()}")
    print(f"        {a3.text.splitlines()[2].strip()}")
    print("""
      The retrieval-only rung is the one teams skip, and it is the most
      valuable. The user gets the two passages that answer their question,
      just without synthesis. For an ISC analyst chasing a quantity mismatch
      that is often ENTIRELY sufficient — and it costs nothing, needs no
      model, and cannot hallucinate.""")

    section("non-transient error — the chain aborts instead of paying 3x")
    chain4 = FallbackChain(deadline_s=2.0, metrics=m)
    from failure_lab import ContentFiltered
    for name in ("primary", "mini", "fallback-region"):
        chain4.add(name, make_rung(
            name, FaultInjector(script=[Fail(ContentFiltered, 0.01)]),
            0.01, "x", []))
    a4 = await chain4.run("blocked prompt")
    print(f"      path: {a4.attempts}")
    print("      Stopped after ONE rung. A content filter rejects the same")
    print("      prompt everywhere; trying two more models is pure waste.")


# ---------------------------------------------------------------------------
# PART 2 — cache as a fallback tier
# ---------------------------------------------------------------------------

class StaleWhileErrorCache:
    """Serve stale content when the origin fails.

    Two TTLs, which is the key idea:
      fresh_ttl  — below this, serve from cache without calling the origin
      stale_ttl  — above fresh but below stale, serve ONLY if the origin fails

    This turns an outage into a staleness problem, which is almost always the
    better problem to have.

    THE PERMISSION TRAP: the cache key must include the caller's identity or
    resolved group set. A cache keyed on query text alone will happily serve
    one user's permission-trimmed answer to another user — and it will do so
    most eagerly during an incident, when nobody is watching closely.
    """

    def __init__(self, fresh_ttl: float = 1.0, stale_ttl: float = 60.0) -> None:
        self.fresh_ttl = fresh_ttl
        self.stale_ttl = stale_ttl
        self.entries: dict[str, tuple[float, Answer]] = {}
        self.hits_fresh = 0
        self.hits_stale = 0
        self.misses = 0

    @staticmethod
    def key(query: str, groups: frozenset[str]) -> str:
        # Identity is PART OF THE KEY. Not an afterthought.
        gid = hashlib.sha256(",".join(sorted(groups)).encode()).hexdigest()[:8]
        return f"{gid}:{hashlib.sha256(query.encode()).hexdigest()[:12]}"

    def get_fresh(self, k: str) -> Answer | None:
        e = self.entries.get(k)
        if e and time.monotonic() - e[0] <= self.fresh_ttl:
            self.hits_fresh += 1
            return e[1]
        return None

    def get_stale(self, k: str) -> Answer | None:
        e = self.entries.get(k)
        if e and time.monotonic() - e[0] <= self.stale_ttl:
            self.hits_stale += 1
            a = Answer(**{**e[1].__dict__})
            a.age_seconds = time.monotonic() - e[0]
            return a
        self.misses += 1
        return None

    def put(self, k: str, a: Answer) -> None:
        self.entries[k] = (time.monotonic(), a)


async def part2_cache() -> None:
    banner("PART 2 — stale-while-error: turn an outage into a staleness problem")

    cache = StaleWhileErrorCache(fresh_ttl=0.05, stale_ttl=10.0)
    origin_up = True

    async def origin(query: str) -> Answer:
        if not origin_up:
            raise ServiceUnavailable("model endpoint down")
        await asyncio.sleep(0.01)
        return Answer(text="Invoice 88 was billed for 2 extra units.",
                      source="origin", citations=["c1", "c3"])

    async def serve(query: str, groups: frozenset[str]) -> Answer:
        k = cache.key(query, groups)
        fresh = cache.get_fresh(k)
        if fresh is not None:
            return fresh
        try:
            a = await origin(query)
            cache.put(k, a)
            return a
        except AppError:
            stale = cache.get_stale(k)
            if stale is not None:
                stale.degraded = True
                stale.source = "cache(stale)"
                return stale
            raise

    isc = frozenset({"isc-all"})
    a = await serve("invoice 88 variance", isc)
    print(f"      origin healthy -> {a.source}")

    await asyncio.sleep(0.06)                 # let it go stale-but-usable
    origin_up = False
    a2 = await serve("invoice 88 variance", isc)
    print(f"      origin DOWN    -> {a2.source}  age={a2.age_seconds:.2f}s "
          f"degraded={a2.degraded}")

    try:
        await serve("a question never asked before", isc)
    except AppError as e:
        print(f"      cold query during outage -> {type(e).__name__}: {e}")

    section("the permission trap")
    hr = frozenset({"isc-all", "hr-only"})
    print(f"      key for isc-all group : {cache.key('q', isc)}")
    print(f"      key for hr-only group : {cache.key('q', hr)}")
    print("      Different keys. A query-only key would serve the HR user's")
    print("      cached answer to the ISC user — and stale-while-error makes")
    print("      that MORE likely, because it serves cache hardest during an")
    print("      incident. State your ACL staleness window as a number:")
    print(f"      with stale_ttl={cache.stale_ttl}s you may serve up to "
          f"{cache.stale_ttl}s of stale permissions.")


# ---------------------------------------------------------------------------
# PART 3 — fail open or fail closed?
# ---------------------------------------------------------------------------

async def part3_fail_direction() -> None:
    banner("PART 3 — fail OPEN or fail CLOSED: the decision per component")

    print("""
    When a component fails, you either let traffic through (fail open) or
    block it (fail closed). Getting this backwards is how resilience work
    creates security incidents.

    FAIL CLOSED — deny when the check fails:
      authentication            no token service => no access
      authorization / ACL       cannot resolve groups => serve nothing
      permission-trimmed search => NEVER fall back to an untrimmed index
      content safety on OUTPUT  filter down => do not emit
      export-control checks     classifier down => do not release

    FAIL OPEN — proceed when the check fails:
      analytics / telemetry     metrics sink down => serve the user anyway
      caching                   cache down => go to origin
      personalisation / ranking => serve generic results
      recommendation sidebars   => omit the panel
      non-blocking enrichment   => omit the enrichment

    THE TEST: if this component were removed entirely, would the result be
    merely WORSE, or would it be WRONG/UNSAFE? Worse => fail open. Wrong or
    unsafe => fail closed.

    THE ANTI-PATTERN THAT SHOWS UP IN REAL INCIDENT REPORTS:

        try:
            chunks = await permission_trimmed_search(q, groups)
        except Exception:
            chunks = await plain_search(q)      # "so the demo still works"

    That line is a data leak. It was probably written during a demo, survived
    review because it looks like defensive programming, and will fire for the
    first time during an incident when the search filter service is
    struggling — i.e. when nobody is reading logs carefully.

    The correct version raises, and the fallback chain's next rung is 'no
    answer', not 'unfiltered answer'.""")

    section("demonstration")

    async def trimmed_search(groups: frozenset[str]) -> list[str]:
        raise ServiceUnavailable("ACL resolver unavailable")

    async def wrong_way() -> list[str]:
        try:
            return await trimmed_search(frozenset({"isc-all"}))
        except AppError:
            return ["c1", "c5-HR-CONFIDENTIAL", "c6-LEGAL-CONFIDENTIAL"]

    async def right_way() -> list[str]:
        try:
            return await trimmed_search(frozenset({"isc-all"}))
        except AppError:
            raise        # fail closed; the chain degrades to "no answer"

    print(f"      fail-open version returns: {await wrong_way()}")
    print("        <- two confidential chunks, to a user with only isc-all")
    try:
        await right_way()
    except AppError as e:
        print(f"      fail-closed version raises: {type(e).__name__}")
        print("        <- user gets 'temporarily unavailable', which is correct")


# ---------------------------------------------------------------------------
# PART 4 — cost and quality accounting for degradation
# ---------------------------------------------------------------------------

async def part4_economics() -> None:
    banner("PART 4 — degradation is a cost/quality decision, so measure it")

    print("""
    Every fallback rung has three numbers. If you cannot state them, you
    cannot defend the ladder in a design review:

      rung                 rel. cost   rel. quality   availability added
      -------------------  ---------   ------------   ------------------
      gpt-4o primary          1.00         1.00              baseline
      gpt-4o other region     1.00         1.00         +regional failure
      gpt-4o-mini             0.06         ~0.85        +model/quota failure
      cached (stale <10m)     0.00         ~0.95*       +total model failure
      retrieval-only          0.00         ~0.40        +any generation failure
      honest error            0.00          0.00              —

      * cached quality is high for the question that was asked before, and
        zero for anything else. Cache hit rate is doing all the work in that
        number, so report it alongside.

    WHAT TO TRACK IN PRODUCTION:
      served_by.<rung>        distribution across rungs. If 30% of traffic is
                              served by the mini fallback, your 'primary'
                              is not your primary and your quality metrics
                              are measuring something you did not intend.
      degraded_rate           share of responses labelled degraded.
      fallback_from.<rung>    which rung is failing, so you know where to look.
      chain.exhausted         hard failures. This is your real error rate.

    THE MEASUREMENT TRAP: if fallbacks are unlabelled, your success rate looks
    like 99.9% while a third of users are getting the cheap model. Labelling
    is not just user honesty — it is the only way your own metrics stay true.""")

    m = Metrics()
    for rung, n in [("gpt-4o-primary", 780), ("gpt-4o-mini", 190),
                    ("cache-stale", 25), ("retrieval-only", 4)]:
        m.incr(f"served_by.{rung}", n)
    m.incr("chain.exhausted", 1)
    total = 1000
    print("\n      example distribution over 1000 requests:")
    for k in sorted(m.counters):
        if k.startswith("served_by."):
            print(f"        {k.split('.', 1)[1]:<18} {m.counters[k]:>5} "
                  f"({m.counters[k] / total:>5.1%})")
    degraded = total - m.counters["served_by.gpt-4o-primary"]
    print(f"        {'-' * 18}")
    print(f"        degraded rate      {degraded / total:>5.1%}")
    print(f"        hard failures      {m.counters['chain.exhausted'] / total:>5.1%}")
    print("      A naive 'success rate' here reads 99.9%. The number that")
    print("      matters for user experience is the 22% degraded rate.")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_chain()
    await part2_cache()
    await part3_fail_direction()
    await part4_economics()

    banner("SUMMARY")
    print("""
  * Build the middle rungs: other region, cheaper model, stale cache,
    retrieval-only. Most systems jump from 'perfect' to 'error'.
  * ONE deadline across the whole chain, not per rung.
  * Do not fall back on FAIL_FAST errors — a content filter rejects the same
    prompt everywhere.
  * Label every degraded response. Unlabelled fallbacks corrupt your own
    metrics as well as misleading users.
  * Cache keys include identity. Stale-while-error serves cache hardest during
    an incident, which is exactly when a query-only key leaks.
  * Fail CLOSED on anything that gates access; fail OPEN on anything that only
    improves the result.
  * Never fall back from a permission-trimmed source to an untrimmed one.
  * Track served_by.<rung> — if 30% is served by the fallback, it is not a
    fallback any more.
""")


if __name__ == "__main__":
    asyncio.run(main())
