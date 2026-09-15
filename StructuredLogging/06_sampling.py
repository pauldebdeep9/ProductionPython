"""
06 — Sampling: keeping the traces that matter and paying for the rest.

THE PROBLEM
-----------
At 500 requests/second with 8 spans each, you produce 4,000 spans/second —
roughly 350 million a day. Nobody stores that, and if you did you could not
query it. So you sample.

The question is not WHETHER but WHICH, and the naive answer — keep 1% at
random — throws away exactly the traces you need, because errors and slow
requests are rare by definition.

THE STRATEGIES, and their real trade-offs:
    HEAD    decide at the start of the trace, before you know the outcome.
            Cheap, simple, consistent across services. Blind to outcome.
    TAIL    decide at the end, when you know it failed or was slow.
            Keeps what matters. Needs the whole trace buffered somewhere.
    HYBRID  head-sample a baseline, force-keep on outcome. What most
            production systems actually run.

Run:  python 06_sampling.py
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field

from obs_lab import banner, new_trace_id, section, show

# ---------------------------------------------------------------------------
# A synthetic traffic model
# ---------------------------------------------------------------------------

@dataclass
class Request:
    trace_id: str
    duration_ms: float
    outcome: str          # ok | error | degraded
    tenant: str
    acl_violation: bool = False
    spans: int = 8


def generate_traffic(n: int, seed: int = 5) -> list[Request]:
    """A realistic mix: mostly fast successes, a long tail, rare errors."""
    rng = random.Random(seed)
    out: list[Request] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.015:
            outcome, dur = "error", rng.uniform(3, 40)
        elif r < 0.05:
            outcome, dur = "degraded", rng.uniform(900, 2500)
        elif r < 0.09:
            outcome, dur = "ok", rng.uniform(1800, 6000)     # slow tail
        else:
            outcome, dur = "ok", rng.uniform(200, 900)
        out.append(Request(
            trace_id=new_trace_id(), duration_ms=dur, outcome=outcome,
            tenant=rng.choice(["isc-sg", "isc-eu", "isc-us"]),
            acl_violation=rng.random() < 0.0004,
        ))
    return out


TRAFFIC = generate_traffic(20_000)


# ---------------------------------------------------------------------------
# PART 1 — why uniform random sampling fails
# ---------------------------------------------------------------------------

def part1_uniform() -> None:
    banner("PART 1 — 1% uniform sampling loses what you need")

    rng = random.Random(1)
    kept = [r for r in TRAFFIC if rng.random() < 0.01]

    def counts(rs: list[Request]) -> Counter[str]:
        return Counter(r.outcome for r in rs)

    total, sampled = counts(TRAFFIC), counts(kept)
    print(f"      {'outcome':<12} {'in traffic':>12} {'kept at 1%':>12}")
    print(f"      {'-' * 12} {'-' * 12} {'-' * 12}")
    for k in ("ok", "degraded", "error"):
        print(f"      {k:<12} {total[k]:>12,} {sampled[k]:>12,}")

    violations_total = sum(1 for r in TRAFFIC if r.acl_violation)
    violations_kept = sum(1 for r in kept if r.acl_violation)
    show("ACL violations in traffic", violations_total)
    show("ACL violations kept", violations_kept)

    print("""
    The errors survive in small numbers, which is survivable. The ACL
    violations mostly do not — and those are the ones where losing the trace
    means losing the only record of a security event.

    THE GENERAL PROBLEM: uniform sampling preserves the DISTRIBUTION, which is
    the opposite of what you want. You want to over-represent the rare and
    interesting and under-represent the common and boring.""")


# ---------------------------------------------------------------------------
# PART 2 — head sampling, done consistently
# ---------------------------------------------------------------------------

def head_sample(trace_id: str, rate: float) -> bool:
    """Decide from the TRACE ID, not from a random number.

    THIS IS THE CRITICAL DETAIL. With `random.random() < rate`, each SERVICE
    in a distributed trace makes its own independent decision, so you get
    traces where service A kept its spans and service B did not — a broken
    trace, which is worse than no trace.

    Hashing the trace id makes the decision DETERMINISTIC and IDENTICAL
    everywhere, with no coordination. Every service that sees this trace id
    reaches the same conclusion.

    (The real W3C mechanism is the `sampled` flag in `traceparent`, which
    propagates the upstream decision. Hashing is what the FIRST service does
    to make that decision.)
    """
    digest = hashlib.sha256(trace_id.encode()).digest()
    value = int.from_bytes(digest[:8], "big") / (2 ** 64)
    return value < rate


def part2_head() -> None:
    banner("PART 2 — head sampling must be deterministic")

    section("random per service — traces break")
    rng = random.Random(2)
    decisions = [rng.random() < 0.5 for _ in range(4)]
    show("gateway / rag / retriever / model", decisions)
    show("result", "a partial trace" if len(set(decisions)) > 1 else "consistent")

    section("hash of the trace id — every service agrees")
    for _ in range(4):
        t = new_trace_id()
        agree = {head_sample(t, 0.5) for _ in range(4)}
        show(f"trace {t[:8]}", f"all four services: {agree}")

    section("the rate is honoured across many traces")
    for rate in (0.01, 0.1, 0.5):
        kept = sum(1 for r in TRAFFIC if head_sample(r.trace_id, rate))
        show(f"rate={rate}", f"{kept:,}/{len(TRAFFIC):,} = "
                             f"{kept / len(TRAFFIC):.3f}")

    print("""
    Deterministic, consistent across services, and statistically correct.
    That is all three properties head sampling needs.

    ITS LIMIT is unavoidable: the decision is made before you know whether the
    request failed, so it cannot preferentially keep errors. PART 3.""")


# ---------------------------------------------------------------------------
# PART 3 — tail-ish sampling: decide on outcome
# ---------------------------------------------------------------------------

@dataclass
class SamplingPolicy:
    """The hybrid that most production systems land on.

    Head-sample a baseline for statistics, then FORCE-KEEP anything
    interesting. The force-keeps are rare, so they cost almost nothing.
    """

    base_rate: float = 0.01
    slow_threshold_ms: float = 2000.0
    keep_errors: bool = True
    keep_degraded: bool = True
    keep_violations: bool = True

    kept_by: Counter[str] = field(default_factory=Counter)

    def decide(self, r: Request) -> str | None:
        # ORDER MATTERS: check the force-keeps FIRST, so a trace that is both
        # an error and inside the base sample is attributed to the reason that
        # actually matters — otherwise your "why did we keep this" breakdown
        # is dominated by "baseline".
        if self.keep_violations and r.acl_violation:
            self.kept_by["acl_violation"] += 1
            return "acl_violation"
        if self.keep_errors and r.outcome == "error":
            self.kept_by["error"] += 1
            return "error"
        if self.keep_degraded and r.outcome == "degraded":
            self.kept_by["degraded"] += 1
            return "degraded"
        if r.duration_ms >= self.slow_threshold_ms:
            self.kept_by["slow"] += 1
            return "slow"
        if head_sample(r.trace_id, self.base_rate):
            self.kept_by["baseline"] += 1
            return "baseline"
        return None


def part3_hybrid() -> None:
    banner("PART 3 — hybrid: baseline plus force-keeps")

    policy = SamplingPolicy(base_rate=0.01)
    kept = [r for r in TRAFFIC if policy.decide(r) is not None]

    show("traffic", f"{len(TRAFFIC):,} requests")
    show("kept", f"{len(kept):,} ({len(kept) / len(TRAFFIC):.2%})")
    section("why each trace was kept")
    for reason, n in policy.kept_by.most_common():
        print(f"      {reason:<16} {n:>7,}")

    section("coverage of the things that matter")
    for label, pred in [
        ("errors", lambda r: r.outcome == "error"),
        ("degraded", lambda r: r.outcome == "degraded"),
        ("ACL violations", lambda r: r.acl_violation),
        ("slow (>2s)", lambda r: r.duration_ms >= 2000),
        ("ordinary successes", lambda r: r.outcome == "ok"
         and r.duration_ms < 2000),
    ]:
        total = sum(1 for r in TRAFFIC if pred(r))
        got = sum(1 for r in kept if pred(r))
        pct = got / total if total else 0
        print(f"      {label:<22} {got:>6,}/{total:>7,}  {pct:>7.1%}")

    print("""
    100% of errors, 100% of ACL violations, 100% of slow requests — and about
    1% of the ordinary successes that make up the bulk of traffic.

    BUT READ THE HEADLINE NUMBER AGAIN: 9.77% kept, against a 1% baseline.
    I wrote "the force-keeps are nearly free" in the first draft of this
    script and the measurement contradicted it. They are not free here — they
    are nine times the baseline.

    WHY: this traffic has 3.5% degraded and 5% slow-tail requests, so
    force-keeping both means the force-keeps DOMINATE. The `slow` bucket alone
    is 799 traces against 179 from the baseline.

    THE LESSON, which is more useful than the original claim: force-keeps are
    cheap only if the conditions are RARE, and "rare" is a property of your
    traffic, not of the policy.

      slow_threshold_ms=2000   matched 5% here -> expensive
      slow_threshold_ms=5000   would match ~1% -> cheap
      keep_degraded=True       at a 3.5% degraded rate this is a major cost
                               line, and arguably the right response is to fix
                               the degradation rather than to sample it

    Set the slow threshold from your OWN p99 rather than a round number. A
    threshold at p99 keeps 1% by construction, whatever your distribution is.

    THE HONEST LIMITATION: "tail" sampling here is decided at the end of the
    request IN ONE SERVICE. True distributed tail sampling requires a
    collector that buffers all spans of a trace across all services until the
    trace completes, then decides. That is what the OTel Collector's
    `tailsamplingprocessor` does, and it costs memory proportional to
    (trace rate x trace duration). Worth knowing before you promise it.""")


# ---------------------------------------------------------------------------
# PART 4 — the cost arithmetic
# ---------------------------------------------------------------------------

def part4_cost() -> None:
    banner("PART 4 — what observability actually costs")

    section("the volume, worked through")
    rps = 500
    spans_per_request = 8
    bytes_per_span = 900          # a span with ~15 attributes, JSON
    log_lines = 1                 # one wide event
    bytes_per_log = 1400

    span_gb_day = rps * spans_per_request * bytes_per_span * 86400 / 1e9
    log_gb_day = rps * log_lines * bytes_per_log * 86400 / 1e9
    show("requests/second", f"{rps:,}")
    show("spans/day (unsampled)", f"{rps * spans_per_request * 86400:,.0f}")
    show("span volume/day", f"{span_gb_day:,.0f} GB")
    show("log volume/day (1 wide event)", f"{log_gb_day:,.0f} GB")

    # Illustrative ingestion pricing. Check current rates — this is a
    # magnitude exercise, not a quote.
    price_per_gb = 2.30
    show("unsampled traces/month at $2.30/GB",
         f"${span_gb_day * 30 * price_per_gb:,.0f}")
    show("at 1% baseline + force-keeps (~5%)",
         f"${span_gb_day * 0.05 * 30 * price_per_gb:,.0f}")
    show("logs/month (1 wide event)",
         f"${log_gb_day * 30 * price_per_gb:,.0f}")

    print("""
    TWO OBSERVATIONS FROM THOSE NUMBERS:

    1. TRACES DOMINATE. Eight spans at ~900 bytes each is several times the
       volume of one wide log event. That is why sampling is a trace concern
       first — and why "one wide event, always kept" plus "sampled traces" is
       a better default than the reverse.

    2. THE FIRST LEVER IS SPAN COUNT, not the sampling rate. Going from 25
       spans per request to 8 is a 3x saving that costs you nothing, because
       the 17 you removed were per-chunk spans nobody read. Cut granularity
       before you cut coverage.

    THE THIRD LEVER, and the one people forget: ATTRIBUTE SIZE. A span with a
    chunk-text attribute is not 900 bytes, it is 9,000. Which means the
    content rules in script 04 are also a cost control, not only a privacy
    one.""")

    section("where the bytes go in one span")
    for part, size in [("trace/span/parent ids", 112), ("name + timings", 90),
                       ("kind, status", 40), ("12 small attributes", 420),
                       ("resource attributes", 240)]:
        print(f"      {part:<28} {size:>5} bytes")
    print(f"      {'-' * 28} {'-' * 5}")
    print(f"      {'total':<28} {902:>5} bytes")
    print("      Resource attributes (service name, version, k8s pod, region)")
    print("      repeat on EVERY span. Most backends de-duplicate them; if")
    print("      yours bills raw, that is a quarter of your spend on constants.")


# ---------------------------------------------------------------------------
# PART 5 — what never gets sampled away
# ---------------------------------------------------------------------------

def part5_never_sample() -> None:
    banner("PART 5 — the signals that must survive at 100%")

    print("""
    Sampling applies to TRACES. It must not apply to:

      METRICS         they are already aggregated. Sampling them means your
                      counts are wrong, and a 1%-sampled error count is not
                      an error count.

      SECURITY EVENTS acl.violation, auth failures, permission changes. Losing
                      99% of these makes them useless — a rare event sampled
                      at 1% is an event you do not have.

      AUDIT RECORDS   they are already triggered and low-volume.

      THE WIDE REQUEST EVENT  this is your analytical dataset. If it is
                      sampled, every "what fraction of requests..." question
                      gets an approximate answer with no confidence interval.

    THE SHAPE THAT WORKS:
        metrics       100%, always
        wide event    100%, one per request
        security      100%, always
        traces        1-5% baseline + force-keep on error/slow/violation
        audit         triggered + ~1% random
        DEBUG logs    0% in production

    THE SUBTLE ONE: if traces are sampled and the wide event is not, they must
    share a trace_id so you can pivot from an unsampled event to a sampled
    trace and know when the trace is simply absent. Record the SAMPLING
    DECISION on the wide event — `trace.sampled: false` — so "no trace for
    this id" is distinguishable from "the trace is missing", which are very
    different investigations.""")


def main() -> None:
    part1_uniform()
    part2_head()
    part3_hybrid()
    part4_cost()
    part5_never_sample()

    banner("SUMMARY")
    print("""
  * Uniform sampling preserves the distribution, which is exactly wrong — you
    want to over-represent the rare and interesting.
  * Head sampling must be a HASH OF THE TRACE ID, not `random()`, or services
    disagree and traces break.
  * Hybrid is the practical answer: ~1% baseline + force-keep errors, slow
    requests, degraded responses, and security events. But MEASURE what
    fraction of traffic your force-keep conditions match — at a 5% slow rate
    they dominate the bill. Set the slow threshold from your own p99.
  * Distributed tail sampling needs a buffering collector; know that before
    promising it.
  * Cut SPAN COUNT and ATTRIBUTE SIZE before cutting sampling rate. The
    content rules from script 04 are also a cost control.
  * Never sample metrics, security events, audit records, or the wide request
    event. Record the sampling decision on the wide event so a missing trace
    is explicable.
""")


if __name__ == "__main__":
    main()
