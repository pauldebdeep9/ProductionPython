"""
04 — Timeouts and hedging: the tail is the failure.

THE OBSERVATION THAT DRIVES THIS
--------------------------------
Most "failures" in a healthy distributed system are not errors. They are
requests that took 40 seconds when p50 was 400ms. Nothing threw. Nothing
logged. The user left.

Two tools address this, and they pull in opposite directions:

  TIMEOUTS convert a slow request into a fast failure. Necessary, but they
  destroy work — a request cancelled at 30s consumed 30s of capacity and
  produced nothing.

  HEDGING sends a second request when the first is slow, and takes whichever
  finishes. It converts tail latency into extra load. Done carelessly it is a
  load amplifier that turns a brownout into an outage.

The connective tissue is knowing your latency distribution. Every number in a
timeout or hedge config should be derived from a measured percentile, not
chosen because it is round.

Run:  python 04_timeouts_hedging.py
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time

from failure_lab import Metrics, banner, section

# ---------------------------------------------------------------------------
# A latency model with a realistic tail
# ---------------------------------------------------------------------------

class LatencyModel:
    """Bimodal: a fast mode plus an occasional slow mode.

    Real LLM endpoint latency looks like this — most requests are served from
    a warm path, a few hit a cold start, a queue, a retry inside the provider,
    or a noisy neighbour. A single normal distribution badly understates the
    tail and will make you set timeouts too tight.
    """

    # NOTE ON SCALE: these means are 10x smaller than realistic LLM latency
    # (p50 ~200ms, tail ~2s) so the demo finishes in seconds. The SHAPE is
    # what matters — every conclusion below is about ratios, not absolutes.
    # Multiply every number in this script by 10 to read it as real latency.
    def __init__(self, fast_mean: float = 0.020, slow_mean: float = 0.20,
                 slow_rate: float = 0.08, seed: int = 0) -> None:
        self.fast_mean = fast_mean
        self.slow_mean = slow_mean
        self.slow_rate = slow_rate
        self._rng = random.Random(seed)

    def sample(self) -> float:
        if self._rng.random() < self.slow_rate:
            return self._rng.gammavariate(2.0, self.slow_mean / 2)
        return self._rng.gammavariate(4.0, self.fast_mean / 4)


async def call_with_latency(model: LatencyModel) -> float:
    d = model.sample()
    await asyncio.sleep(d)
    return d


# ---------------------------------------------------------------------------
# PART 1 — derive the timeout from the distribution
# ---------------------------------------------------------------------------

def percentiles(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    def p(q: float) -> float:
        return v[min(len(v) - 1, int(q * len(v)))]
    return {"p50": p(0.50), "p90": p(0.90), "p95": p(0.95),
            "p99": p(0.99), "max": v[-1]}


async def part1_sizing() -> None:
    banner("PART 1 — a timeout is a percentile decision, not a round number")

    model = LatencyModel(seed=1)
    samples = [model.sample() for _ in range(5000)]
    pcts = percentiles(samples)

    print("    measured latency distribution:")
    for k, v in pcts.items():
        print(f"      {k:<5} {v * 1000:>8.0f}ms")

    print(f"\n    {'timeout':>10} {'requests cut off':>18} {'capacity wasted':>18}")
    print(f"    {'-' * 10} {'-' * 18} {'-' * 18}")
    for t in (0.05, 0.10, 0.20, 0.50, 1.00):
        cut = sum(1 for s in samples if s > t)
        # Work destroyed: every cut-off request consumed `t` seconds and
        # produced nothing.
        wasted = cut * t
        total = sum(min(s, t) for s in samples)
        print(f"    {t * 1000:>8.0f}ms {cut / len(samples):>17.1%} "
              f"{wasted / total:>17.1%}")

    print("""
    THE TRADE, made explicit:
      too tight  -> you cut off requests that would have succeeded, and every
                    one of them still consumed the full timeout of capacity
                    before being discarded. A very tight timeout on a slow
                    dependency can spend 100% of your capacity producing 0%
                    of your answers.
      too loose  -> slow requests occupy connections and worker slots,
                    concurrency climbs, and you queue behind them.

    A DEFENSIBLE RULE: set the timeout at roughly p99.5 of the healthy
    distribution, then verify what fraction of capacity that wastes. If a
    p99.5 timeout wastes more than a few percent of capacity, the problem is
    the dependency's tail, not your timeout — and hedging or a fallback is the
    right tool, not a tighter number.

    For LLM calls specifically, size against OUTPUT LENGTH, not a global
    constant. A 2000-token generation legitimately takes 10x a 200-token one,
    so one timeout for both is either useless or hostile. Scale it:
    `timeout = ttft_budget + max_tokens * per_token_budget`.""")


# ---------------------------------------------------------------------------
# PART 2 — hedged requests
# ---------------------------------------------------------------------------

async def hedged_call(
    make_call,
    *,
    hedge_after: float,
    max_hedges: int = 1,
    call_timeout: float = 10.0,
    metrics: Metrics | None = None,
):
    """Send a backup request if the primary has not answered within
    `hedge_after`. Return the first result; cancel the losers.

    The key property: hedging only fires for the SLOW tail. If hedge_after is
    p95, you send at most 5% extra requests but you cut p99 dramatically —
    because the probability of TWO independent requests both hitting the slow
    mode is roughly the square of one doing so.

    PRECONDITIONS, all of which must hold:
      1. The operation is IDEMPOTENT or read-only. Hedging a write duplicates
         it (see 03).
      2. hedge_after >= p90, ideally p95. Set it near p50 and you have doubled
         your traffic, which is a load amplifier, not a latency fix.
      3. You have headroom. Hedging under saturation makes saturation worse:
         the extra requests queue behind the requests they were meant to
         overtake, and everything gets slower.
      4. There is a hedge BUDGET (see part 3), so hedging self-limits when the
         whole dependency is slow rather than when one request is unlucky.
    """
    tasks: set[asyncio.Task] = set()
    if metrics:
        metrics.incr("requests")

    try:
        async with asyncio.timeout(call_timeout):
            primary = asyncio.create_task(make_call(), name="primary")
            tasks.add(primary)
            if metrics:
                metrics.incr("calls_sent")

            for hedge_n in range(max_hedges + 1):
                done, _pending = await asyncio.wait(
                    tasks, timeout=hedge_after,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    winner = done.pop()
                    if metrics:
                        metrics.incr("hedges_used" if winner.get_name() != "primary"
                                     else "primary_won")
                    return winner.result()

                if hedge_n < max_hedges:
                    t = asyncio.create_task(make_call(), name=f"hedge-{hedge_n}")
                    tasks.add(t)
                    if metrics:
                        metrics.incr("calls_sent")
                        metrics.incr("hedges_sent")

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            return done.pop().result()
    finally:
        # ALWAYS cancel and reap the losers. A hedge you forget to cancel is
        # pure waste that keeps consuming the dependency's capacity after you
        # stopped caring about the answer.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def part2_hedging() -> None:
    banner("PART 2 — hedging: trading a little load for a lot of tail")

    n = 250

    async def measure(hedge_after: float | None) -> tuple[dict, Metrics]:
        model = LatencyModel(seed=5)
        m = Metrics()
        lat = []
        for _ in range(n):
            t0 = time.perf_counter()
            if hedge_after is None:
                m.incr("requests")
                m.incr("calls_sent")
                await call_with_latency(model)
            else:
                await hedged_call(lambda: call_with_latency(model),
                                  hedge_after=hedge_after, call_timeout=30.0,
                                  metrics=m)
            lat.append(time.perf_counter() - t0)
        return percentiles(lat), m

    print(f"    {'strategy':<26} {'p50':>8} {'p95':>8} {'p99':>8} "
          f"{'extra load':>11}")
    print(f"    {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 11}")

    base_pcts, base_m = await measure(None)
    base_calls = base_m.counters["calls_sent"]
    print(f"    {'no hedging':<26} {base_pcts['p50'] * 1000:>7.0f}ms "
          f"{base_pcts['p95'] * 1000:>7.0f}ms {base_pcts['p99'] * 1000:>7.0f}ms "
          f"{'—':>11}")

    for ha, label in [(0.020, "hedge at ~p50"),
                      (0.045, "hedge at ~p90"),
                      (0.110, "hedge at ~p95")]:
        p, m = await measure(ha)
        extra = (m.counters["calls_sent"] - base_calls) / base_calls
        print(f"    {label:<26} {p['p50'] * 1000:>7.0f}ms "
              f"{p['p95'] * 1000:>7.0f}ms {p['p99'] * 1000:>7.0f}ms "
              f"{extra:>10.1%}")

    print("""
    Read the extra-load column against the p99 column. Hedging near p50 buys
    tail improvement at a large load cost. Hedging near p95 buys most of the
    same improvement for a small fraction of the extra requests.

    That is the whole design: hedge far enough out that you only pay for the
    requests that were genuinely unlucky.""")


# ---------------------------------------------------------------------------
# PART 3 — hedge budgets, and when hedging makes things worse
# ---------------------------------------------------------------------------

async def part3_hedge_budget() -> None:
    banner("PART 3 — hedging under saturation makes it WORSE")

    print("""
    Hedging assumes the slowness is IDIOSYNCRATIC — this request got unlucky,
    another one probably will not. That assumption fails exactly when the
    dependency is overloaded, because then every request is slow for the same
    reason, and your hedges:
      * do not finish faster (they queue behind the same congestion),
      * add ~2x load to something already saturated,
      * push it further into congestion collapse.

    So hedging needs the same protection retries do: a BUDGET capping hedges
    at a small fraction of requests (5-10%). Above that, stop hedging.

    A budget also gives you a free signal: hedge budget exhaustion means
    "slowness is now systemic, not idiosyncratic", which is precisely the
    condition where you should stop hedging and start shedding.""")

    section("simulated: a saturated dependency where hedging does not help")

    class SaturatedDependency:
        """Latency grows with in-flight concurrency — the queueing effect that
        hedging cannot outrun, because the hedge joins the same queue."""

        def __init__(self, capacity: int = 8, base: float = 0.1) -> None:
            self.capacity = capacity
            self.base = base
            self.in_flight = 0
            self.total_calls = 0

        async def call(self) -> float:
            self.in_flight += 1
            self.total_calls += 1
            try:
                congestion = max(1.0, self.in_flight / self.capacity)
                d = self.base * congestion
                await asyncio.sleep(d)
                return d
            finally:
                self.in_flight -= 1

    async def load_test(hedge: bool) -> tuple[dict, int]:
        dep = SaturatedDependency(capacity=8, base=0.1)
        lat: list[float] = []

        async def one() -> None:
            t0 = time.perf_counter()
            if hedge:
                await hedged_call(dep.call, hedge_after=0.15, call_timeout=30.0)
            else:
                await dep.call()
            lat.append(time.perf_counter() - t0)

        await asyncio.gather(*(one() for _ in range(60)))
        return percentiles(lat), dep.total_calls

    for hedge in (False, True):
        p, calls = await load_test(hedge)
        label = "with hedging" if hedge else "no hedging"
        print(f"      {label:<14} p50={p['p50'] * 1000:>6.0f}ms "
              f"p99={p['p99'] * 1000:>6.0f}ms  calls={calls}")

    print("""
      Hedging under saturation sends substantially more calls and does not
      improve the tail, because the extra calls ARE the congestion. Compare
      this with PART 2, where the dependency had spare capacity and hedging
      worked well. Same technique, opposite result — the difference is
      headroom, which is why 'do we have headroom' belongs in the config
      decision and not just in the postmortem.""")


# ---------------------------------------------------------------------------
# PART 4 — timeouts that destroy work, and cancellation propagation
# ---------------------------------------------------------------------------

async def part4_wasted_work() -> None:
    banner("PART 4 — a timeout without cancellation propagation is a leak")

    work_still_running = []

    async def expensive_operation(name: str) -> str:
        try:
            await asyncio.sleep(1.0)
            work_still_running.append(name)
            return name
        except asyncio.CancelledError:
            raise

    section("client times out, server work is NOT cancelled")
    # This is what happens over plain HTTP without cancellation propagation:
    # your client gives up, but the server keeps generating tokens, keeps
    # holding a connection, and keeps billing you.
    task = asyncio.create_task(expensive_operation("server-side-work"))
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(0.1):
            await asyncio.shield(task)          # shield models "server keeps going"
    print(f"      client gave up after 100ms; server work done? "
          f"{bool(work_still_running)}")
    await asyncio.sleep(1.0)
    print(f"      ...1s later, server work completed anyway: {work_still_running}")
    print("      You paid for those tokens. Nobody read them.")

    section("cancellation propagated properly")
    work_still_running.clear()
    task2 = asyncio.create_task(expensive_operation("cancelled-work"))
    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
        async with asyncio.timeout(0.1):
            await task2                          # no shield: cancel propagates
    await asyncio.sleep(0.2)
    print(f"      server work completed? {bool(work_still_running)}  <- stopped")

    print("""
    PRACTICAL NOTES:
      * With httpx/aiohttp, cancelling the awaiting task DOES close the
        connection, and a well-behaved server notices and aborts. Azure
        OpenAI stops generating when the client disconnects, so client-side
        cancellation is worth real money on long generations.
      * Do NOT wrap the main call in `asyncio.shield` — that is a timeout you
        disabled. Shield only short, must-complete cleanup.
      * For streaming, cancel on client disconnect explicitly. A dropped
        browser tab should stop the generation, and if your framework does not
        propagate that automatically, you are paying for abandoned tokens at
        whatever rate your users close tabs.""")


# ---------------------------------------------------------------------------
# PART 5 — the four timeouts, restated with the numbers that matter
# ---------------------------------------------------------------------------

async def part5_layers() -> None:
    banner("PART 5 — timeout layers for an LLM call")

    print("""
    connect        3-5s     TCP+TLS. A healthy connect is <100ms; anything
                            near the limit means DNS, an NSG, or a dead
                            endpoint. Keep it SHORT so failures are fast.

    write          10s      Sending the request body. Matters when you upload
                            a document for extraction.

    read           15-30s   Gap BETWEEN BYTES. For streaming this is the
                            inter-token gap, so it can be short even for a
                            long generation. This is the timeout that
                            correctly expresses "the stream has stalled".

    pool           5s       Waiting for a free connection. Firing here means
                            your concurrency exceeds your pool — the fix is a
                            semaphore, not a bigger timeout.

    per-attempt    scaled   ttft_budget + max_tokens * per_token_budget.
                            NOT a constant.

    end-to-end     SLA      One deadline for retrieval + rerank + generation +
                            all retries. Everything above nests inside it.

    THE ONE THAT IS ALMOST ALWAYS WRONG: a single total-response timeout on a
    streaming call. A 3-minute generation is legitimate; a 15-second gap
    between tokens is not. Bound the gap, not the total.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_sizing()
    await part2_hedging()
    await part3_hedge_budget()
    await part4_wasted_work()
    await part5_layers()

    banner("SUMMARY")
    print("""
  * Timeouts are percentile decisions. Derive from measured latency; check
    what fraction of capacity the choice wastes.
  * Scale LLM timeouts by max_tokens, not a global constant.
  * Hedging converts tail latency into extra load. Hedge at ~p95, never near
    p50.
  * Hedging requires idempotency AND headroom. Under saturation it makes
    things worse — measured in PART 3.
  * Give hedges a budget; budget exhaustion is your signal that slowness has
    become systemic.
  * Always cancel and reap losing hedges.
  * A timeout without cancellation propagation destroys work and still bills
    you. Never shield the main call.
  * For streams, bound the inter-token gap, not the total duration.
""")


if __name__ == "__main__":
    asyncio.run(main())
