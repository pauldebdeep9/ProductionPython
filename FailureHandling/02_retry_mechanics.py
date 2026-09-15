"""
02 — Retry mechanics: budgets, amplification, and deadline propagation.

WHAT THIS ADDS BEYOND "USE EXPONENTIAL BACKOFF WITH JITTER"
-----------------------------------------------------------
Backoff with jitter is table stakes and takes ten lines. The things that
actually determine whether retries help or hurt:

  1. RETRY AMPLIFICATION. Retries at N layers multiply. 3 retries at each of
     3 layers is 27x load on the bottom service — during an outage, which is
     exactly when it can least absorb it. This is how a partial degradation
     becomes a total one.

  2. RETRY BUDGETS. The fix for amplification is not "retry less". It is a
     BUDGET: retries may consume at most X% of your request volume. Below the
     threshold you retry freely; above it you stop. This bounds the blast
     radius without penalising the healthy case.

  3. DEADLINE PROPAGATION. Every layer must inherit the caller's remaining
     time, not start its own 30s clock. Without this, nested timeouts sum and
     your "30 second" SLA is three minutes.

  4. ATTEMPT ACCOUNTING. If you cannot report attempts-per-request, you do not
     know what load you are actually generating.

Run:  python 02_retry_mechanics.py
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time

from failure_lab import (
    AppError,
    Attempt,
    CallResult,
    ContentFiltered,
    Disposition,
    Fail,
    FaultInjector,
    Metrics,
    Ok,
    RateLimitError,
    ServiceUnavailable,
    banner,
    classify,
    section,
)

# ---------------------------------------------------------------------------
# PART 1 — backoff families, measured
# ---------------------------------------------------------------------------

def b_fixed(attempt: int, base: float, prev: float) -> float:
    return base


def b_exponential(attempt: int, base: float, prev: float) -> float:
    return base * (2 ** attempt)


def b_equal_jitter(attempt: int, base: float, prev: float) -> float:
    """half deterministic + half random. Bounds the minimum wait, which
    matters when you need to guarantee some breathing room."""
    d = base * (2 ** attempt)
    return d / 2 + random.uniform(0, d / 2)


def b_full_jitter(attempt: int, base: float, prev: float) -> float:
    """U(0, base*2^n). Best general default — maximum decorrelation."""
    return random.uniform(0, base * (2 ** attempt))


def b_decorrelated(attempt: int, base: float, prev: float) -> float:
    """min(cap, U(base, prev*3)). Backs off harder under sustained failure
    while still spreading clients. Good when the dependency is genuinely
    down rather than briefly busy."""
    return min(30.0, random.uniform(base, max(prev, base) * 3))


async def part1_backoff_families() -> None:
    banner("PART 1 — backoff families: spread vs total wait")

    families = [
        ("fixed", b_fixed),
        ("exponential", b_exponential),
        ("equal jitter", b_equal_jitter),
        ("full jitter", b_full_jitter),
        ("decorrelated", b_decorrelated),
    ]

    n_clients, base, attempts = 500, 0.1, 5
    random.seed(7)

    # Bucket width matters: too coarse and every family looks synchronised.
    # 0.1s is roughly a service's reaction time — finer than that is noise.
    BUCKET = 0.1
    print(f"    {'family':<14} {'total wait: p50':>16} {'p95':>8} "
          f"{'worst 0.1s bucket':>18} {'buckets':>8}")
    print(f"    {'-' * 14} {'-' * 16} {'-' * 8} {'-' * 18} {'-' * 8}")

    for name, fn in families:
        totals, third_attempt = [], []
        for _ in range(n_clients):
            total, prev = 0.0, base
            for a in range(attempts):
                d = fn(a, base, prev)
                prev = d
                total += d
                if a == 2:
                    third_attempt.append(total)
            totals.append(total)

        buckets: dict[int, int] = {}
        for t in third_attempt:
            k = int(t / BUCKET)
            buckets[k] = buckets.get(k, 0) + 1

        print(f"    {name:<14} {statistics.median(totals):>15.2f}s "
              f"{sorted(totals)[int(0.95 * n_clients)]:>7.2f}s "
              f"{max(buckets.values()):>12}/{n_clients} "
              f"{len(buckets):>8}")

    print("""
    "worst bucket" = the most clients arriving in any single 0.1s window at
    attempt 3, and "buckets" = how many distinct windows they spread over.
    1 bucket holding 500/500 is a perfectly synchronised herd.

    fixed and exponential are fully deterministic: every client waits an
    identical amount, so all 500 hit the recovering service in the same
    instant. Equal jitter spreads over a handful of windows; full jitter and
    decorrelated spread over many.

    Choosing between the three jittered families:
      full jitter    default. maximum spread, lowest total wait.
      equal jitter   when you need a guaranteed MINIMUM gap (e.g. the server
                     explicitly needs 500ms to recover a connection pool).
      decorrelated   when the dependency is down for minutes, not seconds —
                     it escalates the wait faster while still spreading.""")


# ---------------------------------------------------------------------------
# PART 2 — retry amplification: the multiplication nobody models
# ---------------------------------------------------------------------------

async def part2_amplification() -> None:
    banner("PART 2 — retry amplification across layers")

    # Model a 3-tier call path: gateway -> orchestrator -> model endpoint.
    # Each layer independently retries 3 times. All of them look reasonable
    # in isolation. Together they are a load multiplier.

    bottom_calls = 0

    async def model_endpoint() -> str:
        nonlocal bottom_calls
        bottom_calls += 1
        raise ServiceUnavailable("503")

    async def retry_n(fn, n: int):
        last = None
        for _ in range(n):
            try:
                return await fn()
            except AppError as e:
                last = e
        raise last

    section("each layer retries 3x, independently")
    bottom_calls = 0

    async def orchestrator():
        return await retry_n(model_endpoint, 3)

    async def gateway():
        return await retry_n(orchestrator, 3)

    try:
        await retry_n(gateway, 3)
    except AppError:
        pass

    print(f"      ONE user request produced {bottom_calls} calls "
          f"to the model endpoint")
    print(f"      amplification factor: {bottom_calls}x")
    print("""
      Each layer's author wrote something defensible. The system multiplies
      them. With 1,000 users this is 27,000 requests against a dependency
      that is already failing — the retries ARE the outage now.""")

    section("retry at ONE layer only (the one closest to the failure)")
    bottom_calls = 0

    async def orchestrator2():
        return await retry_n(model_endpoint, 3)

    async def gateway2():
        return await orchestrator2()          # no retry here

    try:
        await gateway2()                       # no retry here either
    except AppError:
        pass
    print(f"      calls to model endpoint: {bottom_calls}")

    print("""
      THE RULE: retry at exactly one layer, and make it the layer with the
      most information about the failure — usually the one adjacent to the
      dependency. Every other layer propagates.

      When you genuinely need retries at two layers, the outer one must have a
      strictly smaller budget and must not retry errors the inner layer
      already exhausted retries on. In practice this is hard to keep correct,
      which is why "retry at one layer" is the better default.""")


# ---------------------------------------------------------------------------
# PART 3 — retry budgets
# ---------------------------------------------------------------------------

class RetryBudget:
    """Bounds retries as a FRACTION of request volume, not per-request.

    The insight (from Google's SRE practice and gRPC's retry design): a
    per-request retry cap does nothing to protect a dependency, because every
    request gets its own cap. What protects the dependency is a global ratio.

    Token-bucket form:
      * every REQUEST deposits `ratio` tokens (e.g. 0.2 = 20%)
      * every RETRY withdraws 1 token
      * no tokens => no retry, fail immediately

    When failures are rare, the bucket is full and retries always succeed in
    getting a token. When everything is failing, the bucket drains and retries
    stop — automatically, without a config change, at exactly the moment
    retrying became counterproductive.
    """

    def __init__(self, ratio: float = 0.2, min_per_sec: float = 10.0,
                 capacity: float = 100.0) -> None:
        self.ratio = ratio
        self.min_per_sec = min_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.denied = 0
        self.granted = 0

    def on_request(self) -> None:
        self.tokens = min(self.capacity, self.tokens + self.ratio)

    def try_retry(self) -> bool:
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.granted += 1
            return True
        self.denied += 1
        return False


async def part3_budget() -> None:
    banner("PART 3 — retry budgets bound the blast radius")

    async def run(failure_rate: float, use_budget: bool, n: int = 500):
        injector = FaultInjector(rate=failure_rate, seed=3,
                                 error_cls=ServiceUnavailable, base_latency=0.0)
        budget = RetryBudget(ratio=0.2)
        m = Metrics()

        for _ in range(n):
            m.incr("requests")
            budget.on_request()
            for attempt in range(4):
                m.incr("attempts")
                try:
                    await injector.maybe_fail()
                    m.incr("ok")
                    break
                except AppError:
                    if attempt == 3:
                        m.incr("failed")
                        break
                    if use_budget and not budget.try_retry():
                        m.incr("failed")
                        m.incr("budget_denied")
                        break
        return m, budget

    print(f"    {'scenario':<34} {'attempts':>9} {'amplif.':>8} {'ok':>6}")
    print(f"    {'-' * 34} {'-' * 9} {'-' * 8} {'-' * 6}")

    for rate, label in [(0.05, "5% failure  (healthy)"),
                        (0.90, "90% failure (outage)")]:
        for use_budget in (False, True):
            m, _budget = await run(rate, use_budget)
            tag = "with budget" if use_budget else "no budget"
            print(f"    {label + ', ' + tag:<34} "
                  f"{m.counters['attempts']:>9} "
                  f"{m.amplification():>7.2f}x "
                  f"{m.counters['ok']:>6}")

    print("""
    HEALTHY ROWS: the budget is invisible. Identical attempts, identical
    successes. It costs nothing when you do not need it — which is the
    property that makes it safe to leave switched on permanently.

    OUTAGE ROWS: read this honestly, because there IS a real trade.
    Load against the dying dependency roughly halves (1728 -> 699 attempts,
    3.46x -> 1.40x amplification). But successes also drop (165 -> 56),
    because some of those successes came from the 3rd and 4th retry that the
    budget refused to fund.

    So the budget trades individual-request recovery for system-wide load
    reduction. That is the decision, and it is the right one at scale: those
    165 successes were bought by hammering a service that is already failing,
    which extends the outage for everyone including the 335 requests that
    failed anyway.

    Tune `ratio` to sit where you want on that trade. 0.1-0.2 is a common
    starting point. What you must NOT do is pick it without knowing the trade
    exists — which is why this table prints both columns.""")


# ---------------------------------------------------------------------------
# PART 4 — deadline propagation
# ---------------------------------------------------------------------------

class Deadline:
    """An absolute deadline passed DOWN the call stack.

    The alternative — each layer starting its own relative timeout — is how a
    30s SLA becomes 3 minutes. Deadlines compose; timeouts do not.

    In gRPC this is built in. Over HTTP you propagate it yourself, typically
    as a header (`grpc-timeout`, or an `X-Request-Deadline` you define), and
    every layer computes `remaining()` rather than reading a config value.
    """

    def __init__(self, seconds: float) -> None:
        self._loop = asyncio.get_event_loop()
        self.at = self._loop.time() + seconds

    def remaining(self) -> float:
        return max(0.0, self.at - self._loop.time())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def child(self, max_seconds: float) -> Deadline:
        """A sub-deadline that can only ever be SHORTER than its parent.

        This is the operation that makes the pattern safe: a child cannot
        extend past the parent, so no layer can blow the end-to-end budget by
        configuring itself generously.
        """
        d = Deadline.__new__(Deadline)
        d._loop = self._loop
        d.at = min(self.at, self._loop.time() + max_seconds)
        return d


async def part4_deadlines() -> None:
    banner("PART 4 — deadline propagation vs independent timeouts")

    async def slow_step(name: str, duration: float) -> str:
        await asyncio.sleep(duration)
        return name

    section("WRONG: each layer has its own 0.30s timeout")
    t0 = time.perf_counter()
    completed = []
    for step in ("retrieve", "rerank", "generate"):
        try:
            async with asyncio.timeout(0.30):
                completed.append(await slow_step(step, 0.25))
        except TimeoutError:
            break
    print(f"      completed {completed} in {time.perf_counter() - t0:.2f}s")
    print("      each step was 'within its timeout'; the total blew past 0.30s")

    section("RIGHT: one deadline, propagated")
    t0 = time.perf_counter()
    dl = Deadline(0.30)
    completed = []
    try:
        for step in ("retrieve", "rerank", "generate"):
            if dl.expired():
                raise TimeoutError("deadline exhausted")
            async with asyncio.timeout(dl.remaining()):
                completed.append(await slow_step(step, 0.25))
    except TimeoutError:
        pass
    print(f"      completed {completed} in {time.perf_counter() - t0:.2f}s")
    print("      total bounded by the ONE budget, regardless of step count")

    section("child deadlines cannot extend the parent")
    dl = Deadline(0.20)
    child = dl.child(10.0)          # asks for 10s...
    print(f"      parent remaining: {dl.remaining():.2f}s")
    print(f"      child  remaining: {child.remaining():.2f}s  <- clamped")

    print("""
    THE INTERACTION WITH RETRIES: before spending an attempt, check the
    deadline. Starting a call you know cannot finish burns quota, delays the
    failure the caller is waiting for, and produces a timeout instead of the
    real error. Check first, and clamp the backoff sleep to the remaining
    time so you never sleep past your own deadline.""")


# ---------------------------------------------------------------------------
# PART 5 — a retry executor with all of the above
# ---------------------------------------------------------------------------

async def resilient_call(
    fn,
    *,
    deadline: Deadline,
    budget: RetryBudget | None = None,
    max_attempts: int = 5,
    base_delay: float = 0.05,
    max_delay: float = 2.0,
    metrics: Metrics | None = None,
    op: str = "op",
) -> CallResult:
    """Retry with classification, budget, deadline, jitter, and Retry-After.

    Returns a CallResult carrying the full attempt history rather than just a
    value. That history is what makes a single log line tell the whole story,
    and what makes retry behaviour assertable in tests.

    In production use `tenacity` or `stamina` rather than hand-rolling — but
    you should be able to read their config and know exactly what it does,
    which is what this function is for.
    """
    result = CallResult()
    if metrics:
        metrics.incr("requests")

    for n in range(max_attempts):
        if deadline.expired():
            result.attempts.append(Attempt(n, "deadline_exhausted"))
            result.final_error = TimeoutError(f"{op}: deadline exhausted")
            break

        started = time.perf_counter()
        if metrics:
            metrics.incr("attempts")
        try:
            # The call itself is bounded by whatever is left of the deadline.
            async with asyncio.timeout(deadline.remaining()):
                value = await fn()
        except asyncio.CancelledError:
            raise                                    # never retried, never counted
        except TimeoutError as e:
            result.attempts.append(
                Attempt(n, "timeout", "deadline", elapsed=time.perf_counter() - started)
            )
            result.final_error = e
            break
        except AppError as e:
            elapsed = time.perf_counter() - started
            disp = classify(e)

            if disp is not Disposition.RETRY:
                result.attempts.append(
                    Attempt(n, disp.value, type(e).__name__, elapsed=elapsed)
                )
                result.final_error = e
                if metrics:
                    metrics.incr(f"outcome.{disp.value}")
                break

            if n == max_attempts - 1:
                result.attempts.append(
                    Attempt(n, "exhausted", type(e).__name__, elapsed=elapsed)
                )
                result.final_error = e
                if metrics:
                    metrics.incr("outcome.exhausted")
                break

            if budget is not None and not budget.try_retry():
                result.attempts.append(
                    Attempt(n, "budget_denied", type(e).__name__, elapsed=elapsed)
                )
                result.final_error = e
                if metrics:
                    metrics.incr("outcome.budget_denied")
                break

            # Server guidance beats our curve when present.
            if e.retry_after is not None:
                delay = float(e.retry_after)
            else:
                delay = min(max_delay, random.uniform(0, base_delay * (2 ** n)))
            # Never sleep past the deadline.
            delay = min(delay, deadline.remaining())
            result.attempts.append(
                Attempt(n, "retry", type(e).__name__, delay_before=delay,
                        elapsed=elapsed)
            )
            await asyncio.sleep(delay)
            continue

        result.attempts.append(
            Attempt(n, "ok", elapsed=time.perf_counter() - started)
        )
        result.value = value
        result.succeeded = True
        if metrics:
            metrics.incr("outcome.ok")
        break

    return result


async def part5_executor() -> None:
    banner("PART 5 — the executor, exercised")

    m = Metrics()
    if True:
        budget = RetryBudget(ratio=0.5)

    scenarios = {
        "transient then success": [
            Fail(ServiceUnavailable, 0.01), Fail(ServiceUnavailable, 0.01), Ok(0.01)
        ],
        "429 with Retry-After": [
            Fail(RateLimitError, 0.01, retry_after=0.05), Ok(0.01)
        ],
        "permanent (fails fast)": [Fail(ContentFiltered, 0.01)],
        "never recovers": [Fail(ServiceUnavailable, 0.01)],
    }

    for label, script in scenarios.items():
        inj = FaultInjector(script=script)
        budget.on_request()
        dl = Deadline(1.0)
        r = await resilient_call(inj.maybe_fail, deadline=dl, budget=budget,
                                 metrics=m, op=label)
        status = "OK " if r.succeeded else "ERR"
        print(f"    {status} {label:<26} {r.summary()}")

    print(f"\n    metrics: {m.report('requests', 'attempts')}"
          f"  amp={m.amplification():.2f}x")

    print("""
    The attempt history is the deliverable. `2 attempt(s): retry(RateLimitError)
    -> ok` in one log line beats five interleaved lines you have to reassemble
    at 3am, and it is directly assertable in a test.""")


# ---------------------------------------------------------------------------
# PART 6 — what to alert on
# ---------------------------------------------------------------------------

async def part6_signals() -> None:
    banner("PART 6 — the retry signals worth alerting on")

    print("""
    Emit these four. Most teams have the first and none of the others.

    1. error_rate              — obvious, and the least useful alone. A
                                 healthy-looking error rate is compatible with
                                 every request taking 4 attempts.

    2. retry_amplification     — attempts / requests. Rising amplification is
                                 the EARLIEST signal of a degrading dependency,
                                 usually minutes before error rate moves,
                                 because retries are still succeeding.
                                 Alert: sustained > 1.5.

    3. budget_denied_rate      — retries you refused. Non-zero means you are
                                 actively shedding, i.e. the dependency is bad
                                 enough that you stopped trying. This is a
                                 page, not a warning.

    4. attempts_histogram      — the DISTRIBUTION, not the mean. "p50=1,
                                 p99=4" is healthy. "p50=3" means everything
                                 is retrying and you are one step from
                                 falling over.

    AND THE TRAP: latency percentiles computed over ALL requests hide this
    completely. Failures are usually FASTER than successes (a 503 comes back
    in 5ms), so during an outage your p50 latency IMPROVES. Always break
    latency down by outcome, or your dashboard will look best at the worst
    possible moment.""")

    # Demonstrate the latency-hiding effect concretely.
    m = Metrics()
    for _ in range(900):
        m.observe("all", random.uniform(0.4, 0.8))       # healthy successes
    print(f"\n    healthy:  p50 over all requests = {m.pct('all', 0.5) * 1000:.0f}ms")

    m2 = Metrics()
    for _ in range(100):
        m2.observe("all", random.uniform(0.4, 0.8))      # few slow successes
    for _ in range(900):
        m2.observe("all", random.uniform(0.003, 0.008))  # many fast failures
    print(f"    OUTAGE:   p50 over all requests = {m2.pct('all', 0.5) * 1000:.0f}ms"
          f"   <- looks BETTER while 90% of requests are failing")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_backoff_families()
    await part2_amplification()
    await part3_budget()
    await part4_deadlines()
    await part5_executor()
    await part6_signals()

    banner("SUMMARY")
    print("""
  * Full jitter is the default; equal jitter when you need a minimum gap;
    decorrelated when the dependency is down for minutes.
  * Retry at ONE layer. Nested retries multiply — 3 layers x 3 retries = 27x.
  * Retry budgets bound retries as a fraction of request volume. They cost
    nothing when healthy and shed automatically when not.
  * Propagate an absolute DEADLINE, not per-layer relative timeouts. Child
    deadlines clamp to the parent and can never extend it.
  * Check the deadline BEFORE spending an attempt; clamp backoff to what
    remains.
  * Return the attempt history, don't just log it.
  * Alert on amplification and the attempts distribution, not just error rate.
    And break latency down by outcome, or an outage will look like an
    improvement.
""")


if __name__ == "__main__":
    asyncio.run(main())
