"""
05 — Circuit breakers, bulkheads, and load shedding: containing failure.

THE PROGRESSION
---------------
Retries handle a blip. When the dependency is genuinely down, retries are the
worst thing you can do — you spend your entire deadline, hold connections, and
add load to something already failing.

Three containment patterns, addressing three different questions:

  CIRCUIT BREAKER  "should we call this dependency at all right now?"
                   Stops calling a dead thing. Fails fast, sheds load,
                   probes for recovery.

  BULKHEAD         "can one dependency's failure consume all our capacity?"
                   Isolates resource pools so a slow dependency cannot
                   starve unrelated work.

  LOAD SHEDDING    "we cannot serve everything; what do we drop?"
                   Rejects work at the door, deliberately and cheaply,
                   rather than accepting everything and timing out.

The failure they prevent is the same one: a slow or dead dependency consuming
all your threads/connections/memory, so that a partial outage becomes total.

Run:  python 05_circuit_breakers.py
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from enum import Enum

from failure_lab import (
    AppError,
    FaultInjector,
    Metrics,
    ServiceUnavailable,
    banner,
    section,
)

# ---------------------------------------------------------------------------
# PART 1 — a breaker with a SLIDING WINDOW (not a naive counter)
# ---------------------------------------------------------------------------

class State(Enum):
    CLOSED = "closed"        # normal
    OPEN = "open"            # failing fast, no calls made
    HALF_OPEN = "half_open"  # probing


class ConsecutiveFailureBreaker:
    """The naive version, included so its failure mode is visible.

    Trips after N CONSECUTIVE failures. Problem: a dependency failing 50% of
    the time never produces 5 in a row for long, so the breaker never opens —
    even though half your requests are failing and every one of them is
    burning a full retry budget first.
    """

    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold
        self.consecutive = 0
        self.state = State.CLOSED

    def record(self, ok: bool) -> None:
        if ok:
            self.consecutive = 0
            self.state = State.CLOSED
        else:
            self.consecutive += 1
            if self.consecutive >= self.threshold:
                self.state = State.OPEN


class SlidingWindowBreaker:
    """Production-shaped breaker.

    Improvements over consecutive-failure counting, each addressing a real
    failure mode:

      RATE over a WINDOW  catches partial degradation (50% failing) that
                          consecutive counting misses entirely.
      MINIMUM VOLUME      prevents 1 failure out of 1 call from opening the
                          circuit at 3am when traffic is low. Without this,
                          low-traffic services flap constantly.
      HALF-OPEN PROBES    a limited number of trial calls, so recovery is
                          tested cheaply rather than by dumping full traffic
                          on a service that just came back.
      SLOW CALLS COUNT    a call that takes 30s and succeeds is as damaging as
                          one that fails. Most breakers ignore this, which is
                          why they do not trip during a brownout.
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        failure_rate_threshold: float = 0.5,
        slow_call_threshold: float = 1.0,
        slow_rate_threshold: float = 0.6,
        minimum_calls: int = 10,
        open_duration: float = 0.5,
        half_open_probes: int = 3,
    ) -> None:
        self.window: deque[tuple[bool, float]] = deque(maxlen=window_size)
        self.failure_rate_threshold = failure_rate_threshold
        self.slow_call_threshold = slow_call_threshold
        self.slow_rate_threshold = slow_rate_threshold
        self.minimum_calls = minimum_calls
        self.open_duration = open_duration
        self.half_open_probes = half_open_probes

        self.state = State.CLOSED
        self._opened_at = 0.0
        self._probes_sent = 0
        self._probes_ok = 0
        self.transitions: list[str] = []
        self.rejected = 0

    # -- state machine -----------------------------------------------------

    def _now(self) -> float:
        return time.monotonic()

    def _trip(self, reason: str) -> None:
        self.state = State.OPEN
        self._opened_at = self._now()
        self.window.clear()
        self.transitions.append(f"-> OPEN ({reason})")

    def _close(self) -> None:
        self.state = State.CLOSED
        self.window.clear()
        self.transitions.append("-> CLOSED (recovered)")

    def _half_open(self) -> None:
        self.state = State.HALF_OPEN
        self._probes_sent = 0
        self._probes_ok = 0
        self.transitions.append("-> HALF_OPEN (probing)")

    def _evaluate(self) -> None:
        if len(self.window) < self.minimum_calls:
            return                       # not enough evidence
        n = len(self.window)
        failures = sum(1 for ok, _ in self.window if not ok)
        slow = sum(1 for ok, d in self.window if ok and d >= self.slow_call_threshold)

        if failures / n >= self.failure_rate_threshold:
            self._trip(f"failure rate {failures / n:.0%}")
        elif slow / n >= self.slow_rate_threshold:
            # THE BROWNOUT CASE. Everything "succeeds", slowly, and your
            # threads are all occupied. A failure-rate breaker never fires.
            self._trip(f"slow-call rate {slow / n:.0%}")

    # -- the guarded call --------------------------------------------------

    async def call(self, fn):
        if self.state is State.OPEN:
            if self._now() - self._opened_at >= self.open_duration:
                self._half_open()
            else:
                self.rejected += 1
                raise ServiceUnavailable("circuit OPEN — no call attempted")

        if self.state is State.HALF_OPEN and self._probes_sent >= self.half_open_probes:
            # Cap concurrent probes. Without this, everything waiting on the
            # open circuit floods through the instant it half-opens, which is
            # how a recovering service gets knocked straight back down.
            self.rejected += 1
            raise ServiceUnavailable("circuit HALF_OPEN — probe limit reached")

        started = self._now()
        if self.state is State.HALF_OPEN:
            self._probes_sent += 1

        try:
            result = await fn()
        except Exception:
            if self.state is State.HALF_OPEN:
                self._trip("probe failed")     # straight back to OPEN
            else:
                self.window.append((False, self._now() - started))
                self._evaluate()
            raise
        else:
            duration = self._now() - started
            if self.state is State.HALF_OPEN:
                self._probes_ok += 1
                if self._probes_ok >= self.half_open_probes:
                    self._close()
            else:
                self.window.append((True, duration))
                self._evaluate()
            return result


async def part1_breakers() -> None:
    banner("PART 1 — why consecutive-failure counting is not enough")

    section("dependency failing 50% of the time")
    naive = ConsecutiveFailureBreaker(threshold=5)
    smart = SlidingWindowBreaker(minimum_calls=10, failure_rate_threshold=0.5)
    inj = FaultInjector(rate=0.5, seed=11, base_latency=0.0)

    for _ in range(60):
        ok = True
        try:
            await inj.maybe_fail()
        except AppError:
            ok = False
        naive.record(ok)
        try:
            await smart.call(_make(ok))
        except Exception:
            pass

    print(f"      consecutive-failure breaker: {naive.state.value}")
    print(f"      sliding-window breaker:      {smart.state.value}")
    print(f"      window breaker transitions:  {smart.transitions[:3]}")
    print("""
      At a 50% failure rate the consecutive counter rarely sees 5 in a row, so
      it stays closed while half of all requests fail — each one first burning
      a full retry budget. The rate-over-window breaker trips correctly.""")

    section("the BROWNOUT: everything succeeds, slowly")
    brown = SlidingWindowBreaker(minimum_calls=10, slow_call_threshold=0.05,
                                 slow_rate_threshold=0.6)

    async def slow_but_successful():
        await asyncio.sleep(0.06)
        return "ok"

    for _ in range(15):
        try:
            await brown.call(slow_but_successful)
        except Exception:
            break
    print(f"      state after 15 slow successes: {brown.state.value}")
    print(f"      transitions: {brown.transitions}")
    print("""
      Zero errors. A failure-rate-only breaker never fires, your threads all
      sit occupied, and the service dies of latency rather than errors.
      Counting slow calls is what catches this.""")


def _make(ok: bool):
    async def f():
        if not ok:
            raise ServiceUnavailable("503")
        return "ok"
    return f


# ---------------------------------------------------------------------------
# PART 2 — the full state machine, observed
# ---------------------------------------------------------------------------

async def part2_lifecycle() -> None:
    banner("PART 2 — breaker lifecycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED")

    cb = SlidingWindowBreaker(minimum_calls=5, failure_rate_threshold=0.5,
                              open_duration=0.2, half_open_probes=3)
    healthy = False

    async def dependency():
        if not healthy:
            raise ServiceUnavailable("503")
        return "ok"

    print(f"    {'phase':<26} {'state':<11} {'outcome'}")
    print(f"    {'-' * 26} {'-' * 11} {'-' * 30}")

    async def attempt(label: str) -> None:
        try:
            await cb.call(dependency)
            print(f"    {label:<26} {cb.state.value:<11} ok")
        except AppError as e:
            print(f"    {label:<26} {cb.state.value:<11} {e}")

    for i in range(6):
        await attempt(f"dependency down #{i}")

    for i in range(2):
        await attempt(f"circuit open #{i}")

    print(f"\n    ...waiting {cb.open_duration}s for the recovery window...\n")
    await asyncio.sleep(cb.open_duration + 0.01)

    healthy = True
    for i in range(4):
        await attempt(f"probe #{i}")

    print(f"\n    transitions: {' '.join(cb.transitions)}")
    print(f"    calls rejected without being attempted: {cb.rejected}")
    print("""
    The rejected count is the value delivered: those calls cost ~0 instead of
    a full timeout each. That is the capacity you keep for the parts of the
    system that still work.""")


# ---------------------------------------------------------------------------
# PART 3 — bulkheads
# ---------------------------------------------------------------------------

class Bulkhead:
    """A dedicated concurrency pool per dependency.

    THE FAILURE IT PREVENTS: one slow dependency consumes every worker slot,
    so requests that never touch it also fail. This is the single most common
    way a partial outage becomes a total one.

    Concretely: your service calls Azure OpenAI, Azure AI Search, and a SQL
    database. OpenAI gets slow. With one shared pool of 50, all 50 slots fill
    with OpenAI calls and your health endpoint — which touches nothing —
    starts timing out. With per-dependency bulkheads, OpenAI's 20 slots fill
    and the other 30 keep working.
    """

    def __init__(self, name: str, limit: int, queue_limit: int = 0) -> None:
        self.name = name
        self.limit = limit
        self.queue_limit = queue_limit
        self._sem = asyncio.Semaphore(limit)
        self.in_flight = 0
        self.queued = 0
        self.rejected = 0
        self.peak = 0

    async def run(self, fn):
        if self.queue_limit and self.queued >= self.queue_limit:
            # Reject rather than queue unboundedly. An unbounded queue turns a
            # capacity problem into a latency problem and then into an OOM.
            self.rejected += 1
            raise ServiceUnavailable(f"bulkhead {self.name} full — rejected")

        self.queued += 1
        try:
            async with self._sem:
                self.queued -= 1
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                try:
                    return await fn()
                finally:
                    self.in_flight -= 1
        except BaseException:
            if self.queued > 0 and self.in_flight == 0:
                self.queued -= 1
            raise


async def part3_bulkheads() -> None:
    banner("PART 3 — bulkheads: one slow dependency must not starve the rest")

    async def slow_llm():
        await asyncio.sleep(0.5)
        return "llm"

    async def fast_db():
        await asyncio.sleep(0.01)
        return "db"

    section("SHARED pool of 10 — LLM slowness starves the database")
    shared = Bulkhead("shared", limit=10)
    t0 = time.perf_counter()
    db_latencies: list[float] = []

    async def db_request():
        s = time.perf_counter()
        await shared.run(fast_db)
        db_latencies.append(time.perf_counter() - s)

    async def llm_request():
        await shared.run(slow_llm)

    await asyncio.gather(
        *(llm_request() for _ in range(20)),
        *(db_request() for _ in range(10)),
    )
    print(f"      db request latency: p50={sorted(db_latencies)[5] * 1000:.0f}ms "
          f"max={max(db_latencies) * 1000:.0f}ms")
    print(f"      total elapsed: {time.perf_counter() - t0:.2f}s")
    print("      The DB calls take 10ms of work but waited behind LLM calls.")

    section("SEPARATE bulkheads — the database is unaffected")
    llm_bh = Bulkhead("llm", limit=8)
    db_bh = Bulkhead("db", limit=8)
    t0 = time.perf_counter()
    db_latencies2: list[float] = []

    async def db_request2():
        s = time.perf_counter()
        await db_bh.run(fast_db)
        db_latencies2.append(time.perf_counter() - s)

    async def llm_request2():
        await llm_bh.run(slow_llm)

    await asyncio.gather(
        *(llm_request2() for _ in range(20)),
        *(db_request2() for _ in range(10)),
    )
    print(f"      db request latency: p50={sorted(db_latencies2)[5] * 1000:.0f}ms "
          f"max={max(db_latencies2) * 1000:.0f}ms")
    print(f"      total elapsed: {time.perf_counter() - t0:.2f}s")
    print("""
      DB latency collapsed from ~1000ms to ~10ms — it is now served at its own
      speed instead of inheriting the LLM's.

      BUT NOTE THE TOTAL ELAPSED WENT UP (1.01s -> 1.50s). That is not noise
      and it is not a flaw in the demo: the shared pool gave LLM calls all 10
      slots, while the LLM bulkhead caps them at 8. Isolation costs some
      aggregate throughput.

      That IS the trade. You are buying predictable latency for the healthy
      dependency at the price of some peak throughput for the unhealthy one.
      Almost always worth it — a slow LLM should not take your health checks
      and your database reads down with it — but state it rather than
      pretending isolation is free.

      SIZING: sum of all bulkhead limits may EXCEED total capacity (that is
      the point of pooling), but no single bulkhead should be large enough to
      consume everything. A common shape: each bulkhead <= 40% of total
      workers, so any two dependencies failing still leaves room.""")


# ---------------------------------------------------------------------------
# PART 4 — load shedding and adaptive concurrency
# ---------------------------------------------------------------------------

class AdaptiveLimiter:
    """AIMD concurrency limiting — the algorithm behind TCP congestion control
    and Netflix's concurrency-limits library.

    Additive Increase, Multiplicative Decrease:
      success  -> limit += 1/limit   (creep up)
      failure/
      timeout  -> limit *= 0.9       (back off hard)

    WHY THIS BEATS A STATIC SEMAPHORE: the right concurrency depends on the
    dependency's current capacity, which changes — deployments, scale events,
    noisy neighbours, quota changes. A static limit is right on the day you
    tuned it. AIMD converges continuously and needs no tuning.
    """

    def __init__(self, initial: float = 10.0, min_limit: float = 1.0,
                 max_limit: float = 200.0) -> None:
        self.limit = initial
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.in_flight = 0
        self.shed = 0
        self.history: list[float] = []

    def try_acquire(self) -> bool:
        if self.in_flight >= self.limit:
            self.shed += 1
            return False
        self.in_flight += 1
        return True

    def release(self, *, ok: bool) -> None:
        self.in_flight -= 1
        if ok:
            self.limit = min(self.max_limit, self.limit + 1.0 / self.limit)
        else:
            self.limit = max(self.min_limit, self.limit * 0.9)
        self.history.append(self.limit)


async def part4_adaptive() -> None:
    banner("PART 4 — adaptive concurrency: let the dependency tell you")

    class Backend:
        """Serves `capacity` concurrently; beyond that it fails."""

        def __init__(self, capacity: int) -> None:
            self.capacity = capacity
            self.in_flight = 0

        async def call(self) -> str:
            self.in_flight += 1
            try:
                if self.in_flight > self.capacity:
                    await asyncio.sleep(0.01)
                    raise ServiceUnavailable("overloaded")
                await asyncio.sleep(0.02)
                return "ok"
            finally:
                self.in_flight -= 1

    backend = Backend(capacity=12)
    lim = AdaptiveLimiter(initial=40.0)
    m = Metrics()

    async def request():
        if not lim.try_acquire():
            m.incr("shed")
            return
        ok = False
        try:
            await backend.call()
            ok = True
            m.incr("ok")
        except AppError:
            m.incr("failed")
        finally:
            lim.release(ok=ok)

    for _ in range(30):
        await asyncio.gather(*(request() for _ in range(25)))

    print(f"      backend true capacity: {backend.capacity}")
    print(f"      limiter converged to:  {lim.limit:.1f}")
    print(f"      limit trajectory: {[round(h, 1) for h in lim.history[::80]]}")
    print(f"      {m.report('ok', 'failed', 'shed')}")

    print("""
      The limiter discovered the backend's capacity without being told it.
      When the backend scales up or down, it re-converges automatically.

      SHEDDING IS THE POINT: rejecting a request in microseconds is vastly
      better than accepting it and timing out 30 seconds later. The rejected
      caller can fail over, retry elsewhere, or show a useful message. A
      caller stuck in a 30s timeout can do none of those.

      SHED BY PRIORITY, not arbitrarily: health checks and interactive user
      requests should survive; batch backfill and speculative prefetch should
      be dropped first. A single global limiter cannot express that — you need
      per-class limits or a priority queue.""")


# ---------------------------------------------------------------------------
# PART 5 — composing them
# ---------------------------------------------------------------------------

async def part5_composition() -> None:
    banner("PART 5 — ordering the layers")

    print("""
    Outermost to innermost, and the order matters:

      1. LOAD SHED / admission control
         Cheapest possible rejection. Nothing below this line should ever see
         a request the system has no capacity to serve.

      2. BULKHEAD (per dependency)
         Bounds how much of your capacity this dependency can consume.

      3. CIRCUIT BREAKER (per dependency, per region)
         Skips the call entirely when the dependency is known bad.

      4. RETRY BUDGET
         Global cap on retry volume.

      5. TIMEOUT / DEADLINE
         Per attempt, nested inside the end-to-end deadline.

      6. RETRY with backoff + jitter
         Innermost. Retries happen INSIDE the bulkhead slot so a retrying
         request does not secretly double your real concurrency.

    THE COMMON MISORDERING: putting the breaker inside the retry loop. Then
    each retry checks the breaker, gets rejected instantly, and you burn all
    your attempts in microseconds and report 'exhausted retries' — which is
    technically true and completely misleading. The breaker belongs OUTSIDE
    the retry loop, so an open circuit means 'we did not try', not 'we tried
    four times very quickly'.

    SCOPE THE BREAKER CORRECTLY: one breaker per (dependency, region,
    deployment). A single global breaker for 'Azure OpenAI' will trip on one
    bad region and cut you off from three healthy ones — and a breaker per
    (dependency, endpoint, user) will never accumulate enough calls to trip
    at all.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_breakers()
    await part2_lifecycle()
    await part3_bulkheads()
    await part4_adaptive()
    await part5_composition()

    banner("SUMMARY")
    print("""
  * Consecutive-failure breakers miss partial degradation. Use failure RATE
    over a sliding window, with a minimum call volume.
  * Count SLOW calls as failures, or brownouts never trip the breaker.
  * Half-open must limit concurrent probes, or recovery re-kills the service.
  * Bulkheads stop one slow dependency from consuming all capacity — the main
    mechanism by which partial outages become total.
  * Bound bulkhead queues; reject rather than queue unboundedly.
  * Adaptive (AIMD) limits discover capacity instead of needing tuning.
  * Shedding fast beats timing out slow. Shed by priority.
  * Order: shed -> bulkhead -> breaker -> budget -> timeout -> retry. The
    breaker goes OUTSIDE the retry loop.
  * Scope breakers per (dependency, region, deployment).
""")


if __name__ == "__main__":
    asyncio.run(main())
