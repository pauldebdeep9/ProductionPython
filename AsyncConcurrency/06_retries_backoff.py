"""
06 — Retries: classification, backoff, jitter, deadlines, and circuit breaking.

THE THREE MISTAKES
------------------
  1. Retrying non-retryable errors. A 400 content-filter rejection or a 401
     will fail identically forever. Retrying it burns your deadline and, for
     auth errors, can trip account lockout.
  2. Retrying without jitter. N clients that fail together retry together, in
     lockstep, forever. The retry storm is worse than the original outage —
     this is the "thundering herd", and it is how a brief 503 becomes a 20
     minute one.
  3. Retrying without a global deadline. 5 retries x 60s timeout = a 5 minute
     user-visible hang from something you configured as "30 seconds".

Retries also interact with idempotency. If a request may have been PROCESSED
before the response was lost, retrying duplicates the side effect. For pure
generation that is merely wasteful. For anything that writes — creating an
invoice dispute, posting an approval — it is a correctness bug. Send an
idempotency key.

Run:  python 06_retries_backoff.py
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from fake_llm import (
    AuthError,
    ContentFilterError,
    FakeLLMClient,
    LLMError,
    RateLimitError,
    Timer,
    TransientServerError,
    banner,
)

# ---------------------------------------------------------------------------
# PART 1 — classification comes first
# ---------------------------------------------------------------------------

# The taxonomy is the design. Everything else is mechanism.
RETRYABLE = (RateLimitError, TransientServerError, TimeoutError, ConnectionError)
NON_RETRYABLE = (ContentFilterError, AuthError, ValueError)


def is_retryable(exc: BaseException) -> bool:
    """Single source of truth. Note it does NOT catch CancelledError — that
    must always propagate, never be retried. This is why we check explicit
    types rather than `except Exception`."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    return isinstance(exc, RETRYABLE)


async def part1_classification() -> None:
    banner("PART 1 — classify before you retry")

    samples = [
        RateLimitError("429", retry_after=1.0),
        TransientServerError("503"),
        ContentFilterError("400 content filter"),
        AuthError("401 invalid credential"),
        asyncio.CancelledError(),
    ]
    for e in samples:
        verdict = "RETRY" if is_retryable(e) else "FAIL FAST"
        print(f"    {type(e).__name__:<22} -> {verdict}")

    print("""
  For Azure OpenAI specifically:
     429  retryable, and it usually carries Retry-After — honour it.
     500/502/503/504  retryable.
     408 request timeout  retryable.
     400 content_filter  NOT retryable. Same input, same rejection.
     400 context_length_exceeded  NOT retryable — but IS repairable: truncate
          or re-chunk and try a genuinely different request. That is a
          fallback, not a retry, and belongs in different code.
     401/403  NOT retryable. Note managed-identity token refresh happens
          inside DefaultAzureCredential; a 401 that survives it is a real
          config or permission problem and must page a human.""")


# ---------------------------------------------------------------------------
# PART 2 — backoff strategies compared, with measured spread
# ---------------------------------------------------------------------------

def backoff_none(attempt: int, base: float) -> float:
    return base


def backoff_exponential(attempt: int, base: float) -> float:
    """No jitter. Every client waits exactly the same. Synchronised herd."""
    return base * (2 ** attempt)


def backoff_full_jitter(attempt: int, base: float, *, cap: float = 30.0) -> float:
    """AWS's "full jitter": sleep = U(0, min(cap, base * 2^attempt)).

    Best general-purpose choice. Maximally decorrelates clients. The trade-off
    is that a given client may retry sooner than a pure exponential would,
    which is usually fine and sometimes better.
    """
    return random.uniform(0, min(cap, base * (2 ** attempt)))


def backoff_decorrelated_jitter(prev: float, base: float, *, cap: float = 30.0) -> float:
    """sleep = min(cap, U(base, prev * 3)). Grows faster than full jitter under
    sustained failure while still spreading clients. Use when you care about
    backing off hard from a genuinely down dependency."""
    return min(cap, random.uniform(base, prev * 3))


async def part2_jitter() -> None:
    banner("PART 2 — why jitter is not optional (measured)")

    n_clients = 200
    base = 0.5
    attempt = 3

    for name, fn in [
        ("no backoff", lambda: backoff_none(attempt, base)),
        ("exponential, no jitter", lambda: backoff_exponential(attempt, base)),
        ("full jitter", lambda: backoff_full_jitter(attempt, base)),
    ]:
        waits = [fn() for _ in range(n_clients)]
        # Bucket into 0.5s windows to see the herd.
        buckets: dict[float, int] = {}
        for w in waits:
            b = round(w * 2) / 2
            buckets[b] = buckets.get(b, 0) + 1
        worst = max(buckets.values())
        print(f"\n    {name}")
        print(f"      spread over {len(buckets)} time bucket(s); "
              f"worst bucket holds {worst}/{n_clients} clients")
        if worst > n_clients * 0.5:
            print("      *** these clients all retry simultaneously — herd ***")

    print("""
  The number that matters is the worst bucket. With no jitter, 200/200
  clients hammer the recovering service in the same instant, guaranteeing a
  second failure and a third round. With full jitter the load is spread and
  the service can actually recover.""")


# ---------------------------------------------------------------------------
# PART 3 — a production-shaped retry wrapper
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.1
    max_delay: float = 5.0
    # The deadline is the important field. Without it, max_attempts x timeout
    # is your real worst case and it is always larger than you think.
    overall_deadline: float | None = 2.0


async def with_retries(
    fn,
    *,
    policy: RetryPolicy,
    op_name: str = "op",
    trace: list[str] | None = None,
):
    """Retry an async callable, honouring classification, jitter, Retry-After,
    and a global deadline.

    In real code use `tenacity` rather than hand-rolling — but you should be
    able to read tenacity's config and know what it does, which is what this
    function is for.
    """
    loop = asyncio.get_running_loop()
    deadline = (
        loop.time() + policy.overall_deadline
        if policy.overall_deadline is not None
        else None
    )
    last_exc: BaseException | None = None

    for attempt in range(policy.max_attempts):
        # Check the deadline BEFORE spending an attempt. Starting a call you
        # know cannot finish in budget wastes quota and delays the failure.
        if deadline is not None and loop.time() >= deadline:
            if trace is not None:
                trace.append(f"{op_name}: deadline exhausted before attempt {attempt}")
            break

        try:
            result = await fn()
            if trace is not None and attempt:
                trace.append(f"{op_name}: succeeded on attempt {attempt + 1}")
            return result

        except asyncio.CancelledError:
            # Never retry a cancellation. Propagate immediately.
            raise

        except BaseException as exc:
            if not is_retryable(exc):
                if trace is not None:
                    trace.append(f"{op_name}: {type(exc).__name__} not retryable, failing fast")
                raise
            last_exc = exc

            if attempt == policy.max_attempts - 1:
                break

            # Honour the server's own guidance when it gives us any. A server
            # saying "wait 1.0s" knows more about its recovery than your
            # exponential curve does.
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = float(retry_after)
            else:
                delay = backoff_full_jitter(attempt, policy.base_delay,
                                            cap=policy.max_delay)

            # Never sleep past the deadline — clamp it.
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                delay = min(delay, remaining)

            if trace is not None:
                trace.append(
                    f"{op_name}: attempt {attempt + 1} failed "
                    f"({type(exc).__name__}), sleeping {delay:.3f}s"
                )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


async def part3_wrapper() -> None:
    banner("PART 3 — retry wrapper in action")

    # A flaky endpoint: 60% of calls hit a transient failure.
    flaky = FakeLLMClient(seed=31, base_latency_s=0.02, jitter_s=0.0,
                          server_error_rate=0.6)

    print("\n  A) transient failures, eventually succeeding")
    trace: list[str] = []
    with Timer("retried call"):
        try:
            r = await with_retries(
                lambda: flaky.complete("classify PO exception"),
                policy=RetryPolicy(max_attempts=6, base_delay=0.05,
                                   overall_deadline=3.0),
                op_name="complete",
                trace=trace,
            )
            print(f"      result: {r.text[:44]}")
        except LLMError as e:
            print(f"      gave up: {e}")
    for line in trace:
        print(f"      {line}")

    print("\n  B) non-retryable error fails immediately (no wasted attempts)")
    trace2: list[str] = []

    async def content_filtered():
        raise ContentFilterError("400 content_filter: prompt rejected")

    with Timer("non-retryable call"):
        try:
            await with_retries(content_filtered,
                               policy=RetryPolicy(max_attempts=6),
                               op_name="complete", trace=trace2)
        except ContentFilterError:
            pass
    for line in trace2:
        print(f"      {line}")

    print("\n  C) deadline caps total time regardless of max_attempts")
    always_down = FakeLLMClient(seed=32, base_latency_s=0.02,
                                server_error_rate=1.0)
    trace3: list[str] = []
    with Timer("20 attempts allowed, 0.5s deadline"):
        try:
            await with_retries(
                lambda: always_down.complete("x"),
                policy=RetryPolicy(max_attempts=20, base_delay=0.05,
                                   overall_deadline=0.5),
                op_name="complete", trace=trace3,
            )
        except LLMError:
            print(f"      failed after {len(trace3)} logged events, "
                  f"bounded by the deadline not the attempt count")


# ---------------------------------------------------------------------------
# PART 4 — circuit breaker: stop retrying a corpse
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Three states. CLOSED = normal. OPEN = fail instantly without calling.
    HALF_OPEN = let one probe through to test recovery.

    Retries handle a blip. A circuit breaker handles an outage. Without one,
    every request pays the full retry budget while the dependency is down,
    so your latency and thread/connection usage explode exactly when you can
    least afford it. With one, you fail fast and shed load.
    """

    def __init__(self, threshold: int = 5, recovery_time: float = 1.0) -> None:
        self.threshold = threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.state = "CLOSED"
        self._opened_at = 0.0

    def _now(self) -> float:
        return time.monotonic()

    async def call(self, fn):
        if self.state == "OPEN":
            if self._now() - self._opened_at >= self.recovery_time:
                self.state = "HALF_OPEN"
            else:
                raise TransientServerError("circuit OPEN — failing fast, no call made")

        try:
            result = await fn()
        except Exception:
            self.failures += 1
            if self.state == "HALF_OPEN" or self.failures >= self.threshold:
                self.state = "OPEN"
                self._opened_at = self._now()
            raise
        else:
            # A single success in HALF_OPEN closes the circuit. Some designs
            # require N consecutive successes; that is safer for flappy
            # dependencies and worth doing if you see the circuit oscillate.
            self.failures = 0
            self.state = "CLOSED"
            return result


async def part4_breaker() -> None:
    banner("PART 4 — circuit breaker")

    down = FakeLLMClient(seed=41, base_latency_s=0.05, server_error_rate=1.0)
    cb = CircuitBreaker(threshold=3, recovery_time=0.3)

    print("\n  hammering a dead dependency:")
    for i in range(7):
        t0 = time.monotonic()
        try:
            await cb.call(lambda: down.complete("q"))
            outcome = "OK"
        except Exception as e:
            outcome = type(e).__name__
        dt = (time.monotonic() - t0) * 1000
        print(f"    req {i}: state={cb.state:<10} {outcome:<22} {dt:6.1f}ms")

    print("\n    Note the latency collapse once the circuit opens: we stop")
    print("    making calls at all. That is the point — you shed load instead")
    print("    of queueing behind a dependency that cannot serve you.")

    print("\n  after recovery window, dependency healthy again:")
    await asyncio.sleep(0.35)
    healthy = FakeLLMClient(seed=42, base_latency_s=0.02, server_error_rate=0.0)
    for i in range(3):
        try:
            await cb.call(lambda: healthy.complete("q"))
            print(f"    req {i}: state={cb.state} OK")
        except Exception as e:
            print(f"    req {i}: state={cb.state} {type(e).__name__}")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_classification()
    await part2_jitter()
    await part3_wrapper()
    await part4_breaker()

    banner("REVIEW CHECKLIST — retries")
    print("""
  [ ] Is there an explicit retryable/non-retryable classification, or does
      the code retry on bare `Exception`?
  [ ] Is CancelledError excluded from retry? (retrying a cancellation is a
      hang waiting to happen)
  [ ] Is there jitter? Check the actual formula, not the word "backoff".
  [ ] Is there a cap on the delay?
  [ ] Is there an OVERALL deadline, not just max_attempts?
  [ ] Is Retry-After honoured when present?
  [ ] Is the retried operation idempotent, or is an idempotency key sent?
  [ ] Is the semaphore acquired OUTSIDE the retry loop, so retries do not
      multiply real concurrency?
  [ ] Is there a circuit breaker for sustained outages?
  [ ] Are retry counts emitted as a metric? A rising retry rate is the
      earliest warning you get of a degrading dependency.
""")


if __name__ == "__main__":
    asyncio.run(main())
