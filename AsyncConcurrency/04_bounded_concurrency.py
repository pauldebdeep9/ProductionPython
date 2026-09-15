"""
04 — Bounded concurrency: semaphores, rate limits, and backpressure.

WHY UNBOUNDED FAN-OUT IS A BUG, NOT AN OPTIMISATION
---------------------------------------------------
`await asyncio.gather(*(embed(d) for d in documents))` over 50,000 documents
does not "go fast". It:
  * builds 50,000 Task objects and 50,000 in-flight HTTP requests,
  * exhausts the connection pool, then file descriptors,
  * loads every document into memory simultaneously,
  * and gets you rate-limited into a retry storm that makes throughput *worse*
    than a sequential loop.

The fix is a concurrency limit. But there are actually THREE distinct limits,
and conflating them is the common error:

  1. CONCURRENCY  — how many requests are in flight at once. Bounded by a
                    Semaphore. Protects *your* memory, sockets, and the
                    downstream service's connection budget.
  2. RATE         — requests (or tokens) per minute. Bounded by a token
                    bucket. This is what Azure OpenAI actually enforces via
                    RPM/TPM quota on a deployment.
  3. BATCH SIZE   — how many items per request. Reduces total requests. Always
                    apply this FIRST; it is strictly cheaper than concurrency.

A semaphore of 8 does NOT keep you under 60 RPM if each call takes 50ms.
You need both.

Run:  python 04_bounded_concurrency.py
"""

from __future__ import annotations

import asyncio
import time

from fake_llm import FakeLLMClient, Timer, banner

# ---------------------------------------------------------------------------
# PART 1 — the semaphore pattern
# ---------------------------------------------------------------------------

async def part1_semaphore() -> None:
    banner("PART 1 — Semaphore: bound the number of in-flight requests")

    client = FakeLLMClient(seed=11, base_latency_s=0.05, jitter_s=0.0)

    # An observer so we can prove the bound actually holds.
    live = 0
    peak = 0

    async def guarded(sem: asyncio.Semaphore, doc: int) -> None:
        nonlocal live, peak
        # THE PATTERN: acquire around the *entire* unit of work, including any
        # retries. Acquiring only around the HTTP call and then retrying
        # outside the semaphore silently multiplies your real concurrency.
        async with sem:
            live += 1
            peak = max(peak, live)
            try:
                await client.complete(f"doc {doc}")
            finally:
                live -= 1

    for limit in (4, 16):
        live = peak = 0
        sem = asyncio.Semaphore(limit)
        with Timer(f"64 docs, semaphore limit={limit}"):
            async with asyncio.TaskGroup() as tg:
                for i in range(64):
                    tg.create_task(guarded(sem, i))
        print(f"     observed peak in-flight: {peak} (limit was {limit})")

    print("""
  GOTCHA: `asyncio.Semaphore` is NOT thread-safe and is bound to the loop it
  was first awaited on. Create it inside your async context, never as a
  module-level global that might outlive a loop — under pytest-asyncio, a
  module-level semaphore reused across tests binds to a dead loop and raises
  a confusing "attached to a different loop" error.""")


# ---------------------------------------------------------------------------
# PART 2 — a reusable bounded-map helper
# ---------------------------------------------------------------------------

async def bounded_map(
    coro_fn,
    items: list,
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list:
    """Apply an async function over items with bounded concurrency, preserving
    input order in the output.

    This is the single most reused helper in LLM pipeline code. Worth having
    in a shared internal library rather than reimplemented per repo.

    Why order preservation matters: you almost always need to zip results back
    to the source records (doc_id, invoice line, chunk offset). Losing that
    correspondence turns a debuggable pipeline into a guessing game.
    """
    sem = asyncio.Semaphore(limit)
    results: list = [None] * len(items)

    async def run(idx: int, item) -> None:
        async with sem:
            try:
                results[idx] = await coro_fn(item)
            except Exception as e:
                if not return_exceptions:
                    raise
                results[idx] = e

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            tg.create_task(run(i, item))
    return results


async def part2_bounded_map() -> None:
    banner("PART 2 — a reusable order-preserving bounded map")

    client = FakeLLMClient(seed=12, base_latency_s=0.03, jitter_s=0.06)

    async def summarise(doc: str) -> str:
        c = await client.complete(f"summarise {doc}")
        return c.text[-12:]

    docs = [f"po-{i:03d}" for i in range(20)]
    with Timer("bounded_map over 20 docs, limit=5"):
        out = await bounded_map(summarise, docs, limit=5)

    print(f"     results are positionally aligned with inputs: {len(out)} items")
    print(f"     first: {docs[0]} -> {out[0]}")
    print(f"     last:  {docs[-1]} -> {out[-1]}")


# ---------------------------------------------------------------------------
# PART 3 — token-bucket rate limiting (RPM / TPM)
# ---------------------------------------------------------------------------

class TokenBucket:
    """Classic token bucket, async-safe for a single event loop.

    `rate` tokens are added per second, up to `capacity`. Each acquire waits
    until enough tokens exist. Capacity > rate allows a controlled burst,
    which matters because real quota windows are not instantaneous.

    Use one bucket per DEPLOYMENT, not per client object — the quota is
    enforced on the Azure OpenAI deployment, so two client instances pointed at
    the same deployment must share the bucket or you will exceed quota.

    For TPM (tokens-per-minute) limiting, call acquire(n) with your *estimated*
    prompt+completion token count, then reconcile against the actual usage the
    API returns. Estimating low is how you get surprise 429s.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._updated = time.monotonic()
        # A lock, because the check-then-deduct below spans an await and is
        # therefore NOT atomic. This is the canonical case where async code
        # genuinely needs a lock. See 09_pitfalls.py.
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        # monotonic(), not time(): wall-clock can jump backwards on NTP sync
        # and hand you a negative elapsed, which silently breaks the limiter.
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self, n: float = 1.0) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait = deficit / self.rate
            # Sleep OUTSIDE the lock, or you serialise every waiter behind the
            # first one and destroy throughput. Easy mistake to make.
            await asyncio.sleep(wait)


async def part3_rate_limit() -> None:
    banner("PART 3 — token bucket: respecting RPM/TPM quota")

    client = FakeLLMClient(seed=13, base_latency_s=0.01, jitter_s=0.0)
    bucket = TokenBucket(rate=20.0, capacity=5.0)  # 20 req/s, burst of 5
    sem = asyncio.Semaphore(10)

    stamps: list[float] = []

    async def call(i: int) -> None:
        # Order matters: take the rate token FIRST, then the concurrency slot.
        # Reversed, you hold a scarce concurrency slot while merely waiting for
        # rate, which needlessly caps throughput.
        await bucket.acquire(1.0)
        async with sem:
            stamps.append(time.monotonic())
            await client.complete(f"req {i}")

    t0 = time.monotonic()
    with Timer("30 requests at 20 req/s with burst 5"):
        async with asyncio.TaskGroup() as tg:
            for i in range(30):
                tg.create_task(call(i))

    # Verify the limiter actually worked: bucket every request into 1s windows.
    windows: dict[int, int] = {}
    for s in stamps:
        w = int(s - t0)
        windows[w] = windows.get(w, 0) + 1
    print("     requests per 1s window:", dict(sorted(windows.items())))
    print("     expected in window 0: capacity(5) drained instantly + rate(20)")
    print("     refilled over that second = ~25. Check the number above against")
    print("     that prediction — a limiter you have not verified arithmetically")
    print("     is a limiter you will discover is wrong via a 429 storm.")


# ---------------------------------------------------------------------------
# PART 4 — backpressure: the limit that isn't a semaphore
# ---------------------------------------------------------------------------

async def part4_backpressure() -> None:
    banner("PART 4 — backpressure: bounding MEMORY, not just concurrency")

    # A semaphore bounds in-flight *requests*. It does NOT bound how much data
    # you have pulled into memory. This is the failure mode of:
    #
    #     all_docs = [load(p) for p in every_path]     # 50k PDFs in RAM
    #     await bounded_map(embed, all_docs, limit=8)  # "but I bounded it!"
    #
    # You bounded the requests and OOM'd on the inputs. The fix is a bounded
    # queue feeding a lazy producer, so the producer STOPS when consumers lag.

    queue: asyncio.Queue = asyncio.Queue(maxsize=4)  # <-- the backpressure
    produced: list[int] = []
    consumed: list[int] = []

    async def producer() -> None:
        for i in range(12):
            # `put` blocks once the queue is full. That block propagates all
            # the way back to whatever is generating work — which is exactly
            # what you want: the slowest stage sets the pace.
            await queue.put(i)
            produced.append(i)
        for _ in range(2):
            await queue.put(None)  # sentinel per consumer

    async def consumer(cid: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            await asyncio.sleep(0.05)  # slow consumer
            consumed.append(item)
            queue.task_done()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        for c in range(2):
            tg.create_task(consumer(c))

    print(f"     produced {len(produced)}, consumed {len(consumed)}")
    print("     queue maxsize=4 meant at most 4 items were ever buffered,")
    print("     regardless of how fast the producer could have gone.")
    print("""
  REMEMBER: three separate limits, three separate mechanisms.
     in-flight requests  -> Semaphore
     requests/tokens per minute -> TokenBucket (shared per deployment)
     memory / buffered work -> bounded asyncio.Queue
  A design that has only the first one will still fall over.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_semaphore()
    await part2_bounded_map()
    await part3_rate_limit()
    await part4_backpressure()

    banner("SIZING GUIDANCE")
    print("""
  Start from the constraint, not from a round number:

    limit = min(
        downstream_quota_rpm / 60 * avg_latency_s,   # Little's Law
        connection_pool_size,                         # httpx default is 100
        memory_budget / avg_item_bytes,
    )

  Little's Law is the useful one: concurrency = arrival_rate x latency.
  To sustain 100 req/s at 400ms average latency you need ~40 in flight.
  Set the semaphore from that number, then verify with a load test rather
  than trusting the arithmetic — tail latency, not average, sets the real
  requirement.

  And make the limit CONFIGURABLE. It is the first thing you will tune in
  production and you do not want a redeploy to change it.
""")


if __name__ == "__main__":
    asyncio.run(main())
