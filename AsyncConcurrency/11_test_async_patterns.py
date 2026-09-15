"""
11 — Testing async code.

Run:  pip install pytest pytest-asyncio
      pytest 11_test_async_patterns.py -v

WHY THIS IS A SEPARATE TOPIC
----------------------------
Async bugs are concurrency bugs, and concurrency bugs are the ones that pass
tests. The specific traps:

  * Tests that pass because they are effectively sequential. If your test
    awaits each call in turn, you never exercise the interleaving where the
    bug lives.
  * Tests that sleep. `await asyncio.sleep(0.5)` to "let things settle" makes
    the suite slow AND flaky — it is a race with a longer fuse.
  * Tests that never exercise cancellation. Cancellation paths are the least
    tested and most load-bearing code in an async service.
  * Loop-bound fixtures reused across tests (see PITFALL 6).

The techniques below address each. They are ordinary pytest — the only async
addition is `pytest-asyncio`.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
import pytest_asyncio

from fake_llm import (
    ContentFilterError,
    FakeLLMClient,
    TransientServerError,
)

# Applies asyncio mode to every test in the module, so individual tests do not
# each need @pytest.mark.asyncio. In pyproject.toml the equivalent is:
#     [tool.pytest.ini_options]
#     asyncio_mode = "auto"
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures: build loop-bound objects INSIDE the async fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client() -> FakeLLMClient:
    """Function-scoped, constructed per test.

    GOTCHA: an `async def` fixture must be decorated with
    `@pytest_asyncio.fixture`, NOT `@pytest.fixture`. With plain
    `@pytest.fixture` current pytest-asyncio raises a confusing setup error
    about "no plugin or hook that handled it" — the fixture is never awaited
    and your test receives a coroutine object. Setting
    `asyncio_mode = "auto"` in pyproject.toml makes `@pytest.fixture` work
    too; in strict mode (the default) you need the explicit decorator.

    Why function-scoped: a module-scoped fixture holding an asyncio.Semaphore
    or Queue binds to the first test's event loop and breaks in the second
    (see PITFALL 6). If you need expensive shared setup, share the CONFIG,
    not the loop-bound primitive.
    """
    return FakeLLMClient(seed=1, base_latency_s=0.001, jitter_s=0.0)


# ---------------------------------------------------------------------------
# 1 — Test that concurrency actually happened
# ---------------------------------------------------------------------------

async def test_calls_actually_overlap(client: FakeLLMClient) -> None:
    """Assert on OBSERVED PEAK CONCURRENCY, not on elapsed time.

    A timing assertion (`assert elapsed < 0.5`) is the classic flaky test: it
    fails on a loaded CI runner and passes on your laptop, so people add
    slack until it asserts nothing. Instrumenting the code under test to
    report its own peak concurrency is deterministic and actually tests the
    property you care about.
    """
    live = 0
    peak = 0

    async def tracked(i: int) -> str:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            r = await client.complete(f"q{i}")
            return r.text
        finally:
            live -= 1

    results = await asyncio.gather(*(tracked(i) for i in range(10)))

    assert len(results) == 10
    assert peak > 1, "calls ran sequentially — concurrency was never exercised"


async def test_semaphore_bound_is_respected(client: FakeLLMClient) -> None:
    """The bound is the contract; assert on it directly."""
    limit = 3
    sem = asyncio.Semaphore(limit)
    live = 0
    peak = 0

    async def guarded(i: int) -> None:
        nonlocal live, peak
        async with sem:
            live += 1
            peak = max(peak, live)
            try:
                await client.complete(f"q{i}")
            finally:
                live -= 1

    async with asyncio.TaskGroup() as tg:
        for i in range(20):
            tg.create_task(guarded(i))

    assert peak == limit, f"expected peak {limit}, observed {peak}"


# ---------------------------------------------------------------------------
# 2 — Test cancellation explicitly (including the DOUBLE cancel)
# ---------------------------------------------------------------------------

async def test_cleanup_runs_on_cancellation() -> None:
    released: list[str] = []

    async def work() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            released.append("lock")

    t = asyncio.create_task(work())
    await asyncio.sleep(0)          # let it reach the await
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert released == ["lock"]


async def test_shielded_cleanup_survives_double_cancel() -> None:
    """THE TEST MOST PEOPLE SKIP.

    A single cancellation lets an unshielded `finally` await complete (verified
    in 05). Only a SECOND cancellation kills it — which is exactly what a
    shutdown drain does. So a single-cancel test passes even when the code is
    broken for the case you actually care about.
    """
    written: list[str] = []

    async def audit() -> None:
        await asyncio.sleep(0.02)
        written.append("record")

    async def handler() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(0.5):
                    await asyncio.shield(audit())

    t = asyncio.create_task(handler())
    await asyncio.sleep(0)
    t.cancel()                       # drain
    await asyncio.sleep(0)           # enter finally
    t.cancel()                       # hard cancel — lands on the cleanup await
    with contextlib.suppress(asyncio.CancelledError):
        await t
    await asyncio.sleep(0.05)        # let the shielded write land

    assert written == ["record"], "audit record lost during shutdown drain"


async def test_cancelled_error_is_not_swallowed() -> None:
    """Guards the review rule from 05: every `except CancelledError` re-raises."""

    async def well_behaved() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise  # the re-raise IS the behaviour under test

    t = asyncio.create_task(well_behaved())
    await asyncio.sleep(0)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert t.cancelled()


# ---------------------------------------------------------------------------
# 3 — Test timeouts without sleeping through them
# ---------------------------------------------------------------------------

async def test_timeout_fires_without_real_waiting() -> None:
    """Keep the fake latency and the timeout both tiny.

    Testing a 30s production timeout by waiting 30s is not a test, it is a
    delay. Make the DURATION configurable in the code under test and inject a
    small value. If the duration is hardcoded, that is the defect to fix
    first — untestable timeouts are untuned timeouts.
    """
    async def slow() -> None:
        await asyncio.sleep(1.0)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await slow()


async def test_deadline_is_shared_across_steps() -> None:
    """Verify the budget is end-to-end, not per-step."""
    loop = asyncio.get_running_loop()
    completed_steps = 0

    async def step() -> None:
        nonlocal completed_steps
        await asyncio.sleep(0.02)
        completed_steps += 1

    with pytest.raises(TimeoutError):
        async with asyncio.timeout_at(loop.time() + 0.05):
            for _ in range(10):
                await step()

    # Some steps ran, then the SHARED budget stopped us. If each step had its
    # own 0.05s timeout, all 10 would have completed.
    assert 1 <= completed_steps < 10


# ---------------------------------------------------------------------------
# 4 — Test error classification and retry behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc,should_retry",
    [
        (TransientServerError("503"), True),
        (TimeoutError(), True),
        (ContentFilterError("400"), False),
        (ValueError("bad input"), False),
    ],
    ids=["503", "timeout", "content_filter", "value_error"],
)
async def test_retry_classification(exc: Exception, should_retry: bool) -> None:
    """Parametrise the taxonomy. When someone adds a new error type, this
    table is where they are forced to make a decision about it."""
    attempts = 0

    async def failing():
        nonlocal attempts
        attempts += 1
        raise exc

    RETRYABLE = (TransientServerError, TimeoutError, ConnectionError)

    async def retry(fn, max_attempts: int = 3):
        for i in range(max_attempts):
            try:
                return await fn()
            except RETRYABLE:
                if i == max_attempts - 1:
                    raise
                await asyncio.sleep(0)
            except Exception:
                raise

    with pytest.raises(type(exc)):
        await retry(failing)

    assert attempts == (3 if should_retry else 1)


async def test_retry_eventually_succeeds() -> None:
    """A flaky dependency that recovers. Deterministic — no randomness."""
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TransientServerError("503")
        return "ok"

    async def retry(fn, max_attempts: int = 5):
        for i in range(max_attempts):
            try:
                return await fn()
            except TransientServerError:
                if i == max_attempts - 1:
                    raise
                await asyncio.sleep(0)

    assert await retry(flaky) == "ok"
    assert calls == 3


# ---------------------------------------------------------------------------
# 5 — Test partial failure without losing the failures
# ---------------------------------------------------------------------------

async def test_partial_failure_is_partitioned() -> None:
    """The bug from 02: exceptions collected into a list nobody inspects."""

    async def work(i: int) -> int:
        if i % 3 == 0:
            raise ContentFilterError(f"item {i} rejected")
        return i * 2

    results = await asyncio.gather(*(work(i) for i in range(9)),
                                   return_exceptions=True)

    ok = [r for r in results if not isinstance(r, BaseException)]
    bad = [r for r in results if isinstance(r, BaseException)]

    assert len(ok) == 6
    assert len(bad) == 3
    # The important assertion: positional correspondence survives, so a failure
    # can be traced back to its input.
    assert isinstance(results[0], ContentFilterError)
    assert results[1] == 2


async def test_taskgroup_raises_exception_group() -> None:
    """Guards against the gather -> TaskGroup migration trap: a plain
    `except SomeError` does NOT catch an ExceptionGroup."""

    async def fail(i: int) -> None:
        raise ContentFilterError(f"item {i}")

    caught = 0
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(3):
                tg.create_task(fail(i))
    except* ContentFilterError as eg:
        caught = len(eg.exceptions)

    assert caught == 3, "all sibling failures should surface, not just the first"


# ---------------------------------------------------------------------------
# 6 — Test the blocking-call guard
# ---------------------------------------------------------------------------

async def test_event_loop_is_not_blocked() -> None:
    """A regression test for the class of bug in 03.

    Run a heartbeat alongside the code under test and assert on max lag. Put
    this around any code path where someone might reintroduce a sync SDK call.
    Tune the threshold generously (200ms) so CI noise does not cause flakes —
    you are catching 'blocked for seconds', not 'blocked for 5ms'.
    """
    lags: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            t0 = loop.time()
            await asyncio.sleep(0.005)
            lags.append((loop.time() - t0) - 0.005)

    async def code_under_test() -> None:
        # Correct: yields to the loop.
        for _ in range(20):
            await asyncio.sleep(0.005)

    hb = asyncio.create_task(heartbeat())
    await code_under_test()
    stop.set()
    await hb

    assert max(lags) < 0.2, f"event loop blocked for {max(lags) * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# 7 — Deterministic ordering without sleeps
# ---------------------------------------------------------------------------

async def test_ordering_with_events_not_sleeps() -> None:
    """Coordinate with asyncio.Event, never with sleep.

    `await asyncio.sleep(0.1)  # let the worker start` is a race condition
    with a timer attached. An Event makes the dependency explicit and the test
    both faster and deterministic.
    """
    order: list[str] = []
    worker_started = asyncio.Event()
    may_finish = asyncio.Event()

    async def worker() -> None:
        order.append("worker:start")
        worker_started.set()
        await may_finish.wait()
        order.append("worker:finish")

    t = asyncio.create_task(worker())
    await worker_started.wait()          # deterministic, not timed
    order.append("test:observed_start")
    may_finish.set()
    await t

    assert order == ["worker:start", "test:observed_start", "worker:finish"]


# ---------------------------------------------------------------------------
# 8 — Detect leaked tasks
# ---------------------------------------------------------------------------

async def test_no_tasks_leak() -> None:
    """Assert the code under test leaves nothing running.

    Worth promoting to an autouse fixture across the whole suite: it catches
    fire-and-forget tasks (PITFALL 1) and missing cleanup automatically, on
    every test, for free.
    """
    before = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}

    async def well_behaved() -> None:
        async with asyncio.TaskGroup() as tg:
            for _ in range(3):
                tg.create_task(asyncio.sleep(0.001))

    await well_behaved()
    await asyncio.sleep(0)

    after = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    leaked = after - before
    assert not leaked, f"leaked tasks: {[t.get_name() for t in leaked]}"


# ---------------------------------------------------------------------------
# 9 — Stress the interleaving
# ---------------------------------------------------------------------------

async def test_race_is_caught_under_contention() -> None:
    """Concurrency bugs need contention to appear. A test with 2 tasks will
    not find the race that 50 tasks find. Where a shared resource exists, turn
    the concurrency up in the test well past production levels."""

    class Budget:
        def __init__(self, total: int) -> None:
            self.remaining = total
            self.lock = asyncio.Lock()

        async def spend(self, n: int) -> bool:
            async with self.lock:
                if self.remaining < n:
                    return False
                await asyncio.sleep(0)     # yield inside the critical section
                self.remaining -= n
                return True

    b = Budget(100)
    results = await asyncio.gather(*(b.spend(10) for _ in range(50)))

    assert sum(results) == 10, "exactly 10 spends of 10 should fit in 100"
    assert b.remaining == 0, f"budget went to {b.remaining} — lock is not holding"
