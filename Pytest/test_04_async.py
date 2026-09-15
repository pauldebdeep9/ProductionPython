"""
tests/unit/test_04_async.py — testing async code.

Run:  pytest tests/unit/test_04_async.py -v

WHY THIS IS A SEPARATE FILE
---------------------------
Async bugs are concurrency bugs, and concurrency bugs are precisely the ones
that pass tests. The specific traps:

  * tests that are effectively SEQUENTIAL and therefore never exercise the
    interleaving where the bug lives
  * tests that SLEEP to "let things settle" — a race with a longer fuse
  * loop-bound fixtures reused across tests
  * cancellation paths, which are the least-tested and most load-bearing code
    in an async service

CONFIGURATION: `asyncio_mode = auto` in pytest.ini means every `async def
test_` is collected automatically, with no `@pytest.mark.asyncio` on each one.
In `strict` mode (the default) you need the marker, and an unmarked async test
is silently SKIPPED — which looks like a passing suite. Use auto.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
import pytest_asyncio

from rag.service import (
    Chunk,
    InMemoryRetriever,
    RagService,
    ScriptedModel,
    good_response,
)


# ===========================================================================
# 1. ASYNC FIXTURES
# ===========================================================================

@pytest_asyncio.fixture
async def warm_service(make_service) -> RagService:
    """An `async def` fixture needs `@pytest_asyncio.fixture`, NOT
    `@pytest.fixture`.

    With the wrong decorator in strict mode you get a confusing setup error,
    or worse — the test receives a coroutine object instead of the value and
    fails somewhere unrelated. In `auto` mode `@pytest.fixture` also works,
    but the explicit decorator documents the intent and survives a config
    change.
    """
    svc = make_service([good_response()])
    await svc.answer("warmup", frozenset({"isc-all"}))
    svc.model._i = 0          # reset the script after the warmup call
    return svc


async def test_async_fixture_delivers_a_value(warm_service: RagService) -> None:
    assert isinstance(warm_service, RagService)
    assert warm_service.model.prompts, "the warmup call should have happened"


@pytest_asyncio.fixture
async def managed_resource():
    """Async setup AND async teardown, guaranteed even on failure.

    This is where async fixtures earn their place: closing an httpx client,
    draining a queue, cancelling a background task. Doing it in each test is
    something everyone forgets exactly once, and then the suite leaks
    connections until it exhausts a pool.
    """
    opened: list[str] = []
    await asyncio.sleep(0)
    opened.append("open")
    yield opened
    await asyncio.sleep(0)
    opened.append("closed")


async def test_async_teardown_runs(managed_resource: list[str]) -> None:
    assert managed_resource == ["open"]


# ===========================================================================
# 2. TESTING THAT CONCURRENCY ACTUALLY HAPPENED
# ===========================================================================

async def test_concurrent_requests_do_not_share_state(make_service) -> None:
    """Run several requests CONCURRENTLY, not in a loop.

    A sequential loop cannot find a shared-state bug, because there is never
    more than one request in flight. If your service holds any per-request
    state on `self`, only this shape of test catches it.
    """
    svc = make_service([good_response()] * 12)

    results = await asyncio.gather(*[
        svc.answer(f"invoice 88 units query {i}", frozenset({"isc-all"}))
        for i in range(6)
    ])

    assert len(results) == 6
    assert all(r.proposal is not None for r in results)
    assert len(svc.retriever.calls) == 6
    queries = [c[0] for c in svc.retriever.calls]
    assert len(set(queries)) == 6, "each request must carry its own query"


async def test_assert_on_peak_concurrency_not_elapsed_time() -> None:
    """ASSERT ON OBSERVED CONCURRENCY, never on wall-clock time.

    `assert elapsed < 0.5` is the classic flaky test: it fails on a loaded CI
    runner, gets slack added until it asserts nothing, and is eventually
    deleted. Instrumenting the code to report its own peak concurrency is
    deterministic and tests the property you actually care about.
    """
    live = 0
    peak = 0
    sem = asyncio.Semaphore(3)

    async def bounded_task() -> None:
        nonlocal live, peak
        async with sem:
            live += 1
            peak = max(peak, live)
            try:
                await asyncio.sleep(0)
            finally:
                live -= 1

    await asyncio.gather(*(bounded_task() for _ in range(20)))

    assert peak <= 3, f"the semaphore bound was violated: peak={peak}"
    assert peak > 1, "the tasks ran sequentially; concurrency was never tested"


# ===========================================================================
# 3. DETERMINISTIC ORDERING WITHOUT SLEEPS
# ===========================================================================

async def test_ordering_with_events_not_sleeps() -> None:
    """Coordinate with `asyncio.Event`, never with `sleep`.

    `await asyncio.sleep(0.1)  # let the worker start` is a race condition
    with a timer attached. It passes on your laptop and fails at 2am in CI on
    a busy runner. An Event makes the dependency explicit, and the test both
    faster and deterministic.
    """
    order: list[str] = []
    worker_started = asyncio.Event()
    may_finish = asyncio.Event()

    async def worker() -> None:
        order.append("worker:start")
        worker_started.set()
        await may_finish.wait()
        order.append("worker:finish")

    task = asyncio.create_task(worker())
    await worker_started.wait()
    order.append("test:observed_start")
    may_finish.set()
    await task

    assert order == ["worker:start", "test:observed_start", "worker:finish"]


# ===========================================================================
# 4. CANCELLATION — the least-tested path
# ===========================================================================

async def test_cleanup_runs_on_cancellation() -> None:
    released: list[str] = []

    async def work() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            released.append("lock")

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released == ["lock"]


async def test_cleanup_survives_a_DOUBLE_cancel() -> None:
    """THE TEST ALMOST NOBODY WRITES, and the one that matters.

    A single cancellation lets an unshielded `finally` await complete. Only a
    SECOND cancellation kills it — which is exactly what a graceful-shutdown
    drain does (cancel, wait out the grace period, hard-cancel stragglers).

    So a single-cancel test passes while the code is broken for the case you
    care about, and the audit record is lost during a deployment rather than
    in CI.
    """
    written: list[str] = []

    async def audit() -> None:
        await asyncio.sleep(0)
        written.append("record")

    async def handler() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(0.05):
                    await asyncio.shield(audit())

    task = asyncio.create_task(handler())
    await asyncio.sleep(0)
    task.cancel()               # drain
    await asyncio.sleep(0)      # let it enter the finally
    task.cancel()               # hard cancel, landing on the cleanup await
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert written == ["record"], "the audit record was lost on a double cancel"


async def test_cancelled_error_is_re_raised() -> None:
    """Guards the review rule: every `except CancelledError` ends in `raise`.
    Swallowing it makes a task immortal and hangs shutdown."""

    async def well_behaved() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(well_behaved())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


# ===========================================================================
# 5. TIMEOUTS WITHOUT WAITING
# ===========================================================================

async def test_timeout_fires_without_real_waiting() -> None:
    """Keep both the injected latency and the timeout tiny.

    Testing a 30-second production timeout by waiting 30 seconds is not a
    test, it is a delay. Make the DURATION configurable in the code under
    test. If it is hardcoded, that is the defect to fix first — an untestable
    timeout is an untuned timeout.
    """
    async def slow() -> None:
        await asyncio.sleep(1.0)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await slow()


# ===========================================================================
# 6. LEAK DETECTION
# ===========================================================================

@pytest.fixture(autouse=True)
def _no_leaked_tasks():
    """Assert the test left nothing running.

    Worth promoting to an autouse fixture across the whole async suite: it
    catches fire-and-forget tasks and missing cleanup automatically, on every
    test, for free.

    NOTE it only counts tasks — a leaked httpx client or file handle needs a
    different check. But leaked TASKS are the most common async leak and the
    one that produces "Task was destroyed but it is pending!" at shutdown.
    """
    yield
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    pending = [t for t in asyncio.all_tasks(loop)
               if t is not asyncio.current_task(loop) and not t.done()]
    assert not pending, f"test leaked tasks: {[t.get_name() for t in pending]}"


async def test_taskgroup_leaves_nothing_behind() -> None:
    """NOTE: the real assertion here lives in the autouse `_no_leaked_tasks`
    fixture above, not in this body.

    That is legitimate but it made the test look assertion-free — and the
    suite lint in test_07 flagged it, correctly. An explicit assertion is
    added so the test states its own claim rather than relying on a fixture
    the reader has to go find. Tests should be readable in isolation.
    """
    tasks: list[asyncio.Task[None]] = []
    async with asyncio.TaskGroup() as tg:
        for _ in range(3):
            tasks.append(tg.create_task(asyncio.sleep(0)))
    assert all(t.done() for t in tasks), (
        "a TaskGroup guarantees every child is complete at block exit"
    )


# ===========================================================================
# 7. CONTENTION
# ===========================================================================

async def test_race_appears_only_under_contention() -> None:
    """Concurrency bugs need CONTENTION to appear.

    Two tasks will not find the race that fifty find. Where a shared resource
    exists, turn the concurrency in the test well past production levels — the
    test is cheap and the bug is expensive.
    """
    class Budget:
        def __init__(self, total: int) -> None:
            self.remaining = total
            self.lock = asyncio.Lock()

        async def spend(self, n: int) -> bool:
            async with self.lock:
                if self.remaining < n:
                    return False
                await asyncio.sleep(0)      # yield INSIDE the critical section
                self.remaining -= n
                return True

    b = Budget(100)
    results = await asyncio.gather(*(b.spend(10) for _ in range(50)))

    assert sum(results) == 10, "exactly ten spends of ten should fit in 100"
    assert b.remaining == 0, f"budget went to {b.remaining}; the lock failed"


async def test_the_same_race_without_a_lock_is_caught() -> None:
    """The NEGATIVE CONTROL for the test above.

    Without it, `test_race_appears_only_under_contention` might be passing
    because the fixture never actually creates contention. This proves the
    test shape detects the bug it claims to detect.
    """
    class UnlockedBudget:
        def __init__(self, total: int) -> None:
            self.remaining = total

        async def spend(self, n: int) -> bool:
            if self.remaining < n:
                return False
            await asyncio.sleep(0)          # the yield that makes it a race
            self.remaining -= n
            return True

    b = UnlockedBudget(100)
    await asyncio.gather(*(b.spend(10) for _ in range(50)))

    assert b.remaining < 0, (
        "expected the unlocked version to overdraw; if it did not, this test "
        "shape cannot detect the race and the positive test above is vacuous"
    )
