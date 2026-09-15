"""
03 — The blocking-call trap: how one synchronous line freezes a whole service.

THE FAILURE
-----------
A coroutine that calls a *synchronous* function does not yield. The event loop
is on the same thread, so it stops. Every other task — including unrelated
users' requests, health-check responses, and heartbeat timers — is frozen for
the duration.

Where it actually comes from in AI/ML code, in rough order of frequency:
  1. A sync SDK used because the async one wasn't obvious (`requests`,
     `openai.OpenAI` instead of `openai.AsyncOpenAI`, sync `azure-*` clients).
  2. Tokenisation. `tiktoken.encode` on a 200-page PDF is CPU-bound and slow.
  3. Local embedding / reranking models (sentence-transformers, ONNX,
     cross-encoders). Pure CPU, hundreds of milliseconds.
  4. File I/O: `open().read()` on a large blob, PDF parsing, `pandas.read_*`.
  5. `time.sleep` in retry logic instead of `asyncio.sleep`. Devastating,
     because it scales with your retry count.
  6. Any `.json()` / pydantic validation on a very large payload.

HOW YOU DETECT IT
-----------------
Event-loop lag: schedule a heartbeat that should fire every N ms, then measure
how late it actually fires. Sustained lag above ~50ms means something on the
loop is not yielding. Ship this as a metric — it is the single highest-value
async health signal you can emit, and it is ~15 lines.

Run:  python 03_blocking_calls.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import os
import time

from fake_llm import BlockingLLMClient, Timer, banner

# ---------------------------------------------------------------------------
# The diagnostic tool: an event-loop lag monitor
# ---------------------------------------------------------------------------

class LoopLagMonitor:
    """Measures how late a periodic wake-up actually fires.

    We ask to be woken every `interval`. Any delay beyond that is time the
    loop spent unable to service us — i.e. time some other task held the
    thread without yielding.

    PRODUCTION NOTE: emit `max_lag` as a gauge to Application Insights. Alert
    above ~100ms sustained. This catches a whole class of bug that no amount
    of request-level tracing will show you, because the blocking task looks
    perfectly healthy — it is everything *else* that is slow.
    """

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.samples: list[float] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(self.interval)
            lag = (time.perf_counter() - t0) - self.interval
            self.samples.append(max(0.0, lag))

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="loop-lag-monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            # Await the cancellation so the task actually finishes unwinding.
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def settle(self) -> None:
        """Let the monitor's currently-pending sleep resolve and record.

        SUBTLE BUT IMPORTANT: while the loop is blocked, the monitor cannot
        run, so it records nothing. The lag sample only lands when the loop is
        freed and the monitor's overdue sleep finally returns. If you report
        immediately after the blocking call, you see "no samples" and the lag
        gets misattributed to the *next* phase. Always settle before reporting.
        This is not an artefact of the demo — it is exactly why loop-lag
        metrics can appear one scrape-interval late in production dashboards.
        """
        await asyncio.sleep(self.interval * 2)

    def report(self, label: str) -> None:
        if not self.samples:
            print(f"  {label}: no samples")
            return
        mx = max(self.samples) * 1000
        avg = sum(self.samples) / len(self.samples) * 1000
        verdict = "HEALTHY" if mx < 50 else "*** LOOP BLOCKED ***"
        print(f"  {label:<34} max_lag={mx:7.1f}ms  avg={avg:6.1f}ms  {verdict}")
        self.samples.clear()


# ---------------------------------------------------------------------------
# PART 1 — demonstrate the freeze
# ---------------------------------------------------------------------------

def cpu_bound_tokenise(text: str, rounds: int = 200_000) -> int:
    """Stands in for tiktoken on a large document, or a local embedding model.

    Pure CPU. No I/O. Nothing here can yield to the event loop, because there
    is no await and nothing to await. This is the shape of the problem.
    """
    h = text.encode()
    for _ in range(rounds):
        h = hashlib.sha256(h).digest()
    return len(h)


async def part1_the_freeze() -> None:
    banner("PART 1 — one blocking call freezes every other task")

    mon = LoopLagMonitor(interval=0.005)
    mon.start()

    # -- baseline: healthy loop, everything awaits properly
    await asyncio.sleep(0.3)
    mon.report("baseline (idle loop)")

    # -- the bug: sync CPU work called directly from a coroutine
    async def bad_tokenise(doc: str) -> int:
        return cpu_bound_tokenise(doc)  # no await — blocks the thread

    with Timer("blocking version"):
        await bad_tokenise("supply chain document")
    await mon.settle()
    mon.report("during blocking CPU call")

    # -- the fix: push it to a worker thread. `to_thread` returns an awaitable,
    #    so the loop is free while the OS thread does the work.
    async def good_tokenise(doc: str) -> int:
        return await asyncio.to_thread(cpu_bound_tokenise, doc)

    with Timer("to_thread version"):
        await good_tokenise("supply chain document")
    await mon.settle()
    mon.report("during to_thread CPU call")

    await mon.stop()

    print("""
  NOTE on to_thread and the GIL: for *CPU-bound pure-Python* work, the thread
  still contends for the GIL, so total throughput barely improves. What DOES
  improve is that the event loop keeps running — other requests are served,
  health checks answer, timeouts fire. That is usually the thing you actually
  needed. For genuine CPU parallelism, use a process pool (PART 3).
  Many ML libraries (numpy, torch, onnxruntime, tokenizers-rs) release the GIL
  inside their C extensions, so threads DO give real parallelism there.""")


# ---------------------------------------------------------------------------
# PART 2 — the sync-SDK case, and how to wrap it
# ---------------------------------------------------------------------------

async def part2_sync_sdk() -> None:
    banner("PART 2 — wrapping a synchronous vendor SDK")

    blocking_client = BlockingLLMClient(latency_s=0.10)
    prompts = [f"classify exception {i}" for i in range(6)]

    mon = LoopLagMonitor(interval=0.005)
    mon.start()

    # -- WRONG: looks async, is entirely sequential AND blocks the loop.
    #    This is the most common real-world instance of the bug, because the
    #    code *reads* as if it were concurrent.
    async def call_wrong(p: str) -> str:
        return blocking_client.complete(p)

    with Timer("sync SDK called directly (6 calls)"):
        await asyncio.gather(*(call_wrong(p) for p in prompts))
    await mon.settle()
    mon.report("  loop health (direct)")

    # -- RIGHT: each call goes to the default ThreadPoolExecutor, so the sleeps
    #    (which release the GIL) genuinely overlap.
    async def call_right(p: str) -> str:
        return await asyncio.to_thread(blocking_client.complete, p)

    with Timer("sync SDK via to_thread (6 calls)"):
        await asyncio.gather(*(call_right(p) for p in prompts))
    await mon.settle()
    mon.report("  loop health (to_thread)")

    await mon.stop()

    print("""
  GOTCHA: `to_thread` uses the loop's *default* executor, whose size defaults
  to min(32, cpu_count + 4). Fan out 500 wrapped calls and you do NOT get 500
  concurrent requests — you get a queue behind ~36 threads, and worse, you have
  starved every other to_thread user in the process (including asyncio's own
  DNS resolution). If you must wrap a sync SDK at scale, give it a DEDICATED,
  explicitly sized executor. See PART 3.

  GOTCHA 2: `to_thread` cannot be cancelled. `task.cancel()` makes the *await*
  raise CancelledError, but the thread keeps running to completion in the
  background. There is no way to interrupt it. For long sync work this means a
  timeout does not free the resource — plan capacity accordingly.""")


# ---------------------------------------------------------------------------
# PART 3 — dedicated executors: threads vs processes
# ---------------------------------------------------------------------------

async def part3_executors() -> None:
    banner("PART 3 — dedicated executors, and when to use processes")

    loop = asyncio.get_running_loop()
    docs = [f"document-{i}" for i in range(4)]

    # Heavier per-item work so process startup/pickling cost is amortised.
    # With small items the ProcessPool LOSES — measure before you reach for it.
    HEAVY = 800_000

    n_cpu = os.cpu_count() or 1
    print(f"  os.cpu_count() = {n_cpu}")
    if n_cpu < 2:
        print("  NOTE: only one CPU is visible to this process, so the ProcessPool")
        print("  below CANNOT beat the ThreadPool — there is no second core to use.")
        print("  This is the lesson, not a broken demo: container CPU limits are")
        print("  routinely below what the host reports, and a process pool sized to")
        print("  the host's core count on a 1-vCPU App Service or a K8s pod with")
        print("  `cpu: 1` buys you memory overhead and pickling cost for nothing.")
        print("  Always read the cgroup limit, not the hardware, and MEASURE.")

    # A dedicated thread pool. Sized for YOUR workload, isolated from the
    # default executor so you cannot starve unrelated code.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="tokenise"
    ) as pool, Timer("4 CPU tasks on a dedicated ThreadPool"):
        await asyncio.gather(
            *(
                loop.run_in_executor(pool, cpu_bound_tokenise, d, HEAVY)
                for d in docs
            )
        )

    # A process pool: true parallelism, separate interpreters, no GIL contention.
    # Costs: pickling arguments and results, process startup, no shared state.
    # Worth it only when the work is long enough to amortise that overhead —
    # rule of thumb, >100ms per item.
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        with Timer("4 CPU tasks on a ProcessPool"):
            await asyncio.gather(
                *(
                    loop.run_in_executor(pool, cpu_bound_tokenise, d, HEAVY)
                    for d in docs
                )
            )

    print("""
  REMEMBER the routing rule:
    network I/O           -> native async client. Never a thread.
    sync SDK / file I/O   -> to_thread or a dedicated ThreadPoolExecutor.
    CPU-bound pure Python -> ProcessPoolExecutor (or move it out of the
                             request path entirely, into a batch job).
    CPU-bound but the lib releases the GIL (numpy/torch/onnx/tokenizers)
                          -> threads are fine and much cheaper than processes.""")


# ---------------------------------------------------------------------------
# PART 4 — the debug-mode switch that finds these for you
# ---------------------------------------------------------------------------

async def part4_debug_mode() -> None:
    banner("PART 4 — asyncio debug mode as a CI gate")

    print("""
  Run any async test suite with:

      PYTHONASYNCIODEBUG=1 python -m pytest
      # or programmatically: asyncio.run(main(), debug=True)
      # or per-loop:         loop.slow_callback_duration = 0.1

  Debug mode logs a warning for any callback that occupies the loop longer
  than `slow_callback_duration` (default 0.1s), plus warnings for coroutines
  that were never awaited and for tasks destroyed while pending.

  This is cheap to turn on in CI and catches the blocking-call class before it
  reaches a shared environment. Pair it with the LoopLagMonitor above as a
  runtime metric, and you have both the pre- and post-deployment detector.""")

    # Demonstrate the never-awaited warning that debug mode makes loud.
    async def forgotten() -> None:
        await asyncio.sleep(0)

    c = forgotten()
    print(f"  (a never-awaited coroutine like {c.__name__!r} is flagged by debug mode)")
    c.close()


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_the_freeze()
    await part2_sync_sdk()
    await part3_executors()
    await part4_debug_mode()

    banner("REVIEW CHECKLIST — blocking calls")
    print("""
  In any PR touching async code, grep for:
    [ ] `time.sleep`        -> must be `await asyncio.sleep`
    [ ] `requests.`         -> must be httpx.AsyncClient / aiohttp
    [ ] `open(...).read()`  -> to_thread, or aiofiles, for large files
    [ ] sync vendor client  -> async variant, else to_thread + own executor
    [ ] tokenisers/local models in the request path -> executor or precompute
    [ ] `.encode(`/`.decode(` on large text inside a coroutine
    [ ] any `for` loop over >1k items with no await inside
""")


if __name__ == "__main__":
    asyncio.run(main())
