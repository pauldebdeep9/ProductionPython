"""
05 — Timeouts and cancellation: the semantics people get wrong.

THE MENTAL MODEL
----------------
Cancellation is delivered as an EXCEPTION (`asyncio.CancelledError`) raised at
whatever `await` the task is currently suspended on. It is not a kill switch.
Three consequences that trip everyone up:

  1. A task with no `await` in its current stretch of code CANNOT be cancelled
     until it reaches one. `task.cancel()` on a task doing a tight CPU loop
     does nothing until that loop finishes.
  2. Cancellation is a REQUEST. The task can catch CancelledError, do cleanup,
     and even (badly) refuse to die. `task.cancel()` returning does not mean
     the task stopped — you must `await` it.
  3. `CancelledError` inherits from `BaseException`, not `Exception`. That was
     a deliberate 3.8 change so that `except Exception:` does not accidentally
     swallow it. `except BaseException:` and bare `except:` still do — and both
     are bugs in async code.

WHY IT MATTERS FOR LLM WORK
---------------------------
A user closes the browser tab mid-stream. A request exceeds its SLA. A batch
job is being drained for deployment. In all three cases you want the in-flight
model call abandoned promptly, its connection released, its partial audit
record written, and its cost accounted for. None of that happens by accident.

Run:  python 05_timeouts_cancellation.py
"""

from __future__ import annotations

import asyncio

from fake_llm import FakeLLMClient, Timer, banner

# ---------------------------------------------------------------------------
# PART 1 — asyncio.timeout, the modern API
# ---------------------------------------------------------------------------

async def slow_generation(client: FakeLLMClient, secs: float) -> str:
    """A model call that takes `secs`, with cleanup we can observe."""
    try:
        await asyncio.sleep(secs)
        return "generated answer"
    except asyncio.CancelledError:
        print("      -> generation saw CancelledError, releasing connection")
        raise  # ALWAYS re-raise. Swallowing it breaks the caller's contract.


async def part1_timeout() -> None:
    banner("PART 1 — asyncio.timeout (3.11+) vs wait_for")

    client = FakeLLMClient(seed=21)

    # `asyncio.timeout` is a context manager. It cancels everything inside the
    # block when the deadline passes, then converts the CancelledError into a
    # TimeoutError at the block boundary. Prefer it over `wait_for`: it works
    # over an arbitrary block, not just a single awaitable, and it composes.
    print("\n  A) deadline exceeded")
    with Timer("timeout(0.2) around a 1.0s call"):
        try:
            async with asyncio.timeout(0.2):
                await slow_generation(client, 1.0)
        except TimeoutError:
            # NOTE: since 3.11, asyncio.TimeoutError IS the builtin TimeoutError.
            # Older code catching `asyncio.TimeoutError` still works.
            print("      caught TimeoutError at the block boundary")

    print("\n  B) inside the deadline — no exception")
    async with asyncio.timeout(0.5):
        r = await slow_generation(client, 0.05)
    print(f"      completed: {r}")

    # A deadline (absolute) rather than a delay (relative) is what you want
    # when a budget must be shared across several sequential steps.
    print("\n  C) absolute deadline shared across a multi-step pipeline")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.30
    try:
        async with asyncio.timeout_at(deadline):
            await slow_generation(client, 0.10)   # step 1: retrieval
            print("      step 1 done, budget partially consumed")
            await slow_generation(client, 0.10)   # step 2: rerank
            print("      step 2 done")
            await slow_generation(client, 0.50)   # step 3: generation — too slow
    except TimeoutError:
        print("      overall budget blown at step 3 — this is the right shape:")
        print("      ONE end-to-end budget, not three independent timeouts that")
        print("      can sum to 3x your SLA.")


# ---------------------------------------------------------------------------
# PART 2 — layered timeouts
# ---------------------------------------------------------------------------

async def part2_layered() -> None:
    banner("PART 2 — the four timeouts you actually need")

    print("""
  A single `timeout=30` on your HTTP client is not a timeout strategy. httpx
  distinguishes four, and each catches a different failure:

    connect  (~5s)   TCP + TLS handshake. Catches: DNS failure, dead endpoint,
                     NSG/firewall blackhole. Should be SHORT — a healthy
                     connect is <100ms; 5s means something is wrong.
    read     (~60s)  Gap between bytes. For a STREAMING response this is the
                     inter-token gap, not the total — so it can be short.
                     For non-streaming it must exceed worst-case generation.
    write    (~10s)  Sending the request body. Matters when you upload a
                     large document for extraction.
    pool     (~5s)   Waiting for a free connection from the pool. Firing here
                     means your concurrency exceeds your pool size — the fix
                     is a semaphore, not a bigger timeout.

  Then, ABOVE all of those, one end-to-end deadline per user request
  (asyncio.timeout_at) that bounds retrieval + rerank + generation + retries
  together. Without it, three layers each "correctly" timing out at 60s give
  you a 180s user-visible hang.

  GOTCHA specific to streaming: a total-response timeout is wrong. A 3-minute
  generation is legitimate. What you actually want to detect is a STALLED
  stream — no token for 15s. That is a read timeout, and it is the only one
  that expresses the real requirement.""")

    # Demonstrate the stalled-stream detector, since it has no library form.
    async def token_stream(gap: float, n: int):
        for i in range(n):
            await asyncio.sleep(gap)
            yield f"tok{i} "

    async def consume_with_stall_detection(stream, per_token_timeout: float) -> str:
        """Bound the GAP between tokens, not the total duration."""
        out = []
        it = stream.__aiter__()
        while True:
            try:
                async with asyncio.timeout(per_token_timeout):
                    tok = await it.__anext__()
            except StopAsyncIteration:
                return "".join(out)
            except TimeoutError:
                out.append("[STALLED]")
                return "".join(out)
            out.append(tok)

    print("\n  healthy stream (0.02s gaps, 0.10s tolerance):")
    r = await consume_with_stall_detection(token_stream(0.02, 5), 0.10)
    print(f"      {r}")

    print("  stalled stream (0.30s gaps, 0.10s tolerance):")
    r = await consume_with_stall_detection(token_stream(0.30, 5), 0.10)
    print(f"      {r}")


# ---------------------------------------------------------------------------
# PART 3 — cleanup that must survive cancellation
# ---------------------------------------------------------------------------

async def part3_cleanup() -> None:
    banner("PART 3 — cleanup, finally, and shield")

    # This section reports MEASURED behaviour. The widely-repeated claim that
    # "await in a finally block gets cancelled" is imprecise, and believing the
    # imprecise version leads you to shield things that do not need it.
    #
    # What is actually true: cancellation is delivered ONCE. After
    # CancelledError has been raised at your await, the task is running
    # normally again. A fresh await inside `finally` therefore RUNS TO
    # COMPLETION — unless a SECOND cancellation arrives.
    #
    # The catch: a second cancellation is not exotic. It is precisely what a
    # graceful-shutdown drain does (cancel, wait out the grace period,
    # hard-cancel the stragglers — see PART 5). So cleanup survives an
    # ordinary request timeout and dies during a deployment, which is the
    # worst possible distribution of failures: it works in every test you run
    # and loses audit records only during a rollout.

    audit: list[str] = []

    async def write_audit(trace_id: str) -> None:
        await asyncio.sleep(0.05)  # a real network write to a log sink
        audit.append(trace_id)

    async def handler_naive(trace_id: str) -> str:
        try:
            await asyncio.sleep(1.0)
            return "done"
        finally:
            await write_audit(f"naive:{trace_id}")

    print("\n  A) ONE cancellation (an ordinary timeout) — cleanup SURVIVES")
    audit.clear()
    try:
        async with asyncio.timeout(0.1):
            await handler_naive("t1")
    except (TimeoutError, asyncio.CancelledError):
        pass
    print(f"      audit: {audit}")
    print("      <-- the record was written. Unshielded cleanup is fine here.")

    print("\n  B) TWO cancellations (a shutdown drain) — cleanup is DESTROYED")
    audit.clear()
    t = asyncio.create_task(handler_naive("t2"))
    await asyncio.sleep(0.05)
    t.cancel()               # drain: ask nicely
    await asyncio.sleep(0)   # task enters its finally block
    t.cancel()               # grace expired: hard cancel — lands on the cleanup await
    try:
        await t
    except asyncio.CancelledError:
        pass
    print(f"      audit: {audit}")
    print("      <-- empty. The second cancel hit the await inside `finally`.")

    print("\n  C) caller does not await long enough — cleanup is DESTROYED")
    audit.clear()
    t = asyncio.create_task(handler_naive("t3"))
    await asyncio.sleep(0.05)
    t.cancel()
    try:
        # wait_for re-cancels on ITS timeout, so this both abandons and kills
        # the cleanup. Giving cancellation less time than cleanup needs is the
        # same bug as B, wearing different clothes.
        await asyncio.wait_for(t, timeout=0.01)
    except (TimeoutError, asyncio.CancelledError):
        pass
    await asyncio.sleep(0.2)  # wait well past when cleanup would have finished
    print(f"      audit after waiting 0.2s more: {audit}")
    print("      <-- still empty. It did not finish later; it was killed.")

    # --- The fix, for cleanup that must survive case B and C. ---
    async def handler_shielded(trace_id: str) -> str:
        try:
            await asyncio.sleep(1.0)
            return "done"
        finally:
            # `shield` protects the inner awaitable from cancellation directed
            # at US. It must have its OWN timeout, or a hung log sink turns a
            # prompt shutdown into an indefinite hang — a worse bug than the
            # one you fixed.
            try:
                async with asyncio.timeout(0.5):
                    await asyncio.shield(write_audit(f"shielded:{trace_id}"))
            except TimeoutError:
                pass

    print("\n  D) shielded cleanup survives the double-cancel case")
    audit.clear()
    t = asyncio.create_task(handler_shielded("t4"))
    await asyncio.sleep(0.05)
    t.cancel()
    await asyncio.sleep(0)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.1)  # let the shielded write land
    print(f"      audit: {audit}")

    print("""
  THE ACTUAL RULE (narrower than the folklore version):
    * Unshielded cleanup is fine for ordinary timeouts and single cancels.
    * Shield cleanup that must survive a SHUTDOWN DRAIN — audit rows, lease
      release, quota decrements, span closure.
    * Every shield needs its own bounded timeout.
    * Never shield the main workload; that is a timeout you disabled.
    * Test cleanup with a DOUBLE cancel. A single-cancel test passes even when
      the code is broken for the case you care about.""")


# ---------------------------------------------------------------------------
# PART 4 — uncancellable tasks, and how to not write one
# ---------------------------------------------------------------------------

async def part4_bad_citizen() -> None:
    banner("PART 4 — the task that refuses to die")

    # NOTE: a *truly* immortal task cannot be demonstrated in a script that
    # terminates — `asyncio.run()` cancels leftover tasks and awaits them at
    # exit, so an immortal task hangs the interpreter on shutdown forever.
    # That hang IS the production symptom: a pod that never completes its
    # termination grace period and gets SIGKILLed by the orchestrator, losing
    # all in-flight work. Here the task relents after 2 swallows so we finish.

    swallowed = 0

    async def bad_citizen() -> None:
        nonlocal swallowed
        while True:
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                # THE BUG. Catching and continuing makes the task immortal.
                # Seen in the wild as bare `except:` or `except BaseException:`
                # in a poller loop. (`except Exception:` is safe here, since
                # CancelledError derives from BaseException — that was the
                # whole point of the 3.8 change.)
                swallowed += 1
                print(f"      (bad citizen swallowed cancellation #{swallowed})")
                if swallowed >= 2:
                    print("      (relenting, so this demo can actually exit)")
                    raise
                continue

    t = asyncio.create_task(bad_citizen(), name="bad-citizen")
    await asyncio.sleep(0.06)

    t.cancel()
    try:
        async with asyncio.timeout(0.3):
            await t
    except TimeoutError:
        print("      still alive 0.3s after cancel() — a real shutdown would hang here")

    # `cancelling()` / `uncancel()` (3.11+) let a task legitimately absorb ONE
    # cancellation; this is how TaskGroup and asyncio.timeout coordinate
    # internally. You rarely need them directly — know they exist so the
    # counts in a traceback make sense.
    print(f"      t.cancelling() = {t.cancelling()}")

    t.cancel()
    try:
        await asyncio.wait_for(t, timeout=0.5)
    except (TimeoutError, asyncio.CancelledError):
        pass
    print(f"      task done={t.done()} cancelled={t.cancelled()}")

    print("""
  REVIEW RULE: in any `except asyncio.CancelledError` block, the last
  statement must be `raise`. If it is not, that is a defect. The only
  legitimate exception is a supervisor implementing uncancel() semantics,
  which should be rare enough to require a comment explaining itself.""")


# ---------------------------------------------------------------------------
# PART 5 — graceful shutdown
# ---------------------------------------------------------------------------

async def part5_shutdown() -> None:
    banner("PART 5 — graceful shutdown / drain")

    # On SIGTERM (K8s pod eviction, App Service restart, deployment rollout)
    # you get a grace period — often 30s — to finish in-flight work. The
    # correct behaviour: stop accepting new work, let existing work finish
    # within the budget, then hard-cancel the remainder.

    stop = asyncio.Event()
    completed, abandoned = [], []

    async def worker(wid: int, duration: float) -> None:
        try:
            await asyncio.sleep(duration)
            completed.append(wid)
        except asyncio.CancelledError:
            abandoned.append(wid)
            raise

    tasks = [
        asyncio.create_task(worker(0, 0.05), name="fast"),
        asyncio.create_task(worker(1, 0.10), name="medium"),
        asyncio.create_task(worker(2, 5.00), name="stuck"),
    ]

    # Simulate SIGTERM arriving. In real code:
    #   loop.add_signal_handler(signal.SIGTERM, stop.set)
    asyncio.get_running_loop().call_later(0.01, stop.set)
    await stop.wait()
    print("      SIGTERM received — draining with a 0.30s grace period")

    done, pending = await asyncio.wait(tasks, timeout=0.30)
    print(f"      finished within grace: {len(done)}")

    for t in pending:
        t.cancel()
    # Reap them so nothing is destroyed-while-pending at interpreter exit.
    await asyncio.gather(*pending, return_exceptions=True)

    print(f"      completed={completed} abandoned={abandoned}")
    print("""
      The abandoned work must be RECOVERABLE. That means: idempotent
      operations, a durable queue that redelivers unacked messages, or a
      checkpoint written before the risky step. "It'll be cancelled cleanly"
      is only half a design — the other half is what happens to that work.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_timeout()
    await part2_layered()
    await part3_cleanup()
    await part4_bad_citizen()
    await part5_shutdown()

    banner("SUMMARY")
    print("""
  * Cancellation = CancelledError raised at the current await. Not a kill.
  * Always re-raise CancelledError. `except Exception` is safe; bare
    `except:` and `except BaseException:` are not.
  * `task.cancel()` then `await task` — cancel is a request, await is the
    confirmation. Skipping the await leaks half-finished work.
  * Prefer `asyncio.timeout` / `timeout_at` over `wait_for`; they compose over
    blocks and support shared deadlines.
  * Four transport timeouts + ONE end-to-end deadline per user request.
  * For streams, bound the inter-token gap, not the total duration.
  * `shield` only short, bounded, must-complete cleanup — with its own timeout.
  * Design what happens to abandoned work, not just how it gets abandoned.
""")


if __name__ == "__main__":
    asyncio.run(main())
