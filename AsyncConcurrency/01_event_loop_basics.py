"""
01 — Event loop mechanics: coroutines, tasks, and what "concurrent" means.

THE ONE-SENTENCE MODEL
----------------------
There is a single thread. It runs a loop. The loop keeps a queue of "ready"
callbacks. A coroutine runs on that thread until it hits an `await` on
something not yet finished, at which point it *suspends and returns control to
the loop*, which picks the next ready thing. When the awaited thing completes,
the coroutine is put back on the ready queue.

Consequences you must internalise:
  * async gives you CONCURRENCY (interleaving), not PARALLELISM (simultaneous
    CPU execution). One thread. One core. Always.
  * It only helps when your code spends its time *waiting* — network, disk,
    subprocess. LLM calls are ~99.9% waiting, which is why this fits so well.
  * Any code between two awaits runs to completion with zero interruption.
    That is a gift (no data races on plain attributes) and a curse (one slow
    synchronous line freezes everything — see 03).

Run:  python 01_event_loop_basics.py
"""

from __future__ import annotations

import asyncio

from fake_llm import FakeLLMClient, Timer, banner

# ---------------------------------------------------------------------------
# PART 1 — A coroutine object is not a running thing
# ---------------------------------------------------------------------------

async def greet(name: str) -> str:
    """Calling `greet("x")` does NOT run this body. It builds a coroutine
    object. Nothing happens until something awaits it or wraps it in a Task."""
    await asyncio.sleep(0.05)
    return f"hello {name}"


async def part1_coroutine_vs_task() -> None:
    banner("PART 1 — coroutine objects are lazy; Tasks are scheduled")

    coro = greet("debdeep")
    print(f"  calling greet() returned: {type(coro).__name__} — nothing ran yet")

    # Awaiting it runs it, on the current task, to completion.
    result = await coro
    print(f"  after await: {result!r}")

    # asyncio.create_task() hands the coroutine to the loop to run
    # *concurrently with the current task*. This is the primitive that makes
    # overlap possible. `await` alone never creates concurrency.
    task = asyncio.create_task(greet("justin"))
    print(f"  create_task returned: {type(task).__name__}, done={task.done()}")

    # Yield control once so the loop gets a chance to start the task.
    # asyncio.sleep(0) is the idiomatic "let other tasks run" checkpoint.
    await asyncio.sleep(0)
    print(f"  after sleep(0): task started but not finished, done={task.done()}")

    print(f"  awaiting task: {await task!r}")

    # GOTCHA: a coroutine you never await produces a RuntimeWarning and does
    # nothing. In a large codebase a forgotten `await` on an LLM call is a
    # silent no-op that returns a coroutine object where you expected text.
    # `mypy` catches this; runtime often does not until something downstream
    # tries to use the "string".
    orphan = greet("nobody")
    print(f"  orphan coroutine (never awaited): {orphan!r}")
    orphan.close()  # explicit cleanup so we don't emit a warning


# ---------------------------------------------------------------------------
# PART 2 — sequential vs concurrent, measured
# ---------------------------------------------------------------------------

async def part2_sequential_vs_concurrent() -> None:
    banner("PART 2 — the actual point: overlapping I/O waits")

    prompts = [f"classify invoice exception #{i}" for i in range(8)]

    # -- Sequential: each await blocks THIS task until it resolves. Total time
    #    is the SUM of latencies. This is what a naive port of sync code does.
    client = FakeLLMClient(seed=1, base_latency_s=0.10, jitter_s=0.0)
    with Timer("sequential (8 calls, 0.10s each)") as t_seq:
        for p in prompts:
            await client.complete(p)

    # -- Concurrent: all 8 coroutines are handed to the loop at once. Each one
    #    suspends at its sleep; the loop services the others meanwhile. Total
    #    time is roughly the MAX of latencies, not the sum.
    client2 = FakeLLMClient(seed=1, base_latency_s=0.10, jitter_s=0.0)
    with Timer("concurrent  (8 calls, 0.10s each)") as t_con:
        await asyncio.gather(*(client2.complete(p) for p in prompts))

    print(f"\n  speedup: {t_seq.elapsed / t_con.elapsed:.1f}x")
    print("  NOTE: nothing ran in parallel. One thread. The wall-clock win is")
    print("  purely from overlapping *waiting*, which is 99% of an LLM call.")


# ---------------------------------------------------------------------------
# PART 3 — the loop is cooperative; nothing preempts you
# ---------------------------------------------------------------------------

async def noisy_worker(worker_id: int, log: list[str]) -> None:
    """Three units of work with awaits between them. Watch the interleaving in
    the log: no worker gets to finish before others start, because each `await`
    is a yield point where the loop reschedules."""
    for step in range(3):
        log.append(f"w{worker_id}:s{step}")
        await asyncio.sleep(0.01)


async def part3_interleaving() -> None:
    banner("PART 3 — awaits are the ONLY yield points (cooperative scheduling)")

    log: list[str] = []
    await asyncio.gather(*(noisy_worker(i, log) for i in range(3)))
    print(f"  interleaved execution order: {' '.join(log)}")
    print("  Each worker ran a step, hit `await`, and yielded to the next.")

    # Now the same thing with NO awaits inside the loop body. It runs to
    # completion atomically because there is no yield point.
    log2: list[str] = []

    async def greedy(worker_id: int) -> None:
        for step in range(3):
            log2.append(f"w{worker_id}:s{step}")
            # no await here — nothing can interleave

    await asyncio.gather(*(greedy(i) for i in range(3)))
    print(f"  no-await execution order:    {' '.join(log2)}")
    print("  REMEMBER: 'atomic between awaits' is why you rarely need locks")
    print("  for simple counter/dict updates — but you DO need them whenever a")
    print("  read-modify-write spans an await. See 09_pitfalls.py.")


# ---------------------------------------------------------------------------
# PART 4 — inspecting the loop
# ---------------------------------------------------------------------------

async def part4_introspection() -> None:
    banner("PART 4 — introspection: what is the loop actually holding?")

    client = FakeLLMClient(seed=3, base_latency_s=0.15, jitter_s=0.0)

    # Naming tasks is free and pays for itself the first time you debug a
    # production hang. `asyncio.all_tasks()` shows you what is outstanding.
    tasks = [
        asyncio.create_task(client.complete(f"doc {i}"), name=f"extract-doc-{i}")
        for i in range(4)
    ]
    await asyncio.sleep(0.01)  # let them start

    pending = asyncio.all_tasks()
    print(f"  tasks currently alive: {len(pending)}")
    for t in sorted(pending, key=lambda x: x.get_name()):
        print(f"    - {t.get_name():<20} done={t.done()}")

    await asyncio.gather(*tasks)
    print(f"  client stats: {client.stats()}")

    # asyncio.current_task() tells you where you are — useful for injecting a
    # correlation ID into logs via contextvars (see 10_rag_capstone.py).
    print(f"  current task name: {asyncio.current_task().get_name()}")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_coroutine_vs_task()
    await part2_sequential_vs_concurrent()
    await part3_interleaving()
    await part4_introspection()

    banner("SUMMARY")
    print("""
  * A coroutine object is inert until awaited or wrapped in a Task.
  * `await` on a coroutine = run it here, now, sequentially.
  * `create_task` / `gather` / `TaskGroup` = hand it to the loop for overlap.
  * Concurrency comes from having multiple Tasks, never from `await` alone.
  * One thread. Overlap of *waiting*, not of computation.
  * Code between awaits is uninterruptible — both a safety property and the
    mechanism by which one blocking call freezes the entire service.
""")


if __name__ == "__main__":
    # asyncio.run() creates a fresh loop, runs main to completion, then
    # cancels remaining tasks, runs async generator shutdown hooks, and closes
    # the loop. Use it once, at the top level. Never call it from inside a
    # coroutine, and never use the deprecated get_event_loop()/run_until_complete
    # pattern in new code.
    asyncio.run(main())
