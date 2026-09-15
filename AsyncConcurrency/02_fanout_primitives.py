"""
02 — Fan-out primitives: gather, TaskGroup, as_completed, wait.

THE DECISION YOU ARE ACTUALLY MAKING
------------------------------------
All four run N coroutines concurrently. They differ almost entirely in
**what happens when one of them fails**, and secondarily in **when you get
results**. Choosing wrongly produces one of two production defects:

  (a) One embedding failure kills a 500-document ingestion run that should
      have quarantined the bad doc and continued.
  (b) One failure is silently swallowed into a results list nobody inspects,
      and you ship an index with 12 missing documents that nobody notices for
      six weeks.

Quick reference:

  gather(..., return_exceptions=False)  first exception propagates; siblings
                                        are cancelled but NOT awaited => you
                                        can leak work. Avoid in new code.
  gather(..., return_exceptions=True)   never raises; exceptions arrive as
                                        list elements. Order preserved. Good
                                        for per-item partial failure.
  TaskGroup (3.11+)                     structured concurrency. On failure:
                                        cancels siblings, WAITS for them, then
                                        raises ExceptionGroup. Default choice
                                        for all-or-nothing work.
  as_completed                          results in completion order. Use for
                                        streaming progress / early exit.
  wait(...)                             lowest level; returns (done, pending)
                                        and does NOT raise. Use for timeouts
                                        with FIRST_COMPLETED semantics.

Run:  python 02_fanout_primitives.py
"""

from __future__ import annotations

import asyncio

from fake_llm import (
    ContentFilterError,
    FakeLLMClient,
    Timer,
    banner,
)

# ---------------------------------------------------------------------------
# A worker that fails predictably, so we can compare error semantics
# ---------------------------------------------------------------------------

async def process_doc(doc_id: int, *, fail_on: set[int], delay: float = 0.10) -> str:
    """Simulates 'extract fields from one supply-chain document'.

    Deliberately fails for doc_ids in `fail_on`, after `delay` seconds — i.e.
    partway through the batch, which is the realistic case. A failure at t=0
    would hide the sibling-cancellation behaviour we want to observe.
    """
    await asyncio.sleep(delay)
    if doc_id in fail_on:
        raise ContentFilterError(f"doc {doc_id} tripped the content filter")
    await asyncio.sleep(delay)  # second phase — reachable only if not failed
    return f"extracted(doc={doc_id})"


# ---------------------------------------------------------------------------
# PART 1 — gather, both modes
# ---------------------------------------------------------------------------

async def part1_gather() -> None:
    banner("PART 1 — asyncio.gather")

    # --- Mode A: fail-fast (the default). --------------------------------
    # The first exception propagates immediately. Siblings are *requested* to
    # cancel, but gather does not wait for them to finish unwinding. If a
    # sibling has a `finally:` block that awaits (closing a file, releasing a
    # DB row, writing an audit record), you have now left the scope while that
    # cleanup is still running. That is the core defect TaskGroup fixes.
    print("\n  A) return_exceptions=False (default, fail-fast)")
    try:
        await asyncio.gather(*(process_doc(i, fail_on={2}) for i in range(5)))
    except ContentFilterError as e:
        print(f"     raised immediately: {e}")
        print("     siblings were cancelled but NOT awaited — cleanup may still be running")

    # --- Mode B: collect everything. -------------------------------------
    # Nothing raises. Exceptions are returned *positionally* in the results
    # list. Order matches input order, which is why this is the right tool
    # when you need to correlate result[i] back to input[i].
    print("\n  B) return_exceptions=True (collect partial failures)")
    results = await asyncio.gather(
        *(process_doc(i, fail_on={1, 3}) for i in range(5)),
        return_exceptions=True,
    )

    # GOTCHA: this is where teams lose data. `results` contains exceptions but
    # nothing forces you to look. You MUST partition explicitly. A code review
    # rule worth enforcing: every `return_exceptions=True` call site is
    # followed by an isinstance partition within a few lines.
    ok = [r for r in results if not isinstance(r, BaseException)]
    bad = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
    print(f"     succeeded: {len(ok)}  failed: {len(bad)}")
    for idx, err in bad:
        print(f"       doc {idx}: {type(err).__name__}: {err}")


# ---------------------------------------------------------------------------
# PART 2 — TaskGroup: structured concurrency
# ---------------------------------------------------------------------------

async def part2_taskgroup() -> None:
    banner("PART 2 — asyncio.TaskGroup (Python 3.11+) — the default choice")

    # The guarantee: when the `async with` block exits — normally, by
    # exception, or by cancellation — every task created inside it is
    # finished. No orphans, no leaked work, ever. This is "structured
    # concurrency": task lifetime is bounded by lexical scope, the same way
    # `with open(...)` bounds a file handle's lifetime.

    print("\n  A) happy path")
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(process_doc(i, fail_on=set())) for i in range(4)]
    # <-- all tasks guaranteed complete at this line
    print(f"     results: {[t.result() for t in tasks]}")

    print("\n  B) one failure — siblings cancelled AND awaited")
    cleanup_log: list[str] = []

    async def doc_with_cleanup(doc_id: int) -> str:
        try:
            return await process_doc(doc_id, fail_on={2})
        except asyncio.CancelledError:
            # Real cleanup that itself awaits. Under gather() this may not
            # finish before the caller moves on. Under TaskGroup it always does.
            await asyncio.sleep(0.01)
            cleanup_log.append(f"doc{doc_id} released its lock")
            raise

    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(5):
                tg.create_task(doc_with_cleanup(i))
    except* ContentFilterError as eg:
        # NOTE the `except*` syntax — TaskGroup raises an ExceptionGroup
        # because *multiple* children can fail simultaneously. `except*`
        # matches by type inside the group and gives you a sub-group.
        # Plain `except ContentFilterError` will NOT catch an ExceptionGroup.
        # This is the #1 surprise when migrating gather -> TaskGroup.
        print(f"     ExceptionGroup with {len(eg.exceptions)} matching error(s)")
        for e in eg.exceptions:
            print(f"       {type(e).__name__}: {e}")
    print(f"     cleanup completed before we got here: {cleanup_log}")

    print("\n  C) multiple simultaneous failures collapse into one group")
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(6):
                tg.create_task(process_doc(i, fail_on={1, 2, 3}))
    except* ContentFilterError as eg:
        print(f"     caught {len(eg.exceptions)} errors together — gather() would")
        print("     have shown you only the first, discarding the rest")


# ---------------------------------------------------------------------------
# PART 3 — as_completed: results in finish order
# ---------------------------------------------------------------------------

async def part3_as_completed() -> None:
    banner("PART 3 — as_completed: stream results as they land")

    client = FakeLLMClient(seed=7, base_latency_s=0.05, jitter_s=0.25)

    # Use this when you want to show progress, write results to a sink
    # incrementally, or stop early once you have enough. The cost: you lose
    # the input->output positional correspondence.
    #
    # GOTCHA: the yielded awaitables are NOT the tasks you passed in, so you
    # cannot recover "which input was this?" from them. The fix is to make the
    # coroutine return its own identifier, as below. Forgetting this and then
    # trying to zip results back to inputs by position is a real bug.

    async def labelled(doc_id: int) -> tuple[int, str]:
        c = await client.complete(f"summarise doc {doc_id}")
        return doc_id, c.text[:40]

    coros = [labelled(i) for i in range(6)]
    print("  completion order (note: NOT submission order):")
    with Timer("as_completed total"):
        for fut in asyncio.as_completed(coros):
            doc_id, _text = await fut
            print(f"     doc {doc_id} finished")

    # Early exit: stop as soon as one retriever returns something usable.
    print("\n  early-exit pattern (first acceptable result wins):")

    async def search(source: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"hit from {source}"

    coros2 = [
        search("sharepoint", 0.30),
        search("cache", 0.05),
        search("blob-index", 0.20),
    ]
    for fut in asyncio.as_completed(coros2):
        first = await fut
        print(f"     took: {first}")
        break
    # GOTCHA: breaking out leaves the other coroutines pending. In real code,
    # wrap this in a TaskGroup or explicitly cancel the rest — otherwise you
    # get "Task was destroyed but it is pending!" at shutdown. See PART 4.


# ---------------------------------------------------------------------------
# PART 4 — asyncio.wait: the low-level primitive, done right
# ---------------------------------------------------------------------------

async def part4_wait() -> None:
    banner("PART 4 — asyncio.wait + FIRST_COMPLETED, with correct cleanup")

    async def search(source: str, delay: float) -> str:
        try:
            await asyncio.sleep(delay)
            return f"hit from {source}"
        except asyncio.CancelledError:
            print(f"     {source} was cancelled cleanly")
            raise

    # asyncio.wait takes TASKS (not bare coroutines, since 3.11) and never
    # raises on child failure. It hands back two sets. You decide what to do.
    tasks = {
        asyncio.create_task(search("sharepoint", 0.30), name="sharepoint"),
        asyncio.create_task(search("cache", 0.05), name="cache"),
        asyncio.create_task(search("blob-index", 0.20), name="blob-index"),
    }
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    winner = done.pop()
    print(f"     winner: {winner.result()}")

    # THE PART PEOPLE SKIP: cancel the losers AND await them. Cancellation is
    # a *request*; the task is not finished until you await it and it unwinds.
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    print("     all losers cancelled and reaped — no pending-task warnings")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_gather()
    await part2_taskgroup()
    await part3_as_completed()
    await part4_wait()

    banner("DECISION TABLE")
    print("""
  Need all N to succeed, want clean shutdown on failure
      -> TaskGroup + except*                  (default; use this)

  Need per-item partial failure with input correspondence preserved
      -> gather(return_exceptions=True) + explicit isinstance partition

  Need progress / incremental writes / early exit
      -> as_completed, with self-identifying return values
         (and a TaskGroup around it if you break early)

  Need FIRST_COMPLETED or custom timeout logic
      -> asyncio.wait, then cancel AND await the pending set

  Legacy gather(return_exceptions=False)
      -> avoid: it abandons siblings mid-cleanup
""")


if __name__ == "__main__":
    asyncio.run(main())
