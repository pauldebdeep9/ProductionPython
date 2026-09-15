"""
09 — The pitfalls: async bugs that pass code review.

Each section is a bug you will see in a PR, why it survives review, and the
fix. These are ordered roughly by how often they cost real production time.

Run:  python 09_pitfalls.py
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import time

from fake_llm import FakeLLMClient, banner

# ---------------------------------------------------------------------------
# PITFALL 1 — fire-and-forget tasks get garbage collected
# ---------------------------------------------------------------------------

async def pitfall1_task_gc() -> None:
    banner("PITFALL 1 — create_task() without keeping a reference")

    # The event loop keeps only a WEAK reference to tasks. If you do not hold
    # a strong reference, the task can be collected mid-execution and simply
    # vanish. Silently. This is documented but almost nobody knows it.
    #
    # It survives review because the code looks obviously correct, and it
    # usually works in dev where GC pressure is low. Under load it drops a
    # fraction of your background writes.

    completed: list[int] = []

    async def background_audit_write(i: int) -> None:
        await asyncio.sleep(0.05)
        completed.append(i)

    print("\n  A) no reference held, with aggressive GC")
    for i in range(20):
        asyncio.create_task(background_audit_write(i))  # reference dropped
    gc.collect()  # simulate memory pressure
    await asyncio.sleep(0.2)
    print(f"      completed: {len(completed)}/20")
    if len(completed) == 20:
        print("""
      MEASURED: all 20 survived — the failure did NOT reproduce here, and you
      should know why rather than trusting the folklore.

      The loop holds only a WEAK reference to a Task. But in this example each
      task is parked in `asyncio.sleep`, which registers a TimerHandle that the
      loop holds STRONGLY, and that handle references the task's wakeup
      callback. So the task is transitively reachable and cannot be collected.

      The hazard is real and documented (see the asyncio.create_task docs), but
      it needs a task suspended on a future that nothing else keeps alive — a
      custom future, a third-party awaitable, some C-extension integrations.
      That is rare, non-deterministic, and load-dependent, which is exactly
      what makes it nasty: it will not show up in your tests.

      So: keep the reference. Not because this demo proved you must, but
      because the cost is three lines and the failure mode is a silently
      dropped audit write that you will never trace back.""")

    # THE FIX: hold strong references in a set, and discard on completion so
    # the set does not grow without bound.
    print("\n  B) strong references retained")
    completed.clear()
    background_tasks: set[asyncio.Task] = set()
    for i in range(20):
        t = asyncio.create_task(background_audit_write(i))
        background_tasks.add(t)
        t.add_done_callback(background_tasks.discard)
    gc.collect()
    await asyncio.sleep(0.2)
    print(f"      completed: {len(completed)}/20")

    print("""
      BETTER FIX: do not fire-and-forget at all. Use a TaskGroup, or a
      long-lived worker consuming a queue. Every fire-and-forget task is an
      unowned lifetime — nothing knows if it succeeded, nothing waits for it
      at shutdown, and its exceptions surface (if at all) as
      "Task exception was never retrieved" in a log nobody reads.""")


# ---------------------------------------------------------------------------
# PITFALL 2 — swallowed exceptions in detached tasks
# ---------------------------------------------------------------------------

async def pitfall2_lost_exceptions() -> None:
    banner("PITFALL 2 — exceptions in tasks nobody awaits")

    async def will_fail() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("index write failed — nobody will hear this")

    t = asyncio.create_task(will_fail())
    await asyncio.sleep(0.05)
    print(f"      task done={t.done()}, exception={t.exception()!r}")
    print("      The exception is stored on the Task. If you never call")
    print("      .exception() or .result(), it is reported only at GC time as")
    print("      'Task exception was never retrieved' — often after the")
    print("      process has moved on, with no request context attached.")

    # THE FIX for genuinely detached work: an explicit done-callback that logs
    # with your structured logger, so failures are never silent.
    def report(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"      [handler] background task failed: {type(exc).__name__}: {exc}")

    t2 = asyncio.create_task(will_fail())
    t2.add_done_callback(report)
    await asyncio.sleep(0.05)

    # Set a global handler as a backstop for everything you missed.
    def loop_exception_handler(loop, context):
        print(f"      [loop handler] {context.get('message')}")

    asyncio.get_running_loop().set_exception_handler(loop_exception_handler)
    print("""
      Install loop.set_exception_handler() at startup and route it to your
      structured logger. It is the only place some failures ever appear.""")


# ---------------------------------------------------------------------------
# PITFALL 3 — read-modify-write across an await
# ---------------------------------------------------------------------------

async def pitfall3_race() -> None:
    banner("PITFALL 3 — 'async is single-threaded so I don't need locks'")

    # TRUE:  code between two awaits is atomic; `counter += 1` is safe.
    # FALSE: therefore all shared-state updates are safe.
    #
    # Any read-modify-write that SPANS an await is a race, because the loop
    # can run another task in the gap. This is the async equivalent of a
    # check-then-act TOCTOU bug and it is very easy to write by accident.

    class TokenBudget:
        """Deducts from a shared budget, checking availability first."""

        def __init__(self, total: int) -> None:
            self.remaining = total
            self.overdrafts = 0

        async def spend_racy(self, n: int) -> bool:
            if self.remaining < n:            # CHECK
                return False
            await asyncio.sleep(0.001)        # <-- yield: another task runs here
            self.remaining -= n               # ACT (on a stale read)
            if self.remaining < 0:
                self.overdrafts += 1
            return True

        async def spend_safe(self, n: int, lock: asyncio.Lock) -> bool:
            async with lock:                  # check and act are now atomic
                if self.remaining < n:
                    return False
                await asyncio.sleep(0.001)
                self.remaining -= n
                if self.remaining < 0:
                    self.overdrafts += 1
                return True

    print("\n  A) racy: 50 tasks spending from a budget of 100")
    b = TokenBudget(100)
    await asyncio.gather(*(b.spend_racy(10) for _ in range(50)))
    print(f"      remaining={b.remaining}  overdrafts={b.overdrafts}")
    print("      <-- budget went NEGATIVE. Every task read 'plenty left' before")
    print("      any of them deducted. In a TPM limiter this is a 429 storm;")
    print("      in a spend cap it is a bill.")

    print("\n  B) with a lock")
    b2 = TokenBudget(100)
    lock = asyncio.Lock()
    await asyncio.gather(*(b2.spend_safe(10, lock) for _ in range(50)))
    print(f"      remaining={b2.remaining}  overdrafts={b2.overdrafts}")

    print("""
      REVIEW HEURISTIC: find every `await` inside a method that mutates shared
      state. If a read before it informs a write after it, you need a lock.
      Grep-able smells: `if self.` ... `await` ... `self.x =` in one method.""")


# ---------------------------------------------------------------------------
# PITFALL 4 — contextvars and trace propagation
# ---------------------------------------------------------------------------

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


async def pitfall4_contextvars() -> None:
    banner("PITFALL 4 — request context across tasks (contextvars)")

    async def leaf_operation(label: str) -> str:
        # No trace_id parameter threaded through five call layers — the
        # context variable carries it. This is how you get correlation IDs
        # into logs without polluting every signature.
        return f"{label} ran under trace_id={trace_id_var.get()}"

    async def handle_request(tid: str) -> list[str]:
        trace_id_var.set(tid)
        # KEY BEHAVIOUR: create_task COPIES the current context. Each child
        # sees the trace_id set by its parent at creation time.
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(leaf_operation("retrieval"))
            t2 = tg.create_task(leaf_operation("generation"))
        return [t1.result(), t2.result()]

    results = await asyncio.gather(handle_request("tr-AAA"), handle_request("tr-BBB"))
    for group in results:
        for line in group:
            print(f"      {line}")
    print("      Two concurrent requests, no cross-contamination.")

    print("""
      GOTCHA: the copy is one-way. A `set()` inside a child task does NOT
      propagate back to the parent, and does not reach sibling tasks. If you
      need a child to publish something upward, return it — do not try to
      mutate a contextvar and read it in the parent.

      GOTCHA 2: contextvars do NOT cross `to_thread` cleanly in older
      versions, and never cross a process boundary. If you push work to an
      executor, pass the trace_id explicitly.""")


# ---------------------------------------------------------------------------
# PITFALL 5 — creating the client per request
# ---------------------------------------------------------------------------

async def pitfall5_client_lifecycle() -> None:
    banner("PITFALL 5 — constructing an HTTP client inside the request path")

    print("""
      THE BUG:

          async def handler(req):
              async with httpx.AsyncClient() as c:      # new pool every request
                  return await c.post(AZURE_OPENAI_URL, json=...)

      Every request builds a fresh connection pool, does a fresh TCP and TLS
      handshake, and throws it away. You lose connection reuse entirely,
      adding one or two RTTs plus a TLS negotiation (~100ms+ to a regional
      endpoint) to every single call. Under load you also churn ephemeral
      ports and can exhaust them — the classic SNAT port exhaustion failure on
      Azure App Service, which presents as random connection timeouts that
      look like the model is slow.

      THE FIX: one client for the process lifetime, created at startup and
      closed at shutdown.

          @asynccontextmanager
          async def lifespan(app):
              app.state.http = httpx.AsyncClient(
                  timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5),
                  limits=httpx.Limits(max_connections=100,
                                      max_keepalive_connections=20),
              )
              app.state.aoai = AsyncAzureOpenAI(http_client=app.state.http, ...)
              yield
              await app.state.http.aclose()

          app = FastAPI(lifespan=lifespan)

      Same rule for AsyncAzureOpenAI, Search clients, Cosmos clients, and
      credential objects. DefaultAzureCredential in particular caches tokens
      internally — constructing it per request defeats that cache and can
      trigger throttling on the IMDS endpoint.""")

    # Demonstrate the cost difference in relative terms.
    setup_cost = 0.05

    async def per_request_client() -> None:
        await asyncio.sleep(setup_cost)   # handshake
        await asyncio.sleep(0.01)         # actual call

    async def shared_client() -> None:
        await asyncio.sleep(0.01)         # actual call only

    t0 = time.perf_counter()
    await asyncio.gather(*(per_request_client() for _ in range(20)))
    a = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*(shared_client() for _ in range(20)))
    b = time.perf_counter() - t0
    print(f"      20 calls, new client each time: {a * 1000:.0f}ms")
    print(f"      20 calls, shared client:        {b * 1000:.0f}ms")


# ---------------------------------------------------------------------------
# PITFALL 6 — asyncio.run() called more than once, and loop-bound objects
# ---------------------------------------------------------------------------

async def pitfall6_loops() -> None:
    banner("PITFALL 6 — loop-bound primitives and multiple event loops")

    print("""
      asyncio.Lock, Semaphore, Event, and Queue bind to the running loop the
      first time they are awaited. A module-level instance therefore breaks
      the moment a second loop exists:

          _SEM = asyncio.Semaphore(8)      # module level — WRONG

      Under pytest-asyncio each test gets a fresh loop, so the second test
      fails with "attached to a different loop" or hangs. Same failure in any
      code that calls asyncio.run() twice.

      FIX: create them inside your async setup, or lazily per loop:

          class Clients:
              def __init__(self):
                  self.sem = asyncio.Semaphore(8)   # built inside the loop

      And never call asyncio.run() from inside a coroutine — it will raise
      "cannot be called from a running event loop". If you need to bridge sync
      code into async, the tool is asyncio.run_coroutine_threadsafe() from
      another thread, not a nested run().""")

    sem = asyncio.Semaphore(2)
    async with sem:
        print(f"      semaphore acquired on loop {id(asyncio.get_running_loop())}")


# ---------------------------------------------------------------------------
# PITFALL 7 — gather over a generator that is itself slow
# ---------------------------------------------------------------------------

async def pitfall7_eager_materialisation() -> None:
    banner("PITFALL 7 — gather materialises everything up front")

    client = FakeLLMClient(seed=91, base_latency_s=0.01, jitter_s=0.0)

    print("""
      `asyncio.gather(*(f(x) for x in huge_iterable))` unpacks the generator
      IMMEDIATELY. All N coroutine objects exist before a single one runs. For
      50,000 items that is 50,000 coroutine objects plus whatever each one
      captured — often the full document text.

      Worse, if `huge_iterable` is itself a paged API call, the `*` unpacking
      forces the entire pagination to complete before any work starts. Your
      "streaming" pipeline is now a batch job with a giant memory spike.

      FIX: bounded_map (04) or a queue pipeline (08). Both consume the source
      lazily and keep only `limit` items in flight.""")

    # Show that the coroutines all exist before any runs.
    created: list[str] = []

    def make(i: int):
        created.append(f"c{i}")
        return client.complete(f"q{i}")

    coros = [make(i) for i in range(5)]
    print(f"\n      coroutines created before any execution: {created}")
    await asyncio.gather(*coros)


# ---------------------------------------------------------------------------

async def main() -> None:
    await pitfall1_task_gc()
    await pitfall2_lost_exceptions()
    await pitfall3_race()
    await pitfall4_contextvars()
    await pitfall5_client_lifecycle()
    await pitfall6_loops()
    await pitfall7_eager_materialisation()

    banner("PR REVIEW CHECKLIST — async")
    print("""
  Lifetime & ownership
    [ ] Every create_task has an owner: a TaskGroup, or a retained reference
        plus a done-callback. No bare fire-and-forget.
    [ ] Shutdown path cancels AND awaits every task it created.
    [ ] Clients (httpx, AsyncAzureOpenAI, credentials) live in lifespan, not
        in the request path.

  Correctness
    [ ] No read-modify-write spanning an await without a lock.
    [ ] CancelledError is always re-raised; no bare `except:`.
    [ ] Loop-bound primitives (Lock/Semaphore/Queue/Event) are not module-level.
    [ ] `async for` over a generator that may exit early is inside aclosing().

  Bounds
    [ ] Concurrency bounded by a semaphore sized from a stated constraint.
    [ ] Rate bounded per deployment, not per client object.
    [ ] Memory bounded by a queue maxsize, not by trusting the input size.
    [ ] One end-to-end deadline per request, above the transport timeouts.

  Blocking
    [ ] No time.sleep / requests / sync SDK / heavy CPU in a coroutine.
    [ ] Executors are explicitly sized, not the shared default, at scale.

  Observability
    [ ] trace_id in a contextvar, set once at the edge.
    [ ] Loop-lag metric emitted.
    [ ] Retry counts, queue depths, and per-stage blocked-time emitted.
    [ ] loop.set_exception_handler installed and routed to the real logger.
""")


if __name__ == "__main__":
    asyncio.run(main())
