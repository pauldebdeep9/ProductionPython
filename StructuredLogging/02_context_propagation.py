"""
02 — Context propagation: keeping the trace id attached.

THE FAILURE THIS PREVENTS
-------------------------
A user reports a bad answer and gives you a trace id. You query for it and get
four log lines out of the forty the request produced. The rest have an empty
trace_id because they were emitted from a background task, a thread pool, or a
downstream service that never received the header.

A partially-propagated trace is worse than none, because it looks complete.

THE MECHANISM
-------------
`contextvars` — a per-context store that asyncio understands. The important
property, and the one that makes it work at all:

    asyncio.create_task() COPIES the current context at creation time.

So a task started inside a request inherits the request's trace id
automatically, with no threading of parameters through five call layers.

THE EDGES, which is most of this script:
    * the copy is ONE-WAY (child cannot publish back to parent)
    * `to_thread` copies it, but a raw `ThreadPoolExecutor` does not
    * processes never inherit it
    * background tasks that OUTLIVE the request inherit a stale context
    * HTTP calls need an explicit header

Run:  python 02_context_propagation.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import re
import threading

from obs_lab import (
    SpanKind,
    TelemetryStore,
    Tracer,
    banner,
    make_logger,
    new_span_id,
    new_trace_id,
    principal_var,
    section,
    show,
    span_id_var,
    tenant_var,
    trace_id_var,
    verdict,
)

log, cap = make_logger("isc.ctx")
STORE = TelemetryStore()
TRACER = Tracer(STORE)


# ---------------------------------------------------------------------------
# PART 1 — the automatic case
# ---------------------------------------------------------------------------

async def part1_tasks_inherit() -> None:
    banner("PART 1 — create_task copies the context")

    async def leaf(name: str) -> str:
        # No trace_id parameter. It is simply THERE.
        log.info("", extra={"event": "leaf.done", "leaf": name})
        return f"{name}:{trace_id_var.get()[:8]}"

    async def handle(tid: str, who: str) -> list[str]:
        trace_id_var.set(tid)
        principal_var.set(who)
        async with asyncio.TaskGroup() as tg:
            a = tg.create_task(leaf("retrieve"))
            b = tg.create_task(leaf("generate"))
        return [a.result(), b.result()]

    cap.reset()
    t1, t2 = new_trace_id(), new_trace_id()
    results = await asyncio.gather(
        handle(t1, "debdeep@contoso.com"),
        handle(t2, "hr.lead@contoso.com"),
    )
    for group in results:
        show("results", group)

    section("no cross-contamination between concurrent requests")
    # NOTE the field is `enduser_id`, a SALTED HASH — not `principal`. The
    # ContextFilter pseudonymises it on the way out, because a raw email on
    # every log line is a personal identifier in telemetry (script 04, rule 5).
    # It still groups correctly, which is the whole point of hashing rather
    # than dropping: you keep the ability to say "these two lines are the same
    # user" without storing who that user is.
    by_trace: dict[str, set[str]] = {}
    for r in cap.records:
        by_trace.setdefault(r["trace_id"][:8], set()).add(r["enduser_id"])
    for tid, ids in by_trace.items():
        verdict(len(ids) == 1, f"trace {tid}: enduser_id={ids}")

    print("""
    Two concurrent requests, four log lines, every one correctly attributed —
    with no parameter threading. That is the whole value of contextvars.""")


# ---------------------------------------------------------------------------
# PART 2 — the one-way copy
# ---------------------------------------------------------------------------

async def part2_one_way() -> None:
    banner("PART 2 — the copy is one-way")

    result_var: contextvars.ContextVar[str] = contextvars.ContextVar(
        "result", default="unset")

    async def child() -> None:
        result_var.set("set-by-child")

    result_var.set("set-by-parent")
    await asyncio.create_task(child())
    show("parent sees after the child ran", result_var.get())

    print("""
    The child's `set()` applied to ITS copy. The parent is unchanged, and so
    is every sibling.

    THE PRACTICAL CONSEQUENCE: you cannot use a contextvar to return data
    upward. If a child task computes something the parent needs — a token
    count, a cost, a flag — RETURN it. People discover this by writing a
    "collect the cost in a contextvar" helper that silently reports zero.

    THE ONE EXCEPTION that works, and is the standard pattern: put a MUTABLE
    OBJECT in the contextvar. The copy copies the REFERENCE, so children
    mutating that object are seen by the parent.""")

    section("a mutable accumulator, shared by reference")

    class CostAccumulator:
        def __init__(self) -> None:
            self.total_usd = 0.0
            self.calls = 0
            self._lock = asyncio.Lock()

        async def add(self, usd: float) -> None:
            # Still needs a lock: `+=` spanning an await is a race even in
            # single-threaded async (see the async tutorial, pitfall 3).
            async with self._lock:
                await asyncio.sleep(0)
                self.total_usd += usd
                self.calls += 1

    cost_var: contextvars.ContextVar[CostAccumulator] = contextvars.ContextVar(
        "cost")

    async def model_call(usd: float) -> None:
        await cost_var.get().add(usd)

    acc = CostAccumulator()
    cost_var.set(acc)
    async with asyncio.TaskGroup() as tg:
        for _ in range(5):
            tg.create_task(model_call(0.0012))
    show("parent sees accumulated cost", f"${acc.total_usd:.4f} over {acc.calls} calls")
    print("      The reference was copied, not the object — so mutations are shared.")


# ---------------------------------------------------------------------------
# PART 3 — thread and process boundaries
# ---------------------------------------------------------------------------

async def part3_threads() -> None:
    banner("PART 3 — threads: to_thread copies, a raw executor does not")

    def sync_work(label: str) -> str:
        """Runs on a worker thread. Does it see the context?"""
        return f"{label}: trace={trace_id_var.get()[:8] or '(EMPTY)'}"

    trace_id_var.set(new_trace_id())

    section("asyncio.to_thread")
    show("result", await asyncio.to_thread(sync_work, "to_thread"))
    print("      to_thread uses contextvars.copy_context() internally.")

    section("a raw ThreadPoolExecutor via run_in_executor")
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        r = await loop.run_in_executor(pool, sync_work, "run_in_executor")
    show("result", r)

    section("the fix: copy the context explicitly")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        ctx = contextvars.copy_context()
        r = await loop.run_in_executor(pool, lambda: ctx.run(sync_work, "with copy_context"))
    show("result", r)

    section("a bare threading.Thread")
    out: list[str] = []
    t = threading.Thread(target=lambda: out.append(sync_work("bare thread")))
    t.start()
    t.join()
    show("result", out[0])

    ctx = contextvars.copy_context()
    out2: list[str] = []
    t2 = threading.Thread(target=lambda: out2.append(ctx.run(sync_work, "bare + ctx")))
    t2.start()
    t2.join()
    show("result", out2[0])

    print("""
    THE RULE: any thread you create yourself needs `contextvars.copy_context()`
    captured on the calling thread and `ctx.run(...)` inside the worker.
    `asyncio.to_thread` does it for you; nothing else does.

    PROCESSES NEVER INHERIT IT. A ProcessPoolExecutor pickles arguments across
    a process boundary and contextvars do not cross. Pass the trace id as an
    explicit argument and set it inside the child.

    WHERE THIS BITES IN AN LLM PIPELINE: the CPU-bound work you correctly
    moved off the event loop — tokenisation, a local cross-encoder reranker,
    PDF parsing. Those are exactly the places that lose the trace id, and
    exactly the places you want timing from.""")


# ---------------------------------------------------------------------------
# PART 4 — background tasks and stale context
# ---------------------------------------------------------------------------

async def part4_background() -> None:
    banner("PART 4 — background tasks inherit a STALE context")

    async def slow_background_write(label: str) -> None:
        await asyncio.sleep(0.05)
        log.info("", extra={"event": "background.write", "label": label})

    cap.reset()

    async def handler(tid: str) -> asyncio.Task[None]:
        trace_id_var.set(tid)
        # Fire-and-forget, outliving the request. It captured the context at
        # CREATION, which is what we want — but see the warning below.
        return asyncio.create_task(slow_background_write("audit"))

    t = new_trace_id()
    task = await handler(t)
    trace_id_var.set(new_trace_id())          # a LATER request reuses the loop
    await task

    rec = cap.records[0]
    verdict(rec["trace_id"] == t,
            f"background write carried the ORIGINAL trace {rec['trace_id'][:8]}")

    print("""
    That is correct behaviour and usually what you want: the audit write
    belongs to the request that caused it.

    THE HAZARD is the opposite case — a LONG-LIVED task created during a
    request. A poller, a cache warmer, a queue consumer started inside a
    request handler inherits that request's trace id and stamps it on
    everything it does for the rest of the process's life. You then have one
    trace with fifty thousand spans, which most backends will refuse to render.

    THE FIX: long-lived tasks belong in the application LIFESPAN, not in a
    request handler. If one genuinely must start from a request, clear the
    context first:

        async def worker() -> None:
            trace_id_var.set("")      # detach from the creating request
            ...
        asyncio.create_task(worker())""")


# ---------------------------------------------------------------------------
# PART 5 — across services: W3C trace context
# ---------------------------------------------------------------------------

TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def build_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    """version-traceid-spanid-flags, all lowercase hex.

    The format is exact. A backend that cannot parse it drops the header
    SILENTLY and starts a new trace — which presents as "our traces do not
    link across services" with no error to investigate.
    """
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def parse_traceparent(header: str) -> tuple[str, str, bool] | None:
    m = TRACEPARENT.match(header.strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3) == "01"


async def part5_cross_service() -> None:
    banner("PART 5 — W3C traceparent across a service boundary")

    trace_id_var.set(new_trace_id())
    span_id_var.set(new_span_id())

    header = build_traceparent(trace_id_var.get(), span_id_var.get())
    show("outgoing traceparent", header)

    parsed = parse_traceparent(header)
    assert parsed
    show("downstream parses", f"trace={parsed[0][:8]} parent={parsed[1]} sampled={parsed[2]}")

    section("malformed headers are rejected, not guessed")
    for bad, why in [
        (f"00-{new_trace_id().upper()}-{new_span_id()}-01", "uppercase hex"),
        ("00-abc-def-01", "wrong lengths"),
        (str(__import__("uuid").uuid4()), "a uuid with dashes"),
        ("", "absent"),
    ]:
        show(f"{why:<22}", "REJECTED" if parse_traceparent(bad) is None else "accepted")

    print("""
    THE UUID ROW IS THE COMMON BUG. Someone generates a trace id with
    `str(uuid4())`, it has dashes, the header is malformed, and every
    downstream service starts a fresh trace. Everything looks like it works —
    there are traces, they have spans — but nothing links across services and
    the cause is a formatting detail nobody logs.

    ALWAYS `uuid4().hex` — 32 lowercase hex characters, no dashes.

    THE OTHER HEADER: `tracestate` carries vendor-specific key-value pairs and
    must be forwarded UNCHANGED even if you do not understand it. Dropping it
    breaks sampling decisions made upstream.

    AND: `principal` and `tenant` are NOT trace context. Do not smuggle them
    through `tracestate`. They belong in your own authenticated headers or in
    baggage, and a downstream service must re-derive the principal from the
    token rather than trusting a header — otherwise you have built a
    privilege-escalation vector out of a telemetry field.""")


# ---------------------------------------------------------------------------
# PART 6 — a middleware that wires it all up
# ---------------------------------------------------------------------------

async def part6_middleware() -> None:
    banner("PART 6 — the entry point, done once")

    async def observability_middleware(headers: dict[str, str],
                                       principal: str, tenant: str,
                                       handler) -> dict[str, object]:
        """What every request should pass through. In FastAPI this is
        `@app.middleware("http")` or an ASGI middleware class.

        THE FOUR JOBS:
          1. CONTINUE an incoming trace, or start one.
          2. Populate the context so everything downstream inherits it.
          3. Open a SERVER span.
          4. Return the trace id to the CALLER, so a user reporting a problem
             can quote it.
        """
        incoming = parse_traceparent(headers.get("traceparent", ""))
        trace_id = incoming[0] if incoming else new_trace_id()

        trace_id_var.set(trace_id)
        span_id_var.set(incoming[1] if incoming else "")
        principal_var.set(principal)
        tenant_var.set(tenant)

        with TRACER.span("POST /answer", SpanKind.SERVER,
                         **{"http.method": "POST", "http.route": "/answer",
                            "enduser.id_hash": str(hash(principal))[-8:],
                            "tenant": tenant}) as span:
            result = await handler()
            span.set("http.status_code", 200)

        # Echo it back. This single header is what turns "the answer was
        # wrong yesterday" into a query.
        return {**result, "_headers": {"x-trace-id": trace_id}}

    async def handler() -> dict[str, object]:
        with TRACER.span("retrieve") as s:
            await asyncio.sleep(0.005)
            s.set("retrieval.chunk_count", 4)
        with TRACER.span("generate") as s:
            await asyncio.sleep(0.01)
            s.set("model.deployment", "gpt-4o-mini-prod")
        log.info("", extra={"event": "request.completed", "outcome": "ok"})
        return {"answer": "..."}

    section("a fresh trace")
    STORE.reset()
    cap.reset()
    r = await observability_middleware({}, "debdeep@contoso.com", "isc-sg", handler)
    show("returned header", r["_headers"])
    print(STORE.render_trace(STORE.traces()[0]))

    section("continuing an upstream trace")
    STORE.reset()
    upstream_trace, upstream_span = new_trace_id(), new_span_id()
    r2 = await observability_middleware(
        {"traceparent": build_traceparent(upstream_trace, upstream_span)},
        "debdeep@contoso.com", "isc-sg", handler)
    show("upstream trace id", upstream_trace[:16] + "...")
    show("our trace id", str(r2["_headers"]["x-trace-id"])[:16] + "...")
    root = STORE.root_spans()
    show("our root span's parent", root[0].parent_id if root else
         next(s.parent_id for s in STORE.spans if s.name.startswith("POST")))

    print("""
    Note there is no root span in the second case — our SERVER span has the
    upstream span as its parent, which is exactly right: the trace is one
    continuous tree across both services, and the gateway's span is the root.

    THE `x-trace-id` RESPONSE HEADER is the highest-value line in this whole
    file. Without it, "the answer was wrong yesterday around 3pm" is an
    archaeology project. With it, the user quotes an id and you have the
    complete trace in one query.""")


async def main() -> None:
    await part1_tasks_inherit()
    await part2_one_way()
    await part3_threads()
    await part4_background()
    await part5_cross_service()
    await part6_middleware()

    banner("SUMMARY")
    print("""
  * `create_task` copies the context, so tasks inherit the trace id for free.
  * The copy is ONE-WAY. Return values upward; or share a mutable accumulator
    by reference (with a lock).
  * `to_thread` copies the context; `run_in_executor` and `threading.Thread`
    do NOT — use `contextvars.copy_context()` + `ctx.run(...)`.
  * Processes never inherit it. Pass the trace id explicitly.
  * Long-lived tasks started inside a request inherit a stale trace id and
    stamp it forever. Start them in the lifespan, or clear the context.
  * `traceparent` is 32+16 lowercase hex, no dashes. `uuid4().hex`, never
    `str(uuid4())`. A malformed header is dropped silently.
  * Forward `tracestate` unchanged. Never trust a principal from a header.
  * Set the context in ONE middleware, and return `x-trace-id` to the caller.
""")


if __name__ == "__main__":
    asyncio.run(main())
