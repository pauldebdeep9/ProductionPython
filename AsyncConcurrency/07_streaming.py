"""
07 — Streaming: async generators, token streams, and their sharp edges.

WHY STREAMING IS A CONCURRENCY TOPIC
------------------------------------
Token streaming is the default UX for chat. It changes your concurrency model
in ways that are easy to miss:

  * A streaming request holds a connection for the WHOLE generation, not just
    the latency of one round trip. Your effective concurrency requirement goes
    up by roughly (generation_time / request_time). A semaphore sized for
    non-streaming calls will be badly wrong.
  * A total-duration timeout becomes meaningless; you need inter-token gap
    detection (see 05).
  * Cancellation gets more likely, not less — users close tabs mid-stream.
  * Partial output has to be handled: what do you log, what do you cache, what
    do you tell the user when generation fails at token 400 of 500?

ASYNC GENERATORS HAVE A CLEANUP PROBLEM
---------------------------------------
An async generator suspended at a `yield` has a `finally` block that has not
run. If the consumer stops early, that cleanup runs only when the generator is
garbage-collected — at an unpredictable time, on an unpredictable loop, or
never. `aclosing()` fixes this. This is the async equivalent of relying on
CPython refcounting to close your files: it works until it doesn't.

Run:  python 07_streaming.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

from fake_llm import FakeLLMClient, Timer, banner

# ---------------------------------------------------------------------------
# PART 1 — consuming a token stream, and measuring what users feel
# ---------------------------------------------------------------------------

async def part1_basic_stream() -> None:
    banner("PART 1 — token streaming and the metrics that matter")

    client = FakeLLMClient(seed=51, base_latency_s=0.15, jitter_s=0.0)

    t0 = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    parts: list[str] = []

    async for tok in client.stream("summarise this purchase order", n_tokens=8):
        if ttft is None:
            # TIME TO FIRST TOKEN. This is the number users actually
            # experience as "is it working?". Track it separately from total
            # duration — they move independently and only TTFT drives the
            # perception of responsiveness.
            ttft = time.perf_counter() - t0
        tokens += 1
        parts.append(tok)

    total = time.perf_counter() - t0
    print(f"    output:        {''.join(parts).strip()}")
    print(f"    TTFT:          {ttft * 1000:.0f}ms   <- perceived responsiveness")
    print(f"    total:         {total * 1000:.0f}ms")
    print(f"    inter-token:   {(total - ttft) / max(1, tokens - 1) * 1000:.0f}ms avg")
    print("""
    Emit all three as separate metrics. A regression in TTFT (routing, cold
    start, a slow retrieval step in front of generation) is invisible if you
    only track end-to-end latency.""")


# ---------------------------------------------------------------------------
# PART 2 — the async generator cleanup trap
# ---------------------------------------------------------------------------

async def part2_aclosing() -> None:
    banner("PART 2 — aclosing(): async generators need explicit shutdown")

    released: list[str] = []

    async def streaming_query(name: str):
        """A generator holding a real resource — a DB cursor, an HTTP
        response, a semaphore slot. The `finally` must run to release it."""
        try:
            for i in range(100):
                await asyncio.sleep(0.001)
                yield f"{name}:chunk{i}"
        finally:
            released.append(name)

    # --- WRONG: break out of the loop, leave the generator suspended. ---
    print("\n  A) early break without aclosing")
    async for chunk in streaming_query("leaky"):
        if chunk.endswith("chunk2"):
            break
    print(f"      released so far: {released}")
    print("      <-- the generator is suspended at `yield`; its finally has NOT run.")
    print("      It will run at GC time, on some future loop, or never.")

    # --- RIGHT: aclosing guarantees aclose() on scope exit. ---
    print("\n  B) early break with contextlib.aclosing")
    async with contextlib.aclosing(streaming_query("clean")) as stream:
        async for chunk in stream:
            if chunk.endswith("chunk2"):
                break
    print(f"      released: {released}")
    print("      <-- 'clean' released deterministically at the `async with` exit.")
    if "leaky" in released:
        print("      LOOK CLOSELY: 'leaky' has now appeared too — it was collected")
        print("      at some arbitrary point AFTER we moved on. That is the bug in")
        print("      miniature: the cleanup did happen, just not when you needed it,")
        print("      and under load 'later' can mean 'after the pool is exhausted'.")

    print("""
  RULE: any `async for` over a generator you might exit early — because of a
  break, an exception, a timeout, or a client disconnect — belongs inside
  `contextlib.aclosing`. In an LLM service that is essentially every token
  stream, because disconnects are normal.

  Note also `asyncio.run()` calls `loop.shutdown_asyncgens()` for you at exit.
  If you manage the loop yourself, you must call it, or generator cleanup is
  silently skipped on shutdown.""")


# ---------------------------------------------------------------------------
# PART 3 — composing streams: transform, tee, and fan-in
# ---------------------------------------------------------------------------

async def part3_composition() -> None:
    banner("PART 3 — transforming and merging streams")

    client = FakeLLMClient(seed=52, base_latency_s=0.05, jitter_s=0.0)

    # -- Transformation: a generator that consumes a generator. This is how
    #    you build a pipeline stage — buffering, redaction, citation
    #    rewriting — without materialising the whole response.
    async def redact(stream):
        """Stands in for PII/sensitivity redaction on the way out.

        SUBTLE: a token-by-token redactor cannot match patterns that span
        token boundaries. Real implementations buffer a sliding window. This
        is a genuine correctness issue for DLP on streamed output, and worth
        raising in an architecture review — streaming makes output-side
        filtering strictly harder than batch.
        """
        async for tok in stream:
            yield tok.replace("answer", "[REDACTED]")

    async def numbered(stream):
        i = 0
        async for tok in stream:
            i += 1
            yield f"({i}){tok}"

    print("\n  chained transformations:")
    async with contextlib.aclosing(
        numbered(redact(client.stream("q", n_tokens=5)))
    ) as s:
        out = [tok async for tok in s]
    print(f"    {''.join(out).strip()}")

    # -- Fan-in: merge several streams into one, in arrival order. Useful for
    #    running two models concurrently, or interleaving retrieval progress
    #    with generation tokens.
    async def merge(*streams):
        """Merge N async iterables into one, yielding as items arrive.

        Implemented with a queue and one pump task per source. The sentinel
        counting is the fiddly part — get it wrong and you either hang forever
        or truncate the output.
        """
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()

        async def pump(src, label):
            try:
                async for item in src:
                    await queue.put((label, item))
            finally:
                await queue.put((label, DONE))

        async with asyncio.TaskGroup() as tg:
            for i, s in enumerate(streams):
                tg.create_task(pump(s, f"m{i}"))

            remaining = len(streams)
            while remaining:
                label, item = await queue.get()
                if item is DONE:
                    remaining -= 1
                    continue
                yield label, item

    print("\n  fan-in from two concurrent model streams:")
    fast = FakeLLMClient(seed=53, base_latency_s=0.02, jitter_s=0.0)
    slow = FakeLLMClient(seed=54, base_latency_s=0.08, jitter_s=0.0)
    seen: list[str] = []
    async with contextlib.aclosing(
        merge(fast.stream("a", n_tokens=3), slow.stream("b", n_tokens=3))
    ) as m:
        async for label, tok in m:
            seen.append(label)
    print(f"    arrival order by source: {' '.join(seen)}")
    print("    (fast model's tokens land first — arrival order, not source order)")


# ---------------------------------------------------------------------------
# PART 4 — SSE framing, the wire format you will actually emit
# ---------------------------------------------------------------------------

async def part4_sse() -> None:
    banner("PART 4 — Server-Sent Events framing")

    client = FakeLLMClient(seed=55, base_latency_s=0.02, jitter_s=0.0)

    async def sse_events(prompt: str, trace_id: str):
        """Yields SSE-framed bytes. In FastAPI:

            return StreamingResponse(
                sse_events(prompt, trace_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"},   # disable nginx buffering
            )

        GOTCHA: proxies buffer. If you stream perfectly and the user still
        sees nothing until the end, suspect an intermediate proxy, App Gateway,
        or CDN — not your code. `X-Accel-Buffering: no` handles nginx;
        Azure Front Door and API Management each need their own setting.
        """
        # Send the trace id first so the client can quote it in a bug report.
        yield f"event: meta\ndata: {json.dumps({'trace_id': trace_id})}\n\n"
        try:
            async for tok in client.stream(prompt, n_tokens=5):
                yield f"data: {json.dumps({'delta': tok})}\n\n"
        except asyncio.CancelledError:
            # Client disconnected. Log it as a disconnect, NOT an error — a
            # dashboard that counts disconnects as 5xx will lie to you.
            yield "event: cancelled\ndata: {}\n\n"
            raise
        except Exception as e:
            # CRITICAL: HTTP status was already sent as 200 when the first byte
            # went out. You CANNOT return a 500 now. Mid-stream errors must be
            # signalled in-band as an event the client understands, and your
            # client code must actually handle that event.
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    print("\n  wire output:")
    async with contextlib.aclosing(sse_events("explain the variance", "tr-abc123")) as s:
        async for frame in s:
            print(f"    {frame!r}")

    # Mid-stream failure
    print("\n  mid-stream failure (status 200 already sent):")
    broken = FakeLLMClient(seed=56, base_latency_s=0.01, server_error_rate=1.0)

    async def broken_events():
        yield "event: meta\ndata: {}\n\n"
        try:
            async for tok in broken.stream("q", n_tokens=3):
                yield f"data: {json.dumps({'delta': tok})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': type(e).__name__})}\n\n"

    async with contextlib.aclosing(broken_events()) as s:
        async for frame in s:
            print(f"    {frame!r}")


# ---------------------------------------------------------------------------
# PART 5 — streaming changes your concurrency arithmetic
# ---------------------------------------------------------------------------

async def part5_capacity() -> None:
    banner("PART 5 — streaming and connection-hold time")

    print("""
  Little's Law again, with the streaming correction:

    non-streaming:  a request holds a connection for ~latency
    streaming:      a request holds a connection for TTFT + (tokens x gap)

  Concrete: 500 output tokens at 20ms/token = 10s of connection hold, versus
  perhaps 800ms for the equivalent non-streaming call. To serve 50 concurrent
  users you need ~500 concurrent connections' worth of capacity, not ~40.

  What this changes in your design:
    * Semaphore and connection-pool sizing must be derived from HOLD TIME,
      not per-request latency.
    * Idle-timeout settings on every hop (load balancer, App Gateway, API
      Management, the ASGI server) must exceed your longest generation. The
      default on several Azure front doors is 4 minutes; a long RAG answer
      with a big context can exceed it.
    * Per-user concurrency caps become necessary, or one user with 20 open
      tabs consumes the whole pool.
    * Graceful shutdown gets harder: draining 500 open streams takes as long
      as the longest one. Budget the termination grace period accordingly.""")

    # Demonstrate hold-time difference concretely.
    client = FakeLLMClient(seed=57, base_latency_s=0.05, jitter_s=0.0)

    with Timer("non-streaming call (connection held)"):
        await client.complete("q")

    with Timer("streaming call, 10 tokens (connection held)"):
        async with contextlib.aclosing(client.stream("q", n_tokens=10)) as s:
            async for _ in s:
                pass


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_basic_stream()
    await part2_aclosing()
    await part3_composition()
    await part4_sse()
    await part5_capacity()

    banner("SUMMARY")
    print("""
  * Track TTFT separately from total latency; users feel the former.
  * Wrap every `async for` you might exit early in `contextlib.aclosing`.
  * Generators compose: transform stages consume and re-yield without
    materialising the whole response.
  * Once the first byte is sent the status code is fixed at 200 — mid-stream
    errors must be in-band events, and the client must handle them.
  * Client disconnect is normal traffic, not an error. Count it separately.
  * Streaming multiplies connection hold time; resize pools, semaphores, and
    every idle timeout on the path accordingly.
  * Output-side redaction across token boundaries needs a sliding buffer —
    naive per-token filtering misses spans. Raise this in design review.
""")


if __name__ == "__main__":
    asyncio.run(main())
