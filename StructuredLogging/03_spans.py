"""
03 — Spans: granularity, attributes, events, and status.

WHAT A SPAN IS FOR
------------------
A log line says "this happened". A span says "this happened, it took this
long, and it happened INSIDE that other thing". The tree structure and the
durations are the whole value — if you are not using both, you want a log
line and you are paying for a span.

THE THREE DECISIONS, in order of how often they are made badly:
    1. GRANULARITY — what deserves a span at all
    2. ATTRIBUTES vs EVENTS — a property of the operation, or a moment in it
    3. STATUS — what "error" means, which is narrower than you think

Run:  python 03_spans.py
"""

from __future__ import annotations

import asyncio
import time

from obs_lab import (
    CORPUS,
    SpanKind,
    TelemetryStore,
    Tracer,
    banner,
    new_trace_id,
    print_json,
    section,
    show,
    trace_id_var,
)

STORE = TelemetryStore()
TRACER = Tracer(STORE)


# ---------------------------------------------------------------------------
# PART 1 — granularity
# ---------------------------------------------------------------------------

async def part1_granularity() -> None:
    banner("PART 1 — what deserves a span")

    print("""
    A SPAN IS WARRANTED when all three hold:
      * it has a MEANINGFUL DURATION you would act on
      * it can FAIL independently of its parent
      * you would want to see it separately in a waterfall

    A span is NOT warranted for: a function call that is always fast, a loop
    iteration, a property access, a parse. Those are events, attributes, or
    nothing.

    THE TWO FAILURE MODES:

      TOO COARSE — one span per request. You know it took 4 seconds and
      nothing about where they went. This is the more common one, and it
      makes the trace worthless for its main purpose.

      TOO FINE — a span per chunk in a 200-chunk rerank. The waterfall is
      unreadable, the payload is enormous, and most backends will start
      dropping spans. Aggregate instead: one `rerank` span with
      `rerank.input_count=200`.

    FOR A RAG REQUEST the right granularity is roughly seven spans:
        request (SERVER)
          resolve_acl
          embed_query        (CLIENT — a network call)
          search             (CLIENT)
          rerank             (INTERNAL — CPU, on a thread)
          build_prompt
          generate           (CLIENT)
        Plus a span per TOOL CALL if the agent makes any.""")

    section("too fine, for comparison")
    STORE.reset()
    trace_id_var.set(new_trace_id())
    with TRACER.span("rerank_bad"):
        for c in CORPUS[:6]:
            with TRACER.span(f"score_chunk_{c.chunk_id}"):
                await asyncio.sleep(0.001)
    show("spans produced for 6 chunks", len(STORE.spans))
    show("...for a realistic 200 chunks", "201")

    section("aggregated, which is what you want")
    STORE.reset()
    trace_id_var.set(new_trace_id())
    with TRACER.span("rerank") as s:
        scores = []
        t0 = time.perf_counter()
        for c in CORPUS[:6]:
            await asyncio.sleep(0.001)
            scores.append(len(c.text) / 100)
        s.set("rerank.input_count", 6)
        s.set("rerank.output_count", 3)
        s.set("rerank.model", "cross-encoder-v2")
        # The DISTRIBUTION as attributes, not 200 child spans.
        s.set("rerank.score_max", round(max(scores), 3))
        s.set("rerank.score_min", round(min(scores), 3))
        s.set("rerank.per_item_us", round((time.perf_counter() - t0) / 6 * 1e6))
    show("spans produced", len(STORE.spans))
    print_json(STORE.spans[0].to_dict()["attributes"])
    print("      Same information, one span. Per-item timing preserved as a")
    print("      number rather than as 200 objects.")


# ---------------------------------------------------------------------------
# PART 2 — attributes vs events
# ---------------------------------------------------------------------------

async def part2_attributes_vs_events() -> None:
    banner("PART 2 — attributes describe; events happen")

    print("""
    ATTRIBUTE  a property of the whole operation, known by the time it ends.
               model.deployment, retrieval.chunk_count, http.status_code
    EVENT      a timestamped moment DURING the operation, with no duration.
               cache.miss, retry.attempted, first_token, exception

    THE TEST: does it have a TIME? If "when did that happen within the span"
    is a meaningful question, it is an event. If not, it is an attribute.

    WHY IT MATTERS FOR LLM CALLS SPECIFICALLY: time-to-first-token is the
    number users actually feel, and it is a MOMENT inside the generation span.
    As an attribute it loses its position; as an event the waterfall shows you
    the gap between request start and first token directly.""")

    section("a generation span with both")
    STORE.reset()
    trace_id_var.set(new_trace_id())

    with TRACER.span("generate", SpanKind.CLIENT) as s:
        # Attributes known up front.
        s.set("model.deployment", "gpt-4o-mini-prod")
        s.set("model.name", "gpt-4o-mini")
        s.set("model.version", "2024-07-18")
        s.set("prompt.version", "v3")
        s.set("prompt.fingerprint", "36206a029343")
        s.set("gen.max_tokens", 512)

        await asyncio.sleep(0.02)
        s.event("first_token")            # the moment TTFT is measured from

        tokens = 0
        for _ in range(5):
            await asyncio.sleep(0.004)
            tokens += 1
        s.event("stream_complete", token_count=tokens)

        # Attributes known only at the end.
        s.set("gen.completion_tokens", 96)
        s.set("gen.prompt_tokens", 1840)
        s.set("gen.finish_reason", "stop")
        s.set("gen.cost_usd", 0.000334)

    print_json(STORE.spans[0].to_dict())

    print("""
    NOTE `gen.finish_reason` is an ATTRIBUTE even though it describes how the
    span ended — because it is a property of the operation, not a moment. But
    `first_token` is an EVENT, because the question "how long until it?" is
    exactly what you want to ask.""")


# ---------------------------------------------------------------------------
# PART 3 — status
# ---------------------------------------------------------------------------

async def part3_status() -> None:
    banner("PART 3 — ERROR means 'this operation failed'")

    print("""
    OTel defines three statuses and the discipline around them is the same
    argument as log levels:

      UNSET  the default. You did not make a claim. Fine for most spans.
      OK     set EXPLICITLY, and only when "success" is not obvious from the
             absence of an error. Most spans should stay UNSET.
      ERROR  THIS operation failed.

    THE MISTAKE: marking a span ERROR when something recoverable happened
    inside it. A retried 503 that eventually succeeded is not an error — the
    OPERATION succeeded. Mark the retry as an EVENT, keep the span OK.

    Get this wrong and your error rate is meaningless, because it counts
    handled failures. Then nobody trusts it, and you build a second metric to
    track "real" errors.

    THE LLM-SPECIFIC ONES, which are genuinely ambiguous and need a decision
    written down:

      finish_reason=length      NOT an error. The model did what you asked
                                within the budget you set. Attribute + event.
      content_filter            NOT an error of yours. It is an outcome.
                                Track it as its own counter.
      a refusal                 NOT an error. The model declined. Its own
                                outcome, its own counter.
      malformed JSON, repaired  NOT an error. Event on the span.
      malformed JSON, gave up   ERROR. The operation did not produce a result.
      client disconnected       NOT an error. Its own outcome — counting
                                disconnects as 5xx makes your dashboard lie
                                during a mobile network blip.""")

    section("four spans, four correct statuses")
    STORE.reset()
    trace_id_var.set(new_trace_id())

    with TRACER.span("gen.retried_then_ok", SpanKind.CLIENT) as s:
        s.event("retry.attempted", attempt=1, status_code=503)
        s.event("retry.attempted", attempt=2, status_code=503)
        await asyncio.sleep(0.001)
        s.set("gen.attempts", 3)
        # Status stays UNSET -> becomes OK. Two failures happened INSIDE a
        # successful operation.

    with TRACER.span("gen.truncated", SpanKind.CLIENT) as s:
        s.set("gen.finish_reason", "length")
        s.event("truncated", tokens_used=512)
        s.set("outcome", "repairable")
        # Also not an error.

    with TRACER.span("gen.refused", SpanKind.CLIENT) as s:
        s.set("gen.finish_reason", "stop")
        s.set("outcome", "refusal")
        s.event("refusal.detected")

    try:
        with TRACER.span("gen.exhausted", SpanKind.CLIENT) as s:
            s.event("repair.attempted", attempt=1)
            s.event("repair.attempted", attempt=2)
            raise ValueError("could not obtain valid JSON after 3 attempts")
    except ValueError:
        pass

    print(f"      {'span':<24} {'status':<8} {'events'}")
    print(f"      {'-' * 24} {'-' * 8} {'-' * 40}")
    for sp in STORE.spans:
        events = ", ".join(e.name for e in sp.events)
        print(f"      {sp.name:<24} {sp.status.value:<8} {events[:40]}")

    show("spans with ERROR status", len(STORE.errors()))
    print("      One error out of four. Three recoverable or non-error")
    print("      outcomes, each individually queryable by its attributes.")


# ---------------------------------------------------------------------------
# PART 4 — exceptions on spans
# ---------------------------------------------------------------------------

async def part4_exceptions() -> None:
    banner("PART 4 — recording an exception without leaking")

    STORE.reset()
    trace_id_var.set(new_trace_id())

    class ConfigWithSecret:
        def __init__(self) -> None:
            self.api_key = "sk-LEAKY-abc123-DO-NOT-SHIP"

        def __repr__(self) -> str:
            return f"ConfigWithSecret(api_key={self.api_key!r})"

    cfg = ConfigWithSecret()
    try:
        with TRACER.span("call.failed") as s:
            raise ConnectionError(f"failed connecting with {cfg!r}")
    except ConnectionError:
        pass

    section("what got recorded")
    print_json(STORE.spans[0].to_dict()["events"])
    show("does the span contain the key?",
         "sk-LEAKY" in str(STORE.spans[0].to_dict()))

    print("""
    THE KEY IS PRESENT — because the exception MESSAGE contained it, and we
    faithfully recorded the message. `record_exception` recording type and
    message rather than the object's repr protects you from ONE leak path
    (the repr) and not from the other (someone interpolating into the message).

    That is the config tutorial's point, arriving here: the defence has to be
    at DECLARATION (SecretStr) rather than at the recording site. No amount of
    care in your tracing layer fixes an exception message that was built with
    an f-string around a secret.

    THE OTHER RULE: never attach the exception's LOCALS or a full traceback as
    a span attribute. Some frameworks offer this. In a RAG service the locals
    at the point of failure include the prompt, the chunks, and the config.""")

    section("the same failure, with a message that carries no payload")
    STORE.reset()
    try:
        with TRACER.span("call.failed.clean") as s:
            s.set("upstream.host", "isc-aoai-prod.openai.azure.com")
            raise ConnectionError("connection reset by peer after 3 attempts")
    except ConnectionError:
        pass
    print_json(STORE.spans[0].to_dict()["events"])
    show("contains the key?", "sk-LEAKY" in str(STORE.spans[0].to_dict()))


# ---------------------------------------------------------------------------
# PART 5 — span kind
# ---------------------------------------------------------------------------

async def part5_kinds() -> None:
    banner("PART 5 — span kind, and why backends care")

    print("""
      SERVER    you RECEIVED a request. One per request, the root of your
                part of the trace.
      CLIENT    you CALLED something external. Azure OpenAI, AI Search,
                Key Vault, a tool API.
      INTERNAL  in-process work. Reranking, prompt building, parsing.
      PRODUCER  you enqueued a message.
      CONSUMER  you dequeued one.

    WHY IT IS NOT COSMETIC:
      * Service maps are built from CLIENT/SERVER pairs. Get them wrong and
        your dependency graph is wrong.
      * "Time spent in dependencies" is computed from CLIENT spans. Mark your
        Azure OpenAI call INTERNAL and it vanishes from that number — so a
        latency regression looks like it is in your code.
      * PRODUCER/CONSUMER let a backend link a trace across a queue, where
        parent-child does not apply because they are separated in time.

    THE MOST COMMON MISTAKE is marking everything INTERNAL because it is the
    default. Every network call is CLIENT.""")

    section("a correctly-kinded RAG trace")
    STORE.reset()
    trace_id_var.set(new_trace_id())

    with TRACER.span("POST /answer", SpanKind.SERVER,
                     **{"http.method": "POST", "http.route": "/answer"}):
        with TRACER.span("resolve_acl", SpanKind.CLIENT,
                         **{"peer.service": "microsoft-graph"}):
            await asyncio.sleep(0.004)
        with TRACER.span("embed_query", SpanKind.CLIENT,
                         **{"peer.service": "azure-openai"}):
            await asyncio.sleep(0.006)
        with TRACER.span("search", SpanKind.CLIENT,
                         **{"peer.service": "azure-ai-search"}):
            await asyncio.sleep(0.012)
        with TRACER.span("rerank", SpanKind.INTERNAL):
            await asyncio.sleep(0.008)
        with TRACER.span("build_prompt", SpanKind.INTERNAL):
            await asyncio.sleep(0.001)
        with TRACER.span("generate", SpanKind.CLIENT,
                         **{"peer.service": "azure-openai"}):
            await asyncio.sleep(0.030)

    tid = STORE.traces()[0]
    print(STORE.render_trace(tid))

    client_ms = sum(s.duration_ms for s in STORE.spans
                    if s.kind is SpanKind.CLIENT)
    root = next(s for s in STORE.spans if s.kind is SpanKind.SERVER)
    show("total request", f"{root.duration_ms:.1f}ms")
    show("time in dependencies (CLIENT spans)", f"{client_ms:.1f}ms")
    show("time in our own code", f"{root.duration_ms - client_ms:.1f}ms")
    print("""
      That last line is the number you want during a latency investigation,
      and it is only computable because the kinds are right.""")


# ---------------------------------------------------------------------------
# PART 6 — semantic conventions
# ---------------------------------------------------------------------------

async def part6_conventions() -> None:
    banner("PART 6 — use the standard attribute names")

    print("""
    OTel publishes SEMANTIC CONVENTIONS — agreed attribute names so that
    backends can build dashboards without knowing your service.

    Use them where they exist:
        http.request.method    http.response.status_code   url.path
        server.address         server.port                 error.type
        db.system              db.operation.name
        rpc.method             network.protocol.version

    GEN-AI CONVENTIONS exist and are stabilising. As of writing, the shape is:
        gen_ai.system                  "azure.ai.openai"
        gen_ai.request.model           the model
        gen_ai.request.max_tokens
        gen_ai.request.temperature
        gen_ai.response.model          what actually served — may differ!
        gen_ai.response.finish_reasons
        gen_ai.usage.input_tokens
        gen_ai.usage.output_tokens
        gen_ai.operation.name          "chat", "embeddings"

    THESE ARE STILL MOVING, so check the current spec rather than trusting
    this list. The durable advice is: use the convention where one exists,
    prefix your own with a namespace you own (`isc.`), and never invent a name
    that collides with a standard one — a backend that special-cases
    `gen_ai.usage.input_tokens` will do the wrong thing with your differently
    shaped version.

    THE ONE THAT MATTERS MOST: `gen_ai.response.model` versus
    `gen_ai.request.model`. On an auto-updating Azure deployment they diverge
    when Microsoft moves your deployment to a new snapshot. Recording both is
    how you detect that your outputs changed without any deploy of yours.""")

    section("our attributes, split by namespace")
    STORE.reset()
    trace_id_var.set(new_trace_id())
    with TRACER.span("chat", SpanKind.CLIENT) as s:
        for k, v in {
            "gen_ai.system": "azure.ai.openai",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.request.max_tokens": 512,
            "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
            "gen_ai.response.finish_reasons": ["stop"],
            "gen_ai.usage.input_tokens": 1840,
            "gen_ai.usage.output_tokens": 96,
            "server.address": "isc-aoai-prod.openai.azure.com",
            # Ours, namespaced so it can never collide.
            "isc.prompt.version": "v3",
            "isc.prompt.fingerprint": "36206a029343",
            "isc.retrieval.chunk_count": 4,
            "isc.tenant": "isc-sg",
        }.items():
            s.set(k, v)
    print_json(STORE.spans[0].to_dict()["attributes"])


async def main() -> None:
    await part1_granularity()
    await part2_attributes_vs_events()
    await part3_status()
    await part4_exceptions()
    await part5_kinds()
    await part6_conventions()

    banner("SUMMARY")
    print("""
  * A span needs meaningful duration, independent failure, and a place in the
    waterfall. Otherwise use an event or an attribute.
  * ~7 spans per RAG request. Aggregate per-item work into counts and
    distributions instead of a span per item.
  * Attributes describe the operation; events are moments within it. TTFT is
    an event.
  * ERROR means THIS operation failed. A retried-then-succeeded call is OK
    with retry EVENTS. Truncation, content filtering, refusals, and client
    disconnects are outcomes, not errors.
  * `record_exception` should capture type and message, never the object repr
    or locals — but the real defence against leaks is SecretStr at
    declaration.
  * Span kind drives service maps and dependency-time calculations. Every
    network call is CLIENT.
  * Use OTel semantic conventions where they exist; namespace your own.
    Record both gen_ai.request.model and gen_ai.response.model.
""")


if __name__ == "__main__":
    asyncio.run(main())
