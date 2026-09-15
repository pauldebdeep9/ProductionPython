"""
01 — Structured logging: fields, not sentences.

THE SHIFT
---------
Unstructured logging optimises for a human reading a terminal:

    logger.info(f"Retrieved {len(chunks)} chunks for {user} in {ms}ms")

Structured logging optimises for a machine answering a question:

    logger.info("retrieval complete", extra={
        "event": "retrieval.complete", "chunk_count": len(chunks),
        "principal": user, "duration_ms": ms})

The first is nicer to read one line at a time. The second is the only one that
answers "what is the p95 retrieval latency for tenant X on the new index?"
without a regex.

At the scale where logs matter — thousands per minute across dozens of
replicas — you never read them one at a time. You query them. So the format
should be a query surface, not prose.

THE MOST IMPORTANT SINGLE DECISION is the `event` field: a stable, low-
cardinality name you group by. Everything else in this script follows from
getting that right.

Run:  python 01_structured_logging.py
"""

from __future__ import annotations

import json

from obs_lab import (
    CORPUS,
    banner,
    make_logger,
    new_trace_id,
    principal_var,
    section,
    show,
    tenant_var,
    trace_id_var,
)

log, cap = make_logger("isc.demo")


# ---------------------------------------------------------------------------
# PART 1 — the same event, two ways
# ---------------------------------------------------------------------------

def part1_comparison() -> None:
    banner("PART 1 — prose vs fields")

    chunks = CORPUS[:3]
    duration_ms = 42.7

    section("unstructured")
    prose = (f"Retrieved {len(chunks)} chunks for debdeep@contoso.com "
             f"in {duration_ms}ms using index isc-docs-v3")
    print(f"      INFO  {prose}")

    section("structured")
    trace_id_var.set(new_trace_id())
    principal_var.set("debdeep@contoso.com")
    cap.reset()
    log.info("retrieval complete", extra={
        "event": "retrieval.complete",
        "chunk_count": len(chunks),
        "duration_ms": duration_ms,
        "index": "isc-docs-v3",
    })
    print(f"      {json.dumps(cap.records[0])}")

    print("""
    QUESTIONS THE FIRST FORM CANNOT ANSWER without a regex that breaks the
    first time someone edits the message:

      * p95 duration_ms for event="retrieval.complete", last hour
      * count by index — did the new index change the latency profile?
      * every event for this trace_id, in order
      * chunk_count distribution — are we retrieving 0 chunks more often?

    All four are one query against the second form, and they keep working
    when the message wording changes.

    THE DEEPER POINT: the moment you write a log parser, your log format has
    become an API — with no schema, no versioning, and no test. Emitting
    fields skips that entirely.""")


# ---------------------------------------------------------------------------
# PART 2 — event names
# ---------------------------------------------------------------------------

def part2_event_names() -> None:
    banner("PART 2 — `event` is the field you group by")

    print("""
    RULES FOR EVENT NAMES:

    1. STABLE. `retrieval.complete` forever. If you rename it, every saved
       query, dashboard, and alert referencing it silently returns zero rows —
       which looks like "the problem stopped" rather than "the query broke".

    2. LOW CARDINALITY. A fixed vocabulary of a few dozen. NEVER interpolate:

           event=f"retrieval.{tenant}.complete"      NO

       That is one event name per tenant. Your backend now indexes thousands
       of distinct names, grouping is meaningless, and the cost model punishes
       you. The tenant is a FIELD.

    3. DOTTED, NOUN.VERB, PAST TENSE for things that happened.
           retrieval.complete    generation.started    repair.attempted
           tool.invoked          acl.violation         cache.hit

    4. PAIRED START/END only where duration matters and a span does not
       already cover it. If you have a span, the span IS the duration; a
       separate pair of log lines is duplicate telemetry you pay for twice.

    5. NAME THE OUTCOME, not the code path. `generation.degraded` is useful.
       `entered_fallback_branch_3` is not — it will not survive a refactor and
       it means nothing to anyone but its author.""")

    section("a real vocabulary for a RAG service")
    vocabulary = [
        ("request.received", "SERVER span opens; principal and tenant known"),
        ("acl.resolved", "group set determined; count only, never the groups"),
        ("retrieval.complete", "chunk_count, trimmed_count, duration"),
        ("acl.violation", "SECURITY. A chunk survived that should not have"),
        ("rerank.complete", "input/output counts"),
        ("generation.started", "deployment, prompt version, token budget"),
        ("generation.complete", "finish_reason, tokens, cost"),
        ("generation.degraded", "fell back to a cheaper model or cache"),
        ("repair.attempted", "which repair, which attempt"),
        ("refusal.detected", "the model declined; NOT an error"),
        ("request.completed", "outcome, total duration, cost"),
    ]
    for name, meaning in vocabulary:
        print(f"      {name:<24} {meaning}")

    print("""
    ELEVEN NAMES cover a whole RAG service. That is roughly the right size:
    small enough to hold in your head, granular enough that each one answers
    a distinct question.""")


# ---------------------------------------------------------------------------
# PART 3 — log levels, with an actual decision rule
# ---------------------------------------------------------------------------

def part3_levels() -> None:
    banner("PART 3 — levels: who is woken up?")

    print("""
    The only useful definition of a level is in terms of ACTION.

      CRITICAL  the service cannot serve. Page someone now.
      ERROR     THIS request failed and a human should eventually look.
                Not: a dependency returned a 429 that we successfully retried.
      WARNING   something unexpected that did not fail the request. Degraded
                to a fallback model, a repair was needed, a cache is cold.
      INFO      one or two lines per request describing the outcome. This is
                your default and it is what you will actually query.
      DEBUG     everything else. OFF in production.

    THE MOST COMMON MISTAKE is logging at ERROR for things that were handled.
    A retry that succeeded is not an error; it is a WARNING at most, and
    arguably just a counter. Once your ERROR stream contains handled failures,
    nobody reads it, and the real errors are invisible.

    THE SECOND MOST COMMON is DEBUG in production. In an LLM system that is
    not merely noisy — DEBUG lines are where prompt text and retrieved chunk
    content live, so enabling it moves document content into your telemetry
    workspace. That is why the config tutorial made it a startup-fatal rule.""")

    section("the same failure at three levels")
    cap.reset()
    trace_id_var.set(new_trace_id())

    log.warning("retry succeeded", extra={
        "event": "upstream.retried", "attempt": 2, "status": 503,
        "recovered": True})
    log.error("request failed", extra={
        "event": "request.failed", "outcome": "deadline_exceeded",
        "attempts": 4})
    log.info("request completed", extra={
        "event": "request.completed", "outcome": "ok", "duration_ms": 812})

    skip = {"ts", "level", "logger", "event", "trace_id", "span_id",
            "enduser_id", "tenant"}
    for r in cap.records:
        fields = {k: v for k, v in r.items() if k not in skip}
        print(f"      {r['level']:<8} {r['event']:<22} {fields}")

    print("""
    THE ALERTING CONSEQUENCE: alert on `event="request.failed"` counts, not
    on `level="ERROR"`. The event name is a contract you control; the level is
    a convention that every library you depend on interprets differently.""")


# ---------------------------------------------------------------------------
# PART 4 — wide events vs many narrow ones
# ---------------------------------------------------------------------------

def part4_wide_events() -> None:
    banner("PART 4 — one wide event per request")

    section("narrow: eight lines per request")
    cap.reset()
    trace_id_var.set(new_trace_id())
    for event, fields in [
        ("request.received", {"path": "/answer"}),
        ("acl.resolved", {"group_count": 2}),
        ("retrieval.started", {}),
        ("retrieval.complete", {"chunk_count": 4}),
        ("generation.started", {"deployment": "gpt-4o-mini-prod"}),
        ("generation.complete", {"finish_reason": "stop"}),
        ("response.serialised", {"bytes": 1420}),
        ("request.completed", {"duration_ms": 812}),
    ]:
        log.info("", extra={"event": event, **fields})
    show("lines emitted", len(cap.records))

    section("wide: one line carrying everything")
    cap.reset()
    log.info("request completed", extra={
        "event": "request.completed",
        "outcome": "ok",
        "duration_ms": 812,
        "acl.group_count": 2,
        "retrieval.chunk_count": 4,
        "retrieval.trimmed_count": 2,
        "retrieval.duration_ms": 47,
        "generation.deployment": "gpt-4o-mini-prod",
        "generation.finish_reason": "stop",
        "generation.prompt_tokens": 1840,
        "generation.completion_tokens": 96,
        "generation.duration_ms": 741,
        "cost_usd": 0.000334,
        "repairs": 0,
        "degraded": False,
    })
    print(f"      {json.dumps(cap.records[0])[:150]}...")
    show("lines emitted", len(cap.records))

    print("""
    THE ARGUMENT FOR WIDE: a question like "for requests that degraded, what
    was the chunk_count distribution?" is a single filter-and-group over one
    event type. With narrow events it is a JOIN across log lines on trace_id,
    which most log backends do badly and some cannot do at all.

    Ingestion cost also favours wide. Most backends price per record AND per
    GB; eight records each repeating trace_id, span_id, principal, tenant,
    timestamp, and logger name is a lot of duplicated overhead.

    THE ARGUMENT FOR NARROW: if the request dies at step 3, a wide event
    emitted at the end never happens. You lose everything.

    THE ANSWER — and this is the shape worth adopting:

      * SPANS for the per-step structure and timing. They are emitted even on
        the failure path, they nest, and they are built for this.
      * ONE WIDE LOG EVENT at request completion, summarising the outcome.
      * A handful of narrow events ONLY for things that must be visible
        immediately and independently: acl.violation, refusal.detected.

    Then the wide event answers analytical questions, the trace answers "what
    happened in this one request", and you are not paying to emit the same
    information twice.""")


# ---------------------------------------------------------------------------
# PART 5 — field naming and types
# ---------------------------------------------------------------------------

def part5_fields() -> None:
    banner("PART 5 — field discipline")

    print("""
    NAMESPACE with dots, mirroring your span attributes:
        retrieval.chunk_count      generation.prompt_tokens
    Using the same names in logs and spans means one mental model and one set
    of queries.

    UNITS IN THE NAME, always:
        duration_ms   not  duration
        cost_usd      not  cost
        size_bytes    not  size
    A field called `duration` will be seconds in one place and milliseconds in
    another within six months, and no query will notice.

    TYPES MUST BE STABLE. A field that is sometimes a number and sometimes a
    string breaks aggregation in most backends — often silently, by dropping
    the rows it cannot coerce. Two specific traps:
        * `null` vs absent: pick one and be consistent.
        * a count that is `0` vs `"none"` vs missing.

    BOOLEANS, not sentinel strings: `degraded: true`, not `degraded: "yes"`.

    NEVER put a variable in a field NAME:
        {"tenant_isc_sg_count": 4}          NO
        {"tenant": "isc-sg", "count": 4}    YES
    The first creates unbounded schema growth, which some backends charge for
    and all of them handle badly.""")

    section("the same data, well and badly shaped")
    bad = {"msg": "took 0.812s and cost 3.34e-4", "chunks": "4 of 6"}
    good = {"duration_ms": 812, "cost_usd": 0.000334,
            "retrieval.chunk_count": 4, "retrieval.candidate_count": 6}
    print(f"      bad : {json.dumps(bad)}")
    print(f"      good: {json.dumps(good)}")
    print("      Only the second can be averaged, percentiled, or thresholded.")


# ---------------------------------------------------------------------------
# PART 6 — what NOT to log
# ---------------------------------------------------------------------------

def part6_not_to_log() -> None:
    banner("PART 6 — the fields that must not appear")

    print("""
    NEVER, in any environment:
      * credentials, tokens, connection strings          (config tutorial)
      * raw prompt text
      * retrieved chunk text
      * the user's question, verbatim
      * model output, verbatim

    The last four are the LLM-specific ones and they surprise people, because
    they are exactly what you want when debugging a bad answer.

    THE REASON is that telemetry is a SEPARATE TRUST BOUNDARY. Query access to
    Application Insights is handed out freely for debugging; access to the
    SharePoint documents behind your index is not. Logging chunk text copies
    document content into a store with a different — and larger — reader set,
    a different retention period, and no permission trimming at all.

    WHAT TO LOG INSTEAD, which is almost as useful:
        question_hash         group identical questions without storing them
        question_length       detect truncation and abuse
        chunk_ids             which documents, not their content
        chunk_count           and how many were ACL-trimmed
        prompt_tokens         size without content
        prompt_fingerprint    which template version produced this
        finish_reason         how generation ended

    With chunk_ids and a prompt fingerprint you can RECONSTRUCT the exact
    prompt from your own index, under your own permission checks, when you
    genuinely need it — which is the right place for that decision.

    THE ESCAPE HATCH, and it needs to be a deliberate one: a separate AUDIT
    sink with restricted access, short retention, and an explicit legal basis,
    receiving the full content. Script 04 builds it. What must not happen is
    content ending up in the general telemetry stream because someone needed
    to debug something on a Tuesday.""")

    section("a debuggable log line with no content in it")
    cap.reset()
    trace_id_var.set(new_trace_id())
    tenant_var.set("isc-sg")
    log.info("generation complete", extra={
        "event": "generation.complete",
        "question_hash": "a41f9c02",
        "question_length": 47,
        "retrieval.chunk_ids": ["c1", "c3", "c4"],
        "retrieval.trimmed_count": 2,
        "prompt.fingerprint": "36206a029343",
        "prompt.version": "v3",
        "generation.prompt_tokens": 1840,
        "generation.finish_reason": "stop",
        "model.deployment": "gpt-4o-mini-prod",
        "model.version": "2024-07-18",
    })
    print(f"      {json.dumps(cap.records[0], indent=2)[:520]}")
    print("""
      Everything needed to reproduce and diagnose. No document content, no
      user question, no model output.""")


def main() -> None:
    part1_comparison()
    part2_event_names()
    part3_levels()
    part4_wide_events()
    part5_fields()
    part6_not_to_log()

    banner("SUMMARY")
    print("""
  * Log FIELDS, not sentences. A log parser means your format became an
    unversioned API.
  * `event` is stable, low-cardinality, dotted, past tense. Never interpolate
    a variable into it.
  * Levels are defined by ACTION. A handled retry is not an ERROR. DEBUG is
    off in production — in an LLM system it is where content leaks.
  * Alert on event names, not on level.
  * Spans for per-step structure; ONE wide event per request for analytics;
    narrow events only for things that must be independently visible.
  * Units in field names. Stable types. Never a variable in a field name.
  * Never log prompt text, chunk text, the user's question, or model output.
    Log hashes, ids, counts, and fingerprints — enough to reconstruct under
    your own permission checks.
""")


if __name__ == "__main__":
    main()
