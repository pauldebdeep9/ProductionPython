"""
01 — Taxonomy: deciding what a failure MEANS before deciding what to do.

THE CENTRAL CLAIM
-----------------
Retry logic is the easy part. The hard part — and the part that determines
whether your system is resilient or merely busy — is classification.

Most codebases have a boolean: retryable or not. That is too coarse. You need
at least these five dispositions, and they are genuinely different actions:

  RETRY      same request, later                (429, 503, timeout)
  REPAIR     DIFFERENT request, now             (context too long, bad JSON)
  FALLBACK   different provider/model/path      (region down, model deprecated)
  FAIL_FAST  nothing helps; surface it          (400 validation, content filter)
  ESCALATE   a human must act                   (401, committed quota gone)

Collapsing REPAIR into RETRY is the most expensive mistake, because the failure
looks retryable (it is a transient-shaped 400) but retrying is guaranteed to
fail identically. You burn your whole budget and then surface the error anyway.

THE SECOND AXIS: SIDE-EFFECT CERTAINTY
--------------------------------------
"Can I retry?" is two questions:
    Is it USEFUL?  (will it plausibly succeed)
    Is it SAFE?    (might the side effect already have happened)

A timeout on a read: useful, safe. A timeout on a write: useful, NOT safe.
Treating those the same is how you create duplicate invoice disputes.

Run:  python 01_taxonomy.py
"""

from __future__ import annotations

import asyncio

from failure_lab import (
    AppError,
    AuthError,
    ConnectionLost,
    ContentFiltered,
    ContextLengthExceeded,
    Disposition,
    PoisonItemError,
    QuotaExhausted,
    RateLimitError,
    SchemaValidationError,
    ServiceUnavailable,
    TruncatedOutput,
    UpstreamTimeout,
    ValidationError,
    banner,
    classify,
    section,
)

# ---------------------------------------------------------------------------
# PART 1 — the classification table
# ---------------------------------------------------------------------------

async def part1_table() -> None:
    banner("PART 1 — one table, five dispositions, two safety classes")

    samples: list[AppError] = [
        RateLimitError("429 too many requests", retry_after=2.0),
        ServiceUnavailable("503 upstream unavailable"),
        UpstreamTimeout("read timeout after 30s"),
        ConnectionLost("connection reset by peer"),
        ValidationError("400 missing required field 'deployment'"),
        ContentFiltered("400 content_filter triggered"),
        ContextLengthExceeded("128000 token limit exceeded by 4200"),
        SchemaValidationError("field 'confidence' expected float, got str"),
        TruncatedOutput("finish_reason=length"),
        AuthError("401 token rejected"),
        QuotaExhausted("committed monthly TPM exhausted"),
        PoisonItemError("PDF has no /Root object"),
    ]

    print(f"    {'error':<26} {'disposition':<11} {'safe to retry?':<15} retry_after")
    print(f"    {'-' * 26} {'-' * 11} {'-' * 15} -----------")
    for e in samples:
        d = classify(e)
        if d is not Disposition.RETRY:
            safe = "n/a (no retry)"
        elif e.side_effect_uncertain:
            safe = "NO — may dupe"
        else:
            safe = "yes"
        ra = f"{e.retry_after}s" if e.retry_after else "-"
        print(f"    {type(e).__name__:<26} {d.value:<11} {safe:<15} {ra}")

    print("""
    The two rows worth staring at:

      UpstreamTimeout / ConnectionLost are RETRY but NOT SAFE. The request may
      have been processed. For a read that is fine. For a write it is a
      correctness bug unless you send an idempotency key (see 03).

      ContextLengthExceeded and SchemaValidationError arrive as HTTP 400s,
      which every generic retry library treats as fail-fast. They are neither
      retryable nor fatal — they are REPAIRABLE, and handling them is where
      most of the reliability of a GenAI system actually comes from (see 08).""")


# ---------------------------------------------------------------------------
# PART 2 — cancellation is not an application error
# ---------------------------------------------------------------------------

async def part2_cancellation() -> None:
    banner("PART 2 — CancelledError is not in the taxonomy at all")

    # This is a deliberate assertion in `classify`: cancellation must never be
    # classified, because classifying it invites retrying it, and retrying a
    # cancellation converts a prompt shutdown into a hang.
    try:
        classify(asyncio.CancelledError())
    except AssertionError as e:
        print(f"    classify(CancelledError) -> {e}")

    print("""
    The review rule: `except Exception` is SAFE in Python 3.8+ because
    CancelledError derives from BaseException. `except BaseException` and bare
    `except:` are NOT safe and are defects in async code.

    In a retry loop specifically, catch your own AppError base class, not
    Exception. Anything you did not anticipate should propagate — a retry loop
    that swallows KeyError is hiding a bug behind three extra seconds of
    latency.""")


# ---------------------------------------------------------------------------
# PART 3 — the adapter: translating a vendor SDK into your taxonomy
# ---------------------------------------------------------------------------

# Stand-ins for `openai.*` exception types. Kept local so the tutorial runs
# without the SDK installed.
class VendorAPIStatusError(Exception):
    def __init__(self, status_code: int, body: dict) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


class VendorAPIConnectionError(Exception):
    pass


def translate_vendor_error(exc: Exception) -> AppError:
    """ONE place that knows about the vendor's error shapes.

    Why this belongs in an adapter rather than at each call site:
      * Vendor SDKs reshape their hierarchies between major versions. When
        that happens you edit one function, not forty except-clauses.
      * Swapping Azure OpenAI for Bedrock or Anthropic means writing a second
        adapter, not rewriting resilience logic.
      * The status-code-to-meaning mapping is genuinely subtle (see the 400
        branch below) and deserves to be written down once, with comments.

    ALWAYS `raise ... from exc`. The wrapped error must keep the original
    traceback or you will debug the wrapper.
    """
    if isinstance(exc, VendorAPIConnectionError):
        return ConnectionLost(str(exc))

    if isinstance(exc, VendorAPIStatusError):
        code = exc.status_code
        err = exc.body.get("error", {})
        etype = err.get("code") or err.get("type") or ""

        if code == 429:
            # Two very different things share this code. A burst 429 is a
            # retry; an exhausted committed quota is an escalation, and
            # backing off against it just wastes your deadline.
            if "quota" in etype.lower() or "insufficient_quota" in etype:
                return QuotaExhausted(err.get("message", "quota exhausted"))
            return RateLimitError(
                err.get("message", "rate limited"),
                retry_after=exc.body.get("retry_after"),
            )

        if code in (500, 502, 503, 504):
            return ServiceUnavailable(f"HTTP {code}")

        if code in (401, 403):
            return AuthError(f"HTTP {code}: {err.get('message', '')}")

        if code == 400:
            # THE IMPORTANT BRANCH. Three different 400s, three dispositions.
            if etype == "context_length_exceeded":
                return ContextLengthExceeded(err.get("message", ""))
            if etype == "content_filter":
                return ContentFiltered(err.get("message", ""))
            return ValidationError(err.get("message", "bad request"))

        if code == 408:
            return UpstreamTimeout("HTTP 408")

    return ValidationError(f"unmapped vendor error: {exc!r}")


async def part3_adapter() -> None:
    banner("PART 3 — the adapter layer")

    cases = [
        VendorAPIStatusError(429, {"error": {"code": "rate_limit_exceeded"},
                                   "retry_after": 3.0}),
        VendorAPIStatusError(429, {"error": {"code": "insufficient_quota"}}),
        VendorAPIStatusError(400, {"error": {"code": "context_length_exceeded",
                                             "message": "too long by 4200"}}),
        VendorAPIStatusError(400, {"error": {"code": "content_filter"}}),
        VendorAPIStatusError(400, {"error": {"message": "unknown parameter"}}),
        VendorAPIStatusError(503, {"error": {}}),
        VendorAPIConnectionError("connection reset"),
    ]

    print(f"    {'vendor error':<44} {'mapped to':<24} disposition")
    print(f"    {'-' * 44} {'-' * 24} -----------")
    for v in cases:
        mapped = translate_vendor_error(v)
        label = f"{type(v).__name__}"
        if isinstance(v, VendorAPIStatusError):
            code = v.body.get("error", {}).get("code", "")
            label = f"{v.status_code} {code}"[:44]
        print(f"    {label:<44} {type(mapped).__name__:<24} {classify(mapped).value}")

    print("""
    Note the two 429 rows map to DIFFERENT dispositions, and the three 400
    rows map to three different ones. Any library that dispatches on status
    code alone gets all five of those wrong.""")


# ---------------------------------------------------------------------------
# PART 4 — exception chaining, and what you lose without it
# ---------------------------------------------------------------------------

async def part4_chaining() -> None:
    banner("PART 4 — `raise ... from` is not optional")

    def bad_wrap():
        try:
            raise VendorAPIStatusError(503, {"error": {}})
        except VendorAPIStatusError:
            raise ServiceUnavailable("upstream down")  # noqa: B904 — the bug

    def good_wrap():
        try:
            raise VendorAPIStatusError(503, {"error": {}})
        except VendorAPIStatusError as e:
            raise ServiceUnavailable("upstream down") from e

    section("without `from`")
    try:
        bad_wrap()
    except ServiceUnavailable as e:
        print(f"    __cause__ = {e.__cause__}")
        print(f"    __context__ = {e.__context__}   <- implicit, and suppressed in some formatters")

    section("with `from`")
    try:
        good_wrap()
    except ServiceUnavailable as e:
        print(f"    __cause__ = {e.__cause__}   <- explicit, always rendered")

    print("""
    Python does keep an implicit __context__, so the information is not
    strictly lost — but `from` sets __cause__, which is what "The above
    exception was the direct cause" renders from, and what most log formatters
    and APM agents serialise. Use `from`; it costs five characters.

    Use `from None` deliberately when the inner exception is genuinely noise
    (e.g. a parse failure on a value you are about to reject anyway) — but be
    aware you are destroying evidence, so do it consciously.""")


# ---------------------------------------------------------------------------
# PART 5 — error context: what to attach, and what never to attach
# ---------------------------------------------------------------------------

async def part5_context() -> None:
    banner("PART 5 — error context that makes a failure diagnosable")

    err = ContextLengthExceeded(
        "context window exceeded",
        context={
            # GOOD: identifiers, sizes, counts, config. All of it safe to log,
            # all of it needed to reproduce.
            "trace_id": "tr-9f2a41",
            "doc_id": "po-1001",
            "deployment": "gpt-4o-mini-prod",
            "prompt_tokens": 132_200,
            "limit": 128_000,
            "chunk_count": 47,
            "retrieval_k": 12,
            # NOT PRESENT, deliberately: the prompt itself, the retrieved
            # chunks, the model output, the user's question.
        },
    )

    print("    attached context:")
    for k, v in err.context.items():
        print(f"      {k:<16} {v}")

    print("""
    THE RULE: log identifiers and shapes, never bodies.

    Prompt text and retrieved chunks in your telemetry moves document content
    across a trust boundary your permission model does not cover — the people
    who can read the Application Insights workspace are not the same set as
    the people who could read the source document. An exception object is a
    very easy place to leak that accidentally, because everyone logs
    `exc.context` without reading it.

    What the context above buys you: `prompt_tokens=132200, limit=128000,
    chunk_count=47, retrieval_k=12` tells you the repair immediately — drop k
    to 8 — without anyone needing to see a single word of the document.""")


# ---------------------------------------------------------------------------
# PART 6 — where the boundary goes
# ---------------------------------------------------------------------------

async def part6_boundaries() -> None:
    banner("PART 6 — error boundaries: translate at the edge, dispatch in the core")

    print("""
    ┌─ adapter layer ──────────────────────────────────────────────┐
    │  openai.*, azure.*, httpx.*  ->  AppError subclasses          │
    │  ONE module. The only place vendor types appear.              │
    └───────────────────────────────────────────────────────────────┘
                                 │  AppError only
    ┌─ resilience layer ───────────┴───────────────────────────────┐
    │  retry / repair / fallback / breaker                          │
    │  Dispatches on Disposition. Knows nothing about HTTP.         │
    └───────────────────────────────────────────────────────────────┘
                                 │  AppError or value
    ┌─ domain layer ───────────────┴───────────────────────────────┐
    │  invoice matching, retrieval, extraction                      │
    │  Sees only its own errors + AppError.                         │
    └───────────────────────────────────────────────────────────────┘
                                 │  DomainError
    ┌─ API layer ──────────────────┴───────────────────────────────┐
    │  AppError -> HTTP status + safe client message                │
    │  Internal detail stripped here, and ONLY here.                │
    └───────────────────────────────────────────────────────────────┘

    Two rules that make this hold:

    1. Vendor exception types appear in exactly one module. Grep for
       `openai.` outside the adapter; every hit is a leak that will cost you
       on the next SDK upgrade.

    2. The resilience layer never inspects status codes. If retry logic
       contains `if e.status_code == 429`, the adapter is not doing its job
       and you now have classification in two places that will drift.""")

    # Demonstrate the API-layer mapping.
    section("AppError -> HTTP, at the outermost boundary")
    mapping = {
        Disposition.RETRY: (503, "Temporarily unavailable, please retry"),
        Disposition.REPAIR: (400, "Request could not be processed as submitted"),
        Disposition.FALLBACK: (503, "Temporarily unavailable"),
        Disposition.FAIL_FAST: (400, "Invalid request"),
        Disposition.ESCALATE: (500, "Internal configuration error"),
    }
    for e in [RateLimitError("x"), ContextLengthExceeded("x"),
              ContentFiltered("x"), AuthError("x")]:
        status, msg = mapping[classify(e)]
        print(f"      {type(e).__name__:<24} -> {status}  {msg!r}")

    print("""
      Note AuthError becomes a 500, not a 401. A 401 from YOUR api means the
      CALLER's credential is bad. Your service's inability to authenticate to
      Azure OpenAI is your problem, not theirs, and telling them otherwise
      sends them debugging the wrong thing.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_table()
    await part2_cancellation()
    await part3_adapter()
    await part4_chaining()
    await part5_context()
    await part6_boundaries()

    banner("SUMMARY")
    print("""
  * Five dispositions, not one boolean: RETRY, REPAIR, FALLBACK, FAIL_FAST,
    ESCALATE.
  * REPAIR is the category generic retry libraries cannot express, and it is
    where most GenAI reliability lives.
  * "Can I retry?" is two questions: useful, and SAFE. Side-effect-uncertain
    errors need idempotency keys, not just backoff.
  * One adapter module owns vendor exception types. The resilience layer never
    sees a status code.
  * `raise ... from exc`, always.
  * Error context carries identifiers and shapes, never bodies.
  * CancelledError is not an application error and must never be classified
    or retried.
""")


if __name__ == "__main__":
    asyncio.run(main())
