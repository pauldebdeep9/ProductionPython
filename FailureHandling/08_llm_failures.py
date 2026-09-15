"""
08 — LLM-specific failures: the ones no HTTP retry library can help with.

THE CATEGORY SHIFT
------------------
Everything so far treated the dependency as a black box that returns 200 or
errors. An LLM has a third mode, and it is the dominant one in practice:

    HTTP 200, and the content is wrong.

Nothing failed. No exception was raised. Your retry library saw a success.
And the response is unusable:

  * JSON that does not parse (fenced in markdown, trailing prose, truncated)
  * JSON that parses but violates the schema (missing field, wrong type,
    invented enum value)
  * finish_reason == "length" — the answer is cut off mid-sentence
  * a refusal ("I can't help with that") where a structured answer was
    expected
  * a tool call to a tool that does not exist, or with invented arguments
  * a confident answer with no grounding in the retrieved context

THE KEY DISTINCTION: RETRY vs REPAIR
------------------------------------
A RETRY sends the same request again. At temperature 0 that mostly returns the
same bad output — and note that temperature 0 is NOT deterministic at the API
level, so "mostly" is doing real work in that sentence. You may get lucky. It
is a coin flip you are paying for.

A REPAIR sends a DIFFERENT request informed by the failure: the parse error
appended, the schema restated, the context truncated, max_tokens raised. That
converts a ~50% coin flip into a high-probability fix.

Almost all GenAI reliability work is repair, not retry. This script builds the
repair ladder.

Run:  python 08_llm_failures.py
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from failure_lab import (
    Completion,
    FakeLLM,
    Metrics,
    SchemaValidationError,
    banner,
    section,
)

# ---------------------------------------------------------------------------
# The target schema — an invoice exception disposition
# ---------------------------------------------------------------------------

VALID_DISPOSITIONS = {
    "approve", "reject", "request_credit", "escalate_to_buyer",
    "hold_pending_receipt", "adjust_quantity", "no_action",
}

SCHEMA_DESCRIPTION = """{
  "disposition": one of ["approve","reject","request_credit",
                         "escalate_to_buyer","hold_pending_receipt",
                         "adjust_quantity","no_action"],
  "confidence": float between 0.0 and 1.0,
  "reason": string,
  "evidence_ids": array of strings
}"""


@dataclass
class Disposition:
    disposition: str
    confidence: float
    reason: str
    evidence_ids: list[str] = field(default_factory=list)


class ParseFailure(Exception):
    """Carries WHAT went wrong, in a form that can be fed back to the model.

    That last clause is the whole design. An error message written for a human
    ("invalid JSON at line 3") is much less useful in a repair prompt than one
    written for the model ("Your response was not valid JSON. The error was:
    Expecting ',' delimiter. Return ONLY the JSON object.").
    """

    def __init__(self, message: str, *, raw: str = "", repair_hint: str = "") -> None:
        super().__init__(message)
        self.raw = raw
        self.repair_hint = repair_hint or message


# ---------------------------------------------------------------------------
# PART 1 — the parse ladder: cheap fixes before expensive ones
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Try increasingly permissive strategies before spending another call.

    ORDER MATTERS — each rung costs more than the last:
      1. json.loads         free
      2. strip code fences  free, and fixes the single most common failure
      3. brace matching     free, handles leading/trailing prose
      4. (caller) repair    one extra LLM call
      5. (caller) fallback  a different model

    Skipping rungs 2-3 and going straight to a repair call is the common
    mistake. Markdown fencing accounts for a large share of parse failures and
    costs nothing to strip — paying for a model round trip to fix it is pure
    waste at scale.
    """
    # 1. straight parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. strip ```json fences — models add these constantly despite
    #    instructions, because their training data is full of them
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3. first balanced {...} block, for "Sure! Here is the JSON: {...}"
    start = text.find("{")
    if start >= 0:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise ParseFailure(
        "response was not valid JSON",
        raw=text,
        repair_hint=("Your previous response was not valid JSON. "
                     "Return ONLY a JSON object, with no markdown fences, "
                     "no explanation, and no text before or after it."),
    )


def validate(obj: dict) -> Disposition:
    """Validate against the schema, producing model-readable errors.

    In production use Pydantic — but note that a raw Pydantic ValidationError
    is verbose and structured for humans. Compress it into a short, imperative
    instruction before putting it in a repair prompt; a 400-line validation
    dump wastes context and reads worse than one sentence.
    """
    problems: list[str] = []

    d = obj.get("disposition")
    if d is None:
        problems.append("missing required field 'disposition'")
    elif d not in VALID_DISPOSITIONS:
        # THE HALLUCINATED-ENUM CASE. Models invent plausible-sounding values
        # ("partial_approve") constantly. Listing the valid set in the repair
        # message fixes it almost every time.
        problems.append(
            f"'disposition' was {d!r}, which is not one of the allowed values: "
            f"{sorted(VALID_DISPOSITIONS)}"
        )

    c = obj.get("confidence")
    if c is None:
        problems.append("missing required field 'confidence'")
    elif isinstance(c, str):
        # Models emit "0.85" or "85%" as strings routinely. Coerce what is
        # unambiguous rather than paying for a repair call.
        try:
            obj["confidence"] = float(c.rstrip("%")) / (100 if "%" in c else 1)
            c = obj["confidence"]
        except ValueError:
            problems.append(f"'confidence' was the string {c!r}; expected a float")
    if isinstance(c, (int, float)) and not 0.0 <= float(c) <= 1.0:
        problems.append(f"'confidence' was {c}; must be between 0.0 and 1.0")

    if not isinstance(obj.get("reason"), str) or not obj.get("reason"):
        problems.append("missing or empty field 'reason'")

    ev = obj.get("evidence_ids", [])
    if not isinstance(ev, list):
        problems.append("'evidence_ids' must be an array of strings")

    if problems:
        raise SchemaValidationError(
            "; ".join(problems),
            context={"problems": problems},
        )

    return Disposition(
        disposition=obj["disposition"],
        confidence=float(obj["confidence"]),
        reason=obj["reason"],
        evidence_ids=[str(x) for x in ev],
    )


async def part1_parse_ladder() -> None:
    banner("PART 1 — the free repairs, before you spend a model call")

    cases = {
        "clean JSON": '{"disposition":"approve","confidence":0.9,'
                      '"reason":"quantities match","evidence_ids":["c1"]}',
        "markdown fenced": '```json\n{"disposition":"approve","confidence":0.9,'
                           '"reason":"ok","evidence_ids":[]}\n```',
        "prose preamble": 'Sure! Here is the result:\n'
                          '{"disposition":"reject","confidence":0.7,'
                          '"reason":"no receipt","evidence_ids":[]}\n'
                          'Let me know if you need more.',
        "confidence as string": '{"disposition":"approve","confidence":"0.85",'
                                '"reason":"ok","evidence_ids":[]}',
        "confidence as percent": '{"disposition":"approve","confidence":"85%",'
                                 '"reason":"ok","evidence_ids":[]}',
        "hallucinated enum": '{"disposition":"partial_approve","confidence":0.6,'
                             '"reason":"partial","evidence_ids":[]}',
        "truncated": '{"disposition":"approve","confidence":0.9,"reason":"the '
                     'quantities on the goods receipt mat',
    }

    for label, raw in cases.items():
        try:
            obj = extract_json(raw)
            d = validate(obj)
            print(f"    OK       {label:<22} -> {d.disposition} "
                  f"@ {d.confidence}")
        except ParseFailure as e:
            print(f"    PARSE    {label:<22} -> {e}")
        except SchemaValidationError as e:
            print(f"    SCHEMA   {label:<22} -> {e}")

    print("""
    Five of these seven were fixed for FREE — fence stripping, brace matching,
    and string coercion. Only the hallucinated enum and the truncation need a
    model call, and they need DIFFERENT ones: the enum needs a re-prompt with
    the valid values, the truncation needs a higher max_tokens.

    Measure your own distribution before building anything clever. In most
    systems markdown fencing alone is the largest single bucket, and it is a
    three-line fix.""")


# ---------------------------------------------------------------------------
# PART 2 — the repair loop
# ---------------------------------------------------------------------------

async def call_with_repair(
    llm: FakeLLM,
    prompt: str,
    *,
    max_repairs: int = 2,
    max_tokens: int = 256,
    metrics: Metrics | None = None,
) -> tuple[Disposition, list[str]]:
    """Call, parse, validate; on failure REPAIR the prompt and try again.

    The repair prompt includes:
      * the original task
      * the model's own bad output
      * the specific error
      * an imperative instruction

    That is materially different from retrying the same prompt, and it is why
    this converges where a retry does not.

    BUDGET IT. max_repairs=2 is usually right. If two repairs do not fix it,
    the problem is the prompt or the schema, not the sample — and a third
    attempt is just money. Emit `repairs_used` as a metric: a rising repair
    rate means your prompt has drifted or the model changed under you.
    """
    trail: list[str] = []
    current = prompt
    tokens = max_tokens

    for _attempt in range(max_repairs + 1):
        if metrics:
            metrics.incr("llm_calls")
        completion: Completion = await llm.complete(current, max_tokens=tokens)

        # CHECK finish_reason BEFORE parsing. Truncated JSON produces a
        # confusing parse error when the real problem is the token limit.
        if completion.finish_reason == "length":
            trail.append("truncated")
            if metrics:
                metrics.incr("repair.truncated")
            tokens *= 2                      # the repair IS raising the limit
            current = prompt
            continue

        try:
            obj = extract_json(completion.text)
            result = validate(obj)
            trail.append("ok")
            if metrics:
                metrics.incr("parse.ok")
            return result, trail

        except ParseFailure as e:
            trail.append("parse_failure")
            if metrics:
                metrics.incr("repair.parse")
            current = (
                f"{prompt}\n\n"
                f"Your previous response was:\n{completion.text[:400]}\n\n"
                f"{e.repair_hint}"
            )

        except SchemaValidationError as e:
            trail.append("schema_failure")
            if metrics:
                metrics.incr("repair.schema")
            current = (
                f"{prompt}\n\n"
                f"Your previous response was:\n{completion.text[:400]}\n\n"
                f"It did not match the required schema. Problems: {e}\n"
                f"The schema is:\n{SCHEMA_DESCRIPTION}\n"
                f"Return ONLY a corrected JSON object."
            )

    if metrics:
        metrics.incr("repair.exhausted")
    raise SchemaValidationError(
        f"could not obtain valid output after {max_repairs + 1} attempts",
        context={"trail": trail},
    )


async def part2_repair_loop() -> None:
    banner("PART 2 — repair converges where retry does not")

    m = Metrics()

    section("bad JSON, then correct after one repair")
    llm = FakeLLM("gpt-4o-mini", respond_with=[
        "Here you go! ```json\n{'disposition': 'approve',}\n```",       # broken
        '{"disposition":"approve","confidence":0.88,'
        '"reason":"GR matches PO","evidence_ids":["gr-55"]}',           # fixed
    ])
    d, trail = await call_with_repair(llm, "Classify this exception.", metrics=m)
    print(f"      trail: {trail}")
    print(f"      result: {d.disposition} @ {d.confidence} — {d.reason}")

    section("hallucinated enum, corrected by restating valid values")
    llm2 = FakeLLM("gpt-4o-mini", respond_with=[
        '{"disposition":"partial_approve","confidence":0.6,'
        '"reason":"partial","evidence_ids":[]}',
        '{"disposition":"adjust_quantity","confidence":0.72,'
        '"reason":"2 unit overbill","evidence_ids":["inv-88"]}',
    ])
    d2, trail2 = await call_with_repair(llm2, "Classify.", metrics=m)
    print(f"      trail: {trail2}")
    print(f"      result: {d2.disposition} @ {d2.confidence}")

    section("never converges — repair budget exhausted, fails cleanly")
    llm3 = FakeLLM("gpt-4o-mini", respond_with=["not json at all, ever"])
    try:
        await call_with_repair(llm3, "Classify.", max_repairs=2, metrics=m)
    except SchemaValidationError as e:
        print(f"      {e}")
        print(f"      trail: {e.context['trail']}")

    print("\n      metrics: " + m.report(
        "llm_calls", "parse.ok", "repair.parse",
        "repair.schema", "repair.exhausted"))
    print("""
      Emit repair.* counters. A rising repair rate is the earliest signal that
      a model version changed under you, or that a prompt edit regressed
      output format — usually days before anyone notices bad answers.""")


# ---------------------------------------------------------------------------
# PART 3 — truncation
# ---------------------------------------------------------------------------

async def part3_truncation() -> None:
    banner("PART 3 — finish_reason is a failure signal you must check")

    print("""
    THE BUG: code that reads `response.choices[0].message.content` and never
    looks at `finish_reason`. When the model hits the token limit you get a
    partial answer that LOOKS complete — a sentence that stops mid-clause, or
    JSON missing its closing brace.

    For prose the user sees a truncated answer. For structured output you get
    a parse error that sends you debugging your parser when the real problem
    is max_tokens.

    finish_reason values and what each means:
      stop            normal completion. The only good one.
      length          hit max_tokens. REPAIRABLE: raise the limit, or ask for
                      a continuation, or reduce what you asked for.
      content_filter  output was filtered. NOT repairable by retry; the same
                      prompt produces the same filtered output.
      tool_calls      the model wants a tool. Not a failure — but code that
                      only handles 'stop' treats it as an empty response.

    ALWAYS CHECK IT FIRST, before parsing. A truncation check is one line and
    it converts a mystifying parse error into an obvious cause.""")

    section("truncated structured output, repaired by raising max_tokens")

    class TruncatingLLM(FakeLLM):
        """Returns truncated JSON until max_tokens is large enough."""

        async def complete(self, prompt: str, *, max_tokens: int = 256) -> Completion:
            self.calls += 1
            full = ('{"disposition":"adjust_quantity","confidence":0.81,'
                    '"reason":"invoice billed 402 units against a 400 unit '
                    'goods receipt","evidence_ids":["inv-88","gr-55"]}')
            if max_tokens < 40:
                cut = full[:max_tokens * 4]
                return Completion(text=cut, finish_reason="length",
                                  completion_tokens=max_tokens)
            return Completion(text=full, finish_reason="stop",
                              completion_tokens=len(full) // 4)

    m = Metrics()
    llm = TruncatingLLM("gpt-4o-mini")
    d, trail = await call_with_repair(llm, "Classify.", max_tokens=10,
                                      max_repairs=3, metrics=m)
    print(f"      trail: {trail}")
    print("      max_tokens doubled each time until the answer fit")
    print(f"      result: {d.disposition} — {d.reason[:44]}...")
    print("""
      Note the trail shows 'truncated' BEFORE any parse attempt. Had we parsed
      first we would have seen 'Expecting , delimiter' and gone looking for a
      JSON bug that does not exist.""")


# ---------------------------------------------------------------------------
# PART 4 — refusals, and why they are not errors
# ---------------------------------------------------------------------------

async def part4_refusals() -> None:
    banner("PART 4 — refusals: HTTP 200, no data, and retrying is wrong")

    REFUSAL_MARKERS = (
        "i can't", "i cannot", "i'm unable", "i am unable",
        "i won't", "as an ai", "i'm not able",
    )

    def looks_like_refusal(text: str) -> bool:
        """Heuristic, and worth being honest about its limits.

        This is string matching. It has false positives (a document ABOUT
        refusals), false negatives (a politely-phrased decline), and it is
        language-specific. Newer APIs expose a structured `refusal` field —
        use that where available and treat this as a fallback.

        The reason to have it at all: without it, a refusal flows into your
        parser as 'not valid JSON', triggers repairs, burns the budget, and
        finally surfaces as a schema error. The operator then investigates a
        parsing problem that does not exist.
        """
        t = text.lower()[:200]
        return any(mark in t for mark in REFUSAL_MARKERS)

    samples = [
        ('{"disposition":"approve","confidence":0.9,"reason":"ok",'
         '"evidence_ids":[]}', "structured answer"),
        ("I can't help with classifying this invoice.", "refusal"),
        ("I'm unable to provide that information.", "refusal"),
    ]
    for text, label in samples:
        print(f"      {label:<20} refusal_detected={looks_like_refusal(text)}")

    print("""
    HANDLING, in order of preference:
      1. Detect it explicitly and classify it as its own outcome — NOT as a
         parse error, and NOT as a success.
      2. Do not retry the identical prompt; a refusal is stable.
      3. Consider a repair: if the refusal is caused by ambiguity or by
         something in the retrieved context, a reformulated prompt may work.
         This is a REPAIR, and it should be bounded to one attempt.
      4. Surface it honestly to the user. "The model declined to answer this"
         is a real outcome and a user can act on it. Silently returning an
         empty result is not.

    METRIC: track refusal_rate separately from error_rate and from
    parse_failure_rate. A sudden jump in refusals usually means a prompt
    change, a model version change, or a shift in the input distribution —
    and it will not show up in any of your other dashboards.""")


# ---------------------------------------------------------------------------
# PART 5 — context length: a repair, not a retry
# ---------------------------------------------------------------------------

async def part5_context_length() -> None:
    banner("PART 5 — context overflow is repairable, and the repair is a choice")

    @dataclass
    class Chunk:
        id: str
        tokens: int
        score: float

    chunks = [Chunk(f"c{i}", 1200, 1.0 - i * 0.05) for i in range(20)]
    budget = 8000

    def fit_by_dropping_lowest(chunks: list[Chunk], budget: int) -> list[Chunk]:
        """Strategy A: keep the highest-scoring chunks that fit.

        Simple, preserves relevance ranking. Risk: drops a chunk that was the
        only source for part of the answer, so the model answers confidently
        from partial evidence.
        """
        kept, used = [], 0
        for c in sorted(chunks, key=lambda x: -x.score):
            if used + c.tokens <= budget:
                kept.append(c)
                used += c.tokens
        return kept

    def fit_by_summarising(chunks: list[Chunk], budget: int) -> list[Chunk]:
        """Strategy B: keep the top few verbatim, summarise the rest.

        Preserves coverage at the cost of an extra model call and some
        fidelity. Note it BREAKS CITATION: you can no longer point at an exact
        passage for the summarised portion, which matters a lot if your
        answers must be auditable.
        """
        kept = fit_by_dropping_lowest(chunks, int(budget * 0.7))
        remaining = [c for c in chunks if c not in kept]
        if remaining:
            kept.append(Chunk(f"summary-of-{len(remaining)}", 800, 0.0))
        return kept

    a = fit_by_dropping_lowest(chunks, budget)
    b = fit_by_summarising(chunks, budget)
    print(f"      input: {len(chunks)} chunks, {sum(c.tokens for c in chunks)} tokens")
    print(f"      budget: {budget} tokens")
    print(f"      A) drop lowest-scoring : kept {len(a)}, "
          f"{sum(c.tokens for c in a)} tokens, citations intact")
    print(f"      B) summarise the tail  : kept {len(b)}, "
          f"{sum(c.tokens for c in b)} tokens, +1 model call, "
          f"citations lost for the tail")

    print("""
    THE POINT: 'context_length_exceeded' has several valid repairs and they
    have different costs. Which one you pick is a product decision:

      reduce retrieval k        cheapest; may lose evidence
      drop lowest-scoring       cheap; preserves citations
      summarise the tail        preserves coverage; breaks citation for the
                                summarised part; costs an extra call
      map-reduce over chunks    highest fidelity; N+1 calls; slowest
      route to a longer-context
      model                     no information loss; higher cost per token

    Whatever you choose, do the arithmetic BEFORE calling rather than catching
    the 400 afterwards. Counting tokens is cheap; a rejected request is a
    wasted round trip and, at scale, real latency.

    AND CAP RETRIEVAL PROPERLY: the common bug is `k=20` with variable-length
    chunks. Twenty short chunks fit; twenty long ones do not. Budget by TOKENS,
    not by chunk count.""")


# ---------------------------------------------------------------------------
# PART 6 — the failure taxonomy for LLM output
# ---------------------------------------------------------------------------

async def part6_summary_table() -> None:
    banner("PART 6 — LLM output failures and their correct handling")

    rows = [
        ("markdown-fenced JSON", "strip fences", "free", "no"),
        ("prose around JSON", "brace extraction", "free", "no"),
        ("number as string", "coerce", "free", "no"),
        ("missing field", "repair w/ schema", "1 call", "no"),
        ("hallucinated enum", "repair w/ valid values", "1 call", "no"),
        ("finish_reason=length", "raise max_tokens", "1 call", "no"),
        ("context_length_exceeded", "re-chunk / reduce k", "0-1 call", "no"),
        ("refusal", "detect, surface honestly", "0 calls", "no"),
        ("content_filter", "fail fast", "0 calls", "no"),
        ("tool not in registry", "repair w/ tool list", "1 call", "no"),
        ("ungrounded answer", "verify vs context, reject", "0-1 call", "no"),
        ("429 / 503 / timeout", "retry w/ backoff", "n calls", "YES"),
    ]
    print(f"    {'failure':<26} {'handling':<26} {'cost':<10} retry?")
    print(f"    {'-' * 26} {'-' * 26} {'-' * 10} ------")
    for a, b, c, d in rows:
        print(f"    {a:<26} {b:<26} {c:<10} {d}")

    print("""
    ELEVEN of the twelve are NOT retries. That ratio is the argument of this
    whole script: for a GenAI system, generic HTTP retry machinery covers the
    last row and nothing else.

    AND THE ONE THAT IS NOT IN THIS TABLE: a well-formed, schema-valid,
    confidently-worded answer that is simply WRONG. No parser catches it. The
    controls for that are grounding checks (does every claim cite retrieved
    context?), a deterministic verification gate that re-derives the answer
    independently rather than trusting the model, and an evaluation harness
    that measures how often it happens. Those are correctness controls, not
    failure-handling ones, but they belong on the same risk register — and a
    PoC that demonstrates the pipeline runs without measuring how often it is
    RIGHT cannot support a funding decision.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_parse_ladder()
    await part2_repair_loop()
    await part3_truncation()
    await part4_refusals()
    await part5_context_length()
    await part6_summary_table()

    banner("SUMMARY")
    print("""
  * HTTP 200 with unusable content is the dominant LLM failure mode.
  * REPAIR (a different request, informed by the error) beats RETRY (the same
    request). Temperature 0 is not deterministic, so retry is a coin flip.
  * Do the free repairs first: strip fences, extract braces, coerce types.
    Markdown fencing is usually the largest single bucket.
  * Check finish_reason BEFORE parsing, or truncation masquerades as a parse
    bug.
  * Bound the repair budget at ~2. Beyond that the prompt is the problem.
  * Refusals are their own outcome — not errors, not successes. Track
    refusal_rate separately.
  * Context overflow: compute the budget before calling; pick the repair
    deliberately, because they differ in cost and in whether citations survive.
  * Emit repair.* counters. Rising repair rates detect model and prompt drift
    days before anyone notices bad answers.
""")


if __name__ == "__main__":
    asyncio.run(main())
