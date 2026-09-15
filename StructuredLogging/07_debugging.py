"""
07 — The payoff: answering real questions from telemetry.

THE TEST OF AN OBSERVABILITY DESIGN
-----------------------------------
Not "did we emit a lot of data" but "can we answer the question a person
actually asks at 3am, in one query, without deploying anything".

This script generates a corpus of traces containing several PLANTED problems,
then works through eight real questions against it. Each one names the field
that makes the answer possible — which is the whole argument for having
emitted it.

If you can only take one thing from this tutorial, take the habit of asking
"what query will I run when this breaks?" BEFORE choosing what to instrument.
Instrumentation chosen that way is dramatically smaller and more useful than
instrumentation chosen by adding an attribute every time someone is confused.

Run:  python 07_debugging.py
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from obs_lab import (
    SpanKind,
    StatusCode,
    TelemetryStore,
    Tracer,
    banner,
    new_trace_id,
    section,
    show,
    trace_id_var,
)

STORE = TelemetryStore()
TRACER = Tracer(STORE)


# ---------------------------------------------------------------------------
# Generate a corpus with planted problems
# ---------------------------------------------------------------------------

def generate_corpus(n: int = 600, seed: int = 17) -> None:
    """Six planted problems, mixed into otherwise healthy traffic.

    They are NOT announced. The queries below have to find them, which is the
    point — a dashboard that only shows you what you already knew to look for
    is not observability.
    """
    rng = random.Random(seed)
    tenants = ["isc-sg", "isc-eu", "isc-us"]

    for i in range(n):
        trace_id_var.set(new_trace_id())
        tenant = rng.choice(tenants)

        # PLANT 1: from request 400 onward, isc-eu is routed to a deployment
        # whose model_version differs from the request model.
        drifted = i >= 400 and tenant == "isc-eu"
        # PLANT 2: a prompt version change at request 300 raises token usage.
        prompt_version = "v3" if i < 300 else "v4"
        # PLANT 3: one document dominates retrieval after request 200.
        hot_doc = i >= 200 and rng.random() < 0.55
        # PLANT 4: permission trimming silently stops for isc-us after 500.
        trimming_broken = i >= 500 and tenant == "isc-us"
        # PLANT 5: a slow reranker for large candidate sets.
        candidates = rng.choice([20, 20, 20, 200])
        # PLANT 6: schema repairs rise sharply with the v4 prompt.
        needs_repair = rng.random() < (0.03 if prompt_version == "v3" else 0.22)

        with TRACER.span("POST /answer", SpanKind.SERVER,
                         **{"tenant": tenant, "http.route": "/answer"}) as root:
            with TRACER.span("retrieve", SpanKind.CLIENT,
                             **{"peer.service": "azure-ai-search"}) as s:
                candidate_count = candidates
                trimmed = 0 if trimming_broken else rng.randint(1, 4)
                # A 40-chunk corpus, so baseline appearance for any one doc
                # is ~7%. With only 4 chunks every doc appears in ~75% of
                # requests and "dominance" is an artifact of the generator,
                # not a signal — which is exactly the mistake this query is
                # meant to catch, so the fixture must not contain it.
                pool = [f"c{k}" for k in range(1, 41)]
                chunk_ids = (["c3", *rng.sample(pool, 2)] if hot_doc
                             else rng.sample(pool, 3))
                s.set("retrieval.candidate_count", candidate_count)
                s.set("retrieval.chunk_count", len(chunk_ids))
                s.set("retrieval.trimmed_count", trimmed)
                s.set("retrieval.chunk_ids", chunk_ids)
                s.set("retrieval.doc_ids",
                      ["inv-88" if c == "c3" else f"doc-{c}" for c in chunk_ids])
                STORE.incr("retrieval_trimmed_total", trimmed, tenant=tenant)

            with TRACER.span("rerank", SpanKind.INTERNAL) as s:
                s.set("rerank.input_count", candidate_count)
                # Latency scales with the candidate count — the planted cause.
                dur = 0.0006 * candidate_count
                _busy(dur)
                s.set("rerank.model", "cross-encoder-v2")

            with TRACER.span("generate", SpanKind.CLIENT,
                             **{"peer.service": "azure-openai"}) as s:
                s.set("gen_ai.request.model", "gpt-4o-mini")
                s.set("gen_ai.response.model",
                      "gpt-4o-mini-2024-11-20" if drifted
                      else "gpt-4o-mini-2024-07-18")
                s.set("isc.prompt.version", prompt_version)
                s.set("isc.prompt.fingerprint",
                      "36206a029343" if prompt_version == "v3" else "9b41ce70a2ff")
                base_in = 1600 if prompt_version == "v3" else 2450
                s.set("gen_ai.usage.input_tokens",
                      base_in + rng.randint(-120, 120))
                s.set("gen_ai.usage.output_tokens", rng.randint(60, 180))
                s.event("first_token")
                _busy(0.0008)
                if needs_repair:
                    s.event("repair.attempted", kind="schema", attempt=1)
                    STORE.incr("repairs_total", kind="schema",
                               prompt_version=prompt_version)
                s.set("gen_ai.response.finish_reasons", ["stop"])
                if rng.random() < 0.02:
                    s.set("outcome", "error")
                    s.status = StatusCode.ERROR
                    s.status_message = "ServiceUnavailable: 503"

            root.set("outcome", "error" if any(
                sp.status is StatusCode.ERROR for sp in STORE.spans[-1:]
            ) else "ok")
            root.set("isc.prompt.version", prompt_version)
            STORE.incr("requests_total", tenant=tenant)


def _busy(seconds: float) -> None:
    import time
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------

def q1_single_trace() -> None:
    banner('Q1 — "The answer I got at 14:32 was wrong. Trace id abc123."')

    tid = STORE.traces()[42]
    section(f"the waterfall for {tid[:12]}")
    print(STORE.render_trace(tid))

    section("the attributes that explain the answer")
    for s in STORE.trace(tid):
        if s.name in ("retrieve", "generate"):
            interesting = {k: v for k, v in s.attributes.items()
                           if k.startswith(("retrieval.", "isc.prompt",
                                            "gen_ai.response.model"))}
            print(f"      {s.name}: {interesting}")

    print("""
    ENABLED BY: the `x-trace-id` response header from script 02. Without it
    the user says "around 2:30pm" and you are searching by timestamp across
    every tenant.

    THE CHUNK IDS ARE THE ANSWER HERE. Fetch those chunks from the index —
    under your own permission checks — and you can see exactly what the model
    was given, without ever having stored the content.""")


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------

def q2_latency_regression() -> None:
    banner('Q2 — "p95 latency doubled this afternoon. Where?"')

    section("p95 per span name")
    names = ["POST /answer", "retrieve", "rerank", "generate"]
    for name in names:
        p50 = STORE.percentile(name, 0.50)
        p95 = STORE.percentile(name, 0.95)
        print(f"      {name:<16} p50={p50:>7.2f}ms  p95={p95:>7.2f}ms  "
              f"ratio={p95 / p50 if p50 else 0:>5.1f}x")

    section("rerank, split by input count")
    by_size: dict[int, list[float]] = defaultdict(list)
    for s in STORE.where(name="rerank"):
        by_size[int(s.attributes["rerank.input_count"])].append(s.duration_ms)
    for size, durations in sorted(by_size.items()):
        durations.sort()
        print(f"      input_count={size:<5} n={len(durations):<5} "
              f"p50={durations[len(durations)//2]:>7.2f}ms  "
              f"p95={durations[int(0.95*len(durations))]:>7.2f}ms")

    print("""
    FOUND IT: `rerank` has by far the worst p95/p50 ratio, and splitting by
    `rerank.input_count` shows why — a bimodal candidate count, with the large
    bucket an order of magnitude slower.

    ENABLED BY: `rerank.input_count` as a span attribute. Without it, rerank
    is just "sometimes slow" and you would go looking for a GC pause or a
    noisy neighbour. The fix is a candidate cap, not a bigger machine.

    THE GENERAL PATTERN: a high p95/p50 ratio means a BIMODAL distribution,
    which almost always means an input-dependent code path. Find the input
    dimension and split by it.""")


# ---------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------

def q3_silent_model_change() -> None:
    banner('Q3 — "Answers got worse last Tuesday. Nobody deployed anything."')

    section("requested model vs served model")
    pairs: Counter[tuple[str, str]] = Counter()
    for s in STORE.where(name="generate"):
        pairs[(s.attributes.get("gen_ai.request.model", "?"),
               s.attributes.get("gen_ai.response.model", "?"))] += 1
    for (req, resp), n in pairs.most_common():
        flag = "  <-- DIVERGED" if not resp.startswith(req) or req not in resp else ""
        print(f"      requested={req:<14} served={resp:<26} n={n:<5}{flag}")

    section("which tenant")
    by_tenant: Counter[tuple[str, str]] = Counter()
    for s in STORE.where(name="generate"):
        root = next((r for r in STORE.trace(s.trace_id)
                     if r.kind is SpanKind.SERVER), None)
        if root:
            by_tenant[(root.attributes.get("tenant", "?"),
                       s.attributes.get("gen_ai.response.model", "?"))] += 1
    for (tenant, model), n in sorted(by_tenant.items()):
        print(f"      {tenant:<10} {model:<28} {n}")

    print("""
    FOUND IT: isc-eu is being served by a different model snapshot. The
    deployment auto-updated, or traffic is routed to a different region's
    deployment. No code changed; the outputs did.

    ENABLED BY: recording BOTH `gen_ai.request.model` and
    `gen_ai.response.model`. This is the single highest-value pair of
    attributes in an LLM system, because it is the only way to detect a change
    that originates outside your deploy pipeline — and it is invisible in
    every other signal.""")


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------

def q4_cost_spike() -> None:
    banner('Q4 — "Token spend is up 50% and traffic is flat."')

    section("input tokens by prompt version")
    by_version: dict[str, list[int]] = defaultdict(list)
    for s in STORE.where(name="generate"):
        v = s.attributes.get("isc.prompt.version", "?")
        by_version[v].append(int(s.attributes.get("gen_ai.usage.input_tokens", 0)))
    for v, tokens in sorted(by_version.items()):
        print(f"      prompt {v}: n={len(tokens):<5} "
              f"mean_input_tokens={sum(tokens)/len(tokens):>8.0f}")

    versions = sorted(by_version)
    if len(versions) == 2:
        a, b = (sum(by_version[v]) / len(by_version[v]) for v in versions)
        show("increase", f"{(b - a) / a:+.1%} from {versions[0]} to {versions[1]}")

    section("the fingerprints, so the change is identifiable")
    fps: Counter[tuple[str, str]] = Counter()
    for s in STORE.where(name="generate"):
        fps[(s.attributes.get("isc.prompt.version", "?"),
             s.attributes.get("isc.prompt.fingerprint", "?"))] += 1
    for (v, fp), n in sorted(fps.items()):
        print(f"      {v} -> {fp}  n={n}")

    print("""
    FOUND IT: the v4 prompt template uses substantially more input tokens per
    request. Traffic is flat; the prompt got longer.

    ENABLED BY: `isc.prompt.version` and `isc.prompt.fingerprint` on every
    generation span. Without them, the token increase is real and
    unattributable — you know spend rose and have no way to connect it to a
    change.

    THE FINGERPRINT IS WHAT MAKES IT ACTIONABLE. A version string can be
    forgotten when someone edits the template in place; the hash cannot.""")


# ---------------------------------------------------------------------------
# Q5
# ---------------------------------------------------------------------------

def q5_permission_trimming() -> None:
    banner('Q5 — "Is permission trimming still working?" (nobody asks this)')

    section("trimmed_count by tenant")
    by_tenant: dict[str, list[int]] = defaultdict(list)
    for s in STORE.where(name="retrieve"):
        root = next((r for r in STORE.trace(s.trace_id)
                     if r.kind is SpanKind.SERVER), None)
        if root:
            by_tenant[root.attributes.get("tenant", "?")].append(
                int(s.attributes.get("retrieval.trimmed_count", 0)))
    rates = {t: sum(1 for v in vs if v == 0) / len(vs)
             for t, vs in by_tenant.items()}
    # A FLEET-RELATIVE threshold, which is what a real alert uses. An absolute
    # one ("> 50% zero-trim") is wrong because the correct baseline differs per
    # tenant: a tenant whose users can see everything legitimately trims zero.
    # What is anomalous is a tenant DIVERGING from its peers.
    peer_median = sorted(rates.values())[len(rates) // 2]
    for tenant, values in sorted(by_tenant.items()):
        zeros = sum(1 for v in values if v == 0)
        rate = zeros / len(values)
        anomaly = rate > max(0.05, peer_median * 3 + 0.05)
        print(f"      {tenant:<10} n={len(values):<5} "
              f"mean={sum(values)/len(values):>5.2f}  "
              f"zero_trim={zeros:>4} ({rate:>6.1%})"
              + ("   <-- ANOMALY vs peers" if anomaly else ""))
    show("peer median zero-trim rate", f"{peer_median:.1%}")

    print("""
    FOUND IT: isc-us has a large fraction of requests where NOTHING was
    trimmed, while the other tenants trim on nearly every request. Something
    stopped applying the filter for that tenant.

    THIS IS THE MOST IMPORTANT QUERY IN THE FILE, and the one nobody runs.
    Every affected request returned a fluent, well-cited, completely normal
    answer. There is no error, no latency change, no failed assertion. The
    ONLY externally visible symptom is this integer.

    ENABLED BY: `retrieval.trimmed_count` on every retrieval span.

    SO MAKE IT AN ALERT, not a query someone might run. And make it
    BASELINE-RELATIVE rather than absolute:

        alert when a tenant's zero-trim rate exceeds 3x the fleet median
        over 15 minutes

    An absolute threshold is wrong because the correct baseline genuinely
    differs per tenant — a tenant whose users can all see everything trims
    zero legitimately, and would page you forever. What is anomalous is one
    tenant DIVERGING from its own history and from its peers.

    A permission failure that fails SILENTLY and UPWARD has to be detected by
    a metric, because it will never page you any other way.""")


# ---------------------------------------------------------------------------
# Q6
# ---------------------------------------------------------------------------

def q6_repair_rate() -> None:
    banner('Q6 — "Are we quietly repairing more model output than before?"')

    section("repairs by prompt version")
    repairs: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for s in STORE.where(name="generate"):
        v = str(s.attributes.get("isc.prompt.version", "?"))
        totals[v] += 1
        if any(e.name == "repair.attempted" for e in s.events):
            repairs[v] += 1
    for v in sorted(totals):
        rate = repairs[v] / totals[v]
        print(f"      prompt {v}: {repairs[v]:>4}/{totals[v]:<5} = {rate:>6.1%}"
              + ("   <-- REGRESSION" if rate > 0.1 else ""))

    print("""
    FOUND IT: the v4 prompt produces schema-invalid output roughly seven times
    as often. Every one of those was silently repaired, so the user-visible
    success rate is unchanged — and the extra latency and extra model call are
    being paid on a fifth of requests.

    ENABLED BY: `repair.attempted` as a span EVENT rather than a silent retry.
    A repair loop that does not emit anything is invisible by construction.

    THIS IS AN EARLY-WARNING SIGNAL. A rising repair rate precedes visible
    quality problems, usually by days, because repairs mostly succeed. It is
    the same shape as retry amplification preceding error rate.""")


# ---------------------------------------------------------------------------
# Q7
# ---------------------------------------------------------------------------

def q7_document_dominance() -> None:
    banner('Q7 — "Is one document dominating our answers?"')

    section("appearance rate by doc_id")
    docs: Counter[str] = Counter()
    n_requests = 0
    for s in STORE.where(name="retrieve"):
        n_requests += 1
        for d in set(s.attributes.get("retrieval.doc_ids", [])):
            docs[d] += 1
    rates = [n / n_requests for _, n in docs.most_common()]
    median = sorted(rates)[len(rates) // 2]
    for doc, n in docs.most_common(6):
        rate = n / n_requests
        print(f"      {doc:<14} {n:>5}/{n_requests}  {rate:>6.1%}"
              + ("   <-- DOMINANT" if rate > median * 4 else ""))
    show("median doc appearance rate", f"{median:.1%}")

    print("""
    FOUND IT: `inv-88` appears in roughly 42% of answers against a median of
    ~8% for every other document — a 5x outlier. Possible causes: a chunking
    bug that made it enormous, an embedding collapse, a duplicate ingestion,
    or a genuinely central document.

    NOTE THE THRESHOLD IS RELATIVE, as in Q5. An absolute one ("> 50%") is
    unusable because the right baseline depends entirely on your corpus size:
    with 40 documents, 8% is normal; with 4, everything appears in 75% of
    answers and the query means nothing. Compare against the median, not
    against a number you picked.

    ENABLED BY: `retrieval.doc_ids` — ids, not content.

    WHY IT MATTERS BEYOND CURIOSITY: a document that appears in most answers
    is a single point of failure for answer quality. If it is stale or wrong,
    so is most of your output, and no per-request signal will tell you.""")


# ---------------------------------------------------------------------------
# Q8
# ---------------------------------------------------------------------------

def q8_error_attribution() -> None:
    banner('Q8 — "Which dependency is causing our errors?"')

    section("errors by span name and peer service")
    errs: Counter[tuple[str, str]] = Counter()
    for s in STORE.errors():
        errs[(s.name, str(s.attributes.get("peer.service", "-")))] += 1
    for (name, peer), n in errs.most_common():
        print(f"      {name:<14} peer={peer:<20} {n:>4}")

    section("error rate vs total")
    total_gen = len(STORE.where(name="generate"))
    gen_errs = sum(n for (name, _), n in errs.items() if name == "generate")
    show("generate error rate", f"{gen_errs}/{total_gen} = {gen_errs/total_gen:.1%}")

    print("""
    ENABLED BY: `peer.service` on CLIENT spans, plus a status discipline where
    ERROR means the operation failed rather than "something went wrong
    somewhere inside".

    THE THING TO CHECK NEXT, and it is a trap: is this the error rate of the
    OPERATION or of the individual attempts? If retries happen inside the
    span, a 2% span error rate can sit on top of a 30% attempt failure rate.
    Both numbers matter and they answer different questions — the first is
    what users experience, the second is what the dependency is doing.""")


def main() -> None:
    generate_corpus()
    show("traces generated", len(STORE.traces()))
    show("spans generated", len(STORE.spans))

    q1_single_trace()
    q2_latency_regression()
    q3_silent_model_change()
    q4_cost_spike()
    q5_permission_trimming()
    q6_repair_rate()
    q7_document_dominance()
    q8_error_attribution()

    banner("WHAT MADE EACH ANSWER POSSIBLE")
    print("""
    Q1  a bad answer          <- x-trace-id header + retrieval.chunk_ids
    Q2  latency regression    <- rerank.input_count on the span
    Q3  outputs changed       <- gen_ai.request.model AND response.model
    Q4  cost spike            <- isc.prompt.version + fingerprint
    Q5  permission failure    <- retrieval.trimmed_count
    Q6  quality drift         <- repair.attempted as a span event
    Q7  document dominance    <- retrieval.doc_ids
    Q8  error attribution     <- peer.service + status discipline

    EIGHT ATTRIBUTES AND ONE HEADER answered eight real questions. That is the
    whole return on this tutorial, and it is a much smaller instrumentation
    surface than most services carry — because each one was chosen by asking
    "what will I query when this breaks?" rather than by adding a field every
    time someone was confused.

    THE EXERCISE WORTH DOING on your own service: write down the ten questions
    you would be asked during an incident. Then check that each is answerable
    from what you emit today. The gaps are your instrumentation backlog, and
    the fields nothing depends on are your first cost saving.
""")


if __name__ == "__main__":
    main()
