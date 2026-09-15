"""
05 — Metrics: the third signal, and the one that bankrupts you.

THE THREE SIGNALS, and when each is right
-----------------------------------------
    METRICS  pre-aggregated numbers over time. Cheap, fast, bounded.
             "What is the p95 latency right now?" "Is the error rate rising?"
             Cannot answer anything about a SPECIFIC request.

    TRACES   one request, in structure. Expensive per request, usually
             sampled. "Why was THIS request slow?"

    LOGS     discrete events with arbitrary fields. Expensive at volume.
             "What happened, exactly, in this one case?"

THE WORKFLOW they support together: a METRIC alerts you, a TRACE tells you
where the time went, a LOG tells you what the values were. Skipping metrics
and alerting on log queries is a common and expensive mistake — log queries
are slow, cost per query, and cannot be evaluated every 15 seconds.

THE THING THAT BANKRUPTS YOU is cardinality, and it is almost always one
label somebody added without thinking about the multiplication.

Run:  python 05_metrics.py
"""

from __future__ import annotations

import random
from collections import Counter

from obs_lab import (
    TelemetryStore,
    banner,
    section,
    show,
)

STORE = TelemetryStore()


# ---------------------------------------------------------------------------
# PART 1 — cardinality, multiplied out
# ---------------------------------------------------------------------------

def part1_cardinality() -> None:
    banner("PART 1 — cardinality is a product, not a sum")

    print("""
    A metric with labels creates ONE TIME SERIES PER DISTINCT COMBINATION.
    Storage and cost scale with the number of series, not with the number of
    data points — and the combinations MULTIPLY.""")

    section("a reasonable metric")
    dims = [("deployment", 3), ("outcome", 5), ("tenant", 8)]
    total = 1
    for name, n in dims:
        total *= n
        print(f"      {name:<16} {n:>8} values")
    show("time series", f"{total:,}")

    section("...then someone adds one label")
    for extra, n, why in [
        ("user_id", 5_000, "'so we can see per-user latency'"),
        ("question_hash", 200_000, "'to find slow question types'"),
        ("trace_id", 10_000_000, "'to link metrics to traces'"),
        ("chunk_id", 48_000, "'which documents are slow?'"),
    ]:
        print(f"      + {extra:<15} x{n:>10,}  -> {total * n:>14,} series   {why}")

    print("""
    The last three are all real things people have added. `trace_id` as a
    metric label is the pathological case: it creates one time series per
    request, which is not a metric at all — it is a log line with worse
    ergonomics and a much worse price.

    THE RULE: a metric label must be BOUNDED and SMALL. If you cannot write
    down the complete list of possible values, it is not a label.

    WHERE THE HIGH-CARDINALITY THING BELONGS INSTEAD:
        user_id, trace_id, question_hash, chunk_id  ->  logs and span
                                                        attributes
    Those are designed for high cardinality. Metrics are not.

    THE BRIDGE between them is an EXEMPLAR — a trace_id attached to a
    histogram bucket, so "show me a trace from the p99 bucket" works without
    trace_id ever becoming a label. If your backend supports exemplars, this
    is the feature that removes the temptation entirely.""")

    section("what a bounded label set looks like")
    good = {
        "deployment": ["gpt-4o-mini-prod", "gpt-4o-prod", "gpt-4o-mini-eu"],
        "outcome": ["ok", "degraded", "refused", "filtered", "error"],
        "tenant": ["isc-sg", "isc-eu", "isc-us"],
        "finish_reason": ["stop", "length", "content_filter", "tool_calls"],
    }
    n = 1
    for k, v in good.items():
        n *= len(v)
        print(f"      {k:<16} {len(v)} values: {v[:3]}")
    show("total series for one metric", n)
    print("      Enumerable, stable, and small. That is the test.")


# ---------------------------------------------------------------------------
# PART 2 — the LLM metric set
# ---------------------------------------------------------------------------

def part2_metric_set() -> None:
    banner("PART 2 — the metrics a RAG service actually needs")

    metrics = [
        # (name, type, labels, why)
        ("requests_total", "counter", "tenant, outcome",
         "the denominator for everything else"),
        ("request_duration_seconds", "histogram", "tenant, outcome",
         "end-to-end latency, split by outcome"),
        ("gen_ai_ttft_seconds", "histogram", "deployment",
         "TIME TO FIRST TOKEN — what users feel"),
        ("gen_ai_tokens_total", "counter", "deployment, direction",
         "input/output tokens; the cost driver"),
        ("gen_ai_cost_usd_total", "counter", "deployment, tenant",
         "spend, attributable per tenant"),
        ("gen_ai_finish_reason_total", "counter", "deployment, reason",
         "truncation and filtering rates"),
        ("retrieval_chunks", "histogram", "index",
         "how many chunks came back; a 0 spike is a retrieval outage"),
        ("retrieval_trimmed_total", "counter", "tenant",
         "SECURITY: sustained zero means trimming stopped"),
        ("acl_violations_total", "counter", "tenant",
         "SECURITY: must always be zero"),
        ("repairs_total", "counter", "kind",
         "parse/schema/truncation repairs; detects model drift"),
        ("retry_attempts_total", "counter", "dependency",
         "with requests_total gives AMPLIFICATION"),
        ("upstream_duration_seconds", "histogram", "dependency",
         "which dependency is slow"),
        ("cache_operations_total", "counter", "cache, result",
         "hit rate"),
        ("degraded_responses_total", "counter", "reason",
         "how often users get a reduced answer"),
        ("refusals_total", "counter", "deployment",
         "tracked SEPARATELY from errors"),
    ]
    print(f"      {'metric':<32} {'type':<10} {'labels':<22} why")
    print(f"      {'-' * 32} {'-' * 10} {'-' * 22} {'-' * 28}")
    for name, kind, labels, why in metrics:
        print(f"      {name:<32} {kind:<10} {labels:<22} {why[:30]}")

    print("""
    THE FOUR THAT MOST TEAMS ARE MISSING, and each answers a question nothing
    else can:

      gen_ai_ttft_seconds      total latency hides it completely. A routing
                               change or a cold start moves TTFT while total
                               duration stays flat, and users notice.

      retrieval_trimmed_total  the only externally-visible signal that
                               permission trimming is still running.

      repairs_total            a rising repair rate is the earliest sign a
                               model version changed under you, or that a
                               prompt edit regressed the output format —
                               usually days before anyone reports bad answers.

      retry_attempts_total     divided by requests_total gives amplification,
                               which moves before error rate does because the
                               retries are still succeeding.""")


# ---------------------------------------------------------------------------
# PART 3 — the latency trap
# ---------------------------------------------------------------------------

def part3_latency_by_outcome() -> None:
    banner("PART 3 — always split latency by outcome")

    rng = random.Random(7)
    STORE.reset()

    section("healthy: 98% success")
    for _ in range(1000):
        if rng.random() < 0.98:
            STORE.observe("request_duration_ms", rng.uniform(400, 900),
                          outcome="ok")
        else:
            STORE.observe("request_duration_ms", rng.uniform(3, 12),
                          outcome="error")
    all_healthy = (STORE.histograms["request_duration_ms|outcome=ok"]
                   + STORE.histograms["request_duration_ms|outcome=error"])
    all_healthy.sort()
    show("p50 over ALL requests", f"{all_healthy[len(all_healthy)//2]:.0f}ms")
    show("p50 for outcome=ok",
         f"{STORE.hist_percentile('request_duration_ms|outcome=ok', 0.5):.0f}ms")

    STORE.reset()
    section("OUTAGE: 85% failing fast")
    for _ in range(1000):
        if rng.random() < 0.15:
            STORE.observe("request_duration_ms", rng.uniform(400, 900),
                          outcome="ok")
        else:
            STORE.observe("request_duration_ms", rng.uniform(3, 12),
                          outcome="error")
    all_outage = (STORE.histograms["request_duration_ms|outcome=ok"]
                  + STORE.histograms["request_duration_ms|outcome=error"])
    all_outage.sort()
    show("p50 over ALL requests", f"{all_outage[len(all_outage)//2]:.0f}ms")
    show("p50 for outcome=ok",
         f"{STORE.hist_percentile('request_duration_ms|outcome=ok', 0.5):.0f}ms")

    print("""
    THE UNSPLIT p50 IMPROVED DRAMATICALLY during a total outage, because
    failures return in milliseconds and successes take hundreds. A latency
    dashboard without an outcome split looks BEST at the worst possible
    moment.

    The split version is unchanged, correctly — successful requests are still
    taking the same time. The error rate is what moved.

    SAME ARGUMENT for token and cost metrics: split by finish_reason, or a
    surge of truncated generations looks like a cost saving.""")


# ---------------------------------------------------------------------------
# PART 4 — counters and derived rates
# ---------------------------------------------------------------------------

def part4_derived() -> None:
    banner("PART 4 — the ratios worth alerting on")

    STORE.reset()
    rng = random.Random(11)
    for _ in range(1000):
        STORE.incr("requests_total", tenant="isc-sg", outcome="ok")
        attempts = 1 + (1 if rng.random() < 0.3 else 0) + (
            1 if rng.random() < 0.1 else 0)
        STORE.incr("retry_attempts_total", attempts, dependency="azure-openai")
        if rng.random() < 0.06:
            STORE.incr("repairs_total", kind="schema")
        if rng.random() < 0.03:
            STORE.incr("degraded_responses_total", reason="fallback_model")

    def total(name: str) -> int:
        return sum(v for (n, _), v in STORE.counters.items() if n == name)

    reqs = total("requests_total")
    ratios = [
        ("retry amplification", total("retry_attempts_total") / reqs,
         "> 1.5 sustained", "earliest sign of a degrading dependency"),
        ("repair rate", total("repairs_total") / reqs,
         "> 2x baseline", "model or prompt drift"),
        ("degraded rate", total("degraded_responses_total") / reqs,
         "> 5%", "users getting reduced answers"),
    ]
    print(f"      {'ratio':<24} {'value':>8}  {'alert at':<16} why")
    print(f"      {'-' * 24} {'-' * 8}  {'-' * 16} {'-' * 30}")
    for name, value, threshold, why in ratios:
        print(f"      {name:<24} {value:>8.2f}  {threshold:<16} {why}")

    print("""
    ALERT ON RATIOS, NOT RAW COUNTS. A raw count of 400 errors means nothing
    without knowing whether that is out of 500 requests or 500,000 — so an
    absolute threshold either fires constantly at high traffic or never fires
    at low traffic.

    AND ALWAYS INCLUDE THE DENOMINATOR IN THE SAME QUERY. A ratio computed
    from two separately-scraped counters can produce nonsense at the edges,
    particularly right after a deploy when one counter has reset and the other
    has not.""")


# ---------------------------------------------------------------------------
# PART 5 — cost accounting as a first-class metric
# ---------------------------------------------------------------------------

def part5_cost() -> None:
    banner("PART 5 — cost is a metric, recorded at the call site")

    print("""
    THE RULE: compute cost at the point of the call, from the usage the API
    RETURNED, and record it as a counter labelled by deployment and tenant.

    WHY NOT RECONSTRUCT IT LATER from token counts and a price list:
      * the price list changes, and your historical data silently re-prices
      * you need a mapping from deployment -> model -> price, which drifts
      * provisioned throughput and pay-as-you-go price differently for the
        same model
      * cached input tokens are priced differently again

    WHY NOT USE THE AZURE COST API: it is delayed by hours to a day, and it
    aggregates by resource, not by tenant, use case, or prompt version. It
    tells you what you spent, never which feature spent it.

    THE ATTRIBUTION YOU WANT, and it is only possible if you label at the
    call site:
        by TENANT       -> chargeback and per-tenant caps
        by USE CASE     -> which feature is worth its cost
        by DEPLOYMENT   -> is the expensive model earning its price
        by PROMPT VERSION -> did the new template increase token usage
        by OUTCOME      -> how much are we spending on requests that FAILED""")

    STORE.reset()
    rng = random.Random(3)
    PRICE = {"gpt-4o-mini": (0.00015, 0.00060), "gpt-4o": (0.0025, 0.010)}
    for _ in range(500):
        model = "gpt-4o" if rng.random() < 0.15 else "gpt-4o-mini"
        tenant = rng.choice(["isc-sg", "isc-eu"])
        outcome = "ok" if rng.random() < 0.93 else "error"
        pin, pout = PRICE[model]
        inp, out = rng.randint(800, 3000), rng.randint(50, 400)
        usd = inp / 1000 * pin + out / 1000 * pout
        STORE.observe(f"cost_usd|model={model},tenant={tenant},outcome={outcome}",
                      usd)

    section("spend by model, tenant, and outcome")
    totals: Counter[str] = Counter()
    for key, values in STORE.histograms.items():
        totals[key.split("|", 1)[1]] += sum(values)
    for key, usd in totals.most_common():
        print(f"      {key:<44} ${usd:>8.4f}")

    wasted = sum(v for k, v in totals.items() if "outcome=error" in k)
    total_spend = sum(totals.values())
    show("total", f"${total_spend:.4f}")
    show("spent on FAILED requests", f"${wasted:.4f} ({wasted/total_spend:.1%})")

    print("""
    That last line is the one nobody has. Money spent on requests that
    returned an error is pure waste, and it is invisible unless `outcome` is
    a label on your cost metric.

    It also gives retry policy a price. If amplification is 1.4x and 7% of
    spend is on failed requests, you can put a number on what a circuit
    breaker would save — which is a much better argument than "retries are
    wasteful".""")


# ---------------------------------------------------------------------------
# PART 6 — choosing the signal
# ---------------------------------------------------------------------------

def part6_choosing() -> None:
    banner("PART 6 — which signal for which question")

    rows = [
        ("Is the service healthy right now?", "metric", "cheap, fast, alertable"),
        ("Is p95 latency rising?", "metric", "pre-aggregated"),
        ("Why was THIS request slow?", "trace", "structure and durations"),
        ("What did we retrieve for THIS request?", "trace", "span attributes"),
        ("How many requests used the fallback?", "metric", "a counter"),
        ("What was the prompt for THIS answer?", "audit", "content, restricted"),
        ("Which tenant is driving cost?", "metric", "labelled counter"),
        ("What is the top question this week?", "log", "hash + group by"),
        ("Did permission trimming stop?", "metric", "trimmed_total -> 0"),
        ("Which document leaked?", "trace", "acl.violation event"),
        ("Is the model version changing?", "log", "wide event, group by"),
    ]
    print(f"      {'question':<42} {'signal':<9} why")
    print(f"      {'-' * 42} {'-' * 9} {'-' * 22}")
    for q, signal, why in rows:
        print(f"      {q:<42} {signal:<9} {why}")

    print("""
    NOTE HOW FEW ARE LOGS. Once you have wide events on spans and a decent
    metric set, most log lines are redundant — and logs are usually the most
    expensive of the three per unit of information.

    A REASONABLE STARTING BUDGET for a RAG service:
        metrics   ~15 series-generating metrics, all labels bounded
        traces    100% sampled at low volume; head-sample + keep-all-errors
                  at high volume (script 06)
        logs      ONE wide event per request, plus security events
        audit     triggered + ~1% sample""")


def main() -> None:
    part1_cardinality()
    part2_metric_set()
    part3_latency_by_outcome()
    part4_derived()
    part5_cost()
    part6_choosing()

    banner("SUMMARY")
    print("""
  * Cardinality MULTIPLIES. If you cannot enumerate a label's values, it is
    not a label — it is a span attribute or a log field.
  * `trace_id` as a metric label creates one series per request. Use
    exemplars to bridge instead.
  * Always split latency by OUTCOME. An unsplit p50 improves during an
    outage, because failures are fast.
  * The four metrics most teams lack: TTFT, retrieval_trimmed_total,
    repairs_total, retry_attempts_total.
  * Alert on RATIOS with the denominator in the same query, not raw counts.
  * Record cost at the CALL SITE from returned usage, labelled by tenant,
    deployment, and outcome. Spend-on-failed-requests is otherwise invisible.
  * Metric alerts you, trace localises it, log/audit explains it.
""")


if __name__ == "__main__":
    main()
