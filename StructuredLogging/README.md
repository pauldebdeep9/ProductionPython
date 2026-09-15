# Structured Logging and Tracing — Production Python, GenAI Focus

Ten files, ~4,700 lines. Runnable offline with no dependencies beyond `pytest`.

**Verified state:** 9/9 scripts run clean · `ruff` passes · 26/26 tests pass ·
**mutation-tested** (logging the prompt, dropping `trimmed_count`, and adding
`trace_id` as a metric label are caught by 5 tests) · the capstone scans every
telemetry surface it emits and finds zero content leaks.

---

## Files

| File | Topic | The thing worth taking away |
|---|---|---|
| `obs_lab.py` | Harness | A **queryable** span store, plus a content scanner |
| `01_structured_logging.py` | Fields, not sentences | `event` is stable, low-cardinality, never interpolated |
| `02_context_propagation.py` | Trace IDs across boundaries | `to_thread` propagates; `run_in_executor` does **not** |
| `03_spans.py` | Span design | ERROR means *this operation failed* — narrower than you think |
| `04_what_to_record.py` | The GenAI-specific rules | Telemetry and audit are **different sinks** |
| `05_metrics.py` | Metrics and cardinality | Unsplit p50 **improves** during an outage |
| `06_sampling.py` | Keeping what matters | Head sampling must hash the trace ID, not call `random()` |
| `07_debugging.py` | The payoff | 8 planted problems, found by 8 queries |
| `08_capstone.py` | Everything composed | Audits its own output for leaks |
| `09_test_observability.py` | Testing telemetry | Telemetry is code nothing else exercises |

Run any file: `python 07_debugging.py`. Test: `pytest 09_test_observability.py -v`.

---

## Suggested path

**Two hours** — 04, 07, 02. What you may record, what it buys you, and how the
trace ID survives.

**If you read one file, read 07.** It generates traces with six planted
problems and finds each one. It's the only real test of whether instrumentation
was worth emitting.

**Instrumenting an existing service** — 03 (granularity), then 05
(cardinality), then 06 (sampling).

---

## The central argument

Telemetry is a **separate trust boundary**. Query access to Application
Insights gets handed out for debugging; access to the documents behind your
index is granted per-group and audited. Writing chunk text into telemetry
copies document content into a store with a *larger reader set*, a *different
retention period*, and *no permission trimming* — which breaks the property
your whole retrieval design exists to guarantee.

The resolution is two sinks:

| | Access | Retention | Volume | Contents |
|---|---|---|---|---|
| **Telemetry** | Broad | 30–90 days | High | Ids, counts, hashes, fingerprints |
| **Audit** | Restricted, reads logged | 7–30 days | Triggered + ~1% | Full content |

`chunk_ids` + `prompt.fingerprint` lets you **reconstruct** the prompt from
your own index, under your own permission checks, at the moment you need it.
That's better than storing it.

---

## Measured results

**8 planted problems, 8 queries, 8 finds** (`07`) — a latency regression traced
to `rerank.input_count` (bimodal: 12ms vs 120ms), a silent model-version drift
affecting one tenant, a prompt change that raised input tokens **+52.5%**, a
repair rate jump from **3.0% → 21.0%**, a document appearing in **42%** of
answers against an ~8% median, and permission trimming silently stopping for
one tenant.

**The latency trap** (`05`) — during an 85%-failure outage, p50 over all
requests **improved from 642ms to 8ms**, because failures return in
milliseconds. Split by outcome, it correctly stayed at 642ms.

**Context propagation, measured** (`02`) — `to_thread` carries the trace ID;
`run_in_executor` and `threading.Thread` return `(EMPTY)` until you pass
`contextvars.copy_context()`.

**Hybrid sampling coverage** (`06`) — 100% of errors, ACL violations, and slow
requests kept; ~1.0% of ordinary successes.

**The capstone audits itself** — 0 findings across spans, logs, and metric
labels; the audit sink correctly holds 6 sensitive items.

---

## The bug I shipped in my own harness

The `ContextFilter` injected `principal` verbatim, so **every log line in the
capstone carried a user's email address** — violating the "hash personal
identifiers" rule from script 04, in the very harness that teaches it.

It's a good illustration of why automatic injection is both the right design
*and* a place to be careful: the filter applies to every log line in the
process, so one careless field there is a leak everywhere at once. Now
`enduser_id` is a salted hash, which still groups correctly — you keep "these
two lines are the same user" without storing who that user is. There's a
regression test.

Two claims also had to be corrected against measurement:

**"Force-keeps are nearly free"** (`06`) — the output said 9.77% kept against a
1% baseline. They're nine times the baseline here, because this traffic has 5%
slow-tail requests. Force-keeps are cheap only if the conditions are *rare*,
which is a property of your traffic, not your policy. Set the slow threshold
from your own p99.

**Absolute thresholds in `07`** — my first version flagged a "dominant"
document at 83.7%, which was an artifact of sampling 3 of 4 documents. With a
realistic 40-document corpus the real signal is 42% against an ~8% median. Both
that query and the ACL one now compare against a **peer median**, because the
right baseline depends entirely on your corpus and your tenants.

---

## Decision tables

### Which signal

| Question | Signal |
|---|---|
| Is the service healthy? Is p95 rising? | **Metric** |
| Why was *this* request slow? What did it retrieve? | **Trace** |
| What was the prompt for *this* answer? | **Audit** |
| Top question this week? Model version changing? | **Log** (wide event) |
| Did permission trimming stop? | **Metric** |

### Span status

| Situation | Status |
|---|---|
| Retried 503, eventually succeeded | **OK** + retry events |
| `finish_reason=length` | **OK** — you set that budget |
| Content filtered, refusal, client disconnect | **OK** — outcomes, not errors |
| Malformed JSON, repaired | **OK** + `repair.attempted` event |
| Malformed JSON, gave up | **ERROR** |

Get this wrong and your error rate counts handled failures, nobody trusts it,
and you build a second metric to track "real" errors.

### What must never be sampled

Metrics · security events · audit records · the wide request event. Record the
sampling decision *on* the wide event, so "no trace for this id" is
distinguishable from "the trace is missing."

---

## Review checklist

**Logging**
- [ ] `event` is stable, dotted, past tense, never interpolated
- [ ] Units in field names (`duration_ms`, `cost_usd`); stable types
- [ ] Alert on **event names**, not on `level`
- [ ] DEBUG off in production — it's where content leaks
- [ ] One wide event per request, always emitted

**Context**
- [ ] One middleware sets context and returns `x-trace-id`
- [ ] `uuid4().hex`, never `str(uuid4())` — a malformed traceparent is dropped silently
- [ ] `copy_context()` for any thread you create; explicit for processes
- [ ] Long-lived tasks start in the lifespan, not in a request handler

**Spans**
- [ ] ~7 per request; aggregate per-item work into counts
- [ ] Every network call is `CLIENT` (service maps depend on it)
- [ ] TTFT is an **event**, not an attribute
- [ ] ERROR only when the operation failed
- [ ] Exceptions record type + message, never repr or locals

**Content**
- [ ] No prompt text, chunk text, user questions, or model output
- [ ] Personal identifiers hashed with a stable salt
- [ ] `acl.violation` events carry ids only
- [ ] Search *highlights* are document content under another name

**Metrics**
- [ ] Every label enumerable; `trace_id` is never a label
- [ ] Latency split by outcome
- [ ] TTFT, `retrieval_trimmed_total`, `repairs_total`, `retry_attempts_total`
- [ ] Cost recorded at the call site, labelled by tenant and outcome

---

## The security query nobody runs

`retrieval.trimmed_count`. Permission-trimmed retrieval fails **silently and
upward** — the user gets a fluent, well-cited answer built from a document they
were never allowed to see. No error, no latency change, no failed assertion.

The only externally visible symptom is that integer. So:

```
alert when a tenant's zero-trim rate exceeds 3x the fleet median over 15 min
```

Baseline-relative, not absolute — a tenant whose users can all see everything
trims zero legitimately and would page you forever.

---

## Connections to your existing work

**This is the operational half of the P1 permission invariant.** Your
post-retrieval assertion catches a violation *in the request*; `trimmed_count`
catches the case where the assertion itself was removed or the filter silently
stopped. `07` Q5 finds exactly that, and `08` demo 3 shows what the telemetry
looks like when a retriever stops trimming — the response is indistinguishable
from normal.

**`repair.attempted` as a span event connects to your k/12 reporting.** Script
`07` Q6 detects a repair-rate jump from 3% to 21% purely from span events. For
`isc-agent-ops`, that separates "the scenario failed" from "the scenario
succeeded after two repairs" — currently both land in the same bucket, and the
second is the early warning.

**`gen_ai.request.model` vs `gen_ai.response.model`** is the pair I'd add
first. It's the only way to detect a change originating outside your deploy
pipeline, and given you've established temperature 0 isn't deterministic, it
separates "the model changed" from "the sample moved" — a distinction your
current records can't make.

**Retrofit order:** the `x-trace-id` response header and the `ContextFilter`
first (both are ~20 lines and make everything else queryable), then the seven
spans with correct kinds, then the sensitivity-routed recorder. The scanner
test is worth adding at the same time as the recorder, since it's what keeps
the split honest.

---

That completes the five Tier 1 topics: async concurrency, failure handling,
typing, configuration and secrets, and observability.
