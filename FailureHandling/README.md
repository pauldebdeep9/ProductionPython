# Failure Handling and Retries — Production Python, GenAI Focus

Eleven files, ~6,100 lines, runnable offline with no dependencies beyond
`pytest`. Every number below is **measured** on the container these were built
in, not asserted from documentation.

**Verified state:** 9/9 scripts execute clean with zero stderr warnings ·
`ruff check` (with `E,F,W,I,UP,B,SIM,C4,BLE,RUF,ASYNC`) passes · 44/44 tests
pass · invariant tests confirmed to catch a deliberately injected bug
(mutation-tested).

---

## Files

| File | Topic | The thing worth taking away |
|---|---|---|
| `failure_lab.py` | Taxonomy, fault injection, fake services | Scripted failures, not random ones |
| `01_taxonomy.py` | Classification and adapters | Five dispositions, not one boolean |
| `02_retry_mechanics.py` | Budgets, amplification, deadlines | 3 layers × 3 retries = **27x** load |
| `03_idempotency.py` | Making retries *safe* | Every write has three outcomes, not two |
| `04_timeouts_hedging.py` | Tail latency | Hedge at p95; hedging under saturation backfires |
| `05_circuit_breakers.py` | Containment | Count **slow** calls, or brownouts never trip |
| `06_fallbacks.py` | Graceful degradation | Never fall back toward wider access |
| `07_partial_failure.py` | DLQ, poison pills, batch guards | 400 dead letters is usually 3 problems |
| `08_llm_failures.py` | The GenAI-specific ones | **Repair ≠ retry** |
| `09_capstone_agent.py` | Invoice exception agent | A deterministic gate that doesn't trust the model |
| `10_test_failure_handling.py` | Testing failure paths | Assert on invariants, not elapsed time |

Run any file directly: `python 01_taxonomy.py`. Tests:
`pytest 10_test_failure_handling.py -v`.

---

## Suggested path

**Two hours** — 01, 02, 08. Classification, the amplification trap, and the
LLM-specific failures that generic retry libraries cannot express.

**Reviewing someone's resilience PR** — the checklist below, then whichever
section it points at.

**Designing an agent that writes to a system of record** — 03, then 09. The
idempotency argument is the load-bearing one.

---

## Measured results

**Retry amplification is multiplicative** (`02`) — three layers each retrying
3× produced **27 calls** to the bottom service from one user request. Retrying
at one layer only: 3 calls.

**Retry budgets shed load, at a real cost** (`02`) — at a 90% failure rate,
attempts dropped 1728 → 699 (amplification 3.46x → 1.40x). But successes also
dropped 165 → 56. That trade is the decision; the script prints both columns so
you can't make it accidentally. At a 5% failure rate the budget is invisible
(identical attempts, identical successes).

**Jitter families, by herd spread** (`02`) — 500 clients at attempt 3, bucketed
into 0.1s windows: fixed and exponential put **500/500 in one bucket**; equal
jitter spread over 4; full jitter 7; decorrelated 29.

**Idempotency, demonstrated both ways** (`03`) — a write that succeeds
server-side then loses its response, retried 3×: **3 duplicate disputes filed**
without a key, **1** with one. The naive check-then-write dedup store produced
**5 duplicates** under 5 concurrent requests.

**Hedging works, in the right conditions** (`04`) — with headroom, hedging at
~p95 cut p99 from 240ms to 144ms for **4% extra load**; hedging at ~p50 got p99
to 55ms but cost 46%. Under saturation, hedging sent 60 → **109 calls** and
moved p99 by 3ms (750 → 753). Same technique, opposite result.

**Brownouts need slow-call counting** (`05`) — 15 slow-but-*successful* calls
tripped the sliding-window breaker on slow-call rate. A failure-rate-only
breaker sees zero errors and never fires.

**Consecutive-failure counting misses partial degradation** (`05`) — at a 50%
failure rate the consecutive-failure breaker stayed `closed`; the
sliding-window breaker opened correctly.

**Free repairs cover most LLM parse failures** (`08`) — of seven common
malformations, **five were fixed for free** (fence stripping, brace extraction,
type coercion). Only the hallucinated enum and the truncation needed a model
call — and they needed *different* ones.

**The capstone held every invariant under stress** (`09`) — 60 cases against a
35%-failure primary: 59/60 approved (23 via the fallback model), 1 failed to
the DLQ, **59 proposals written = 59 approved**, breaker opened, retry budget
denied 9 retries, amplification 2.52x.

---

## Two things worth flagging

**Bulkhead isolation is not free.** In `05` part 3, separating the pools cut DB
latency from ~1000ms to ~10ms — but total elapsed went *up*, 1.01s → 1.50s,
because the shared pool had given LLM calls all 10 slots while the bulkhead caps
them at 8. I'd originally written "same wall clock," which the measurement
contradicted. Isolation buys predictable latency for the healthy dependency at
the price of some peak throughput for the unhealthy one. Almost always worth it;
still worth stating.

**Which batch guard fires matters.** In `07` part 4 I predicted the
consecutive-failure rule would trip; the failure-*rate* rule fired first (26%
over 53 items). Both guard the same thing from different angles — the rate rule
catches partial systemic failure, the consecutive rule catches total failure
faster. Keep both, and don't assume which one you're relying on.

---

## The mutation test

`44 passed` means nothing if the tests are vacuous. I deliberately removed the
dedup branch from `FlakyWriteAPI.write` and re-ran:

```
7 failed, 37 passed
  test_invariant_writes_never_duplicate_under_retry
  test_invariants_hold_under_random_failures[1,2,3,5,6,7]
```

Seeds 0 and 4 did **not** catch it — `fail_first_n = seed % 4` is 0 for both, so
no retry happens and no duplicate can occur. That's the seeded chaos sweep
working as designed: the randomness explores the space, the seed makes each
point reproducible, and `pytest -k 'seed3'` reruns the exact failure.

---

## Decision tables

### Disposition, not "retryable"

| Disposition | Meaning | Examples |
|---|---|---|
| `RETRY` | Same request, later | 429, 503, timeout |
| `REPAIR` | **Different** request, now | context too long, bad JSON, truncation |
| `FALLBACK` | Different provider/model/path | regional outage, model deprecated |
| `FAIL_FAST` | Nothing helps | 400 validation, content filter |
| `ESCALATE` | A human must act | 401, committed quota exhausted |

`REPAIR` is the category generic retry libraries cannot express, and it's where
most GenAI reliability lives.

### The second axis: is the retry *safe*?

| Error | Useful to retry? | Safe to retry? |
|---|---|---|
| 429, 503 | yes | yes |
| timeout, connection reset | yes | **no — side effect may have occurred** |
| 400 validation | no | — |

### Layer ordering

```
load shed  →  bulkhead  →  circuit breaker  →  retry budget  →  timeout  →  retry
```

The common misordering is putting the breaker **inside** the retry loop. Then
each retry checks the breaker, gets rejected instantly, and you burn all
attempts in microseconds and report "exhausted retries" — technically true,
completely misleading. An open circuit should mean *we did not try*.

### LLM output failures — 11 of 12 are not retries

| Failure | Handling | Cost | Retry? |
|---|---|---|---|
| markdown-fenced JSON | strip fences | free | no |
| prose around JSON | brace extraction | free | no |
| number as string | coerce | free | no |
| missing field | repair w/ schema | 1 call | no |
| hallucinated enum | repair w/ valid values | 1 call | no |
| `finish_reason=length` | raise max_tokens | 1 call | no |
| context_length_exceeded | re-chunk / reduce k | 0–1 call | no |
| refusal | detect, surface honestly | 0 calls | no |
| content_filter | fail fast | 0 calls | no |
| tool not in registry | repair w/ tool list | 1 call | no |
| ungrounded answer | verify vs context | 0–1 call | no |
| 429 / 503 / timeout | retry w/ backoff | n calls | **yes** |

---

## PR review checklist

**Classification**
- [ ] Explicit dispositions, not a retryable boolean
- [ ] Vendor exception types appear in exactly one adapter module
- [ ] Resilience layer never inspects status codes
- [ ] `raise ... from exc` everywhere
- [ ] `CancelledError` never classified or retried
- [ ] Error context carries identifiers and shapes, **never** prompt/chunk bodies

**Retries**
- [ ] Retry at exactly one layer
- [ ] Jitter present — check the formula, not the word "backoff"
- [ ] Retry budget as a fraction of request volume
- [ ] Absolute deadline propagated; child deadlines clamp to parent
- [ ] Deadline checked *before* spending an attempt; backoff clamped to remaining
- [ ] `Retry-After` honoured
- [ ] Attempt history returned, not just logged

**Idempotency**
- [ ] Every write tool takes a key
- [ ] Key generated **once** per logical operation, persisted with the work item
- [ ] Server reserves the key atomically before doing work
- [ ] Same key + different payload → reject, never replay
- [ ] Keys scoped per tenant; dedup store TTL stated as a number

**Containment**
- [ ] Breaker uses failure *rate* over a window, with minimum call volume
- [ ] Slow calls counted as failures
- [ ] Half-open limits concurrent probes
- [ ] Breaker scoped per (dependency, region, deployment)
- [ ] Bulkheads per dependency; queues bounded
- [ ] Breaker outside the retry loop

**Degradation**
- [ ] Middle rungs exist (other region, cheaper model, stale cache, retrieval-only)
- [ ] One deadline across the whole chain
- [ ] No fallback on `FAIL_FAST` errors
- [ ] Every degraded response **labelled**
- [ ] Cache keys include identity
- [ ] Fail **closed** on anything gating access
- [ ] `served_by.<rung>` tracked

**LLM-specific**
- [ ] `finish_reason` checked before parsing
- [ ] Free repairs (fences, braces, coercion) before any model call
- [ ] Repair prompt actually differs from the original
- [ ] Repair budget bounded (~2)
- [ ] `repair.*` counters emitted separately from retries
- [ ] Refusals detected and tracked separately

**Batch**
- [ ] Quarantine, don't abort or swallow; count reported as a result
- [ ] Dead letters store a payload *reference*
- [ ] Poison detection via delivery count
- [ ] Batch-level guard for systemic failure
- [ ] Alert on DLQ growth *rate*, not depth
- [ ] Replay is idempotent

**Observability**
- [ ] `retry_amplification` (attempts/requests) — alert >1.5 sustained
- [ ] Attempts *distribution*, not mean
- [ ] Latency broken down **by outcome**

That last one is worth its own note: failures are usually faster than
successes, so p50 latency computed over all requests **improves** during an
outage. `02` part 6 demonstrates this — a healthy p50 of ~600ms drops to ~5ms
while 90% of requests are failing.

---

## Azure-specific notes

- **Two different 429s.** A burst rate-limit is `RETRY`; `insufficient_quota` is
  `ESCALATE`. Backing off against exhausted committed quota just wastes the
  deadline.
- **Three different 400s.** `context_length_exceeded` → REPAIR,
  `content_filter` → FAIL_FAST, everything else → validation. Any library
  dispatching on status code alone gets all of these wrong.
- **Deployment name ≠ model name.** Code hardcoding `gpt-4o-mini` as a
  deployment breaks on the first tenant that named it differently.
- **Service Bus `MaxDeliveryCount`** is poison-pill handling. If you build your
  own consumer loop, you must build it yourself.
- **Client cancellation stops generation.** Azure OpenAI halts when the client
  disconnects, so propagating cancellation is worth real money on long outputs.
  Never wrap the main call in `asyncio.shield`.

---

## Connections to your existing work

**`isc-agent-ops` already has the hardest control right.** The deterministic
approval gate that re-derives exceptions independently rather than trusting the
agent is exactly what `09` builds — and demo 2 shows why: the model proposed
`approve` with **0.97 confidence** for a genuine 402-vs-400 overbill, and the
arithmetic caught it. Confidence is not evidence. What `09` adds is making the
consistency policy a data structure (`CONSISTENT`) rather than prose in a
prompt, so it's diffable and testable.

**The `write_proposal` tool should take an idempotency key.** You have exactly
one non-read-only tool, which is the right design. `03` argues the remaining
piece: a key derived from `case_id:trace_id` — identifiers you already have,
stable across retries, no extra state. Without it, a redelivery or a pod restart
between the write and the audit record double-files a dispute.

**The stuck-loop finding is a batch guard, not a retry fix.** Two of your 12
scenarios re-queried argument-free tools repeatedly. `07` part 4's `BatchGuard`
is the general shape — a consecutive-failure or repeated-identical-call limit
that aborts with a *specific* reason. More robust than trying to detect the loop
pattern itself.

**Retry vs repair maps onto your k/n reporting.** If a scenario needed three
attempts because the model emitted fenced JSON twice, that's a repair, and it
belongs in a different column from a scenario that needed three attempts because
the endpoint 503'd. Collapsing them into one "attempts" number hides which
problem you actually have — and `08`'s `repair.*` counters are the split.

**Retrofit order, if you're adding this to `isc-agent-ops`:** the idempotency
key first (smallest change, biggest correctness win), then the repair/retry
counter split, then the batch guard. All three are additive and none requires
restructuring the agent loop.
