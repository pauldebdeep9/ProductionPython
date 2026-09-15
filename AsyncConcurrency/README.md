# Async Concurrency for Production Python — GenAI/ML Focus

Twelve files, ~4,500 lines, all runnable offline with zero dependencies beyond
`pytest` for the test module. Every number quoted below is **measured** on the
container these were built in, not asserted from documentation.

**Verified state:** 10/10 scripts execute clean with zero stderr warnings ·
`ruff check` (with `E,F,W,I,UP,B,SIM,C4,BLE,RUF,ASYNC`) passes · 18/18 tests pass.

---

## Files

| File | Topic | The thing worth taking away |
|---|---|---|
| `fake_llm.py` | Offline LLM/embedding stub | Seeded per-client RNG; error taxonomy mirrors a real SDK |
| `01_event_loop_basics.py` | Coroutines, tasks, scheduling | `await` alone never creates concurrency |
| `02_fanout_primitives.py` | gather / TaskGroup / as_completed / wait | The choice is about **failure semantics**, not speed |
| `03_blocking_calls.py` | The blocking trap | Loop-lag monitor: ~15 lines, highest-value async metric you can emit |
| `04_bounded_concurrency.py` | Semaphore, token bucket, backpressure | Three *different* limits; having only one still falls over |
| `05_timeouts_cancellation.py` | Deadlines, cancellation, shutdown | Cleanup survives one cancel and dies on the second |
| `06_retries_backoff.py` | Classification, jitter, breaker | Classification is the design; backoff is mechanism |
| `07_streaming.py` | Async generators, SSE | Once byte one is sent, status is fixed at 200 |
| `08_pipeline_queue.py` | Staged ingestion pipeline | Blocked-time per stage identifies the bottleneck directly |
| `09_pitfalls.py` | Bugs that pass review | Read-modify-write across an `await` is a race |
| `10_rag_capstone.py` | Everything composed | Permission-trimming as a concurrency problem |
| `11_test_async_patterns.py` | Testing async code | Assert on peak concurrency, never on elapsed time |

Run any file directly: `python 01_event_loop_basics.py`. Tests:
`pytest 11_test_async_patterns.py -v`.

---

## Suggested path

**If you have two hours** — 01, 03, 05, 09. That is the mechanics, the failure
you will actually cause, the failure you will actually be asked about, and the
review checklist.

**If you're prepping to review someone's PR** — start at 09's checklist, then
read backwards into whichever section it points at.

**If you're designing a pipeline** — 04, then 08. If you're designing a
request-serving service — 05, then 07, then 10.

---

## Measured results

These are the outputs that make the arguments, reproduced so you can check them
against your own run.

**Concurrency is real, parallelism isn't** (`01`) — 8 calls at 0.10s each:
0.803s sequential → 0.101s concurrent, **8.0x**. One thread throughout. The win
is entirely overlapped *waiting*.

**One blocking call freezes everything** (`03`) — event-loop lag during a sync
SDK call: **max 596.6ms**, avg 298.4ms. Same work via `to_thread`: **max 1.0ms**.
Wall clock also improved 0.602s → 0.201s, but the lag number is the one that
matters, because it's what every *other* request experiences.

**Jitter is not optional** (`06`) — 200 clients retrying at attempt 3. Without
jitter: 200/200 land in the same time bucket. With full jitter: worst bucket
holds 35/200, spread across 9 buckets.

**Circuit breakers shed load** (`06`) — latency collapses from ~70ms to **0.0ms**
once the circuit opens, because no call is made at all.

**Instrumentation predicts the fix** (`08`) — per-stage metrics identified
`embed` as the bottleneck (util 100%, blocked 0.00s). Of four tuning configs,
only "more embedders" moved the number: 1.13s → **0.69s**. Adding downloaders or
parsers changed nothing.

**Races are real in single-threaded code** (`09`) — 50 tasks spending 10 each
from a budget of 100, with a check-then-act spanning an `await`: budget ended at
**−400** with 40 overdrafts. With a lock: 0 and 0.

---

## Two results that contradicted what I expected

Reported plainly, because both change the advice.

### 1. `finally`-block cleanup survives a *single* cancellation

The common claim is that an `await` inside `finally` gets cancelled and your
audit write is lost. **Measured:** it completes fine. Cancellation is delivered
once; after `CancelledError` has been raised, the task is running normally and a
fresh `await` runs to completion.

It's the **second** cancellation that kills cleanup — which is precisely what a
graceful-shutdown drain does (cancel, wait out the grace period, hard-cancel the
stragglers).

| Scenario | Audit record written? |
|---|---|
| One cancel (ordinary request timeout) | ✅ yes |
| Two cancels (shutdown drain) | ❌ **lost** |
| Caller doesn't await long enough | ❌ **lost** |
| Shielded + bounded timeout, two cancels | ✅ yes |

The practical consequence: **test cleanup with a double cancel.** A single-cancel
test passes even when the code is broken for the case you care about — so this
fails only during a deployment, never in CI. `11_test_async_patterns.py` has that
test written out.

### 2. The fire-and-forget task-GC hazard did not reproduce

20/20 tasks completed with no reference held, even under forced `gc.collect()`.
The reason: those tasks were parked in `asyncio.sleep`, which registers a
`TimerHandle` the loop holds *strongly*, transitively keeping the task alive.

The hazard is real and documented, but it needs a task suspended on a future
nothing else references. Keep the reference anyway — the cost is three lines and
the failure mode is a silently dropped write. But don't repeat the folklore
version; it's more specific than people say.

---

## Decision tables

### Fan-out primitive

| Need | Use |
|---|---|
| All N must succeed, clean shutdown on failure | `TaskGroup` + `except*` |
| Per-item partial failure, input correspondence preserved | `gather(return_exceptions=True)` + explicit `isinstance` partition |
| Progress / incremental writes / early exit | `as_completed` with self-identifying return values |
| FIRST_COMPLETED or custom timeout logic | `asyncio.wait`, then cancel **and await** the pending set |
| — | Avoid `gather(return_exceptions=False)`: it abandons siblings mid-cleanup |

### Where does this work belong?

| Work | Route |
|---|---|
| Network I/O | Native async client. Never a thread. |
| Sync SDK, file I/O | `to_thread`, or a dedicated `ThreadPoolExecutor` at scale |
| CPU-bound pure Python | `ProcessPoolExecutor`, or move it out of the request path |
| CPU-bound, GIL released (numpy/torch/onnx/tokenizers) | Threads — much cheaper than processes |

### Three limits, three mechanisms

| Bound | Mechanism | Scope |
|---|---|---|
| In-flight requests | `Semaphore` | Per process |
| Requests/tokens per minute | `TokenBucket` | **Per deployment** — the quota is shared |
| Buffered work / memory | Bounded `asyncio.Queue` | Per pipeline stage |

Size the semaphore from Little's Law: `concurrency = arrival_rate × latency`.
100 req/s at 400ms ⇒ ~40 in flight. For **streaming**, substitute connection
*hold* time: 500 tokens at 20ms/token is 10s held, not 800ms — so 50 concurrent
users needs ~500 connections' worth of capacity, not ~40.

---

## PR review checklist

**Lifetime & ownership**
- [ ] Every `create_task` has an owner — a TaskGroup, or a retained reference plus a done-callback
- [ ] Shutdown cancels **and awaits** every task it created
- [ ] Clients (`httpx`, `AsyncAzureOpenAI`, credentials) live in `lifespan`, not the request path

**Correctness**
- [ ] No read-modify-write spanning an `await` without a lock
- [ ] `CancelledError` always re-raised; no bare `except:` or `except BaseException:`
- [ ] Loop-bound primitives (Lock/Semaphore/Queue/Event) not module-level
- [ ] `async for` that may exit early wrapped in `contextlib.aclosing`

**Bounds**
- [ ] Concurrency bounded by a semaphore sized from a *stated constraint*
- [ ] Rate bounded per deployment, not per client object
- [ ] Memory bounded by queue `maxsize`, not by trusting input size
- [ ] One end-to-end deadline per request, above the four transport timeouts
- [ ] Retries sit **inside** the semaphore, so retrying doesn't double real concurrency

**Blocking**
- [ ] No `time.sleep` / `requests` / sync SDK / heavy CPU inside a coroutine
- [ ] Executors explicitly sized, not the shared default, at scale
- [ ] Reranker / tokeniser / PDF parse goes through `to_thread`

**Retries**
- [ ] Explicit retryable vs non-retryable classification, not bare `Exception`
- [ ] Jitter present — check the formula, not the word "backoff"
- [ ] `Retry-After` honoured when present
- [ ] Overall deadline, not just `max_attempts`
- [ ] Operation idempotent, or an idempotency key sent
- [ ] Circuit breaker for sustained outages

**Observability**
- [ ] `trace_id` in a contextvar, set once at the edge
- [ ] Loop-lag metric emitted (alert >100ms sustained)
- [ ] Retry counts, queue depths, per-stage blocked-time emitted
- [ ] `loop.set_exception_handler` installed and routed to the real logger
- [ ] Logs carry IDs and counts, **never** prompt or chunk bodies

---

## The Azure-specific notes worth extracting

Scattered through the scripts, collected here:

- **Deployment name ≠ model name.** Code hardcoding `"gpt-4o-mini"` as a
  deployment name breaks on the first tenant that named it differently.
- **429 carries `Retry-After`.** Honour it over your own exponential curve.
- **`400 content_filter`** is not retryable. **`400 context_length_exceeded`** is
  not retryable either — but it *is* repairable by re-chunking, which is a
  fallback, not a retry, and belongs in different code.
- **SNAT port exhaustion** on App Service is what per-request `httpx.AsyncClient`
  construction looks like in production: random connection timeouts that read as
  "the model is slow".
- **`DefaultAzureCredential` caches tokens internally.** Constructing it per
  request defeats that cache and can throttle the IMDS endpoint.
- **Proxies buffer SSE.** If you stream correctly and users still see nothing
  until the end, suspect Front Door / API Management / nginx before your code.
  `X-Accel-Buffering: no` covers nginx; the Azure hops each need their own setting.
- **Idle timeouts** on every hop must exceed your longest generation. Several
  Azure front doors default to 4 minutes.
- **`os.cpu_count()` lies in containers.** A process pool sized to the host's
  core count on a 1-vCPU App Service buys memory overhead for nothing. Read the
  cgroup limit. (This container reports 1 CPU, which is why script 03's process
  pool shows no win — the demo reports and interprets its own result rather than
  pretending otherwise.)

---

## Where this connects to `isc-docint` and `isc-agent-ops`

**The P2 step-budget finding is script 04 and 05.** Your note that the step
budget bounds model turns rather than tool calls, so parallel batching bounds the
wrong thing, is exactly the distinction between a *concurrency* limit and a
*rate* limit. The fix is a semaphore around tool execution plus a separate
counter on tool invocations, not a larger step budget.

**The stuck loops re-querying argument-free tools** are a cancellation and
deadline problem. An end-to-end `timeout_at` budget on the agent loop bounds it
regardless of how the model misbehaves — which is more robust than trying to
detect the loop pattern.

**Permission-trimmed retrieval is a concurrency invariant**, which is the
argument `10_rag_capstone.py` makes structurally. The four ways it breaks are all
concurrency-shaped: a cache keyed on query text alone, an ACL that goes stale
mid-request, a fan-out where one source applies the filter and another doesn't,
and a partial-failure path falling back to an unfiltered index. None is a
cryptography bug. The post-retrieval assertion in the capstone is cheap (a set
intersection per chunk) and is the only thing that makes a silent upward failure
loud.

**Retrofit rather than rebuild.** The higher-leverage move is adding one Tier 1
item at a time to `isc-agent-ops` — bounded concurrency, classified retries,
structured traces with correlation IDs — rather than standing up a new service to
"learn production Python." That gives you the review vocabulary without a new
project, and it fixes the loop-bounding defect you already documented.
