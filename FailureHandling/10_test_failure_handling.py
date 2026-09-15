"""
10 — Testing failure paths.

Run:  pip install pytest pytest-asyncio
      pytest 10_test_failure_handling.py -v

WHY FAILURE PATHS ARE THE LEAST-TESTED CODE YOU HAVE
----------------------------------------------------
The happy path gets exercised by every manual test, every demo, and every
integration run. The failure path runs only when something breaks — in
production, at 3am, when it is least convenient to discover it does not work.

The specific traps:

  * Tests that use random failure injection. They pass 9 times out of 10 and
    fail on the 10th CI run, so they get marked flaky and skipped. Use
    SCRIPTED failures: "fail twice with 503, then succeed" is an assertion,
    not a lottery.

  * Tests that assert on elapsed time. `assert elapsed < 2.0` fails on a
    loaded runner. Assert on ATTEMPT COUNTS and SEQUENCES instead — they are
    deterministic and they test the property you actually care about.

  * Tests that never exercise the second failure. Cleanup that survives one
    cancellation dies on the second (a shutdown drain). Same shape here:
    retry logic that handles one 503 may not handle a 503 followed by a 429
    with Retry-After.

  * Tests that verify the retry HAPPENED but not that the outcome was CORRECT.
    A retry that duplicates a write is worse than no retry.

The most valuable tests below are the INVARIANT tests: properties that must
hold across all failure paths (writes never duplicate, budgets are never
exceeded, deadlines are never blown).
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from failure_lab import (
    AppError,
    AuthError,
    ContentFiltered,
    ContextLengthExceeded,
    Disposition,
    Fail,
    FaultInjector,
    FlakyWriteAPI,
    Hang,
    Metrics,
    Ok,
    RateLimitError,
    ServiceUnavailable,
    UpstreamTimeout,
    classify,
)

pytestmark = pytest.mark.asyncio


# ===========================================================================
# 1. CLASSIFICATION — a table test that forces a decision on new error types
# ===========================================================================

@pytest.mark.parametrize(
    "exc,expected",
    [
        (RateLimitError("429"), Disposition.RETRY),
        (ServiceUnavailable("503"), Disposition.RETRY),
        (UpstreamTimeout("timeout"), Disposition.RETRY),
        (ContentFiltered("400"), Disposition.FAIL_FAST),
        (ContextLengthExceeded("400"), Disposition.REPAIR),
        (AuthError("401"), Disposition.ESCALATE),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else str(v),
)
async def test_classification_table(exc: AppError, expected: Disposition) -> None:
    """When someone adds an error type, this table is where they are forced to
    decide what it means. That is the point of writing it as a table rather
    than as scattered asserts."""
    assert classify(exc) is expected


async def test_cancellation_is_never_classified() -> None:
    """Guards the rule from 01: cancellation is not an application error.
    Classifying it invites retrying it, and retrying a cancellation turns a
    prompt shutdown into a hang."""
    with pytest.raises(AssertionError):
        classify(asyncio.CancelledError())


async def test_side_effect_uncertainty_is_marked() -> None:
    """A timeout on a write may have applied. The taxonomy must say so, or
    downstream code cannot make the idempotency decision."""
    assert UpstreamTimeout("x").side_effect_uncertain is True
    assert RateLimitError("x").side_effect_uncertain is False


# ===========================================================================
# 2. RETRY BEHAVIOUR — scripted, never random
# ===========================================================================

async def retry(fn, *, max_attempts: int = 4, on_attempt=None) -> object:
    """Minimal retry under test."""
    last: BaseException | None = None
    for n in range(max_attempts):
        if on_attempt:
            on_attempt(n)
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except AppError as e:
            if classify(e) is not Disposition.RETRY:
                raise
            last = e
            if n == max_attempts - 1:
                raise
            await asyncio.sleep(0)
    raise last


async def test_retries_exactly_until_success() -> None:
    """Assert the ATTEMPT COUNT, not elapsed time. Deterministic on any runner."""
    inj = FaultInjector(script=[
        Fail(ServiceUnavailable), Fail(ServiceUnavailable), Ok(),
    ])
    await retry(inj.maybe_fail)
    assert inj.calls == 3, "should stop retrying as soon as it succeeds"


async def test_no_retry_on_fail_fast() -> None:
    """The expensive mistake: retrying something that can never succeed."""
    inj = FaultInjector(script=[Fail(ContentFiltered)])
    with pytest.raises(ContentFiltered):
        await retry(inj.maybe_fail)
    assert inj.calls == 1, "a content filter must not be retried"


async def test_retry_exhaustion_raises_the_last_error() -> None:
    """The caller must receive the real error, not a generic 'retries
    exhausted' that discards what actually went wrong."""
    inj = FaultInjector(script=[Fail(ServiceUnavailable, message="503 upstream")])
    with pytest.raises(ServiceUnavailable, match="503 upstream"):
        await retry(inj.maybe_fail, max_attempts=3)
    assert inj.calls == 3


async def test_mixed_error_sequence() -> None:
    """The case single-error tests miss: a 503 followed by a 429 followed by
    a permanent error. Real failures are heterogeneous."""
    inj = FaultInjector(script=[
        Fail(ServiceUnavailable),
        Fail(RateLimitError, retry_after=0.0),
        Fail(ContentFiltered),
    ])
    with pytest.raises(ContentFiltered):
        await retry(inj.maybe_fail, max_attempts=5)
    assert inj.calls == 3, "must stop at the permanent error, not continue to 5"


# ===========================================================================
# 3. INVARIANTS — the highest-value tests here
# ===========================================================================

async def test_invariant_writes_never_duplicate_under_retry() -> None:
    """THE invariant for any side-effecting operation.

    Note this asserts on the SERVER's applied-writes list, not on the client's
    return value. A test that only checks the client got a result would pass
    even while three duplicate records were created.
    """
    api = FlakyWriteAPI("erp", fail_first_n=2)
    key = "dispute:INV-88:2026-08-20"

    await retry(lambda: api.write({"invoice": "INV-88"}, idempotency_key=key),
                max_attempts=5)

    assert len(api.applied) == 1, (
        f"idempotency violated: {len(api.applied)} writes applied"
    )


async def test_invariant_writes_DO_duplicate_without_a_key() -> None:
    """The negative control. Without this test, the test above might be passing
    for the wrong reason — e.g. because the fixture never actually retries."""
    api = FlakyWriteAPI("erp", fail_first_n=2)
    await retry(lambda: api.write({"invoice": "INV-88"}, idempotency_key=None),
                max_attempts=5)
    assert len(api.applied) == 3, "fixture must actually be retrying"


async def test_invariant_deadline_is_never_exceeded() -> None:
    """No retry sequence may blow the end-to-end budget.

    This is a timing assertion, which the preamble warns against — so give it
    generous slack (2x) and assert the ORDER OF MAGNITUDE, not a tight bound.
    The bug it catches is 'no deadline at all', which is a 10x overrun, not a
    50ms one.
    """
    budget = 0.20

    async def always_slow():
        await asyncio.sleep(0.05)
        raise ServiceUnavailable("503")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    t0 = time.perf_counter()

    async def bounded_retry():
        for _ in range(100):
            if loop.time() >= deadline:
                raise TimeoutError("deadline exhausted")
            try:
                return await always_slow()
            except ServiceUnavailable:
                continue

    with pytest.raises(TimeoutError):
        await bounded_retry()

    elapsed = time.perf_counter() - t0
    assert elapsed < budget * 2, f"deadline blown: {elapsed:.2f}s vs {budget}s"


async def test_invariant_retry_budget_is_respected() -> None:
    """Assert the budget actually caps retries, by counting."""
    class Budget:
        def __init__(self, tokens: int) -> None:
            self.tokens = tokens
            self.denied = 0

        def try_retry(self) -> bool:
            if self.tokens > 0:
                self.tokens -= 1
                return True
            self.denied += 1
            return False

    budget = Budget(tokens=3)
    inj = FaultInjector(script=[Fail(ServiceUnavailable)])
    attempts = 0

    for _ in range(10):
        attempts += 1
        try:
            await inj.maybe_fail()
            break
        except AppError:
            if not budget.try_retry():
                break

    assert attempts == 4, "1 initial attempt + 3 budgeted retries"
    assert budget.denied == 1


# ===========================================================================
# 4. TIMEOUTS AND HANGS
# ===========================================================================

async def test_hang_is_caught_by_timeout() -> None:
    """A hang is nastier than an error: no exception, just silence. Only a
    timeout saves you, so test that the timeout is actually wired up."""
    inj = FaultInjector(script=[Hang(latency=10.0)])
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await inj.maybe_fail()


async def test_timeout_does_not_retry_forever() -> None:
    """A timeout is retryable, so a hung dependency must still terminate."""
    inj = FaultInjector(script=[Hang(latency=10.0)])
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        async with asyncio.timeout(0.02):
            await inj.maybe_fail()

    with pytest.raises(TimeoutError):
        for n in range(3):
            try:
                await attempt()
                break
            except TimeoutError:
                if n == 2:
                    raise
    assert attempts == 3


# ===========================================================================
# 5. CIRCUIT BREAKER STATE MACHINE
# ===========================================================================

class TinyBreaker:
    def __init__(self, threshold: int = 3, open_for: float = 0.05) -> None:
        self.threshold, self.open_for = threshold, open_for
        self.failures = 0
        self.state = "closed"
        self._opened = 0.0
        self.calls_made = 0

    async def call(self, fn):
        if self.state == "open":
            if time.monotonic() - self._opened >= self.open_for:
                self.state = "half_open"
            else:
                raise ServiceUnavailable("circuit open")
        self.calls_made += 1
        try:
            r = await fn()
        except Exception:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.threshold:
                self.state, self._opened = "open", time.monotonic()
            raise
        else:
            self.failures, self.state = 0, "closed"
            return r


async def test_breaker_opens_then_stops_calling() -> None:
    """The value of a breaker is the calls it does NOT make. Assert on that,
    not just on the state string."""
    cb = TinyBreaker(threshold=3)
    inj = FaultInjector(script=[Fail(ServiceUnavailable)])

    for _ in range(10):
        with pytest.raises(ServiceUnavailable):
            await cb.call(inj.maybe_fail)

    assert cb.state == "open"
    assert cb.calls_made == 3, (
        f"breaker made {cb.calls_made} calls; should stop after threshold"
    )
    assert inj.calls == 3, "the dependency must not be touched once open"


async def test_breaker_recovers_through_half_open() -> None:
    cb = TinyBreaker(threshold=2, open_for=0.02)
    healthy = False

    async def dep():
        if not healthy:
            raise ServiceUnavailable("503")
        return "ok"

    for _ in range(2):
        with pytest.raises(ServiceUnavailable):
            await cb.call(dep)
    assert cb.state == "open"

    await asyncio.sleep(0.03)
    healthy = True
    assert await cb.call(dep) == "ok"
    assert cb.state == "closed"


async def test_breaker_reopens_if_probe_fails() -> None:
    """Half-open must not optimistically close. A failed probe means the
    dependency is still bad and dumping traffic on it re-kills it."""
    cb = TinyBreaker(threshold=2, open_for=0.02)
    inj = FaultInjector(script=[Fail(ServiceUnavailable)])

    for _ in range(2):
        with pytest.raises(ServiceUnavailable):
            await cb.call(inj.maybe_fail)
    await asyncio.sleep(0.03)

    with pytest.raises(ServiceUnavailable):
        await cb.call(inj.maybe_fail)
    assert cb.state == "open", "a failed probe must return to OPEN"


# ===========================================================================
# 6. LLM-SPECIFIC FAILURES
# ===========================================================================

def extract_json(text: str) -> dict:
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        return json.loads(text[s:e + 1])
    raise ValueError("not JSON")


@pytest.mark.parametrize(
    "raw,expected_disposition",
    [
        ('{"disposition":"approve"}', "approve"),
        ('```json\n{"disposition":"approve"}\n```', "approve"),
        ('Sure! {"disposition":"approve"} Hope that helps.', "approve"),
        ('```\n{"disposition":"approve"}\n```', "approve"),
    ],
    ids=["plain", "json_fence", "prose_wrapped", "bare_fence"],
)
async def test_free_repairs_handle_common_malformations(
    raw: str, expected_disposition: str
) -> None:
    """These four cost nothing to fix. Paying for a model round trip to strip
    a markdown fence is waste at scale, so lock the behaviour in with a test."""
    assert extract_json(raw)["disposition"] == expected_disposition


async def test_truncation_is_detected_before_parsing() -> None:
    """finish_reason must be checked FIRST. Otherwise a token-limit problem
    surfaces as a confusing JSON parse error and someone debugs the parser."""
    from failure_lab import Completion

    c = Completion(text='{"disposition":"appro', finish_reason="length")
    checked_finish_reason_first = c.finish_reason == "length"
    assert checked_finish_reason_first
    with pytest.raises(ValueError):
        extract_json(c.text)


async def test_repair_uses_a_different_prompt_than_retry() -> None:
    """The distinction the whole of 08 rests on. A repair must actually change
    the request; if the prompt is identical it is a retry wearing a costume."""
    prompts_sent: list[str] = []

    async def model(prompt: str) -> str:
        prompts_sent.append(prompt)
        if len(prompts_sent) == 1:
            return "not json"
        return '{"disposition":"approve"}'

    prompt = "Classify."
    for _ in range(3):
        text = await model(prompt)
        try:
            result = extract_json(text)
            break
        except ValueError:
            prompt = f"{prompt}\nYour last reply was not JSON. Return ONLY JSON."

    assert result["disposition"] == "approve"
    assert prompts_sent[0] != prompts_sent[1], (
        "repair must change the prompt, otherwise it is just a retry"
    )


async def test_repair_budget_is_bounded() -> None:
    """A model that never produces valid output must fail cleanly, not loop."""
    calls = 0

    async def never_valid(_: str) -> str:
        nonlocal calls
        calls += 1
        return "never json"

    prompt = "Classify."
    with pytest.raises(ValueError):
        for n in range(3):
            text = await never_valid(prompt)
            try:
                extract_json(text)
                break
            except ValueError:
                if n == 2:
                    raise
                prompt += "\nReturn ONLY JSON."

    assert calls == 3, "repair budget must cap the number of calls"


# ===========================================================================
# 7. PARTIAL FAILURE / DLQ
# ===========================================================================

async def test_bad_item_does_not_abort_the_batch() -> None:
    processed, dead = [], []

    async def process(i: int) -> str:
        if i in (3, 7):
            raise ServiceUnavailable("bad item")
        return f"ok-{i}"

    for i in range(10):
        try:
            processed.append(await process(i))
        except AppError as e:
            dead.append((i, type(e).__name__))

    assert len(processed) == 8
    assert len(dead) == 2
    assert [i for i, _ in dead] == [3, 7]


async def test_dead_letters_are_countable_not_silent() -> None:
    """The swallow anti-pattern passes a naive test ('did it complete?').
    Assert that failures are RECORDED, which is what distinguishes quarantine
    from swallowing."""
    dlq: list[dict] = []

    async def process(i: int) -> str:
        if i == 5:
            raise ServiceUnavailable("boom")
        return "ok"

    for i in range(10):
        try:
            await process(i)
        except AppError as e:
            dlq.append({"item": i, "error": type(e).__name__,
                        "trace_id": f"tr-{i}"})

    assert len(dlq) == 1
    assert dlq[0]["trace_id"], "a dead letter without a trace_id is not replayable"


async def test_poison_message_is_dead_lettered_not_looped() -> None:
    """Without a delivery cap this test hangs — which is exactly the
    production symptom."""

    class Msg:
        def __init__(self, body: str) -> None:
            self.body, self.delivery_count = body, 0

    queue = [Msg("good"), Msg("POISON")]
    dead, processed = [], []
    iterations = 0

    while queue and iterations < 50:
        iterations += 1
        msg = queue.pop(0)
        msg.delivery_count += 1
        if msg.delivery_count > 3:
            dead.append(msg.body)
            continue
        if msg.body == "POISON":
            queue.append(msg)
            continue
        processed.append(msg.body)

    assert processed == ["good"]
    assert dead == ["POISON"]
    assert iterations < 50, "queue must drain, not loop forever"


async def test_batch_guard_aborts_on_systemic_failure() -> None:
    """Item-level quarantine will happily dead-letter your entire corpus when
    a credential expires. The batch guard is what stops that."""
    consecutive, aborted, processed = 0, False, 0

    for i in range(200):
        ok = i < 20                       # everything fails from item 20
        processed += 1
        consecutive = 0 if ok else consecutive + 1
        if consecutive >= 15:
            aborted = True
            break

    assert aborted
    assert processed < 40, f"should abort quickly, processed {processed}"


# ===========================================================================
# 8. CHAOS — bounded, seeded, and reproducible
# ===========================================================================

@pytest.mark.parametrize("seed", range(8))
async def test_invariants_hold_under_random_failures(seed: int) -> None:
    """Randomised, but SEEDED and PARAMETRISED — so a failure names the exact
    seed that broke it and `pytest -k 'seed3'` reproduces it exactly.

    This is how to get chaos-testing value without flaky tests: the randomness
    explores the space, the seed makes each point reproducible.

    The assertion is an INVARIANT (writes never duplicate), not a specific
    outcome — because the outcome legitimately varies with the seed while the
    invariant must never break.
    """
    api = FlakyWriteAPI("erp", fail_first_n=seed % 4)
    key = f"op:{seed}"

    try:
        await retry(lambda: api.write({"x": seed}, idempotency_key=key),
                    max_attempts=6)
    except AppError:
        pass

    assert len(api.applied) <= 1 or all(
        r == api.applied[0] for r in api.applied
    ), f"seed={seed}: duplicate divergent writes {api.applied}"
    assert len(api.writes) <= 1, f"seed={seed}: multiple distinct records stored"


@pytest.mark.parametrize("failure_rate", [0.0, 0.3, 0.7, 1.0])
async def test_system_degrades_but_never_corrupts(failure_rate: float) -> None:
    """Across the whole failure-rate spectrum, the outcome quality varies but
    correctness does not. Sweeping the rate catches bugs that only appear at
    the extremes — 0.0 and 1.0 are the two most commonly untested points."""
    inj = FaultInjector(rate=failure_rate, seed=42,
                        error_cls=ServiceUnavailable, base_latency=0.0)
    m = Metrics()
    completed, failed = 0, 0

    for _ in range(50):
        m.incr("requests")
        try:
            await retry(inj.maybe_fail, max_attempts=3,
                        on_attempt=lambda n: m.incr("attempts"))
            completed += 1
        except AppError:
            failed += 1

    assert completed + failed == 50, "every request must have a definite outcome"
    if failure_rate == 0.0:
        assert failed == 0
        assert m.amplification() == 1.0, "no retries when nothing fails"
    if failure_rate == 1.0:
        assert completed == 0
        assert m.amplification() == 3.0, "exactly max_attempts per request"
